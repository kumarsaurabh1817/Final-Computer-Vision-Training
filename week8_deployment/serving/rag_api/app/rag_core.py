# ---- rag_core.py: GENERATED for the RAG API — do not edit here ----
# The Week 6 pipeline and Week 7 agent, concatenated from week6_multimodal_rag/tools/shared and
# week7_agentic_rag/tools/shared by week8_deployment/tools/inline_shared.py.
from pathlib import Path

IN_COLAB = False
HERE = Path(__file__).resolve().parent


def build_corpus(*args, **kwargs):
    """The API only loads a prebuilt Week 6 index; building the corpus happens in the Week 6 notebooks."""
    raise RuntimeError("build the Week 6 index first (week6_multimodal_rag notebooks); the API only loads it")


# ---- storage.py: shared helper code (canonical copy in week6_multimodal_rag/tools/shared/) ----
# Where files live. Colab wipes /content when the session ends, so everything a later day or week needs
# (corpus, parsed elements, index) is written to Google Drive when it is mounted:
#     My Drive/cv_course_artifacts/<name>/
# Without Drive (or when running locally) the same layout is created under cv_course_artifacts/.
import os
from pathlib import Path

USE_DRIVE = True  # Colab: mount Google Drive and keep artifacts there
DRIVE_ROOT = Path("/content/drive/MyDrive")


def get_secret(name: str) -> str | None:
    """Environment variable first, then Colab secrets (key icon in the left sidebar)."""
    if os.environ.get(name):
        return os.environ[name]
    try:
        from google.colab import userdata
        return userdata.get(name)
    except Exception:
        return None


def mount_drive() -> Path | None:
    """Mount Google Drive once (asks for permission) and return My Drive, or None outside Colab."""
    if not (IN_COLAB and USE_DRIVE):
        return None
    if not DRIVE_ROOT.exists():
        from google.colab import drive
        drive.mount("/content/drive")
    return DRIVE_ROOT if DRIVE_ROOT.exists() else None


def artifact_dir(name: str) -> Path:
    """A folder that survives the Colab session (Drive) or the repo-level cv_course_artifacts/ folder locally."""
    drive = mount_drive()
    if drive is not None:
        base = drive / "cv_course_artifacts"
    elif IN_COLAB:
        base = Path("/content/cv_course_artifacts")
    else:
        base = HERE.parent.parent / "cv_course_artifacts"
    path = base / name
    path.mkdir(parents=True, exist_ok=True)
    return path


# ---- parse.py: shared helper code (canonical copy in week6_multimodal_rag/tools/shared/) ----
# PDF -> structured elements with metadata (Week 6 Day 1, reused by every later day).
#
#     digital page:  PyMuPDF text blocks (font size -> titles) + find_tables() + raster images + clustered vector drawings
#     scanned page:  render -> Tesseract (paragraphs) + Table Transformer (table regions) + OpenCV (figure regions)
#
# Every element carries provenance: doc_id, page, bbox (PDF points, origin top-left), type, section, parser.
import re
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np
import pymupdf
import pytesseract
import torch
from PIL import Image


@dataclass
class Element:
    id: str
    doc_id: str
    page: int
    type: str                 # title | text | table | figure
    bbox: list                # [x0, y0, x1, y1] in PDF points
    text: str = ""            # text; table as markdown; figure: its printed caption ("Figure 2: ...")
    section: str = ""         # nearest heading above
    rows: list | None = None  # table cells
    image_path: str | None = None  # figure / table crop
    caption: str = ""         # VLM caption of a figure (Day 2)
    parser: str = "pymupdf"

    def to_dict(self) -> dict:
        return asdict(self)


def rows_to_markdown(rows: list) -> str:
    rows = [[re.sub(r"\s+", " ", str(c or "")).strip() for c in r] for r in rows if any(c for c in r)]
    if not rows:
        return ""
    w = max(len(r) for r in rows)
    rows = [r + [""] * (w - len(r)) for r in rows]
    return "\n".join(["| " + " | ".join(rows[0]) + " |", "|" + "---|" * w] + ["| " + " | ".join(r) + " |" for r in rows[1:]])


def _area(b) -> float:
    return max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])


def overlap_frac(a, b) -> float:
    """Fraction of box a covered by box b."""
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    return ix * iy / max(1e-6, _area(a))


def dedupe_boxes(boxes: list, thr: float = 0.6) -> list:
    out = []
    for b in sorted(boxes, key=_area, reverse=True):
        if not any(overlap_frac(b, o) > thr for o in out):
            out.append(b)
    return out


def vector_figure_regions(page, exclude: list, min_frac: float = 0.03, pad: int = 8) -> list:
    """Vector graphics (plots drawn as paths) -> figure boxes: paint every drawing's rect on a mask, dilate, connected components."""
    w, h = int(page.rect.width) + 1, int(page.rect.height) + 1
    mask = np.zeros((h, w), np.uint8)
    n = 0
    for d in page.get_drawings():
        r = d.get("rect")
        if r is None:
            continue
        if r.width > 0.9 * w and r.height < 4:  # full-width rule lines are not figures
            continue
        if r.width * r.height > 0.4 * w * h:  # page backgrounds and frames
            continue
        cv2.rectangle(mask, (int(r.x0), int(r.y0)), (int(r.x1), int(r.y1)), 255, -1)
        n += 1
    if n == 0:
        return []
    mask = cv2.dilate(mask, np.ones((2 * pad + 1, 2 * pad + 1), np.uint8))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for c in contours:
        x, y, bw, bh = cv2.boundingRect(c)
        b = [x + pad / 2, y + pad / 2, x + bw - pad / 2, y + bh - pad / 2]
        if _area(b) > min_frac * w * h and not any(overlap_frac(b, e) > 0.5 for e in exclude):
            boxes.append(b)
    return boxes


def text_blocks(page) -> list[dict]:
    """PyMuPDF text blocks with the largest span size (used to tell headings from body text)."""
    out = []
    for b in page.get_text("dict")["blocks"]:
        if b.get("type") != 0:
            continue
        lines, sizes = [], []
        for line in b["lines"]:
            spans = [s for s in line["spans"] if s["text"].strip()]
            if spans:
                lines.append("".join(s["text"] for s in spans))
                sizes += [s["size"] for s in spans]
        text = re.sub(r"\s+", " ", " ".join(lines)).strip()
        if text:
            out.append({"bbox": list(b["bbox"]), "text": text, "size": max(sizes)})
    return out


def figure_caption_near(box, blocks: list[dict], max_gap: float = 70) -> str:
    """The 'Figure N: ...' text right below (or above) a figure, horizontally overlapping it."""
    best, best_gap = "", max_gap
    for b in blocks:
        if not re.match(r"^(fig(ure)?\.?\s*\d+)", b["text"], re.I):
            continue
        if min(box[2], b["bbox"][2]) - max(box[0], b["bbox"][0]) <= 0:
            continue
        gap = b["bbox"][1] - box[3] if b["bbox"][1] >= box[3] - 5 else box[1] - b["bbox"][3]
        if -5 <= gap < best_gap:
            best, best_gap = b["text"], gap
    return best


def crop_region(page, box, path: Path, dpi: int = 150) -> str:
    pix = page.get_pixmap(dpi=dpi, clip=pymupdf.Rect(*box))
    Image.frombytes("RGB", (pix.width, pix.height), pix.samples).save(path)
    return str(path)


class TableDetector:
    """Microsoft Table Transformer (DETR) table detection on page images. Loaded lazily."""

    def __init__(self, device: str = "cpu", threshold: float = 0.8):
        from transformers import AutoImageProcessor, TableTransformerForObjectDetection
        self.proc = AutoImageProcessor.from_pretrained("microsoft/table-transformer-detection")
        self.model = TableTransformerForObjectDetection.from_pretrained("microsoft/table-transformer-detection").to(device).eval()
        self.device, self.threshold = device, threshold

    @torch.inference_mode()
    def detect(self, img: Image.Image) -> list[list[float]]:
        inputs = self.proc(images=img, return_tensors="pt").to(self.device)
        out = self.model(**inputs)
        res = self.proc.post_process_object_detection(out, threshold=self.threshold, target_sizes=[img.size[::-1]])[0]
        return [b.tolist() for b, lab in zip(res["boxes"], res["labels"])
                if self.model.config.id2label[int(lab)].startswith("table")]


def figure_regions_cv(img: Image.Image, word_boxes: list, exclude: list, min_frac: float = 0.03) -> list:
    """Scanned page: ink that is not text and not a table -> figure boxes (pixels)."""
    gray = np.asarray(img.convert("L"))
    ink = (gray < 200).astype(np.uint8) * 255
    for x0, y0, x1, y1 in word_boxes:
        ink[int(y0):int(y1), int(x0):int(x1)] = 0
    for x0, y0, x1, y1 in exclude:
        ink[int(y0):int(y1), int(x0):int(x1)] = 0
    ink = cv2.morphologyEx(ink, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))  # drop scan noise
    ink = cv2.dilate(ink, np.ones((25, 25), np.uint8))
    contours, _ = cv2.findContours(ink, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    h, w = gray.shape
    boxes = []
    for c in contours:
        x, y, bw, bh = cv2.boundingRect(c)
        if bw * bh > min_frac * w * h:
            boxes.append([x, y, x + bw, y + bh])
    return boxes


def parse_scanned_page(page, pno: int, doc_id: str, out_dir: Path, dpi: int, detector: "TableDetector | None") -> list[Element]:
    pix = page.get_pixmap(dpi=dpi)
    img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
    s = 72 / dpi  # pixels -> PDF points
    data = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT, config="--psm 3")
    paras, words = {}, []
    for i, t in enumerate(data["text"]):
        if not t.strip() or float(data["conf"][i]) < 30:
            continue
        box = [data["left"][i], data["top"][i], data["left"][i] + data["width"][i], data["top"][i] + data["height"][i]]
        words.append(box)
        paras.setdefault((data["block_num"][i], data["par_num"][i]), []).append((t, box))
    tables = detector.detect(img) if detector is not None else []
    figs = figure_regions_cv(img, words, tables)
    out = []
    for i, box in enumerate(tables):
        crop = img.crop([int(v) for v in box])
        rows = [re.split(r"\s{2,}|\s\|\s", line.strip()) for line in pytesseract.image_to_string(crop, config="--psm 6").splitlines() if line.strip()]
        out.append(Element("", doc_id, pno, "table", [v * s for v in box], rows_to_markdown(rows), rows=rows, parser="ocr+tatr"))
    para_items = []
    heights = [b[3] - b[1] for b in words] or [10]
    med_h = float(np.median(heights))
    for (_, _), items in paras.items():
        box = [min(b[0] for _, b in items), min(b[1] for _, b in items), max(b[2] for _, b in items), max(b[3] for _, b in items)]
        if any(overlap_frac(box, r) > 0.5 for r in tables + figs):
            continue
        text = " ".join(t for t, _ in items)
        h = float(np.median([b[3] - b[1] for _, b in items]))
        typ = "title" if h > 1.35 * med_h and len(text) < 120 else "text"
        para_items.append({"bbox": [v * s for v in box], "text": text, "type": typ})
    blocks = [{"bbox": p["bbox"], "text": p["text"]} for p in para_items]
    for i, box in enumerate(figs):
        pbox = [v * s for v in box]
        path = out_dir / f"{doc_id}_p{pno}_fig{i}.png"
        img.crop([int(v) for v in box]).save(path)
        out.append(Element("", doc_id, pno, "figure", pbox, figure_caption_near(pbox, blocks), image_path=str(path), parser="ocr+cv"))
    out += [Element("", doc_id, pno, p["type"], p["bbox"], p["text"], parser="ocr") for p in para_items]
    return out


def absorb_labels(box, blocks: list[dict], gap: float = 15, max_chars: int = 40) -> list:
    """Grow a figure box over the short text around it (axis ticks, legends, plot titles) so it is part of the figure."""
    grown = list(box)
    changed = True
    while changed:
        changed = False
        for b in blocks:
            bb = b["bbox"]
            if len(b["text"]) > max_chars or re.match(r"^(fig(ure)?|table)\.?\s*\d", b["text"], re.I):
                continue
            near = bb[0] < grown[2] + gap and bb[2] > grown[0] - gap and bb[1] < grown[3] + gap and bb[3] > grown[1] - gap
            if near and overlap_frac(bb, grown) < 0.99:
                grown = [min(grown[0], bb[0]), min(grown[1], bb[1]), max(grown[2], bb[2]), max(grown[3], bb[3])]
                changed = True
    return grown


def table_rows(page, t) -> list[list[str]]:
    """Cell texts from the words inside each cell box (more robust than Table.extract() for right-aligned numbers)."""
    try:
        return [[" ".join(w[4] for w in page.get_text("words", clip=pymupdf.Rect(c))) if c else "" for c in row.cells]
                for row in t.rows]
    except Exception:
        return t.extract()


def parse_digital_page(page, pno: int, doc_id: str, out_dir: Path, dpi: int) -> list[Element]:
    page_area = page.rect.width * page.rect.height
    tables = []
    try:
        for t in page.find_tables().tables:
            rows = table_rows(page, t)
            if len(rows) >= 2 and max(len(r) for r in rows) >= 2:
                tables.append((list(t.bbox), rows))
    except Exception:
        pass
    tboxes = [b for b, _ in tables]
    figs = []
    for img in page.get_images(full=True):
        for r in page.get_image_rects(img[0]):
            if r.width * r.height > 0.02 * page_area:
                figs.append([r.x0, r.y0, r.x1, r.y1])
    figs = dedupe_boxes(figs + vector_figure_regions(page, exclude=tboxes + figs))
    blocks = text_blocks(page)
    figs = [absorb_labels(b, blocks) for b in figs]
    # body size = median over characters, so a few short headings cannot shift it
    body = float(np.median(np.repeat([b["size"] for b in blocks], [len(b["text"]) for b in blocks]))) if blocks else 10.0
    out = []
    for b in blocks:
        if any(overlap_frac(b["bbox"], r) > 0.6 for r in tboxes + figs):
            continue  # text inside a table or a chart (cells, axis labels) belongs to that element
        typ = "title" if b["size"] >= 1.2 * body and len(b["text"]) < 120 else "text"
        out.append(Element("", doc_id, pno, typ, b["bbox"], b["text"]))
    for box, rows in tables:
        out.append(Element("", doc_id, pno, "table", box, rows_to_markdown(rows), rows=rows, parser="pymupdf.find_tables"))
    for i, box in enumerate(figs):
        path = crop_region(page, box, out_dir / f"{doc_id}_p{pno}_fig{i}.png", dpi)
        out.append(Element("", doc_id, pno, "figure", box, figure_caption_near(box, blocks), image_path=path, parser="pymupdf.images+drawings"))
    return out


def reading_order(els: list[Element], page_width: float) -> list[Element]:
    """Sort top-to-bottom, left column before right column (good enough for 1- and 2-column layouts)."""
    def key(e):
        full = (e.bbox[2] - e.bbox[0]) > 0.6 * page_width
        col = 0 if full or e.bbox[0] < 0.45 * page_width else 1
        return (e.page, col if not full else 0, e.bbox[1], e.bbox[0])
    return sorted(els, key=key)


def parse_pdf(path, doc_id: str, out_dir: Path, dpi: int = 150, ocr_min_words: int = 25,
              detector: "TableDetector | None" = None) -> list[Element]:
    """All elements of one PDF, in reading order, with ids '<doc>:p<page>:<type><n>' and section headings filled in."""
    out_dir.mkdir(parents=True, exist_ok=True)
    els = []
    doc = pymupdf.open(path)
    for pno, page in enumerate(doc, 1):
        if len(page.get_text("words")) < ocr_min_words:
            page_els = parse_scanned_page(page, pno, doc_id, out_dir, dpi, detector)
        else:
            page_els = parse_digital_page(page, pno, doc_id, out_dir, dpi)
        els += reading_order(page_els, page.rect.width)
    section, counts = "", {}
    for e in els:
        if e.type == "title":
            section = e.text
        e.section = section
        counts[(e.page, e.type)] = counts.get((e.page, e.type), 0) + 1
        e.id = f"{doc_id}:p{e.page}:{e.type}{counts[(e.page, e.type)]}"
    return els


# ---- llm.py: shared helper code (canonical copy in week6_multimodal_rag/tools/shared/) ----
# One chat interface over two backends:
#     "local"       Qwen3-VL on the Colab GPU via transformers (default: no account, nothing leaves the VM)
#     "openrouter"  any vision model on openrouter.ai (OpenAI-compatible API; set OPENROUTER_API_KEY in Colab secrets)
# Messages use one format for both:
#     [{"role": "system" | "user" | "assistant",
#       "content": "text" or [{"type": "text", "text": ...}, {"type": "image", "image": <PIL.Image>}]}]
# Plus the Week 5 Day 3 JSON helpers: extract the first JSON object, validate with pydantic, repair once.
import base64
import io
import json
import re
import time

import requests
import torch
from PIL import Image
from pydantic import BaseModel, ValidationError


def fit_image(img: Image.Image, max_side: int | None) -> Image.Image:
    img = img.convert("RGB")
    if max_side and max(img.size) > max_side:
        s = max_side / max(img.size)
        img = img.resize((max(1, round(img.width * s)), max(1, round(img.height * s))), Image.BICUBIC)
    return img


def default_dtype() -> torch.dtype:
    """bf16 on Ampere+ (A100, L4), fp16 on T4 (no native bf16), fp32 on CPU."""
    if torch.cuda.is_available():
        return torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    return torch.float32


class LLM:
    def __init__(self, backend: str = "local", model_id: str | None = None, max_side: int = 1024):
        self.backend, self.max_side = backend, max_side
        self.calls, self.seconds, self.tokens_in, self.tokens_out = 0, 0.0, 0, 0
        if backend == "local":
            from transformers import AutoModelForImageTextToText, AutoProcessor
            self.model_id = model_id or "Qwen/Qwen3-VL-4B-Instruct"
            self.processor = AutoProcessor.from_pretrained(self.model_id)
            self.model = AutoModelForImageTextToText.from_pretrained(
                self.model_id, dtype=default_dtype(), device_map="auto", attn_implementation="sdpa").eval()
        elif backend == "openrouter":
            self.model_id = model_id or "qwen/qwen3-vl-8b-instruct"
            self.key = get_secret("OPENROUTER_API_KEY")
            if not self.key:
                raise RuntimeError("Set OPENROUTER_API_KEY in Colab secrets (or use backend='local').")
        else:
            raise ValueError(f"unknown backend {backend!r}")

    def _normalize(self, messages: list) -> list:
        out = []
        for m in messages:
            content = m["content"]
            if isinstance(content, str):
                content = [{"type": "text", "text": content}]
            parts = [{"type": "image", "image": fit_image(p["image"], self.max_side)} if p["type"] == "image" else p for p in content]
            out.append({"role": m["role"], "content": parts})
        return out

    @torch.inference_mode()
    def _local(self, msgs: list, max_new_tokens: int) -> str:
        inputs = self.processor.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True, return_dict=True,
                                                    return_tensors="pt").to(self.model.device)
        out = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        new = out[:, inputs["input_ids"].shape[1]:]
        self.tokens_in += int(inputs["input_ids"].shape[1])
        self.tokens_out += int((new[0] != self.processor.tokenizer.pad_token_id).sum())
        return self.processor.batch_decode(new, skip_special_tokens=True)[0].strip()

    def _remote(self, msgs: list, max_new_tokens: int) -> str:
        def part(p):
            if p["type"] == "text":
                return p
            buf = io.BytesIO()
            p["image"].save(buf, format="PNG")
            return {"type": "image_url", "image_url": {"url": "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()}}
        body = {"model": self.model_id, "temperature": 0, "max_tokens": max_new_tokens,
                "messages": [{"role": m["role"], "content": [part(p) for p in m["content"]]} for m in msgs]}
        for attempt in range(4):
            r = requests.post("https://openrouter.ai/api/v1/chat/completions", json=body, timeout=180,
                              headers={"Authorization": f"Bearer {self.key}"})
            if r.status_code == 429 and attempt < 3:  # free tiers rate-limit: back off and retry
                time.sleep(5 * 2 ** attempt)
                continue
            r.raise_for_status()
            j = r.json()
            usage = j.get("usage") or {}
            self.tokens_in += usage.get("prompt_tokens", 0)
            self.tokens_out += usage.get("completion_tokens", 0)
            return (j["choices"][0]["message"]["content"] or "").strip()
        raise RuntimeError("OpenRouter kept rate-limiting")

    def chat(self, messages: list, max_new_tokens: int = 512) -> str:
        t0 = time.perf_counter()
        msgs = self._normalize(messages)
        text = self._local(msgs, max_new_tokens) if self.backend == "local" else self._remote(msgs, max_new_tokens)
        self.calls += 1
        self.seconds += time.perf_counter() - t0
        return text

    def ask(self, prompt: str, images: list = (), system: str | None = None, max_new_tokens: int = 512) -> str:
        content = [{"type": "image", "image": im} for im in images] + [{"type": "text", "text": prompt}]
        msgs = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": content}]
        return self.chat(msgs, max_new_tokens)

    def stats(self) -> dict:
        return {"calls": self.calls, "seconds": round(self.seconds, 1), "tokens_in": self.tokens_in, "tokens_out": self.tokens_out}


def extract_json(text: str) -> str | None:
    """The first balanced {...} object in a reply (tolerates ```json fences and chatter around it)."""
    text = re.sub(r"```(?:json)?", "", text)
    start = text.find("{")
    if start < 0:
        return None
    depth, in_str, esc = 0, False, False
    for i, ch in enumerate(text[start:], start):
        if in_str:
            esc = (ch == "\\") and not esc
            if ch == '"' and not esc:
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start:i + 1]
    return None


def validate_json(text: str, schema: type[BaseModel]) -> tuple[dict | None, BaseModel | None, str | None]:
    blob = extract_json(text)
    if blob is None:
        return None, None, "no JSON object found in the reply"
    try:
        data = json.loads(blob)
    except json.JSONDecodeError as exc:
        return None, None, f"invalid JSON: {exc}"
    try:
        return data, schema.model_validate(data), None
    except ValidationError as exc:
        return data, None, "; ".join(f"{'.'.join(map(str, e['loc']))}: {e['msg']}" for e in exc.errors()[:6])


def ask_json(llm: "LLM", messages: list, schema: type[BaseModel], max_new_tokens: int = 512, repairs: int = 1):
    """Schema in the prompt -> validate -> send the error back (Week 5 'schema_repair'). Returns (obj | None, raw, attempts)."""
    schema_txt = ("\nReply with one JSON object only, valid against this JSON Schema:\n"
                  + json.dumps(schema.model_json_schema(), separators=(",", ":")))
    msgs = [dict(m) for m in messages]
    last = msgs[-1]
    last["content"] = (last["content"] + schema_txt) if isinstance(last["content"], str) else \
        list(last["content"]) + [{"type": "text", "text": schema_txt}]
    for attempt in range(1, repairs + 2):
        raw = llm.chat(msgs, max_new_tokens)
        _, obj, err = validate_json(raw, schema)
        if obj is not None or attempt > repairs:
            return obj, raw, attempt
        msgs = msgs + [{"role": "assistant", "content": raw},
                       {"role": "user", "content": f"That reply is not valid: {err}. Return the corrected JSON object only."}]
    return None, raw, attempt


# ---- represent.py: shared helper code (canonical copy in week6_multimodal_rag/tools/shared/) ----
# Elements -> retrievable items (Week 6 Day 2):
#     text   : chunks (section-aware by default), prefixed with document title + section for context
#     table  : markdown with its "Table N:" caption line
#     figure : printed caption + a VLM caption that reads the chart (searchable text) + the crop (for CLIP and the VLM)
# Plus the two embedding models: a text embedder (BGE) and CLIP (image <-> text).
import re

import numpy as np
import open_clip
import torch
from langchain_text_splitters import RecursiveCharacterTextSplitter
from PIL import Image
from sentence_transformers import SentenceTransformer

FIGURE_PROMPT = ("This figure comes from the document '{title}', page {page}. Its printed caption is: '{printed}'.\n"
                 "Describe it for a search index in 3-5 sentences: the chart type, its title, the axes or categories, "
                 "every data value and label you can read (copy numbers exactly), and the main takeaway. No preamble.")


def doc_titles(elements: list, docs: list[dict] | None = None) -> dict:
    titles = {d["doc_id"]: d["title"] for d in (docs or [])}
    for e in elements:
        if e.doc_id not in titles and e.type == "title" and e.page == 1:
            titles[e.doc_id] = e.text
    return titles


def caption_figures(elements: list, llm: "LLM", titles: dict, max_new_tokens: int = 220, verbose: bool = True) -> None:
    """Fill Element.caption for every figure (one VLM call each)."""
    figs = [e for e in elements if e.type == "figure" and e.image_path and not e.caption]
    for i, e in enumerate(figs):
        prompt = FIGURE_PROMPT.format(title=titles.get(e.doc_id, e.doc_id), page=e.page, printed=e.text or "none")
        e.caption = llm.ask(prompt, [Image.open(e.image_path)], max_new_tokens=max_new_tokens)
        if verbose and i % 5 == 0:
            print(f"captioned {i + 1}/{len(figs)}: {e.id} -> {e.caption[:90]}...")


def _table_caption(e, elements: list) -> str:
    near = [x for x in elements if x.doc_id == e.doc_id and x.page == e.page and x.type == "text" and re.match(r"^table\s*\d", x.text, re.I)]
    return near[0].text if near else ""


def _split_with_pages(parts: list[tuple[str, int]], splitter) -> list[tuple[str, list[int]]]:
    """Join (text, page) parts, split, and give each chunk the pages its characters came from."""
    joined, spans = "", []
    for text, page in parts:
        spans.append((len(joined), len(joined) + len(text), page))
        joined += text + "\n"
    out, cursor = [], 0
    for chunk in splitter.split_text(joined):
        start = joined.find(chunk, max(0, cursor - 400))
        start = start if start >= 0 else joined.find(chunk)
        end = start + len(chunk)
        cursor = start + 1
        pages = sorted({p for s, e, p in spans if s < end and e > start}) or [parts[0][1]]
        out.append((chunk, pages))
    return out


def make_items(elements: list, titles: dict, strategy: str = "section", chunk_size: int = 700, overlap: int = 100) -> list[dict]:
    """
    Retrievable items. strategy:
      "section"  split each section separately, prefix 'Title > Section' (default)
      "fixed"    split the whole document text into fixed-size windows, no headings
      "element"  one item per text block
    """
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap)
    fixed = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap, separators=[" ", ""])
    items = []

    def add(doc_id, typ, text, pages, bbox=None, image_path=None, element_ids=()):
        items.append({"id": f"{doc_id}:{typ}{sum(1 for i in items if i['doc_id'] == doc_id and i['type'] == typ) + 1}",
                      "doc_id": doc_id, "doc_title": titles.get(doc_id, doc_id), "type": typ, "page": pages[0], "pages": pages,
                      "bbox": bbox, "text": text, "image_path": image_path, "element_ids": list(element_ids)})

    for doc_id in dict.fromkeys(e.doc_id for e in elements):
        els = [e for e in elements if e.doc_id == doc_id]
        title = titles.get(doc_id, doc_id)
        texts = [e for e in els if e.type in ("text", "title")]
        if strategy == "element":
            for e in texts:
                if e.type == "text":
                    add(doc_id, "text", f"{title} > {e.section}\n{e.text}", [e.page], e.bbox, element_ids=[e.id])
        elif strategy == "fixed":
            for chunk, pages in _split_with_pages([(e.text, e.page) for e in texts], fixed):
                add(doc_id, "text", chunk, pages)
        else:
            groups = []
            for e in texts:
                if not groups or groups[-1][0] != e.section:
                    groups.append((e.section, []))
                if e.type == "text":
                    groups[-1][1].append(e)
            for section, members in groups:
                if members:
                    for chunk, pages in _split_with_pages([(e.text, e.page) for e in members], splitter):
                        add(doc_id, "text", f"{title} > {section}\n{chunk}", pages, element_ids=[m.id for m in members])
        for e in els:
            if e.type == "table":
                add(doc_id, "table", f"{title} > {e.section}\nTable on page {e.page}. {_table_caption(e, els)}\n{e.text}",
                    [e.page], e.bbox, element_ids=[e.id])
            elif e.type == "figure":
                add(doc_id, "figure", f"{title} > {e.section}\nFigure on page {e.page}. {e.text}\n{e.caption}".strip(),
                    [e.page], e.bbox, e.image_path, [e.id])
    return items


class TextEmbedder:
    """BGE-small: 384-d, strong for its size. Queries get BGE's retrieval instruction prefix; passages do not."""

    def __init__(self, model_id: str = "BAAI/bge-small-en-v1.5", device: str = "cpu"):
        self.model_id = model_id
        self.model = SentenceTransformer(model_id, device=device)
        self.prefix = "Represent this sentence for searching relevant passages: " if "bge" in model_id else ""

    def passages(self, texts: list[str]) -> np.ndarray:
        return self.model.encode(texts, batch_size=64, normalize_embeddings=True, convert_to_numpy=True).astype(np.float32)

    def queries(self, texts: list[str]) -> np.ndarray:
        return self.model.encode([self.prefix + t for t in texts], normalize_embeddings=True, convert_to_numpy=True).astype(np.float32)


class ClipEmbedder:
    """open_clip model: images and texts -> unit vectors in one space (Week 5 Day 5)."""

    def __init__(self, arch: str = "ViT-B-32", pretrained: str = "openai", device: str = "cpu"):
        self.name, self.device = f"{arch}/{pretrained}", device
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(arch, pretrained=pretrained, device=device)
        self.model.eval()
        self.tokenizer = open_clip.get_tokenizer(arch)

    @torch.inference_mode()
    def images(self, imgs: list, bs: int = 32) -> np.ndarray:
        out = []
        for i in range(0, len(imgs), bs):
            batch = [Image.open(x).convert("RGB") if isinstance(x, (str, Path)) else x.convert("RGB") for x in imgs[i:i + bs]]
            x = torch.stack([self.preprocess(im) for im in batch]).to(self.device)
            out.append(torch.nn.functional.normalize(self.model.encode_image(x).float(), dim=-1).cpu())
        return torch.cat(out).numpy().astype(np.float32) if out else np.zeros((0, 512), np.float32)

    @torch.inference_mode()
    def texts(self, texts: list[str], bs: int = 128) -> np.ndarray:
        out = []
        for i in range(0, len(texts), bs):
            t = self.model.encode_text(self.tokenizer(texts[i:i + bs]).to(self.device))
            out.append(torch.nn.functional.normalize(t.float(), dim=-1).cpu())
        return torch.cat(out).numpy().astype(np.float32)


# ---- index.py: shared helper code (canonical copy in week6_multimodal_rag/tools/shared/) ----
# Multimodal index (Week 6 Day 3) + retrieval (Day 4).
#
# Two vector spaces, one metadata store:
#     text space  (BGE)  : every item's text  -> text chunks, table markdown, figure captions ("caption-as-text")
#     image space (CLIP) : every figure crop  -> reached by a CLIP text query or a query image ("raw embedding")
# Each hit carries its provenance (doc, page, bbox, type, element ids, image path) all the way to the answer.
import json

import faiss
import numpy as np
from PIL import Image


class MultimodalIndex:
    def __init__(self, text_emb: "TextEmbedder", clip: "ClipEmbedder"):
        self.text_emb, self.clip = text_emb, clip
        self.items: list[dict] = []
        self.text_index = self.image_index = None
        self.image_rows: list[int] = []  # image-space row -> item index

    def build(self, items: list[dict]) -> "MultimodalIndex":
        self.items = items
        tv = self.text_emb.passages([it["text"] for it in items])
        self.text_index = faiss.IndexFlatIP(tv.shape[1])
        self.text_index.add(tv)
        self.image_rows = [i for i, it in enumerate(items) if it.get("image_path")]
        iv = self.clip.images([items[i]["image_path"] for i in self.image_rows])
        self.image_index = faiss.IndexFlatIP(iv.shape[1] if len(iv) else 512)
        if len(iv):
            self.image_index.add(iv)
        return self

    def _hits(self, index, qv: np.ndarray, k: int, rows: list[int] | None, space: str, types=None, doc_ids=None) -> list[dict]:
        if index is None or index.ntotal == 0:
            return []
        scores, ids = index.search(qv.reshape(1, -1).astype(np.float32), min(index.ntotal, max(k * 5, 50)))
        out = []
        for s, i in zip(scores[0], ids[0]):
            if i < 0:
                continue
            it = self.items[rows[i] if rows is not None else i]
            if (types and it["type"] not in types) or (doc_ids and it["doc_id"] not in doc_ids):
                continue  # filters are applied after an over-fetch (FAISS flat indexes have no payload filters)
            out.append({**it, "score": float(s), "space": space})
            if len(out) >= k:
                break
        return out

    def search_text(self, query: str, k: int = 5, **filters) -> list[dict]:
        return self._hits(self.text_index, self.text_emb.queries([query])[0], k, None, "text", **filters)

    def search_image_by_text(self, query: str, k: int = 5, **filters) -> list[dict]:
        return self._hits(self.image_index, self.clip.texts([query])[0], k, self.image_rows, "clip_text->image", **filters)

    def search_by_image(self, image, query: str | None = None, alpha: float = 0.5, k: int = 5, **filters) -> list[dict]:
        v = self.clip.images([image])[0]
        if query:
            v = v + alpha * self.clip.texts([query])[0]
            v = v / np.linalg.norm(v)
        return self._hits(self.image_index, v, k, self.image_rows, "clip_image->image", **filters)

    def save(self, folder: Path) -> None:
        folder.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.text_index, str(folder / "text.faiss"))
        faiss.write_index(self.image_index, str(folder / "image.faiss"))
        (folder / "items.json").write_text(json.dumps({"text_model": self.text_emb.model_id, "clip_model": self.clip.name,
                                                       "image_rows": self.image_rows, "items": self.items}, indent=1), encoding="utf-8")

    @classmethod
    def load(cls, folder: Path, text_emb: "TextEmbedder", clip: "ClipEmbedder") -> "MultimodalIndex":
        meta = json.loads((folder / "items.json").read_text(encoding="utf-8"))
        if meta["text_model"] != text_emb.model_id or meta["clip_model"] != clip.name:
            raise ValueError(f"index built with {meta['text_model']} / {meta['clip_model']}")
        obj = cls(text_emb, clip)
        obj.items, obj.image_rows = meta["items"], meta["image_rows"]
        obj.text_index = faiss.read_index(str(folder / "text.faiss"))
        obj.image_index = faiss.read_index(str(folder / "image.faiss"))
        return obj


def rrf_fuse(result_lists: list[list[dict]], k: int, c: int = 60) -> list[dict]:
    """Reciprocal-rank fusion over item ids; a fused hit remembers every space that found it."""
    scores, best = {}, {}
    for hits in result_lists:
        for rank, h in enumerate(hits, 1):
            scores[h["id"]] = scores.get(h["id"], 0.0) + 1.0 / (c + rank)
            if h["id"] not in best:
                best[h["id"]] = {**h, "spaces": [h["space"]]}
            elif h["space"] not in best[h["id"]]["spaces"]:
                best[h["id"]]["spaces"].append(h["space"])
    order = sorted(scores, key=scores.get, reverse=True)[:k]
    return [{**best[i], "score": scores[i]} for i in order]


class Reranker:
    """Cross-encoder: reads (query, passage) together. Slower than embeddings, sharper ranking of the top candidates."""

    def __init__(self, model_id: str = "cross-encoder/ms-marco-MiniLM-L-6-v2", device: str = "cpu"):
        from sentence_transformers import CrossEncoder
        self.model = CrossEncoder(model_id, device=device)

    def rerank(self, query: str, hits: list[dict], k: int) -> list[dict]:
        if not hits:
            return hits
        scores = self.model.predict([(query, h["text"][:1500]) for h in hits])
        return [{**h, "rerank_score": float(s)} for s, h in sorted(zip(scores, hits), key=lambda x: -x[0])][:k]


def retrieve(index: MultimodalIndex, query: str | None = None, image=None, k: int = 5, mode: str = "hybrid",
             reranker: "Reranker | None" = None, types=None, doc_ids=None) -> list[dict]:
    """
    mode "text"   : text space only (chunks, tables, figure captions-as-text)
         "image"  : CLIP image space only (text->image, or image->image when an image is given)
         "hybrid" : both, fused with RRF (default)
    A query image adds image->image search (composed with the text when both are given).
    """
    filters = {"types": types, "doc_ids": doc_ids}
    lists = []
    if query and mode in ("text", "hybrid"):
        lists.append(index.search_text(query, k * 3, **filters))
    if query and mode in ("image", "hybrid") and image is None:
        lists.append(index.search_image_by_text(query, k * 3, **filters))
    if image is not None:
        lists.append(index.search_by_image(image, query if mode != "image" else None, k=k * 3, **filters))
    fused = rrf_fuse(lists, k * 3 if reranker else k)
    return reranker.rerank(query, fused, k) if reranker and query else fused[:k]


# ---- generate.py: shared helper code (canonical copy in week6_multimodal_rag/tools/shared/) ----
# Grounded generation (Week 6 Day 5): retrieved items -> numbered sources [S1..Sn] (text + actual figure images)
# -> VLM answer in a JSON schema with citations -> citations mapped back to item ids, docs, pages.
import time

from PIL import Image
from pydantic import BaseModel, Field


class GroundedAnswer(BaseModel):
    answer: str = Field(description="the answer, using only the sources; numbers copied exactly")
    citations: list[str] = Field(description="source labels used, e.g. ['S1', 'S3']")
    insufficient_context: bool = Field(description="true if the sources do not contain the answer")


ANSWER_RULES = ("Answer the question using ONLY the numbered sources below. Cite every source you use as [S#]. "
                "Copy numbers exactly as written in the sources. Figures are attached as images; read values from them. "
                "If the sources do not contain the answer, set insufficient_context to true and answer "
                "\"I don't know based on the documents.\"")


def build_context(hits: list[dict], max_images: int = 3, max_chars: int = 1200) -> tuple[str, list, dict]:
    """Numbered source block, the images to attach, and label -> hit."""
    lines, images, labels = [], [], {}
    for i, h in enumerate(hits, 1):
        label = f"S{i}"
        labels[label] = h
        note = ""
        if h.get("image_path") and len(images) < max_images:
            images.append(Image.open(h["image_path"]))
            note = f" [shown as attached image #{len(images)}]"
        lines.append(f"[{label}] ({h['type']}, {h['doc_title']}, page {h['page']}){note}\n{h['text'][:max_chars]}")
    return "\n\n".join(lines), images, labels


def generate_answer(llm: "LLM", question: str, hits: list[dict], query_image=None, max_images: int = 3,
                    use_images: bool = True, max_new_tokens: int = 300) -> dict:
    """One grounded answer. Returns answer, citations (labels and item ids), validity, latency."""
    t0 = time.perf_counter()
    context, images, labels = build_context(hits, max_images if use_images else 0)
    content = []
    if query_image is not None:
        content += [{"type": "text", "text": "The user's query image:"}, {"type": "image", "image": query_image}]
    content += [{"type": "image", "image": im} for im in images]
    content.append({"type": "text", "text": f"{ANSWER_RULES}\n\nSOURCES:\n{context}\n\nQUESTION: {question}"})
    obj, raw, attempts = ask_json(llm, [{"role": "user", "content": content}], GroundedAnswer, max_new_tokens)
    if obj is None:
        return {"answer": raw, "citations": [], "cited_items": [], "insufficient": False, "valid": False,
                "attempts": attempts, "latency_s": time.perf_counter() - t0, "sources": labels}
    cited = [c.strip("[] ") for c in obj.citations]
    return {"answer": obj.answer, "citations": cited, "cited_items": [labels[c] for c in cited if c in labels],
            "insufficient": obj.insufficient_context, "valid": True, "attempts": attempts,
            "latency_s": time.perf_counter() - t0, "sources": labels}


# ---- pipeline.py: shared helper code (canonical copy in week6_multimodal_rag/tools/shared/) ----
# The whole Week 6 pipeline behind one class, cached on Drive:
#     corpus -> parse -> caption figures -> items -> index        (built once, ~5 min on a T4)
#     query (+ image) -> retrieve -> grounded answer with citations
# Week 7's agent and Week 8's RAG API load the same folder: My Drive/cv_course_artifacts/week6_rag/
import json
import time

from PIL import Image


class MultimodalRAG:
    def __init__(self, work_dir: Path, llm: "LLM", text_emb: "TextEmbedder", clip: "ClipEmbedder"):
        self.work_dir, self.llm, self.text_emb, self.clip = work_dir, llm, text_emb, clip
        self.index: MultimodalIndex | None = None
        self.elements: list = []
        self.docs: list[dict] = []
        self.qa: list[dict] = []

    @property
    def ready(self) -> bool:
        return (self.work_dir / "index" / "items.json").exists() and (self.work_dir / "elements.jsonl").exists()

    def build(self, seed: int = 7, real_papers: bool = True, detector: "TableDetector | None" = None,
              reuse_elements: bool = True) -> "MultimodalRAG":
        """Corpus -> elements -> captions -> items -> index. Reuses parsed/captioned elements saved by an earlier day."""
        t0 = time.perf_counter()
        self.docs, _, self.qa = build_corpus(self.work_dir, seed=seed, real_papers=real_papers)
        cached = self.work_dir / "elements.jsonl"
        old = [Element(**json.loads(line)) for line in open(cached, encoding="utf-8")] if reuse_elements and cached.exists() else []
        have = {e.doc_id for e in old}
        self.elements = [e for e in old if e.doc_id in {d["doc_id"] for d in self.docs}]
        for d in self.docs:
            if d["doc_id"] not in have:
                self.elements += parse_pdf(d["path"], d["doc_id"], self.work_dir / "crops", detector=detector)
        titles = doc_titles(self.elements, self.docs)
        caption_figures(self.elements, self.llm, titles)  # only figures without a caption yet
        self.index = MultimodalIndex(self.text_emb, self.clip).build(make_items(self.elements, titles))
        self.save()
        print(f"built in {time.perf_counter() - t0:.0f}s: {len(self.docs)} docs, {len(self.elements)} elements, "
              f"{len(self.index.items)} items -> {self.work_dir}")
        return self

    def save(self) -> None:
        with open(self.work_dir / "elements.jsonl", "w", encoding="utf-8") as fh:
            for e in self.elements:
                fh.write(json.dumps(e.to_dict()) + "\n")
        self.index.save(self.work_dir / "index")

    def load(self) -> "MultimodalRAG":
        corpus = json.loads((self.work_dir / "corpus.json").read_text(encoding="utf-8"))
        self.docs, self.qa = corpus["docs"], corpus["qa"]
        self.elements = [Element(**json.loads(line)) for line in open(self.work_dir / "elements.jsonl", encoding="utf-8")]
        self.index = MultimodalIndex.load(self.work_dir / "index", self.text_emb, self.clip)
        print(f"loaded {len(self.docs)} docs, {len(self.elements)} elements, {len(self.index.items)} items from {self.work_dir}")
        return self

    def build_or_load(self, rebuild: bool = False, **kw) -> "MultimodalRAG":
        return self.load() if self.ready and not rebuild else self.build(**kw)

    def ask(self, question: str, image=None, k: int = 5, mode: str = "hybrid", reranker=None, use_images: bool = True) -> dict:
        t0 = time.perf_counter()
        img = Image.open(image) if isinstance(image, (str, Path)) else image
        hits = retrieve(self.index, question, img, k=k, mode=mode, reranker=reranker)
        out = generate_answer(self.llm, question, hits, query_image=img, use_images=use_images)
        out.update({"hits": hits, "total_latency_s": time.perf_counter() - t0})
        return out


# ---- evaluate.py: shared helper code (canonical copy in week6_multimodal_rag/tools/shared/) ----
# Scoring against the corpus ground truth (Weeks 6-7).
#     source_recall : fraction of expected (doc, page, type) sources found in the retrieved hits
#     answer_correct: numbers within 1 %, names/labels as case-insensitive substrings, abstention for unanswerable
#     citation_ok   : at least one cited item is an expected source
#     grounded_nums : every number in the answer appears in the cited sources (a hallucination check)
import re

ABSTAIN = re.compile(r"(don't know|do not know|not (?:found|mentioned|available|provided|stated|specified|in the (?:documents|sources|context))"
                     r"|insufficient|cannot (?:find|determine|answer)|no information|unknown)", re.I)


def norm(s) -> str:
    return re.sub(r"[^a-z0-9%.]+", " ", str(s).lower()).strip()


def numbers_in(s: str) -> list[float]:
    out = []
    for m in re.findall(r"-?\d[\d,]*(?:\.\d+)?", str(s)):
        try:
            out.append(float(m.replace(",", "")))
        except ValueError:
            pass
    return out


def is_abstention(pred: str) -> bool:
    return not str(pred).strip() or bool(ABSTAIN.search(str(pred)))


def answer_correct(pred: str, qa: dict) -> bool:
    if qa["answer"] is None:
        return is_abstention(pred)
    if qa.get("number") is not None:
        target = float(qa["number"])
        return any(abs(x - target) <= max(0.01 * abs(target), 0.05) for x in numbers_in(pred))
    golds = qa["answer"] if isinstance(qa["answer"], list) else [qa["answer"]]
    return all(norm(g) in norm(pred) for g in golds)


def hit_matches(hit: dict, source: list) -> bool:
    doc, page, typ = source
    return hit["doc_id"] == doc and page in hit.get("pages", [hit["page"]]) and hit["type"] == typ


def source_recall(hits: list[dict], qa: dict) -> float:
    if not qa["sources"]:
        return float("nan")
    return sum(any(hit_matches(h, s) for h in hits) for s in qa["sources"]) / len(qa["sources"])


def citation_ok(cited_items: list[dict], qa: dict) -> bool:
    if not qa["sources"]:
        return not cited_items
    return any(hit_matches(h, s) for h in cited_items for s in qa["sources"])


def grounded_numbers(answer: str, cited_items: list[dict]) -> bool:
    """Numbers in the answer must appear in the cited text (figure values may only be in the image: then not checkable)."""
    nums = [n for n in numbers_in(answer) if not (1900 <= n <= 2100)]  # years are context, not claims
    if not nums:
        return True
    pool = numbers_in(" ".join(h["text"] for h in cited_items))
    return all(any(abs(n - p) <= max(0.01 * abs(n), 0.05) for p in pool) for n in nums)


# ---- agent.py: shared helper code (canonical copy in week7_agentic_rag/tools/shared/) ----
# Agentic multimodal RAG on top of the Week 6 pipeline (Week 7, reused by the Week 8 RAG API).
#
#   Toolbox      parse_document, search_metadata, retrieve_multimodal, caption_image, answer_vqa  (every call logged)
#   react_agent  a hand-written ReAct loop: the LLM picks one tool per turn as JSON (Day 1)
#   Agent        a LangGraph state machine (Days 2-5):
#
#       START -> route --direct--> retrieve_direct ------------------------------> synthesize -> verify -> END
#                    \--agentic--> plan -> execute <-> rewrite (self-correction)      ^   \--retry--/
#                                              \-- all steps done --------------------/
import difflib
import json
import re
import time
from typing import Literal, TypedDict

from langgraph.graph import END, START, StateGraph
from PIL import Image
from pydantic import BaseModel, Field

TYPES = {"text": ["text"], "table": ["table"], "figure": ["figure"], "image": None}


def compact(h: dict, chars: int = 220) -> dict:
    """What a tool shows the LLM about a hit (full records stay in the toolbox registry)."""
    return {"id": h["id"], "type": h["type"], "doc": h["doc_title"], "page": h["page"], "text": h["text"][:chars]}


class Toolbox:
    """The agent's tools over a loaded MultimodalRAG. Every call is appended to self.log with inputs, output and time."""

    def __init__(self, rag: "MultimodalRAG", query_image: Image.Image | None = None):
        self.rag, self.query_image = rag, query_image
        self.items = {it["id"]: it for it in rag.index.items}
        self.seen: dict[str, dict] = {}  # every item a tool returned: the agent may cite only these
        self.log: list[dict] = []

    def spec(self) -> str:
        docs = "; ".join(f"{d['doc_id']} = {d['title']}" for d in self.rag.docs)
        return (
            "TOOLS\n"
            "- search_metadata(query: str, doc_id: str|null, type: 'text'|'table'|'figure'|null): list documents whose title matches, "
            "or the elements of one document (ids, types, pages). Cheap; use it to find the right document.\n"
            "- parse_document(doc_id: str): structure of one document: pages, sections, tables and figures with their captions.\n"
            "- retrieve_multimodal(query: str, k: int=5, types: list|null, doc_ids: list|null, use_query_image: bool=false): "
            "semantic search over text chunks, tables and figure captions (+ CLIP over figure images). Returns item ids and snippets.\n"
            "- caption_image(item_id: str): detailed description of a figure item, or of the user's image if item_id is 'query_image'.\n"
            "- answer_vqa(question: str, item_ids: list[str], include_query_image: bool=false): ask the vision model a question "
            "while it looks at those figures (and the user's image). Use it to read values from charts.\n"
            f"DOCUMENTS: {docs}")

    def call(self, name: str, args: dict) -> object:
        t0 = time.perf_counter()
        fn = {"search_metadata": self.search_metadata, "parse_document": self.parse_document, "retrieve_multimodal": self.retrieve_multimodal,
              "caption_image": self.caption_image, "answer_vqa": self.answer_vqa}.get(name)
        try:
            out = fn(**args) if fn else {"error": f"unknown tool {name!r}"}
        except TypeError as exc:
            out = {"error": f"bad arguments for {name}: {exc}"}
        self.log.append({"tool": name, "args": {k: v for k, v in args.items() if k != "image"}, "seconds": round(time.perf_counter() - t0, 2),
                         "output": json.dumps(out, default=str)[:600]})
        return out

    def search_metadata(self, query: str = "", doc_id: str | None = None, type: str | None = None) -> list[dict]:
        if doc_id:
            return [{"id": it["id"], "type": it["type"], "page": it["page"], "text": it["text"][:100]}
                    for it in self.items.values() if it["doc_id"] == doc_id and (type is None or it["type"] == type)][:20]
        words = set(re.findall(r"[a-z0-9]+", query.lower()))
        scored = [(len(words & set(re.findall(r"[a-z0-9]+", (d["title"] + " " + d["doc_id"]).lower()))), d) for d in self.rag.docs]
        return [{"doc_id": d["doc_id"], "title": d["title"], "kind": d["kind"]} for s, d in sorted(scored, key=lambda x: -x[0]) if s > 0][:5] \
            or [{"doc_id": d["doc_id"], "title": d["title"]} for d in self.rag.docs]

    def parse_document(self, doc_id: str) -> dict:
        els = [e for e in self.rag.elements if e.doc_id == doc_id]
        if not els:
            return {"error": f"no document {doc_id!r}"}
        return {"doc_id": doc_id, "pages": max(e.page for e in els),
                "sections": list(dict.fromkeys(e.text for e in els if e.type == "title"))[:12],
                "tables": [{"page": e.page, "first_row": (e.rows or [[""]])[0]} for e in els if e.type == "table"],
                "figures": [{"page": e.page, "printed_caption": e.text[:120]} for e in els if e.type == "figure"]}

    def retrieve_multimodal(self, query: str, k: int = 5, types: list | None = None, doc_ids: list | None = None,
                            use_query_image: bool = False, mode: str = "hybrid") -> list[dict]:
        img = self.query_image if use_query_image else None
        hits = retrieve(self.rag.index, query, img, k=k, mode=mode, types=types, doc_ids=doc_ids)
        for h in hits:
            self.seen[h["id"]] = h
        return [compact(h) for h in hits]

    def caption_image(self, item_id: str) -> str:
        if item_id == "query_image" and self.query_image is not None:
            return self.rag.llm.ask("Describe this chart: type, title, categories and every value you can read.", [self.query_image],
                                    max_new_tokens=200)
        it = self.items.get(item_id)
        if not it or not it.get("image_path"):
            return f"{item_id} is not a figure"
        self.seen[item_id] = {**it, "score": 0.0, "space": "caption_image"}
        return it["text"][-600:]

    def answer_vqa(self, question: str, item_ids: list, include_query_image: bool = False) -> str:
        imgs, names = [], []
        if include_query_image and self.query_image is not None:
            imgs.append(self.query_image)
            names.append("image 1 is the USER'S image")
        for iid in item_ids[:3]:
            it = self.items.get(iid)
            if it and it.get("image_path"):
                imgs.append(Image.open(it["image_path"]))
                names.append(f"image {len(imgs)} is {iid} ({it['doc_title']}, page {it['page']})")
                self.seen[iid] = {**it, "score": 0.0, "space": "answer_vqa"}
        if not imgs:
            return "no figure images among those item ids"
        return self.rag.llm.ask(f"{'; '.join(names)}.\n{question}\nRead values exactly from the images. Answer in 1-3 sentences.",
                                imgs, max_new_tokens=160)


# ---------------------------------------------------------------------------------------------
# Day 1: a hand-written ReAct loop
# ---------------------------------------------------------------------------------------------
class ReActStep(BaseModel):
    thought: str = Field(description="one or two sentences of reasoning")
    action: Literal["search_metadata", "parse_document", "retrieve_multimodal", "caption_image", "answer_vqa", "final_answer"]
    args: dict = Field(default_factory=dict, description="tool arguments; for final_answer: {answer: str, citations: [item ids]}")


REACT_SYSTEM = ("You answer questions about a document collection by calling tools, one per turn (ReAct: think, act, observe). "
                "Use retrieve_multimodal to find evidence and answer_vqa to read values from figures. Cite item ids you actually saw. "
                "When you have the answer, use action final_answer with args {\"answer\": ..., \"citations\": [item ids]}. "
                "If the documents do not contain the answer, give final_answer \"I don't know based on the documents.\"")


def react_agent(question: str, rag: "MultimodalRAG", image=None, max_steps: int = 6, verbose: bool = True) -> dict:
    """Plain loop, no framework: prompt -> JSON action -> run tool -> observation -> repeat (capped at max_steps)."""
    t0, calls0 = time.perf_counter(), rag.llm.calls
    tb = Toolbox(rag, image)
    user = [{"type": "text", "text": f"{tb.spec()}\n\nQUESTION: {question}"}]
    if image is not None:
        user = [{"type": "text", "text": "The user attached this image (refer to it as 'query_image'):"}, {"type": "image", "image": image}] + user
    msgs = [{"role": "system", "content": REACT_SYSTEM}, {"role": "user", "content": user}]
    trace, final = [], None
    for step in range(1, max_steps + 1):
        obj, raw, _ = ask_json(rag.llm, msgs, ReActStep, max_new_tokens=300)
        if obj is None:
            trace.append({"step": step, "error": "invalid action JSON", "raw": raw[:200]})
            msgs += [{"role": "assistant", "content": raw}, {"role": "user", "content": "Reply with one valid JSON action."}]
            continue
        if verbose:
            print(f"  [{step}] {obj.thought[:110]}\n      -> {obj.action}({json.dumps(obj.args)[:120]})")
        if obj.action == "final_answer":
            final = obj.args
            trace.append({"step": step, "thought": obj.thought, "action": "final_answer", "args": obj.args})
            break
        args = dict(obj.args)
        if obj.action == "retrieve_multimodal":
            args.setdefault("query", question)
        obs = tb.call(obj.action, args)
        trace.append({"step": step, "thought": obj.thought, "action": obj.action, "args": obj.args, "observation": json.dumps(obs, default=str)[:400]})
        msgs += [{"role": "assistant", "content": obj.model_dump_json()},
                 {"role": "user", "content": f"Observation: {json.dumps(obs, default=str)[:1800]}\nNext action (JSON)."}]
    if final is None:  # step budget exhausted: answer from whatever was seen (a guardrail, not a crash)
        out = generate_answer(rag.llm, question, list(tb.seen.values())[:6], query_image=image)
        final = {"answer": out["answer"], "citations": [h["id"] for h in out["cited_items"]], "forced": True}
    cited = [tb.seen.get(c) or tb.items.get(c) for c in final.get("citations", []) or []]
    return {"answer": str(final.get("answer", "")), "cited_items": [c for c in cited if c], "trace": trace, "tool_log": tb.log,
            "steps": len(trace), "tool_calls": len(tb.log), "llm_calls": rag.llm.calls - calls0,
            "seconds": time.perf_counter() - t0, "forced": final.get("forced", False)}


# ---------------------------------------------------------------------------------------------
# Days 2-5: a LangGraph agent with routing, planning, self-correction and verification
# ---------------------------------------------------------------------------------------------
class RouteDecision(BaseModel):
    route: Literal["direct", "agentic"]
    reason: str = Field(max_length=300)


class PlanStep(BaseModel):
    subquestion: str
    modality: Literal["text", "table", "figure", "image"] = Field(description="where the answer lives; 'image' = match the user's image")
    document: str | None = Field(default=None, description="company or paper name if the step is about one document")


class Plan(BaseModel):
    steps: list[PlanStep] = Field(min_length=1, max_length=4)


class Grade(BaseModel):
    sufficient: bool = Field(description="does the evidence contain the answer to the sub-question?")
    answer: str | None = Field(description="the answer to the sub-question if sufficient, copied exactly")
    missing: str | None = Field(description="what is missing if not sufficient")


class Rewrite(BaseModel):
    query: str
    modality: Literal["text", "table", "figure", "image"]
    document: str | None = None


MULTI_HOP = re.compile(r"\b(compare|versus|vs\.?|higher|lower|more|fewer|both|and who|among|difference|differ|like this|same data)\b", re.I)


def heuristic_route(question: str, has_image: bool) -> str:
    return "agentic" if has_image or MULTI_HOP.search(question) else "direct"


class AgentState(TypedDict, total=False):
    question: str
    image: object
    route: str
    plan: list
    current: int
    attempts: int
    need_retry: bool
    missing: str
    evidence: list
    findings: list
    answer: str
    citations: list
    cited_items: list
    insufficient: bool
    verified: bool
    feedback: str
    synth_rounds: int
    trace: list


class Agent:
    def __init__(self, rag: "MultimodalRAG", k: int = 4, max_retries: int = 2, self_correct: bool = True, verify: bool = True,
                 llm_router: bool = True, max_tool_calls: int = 14, verbose: bool = True, force_route: str | None = None,
                 plan_override=None):
        self.rag, self.k, self.max_retries, self.self_correct, self.do_verify = rag, k, max_retries, self_correct, verify
        self.llm_router, self.max_tool_calls, self.verbose = llm_router, max_tool_calls, verbose
        self.force_route = force_route      # "direct" / "agentic": skip the router (ablations)
        self.plan_override = plan_override  # fn(steps, question) -> steps: inject a bad plan to test self-correction
        self.tb: Toolbox | None = None
        g = StateGraph(AgentState)
        for name in ("route", "retrieve_direct", "plan", "execute", "rewrite", "synthesize", "verify"):
            g.add_node(name, getattr(self, f"_{name}"))
        g.add_edge(START, "route")
        g.add_conditional_edges("route", lambda s: s["route"], {"direct": "retrieve_direct", "agentic": "plan"})
        g.add_edge("retrieve_direct", "synthesize")
        g.add_edge("plan", "execute")
        g.add_conditional_edges("execute", self._after_execute, {"rewrite": "rewrite", "execute": "execute", "synthesize": "synthesize"})
        g.add_edge("rewrite", "execute")
        g.add_edge("synthesize", "verify")
        g.add_conditional_edges("verify", lambda s: "done" if s.get("verified", True) or s.get("synth_rounds", 0) >= 2 else "retry",
                                {"done": END, "retry": "synthesize"})
        self.graph = g.compile()

    # -- helpers ------------------------------------------------------------------------------
    def _log(self, s: AgentState, node: str, **info) -> list:
        if self.verbose:
            print(f"  [{node}] " + " | ".join(f"{k}={str(v)[:110]}" for k, v in info.items()))
        return s.get("trace", []) + [{"node": node, **info}]

    def resolve_doc(self, name: str | None) -> str | None:
        if not name:
            return None
        name = name.lower()
        best, best_r = None, 0.0
        for d in self.rag.docs:
            title = d["title"].lower()
            r = 1.0 if (name in title or d["doc_id"].replace("_", " ") in name) else difflib.SequenceMatcher(None, name, title).ratio()
            if r > best_r:
                best, best_r = d["doc_id"], r
        return best if best_r >= 0.5 else None

    def _budget_left(self) -> bool:
        return len(self.tb.log) < self.max_tool_calls

    # -- nodes ----------------------------------------------------------------------------------
    def _route(self, s: AgentState) -> dict:
        has_img = s.get("image") is not None
        route, reason = heuristic_route(s["question"], has_img), "heuristic"
        if self.force_route:
            route, reason = self.force_route, "forced"
        elif self.llm_router and not has_img:
            obj, _, _ = ask_json(self.rag.llm, [{"role": "user", "content":
                "Decide how to answer this question over a collection of company annual reports and research papers.\n"
                "'direct' = one search finds the answer (a single fact about one document).\n"
                "'agentic' = needs several searches or steps (comparing documents, combining facts, finding an entity first).\n"
                f"QUESTION: {s['question']}"}], RouteDecision, max_new_tokens=120)
            if obj is not None:
                route, reason = obj.route, obj.reason
        return {"route": route, "trace": self._log(s, "route", route=route, reason=reason), "evidence": [], "findings": [],
                "synth_rounds": 0, "feedback": ""}

    def _retrieve_direct(self, s: AgentState) -> dict:
        hits = retrieve(self.rag.index, s["question"], s.get("image"), k=5)
        self.tb.log.append({"tool": "retrieve_multimodal", "args": {"query": s["question"]}, "seconds": 0, "output": str([h["id"] for h in hits])})
        return {"evidence": hits, "trace": self._log(s, "retrieve_direct", hits=[h["id"] for h in hits])}

    def _plan(self, s: AgentState) -> dict:
        img_note = "The user attached an image; use modality 'image' for the step that must match it against the documents.\n" \
            if s.get("image") is not None else ""
        docs = "; ".join(d["title"] for d in self.rag.docs)
        obj, _, _ = ask_json(self.rag.llm, [{"role": "user", "content":
            f"Break the question into 1-4 simple sub-questions, each answerable from ONE place in ONE document.\n"
            f"For each, say where the answer lives: 'text' (prose), 'table' (numbers by quarter), 'figure' (charts: regions, "
            f"segments, headcount) or 'image'. Name the document when known.\n{img_note}DOCUMENTS: {docs}\nQUESTION: {s['question']}"}],
            Plan, max_new_tokens=400)
        steps = [st.model_dump() for st in obj.steps] if obj else [{"subquestion": s["question"], "modality": "image" if s.get("image") is not None else "text", "document": None}]
        for st in steps:
            st["query"] = st["subquestion"]
        if self.plan_override is not None:
            steps = self.plan_override(steps, s["question"])
        return {"plan": steps, "current": 0, "attempts": 0, "need_retry": False,
                "trace": self._log(s, "plan", steps=[(st["subquestion"], st["modality"], st["document"]) for st in steps])}

    def _grade(self, subq: str, hits: list[dict], note: str | None) -> Grade:
        ev = "\n".join(f"[{h['id']}] ({h['type']}, {h['doc_title']}, p{h['page']}) {h['text'][:500]}" for h in hits)
        if note:
            ev += f"\n[vision model reading the figures] {note}"
        obj, _, _ = ask_json(self.rag.llm, [{"role": "user", "content":
            f"SUB-QUESTION: {subq}\nEVIDENCE:\n{ev}\nDoes the evidence contain the answer? Be strict: the right document and the right "
            "value must both be there."}], Grade, max_new_tokens=200)
        return obj or Grade(sufficient=bool(hits), answer=note, missing=None)

    def _execute(self, s: AgentState) -> dict:
        i, plan = s["current"], [dict(p) for p in s["plan"]]
        step = plan[i]
        doc = self.resolve_doc(step.get("document"))
        if step["modality"] == "image" and s.get("image") is not None:
            raw = self.tb.call("retrieve_multimodal", {"query": step["query"], "k": self.k, "use_query_image": True, "mode": "image"})
        else:
            raw = self.tb.call("retrieve_multimodal", {"query": step["query"], "k": self.k, "types": TYPES[step["modality"]],
                                                       "doc_ids": [doc] if doc else None})
        hits = [self.tb.seen[h["id"]] for h in raw if isinstance(h, dict) and h.get("id") in self.tb.seen]
        note = None
        figs = [h["id"] for h in hits if h.get("image_path")][:2]
        if figs and step["modality"] in ("figure", "image") and self._budget_left():
            note = self.tb.call("answer_vqa", {"question": step["subquestion"], "item_ids": figs,
                                               "include_query_image": step["modality"] == "image"})
        grade = self._grade(step["subquestion"], hits, note) if self.self_correct else Grade(sufficient=True, answer=note, missing=None)
        attempts = s.get("attempts", 0)
        trace = self._log(s, "execute", step=i, query=step["query"], modality=step["modality"], doc=doc,
                          hits=[h["id"] for h in hits], vqa=note, sufficient=grade.sufficient, answer=grade.answer)
        if not grade.sufficient and attempts < self.max_retries and self._budget_left():
            return {"need_retry": True, "missing": grade.missing or "", "trace": trace, "plan": plan}
        evidence = s.get("evidence", []) + [h for h in hits if h["id"] not in {e["id"] for e in s.get("evidence", [])}]
        finding = {"step": i, "subquestion": step["subquestion"], "answer": grade.answer if grade.sufficient else None,
                   "vqa": note, "sources": [h["id"] for h in hits], "attempts": attempts}
        return {"need_retry": False, "evidence": evidence, "findings": s.get("findings", []) + [finding], "current": i + 1,
                "attempts": 0, "trace": trace, "plan": plan}

    def _after_execute(self, s: AgentState) -> str:
        if s.get("need_retry"):
            return "rewrite"
        return "execute" if s["current"] < len(s["plan"]) and self._budget_left() else "synthesize"

    def _rewrite(self, s: AgentState) -> dict:
        plan = [dict(p) for p in s["plan"]]
        step = plan[s["current"]]
        obj, _, _ = ask_json(self.rag.llm, [{"role": "user", "content":
            f"A search did not find the answer.\nSUB-QUESTION: {step['subquestion']}\nLAST QUERY: {step['query']} "
            f"(modality {step['modality']}, document {step.get('document')})\nMISSING: {s.get('missing')}\n"
            "Write a better search query. Switch modality if the fact is more likely in a table (quarterly numbers) or a figure "
            "(regions, segment shares, headcount), and name the document if you can.\n"
            f"DOCUMENTS: {'; '.join(d['title'] for d in self.rag.docs)}"}], Rewrite, max_new_tokens=150)
        order = ["text", "table", "figure"]
        if obj is None or (obj.query == step["query"] and obj.modality == step["modality"]):
            nxt = order[(order.index(step["modality"]) + 1) % 3] if step["modality"] in order else "text"
            obj = Rewrite(query=step["subquestion"], modality=nxt, document=step.get("document"))  # deterministic fallback
        step.update(query=obj.query, modality=obj.modality, document=obj.document or step.get("document"))
        return {"plan": plan, "attempts": s.get("attempts", 0) + 1, "need_retry": False,
                "trace": self._log(s, "rewrite", query=obj.query, modality=obj.modality, document=step["document"])}

    def _synthesize(self, s: AgentState) -> dict:
        ev = s.get("evidence", [])[:8]
        findings = "\n".join(f"- {f['subquestion']} -> {f['answer'] or f['vqa'] or 'not found'} (from {', '.join(f['sources'][:3])})"
                             for f in s.get("findings", []))
        question = s["question"]
        if findings:
            question += f"\n\nFINDINGS FROM EARLIER STEPS (each from the listed source ids):\n{findings}"
        if s.get("feedback"):
            question += f"\n\nA CHECK FAILED ON YOUR PREVIOUS ANSWER: {s['feedback']} Fix it."
        out = generate_answer(self.rag.llm, question, ev, query_image=s.get("image"))
        return {"answer": out["answer"], "citations": out["citations"], "cited_items": out["cited_items"], "insufficient": out["insufficient"],
                "synth_rounds": s.get("synth_rounds", 0) + 1,
                "trace": self._log(s, "synthesize", answer=out["answer"], citations=out["citations"])}

    def _verify(self, s: AgentState) -> dict:
        if not self.do_verify:
            return {"verified": True}
        problems = []
        found = [f for f in s.get("findings", []) if f["answer"]]
        if is_abstention(s["answer"]) and found:
            problems.append(f"you abstained but the findings contain answers: {[f['answer'] for f in found]}")
        pool = " ".join(h["text"] for h in s.get("evidence", [])) + " " + " ".join(str(f["answer"] or "") + " " + str(f["vqa"] or "")
                                                                                  for f in s.get("findings", []))
        nums = [n for n in numbers_in(s["answer"]) if not 1900 <= n <= 2100]
        unsupported = [n for n in nums if not any(abs(n - p) <= max(0.01 * abs(n), 0.05) for p in numbers_in(pool))]
        if unsupported:
            problems.append(f"numbers {unsupported} do not appear in the evidence")
        verified = not problems
        return {"verified": verified, "feedback": "; ".join(problems),
                "trace": self._log(s, "verify", verified=verified, problems=problems)}

    # -- entry point ------------------------------------------------------------------------------
    def run(self, question: str, image=None) -> dict:
        if isinstance(image, (str, Path)):
            image = Image.open(image)
        t0, calls0 = time.perf_counter(), self.rag.llm.calls
        self.tb = Toolbox(self.rag, image)
        if self.verbose:
            print(f"Q: {question}")
        s = self.graph.invoke({"question": question, "image": image, "trace": []}, config={"recursion_limit": 40})
        retries = sum(1 for t in s.get("trace", []) if t["node"] == "rewrite")
        return {"answer": s.get("answer", ""), "cited_items": s.get("cited_items", []), "route": s.get("route"), "plan": s.get("plan", []),
                "findings": s.get("findings", []), "evidence": s.get("evidence", []), "verified": s.get("verified"),
                "retries": retries, "synth_rounds": s.get("synth_rounds", 0), "trace": s.get("trace", []), "tool_log": self.tb.log,
                "tool_calls": len(self.tb.log), "llm_calls": self.rag.llm.calls - calls0, "seconds": time.perf_counter() - t0}
