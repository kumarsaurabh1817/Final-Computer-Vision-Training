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
