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
