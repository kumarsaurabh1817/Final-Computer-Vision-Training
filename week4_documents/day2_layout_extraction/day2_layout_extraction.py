"""
Week 4 · Day 2 — Layout-aware extraction: LayoutLMv3 (OCR + layout) and Donut (OCR-free) to JSON.

Colab-first script (percent-format cells). Build the notebook with:
    python ../tools/py_to_ipynb.py day2_layout_extraction/day2_layout_extraction.py
Datasets: FUNSD (https://guillaumejaume.github.io/FUNSD/), CORD-v2 (https://huggingface.co/datasets/naver-clova-ix/cord-v2)
"""

# %% [markdown]
# # Week 4 · Day 2 — Layout-Aware Extraction (LayoutLMv3 & Donut)
#
# **Goal:** go from "a bag of OCR words" to **structured key-value JSON**, and see for yourself why the *position* of words matters as much as the words.
#
# | Topic | What you will build |
# |---|---|
# | LayoutLMv3 | Fine-tune `microsoft/layoutlmv3-base` for token classification (HEADER / QUESTION / ANSWER) on FUNSD forms |
# | Why layout matters | Same task with a text-only `roberta-base` (no boxes, no image) + LayoutLMv3 with its boxes zeroed at inference |
# | Key-value linking | Turn predicted QUESTION / ANSWER entities into `{"key": "value"}` JSON and score the pairs against FUNSD's ground-truth links |
# | Donut | OCR-free image → JSON with `naver-clova-ix/donut-base-finetuned-cord-v2` on CORD receipts; field-level F1 |
# | Domain gap | Both models on this week's synthetic invoices |
#
# **Data:** [FUNSD](https://guillaumejaume.github.io/FUNSD/) (149 train / 50 test scanned forms, entity labels + question→answer links) and [CORD-v2](https://huggingface.co/datasets/naver-clova-ix/cord-v2) (Indonesian receipts with JSON ground truth; test split = 100).
#
# > Runtime: **T4 GPU** required for fine-tuning (~10 min with `FULL_RUN=True`). Licences: LayoutLMv3 weights are CC BY-NC-SA 4.0 (non-commercial); Donut is MIT.

# %% [markdown]
# ## 0 · Install dependencies

# %%
import shutil
import subprocess
import sys


def pip_install(*packages: str) -> None:
    """Install packages into the running interpreter (works in Colab and plain Python)."""
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *packages])


if shutil.which("tesseract") is None and shutil.which("apt-get"):
    subprocess.run("apt-get -qq update && apt-get -qq install -y tesseract-ocr > /dev/null", shell=True, check=True)
pip_install("transformers>=4.46,<5", "datasets>=3.0", "seqeval==1.2.2", "sentencepiece", "pytesseract==0.3.13",
            "rapidfuzz>=3.9")

# %% [markdown]
# ## 1 · Imports, configuration, and output folder

# %%
import json
import random
import re
import time
import urllib.request
import zipfile
from collections import Counter
from pathlib import Path

import matplotlib

IN_NOTEBOOK = "ipykernel" in sys.modules
if not IN_NOTEBOOK:
    matplotlib.use("Agg")  # headless when run as a script

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytesseract
import torch
from PIL import Image
from seqeval.metrics import classification_report
from seqeval.metrics import f1_score as seq_f1
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    DonutProcessor,
    LayoutLMv3ForTokenClassification,
    LayoutLMv3Processor,
    VisionEncoderDecoderModel,
    get_linear_schedule_with_warmup,
)

IN_COLAB = "google.colab" in sys.modules
try:
    HERE = Path(__file__).resolve().parent
except NameError:  # running inside a notebook
    HERE = Path.cwd()

OUTPUT_DIR = Path("/content/week4_outputs/day2") if IN_COLAB else HERE.parent / "results" / "day2"
DATA_DIR = Path("/content/datasets") if IN_COLAB else HERE.parent / "data"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)

FULL_RUN = True  # False = quick smoke test (few steps, few receipts)
TRAIN_STEPS = 800 if FULL_RUN else 40
BATCH_SIZE = 4
LR = 2e-5
CHUNK_WORDS = 150  # FUNSD forms can exceed 512 tokens; train/predict on windows of words
N_CORD = 100 if FULL_RUN else 8
TRAIN_TEXT_BASELINE = True
SAVE_MODEL = False  # True writes ~500 MB of LayoutLMv3 weights to OUTPUT_DIR
SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_AMP = DEVICE.type == "cuda"
print(f"PyTorch {torch.__version__} | device={DEVICE} | FULL_RUN={FULL_RUN}")
print("Outputs ->", OUTPUT_DIR)


def save_fig(fig, name: str) -> None:
    """Save a figure to OUTPUT_DIR and show it inline when in a notebook."""
    path = OUTPUT_DIR / name
    fig.savefig(path, dpi=120, bbox_inches="tight")
    if IN_NOTEBOOK:
        plt.show()
    plt.close(fig)
    print(f"saved -> {path}")


# %%
# ---- invoice_synth.py: helper code for this day (self-contained, no external project files) ----
# Synthetic invoice generator with exact ground truth (Week 4, shared by Days 1–5).
#
# Real labeled invoices are scarce and often confidential, so every day this week also
# renders its own: the same generator + seed always gives the same documents, and the
# ground truth is already in the Week 4 common JSON schema (InvoiceDoc).
#
# Layouts (each one is a "hard case" from the Day 4 plan):
#     standard   ruled table, "Label: value" header block, USD
#     layout_b   different labels ("Bill No.", "Amount Due"), vendor on the right, borderless table, EUR
#     merged     two-level table header ("Pricing" spans two columns) + section rows spanning the table
#     multipage  25–34 line items, table continues on page 2 with a repeated header
#     rotated    standard layout scanned crooked: 3–7° skew, noise, blur, JPEG artifacts
#     layout_c   held out for Day 5 evaluation: unseen labels ("Document No:", "Balance:"), GBP
import io
import json
import random
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

PAGE_W, PAGE_H = 1240, 1754  # A4 at 150 dpi
MARGIN = 90
LAYOUTS = ("standard", "layout_b", "merged", "multipage", "rotated")

_FONT_FILES = {
    False: ["DejaVuSans.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "arial.ttf",
            "C:/Windows/Fonts/arial.ttf", "/Library/Fonts/Arial.ttf"],
    True: ["DejaVuSans-Bold.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", "arialbd.ttf",
           "C:/Windows/Fonts/arialbd.ttf", "/Library/Fonts/Arial Bold.ttf"],
}
_FONTS: dict = {}


def get_font(size: int, bold: bool = False):
    key = (size, bold)
    if key not in _FONTS:
        for path in _FONT_FILES[bold]:
            try:
                _FONTS[key] = ImageFont.truetype(path, size)
                break
            except OSError:
                continue
        else:
            _FONTS[key] = ImageFont.load_default(size=size)
    return _FONTS[key]


VENDORS = [
    ("Northwind Traders", "120 Harbor Way", "Seattle, WA 98101"),
    ("Contoso Supplies Ltd", "48 Kingsway Road", "Manchester M19 2LP"),
    ("Fabrikam Industrial", "7700 Mill Creek Drive", "Austin, TX 73301"),
    ("Acme Office Goods", "15 Market Street", "Denver, CO 80202"),
    ("Blue Yonder Logistics", "301 Canal Street", "Rotterdam 3011 AA"),
    ("Tailspin Components", "92 Riverside Avenue", "Portland, OR 97205"),
    ("Litware Systems Inc", "2200 Innovation Pkwy", "San Jose, CA 95134"),
    ("Proseware Analytics", "11 Rue de Lyon", "Paris 75012"),
]
CUSTOMERS = [
    ("Woodgrove Bank", "500 Finance Plaza", "Chicago, IL 60601"),
    ("Adventure Works Cycles", "1 Trailhead Road", "Boulder, CO 80301"),
    ("Margie's Travel", "77 Ocean Drive", "Miami, FL 33139"),
    ("Coho Winery", "9 Vineyard Lane", "Napa, CA 94558"),
    ("Alpine Ski House", "3 Summit Road", "Innsbruck 6020"),
    ("Lucerne Publishing", "68 Book Street", "Boston, MA 02108"),
    ("Wide World Importers", "400 Dock Road", "Hamburg 20457"),
    ("Trey Research", "18 Lab Court", "Cambridge CB2 1TN"),
]
ITEM_POOL = {
    "Hardware": [("Laptop stand aluminium", 35, 90), ("27-inch monitor", 180, 420), ("Wireless keyboard", 25, 80),
                 ("USB-C docking station", 90, 260), ("Ergonomic office chair", 150, 480), ("Noise cancelling headset", 60, 220)],
    "Services": [("On-site installation", 80, 300), ("Annual support plan", 400, 1500), ("Network configuration", 120, 600),
                 ("Data migration per TB", 60, 220), ("Security audit", 900, 2500), ("Staff training session", 250, 800)],
    "Supplies": [("A4 copy paper box", 18, 45), ("Toner cartridge black", 40, 130), ("Whiteboard markers 12 pack", 8, 22),
                 ("Desk organiser set", 12, 40), ("Shipping labels roll", 9, 30), ("Archive boxes 10 pack", 20, 60)],
}
DATE_FORMATS = ["%B %d, %Y", "%Y-%m-%d", "%d %b %Y", "%m/%d/%Y"]

# Per-layout vocabulary. The labels deliberately differ so a rule-based extractor tuned on one layout breaks on another.
LAYOUT_STYLE = {
    "standard": dict(title="INVOICE", inv="Invoice Number:", date="Invoice Date:", due="Due Date:", bill="Bill To:",
                     cols=["Description", "Qty", "Unit Price", "Amount"], sub="Subtotal:", tax="Tax ({rate}%):",
                     total="Total:", currency=("USD", "$"), ruled=True),
    "layout_b": dict(title="Sales Invoice", inv="Bill No.", date="Date of Issue", due="Payment Due", bill="Customer",
                     cols=["Item", "Quantity", "Rate", "Line Total"], sub="Net Amount", tax="VAT ({rate}%)",
                     total="Amount Due", currency=("EUR", "€"), ruled=False),
}
LAYOUT_STYLE["merged"] = dict(LAYOUT_STYLE["standard"])
LAYOUT_STYLE["multipage"] = dict(LAYOUT_STYLE["standard"])
LAYOUT_STYLE["rotated"] = dict(LAYOUT_STYLE["standard"])
# Held out until Day 5: vocabulary none of the Day 4 rules were written for.
LAYOUT_STYLE["layout_c"] = dict(title="Tax Invoice", inv="Document No:", date="Issued:", due="Payable By:", bill="Client:",
                                cols=["Product", "Units", "Price Each", "Ext. Price"], sub="Sum before tax:",
                                tax="GST ({rate}%):", total="Balance:", currency=("GBP", "£"), ruled=True)
HELDOUT_LAYOUTS = ("layout_c",)


def fmt_money(x: float, sym: str) -> str:
    return f"{sym}{x:,.2f}"


@dataclass
class SynthDoc:
    doc_id: str
    layout: str
    pages: list            # PIL RGB images
    truth: dict            # Week 4 common schema (InvoiceDoc) as a plain dict
    lines: list = field(default_factory=list)  # every drawn text run: {"page", "text", "bbox"} (pre-degradation coords)

    def pdf_bytes(self) -> bytes:
        buf = io.BytesIO()
        self.pages[0].save(buf, format="PDF", save_all=True, append_images=self.pages[1:], resolution=150)
        return buf.getvalue()

    def png_bytes(self, page_index: int) -> bytes:
        buf = io.BytesIO()
        self.pages[page_index].save(buf, format="PNG")
        return buf.getvalue()

    def save(self, out_dir) -> Path:
        out = Path(out_dir) / self.doc_id
        out.mkdir(parents=True, exist_ok=True)
        for i, page in enumerate(self.pages, 1):
            page.save(out / f"page{i}.png")
        (out / "document.pdf").write_bytes(self.pdf_bytes())
        (out / "truth.json").write_text(json.dumps(self.truth, indent=2), encoding="utf-8")
        return out


class _Page:
    def __init__(self, page_no: int, record: list):
        self.img = Image.new("RGB", (PAGE_W, PAGE_H), "white")
        self.draw = ImageDraw.Draw(self.img)
        self.page_no = page_no
        self.record = record

    def text(self, x, y, s, size=22, bold=False, anchor="la", fill=(25, 25, 25)):
        font = get_font(size, bold)
        self.draw.text((x, y), s, font=font, fill=fill, anchor=anchor)
        bbox = self.draw.textbbox((x, y), s, font=font, anchor=anchor)
        self.record.append({"page": self.page_no, "text": s, "bbox": [int(v) for v in bbox]})
        return bbox


def _invoice_data(rng: random.Random, layout: str) -> dict:
    style = LAYOUT_STYLE[layout]
    vendor, customer = rng.choice(VENDORS), rng.choice(CUSTOMERS)
    n_items = rng.randint(25, 34) if layout == "multipage" else rng.randint(4, 8)
    sections = list(ITEM_POOL)
    if layout == "merged":
        sections = rng.sample(sections, k=rng.choice([2, 3]))
    items = []
    for _ in range(n_items):
        sec = rng.choice(sections)
        desc, lo, hi = rng.choice(ITEM_POOL[sec])
        qty = rng.randint(1, 12) if hi < 300 else rng.randint(1, 3)
        price = round(rng.uniform(lo, hi), 2)
        items.append(dict(section=sec, description=desc, quantity=qty, unit_price=price, amount=round(qty * price, 2)))
    if layout == "merged":
        items.sort(key=lambda it: sections.index(it["section"]))
    issue = date(2026, 1, 5) + timedelta(days=rng.randint(0, 280))
    due = issue + timedelta(days=rng.choice([15, 30, 45]))
    rate = 20 if layout == "layout_b" else rng.choice([5, 8, 10])
    subtotal = round(sum(it["amount"] for it in items), 2)
    tax = round(subtotal * rate / 100, 2)
    inv_no = f"B{rng.randint(100000, 999999)}" if layout == "layout_b" else f"INV-{issue.year}-{rng.randint(1000, 9999)}"
    return dict(style=style, vendor=vendor, customer=customer, items=items, issue=issue, due=due,
                date_fmt=rng.choice(DATE_FORMATS), rate=rate, subtotal=subtotal, tax=tax,
                total=round(subtotal + tax, 2), inv_no=inv_no, currency=style["currency"])


def _table_spec(data: dict, layout: str):
    """Build header rows and body rows as cell specs: {text, col, colspan, rowspan, align, bold, fill}."""
    cols = data["style"]["cols"]
    sym = data["currency"][1]
    if layout == "merged":
        header = [
            [dict(text=cols[0], col=0, rowspan=2), dict(text=cols[1], col=1, rowspan=2, align="r"),
             dict(text="Pricing", col=2, colspan=2, align="c")],
            [dict(text=cols[2], col=2, align="r"), dict(text=cols[3], col=3, align="r")],
        ]
    else:
        header = [[dict(text=c, col=i, align="l" if i == 0 else "r") for i, c in enumerate(cols)]]
    for row in header:
        for cell in row:
            cell.update(bold=True, fill=(225, 230, 240))
    body, section = [], None
    for it in data["items"]:
        if layout == "merged" and it["section"] != section:
            section = it["section"]
            body.append(([dict(text=section, col=0, colspan=4, bold=True, fill=(240, 240, 240))], None))
        body.append(([dict(text=it["description"], col=0),
                      dict(text=str(it["quantity"]), col=1, align="r"),
                      dict(text=fmt_money(it["unit_price"], sym), col=2, align="r"),
                      dict(text=fmt_money(it["amount"], sym), col=3, align="r")], it))
    return header, body


def _render(doc_id: str, layout: str, data: dict) -> SynthDoc:
    style, sym = data["style"], data["currency"][1]
    record: list = []
    pages = [_Page(1, record)]
    page = pages[0]
    right = PAGE_W - MARGIN
    dfmt = data["date_fmt"]

    # Header block ---------------------------------------------------------
    vname, vaddr1, vaddr2 = data["vendor"]
    cname, caddr1, caddr2 = data["customer"]
    if layout == "layout_b":
        page.text(right, 80, vname, 34, True, anchor="ra")
        page.text(right, 125, vaddr1, 20, anchor="ra")
        page.text(right, 152, vaddr2, 20, anchor="ra")
        page.text(MARGIN, 80, style["title"], 40, True)
        kv_rows = [(style["inv"], data["inv_no"]), (style["date"], data["issue"].strftime(dfmt)),
                   (style["due"], data["due"].strftime(dfmt))]
        for i, (lab, val) in enumerate(kv_rows):
            y = 200 + i * 36
            page.text(MARGIN, y, lab, 20, True)
            page.text(MARGIN + 260, y, val, 20)
        page.draw.rectangle([MARGIN - 12, 188, MARGIN + 560, 200 + 3 * 36], outline=(160, 160, 160))
        page.text(700, 200, style["bill"], 20, True)
        page.text(700, 236, cname, 20)
        page.text(700, 263, caddr1, 20)
        page.text(700, 290, caddr2, 20)
    else:
        page.text(MARGIN, 80, vname, 38, True)
        page.text(MARGIN, 132, vaddr1, 20)
        page.text(MARGIN, 159, vaddr2, 20)
        page.text(right, 80, style["title"], 46, True, anchor="ra")
        kv_rows = [(style["inv"], data["inv_no"]), (style["date"], data["issue"].strftime(dfmt)),
                   (style["due"], data["due"].strftime(dfmt))]
        for i, (lab, val) in enumerate(kv_rows):
            y = 170 + i * 35
            page.text(720, y, lab, 20, True)
            page.text(right, y, val, 20, anchor="ra")
        page.text(MARGIN, 300, style["bill"], 20, True)
        page.text(MARGIN, 335, cname, 20)
        page.text(MARGIN, 362, caddr1, 20)
        page.text(MARGIN, 389, caddr2, 20)

    # Table ------------------------------------------------------------------
    col_x = [MARGIN, 640, 770, 960, right]
    row_h = 40
    top_y, bottom_limit = 460, PAGE_H - 170
    header, body = _table_spec(data, layout)
    ruled = style["ruled"]
    n_cols = 4

    def draw_row(pg: _Page, y: int, cells: list):
        for cell in cells:
            x0, x1 = col_x[cell["col"]], col_x[cell["col"] + cell.get("colspan", 1)]
            y0, y1 = y, y + row_h * cell.get("rowspan", 1)
            if cell.get("fill") and ruled:
                pg.draw.rectangle([x0, y0, x1, y1], fill=cell["fill"])
            if ruled:
                pg.draw.rectangle([x0, y0, x1, y1], outline=(90, 90, 90), width=1)
            align = cell.get("align", "l")
            cy = (y0 + y1) // 2
            if align == "r":
                pg.text(x1 - 10, cy, cell["text"], 19, cell.get("bold", False), anchor="rm")
            elif align == "c":
                pg.text((x0 + x1) // 2, cy, cell["text"], 19, cell.get("bold", False), anchor="mm")
            else:
                pg.text(x0 + 10, cy, cell["text"], 19, cell.get("bold", False), anchor="lm")

    def draw_header(pg: _Page, y: int) -> int:
        for r, row in enumerate(header):
            draw_row(pg, y + r * row_h, row)
        y += row_h * len(header)
        if not ruled:
            pg.draw.line([MARGIN, y, right, y], fill=(60, 60, 60), width=2)
        return y

    y = draw_header(page, top_y)
    truth_cells = []
    for r, row in enumerate(header):
        for cell in row:
            truth_cells.append(dict(row=r, col=cell["col"], row_span=cell.get("rowspan", 1),
                                    col_span=cell.get("colspan", 1), text=cell["text"], is_header=True))
    r = len(header)
    for cells, _ in body:
        if y + row_h > bottom_limit:
            pages.append(_Page(len(pages) + 1, record))
            page = pages[-1]
            page.text(MARGIN, 80, f"{vname} - {style['title']} {data['inv_no']} (continued)", 20, True)
            y = draw_header(page, 140)
        draw_row(page, y, cells)
        for cell in cells:
            truth_cells.append(dict(row=r, col=cell["col"], row_span=1, col_span=cell.get("colspan", 1),
                                    text=cell["text"], is_header=False))
        y += row_h
        r += 1

    # Totals -------------------------------------------------------------------
    if y + 4 * 38 > PAGE_H - 120:
        pages.append(_Page(len(pages) + 1, record))
        page = pages[-1]
        y = 140
    y += 30
    rate = data["rate"]
    for lab, val, bold in [(style["sub"], data["subtotal"], False), (style["tax"].format(rate=rate), data["tax"], False),
                           (style["total"], data["total"], True)]:
        page.text(770, y, lab, 21, bold)
        page.text(right - 10, y, fmt_money(val, sym), 21, bold, anchor="ra")
        y += 38
    for p in pages:
        p.text(MARGIN, PAGE_H - 110, "Thank you for your business. Payment terms: Net 30.", 18, fill=(90, 90, 90))
        if len(pages) > 1:
            p.text(right, PAGE_H - 110, f"Page {p.page_no} of {len(pages)}", 18, anchor="ra", fill=(90, 90, 90))

    truth = dict(
        doc_id=doc_id,
        fields=dict(invoice_number=data["inv_no"], invoice_date=data["issue"].isoformat(), due_date=data["due"].isoformat(),
                    vendor_name=vname, bill_to=cname, subtotal=data["subtotal"], tax=data["tax"], total=data["total"],
                    currency=data["currency"][0]),
        line_items=[dict(description=it["description"], quantity=float(it["quantity"]), unit_price=it["unit_price"],
                         amount=it["amount"]) for _, it in body if it is not None],
        tables=[dict(page=1, n_rows=r, n_cols=n_cols, cells=truth_cells)],
        meta=dict(source="ground_truth", pages=len(pages), layout=layout),
    )
    return SynthDoc(doc_id, layout, [p.img for p in pages], truth, record)


def degrade_scan(img: Image.Image, rng: random.Random | None = None, angle: float = 0.0, noise: float = 0.0,
                 blur: float = 0.0, jpeg_quality: int | None = None, contrast: float = 1.0) -> Image.Image:
    """Make a clean render look like a cheap scan: rotate, lower contrast, blur, add Gaussian noise, JPEG-compress."""
    rng = rng or random.Random(0)
    out = img.convert("RGB")
    if angle:
        out = out.rotate(angle, resample=Image.BICUBIC, expand=False, fillcolor="white")
    if contrast != 1.0:
        arr = np.asarray(out).astype(np.float32)
        out = Image.fromarray(np.clip((arr - 255) * contrast + 255, 0, 255).astype(np.uint8))
    if blur:
        out = out.filter(ImageFilter.GaussianBlur(blur))
    if noise:
        arr = np.asarray(out).astype(np.float32)
        gen = np.random.default_rng(rng.randint(0, 2**31))
        arr += gen.normal(0, noise, arr.shape[:2])[..., None]
        out = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
    if jpeg_quality:
        buf = io.BytesIO()
        out.save(buf, format="JPEG", quality=jpeg_quality)
        out = Image.open(io.BytesIO(buf.getvalue())).convert("RGB")
    return out


def make_invoice(doc_id: str, layout: str = "standard", seed: int = 0) -> SynthDoc:
    """Render one invoice. Same (doc_id, layout, seed) -> identical pixels and truth."""
    if layout not in LAYOUT_STYLE:
        raise ValueError(f"layout must be one of {LAYOUTS + HELDOUT_LAYOUTS}")
    rng = random.Random(f"{seed}-{doc_id}-{layout}")
    doc = _render(doc_id, layout, _invoice_data(rng, layout))
    if layout == "rotated":
        angle = rng.uniform(3, 7) * rng.choice([-1, 1])
        doc.pages = [degrade_scan(p, rng, angle=angle, noise=10, blur=0.7, jpeg_quality=55, contrast=0.85) for p in doc.pages]
        doc.truth["meta"]["skew_deg"] = round(angle, 2)
    return doc


def make_invoice_set(n_per_layout: int = 1, seed: int = 0, layouts=LAYOUTS, prefix: str = "inv") -> list[SynthDoc]:
    """A small labeled corpus: n documents of each layout."""
    return [make_invoice(f"{prefix}_{layout}_{i:02d}", layout, seed) for layout in layouts for i in range(n_per_layout)]

# %% [markdown]
# ## 2 · FUNSD as entities, BIO labels, and key→value links
#
# Every FUNSD entity has a label (`header`, `question`, `answer`, `other`), its words with boxes, and `linking` pairs. A question→answer link *is* a key-value pair: `"Date:" → "12/10/98"`.
#
# For token classification each word gets a **BIO** tag: `B-QUESTION` on the first word of a question entity, `I-QUESTION` on the rest, `O` for `other`. LayoutLMv3 wants boxes on a 0–1000 grid regardless of image size.

# %%
FUNSD_URL = "https://guillaumejaume.github.io/FUNSD/dataset.zip"
LABELS = ["O", "B-HEADER", "I-HEADER", "B-QUESTION", "I-QUESTION", "B-ANSWER", "I-ANSWER"]
L2ID = {l: i for i, l in enumerate(LABELS)}


def get_funsd() -> Path:
    root = DATA_DIR / "funsd"
    if not (root / "dataset" / "testing_data").is_dir():
        zip_path = DATA_DIR / "funsd.zip"
        if not zip_path.exists():
            urllib.request.urlretrieve(FUNSD_URL, zip_path)
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(root)
    return root / "dataset"


def norm_box(box, w, h) -> list[int]:
    x0, y0, x1, y1 = box
    return [max(0, min(1000, int(1000 * v / s))) for v, s in ((x0, w), (y0, h), (x1, w), (y1, h))]


def load_funsd(split: str) -> list[dict]:
    split_dir = get_funsd() / split
    docs = []
    for ann_path in sorted((split_dir / "annotations").glob("*.json")):
        ann = json.loads(ann_path.read_text(encoding="utf-8"))
        image = Image.open(split_dir / "images" / f"{ann_path.stem}.png").convert("RGB")
        w, h = image.size
        words, boxes, labels, ent_of_word, entities, links = [], [], [], [], {}, set()
        for ent in ann["form"]:
            ws = [x for x in ent["words"] if x["text"].strip()]
            if not ws:
                continue
            tag = ent["label"].upper()
            entities[ent["id"]] = {"label": tag, "text": " ".join(x["text"] for x in ws), "box": norm_box(ent["box"], w, h)}
            for i, x in enumerate(ws):
                words.append(x["text"])
                boxes.append(norm_box(x["box"], w, h))
                labels.append("O" if tag == "OTHER" else f"{'B' if i == 0 else 'I'}-{tag}")
                ent_of_word.append(ent["id"])
            for a, b in ent["linking"]:
                links.add((a, b))
        pairs = set()
        for a, b in links:
            if a in entities and b in entities:
                la, lb = entities[a]["label"], entities[b]["label"]
                if (la, lb) == ("QUESTION", "ANSWER"):
                    pairs.add((a, b))
                elif (la, lb) == ("ANSWER", "QUESTION"):
                    pairs.add((b, a))
        docs.append({"id": ann_path.stem, "image": image, "words": words, "boxes": boxes, "labels": labels,
                     "entities": entities, "pairs": pairs})
    return docs


TRAIN, TEST = load_funsd("training_data"), load_funsd("testing_data")
print(f"train {len(TRAIN)} forms | test {len(TEST)} forms")
counts = Counter(l for d in TRAIN for l in d["labels"])
print("train word-label counts:", dict(counts))
print("example key-value pairs (ground truth):")
d0 = TEST[0]
for q, a in list(d0["pairs"])[:5]:
    print(f"  {d0['entities'][q]['text']!r:40} -> {d0['entities'][a]['text']!r}")

COLORS = {"HEADER": "tab:purple", "QUESTION": "tab:blue", "ANSWER": "tab:green", "OTHER": "tab:gray"}
fig, ax = plt.subplots(figsize=(9, 11))
ax.imshow(d0["image"])
W, H = d0["image"].size
for ent in d0["entities"].values():
    x0, y0, x1, y1 = [v / 1000 * s for v, s in zip(ent["box"], (W, H, W, H))]
    ax.add_patch(plt.Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, lw=1.2, color=COLORS[ent["label"]]))
for q, a in d0["pairs"]:
    bq, ba = d0["entities"][q]["box"], d0["entities"][a]["box"]
    ax.plot([bq[2] / 1000 * W, ba[0] / 1000 * W], [(bq[1] + bq[3]) / 2000 * H, (ba[1] + ba[3]) / 2000 * H], "r-", lw=0.8)
ax.set_title("FUNSD: header (purple), question (blue), answer (green), links (red)")
ax.axis("off")
save_fig(fig, "day2_funsd_annotations.png")

# %% [markdown]
# ## 3 · Encode word windows for LayoutLMv3 and for a text-only baseline
#
# `LayoutLMv3Processor(apply_ocr=False)` takes *your* words and boxes (FUNSD ships them; in production they come from Day 1's OCR), tokenizes words into sub-words, repeats each word's box for its sub-words, resizes the image to 224×224 patches, and puts the label on the first sub-word only (`-100` elsewhere, ignored by the loss).
#
# The text-only baseline gets exactly the same words and labels, but no boxes and no image.

# %%
LMV3_ID = "microsoft/layoutlmv3-base"
processor = LayoutLMv3Processor.from_pretrained(LMV3_ID, apply_ocr=False)
text_tok = AutoTokenizer.from_pretrained("roberta-base", add_prefix_space=True)


def windows(doc: dict) -> list[tuple[int, int]]:
    n = len(doc["words"])
    return [(s, min(n, s + CHUNK_WORDS)) for s in range(0, n, CHUNK_WORDS)] or [(0, 0)]


def encode_layout(doc: dict, s: int, e: int, zero_boxes: bool = False) -> dict:
    boxes = [[0, 0, 0, 0]] * (e - s) if zero_boxes else doc["boxes"][s:e]
    enc = processor(doc["image"], doc["words"][s:e], boxes=boxes, word_labels=[L2ID[l] for l in doc["labels"][s:e]],
                    truncation=True, padding="max_length", max_length=512, return_tensors="pt")
    return {k: v.squeeze(0) for k, v in enc.items()}


def encode_text(doc: dict, s: int, e: int) -> dict:
    enc = text_tok(doc["words"][s:e], is_split_into_words=True, truncation=True, padding="max_length", max_length=512,
                   return_tensors="pt")
    labels, prev = [], None
    for wid in enc.word_ids(0):
        labels.append(-100 if wid is None or wid == prev else L2ID[doc["labels"][s + wid]])
        prev = wid
    out = {k: v.squeeze(0) for k, v in enc.items()}
    out["labels"] = torch.tensor(labels)
    return out


def encode_all(docs, fn, **kw) -> list[dict]:
    return [dict(fn(d, s, e, **kw), doc=i, start=s, end=e) for i, d in enumerate(docs) for s, e in windows(d) if e > s]


t0 = time.perf_counter()
TRAIN_LMV3, TEST_LMV3 = encode_all(TRAIN, encode_layout), encode_all(TEST, encode_layout)
TRAIN_TXT, TEST_TXT = encode_all(TRAIN, encode_text), encode_all(TEST, encode_text)
print(f"windows: train {len(TRAIN_LMV3)} | test {len(TEST_LMV3)} | encoded in {time.perf_counter() - t0:.1f}s")
ex = TRAIN_LMV3[0]
print({k: tuple(v.shape) for k, v in ex.items() if torch.is_tensor(v)})
kept = sum(int((x["labels"] != -100).sum()) for x in TRAIN_LMV3)
print(f"train words kept after 512-token truncation: {kept}/{sum(x['end'] - x['start'] for x in TRAIN_LMV3)}")

# %% [markdown]
# ## 4 · Fine-tune (plain PyTorch loop, same as Weeks 1–2)
#
# One loop trains both models: AdamW, linear warm-up/decay, mixed precision on GPU. The model computes cross-entropy over the 7 labels itself when `labels` is passed.

# %%
TENSOR_KEYS = ("input_ids", "attention_mask", "bbox", "pixel_values", "labels")


def collate(items: list[dict]) -> dict:
    return {k: torch.stack([x[k] for x in items]).to(DEVICE) for k in TENSOR_KEYS if k in items[0]}


def train_model(model, data: list[dict], steps: int, name: str) -> list[float]:
    model.to(DEVICE).train()
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    sched = get_linear_schedule_with_warmup(opt, int(0.1 * steps), steps)
    scaler = torch.amp.GradScaler("cuda", enabled=USE_AMP)
    losses, step = [], 0
    t0 = time.perf_counter()
    while step < steps:
        order = np.random.permutation(len(data))
        for i in range(0, len(order), BATCH_SIZE):
            batch = collate([data[j] for j in order[i:i + BATCH_SIZE]])
            with torch.autocast(DEVICE.type, dtype=torch.float16, enabled=USE_AMP):
                loss = model(**batch).loss
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            losses.append(loss.item())
            step += 1
            if step % 100 == 0 or step == steps:
                print(f"[{name}] step {step}/{steps} loss {np.mean(losses[-50:]):.4f} ({time.perf_counter() - t0:.0f}s)")
            if step >= steps:
                break
    return losses


@torch.no_grad()
def predict_words(model, data: list[dict], docs: list[dict]) -> list[list[str]]:
    """Per-document word labels: the prediction at each word's first sub-token (labels != -100 marks them)."""
    model.to(DEVICE).eval()
    preds = [["O"] * len(d["words"]) for d in docs]
    for i in range(0, len(data), 8):
        chunk = data[i:i + 8]
        with torch.autocast(DEVICE.type, dtype=torch.float16, enabled=USE_AMP):
            logits = model(**{k: v for k, v in collate(chunk).items() if k != "labels"}).logits
        for x, lg in zip(chunk, logits.argmax(-1).cpu()):
            first = (x["labels"] != -100).nonzero().flatten()
            for k, pos in enumerate(first.tolist()):
                preds[x["doc"]][x["start"] + k] = LABELS[lg[pos]]
    return preds


lmv3 = LayoutLMv3ForTokenClassification.from_pretrained(LMV3_ID, num_labels=len(LABELS), id2label=dict(enumerate(LABELS)),
                                                        label2id=L2ID)
LOSS_LMV3 = train_model(lmv3, TRAIN_LMV3, TRAIN_STEPS, "LayoutLMv3")
PRED_LMV3 = predict_words(lmv3, TEST_LMV3, TEST)

if TRAIN_TEXT_BASELINE:
    roberta = AutoModelForTokenClassification.from_pretrained("roberta-base", num_labels=len(LABELS),
                                                              id2label=dict(enumerate(LABELS)), label2id=L2ID)
    LOSS_TXT = train_model(roberta, TRAIN_TXT, TRAIN_STEPS, "RoBERTa text-only")
    PRED_TXT = predict_words(roberta, TEST_TXT, TEST)

# %% [markdown]
# ## 5 · Entity-level F1, and how much layout is worth
#
# `seqeval` scores whole entities: a predicted QUESTION counts only if its span and type both match exactly.
#
# The zero-box run reuses the trained LayoutLMv3 but hides every box at inference. It is not a fair model (it never trained that way), so read it as "how much does the trained model *rely* on position", while RoBERTa is the fair text-only baseline.

# %%
Y_TRUE = [d["labels"] for d in TEST]
PRED_ZERO = predict_words(lmv3, encode_all(TEST, encode_layout, zero_boxes=True), TEST)
RESULTS = {"LayoutLMv3 (text + layout + image)": PRED_LMV3, "LayoutLMv3, boxes zeroed at inference": PRED_ZERO}
if TRAIN_TEXT_BASELINE:
    RESULTS["RoBERTa (text only)"] = PRED_TXT
ENTITY_F1 = {name: seq_f1(Y_TRUE, p) for name, p in RESULTS.items()}
for name, f in ENTITY_F1.items():
    print(f"{name:42s} entity F1 = {f:.3f}")
print("\nLayoutLMv3 per entity type:\n", classification_report(Y_TRUE, PRED_LMV3, digits=3))

fig, axes = plt.subplots(1, 2, figsize=(14, 4))
axes[0].plot(pd.Series(LOSS_LMV3).rolling(20).mean(), label="LayoutLMv3")
if TRAIN_TEXT_BASELINE:
    axes[0].plot(pd.Series(LOSS_TXT).rolling(20).mean(), label="RoBERTa text-only")
axes[0].set_xlabel("step")
axes[0].set_ylabel("training loss (20-step mean)")
axes[0].legend()
axes[1].barh(list(ENTITY_F1), list(ENTITY_F1.values()), color=["tab:blue", "tab:orange", "tab:gray"][:len(ENTITY_F1)])
axes[1].set_xlim(0, 1)
axes[1].set_title("FUNSD test entity F1")
save_fig(fig, "day2_layout_vs_text.png")

# %% [markdown]
# ## 6 · From entities to key-value JSON
#
# The token classifier tells you *what* each word is, not *which answer belongs to which question*. A geometric linker closes the gap: each ANSWER links to the nearest QUESTION to its left on the same line, or directly above it. (LayoutLMv3 can also be trained for relation extraction, but this heuristic is the common production baseline.)
#
# Scored two ways:
# - **oracle entities**: the linker on ground-truth entities, which isolates linking errors;
# - **predicted entities**: the full pipeline, the number that matters.

# %%
def entities_from_bio(doc: dict, labels: list[str]) -> list[dict]:
    ents, cur = [], None
    for i, lab in enumerate(labels):
        if lab == "O":
            cur = None
            continue
        bio, tag = lab.split("-", 1)
        if cur is not None and bio == "I" and cur["label"] == tag:
            cur["idx"].append(i)
        else:  # B- tag, or an I- tag that does not continue the open entity
            cur = {"label": tag, "idx": [i]}
            ents.append(cur)
    for e in ents:
        bs = [doc["boxes"][i] for i in e["idx"]]
        e["text"] = " ".join(doc["words"][i] for i in e["idx"])
        e["box"] = [min(b[0] for b in bs), min(b[1] for b in bs), max(b[2] for b in bs), max(b[3] for b in bs)]
    return ents


def link_pairs(ents: list[dict], max_dist: int = 250) -> list[tuple[dict, dict]]:
    qs = [e for e in ents if e["label"] == "QUESTION"]
    pairs = []
    for a in (e for e in ents if e["label"] == "ANSWER"):
        best, best_d = None, max_dist
        for q in qs:
            ax0, ay0, ax1, ay1 = a["box"]
            qx0, qy0, qx1, qy1 = q["box"]
            v_overlap = min(ay1, qy1) - max(ay0, qy0)
            h_overlap = min(ax1, qx1) - max(ax0, qx0)
            if v_overlap > 0 and qx1 <= ax0 + 15:
                dist = ax0 - qx1  # same line, question on the left
            elif h_overlap > 0 and qy1 <= ay0 + 5:
                dist = 1.5 * (ay0 - qy1)  # question directly above
            else:
                continue
            if dist < best_d:
                best, best_d = q, dist
        if best is not None:
            pairs.append((best, a))
    return pairs


def norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", s.lower()).strip()


def pair_prf(pred_pairs: set, gt_pairs: set) -> dict:
    tp = len(pred_pairs & gt_pairs)
    p, r = tp / max(1, len(pred_pairs)), tp / max(1, len(gt_pairs))
    return {"precision": p, "recall": r, "f1": 2 * p * r / max(1e-9, p + r)}


def kv_scores(pred_labels: list[list[str]] | None) -> dict:
    agg = Counter()
    for k, doc in enumerate(TEST):
        gt = {(norm(doc["entities"][q]["text"]), norm(doc["entities"][a]["text"])) for q, a in doc["pairs"]}
        labels = doc["labels"] if pred_labels is None else pred_labels[k]
        pred = {(norm(q["text"]), norm(a["text"])) for q, a in link_pairs(entities_from_bio(doc, labels))}
        agg["tp"] += len(pred & gt)
        agg["pred"] += len(pred)
        agg["gt"] += len(gt)
    p, r = agg["tp"] / max(1, agg["pred"]), agg["tp"] / max(1, agg["gt"])
    return {"precision": p, "recall": r, "f1": 2 * p * r / max(1e-9, p + r)}


KV = {"oracle entities + linker": kv_scores(None), "LayoutLMv3 entities + linker": kv_scores(PRED_LMV3)}
if TRAIN_TEXT_BASELINE:
    KV["RoBERTa entities + linker"] = kv_scores(PRED_TXT)
KV_DF = pd.DataFrame(KV).T.round(3)
print(KV_DF.to_string())


def to_kv_json(doc: dict, labels: list[str]) -> dict:
    ents = entities_from_bio(doc, labels)
    out = {"doc_id": doc["id"], "headers": [e["text"] for e in ents if e["label"] == "HEADER"], "fields": {}}
    for q, a in link_pairs(ents):
        key = q["text"].strip(" :")
        out["fields"][key] = f"{out['fields'][key]} {a['text']}" if key in out["fields"] else a["text"]
    return out


example_json = to_kv_json(TEST[0], PRED_LMV3[0])
(OUTPUT_DIR / "day2_funsd_kv_example.json").write_text(json.dumps(example_json, indent=2), encoding="utf-8")
print(json.dumps(example_json, indent=2)[:1500])

# %% [markdown]
# ## 7 · LayoutLMv3 on an invoice: OCR words in, key-value JSON out
#
# In production nobody hands you word boxes. Here Tesseract (Day 1) supplies them for a synthetic invoice, and the FUNSD-trained model labels them. The model has never seen an invoice; watch which labels transfer (questions like "Invoice Date:") and which do not (table cells).

# %%
inv = make_invoice("lmv3_demo", "standard", SEED)
img = inv.pages[0]
data = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT)
ocr_words = [(t, [data["left"][i], data["top"][i], data["left"][i] + data["width"][i], data["top"][i] + data["height"][i]])
             for i, t in enumerate(data["text"]) if t.strip()]
inv_doc = {"id": inv.doc_id, "image": img, "words": [w for w, _ in ocr_words],
           "boxes": [norm_box(b, *img.size) for _, b in ocr_words], "labels": ["O"] * len(ocr_words)}
inv_pred = predict_words(lmv3, encode_all([inv_doc], encode_layout), [inv_doc])[0]
print(json.dumps(to_kv_json(inv_doc, inv_pred), indent=2))
print("truth:", json.dumps(inv.truth["fields"]))

fig, ax = plt.subplots(figsize=(9, 12))
ax.imshow(img)
for ent in entities_from_bio(inv_doc, inv_pred):
    x0, y0, x1, y1 = [v / 1000 * s for v, s in zip(ent["box"], img.size * 2)]
    ax.add_patch(plt.Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, lw=1.2, color=COLORS[ent["label"]]))
ax.set_title("FUNSD-trained LayoutLMv3 on an invoice (blue=question, green=answer)")
ax.axis("off")
save_fig(fig, "day2_lmv3_on_invoice.png")

# %% [markdown]
# ## 8 · Donut: OCR-free document understanding
#
# Donut = Swin Transformer image encoder + BART text decoder. It reads pixels and *generates* an XML-like token sequence (`<s_menu><s_nm>Ice Tea</s_nm>…`) that `processor.token2json` turns into JSON. No OCR engine, no boxes, no linker — but also no word positions to show *where* a value came from, and the output schema is whatever the model was fine-tuned on (here: CORD receipts).
#
# **Field-level F1** (from the Donut paper): flatten the JSON to `(path, value)` leaves like `menu.nm = "ice tea"` and compute multiset F1 against ground truth.

# %%
from datasets import load_dataset  # noqa: E402

DONUT_ID = "naver-clova-ix/donut-base-finetuned-cord-v2"
donut_proc = DonutProcessor.from_pretrained(DONUT_ID)
donut = VisionEncoderDecoderModel.from_pretrained(DONUT_ID).to(DEVICE).eval()
if USE_AMP:
    donut.half()


@torch.no_grad()
def donut_parse(image: Image.Image, task: str = "<s_cord-v2>") -> dict:
    pixel_values = donut_proc(image.convert("RGB"), return_tensors="pt").pixel_values.to(DEVICE, donut.dtype)
    prompt = donut_proc.tokenizer(task, add_special_tokens=False, return_tensors="pt").input_ids.to(DEVICE)
    out = donut.generate(pixel_values, decoder_input_ids=prompt, max_length=donut.decoder.config.max_position_embeddings,
                         pad_token_id=donut_proc.tokenizer.pad_token_id, eos_token_id=donut_proc.tokenizer.eos_token_id,
                         use_cache=True, bad_words_ids=[[donut_proc.tokenizer.unk_token_id]], num_beams=1)
    seq = donut_proc.batch_decode(out)[0]
    seq = seq.replace(donut_proc.tokenizer.eos_token, "").replace(donut_proc.tokenizer.pad_token, "")
    seq = re.sub(r"<.*?>", "", seq, count=1).strip()  # drop the task start token
    return donut_proc.token2json(seq)


def flatten(obj, prefix: str = "") -> list[tuple[str, str]]:
    if isinstance(obj, dict):
        return [kv for k, v in obj.items() for kv in flatten(v, f"{prefix}.{k}" if prefix else k)]
    if isinstance(obj, list):
        return [kv for v in obj for kv in flatten(v, prefix)]
    return [(prefix, re.sub(r"\s+", "", str(obj).lower()))]


def field_f1(pred: dict, gt: dict) -> tuple[int, int, int]:
    p, g = Counter(flatten(pred)), Counter(flatten(gt))
    return sum((p & g).values()), sum(p.values()), sum(g.values())


def cord_total(d: dict):
    t = d.get("total")
    t = t[0] if isinstance(t, list) and t else t
    return re.sub(r"\D", "", str(t.get("total_price", ""))) if isinstance(t, dict) else None


def n_menu(d: dict) -> int:
    m = d.get("menu", [])
    return len(m) if isinstance(m, list) else 1


cord = load_dataset("naver-clova-ix/cord-v2", split="test", streaming=True).take(N_CORD)
rows, examples = [], []
for i, sample in enumerate(cord):
    gt = json.loads(sample["ground_truth"])["gt_parse"]
    t0 = time.perf_counter()
    pred = donut_parse(sample["image"])
    dt = time.perf_counter() - t0
    tp, npred, ngt = field_f1(pred, gt)
    rows.append({"tp": tp, "pred": npred, "gt": ngt, "total_price_correct": cord_total(gt) == cord_total(pred),
                 "n_items_correct": n_menu(gt) == n_menu(pred), "latency_s": dt})
    if i < 3:
        examples.append((sample["image"], pred, gt))
DONUT_DF = pd.DataFrame(rows)
P = DONUT_DF.tp.sum() / max(1, DONUT_DF.pred.sum())
R = DONUT_DF.tp.sum() / max(1, DONUT_DF.gt.sum())
DONUT_SCORES = {"field_precision": P, "field_recall": R, "field_f1": 2 * P * R / max(1e-9, P + R),
                "total_price_accuracy": DONUT_DF.total_price_correct.mean(), "item_count_accuracy": DONUT_DF.n_items_correct.mean(),
                "latency_s_per_page": DONUT_DF.latency_s.mean()}
print(json.dumps({k: round(float(v), 3) for k, v in DONUT_SCORES.items()}, indent=2))

img0, pred0, gt0 = examples[0]
fig, ax = plt.subplots(figsize=(5, 8))
ax.imshow(img0)
ax.axis("off")
ax.set_title("CORD test receipt")
save_fig(fig, "day2_donut_receipt.png")
print("Donut:", json.dumps(pred0, indent=1, ensure_ascii=False)[:800])
print("Truth:", json.dumps(gt0, indent=1, ensure_ascii=False)[:800])

# %% [markdown]
# ### Donut out of domain
#
# The same model on an invoice: it can only answer in the CORD schema (`menu`, `sub_total`, `total`). A new document type means a new fine-tune with labeled JSON — the main cost of the OCR-free approach.

# %%
for layout in ("standard", "layout_b"):
    inv = make_invoice(f"donut_{layout}", layout, SEED)
    out = donut_parse(inv.pages[0])
    print(f"\n{layout}: {json.dumps(out, ensure_ascii=False)[:600]}")
    print(f"truth total = {inv.truth['fields']['total']}, items = {len(inv.truth['line_items'])}")

# %% [markdown]
# ## 9 · Save results

# %%
if SAVE_MODEL:
    lmv3.save_pretrained(OUTPUT_DIR / "layoutlmv3_funsd")
    processor.save_pretrained(OUTPUT_DIR / "layoutlmv3_funsd")
summary = {
    "funsd": {"train_forms": len(TRAIN), "test_forms": len(TEST), "train_steps": TRAIN_STEPS,
              "entity_f1": ENTITY_F1, "kv_pairs": KV_DF.to_dict("index")},
    "donut_cord": {"n_receipts": len(DONUT_DF), **DONUT_SCORES},
    "full_run": FULL_RUN,
}
with open(OUTPUT_DIR / "day2_summary.json", "w") as f:
    json.dump(summary, f, indent=2, default=float)
print(json.dumps(summary, indent=2, default=float))

# %% [markdown]
# ## Takeaways (fill in with your numbers)
#
# - FUNSD entity F1: LayoutLMv3 ___ vs RoBERTa text-only ___ → layout + image are worth ___ points. Zeroing boxes drops LayoutLMv3 to ___.
# - Key-value pair F1: oracle entities ___ (linker ceiling) vs predicted ___. Which loses more points, entity labeling or linking?
# - Donut on CORD: field F1 ___, total-price accuracy ___ at ___ s/page. On an invoice it outputs ___.
#
# **LayoutLMv3 vs Donut:** LayoutLMv3 depends on OCR quality (Day 1) but gives word-level provenance and transfers its labels (question/answer) across document types. Donut skips OCR errors and the linker, but hallucination is possible, every schema needs its own fine-tune, and there are no boxes to verify against. Tomorrow: what a managed service gives you without any training at all.
