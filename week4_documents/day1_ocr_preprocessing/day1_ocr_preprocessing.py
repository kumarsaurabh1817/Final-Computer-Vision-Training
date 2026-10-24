"""
Week 4 · Day 1 — OCR landscape (Tesseract vs EasyOCR vs PaddleOCR) and scan preprocessing.

Colab-first script (percent-format cells). Build the notebook with:
    python ../tools/py_to_ipynb.py day1_ocr_preprocessing/day1_ocr_preprocessing.py
Datasets: FUNSD (https://guillaumejaume.github.io/FUNSD/) + synthetic invoices rendered in this script
"""

# %% [markdown]
# # Week 4 · Day 1 — OCR Landscape & Preprocessing
#
# **Goal:** extract raw text from scanned documents with three self-hosted OCR engines, measure them properly, and find out how much preprocessing (deskew, denoise, threshold) actually buys you.
#
# | Topic | What you will build |
# |---|---|
# | OCR engines | One wrapper per engine returning the same `{"text", "conf", "bbox"}` word list |
# | OCR metrics | Order-free **word F1** and reading-order **CER** (character error rate) against ground truth |
# | Engine comparison | Tesseract 5 vs EasyOCR vs PaddleOCR on real scanned forms (FUNSD): accuracy and latency |
# | Preprocessing | Grayscale → denoise → deskew (projection profile) → CLAHE → Otsu / adaptive threshold |
# | Ablation | Degradation (skew, noise, blur, low contrast) × preprocessing pipeline → word F1 heatmap |
#
# **Data**
# - [FUNSD](https://guillaumejaume.github.io/FUNSD/): 199 real, noisy scanned forms with word-level boxes and text (~16 MB). The test split (50 forms) is used here.
# - Synthetic invoices rendered by the generator in section 2: clean renders with exact text ground truth, degraded on purpose so every preprocessing step has something to fix.
#
# > Runtime: **Runtime → Change runtime type → T4 GPU** (EasyOCR uses it), then run all cells in order.

# %% [markdown]
# ## 0 · Install dependencies
#
# Tesseract is a system binary (`apt`), `pytesseract` is only its Python wrapper. PaddleOCR is optional because `paddlepaddle` is a large install; switch `RUN_PADDLE` off if it fails in your runtime.

# %%
import shutil
import subprocess
import sys

RUN_PADDLE = True


def pip_install(*packages: str) -> None:
    """Install packages into the running interpreter (works in Colab and plain Python)."""
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *packages])


if shutil.which("tesseract") is None and shutil.which("apt-get"):
    subprocess.run("apt-get -qq update && apt-get -qq install -y tesseract-ocr > /dev/null", shell=True, check=True)
pip_install("pytesseract==0.3.13", "easyocr==1.7.2", "rapidfuzz>=3.9")
if RUN_PADDLE:
    try:
        pip_install("paddlepaddle>=3.0,<4", "paddleocr>=3.0,<4")
    except subprocess.CalledProcessError as exc:
        print(f"PaddleOCR install failed ({exc}); continuing without it")
        RUN_PADDLE = False

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

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytesseract
import torch
from PIL import Image
from rapidfuzz.distance import Levenshtein

IN_COLAB = "google.colab" in sys.modules
try:
    HERE = Path(__file__).resolve().parent
except NameError:  # running inside a notebook
    HERE = Path.cwd()

OUTPUT_DIR = Path("/content/week4_outputs/day1") if IN_COLAB else HERE.parent / "results" / "day1"
DATA_DIR = Path("/content/datasets") if IN_COLAB else HERE.parent / "data"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)

FULL_RUN = True  # False = quick smoke test on a handful of pages
N_FUNSD = 50 if FULL_RUN else 6
N_SYNTH = 2 if FULL_RUN else 1  # synthetic invoices per layout
SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Tesseract {pytesseract.get_tesseract_version()} | device={DEVICE} | FULL_RUN={FULL_RUN}")
print("Outputs ->", OUTPUT_DIR)


def save_fig(fig, name: str) -> None:
    """Save a figure to OUTPUT_DIR and show it inline when in a notebook."""
    path = OUTPUT_DIR / name
    fig.savefig(path, dpi=120, bbox_inches="tight")
    if IN_NOTEBOOK:
        plt.show()
    plt.close(fig)
    print(f"saved -> {path}")


# %% [markdown]
# ## 2 · Helper code: preprocessing steps and a synthetic invoice generator
#
# The first cell holds the preprocessing functions (grayscale, denoise, deskew, CLAHE, threshold). The second renders invoices with exact ground truth, so OCR can be scored without hand labeling. Read both before moving on.

# %%
# ---- preprocess.py: helper code for this day (self-contained, no external project files) ----
# Scan preprocessing for OCR (Week 4, shared by Days 1, 4, 5).
#
# Every function takes and returns a numpy uint8 image (grayscale unless noted), so the
# steps can be chained in any order and switched on/off for ablations:
#
#     gray -> denoise -> deskew -> normalize contrast -> binarize
#
# day scripts run locally.
import cv2
import numpy as np
from PIL import Image


def to_gray(img) -> np.ndarray:
    """PIL image or RGB/BGR/gray array -> uint8 grayscale array."""
    if isinstance(img, Image.Image):
        img = np.asarray(img.convert("RGB"))
    if img.ndim == 3:
        return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    return img.copy()


def denoise(gray: np.ndarray, method: str = "median", strength: int = 3) -> np.ndarray:
    """Remove sensor / speckle noise. Median is fast and keeps stroke edges; NLM is stronger but slower."""
    if method == "median":
        return cv2.medianBlur(gray, strength if strength % 2 else strength + 1)
    if method == "nlm":
        return cv2.fastNlMeansDenoising(gray, None, h=10 * strength / 3, templateWindowSize=7, searchWindowSize=21)
    if method == "gaussian":
        return cv2.GaussianBlur(gray, (0, 0), strength / 3)
    raise ValueError(f"unknown denoise method {method!r}")


def normalize_contrast(gray: np.ndarray, clip_limit: float = 2.0) -> np.ndarray:
    """CLAHE: local histogram equalization. Rescues faded or unevenly lit scans."""
    return cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8)).apply(gray)


def binarize(gray: np.ndarray, method: str = "otsu", block_size: int = 31, c: int = 15) -> np.ndarray:
    """Black text on white. Otsu = one global threshold; adaptive = per-neighbourhood threshold (handles shadows)."""
    if method == "otsu":
        _, out = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return out
    if method == "adaptive":
        return cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, block_size, c)
    raise ValueError(f"unknown binarize method {method!r}")


def estimate_skew(gray: np.ndarray, max_angle: float = 10.0, step: float = 0.25) -> float:
    """
    Projection-profile skew estimate, in degrees (counter-clockwise positive, like cv2/PIL).

    Rotating a page so that text lines are horizontal makes the row sums of ink
    "spiky" (lines vs gaps), so their variance peaks at the correct angle. Searched
    coarse-to-fine on a downscaled, inverted, binarized copy.
    """
    small = gray
    scale = 1000 / max(gray.shape)
    if scale < 1:
        small = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    ink = 255 - binarize(small, "otsu")
    h, w = ink.shape
    center = (w / 2, h / 2)

    def score(angle: float) -> float:
        m = cv2.getRotationMatrix2D(center, angle, 1.0)
        rot = cv2.warpAffine(ink, m, (w, h), flags=cv2.INTER_NEAREST, borderValue=0)
        return float(np.var(rot.sum(axis=1, dtype=np.float64)))

    coarse = np.arange(-max_angle, max_angle + 1e-6, 1.0)
    best = max(coarse, key=score)
    fine = np.arange(best - 1.0, best + 1.0 + 1e-6, step)
    # The page was rotated by +a; we rotated it back by `best`, so the page's skew is -best.
    return -float(max(fine, key=score))


def rotate(img: np.ndarray, angle: float, border: int = 255) -> np.ndarray:
    """Rotate counter-clockwise by `angle` degrees around the centre, keeping the canvas size."""
    h, w = img.shape[:2]
    m = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    fill = (border,) * 3 if img.ndim == 3 else border
    return cv2.warpAffine(img, m, (w, h), flags=cv2.INTER_CUBIC, borderValue=fill)


def deskew(img: np.ndarray, max_angle: float = 10.0, min_angle: float = 0.3) -> tuple[np.ndarray, float]:
    """Estimate and undo skew. Returns (image, estimated skew). Works on gray or RGB arrays."""
    angle = estimate_skew(to_gray(img), max_angle=max_angle)
    if abs(angle) < min_angle:
        return img.copy(), angle
    return rotate(img, -angle), angle


def upscale_if_small(gray: np.ndarray, min_height: int = 1500) -> np.ndarray:
    """OCR engines like ~25–35 px cap height. Upscale low-resolution scans before recognition."""
    if gray.shape[0] >= min_height:
        return gray
    f = min_height / gray.shape[0]
    return cv2.resize(gray, None, fx=f, fy=f, interpolation=cv2.INTER_CUBIC)


PIPELINES = {
    "none": [],
    "gray+otsu": ["otsu"],
    "denoise+otsu": ["denoise", "otsu"],
    "deskew+otsu": ["deskew", "otsu"],
    "full": ["denoise", "deskew", "clahe", "adaptive"],
}


def preprocess(img, steps: list[str]) -> np.ndarray:
    """Apply named steps in order. Returns a grayscale uint8 array (or the gray input if steps is empty)."""
    out = to_gray(img)
    for step in steps:
        if step == "denoise":
            out = denoise(out, "median", 3)
        elif step == "nlm":
            out = denoise(out, "nlm", 3)
        elif step == "deskew":
            out, _ = deskew(out)
        elif step == "clahe":
            out = normalize_contrast(out)
        elif step == "otsu":
            out = binarize(out, "otsu")
        elif step == "adaptive":
            out = binarize(out, "adaptive")
        elif step == "upscale":
            out = upscale_if_small(out)
        else:
            raise ValueError(f"unknown step {step!r}")
    return out

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
# ## 3 · Download FUNSD
#
# FUNSD annotations group words into entities (`header`, `question`, `answer`, `other`). Today only the words matter: each has its own text and box, which is exactly what an OCR engine should return.

# %%
FUNSD_URL = "https://guillaumejaume.github.io/FUNSD/dataset.zip"


def get_funsd() -> Path:
    root = DATA_DIR / "funsd"
    if not (root / "dataset" / "testing_data").is_dir():
        zip_path = DATA_DIR / "funsd.zip"
        if not zip_path.exists():
            urllib.request.urlretrieve(FUNSD_URL, zip_path)
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(root)
    return root / "dataset"


def load_funsd(split: str = "testing_data", limit: int | None = None) -> list[dict]:
    split_dir = get_funsd() / split
    docs = []
    for ann_path in sorted((split_dir / "annotations").glob("*.json"))[:limit]:
        ann = json.loads(ann_path.read_text(encoding="utf-8"))
        words = [{"text": w["text"], "bbox": w["box"]} for ent in ann["form"] for w in ent["words"] if w["text"].strip()]
        docs.append({"id": ann_path.stem, "image": Image.open(split_dir / "images" / f"{ann_path.stem}.png").convert("RGB"),
                     "words": words})
    return docs


FUNSD = load_funsd(limit=N_FUNSD)
print(f"FUNSD test forms: {len(FUNSD)} | words in first form: {len(FUNSD[0]['words'])} | size {FUNSD[0]['image'].size}")


def synth_gt_words(doc) -> list[dict]:
    """Split each drawn text run of a synthetic page into words with proportional boxes."""
    words = []
    for rec in doc.lines:
        if rec["page"] != 1:
            continue
        x0, y0, x1, y1 = rec["bbox"]
        toks = rec["text"].split()
        n = sum(len(t) + 1 for t in toks)
        x = x0
        for t in toks:
            w = (x1 - x0) * len(t) / n
            words.append({"text": t, "bbox": [x, y0, x + w, y1]})
            x += w + (x1 - x0) / n
    return words


SYNTH = [d for d in make_invoice_set(N_SYNTH, seed=SEED, layouts=("standard", "layout_b", "merged"))]
for d in SYNTH:
    d.gt_words = synth_gt_words(d)
print(f"synthetic invoices: {len(SYNTH)}")

fig, axes = plt.subplots(1, 3, figsize=(16, 7))
axes[0].imshow(FUNSD[0]["image"])
axes[0].set_title(f"FUNSD {FUNSD[0]['id']} (real scan)")
axes[1].imshow(SYNTH[0].pages[0])
axes[1].set_title(f"{SYNTH[0].doc_id} (clean render)")
axes[2].imshow(make_invoice("demo", "rotated", SEED).pages[0])
axes[2].set_title("same layout, 'rotated' scan degradation")
for ax in axes:
    ax.axis("off")
save_fig(fig, "day1_samples.png")

# %% [markdown]
# ## 4 · Measuring OCR
#
# Two complementary numbers:
#
# - **Word F1 (order-free).** Multiset of normalized words (lowercase, punctuation stripped) predicted vs ground truth. Ignores reading order, so a layout the engine reads column-by-column is not punished.
# - **CER (character error rate)** = Levenshtein edit distance / number of GT characters, after putting both word lists in the same reading order (top-to-bottom lines, then left-to-right). Sensitive to every substitution such as `0`↔`O` and `1`↔`l`, which is what breaks invoice numbers and amounts.
#
# CER can exceed 1.0 when the engine hallucinates lots of extra text.

# %%
def norm_word(w: str) -> str:
    return re.sub(r"^[^\w$€£]+|[^\w%]+$", "", w.lower())


def word_f1(pred: list[dict], gt: list[dict]) -> dict:
    p = Counter(x for x in (norm_word(w["text"]) for w in pred) if x)
    g = Counter(x for x in (norm_word(w["text"]) for w in gt) if x)
    tp = sum((p & g).values())
    prec = tp / max(1, sum(p.values()))
    rec = tp / max(1, sum(g.values()))
    return {"precision": prec, "recall": rec, "f1": 2 * prec * rec / max(1e-9, prec + rec)}


def reading_order_text(words: list[dict]) -> str:
    """Group words into lines by vertical centre, then read lines top-down and words left-right."""
    if not words:
        return ""
    med_h = np.median([w["bbox"][3] - w["bbox"][1] for w in words])
    lines: list[list[dict]] = []
    for w in sorted(words, key=lambda w: (w["bbox"][1] + w["bbox"][3]) / 2):
        cy = (w["bbox"][1] + w["bbox"][3]) / 2
        if lines and abs(cy - np.mean([(v["bbox"][1] + v["bbox"][3]) / 2 for v in lines[-1]])) < 0.6 * med_h:
            lines[-1].append(w)
        else:
            lines.append([w])
    return "\n".join(" ".join(w["text"] for w in sorted(line, key=lambda w: w["bbox"][0])) for line in lines)


def cer(pred: list[dict], gt: list[dict]) -> float:
    ref = reading_order_text(gt)
    return Levenshtein.distance(reading_order_text(pred), ref) / max(1, len(ref))


# Sanity check: GT vs itself, and GT with one word dropped
gt0 = FUNSD[0]["words"]
print("perfect:", word_f1(gt0, gt0), "CER", cer(gt0, gt0))
print("one word missing:", {k: round(v, 3) for k, v in word_f1(gt0[1:], gt0).items()}, "CER", round(cer(gt0[1:], gt0), 3))

# %% [markdown]
# ## 5 · Three OCR engines behind one interface
#
# | Engine | Detection → recognition | Strengths | Weak spots |
# |---|---|---|---|
# | **Tesseract 5** | Classical page-layout analysis → LSTM line recognizer | CPU-only, fast, 100+ languages, word boxes + confidences, page segmentation modes | Needs clean, upright, high-contrast input; struggles with noise, skew, handwriting |
# | **EasyOCR** | CRAFT text detector → CRNN recognizer (PyTorch) | Two-line setup, GPU, 80+ languages, robust to noise | Returns phrases, not words; slower on CPU |
# | **PaddleOCR (PP-OCRv5)** | DB detector → SVTR recognizer, optional orientation/unwarping | Usually most accurate on dense or rotated text; strong on Chinese/multilingual | Heavy dependency (`paddlepaddle`), API changed in 3.x |
#
# EasyOCR and PaddleOCR return a box per text *line/phrase*. To compare like with like, each wrapper splits phrases into words with proportional boxes.

# %%
def split_phrase(text: str, box, conf: float) -> list[dict]:
    x0, y0, x1, y1 = box
    toks = text.split()
    n = sum(len(t) + 1 for t in toks) or 1
    out, x = [], x0
    for t in toks:
        w = (x1 - x0) * len(t) / n
        out.append({"text": t, "conf": conf, "bbox": [x, y0, x + w, y1]})
        x += w + (x1 - x0) / n
    return out


def ocr_tesseract(img, psm: int = 3) -> list[dict]:
    data = pytesseract.image_to_data(img, config=f"--psm {psm}", output_type=pytesseract.Output.DICT)
    out = []
    for i, t in enumerate(data["text"]):
        if t.strip():
            x, y, w, h = data["left"][i], data["top"][i], data["width"][i], data["height"][i]
            out.append({"text": t.strip(), "conf": float(data["conf"][i]) / 100, "bbox": [x, y, x + w, y + h]})
    return out


import easyocr  # noqa: E402

EASY = easyocr.Reader(["en"], gpu=DEVICE == "cuda", verbose=False)


def ocr_easyocr(img) -> list[dict]:
    arr = np.asarray(img) if isinstance(img, Image.Image) else img
    out = []
    for pts, text, conf in EASY.readtext(arr):
        xs, ys = [p[0] for p in pts], [p[1] for p in pts]
        out += split_phrase(text, [min(xs), min(ys), max(xs), max(ys)], float(conf))
    return out


PADDLE = None
if RUN_PADDLE:
    try:
        from paddleocr import PaddleOCR

        PADDLE = PaddleOCR(lang="en", use_doc_orientation_classify=False, use_doc_unwarping=False,
                           use_textline_orientation=False)
    except Exception as exc:  # version / install problems should not stop the day
        print(f"PaddleOCR unavailable ({exc})")


def ocr_paddle(img) -> list[dict]:
    arr = np.asarray(img.convert("RGB") if isinstance(img, Image.Image) else img)
    if arr.ndim == 2:
        arr = cv2.cvtColor(arr, cv2.COLOR_GRAY2RGB)
    res = PADDLE.predict(cv2.cvtColor(arr, cv2.COLOR_RGB2BGR))[0]
    out = []
    for text, score, poly in zip(res["rec_texts"], res["rec_scores"], res["rec_polys"]):
        poly = np.asarray(poly)
        out += split_phrase(text, [poly[:, 0].min(), poly[:, 1].min(), poly[:, 0].max(), poly[:, 1].max()], float(score))
    return out


ENGINES = {"tesseract": ocr_tesseract, "easyocr": ocr_easyocr}
if PADDLE is not None:
    ENGINES["paddleocr"] = ocr_paddle
for name, fn in ENGINES.items():  # warm-up (model loading / CUDA init is not part of latency)
    fn(SYNTH[0].pages[0])
print("engines:", list(ENGINES))

# %% [markdown]
# ## 6 · Engine comparison on real scans (FUNSD) and clean invoices
#
# Latency is wall-clock per page on this runtime, after warm-up.

# %%
def run_engines(docs: list[dict], dataset: str) -> pd.DataFrame:
    rows = []
    for d in docs:
        for name, fn in ENGINES.items():
            t0 = time.perf_counter()
            pred = fn(d["image"])
            dt = time.perf_counter() - t0
            rows.append({"dataset": dataset, "doc": d["id"], "engine": name, **word_f1(pred, d["words"]),
                         "cer": cer(pred, d["words"]), "latency_s": dt})
    return pd.DataFrame(rows)


synth_docs = [{"id": d.doc_id, "image": d.pages[0], "words": d.gt_words} for d in SYNTH]
ENGINE_DF = pd.concat([run_engines(FUNSD, "FUNSD (real scans)"), run_engines(synth_docs, "synthetic (clean)")])
ENGINE_TABLE = ENGINE_DF.groupby(["dataset", "engine"])[["precision", "recall", "f1", "cer", "latency_s"]].mean().round(3)
ENGINE_TABLE.to_csv(OUTPUT_DIR / "day1_engine_comparison.csv")
print(ENGINE_TABLE.to_string())

fig, axes = plt.subplots(1, 3, figsize=(16, 4))
for ax, metric in zip(axes, ["f1", "cer", "latency_s"]):
    ENGINE_TABLE[metric].unstack("engine").plot.bar(ax=ax, rot=0)
    ax.set_title(metric + (" (lower is better)" if metric != "f1" else " (higher is better)"))
save_fig(fig, "day1_engine_comparison.png")

# %% [markdown]
# ### What each engine actually returned
#
# Green boxes are words that appear in the ground truth; red boxes are words that do not (misreads or noise). Look for where each engine fails: tiny print, checkboxes, stamps, handwriting.

# %%
def draw_words(ax, img, pred, gt, title):
    ax.imshow(img)
    gt_set = Counter(norm_word(w["text"]) for w in gt)
    for w in pred:
        ok = gt_set[norm_word(w["text"])] > 0
        x0, y0, x1, y1 = w["bbox"]
        ax.add_patch(plt.Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, lw=0.8, color="green" if ok else "red"))
    ax.set_title(title, fontsize=10)
    ax.axis("off")


doc = FUNSD[min(3, len(FUNSD) - 1)]
fig, axes = plt.subplots(1, len(ENGINES), figsize=(6 * len(ENGINES), 8))
for ax, (name, fn) in zip(np.atleast_1d(axes), ENGINES.items()):
    pred = fn(doc["image"])
    draw_words(ax, doc["image"], pred, doc["words"], f"{name}: F1={word_f1(pred, doc['words'])['f1']:.2f}")
save_fig(fig, "day1_engine_boxes.png")

# %% [markdown]
# ## 7 · Preprocessing, step by step
#
# | Step | Fixes | Implementation (section 2) |
# |---|---|---|
# | Grayscale | Colour carries no text information | `cv2.cvtColor` |
# | Denoise | Scanner speckle that becomes fake characters | Median blur (fast) or non-local means (strong) |
# | **Deskew** | Rotated lines; recognizers expect horizontal text | Projection profile: rotate, sum ink per row, keep the angle whose row profile has the highest variance |
# | CLAHE | Faded / unevenly lit scans | Local histogram equalization |
# | Threshold | Separates ink from paper | Otsu = one global threshold; adaptive = per-neighbourhood (handles shadows) |
#
# The degraded invoice below is the `rotated` layout: skew, blur, Gaussian noise, JPEG artifacts and lowered contrast.

# %%
scan = make_invoice("prep_demo", "rotated", SEED)
true_skew = scan.truth["meta"]["skew_deg"]
g = to_gray(scan.pages[0])
steps = {"input (degraded)": g}
steps["median denoise"] = denoise(g, "median", 3)
deskewed, est = deskew(steps["median denoise"])
steps[f"deskew (est {est:+.2f}°, true {true_skew:+.2f}°)"] = deskewed
steps["CLAHE"] = normalize_contrast(deskewed)
steps["Otsu threshold"] = binarize(steps["CLAHE"], "otsu")
steps["adaptive threshold"] = binarize(steps["CLAHE"], "adaptive")

fig, axes = plt.subplots(2, 3, figsize=(16, 14))
for ax, (title, im) in zip(axes.ravel(), steps.items()):
    ax.imshow(im[150:850, 50:1200], cmap="gray")
    ax.set_title(title)
    ax.axis("off")
save_fig(fig, "day1_preprocessing_steps.png")

# Skew estimator accuracy over a sweep of known angles
angles = np.linspace(-8, 8, 9)
est_angles = [estimate_skew(to_gray(SYNTH[0].pages[0].rotate(a, fillcolor="white"))) for a in angles]
print("skew sweep  true:", np.round(angles, 1))
print("            est :", np.round(est_angles, 2))
print(f"mean abs error {np.mean(np.abs(np.array(est_angles) - angles)):.3f}°")

# %% [markdown]
# ## 8 · Ablation: degradation × preprocessing pipeline
#
# Each synthetic invoice is degraded in one way at a time, then OCR'd by Tesseract after each pipeline from `PIPELINES`. Tesseract is used because it is the engine most sensitive to input quality (and the fastest).

# %%
DEGRADATIONS = {
    "clean": {},
    "skew 5°": {"angle": 5.0},
    "noise σ=25": {"noise": 25},
    "blur 1.5": {"blur": 1.5},
    "low contrast + jpeg": {"contrast": 0.35, "jpeg_quality": 30},
    "all of the above": {"angle": -4.0, "noise": 18, "blur": 1.0, "contrast": 0.5, "jpeg_quality": 40},
}

rows = []
for d in SYNTH:
    for deg_name, kw in DEGRADATIONS.items():
        img = degrade_scan(d.pages[0], random.Random(SEED), **kw)
        for pipe_name, pipe in PIPELINES.items():
            pre = preprocess(img, pipe)
            pred = ocr_tesseract(pre)
            rows.append({"doc": d.doc_id, "degradation": deg_name, "pipeline": pipe_name,
                         **word_f1(pred, d.gt_words), "cer": cer(pred, d.gt_words)})
ABL = pd.DataFrame(rows)
heat = ABL.pivot_table(index="degradation", columns="pipeline", values="f1").loc[list(DEGRADATIONS), list(PIPELINES)]
ABL.to_csv(OUTPUT_DIR / "day1_preprocessing_ablation.csv", index=False)
print(heat.round(3).to_string())

fig, ax = plt.subplots(figsize=(10, 5))
im = ax.imshow(heat.values, cmap="RdYlGn", vmin=max(0, heat.values.min() - 0.05), vmax=1)
ax.set_xticks(range(heat.shape[1]), heat.columns, rotation=20)
ax.set_yticks(range(heat.shape[0]), heat.index)
for i in range(heat.shape[0]):
    for j in range(heat.shape[1]):
        ax.text(j, i, f"{heat.values[i, j]:.2f}", ha="center", va="center", fontsize=9)
ax.set_title("Tesseract word F1: degradation (rows) × preprocessing (columns)")
fig.colorbar(im, ax=ax)
save_fig(fig, "day1_preprocessing_heatmap.png")

# %% [markdown]
# Read the heatmap row by row: preprocessing is not "always good". Thresholding a clean render can *lower* F1 (thin strokes break), deskew only matters on the skew rows, and denoising matters on the noise rows. Production pipelines pick steps per document source, or detect the problem first (estimate skew, measure noise) and only then fix it.

# %% [markdown]
# ## 9 · Do the other engines need preprocessing?

# %%
worst = DEGRADATIONS["all of the above"]
rows = []
for d in SYNTH:
    img = degrade_scan(d.pages[0], random.Random(SEED), **worst)
    for pipe_name in ("none", "deskew+otsu", "full"):
        pre = preprocess(img, PIPELINES[pipe_name])
        pre_rgb = cv2.cvtColor(pre, cv2.COLOR_GRAY2RGB)
        for name, fn in ENGINES.items():
            pred = fn(pre_rgb)
            rows.append({"engine": name, "pipeline": pipe_name, **word_f1(pred, d.gt_words), "cer": cer(pred, d.gt_words)})
ENG_PRE = pd.DataFrame(rows).groupby(["engine", "pipeline"])[["f1", "cer"]].mean().round(3)
ENG_PRE.to_csv(OUTPUT_DIR / "day1_engines_on_degraded.csv")
print(ENG_PRE.unstack("pipeline").to_string())

# %% [markdown]
# Deep-learning engines (EasyOCR, PaddleOCR) were trained on noisy, rotated crops and usually lose less on degraded input, and gain less from preprocessing, than Tesseract.

# %% [markdown]
# ## 10 · Tesseract page segmentation modes (PSM)
#
# PSM tells Tesseract what layout to expect. The default (3) runs full page-layout analysis; 4 assumes a single column of variable-size text; 6 one uniform block; 11 sparse text in any order. Forms and invoices are "sparse": labels and values float around the page.

# %%
rows = []
for psm in (3, 4, 6, 11):
    for dataset, d in [("synthetic", d) for d in synth_docs] + [("FUNSD", d) for d in FUNSD[:10]]:
        pred = ocr_tesseract(d["image"], psm=psm)
        rows.append({"psm": psm, "dataset": dataset, **word_f1(pred, d["words"]),
                     "cer": cer(pred, d["words"])})
PSM = pd.DataFrame(rows).groupby(["dataset", "psm"])[["f1", "cer"]].mean().round(3)
print(PSM.unstack("dataset").to_string())

# %% [markdown]
# ## 11 · Are confidence scores useful?
#
# If low-confidence words are the wrong ones, confidence can route a document to human review. Bucket every Tesseract word on FUNSD by confidence and check how often it matches a GT word.

# %%
rows = []
for d in FUNSD:
    gt_set = Counter(norm_word(w["text"]) for w in d["words"])
    for w in ocr_tesseract(d["image"]):
        rows.append({"conf": w["conf"], "correct": gt_set[norm_word(w["text"])] > 0})
CONF = pd.DataFrame(rows)
CONF["bucket"] = pd.cut(CONF["conf"], [-0.01, 0.3, 0.5, 0.7, 0.85, 0.95, 1.0])
calib = CONF.groupby("bucket", observed=True)["correct"].agg(["mean", "count"])
print(calib.round(3).to_string())

fig, ax = plt.subplots(figsize=(8, 4))
ax.bar(range(len(calib)), calib["mean"], tick_label=[str(b) for b in calib.index])
ax.set_ylabel("fraction of words that are correct")
ax.set_xlabel("Tesseract confidence bucket")
ax.set_title("Is Tesseract's confidence calibrated? (FUNSD)")
save_fig(fig, "day1_confidence_calibration.png")

# %% [markdown]
# ## 12 · Save results

# %%
summary = {
    "datasets": {"funsd_test_forms": len(FUNSD), "synthetic_invoices": len(SYNTH)},
    "engines": list(ENGINES),
    "engine_comparison": {f"{ds} | {eng}": row for (ds, eng), row in ENGINE_TABLE.to_dict("index").items()},
    "preprocessing_f1_heatmap": heat.round(4).to_dict(),
    "skew_estimator_mae_deg": float(np.mean(np.abs(np.array(est_angles) - angles))),
    "full_run": FULL_RUN,
}
with open(OUTPUT_DIR / "day1_summary.json", "w") as f:
    json.dump(summary, f, indent=2, default=float)
print(json.dumps(summary["engine_comparison"], indent=2, default=float))

# %% [markdown]
# ## Takeaways (fill in with your numbers)
#
# - FUNSD word F1: Tesseract ___ / EasyOCR ___ / PaddleOCR ___; latency ___ / ___ / ___ s per page.
# - On clean renders all engines are close to perfect; real scans separate them. Which one wins on FUNSD, and by how much?
# - Biggest preprocessing gain for Tesseract: ___ on the ___ row. Which step caused it?
# - A step that *hurt*: ___. Why?
# - Confidence below ___ means a word is wrong more often than right — a usable review trigger?
#
# **When each engine wins:** Tesseract for clean, printed, high-volume CPU workloads; EasyOCR for quick multilingual prototypes on GPU; PaddleOCR when accuracy on dense, small, or rotated text matters most. Tomorrow: plain OCR gives you text, but not which text is a *key* and which is its *value*. Layout-aware models fix that.
