"""
Week 4 · Day 4 — Key-value + table deep dive: hard cases and one common JSON schema for every tool.

Colab-first script (percent-format cells). Build the notebook with:
    python ../tools/py_to_ipynb.py day4_kv_tables/day4_kv_tables.py
Documents: the Day 3 synthetic invoice set (same seed, so cached managed responses are reused for free).
"""

# %% [markdown]
# # Week 4 · Day 4 — KV & Table Deep Dive
#
# **Goal:** push every approach through the hard cases — merged cells, multi-page tables, rotated scans, unfamiliar layouts — and normalize every output into **one JSON schema** so the results are directly comparable.
#
# | Hard case | Layout | What breaks | Fix built here |
# |---|---|---|---|
# | Rotated scan | `rotated` | OCR lines become diagonal; table rows mix | Projection-profile deskew before OCR |
# | Varied layout | `layout_b` | Labels differ ("Bill No.", "Amount Due"), table has no ruling lines | Label synonym rules; Table Transformer handles borderless tables |
# | Merged cells | `merged` | Two-level header ("Pricing" spans 2 columns), section rows spanning the table | TATR spanning-cell detection → `row_span`/`col_span` in the schema |
# | Multi-page table | `multipage` | Table split across pages with a repeated header | `stitch_tables`: same columns + repeated header → one logical table |
#
# **Approaches compared**
# 1. **Self-hosted, rules only:** Tesseract words + label rules + geometric table rules (no ML beyond OCR).
# 2. **Self-hosted, Table Transformer:** same KV rules, tables by Microsoft TATR (detection + structure recognition).
# 3. **Azure prebuilt-invoice** and 4. **AWS Textract** (AnalyzeExpense + AnalyzeDocument), from Day 3's cache or live calls.
#
# > Runtime: **T4 GPU** recommended (Table Transformer); CPU works, just slower.

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
pip_install("pytesseract==0.3.13", "transformers>=4.46,<5", "timm", "rapidfuzz>=3.9", "python-dateutil", "pydantic>=2.6",
            "azure-ai-documentintelligence==1.0.2", "boto3>=1.34")

# %% [markdown]
# ## 1 · Imports, configuration, and output folder

# %%
import json
from pathlib import Path

import matplotlib

IN_NOTEBOOK = "ipykernel" in sys.modules
if not IN_NOTEBOOK:
    matplotlib.use("Agg")  # headless when run as a script

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image
from rapidfuzz import fuzz

IN_COLAB = "google.colab" in sys.modules
try:
    HERE = Path(__file__).resolve().parent
except NameError:  # running inside a notebook
    HERE = Path.cwd()

OUTPUT_DIR = Path("/content/week4_outputs/day4") if IN_COLAB else HERE.parent / "results" / "day4"
RAW_DIR = OUTPUT_DIR.parent / "day3" / "raw"  # Day 3's response cache: same documents -> no new charges
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SEED = 7  # must match Day 3 so the documents (and cache keys) are identical
MAX_PAGES, MAX_USD = 40, 1.00
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"device={DEVICE} | managed-response cache: {RAW_DIR} ({'found' if RAW_DIR.exists() else 'empty'})")
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
# ## 2 · Helper code
#
# Five cells: preprocessing, the invoice generator, the common schema, the self-hosted pipeline and the managed-service helpers. Today's main reading is the schema cell (normalizers, `stitch_tables`, `line_items_from_table`) and the self-hosted pipeline cell.

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

# %%
# ---- doc_schema.py: helper code for this day (self-contained, no external project files) ----
# The Week 4 common JSON schema + the normalization helpers every extractor shares.
#
# Every approach (Tesseract rules + Table Transformer, Donut, Azure, Textract) ends in an
# `InvoiceDoc`. Only then are outputs comparable: same field names, ISO dates, floats for
# money, tables as explicit cells with row/col spans.
#
#     {
#       "doc_id": "inv_standard_00",
#       "fields": {"invoice_number": "INV-2026-4821", "invoice_date": "2026-03-14", "due_date": "2026-04-13",
#                  "vendor_name": "Northwind Traders", "bill_to": "Woodgrove Bank",
#                  "subtotal": 1520.0, "tax": 121.6, "total": 1641.6, "currency": "USD"},
#       "line_items": [{"description": "27-inch monitor", "quantity": 2, "unit_price": 310.0, "amount": 620.0}],
#       "tables": [{"page": 1, "n_rows": 6, "n_cols": 4,
#                   "cells": [{"row": 0, "col": 0, "row_span": 1, "col_span": 1, "text": "Description", "is_header": true}]}],
#       "meta": {"source": "azure", "pages": 1, "latency_s": 2.4, "cost_usd": 0.01}
#     }
import re
from datetime import date

from dateutil import parser as date_parser
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, model_validator
from rapidfuzz import fuzz

FIELD_NAMES = ("invoice_number", "invoice_date", "due_date", "vendor_name", "bill_to",
               "subtotal", "tax", "total", "currency")
MONEY_FIELDS = ("subtotal", "tax", "total")
DATE_FIELDS = ("invoice_date", "due_date")


class Cell(BaseModel):
    row: int = Field(ge=0)
    col: int = Field(ge=0)
    row_span: int = Field(default=1, ge=1)
    col_span: int = Field(default=1, ge=1)
    text: str = ""
    is_header: bool = False


class Table(BaseModel):
    page: int = Field(default=1, ge=1)
    n_rows: int = Field(ge=0)
    n_cols: int = Field(ge=0)
    cells: list[Cell] = []
    _span_pages: int = PrivateAttr(default=1)  # pages covered after stitching (not serialized)

    @model_validator(mode="after")
    def _cells_inside_grid(self):
        for c in self.cells:
            if c.row + c.row_span > self.n_rows or c.col + c.col_span > self.n_cols:
                raise ValueError(f"cell ({c.row},{c.col}) span {c.row_span}x{c.col_span} outside {self.n_rows}x{self.n_cols}")
        return self


class LineItem(BaseModel):
    description: str | None = None
    quantity: float | None = None
    unit_price: float | None = None
    amount: float | None = None


class Fields(BaseModel):
    invoice_number: str | None = None
    invoice_date: str | None = Field(default=None, pattern=r"^\d{4}-\d{2}-\d{2}$")
    due_date: str | None = Field(default=None, pattern=r"^\d{4}-\d{2}-\d{2}$")
    vendor_name: str | None = None
    bill_to: str | None = None
    subtotal: float | None = None
    tax: float | None = None
    total: float | None = None
    currency: str | None = Field(default=None, pattern=r"^[A-Z]{3}$")


class Meta(BaseModel):
    model_config = ConfigDict(extra="allow")
    source: str
    pages: int = 1
    latency_s: float = 0.0
    cost_usd: float = 0.0


class InvoiceDoc(BaseModel):
    doc_id: str
    fields: Fields
    line_items: list[LineItem] = []
    tables: list[Table] = []
    meta: Meta


# ---------------------------------------------------------------------------------------------
# Value normalizers: raw strings from any tool -> canonical Python values
# ---------------------------------------------------------------------------------------------
CURRENCY_SYMBOLS = {"$": "USD", "€": "EUR", "£": "GBP", "₹": "INR", "¥": "JPY"}
_MONEY_RE = re.compile(r"-?\d[\d.,\s]*")


def norm_text(s) -> str:
    """Lowercase, unify whitespace, drop surrounding punctuation. For comparing strings across tools."""
    if s is None:
        return ""
    s = re.sub(r"\s+", " ", str(s)).strip().lower()
    return s.strip(" .:;,#|-")


def parse_money(s) -> float | None:
    """'$1,234.50' / '1.234,50 €' / 'USD 99' / 99.5 -> float. None if no number is found."""
    if s is None:
        return None
    if isinstance(s, (int, float)):
        return float(s)
    m = _MONEY_RE.search(str(s))
    if not m:
        return None
    num = m.group(0).replace(" ", "").strip(".,")
    if "," in num and "." in num:
        # whichever separator comes last is the decimal point
        num = num.replace(",", "") if num.rfind(".") > num.rfind(",") else num.replace(".", "").replace(",", ".")
    elif "," in num:
        head, _, tail = num.rpartition(",")
        num = f"{head.replace(',', '')}.{tail}" if len(tail) == 2 else num.replace(",", "")
    elif num.count(".") > 1:
        head, _, tail = num.rpartition(".")
        num = f"{head.replace('.', '')}.{tail}"
    try:
        return round(float(num), 2)
    except ValueError:
        return None


def parse_number(s) -> float | None:
    if s is None:
        return None
    if isinstance(s, (int, float)):
        return float(s)
    m = re.search(r"-?\d+(?:[.,]\d+)?", str(s))
    return float(m.group(0).replace(",", ".")) if m else None


def parse_date(s) -> str | None:
    """Any common date string -> 'YYYY-MM-DD'. Month-first for ambiguous 03/04/2026 (US convention)."""
    if s is None:
        return None
    if isinstance(s, date):
        return s.isoformat()
    s = str(s).strip()
    if not re.search(r"\d", s):
        return None
    try:
        return date_parser.parse(s, fuzzy=True, dayfirst=False, default=None).date().isoformat()
    except (ValueError, OverflowError, TypeError, AttributeError):
        return None


def detect_currency(*texts) -> str | None:
    for t in texts:
        if not t:
            continue
        t = str(t)
        for sym, code in CURRENCY_SYMBOLS.items():
            if sym in t:
                return code
        m = re.search(r"\b(USD|EUR|GBP|INR|JPY|CAD|AUD)\b", t.upper())
        if m:
            return m.group(1)
    return None


def normalize_fields(raw: dict) -> dict:
    """Raw field strings (any tool) -> Fields-compatible dict with canonical types."""
    out = {}
    for name in FIELD_NAMES:
        val = raw.get(name)
        if name in MONEY_FIELDS:
            out[name] = parse_money(val)
        elif name in DATE_FIELDS:
            out[name] = parse_date(val)
        elif name == "currency":
            out[name] = val if isinstance(val, str) and re.fullmatch(r"[A-Z]{3}", val or "") else \
                detect_currency(val, raw.get("total"), raw.get("subtotal"))
        else:
            out[name] = re.sub(r"\s+", " ", str(val)).strip(" :#") if val not in (None, "") else None
    return out


# ---------------------------------------------------------------------------------------------
# Table helpers shared by every normalizer
# ---------------------------------------------------------------------------------------------
def table_to_grid(table: Table, fill_spans: bool = False) -> list[list[str]]:
    """Dense rows x cols text grid. Covered positions of a span get '' (or the span text if fill_spans)."""
    grid = [["" for _ in range(table.n_cols)] for _ in range(table.n_rows)]
    for c in table.cells:
        for r in range(c.row, c.row + c.row_span):
            for k in range(c.col, c.col + c.col_span):
                if (r, k) == (c.row, c.col) or fill_spans:
                    grid[r][k] = c.text
    return grid


def header_row_count(table: Table) -> int:
    rows = [c.row + c.row_span for c in table.cells if c.is_header]
    return max(rows) if rows else 0


def column_names(table: Table) -> list[str]:
    """Join all header texts covering each column: two-level headers become 'Pricing Unit Price'."""
    n_head = header_row_count(table) or 1
    names = [[] for _ in range(table.n_cols)]
    for c in sorted(table.cells, key=lambda c: (c.row, c.col)):
        if c.row < n_head:
            for k in range(c.col, c.col + c.col_span):
                if c.text and c.text not in names[k]:
                    names[k].append(c.text)
    return [" ".join(n) for n in names]


COLUMN_SYNONYMS = {
    "description": ["description", "item", "product", "service", "details", "particulars"],
    "quantity": ["qty", "quantity", "units", "hours"],
    "unit_price": ["unit price", "rate", "price per", "unit cost", "price"],
    "amount": ["amount", "line total", "total", "ext price", "net"],
}


def map_columns(names: list[str]) -> dict[str, int]:
    """Header names -> {line-item field: column index}. Most specific synonyms are tried first."""
    mapping: dict[str, int] = {}
    for fld in ("quantity", "unit_price", "amount", "description"):
        best, best_score = None, 0
        for i, name in enumerate(names):
            if i in mapping.values():
                continue
            low = norm_text(name)
            for syn in COLUMN_SYNONYMS[fld]:
                score = 100 if re.search(rf"\b{re.escape(syn)}\b", low) else fuzz.partial_ratio(syn, low)
                if score > best_score and score >= 85:
                    best, best_score = i, score
        if best is not None:
            mapping[fld] = best
    return mapping


def line_item_score(table: Table) -> int:
    """How much does this table look like a line-item table? (# of recognised columns, then # rows)."""
    return len(map_columns(column_names(table))) * 1000 + table.n_rows


def primary_table(tables: list[Table]) -> Table | None:
    return max(tables, key=line_item_score) if tables else None


def line_items_from_table(table: Table | None) -> list[LineItem]:
    """Body rows of the line-item table -> LineItem list. Section/spanning rows and empty rows are skipped."""
    if table is None:
        return []
    mapping = map_columns(column_names(table))
    if "description" not in mapping and "amount" not in mapping:
        return []
    grid = table_to_grid(table)
    spans = {(c.row, c.col) for c in table.cells if c.col_span >= max(2, table.n_cols - 1)}
    items = []
    for r in range(header_row_count(table) or 1, table.n_rows):
        if any((r, k) in spans for k in range(table.n_cols)):
            continue  # section header like "Hardware" spanning the row
        row = grid[r]
        get = lambda f: row[mapping[f]] if f in mapping else None  # noqa: E731
        item = LineItem(description=(get("description") or "").strip() or None, quantity=parse_number(get("quantity")),
                        unit_price=parse_money(get("unit_price")), amount=parse_money(get("amount")))
        if item.amount is None and item.unit_price is None:
            continue
        items.append(item)
    return items


def _same_header(a: Table, b: Table) -> bool:
    na, nb = [norm_text(x) for x in column_names(a)], [norm_text(x) for x in column_names(b)]
    return len(na) == len(nb) and fuzz.ratio(" ".join(na), " ".join(nb)) >= 85


def stitch_tables(tables: list[Table]) -> list[Table]:
    """
    Merge a table that continues on the next page into one logical table.

    Rule: table T2 on page p+1 continues T1 on page p when they have the same column count
    and either T2 repeats T1's header (header rows dropped) or T2 has no header at all.
    """
    out: list[Table] = []
    for t in sorted(tables, key=lambda t: t.page):
        prev = out[-1] if out else None
        if prev is not None and t.page == prev.page + prev._span_pages and t.n_cols == prev.n_cols:
            repeats = header_row_count(t) > 0 and _same_header(prev, t)
            if repeats or header_row_count(t) == 0:
                skip = header_row_count(t) if repeats else 0
                cells = list(prev.cells)
                for c in t.cells:
                    if c.row >= skip:
                        cells.append(c.model_copy(update={"row": c.row - skip + prev.n_rows, "is_header": False}))
                merged = Table(page=prev.page, n_rows=prev.n_rows + t.n_rows - skip, n_cols=prev.n_cols, cells=cells)
                merged._span_pages = prev._span_pages + 1
                out[-1] = merged
                continue
        out.append(t)
    return out


def finalize(doc_id: str, raw_fields: dict, tables: list[Table], source: str, pages: int,
             latency_s: float = 0.0, cost_usd: float = 0.0, line_items: list | None = None, **meta) -> InvoiceDoc:
    """Last step of every normalizer: stitch tables, derive line items if the tool gave none, validate."""
    tables = stitch_tables(tables)
    if line_items is None:
        line_items = line_items_from_table(primary_table(tables))
    return InvoiceDoc(doc_id=doc_id, fields=Fields(**normalize_fields(raw_fields)), line_items=line_items, tables=tables,
                      meta=Meta(source=source, pages=pages, latency_s=round(latency_s, 3), cost_usd=round(cost_usd, 4), **meta))

# %%
# ---- selfhosted.py: helper code for this day (self-contained, no external project files) ----
# Self-hosted invoice extraction (Week 4, Days 4–5): everything runs on your own machine/GPU.
#
#     page image -> deskew -> Tesseract words -> phrases
#                -> key-value rules (label synonyms + "value is right of / below the label")
#                -> Table Transformer: table detection -> structure (rows, columns, header, spanning cells)
#                -> words assigned to cells -> stitched across pages -> InvoiceDoc (common schema)
import re
import time

import numpy as np
import pytesseract
import torch
from PIL import Image
from rapidfuzz import fuzz



# ---------------------------------------------------------------------------------------------
# OCR words -> lines -> phrases
# ---------------------------------------------------------------------------------------------
def ocr_words(img, psm: int = 3, lang: str = "eng", min_conf: float = 0) -> list[dict]:
    """Tesseract word boxes: [{"text", "conf", "bbox": [x0, y0, x1, y1]}]."""
    data = pytesseract.image_to_data(img, lang=lang, config=f"--psm {psm}", output_type=pytesseract.Output.DICT)
    words = []
    for i, text in enumerate(data["text"]):
        text = text.strip()
        conf = float(data["conf"][i])
        if not text or conf < min_conf:
            continue
        x, y, w, h = data["left"][i], data["top"][i], data["width"][i], data["height"][i]
        words.append({"text": text, "conf": conf, "bbox": [x, y, x + w, y + h]})
    return words


def group_lines(words: list[dict], y_tol: float = 0.6) -> list[list[dict]]:
    """Cluster words into visual lines by vertical centre (tolerance = y_tol x median word height)."""
    if not words:
        return []
    med_h = float(np.median([w["bbox"][3] - w["bbox"][1] for w in words]))
    lines: list[list[dict]] = []
    for w in sorted(words, key=lambda w: (w["bbox"][1] + w["bbox"][3]) / 2):
        cy = (w["bbox"][1] + w["bbox"][3]) / 2
        if lines:
            last = lines[-1]
            ly = np.mean([(v["bbox"][1] + v["bbox"][3]) / 2 for v in last])
            if abs(cy - ly) <= y_tol * med_h:
                last.append(w)
                continue
        lines.append([w])
    return [sorted(line, key=lambda w: w["bbox"][0]) for line in lines]


def _merge_box(ws):
    return [min(w["bbox"][0] for w in ws), min(w["bbox"][1] for w in ws),
            max(w["bbox"][2] for w in ws), max(w["bbox"][3] for w in ws)]


def split_phrases(line: list[dict], gap_factor: float = 1.2) -> list[dict]:
    """Split a line where the horizontal gap exceeds gap_factor x word height: 'Invoice Number:' | 'INV-2026-0412'."""
    phrases, cur = [], [line[0]]
    for prev, w in zip(line, line[1:]):
        h = max(prev["bbox"][3] - prev["bbox"][1], w["bbox"][3] - w["bbox"][1])
        if w["bbox"][0] - prev["bbox"][2] > gap_factor * h:
            phrases.append(cur)
            cur = []
        cur.append(w)
    phrases.append(cur)
    return [{"text": " ".join(w["text"] for w in p), "bbox": _merge_box(p), "words": p,
             "height": float(np.median([w["bbox"][3] - w["bbox"][1] for w in p]))} for p in phrases]


def page_phrases(words: list[dict]) -> list[list[dict]]:
    return [split_phrases(line) for line in group_lines(words)]


# ---------------------------------------------------------------------------------------------
# Key-value extraction with label rules
# ---------------------------------------------------------------------------------------------
FIELD_LABELS = {  # anchored at the start of a phrase, so "Due Date" never matches the generic "Date"
    "invoice_number": r"(invoice\s*(number|no\.?|num|#|id)|bill\s*(no\.?|number|#)|reference|ref\.?\s*no\.?)",
    "due_date": r"(due\s*date|payment\s*due|pay\s*by|date\s*due)",
    "invoice_date": r"(invoice\s*date|date\s*of\s*issue|issue\s*date|bill\s*date|date)",
    "bill_to": r"(bill(ed)?\s*to|customer|sold\s*to|invoice\s*to)",
    "subtotal": r"(sub\s*-?\s*total|net\s*amount|net\s*total)",
    "tax": r"(tax|vat|gst|sales\s*tax)",
    "total": r"(total\s*due|amount\s*due|grand\s*total|balance\s*due|total)",
}
TITLE_WORDS = re.compile(r"^(tax\s*|sales\s*)?(invoice|bill|receipt|statement)$", re.I)
PERCENT = re.compile(r"\(?\s*\d+(\.\d+)?\s*%\s*\)?")  # "Tax (8%):" -> the 8 is a rate, not the value


def _inside(box, region, min_frac=0.5) -> bool:
    ix = max(0, min(box[2], region[2]) - max(box[0], region[0]))
    iy = max(0, min(box[3], region[3]) - max(box[1], region[1]))
    area = max(1, (box[2] - box[0]) * (box[3] - box[1]))
    return ix * iy / area >= min_frac


def _value_after_label(phrases_by_line, li, pi, rest: str) -> tuple[str | None, list]:
    """Value = rest of the label phrase, else the next phrase on the line, else the phrase just below."""
    phrase = phrases_by_line[li][pi]
    rest = rest.strip(" :#.-")
    if rest:
        return rest, phrase["bbox"]
    line = phrases_by_line[li]
    if pi + 1 < len(line):
        return line[pi + 1]["text"].strip(" :"), line[pi + 1]["bbox"]
    x0, x1 = phrase["bbox"][0], phrase["bbox"][2]
    for below in phrases_by_line[li + 1: li + 3]:
        for cand in below:
            if cand["bbox"][0] < x1 and cand["bbox"][2] > x0:
                return cand["text"].strip(" :"), cand["bbox"]
    return None, []


def extract_fields(words: list[dict], exclude_regions: list | None = None, page_index: int = 0) -> dict:
    """Raw field strings from one page. Header fields keep the first match; totals keep the last (bottom-most)."""
    exclude_regions = exclude_regions or []
    lines = [[p for p in line if not any(_inside(p["bbox"], r) for r in exclude_regions)] for line in page_phrases(words)]
    lines = [line for line in lines if line]
    found: dict = {}
    for li, line in enumerate(lines):
        for pi, phrase in enumerate(line):
            text = PERCENT.sub("", phrase["text"]).strip()
            low = text.lower()
            if TITLE_WORDS.match(low):
                continue
            for name, pattern in FIELD_LABELS.items():
                m = re.match(rf"^{pattern}\b[\s:#.]*", low)
                if name == "total" and re.match(r"^sub", low):
                    continue
                if not m:
                    continue
                value, _ = _value_after_label(lines, li, pi, text[m.end():])
                if value is None:
                    break
                if name in ("subtotal", "tax", "total"):
                    if not re.search(r"\d", value):
                        break
                    found[name] = value  # keep the last one on the page
                elif name not in found:
                    found[name] = value
                break
    if page_index == 0 and lines:
        # Vendor name: the tallest phrase in the top quarter that is not the document title.
        page_h = max(p["bbox"][3] for line in lines for p in line)
        cands = [p for line in lines for p in line
                 if p["bbox"][1] < 0.25 * page_h and not TITLE_WORDS.match(p["text"].strip()) and len(p["text"]) > 3]
        if cands:
            found["vendor_name"] = max(cands, key=lambda p: p["height"])["text"]
    return found


# ---------------------------------------------------------------------------------------------
# Tables with Microsoft Table Transformer (TATR)
# ---------------------------------------------------------------------------------------------
_IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
_IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def _to_tatr_tensor(img: Image.Image, max_size: int) -> torch.Tensor:
    w, h = img.size
    scale = max_size / max(w, h)
    img = img.convert("RGB").resize((round(w * scale), round(h * scale)), Image.BILINEAR)
    t = torch.from_numpy(np.asarray(img).copy()).permute(2, 0, 1).float() / 255
    return ((t - _IMAGENET_MEAN) / _IMAGENET_STD).unsqueeze(0)


def _iou(a, b) -> float:
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def _dedupe(objs: list[dict], thr: float = 0.5) -> list[dict]:
    keep = []
    for o in sorted(objs, key=lambda o: -o["score"]):
        if all(_iou(o["bbox"], k["bbox"]) < thr for k in keep):
            keep.append(o)
    return keep


class TableExtractor:
    """Table Transformer detection + structure recognition, loaded lazily on first use."""

    DET_ID = "microsoft/table-transformer-detection"
    STRUCT_ID = "microsoft/table-transformer-structure-recognition-v1.1-all"

    def __init__(self, device=None, det_threshold: float = 0.7, struct_threshold: float = 0.5):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.det_threshold, self.struct_threshold = det_threshold, struct_threshold
        self._det = self._struct = None

    def _load(self):
        if self._det is None:
            from transformers import TableTransformerForObjectDetection

            self._det = TableTransformerForObjectDetection.from_pretrained(self.DET_ID).to(self.device).eval()
            self._struct = TableTransformerForObjectDetection.from_pretrained(self.STRUCT_ID).to(self.device).eval()

    @torch.no_grad()
    def _predict(self, model, img: Image.Image, max_size: int, threshold: float) -> list[dict]:
        out = model(pixel_values=_to_tatr_tensor(img, max_size).to(self.device))
        probs = out.logits.softmax(-1)[0, :, :-1]  # last class = "no object"
        scores, labels = probs.max(-1)
        w, h = img.size
        objs = []
        for s, l, (cx, cy, bw, bh) in zip(scores.tolist(), labels.tolist(), out.pred_boxes[0].tolist()):
            if s >= threshold:
                objs.append({"label": model.config.id2label[l], "score": s,
                             "bbox": [(cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h]})
        return objs

    def detect(self, img: Image.Image) -> list[list[float]]:
        self._load()
        objs = self._predict(self._det, img, 800, self.det_threshold)
        return [o["bbox"] for o in _dedupe([o for o in objs if o["label"].startswith("table")])]

    def structure(self, crop: Image.Image) -> dict[str, list[dict]]:
        self._load()
        objs = self._predict(self._struct, crop, 1000, self.struct_threshold)
        groups: dict[str, list[dict]] = {}
        for o in objs:
            groups.setdefault(o["label"], []).append(o)
        rows = sorted(_dedupe(groups.get("table row", [])), key=lambda o: o["bbox"][1])
        cols = sorted(_dedupe(groups.get("table column", [])), key=lambda o: o["bbox"][0])
        return {"rows": rows, "cols": cols, "header": groups.get("table column header", []),
                "spans": _dedupe(groups.get("table spanning cell", []), 0.3)}


def _overlap_1d(a0, a1, b0, b1) -> float:
    return max(0.0, min(a1, b1) - max(a0, b0))


def cells_from_structure(struct: dict, words: list[dict], offset=(0, 0), page: int = 1) -> Table | None:
    """Rows x columns grid -> cells; spanning cells merge the grid cells they cover; words go to the best cell."""
    ox, oy = offset
    rows = [[o["bbox"][0] + ox, o["bbox"][1] + oy, o["bbox"][2] + ox, o["bbox"][3] + oy] for o in struct["rows"]]
    cols = [[o["bbox"][0] + ox, o["bbox"][1] + oy, o["bbox"][2] + ox, o["bbox"][3] + oy] for o in struct["cols"]]
    if len(rows) < 2 or len(cols) < 2:
        return None
    texts: dict[tuple, list] = {}
    for w in words:
        x0, y0, x1, y1 = w["bbox"]
        r_ov = [_overlap_1d(y0, y1, r[1], r[3]) for r in rows]
        c_ov = [_overlap_1d(x0, x1, c[0], c[2]) for c in cols]
        if max(r_ov) <= 0 or max(c_ov) <= 0:
            continue
        texts.setdefault((int(np.argmax(r_ov)), int(np.argmax(c_ov))), []).append(w)

    covered: dict[tuple, tuple] = {}
    for s in struct["spans"]:
        sx0, sy0, sx1, sy1 = s["bbox"][0] + ox, s["bbox"][1] + oy, s["bbox"][2] + ox, s["bbox"][3] + oy
        rr = [i for i, r in enumerate(rows) if _overlap_1d(sy0, sy1, r[1], r[3]) > 0.5 * (r[3] - r[1])]
        cc = [i for i, c in enumerate(cols) if _overlap_1d(sx0, sx1, c[0], c[2]) > 0.5 * (c[2] - c[0])]
        if not rr or not cc or len(rr) * len(cc) < 2:
            continue
        block = [(r, c) for r in range(min(rr), max(rr) + 1) for c in range(min(cc), max(cc) + 1)]
        if any(b in covered for b in block):
            continue
        for b in block:
            covered[b] = (min(rr), min(cc), max(rr) - min(rr) + 1, max(cc) - min(cc) + 1)

    header_rows = set()
    for hdr in struct["header"]:
        hy0, hy1 = hdr["bbox"][1] + oy, hdr["bbox"][3] + oy
        header_rows |= {i for i, r in enumerate(rows) if _overlap_1d(hy0, hy1, r[1], r[3]) > 0.5 * (r[3] - r[1])}
    if not header_rows:
        header_rows = {0}

    def join(ws):
        return " ".join(w["text"] for w in sorted(ws, key=lambda w: (round(w["bbox"][1] / 15), w["bbox"][0])))

    cells, done = [], set()
    for r in range(len(rows)):
        for c in range(len(cols)):
            if (r, c) in done:
                continue
            if (r, c) in covered:
                r0, c0, rs, cs = covered[(r, c)]
                block = [(i, j) for i in range(r0, r0 + rs) for j in range(c0, c0 + cs)]
                done.update(block)
                ws = [w for b in block for w in texts.get(b, [])]
                cells.append(Cell(row=r0, col=c0, row_span=rs, col_span=cs, text=join(ws), is_header=r0 in header_rows))
            else:
                done.add((r, c))
                cells.append(Cell(row=r, col=c, text=join(texts.get((r, c), [])), is_header=r in header_rows))
    table = Table(page=page, n_rows=len(rows), n_cols=len(cols), cells=cells)
    return _drop_empty(table)


def _drop_empty(table: Table) -> Table:
    """Remove grid rows/columns with no text at all (TATR sometimes predicts a sliver row/column)."""
    used_rows = {r for c in table.cells if c.text for r in range(c.row, c.row + c.row_span)}
    used_cols = {k for c in table.cells if c.text for k in range(c.col, c.col + c.col_span)}
    rmap = {r: i for i, r in enumerate(sorted(used_rows))}
    cmap = {k: i for i, k in enumerate(sorted(used_cols))}
    cells = []
    for c in table.cells:
        rr = [rmap[r] for r in range(c.row, c.row + c.row_span) if r in rmap]
        cc = [cmap[k] for k in range(c.col, c.col + c.col_span) if k in cmap]
        if rr and cc and (c.text or (c.row in rmap and c.col in cmap)):
            cells.append(c.model_copy(update={"row": rr[0], "col": cc[0], "row_span": len(rr), "col_span": len(cc)}))
    return Table(page=table.page, n_rows=len(rmap), n_cols=len(cmap), cells=cells)


# ---------------------------------------------------------------------------------------------
# Fallback: rule-based table from word geometry (no model)
# ---------------------------------------------------------------------------------------------
HEADER_HINTS = ("description", "item", "product", "qty", "quantity", "units", "price", "rate", "amount", "total")
STOP_HINTS = re.compile(r"^(sub\s*total|net\s*amount|net\s*total|thank you|tax|vat)", re.I)


def find_table_region(words: list[dict], page_size) -> list[float] | None:
    """Header line (>= 2 column keywords) down to the first totals line."""
    lines = group_lines(words)
    top = None
    for i, line in enumerate(lines):
        low = " ".join(w["text"].lower() for w in line)
        if top is None and sum(h in low for h in HEADER_HINTS) >= 2:
            top = i
        elif top is not None and STOP_HINTS.match(split_phrases(line)[0]["text"].lower().strip()):
            box = _merge_box([w for ln in lines[top:i] for w in ln])
            return [0, box[1] - 5, page_size[0], box[3] + 5]
    if top is not None:
        box = _merge_box([w for ln in lines[top:] for w in ln])
        return [0, box[1] - 5, page_size[0], box[3] + 5]
    return None


def rule_table(words: list[dict], region, page: int = 1) -> Table | None:
    """Columns from the header phrases' x-ranges; one row per text line; single-phrase text rows span the table."""
    inside = [w for w in words if _inside(w["bbox"], region)]
    lines = group_lines(inside)
    if len(lines) < 2:
        return None
    header = split_phrases(lines[0])
    if len(header) < 2:
        return None
    bounds = [(header[i]["bbox"][2] + header[i + 1]["bbox"][0]) / 2 for i in range(len(header) - 1)]
    col_of = lambda x: int(np.searchsorted(bounds, x))  # noqa: E731
    cells = [Cell(row=0, col=i, text=p["text"], is_header=True) for i, p in enumerate(header)]
    n_cols = len(header)
    for r, line in enumerate(lines[1:], start=1):
        phrases = split_phrases(line)
        if len(phrases) == 1 and not re.search(r"\d", phrases[0]["text"]):
            cells.append(Cell(row=r, col=0, col_span=n_cols, text=phrases[0]["text"]))
            continue
        row_txt: dict[int, list[str]] = {}
        for w in line:
            row_txt.setdefault(col_of((w["bbox"][0] + w["bbox"][2]) / 2), []).append(w["text"])
        cells += [Cell(row=r, col=k, text=" ".join(row_txt.get(k, []))) for k in range(n_cols)]
    return Table(page=page, n_rows=len(lines), n_cols=n_cols, cells=cells)


# ---------------------------------------------------------------------------------------------
# Full self-hosted pipeline
# ---------------------------------------------------------------------------------------------
def extract_tables(img: Image.Image, words: list[dict], extractor: "TableExtractor | None", page: int = 1,
                   pad: int = 20) -> tuple[list[Table], list[list[float]], str]:
    """Returns (tables, table regions, method used)."""
    regions = extractor.detect(img) if extractor is not None else []
    method = "tatr"
    if not regions:
        region = find_table_region(words, img.size)
        regions = [region] if region else []
        method = "keyword-region+tatr" if extractor is not None else "rules"
    tables = []
    for reg in regions:
        x0, y0 = max(0, int(reg[0]) - pad), max(0, int(reg[1]) - pad)
        x1, y1 = min(img.width, int(reg[2]) + pad), min(img.height, int(reg[3]) + pad)
        table = None
        if extractor is not None:
            struct = extractor.structure(img.crop((x0, y0, x1, y1)))
            in_words = [w for w in words if _inside(w["bbox"], [x0, y0, x1, y1])]
            table = cells_from_structure(struct, in_words, offset=(x0, y0), page=page)
        if table is None or table.n_rows < 2 or table.n_cols < 2:
            table = rule_table(words, [x0, y0, x1, y1], page=page)
            method = "rules"
        if table is not None:
            tables.append(table)
    return tables, regions, method


def run_selfhosted(doc_id: str, pages: list[Image.Image], extractor: "TableExtractor | None" = None,
                   do_deskew: bool = True, gpu_usd_per_hour: float = 0.35) -> tuple[InvoiceDoc, dict]:
    """Whole pipeline for one document. Returns (InvoiceDoc, debug info per page)."""
    t0 = time.perf_counter()
    raw: dict = {}
    tables: list[Table] = []
    debug = []
    for i, page in enumerate(pages):
        arr = np.asarray(page.convert("RGB"))
        angle = 0.0
        if do_deskew:
            arr, angle = deskew(arr)
        img = Image.fromarray(arr)
        words = ocr_words(img)
        page_tables, regions, method = extract_tables(img, words, extractor, page=i + 1)
        fields = extract_fields(words, exclude_regions=regions, page_index=i)
        for k, v in fields.items():
            if k in ("subtotal", "tax", "total"):
                raw[k] = v  # totals: last page wins
            else:
                raw.setdefault(k, v)
        tables += page_tables
        debug.append({"page": i + 1, "skew": angle, "n_words": len(words), "table_method": method,
                      "regions": regions, "image": img, "words": words})
    latency = time.perf_counter() - t0
    doc = finalize(doc_id, raw, tables, source="selfhosted", pages=len(pages), latency_s=latency,
                   cost_usd=latency / 3600 * gpu_usd_per_hour, raw_fields=raw)
    return doc, {"pages": debug}

# %%
# ---- managed.py: helper code for this day (self-contained, no external project files) ----
# Managed document-intelligence services (Week 4, Days 3–5): Azure AI Document Intelligence and AWS Textract.
#
# Three rules this module enforces, because cloud OCR bills per page:
#   1. Every call goes through a `CostMeter` with a page cap and a USD cap. Exceeding either raises before calling.
#   2. Every raw response is cached as JSON on disk. Re-running a notebook re-reads the cache and costs nothing.
#   3. Parsers work on the cached plain-dict form, so normalization can be developed offline.
#
# Credentials come from environment variables or Colab secrets (key icon in the left sidebar):
#     AZURE_DI_ENDPOINT, AZURE_DI_KEY
#     AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_DEFAULT_REGION
import json
import os
import time
from pathlib import Path


# List prices in USD per page (pay-as-you-go, first pricing tier). Check the provider pricing pages before
# relying on them; they change and vary by region. Free tiers (Azure F0: 500 pages/month; Textract: limited
# free pages for the first 3 months) make a capped test set cost nothing.
PRICE_PER_PAGE = {
    "azure:prebuilt-read": 0.0015,
    "azure:prebuilt-layout": 0.01,
    "azure:prebuilt-invoice": 0.01,
    "azure:prebuilt-receipt": 0.01,
    "azure:custom": 0.03,
    "textract:expense": 0.01,
    "textract:tables": 0.015,
    "textract:forms": 0.05,
    "textract:tables+forms": 0.065,
}


def get_secret(name: str) -> str | None:
    """Environment variable first, then Colab secrets."""
    if os.environ.get(name):
        return os.environ[name]
    try:
        from google.colab import userdata  # type: ignore

        return userdata.get(name)
    except Exception:
        return None


class BudgetExceeded(RuntimeError):
    pass


class CostMeter:
    """Hard cap on pages and dollars across all managed calls in a session."""

    def __init__(self, max_pages: int = 40, max_usd: float = 1.0):
        self.max_pages, self.max_usd = max_pages, max_usd
        self.pages, self.usd, self.log = 0, 0.0, []

    def charge(self, sku: str, pages: int, doc_id: str = "") -> float:
        cost = PRICE_PER_PAGE[sku] * pages
        if self.pages + pages > self.max_pages or self.usd + cost > self.max_usd:
            raise BudgetExceeded(f"{sku} x{pages} would exceed the cap "
                                 f"({self.pages}/{self.max_pages} pages, ${self.usd:.3f}/${self.max_usd:.2f})")
        self.pages += pages
        self.usd += cost
        self.log.append({"sku": sku, "pages": pages, "usd": cost, "doc_id": doc_id})
        return cost

    def refund(self, sku: str, pages: int) -> None:
        """Undo a charge whose call failed (rejected requests are not billed)."""
        self.pages -= pages
        self.usd -= PRICE_PER_PAGE[sku] * pages
        self.log.append({"sku": sku, "pages": -pages, "usd": -PRICE_PER_PAGE[sku] * pages, "doc_id": "refund"})

    def summary(self) -> str:
        return f"{self.pages} billable pages, ${self.usd:.3f} (caps: {self.max_pages} pages, ${self.max_usd:.2f})"


def _cached(cache_path: Path | None, call, meter: "CostMeter | None" = None, sku: str = "", pages: int = 0):
    """Return (response_dict, latency_s, from_cache). Latency of a cached call is the latency recorded when it ran."""
    if cache_path is not None and cache_path.exists():
        blob = json.loads(cache_path.read_text(encoding="utf-8"))
        return blob["response"], blob["latency_s"], True
    t0 = time.perf_counter()
    try:
        response = call()
    except Exception:
        if meter is not None:
            meter.refund(sku, pages)
        raise
    latency = time.perf_counter() - t0
    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps({"latency_s": latency, "response": response}, default=str), encoding="utf-8")
    return response, latency, False


# ---------------------------------------------------------------------------------------------
# Azure AI Document Intelligence (SDK: azure-ai-documentintelligence 1.x, API 2024-11-30)
# ---------------------------------------------------------------------------------------------
def azure_client():
    endpoint, key = get_secret("AZURE_DI_ENDPOINT"), get_secret("AZURE_DI_KEY")
    if not endpoint or not key:
        return None
    from azure.ai.documentintelligence import DocumentIntelligenceClient
    from azure.core.credentials import AzureKeyCredential

    return DocumentIntelligenceClient(endpoint=endpoint, credential=AzureKeyCredential(key))


def azure_analyze(client, model_id: str, data: bytes, n_pages: int, meter: CostMeter, cache_path: Path | None = None,
                  doc_id: str = "", key_value_pairs: bool = False) -> tuple[dict, float, float]:
    """Run one Azure model on a PDF/image. Returns (result as dict, latency_s, cost_usd)."""
    sku = f"azure:{model_id}" if f"azure:{model_id}" in PRICE_PER_PAGE else "azure:custom"
    if cache_path is not None and cache_path.exists():
        resp, latency, _ = _cached(cache_path, None)
        return resp, latency, PRICE_PER_PAGE[sku] * n_pages

    def call():
        from azure.ai.documentintelligence.models import AnalyzeDocumentRequest, DocumentAnalysisFeature

        kwargs = {"features": [DocumentAnalysisFeature.KEY_VALUE_PAIRS]} if key_value_pairs else {}
        poller = client.begin_analyze_document(model_id, AnalyzeDocumentRequest(bytes_source=data), **kwargs)
        return poller.result().as_dict()

    cost = meter.charge(sku, n_pages, doc_id)
    resp, latency, _ = _cached(cache_path, call, meter, sku, n_pages)
    return resp, latency, cost


def _az_value(f: dict | None):
    """DocumentField dict -> plain Python value (amounts as float, dates as ISO strings)."""
    if not f:
        return None
    t = f.get("type")
    if t == "currency" and f.get("valueCurrency"):
        return f["valueCurrency"].get("amount")
    if t == "date" and f.get("valueDate"):
        return f["valueDate"]
    if t == "number" and f.get("valueNumber") is not None:
        return f["valueNumber"]
    if t == "string" and f.get("valueString"):
        return f["valueString"]
    if t == "address" and f.get("content"):
        return f["content"]
    return f.get("content")


AZURE_INVOICE_MAP = {"invoice_number": ["InvoiceId"], "invoice_date": ["InvoiceDate"], "due_date": ["DueDate"],
                     "vendor_name": ["VendorName"], "bill_to": ["CustomerName", "BillingAddressRecipient"],
                     "subtotal": ["SubTotal"], "tax": ["TotalTax"], "total": ["InvoiceTotal", "AmountDue"]}


def azure_fields(result: dict) -> dict:
    docs = result.get("documents") or []
    if not docs:
        return {}
    fields = docs[0].get("fields", {})
    raw = {}
    for name, keys in AZURE_INVOICE_MAP.items():
        for k in keys:
            if fields.get(k):
                raw[name] = _az_value(fields[k])
                break
    for k in ("InvoiceTotal", "AmountDue", "SubTotal"):
        cur = (fields.get(k) or {}).get("valueCurrency") or {}
        if cur.get("currencyCode"):
            raw["currency"] = cur["currencyCode"]
            break
    return raw


def azure_line_items(result: dict) -> list:
    docs = result.get("documents") or []
    items = ((docs[0].get("fields") or {}).get("Items") or {}).get("valueArray", []) if docs else []
    out = []
    for it in items:
        obj = it.get("valueObject") or {}
        out.append(LineItem(description=_az_value(obj.get("Description")), quantity=parse_number(_az_value(obj.get("Quantity"))),
                            unit_price=parse_money(_az_value(obj.get("UnitPrice"))), amount=parse_money(_az_value(obj.get("Amount")))))
    return out


def azure_tables(result: dict) -> list:
    tables = []
    for t in result.get("tables") or []:
        page = ((t.get("boundingRegions") or [{}])[0]).get("pageNumber", 1)
        cells = [Cell(row=c["rowIndex"], col=c["columnIndex"], row_span=c.get("rowSpan") or 1, col_span=c.get("columnSpan") or 1,
                      text=(c.get("content") or "").strip(), is_header=c.get("kind") == "columnHeader") for c in t.get("cells", [])]
        tables.append(Table(page=page, n_rows=t["rowCount"], n_cols=t["columnCount"], cells=cells))
    return tables


def azure_kv_pairs(result: dict) -> list[tuple[str, str, float]]:
    """prebuilt-layout with the keyValuePairs feature: generic (key, value, confidence) triples."""
    return [((kv.get("key") or {}).get("content", ""), (kv.get("value") or {}).get("content", ""), kv.get("confidence", 0.0))
            for kv in result.get("keyValuePairs") or []]


def azure_to_common(doc_id: str, result: dict, latency_s: float, cost_usd: float) -> "InvoiceDoc":
    items = azure_line_items(result)
    return finalize(doc_id, azure_fields(result), azure_tables(result), source="azure", pages=len(result.get("pages") or [1]),
                    latency_s=latency_s, cost_usd=cost_usd, line_items=items or None)


# ---------------------------------------------------------------------------------------------
# AWS Textract (boto3). Synchronous APIs take one page per call for images.
# ---------------------------------------------------------------------------------------------
def textract_client():
    if not get_secret("AWS_ACCESS_KEY_ID") or not get_secret("AWS_SECRET_ACCESS_KEY"):
        return None
    import boto3

    return boto3.client("textract", region_name=get_secret("AWS_DEFAULT_REGION") or "us-east-1",
                        aws_access_key_id=get_secret("AWS_ACCESS_KEY_ID"),
                        aws_secret_access_key=get_secret("AWS_SECRET_ACCESS_KEY"))


def textract_analyze_document(client, png: bytes, meter: CostMeter, features=("TABLES", "FORMS"),
                              cache_path: Path | None = None, doc_id: str = "") -> tuple[dict, float, float]:
    sku = "textract:" + "+".join(f.lower() for f in sorted(features, reverse=True))
    if cache_path is not None and cache_path.exists():
        resp, latency, _ = _cached(cache_path, None)
        return resp, latency, PRICE_PER_PAGE[sku]
    cost = meter.charge(sku, 1, doc_id)
    resp, latency, _ = _cached(cache_path, lambda: client.analyze_document(Document={"Bytes": png}, FeatureTypes=list(features)),
                               meter, sku, 1)
    return resp, latency, cost


def textract_analyze_expense(client, png: bytes, meter: CostMeter, cache_path: Path | None = None,
                             doc_id: str = "") -> tuple[dict, float, float]:
    if cache_path is not None and cache_path.exists():
        resp, latency, _ = _cached(cache_path, None)
        return resp, latency, PRICE_PER_PAGE["textract:expense"]
    cost = meter.charge("textract:expense", 1, doc_id)
    resp, latency, _ = _cached(cache_path, lambda: client.analyze_expense(Document={"Bytes": png}), meter, "textract:expense", 1)
    return resp, latency, cost


def _children(block: dict, by_id: dict, rel_type: str = "CHILD") -> list[dict]:
    ids = [i for r in block.get("Relationships", []) if r["Type"] == rel_type for i in r["Ids"]]
    return [by_id[i] for i in ids if i in by_id]


def _block_text(block: dict, by_id: dict) -> str:
    words = [b["Text"] for b in _children(block, by_id) if b["BlockType"] == "WORD"]
    marks = [b.get("SelectionStatus", "") for b in _children(block, by_id) if b["BlockType"] == "SELECTION_ELEMENT"]
    return " ".join(words + [m for m in marks if m == "SELECTED"])


def textract_kv_pairs(resp: dict) -> list[tuple[str, str, float]]:
    by_id = {b["Id"]: b for b in resp.get("Blocks", [])}
    pairs = []
    for b in by_id.values():
        if b["BlockType"] == "KEY_VALUE_SET" and "KEY" in b.get("EntityTypes", []):
            values = _children(b, by_id, "VALUE")
            pairs.append((_block_text(b, by_id), " ".join(_block_text(v, by_id) for v in values), b.get("Confidence", 0.0)))
    return pairs


def textract_tables(resp: dict, page: int = 1) -> list:
    """TABLE -> CELL (+ MERGED_CELL spans). Merged cells replace the cells they cover, with their joined text."""
    by_id = {b["Id"]: b for b in resp.get("Blocks", [])}
    tables = []
    for t in (b for b in by_id.values() if b["BlockType"] == "TABLE"):
        cells = [c for c in _children(t, by_id) if c["BlockType"] == "CELL"]
        merged = [m for m in _children(t, by_id, "MERGED_CELL") if m["BlockType"] == "MERGED_CELL"]
        if not cells:
            continue
        covered, out = set(), []
        for m in merged:
            kids = _children(m, by_id)
            covered |= {k["Id"] for k in kids}
            kid_texts = [_block_text(k, by_id) for k in sorted(kids, key=lambda k: (k["RowIndex"], k["ColumnIndex"]))]
            text = " ".join(x for x in kid_texts if x)
            out.append(Cell(row=m["RowIndex"] - 1, col=m["ColumnIndex"] - 1, row_span=m["RowSpan"], col_span=m["ColumnSpan"],
                            text=text, is_header="COLUMN_HEADER" in m.get("EntityTypes", []) or
                            any("COLUMN_HEADER" in k.get("EntityTypes", []) for k in kids)))
        for c in cells:
            if c["Id"] in covered:
                continue
            out.append(Cell(row=c["RowIndex"] - 1, col=c["ColumnIndex"] - 1, row_span=c.get("RowSpan", 1), col_span=c.get("ColumnSpan", 1),
                            text=_block_text(c, by_id), is_header="COLUMN_HEADER" in c.get("EntityTypes", [])))
        n_rows = max(c.row + c.row_span for c in out)
        n_cols = max(c.col + c.col_span for c in out)
        tables.append(Table(page=page, n_rows=n_rows, n_cols=n_cols, cells=out))
    return tables


TEXTRACT_EXPENSE_MAP = {"invoice_number": ["INVOICE_RECEIPT_ID"], "invoice_date": ["INVOICE_RECEIPT_DATE", "ORDER_DATE"],
                        "due_date": ["DUE_DATE"], "vendor_name": ["VENDOR_NAME", "NAME"], "bill_to": ["RECEIVER_NAME", "CUSTOMER_NUMBER"],
                        "subtotal": ["SUBTOTAL"], "tax": ["TAX"], "total": ["TOTAL", "AMOUNT_DUE"]}


def textract_expense_fields(resp: dict) -> tuple[dict, list]:
    """AnalyzeExpense -> (raw fields by Textract type, line items)."""
    by_type: dict[str, list[str]] = {}
    items = []
    for doc in resp.get("ExpenseDocuments", []):
        for f in doc.get("SummaryFields", []):
            typ = (f.get("Type") or {}).get("Text", "OTHER")
            val = (f.get("ValueDetection") or {}).get("Text")
            if val:
                by_type.setdefault(typ, []).append(val)
        for group in doc.get("LineItemGroups", []):
            for li in group.get("LineItems", []):
                vals = {(e.get("Type") or {}).get("Text"): (e.get("ValueDetection") or {}).get("Text")
                        for e in li.get("LineItemExpenseFields", [])}
                items.append(LineItem(description=vals.get("ITEM"), quantity=parse_number(vals.get("QUANTITY")),
                                      unit_price=parse_money(vals.get("UNIT_PRICE")), amount=parse_money(vals.get("PRICE"))))
    raw = {}
    for name, types in TEXTRACT_EXPENSE_MAP.items():
        for t in types:
            if by_type.get(t):
                raw[name] = by_type[t][-1] if name in ("subtotal", "tax", "total") else by_type[t][0]
                break
    return raw, items


def textract_to_common(doc_id: str, pages: list[tuple[dict | None, dict | None]], latency_s: float, cost_usd: float) -> "InvoiceDoc":
    """pages = [(analyze_document response, analyze_expense response) per page]; either may be None."""
    raw: dict = {}
    tables, items = [], []
    for p, (doc_resp, exp_resp) in enumerate(pages, 1):
        if doc_resp:
            tables += textract_tables(doc_resp, page=p)
        if exp_resp:
            fields, page_items = textract_expense_fields(exp_resp)
            items += page_items
            for k, v in fields.items():
                if k in ("subtotal", "tax", "total"):
                    raw[k] = v
                else:
                    raw.setdefault(k, v)
    return finalize(doc_id, raw, tables, source="textract", pages=len(pages), latency_s=latency_s, cost_usd=cost_usd,
                    line_items=items or None)

# %% [markdown]
# ## 3 · The hard-case set and the target schema

# %%
DOCS = make_invoice_set(1, seed=SEED, prefix="d3")
BY_LAYOUT = {d.layout: d for d in DOCS}

fig, axes = plt.subplots(1, len(DOCS) + 1, figsize=(4 * (len(DOCS) + 1), 6))
panels = [(d.doc_id, d.pages[0]) for d in DOCS] + [(f"{BY_LAYOUT['multipage'].doc_id} p2", BY_LAYOUT["multipage"].pages[1])]
for ax, (title, img) in zip(axes, panels):
    ax.imshow(img)
    ax.set_title(title, fontsize=9)
    ax.axis("off")
save_fig(fig, "day4_hard_cases.png")

schema = InvoiceDoc.model_json_schema()
(OUTPUT_DIR / "invoice_schema.json").write_text(json.dumps(schema, indent=2), encoding="utf-8")
print("JSON Schema written to invoice_schema.json — top-level properties:", list(schema["properties"]))
print(json.dumps(BY_LAYOUT["merged"].truth, indent=1)[:1200])

# %% [markdown]
# ## 4 · Normalizing values: the boring part that decides your accuracy
#
# Every tool returns strings in its own format. The schema stores canonical values, so a correct extraction is never scored wrong because of formatting.

# %%
examples = {
    "money": ["$1,234.50", "1.234,50 €", "€ 99", "USD 1,000", "8,108.48"],
    "date": ["March 04, 2026", "2026-03-14", "14 Mar 2026", "03/14/2026", "Due: 04/13/2026"],
    "currency": ["$12.00", "€5", "INR 400", "1,000"],
}
print(pd.DataFrame({"money in": examples["money"], "parse_money": [parse_money(x) for x in examples["money"]],
                    "date in": examples["date"], "parse_date": [parse_date(x) for x in examples["date"]]}).to_string(index=False))
print({x: detect_currency(x) for x in examples["currency"]})
print("\nAmbiguity to remember: '03/04/2026' ->", parse_date("03/04/2026"),
      "(US month-first). European documents need dayfirst=True: a per-source setting, not a guess.")

# %% [markdown]
# ## 5 · Hard case: rotated scan
#
# Tesseract assumes horizontal text lines. Compare key-value extraction with and without deskew on the rotated invoice; then sweep the angle.

# %%
EXTRACTOR = TableExtractor(device=DEVICE)


def field_hits(pred: dict, truth: dict) -> dict:
    out = {}
    for k in FIELD_NAMES:
        p, t = pred.get(k), truth.get(k)
        out[k] = (p is not None and abs(p - t) < 0.01) if k in MONEY_FIELDS else norm_text(p) == norm_text(t)
    return out


d = BY_LAYOUT["rotated"]
rows = []
for do_deskew in (False, True):
    doc, dbg = run_selfhosted(d.doc_id, d.pages, EXTRACTOR, do_deskew=do_deskew)
    hits = field_hits(doc.fields.model_dump(), d.truth["fields"])
    rows.append({"deskew": do_deskew, "estimated_skew": round(dbg["pages"][0]["skew"], 2), "true_skew": d.truth["meta"]["skew_deg"],
                 "fields_correct": f"{sum(hits.values())}/9", "items": f"{len(doc.line_items)}/{len(d.truth['line_items'])}",
                 "table": f"{doc.tables[0].n_rows}x{doc.tables[0].n_cols}" if doc.tables else "none"})
    if do_deskew:
        fig, axes = plt.subplots(1, 2, figsize=(14, 9))
        axes[0].imshow(d.pages[0])
        axes[0].set_title("input")
        axes[1].imshow(dbg["pages"][0]["image"])
        for w in dbg["pages"][0]["words"]:
            x0, y0, x1, y1 = w["bbox"]
            axes[1].add_patch(plt.Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, lw=0.5, color="tab:blue"))
        for reg in dbg["pages"][0]["regions"]:
            axes[1].add_patch(plt.Rectangle((reg[0], reg[1]), reg[2] - reg[0], reg[3] - reg[1], fill=False, lw=2, color="tab:red"))
        axes[1].set_title(f"deskewed by {dbg['pages'][0]['skew']:+.2f}°: OCR words (blue), table region (red)")
        for ax in axes:
            ax.axis("off")
        save_fig(fig, "day4_rotated_deskew.png")
print(pd.DataFrame(rows).to_string(index=False))

sweep = []
base = BY_LAYOUT["standard"]
for angle in (0, 2, 4, 6, 8):
    page = degrade_scan(base.pages[0], angle=angle, noise=6)
    for do_deskew in (False, True):
        doc, _ = run_selfhosted(base.doc_id, [page], None, do_deskew=do_deskew)
        sweep.append({"angle": angle, "deskew": do_deskew, "fields_correct": sum(field_hits(doc.fields.model_dump(), base.truth["fields"]).values()),
                      "items_found": len(doc.line_items)})
SWEEP = pd.DataFrame(sweep).pivot(index="angle", columns="deskew", values=["fields_correct", "items_found"])
print("\nskew sweep (rules-only pipeline, standard invoice):\n", SWEEP.to_string())

# %% [markdown]
# ## 6 · Hard case: merged cells with Table Transformer
#
# Table Transformer (TATR, DETR architecture) detects six object classes inside a table crop: `table`, `table row`, `table column`, `table column header`, `table projected row header`, `table spanning cell`. Cells are row × column intersections; a spanning-cell box merges the grid cells it covers. Below: what TATR predicts on the merged-header invoice, the resulting grid, and what the rule-based fallback makes of the same table.

# %%
d = BY_LAYOUT["merged"]
img = d.pages[0]
regions = EXTRACTOR.detect(img)
print("TATR table detections:", [[round(v) for v in r] for r in regions] or "none -> keyword region fallback")
region = regions[0] if regions else (find_table_region(ocr_words(img), img.size) or [0, 0, img.width, img.height])
pad = 20
x0, y0 = max(0, int(region[0]) - pad), max(0, int(region[1]) - pad)
x1, y1 = min(img.width, int(region[2]) + pad), min(img.height, int(region[3]) + pad)
crop = img.crop((x0, y0, x1, y1))
struct = EXTRACTOR.structure(crop)
print({k: len(v) for k, v in struct.items()})

fig, axes = plt.subplots(1, 3, figsize=(20, 6))
for ax, key, color in zip(axes, ("rows", "cols", "spans"), ("tab:blue", "tab:green", "tab:red")):
    ax.imshow(crop)
    for o in struct[key] + (struct["header"] if key == "spans" else []):
        bx = o["bbox"]
        ax.add_patch(plt.Rectangle((bx[0], bx[1]), bx[2] - bx[0], bx[3] - bx[1], fill=False, lw=1.5,
                                   color="tab:purple" if o["label"] == "table column header" else color))
    ax.set_title({"rows": "table rows", "cols": "table columns", "spans": "spanning cells (red), column header (purple)"}[key])
    ax.axis("off")
save_fig(fig, "day4_tatr_structure.png")

words = ocr_words(img)
tatr_table = cells_from_structure(struct, [w for w in words if x0 <= w["bbox"][0] <= x1 and y0 <= w["bbox"][1] <= y1], offset=(x0, y0))
truth_table = Table.model_validate(d.truth["tables"][0])
print("\nTATR grid (spans shown once, at their top-left cell):")
print(pd.DataFrame(table_to_grid(tatr_table)).to_string(index=False, header=False))
print("spans:", [(c.row, c.col, c.row_span, c.col_span, c.text) for c in tatr_table.cells if c.row_span > 1 or c.col_span > 1])
print("columns ->", column_names(tatr_table), "->", map_columns(column_names(tatr_table)))
print("\ntruth spans:", [(c.row, c.col, c.row_span, c.col_span, c.text) for c in truth_table.cells if c.row_span > 1 or c.col_span > 1])
rule_tab = rule_table(words, [x0, y0, x1, y1])
print("\nrules-only table:", f"{rule_tab.n_rows}x{rule_tab.n_cols}, columns {column_names(rule_tab)}" if rule_tab else "none")

# %% [markdown]
# ## 7 · Hard case: a table across two pages
#
# Each page yields its own table. `stitch_tables` merges page *p+1*'s table into page *p*'s when the column count matches and the header repeats (header rows dropped) or is absent.

# %%
d = BY_LAYOUT["multipage"]
page_tables = []
for i, page in enumerate(d.pages):
    tabs, _, method = extract_tables(page, ocr_words(page), EXTRACTOR, page=i + 1)
    page_tables += tabs
    print(f"page {i + 1}: {[(t.n_rows, t.n_cols) for t in tabs]} via {method}")
stitched = stitch_tables(page_tables)
print(f"before stitching: {len(page_tables)} tables, {sum(t.n_rows for t in page_tables)} rows | after: {len(stitched)} table(s), "
      f"{[t.n_rows for t in stitched]} rows | truth: {d.truth['tables'][0]['n_rows']} rows (1 header + {len(d.truth['line_items'])} items)")
print("line items recovered:", len(line_items_from_table(primary_table(stitched))), "/", len(d.truth["line_items"]))

# %% [markdown]
# ## 8 · Run every approach on every hard case → common JSON
#
# Each result is validated by pydantic on construction (`finalize`), so "produces valid JSON" is guaranteed rather than hoped for. Files go to `normalized/<approach>/<doc_id>.json`.

# %%
AZ, TX = azure_client(), textract_client()
METER = CostMeter(MAX_PAGES, MAX_USD)
cached_azure = RAW_DIR / "azure"
cached_textract = RAW_DIR / "textract"
print(f"Azure: {'live' if AZ else ('cache only' if cached_azure.exists() else 'unavailable')} | "
      f"Textract: {'live' if TX else ('cache only' if cached_textract.exists() else 'unavailable')}")


def run_azure(d) -> "InvoiceDoc | None":
    path = RAW_DIR / "azure" / f"{d.doc_id}.json"
    if AZ is None and not path.exists():
        return None
    try:
        res, lat, cost = azure_analyze(AZ, "prebuilt-invoice", d.pdf_bytes(), len(d.pages), METER, path, d.doc_id)
    except Exception as exc:
        print(f"azure {d.doc_id}: {exc}")
        return None
    return azure_to_common(d.doc_id, res, lat, cost)


def run_textract(d) -> "InvoiceDoc | None":
    paths = [(RAW_DIR / "textract" / f"{d.doc_id}_p{p + 1}.json", RAW_DIR / "textract_expense" / f"{d.doc_id}_p{p + 1}.json")
             for p in range(len(d.pages))]
    if TX is None and not all(a.exists() and b.exists() for a, b in paths):
        return None
    pages, lat, cost = [], 0.0, 0.0
    try:
        for p, (pa, pb) in enumerate(paths):
            png = d.png_bytes(p)
            r1, l1, c1 = textract_analyze_document(TX, png, METER, ("TABLES", "FORMS"), pa, d.doc_id)
            r2, l2, c2 = textract_analyze_expense(TX, png, METER, pb, d.doc_id)
            pages.append((r1, r2))
            lat, cost = lat + l1 + l2, cost + c1 + c2
    except Exception as exc:
        print(f"textract {d.doc_id}: {exc}")
        return None
    return textract_to_common(d.doc_id, pages, lat, cost)


APPROACHES = {
    "selfhosted_rules": lambda d: run_selfhosted(d.doc_id, d.pages, None)[0],
    "selfhosted_tatr": lambda d: run_selfhosted(d.doc_id, d.pages, EXTRACTOR)[0],
    "azure": run_azure,
    "textract": run_textract,
}
OUTPUTS: dict[tuple[str, str], InvoiceDoc] = {}
for d in DOCS:
    for name, fn in APPROACHES.items():
        doc = fn(d)
        if doc is None:
            continue
        doc = InvoiceDoc.model_validate(json.loads(doc.model_dump_json()))  # round-trip proves the JSON is schema-valid
        OUTPUTS[(name, d.doc_id)] = doc
        path = OUTPUT_DIR / "normalized" / name / f"{d.doc_id}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(doc.model_dump_json(indent=2), encoding="utf-8")
print(f"{len(OUTPUTS)} normalized documents written | managed spend this run: {METER.summary()}")
print(OUTPUTS[("selfhosted_tatr", BY_LAYOUT["standard"].doc_id)].model_dump_json(indent=1)[:1500])

# %% [markdown]
# ## 9 · Hard-case scoreboard
#
# Quick checks per (approach, layout); Day 5 replaces these with proper metrics (TEDS, cell F1) on a larger held-out set.
#
# - **fields**: exact matches out of 9 after normalization
# - **items**: truth line items whose amount matches a predicted item (±0.01) with a similar description
# - **table shape**: predicted primary table rows × cols vs truth

# %%
def items_matched(pred: list, truth: list[dict]) -> int:
    used, hits = set(), 0
    for t in truth:
        for j, p in enumerate(pred):
            if j not in used and p.amount is not None and abs(p.amount - t["amount"]) < 0.01 and \
                    fuzz.token_set_ratio(p.description or "", t["description"]) >= 80:
                used.add(j)
                hits += 1
                break
    return hits


rows = []
for (name, doc_id), doc in OUTPUTS.items():
    d = next(x for x in DOCS if x.doc_id == doc_id)
    hits = field_hits(doc.fields.model_dump(), d.truth["fields"])
    tt, pt = d.truth["tables"][0], primary_table(doc.tables)
    rows.append({"approach": name, "layout": d.layout, "fields": sum(hits.values()),
                 "items": f"{items_matched(doc.line_items, d.truth['line_items'])}/{len(d.truth['line_items'])}",
                 "table_shape": f"{pt.n_rows}x{pt.n_cols}" if pt else "-", "truth_shape": f"{tt['n_rows']}x{tt['n_cols']}",
                 "spans_found": sum(1 for c in (pt.cells if pt else []) if c.row_span > 1 or c.col_span > 1),
                 "latency_s": doc.meta.latency_s, "missed_fields": ", ".join(k for k, v in hits.items() if not v)})
BOARD = pd.DataFrame(rows).sort_values(["layout", "approach"])
BOARD.to_csv(OUTPUT_DIR / "day4_hard_case_board.csv", index=False)
print(BOARD.to_string(index=False))

pivot = BOARD.pivot(index="approach", columns="layout", values="fields")
fig, ax = plt.subplots(figsize=(9, 3 + 0.4 * len(pivot)))
im = ax.imshow(pivot.values, cmap="RdYlGn", vmin=0, vmax=9)
ax.set_xticks(range(pivot.shape[1]), pivot.columns)
ax.set_yticks(range(pivot.shape[0]), pivot.index)
for i in range(pivot.shape[0]):
    for j in range(pivot.shape[1]):
        ax.text(j, i, f"{pivot.values[i, j]:.0f}/9", ha="center", va="center")
ax.set_title("Fields correct per hard case")
save_fig(fig, "day4_fields_heatmap.png")

# %% [markdown]
# ## 10 · Where did it go wrong? Field-by-field diff
#
# For each layout, every approach's value next to the truth. This is the table to read before tuning anything.

# %%
for layout in ("layout_b", "merged"):
    d = BY_LAYOUT[layout]
    diff = {"truth": d.truth["fields"]}
    for name in APPROACHES:
        if (name, d.doc_id) in OUTPUTS:
            diff[name] = OUTPUTS[(name, d.doc_id)].fields.model_dump()
    print(f"\n=== {layout} ===")
    print(pd.DataFrame(diff).to_string())

# %% [markdown]
# ## 11 · Save results

# %%
summary = {
    "documents": {d.doc_id: d.layout for d in DOCS},
    "approaches": sorted({a for a, _ in OUTPUTS}),
    "board": BOARD.drop(columns=["missed_fields"]).to_dict("records"),
    "skew_sweep": SWEEP.reset_index().to_dict("records"),
    "managed_spend": {"pages": METER.pages, "usd": round(METER.usd, 4)},
}
with open(OUTPUT_DIR / "day4_summary.json", "w") as f:
    json.dump(summary, f, indent=2, default=str)
print("normalized JSON per approach in", OUTPUT_DIR / "normalized")

# %% [markdown]
# ## Takeaways (fill in with your numbers)
#
# - Rotated: deskew moved fields from ___/9 to ___/9. Above ___° the pipeline breaks even with deskew.
# - Merged header: rules-only produced ___; TATR found ___ spanning cells and recovered ___/___ line items. Azure / Textract spans: ___ / ___.
# - Multi-page: stitching turned ___ tables into 1 with ___ rows (truth ___).
# - Layout B: the rule-based KV extractor needed the synonyms "Bill No." / "Amount Due" — what happens to a layout whose labels you never listed? Managed models are trained on thousands of layouts: ___ vs ___ fields.
#
# The schema is the contract for Week 6's RAG ingestion layer: a table with explicit spans and a provenance-carrying `meta` block is what a retriever can cite. Tomorrow: score all of this properly on a held-out set and put a price on it.
