"""
Week 4 · Day 3 — Managed document intelligence: Azure AI Document Intelligence and AWS Textract.

Colab-first script (percent-format cells). Build the notebook with:
    python ../tools/py_to_ipynb.py day3_managed_services/day3_managed_services.py
Documents: synthetic invoices rendered in this script (one per hard-case layout).
"""

# %% [markdown]
# # Week 4 · Day 3 — Managed Services (Azure Document Intelligence & AWS Textract)
#
# **Goal:** extract key-value pairs and tables from the *same* invoices with two managed cloud services, understand what each API returns, and do it without a surprise bill.
#
# | Topic | What you will build |
# |---|---|
# | Azure AI Document Intelligence | `prebuilt-invoice` (typed fields + line items + tables) and `prebuilt-layout` with the key-value add-on; how custom models fit in |
# | AWS Textract | `AnalyzeDocument` (FORMS + TABLES, incl. merged cells) and `AnalyzeExpense` (invoice/receipt fields + line items) |
# | Response parsing | Walk Azure's field tree and Textract's block graph; tested offline on hand-written mock responses |
# | Cost control | A page/USD cap that refuses the call before it is made, and an on-disk response cache so re-runs are free |
# | First look at normalization | Both outputs mapped into the Week 4 common schema and compared with ground truth (Day 4 goes deep) |
#
# **No cloud account?** Every cell still runs: the parsers are exercised on mock responses and the API cells print setup instructions and skip. Add credentials later and re-run.
#
# > Runtime: CPU is enough today.

# %% [markdown]
# ## 0 · Install dependencies

# %%
import subprocess
import sys


def pip_install(*packages: str) -> None:
    """Install packages into the running interpreter (works in Colab and plain Python)."""
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *packages])


pip_install("azure-ai-documentintelligence==1.0.2", "boto3>=1.34", "rapidfuzz>=3.9", "python-dateutil", "pydantic>=2.6")

# %% [markdown]
# ## 1 · Imports, configuration, and output folder
#
# `MAX_PAGES` / `MAX_USD` cap all billable calls in this session. The defaults (40 pages, $1) are far above what this notebook needs (~6 pages × 2 Azure models + 6 pages × 2 Textract APIs).

# %%
import json
from pathlib import Path

import matplotlib

IN_NOTEBOOK = "ipykernel" in sys.modules
if not IN_NOTEBOOK:
    matplotlib.use("Agg")  # headless when run as a script

import matplotlib.pyplot as plt
import pandas as pd

IN_COLAB = "google.colab" in sys.modules
try:
    HERE = Path(__file__).resolve().parent
except NameError:  # running inside a notebook
    HERE = Path.cwd()

OUTPUT_DIR = Path("/content/week4_outputs/day3") if IN_COLAB else HERE.parent / "results" / "day3"
RAW_DIR = OUTPUT_DIR / "raw"  # cached API responses: reruns cost nothing. Day 4 and 5 reuse this folder.
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MAX_PAGES, MAX_USD = 40, 1.00
SEED = 7
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
# - **Common JSON schema** (pydantic) and value normalizers. Read its header comment: it is the target format for the whole week.
# - **Synthetic invoice generator**: the invoices we send to the cloud (no confidential documents leave your machine today).
# - **Managed-service helpers**: API calls with cost cap + cache, and parsers for both providers.

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
# ## 3 · How the two APIs shape their answers (offline, mock responses)
#
# **Azure** returns a *document model*: `documents[0].fields` is a typed tree (`InvoiceTotal` is a `currency` with `amount` and `currencyCode`; `Items` is an array of objects). Tables are listed separately with explicit `rowSpan` / `columnSpan` and `kind="columnHeader"`.
#
# **Textract** returns a flat *block graph*: `PAGE → LINE → WORD`, `KEY_VALUE_SET` blocks linked by `VALUE` relationships, `TABLE → CELL → WORD`, and `MERGED_CELL` blocks that group the cells they cover. You reconstruct structure by following relationship IDs.
#
# The two tiny responses below have the shapes the real APIs return, including a merged header cell. They let the parsers be tested without an account.

# %%
AZURE_MOCK = {
    "pages": [{"pageNumber": 1, "width": 8.27, "height": 11.69, "unit": "inch"}],
    "documents": [{"docType": "invoice", "fields": {
        "InvoiceId": {"type": "string", "valueString": "INV-1", "content": "INV-1", "confidence": 0.98},
        "InvoiceDate": {"type": "date", "valueDate": "2026-03-14", "content": "March 14, 2026"},
        "VendorName": {"type": "string", "valueString": "Northwind Traders", "content": "Northwind Traders"},
        "CustomerName": {"type": "string", "valueString": "Woodgrove Bank", "content": "Woodgrove Bank"},
        "SubTotal": {"type": "currency", "valueCurrency": {"amount": 100.0, "currencyCode": "USD"}, "content": "$100.00"},
        "TotalTax": {"type": "currency", "valueCurrency": {"amount": 8.0, "currencyCode": "USD"}, "content": "$8.00"},
        "InvoiceTotal": {"type": "currency", "valueCurrency": {"amount": 108.0, "currencyCode": "USD"}, "content": "$108.00"},
        "Items": {"type": "array", "valueArray": [{"type": "object", "valueObject": {
            "Description": {"type": "string", "valueString": "Widget"}, "Quantity": {"type": "number", "valueNumber": 2},
            "UnitPrice": {"type": "currency", "valueCurrency": {"amount": 50.0}},
            "Amount": {"type": "currency", "valueCurrency": {"amount": 100.0}}}}]}}}],
    "tables": [{"rowCount": 3, "columnCount": 3, "boundingRegions": [{"pageNumber": 1}], "cells": [
        {"rowIndex": 0, "columnIndex": 0, "rowSpan": 2, "content": "Description", "kind": "columnHeader"},
        {"rowIndex": 0, "columnIndex": 1, "columnSpan": 2, "content": "Pricing", "kind": "columnHeader"},
        {"rowIndex": 1, "columnIndex": 1, "content": "Unit Price", "kind": "columnHeader"},
        {"rowIndex": 1, "columnIndex": 2, "content": "Amount", "kind": "columnHeader"},
        {"rowIndex": 2, "columnIndex": 0, "content": "Widget"},
        {"rowIndex": 2, "columnIndex": 1, "content": "$50.00"},
        {"rowIndex": 2, "columnIndex": 2, "content": "$100.00"}]}],
    "keyValuePairs": [{"key": {"content": "Invoice Number:"}, "value": {"content": "INV-1"}, "confidence": 0.91}],
}


def _w(i, text):
    return {"Id": f"w{i}", "BlockType": "WORD", "Text": text}


def _cell(cid, r, c, word_ids=(), header=False):
    b = {"Id": cid, "BlockType": "CELL", "RowIndex": r, "ColumnIndex": c, "RowSpan": 1, "ColumnSpan": 1}
    if word_ids:
        b["Relationships"] = [{"Type": "CHILD", "Ids": list(word_ids)}]
    if header:
        b["EntityTypes"] = ["COLUMN_HEADER"]
    return b


TEXTRACT_MOCK = {"Blocks": [
    _w(1, "Item"), _w(2, "Pricing"), _w(3, "Widget"), _w(4, "$50.00"), _w(5, "$100.00"), _w(6, "Invoice"), _w(7, "No:"), _w(8, "INV-1"),
    {"Id": "t1", "BlockType": "TABLE", "Relationships": [{"Type": "CHILD", "Ids": ["c11", "c12", "c13", "c21", "c22", "c23"]},
                                                         {"Type": "MERGED_CELL", "Ids": ["m1"]}]},
    _cell("c11", 1, 1, ["w1"], True), _cell("c12", 1, 2, ["w2"], True), _cell("c13", 1, 3, (), True),
    _cell("c21", 2, 1, ["w3"]), _cell("c22", 2, 2, ["w4"]), _cell("c23", 2, 3, ["w5"]),
    {"Id": "m1", "BlockType": "MERGED_CELL", "RowIndex": 1, "ColumnIndex": 2, "RowSpan": 1, "ColumnSpan": 2,
     "EntityTypes": ["COLUMN_HEADER"], "Relationships": [{"Type": "CHILD", "Ids": ["c12", "c13"]}]},
    {"Id": "k1", "BlockType": "KEY_VALUE_SET", "EntityTypes": ["KEY"], "Confidence": 93.0,
     "Relationships": [{"Type": "VALUE", "Ids": ["v1"]}, {"Type": "CHILD", "Ids": ["w6", "w7"]}]},
    {"Id": "v1", "BlockType": "KEY_VALUE_SET", "EntityTypes": ["VALUE"], "Relationships": [{"Type": "CHILD", "Ids": ["w8"]}]},
]}
EXPENSE_MOCK = {"ExpenseDocuments": [{
    "SummaryFields": [{"Type": {"Text": "INVOICE_RECEIPT_ID"}, "ValueDetection": {"Text": "INV-1"}},
                      {"Type": {"Text": "VENDOR_NAME"}, "ValueDetection": {"Text": "Northwind Traders"}},
                      {"Type": {"Text": "TOTAL"}, "ValueDetection": {"Text": "$108.00"}}],
    "LineItemGroups": [{"LineItems": [{"LineItemExpenseFields": [
        {"Type": {"Text": "ITEM"}, "ValueDetection": {"Text": "Widget"}}, {"Type": {"Text": "QUANTITY"}, "ValueDetection": {"Text": "2"}},
        {"Type": {"Text": "UNIT_PRICE"}, "ValueDetection": {"Text": "$50.00"}}, {"Type": {"Text": "PRICE"}, "ValueDetection": {"Text": "$100.00"}}]}]}],
}]}

print("Azure fields      :", azure_fields(AZURE_MOCK))
print("Azure line items  :", [i.model_dump() for i in azure_line_items(AZURE_MOCK)])
print("Azure KV pairs    :", azure_kv_pairs(AZURE_MOCK))
print("Azure table grid  :", table_to_grid(azure_tables(AZURE_MOCK)[0]), "-> columns", column_names(azure_tables(AZURE_MOCK)[0]))
print("Textract KV pairs :", textract_kv_pairs(TEXTRACT_MOCK))
tt = textract_tables(TEXTRACT_MOCK)[0]
print("Textract table    :", table_to_grid(tt), "| merged:", [(c.row, c.col, c.row_span, c.col_span, c.text) for c in tt.cells if c.col_span > 1])
print("Textract expense  :", textract_expense_fields(EXPENSE_MOCK))
assert azure_fields(AZURE_MOCK)["total"] == 108.0 and textract_kv_pairs(TEXTRACT_MOCK)[0][:2] == ("Invoice No:", "INV-1")
assert any(c.col_span == 2 and c.text == "Pricing" for c in tt.cells)
mock_doc = azure_to_common("mock", AZURE_MOCK, 1.0, 0.01)
print("\nAzure mock in the common schema:\n", mock_doc.model_dump_json(indent=1)[:900])

# %% [markdown]
# ## 4 · Credentials
#
# **Azure** — Portal → *Create a resource* → **Document Intelligence** → pricing tier **F0** (free: 500 pages/month, first 2 pages of each document) → *Keys and Endpoint*. Set `AZURE_DI_ENDPOINT` and `AZURE_DI_KEY`.
#
# **AWS** — IAM → user with the `AmazonTextractFullAccess` policy → access key. Set `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_DEFAULT_REGION` (Textract regions include us-east-1, eu-west-1, ap-south-1…).
#
# In Colab add them under the **key icon (Secrets)** and grant this notebook access; never paste keys into a cell you commit.

# %%
AZ = azure_client()
TX = textract_client()
METER = CostMeter(MAX_PAGES, MAX_USD)
print(f"Azure Document Intelligence: {'configured' if AZ else 'not configured (cells will skip)'}")
print(f"AWS Textract              : {'configured' if TX else 'not configured (cells will skip)'}")

# %% [markdown]
# ## 5 · The document set
#
# One invoice per hard-case layout. Saved as PNG + PDF + `truth.json` so you can also upload them by hand to [Azure Document Intelligence Studio](https://documentintelligence.ai.azure.com/studio) or the [Textract console](https://console.aws.amazon.com/textract/) and compare with what the API returns here.

# %%
DOCS = make_invoice_set(1, seed=SEED, prefix="d3")
for d in DOCS:
    d.save(OUTPUT_DIR / "docs")
print(pd.DataFrame([{"doc_id": d.doc_id, "layout": d.layout, "pages": len(d.pages), "items": len(d.truth["line_items"]),
                     "total": d.truth["fields"]["total"]} for d in DOCS]).to_string(index=False))
total_pages = sum(len(d.pages) for d in DOCS)
print(f"\n{total_pages} pages. Estimated cost if run uncached: "
      f"Azure invoice+layout ${total_pages * (PRICE_PER_PAGE['azure:prebuilt-invoice'] + PRICE_PER_PAGE['azure:prebuilt-layout']):.2f}, "
      f"Textract tables+forms+expense ${total_pages * (PRICE_PER_PAGE['textract:tables+forms'] + PRICE_PER_PAGE['textract:expense']):.2f}")

# %% [markdown]
# ## 6 · Azure AI Document Intelligence
#
# | Model | Returns | Typical use |
# |---|---|---|
# | `prebuilt-read` | Lines, words, languages | Plain OCR |
# | `prebuilt-layout` | + tables, selection marks, paragraphs/roles; key-value pairs with the `keyValuePairs` feature | Any document, no schema |
# | `prebuilt-invoice` / `prebuilt-receipt` | + typed fields (`InvoiceId`, `InvoiceTotal`, `Items[]`) | Known document types, zero training |
# | Custom template / neural | Your own fields, trained from ~5+ labeled samples in Studio | Your own forms |
#
# Multi-page PDFs go in one call. `begin_analyze_document` is a long-running operation: it returns a poller and the result arrives after polling.

# %%
AZ_RESULTS: dict[str, tuple] = {}
if AZ is None:
    print("Azure not configured — skipping. (Set AZURE_DI_ENDPOINT / AZURE_DI_KEY and re-run.)")
else:
    for d in DOCS:
        try:
            res, lat, cost = azure_analyze(AZ, "prebuilt-invoice", d.pdf_bytes(), len(d.pages), METER,
                                           RAW_DIR / "azure" / f"{d.doc_id}.json", d.doc_id)
        except BudgetExceeded as exc:
            print("STOP:", exc)
            break
        except Exception as exc:  # bad key/endpoint, quota, network: report and stop calling this provider
            print(f"Azure call failed for {d.doc_id}: {exc}")
            break
        AZ_RESULTS[d.doc_id] = (res, lat, cost)
        print(f"{d.doc_id:22s} {lat:5.2f}s  ${cost:.3f}  fields={len(res['documents'][0]['fields']) if res.get('documents') else 0}"
              f"  tables={len(res.get('tables') or [])}")
    print(METER.summary())

# %%
if AZ_RESULTS:
    d = DOCS[0]
    res = AZ_RESULTS[d.doc_id][0]
    fields = res["documents"][0]["fields"]
    print(pd.DataFrame([{"field": k, "type": v.get("type"), "content": v.get("content"), "value": _az_value(v),
                         "confidence": v.get("confidence")} for k, v in fields.items() if k != "Items"]).to_string(index=False))
    for t in azure_tables(res):
        print(f"\ntable on page {t.page}: {t.n_rows}x{t.n_cols}, columns = {column_names(t)}")
        print(pd.DataFrame(table_to_grid(t)).head(6).to_string(index=False, header=False))

    # Overlay: Azure polygons are in the page's unit (inches for PDF input, pixels for images)
    page = res["pages"][0]
    img = d.pages[0]
    sx, sy = img.width / page["width"], img.height / page["height"]
    fig, ax = plt.subplots(figsize=(9, 12))
    ax.imshow(img)
    for k, v in fields.items():
        for br in v.get("boundingRegions", []):
            if br["pageNumber"] == 1:
                xs, ys = br["polygon"][0::2], br["polygon"][1::2]
                ax.add_patch(plt.Polygon([(x * sx, y * sy) for x, y in zip(xs, ys)], fill=False, color="tab:red", lw=1.2))
                ax.text(min(xs) * sx, min(ys) * sy - 4, k, color="tab:red", fontsize=7)
    ax.set_title("Azure prebuilt-invoice fields")
    ax.axis("off")
    save_fig(fig, "day3_azure_fields.png")

# %% [markdown]
# ### Generic key-value pairs with `prebuilt-layout`
#
# No schema: Azure returns whatever looks like `key: value`. Useful for unknown forms, but the keys are raw text ("Bill No.", "Invoice Number:") that you must map to your schema yourself — exactly the normalization problem of Day 4.

# %%
if AZ is not None:
    d = next(x for x in DOCS if x.layout == "layout_b")
    try:
        res, lat, cost = azure_analyze(AZ, "prebuilt-layout", d.pdf_bytes(), len(d.pages), METER,
                                       RAW_DIR / "azure_layout" / f"{d.doc_id}.json", d.doc_id, key_value_pairs=True)
        print(pd.DataFrame(azure_kv_pairs(res), columns=["key", "value", "confidence"]).to_string(index=False))
    except Exception as exc:
        print("skipped:", exc)

# %% [markdown]
# ### Custom models (when prebuilt is not enough)
#
# For your own forms: upload ~5–10 samples to a blob container → label fields in **Document Intelligence Studio** (Custom extraction model) → *Train* (template = fixed layout, fast; neural = varied layouts) → you get a `model_id`. Calling it is the same API; set `AZURE_CUSTOM_MODEL_ID` to try. Programmatic training uses `DocumentIntelligenceAdministrationClient.begin_build_document_model` with a SAS URL to the labeled container.

# %%
CUSTOM_ID = get_secret("AZURE_CUSTOM_MODEL_ID")
if AZ is not None and CUSTOM_ID:
    d = DOCS[0]
    res, lat, cost = azure_analyze(AZ, CUSTOM_ID, d.pdf_bytes(), len(d.pages), METER, RAW_DIR / "azure_custom" / f"{d.doc_id}.json", d.doc_id)
    print({k: _az_value(v) for k, v in res["documents"][0]["fields"].items()})
else:
    print("No AZURE_CUSTOM_MODEL_ID set — custom model call skipped.")

# %% [markdown]
# ## 7 · AWS Textract
#
# | API | Returns | Price/page (list) |
# |---|---|---|
# | `DetectDocumentText` | Lines + words | lowest |
# | `AnalyzeDocument` `FeatureTypes=["TABLES"]` | + tables, cells, merged cells, column headers | ~$0.015 |
# | `AnalyzeDocument` `["FORMS"]` | + key-value sets | ~$0.05 |
# | `AnalyzeExpense` | Normalized invoice/receipt summary fields + line items | ~$0.01 |
#
# Synchronous calls take one image page (PNG/JPEG) or a single-page PDF. Multi-page PDFs need the asynchronous `Start…`/`Get…` APIs with the file in S3; here each page is sent as an image.

# %%
TX_RESULTS: dict[str, tuple] = {}
if TX is None:
    print("Textract not configured — skipping. (Set AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY / AWS_DEFAULT_REGION and re-run.)")
else:
    for d in DOCS:
        pages, lat_sum, cost_sum = [], 0.0, 0.0
        try:
            for p in range(len(d.pages)):
                png = d.png_bytes(p)
                doc_resp, l1, c1 = textract_analyze_document(TX, png, METER, ("TABLES", "FORMS"),
                                                             RAW_DIR / "textract" / f"{d.doc_id}_p{p + 1}.json", d.doc_id)
                exp_resp, l2, c2 = textract_analyze_expense(TX, png, METER, RAW_DIR / "textract_expense" / f"{d.doc_id}_p{p + 1}.json",
                                                            d.doc_id)
                pages.append((doc_resp, exp_resp))
                lat_sum += l1 + l2
                cost_sum += c1 + c2
        except BudgetExceeded as exc:
            print("STOP:", exc)
            break
        except Exception as exc:  # bad credentials, region without Textract, throttling
            print(f"Textract call failed for {d.doc_id}: {exc}")
            break
        TX_RESULTS[d.doc_id] = (pages, lat_sum, cost_sum)
        n_blocks = sum(len(x[0].get("Blocks", [])) for x in pages)
        print(f"{d.doc_id:22s} {lat_sum:5.2f}s  ${cost_sum:.3f}  blocks={n_blocks}")
    print(METER.summary())

# %%
if TX_RESULTS:
    d = next(x for x in DOCS if x.layout == "merged")
    doc_resp, exp_resp = TX_RESULTS[d.doc_id][0][0]
    print("FORMS key-value pairs:")
    print(pd.DataFrame(textract_kv_pairs(doc_resp), columns=["key", "value", "confidence"]).to_string(index=False))
    for t in textract_tables(doc_resp):
        print(f"\ntable {t.n_rows}x{t.n_cols}; merged cells:",
              [(c.row, c.col, c.row_span, c.col_span, c.text) for c in t.cells if c.row_span > 1 or c.col_span > 1])
        print(pd.DataFrame(table_to_grid(t)).head(8).to_string(index=False, header=False))
    raw, items = textract_expense_fields(exp_resp)
    print("\nAnalyzeExpense summary fields:", raw)
    print("line items:", len(items))

    img = d.pages[0]
    fig, ax = plt.subplots(figsize=(9, 12))
    ax.imshow(img)
    colors = {"TABLE": "tab:blue", "MERGED_CELL": "tab:red", "KEY_VALUE_SET": "tab:green"}
    for b in doc_resp["Blocks"]:
        if b["BlockType"] in colors and "Geometry" in b:
            bb = b["Geometry"]["BoundingBox"]
            ax.add_patch(plt.Rectangle((bb["Left"] * img.width, bb["Top"] * img.height), bb["Width"] * img.width,
                                       bb["Height"] * img.height, fill=False, color=colors[b["BlockType"]], lw=1.2))
    ax.set_title("Textract: tables (blue), merged cells (red), key/value sets (green)")
    ax.axis("off")
    save_fig(fig, "day3_textract_blocks.png")

# %% [markdown]
# ## 8 · Optional third provider: Google Document AI
#
# Create a processor (Invoice Parser or Form Parser) in the Google Cloud console, then set `GCP_PROJECT_ID`, `GCP_LOCATION` (`us`/`eu`), `GCP_PROCESSOR_ID` and authenticate (`gcloud auth application-default login`, or `google.colab.auth.authenticate_user()` in Colab). The response's `document.entities` carry `type_`, `mention_text`, and a `normalized_value`.

# %%
GCP = {k: get_secret(k) for k in ("GCP_PROJECT_ID", "GCP_LOCATION", "GCP_PROCESSOR_ID")}
if all(GCP.values()):
    pip_install("google-cloud-documentai>=2.29")
    from google.api_core.client_options import ClientOptions
    from google.cloud import documentai

    client = documentai.DocumentProcessorServiceClient(
        client_options=ClientOptions(api_endpoint=f"{GCP['GCP_LOCATION']}-documentai.googleapis.com"))
    name = client.processor_path(GCP["GCP_PROJECT_ID"], GCP["GCP_LOCATION"], GCP["GCP_PROCESSOR_ID"])
    d = DOCS[0]
    result = client.process_document(request=documentai.ProcessRequest(
        name=name, raw_document=documentai.RawDocument(content=d.pdf_bytes(), mime_type="application/pdf")))
    for e in result.document.entities:
        print(f"{e.type_:25s} {e.mention_text!r:40} conf={e.confidence:.2f}")
else:
    print("Google Document AI not configured — skipped (optional).")

# %% [markdown]
# ## 9 · Both services in the common schema, against ground truth
#
# `azure_to_common` and `textract_to_common` (section 2) map provider names to ours (`InvoiceTotal`→`total`, `INVOICE_RECEIPT_ID`→`invoice_number`), parse money/dates, stitch tables across pages, and validate with pydantic. A quick exact-match check per field; tomorrow covers the hard cases, Day 5 the full scoring.

# %%
def field_hits(pred: dict, truth: dict) -> dict:
    out = {}
    for k in FIELD_NAMES:
        p, t = pred.get(k), truth.get(k)
        if k in MONEY_FIELDS:
            out[k] = p is not None and abs(p - t) < 0.01
        else:
            out[k] = norm_text(p) == norm_text(t)
    return out


NORMALIZED = {}
rows = []
for d in DOCS:
    truth = d.truth["fields"]
    for provider, results, fn in (("azure", AZ_RESULTS, azure_to_common), ("textract", TX_RESULTS, textract_to_common)):
        if d.doc_id not in results:
            continue
        doc = fn(d.doc_id, *results[d.doc_id])
        NORMALIZED[(provider, d.doc_id)] = doc
        out_path = OUTPUT_DIR / "normalized" / provider / f"{d.doc_id}.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(doc.model_dump_json(indent=2), encoding="utf-8")
        hits = field_hits(doc.fields.model_dump(), truth)
        rows.append({"provider": provider, "layout": d.layout, **hits, "fields_correct": sum(hits.values()),
                     "items": f"{len(doc.line_items)}/{len(d.truth['line_items'])}", "latency_s": doc.meta.latency_s,
                     "cost_usd": doc.meta.cost_usd})
QUICK = pd.DataFrame(rows)
if len(QUICK):
    print(QUICK.to_string(index=False))
    QUICK.to_csv(OUTPUT_DIR / "day3_quick_check.csv", index=False)
else:
    print("No managed results yet — configure at least one provider to fill this table.")

# %% [markdown]
# ## 10 · Save results

# %%
summary = {
    "documents": [d.doc_id for d in DOCS],
    "azure_configured": AZ is not None,
    "textract_configured": TX is not None,
    "billing": {"pages": METER.pages, "usd": round(METER.usd, 4), "log": METER.log},
    "quick_check": QUICK.to_dict("records") if len(QUICK) else [],
}
with open(OUTPUT_DIR / "day3_summary.json", "w") as f:
    json.dump(summary, f, indent=2, default=float)
print(json.dumps({k: v for k, v in summary.items() if k != "quick_check"}, indent=2, default=float))
print("Raw responses cached in", RAW_DIR, "— download this folder; Day 4 and Day 5 can reuse it without paying again.")

# %% [markdown]
# ## Takeaways (fill in with your numbers)
#
# - Azure prebuilt-invoice: ___/9 fields right on `standard`, ___ on `layout_b`; latency ___ s/doc.
# - Textract AnalyzeExpense: ___/9 fields; did `AnalyzeDocument` report the "Pricing" merged cell? ___
# - Which service read the rotated scan better? ___
# - Total spend today: $___ for ___ pages.
#
# **Data privacy and residency:** every page above left your machine and was processed in the provider's region. Both services state that inputs are not used to train their models, but contracts, PII, and health or financial records may still require a specific region, a private endpoint, or an on-prem/container deployment (Azure offers Document Intelligence containers for disconnected use). Write down what your organisation allows before choosing a managed service.
