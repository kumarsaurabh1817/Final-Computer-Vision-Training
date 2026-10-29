"""
Week 5 · Day 3 — Natural-language explanations of CV output + schema-valid JSON (prompting, validate-and-repair, constrained decoding).

Colab-first script (percent-format cells). Build the notebook with:
    python ../tools/py_to_ipynb.py day3_structured_explanations/day3_structured_explanations.py
Datasets: Flickr8k from Kaggle (https://www.kaggle.com/datasets/adityajn105/flickr8k),
          the Week 4 held-out synthetic invoices (rendered in code, seed 2026).
"""

# %% [markdown]
# # Week 5 · Day 3 — Explaining CV Output and Enforcing JSON Schemas
#
# **Goal:** make a VLM's output something a program can consume. Two jobs, one technique:
#
# 1. **Explain CV output.** Give the VLM the image *and* what your Week 3 detector and Week 2-style classifier said, and get back a human-readable explanation — which detections it agrees with, which look wrong, what was missed.
# 2. **Extract structured data.** One call per invoice that returns the whole Week 4 `Fields` record as JSON (Day 2 needed nine questions).
#
# Both must be **schema-valid JSON**, every time. We compare three ways of getting there and measure compliance:
#
# | Mode | How | Guarantee |
# |---|---|---|
# | `prompt_only` | Ask for JSON and describe the keys in words | None — parse and hope |
# | `schema_repair` | Put the JSON Schema in the prompt, validate with pydantic, send the error back and retry (max 2) | Valid *if* a retry succeeds; costs extra calls |
# | `constrained` | [Outlines](https://dottxt-ai.github.io/outlines/) masks the logits so only tokens that keep the output schema-valid can be generated | Valid by construction (structure, types, enums, patterns) — not *correct* |
#
# **Data:** [Flickr8k on Kaggle](https://www.kaggle.com/datasets/adityajn105/flickr8k) for scenes, and the Week 4 held-out invoices (`d5_*`, seed 2026) so the extraction numbers line up with Week 4 Day 5 and Day 2.
#
# > Runtime: **T4 GPU**, ~25–35 min with `FULL_RUN=True`.

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
pip_install("transformers>=4.57", "accelerate>=1.0", "kagglehub>=0.3", "outlines>=1.0", "ultralytics>=8.3",
            "pytesseract==0.3.13", "rapidfuzz>=3.9", "python-dateutil", "pydantic>=2.6", "tabulate")

# %% [markdown]
# ## 1 · Imports, configuration, and output folder

# %%
import io
import json
import os
import random
import re
import time
from collections import Counter
from pathlib import Path
from typing import Literal

import matplotlib

IN_NOTEBOOK = "ipykernel" in sys.modules
if not IN_NOTEBOOK:
    matplotlib.use("Agg")  # headless when run as a script

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytesseract
import torch
import torchvision
from PIL import Image
from pydantic import BaseModel, Field, ValidationError
from rapidfuzz import fuzz
from rapidfuzz.distance import Levenshtein
from transformers import AutoModelForImageTextToText, AutoProcessor
from ultralytics import YOLO

IN_COLAB = "google.colab" in sys.modules
try:
    HERE = Path(__file__).resolve().parent
except NameError:  # running inside a notebook
    HERE = Path.cwd()

OUTPUT_DIR = Path("/content/week5_outputs/day3") if IN_COLAB else HERE.parent / "results" / "day3"
DATA_DIR = Path("/content/datasets") if IN_COLAB else HERE.parent / "data"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)

FULL_RUN = True  # False = quick smoke test
VLM_ID = "Qwen/Qwen3-VL-4B-Instruct"
N_SCENES = 20 if FULL_RUN else 4
N_PER_LAYOUT = 2 if FULL_RUN else 1  # Week 4 Day 5 held-out ids d5_<layout>_00, _01
HELDOUT_SEED = 2026
MAX_REPAIRS = 2
DET_CONF = 0.35
DOC_MAX_SIDE = 1600
GPU_USD_PER_HOUR = 0.35
SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
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


# %% [markdown]
# ## 2 · Helper code
#
# **Kaggle credentials:** Colab **key icon (Secrets)** → `KAGGLE_API_TOKEN`, or `KAGGLE_USERNAME` + `KAGGLE_KEY`. Or unzip Flickr8k into `/content/datasets/flickr8k/`. Week 4 blocks are copied unchanged so documents, truth and scoring match Week 4 Day 5.

# %%
# ---- data.py: helper code for this day (self-contained, no external project files) ----
def get_secret(name: str) -> str | None:
    """Environment variable first, then Colab secrets."""
    if os.environ.get(name):
        return os.environ[name]
    try:
        from google.colab import userdata
        return userdata.get(name)
    except Exception:
        return None


USE_DRIVE = True  # Colab: look for datasets in Google Drive before downloading
DRIVE_DATASETS = Path("/content/drive/MyDrive/datasets")


def drive_datasets() -> Path | None:
    """Mount Google Drive (Colab only, asks for permission once) and return the datasets folder, or None."""
    if not (IN_COLAB and USE_DRIVE):
        return None
    if not Path("/content/drive/MyDrive").exists():
        from google.colab import drive
        drive.mount("/content/drive")
    return DRIVE_DATASETS if DRIVE_DATASETS.exists() else None


def kaggle_dataset(slug: str, local_name: str) -> Path:
    """
    Folder of a Kaggle dataset, found or downloaded once.

    1. DATA_DIR/<local_name>, if it is already there (unzipped earlier in this session, or by you).
    2. Google Drive: DRIVE_DATASETS/<local_name>.zip is unzipped to DATA_DIR (local disk is much faster than
       reading thousands of files from Drive); an unzipped DRIVE_DATASETS/<local_name>/ folder is used directly.
    3. kagglehub.dataset_download(slug). Public datasets usually need no login; otherwise it uses KAGGLE_API_TOKEN
       or KAGGLE_USERNAME + KAGGLE_KEY (env or Colab secrets) or ~/.kaggle/kaggle.json. kagglehub caches downloads.
    """
    manual = DATA_DIR / local_name
    if manual.exists() and any(manual.iterdir()):
        return manual
    drive_dir = drive_datasets()
    if drive_dir is not None:
        zip_path = drive_dir / f"{local_name}.zip"
        if zip_path.exists():
            import zipfile
            print(f"unzipping {zip_path} -> {manual} ...")
            with zipfile.ZipFile(zip_path) as zf:
                zf.extractall(manual)
            return manual
        if (drive_dir / local_name).is_dir():
            print(f"using {drive_dir / local_name} (unzipped on Drive; a .zip would load faster)")
            return drive_dir / local_name
    for key in ("KAGGLE_API_TOKEN", "KAGGLE_USERNAME", "KAGGLE_KEY"):
        val = get_secret(key)
        if val:
            os.environ[key] = val
    import kagglehub
    path = Path(kagglehub.dataset_download(slug))
    print(f"{slug} -> {path}")
    return path


def clean_caption(c: str) -> str:
    return re.sub(r"\s+", " ", str(c)).strip().rstrip(" .")


def load_flickr8k(root: Path) -> tuple[Path, dict[str, list[str]]]:
    """Kaggle adityajn105/flickr8k layout: Images/*.jpg + captions.txt (CSV: image,caption). Returns (image_dir, {name: [5 captions]})."""
    cap_file = next(root.rglob("captions.txt"))
    image_dir = next(root.rglob("*.jpg")).parent
    df = pd.read_csv(cap_file)
    caps: dict[str, list[str]] = {}
    for name, cap in zip(df["image"], df["caption"]):
        caps.setdefault(name, []).append(clean_caption(cap))
    return image_dir, {k: v for k, v in caps.items() if (image_dir / k).exists()}


def flickr_split(caps: dict, n_test: int = 1000, seed: int = 42) -> tuple[list[str], list[str]]:
    """Deterministic test/train split by image name. Same seed on every Week 5 day -> same test images."""
    names = sorted(caps)
    random.Random(seed).shuffle(names)
    return names[:n_test], names[n_test:]

# %%
# ---- vlm.py: helper code for this day (self-contained, no external project files) ----
# A thin wrapper around any Hugging Face image-text-to-text model (Qwen3-VL, LLaVA-OneVision, ...):
#     messages -> chat template (inserts image placeholder tokens) -> processor (pixels -> patches) -> generate
def fit_image(img: Image.Image, max_side: int | None) -> Image.Image:
    """RGB copy with the longest side capped. The VLM's visual-token count grows with pixel count."""
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


class VLM:
    def __init__(self, model_id: str, dtype: torch.dtype | None = None, max_side: int | None = 1280):
        self.model_id = model_id
        self.max_side = max_side
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.processor.tokenizer.padding_side = "left"  # batched generation needs left padding
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_id, dtype=dtype or default_dtype(), device_map="auto", attn_implementation="sdpa").eval()
        self.last: dict = {}

    def messages(self, images: list, prompt: str, system: str | None = None, max_side: int | None = -1) -> list:
        side = self.max_side if max_side == -1 else max_side
        content = [{"type": "image", "image": fit_image(im, side)} for im in images] + [{"type": "text", "text": prompt}]
        msgs = [{"role": "system", "content": [{"type": "text", "text": system}]}] if system else []
        return msgs + [{"role": "user", "content": content}]

    def encode(self, conversations: list) -> dict:
        return self.processor.apply_chat_template(conversations, tokenize=True, add_generation_prompt=True, return_dict=True,
                                                  return_tensors="pt", padding=True).to(self.model.device)

    def visual_tokens(self, inputs: dict) -> int:
        """Number of image placeholder tokens in the first sequence (the processor expands one per visual token)."""
        tok = getattr(self.processor, "image_token", None)
        if tok is None:
            return 0
        tid = self.processor.tokenizer.convert_tokens_to_ids(tok)
        return int((inputs["input_ids"][0] == tid).sum())

    @torch.inference_mode()
    def generate(self, conversations: list, max_new_tokens: int = 128) -> list[str]:
        inputs = self.encode(conversations)
        t0 = time.perf_counter()
        out = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        latency = time.perf_counter() - t0
        new = out[:, inputs["input_ids"].shape[1]:]
        n_out = int((new[0] != self.processor.tokenizer.pad_token_id).sum())
        self.last = {"latency_s": latency, "input_tokens": int(inputs["input_ids"].shape[1]), "output_tokens": n_out,
                     "visual_tokens": self.visual_tokens(inputs)}
        return [t.strip() for t in self.processor.batch_decode(new, skip_special_tokens=True)]

    def ask(self, images, prompt: str, max_new_tokens: int = 128, system: str | None = None, max_side: int | None = -1) -> str:
        images = images if isinstance(images, (list, tuple)) else [images]
        return self.generate([self.messages(list(images), prompt, system, max_side)], max_new_tokens)[0]

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


def field_scores(pred: Fields, truth: Fields) -> dict:
    """Week 4 Day 5 scorer: per field (exact match, ANLS-style similarity)."""
    out = {}
    for k in FIELD_NAMES:
        p, t = getattr(pred, k), getattr(truth, k)
        if k in MONEY_FIELDS:
            exact = p is not None and abs(p - t) < 0.01
            sim = float(exact)
        else:
            exact = norm_text(p) == norm_text(t)
            sim = Levenshtein.normalized_similarity(norm_text(p), norm_text(t)) if p is not None else 0.0
            sim = sim if sim >= 0.5 else 0.0
        out[k] = (exact, sim)
    return out

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


def run_rules(d: "SynthDoc") -> tuple[dict, float, str]:
    """Week 4 self-hosted KV path (deskew -> Tesseract -> label rules). Returns (raw fields, latency, full OCR text)."""
    t0 = time.perf_counter()
    raw, texts = {}, []
    for i, page in enumerate(d.pages):
        arr, _ = deskew(np.asarray(page.convert("RGB")))
        words = ocr_words(Image.fromarray(arr))
        texts.append(" ".join(w["text"] for w in words))
        for k, v in extract_fields(words, page_index=i).items():
            if k in MONEY_FIELDS:
                raw[k] = v
            else:
                raw.setdefault(k, v)
    return raw, time.perf_counter() - t0, "\n".join(texts)


def grounded(fname: str, value, ocr_text: str) -> bool:
    """Does an independent OCR pass find this value on the page? (Day 2 hallucination check.)"""
    if value is None:
        return True
    if fname in MONEY_FIELDS:
        return round(float(value), 2) in {parse_money(x) for x in re.findall(r"\d[\d,]*\.\d{2}", ocr_text)}
    if fname in DATE_FIELDS:
        found = {parse_date(m) for m in re.findall(r"[A-Za-z]{3,9} \d{1,2}, \d{4}|\d{4}-\d{2}-\d{2}|\d{1,2} [A-Za-z]{3} \d{4}|\d{1,2}/\d{1,2}/\d{4}", ocr_text)}
        return value in found
    if fname == "currency":
        return True
    return fuzz.partial_ratio(norm_text(value), norm_text(ocr_text)) >= 90


# %%
# ---- structured.py: helper code for this day (self-contained, no external project files) ----
# Three ways to get schema-valid JSON out of a VLM, behind one function: structured_call(mode, ...).
def extract_json(text: str) -> str | None:
    """The first balanced {...} object in a model reply (tolerates ```json fences and chatter around it)."""
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


def validate(text: str, schema: type[BaseModel]) -> tuple[dict | None, BaseModel | None, str | None]:
    """(parsed JSON or None, validated object or None, error message or None)."""
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


def png_backed(img: Image.Image) -> Image.Image:
    """Outlines' Image input needs a PIL image with a .format; round-trip through PNG bytes."""
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    out = Image.open(io.BytesIO(buf.getvalue()))
    out.load()
    return out


class ConstrainedRunner:
    """Outlines constrained decoding around the same HF model and processor the VLM wrapper uses."""

    def __init__(self, vlm: "VLM"):
        import outlines
        from outlines.inputs import Image as OutlinesImage
        self.vlm, self.OutlinesImage = vlm, OutlinesImage
        self.model = outlines.from_transformers(vlm.model, vlm.processor)

    def __call__(self, images: list, prompt: str, schema: type[BaseModel], max_new_tokens: int, max_side: int) -> str:
        imgs = [png_backed(fit_image(im, max_side)) for im in images]
        conv = [{"role": "user", "content": [{"type": "image"} for _ in imgs] + [{"type": "text", "text": prompt}]}]
        text = self.vlm.processor.apply_chat_template(conv, tokenize=False, add_generation_prompt=True)
        out = self.model([text, *[self.OutlinesImage(im) for im in imgs]], output_type=schema, max_new_tokens=max_new_tokens)
        return out if isinstance(out, str) else json.dumps(out)


def schema_prompt(schema: type[BaseModel]) -> str:
    return ("Reply with a single JSON object and nothing else. It must validate against this JSON Schema:\n"
            + json.dumps(schema.model_json_schema(), separators=(",", ":")))


def structured_call(mode: str, images: list, task: str, schema: type[BaseModel], keys_hint: str,
                    max_new_tokens: int = 400, max_side: int | None = None) -> dict:
    """Run one structured request. Returns raw text, parsed JSON, validated object, attempts, latency."""
    side = max_side or VLM_MAX_SIDE
    t0 = time.perf_counter()
    if mode == "constrained":
        try:
            raw = CONSTRAINED(images, task + "\n" + schema_prompt(schema), schema, max_new_tokens, side)
        except Exception as exc:  # generation hit max_new_tokens mid-object, or an Outlines/transformers mismatch
            raw = f"<constrained generation failed: {exc!r}>"[:500]
        data, obj, err = validate(raw, schema)
        return {"raw": raw, "data": data, "obj": obj, "error": err, "attempts": 1, "latency_s": time.perf_counter() - t0}
    if mode == "prompt_only":
        prompt = task + f"\nAnswer in JSON with these keys: {keys_hint}"
        raw = QWEN.ask(images, prompt, max_new_tokens=max_new_tokens, max_side=side)
        data, obj, err = validate(raw, schema)
        return {"raw": raw, "data": data, "obj": obj, "error": err, "attempts": 1, "latency_s": time.perf_counter() - t0}
    # schema_repair: validate, then show the model its own answer and the validation error
    prompt = task + "\n" + schema_prompt(schema)
    conv = QWEN.messages(images, prompt, max_side=side)
    for attempt in range(1, MAX_REPAIRS + 2):
        raw = QWEN.generate([conv], max_new_tokens=max_new_tokens)[0]
        data, obj, err = validate(raw, schema)
        if obj is not None or attempt > MAX_REPAIRS:
            break
        conv = conv + [{"role": "assistant", "content": [{"type": "text", "text": raw}]},
                       {"role": "user", "content": [{"type": "text", "text": f"That reply is not valid: {err}. "
                                                     "Return the corrected JSON object only."}]}]
    return {"raw": raw, "data": data, "obj": obj, "error": err, "attempts": attempt, "latency_s": time.perf_counter() - t0}


# %% [markdown]
# ## 3 · Load the VLM and the constrained-decoding wrapper

# %%
VLM_MAX_SIDE = 896
QWEN = VLM(VLM_ID, max_side=VLM_MAX_SIDE)
try:
    CONSTRAINED = ConstrainedRunner(QWEN)
    CONSTRAINED_ERROR = None
except Exception as exc:  # keep the notebook useful even if the installed Outlines version has a different API
    CONSTRAINED, CONSTRAINED_ERROR = None, repr(exc)
    print("Constrained decoding unavailable:", CONSTRAINED_ERROR)
MODES = ["prompt_only", "schema_repair"] + (["constrained"] if CONSTRAINED else [])
print("modes:", MODES)

# %% [markdown]
# ## Part A — Natural-language explanations of CV output
#
# ### 4 · Run the CV models: YOLO11n (detection) and ResNet-50 (classification)

# %%
FLICKR_ROOT = kaggle_dataset("adityajn105/flickr8k", "flickr8k")
FLICKR_DIR, FLICKR_CAPS = load_flickr8k(FLICKR_ROOT)
FLICKR_TEST, _ = flickr_split(FLICKR_CAPS)

detector = YOLO("yolo11n.pt")
weights = torchvision.models.ResNet50_Weights.DEFAULT
classifier = torchvision.models.resnet50(weights=weights).eval().to(DEVICE)
cls_tf = weights.transforms()
CATEGORIES = weights.meta["categories"]


def where(box, w, h) -> str:
    cx, cy = (box[0] + box[2]) / 2 / w, (box[1] + box[3]) / 2 / h
    if (box[2] - box[0]) * (box[3] - box[1]) > 0.6 * w * h:
        return "whole image"
    return "left" if cx < 1 / 3 else "right" if cx > 2 / 3 else ("top" if cy < 1 / 3 else "bottom" if cy > 2 / 3 else "center")


def detect(img: Image.Image) -> list[dict]:
    res = detector(img, conf=DET_CONF, verbose=False)[0]
    return [{"label": res.names[int(c)], "confidence": round(float(s), 2), "where": where(b.tolist(), img.width, img.height)}
            for b, c, s in zip(res.boxes.xyxy, res.boxes.cls, res.boxes.conf)]


@torch.inference_mode()
def classify(img: Image.Image, k: int = 3) -> list[str]:
    probs = classifier(cls_tf(img).unsqueeze(0).to(DEVICE)).softmax(-1)[0]
    top = probs.topk(k)
    return [f"{CATEGORIES[i]} ({p:.2f})" for p, i in zip(top.values.tolist(), top.indices.tolist())]


SCENES = []
for name in FLICKR_TEST[200:]:
    img = Image.open(FLICKR_DIR / name).convert("RGB")
    dets = detect(img)
    if dets:
        SCENES.append({"name": name, "image": img, "detections": dets, "classifier": classify(img), "captions": FLICKR_CAPS[name]})
    if len(SCENES) >= N_SCENES:
        break
print(f"{len(SCENES)} scenes | example detections: {SCENES[0]['detections'][:4]} | classifier: {SCENES[0]['classifier']}")

# %% [markdown]
# ### 5 · The explanation schema
#
# Enums (`Literal`) and bounds make the output machine-checkable: a dashboard can colour `classifier_assessment`, an alert can fire on `likely_false_positives`.

# %%
class ObjectNote(BaseModel):
    label: str = Field(description="detector class name, as given", max_length=40)
    count: int = Field(ge=0, le=50, description="how many the detector reported")
    where: Literal["left", "center", "right", "top", "bottom", "whole image"]


class SceneExplanation(BaseModel):
    summary: str = Field(min_length=10, max_length=400, description="2-3 sentence explanation for a non-expert")
    objects: list[ObjectNote] = Field(description="the detector's objects you can confirm in the image")
    likely_false_positives: list[str] = Field(description="detector labels that do not match the image")
    missed_objects: list[str] = Field(description="clearly visible objects the detector did not report")
    classifier_assessment: Literal["agree", "partially_agree", "disagree"]
    confidence: Literal["low", "medium", "high"]


SCENE_KEYS = ("summary, objects (list of {label, count, where}), likely_false_positives (list), missed_objects (list), "
              "classifier_assessment (agree / partially_agree / disagree), confidence (low / medium / high)")


def scene_task(dets: list[dict], cls: list[str]) -> str:
    return ("You are checking the output of a computer-vision system against the image.\n"
            f"Object detector (YOLO11n, COCO classes) output: {json.dumps(dets)}\n"
            f"Image classifier (ResNet-50, ImageNet) top-3: {json.dumps(cls)}\n"
            "Explain the result for a non-expert. Look at the image yourself: confirm the detector's objects with counts, "
            "list detections that look wrong, list important objects it missed, and say whether the classifier's top label fits.")


print(json.dumps(SceneExplanation.model_json_schema(), indent=1)[:900], "...")

# %% [markdown]
# ### 6 · Compliance: the same task in all three modes
#
# Beyond "is it valid JSON", we check that the explanation is **consistent with the CV output it explains**: every confirmed object is a label the detector actually produced, with the detector's count.

# %%
def singular(s: str) -> str:
    s = s.strip().lower()
    return s[:-1] if s.endswith("s") and not s.endswith("ss") else s


def consistency(obj: SceneExplanation | None, dets: list[dict]) -> tuple[float, float]:
    """(fraction of objects whose label the detector produced, fraction whose count matches the detector)."""
    if obj is None or not obj.objects:
        return (np.nan, np.nan)
    counts = Counter(singular(d["label"]) for d in dets)
    labels_ok = [singular(o.label) in counts for o in obj.objects]
    counts_ok = [counts.get(singular(o.label)) == o.count for o in obj.objects]
    return float(np.mean(labels_ok)), float(np.mean(counts_ok))


SCENE_ROWS, SCENE_OUT = [], {}
for mode in MODES:
    for s in SCENES:
        r = structured_call(mode, [s["image"]], scene_task(s["detections"], s["classifier"]), SceneExplanation, SCENE_KEYS)
        lab, cnt = consistency(r["obj"], s["detections"])
        SCENE_OUT[(mode, s["name"])] = r
        SCENE_ROWS.append({"mode": mode, "image": s["name"], "parsed": r["data"] is not None, "valid": r["obj"] is not None,
                           "attempts": r["attempts"], "latency_s": r["latency_s"], "labels_consistent": lab, "counts_consistent": cnt,
                           "error": r["error"]})
    print(f"{mode}: done")
SCENE_DF = pd.DataFrame(SCENE_ROWS)
SCENE_DF.to_csv(OUTPUT_DIR / "day3_scene_compliance.csv", index=False)
COMPLIANCE = SCENE_DF.groupby("mode").agg(json_parse_rate=("parsed", "mean"), schema_valid_rate=("valid", "mean"),
                                         mean_attempts=("attempts", "mean"), latency_s=("latency_s", "mean"),
                                         labels_consistent=("labels_consistent", "mean"), counts_consistent=("counts_consistent", "mean"))
print(COMPLIANCE.round(3).to_string())
print("\nTypical validation errors:\n", SCENE_DF[~SCENE_DF.valid].error.value_counts().head(6).to_string())

# %%
best_mode = "constrained" if "constrained" in MODES else "schema_repair"
fig, axes = plt.subplots(1, 4, figsize=(24, 6))
for ax, s in zip(axes, SCENES[:4]):
    r = SCENE_OUT[(best_mode, s["name"])]
    ax.imshow(s["image"])
    ax.axis("off")
    det_txt = ", ".join(f"{d['label']} {d['confidence']}" for d in s["detections"][:5])
    expl = r["obj"].summary if r["obj"] else f"INVALID: {r['error']}"
    ax.set_title(f"YOLO: {det_txt}\nResNet: {s['classifier'][0]}\n\n{expl[:220]}", fontsize=7, wrap=True)
fig.suptitle(f"VLM explanations of detector + classifier output ({best_mode})")
save_fig(fig, "day3_explanations.png")
for s in SCENES[:2]:
    r = SCENE_OUT[(best_mode, s["name"])]
    print(json.dumps(r["data"], indent=1) if r["data"] else r["raw"][:400])

# %% [markdown]
# ### 7 · Does the VLM blindly trust the detector?
#
# Inject one fake, high-confidence detection of a class that is not in the image (e.g. `giraffe 0.88`) and see whether the explanation **flags it** as a likely false positive or **repeats it** as a confirmed object. A useful explainer must be able to disagree with the model it explains.

# %%
FAKES = ["giraffe", "zebra", "elephant", "airplane", "toilet", "pizza", "refrigerator", "surfboard"]
inj_rows = []
for s in SCENES:
    text = " ".join(s["captions"]).lower() + " " + " ".join(d["label"] for d in s["detections"])
    fake = next((f for f in FAKES if f not in text), "giraffe")
    dets = s["detections"] + [{"label": fake, "confidence": 0.88, "where": random.choice(["left", "right", "center"])}]
    r = structured_call(best_mode, [s["image"]], scene_task(dets, s["classifier"]), SceneExplanation, SCENE_KEYS)
    o = r["obj"]
    inj_rows.append({"image": s["name"], "fake": fake, "valid": o is not None,
                     "flagged_fake": bool(o and any(fake in x.lower() for x in o.likely_false_positives)),
                     "confirmed_fake": bool(o and any(singular(fake) == singular(x.label) for x in o.objects)),
                     "fake_in_summary": bool(o and fake in o.summary.lower())})
INJ = pd.DataFrame(inj_rows)
INJ.to_csv(OUTPUT_DIR / "day3_injected_false_positive.csv", index=False)
print(INJ[["valid", "flagged_fake", "confirmed_fake", "fake_in_summary"]].mean().round(3).to_string())

# %% [markdown]
# ## Part B — Schema-constrained invoice extraction, benchmarked against Week 4
#
# ### 8 · One call per invoice → the Week 4 `Fields` record
#
# Every key is **required but nullable**: the model must output all nine keys and use `null` when a value is absent, instead of silently dropping keys. Dates and currency carry regex patterns, money is `number`.

# %%
class InvoiceExtraction(BaseModel):
    invoice_number: str | None = Field(description="invoice / bill / document number as printed")
    invoice_date: str | None = Field(pattern=r"^\d{4}-\d{2}-\d{2}$", description="issue date as YYYY-MM-DD")
    due_date: str | None = Field(pattern=r"^\d{4}-\d{2}-\d{2}$", description="payment due date as YYYY-MM-DD")
    vendor_name: str | None = Field(description="company that issued the invoice")
    bill_to: str | None = Field(description="customer the invoice is billed to (name only)")
    subtotal: float | None = Field(description="amount before tax")
    tax: float | None = Field(description="tax amount (not the rate)")
    total: float | None = Field(description="final amount due including tax")
    currency: str | None = Field(pattern=r"^[A-Z]{3}$", description="ISO 4217 code, e.g. USD")


INVOICE_TASK = "Extract the key fields of this invoice. Read values exactly from the document; use null if a field is not present."
INVOICE_KEYS = ", ".join(FIELD_NAMES) + " (dates as YYYY-MM-DD, amounts as numbers)"

EVAL_LAYOUTS = LAYOUTS + HELDOUT_LAYOUTS
DOCS = make_invoice_set(N_PER_LAYOUT, seed=HELDOUT_SEED, layouts=EVAL_LAYOUTS, prefix="d5")
TRUTH = {d.doc_id: Fields(**d.truth["fields"]) for d in DOCS}
print(f"{len(DOCS)} invoices")


def score(approach: str, d, raw_fields: dict | None, latency: float, valid: bool, ocr_text: str, attempts: int = 1) -> dict:
    """Accuracy is computed from whatever JSON came back (normalized like Week 4); schema validity is reported separately."""
    pred = Fields(**normalize_fields(raw_fields or {}))
    fs = field_scores(pred, TRUTH[d.doc_id])
    p = pred.model_dump()
    arith_ok = None not in (p["subtotal"], p["tax"], p["total"]) and abs(p["subtotal"] + p["tax"] - p["total"]) <= 0.02
    ungrounded = [k for k in FIELD_NAMES if not grounded(k, p[k], ocr_text)]
    wrong = [k for k, (e, _) in fs.items() if not e]
    return {"approach": approach, "doc_id": d.doc_id, "layout": d.layout, "pages": len(d.pages), "schema_valid": valid,
            "field_acc": np.mean([e for e, _ in fs.values()]), "field_anls": np.mean([s for _, s in fs.values()]),
            "latency_s_per_page": latency / len(d.pages), "attempts": attempts, "arith_ok": arith_ok,
            "needs_review": (not arith_ok) or bool(ungrounded), "has_error": bool(wrong),
            "wrong_fields": ",".join(wrong), "ungrounded_fields": ",".join(ungrounded)}


INV_ROWS = []
for d in DOCS:
    raw_rules, lat_rules, ocr_text = run_rules(d)
    INV_ROWS.append(score("selfhosted_rules", d, raw_rules, lat_rules, True, ocr_text))
    for mode in MODES:
        r = structured_call(mode, d.pages, INVOICE_TASK, InvoiceExtraction, INVOICE_KEYS, max_new_tokens=300, max_side=DOC_MAX_SIDE)
        INV_ROWS.append(score(f"vlm_{mode}", d, r["data"], r["latency_s"], r["obj"] is not None, ocr_text, r["attempts"]))
    print(d.doc_id, {row["approach"]: round(row["field_acc"], 2) for row in INV_ROWS[-1 - len(MODES):]})
INV = pd.DataFrame(INV_ROWS)
INV["cost_per_page"] = INV.latency_s_per_page / 3600 * GPU_USD_PER_HOUR
INV.to_csv(OUTPUT_DIR / "day3_invoice_scores.csv", index=False)
print(INV.groupby("approach")[["schema_valid", "field_acc", "field_anls", "latency_s_per_page", "attempts"]].mean().round(3).to_string())

# %% [markdown]
# ### 9 · Valid ≠ correct: validation still catches wrong numbers
#
# Constrained decoding guarantees the *shape*. It cannot stop the model writing `1,284.50` when the page says `1,248.50`. The `needs_review` flag (arithmetic check or a value the OCR pass cannot find) is the safety net: how many wrong documents does it route to a human, and how many correct ones does it bother them with?

# %%
for approach, g in INV[INV.approach.str.startswith("vlm_")].groupby("approach"):
    caught = (g.needs_review & g.has_error).sum()
    print(f"{approach:20s} docs with an error: {g.has_error.sum():2d} | flagged: {g.needs_review.sum():2d} | "
          f"errors caught: {caught}/{g.has_error.sum()} | false alarms: {(g.needs_review & ~g.has_error).sum()}")
print("\nBy layout (field accuracy):")
print(INV.pivot_table(index="approach", columns="layout", values="field_acc").reindex(columns=list(EVAL_LAYOUTS)).round(2).to_string())

# %% [markdown]
# ### 10 · Three-way comparison: self-hosted OCR vs managed service vs VLM
#
# Rows for the managed services come from your Week 4 Day 5 `day5_scores_per_doc.csv` and Day 2's per-field VQA from `day2_invoice_scores.csv` — upload either file to `/content/` (or keep the repo's `results/` folders) and they are filtered to **the same document ids**. Without them, the table shows this notebook's approaches only.

# %%
def find_file(name: str, *dirs: Path) -> Path | None:
    for d in (Path("/content"), *dirs):
        if (d / name).exists():
            return d / name
    return None


METRICS = ["field_acc", "field_anls", "latency_s_per_page", "cost_per_page"]
ids = {d.doc_id for d in DOCS}
frames = [INV]
w4 = find_file("day5_scores_per_doc.csv", HERE.parent.parent / "week4_documents" / "results" / "day5")
if w4:
    df = pd.read_csv(w4)
    frames.append(df[df.doc_id.isin(ids) & df.approach.isin(["azure", "textract", "selfhosted_tatr"])].assign(schema_valid=True))
d2 = find_file("day2_invoice_scores.csv", HERE.parent / "results" / "day2", Path("/content/week5_outputs/day2"))
if d2:
    df = pd.read_csv(d2)
    frames.append(df[df.doc_id.isin(ids) & (df.approach == "vlm_vqa")].assign(schema_valid=True))
print("Week 4 scores:", w4, "| Day 2 scores:", d2)
ALL = pd.concat(frames, ignore_index=True)
THREE_WAY = ALL.groupby("approach").agg(**{m: (m, "mean") for m in METRICS}, schema_valid_rate=("schema_valid", "mean"),
                                        docs=("doc_id", "nunique"))
THREE_WAY["cost_per_1k_pages"] = THREE_WAY.pop("cost_per_page") * 1000
unseen = ALL[ALL.layout == "layout_c"].groupby("approach").field_acc.mean()
THREE_WAY["unseen_layout_field_acc"] = unseen
FAMILY = {"selfhosted_rules": "self-hosted OCR", "selfhosted_tatr": "self-hosted OCR", "azure": "managed", "textract": "managed"}
THREE_WAY.insert(0, "family", [FAMILY.get(a, "VLM") for a in THREE_WAY.index])
QUAL = {
    "self-hosted OCR": {"setup": "low-medium", "privacy": "on-prem", "new layout": "write new rules", "typical failure": "missed label variants"},
    "managed": {"setup": "low (keys)", "privacy": "leaves premises", "new layout": "prebuilt generalizes", "typical failure": "per-page cost, vendor schema"},
    "VLM": {"setup": "medium (GPU, 8 GB weights)", "privacy": "on-prem", "new layout": "zero-shot via prompt",
            "typical failure": "confident wrong digits; slow"},
}
for col in ("setup", "privacy", "new layout", "typical failure"):
    THREE_WAY[col] = [QUAL[f][col] for f in THREE_WAY.family]
THREE_WAY = THREE_WAY.round(3).sort_values("field_acc", ascending=False)
print(THREE_WAY.to_string())
THREE_WAY.to_csv(OUTPUT_DIR / "three_way_comparison.csv")

best = THREE_WAY.index[0]
fast = THREE_WAY.latency_s_per_page.idxmin()
vlm_rows = THREE_WAY[THREE_WAY.family == "VLM"]
lines = [f"# Three-way comparison (Week 4 held-out invoices, seed {HELDOUT_SEED}, {len(DOCS)} documents)", "",
         THREE_WAY.to_markdown(), "",
         f"- Most accurate: **{best}** (field accuracy {THREE_WAY.loc[best, 'field_acc']:.2f}).",
         f"- Fastest: **{fast}** ({THREE_WAY.loc[fast, 'latency_s_per_page']:.2f} s/page)."]
if len(vlm_rows):
    v = vlm_rows.field_acc.idxmax()
    lines.append(f"- Best VLM mode: **{v}** — accuracy {vlm_rows.loc[v, 'field_acc']:.2f}, schema-valid {vlm_rows.loc[v, 'schema_valid_rate']:.0%}, "
                 f"unseen layout {vlm_rows.loc[v, 'unseen_layout_field_acc']:.2f}.")
(OUTPUT_DIR / "three_way_comparison.md").write_text("\n".join(lines), encoding="utf-8")
print("\n".join(lines[-3:]))

# %%
fig, ax = plt.subplots(figsize=(8, 5))
colors = {"self-hosted OCR": "tab:blue", "managed": "tab:orange", "VLM": "tab:green"}
for name, row in THREE_WAY.iterrows():
    ax.scatter(row.latency_s_per_page, row.field_acc, s=140, color=colors[row.family], label=row.family)
    ax.annotate(name, (row.latency_s_per_page, row.field_acc), textcoords="offset points", xytext=(6, 4), fontsize=8)
handles, labels = ax.get_legend_handles_labels()
ax.legend(dict(zip(labels, handles)).values(), dict(zip(labels, handles)).keys())
ax.set_xlabel("latency per page (s)")
ax.set_ylabel("field accuracy")
ax.set_title("Accuracy vs latency on the Week 4 held-out invoices")
ax.grid(alpha=0.3)
save_fig(fig, "day3_three_way.png")

(OUTPUT_DIR / "day3_summary.json").write_text(json.dumps({
    "model": VLM_ID, "modes": MODES, "constrained_error": CONSTRAINED_ERROR,
    "scene_compliance": COMPLIANCE.round(4).to_dict("index"), "injected_fp": INJ[["flagged_fake", "confirmed_fake"]].mean().to_dict(),
    "three_way": THREE_WAY.drop(columns=["setup", "privacy", "new layout", "typical failure"]).to_dict("index"),
}, indent=2, default=str), encoding="utf-8")
print("saved ->", OUTPUT_DIR / "day3_summary.json")

# %% [markdown]
# ## Checklist
#
# - [ ] Schema-valid rate per mode, with the validation errors that `prompt_only` produces.
# - [ ] Explanations are consistent with the detector output (labels and counts), and I measured whether the VLM flags an injected false positive.
# - [ ] One-call invoice extraction scored with Week 4's metric on Week 4's documents; three-way table saved.
# - [ ] I can say why constrained decoding makes output *valid* but not *correct*, and what catches the rest.
#
# **Next (Day 4):** captioning — turning any image or figure into searchable text.
