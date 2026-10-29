"""
Week 5 · Day 4 — Image captioning: BLIP, BLIP-2 and Qwen3-VL on Flickr8k and on figures; caption-as-text vs raw-embedding indexes.

Colab-first script (percent-format cells). Build the notebook with:
    python ../tools/py_to_ipynb.py day4_image_captioning/day4_image_captioning.py
Dataset: Flickr8k from Kaggle (https://www.kaggle.com/datasets/adityajn105/flickr8k)
"""

# %% [markdown]
# # Week 5 · Day 4 — Image Captioning
#
# **Goal:** turn any image — a photo, a chart, a scanned figure — into **searchable text**, measure how good that text is, and find out when a caption is a better index than a raw image embedding. Week 6 indexes every figure in a document corpus with exactly these functions.
#
# | Topic | What you will build / measure |
# |---|---|
# | Three generations of captioners | BLIP (encoder-decoder, 2022), BLIP-2 (frozen ViT + Q-Former + frozen OPT-2.7B, 2023), Qwen3-VL (instruction-tuned VLM, promptable style) |
# | Reference metrics | BLEU-4, ROUGE-L, CIDEr against human captions — plus a **human baseline** (one human caption scored against the other four) |
# | Reference-free metric | CLIPScore: does the caption match the *image*, regardless of wording? |
# | Figures | Captions of synthetic charts with known content: are the title, numbers and trend in the caption? |
# | Caption vs embedding | Retrieve images by text with (a) BM25 over captions, (b) dense text embeddings of captions, (c) CLIP image embeddings, (d) a hybrid |
# | Reusable utility | `caption_image(img, style)` for Week 6 |
#
# **Data:** [Flickr8k on Kaggle](https://www.kaggle.com/datasets/adityajn105/flickr8k) (same 1,000-image test split as Days 1–3) plus charts rendered in code.
#
# > Runtime: **T4 GPU**, ~25–35 min with `FULL_RUN=True`. Models are loaded one at a time and freed, so peak memory stays under 10 GB.

# %% [markdown]
# ## 0 · Install dependencies

# %%
import subprocess
import sys


def pip_install(*packages: str) -> None:
    """Install packages into the running interpreter (works in Colab and plain Python)."""
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *packages])


pip_install("transformers>=4.57", "accelerate>=1.0", "kagglehub>=0.3", "open_clip_torch>=2.24", "pycocoevalcap==1.2",
            "sentence-transformers>=3.0", "rank_bm25==0.2.2")

# %% [markdown]
# ## 1 · Imports, configuration, and output folder

# %%
import gc
import io
import json
import os
import random
import re
import time
from pathlib import Path

import matplotlib

IN_NOTEBOOK = "ipykernel" in sys.modules
if not IN_NOTEBOOK:
    matplotlib.use("Agg")  # headless when run as a script

import matplotlib.pyplot as plt
import numpy as np
import open_clip
import pandas as pd
import torch
from PIL import Image
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.rouge.rouge import Rouge
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from transformers import (
    AutoModelForImageTextToText,
    AutoProcessor,
    Blip2ForConditionalGeneration,
    Blip2Processor,
    BlipForConditionalGeneration,
    BlipProcessor,
)

IN_COLAB = "google.colab" in sys.modules
try:
    HERE = Path(__file__).resolve().parent
except NameError:  # running inside a notebook
    HERE = Path.cwd()

OUTPUT_DIR = Path("/content/week5_outputs/day4") if IN_COLAB else HERE.parent / "results" / "day4"
DATA_DIR = Path("/content/datasets") if IN_COLAB else HERE.parent / "data"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)

FULL_RUN = True  # False = quick smoke test
VLM_ID = "Qwen/Qwen3-VL-4B-Instruct"
BLIP_ID = "Salesforce/blip-image-captioning-large"
BLIP2_ID = "Salesforce/blip2-opt-2.7b"
RUN_BLIP2 = True  # ~7.5 GB download; set False to save time
N_EVAL = 150 if FULL_RUN else 12  # captioned by every model; also the retrieval gallery
BATCH = 8
QWEN_BATCH = 4
TEXT_EMBEDDER = "sentence-transformers/all-MiniLM-L6-v2"
SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
FP = torch.float16 if DEVICE == "cuda" else torch.float32
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


def free(*names: str) -> None:
    """Delete global model objects by name and return their GPU memory."""
    for n in names:
        globals().pop(n, None)
    gc.collect()
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
        print(f"GPU memory in use: {torch.cuda.memory_allocated() / 1e9:.1f} GB")


# %% [markdown]
# ## 2 · Helper code
#
# **Kaggle credentials:** Colab **key icon (Secrets)** → `KAGGLE_API_TOKEN`, or `KAGGLE_USERNAME` + `KAGGLE_KEY`. Or unzip Flickr8k into `/content/datasets/flickr8k/`.

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

# %% [markdown]
# ## 3 · Data: Flickr8k test images and synthetic figures
#
# Each image has 5 human captions. For a fair human baseline, models and the held-out human caption are all scored against the **same 4 references** (captions 2–5).

# %%
FLICKR_ROOT = kaggle_dataset("adityajn105/flickr8k", "flickr8k")
IMAGE_DIR, CAPTIONS = load_flickr8k(FLICKR_ROOT)
TEST_NAMES, _ = flickr_split(CAPTIONS)
EVAL = TEST_NAMES[:N_EVAL]
IMAGES = {n: Image.open(IMAGE_DIR / n).convert("RGB") for n in EVAL}
print(f"eval images: {len(EVAL)}")


def fig_to_image(fig) -> Image.Image:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
    plt.close(fig)
    return Image.open(io.BytesIO(buf.getvalue())).convert("RGB")


def make_figures() -> list[dict]:
    """Charts with known content. 'facts' are words a good caption should contain."""
    out = []
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(["Q1", "Q2", "Q3", "Q4"], [4.2, 5.1, 3.8, 6.4], color="tab:blue")
    ax.set_title("Quarterly revenue 2025 (USD millions)")
    out.append({"id": "fig_bar_revenue", "image": fig_to_image(fig), "query": "bar chart of quarterly revenue in 2025",
                "facts": ["revenue", "quarter|q1|q2|q3|q4", "bar", "6.4|q4"]})
    fig, ax = plt.subplots(figsize=(6, 4))
    months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun"]
    ax.plot(months, [2, 4, 9, 14, 18, 22], marker="o", label="Oslo")
    ax.plot(months, [10, 12, 15, 18, 22, 26], marker="s", label="Rome")
    ax.set_title("Average temperature (°C)")
    ax.legend()
    out.append({"id": "fig_line_temperature", "image": fig_to_image(fig), "query": "line chart comparing temperature in Oslo and Rome",
                "facts": ["temperature", "oslo", "rome", "line"]})
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.pie([45, 30, 15, 10], labels=["Android", "iOS", "Windows", "Other"], autopct="%1.0f%%")
    ax.set_title("Device market share")
    out.append({"id": "fig_pie_market", "image": fig_to_image(fig), "query": "pie chart of device market share",
                "facts": ["market share", "android", "ios", "pie|45"]})
    return out


FIGURES = make_figures()
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for ax, f in zip(axes, FIGURES):
    ax.imshow(f["image"])
    ax.set_title(f["id"])
    ax.axis("off")
save_fig(fig, "day4_figures.png")

CAPS: dict[str, dict[str, str]] = {}  # model -> {image id: caption}
LAT: dict[str, float] = {}  # model -> seconds per image

# %% [markdown]
# ## 4 · BLIP (2022): image encoder + text decoder trained for captioning
#
# One ViT encoder and a BERT-style decoder with cross-attention to image features, pretrained on 129M image-text pairs with *CapFilt* (a captioner writes synthetic captions, a filter drops noisy ones). Fast, batchable, short generic captions.

# %%
def run_batched(fn, items: list, bs: int) -> tuple[list[str], float]:
    """fn(list of PIL) -> list of captions. Returns (captions, seconds per image)."""
    out, t0 = [], time.perf_counter()
    for i in range(0, len(items), bs):
        out += fn(items[i:i + bs])
    return out, (time.perf_counter() - t0) / max(1, len(items))


blip_proc = BlipProcessor.from_pretrained(BLIP_ID)
blip = BlipForConditionalGeneration.from_pretrained(BLIP_ID, dtype=FP).to(DEVICE).eval()


@torch.inference_mode()
def blip_caption(imgs: list) -> list[str]:
    inputs = blip_proc(images=imgs, return_tensors="pt").to(DEVICE, FP)
    ids = blip.generate(**inputs, max_new_tokens=30, num_beams=3)
    return [c.strip() for c in blip_proc.batch_decode(ids, skip_special_tokens=True)]


caps, LAT["blip"] = run_batched(blip_caption, [IMAGES[n] for n in EVAL], BATCH)
CAPS["blip"] = dict(zip(EVAL, caps))
CAPS["blip"].update(zip([f["id"] for f in FIGURES], blip_caption([f["image"] for f in FIGURES])))
print(f"BLIP: {LAT['blip']:.3f} s/image |", CAPS["blip"][EVAL[0]])
free("blip", "blip_proc")

# %% [markdown]
# ## 5 · BLIP-2 (2023): a Q-Former bridges a frozen ViT and a frozen LLM
#
# BLIP-2 freezes both an EVA ViT-g and OPT-2.7B, and trains only the **Q-Former** — 32 learned query vectors that cross-attend to the image and come out as 32 "soft prompt" tokens for the LLM. Same idea as Day 1's projector, but with a fixed token budget regardless of resolution. It also answers questions with the prompt `Question: … Answer:`.

# %%
if RUN_BLIP2:
    blip2_proc = Blip2Processor.from_pretrained(BLIP2_ID)
    blip2 = Blip2ForConditionalGeneration.from_pretrained(BLIP2_ID, dtype=FP, device_map="auto").eval()

    @torch.inference_mode()
    def blip2_generate(imgs: list, prompt: str | None = None) -> list[str]:
        kwargs = {"text": [prompt] * len(imgs)} if prompt else {}
        inputs = blip2_proc(images=imgs, return_tensors="pt", **kwargs).to(blip2.device, FP)
        ids = blip2.generate(**inputs, max_new_tokens=30, num_beams=3)
        return [c.strip() for c in blip2_proc.batch_decode(ids, skip_special_tokens=True)]

    caps, LAT["blip2"] = run_batched(blip2_generate, [IMAGES[n] for n in EVAL], BATCH)
    CAPS["blip2"] = dict(zip(EVAL, caps))
    CAPS["blip2"].update(zip([f["id"] for f in FIGURES], blip2_generate([f["image"] for f in FIGURES])))
    print(f"BLIP-2: {LAT['blip2']:.3f} s/image |", CAPS["blip2"][EVAL[0]])
    for q in ["Question: what is the person doing? Answer:", "Question: what is the weather like? Answer:"]:
        print(q, "->", blip2_generate([IMAGES[EVAL[1]]], q)[0])
    free("blip2", "blip2_proc")

# %% [markdown]
# ## 6 · Qwen3-VL: captions on demand, in any style
#
# An instruction-tuned VLM is a *promptable* captioner. Two styles matter for retrieval: a **short** caption (comparable with Flickr's human captions) and a **detailed, search-oriented** one (subjects, actions, setting, colours, visible text) for indexing.

# %%
QWEN = VLM(VLM_ID, max_side=768)
CAPTION_PROMPTS = {
    "short": "Write a one-sentence caption for this photo, like a human would: the main subject and what it is doing.",
    "detailed": ("Describe this image for a search index in 2-3 sentences: main subjects, actions, setting, colours, "
                 "and any visible text, numbers or chart content. No preamble."),
}


def caption_image(img: Image.Image, style: str = "detailed", max_new_tokens: int = 90) -> str:
    """Reusable captioning utility (Week 6 imports this pattern): one image -> one caption string."""
    return QWEN.ask(img, CAPTION_PROMPTS[style], max_new_tokens=max_new_tokens)


def qwen_batch(style: str, max_new_tokens: int):
    def fn(imgs: list) -> list[str]:
        try:
            return QWEN.generate([QWEN.messages([im], CAPTION_PROMPTS[style]) for im in imgs], max_new_tokens=max_new_tokens)
        except Exception as exc:  # some processor versions cannot batch images; fall back to one at a time
            print("batched generation failed, falling back to sequential:", repr(exc)[:120])
            return [caption_image(im, style, max_new_tokens) for im in imgs]
    return fn


for style, tokens in (("short", 40), ("detailed", 90)):
    key = f"qwen3vl_{style}"
    caps, LAT[key] = run_batched(qwen_batch(style, tokens), [IMAGES[n] for n in EVAL], QWEN_BATCH)
    CAPS[key] = dict(zip(EVAL, caps))
    CAPS[key].update({f["id"]: caption_image(f["image"], style, tokens) for f in FIGURES})
    print(f"{key}: {LAT[key]:.2f} s/image |", CAPS[key][EVAL[0]])

# %% [markdown]
# ## 7 · Reference metrics vs a human baseline
#
# - **BLEU-4**: n-gram precision (up to 4-grams) with a brevity penalty.
# - **ROUGE-L**: longest common subsequence F-score.
# - **CIDEr**: TF-IDF-weighted n-gram similarity to the consensus of references — the standard captioning metric; rewards the words humans agree on.
#
# All three reward *wording like the references*. A correct but detailed caption scores low — watch the `qwen3vl_detailed` row.

# %%
def tok(s: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", s.lower()))


def reference_scores(hyps: dict[str, str]) -> dict:
    gts = {n: [tok(c) for c in CAPTIONS[n][1:]] for n in EVAL}
    res = {n: [tok(hyps[n])] for n in EVAL}
    bleu, _ = Bleu(4).compute_score(gts, res)
    rouge, _ = Rouge().compute_score(gts, res)
    cider, _ = Cider().compute_score(gts, res)
    return {"BLEU-1": bleu[0], "BLEU-4": bleu[3], "ROUGE-L": rouge, "CIDEr": cider,
            "words": float(np.mean([len(hyps[n].split()) for n in EVAL]))}


HUMAN = {n: CAPTIONS[n][0] for n in EVAL}
REF = pd.DataFrame({"human (held-out)": reference_scores(HUMAN), **{m: reference_scores(c) for m, c in CAPS.items()}}).T
print(REF.round(3).to_string())

# %% [markdown]
# ## 8 · CLIPScore: reference-free caption quality
#
# [CLIPScore](https://arxiv.org/abs/2104.08718) = 2.5 × max(cos(CLIP(image), CLIP(caption)), 0). It asks "does this text describe this image?", so a detailed caption is not punished for using different words. Caveat: CLIP's text encoder reads only 77 tokens, and it is biased toward what CLIP was trained on.

# %%
clip_model, _, clip_pre = open_clip.create_model_and_transforms("ViT-B-32", pretrained="openai", device=DEVICE)
clip_model.eval()
clip_tok = open_clip.get_tokenizer("ViT-B-32")


@torch.inference_mode()
def clip_images(imgs: list, bs: int = 64) -> np.ndarray:
    out = []
    for i in range(0, len(imgs), bs):
        x = torch.stack([clip_pre(im) for im in imgs[i:i + bs]]).to(DEVICE)
        out.append(torch.nn.functional.normalize(clip_model.encode_image(x).float(), dim=-1).cpu())
    return torch.cat(out).numpy()


@torch.inference_mode()
def clip_texts(texts: list[str], bs: int = 256) -> np.ndarray:
    out = []
    for i in range(0, len(texts), bs):
        out.append(torch.nn.functional.normalize(clip_model.encode_text(clip_tok(texts[i:i + bs]).to(DEVICE)).float(), dim=-1).cpu())
    return torch.cat(out).numpy()


IMG_EMB = clip_images([IMAGES[n] for n in EVAL])


def clip_score(hyps: dict[str, str]) -> float:
    t = clip_texts([hyps[n] for n in EVAL])
    return float(np.mean(2.5 * np.clip((IMG_EMB * t).sum(1), 0, None)))


REF["CLIPScore"] = [clip_score(HUMAN)] + [clip_score(c) for c in CAPS.values()]
REF["s_per_image"] = [np.nan] + [LAT[m] for m in CAPS]
REF = REF.round(3)
print(REF.to_string())
REF.to_csv(OUTPUT_DIR / "day4_caption_metrics.csv")

fig, axes = plt.subplots(1, 3, figsize=(18, 4))
for ax, col in zip(axes, ["CIDEr", "CLIPScore", "words"]):
    REF[col].plot.barh(ax=ax, color=["grey"] + ["tab:blue"] * (len(REF) - 1))
    ax.set_title(col)
    ax.grid(axis="x", alpha=0.3)
fig.suptitle("Reference metrics reward human-like wording; CLIPScore rewards matching the image")
save_fig(fig, "day4_caption_metrics.png")

# %%
fig, axes = plt.subplots(2, 3, figsize=(20, 11))
for ax, n in zip(axes.flat, EVAL[:6]):
    ax.imshow(IMAGES[n])
    ax.axis("off")
    lines = [f"HUMAN: {CAPTIONS[n][0]}"] + [f"{m}: {CAPS[m][n]}" for m in CAPS]
    ax.set_title("\n".join(l[:110] for l in lines), fontsize=7, loc="left")
save_fig(fig, "day4_caption_examples.png")

# %% [markdown]
# ## 9 · Captioning figures: does the caption carry the facts?
#
# For a RAG system a chart caption is only useful if it contains what someone would search for: the title, the series, the key numbers. Fact recall = fraction of each figure's fact patterns found in its caption.

# %%
fig_rows = []
for f in FIGURES:
    for m in CAPS:
        cap = CAPS[m][f["id"]].lower()
        hits = [bool(re.search(p, cap)) for p in f["facts"]]
        fig_rows.append({"figure": f["id"], "model": m, "fact_recall": np.mean(hits), "caption": CAPS[m][f["id"]][:160]})
FIG_DF = pd.DataFrame(fig_rows)
print(FIG_DF.pivot_table(index="model", columns="figure", values="fact_recall").round(2).to_string())
print(FIG_DF[FIG_DF.figure == "fig_bar_revenue"][["model", "caption"]].to_string(index=False))

# %% [markdown]
# ## 10 · Caption-as-text vs raw embedding: which index finds the image?
#
# Query = a held-out human caption; the gallery = the eval images (+ the 3 figures). Five indexes:
#
# | Index | Image is represented as | Query is matched with |
# |---|---|---|
# | `bm25_qwen_detailed` | Qwen3-VL detailed caption (words) | BM25 keyword scoring |
# | `dense_blip` / `dense_qwen_detailed` | caption → MiniLM sentence embedding | MiniLM embedding of the query |
# | `clip_image` | CLIP image embedding | CLIP text embedding of the query |
# | `hybrid` | z-scored `clip_image` + `dense_qwen_detailed` | both |

# %%
text_model = SentenceTransformer(TEXT_EMBEDDER, device=DEVICE)
GALLERY = EVAL + [f["id"] for f in FIGURES]
GALLERY_IMG_EMB = np.vstack([IMG_EMB, clip_images([f["image"] for f in FIGURES])])
queries = [(CAPTIONS[n][0], i) for i, n in enumerate(EVAL)]
fig_queries = [(f["query"], len(EVAL) + j) for j, f in enumerate(FIGURES)]


def zscore(s: np.ndarray) -> np.ndarray:
    return (s - s.mean(1, keepdims=True)) / (s.std(1, keepdims=True) + 1e-9)


def score_matrices(qs: list[str]) -> dict[str, np.ndarray]:
    q_dense = text_model.encode(qs, normalize_embeddings=True, convert_to_numpy=True)
    out = {}
    for m in ("blip", "qwen3vl_detailed"):
        g = text_model.encode([CAPS[m][g] for g in GALLERY], normalize_embeddings=True, convert_to_numpy=True)
        out[f"dense_{'qwen_detailed' if m.startswith('qwen') else m}"] = q_dense @ g.T
    bm25 = BM25Okapi([tok(CAPS["qwen3vl_detailed"][g]).split() for g in GALLERY])
    out["bm25_qwen_detailed"] = np.array([bm25.get_scores(tok(q).split()) for q in qs])
    out["clip_image"] = clip_texts(qs) @ GALLERY_IMG_EMB.T
    out["hybrid"] = zscore(out["clip_image"]) + zscore(out["dense_qwen_detailed"])
    return out


def recall_table(qs: list[tuple[str, int]]) -> pd.DataFrame:
    mats = score_matrices([q for q, _ in qs])
    gt = np.array([i for _, i in qs])
    rows = {}
    for name, s in mats.items():
        ranks = (s > s[np.arange(len(gt)), gt][:, None]).sum(1) + 1
        rows[name] = {"R@1": np.mean(ranks <= 1), "R@5": np.mean(ranks <= 5), "R@10": np.mean(ranks <= 10), "MRR": np.mean(1 / ranks)}
    return pd.DataFrame(rows).T.round(3)


RETR = recall_table(queries)
FIG_RETR = recall_table(fig_queries)
print(f"Photo queries (gallery {len(GALLERY)}):\n{RETR.to_string()}")
print(f"\nFigure queries (find the chart among {len(GALLERY)} images):\n{FIG_RETR.to_string()}")
RETR.to_csv(OUTPUT_DIR / "day4_caption_vs_embedding.csv")

fig, ax = plt.subplots(figsize=(8, 4))
RETR[["R@1", "R@5", "R@10"]].plot.bar(ax=ax)
ax.set_ylim(0, 1)
ax.set_title("Text→image retrieval: caption indexes vs CLIP image embedding")
ax.grid(axis="y", alpha=0.3)
save_fig(fig, "day4_caption_vs_embedding.png")

# %% [markdown]
# Typical pattern: CLIP wins on *visual* queries ("a dog jumping through snow") because the embedding keeps information no caption wrote down; captions win on *text-in-image* queries (chart titles, product names, numbers) and are explainable and editable; the hybrid is the most robust. Week 6 indexes both.

# %% [markdown]
# ## 11 · Save captions for Day 5 and Week 6

# %%
cap_rows = [{"image": k, "model": m, "caption": c} for m, d in CAPS.items() for k, c in d.items()]
pd.DataFrame(cap_rows).to_csv(OUTPUT_DIR / "day4_captions.csv", index=False)
(OUTPUT_DIR / "day4_summary.json").write_text(json.dumps({
    "n_eval": len(EVAL), "metrics": REF.to_dict("index"), "figure_fact_recall": FIG_DF.groupby("model").fact_recall.mean().to_dict(),
    "retrieval_photo": RETR.to_dict("index"), "retrieval_figures": FIG_RETR.to_dict("index"),
}, indent=2, default=float), encoding="utf-8")
print("saved ->", OUTPUT_DIR / "day4_captions.csv", "and day4_summary.json")

# %% [markdown]
# ## Checklist
#
# - [ ] Caption metrics for BLIP, BLIP-2 and two Qwen3-VL styles, next to a human baseline.
# - [ ] I can explain why the detailed caption scores low on CIDEr but well on CLIPScore and retrieval.
# - [ ] I know which captioners keep chart facts (titles, numbers) and which do not.
# - [ ] I have numbers for caption-as-text vs raw-embedding retrieval, on photos and on figures.
# - [ ] `caption_image(img, style)` is ready to reuse.
#
# **Next (Day 5):** CLIP embeddings in depth — text↔image and image↔image retrieval, zero-shot classification, and a reusable index.
