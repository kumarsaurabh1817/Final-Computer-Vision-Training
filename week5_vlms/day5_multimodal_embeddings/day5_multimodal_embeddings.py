"""
Week 5 · Day 5 — Multimodal embeddings with CLIP / SigLIP: text<->image retrieval, zero-shot classification, and a reusable FAISS index.

Colab-first script (percent-format cells). Build the notebook with:
    python ../tools/py_to_ipynb.py day5_multimodal_embeddings/day5_multimodal_embeddings.py
Datasets: Flickr8k from Kaggle (https://www.kaggle.com/datasets/adityajn105/flickr8k), CIFAR-10 test (torchvision download).
"""

# %% [markdown]
# # Week 5 · Day 5 — Multimodal Embeddings (CLIP)
#
# **Goal:** put images and text in **one vector space**, so "nearest neighbour" works across modalities: a text query finds an image, an image finds text, an image finds similar images. This is the retrieval primitive Week 6's multimodal RAG is built on.
#
# | Topic | What you will build / measure |
# |---|---|
# | Contrastive training | How CLIP's symmetric InfoNCE loss creates a shared space (and how SigLIP's sigmoid loss differs) |
# | Benchmark | Text→image and image→text Recall@1/5/10 on 1,000 Flickr8k images × 5,000 captions, for several CLIP/SigLIP models; speed and storage |
# | Anatomy of the space | Matched vs unmatched similarity distributions, the **modality gap**, a similarity matrix |
# | Retrieval demos | text→image, image→image, image→text, and **text+image** composed queries |
# | Zero-shot classification | CIFAR-10 with prompt templates and prompt ensembling — vs your Week 1 CNN |
# | Caption vs embedding vs hybrid | Same 1,000-image gallery indexed by caption text, by CLIP image vectors, and by both |
# | Reusable index | `MultimodalIndex` on FAISS: add images (+ captions), search by text / image / text+image, save and load — for Week 6 |
#
# **Data:** [Flickr8k on Kaggle](https://www.kaggle.com/datasets/adityajn105/flickr8k) (same test split as Days 1–4); CIFAR-10 test set (downloaded by torchvision).
#
# > Runtime: **T4 GPU**, ~15–20 min with `FULL_RUN=True`. No VLM today — CLIP models are 150–430 M parameters.

# %% [markdown]
# ## 0 · Install dependencies

# %%
import subprocess
import sys


def pip_install(*packages: str) -> None:
    """Install packages into the running interpreter (works in Colab and plain Python)."""
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *packages])


pip_install("open_clip_torch>=2.24", "faiss-cpu>=1.8", "kagglehub>=0.3", "sentence-transformers>=3.0", "transformers>=4.57")

# %% [markdown]
# ## 1 · Imports, configuration, and output folder

# %%
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

import faiss
import matplotlib.pyplot as plt
import numpy as np
import open_clip
import pandas as pd
import torch
import torchvision
from PIL import Image
from sentence_transformers import SentenceTransformer
from transformers import BlipForConditionalGeneration, BlipProcessor

IN_COLAB = "google.colab" in sys.modules
try:
    HERE = Path(__file__).resolve().parent
except NameError:  # running inside a notebook
    HERE = Path.cwd()

OUTPUT_DIR = Path("/content/week5_outputs/day5") if IN_COLAB else HERE.parent / "results" / "day5"
DATA_DIR = Path("/content/datasets") if IN_COLAB else HERE.parent / "data"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)

FULL_RUN = True  # False = quick smoke test
N_GALLERY = 1000 if FULL_RUN else 100  # Flickr8k test split
N_CIFAR = 2000 if FULL_RUN else 300
MODELS = [("ViT-B-32", "openai"), ("ViT-B-16", "openai"), ("ViT-B-16-SigLIP", "webli")]
if FULL_RUN:
    MODELS.append(("ViT-L-14", "openai"))
MAIN_MODEL = ("ViT-B-32", "openai")  # used for the anatomy, demos and the saved index
BLIP_ID = "Salesforce/blip-image-captioning-large"
TEXT_EMBEDDER = "sentence-transformers/all-MiniLM-L6-v2"
SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"PyTorch {torch.__version__} | device={DEVICE} | FULL_RUN={FULL_RUN} | FAISS {faiss.__version__}")
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
# ---- embed.py: helper code for this day (self-contained, no external project files) ----
# One interface over any open_clip model: images / texts -> L2-normalized float32 vectors in the same space.
class Embedder:
    def __init__(self, arch: str, pretrained: str):
        self.name = f"{arch}/{pretrained}"
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(arch, pretrained=pretrained, device=DEVICE)
        self.model.eval()
        self.tokenizer = open_clip.get_tokenizer(arch)

    def _autocast(self):
        return torch.autocast("cuda", dtype=torch.float16) if DEVICE == "cuda" else torch.autocast("cpu", enabled=False)

    @torch.inference_mode()
    def images(self, imgs: list, bs: int = 64) -> np.ndarray:
        """PIL images or file paths -> (n, d) unit vectors."""
        out = []
        for i in range(0, len(imgs), bs):
            batch = [Image.open(x).convert("RGB") if isinstance(x, (str, Path)) else x.convert("RGB") for x in imgs[i:i + bs]]
            x = torch.stack([self.preprocess(im) for im in batch]).to(DEVICE)
            with self._autocast():
                out.append(torch.nn.functional.normalize(self.model.encode_image(x).float(), dim=-1).cpu())
        return torch.cat(out).numpy().astype(np.float32)

    @torch.inference_mode()
    def texts(self, texts: list[str], bs: int = 256) -> np.ndarray:
        out = []
        for i in range(0, len(texts), bs):
            with self._autocast():
                t = self.model.encode_text(self.tokenizer(texts[i:i + bs]).to(DEVICE))
            out.append(torch.nn.functional.normalize(t.float(), dim=-1).cpu())
        return torch.cat(out).numpy().astype(np.float32)

    @property
    def params_m(self) -> float:
        return sum(p.numel() for p in self.model.parameters()) / 1e6


def ranks_of(scores: np.ndarray, gt: list[set[int]]) -> np.ndarray:
    """1-based rank of the best-ranked correct item for every query row."""
    order = np.argsort(-scores, axis=1)
    return np.array([next(r for r, j in enumerate(row, 1) if j in g) for row, g in zip(order, gt)])


def recall_at(ranks: np.ndarray, prefix: str) -> dict:
    return {f"{prefix}R@1": np.mean(ranks <= 1), f"{prefix}R@5": np.mean(ranks <= 5), f"{prefix}R@10": np.mean(ranks <= 10),
            f"{prefix}MedR": float(np.median(ranks))}


# %% [markdown]
# ## 3 · How CLIP learns a shared space
#
# A batch of N (image, caption) pairs gives an N×N similarity matrix (cosine × a learned temperature). CLIP's loss is cross-entropy over the rows (each image must pick its caption) **and** the columns (each caption must pick its image) — the diagonal is pulled up, everything else pushed down. SigLIP replaces the softmax with an independent sigmoid per pair, so it does not need huge batches. Below: the loss on a toy batch, before and after "training" makes the diagonal win.

# %%
def clip_loss(sim: torch.Tensor, temperature: float = 0.07) -> torch.Tensor:
    logits = sim / temperature
    labels = torch.arange(sim.shape[0])
    return (torch.nn.functional.cross_entropy(logits, labels) + torch.nn.functional.cross_entropy(logits.T, labels)) / 2


def siglip_loss(sim: torch.Tensor, t: float = 10.0, b: float = -10.0) -> torch.Tensor:
    labels = 2 * torch.eye(sim.shape[0]) - 1  # +1 on the diagonal, -1 elsewhere
    return -torch.nn.functional.logsigmoid(labels * (sim * t + b)).sum() / sim.shape[0]


torch.manual_seed(0)
random_sim = torch.nn.functional.normalize(torch.randn(8, 64), dim=1) @ torch.nn.functional.normalize(torch.randn(8, 64), dim=1).T
aligned_sim = random_sim.clone().fill_diagonal_(0.9)
print(f"CLIP loss   random={clip_loss(random_sim):.3f}  aligned={clip_loss(aligned_sim):.3f}")
print(f"SigLIP loss random={siglip_loss(random_sim):.3f}  aligned={siglip_loss(aligned_sim):.3f}")

# %% [markdown]
# ## 4 · Data

# %%
FLICKR_ROOT = kaggle_dataset("adityajn105/flickr8k", "flickr8k")
IMAGE_DIR, CAPTIONS = load_flickr8k(FLICKR_ROOT)
TEST_NAMES, TRAIN_NAMES = flickr_split(CAPTIONS)
GALLERY = TEST_NAMES[:N_GALLERY]
GALLERY_PATHS = [IMAGE_DIR / n for n in GALLERY]
CAP_TEXTS, CAP_OWNER = [], []
for i, n in enumerate(GALLERY):
    for c in CAPTIONS[n]:
        CAP_TEXTS.append(c)
        CAP_OWNER.append(i)
CAP_OWNER = np.array(CAP_OWNER)
print(f"gallery: {len(GALLERY)} images, {len(CAP_TEXTS)} captions")

# %% [markdown]
# ## 5 · Benchmark: which embedding model retrieves best?
#
# - **Text→image:** each of the 5,000 captions is a query; the hit is its own image among 1,000.
# - **Image→text:** each image is a query; a hit is any of its 5 captions among 5,000.

# %%
BENCH, EMB = [], {}
for arch, pre in MODELS:
    emb = Embedder(arch, pre)
    t0 = time.perf_counter()
    img = emb.images(GALLERY_PATHS)
    t_img = (time.perf_counter() - t0) / len(GALLERY_PATHS)
    t0 = time.perf_counter()
    txt = emb.texts(CAP_TEXTS)
    t_txt = (time.perf_counter() - t0) / len(CAP_TEXTS)
    sim = txt @ img.T  # (captions, images)
    t2i = ranks_of(sim, [{o} for o in CAP_OWNER])
    i2t = ranks_of(sim.T, [set(np.where(CAP_OWNER == i)[0]) for i in range(len(GALLERY))])
    BENCH.append({"model": emb.name, "params_M": round(emb.params_m), "dim": img.shape[1], **recall_at(t2i, "t2i_"),
                  **recall_at(i2t, "i2t_"), "ms_per_image": 1000 * t_img, "ms_per_text": 1000 * t_txt,
                  "GB_per_1M_images_fp32": img.shape[1] * 4 / 1e3})
    EMB[(arch, pre)] = (img, txt)
    print(f"{emb.name}: t2i R@1={BENCH[-1]['t2i_R@1']:.3f}  i2t R@1={BENCH[-1]['i2t_R@1']:.3f}")
    if (arch, pre) == MAIN_MODEL:
        MAIN = emb
    else:
        del emb
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
BENCH_DF = pd.DataFrame(BENCH).set_index("model").round(3)
print(BENCH_DF.to_string())
BENCH_DF.to_csv(OUTPUT_DIR / "day5_retrieval_benchmark.csv")

fig, ax = plt.subplots(figsize=(9, 4))
BENCH_DF[["t2i_R@1", "t2i_R@5", "i2t_R@1", "i2t_R@5"]].plot.bar(ax=ax)
ax.set_ylim(0, 1)
ax.set_title(f"Flickr8k retrieval ({len(GALLERY)} images, {len(CAP_TEXTS)} captions)")
ax.grid(axis="y", alpha=0.3)
plt.xticks(rotation=15)
save_fig(fig, "day5_benchmark.png")

# %% [markdown]
# ## 6 · Anatomy of the shared space
#
# 1. **Matched vs unmatched similarity.** CLIP cosines are small (≈0.2–0.35 for a match): ranking works, but an absolute threshold must be tuned per model.
# 2. **Modality gap.** Image vectors and text vectors live in two separate cones; a matched image-text pair is *less* similar than two random images. This is why "is the nearest neighbour above 0.8?" logic from text-only RAG does not transfer.
# 3. **Similarity matrix** for 8 images × their first captions: the diagonal should win each row.

# %%
img, txt = EMB[MAIN_MODEL]
first_cap = np.array([np.where(CAP_OWNER == i)[0][0] for i in range(len(GALLERY))])
matched = (img * txt[first_cap]).sum(1)
perm = np.random.default_rng(SEED).permutation(len(GALLERY))
unmatched = (img * txt[first_cap[perm]]).sum(1)[perm != np.arange(len(GALLERY))]
ii = (img[:300] @ img[:300].T)[np.triu_indices(300, 1)]
tt = (txt[first_cap[:300]] @ txt[first_cap[:300]].T)[np.triu_indices(300, 1)]
print(f"mean cosine: matched img-txt {matched.mean():.3f} | random img-txt {unmatched.mean():.3f} | "
      f"random img-img {ii.mean():.3f} | random txt-txt {tt.mean():.3f}")
gap = np.linalg.norm(img.mean(0) - txt.mean(0))
print(f"modality gap ||mean(img) - mean(txt)|| = {gap:.3f}")

fig, axes = plt.subplots(1, 3, figsize=(20, 5))
axes[0].hist(unmatched, bins=50, alpha=0.6, density=True, label="unmatched")
axes[0].hist(matched, bins=50, alpha=0.6, density=True, label="matched")
axes[0].set_xlabel("cosine similarity")
axes[0].legend()
axes[0].set_title("image-caption similarity")
both = np.vstack([img[:300], txt[first_cap[:300]]])
both = both - both.mean(0)
_, _, vt = np.linalg.svd(both, full_matrices=False)
proj = both @ vt[:2].T
axes[1].scatter(proj[:300, 0], proj[:300, 1], s=8, label="images")
axes[1].scatter(proj[300:, 0], proj[300:, 1], s=8, label="captions")
for k in range(0, 300, 30):
    axes[1].plot([proj[k, 0], proj[300 + k, 0]], [proj[k, 1], proj[300 + k, 1]], color="grey", lw=0.5)
axes[1].legend()
axes[1].set_title("PCA: the modality gap (grey = matched pairs)")
m = txt[first_cap[:8]] @ img[:8].T
axes[2].imshow(m, cmap="viridis")
axes[2].set_xlabel("image")
axes[2].set_ylabel("caption")
for r in range(8):
    for c in range(8):
        axes[2].text(c, r, f"{m[r, c]:.2f}", ha="center", va="center", fontsize=7, color="white")
axes[2].set_title("caption × image similarity")
save_fig(fig, "day5_space_anatomy.png")

# %% [markdown]
# ## 7 · Retrieval demos: text→image, image→image, image→text, text+image
#
# The **text+image** query is the primitive Week 6 needs for "find documents with a diagram like this one, but about X": embed both, add them (weight α on the text), re-normalize, search the image index.

# %%
def show_hits(query_title: str, hits: list[int], scores: np.ndarray, fname: str, query_img: Image.Image | None = None) -> None:
    n = len(hits) + (query_img is not None)
    fig, axes = plt.subplots(1, n, figsize=(3.2 * n, 3.2))
    axes = list(np.atleast_1d(axes))
    if query_img is not None:
        ax = axes.pop(0)
        ax.imshow(query_img)
        ax.set_title("QUERY", color="red")
        ax.axis("off")
    for ax, h in zip(axes, hits):
        ax.imshow(Image.open(GALLERY_PATHS[h]))
        ax.set_title(f"{scores[h]:.3f}", fontsize=9)
        ax.axis("off")
    fig.suptitle(query_title)
    save_fig(fig, fname)


def norm(v: np.ndarray) -> np.ndarray:
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


for k, q in enumerate(["a dog running through snow", "children playing in a fountain", "a man climbing a rock wall"]):
    s = (MAIN.texts([q]) @ img.T)[0]
    show_hits(f"text → image: '{q}'", list(np.argsort(-s)[:5]), s, f"day5_t2i_{k}.png")

q_idx = 3
s = img @ img[q_idx]
show_hits("image → image (the query itself is excluded)", [h for h in np.argsort(-s) if h != q_idx][:5], s, "day5_i2i.png",
          Image.open(GALLERY_PATHS[q_idx]))
s_txt = txt @ img[q_idx]
print("image → text, top 3 captions:", [CAP_TEXTS[j] for j in np.argsort(-s_txt)[:3]], "| truth:", CAPTIONS[GALLERY[q_idx]][0])

dog = next((i for i, n in enumerate(GALLERY) if re.search(r"\bdog\b", CAPTIONS[n][0]) and "grass" in " ".join(CAPTIONS[n])), 0)
modifier = "in the snow"
for alpha in (0.0, 0.6, 1.2):
    qv = norm(img[dog] + alpha * MAIN.texts([modifier])[0])
    s = img @ qv
    show_hits(f"image + '{modifier}' (α={alpha})", [h for h in np.argsort(-s) if h != dog][:5], s, f"day5_compose_a{alpha}.png",
              Image.open(GALLERY_PATHS[dog]))

# %% [markdown]
# ## 8 · Zero-shot classification on CIFAR-10
#
# No training: the class names become text prompts, and each image is assigned to the nearest prompt. Prompt wording matters ("a photo of a dog" ≫ "dog"), and averaging the text embeddings of many templates (**prompt ensembling**) adds a few points. Your Week 1 CNN needed 50,000 labeled images to pass 70%.

# %%
drive_dir = drive_datasets()
cifar_root = drive_dir if drive_dir is not None and (drive_dir / "cifar-10-batches-py").is_dir() else DATA_DIR / "cifar10"
cifar = torchvision.datasets.CIFAR10(root=str(cifar_root), train=False, download=True)  # skips the download if found
print("CIFAR-10 from", cifar_root)
idx = np.random.default_rng(SEED).choice(len(cifar), N_CIFAR, replace=False)
cifar_imgs = [cifar[i][0] for i in idx]
cifar_y = np.array([cifar[i][1] for i in idx])
TEMPLATES = ["a photo of a {}.", "a blurry photo of a {}.", "a low resolution photo of a {}.", "a photo of the small {}.",
             "a photo of the big {}.", "a pixelated photo of a {}.", "a close-up photo of a {}.", "an image of a {}."]


def zero_shot(emb: "Embedder", img_vecs: np.ndarray, templates: list[str]) -> float:
    cls = np.stack([norm(emb.texts([t.format(c) for t in templates]).mean(0)) for c in cifar.classes])
    return float(np.mean((img_vecs @ cls.T).argmax(1) == cifar_y))


ZS = []
for arch, pre in MODELS:
    emb = MAIN if (arch, pre) == MAIN_MODEL else Embedder(arch, pre)
    vecs = emb.images(cifar_imgs)
    ZS.append({"model": emb.name, "label only '{}'": zero_shot(emb, vecs, ["{}"]),
               "'a photo of a {}.'": zero_shot(emb, vecs, TEMPLATES[:1]), "8-template ensemble": zero_shot(emb, vecs, TEMPLATES)})
    print(ZS[-1])
    if emb is not MAIN:
        del emb
        if DEVICE == "cuda":
            torch.cuda.empty_cache()
ZS_DF = pd.DataFrame(ZS).set_index("model").round(3)
print(ZS_DF.to_string())
ZS_DF.to_csv(OUTPUT_DIR / "day5_zero_shot_cifar10.csv")

# %% [markdown]
# ## 9 · Caption-as-text vs raw embedding vs hybrid (full gallery)
#
# Day 4 compared these on 150 images; here on all 1,000. Every gallery image gets a BLIP caption (fast, batched), then:
#
# - `caption_dense`: query and caption embedded with MiniLM (a text-only sentence encoder)
# - `caption_clip_text`: query and caption both through CLIP's *text* encoder (text↔text, no modality gap)
# - `clip_image`: CLIP text query vs CLIP image vector
# - `hybrid_rrf`: reciprocal-rank fusion of `clip_image` and `caption_dense`
#
# Queries: one human caption per image (the first one).

# %%
caps_csv = OUTPUT_DIR / "day5_blip_captions.csv"
if caps_csv.exists():
    GEN_CAPS = pd.read_csv(caps_csv).set_index("image").caption.reindex(GALLERY).tolist()
else:
    bproc = BlipProcessor.from_pretrained(BLIP_ID)
    fp = torch.float16 if DEVICE == "cuda" else torch.float32
    blip = BlipForConditionalGeneration.from_pretrained(BLIP_ID, dtype=fp).to(DEVICE).eval()
    GEN_CAPS = []
    with torch.inference_mode():
        for i in range(0, len(GALLERY_PATHS), 32):
            batch = [Image.open(p).convert("RGB") for p in GALLERY_PATHS[i:i + 32]]
            ids = blip.generate(**bproc(images=batch, return_tensors="pt").to(DEVICE, fp), max_new_tokens=30, num_beams=3)
            GEN_CAPS += [c.strip() for c in bproc.batch_decode(ids, skip_special_tokens=True)]
    pd.DataFrame({"image": GALLERY, "caption": GEN_CAPS}).to_csv(caps_csv, index=False)
    del blip
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
print("example:", GALLERY[0], "->", GEN_CAPS[0])

text_model = SentenceTransformer(TEXT_EMBEDDER, device=DEVICE)
queries = [CAPTIONS[n][0] for n in GALLERY]
gt = [{i} for i in range(len(GALLERY))]
q_dense = text_model.encode(queries, normalize_embeddings=True, convert_to_numpy=True)
c_dense = text_model.encode(GEN_CAPS, normalize_embeddings=True, convert_to_numpy=True)
q_clip = txt[first_cap]  # CLIP text vectors of the first captions = the queries
SCORES = {"caption_dense": q_dense @ c_dense.T, "caption_clip_text": q_clip @ MAIN.texts(GEN_CAPS).T, "clip_image": q_clip @ img.T}


def rrf(*score_mats: np.ndarray, k: int = 60) -> np.ndarray:
    """Reciprocal-rank fusion: sum of 1/(k + rank) over the input rankings. Robust to different score scales."""
    out = np.zeros_like(score_mats[0])
    for s in score_mats:
        r = np.argsort(np.argsort(-s, axis=1), axis=1) + 1
        out += 1.0 / (k + r)
    return out


SCORES["hybrid_rrf"] = rrf(SCORES["clip_image"], SCORES["caption_dense"])
REPR = pd.DataFrame({name: recall_at(ranks_of(s, gt), "") for name, s in SCORES.items()}).T.round(3)
print(REPR.to_string())
REPR.to_csv(OUTPUT_DIR / "day5_caption_vs_embedding.csv")

# Where do they disagree? Queries the image vector gets right but the caption misses, and vice versa.
r_img, r_cap = ranks_of(SCORES["clip_image"], gt), ranks_of(SCORES["caption_dense"], gt)
for title, sel in (("image vector wins", np.where((r_img == 1) & (r_cap > 10))[0][:3]),
                   ("caption wins", np.where((r_cap == 1) & (r_img > 10))[0][:3])):
    print(f"\n{title}:")
    for i in sel:
        print(f"  query: {queries[i]!r} | BLIP caption: {GEN_CAPS[i]!r}")

# %% [markdown]
# ## 10 · A reusable multimodal index (FAISS) for Week 6
#
# `IndexFlatIP` on unit vectors = exact cosine search. 1,000–100,000 items need nothing more; beyond that, swap in `IndexHNSWFlat` or IVF-PQ. Every item carries metadata (id, path, caption, source), so a hit can be traced back — the provenance Week 6 needs.

# %%
class MultimodalIndex:
    """Images (+ optional captions) in one CLIP space. Search by text, image, or text+image."""

    def __init__(self, embedder: "Embedder"):
        self.emb = embedder
        self.index = None
        self.meta: list[dict] = []

    def add_images(self, images: list, metas: list[dict]) -> None:
        vecs = self.emb.images(images)
        if self.index is None:
            self.index = faiss.IndexFlatIP(vecs.shape[1])
        self.index.add(vecs)
        self.meta += metas

    def _search(self, qv: np.ndarray, k: int) -> list[dict]:
        scores, ids = self.index.search(norm(qv.reshape(1, -1)).astype(np.float32), k)
        return [{**self.meta[i], "score": float(s)} for s, i in zip(scores[0], ids[0]) if i >= 0]

    def search_text(self, query: str, k: int = 5) -> list[dict]:
        return self._search(self.emb.texts([query])[0], k)

    def search_image(self, image, k: int = 5) -> list[dict]:
        return self._search(self.emb.images([image])[0], k)

    def search_text_image(self, image, text: str, alpha: float = 0.6, k: int = 5) -> list[dict]:
        return self._search(self.emb.images([image])[0] + alpha * self.emb.texts([text])[0], k)

    def save(self, folder: Path) -> None:
        folder.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(folder / "images.faiss"))
        (folder / "meta.json").write_text(json.dumps({"model": self.emb.name, "items": self.meta}, indent=1), encoding="utf-8")

    @classmethod
    def load(cls, folder: Path, embedder: "Embedder") -> "MultimodalIndex":
        info = json.loads((folder / "meta.json").read_text(encoding="utf-8"))
        assert info["model"] == embedder.name, f"index built with {info['model']}, not {embedder.name}"
        obj = cls(embedder)
        obj.index, obj.meta = faiss.read_index(str(folder / "images.faiss")), info["items"]
        return obj


INDEX = MultimodalIndex(MAIN)
INDEX.add_images(GALLERY_PATHS, [{"id": n, "path": str(p), "caption": c, "source": "flickr8k"} for n, p, c in zip(GALLERY, GALLERY_PATHS, GEN_CAPS)])
INDEX.save(OUTPUT_DIR / "index")
INDEX2 = MultimodalIndex.load(OUTPUT_DIR / "index", MAIN)
t0 = time.perf_counter()
hits = INDEX2.search_text("two dogs playing on the beach", k=3)
print(f"query latency {1000 * (time.perf_counter() - t0):.1f} ms | index size {INDEX2.index.ntotal}")
for h in hits:
    print(f"  {h['score']:.3f}  {h['id']}  ({h['caption']})")

# %% [markdown]
# ## 11 · Success gate: caption + embed an *arbitrary* image, then find it with a text query
#
# Take an image the index has never seen (a Flickr *training* image and a chart rendered right now), caption it, add it to the index with its caption as metadata, and retrieve it by describing it in words.

# %%
fig, ax = plt.subplots(figsize=(5, 4))
ax.bar(["2022", "2023", "2024", "2025"], [12, 18, 25, 31], color="tab:green")
ax.set_title("Solar installations per year")
fig.canvas.draw()
chart = Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[..., :3])
plt.close(fig)
new_name = TRAIN_NAMES[0]
new_imgs = [Image.open(IMAGE_DIR / new_name).convert("RGB"), chart]
bproc = BlipProcessor.from_pretrained(BLIP_ID)
fp = torch.float16 if DEVICE == "cuda" else torch.float32
blip = BlipForConditionalGeneration.from_pretrained(BLIP_ID, dtype=fp).to(DEVICE).eval()
with torch.inference_mode():
    new_caps = [c.strip() for c in bproc.batch_decode(blip.generate(**bproc(images=new_imgs, return_tensors="pt").to(DEVICE, fp),
                                                                    max_new_tokens=30), skip_special_tokens=True)]
INDEX2.add_images(new_imgs, [{"id": new_name, "caption": new_caps[0], "source": "new photo"},
                             {"id": "solar_chart", "caption": new_caps[1], "source": "new chart"}])
GATE = []
for target, q in ((new_name, CAPTIONS[new_name][1]), ("solar_chart", "bar chart of solar installations per year")):
    hits = INDEX2.search_text(q, k=5)
    rank = next((r for r, h in enumerate(hits, 1) if h["id"] == target), None)
    GATE.append({"target": target, "query": q, "caption": next(m["caption"] for m in INDEX2.meta if m["id"] == target), "rank_in_top5": rank})
GATE_DF = pd.DataFrame(GATE)
print(GATE_DF.to_string(index=False))
print("GATE PASSED" if GATE_DF.rank_in_top5.notna().all() else "GATE: at least one target not in the top 5 — inspect the caption/query")

# %%
(OUTPUT_DIR / "day5_summary.json").write_text(json.dumps({
    "gallery": len(GALLERY), "benchmark": BENCH_DF.to_dict("index"), "zero_shot_cifar10": ZS_DF.to_dict("index"),
    "caption_vs_embedding": REPR.to_dict("index"), "modality_gap": float(gap), "gate": GATE,
}, indent=2, default=float), encoding="utf-8")
print("saved ->", OUTPUT_DIR / "day5_summary.json", "| index ->", OUTPUT_DIR / "index")

# %% [markdown]
# ## Checklist
#
# - [ ] Recall@K for text→image and image→text for several models, with speed and storage per vector.
# - [ ] I can explain the modality gap and why absolute similarity thresholds must be tuned per model.
# - [ ] Text+image composed queries work, and I know what α does.
# - [ ] Zero-shot CIFAR-10 accuracy with and without prompt ensembling, next to my Week 1 CNN.
# - [ ] Caption vs embedding vs hybrid numbers on 1,000 images, plus examples where each wins.
# - [ ] `MultimodalIndex` saved to disk, reloads, and finds a brand-new image from a text description (the Week 6 primitive).
