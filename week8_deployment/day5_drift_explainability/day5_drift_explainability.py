"""
Week 8 · Day 5 — Drift experiment (clean vs corrupted / class-imbalanced streams; KS, PSI, embedding and label-shift detectors,
threshold tuned on both streams, Evidently report) and Grad-CAM explainability (+ the API's /drift and /gradcam endpoints).

Colab-first script (percent-format cells). Build the notebook with:
    python ../tools/inline_shared.py && python ../tools/py_to_ipynb.py
Dataset: Intel Image Classification (My Drive/datasets/intel-image-classification/ or Kaggle).
"""

# %% [markdown]
# # Week 8 · Day 5 — Drift Detection & Explainability
#
# **Goal:** prove that monitoring works. Build a reference distribution from clean test data, then feed the model **deliberately shifted** streams — same classes but corrupted (blur, brightness, contrast, JPEG, noise; CIFAR-10-C / ImageNet-C style) or class-imbalanced — and show that the monitor **fires on drift and stays quiet on clean data**. Then explain individual predictions with **Grad-CAM**.
#
# | Topic | What you will build / measure |
# |---|---|
# | Reference | Input statistics (brightness, contrast, sharpness, colourfulness), confidence, entropy, penultimate-layer embeddings, predicted-class mix |
# | Streams | Clean windows; corruption windows at severities 1–5; a class-imbalanced window (label shift) |
# | Detectors | KS tests on input stats and embeddings (PCA), PSI on confidence, χ² on predicted classes |
# | Thresholds | Tuned on bootstrap clean windows (false-alarm rate) — then checked on drift windows (detection rate) |
# | The chart | One timeline: clean windows below the threshold, injected drift above it |
# | Evidently | A drift report (HTML) on the same features |
# | Grad-CAM | Heatmaps for correct and wrong predictions, and for corrupted inputs; the API's `/gradcam` and `/drift` endpoints |
#
# > Runtime: **T4 GPU**, ~15 min.

# %% [markdown]
# ## 0 · Install dependencies

# %%
import subprocess
import sys


def pip_install(*packages: str) -> None:
    """Install packages into the running interpreter (works in Colab and plain Python)."""
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *packages])


pip_install("kagglehub>=0.3", "grad-cam>=1.5", "evidently", "scipy", "onnxruntime-gpu>=1.19", "fastapi>=0.110",
            "uvicorn[standard]>=0.29", "python-multipart", "requests")

# %% [markdown]
# ## 1 · Imports, configuration, and output folder

# %%
import io
import json
import os
import random
import shutil
import time
from pathlib import Path

import matplotlib

IN_NOTEBOOK = "ipykernel" in sys.modules
if not IN_NOTEBOOK:
    matplotlib.use("Agg")  # headless when run as a script

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import torch
from PIL import ImageEnhance, ImageFilter
from scipy.stats import chisquare, ks_2samp

IN_COLAB = "google.colab" in sys.modules
try:
    HERE = Path(__file__).resolve().parent
except NameError:  # running inside a notebook
    HERE = Path.cwd()

OUTPUT_DIR = Path("/content/week8_outputs/day5") if IN_COLAB else HERE.parent / "results" / "day5"
DATA_DIR = Path("/content/datasets") if IN_COLAB else HERE.parent / "data"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)

FULL_RUN = True
TEST_PER_CLASS = 350 if FULL_RUN else 120  # half for the reference, half for the streams
WINDOW = 150 if FULL_RUN else 60
N_BOOTSTRAP = 40  # clean windows resampled to estimate the false-alarm rate
TARGET_FALSE_ALARM = 0.05
CORRUPTIONS = ["gaussian_blur", "brightness", "contrast", "jpeg", "gaussian_noise"]
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"PyTorch {torch.__version__} | device={DEVICE} | window={WINDOW}")


def save_fig(fig, name: str) -> None:
    """Save a figure to OUTPUT_DIR and show it inline when in a notebook."""
    path = OUTPUT_DIR / name
    fig.savefig(path, dpi=120, bbox_inches="tight")
    if IN_NOTEBOOK:
        plt.show()
    plt.close(fig)
    print(f"saved -> {path}")


# %% [markdown]
# ## 2 · Helper code (shared by Week 8; canonical copies in `tools/shared/`)

# %%
# ==== shared: storage ====
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
# ==== end shared: storage ====

# %%
# ==== shared: classifier ====
# ---- classifier.py: shared helper code (canonical copy in week8_deployment/tools/shared/) ----
# The Week 2 classifier, reused all of Week 8: Intel Image Classification (6 scenes), ResNet-18 with a
# Dropout + Linear head, ImageNet normalization, 224 px centre crop.
#
#   data       My Drive/datasets/intel-image-classification/ (your folder) -> intel-image-classification.zip on Drive
#              -> kagglehub (puneet6060/intel-image-classification). A fixed per-class subset is copied to local disk.
#   checkpoint Week 2's pretrained_aug.pt if you put it on Drive, else a 3-epoch fine-tune here (saved for later days).
import hashlib
import io
import itertools
import os
import random
import shutil
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import ResNet18_Weights, resnet18

CLASSES = ["buildings", "forest", "glacier", "mountain", "sea", "street"]  # ImageFolder order (alphabetical), as in Week 2
IMAGENET_MEAN, IMAGENET_STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
EVAL_TF = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(),
                              transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)])
TRAIN_TF = transforms.Compose([transforms.RandomResizedCrop(224, scale=(0.7, 1.0)), transforms.RandomHorizontalFlip(),
                               transforms.ColorJitter(0.2, 0.2, 0.2), transforms.ToTensor(), transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)])


def intel_root(data_dir: Path) -> Path:
    """Folder containing seg_train/ and seg_test/: Drive folder, Drive zip, local copy or kagglehub download."""
    drive = mount_drive()
    candidates = []
    if drive is not None:
        candidates += [drive / "datasets" / "intel-image-classification"]
        z = drive / "datasets" / "intel-image-classification.zip"
        if z.exists() and not (data_dir / "intel").exists():
            print(f"unzipping {z} ...")
            with zipfile.ZipFile(z) as zf:
                zf.extractall(data_dir / "intel")
    candidates.append(data_dir / "intel")
    for c in candidates:
        if c.exists() and ((c / "seg_test").exists() or any(c.glob("*/seg_test"))):
            return c
    import kagglehub
    for key in ("KAGGLE_API_TOKEN", "KAGGLE_USERNAME", "KAGGLE_KEY"):
        val = get_secret(key)
        if val:
            os.environ[key] = val
    return Path(kagglehub.dataset_download("puneet6060/intel-image-classification"))


def locate_split(root: Path, split: str) -> Path:
    """seg_train / seg_test folder holding the six class folders (Kaggle zips nest it twice)."""
    for cand in itertools.chain([root / split / split, root / split], root.rglob(split)):  # rglob only if needed
        if cand.is_dir() and set(CLASSES).issubset({p.name for p in cand.iterdir() if p.is_dir()}):
            return cand
    raise FileNotFoundError(f"no {split}/ with the six class folders under {root}")


def sample_split(split_dir: Path, per_class: int, seed: int = 42) -> list[tuple[Path, int]]:
    rng = random.Random(seed)
    out = []
    for label, cls in enumerate(CLASSES):
        files = sorted((split_dir / cls).glob("*.jpg"))
        out += [(p, label) for p in rng.sample(files, min(per_class, len(files)))]
    return out


def localize(samples: list, dest: Path, workers: int = 32) -> list[tuple[Path, int]]:
    """Copy files to local disk in parallel (reading thousands of small files from Drive is slow; local disk is not)."""
    dest.mkdir(parents=True, exist_ok=True)

    def copy(item):
        src, label = item
        dst = dest / CLASSES[label] / src.name
        if not dst.exists():
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(src, dst)
        return dst, label
    with ThreadPoolExecutor(workers) as ex:
        return list(ex.map(copy, samples))


def load_intel(data_dir: Path, train_per_class: int, test_per_class: int, seed: int = 42) -> tuple[list, list]:
    t0 = time.perf_counter()
    root = intel_root(data_dir)
    train = localize(sample_split(locate_split(root, "seg_train"), train_per_class, seed), data_dir / "local" / "train")
    test = localize(sample_split(locate_split(root, "seg_test"), test_per_class, seed), data_dir / "local" / "test")
    print(f"Intel data from {root}: train {len(train)} / test {len(test)} images ({time.perf_counter() - t0:.0f}s)")
    return train, test


def dataset_hash(samples: list) -> str:
    """Content hash of a sample list (relative names + file bytes): the data version a result was produced on."""
    h = hashlib.sha256()
    for p, label in sorted(samples, key=lambda x: (x[1], Path(x[0]).name)):
        h.update(f"{label}/{Path(p).name}".encode())
        h.update(Path(p).read_bytes())
    return h.hexdigest()[:16]


class ImageList(Dataset):
    def __init__(self, samples: list, tf=EVAL_TF, corrupt=None):
        self.samples, self.tf, self.corrupt = samples, tf, corrupt

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, i):
        p, label = self.samples[i]
        img = Image.open(p).convert("RGB")
        if self.corrupt is not None:
            img = self.corrupt(img)
        return self.tf(img), label


def make_loader(samples: list, train: bool = False, bs: int = 64, corrupt=None, seed: int = 42, augment: bool | None = None) -> DataLoader:
    """train=True shuffles (and augments unless augment=False); seeded so the batch order is reproducible."""
    g = torch.Generator().manual_seed(seed)
    tf = TRAIN_TF if (train if augment is None else augment) else EVAL_TF
    return DataLoader(ImageList(samples, tf, corrupt), batch_size=bs, shuffle=train, generator=g,
                      num_workers=2, pin_memory=torch.cuda.is_available())


def build_model(num_classes: int = len(CLASSES), pretrained: bool = True) -> nn.Module:
    """Week 2 architecture: ResNet-18, fc = Dropout(0.3) + Linear(512, 6)."""
    model = resnet18(weights=ResNet18_Weights.DEFAULT if pretrained else None)
    model.fc = nn.Sequential(nn.Dropout(p=0.3), nn.Linear(model.fc.in_features, num_classes))
    return model


def run_epoch(model, loader, device, optimizer=None, loss_fn=None) -> dict:
    train = optimizer is not None
    model.train(train)
    loss_fn = loss_fn or nn.CrossEntropyLoss()
    total, correct, loss_sum = 0, 0, 0.0
    with torch.set_grad_enabled(train):
        for x, y in loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            out = model(x)
            loss = loss_fn(out, y)
            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
            loss_sum += float(loss) * len(y)
            correct += int((out.argmax(1) == y).sum())
            total += len(y)
    return {"loss": loss_sum / max(1, total), "accuracy": correct / max(1, total)}


def fine_tune(model, train_loader, val_loader, device, epochs: int = 3, head_lr: float = 1e-3, backbone_lr: float = 1e-4,
              log=None) -> nn.Module:
    """Week 2 recipe: differential learning rates (small for the pretrained backbone), cosine schedule."""
    model.to(device)
    head = list(model.fc.parameters())
    head_ids = {id(p) for p in head}
    opt = torch.optim.AdamW([{"params": [p for p in model.parameters() if id(p) not in head_ids], "lr": backbone_lr},
                             {"params": head, "lr": head_lr}], weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    for ep in range(1, epochs + 1):
        tr = run_epoch(model, train_loader, device, opt)
        va = run_epoch(model, val_loader, device)
        sched.step()
        print(f"  epoch {ep}/{epochs}: train acc {tr['accuracy']:.3f} | val acc {va['accuracy']:.3f}")
        if log:
            log(ep, tr, va)
    return model


def find_week2_checkpoint() -> Path | None:
    drive = mount_drive()
    cands = [Path("/content/pretrained_aug.pt")]
    if drive is not None:
        cands += [drive / "checkpoints" / "week2" / "pretrained_aug.pt", drive / "cv_course_artifacts" / "week8" / "resnet18_week2.pt"]
    cands += [HERE.parent.parent / "week2_classification" / "day5_ablation_evaluation" / "outputs" / "models" / "pretrained_aug.pt",
              HERE.parent.parent / "cv_course_artifacts" / "week8" / "resnet18_week2.pt"]
    return next((c for c in cands if c.exists()), None)


def load_or_train_classifier(train: list, test: list, device, epochs: int = 3) -> tuple[nn.Module, str]:
    """The Week 2 fine-tuned model. Trains (and saves to the week8 artifact folder) only if no checkpoint is found."""
    model = build_model()
    ckpt = find_week2_checkpoint()
    if ckpt is not None:
        model.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=True))
        return model.to(device).eval(), str(ckpt)
    print("no Week 2 checkpoint found: fine-tuning ResNet-18 with the Week 2 recipe")
    fine_tune(model, make_loader(train, train=True), make_loader(test), device, epochs=epochs)
    out = artifact_dir("week8") / "resnet18_week2.pt"
    torch.save(model.state_dict(), out)
    return model.eval(), f"trained here -> {out}"


@torch.inference_mode()
def predict(model, loader, device) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(labels, predictions, softmax probabilities) over a loader."""
    model.eval()
    ys, ps = [], []
    for x, y in loader:
        ps.append(torch.softmax(model(x.to(device)).float(), 1).cpu())
        ys.append(y)
    probs = torch.cat(ps).numpy()
    return torch.cat(ys).numpy(), probs.argmax(1), probs


def accuracy_report(y: np.ndarray, pred: np.ndarray) -> dict:
    per_class = {c: float((pred[y == i] == i).mean()) for i, c in enumerate(CLASSES) if (y == i).any()}
    return {"accuracy": float((pred == y).mean()), "worst_class": min(per_class, key=per_class.get),
            "worst_class_acc": min(per_class.values()), **{f"acc_{c}": v for c, v in per_class.items()}}


def model_size_mb(model) -> float:
    buf = io.BytesIO()
    torch.save(model.state_dict(), buf)
    return buf.getbuffer().nbytes / 1e6


def latency_ms(fn, x, warmup: int = 5, iters: int = 30) -> float:
    """Median wall-clock ms per call of fn(x) (CUDA synchronized)."""
    for _ in range(warmup):
        fn(x)
    times = []
    for _ in range(iters):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn(x)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000)
    return float(np.median(times))
# ==== end shared: classifier ====

# %% [markdown]
# ## 3 · Data, model and the corruption functions
#
# Corruptions follow the ImageNet-C families at five severities, implemented with Pillow / NumPy (the `imagecorruptions` package does the same with more types, but pins old scikit-image versions).

# %%
TRAIN, TEST = load_intel(DATA_DIR, 100, TEST_PER_CLASS, SEED)
MODEL, SOURCE = load_or_train_classifier(TRAIN, TEST, DEVICE)
MODEL = MODEL.to(DEVICE).eval()
rng = random.Random(SEED)
shuffled = TEST[:]
rng.shuffle(shuffled)
REF_SET, STREAM_SET = shuffled[: len(shuffled) // 2], shuffled[len(shuffled) // 2:]
print(f"model: {SOURCE} | reference {len(REF_SET)} images | stream pool {len(STREAM_SET)} images")

SEVERITY = {"gaussian_blur": [1, 2, 3, 4, 6], "brightness": [1.3, 1.6, 1.9, 2.2, 2.6], "contrast": [0.75, 0.55, 0.4, 0.3, 0.2],
            "jpeg": [30, 18, 12, 8, 5], "gaussian_noise": [8, 16, 24, 36, 50]}


def corruption(kind: str, severity: int):
    v = SEVERITY[kind][severity - 1]

    def f(img: Image.Image) -> Image.Image:
        if kind == "gaussian_blur":
            return img.filter(ImageFilter.GaussianBlur(v))
        if kind == "brightness":
            return ImageEnhance.Brightness(img).enhance(v)
        if kind == "contrast":
            return ImageEnhance.Contrast(img).enhance(v)
        if kind == "jpeg":
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=v)
            return Image.open(io.BytesIO(buf.getvalue())).convert("RGB")
        arr = np.asarray(img).astype(np.float32) + np.random.default_rng(severity).normal(0, v, (img.height, img.width, 3))
        return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
    return f


sample = Image.open(STREAM_SET[0][0]).convert("RGB")
fig, axes = plt.subplots(len(CORRUPTIONS), 5, figsize=(15, 3 * len(CORRUPTIONS)))
for i, kind in enumerate(CORRUPTIONS):
    for s in range(1, 6):
        axes[i, s - 1].imshow(corruption(kind, s)(sample))
        axes[i, s - 1].set_title(f"{kind} s{s}", fontsize=8)
        axes[i, s - 1].axis("off")
save_fig(fig, "day5_corruptions.png")

# %% [markdown]
# ## 4 · What the monitor measures
#
# For every image: **input statistics** (computed without the model — they catch camera/pipeline changes), the model's **confidence** and **entropy**, its **predicted class**, and the 512-d **embedding** before the classifier head (captured with a forward hook).

# %%
EMB = {}
MODEL.fc.register_forward_hook(lambda m, inp, out: EMB.__setitem__("x", inp[0].detach()))


def image_stats(img: Image.Image) -> dict:
    gray = np.asarray(img.convert("L"), dtype=np.float32)
    rgb = np.asarray(img, dtype=np.float32)
    rg, yb = rgb[..., 0] - rgb[..., 1], 0.5 * (rgb[..., 0] + rgb[..., 1]) - rgb[..., 2]
    return {"brightness": gray.mean() / 255.0, "contrast": gray.std() / 255.0,
            "sharpness": float(cv2.Laplacian(gray, cv2.CV_32F).var()),
            "colorfulness": float(np.hypot(rg.std(), yb.std()) + 0.3 * np.hypot(rg.mean(), yb.mean()))}


@torch.inference_mode()
def observe(samples: list, corrupt=None, bs: int = 64) -> tuple[pd.DataFrame, np.ndarray]:
    rows, embs = [], []
    for i in range(0, len(samples), bs):
        imgs, labels = [], []
        for p, y in samples[i:i + bs]:
            img = Image.open(p).convert("RGB")
            imgs.append(corrupt(img) if corrupt else img)
            labels.append(y)
        x = torch.stack([EVAL_TF(im) for im in imgs]).to(DEVICE)
        probs = torch.softmax(MODEL(x), 1).cpu().numpy()
        embs.append(EMB["x"].cpu().numpy())
        for im, y, pr in zip(imgs, labels, probs):
            rows.append({**image_stats(im), "confidence": float(pr.max()), "entropy": float(-(pr * np.log(pr + 1e-12)).sum()),
                         "pred": int(pr.argmax()), "label": y})
    return pd.DataFrame(rows), np.concatenate(embs)


REF, REF_EMB = observe(REF_SET)
emb_mean = REF_EMB.mean(0)
_, _, VT = np.linalg.svd(REF_EMB - emb_mean, full_matrices=False)
PCA = VT[:5]  # top-5 principal directions of the reference embeddings
REF_PCA = (REF_EMB - emb_mean) @ PCA.T
print(f"reference accuracy {(REF.pred == REF.label).mean():.3f} | mean confidence {REF.confidence.mean():.3f}")
print(REF.describe().loc[["mean", "std"]].round(3).to_string())

# %% [markdown]
# ## 5 · Detectors
#
# | Detector | Statistic | Catches |
# |---|---|---|
# | Input KS | two-sample Kolmogorov–Smirnov p-value per input statistic (Bonferroni over 4) | brightness/contrast/blur/noise changes, before any model is involved |
# | Confidence PSI | Population Stability Index over reference-decile bins | the model becoming unsure |
# | Embedding KS | KS on the top-5 PCA components of the embeddings (Bonferroni over 5) | changes in what the network "sees" |
# | Label shift | χ² test of predicted-class counts vs reference proportions | class mix changes (no corruption at all) |
#
# PSI = Σ (cur − ref) · ln(cur / ref) over bins; rule of thumb 0.1 = moderate, 0.25 = large shift. Here the thresholds are **tuned** (section 6).

# %%
STATS = ["brightness", "contrast", "sharpness", "colorfulness"]
EDGES = np.quantile(REF.confidence, np.linspace(0, 1, 11))
EDGES[0], EDGES[-1] = 0.0, 1.0
REF_FRAC = np.histogram(REF.confidence, bins=EDGES)[0] / len(REF)
REF_CLASS = np.bincount(REF.pred, minlength=len(CLASSES)) / len(REF)


def psi(values: np.ndarray) -> float:
    cur = np.histogram(values, bins=EDGES)[0] / len(values)
    return float(np.sum((cur - REF_FRAC) * np.log((cur + 1e-4) / (REF_FRAC + 1e-4))))


def detect(win: pd.DataFrame, emb: np.ndarray) -> dict:
    p_input = min(ks_2samp(win[s], REF[s]).pvalue for s in STATS) * len(STATS)
    pca = (emb - emb_mean) @ PCA.T
    p_emb = min(ks_2samp(pca[:, j], REF_PCA[:, j]).pvalue for j in range(PCA.shape[0])) * PCA.shape[0]
    counts = np.bincount(win.pred, minlength=len(CLASSES))
    p_label = chisquare(counts, REF_CLASS * counts.sum()).pvalue
    return {"psi_confidence": psi(win.confidence.values), "p_input": min(1.0, p_input), "p_embedding": min(1.0, p_emb),
            "p_label": float(p_label), "accuracy": float((win.pred == win.label).mean()), "mean_confidence": float(win.confidence.mean())}


# %% [markdown]
# ## 6 · Tune thresholds on clean data first
#
# A monitor that always fires is as useless as one that never fires. Resample many **clean** windows from the stream pool (bootstrap), compute every statistic, and set each threshold so the clean false-alarm rate is ≈ 5 % overall. Only then look at drift.

# %%
CLEAN_ALL, CLEAN_EMB = observe(STREAM_SET)
boot = []
r = np.random.default_rng(0)
for _ in range(N_BOOTSTRAP):
    idx = r.choice(len(CLEAN_ALL), WINDOW, replace=False)
    boot.append(detect(CLEAN_ALL.iloc[idx].reset_index(drop=True), CLEAN_EMB[idx]))
BOOT = pd.DataFrame(boot)
q = 1 - TARGET_FALSE_ALARM / 4  # four detectors share the false-alarm budget
THRESH = {"psi_confidence": float(max(0.1, BOOT.psi_confidence.quantile(q))),
          "p_input": float(min(0.01, BOOT.p_input.quantile(1 - q))),
          "p_embedding": float(min(0.01, BOOT.p_embedding.quantile(1 - q))),
          "p_label": float(min(0.01, BOOT.p_label.quantile(1 - q)))}
print("thresholds:", {k: round(v, 5) for k, v in THRESH.items()})


def alerts(d: dict) -> list[str]:
    fired = []
    if d["psi_confidence"] > THRESH["psi_confidence"]:
        fired.append("confidence PSI")
    if d["p_input"] < THRESH["p_input"]:
        fired.append("input KS")
    if d["p_embedding"] < THRESH["p_embedding"]:
        fired.append("embedding KS")
    if d["p_label"] < THRESH["p_label"]:
        fired.append("label shift")
    return fired


fa = np.mean([bool(alerts(d)) for d in boot])
print(f"false-alarm rate on {N_BOOTSTRAP} clean bootstrap windows: {fa:.3f}")

# %% [markdown]
# ## 7 · The stream: clean windows, then injected drift

# %%
windows = []
pool = STREAM_SET[:]
random.Random(1).shuffle(pool)
for i in range(4):  # clean windows (fresh, not bootstrap)
    chunk = pool[(i * WINDOW) % len(pool):][:WINDOW]
    windows.append(("clean", 0, chunk, None))
for s in range(1, 6):
    windows.append(("gaussian_blur", s, random.Random(10 + s).sample(STREAM_SET, WINDOW), corruption("gaussian_blur", s)))
for j, kind in enumerate(CORRUPTIONS[1:]):
    windows.append((kind, 3, random.Random(20 + j).sample(STREAM_SET, WINDOW), corruption(kind, 3)))
imbalanced = [x for x in STREAM_SET if CLASSES[x[1]] in ("sea", "glacier")]
windows.append(("class_imbalance (sea+glacier)", 0, random.Random(5).sample(imbalanced, min(WINDOW, len(imbalanced))), None))
windows.append(("clean", 0, random.Random(99).sample(STREAM_SET, WINDOW), None))

rows = []
for i, (kind, sev, chunk, fn) in enumerate(windows):
    win, emb = observe(chunk, fn)
    d = detect(win, emb)
    fired = alerts(d)
    rows.append({"window": i, "stream": kind, "severity": sev, "drifted": kind != "clean", "alert": bool(fired),
                 "fired": ", ".join(fired), **d})
STREAM = pd.DataFrame(rows)
print(STREAM[["window", "stream", "severity", "alert", "fired", "psi_confidence", "p_input", "p_embedding", "p_label", "accuracy",
              "mean_confidence"]].round(4).to_string(index=False))
tp = (STREAM.alert & STREAM.drifted).sum()
fp = (STREAM.alert & ~STREAM.drifted).sum()
print(f"\ndrift windows detected: {tp}/{STREAM.drifted.sum()} | clean windows that alerted: {fp}/{(~STREAM.drifted).sum()}")
STREAM.to_csv(OUTPUT_DIR / "day5_drift_stream.csv", index=False)

# %% [markdown]
# ### The deliverable chart
#
# Top: confidence PSI per window with its threshold. Bottom: the strongest KS evidence per window as −log10(p) (higher = more surprising) with the p-value threshold. Shaded = injected drift. Accuracy (dots, right axis) shows *why* it matters: drift that the monitor catches is drift that costs accuracy.

# %%
fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
x = STREAM.window.values
for ax in axes:
    for i, drifted in zip(x, STREAM.drifted):
        if drifted:
            ax.axvspan(i - 0.5, i + 0.5, color="tab:red", alpha=0.08)
axes[0].bar(x, STREAM.psi_confidence, color=["tab:red" if a else "tab:green" for a in STREAM.alert])
axes[0].axhline(THRESH["psi_confidence"], ls="--", color="k", label=f"PSI threshold {THRESH['psi_confidence']:.2f}")
axes[0].set_ylabel("confidence PSI")
axes[0].legend(loc="upper left")
ax2 = axes[0].twinx()
ax2.plot(x, STREAM.accuracy, "o-", color="tab:blue", label="accuracy")
ax2.set_ylabel("accuracy", color="tab:blue")
ax2.set_ylim(0, 1)
evidence = -np.log10(np.clip(STREAM[["p_input", "p_embedding", "p_label"]].min(axis=1), 1e-300, 1))
axes[1].bar(x, evidence, color=["tab:red" if a else "tab:green" for a in STREAM.alert])
axes[1].axhline(-np.log10(min(THRESH["p_input"], THRESH["p_embedding"], THRESH["p_label"])), ls="--", color="k", label="p threshold")
axes[1].set_ylabel("-log10(min KS / chi2 p)")
axes[1].set_xticks(x, [f"{s}\ns{v}" if v else s for s, v in zip(STREAM.stream, STREAM.severity)], rotation=60, fontsize=8)
axes[1].legend(loc="upper left")
fig.suptitle("Clean windows stay below the thresholds; injected drift goes above (red = alert)")
save_fig(fig, "day5_drift_timeline.png")

# %% [markdown]
# ### Accuracy under corruption, per class
#
# The same severity hurts classes differently (blur destroys the texture that separates forest from mountain; brightness washes out glacier vs sea).

# %%
per_class = {}
for kind in CORRUPTIONS:
    win, _ = observe(STREAM_SET, corruption(kind, 3))
    per_class[kind] = {c: float((win.pred[win.label == i] == i).mean()) for i, c in enumerate(CLASSES)}
per_class["clean"] = {c: float((CLEAN_ALL.pred[CLEAN_ALL.label == i] == i).mean()) for i, c in enumerate(CLASSES)}
PC = pd.DataFrame(per_class).T.round(3)
print(PC.to_string())

# %% [markdown]
# ## 8 · Evidently report
#
# [Evidently](https://docs.evidentlyai.com/) runs a per-column drift test suite and renders an HTML dashboard — the same idea as section 5 packaged for monitoring jobs. The API changed in Evidently 0.7; both versions are tried.

# %%
cur, _ = observe(random.Random(3).sample(STREAM_SET, WINDOW), corruption("gaussian_blur", 3))
cols = STATS + ["confidence", "entropy"]
html = OUTPUT_DIR / "day5_evidently_report.html"
try:
    try:
        from evidently import Report
        from evidently.presets import DataDriftPreset
        result = Report([DataDriftPreset()]).run(current_data=cur[cols], reference_data=REF[cols])
        result.save_html(str(html))
    except ImportError:
        from evidently.metric_preset import DataDriftPreset
        from evidently.report import Report
        rep = Report(metrics=[DataDriftPreset()])
        rep.run(reference_data=REF[cols], current_data=cur[cols])
        rep.save_html(str(html))
    print("Evidently report ->", html)
except Exception as exc:
    print("Evidently report skipped:", repr(exc)[:200])

# %% [markdown]
# ## 9 · Grad-CAM: which pixels drove the prediction?
#
# Grad-CAM weights the last convolutional block's feature maps by the gradient of the predicted class score and keeps the positive part — a coarse heatmap of the evidence. Correct predictions should light up the scene content (trees, peaks, water); wrong ones often reveal a shortcut.

# %%
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

MEAN_T, STD_T = torch.tensor(IMAGENET_MEAN).view(3, 1, 1), torch.tensor(IMAGENET_STD).view(3, 1, 1)


def gradcam(img: Image.Image) -> tuple[np.ndarray, int, float]:
    x = EVAL_TF(img)[None].to(DEVICE)
    with torch.no_grad():
        probs = torch.softmax(MODEL(x), 1)[0]
    cls = int(probs.argmax())
    with GradCAM(model=MODEL, target_layers=[MODEL.layer4[-1]]) as cam:
        mask = cam(input_tensor=x, targets=[ClassifierOutputTarget(cls)])[0]
    rgb = (x[0].cpu() * STD_T + MEAN_T).permute(1, 2, 0).numpy().clip(0, 1).astype(np.float32)
    return show_cam_on_image(rgb, mask, use_rgb=True), cls, float(probs[cls])


right = CLEAN_ALL.index[CLEAN_ALL.pred == CLEAN_ALL.label][:4].tolist()
wrong = CLEAN_ALL.index[CLEAN_ALL.pred != CLEAN_ALL.label][:4].tolist()
picks = [(STREAM_SET[i], "correct") for i in right] + [(STREAM_SET[i], "wrong") for i in wrong]
fig, axes = plt.subplots(2, len(picks), figsize=(3 * len(picks), 6.5))
for j, ((p, y), tag) in enumerate(picks):
    img = Image.open(p).convert("RGB")
    overlay, cls, conf = gradcam(img)
    axes[0, j].imshow(img.resize((224, 224)))
    axes[0, j].set_title(f"{tag}: true {CLASSES[y]}", fontsize=8, color="green" if tag == "correct" else "red")
    axes[1, j].imshow(overlay)
    axes[1, j].set_title(f"pred {CLASSES[cls]} ({conf:.2f})", fontsize=8)
for ax in axes.flat:
    ax.axis("off")
save_fig(fig, "day5_gradcam.png")

img = Image.open(picks[0][0][0]).convert("RGB")
fig, axes = plt.subplots(1, 4, figsize=(13, 3.6))
for ax, (name, fn) in zip(axes, [("clean", None), ("blur s4", corruption("gaussian_blur", 4)), ("brightness s4", corruption("brightness", 4)),
                                 ("noise s4", corruption("gaussian_noise", 4))]):
    overlay, cls, conf = gradcam(fn(img) if fn else img)
    ax.imshow(overlay)
    ax.set_title(f"{name}: {CLASSES[cls]} ({conf:.2f})", fontsize=9)
    ax.axis("off")
save_fig(fig, "day5_gradcam_under_drift.png")

# %% [markdown]
# ## 10 · Hand the reference to the API, and watch `/drift` fire
#
# `reference.json` is what Day 3's `DriftMonitor` compares with: a confidence sample with its decile bins, a brightness sample and the tuned thresholds. If Day 3's service folder is on Drive, start it, send clean images, then blurred ones, and read `/drift` after each batch. `/gradcam` returns the heatmap over HTTP.

# %%
ART = artifact_dir("week8")
reference = {"confidence_sample": REF.confidence.round(5).tolist(), "brightness_sample": REF.brightness.round(5).tolist(),
             "confidence_bin_edges": EDGES.tolist(), "confidence_bin_frac": REF_FRAC.tolist(), "mean_confidence": float(REF.confidence.mean()),
             "psi_threshold": THRESH["psi_confidence"], "p_threshold": THRESH["p_input"], "window": WINDOW}
(ART / "reference.json").write_text(json.dumps(reference), encoding="utf-8")
print("reference ->", ART / "reference.json")

API_LOG = []
serving = ART / "serving" / "classifier_api"
if (serving / "app" / "main.py").exists():
    local = Path("/content/serving_day5") if IN_COLAB else OUTPUT_DIR / "serving_day5"
    if local.exists():
        shutil.rmtree(local)
    shutil.copytree(serving, local)
    shutil.copy(ART / "reference.json", local / "model" / "reference.json")
    proc = subprocess.Popen([sys.executable, "-m", "uvicorn", "app.main:app", "--port", "8010"], cwd=local,
                            env={**os.environ, "MODEL_DIR": str(local / "model")}, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
    for _ in range(60):
        try:
            if requests.get("http://127.0.0.1:8010/health", timeout=2).ok:
                break
        except requests.ConnectionError:
            time.sleep(2)

    def send(img: Image.Image, endpoint: str = "predict") -> requests.Response:
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=95)
        return requests.post(f"http://127.0.0.1:8010/{endpoint}", files={"file": ("x.jpg", buf.getvalue(), "image/jpeg")}, timeout=60)

    blur = corruption("gaussian_blur", 4)
    for phase, fn in (("clean", None), ("blurred", blur)):
        for p, _ in random.Random(7).sample(STREAM_SET, 200):
            im = Image.open(p).convert("RGB")
            send(fn(im) if fn else im)
        rep = requests.get("http://127.0.0.1:8010/drift").json()
        API_LOG.append({"after": phase, **rep})
        print(f"/drift after 200 {phase} requests:", rep)
    r = send(Image.open(STREAM_SET[0][0]).convert("RGB"), "gradcam")
    print("/gradcam ->", r.status_code, r.headers.get("content-type"), r.headers.get("X-Predicted-Label"))
    proc.terminate()
else:
    print("run Day 3 first to test the API endpoints (its service folder is saved to Drive)")

(OUTPUT_DIR / "day5_summary.json").write_text(json.dumps({
    "thresholds": THRESH, "false_alarm_rate_bootstrap": fa, "detected": int(tp), "drift_windows": int(STREAM.drifted.sum()),
    "clean_alerts": int(fp), "stream": STREAM.round(4).to_dict("records"), "per_class_under_corruption": PC.to_dict("index"),
    "api": API_LOG}, indent=2, default=float), encoding="utf-8")
print("saved ->", OUTPUT_DIR)

# %% [markdown]
# ## Checklist
#
# - [ ] Thresholds tuned on clean bootstrap windows (false-alarm rate reported) before looking at drift.
# - [ ] The timeline chart: clean windows quiet, corrupted and class-imbalanced windows alert — and which detector fired for each.
# - [ ] Accuracy per window and per class under corruption: the monitor catches drift that actually costs accuracy.
# - [ ] An Evidently drift report.
# - [ ] Grad-CAM heatmaps for right and wrong predictions and under corruption; `/gradcam` and `/drift` work over HTTP.
#
# **Weekend:** assemble the write-up — optimization table (Day 1), runtime benchmark (Day 2), serving architecture (Day 3), MLflow/DVC (Day 4), the drift chart and Grad-CAM examples (Day 5). Tag a release.
