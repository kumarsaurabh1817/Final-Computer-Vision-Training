"""
Week 8 · Day 1 — Optimizing the Week 2 classifier: INT8 quantization, pruning (unstructured and structured), distillation.

Colab-first script (percent-format cells). Build the notebook with:
    python ../tools/inline_shared.py && python ../tools/py_to_ipynb.py
Dataset: Intel Image Classification (Kaggle puneet6060/intel-image-classification) — read from
         My Drive/datasets/intel-image-classification/ if it is there.
"""

# %% [markdown]
# # Week 8 · Day 1 — Model Optimization
#
# **Goal:** make the Week 2 classifier (ResNet-18 fine-tuned on Intel scenes) smaller and faster, and know exactly what each step costs in accuracy — overall **and per class**.
#
# | Technique | What changes | Measured |
# |---|---|---|
# | Baseline FP32 | — | accuracy, worst class, size, CPU and GPU latency |
# | Dynamic INT8 | `Linear` weights stored as INT8, activations quantized on the fly | same |
# | Static INT8 (PTQ, FX) | weights **and** activations INT8, ranges calibrated on 256 images; conv+bn+relu fused | same + per-class deltas |
# | Unstructured pruning | 30/50/70 % of conv weights zeroed (global L1); fine-tuned 1 epoch | accuracy, sparsity, compressed size, latency |
# | Structured pruning | 30 % of channels removed (torch-pruning) — a genuinely smaller network | params, MACs, latency, accuracy after fine-tuning |
# | Distillation | MobileNetV3-Small student learns from the ResNet-18 teacher's soft labels | KD vs plain training |
#
# **Data:** your Drive folder **My Drive/datasets/intel-image-classification/** (or the Kaggle download). A fixed subset — 400 train / 300 test images per class — is copied to local disk.
#
# **Checkpoint:** put Week 2's `pretrained_aug.pt` in **My Drive/checkpoints/week2/** to optimize *your* model; otherwise the notebook fine-tunes one with the Week 2 recipe (3 epochs, ~4 min) and saves it to **My Drive/cv_course_artifacts/week8/** for Days 2–5.
#
# > Runtime: **T4 GPU** (quantized models run on the CPU), ~25 min.

# %% [markdown]
# ## 0 · Install dependencies

# %%
import subprocess
import sys


def pip_install(*packages: str) -> None:
    """Install packages into the running interpreter (works in Colab and plain Python)."""
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *packages])


pip_install("kagglehub>=0.3", "torch-pruning>=1.4")

# %% [markdown]
# ## 1 · Imports, configuration, and output folder

# %%
import copy
import gzip
import json
import os
import random
import time
from pathlib import Path

import matplotlib

IN_NOTEBOOK = "ipykernel" in sys.modules
if not IN_NOTEBOOK:
    matplotlib.use("Agg")  # headless when run as a script

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.prune as prune
from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small

IN_COLAB = "google.colab" in sys.modules
try:
    HERE = Path(__file__).resolve().parent
except NameError:  # running inside a notebook
    HERE = Path.cwd()

OUTPUT_DIR = Path("/content/week8_outputs/day1") if IN_COLAB else HERE.parent / "results" / "day1"
DATA_DIR = Path("/content/datasets") if IN_COLAB else HERE.parent / "data"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)

FULL_RUN = True
TRAIN_PER_CLASS = 400 if FULL_RUN else 100
TEST_PER_CLASS = 300 if FULL_RUN else 60
KD_EPOCHS = 3 if FULL_RUN else 1
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
torch.set_num_threads(os.cpu_count() or 2)
print(f"PyTorch {torch.__version__} | device={DEVICE} | CPU threads={torch.get_num_threads()} | FULL_RUN={FULL_RUN}")


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
# ## 3 · Data and the Week 2 model

# %%
TRAIN, TEST = load_intel(DATA_DIR, TRAIN_PER_CLASS, TEST_PER_CLASS, SEED)
TRAIN_LOADER, TEST_LOADER = make_loader(TRAIN, train=True), make_loader(TEST)
CALIB = make_loader(random.Random(0).sample(TRAIN, 256), bs=32)
MODEL, SOURCE = load_or_train_classifier(TRAIN, TEST, DEVICE)
print("model:", SOURCE)
X1 = torch.randn(1, 3, 224, 224)
X32 = torch.randn(32, 3, 224, 224)

# %% [markdown]
# ## 4 · The measuring stick
#
# Every variant gets the same row: accuracy and **worst-class accuracy** on the 1,800-image test subset, parameters, size of the saved weights, CPU latency at batch 1 (the serving case) and GPU latency where the variant runs on GPU.

# %%
ROWS, PER_CLASS = [], {}


@torch.inference_mode()
def record(name: str, model: nn.Module, device: str, gpu: bool = True, size_mb: float | None = None, note: str = "") -> dict:
    model = model.to(device).eval()
    y, pred, _ = predict(model, TEST_LOADER, device)
    rep = accuracy_report(y, pred)
    PER_CLASS[name] = {c: rep[f"acc_{c}"] for c in CLASSES}
    cpu_model = model.cpu() if device != "cpu" else model
    row = {"variant": name, "accuracy": rep["accuracy"], "worst_class": rep["worst_class"], "worst_class_acc": rep["worst_class_acc"],
           "params_M": sum(p.numel() for p in cpu_model.parameters()) / 1e6, "size_MB": size_mb if size_mb is not None else model_size_mb(cpu_model),
           "cpu_ms_b1": latency_ms(cpu_model, X1, iters=20), "note": note}
    if gpu and DEVICE == "cuda":
        g = cpu_model.to("cuda")
        row["gpu_ms_b1"] = latency_ms(g, X1.cuda())
        row["gpu_img_per_s_b32"] = 32 / (latency_ms(g, X32.cuda()) / 1000)
        g.cpu()
    ROWS.append(row)
    print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()})
    return row


record("fp32 (baseline)", copy.deepcopy(MODEL), DEVICE)

# %% [markdown]
# ## 5 · Quantization: INT8
#
# - **Dynamic** quantization converts `Linear` weights to INT8 and quantizes activations at run time. One line, but ResNet-18 is almost all convolutions, so expect little.
# - **Static post-training quantization (PTQ)** quantizes weights *and* activations. It needs **calibration**: run a few hundred real images through the model with observers to learn each activation's range. FX graph mode fuses Conv+BN+ReLU first. INT8 kernels (fbgemm / x86) run on the **CPU**.

# %%
dyn = torch.ao.quantization.quantize_dynamic(copy.deepcopy(MODEL).cpu().eval(), {nn.Linear}, dtype=torch.qint8)
record("int8 dynamic (Linear only)", dyn, "cpu", gpu=False)

PTQ = None
try:
    from torch.ao.quantization import get_default_qconfig_mapping
    from torch.ao.quantization.quantize_fx import convert_fx, prepare_fx

    fp = copy.deepcopy(MODEL).cpu().eval()
    prepared = prepare_fx(fp, get_default_qconfig_mapping("x86"), example_inputs=(X1,))
    with torch.inference_mode():
        for x, _ in CALIB:  # calibration: observers record activation ranges
            prepared(x)
    PTQ = convert_fx(prepared)
    record("int8 static PTQ (FX)", PTQ, "cpu", gpu=False, note="256 calibration images")
except Exception as exc:  # torch.ao.quantization is being phased out in favour of torchao; Day 2's ONNX INT8 path still works
    print("FX static quantization unavailable in this PyTorch build:", repr(exc)[:200])

# %% [markdown]
# ### Per-class check
#
# Quantization error is not spread evenly: classes that depend on subtle textures (glacier vs mountain, buildings vs street) can lose more than the average suggests.

# %%
pc = pd.DataFrame(PER_CLASS).T
delta = (pc - pc.loc["fp32 (baseline)"]).round(3)
print("Per-class accuracy:\n", pc.round(3).to_string())
print("\nChange vs FP32:\n", delta.to_string())

# %% [markdown]
# ## 6 · Pruning
#
# **Unstructured** (global L1): zero the smallest weights across all convolutions. The tensors keep their shape, so a dense GPU/CPU kernel is *not* faster — the gain is compressibility (and speed only with sparse kernels or hardware). **Structured**: remove whole channels, producing a genuinely smaller network that is faster everywhere.

# %%
def conv_params(m: nn.Module):
    return [(mod, "weight") for mod in m.modules() if isinstance(mod, nn.Conv2d)]


def sparsity(m: nn.Module) -> float:
    w = [mod.weight for mod, _ in conv_params(m)]
    return float(sum((t == 0).sum() for t in w) / sum(t.numel() for t in w))


def gzip_mb(m: nn.Module) -> float:
    buf = io.BytesIO()
    torch.save(m.state_dict(), buf)
    return len(gzip.compress(buf.getvalue())) / 1e6


one_epoch_opt = lambda m: torch.optim.AdamW(m.parameters(), lr=1e-4, weight_decay=1e-4)  # noqa: E731
for amount in (0.3, 0.5, 0.7):
    m = copy.deepcopy(MODEL).to(DEVICE)
    prune.global_unstructured(conv_params(m), pruning_method=prune.L1Unstructured, amount=amount)
    y, pred, _ = predict(m, TEST_LOADER, DEVICE)
    before = accuracy_report(y, pred)["accuracy"]
    run_epoch(m, TRAIN_LOADER, DEVICE, one_epoch_opt(m))  # masks stay applied while fine-tuning
    for mod, name in conv_params(m):
        prune.remove(mod, name)  # make the zeros permanent
    record(f"unstructured prune {int(amount * 100)}% + 1 ep", m, DEVICE, size_mb=gzip_mb(m),
           note=f"sparsity {sparsity(m):.2f}; acc before fine-tune {before:.3f}; size = gzip of weights")
print(f"gzip size of the dense FP32 weights for comparison: {gzip_mb(MODEL.cpu()):.1f} MB")

# %%
STRUCT = None
try:
    import torch_pruning as tp

    m = copy.deepcopy(MODEL).cpu().eval()
    macs0, params0 = tp.utils.count_ops_and_params(m, X1)
    pruner = tp.pruner.MetaPruner(m, X1, importance=tp.importance.MagnitudeImportance(p=2), pruning_ratio=0.3,
                                  ignored_layers=[m.fc[1]])  # never prune the output classes
    pruner.step()
    macs1, params1 = tp.utils.count_ops_and_params(m, X1)
    print(f"structured 30%: MACs {macs0 / 1e9:.2f}G -> {macs1 / 1e9:.2f}G | params {params0 / 1e6:.1f}M -> {params1 / 1e6:.1f}M")
    y, pred, _ = predict(m.to(DEVICE), TEST_LOADER, DEVICE)
    before = accuracy_report(y, pred)["accuracy"]
    fine_tune(m, TRAIN_LOADER, TEST_LOADER, DEVICE, epochs=2, head_lr=1e-3, backbone_lr=3e-4)
    STRUCT = m
    record("structured prune 30% + 2 ep", m, DEVICE, note=f"MACs {macs1 / macs0:.0%} of baseline; acc before fine-tune {before:.3f}")
except Exception as exc:
    print("structured pruning skipped:", repr(exc)[:200])

# %% [markdown]
# ## 7 · Distillation: a smaller student learns from the teacher
#
# The student (MobileNetV3-Small, ImageNet-pretrained, ~1.5M params) is trained on a mix of the true labels and the teacher's **soft** probabilities at temperature T. Soft labels carry "dark knowledge": a glacier photo that the teacher thinks is 20 % mountain teaches the student which classes are confusable.
#
# `loss = α · T² · KL(softmax(student/T) ‖ softmax(teacher/T)) + (1 − α) · CE(student, y)`

# %%
def make_student() -> nn.Module:
    s = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.DEFAULT)
    s.classifier[-1] = nn.Linear(s.classifier[-1].in_features, len(CLASSES))
    return s


def train_student(student: nn.Module, teacher: nn.Module | None, epochs: int, T: float = 4.0, alpha: float = 0.7) -> nn.Module:
    student.to(DEVICE)
    if teacher is not None:
        teacher.to(DEVICE).eval()
    opt = torch.optim.AdamW(student.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    for ep in range(epochs):
        student.train()
        for x, y in TRAIN_LOADER:
            x, y = x.to(DEVICE), y.to(DEVICE)
            s = student(x)
            loss = F.cross_entropy(s, y)
            if teacher is not None:
                with torch.no_grad():
                    t = teacher(x)
                kd = F.kl_div(F.log_softmax(s / T, 1), F.softmax(t / T, 1), reduction="batchmean") * T * T
                loss = alpha * kd + (1 - alpha) * loss
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        sched.step()
        print(f"  {'KD' if teacher is not None else 'CE'} epoch {ep + 1}/{epochs}: test acc {run_epoch(student, TEST_LOADER, DEVICE)['accuracy']:.3f}")
    return student.eval()


torch.manual_seed(SEED)
student_ce = train_student(make_student(), None, KD_EPOCHS)
record("mobilenetv3-small (CE only)", student_ce, DEVICE, note=f"{KD_EPOCHS} epochs, labels only")
torch.manual_seed(SEED)
student_kd = train_student(make_student(), copy.deepcopy(MODEL), KD_EPOCHS)
record("mobilenetv3-small (distilled)", student_kd, DEVICE, note=f"{KD_EPOCHS} epochs, T=4, alpha=0.7")

# %% [markdown]
# ## 8 · The trade-off table

# %%
RESULTS = pd.DataFrame(ROWS).set_index("variant")
base = RESULTS.loc["fp32 (baseline)"]
RESULTS["acc_delta"] = RESULTS.accuracy - base.accuracy
RESULTS["cpu_speedup"] = base.cpu_ms_b1 / RESULTS.cpu_ms_b1
RESULTS["size_ratio"] = RESULTS.size_MB / base.size_MB
cols = ["accuracy", "acc_delta", "worst_class", "worst_class_acc", "params_M", "size_MB", "size_ratio", "cpu_ms_b1", "cpu_speedup",
        "gpu_ms_b1", "gpu_img_per_s_b32", "note"]
print(RESULTS[[c for c in cols if c in RESULTS]].round(3).to_string())
RESULTS.round(4).to_csv(OUTPUT_DIR / "day1_optimization_table.csv")
(OUTPUT_DIR / "day1_optimization_table.md").write_text(RESULTS[[c for c in cols if c in RESULTS]].round(3).to_markdown(), encoding="utf-8")

fig, ax = plt.subplots(figsize=(9, 5))
for name, r in RESULTS.iterrows():
    ax.scatter(r.cpu_ms_b1, r.accuracy, s=40 + 8 * r.size_MB)
    ax.annotate(name, (r.cpu_ms_b1, r.accuracy), textcoords="offset points", xytext=(5, 4), fontsize=8)
ax.set_xlabel("CPU latency, batch 1 (ms)")
ax.set_ylabel("test accuracy")
ax.set_title("Accuracy vs CPU latency (marker size = model size)")
ax.grid(alpha=0.3)
save_fig(fig, "day1_tradeoffs.png")

# %% [markdown]
# ## 9 · Save the variants for Days 2–5

# %%
ART = artifact_dir("week8")
torch.save(MODEL.cpu().state_dict(), ART / "resnet18_week2.pt")
torch.save(student_kd.cpu().state_dict(), ART / "mobilenetv3_small_kd.pt")
if PTQ is not None:
    try:
        torch.jit.save(torch.jit.trace(PTQ, X1), str(ART / "resnet18_int8_ptq.torchscript.pt"))
    except Exception as exc:
        print("could not TorchScript the INT8 model:", repr(exc)[:150])
(ART / "labels.json").write_text(json.dumps(CLASSES), encoding="utf-8")
(OUTPUT_DIR / "day1_summary.json").write_text(json.dumps({"checkpoint": SOURCE, "results": RESULTS.round(4).to_dict("index"),
                                                          "per_class": PER_CLASS}, indent=2, default=str), encoding="utf-8")
print("saved ->", ART)

# %% [markdown]
# ## Checklist
#
# - [ ] One table: accuracy (overall and worst class), params, size, CPU/GPU latency for FP32, INT8 dynamic, INT8 static, pruned, distilled.
# - [ ] Per-class accuracy change after quantization checked, not just the average.
# - [ ] I can explain why 70 % unstructured sparsity does not make the dense model faster, and what structured pruning does instead.
# - [ ] KD student vs the same student trained on labels only.
#
# **Next (Day 2):** export to ONNX and run it with ONNX Runtime (CPU, CUDA, TensorRT), plus ONNX INT8.
