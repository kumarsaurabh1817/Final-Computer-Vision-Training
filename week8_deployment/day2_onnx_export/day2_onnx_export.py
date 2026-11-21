"""
Week 8 · Day 2 — Export & accelerate: PyTorch -> ONNX -> ONNX Runtime (CPU, CUDA, TensorRT) and ONNX INT8, benchmarked.

Colab-first script (percent-format cells). Build the notebook with:
    python ../tools/inline_shared.py && python ../tools/py_to_ipynb.py
Dataset: Intel Image Classification (My Drive/datasets/intel-image-classification/ or Kaggle).
"""

# %% [markdown]
# # Week 8 · Day 2 — Export to ONNX and Accelerate
#
# **Goal:** take the model out of PyTorch. ONNX is a portable graph format; **ONNX Runtime** (ORT) runs it on CPU, CUDA or TensorRT with graph optimizations — and without PyTorch in the serving image (Day 3).
#
# | Topic | What you will build / measure |
# |---|---|
# | Export | `torch.onnx.export` with a dynamic batch axis; `onnx.checker`; numerical parity with PyTorch |
# | Graph optimizations | ORT's fused/folded graph saved to disk |
# | Execution providers | CPU, CUDA, TensorRT (FP16) — batch-1 latency and batch-32 throughput vs PyTorch on the same fixed batch |
# | ONNX INT8 | Static QDQ quantization with a calibration reader — accuracy, size, CPU latency |
# | Accuracy check | Every runtime on the same 1,800 test images, overall and worst class |
#
# > Runtime: **T4 GPU**, ~10–15 min. Uses the checkpoint saved on Day 1 (My Drive/cv_course_artifacts/week8/), or trains one.

# %% [markdown]
# ## 0 · Install dependencies

# %%
import subprocess
import sys


def pip_install(*packages: str) -> None:
    """Install packages into the running interpreter (works in Colab and plain Python)."""
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *packages])


pip_install("kagglehub>=0.3", "onnx>=1.16", "onnxruntime-gpu>=1.19", "onnxscript")

# %% [markdown]
# ## 1 · Imports, configuration, and output folder

# %%
import copy
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
import onnx
import onnxruntime as ort
import pandas as pd
import torch
from onnxruntime.quantization import CalibrationDataReader, QuantFormat, QuantType, quantize_static

IN_COLAB = "google.colab" in sys.modules
try:
    HERE = Path(__file__).resolve().parent
except NameError:  # running inside a notebook
    HERE = Path.cwd()

OUTPUT_DIR = Path("/content/week8_outputs/day2") if IN_COLAB else HERE.parent / "results" / "day2"
DATA_DIR = Path("/content/datasets") if IN_COLAB else HERE.parent / "data"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)

FULL_RUN = True
TRAIN_PER_CLASS = 400 if FULL_RUN else 100
TEST_PER_CLASS = 300 if FULL_RUN else 60
OPSET = 17
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"PyTorch {torch.__version__} | ONNX Runtime {ort.__version__} | providers {ort.get_available_providers()}")


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
# ## 3 · Data and model

# %%
TRAIN, TEST = load_intel(DATA_DIR, TRAIN_PER_CLASS, TEST_PER_CLASS, SEED)
TEST_LOADER = make_loader(TEST, bs=32)
MODEL, SOURCE = load_or_train_classifier(TRAIN, TEST, DEVICE)
MODEL = MODEL.cpu().eval()
ART = artifact_dir("week8")
print("model:", SOURCE, "| artifacts ->", ART)

# A fixed benchmark batch of real test images, identical for every runtime
X_TEST = torch.stack([EVAL_TF(Image.open(p).convert("RGB")) for p, _ in TEST[:32]])
Y_ALL = np.array([y for _, y in TEST])

# %% [markdown]
# ## 4 · Export to ONNX
#
# The exporter traces the model with an example input and writes the graph + weights. `dynamic_axes` keeps the batch dimension free so one file serves batch 1 and batch 32. Then: the ONNX checker validates the graph, and we compare logits with PyTorch on the same batch (max absolute difference ~1e-5 is float noise; anything large is an export bug).

# %%
def export_onnx(model: torch.nn.Module, path: Path) -> Path:
    kw = dict(input_names=["input"], output_names=["logits"], opset_version=OPSET,
              dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}})
    try:
        torch.onnx.export(model, torch.randn(1, 3, 224, 224), str(path), dynamo=False, **kw)  # classic TorchScript exporter
    except TypeError:  # very old torch without the dynamo flag
        torch.onnx.export(model, torch.randn(1, 3, 224, 224), str(path), **kw)
    onnx.checker.check_model(onnx.load(str(path)))
    return path


FP32_ONNX = export_onnx(MODEL, ART / "resnet18_fp32.onnx")
print(f"exported {FP32_ONNX} ({FP32_ONNX.stat().st_size / 1e6:.1f} MB)")
g = onnx.load(str(FP32_ONNX))
print("inputs:", [(i.name, [d.dim_param or d.dim_value for d in i.type.tensor_type.shape.dim]) for i in g.graph.input],
      "| nodes:", len(g.graph.node), "| op types:", sorted({n.op_type for n in g.graph.node}))

sess = ort.InferenceSession(str(FP32_ONNX), providers=["CPUExecutionProvider"])
with torch.inference_mode():
    ref = MODEL(X_TEST).numpy()
out = sess.run(None, {"input": X_TEST.numpy()})[0]
print(f"parity: max |logit diff| = {np.abs(ref - out).max():.2e} | same top-1: {(ref.argmax(1) == out.argmax(1)).mean():.3f}")

# %% [markdown]
# ### Graph optimizations
#
# ORT fuses Conv+BN, folds constants and removes no-ops when it loads a model. Saving the optimized graph shows what changed (and speeds up later loads).

# %%
so = ort.SessionOptions()
so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
so.optimized_model_filepath = str(ART / "resnet18_fp32.opt.onnx")
ort.InferenceSession(str(FP32_ONNX), so, providers=["CPUExecutionProvider"])
opt = onnx.load(so.optimized_model_filepath)
print(f"nodes: {len(g.graph.node)} -> {len(opt.graph.node)} after ORT_ENABLE_ALL")

# %% [markdown]
# ## 5 · ONNX INT8 (static, QDQ)
#
# The ONNX equivalent of Day 1's static PTQ: a `CalibrationDataReader` feeds ~256 training images, ORT computes activation ranges and inserts QuantizeLinear/DequantizeLinear pairs around INT8 ops. Runs anywhere ORT runs — no PyTorch quantization backend needed.

# %%
class CalibReader(CalibrationDataReader):
    def __init__(self, samples: list, n: int = 256, bs: int = 16):
        imgs = random.Random(0).sample(samples, min(n, len(samples)))
        self.batches = iter([np.stack([EVAL_TF(Image.open(p).convert("RGB")).numpy() for p, _ in imgs[i:i + bs]])
                             for i in range(0, len(imgs), bs)])

    def get_next(self):
        b = next(self.batches, None)
        return None if b is None else {"input": b}


INT8_ONNX = ART / "resnet18_int8.onnx"
try:
    from onnxruntime.quantization.shape_inference import quant_pre_process
    pre = ART / "resnet18_fp32.pre.onnx"
    quant_pre_process(str(FP32_ONNX), str(pre))
except Exception as exc:
    print("pre-processing skipped:", repr(exc)[:120])
    pre = FP32_ONNX
quantize_static(str(pre), str(INT8_ONNX), CalibReader(TRAIN), quant_format=QuantFormat.QDQ,
                activation_type=QuantType.QUInt8, weight_type=QuantType.QInt8, per_channel=True)
print(f"INT8 model: {INT8_ONNX.stat().st_size / 1e6:.1f} MB (FP32 {FP32_ONNX.stat().st_size / 1e6:.1f} MB)")

# %% [markdown]
# ## 6 · Every runtime on the same inputs
#
# Latency: median of 30 runs on the fixed batch (batch 1 = first image, batch 32 = all), after warm-up. Accuracy: all test images. TensorRT builds an engine on first use (can take a minute) — it is optional and skipped if the TensorRT libraries are missing.

# %%
def ort_session(path: Path, provider: str):
    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    providers = {"cpu": ["CPUExecutionProvider"], "cuda": ["CUDAExecutionProvider", "CPUExecutionProvider"],
                 "tensorrt": [("TensorrtExecutionProvider", {"trt_fp16_enable": True, "trt_engine_cache_enable": True,
                                                             "trt_engine_cache_path": str(ART / "trt_cache")}),
                              "CUDAExecutionProvider", "CPUExecutionProvider"]}[provider]
    s = ort.InferenceSession(str(path), opts, providers=providers)
    return s, s.get_providers()[0]


def ort_accuracy(s) -> dict:
    preds = []
    for x, _ in TEST_LOADER:
        preds.append(s.run(None, {"input": x.numpy()})[0].argmax(1))
    return accuracy_report(Y_ALL, np.concatenate(preds))


RUNS = []


def bench(name: str, run_fn, acc: dict, size_mb: float) -> None:
    b1 = latency_ms(run_fn, X_TEST[:1])
    b32 = latency_ms(run_fn, X_TEST)
    RUNS.append({"runtime": name, "accuracy": acc["accuracy"], "worst_class_acc": acc["worst_class_acc"], "size_MB": size_mb,
                 "ms_b1": b1, "img_per_s_b32": 32 / (b32 / 1000)})
    print(RUNS[-1])


fp32_mb = FP32_ONNX.stat().st_size / 1e6
y, pred, _ = predict(MODEL, TEST_LOADER, "cpu")
with torch.inference_mode():
    bench("pytorch cpu fp32", lambda x: MODEL(x), accuracy_report(y, pred), model_size_mb(MODEL))
    if DEVICE == "cuda":
        gpu = copy.deepcopy(MODEL).cuda().eval()
        y, pred, _ = predict(gpu, TEST_LOADER, "cuda")
        bench("pytorch cuda fp32", lambda x: gpu(x.cuda()).cpu(), accuracy_report(y, pred), model_size_mb(MODEL))
        gpu16 = copy.deepcopy(MODEL).cuda().half().eval()
        p16 = np.concatenate([gpu16(x.cuda().half()).argmax(1).cpu().numpy() for x, _ in TEST_LOADER])
        bench("pytorch cuda fp16", lambda x: gpu16(x.cuda().half()).cpu(), accuracy_report(Y_ALL, p16), model_size_mb(MODEL) / 2)

for path, label, mb in ((FP32_ONNX, "fp32", fp32_mb), (INT8_ONNX, "int8", INT8_ONNX.stat().st_size / 1e6)):
    for provider in ("cpu", "cuda", "tensorrt"):
        if provider != "cpu" and DEVICE != "cuda":
            continue
        if label == "int8" and provider == "tensorrt":
            continue  # QDQ INT8 in TensorRT needs explicit-precision setup; out of scope here
        try:
            s, used = ort_session(path, provider)
            if provider != "cpu" and used == "CPUExecutionProvider":
                print(f"{provider} provider not available for {label}; skipped")
                continue
            bench(f"onnxruntime {provider} {label}", lambda x, s=s: s.run(None, {"input": x.numpy()})[0], ort_accuracy(s), mb)
        except Exception as exc:
            print(f"onnxruntime {provider} {label} skipped:", repr(exc)[:200])

BENCH = pd.DataFrame(RUNS).set_index("runtime")
BENCH["speedup_vs_torch_cpu_b1"] = BENCH.loc["pytorch cpu fp32", "ms_b1"] / BENCH.ms_b1
print(BENCH.round(3).to_string())
BENCH.round(4).to_csv(OUTPUT_DIR / "day2_runtime_benchmark.csv")

fig, axes = plt.subplots(1, 2, figsize=(15, 4.5))
BENCH.ms_b1.sort_values().plot.barh(ax=axes[0], title="batch-1 latency (ms, lower is better)")
BENCH.img_per_s_b32.sort_values().plot.barh(ax=axes[1], title="batch-32 throughput (images/s, higher is better)")
for ax in axes:
    ax.grid(axis="x", alpha=0.3)
save_fig(fig, "day2_runtimes.png")

# %% [markdown]
# ## 7 · The distilled student too
#
# Day 1's MobileNetV3-Small, exported and run the same way: the cheapest model to serve on a CPU.

# %%
student_path = ART / "mobilenetv3_small_kd.pt"
if student_path.exists():
    from torchvision.models import mobilenet_v3_small
    st = mobilenet_v3_small()
    st.classifier[-1] = torch.nn.Linear(st.classifier[-1].in_features, len(CLASSES))
    st.load_state_dict(torch.load(student_path, map_location="cpu", weights_only=True))
    st_onnx = export_onnx(st.eval(), ART / "mobilenetv3_small_kd.onnx")
    s, _ = ort_session(st_onnx, "cpu")
    acc = ort_accuracy(s)
    print(f"student ONNX on CPU: accuracy {acc['accuracy']:.3f} | {latency_ms(lambda x: s.run(None, {'input': x.numpy()}), X_TEST[:1]):.2f} ms "
          f"at batch 1 | {st_onnx.stat().st_size / 1e6:.1f} MB")
else:
    print("run Day 1 to create the distilled student")

(OUTPUT_DIR / "day2_summary.json").write_text(json.dumps({"benchmark": BENCH.round(4).to_dict("index"), "onnx": str(FP32_ONNX),
                                                          "onnx_int8": str(INT8_ONNX), "checkpoint": SOURCE}, indent=2, default=str), encoding="utf-8")
print("ONNX models ->", ART)

# %% [markdown]
# ## Checklist
#
# - [ ] ONNX export passes the checker and matches PyTorch logits (max diff ~1e-5, same top-1).
# - [ ] Latency and throughput for PyTorch vs ORT CPU / CUDA / TensorRT on the same batch.
# - [ ] ONNX INT8: size, accuracy (overall + worst class) and CPU latency vs FP32.
# - [ ] I know which runtime I would serve with on a CPU box and on a GPU box, and why.
#
# **Next (Day 3):** serve the ONNX model behind FastAPI in a Docker image — then the RAG system the same way.
