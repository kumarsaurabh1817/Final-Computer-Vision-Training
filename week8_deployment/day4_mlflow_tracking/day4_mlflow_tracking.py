"""
Week 8 · Day 4 — Experiment tracking & versioning: MLflow runs, comparison and Model Registry; DVC data versioning; reproducing a run.

Colab-first script (percent-format cells). Build the notebook with:
    python ../tools/inline_shared.py && python ../tools/py_to_ipynb.py
Dataset: Intel Image Classification (My Drive/datasets/intel-image-classification/ or Kaggle).
"""

# %% [markdown]
# # Week 8 · Day 4 — Experiment Tracking & Versioning
#
# **Goal:** any number you report can be traced to *the code, the parameters, the data and the model* that produced it — and re-produced. The Week 2 training is instrumented with MLflow, the data is versioned with DVC (and a content hash), the best model goes into the MLflow **Model Registry** with an alias, and one logged run is reproduced from its record.
#
# | Topic | What you will build / measure |
# |---|---|
# | MLflow tracking | Params, per-epoch metrics, tags (data hash, DVC md5), artifacts (confusion matrix, per-class CSV), the model with an input example |
# | Comparing runs | `mlflow.search_runs` → a table and a plot of three training configurations |
# | Model Registry | Register the best run's model, alias it `champion`, load it back by alias and re-check its accuracy |
# | DVC | `dvc add` the dataset subset, read the `.dvc` pointer (md5, files), detect a change with `dvc status`, restore it with `dvc checkout` |
# | Reproducibility | Re-run the champion's parameters on the same data hash with deterministic settings; compare metrics |
#
# > Runtime: **T4 GPU**, ~15 min. The MLflow database and artifacts are copied to **My Drive/cv_course_artifacts/week8/mlflow/** at the end.

# %% [markdown]
# ## 0 · Install dependencies

# %%
import subprocess
import sys


def pip_install(*packages: str) -> None:
    """Install packages into the running interpreter (works in Colab and plain Python)."""
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *packages])


pip_install("kagglehub>=0.3", "mlflow>=2.14", "dvc>=3.0")

# %% [markdown]
# ## 1 · Imports, configuration, and output folder
#
# Determinism has to be switched on **before** CUDA is initialized: `CUBLAS_WORKSPACE_CONFIG` and `torch.use_deterministic_algorithms`.

# %%
import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")  # deterministic cuBLAS (must precede CUDA init)

import json
import random
import re
import shutil
import time
from pathlib import Path

import matplotlib

IN_NOTEBOOK = "ipykernel" in sys.modules
if not IN_NOTEBOOK:
    matplotlib.use("Agg")  # headless when run as a script

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
import torch
from mlflow.tracking import MlflowClient
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

IN_COLAB = "google.colab" in sys.modules
try:
    HERE = Path(__file__).resolve().parent
except NameError:  # running inside a notebook
    HERE = Path.cwd()

OUTPUT_DIR = Path("/content/week8_outputs/day4") if IN_COLAB else HERE.parent / "results" / "day4"
DATA_DIR = Path("/content/datasets") if IN_COLAB else HERE.parent / "data"
MLFLOW_DIR = Path("/content/mlflow") if IN_COLAB else HERE.parent / "results" / "mlflow"
DVC_DIR = Path("/content/dvc_project") if IN_COLAB else HERE.parent / "results" / "dvc_project"
for d in (OUTPUT_DIR, DATA_DIR, MLFLOW_DIR):
    d.mkdir(parents=True, exist_ok=True)

FULL_RUN = True
TRAIN_PER_CLASS = 300 if FULL_RUN else 80
TEST_PER_CLASS = 150 if FULL_RUN else 40
EPOCHS = 2
EXPERIMENT = "week8-intel-resnet18"
REGISTERED_NAME = "intel-scene-classifier"
SEED = 42
torch.use_deterministic_algorithms(True, warn_only=True)
torch.backends.cudnn.benchmark = False
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"PyTorch {torch.__version__} | MLflow {mlflow.__version__} | device={DEVICE}")


def save_fig(fig, name: str) -> Path:
    """Save a figure to OUTPUT_DIR and show it inline when in a notebook."""
    path = OUTPUT_DIR / name
    fig.savefig(path, dpi=120, bbox_inches="tight")
    if IN_NOTEBOOK:
        plt.show()
    plt.close(fig)
    print(f"saved -> {path}")
    return path


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


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
# ## 3 · Data, and its version
#
# A **content hash** of the exact files (names + bytes) is the cheapest data version: if the hash logged with a run equals the hash of the data you have now, you are training on the same data.

# %%
TRAIN, TEST = load_intel(DATA_DIR, TRAIN_PER_CLASS, TEST_PER_CLASS, SEED)
DATA_HASH = {"train": dataset_hash(TRAIN), "test": dataset_hash(TEST)}
print("data version:", DATA_HASH)

# %% [markdown]
# ## 4 · DVC: version the dataset next to the code
#
# DVC stores the data in its cache and puts a small `.dvc` pointer file (md5, size, file count) in Git. Changing a file changes the md5; `dvc status` sees it and `dvc checkout` restores the version the pointer names. With a remote (S3, GCS, Google Drive, SSH) `dvc push/pull` shares the data like `git push/pull` shares code.

# %%
def sh(cmd: str, cwd: Path = DVC_DIR) -> str:
    r = subprocess.run(cmd, shell=True, cwd=cwd, capture_output=True, text=True)
    out = (r.stdout + r.stderr).strip()
    print(f"$ {cmd}\n{out}\n" if out else f"$ {cmd}")
    return out


DVC_MD5 = None
try:
    if DVC_DIR.exists():
        shutil.rmtree(DVC_DIR)
    (DVC_DIR / "data").mkdir(parents=True)
    shutil.copytree(DATA_DIR / "local" / "train", DVC_DIR / "data" / "intel_subset" / "train")
    shutil.copytree(DATA_DIR / "local" / "test", DVC_DIR / "data" / "intel_subset" / "test")
    sh("git init -q && git config user.email 'you@example.com' && git config user.name 'course'")
    sh("dvc init -q && git add . && git commit -qm 'init dvc'")
    sh("dvc add data/intel_subset")
    sh("git add data/intel_subset.dvc data/.gitignore && git commit -qm 'data: intel subset v1'")
    print((DVC_DIR / "data" / "intel_subset.dvc").read_text())
    DVC_MD5 = re.search(r"md5: (\S+)", (DVC_DIR / "data" / "intel_subset.dvc").read_text()).group(1)
    victim = next((DVC_DIR / "data" / "intel_subset" / "train" / CLASSES[0]).glob("*.jpg"))
    victim.unlink()  # simulate someone changing the data
    sh("dvc status")
    sh("dvc checkout data/intel_subset.dvc")
    sh("dvc status")
except Exception as exc:
    print("DVC demo skipped:", repr(exc)[:200])
print("DVC md5 of the dataset:", DVC_MD5)

# %% [markdown]
# ## 5 · MLflow: instrument the Week 2 training
#
# Tracking store: a SQLite file (needed for the Model Registry). Each run logs:
#
# - **params** — architecture, learning rates, epochs, augmentation, batch size, seed;
# - **tags** — data hash, DVC md5, checkpoint source;
# - **metrics** — train/val loss and accuracy per epoch; final test accuracy, worst-class accuracy;
# - **artifacts** — confusion matrix, per-class CSV, the model (`mlflow.pytorch.log_model` with an input example → signature).

# %%
mlflow.set_tracking_uri(f"sqlite:///{MLFLOW_DIR / 'mlflow.db'}")
CLIENT = MlflowClient()
exp = CLIENT.get_experiment_by_name(EXPERIMENT)
EXP_ID = exp.experiment_id if exp else CLIENT.create_experiment(EXPERIMENT, artifact_location=(MLFLOW_DIR / "artifacts").as_uri())
mlflow.set_experiment(EXPERIMENT)


def log_model(model, example: np.ndarray):
    try:
        return mlflow.pytorch.log_model(model, name="model", input_example=example)  # MLflow 3
    except TypeError:
        return mlflow.pytorch.log_model(model, artifact_path="model", input_example=example)  # MLflow 2


def train_run(params: dict, run_name: str) -> tuple[str, dict]:
    seed_everything(params["seed"])
    model = build_model()
    train_loader = make_loader(TRAIN, train=True, augment=params["augment"], bs=params["batch_size"], seed=params["seed"])
    test_loader = make_loader(TEST, bs=64)
    with mlflow.start_run(run_name=run_name) as run:
        mlflow.log_params(params)
        mlflow.set_tags({"data_hash_train": DATA_HASH["train"], "data_hash_test": DATA_HASH["test"], "dvc_md5": str(DVC_MD5),
                         "torch": torch.__version__, "week": "8"})

        def log(ep, tr, va):
            mlflow.log_metrics({"train_loss": tr["loss"], "train_acc": tr["accuracy"], "val_loss": va["loss"], "val_acc": va["accuracy"]}, step=ep)
        fine_tune(model, train_loader, test_loader, DEVICE, epochs=params["epochs"], head_lr=params["head_lr"],
                  backbone_lr=params["backbone_lr"], log=log)
        y, pred, _ = predict(model, test_loader, DEVICE)
        rep = accuracy_report(y, pred)
        mlflow.log_metrics({"test_acc": rep["accuracy"], "test_worst_class_acc": rep["worst_class_acc"]})
        fig, ax = plt.subplots(figsize=(6, 6))
        ConfusionMatrixDisplay(confusion_matrix(y, pred, normalize="true"), display_labels=CLASSES).plot(ax=ax, values_format=".2f",
                                                                                                         colorbar=False, xticks_rotation=45)
        mlflow.log_figure(fig, "confusion_matrix.png")
        plt.close(fig)
        per_class = OUTPUT_DIR / f"{run_name}_per_class.csv"
        pd.Series({c: rep[f"acc_{c}"] for c in CLASSES}, name="accuracy").to_csv(per_class)
        mlflow.log_artifact(str(per_class))
        info = log_model(model.cpu().eval(), np.random.rand(1, 3, 224, 224).astype(np.float32))
        mlflow.set_tag("model_uri", info.model_uri)
        print(f"{run_name}: test acc {rep['accuracy']:.4f} (run {run.info.run_id})")
        return run.info.run_id, rep


BASE = {"arch": "resnet18", "epochs": EPOCHS, "batch_size": 64, "head_lr": 1e-3, "backbone_lr": 1e-4, "augment": True, "seed": SEED}
CONFIGS = {"baseline": BASE, "higher_backbone_lr": {**BASE, "backbone_lr": 3e-4}, "no_augmentation": {**BASE, "augment": False}}
RUN_IDS = {name: train_run(p, name)[0] for name, p in CONFIGS.items()}

# %% [markdown]
# ## 6 · Compare runs

# %%
RUNS = mlflow.search_runs(experiment_names=[EXPERIMENT], filter_string="tags.week = '8'", order_by=["metrics.test_acc DESC"])
cols = ["tags.mlflow.runName", "metrics.test_acc", "metrics.test_worst_class_acc", "params.backbone_lr", "params.augment",
        "tags.data_hash_train", "run_id"]
print(RUNS[[c for c in cols if c in RUNS]].head(10).to_string(index=False))

fig, ax = plt.subplots(figsize=(8, 4))
for name, rid in RUN_IDS.items():
    hist = CLIENT.get_metric_history(rid, "val_acc")
    ax.plot([m.step for m in hist], [m.value for m in hist], "o-", label=name)
ax.set_xlabel("epoch")
ax.set_ylabel("validation accuracy")
ax.legend()
ax.grid(alpha=0.3)
save_fig(fig, "day4_runs.png")

# %% [markdown]
# ## 7 · Model Registry: register, alias, load by alias
#
# The registry gives a model a **name** and numbered **versions**; an **alias** (`champion`, `challenger`) is a movable pointer. Serving code loads `models:/intel-scene-classifier@champion` and never hard-codes a run id — promoting a new model is moving the alias.

# %%
best_name = max(RUN_IDS, key=lambda n: CLIENT.get_run(RUN_IDS[n]).data.metrics["test_acc"])
best_run = CLIENT.get_run(RUN_IDS[best_name])
mv = mlflow.register_model(best_run.data.tags["model_uri"], REGISTERED_NAME)
CLIENT.set_registered_model_alias(REGISTERED_NAME, "champion", mv.version)
CLIENT.update_model_version(REGISTERED_NAME, mv.version, description=f"{best_name}: test acc {best_run.data.metrics['test_acc']:.4f}, "
                                                                        f"data {DATA_HASH['train']}")
print(f"registered {REGISTERED_NAME} v{mv.version} from run '{best_name}' and set alias 'champion'")

champion = mlflow.pytorch.load_model(f"models:/{REGISTERED_NAME}@champion").to(DEVICE).eval()
y, pred, _ = predict(champion, make_loader(TEST), DEVICE)
print(f"champion loaded by alias: test acc {accuracy_report(y, pred)['accuracy']:.4f} "
      f"(logged {best_run.data.metrics['test_acc']:.4f})")

art = artifact_dir("week8")
for f in ("resnet18_fp32.onnx", "resnet18_int8.onnx"):
    if (art / f).exists():
        with mlflow.start_run(run_id=best_run.info.run_id):
            mlflow.log_artifact(str(art / f), artifact_path="onnx")  # the deployable files travel with the run
        print("attached", f, "to the champion run")

# %% [markdown]
# ## 8 · Reproduce the champion from its record
#
# Everything needed is in the run: parameters, and the data hash to check that the data is the same. Re-train with exactly those parameters and compare. With deterministic algorithms and the same data order, the metrics should match to within GPU floating-point noise.

# %%
record = CLIENT.get_run(RUN_IDS[best_name])
params = {k: (v == "True" if v in ("True", "False") else type(BASE[k])(v)) for k, v in record.data.params.items()}
assert record.data.tags["data_hash_train"] == dataset_hash(TRAIN), "the training data changed since the run was logged"
_, rep = train_run(params, f"{best_name}_reproduction")
orig = record.data.metrics["test_acc"]
REPRO = {"original_test_acc": orig, "reproduced_test_acc": rep["accuracy"], "abs_diff": abs(orig - rep["accuracy"]),
         "params": params, "data_hash": DATA_HASH["train"]}
print(json.dumps(REPRO, indent=1))
print("REPRODUCED" if REPRO["abs_diff"] <= 0.005 else "NOT REPRODUCED within 0.5 points: check nondeterministic ops / data order")

# %% [markdown]
# ## 9 · Keep the tracking store

# %%
dst = artifact_dir("week8") / "mlflow"
if dst.exists():
    shutil.rmtree(dst)
shutil.copytree(MLFLOW_DIR, dst)
(OUTPUT_DIR / "day4_summary.json").write_text(json.dumps({"runs": {n: CLIENT.get_run(r).data.metrics for n, r in RUN_IDS.items()},
                                                          "champion": {"name": best_name, "version": mv.version}, "reproduction": REPRO,
                                                          "data_hash": DATA_HASH, "dvc_md5": DVC_MD5}, indent=2, default=str), encoding="utf-8")
print("MLflow store ->", dst)
print(f"Browse it locally:  mlflow ui --backend-store-uri sqlite:///{dst / 'mlflow.db'}")

# %% [markdown]
# ## Checklist
#
# - [ ] Three runs with params, per-epoch metrics, tags (data hash, DVC md5) and artifacts; compared in a table and a plot.
# - [ ] Best model registered, aliased `champion`, loaded by alias, accuracy re-checked.
# - [ ] DVC pointer file read; a data change detected by `dvc status` and undone by `dvc checkout`.
# - [ ] One run reproduced from its record, with the difference reported.
#
# **Next (Day 5):** detect drift in a stream of incoming images, and explain predictions with Grad-CAM.
