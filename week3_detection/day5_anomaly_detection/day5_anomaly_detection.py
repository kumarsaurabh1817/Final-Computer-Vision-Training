"""
Week 3 · Day 5 — Anomaly detection on MVTec AD: autoencoder reconstruction vs embedding distance
(PatchCore built from scratch) vs the anomalib library (PatchCore + PaDiM).

Colab-first script (percent-format cells). Build the notebook with:
    python ../tools/py_to_ipynb.py day5_anomaly_detection/day5_anomaly_detection.py
Kaggle dataset: https://www.kaggle.com/datasets/ipythonx/mvtec-ad
"""

# %% [markdown]
# # Week 3 · Day 5 — Anomaly Detection for Defects
#
# **Goal:** detect defective parts when you **only have images of good parts** for training — the normal situation on a production line, where defects are rare and endlessly varied.
#
# | Method | Family | Idea |
# |---|---|---|
# | A · Convolutional autoencoder | Reconstruction | Learn to rebuild normal images; defects reconstruct badly → high pixel error |
# | B · PatchCore (from scratch) | Embedding distance | Store patch features of normal images from a frozen ImageNet CNN; a test patch far from every stored patch is anomalous |
# | C · anomalib PatchCore + PaDiM | Embedding distance (library) | Reference implementations; PaDiM fits a Gaussian per patch position |
#
# **Dataset:** [MVTec AD on Kaggle (ipythonx/mvtec-ad)](https://www.kaggle.com/datasets/ipythonx/mvtec-ad) — 15 industrial categories; `train/` contains only good images, `test/` contains good + defect types with pixel masks in `ground_truth/`. Only the chosen category folder is downloaded (`bottle` ≈ 150 MB) when possible; otherwise the full ~5 GB dataset.
#
# **Metrics:** image-level AUROC (is this part defective?), pixel-level AUROC (where is the defect?), and precision / recall / F1 at a threshold calibrated **on good images only**.
#
# > Runtime: **T4 GPU**. ~10–15 min for one category including anomalib.

# %% [markdown]
# ## 0 · Install dependencies
#
# anomalib is installed first, before torch is imported, so the kernel never runs with mixed package versions. If Colab says a restart is needed after this cell, use **Runtime → Restart session** and continue from section 1 (don't re-run the install).

# %%
import subprocess
import sys


def pip_install(*packages: str) -> None:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *packages])


RUN_ANOMALIB = True  # set False to skip the library comparison (part C)
pip_install("kagglehub==1.0.2", *(["anomalib==2.7.0"] if RUN_ANOMALIB else []))

# %% [markdown]
# ## 1 · Configuration

# %%
import json
import time
from collections import defaultdict
from pathlib import Path

import matplotlib

IN_NOTEBOOK = "ipykernel" in sys.modules
if not IN_NOTEBOOK:
    matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import f1_score, precision_recall_curve, precision_score, recall_score, roc_auc_score
from torchvision.models import Wide_ResNet50_2_Weights, wide_resnet50_2
from torchvision.models.feature_extraction import create_feature_extractor
from torchvision.transforms.functional import gaussian_blur

CATEGORY = "bottle"  # try "screw" (hard), "carpet" (texture), "transistor" (logical defects)
FULL_RUN = True
SEED = 42
IMG_SIZE = 224
AE_EPOCHS = 150 if FULL_RUN else 5
AE_BATCH = 16
AE_LR = 1e-3
CORESET_RATIO = 0.1
HOLDOUT_GOOD_FRACTION = 0.1  # good training images held out to calibrate the threshold
THRESHOLD_QUANTILE = 0.99

IN_COLAB = "google.colab" in sys.modules
try:
    HERE = Path(__file__).resolve().parent
except NameError:
    HERE = Path.cwd()
OUTPUT_DIR = (Path("/content/week3_outputs/day5") if IN_COLAB else HERE.parent / "results" / "day5") / CATEGORY
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(SEED)
np.random.seed(SEED)
print({"category": CATEGORY, "device": str(DEVICE), "full_run": FULL_RUN})


def save_fig(fig, name: str) -> None:
    path = OUTPUT_DIR / name
    fig.savefig(path, dpi=120, bbox_inches="tight")
    if IN_NOTEBOOK:
        plt.show()
    plt.close(fig)
    print(f"saved -> {path}")


# %% [markdown]
# ## 2 · Download one MVTec AD category from Kaggle

# %%
import kagglehub

KAGGLE_HANDLE = "ipythonx/mvtec-ad"


def locate_category(root: Path) -> Path:
    for cand in [root, root / CATEGORY, *root.rglob(CATEGORY)]:
        if (cand / "train" / "good").is_dir() and (cand / "test").is_dir():
            return cand
    raise FileNotFoundError(f"{CATEGORY}/train/good not found below {root}")


try:
    CATEGORY_DIR = locate_category(Path(kagglehub.dataset_download(KAGGLE_HANDLE, path=CATEGORY)))
except Exception as exc:
    print(f"Single-folder download failed ({exc}); downloading the full dataset (~5 GB)")
    CATEGORY_DIR = locate_category(Path(kagglehub.dataset_download(KAGGLE_HANDLE)))
MVTEC_ROOT = CATEGORY_DIR.parent  # anomalib expects <root>/<category>/...
print("Category dir:", CATEGORY_DIR)

train_paths = sorted((CATEGORY_DIR / "train" / "good").glob("*.png"))
test_items = []  # (path, defect_type, label, mask_path or None)
for defect_dir in sorted(p for p in (CATEGORY_DIR / "test").iterdir() if p.is_dir()):
    for p in sorted(defect_dir.glob("*.png")):
        is_good = defect_dir.name == "good"
        mask = None if is_good else CATEGORY_DIR / "ground_truth" / defect_dir.name / f"{p.stem}_mask.png"
        test_items.append((p, defect_dir.name, 0 if is_good else 1, mask))

print(f"train (good only): {len(train_paths)}")
print("test:", pd.Series([d for _, d, _, _ in test_items]).value_counts().to_dict())

# %% [markdown]
# ## 3 · Look at the data
#
# Note how subtle some defects are, and that the training set contains **zero** defect examples. This is one-class learning: the model only learns what "normal" looks like.

# %%
defect_types = sorted({d for _, d, _, _ in test_items})
fig, axes = plt.subplots(2, len(defect_types), figsize=(4 * len(defect_types), 8))
for col, dtype in enumerate(defect_types):
    path, _, label, mask_path = next(t for t in test_items if t[1] == dtype)
    axes[0, col].imshow(Image.open(path).convert("RGB"))
    axes[0, col].set_title(dtype)
    if mask_path is not None and mask_path.exists():
        axes[1, col].imshow(Image.open(mask_path), cmap="gray")
        axes[1, col].set_title("GT defect mask")
    else:
        axes[1, col].imshow(np.zeros((8, 8)), cmap="gray", vmin=0, vmax=1)
        axes[1, col].set_title("no defect")
    for ax in axes[:, col]:
        ax.axis("off")
save_fig(fig, "day5_dataset_overview.png")

# %% [markdown]
# ## 4 · Load tensors and hold out good images for threshold calibration
#
# In production you cannot tune the decision threshold on defects you have never seen. We hold out 10 % of the **good** training images and set the threshold at the 99th percentile of their anomaly scores. The F1-optimal threshold on the test set is reported separately as an optimistic upper bound.

# %%
def load_image(path: Path) -> torch.Tensor:
    img = Image.open(path).convert("RGB").resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
    return torch.from_numpy(np.asarray(img, dtype=np.float32) / 255).permute(2, 0, 1)


def load_mask(path: Path | None) -> torch.Tensor:
    if path is None or not path.exists():
        return torch.zeros(IMG_SIZE, IMG_SIZE)
    m = Image.open(path).convert("L").resize((IMG_SIZE, IMG_SIZE), Image.NEAREST)
    return torch.from_numpy((np.asarray(m) > 127).astype(np.float32))


perm = np.random.default_rng(SEED).permutation(len(train_paths))
n_hold = max(5, int(len(train_paths) * HOLDOUT_GOOD_FRACTION))
fit_x = torch.stack([load_image(train_paths[i]) for i in perm[n_hold:]])
hold_x = torch.stack([load_image(train_paths[i]) for i in perm[:n_hold]])
test_x = torch.stack([load_image(p) for p, _, _, _ in test_items])
test_masks = torch.stack([load_mask(m) for _, _, _, m in test_items])
test_labels = np.array([lab for _, _, lab, _ in test_items])
test_types = np.array([d for _, d, _, _ in test_items])
print(f"fit={len(fit_x)} holdout_good={len(hold_x)} test={len(test_x)} ({test_labels.sum()} defective)")

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406], device=DEVICE).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225], device=DEVICE).view(1, 3, 1, 1)


def smooth(maps: torch.Tensor, sigma: float = 4.0) -> torch.Tensor:
    """Gaussian-blur (B, H, W) anomaly maps; standard post-processing for both methods."""
    k = int(2 * round(3 * sigma) + 1)
    return gaussian_blur(maps.unsqueeze(1), kernel_size=[k, k], sigma=[sigma, sigma]).squeeze(1)


# %% [markdown]
# ## 5 · Evaluation helpers (shared by every method)

# %%
RESULTS = {}
MAPS = {}


def evaluate_method(name: str, test_scores: np.ndarray, holdout_scores: np.ndarray, maps: torch.Tensor | None, ms_per_img: float):
    image_auroc = roc_auc_score(test_labels, test_scores)
    pixel_auroc = roc_auc_score(test_masks.flatten().numpy().astype(int), maps.flatten().numpy()) if maps is not None else np.nan
    threshold = float(np.quantile(holdout_scores, THRESHOLD_QUANTILE))
    pred = (test_scores >= threshold).astype(int)
    p, r, t = precision_recall_curve(test_labels, test_scores)
    f1s = 2 * p * r / np.maximum(p + r, 1e-9)
    per_type = {d: float(pred[test_types == d].mean()) for d in defect_types}  # flag rate per type
    RESULTS[name] = {
        "image_AUROC": image_auroc,
        "pixel_AUROC": pixel_auroc,
        "threshold": threshold,
        "precision": precision_score(test_labels, pred, zero_division=0),
        "recall": recall_score(test_labels, pred, zero_division=0),
        "F1": f1_score(test_labels, pred, zero_division=0),
        "F1_oracle": float(f1s[:-1].max()) if len(f1s) > 1 else np.nan,
        "false_alarm_rate_on_good": per_type.get("good", np.nan),
        "flag_rate_per_type": per_type,
        "ms_per_img": ms_per_img,
    }
    if maps is not None:
        MAPS[name] = maps
    RESULTS[name]["scores"] = test_scores
    r_ = RESULTS[name]
    print(f"{name}: image AUROC={image_auroc:.3f} pixel AUROC={pixel_auroc:.3f} | "
          f"@thr={threshold:.4f}: P={r_['precision']:.3f} R={r_['recall']:.3f} F1={r_['F1']:.3f} "
          f"(oracle F1={r_['F1_oracle']:.3f}) | {ms_per_img:.1f} ms/img")


# %% [markdown]
# ## Method A · Convolutional autoencoder (reconstruction error)
#
# The encoder squeezes a 224×224×3 image into a 14×14×16 code; the decoder rebuilds it. Trained only on good parts, it should rebuild normal structure well and defects poorly. Known weakness: a capable autoencoder also generalizes to defects and reconstructs them too, which lowers its AUROC. That trade-off is the reason embedding methods took over.

# %%
def conv_block(i, o):
    return nn.Sequential(nn.Conv2d(i, o, 4, 2, 1), nn.BatchNorm2d(o), nn.LeakyReLU(0.2, inplace=True))


def deconv_block(i, o):
    return nn.Sequential(nn.ConvTranspose2d(i, o, 4, 2, 1), nn.BatchNorm2d(o), nn.ReLU(inplace=True))


class ConvAutoencoder(nn.Module):
    def __init__(self, latent_channels: int = 16):
        super().__init__()
        self.encoder = nn.Sequential(conv_block(3, 32), conv_block(32, 64), conv_block(64, 128), conv_block(128, 256),
                                     nn.Conv2d(256, latent_channels, 3, 1, 1))  # 224 -> 14x14 bottleneck
        self.decoder = nn.Sequential(nn.Conv2d(latent_channels, 256, 3, 1, 1), nn.ReLU(inplace=True),
                                     deconv_block(256, 128), deconv_block(128, 64), deconv_block(64, 32),
                                     nn.ConvTranspose2d(32, 3, 4, 2, 1), nn.Sigmoid())

    def forward(self, x):
        return self.decoder(self.encoder(x))


ae = ConvAutoencoder().to(DEVICE)
opt = torch.optim.Adam(ae.parameters(), lr=AE_LR)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=AE_EPOCHS)
ae_losses = []
t0 = time.time()
for epoch in range(1, AE_EPOCHS + 1):
    ae.train()
    order = torch.randperm(len(fit_x))
    total = 0.0
    for i in range(0, len(order), AE_BATCH):
        x = fit_x[order[i : i + AE_BATCH]].to(DEVICE)
        loss = F.mse_loss(ae(x), x)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        total += loss.item() * len(x)
    sched.step()
    ae_losses.append(total / len(fit_x))
    if epoch % max(1, AE_EPOCHS // 10) == 0 or epoch == 1:
        print(f"epoch {epoch:3d}/{AE_EPOCHS} recon MSE={ae_losses[-1]:.5f}")
print(f"Autoencoder trained in {time.time() - t0:.0f}s")


@torch.no_grad()
def ae_score(images: torch.Tensor):
    ae.eval()
    maps = []
    for i in range(0, len(images), 32):
        x = images[i : i + 32].to(DEVICE)
        maps.append(smooth(((ae(x) - x) ** 2).mean(1)).cpu())
    maps = torch.cat(maps)
    return maps, maps.flatten(1).max(1).values.numpy()


t0 = time.perf_counter()
ae_maps, ae_scores = ae_score(test_x)
ae_ms = (time.perf_counter() - t0) * 1000 / len(test_x)
_, ae_hold = ae_score(hold_x)
evaluate_method("A_autoencoder", ae_scores, ae_hold, ae_maps, ae_ms)

with torch.no_grad():
    sample = test_x[test_labels == 1][:4].to(DEVICE)
    recon = ae.eval()(sample).cpu()
fig, axes = plt.subplots(3, 4, figsize=(16, 12))
for i in range(len(sample)):
    axes[0, i].imshow(sample[i].cpu().permute(1, 2, 0))
    axes[1, i].imshow(recon[i].permute(1, 2, 0))
    axes[2, i].imshow(((recon[i] - sample[i].cpu()) ** 2).mean(0), cmap="inferno")
    for row, title in enumerate(["input (defective)", "reconstruction", "squared error"]):
        axes[row, i].set_title(title)
        axes[row, i].axis("off")
save_fig(fig, "day5_autoencoder_reconstructions.png")

# %% [markdown]
# ## Method B · PatchCore from scratch (embedding distance)
#
# 1. **Features:** a frozen ImageNet WideResNet-50 gives mid-level features from `layer2` (28×28) and `layer3` (14×14, upsampled). A 3×3 average pool adds local context. Every spatial position becomes a 1536-d **patch embedding**.
# 2. **Memory bank:** collect patch embeddings from all good training images, then keep a 10 % **greedy coreset** that covers the feature space (k-center selection).
# 3. **Scoring:** a test patch's anomaly score is the distance to its nearest neighbour in the memory bank. The image score is the maximum patch score: one bad patch makes the part bad.

# %%
backbone = wide_resnet50_2(weights=Wide_ResNet50_2_Weights.IMAGENET1K_V1)
extractor = create_feature_extractor(backbone, {"layer2": "l2", "layer3": "l3"}).to(DEVICE).eval()


@torch.no_grad()
def patch_embeddings(images: torch.Tensor, batch: int = 32) -> torch.Tensor:
    """(N,3,H,W) in [0,1] -> (N, 28, 28, 1536) patch features."""
    out = []
    for i in range(0, len(images), batch):
        x = (images[i : i + batch].to(DEVICE) - IMAGENET_MEAN) / IMAGENET_STD
        feats = extractor(x)
        l2 = F.avg_pool2d(feats["l2"], 3, 1, 1)
        l3 = F.interpolate(F.avg_pool2d(feats["l3"], 3, 1, 1), size=l2.shape[-2:], mode="bilinear", align_corners=False)
        out.append(torch.cat([l2, l3], dim=1).permute(0, 2, 3, 1))
    return torch.cat(out)


@torch.no_grad()
def greedy_coreset(features: torch.Tensor, ratio: float, proj_dim: int = 128) -> torch.Tensor:
    """k-center greedy selection on a random projection (as in the PatchCore paper)."""
    n = len(features)
    k = max(1, int(n * ratio))
    g = torch.Generator(device=features.device).manual_seed(SEED)
    proj = torch.randn(features.shape[1], proj_dim, device=features.device, generator=g) / proj_dim**0.5
    z = features @ proj
    selected = [torch.randint(n, (1,), device=features.device, generator=g)[0]]
    min_dist = ((z - z[selected[0]]) ** 2).sum(1)
    for _ in range(k - 1):
        idx = torch.argmax(min_dist)
        selected.append(idx)
        min_dist = torch.minimum(min_dist, ((z - z[idx]) ** 2).sum(1))
    return torch.stack(selected)


t0 = time.time()
train_feats = patch_embeddings(fit_x)
grid = train_feats.shape[1:3]
bank_full = train_feats.reshape(-1, train_feats.shape[-1])
memory_bank = bank_full[greedy_coreset(bank_full, CORESET_RATIO)]
del train_feats
print(f"Patch features: {tuple(bank_full.shape)} -> coreset memory bank {tuple(memory_bank.shape)} "
      f"({time.time() - t0:.0f}s)")
del bank_full
torch.cuda.empty_cache() if DEVICE.type == "cuda" else None


@torch.no_grad()
def patchcore_score(images: torch.Tensor):
    maps = []
    for i in range(0, len(images), 8):
        feats = patch_embeddings(images[i : i + 8])
        b = feats.shape[0]
        d = torch.cdist(feats.reshape(-1, feats.shape[-1]), memory_bank).min(1).values  # nearest-neighbour distance
        d = d.reshape(b, 1, *grid)
        maps.append(smooth(F.interpolate(d, size=(IMG_SIZE, IMG_SIZE), mode="bilinear", align_corners=False)[:, 0]).cpu())
    maps = torch.cat(maps)
    return maps, maps.flatten(1).max(1).values.numpy()


t0 = time.perf_counter()
pc_maps, pc_scores = patchcore_score(test_x)
pc_ms = (time.perf_counter() - t0) * 1000 / len(test_x)
_, pc_hold = patchcore_score(hold_x)
evaluate_method("B_patchcore_scratch", pc_scores, pc_hold, pc_maps, pc_ms)

# %% [markdown]
# ## Method C · anomalib PatchCore and PaDiM
#
# The library versions add details we skipped (feature-dimension reduction, score re-weighting, normalization, adaptive thresholds). If anomalib's API changes in a future release this cell reports the error and the rest of the notebook still runs.

# %%
ANOMALIB_ROWS = {}
if RUN_ANOMALIB:
    try:
        from anomalib.data import MVTecAD
        from anomalib.engine import Engine
        from anomalib.models import Padim, Patchcore

        for model_name, model_cls in [("C_anomalib_patchcore", Patchcore), ("C_anomalib_padim", Padim)]:
            datamodule = MVTecAD(root=str(MVTEC_ROOT), category=CATEGORY, train_batch_size=32, eval_batch_size=32, num_workers=2)
            engine = Engine(max_epochs=1, default_root_dir=str(OUTPUT_DIR / "anomalib"))
            model = model_cls()
            t0 = time.time()
            engine.fit(datamodule=datamodule, model=model)
            test_out = engine.test(datamodule=datamodule, model=model)[0]
            elapsed = time.time() - t0
            print(model_name, {k: round(float(v), 4) for k, v in test_out.items()})
            pick = lambda key: next((float(v) for k, v in test_out.items() if k.lower() == key), np.nan)
            ANOMALIB_ROWS[model_name] = {"image_AUROC": pick("image_auroc"), "pixel_AUROC": pick("pixel_auroc"),
                                         "F1": pick("image_f1score"), "fit_plus_test_s": elapsed}
    except Exception as exc:
        print(f"anomalib run failed: {type(exc).__name__}: {exc}")
        ANOMALIB_ROWS["C_anomalib"] = {"error": f"{type(exc).__name__}: {exc}"}

# %% [markdown]
# ## 6 · Compare all methods

# %%
table_rows = []
for name, r in RESULTS.items():
    table_rows.append({"method": name, "image_AUROC": r["image_AUROC"], "pixel_AUROC": r["pixel_AUROC"],
                       "precision@thr": r["precision"], "recall@thr": r["recall"], "F1@thr": r["F1"],
                       "F1_oracle": r["F1_oracle"], "false_alarms_on_good": r["false_alarm_rate_on_good"],
                       "ms_per_img": r["ms_per_img"]})
for name, r in ANOMALIB_ROWS.items():
    table_rows.append({"method": name, **{k: v for k, v in r.items() if k != "error"}})
comparison = pd.DataFrame(table_rows).set_index("method")
print(comparison.round(3).to_string())
comparison.to_csv(OUTPUT_DIR / "day5_comparison.csv")

print("\nShare of test images flagged as defective, per type (good = false-alarm rate):")
print(pd.DataFrame({n: r["flag_rate_per_type"] for n, r in RESULTS.items()}).round(3).to_string())

fig, axes = plt.subplots(1, len(RESULTS), figsize=(7 * len(RESULTS), 4.5))
for ax, (name, r) in zip(np.atleast_1d(axes), RESULTS.items()):
    scores = r["scores"]
    bins = np.linspace(scores.min(), scores.max(), 40)
    ax.hist(scores[test_labels == 0], bins=bins, alpha=0.6, label="good", color="tab:green")
    ax.hist(scores[test_labels == 1], bins=bins, alpha=0.6, label="defective", color="tab:red")
    ax.axvline(r["threshold"], color="black", ls="--", label="calibrated threshold")
    ax.set_title(f"{name} (AUROC {r['image_AUROC']:.3f})")
    ax.set_xlabel("image anomaly score")
    ax.legend()
save_fig(fig, "day5_score_distributions.png")

# %% [markdown]
# ## 7 · Flag defective samples (the deliverable demo)
#
# The best method by image AUROC flags each test part as **DEFECT** or **OK** at the calibrated threshold and shows *where* it thinks the defect is. Red titles mark mistakes.

# %%
best = max(RESULTS, key=lambda n: RESULTS[n]["image_AUROC"])
scores, thr, maps = RESULTS[best]["scores"], RESULTS[best]["threshold"], MAPS[best]
flags = scores >= thr
vmax = float(np.quantile(maps.numpy(), 0.999))

rng = np.random.default_rng(SEED)
picks = []
for dtype in defect_types:
    idx = np.where(test_types == dtype)[0]
    picks.extend(rng.choice(idx, size=min(2, len(idx)), replace=False).tolist())
picks = picks[:12]
missed = np.where((test_labels == 1) & ~flags)[0]
false_alarms = np.where((test_labels == 0) & flags)[0]

fig, axes = plt.subplots(3, len(picks), figsize=(2.6 * len(picks), 8.5))
for col, i in enumerate(picks):
    verdict = "DEFECT" if flags[i] else "OK"
    correct = flags[i] == bool(test_labels[i])
    axes[0, col].imshow(test_x[i].permute(1, 2, 0))
    axes[0, col].set_title(f"{test_types[i]}\n{verdict} ({scores[i]:.2f})", fontsize=8, color="black" if correct else "red")
    axes[1, col].imshow(test_masks[i], cmap="gray", vmin=0, vmax=1)
    axes[1, col].set_title("GT mask", fontsize=8)
    axes[2, col].imshow(test_x[i].permute(1, 2, 0))
    axes[2, col].imshow(maps[i], cmap="jet", alpha=0.5, vmin=0, vmax=vmax)
    axes[2, col].set_title("anomaly map", fontsize=8)
    for ax in axes[:, col]:
        ax.axis("off")
fig.suptitle(f"{CATEGORY}: {best} at threshold {thr:.3f} — missed defects: {len(missed)}, false alarms: {len(false_alarms)}")
save_fig(fig, "day5_flagged_samples.png")

flag_report = pd.DataFrame({"file": [str(p.relative_to(CATEGORY_DIR)) for p, _, _, _ in test_items], "type": test_types,
                            "label": test_labels, "score": scores, "flagged": flags})
flag_report.to_csv(OUTPUT_DIR / "day5_flag_report.csv", index=False)
print(flag_report.sort_values("score", ascending=False).head(10).to_string(index=False))

# %% [markdown]
# ## 8 · Save results

# %%
summary = {
    "dataset": f"{KAGGLE_HANDLE} (Kaggle) / {CATEGORY}",
    "train_good": len(fit_x),
    "holdout_good": len(hold_x),
    "test": {"total": len(test_items), "defective": int(test_labels.sum())},
    "threshold_rule": f"{THRESHOLD_QUANTILE:.0%} quantile of held-out good scores",
    "best_method": best,
    "methods": {n: {k: v for k, v in r.items() if k != "scores"} for n, r in RESULTS.items()},
    "anomalib": ANOMALIB_ROWS,
    "full_run": FULL_RUN,
}
with open(OUTPUT_DIR / "day5_summary.json", "w") as f:
    json.dump(summary, f, indent=2, default=float)
print(json.dumps(summary, indent=2, default=float))

# %% [markdown]
# ## Takeaways (fill in with your numbers)
#
# - Image AUROC: autoencoder ___ vs PatchCore (ours) ___ vs anomalib PatchCore ___ / PaDiM ___.
# - Why the gap: pretrained features already separate "scratch" from "glass", while an autoencoder has to learn everything from ~190 images and tends to reconstruct defects too.
# - The calibrated threshold gives recall ___ with ___ false alarms on good parts. In a factory, which is costlier: a missed defect or a false reject? Move `THRESHOLD_QUANTILE` accordingly.
# - Re-run with `CATEGORY = "screw"` or `"carpet"`: textures behave differently from objects.
#
# Week 3 is done: detection (Day 1–2), segmentation (Day 3), tracking (Day 4), anomaly detection (Day 5).
