"""
Day 3: Custom Dataset Pipeline — Dataset, DataLoader & Transforms
==================================================================

Objectives:
- Pick a custom Kaggle image-classification dataset (Intel Image Classification)
- Build the full data pipeline: Dataset, DataLoader, transforms
- Understand ImageFolder vs writing a custom Dataset class
- Visualise sample batches and inspect class distributions
- Compute dataset-specific mean/std for proper normalisation
- Plug the custom pipeline into Day 2's transfer-learning code and train

Materials:
- PyTorch — Datasets & DataLoaders:
    https://pytorch.org/tutorials/beginner/basics/data_tutorial.html
- Kaggle Datasets — image classification:
    https://www.kaggle.com/datasets?tags=13302-Image+Classification

Dataset:
  Intel Image Classification (default) — ~25k images, 6 classes:
  buildings, forest, glacier, mountain, sea, street
  https://www.kaggle.com/datasets/puneet6060/intel-image-classification

  If running locally, download from Kaggle and extract to:
      week2_classification/data/intel-image-classification/
          seg_train/seg_train/<class_folders>/
          seg_test/seg_test/<class_folders>/
          seg_pred/seg_pred/<images>

  If running on Colab, the dataset is expected on Google Drive at:
      /content/drive/MyDrive/datasets/intel-image-classification/

Run:
    # Quick exploration (no training)
    python day3_custom_dataset.py --explore-only

    # Full: explore + train feature-extraction + fine-tuning
    python day3_custom_dataset.py --epochs 10 --no-wandb

    # Use a different backbone
    python day3_custom_dataset.py --backbone efficientnet_b0 --epochs 10
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split, WeightedRandomSampler
import torchvision
import torchvision.transforms as T
import torchvision.models as models
from torchvision.models import (
    ResNet18_Weights, ResNet50_Weights,
    EfficientNet_B0_Weights, EfficientNet_B3_Weights,
)
from torchvision.datasets import ImageFolder
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from collections import Counter
import time
import argparse
import json
import os

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("⚠️  wandb not installed — run: pip install wandb")

# ── Directories ───────────────────────────────────────────────────────────────
BASE_DIR    = Path(__file__).parent.parent
RESULTS_DIR = BASE_DIR / "results"
MODEL_DIR   = BASE_DIR / "models"
DATA_DIR    = BASE_DIR / "data"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Reproducibility
torch.manual_seed(42)
np.random.seed(42)

# ImageNet stats — used when backbone was pretrained on ImageNet
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)


# =============================================================================
# 1. DATASET DISCOVERY — FIND AND VALIDATE THE DATASET
# =============================================================================

def find_dataset(data_dir: str = None) -> tuple[Path, Path, list[str]]:
    """
    Locate the Intel Image Classification dataset.
    Returns (train_dir, test_dir, class_names).

    Expected structure:
        <root>/seg_train/seg_train/<class_folders>/
        <root>/seg_test/seg_test/<class_folders>/
    """
    if data_dir:
        root = Path(data_dir)
    else:
        root = DATA_DIR / "intel-image-classification"

    # Handle the nested directory structure of this Kaggle dataset
    train_dir = root / "seg_train" / "seg_train"
    test_dir  = root / "seg_test" / "seg_test"

    # Fallback: try non-nested structure
    if not train_dir.exists():
        train_dir = root / "seg_train"
    if not test_dir.exists():
        test_dir = root / "seg_test"

    if not train_dir.exists():
        raise FileNotFoundError(
            f"Training data not found at {train_dir}\n"
            f"Download the Intel Image Classification dataset from Kaggle:\n"
            f"  https://www.kaggle.com/datasets/puneet6060/intel-image-classification\n"
            f"Extract to: {root}"
        )

    class_names = sorted([d.name for d in train_dir.iterdir() if d.is_dir()])
    print(f"✅ Dataset found at: {root}")
    print(f"   Train dir: {train_dir}")
    print(f"   Test dir:  {test_dir}")
    print(f"   Classes ({len(class_names)}): {class_names}")

    return train_dir, test_dir, class_names


# =============================================================================
# 2. DATASET ANALYSIS — UNDERSTAND YOUR DATA BEFORE TRAINING
# =============================================================================

def analyse_dataset(train_dir: Path, test_dir: Path, class_names: list[str]):
    """
    Analyse the dataset: count images per class, check for imbalance,
    sample image sizes, and plot the class distribution.
    """
    print("\n" + "=" * 60)
    print("DATASET ANALYSIS")
    print("=" * 60)

    # Count images per class
    train_counts = {}
    test_counts = {}
    sample_sizes = []

    for cls in class_names:
        train_cls_dir = train_dir / cls
        test_cls_dir  = test_dir / cls

        train_imgs = list(train_cls_dir.glob("*")) if train_cls_dir.exists() else []
        test_imgs  = list(test_cls_dir.glob("*")) if test_cls_dir.exists() else []

        train_counts[cls] = len(train_imgs)
        test_counts[cls]  = len(test_imgs)

        # Sample a few image sizes
        for img_path in train_imgs[:5]:
            try:
                img = Image.open(img_path)
                sample_sizes.append(img.size)
            except Exception:
                pass

    total_train = sum(train_counts.values())
    total_test  = sum(test_counts.values())

    print(f"\n  {'Class':<15} {'Train':>8} {'Test':>8} {'% of Train':>12}")
    print("  " + "-" * 45)
    for cls in class_names:
        pct = 100 * train_counts[cls] / total_train
        print(f"  {cls:<15} {train_counts[cls]:>8} {test_counts[cls]:>8} {pct:>11.1f}%")
    print("  " + "-" * 45)
    print(f"  {'TOTAL':<15} {total_train:>8} {total_test:>8}")

    # Check for imbalance
    counts = list(train_counts.values())
    ratio = max(counts) / min(counts) if min(counts) > 0 else float('inf')
    print(f"\n  Class imbalance ratio (max/min): {ratio:.2f}")
    if ratio > 2.0:
        print("  ⚠️  Significant imbalance detected — consider WeightedRandomSampler")
    else:
        print("  ✅ Classes are reasonably balanced")

    # Image sizes
    if sample_sizes:
        widths  = [s[0] for s in sample_sizes]
        heights = [s[1] for s in sample_sizes]
        print(f"\n  Sample image sizes:")
        print(f"    Width  range: {min(widths)} – {max(widths)}")
        print(f"    Height range: {min(heights)} – {max(heights)}")

    # Plot class distribution
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    colors = plt.cm.Set3(np.linspace(0, 1, len(class_names)))

    axes[0].barh(class_names, [train_counts[c] for c in class_names], color=colors)
    axes[0].set_xlabel("Number of Images")
    axes[0].set_title("Training Set Distribution")
    for i, cls in enumerate(class_names):
        axes[0].text(train_counts[cls] + 10, i, str(train_counts[cls]),
                     va='center', fontsize=9)

    axes[1].barh(class_names, [test_counts[c] for c in class_names], color=colors)
    axes[1].set_xlabel("Number of Images")
    axes[1].set_title("Test Set Distribution")
    for i, cls in enumerate(class_names):
        axes[1].text(test_counts[cls] + 5, i, str(test_counts[cls]),
                     va='center', fontsize=9)

    plt.tight_layout()
    save_path = RESULTS_DIR / "day3_class_distribution.png"
    plt.savefig(save_path, dpi=120)
    plt.close()
    print(f"\n  Distribution plot saved → {save_path}")

    return train_counts, test_counts


# =============================================================================
# 3. COMPUTE DATASET MEAN & STD
# =============================================================================

def compute_mean_std(dataset_dir: Path, img_size: int = 224,
                     max_samples: int = 2000) -> tuple[tuple, tuple]:
    """
    Compute per-channel mean and std of the dataset.

    For transfer learning we typically use ImageNet stats, but computing
    your dataset's own stats is good practice and can help when the domain
    is very different from ImageNet (e.g., medical, satellite).
    """
    print("\n" + "=" * 60)
    print("COMPUTING DATASET MEAN & STD")
    print("=" * 60)

    transform = T.Compose([
        T.Resize((img_size, img_size)),
        T.ToTensor(),  # [0, 1] range, (C, H, W)
    ])

    dataset = ImageFolder(str(dataset_dir), transform=transform)

    # Sample a subset for speed
    n = min(max_samples, len(dataset))
    indices = np.random.choice(len(dataset), n, replace=False)

    mean = torch.zeros(3)
    std  = torch.zeros(3)

    for idx in indices:
        img, _ = dataset[idx]
        for c in range(3):
            mean[c] += img[c].mean()
            std[c]  += img[c].std()

    mean /= n
    std  /= n

    print(f"  Samples used: {n}")
    print(f"  Mean: ({mean[0]:.4f}, {mean[1]:.4f}, {mean[2]:.4f})")
    print(f"  Std:  ({std[0]:.4f}, {std[1]:.4f}, {std[2]:.4f})")
    print(f"\n  For comparison — ImageNet:")
    print(f"    Mean: (0.4850, 0.4560, 0.4060)")
    print(f"    Std:  (0.2290, 0.2240, 0.2250)")
    print(f"\n  → Using ImageNet stats for transfer learning (pretrained backbone)")

    return tuple(mean.tolist()), tuple(std.tolist())


# =============================================================================
# 4. CUSTOM DATASET CLASS (educational — shows what ImageFolder does internally)
# =============================================================================

class CustomImageDataset(Dataset):
    """
    A custom Dataset that loads images from a directory structure:
        root/class1/img001.jpg
        root/class2/img002.jpg
        ...

    This is educational — in practice, use torchvision.datasets.ImageFolder.
    But knowing how to write one is essential for:
      - Non-standard directory layouts
      - CSV-based label files
      - Multi-label classification
      - On-the-fly preprocessing
    """

    def __init__(self, root_dir: str, transform=None):
        self.root_dir = Path(root_dir)
        self.transform = transform

        # Discover classes from subdirectory names
        self.classes = sorted([d.name for d in self.root_dir.iterdir() if d.is_dir()])
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}

        # Build list of (image_path, label) pairs
        self.samples = []
        for cls in self.classes:
            cls_dir = self.root_dir / cls
            for img_path in sorted(cls_dir.glob("*")):
                if img_path.suffix.lower() in {'.jpg', '.jpeg', '.png', '.bmp', '.gif'}:
                    self.samples.append((str(img_path), self.class_to_idx[cls]))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        img_path, label = self.samples[idx]

        # Load image and convert to RGB (handles grayscale, RGBA, etc.)
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label


# =============================================================================
# 5. DATA PIPELINE — TRANSFORMS & DATALOADERS
# =============================================================================

def get_transforms(img_size: int = 224) -> tuple[T.Compose, T.Compose]:
    """
    Create training and evaluation transforms.

    Training:  augmentation (random crop, flip) + normalisation
    Eval:      deterministic (resize + center crop) + normalisation

    ⚠️ IMPORTANT: We use ImageNet normalisation stats because our backbone
    was pretrained on ImageNet. Using dataset-specific stats would mismatch
    the feature distributions the backbone expects.
    """
    transform_train = T.Compose([
        T.Resize(int(img_size * 1.15)),       # Slight over-resize for cropping margin
        T.RandomCrop(img_size),                # Random spatial crop (augmentation)
        T.RandomHorizontalFlip(p=0.5),         # 50% horizontal flip (augmentation)
        T.ColorJitter(                         # Slight colour variation
            brightness=0.1, contrast=0.1,
            saturation=0.1, hue=0.05,
        ),
        T.ToTensor(),                          # (H,W,C) uint8 → (C,H,W) float [0,1]
        T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    transform_eval = T.Compose([
        T.Resize(int(img_size * 1.15)),
        T.CenterCrop(img_size),                # Deterministic center crop
        T.ToTensor(),
        T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    return transform_train, transform_eval


def get_data_loaders(train_dir: Path, test_dir: Path,
                     img_size: int = 224, batch_size: int = 32,
                     val_split: float = 0.15, num_workers: int = 2,
                     use_custom_dataset: bool = False,
                     ) -> tuple[DataLoader, DataLoader, DataLoader, list[str]]:
    """
    Build train/val/test DataLoaders.

    Args:
        train_dir:   Path to training images (class-per-folder).
        test_dir:    Path to test images (class-per-folder).
        img_size:    Target image size (224 for most backbones).
        batch_size:  Batch size.
        val_split:   Fraction of training data to use as validation.
        num_workers: DataLoader workers.
        use_custom_dataset: If True, use our CustomImageDataset class
                            (educational). Otherwise use ImageFolder.
    """
    transform_train, transform_eval = get_transforms(img_size)

    # ── Load datasets ─────────────────────────────────────────────────────
    DatasetClass = CustomImageDataset if use_custom_dataset else ImageFolder

    full_train = DatasetClass(str(train_dir), transform=transform_train)
    test_set   = DatasetClass(str(test_dir),  transform=transform_eval)

    class_names = full_train.classes

    # ── Train / Val split ─────────────────────────────────────────────────
    val_size   = int(len(full_train) * val_split)
    train_size = len(full_train) - val_size
    train_set, val_set = random_split(
        full_train, [train_size, val_size],
        generator=torch.Generator().manual_seed(42),
    )

    # ── DataLoaders ───────────────────────────────────────────────────────
    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True,
    )
    val_loader = DataLoader(
        val_set, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )
    test_loader = DataLoader(
        test_set, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )

    print(f"\n  Data pipeline ready:")
    print(f"    Dataset class: {DatasetClass.__name__}")
    print(f"    Splits — Train: {train_size:,}  Val: {val_size:,}  Test: {len(test_set):,}")
    print(f"    Classes ({len(class_names)}): {class_names}")
    print(f"    Batch size: {batch_size}  |  Image size: {img_size}px")
    print(f"    Batches per epoch — Train: {len(train_loader)}  Val: {len(val_loader)}  Test: {len(test_loader)}")

    return train_loader, val_loader, test_loader, class_names


# =============================================================================
# 6. VISUALISE SAMPLE BATCHES — ALWAYS SANITY-CHECK!
# =============================================================================

def visualise_batch(data_loader: DataLoader, class_names: list[str],
                    title: str = "Sample Batch", n_images: int = 16):
    """
    Display a grid of images from a DataLoader to sanity-check the pipeline.
    Un-normalises images for proper display.
    """
    print(f"\n  Visualising: {title}")

    imgs, labels = next(iter(data_loader))

    # Un-normalise for display
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std  = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    imgs_display = imgs[:n_images] * std + mean
    imgs_display = imgs_display.clamp(0, 1)

    n = min(n_images, len(imgs_display))
    cols = min(8, n)
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2, rows * 2.3))
    if rows == 1:
        axes = [axes] if cols == 1 else axes
    axes_flat = np.array(axes).flat

    for i, ax in enumerate(axes_flat):
        if i < n:
            img = imgs_display[i].permute(1, 2, 0).numpy()
            ax.imshow(np.clip(img, 0, 1))
            ax.set_title(class_names[labels[i]], fontsize=9)
        ax.axis('off')

    plt.suptitle(title, fontsize=13, fontweight='bold')
    plt.tight_layout()
    save_path = RESULTS_DIR / f"day3_{title.lower().replace(' ', '_')}.png"
    plt.savefig(save_path, dpi=120)
    plt.close()
    print(f"  Saved → {save_path}")


# =============================================================================
# 7. TRAINING PIPELINE (reuses Day 2 patterns)
# =============================================================================

SUPPORTED_BACKBONES = {
    "resnet18":       (models.resnet18,       ResNet18_Weights.DEFAULT),
    "resnet50":       (models.resnet50,       ResNet50_Weights.DEFAULT),
    "efficientnet_b0":(models.efficientnet_b0, EfficientNet_B0_Weights.DEFAULT),
    "efficientnet_b3":(models.efficientnet_b3, EfficientNet_B3_Weights.DEFAULT),
}


def build_model(backbone_name: str, num_classes: int, freeze_backbone: bool) -> nn.Module:
    """Load pretrained backbone, replace head, optionally freeze."""
    builder, weights = SUPPORTED_BACKBONES[backbone_name]
    model = builder(weights=weights)

    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False
        print(f"  ✓ Backbone frozen — only training the new head")
    else:
        print(f"  ✓ All layers unfrozen — full fine-tuning with low LR")

    # Replace classifier head
    if "resnet" in backbone_name:
        in_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(in_features, num_classes),
        )
        print(f"  ✓ Head replaced: Linear({in_features} → {num_classes})")
    elif "efficientnet" in backbone_name:
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(in_features, num_classes),
        )
        print(f"  ✓ Head replaced: Linear({in_features} → {num_classes})")

    # Re-enable gradients on the new head
    head = model.fc if "resnet" in backbone_name else model.classifier[-1]
    for param in head.parameters():
        param.requires_grad = True

    return model.to(DEVICE)


def train_one_epoch(model, loader, criterion, optimizer) -> tuple[float, float]:
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        out = model(imgs)
        loss = criterion(out, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * imgs.size(0)
        _, pred = out.max(1)
        total += labels.size(0)
        correct += pred.eq(labels).sum().item()
    return total_loss / total, 100.0 * correct / total


@torch.no_grad()
def evaluate(model, loader, criterion) -> tuple[float, float, np.ndarray, np.ndarray]:
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []
    for imgs, labels in loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        out = model(imgs)
        loss = criterion(out, labels)
        total_loss += loss.item() * imgs.size(0)
        _, pred = out.max(1)
        total += labels.size(0)
        correct += pred.eq(labels).sum().item()
        all_preds.extend(pred.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    return (total_loss / total, 100.0 * correct / total,
            np.array(all_preds), np.array(all_labels))


def run_training(backbone_name: str, num_classes: int, freeze_backbone: bool,
                 train_loader: DataLoader, val_loader: DataLoader,
                 test_loader: DataLoader, class_names: list[str],
                 epochs: int, lr_head: float, lr_backbone: float,
                 weight_decay: float, wandb_run=None) -> dict:
    """Run full training loop — feature extraction or fine-tuning."""
    mode = "feature_extraction" if freeze_backbone else "full_finetune"
    label = "FROZEN backbone" if freeze_backbone else "UNFROZEN backbone"

    print(f"\n{'─' * 60}")
    print(f"  Training: {backbone_name.upper()}  [{label}]")
    print(f"  Dataset: Custom ({num_classes} classes)  |  Epochs: {epochs}")
    print(f"  LR head: {lr_head}  |  LR backbone: {lr_backbone}")
    print(f"{'─' * 60}")

    model = build_model(backbone_name, num_classes, freeze_backbone)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_p   = sum(p.numel() for p in model.parameters())
    print(f"  Params: {trainable:,} trainable / {total_p:,} total ({100*trainable/total_p:.1f}%)")

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    # Separate parameter groups for two-LR strategy
    if freeze_backbone:
        head_params = [p for p in model.parameters() if p.requires_grad]
        optimizer = optim.AdamW(head_params, lr=lr_head, weight_decay=weight_decay)
    else:
        if "resnet" in backbone_name:
            head_module = model.fc
        else:
            head_module = model.classifier[-1]

        head_param_ids = {id(p) for p in head_module.parameters()}
        backbone_params = [p for p in model.parameters()
                           if id(p) not in head_param_ids]
        head_params     = list(head_module.parameters())

        optimizer = optim.AdamW([
            {"params": backbone_params, "lr": lr_backbone},
            {"params": head_params,     "lr": lr_head},
        ], weight_decay=weight_decay)

    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_val_acc = 0.0

    t_start = time.time()
    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_acc, _, _ = evaluate(model, val_loader, criterion)
        scheduler.step()

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        lr_now = optimizer.param_groups[0]["lr"]
        print(f"  Ep {epoch:2d}/{epochs}  LR={lr_now:.2e}  "
              f"Train {train_loss:.3f}/{train_acc:.1f}%  "
              f"Val {val_loss:.3f}/{val_acc:.1f}%")

        if wandb_run:
            wandb_run.log({
                f"{mode}/epoch": epoch,
                f"{mode}/train_loss": train_loss, f"{mode}/train_acc": train_acc,
                f"{mode}/val_loss": val_loss, f"{mode}/val_acc": val_acc,
                f"{mode}/lr": lr_now,
            })

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), MODEL_DIR / f"week2_day3_{backbone_name}_{mode}.pt")

    elapsed = time.time() - t_start
    print(f"\n  Best val accuracy: {best_val_acc:.1f}%  (training took {elapsed:.0f}s)")

    # Final test evaluation
    model.load_state_dict(torch.load(MODEL_DIR / f"week2_day3_{backbone_name}_{mode}.pt",
                                     weights_only=True))
    test_loss, test_acc, test_preds, test_labels = evaluate(model, test_loader, criterion)
    print(f"  Test accuracy: {test_acc:.1f}%")

    if wandb_run:
        wandb_run.log({f"{mode}/test_acc": test_acc, f"{mode}/test_loss": test_loss})

    # Per-class accuracy
    print(f"\n  Per-class accuracy:")
    for i, cls in enumerate(class_names):
        mask = test_labels == i
        if mask.sum() > 0:
            cls_acc = 100.0 * (test_preds[mask] == i).sum() / mask.sum()
            print(f"    {cls:<15} {cls_acc:.1f}%  ({mask.sum()} samples)")

    history["best_val_acc"] = best_val_acc
    history["test_acc"]     = test_acc
    history["mode"]         = mode
    history["label"]        = label
    return history


# =============================================================================
# 8. COMPARISON PLOT
# =============================================================================

def plot_comparison(feat_history: dict, finetune_history: dict,
                    backbone_name: str, dataset_name: str = "Custom"):
    """Side-by-side training curves for both transfer learning modes."""
    epochs = range(1, len(feat_history["train_acc"]) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f"Day 3 — {backbone_name.upper()} on {dataset_name}", fontsize=14)

    colors = {"feat_train": "#4C72B0", "feat_val": "#55A868",
              "ft_train":   "#C44E52", "ft_val":   "#DD8452"}

    for i, (metric, ylabel) in enumerate([("acc", "Accuracy (%)"), ("loss", "Loss")]):
        ax = axes[i]
        ax.plot(epochs, feat_history[f"train_{metric}"],
                color=colors["feat_train"], linestyle="--", label="FeatExt Train")
        ax.plot(epochs, feat_history[f"val_{metric}"],
                color=colors["feat_val"], label="FeatExt Val")
        ax.plot(epochs, finetune_history[f"train_{metric}"],
                color=colors["ft_train"], linestyle="--", label="FineTune Train")
        ax.plot(epochs, finetune_history[f"val_{metric}"],
                color=colors["ft_val"], label="FineTune Val")
        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylabel)
        ax.set_title(f"Training {ylabel}")
        ax.legend()
        ax.grid(alpha=0.3)

    plt.tight_layout()
    save_path = RESULTS_DIR / f"day3_{backbone_name}_custom_comparison.png"
    plt.savefig(save_path, dpi=120)
    plt.close()
    print(f"\n  Comparison plot saved → {save_path}")


# =============================================================================
# MAIN
# =============================================================================

def main(args):
    print("\n" + "█" * 60)
    print("  Week 2 · Day 3 — Custom Dataset Pipeline")
    print("█" * 60)
    print(f"  Device: {DEVICE}")

    # ── 1. Find and validate dataset ──────────────────────────────────────
    train_dir, test_dir, class_names = find_dataset(args.data_dir)
    num_classes = len(class_names)

    # ── 2. Analyse the dataset ────────────────────────────────────────────
    train_counts, test_counts = analyse_dataset(train_dir, test_dir, class_names)

    # ── 3. Compute dataset mean/std ───────────────────────────────────────
    compute_mean_std(train_dir, img_size=args.img_size)

    # ── 4. Build data pipeline ────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("BUILDING DATA PIPELINE")
    print("=" * 60)

    img_size = 300 if "b3" in args.backbone else args.img_size

    train_loader, val_loader, test_loader, class_names = get_data_loaders(
        train_dir, test_dir,
        img_size=img_size,
        batch_size=args.batch_size,
        val_split=args.val_split,
        num_workers=args.num_workers,
        use_custom_dataset=args.use_custom_dataset,
    )

    # ── 5. Visualise sample batches ───────────────────────────────────────
    visualise_batch(train_loader, class_names, title="Training Batch (augmented)")
    visualise_batch(val_loader, class_names, title="Validation Batch (no augment)")

    if args.explore_only:
        print("\n✅ Exploration complete. Use --epochs N to train.")
        return

    # ── 6. Training ───────────────────────────────────────────────────────
    wandb_run = None
    if WANDB_AVAILABLE and not args.no_wandb:
        wandb_run = wandb.init(
            project="cv-week2-custom-dataset",
            name=f"day3-{args.backbone}",
            config=vars(args),
        )

    # Strategy A: Feature Extraction
    feat_history = run_training(
        backbone_name=args.backbone, num_classes=num_classes,
        freeze_backbone=True,
        train_loader=train_loader, val_loader=val_loader,
        test_loader=test_loader, class_names=class_names,
        epochs=args.epochs, lr_head=args.lr_head,
        lr_backbone=args.lr_backbone,
        weight_decay=args.weight_decay, wandb_run=wandb_run,
    )

    # Strategy B: Full Fine-Tuning
    finetune_history = run_training(
        backbone_name=args.backbone, num_classes=num_classes,
        freeze_backbone=False,
        train_loader=train_loader, val_loader=val_loader,
        test_loader=test_loader, class_names=class_names,
        epochs=args.epochs, lr_head=args.lr_head,
        lr_backbone=args.lr_backbone,
        weight_decay=args.weight_decay, wandb_run=wandb_run,
    )

    # ── 7. Comparison ─────────────────────────────────────────────────────
    plot_comparison(feat_history, finetune_history, args.backbone)

    print("\n" + "=" * 60)
    print("RESULTS SUMMARY — CUSTOM DATASET")
    print("=" * 60)
    print(f"  {'Strategy':<25} {'Val Acc':>10} {'Test Acc':>10}")
    print("  " + "-" * 47)
    print(f"  {'Feature Extraction':<25} {feat_history['best_val_acc']:>9.1f}%"
          f" {feat_history['test_acc']:>9.1f}%")
    print(f"  {'Full Fine-Tuning':<25} {finetune_history['best_val_acc']:>9.1f}%"
          f" {finetune_history['test_acc']:>9.1f}%")

    winner = ("Full Fine-Tuning" if finetune_history["test_acc"] > feat_history["test_acc"]
              else "Feature Extraction")
    print(f"\n  Winner on test set: ✅ {winner}")

    # Save results
    results = {
        "dataset": "Intel Image Classification" if not args.data_dir else args.data_dir,
        "backbone": args.backbone,
        "epochs": args.epochs,
        "num_classes": num_classes,
        "class_names": class_names,
        "feature_extraction": {
            "best_val_acc": feat_history["best_val_acc"],
            "test_acc": feat_history["test_acc"],
        },
        "full_finetune": {
            "best_val_acc": finetune_history["best_val_acc"],
            "test_acc": finetune_history["test_acc"],
        },
    }
    results_path = RESULTS_DIR / f"day3_{args.backbone}_custom_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results JSON saved → {results_path}")

    if wandb_run:
        wandb_run.summary.update(results)
        wandb_run.finish()

    print(f"\n✅ Day 3 complete.")
    print("   Next: Day 4 — Data augmentation ablation with Albumentations")


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Week 2 Day 3 — Custom Dataset Pipeline")
    parser.add_argument("--data-dir", type=str, default=None,
                        help="Path to dataset root (default: data/intel-image-classification)")
    parser.add_argument("--backbone", type=str, default="resnet18",
                        choices=list(SUPPORTED_BACKBONES.keys()),
                        help="Backbone to use")
    parser.add_argument("--epochs", type=int, default=10,
                        help="Training epochs per strategy")
    parser.add_argument("--batch-size", type=int, default=32,
                        help="Batch size")
    parser.add_argument("--img-size", type=int, default=224,
                        help="Image size (pixels)")
    parser.add_argument("--lr-head", type=float, default=1e-3,
                        help="Learning rate for classifier head")
    parser.add_argument("--lr-backbone", type=float, default=1e-5,
                        help="Learning rate for backbone (fine-tuning only)")
    parser.add_argument("--weight-decay", type=float, default=1e-4,
                        help="AdamW weight decay")
    parser.add_argument("--val-split", type=float, default=0.15,
                        help="Fraction of training data for validation")
    parser.add_argument("--num-workers", type=int, default=2,
                        help="DataLoader workers")
    parser.add_argument("--use-custom-dataset", action="store_true",
                        help="Use CustomImageDataset instead of ImageFolder")
    parser.add_argument("--explore-only", action="store_true",
                        help="Only run dataset exploration, no training")
    parser.add_argument("--no-wandb", action="store_true",
                        help="Disable Weights & Biases logging")
    args = parser.parse_args()
    main(args)
