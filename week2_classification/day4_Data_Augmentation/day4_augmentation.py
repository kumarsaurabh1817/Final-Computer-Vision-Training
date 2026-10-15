"""
Day 4: Data Augmentation Ablation with Albumentations
======================================================

Objectives:
- Introduce Albumentations as the augmentation library (faster, more flexible
  than torchvision transforms, detection/segmentation-ready for Week 3)
- Establish a NO-AUGMENTATION baseline (resize + center crop + normalize only)
- Build incremental augmentation pipelines:
    1. Baseline     — no augmentation
    2. Spatial      — HorizontalFlip + ShiftScaleRotate + RandomCrop
    3. Color        — RandomBrightnessContrast + HueSaturationValue + RGBShift
    4. Advanced     — CoarseDropout + GaussNoise + GaussianBlur
    5. Full         — all combined
- Run a controlled ablation: same backbone, same LR, same epochs, same split
- Compare accuracy, build ablation table, visualise augmented samples

Dataset:
  Intel Image Classification (same as Day 3) — ~25k images, 6 classes
  Expected on Google Drive at:
      /content/drive/MyDrive/datasets/intel-image-classification/

  Or locally at:
      week2_classification/data/intel-image-classification/

Materials:
- Albumentations — Documentation:
    https://albumentations.ai/docs/
- Albumentations — PyTorch examples:
    https://albumentations.ai/docs/examples/pytorch_classification/

Run:
    # Quick: explore augmentations visually (no training)
    python day4_augmentation.py --explore-only

    # Full ablation (5 experiments × N epochs each)
    python day4_augmentation.py --epochs 10 --no-wandb

    # Try with a different backbone
    python day4_augmentation.py --backbone efficientnet_b0 --epochs 10
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision
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
from collections import Counter, OrderedDict
import time
import argparse
import json
import os
import copy

try:
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
    ALBU_AVAILABLE = True
except ImportError:
    ALBU_AVAILABLE = False
    print("❌ albumentations not installed — run: pip install albumentations")

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
# 1. DATASET DISCOVERY
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
# 2. ALBUMENTATIONS PIPELINES — THE CORE OF TODAY
# =============================================================================

def get_augmentation_pipelines(img_size: int = 224) -> OrderedDict:
    """
    Build the five augmentation pipelines for the ablation study.

    Returns an OrderedDict mapping experiment name → A.Compose.
    Each pipeline ends with Normalize + ToTensorV2 (required for PyTorch).

    CRITICAL: Evaluation transforms are ALWAYS the same (deterministic)
    across all experiments — only training transforms differ.
    """
    # ── Common tail: Normalize + ToTensorV2 ──────────────────────────────
    norm_tail = [
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ]

    resize_h = int(img_size * 1.15)  # slight over-resize for crop margin

    pipelines = OrderedDict()

    # ── 1. Baseline: No augmentation ─────────────────────────────────────
    # Only resize + center crop + normalize
    # This is our controlled reference — everything else is compared to this
    pipelines["baseline"] = A.Compose([
        A.Resize(resize_h, resize_h),
        A.CenterCrop(img_size, img_size),
        *norm_tail,
    ])

    # ── 2. Spatial augmentations ─────────────────────────────────────────
    # HorizontalFlip, ShiftScaleRotate, RandomCrop
    # These change the geometry of the image
    pipelines["spatial"] = A.Compose([
        A.Resize(resize_h, resize_h),
        A.RandomCrop(img_size, img_size),
        A.HorizontalFlip(p=0.5),
        A.ShiftScaleRotate(
            shift_limit=0.05,     # ±5% translation
            scale_limit=0.1,      # ±10% zoom
            rotate_limit=15,      # ±15° rotation
            border_mode=0,        # constant border
            p=0.5,
        ),
        *norm_tail,
    ])

    # ── 3. Color augmentations ───────────────────────────────────────────
    # RandomBrightnessContrast, HueSaturationValue, RGBShift
    # These change pixel values but not geometry
    pipelines["color"] = A.Compose([
        A.Resize(resize_h, resize_h),
        A.CenterCrop(img_size, img_size),
        A.RandomBrightnessContrast(
            brightness_limit=0.2,
            contrast_limit=0.2,
            p=0.5,
        ),
        A.HueSaturationValue(
            hue_shift_limit=10,
            sat_shift_limit=20,
            val_shift_limit=20,
            p=0.3,
        ),
        A.RGBShift(r_shift_limit=15, g_shift_limit=15, b_shift_limit=15, p=0.3),
        *norm_tail,
    ])

    # ── 4. Advanced augmentations ────────────────────────────────────────
    # CoarseDropout (Cutout), GaussNoise, GaussianBlur
    # These simulate occlusion and sensor noise
    pipelines["advanced"] = A.Compose([
        A.Resize(resize_h, resize_h),
        A.CenterCrop(img_size, img_size),
        # albumentations >=2.0: CoarseDropout & GaussNoise API renamed
        A.CoarseDropout(
            num_holes_range=(1, 8),
            hole_height_range=(img_size // 16, img_size // 8),
            hole_width_range=(img_size // 16, img_size // 8),
            fill=0,
            p=0.3,
        ),
        A.OneOf([
            A.GaussNoise(std_range=(3.0, 7.0)),  # sqrt(10)~3.2, sqrt(50)~7.1
            A.GaussianBlur(blur_limit=(3, 7)),
        ], p=0.3),
        *norm_tail,
    ])

    # ── 5. Full augmentation ─────────────────────────────────────────────
    # Everything combined — the "kitchen sink"
    pipelines["full"] = A.Compose([
        A.Resize(resize_h, resize_h),
        A.RandomCrop(img_size, img_size),
        A.HorizontalFlip(p=0.5),
        A.ShiftScaleRotate(
            shift_limit=0.05,
            scale_limit=0.1,
            rotate_limit=15,
            border_mode=0,
            p=0.5,
        ),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.4),
        A.HueSaturationValue(
            hue_shift_limit=10,
            sat_shift_limit=20,
            val_shift_limit=20,
            p=0.3,
        ),
        A.OneOf([
            A.GaussNoise(std_range=(3.0, 7.0)),  # sqrt(10)~3.2, sqrt(50)~7.1
            A.GaussianBlur(blur_limit=(3, 7)),
        ], p=0.2),
        A.CoarseDropout(
            num_holes_range=(1, 8),
            hole_height_range=(img_size // 16, img_size // 8),
            hole_width_range=(img_size // 16, img_size // 8),
            fill=0,
            p=0.3,
        ),
        *norm_tail,
    ])

    return pipelines


def get_eval_transform(img_size: int = 224) -> A.Compose:
    """
    Deterministic evaluation transform — SAME for all experiments.
    No randomness, no augmentation.
    """
    resize_h = int(img_size * 1.15)
    return A.Compose([
        A.Resize(resize_h, resize_h),
        A.CenterCrop(img_size, img_size),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])


# =============================================================================
# 3. ALBUMENTATIONS DATASET WRAPPER
# =============================================================================

class AlbumentationsDataset(Dataset):
    """
    Wraps a torchvision ImageFolder dataset to use Albumentations transforms.

    ImageFolder returns (PIL Image, label).
    Albumentations expects NumPy arrays (H, W, C), uint8.
    This wrapper handles the conversion.
    """

    def __init__(self, image_folder_dataset, albu_transform=None):
        """
        Args:
            image_folder_dataset: A torchvision ImageFolder dataset
                                  with transform=None (no torchvision transforms!)
            albu_transform:       An albumentations.Compose pipeline
        """
        self.dataset = image_folder_dataset
        self.albu_transform = albu_transform

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        # ImageFolder returns (PIL Image, label) when transform=None
        pil_image, label = self.dataset[idx]

        # Convert PIL → NumPy (H, W, C) uint8 — what Albumentations expects
        image = np.array(pil_image)

        if self.albu_transform:
            augmented = self.albu_transform(image=image)
            image = augmented["image"]  # Now a (C, H, W) float32 tensor

        return image, label

    @property
    def classes(self):
        return self.dataset.classes


# =============================================================================
# 4. DATA LOADERS
# =============================================================================

def get_data_loaders(train_dir: Path, test_dir: Path,
                     train_transform, eval_transform,
                     batch_size: int = 32, val_split: float = 0.15,
                     num_workers: int = 2,
                     ) -> tuple[DataLoader, DataLoader, DataLoader, list[str]]:
    """
    Build train/val/test DataLoaders with Albumentations transforms.

    Args:
        train_dir:       Path to training images (class-per-folder).
        test_dir:        Path to test images (class-per-folder).
        train_transform: Albumentations Compose for training.
        eval_transform:  Albumentations Compose for eval (deterministic).
        batch_size:      Batch size.
        val_split:       Fraction of training data for validation.
        num_workers:     DataLoader workers.
    """
    # Load RAW datasets (no transforms — Albumentations handles this)
    raw_train = ImageFolder(str(train_dir), transform=None)
    raw_test  = ImageFolder(str(test_dir),  transform=None)
    class_names = raw_train.classes

    # ── Train / Val split ─────────────────────────────────────────────────
    val_size   = int(len(raw_train) * val_split)
    train_size = len(raw_train) - val_size
    train_subset, val_subset = random_split(
        raw_train, [train_size, val_size],
        generator=torch.Generator().manual_seed(42),  # SAME split every run!
    )

    # Wrap with Albumentations
    train_dataset = AlbumentationsSubsetDataset(train_subset, train_transform)
    val_dataset   = AlbumentationsSubsetDataset(val_subset,   eval_transform)
    test_dataset  = AlbumentationsDataset(raw_test, eval_transform)

    # ── DataLoaders ───────────────────────────────────────────────────────
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )

    print(f"    Splits — Train: {train_size:,}  Val: {val_size:,}  Test: {len(raw_test):,}")
    print(f"    Batch size: {batch_size}")

    return train_loader, val_loader, test_loader, class_names


class AlbumentationsSubsetDataset(Dataset):
    """
    Wraps a torch.utils.data.Subset (from random_split) to apply
    Albumentations transforms. The Subset returns (PIL Image, label)
    from the underlying ImageFolder.
    """

    def __init__(self, subset, albu_transform=None):
        self.subset = subset
        self.albu_transform = albu_transform

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, idx):
        pil_image, label = self.subset[idx]
        image = np.array(pil_image)

        if self.albu_transform:
            augmented = self.albu_transform(image=image)
            image = augmented["image"]

        return image, label


# =============================================================================
# 5. VISUALISE AUGMENTED SAMPLES
# =============================================================================

def visualise_augmentations(train_dir: Path, class_names: list[str],
                            pipelines: OrderedDict, img_size: int = 224):
    """
    Show the same image under each augmentation pipeline.
    Critical sanity check — do the augmentations look reasonable?
    """
    print("\n" + "=" * 60)
    print("VISUALISING AUGMENTATIONS")
    print("=" * 60)

    # Pick one sample image per class
    sample_images = []
    sample_labels = []
    for cls_name in class_names[:3]:  # Show 3 classes
        cls_dir = train_dir / cls_name
        img_path = sorted(cls_dir.glob("*"))[0]
        img = np.array(Image.open(img_path).convert("RGB"))
        sample_images.append(img)
        sample_labels.append(cls_name)

    n_pipelines = len(pipelines)
    n_images = len(sample_images)

    fig, axes = plt.subplots(n_images, n_pipelines + 1, figsize=(3 * (n_pipelines + 1), 3 * n_images))
    if n_images == 1:
        axes = axes[np.newaxis, :]

    # Column 0: Original
    for i, (img, label) in enumerate(zip(sample_images, sample_labels)):
        axes[i, 0].imshow(img)
        axes[i, 0].set_title(f"Original\n({label})", fontsize=9)
        axes[i, 0].axis("off")

    # Columns 1+: Each augmentation pipeline
    for j, (name, pipeline) in enumerate(pipelines.items(), start=1):
        for i, img in enumerate(sample_images):
            augmented = pipeline(image=img.copy())["image"]

            # Un-normalize for display
            display = augmented.float()
            mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
            std  = torch.tensor(IMAGENET_STD).view(3, 1, 1)
            display = display * std + mean
            display = display.permute(1, 2, 0).numpy()
            display = np.clip(display, 0, 1)

            axes[i, j].imshow(display)
            axes[i, j].set_title(name, fontsize=9)
            axes[i, j].axis("off")

    plt.suptitle("Augmentation Pipeline Comparison", fontsize=14, fontweight="bold")
    plt.tight_layout()
    save_path = RESULTS_DIR / "day4_augmentation_comparison.png"
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  Augmentation comparison saved → {save_path}")


def visualise_single_pipeline(train_dir: Path, pipeline_name: str,
                               pipeline: A.Compose, n_examples: int = 8):
    """
    Show N random augmented versions of the same image using one pipeline.
    This shows the diversity of augmentations.
    """
    # Pick a sample image
    cls_dirs = sorted([d for d in train_dir.iterdir() if d.is_dir()])
    img_path = sorted(cls_dirs[0].glob("*"))[0]
    original = np.array(Image.open(img_path).convert("RGB"))

    fig, axes = plt.subplots(2, n_examples // 2 + 1, figsize=(16, 6))
    axes_flat = axes.flat

    # First image: original
    axes_flat[0].imshow(original)
    axes_flat[0].set_title("Original", fontsize=9, fontweight="bold")
    axes_flat[0].axis("off")

    # Rest: augmented versions
    for i in range(1, n_examples + 1):
        augmented = pipeline(image=original.copy())["image"]

        # Un-normalize
        display = augmented.float()
        mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
        std  = torch.tensor(IMAGENET_STD).view(3, 1, 1)
        display = display * std + mean
        display = display.permute(1, 2, 0).numpy()
        display = np.clip(display, 0, 1)

        if i < len(axes_flat):
            axes_flat[i].imshow(display)
            axes_flat[i].set_title(f"Aug #{i}", fontsize=9)
            axes_flat[i].axis("off")

    # Hide unused axes
    for i in range(n_examples + 1, len(axes_flat)):
        axes_flat[i].axis("off")

    plt.suptitle(f"'{pipeline_name}' — Multiple Augmentations of Same Image",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    save_path = RESULTS_DIR / f"day4_{pipeline_name}_diversity.png"
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  Diversity plot saved → {save_path}")


# =============================================================================
# 6. MODEL BUILDER (same as Day 3)
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

    # Replace classifier head
    if "resnet" in backbone_name:
        in_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(in_features, num_classes),
        )
    elif "efficientnet" in backbone_name:
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(in_features, num_classes),
        )

    # Re-enable gradients on the new head
    head = model.fc if "resnet" in backbone_name else model.classifier[-1]
    for param in head.parameters():
        param.requires_grad = True

    return model.to(DEVICE)


# =============================================================================
# 7. TRAINING & EVALUATION
# =============================================================================

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


def run_ablation_experiment(
    exp_name: str,
    backbone_name: str,
    num_classes: int,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    class_names: list[str],
    epochs: int,
    lr: float,
    weight_decay: float,
    wandb_run=None,
) -> dict:
    """
    Run one experiment in the ablation study.
    Uses full fine-tuning (unfrozen backbone, two-LR strategy).
    """
    print(f"\n{'─' * 60}")
    print(f"  Experiment: {exp_name.upper()}")
    print(f"  Backbone: {backbone_name}  |  Epochs: {epochs}  |  LR: {lr}")
    print(f"{'─' * 60}")

    model = build_model(backbone_name, num_classes, freeze_backbone=False)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_p   = sum(p.numel() for p in model.parameters())
    print(f"  Params: {trainable:,} trainable / {total_p:,} total")

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    # Two-LR strategy: low LR for backbone, normal LR for head
    if "resnet" in backbone_name:
        head_module = model.fc
    else:
        head_module = model.classifier[-1]

    head_param_ids = {id(p) for p in head_module.parameters()}
    backbone_params = [p for p in model.parameters()
                       if p.requires_grad and id(p) not in head_param_ids]
    head_params     = list(head_module.parameters())

    optimizer = optim.AdamW([
        {"params": backbone_params, "lr": lr * 0.01},  # backbone: 100× lower
        {"params": head_params,     "lr": lr},
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
                f"{exp_name}/epoch": epoch,
                f"{exp_name}/train_loss": train_loss, f"{exp_name}/train_acc": train_acc,
                f"{exp_name}/val_loss": val_loss, f"{exp_name}/val_acc": val_acc,
                f"{exp_name}/lr": lr_now,
            })

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(),
                       MODEL_DIR / f"week2_day4_{backbone_name}_{exp_name}.pt")

    elapsed = time.time() - t_start
    print(f"\n  Best val accuracy: {best_val_acc:.1f}%  (training took {elapsed:.0f}s)")

    # Final test evaluation
    best_path = MODEL_DIR / f"week2_day4_{backbone_name}_{exp_name}.pt"
    if best_path.exists():
        model.load_state_dict(torch.load(best_path, weights_only=True))
    test_loss, test_acc, test_preds, test_labels = evaluate(model, test_loader, criterion)
    print(f"  Test accuracy: {test_acc:.1f}%")

    # Per-class accuracy
    per_class_acc = {}
    print(f"\n  Per-class accuracy:")
    for i, cls in enumerate(class_names):
        mask = test_labels == i
        if mask.sum() > 0:
            cls_acc = 100.0 * (test_preds[mask] == i).sum() / mask.sum()
            per_class_acc[cls] = float(cls_acc)
            print(f"    {cls:<15} {cls_acc:.1f}%  ({mask.sum()} samples)")

    if wandb_run:
        wandb_run.log({f"{exp_name}/test_acc": test_acc})

    return {
        "exp_name": exp_name,
        "best_val_acc": best_val_acc,
        "test_acc": test_acc,
        "per_class_acc": per_class_acc,
        "train_time_s": elapsed,
        "history": history,
    }


# =============================================================================
# 8. ABLATION TABLE & PLOTS
# =============================================================================

def build_ablation_table(results: list[dict], backbone_name: str):
    """
    Print and save the ablation results table.
    """
    print("\n" + "=" * 70)
    print("ABLATION RESULTS TABLE")
    print("=" * 70)
    print(f"  {'Experiment':<15} {'Val Acc':>10} {'Test Acc':>10} "
          f"{'Δ vs Base':>10} {'Time (s)':>10}")
    print("  " + "-" * 57)

    baseline_test = results[0]["test_acc"]

    for r in results:
        delta = r["test_acc"] - baseline_test
        delta_str = f"{delta:+.1f}%" if r["exp_name"] != "baseline" else "—"
        print(f"  {r['exp_name']:<15} {r['best_val_acc']:>9.1f}% "
              f"{r['test_acc']:>9.1f}% {delta_str:>10} {r['train_time_s']:>9.0f}")

    print("  " + "-" * 57)

    # Find winner
    best = max(results, key=lambda x: x["test_acc"])
    print(f"\n  ✅ Best on test set: {best['exp_name']} ({best['test_acc']:.1f}%)")

    if best["exp_name"] == "baseline":
        print("  ⚠️  Augmentation didn't help! Possible reasons:")
        print("     - Dataset is already large/diverse enough")
        print("     - Augmentations are too aggressive")
        print("     - Not enough epochs for augmented models to converge")
    else:
        improvement = best["test_acc"] - baseline_test
        print(f"  ↑  Improvement over baseline: +{improvement:.1f}%")


def plot_ablation_curves(results: list[dict], backbone_name: str):
    """
    Plot training/validation curves for all experiments on the same axes.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f"Day 4 Ablation — {backbone_name.upper()}", fontsize=14)

    colors = ["#4C72B0", "#55A868", "#C44E52", "#8172B2", "#CCB974"]

    for i, (metric, ylabel) in enumerate([("acc", "Accuracy (%)"), ("loss", "Loss")]):
        ax = axes[i]
        for j, r in enumerate(results):
            epochs = range(1, len(r["history"][f"val_{metric}"]) + 1)
            color = colors[j % len(colors)]
            ax.plot(epochs, r["history"][f"val_{metric}"],
                    color=color, label=r["exp_name"], linewidth=2)
            ax.plot(epochs, r["history"][f"train_{metric}"],
                    color=color, linestyle="--", alpha=0.4)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylabel)
        ax.set_title(f"Validation {ylabel}")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    plt.tight_layout()
    save_path = RESULTS_DIR / f"day4_{backbone_name}_ablation_curves.png"
    plt.savefig(save_path, dpi=120)
    plt.close()
    print(f"\n  Ablation curves saved → {save_path}")


def plot_per_class_comparison(results: list[dict], class_names: list[str],
                               backbone_name: str):
    """
    Bar chart comparing per-class accuracy across augmentation experiments.
    """
    n_classes = len(class_names)
    n_experiments = len(results)
    x = np.arange(n_classes)
    width = 0.8 / n_experiments

    colors = ["#4C72B0", "#55A868", "#C44E52", "#8172B2", "#CCB974"]

    fig, ax = plt.subplots(figsize=(14, 6))

    for j, r in enumerate(results):
        accs = [r["per_class_acc"].get(cls, 0) for cls in class_names]
        ax.bar(x + j * width, accs, width, label=r["exp_name"],
               color=colors[j % len(colors)], alpha=0.85)

    ax.set_xlabel("Class")
    ax.set_ylabel("Test Accuracy (%)")
    ax.set_title(f"Per-Class Accuracy — Augmentation Ablation ({backbone_name.upper()})")
    ax.set_xticks(x + width * (n_experiments - 1) / 2)
    ax.set_xticklabels(class_names, rotation=30, ha="right")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    save_path = RESULTS_DIR / f"day4_{backbone_name}_per_class_comparison.png"
    plt.savefig(save_path, dpi=120)
    plt.close()
    print(f"  Per-class comparison saved → {save_path}")


# =============================================================================
# MAIN
# =============================================================================

def main(args):
    print("\n" + "█" * 60)
    print("  Week 2 · Day 4 — Data Augmentation Ablation")
    print("█" * 60)
    print(f"  Device: {DEVICE}")

    if not ALBU_AVAILABLE:
        print("❌ Cannot proceed without albumentations. Install it:")
        print("   pip install albumentations")
        return

    # ── 1. Find dataset ───────────────────────────────────────────────────
    train_dir, test_dir, class_names = find_dataset(args.data_dir)
    num_classes = len(class_names)

    # ── 2. Build augmentation pipelines ───────────────────────────────────
    img_size = 300 if "b3" in args.backbone else args.img_size
    pipelines = get_augmentation_pipelines(img_size)
    eval_transform = get_eval_transform(img_size)

    print(f"\n  Augmentation experiments ({len(pipelines)}):")
    for name in pipelines:
        print(f"    • {name}")

    # ── 3. Visualise augmentations ────────────────────────────────────────
    visualise_augmentations(train_dir, class_names, pipelines, img_size)
    visualise_single_pipeline(train_dir, "full", pipelines["full"])

    if args.explore_only:
        print("\n✅ Exploration complete. Use --epochs N to train.")
        return

    # ── 4. Run the ablation ───────────────────────────────────────────────
    wandb_run = None
    if WANDB_AVAILABLE and not args.no_wandb:
        wandb_run = wandb.init(
            project="cv-week2-augmentation",
            name=f"day4-ablation-{args.backbone}",
            config=vars(args),
        )

    all_results = []

    for exp_name, train_transform in pipelines.items():
        print(f"\n{'━' * 60}")
        print(f"  EXPERIMENT: {exp_name.upper()}")
        print(f"{'━' * 60}")

        # Build fresh data loaders for this experiment
        train_loader, val_loader, test_loader, _ = get_data_loaders(
            train_dir, test_dir,
            train_transform=train_transform,
            eval_transform=eval_transform,
            batch_size=args.batch_size,
            val_split=args.val_split,
            num_workers=args.num_workers,
        )

        result = run_ablation_experiment(
            exp_name=exp_name,
            backbone_name=args.backbone,
            num_classes=num_classes,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            class_names=class_names,
            epochs=args.epochs,
            lr=args.lr,
            weight_decay=args.weight_decay,
            wandb_run=wandb_run,
        )
        all_results.append(result)

    # ── 5. Results ────────────────────────────────────────────────────────
    build_ablation_table(all_results, args.backbone)
    plot_ablation_curves(all_results, args.backbone)
    plot_per_class_comparison(all_results, class_names, args.backbone)

    # Save results JSON
    results_json = {
        "dataset": "Intel Image Classification",
        "backbone": args.backbone,
        "epochs": args.epochs,
        "num_classes": num_classes,
        "class_names": class_names,
        "experiments": [
            {
                "name": r["exp_name"],
                "best_val_acc": r["best_val_acc"],
                "test_acc": r["test_acc"],
                "per_class_acc": r["per_class_acc"],
                "train_time_s": r["train_time_s"],
            }
            for r in all_results
        ],
    }
    results_path = RESULTS_DIR / f"day4_{args.backbone}_ablation_results.json"
    with open(results_path, "w") as f:
        json.dump(results_json, f, indent=2)
    print(f"\n  Results JSON saved → {results_path}")

    if wandb_run:
        wandb_run.summary.update({
            "best_experiment": max(all_results, key=lambda x: x["test_acc"])["exp_name"],
            "best_test_acc": max(r["test_acc"] for r in all_results),
            "baseline_test_acc": all_results[0]["test_acc"],
        })
        wandb_run.finish()

    print(f"\n✅ Day 4 complete.")
    print("   Next: Day 5 — Full ablation study with confusion matrix & per-class metrics")


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Week 2 Day 4 — Data Augmentation Ablation with Albumentations"
    )
    parser.add_argument("--data-dir", type=str, default=None,
                        help="Path to dataset root (default: data/intel-image-classification)")
    parser.add_argument("--backbone", type=str, default="resnet18",
                        choices=list(SUPPORTED_BACKBONES.keys()),
                        help="Backbone to use")
    parser.add_argument("--epochs", type=int, default=10,
                        help="Training epochs per experiment")
    parser.add_argument("--batch-size", type=int, default=32,
                        help="Batch size")
    parser.add_argument("--img-size", type=int, default=224,
                        help="Image size (pixels)")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="Learning rate for head (backbone gets lr×0.01)")
    parser.add_argument("--weight-decay", type=float, default=1e-4,
                        help="AdamW weight decay")
    parser.add_argument("--val-split", type=float, default=0.15,
                        help="Fraction of training data for validation")
    parser.add_argument("--num-workers", type=int, default=2,
                        help="DataLoader workers")
    parser.add_argument("--explore-only", action="store_true",
                        help="Only visualise augmentations, no training")
    parser.add_argument("--no-wandb", action="store_true",
                        help="Disable Weights & Biases logging")
    args = parser.parse_args()
    main(args)
