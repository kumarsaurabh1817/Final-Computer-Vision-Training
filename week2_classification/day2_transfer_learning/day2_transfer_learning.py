"""
Day 2: Transfer Learning — Feature Extraction vs Fine-Tuning
============================================================

Objectives:
- Understand two transfer-learning strategies:
    A) Feature extraction  — freeze the backbone, only train the new head
    B) Full fine-tuning    — unfreeze the backbone with a tiny LR
- Know *why* the LR must be 10–100× smaller when fine-tuning
- Replace the classifier head (fc / classifier) for an arbitrary N-class task
- Compare the two strategies on a small dataset to see which wins and when
- Save and inspect the modified model

Materials:
- PyTorch Transfer Learning Tutorial:
    https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html
- CS231n — Transfer Learning notes:
    https://cs231n.github.io/transfer-learning/

Dataset used:
  By default we use CIFAR-10 (always available) so you can run this script
  immediately without downloading anything extra. On Day 3 you will swap in
  your own Kaggle dataset — the data-pipeline scaffolding is already in place.

Run:
    # Quick smoke-test (5 epochs each mode, CIFAR-10)
    python day2_transfer_learning.py --epochs 5 --no-wandb

    # Full comparison (20 epochs each, with W&B logging)
    python day2_transfer_learning.py --epochs 20

    # Try a different backbone
    python day2_transfer_learning.py --backbone efficientnet_b0 --epochs 10
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import torchvision
import torchvision.models as models
from torchvision.models import (
    ResNet18_Weights, ResNet50_Weights,
    EfficientNet_B0_Weights, EfficientNet_B3_Weights,
)
import torchvision.transforms as T
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import time
import argparse
import json

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("⚠️  wandb not installed — run: pip install wandb")

# ── Directories ───────────────────────────────────────────────────────────────
BASE_DIR   = Path(__file__).parent.parent
RESULTS_DIR = BASE_DIR / "results"
MODEL_DIR   = BASE_DIR / "models"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Reproducibility
torch.manual_seed(42)
np.random.seed(42)


# =============================================================================
# 1. MODEL BUILDER — replace head, freeze/unfreeze
# =============================================================================

SUPPORTED_BACKBONES = {
    "resnet18":       (models.resnet18,       ResNet18_Weights.DEFAULT),
    "resnet50":       (models.resnet50,       ResNet50_Weights.DEFAULT),
    "efficientnet_b0":(models.efficientnet_b0, EfficientNet_B0_Weights.DEFAULT),
    "efficientnet_b3":(models.efficientnet_b3, EfficientNet_B3_Weights.DEFAULT),
}


def build_model(backbone_name: str, num_classes: int, freeze_backbone: bool) -> nn.Module:
    """
    Load a pretrained backbone, replace its classifier head, and optionally
    freeze all backbone parameters so only the head is trained.

    Args:
        backbone_name:   One of SUPPORTED_BACKBONES keys.
        num_classes:     Number of output classes for the new task.
        freeze_backbone: True → feature extraction only.
                         False → full fine-tuning (backbone + head).

    Returns:
        model (nn.Module) ready to move to DEVICE.
    """
    builder, weights = SUPPORTED_BACKBONES[backbone_name]
    model = builder(weights=weights)

    # ── Step 1: Freeze all backbone parameters ────────────────────────────
    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False
        print(f"  ✓ Backbone frozen — only training the new head")
    else:
        print(f"  ✓ All layers unfrozen — full fine-tuning with low LR")

    # ── Step 2: Replace the classification head ───────────────────────────
    #
    # The head location differs per family:
    #   ResNet:      model.fc              Linear(in_features → 1000)
    #   EfficientNet: model.classifier[-1] Linear(in_features → 1000)
    #
    if "resnet" in backbone_name:
        in_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(in_features, num_classes),
        )
        print(f"  ✓ ResNet head replaced: Linear({in_features} → {num_classes})")

    elif "efficientnet" in backbone_name:
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(in_features, num_classes),
        )
        print(f"  ✓ EfficientNet head replaced: Linear({in_features} → {num_classes})")

    else:
        raise ValueError(f"Unsupported backbone: {backbone_name}")

    # ── Step 3: Re-enable gradients on the new head ───────────────────────
    #   (required when freeze_backbone=True — the new head must train)
    head = model.fc if "resnet" in backbone_name else model.classifier[-1]
    for param in head.parameters():
        param.requires_grad = True

    return model.to(DEVICE)


def count_trainable_params(model: nn.Module) -> tuple[int, int]:
    """Return (trainable_params, total_params)."""
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return trainable, total


def print_param_summary(model: nn.Module, label: str):
    trainable, total = count_trainable_params(model)
    frozen = total - trainable
    print(f"\n  [{label}] Parameter summary:")
    print(f"    Total      : {total:>12,}")
    print(f"    Trainable  : {trainable:>12,}  ({100*trainable/total:.1f}%)")
    print(f"    Frozen     : {frozen:>12,}  ({100*frozen/total:.1f}%)")


# =============================================================================
# 2. DATA PIPELINE
# =============================================================================
#
# We use CIFAR-10 here so the script runs without any extra downloads.
# On Day 3 you will plug in a Kaggle dataset — swap `get_cifar10_loaders`
# with your own `get_custom_loaders` following the same interface.
#

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD  = (0.2470, 0.2435, 0.2616)

# ImageNet stats — used when the backbone was pretrained on ImageNet
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)

CIFAR10_CLASSES = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                   'dog', 'frog', 'horse', 'ship', 'truck']


def get_cifar10_loaders(batch_size: int = 64,
                         val_split: float = 0.1,
                         img_size: int = 224,
                         num_workers: int = 2
                         ) -> tuple[DataLoader, DataLoader, DataLoader]:
    """
    CIFAR-10 data loaders with ImageNet normalisation.

    We resize 32×32 → img_size (224 by default) because pretrained backbones
    expect ~224 px inputs — this is the standard approach for transfer learning
    on small-image datasets.
    """
    transform_train = T.Compose([
        T.Resize(img_size),
        T.RandomHorizontalFlip(),
        T.RandomCrop(img_size, padding=8),
        T.ToTensor(),
        T.Normalize(IMAGENET_MEAN, IMAGENET_STD),   # ImageNet stats!
    ])

    transform_eval = T.Compose([
        T.Resize(img_size),
        T.CenterCrop(img_size),
        T.ToTensor(),
        T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    data_root = str(BASE_DIR / "data")

    full_train = torchvision.datasets.CIFAR10(
        root=data_root, train=True, download=True, transform=transform_train
    )
    test_set = torchvision.datasets.CIFAR10(
        root=data_root, train=False, download=True, transform=transform_eval
    )

    val_size   = int(len(full_train) * val_split)
    train_size = len(full_train) - val_size
    train_set, val_set = random_split(full_train, [train_size, val_size],
                                       generator=torch.Generator().manual_seed(42))

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True)
    val_loader   = DataLoader(val_set,   batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True)
    test_loader  = DataLoader(test_set,  batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True)

    print(f"\n  Dataset: CIFAR-10  (resized to {img_size}px)")
    print(f"  Splits — Train: {train_size:,}  Val: {val_size:,}  Test: {len(test_set):,}")
    return train_loader, val_loader, test_loader


# =============================================================================
# 3. TRAINING & EVALUATION
# =============================================================================

def train_one_epoch(model, loader, criterion, optimizer) -> tuple[float, float]:
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        out  = model(imgs)
        loss = criterion(out, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * imgs.size(0)
        _, pred = out.max(1)
        total   += labels.size(0)
        correct += pred.eq(labels).sum().item()
    return total_loss / total, 100.0 * correct / total


@torch.no_grad()
def evaluate(model, loader, criterion) -> tuple[float, float]:
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        out  = model(imgs)
        loss = criterion(out, labels)
        total_loss += loss.item() * imgs.size(0)
        _, pred = out.max(1)
        total   += labels.size(0)
        correct += pred.eq(labels).sum().item()
    return total_loss / total, 100.0 * correct / total


def run_training(
    backbone_name: str,
    num_classes:   int,
    freeze_backbone: bool,
    train_loader:  DataLoader,
    val_loader:    DataLoader,
    test_loader:   DataLoader,
    epochs:        int,
    lr_head:       float,
    lr_backbone:   float,
    weight_decay:  float,
    wandb_run,
) -> dict:
    """
    Run a full training loop.
    Returns a history dict with train/val loss & accuracy per epoch.

    LR strategy:
    ┌──────────────────────┬─────────────────────────────────────────────────┐
    │ Mode                 │ LR logic                                        │
    ├──────────────────────┼─────────────────────────────────────────────────┤
    │ Feature extraction   │ Only head params are trainable → single LR       │
    │ Full fine-tuning     │ Backbone params get lr_backbone (<<head LR)      │
    │                      │ Head params get lr_head (normal LR)              │
    └──────────────────────┴─────────────────────────────────────────────────┘
    The two-LR approach prevents "catastrophic forgetting" — blowing away
    the rich ImageNet representations with too large a gradient step.
    """
    mode = "feature_extraction" if freeze_backbone else "full_finetune"
    label = "FROZEN backbone" if freeze_backbone else "UNFROZEN backbone"

    print(f"\n{'─'*60}")
    print(f"  Training: {backbone_name.upper()}  [{label}]")
    print(f"  Epochs: {epochs}  |  LR head: {lr_head}  |  LR backbone: {lr_backbone}")
    print(f"{'─'*60}")

    model = build_model(backbone_name, num_classes, freeze_backbone)
    print_param_summary(model, mode)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    # ── Separate parameter groups ──────────────────────────────────────────
    if freeze_backbone:
        # Only the head is trainable — simple single-LR optimiser
        head_params = [p for p in model.parameters() if p.requires_grad]
        optimizer = optim.AdamW(head_params, lr=lr_head, weight_decay=weight_decay)
    else:
        # Backbone + head — different LRs per group
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

    # Cosine annealing scheduler — smooth LR decay
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_val_acc = 0.0

    t_start = time.time()
    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer)
        val_loss,   val_acc   = evaluate(model, val_loader, criterion)
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
                f"{mode}/train_loss": train_loss,
                f"{mode}/train_acc": train_acc,
                f"{mode}/val_loss": val_loss,
                f"{mode}/val_acc": val_acc,
                f"{mode}/lr": lr_now,
            })

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), MODEL_DIR / f"week2_{backbone_name}_{mode}.pt")

    elapsed = time.time() - t_start
    print(f"\n  Best val accuracy: {best_val_acc:.1f}%  (training took {elapsed:.0f}s)")

    # ── Final test evaluation ─────────────────────────────────────────────
    model.load_state_dict(torch.load(MODEL_DIR / f"week2_{backbone_name}_{mode}.pt",
                                     weights_only=True))
    test_loss, test_acc = evaluate(model, test_loader, criterion)
    print(f"  Test accuracy: {test_acc:.1f}%")

    if wandb_run:
        wandb_run.log({
            f"{mode}/test_acc": test_acc,
            f"{mode}/test_loss": test_loss,
        })

    history["best_val_acc"] = best_val_acc
    history["test_acc"] = test_acc
    history["mode"] = mode
    history["label"] = label
    return history


# =============================================================================
# 4. VISUALISATION
# =============================================================================

def plot_comparison(feat_history: dict, finetune_history: dict,
                    backbone_name: str):
    """Side-by-side training curves for both transfer learning modes."""
    epochs = range(1, len(feat_history["train_acc"]) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f"Transfer Learning Comparison — {backbone_name.upper()}", fontsize=14)

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
    save_path = RESULTS_DIR / f"day2_{backbone_name}_comparison.png"
    plt.savefig(save_path, dpi=120)
    plt.close()
    print(f"\n  Comparison plot saved → {save_path}")


# =============================================================================
# 5. CONCEPTUAL EXPLAINER — printed at startup
# =============================================================================

def print_transfer_learning_explainer():
    print("\n" + "="*60)
    print("TRANSFER LEARNING — CONCEPT EXPLAINER")
    print("="*60)
    explainer = """
Why transfer learning?
  Training from scratch on a small dataset (< 50 k images) almost always
  leads to overfitting. Pretrained backbones carry rich, general-purpose
  features learned from 1.2 M ImageNet images — edges, textures, shapes —
  that transfer well to almost any visual task.

Strategy A — Feature Extraction (freeze backbone):
  • Freeze ALL backbone weights.
  • Only train the new classification head.
  • Training is FAST (no backbone gradients to compute).
  • Works well when your dataset is SMALL or VERY similar to ImageNet.
  • Risk: the frozen features may not be optimal for your domain.

Strategy B — Full Fine-Tuning (unfreeze backbone):
  • Allow ALL weights to update.
  • Use a MUCH smaller LR for the backbone (typically 10–100× lower).
  • Training is SLOWER but often more accurate.
  • Works well when your dataset is LARGE or your domain is DIFFERENT.
  • Risk: 'catastrophic forgetting' if backbone LR is too high.

Typical LR values:
  Head      :  1e-3 to 1e-4   (learning from scratch)
  Backbone  :  1e-5 to 1e-6   (gentle nudge, don't destroy pretrained features)

Heuristic:
  1. Always start with feature extraction as a strong baseline.
  2. If the domain is different from ImageNet, try fine-tuning.
  3. Use discriminative learning rates (head > backbone).
"""
    print(explainer)


# =============================================================================
# MAIN
# =============================================================================

def main(args):
    print("\n" + "█"*60)
    print("  Week 2 · Day 2 — Transfer Learning")
    print("█"*60)
    print(f"  Device: {DEVICE}  |  Backbone: {args.backbone}")

    print_transfer_learning_explainer()

    # ── W&B ──────────────────────────────────────────────────────────────
    wandb_run = None
    if WANDB_AVAILABLE and not args.no_wandb:
        wandb_run = wandb.init(
            project="cv-week2-transfer-learning",
            name=f"{args.backbone}-comparison",
            config=vars(args),
        )

    # ── Data ──────────────────────────────────────────────────────────────
    img_size = 300 if "b3" in args.backbone else 224
    train_loader, val_loader, test_loader = get_cifar10_loaders(
        batch_size=args.batch_size,
        img_size=img_size,
        num_workers=args.num_workers,
    )
    num_classes = 10   # CIFAR-10

    # ── Strategy A: Feature Extraction ────────────────────────────────────
    feat_history = run_training(
        backbone_name=args.backbone,
        num_classes=num_classes,
        freeze_backbone=True,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        epochs=args.epochs,
        lr_head=args.lr_head,
        lr_backbone=args.lr_backbone,
        weight_decay=args.weight_decay,
        wandb_run=wandb_run,
    )

    # ── Strategy B: Full Fine-Tuning ──────────────────────────────────────
    finetune_history = run_training(
        backbone_name=args.backbone,
        num_classes=num_classes,
        freeze_backbone=False,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        epochs=args.epochs,
        lr_head=args.lr_head,
        lr_backbone=args.lr_backbone,
        weight_decay=args.weight_decay,
        wandb_run=wandb_run,
    )

    # ── Comparison ────────────────────────────────────────────────────────
    plot_comparison(feat_history, finetune_history, args.backbone)

    print("\n" + "="*60)
    print("RESULTS SUMMARY")
    print("="*60)
    print(f"  {'Strategy':<25} {'Val Acc':>10} {'Test Acc':>10}")
    print("  " + "-"*47)
    print(f"  {'Feature Extraction':<25} {feat_history['best_val_acc']:>9.1f}%"
          f" {feat_history['test_acc']:>9.1f}%")
    print(f"  {'Full Fine-Tuning':<25} {finetune_history['best_val_acc']:>9.1f}%"
          f" {finetune_history['test_acc']:>9.1f}%")

    winner = ("Full Fine-Tuning" if finetune_history["test_acc"] > feat_history["test_acc"]
              else "Feature Extraction")
    print(f"\n  Winner on test set: ✅ {winner}")
    print("\n  Key takeaway:")
    if finetune_history["test_acc"] > feat_history["test_acc"]:
        diff = finetune_history["test_acc"] - feat_history["test_acc"]
        print(f"    Fine-tuning gained +{diff:.1f}% over feature extraction.")
        print("    → The domain shift from ImageNet to CIFAR-10 justifies updating the backbone.")
    else:
        diff = feat_history["test_acc"] - finetune_history["test_acc"]
        print(f"    Feature extraction matched or beat fine-tuning (+{diff:.1f}%).")
        print("    → ImageNet features transfer well; simpler approach is sufficient.")

    # ── Save results JSON ─────────────────────────────────────────────────
    results = {
        "backbone": args.backbone,
        "epochs":   args.epochs,
        "feature_extraction": {
            "best_val_acc": feat_history["best_val_acc"],
            "test_acc":     feat_history["test_acc"],
        },
        "full_finetune": {
            "best_val_acc": finetune_history["best_val_acc"],
            "test_acc":     finetune_history["test_acc"],
        },
    }
    results_path = RESULTS_DIR / f"day2_{args.backbone}_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results JSON saved → {results_path}")

    if wandb_run:
        wandb_run.summary.update(results)
        wandb_run.finish()

    print(f"\n✅ Day 2 complete.")
    print("   Next: Day 3 — Custom dataset pipeline + Albumentations augmentation")


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Week 2 Day 2 — Transfer Learning")
    parser.add_argument("--backbone", type=str, default="resnet18",
                        choices=list(SUPPORTED_BACKBONES.keys()),
                        help="Backbone to use for both strategies")
    parser.add_argument("--epochs",     type=int,   default=10,
                        help="Training epochs per strategy")
    parser.add_argument("--batch-size", type=int,   default=64,
                        help="DataLoader batch size")
    parser.add_argument("--lr-head",    type=float, default=1e-3,
                        help="Learning rate for the classifier head")
    parser.add_argument("--lr-backbone",type=float, default=1e-5,
                        help="Learning rate for the frozen backbone (fine-tuning only)")
    parser.add_argument("--weight-decay",type=float,default=1e-4,
                        help="AdamW weight decay")
    parser.add_argument("--num-workers",type=int,   default=2,
                        help="DataLoader workers")
    parser.add_argument("--no-wandb", action="store_true",
                        help="Disable Weights & Biases logging")
    args = parser.parse_args()
    main(args)
