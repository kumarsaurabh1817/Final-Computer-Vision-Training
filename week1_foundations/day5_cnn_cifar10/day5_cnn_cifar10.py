"""
Day 5: Train CNN on CIFAR-10 — Validation, Tuning & W&B Logging
================================================================

Objectives:
- Build a polished CNN training pipeline with proper train/val/test splits
- Tune learning rate and batch size
- Log everything to Weights & Biases
- Achieve ≥70% test accuracy (Week 1 success gate)

Materials:
- PyTorch — Training a Classifier (CIFAR-10)
- Weights & Biases — PyTorch Integration

This is the DELIVERABLE script for Week 1.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import torchvision
import torchvision.transforms as T
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import time
import argparse

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("⚠️  wandb not installed. Run: pip install wandb")
    print("   Continuing without W&B logging.\n")

from utils import plot_learning_curves, plot_confusion_matrix, count_parameters

# Directories
OUTPUT_DIR = Path(__file__).parent.parent / "results"
MODEL_DIR = Path(__file__).parent.parent / "models"
OUTPUT_DIR.mkdir(exist_ok=True)
MODEL_DIR.mkdir(exist_ok=True)

# Reproducibility
torch.manual_seed(42)
np.random.seed(42)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CIFAR10_CLASSES = ('plane', 'car', 'bird', 'cat', 'deer',
                   'dog', 'frog', 'horse', 'ship', 'truck')


# =============================================================================
# MODEL
# =============================================================================

class CIFAR10CNN(nn.Module):
    """
    An improved CNN for CIFAR-10 with BatchNorm and Dropout.

    Architecture:
        Conv(3→32) → BN → ReLU → Conv(32→32) → BN → ReLU → MaxPool → Dropout
        Conv(32→64) → BN → ReLU → Conv(64→64) → BN → ReLU → MaxPool → Dropout
        Flatten → Linear(4096→512) → BN → ReLU → Dropout → Linear(512→10)
    """

    def __init__(self, num_classes=10, dropout_rate=0.25):
        super().__init__()

        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(dropout_rate),

            # Block 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(dropout_rate),
        )

        self.classifier = nn.Sequential(
            nn.Linear(64 * 8 * 8, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


# =============================================================================
# DATA
# =============================================================================

def get_data_loaders(batch_size=64, val_split=0.1, num_workers=2):
    """Create train/val/test data loaders with proper splits."""
    transform_train = T.Compose([
        T.ToTensor(),
        T.Normalize((0.4914, 0.4822, 0.4465),   # CIFAR-10 mean
                     (0.2470, 0.2435, 0.2616)),   # CIFAR-10 std
    ])

    transform_test = T.Compose([
        T.ToTensor(),
        T.Normalize((0.4914, 0.4822, 0.4465),
                     (0.2470, 0.2435, 0.2616)),
    ])

    # Full training set
    full_train = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform_train
    )

    # Split into train and validation
    val_size = int(len(full_train) * val_split)
    train_size = len(full_train) - val_size
    train_set, val_set = random_split(full_train, [train_size, val_size])

    # Test set
    test_set = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform_test
    )

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=True)

    print(f"Data splits: Train={train_size}, Val={val_size}, Test={len(test_set)}")
    return train_loader, val_loader, test_loader


# =============================================================================
# TRAINING
# =============================================================================

def train_one_epoch(model, loader, criterion, optimizer):
    """Train for one epoch."""
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for inputs, labels in loader:
        inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    return total_loss / total, 100. * correct / total


@torch.no_grad()
def evaluate(model, loader, criterion):
    """Evaluate and return loss, accuracy, all predictions and labels."""
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []

    for inputs, labels in loader:
        inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        total_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    return (total_loss / total, 100. * correct / total,
            np.array(all_preds), np.array(all_labels))


# =============================================================================
# MAIN TRAINING LOOP
# =============================================================================

def main(args):
    """Main training function."""
    print("=" * 60)
    print("Week 1 Deliverable — CNN on CIFAR-10")
    print("=" * 60)
    print(f"Device: {DEVICE}")
    print(f"Config: epochs={args.epochs}, lr={args.lr}, bs={args.batch_size}")

    # Initialize W&B
    if WANDB_AVAILABLE and not args.no_wandb:
        wandb.init(
            project="cv-week1-cifar10",
            config={
                "epochs": args.epochs,
                "learning_rate": args.lr,
                "batch_size": args.batch_size,
                "optimizer": "SGD",
                "momentum": 0.9,
                "weight_decay": args.weight_decay,
                "architecture": "CIFAR10CNN",
                "dataset": "CIFAR-10",
            },
        )
        use_wandb = True
    else:
        use_wandb = False

    # Data
    train_loader, val_loader, test_loader = get_data_loaders(
        batch_size=args.batch_size
    )

    # Model
    model = CIFAR10CNN(num_classes=10).to(DEVICE)
    print(f"\nModel parameters: {count_parameters(model):,}")

    # Loss, optimizer, scheduler
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        model.parameters(),
        lr=args.lr,
        momentum=0.9,
        weight_decay=args.weight_decay,
    )
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.5)

    # Training history
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': [],
    }

    best_val_acc = 0.0
    start = time.time()

    for epoch in range(args.epochs):
        current_lr = optimizer.param_groups[0]['lr']
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_acc, _, _ = evaluate(model, val_loader, criterion)
        scheduler.step()

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        print(f"Epoch {epoch+1:2d}/{args.epochs} | "
              f"LR: {current_lr:.5f} | "
              f"Train: {train_loss:.4f} / {train_acc:.1f}% | "
              f"Val: {val_loss:.4f} / {val_acc:.1f}%")

        # Log to W&B
        if use_wandb:
            wandb.log({
                "epoch": epoch + 1,
                "train/loss": train_loss,
                "train/accuracy": train_acc,
                "val/loss": val_loss,
                "val/accuracy": val_acc,
                "learning_rate": current_lr,
            })

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
            }, MODEL_DIR / "best_cifar10_cnn.pt")

    elapsed = time.time() - start
    print(f"\nTraining complete in {elapsed:.1f}s")
    print(f"Best validation accuracy: {best_val_acc:.1f}%")

    # ── Final evaluation on test set ──
    print("\n--- Final Test Evaluation ---")
    checkpoint = torch.load(MODEL_DIR / "best_cifar10_cnn.pt", weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'])
    test_loss, test_acc, test_preds, test_labels = evaluate(model, test_loader, criterion)
    print(f"Test Accuracy: {test_acc:.1f}%")
    print(f"{'✅ SUCCESS' if test_acc >= 70 else '❌ BELOW 70%'}: "
          f"Target was ≥70%, got {test_acc:.1f}%")

    # Log final test metrics
    if use_wandb:
        wandb.log({"test/accuracy": test_acc, "test/loss": test_loss})
        wandb.summary["best_val_accuracy"] = best_val_acc
        wandb.summary["test_accuracy"] = test_acc

    # ── Plots ──
    plot_learning_curves(history, save_path=OUTPUT_DIR / "day5_learning_curves.png")
    plot_confusion_matrix(test_labels, test_preds, CIFAR10_CLASSES,
                          save_path=OUTPUT_DIR / "day5_confusion_matrix.png")

    if use_wandb:
        wandb.log({
            "learning_curves": wandb.Image(str(OUTPUT_DIR / "day5_learning_curves.png")),
            "confusion_matrix": wandb.Image(str(OUTPUT_DIR / "day5_confusion_matrix.png")),
        })
        wandb.finish()

    print(f"\nResults saved to: {OUTPUT_DIR}")
    print(f"Model saved to:  {MODEL_DIR / 'best_cifar10_cnn.pt'}")


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Week 1 — Train CNN on CIFAR-10")
    parser.add_argument("--epochs", type=int, default=30, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=0.01, help="Learning rate")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="Weight decay")
    parser.add_argument("--no-wandb", action="store_true", help="Disable W&B logging")
    args = parser.parse_args()
    main(args)
