"""
Day 3: The Training Loop — Forward, Loss, Backward, Step
=========================================================

Objectives:
- Build a 2-layer MLP from scratch (no nn.Sequential shortcuts)
- Train it on CIFAR-10 or FashionMNIST
- Understand EVERY line of the training loop
- Watch accuracy climb and plot learning curves

Materials:
- 3Blue1Brown — Neural Networks (video series)
- PyTorch — Optimizing Model Parameters

⚠️ NO pretrained models — the goal is to understand the mechanics.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as T
import matplotlib.pyplot as plt
from pathlib import Path
import time

# Output directory for plots
OUTPUT_DIR = Path(__file__).parent.parent / "results"
OUTPUT_DIR.mkdir(exist_ok=True)

# Reproducibility
torch.manual_seed(42)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")


# =============================================================================
# PART 1: DATASET
# =============================================================================

def get_data(dataset_name="CIFAR10", batch_size=64):
    """
    Load CIFAR-10 (or FashionMNIST) with minimal transforms.

    CIFAR-10: 60k 32x32 color images in 10 classes (50k train, 10k test)
    FashionMNIST: 60k 28x28 grayscale images in 10 classes
    """
    if dataset_name == "CIFAR10":
        transform = T.Compose([
            T.ToTensor(),                          # (H,W,C) → (C,H,W) [0,1]
            T.Normalize((0.5, 0.5, 0.5),           # Normalize to [-1, 1]
                        (0.5, 0.5, 0.5)),
        ])
        train_set = torchvision.datasets.CIFAR10(
            root='./data', train=True, download=True, transform=transform
        )
        test_set = torchvision.datasets.CIFAR10(
            root='./data', train=False, download=True, transform=transform
        )
        input_size = 32 * 32 * 3  # Flatten for MLP
        classes = ('plane', 'car', 'bird', 'cat', 'deer',
                   'dog', 'frog', 'horse', 'ship', 'truck')
    else:
        transform = T.Compose([
            T.ToTensor(),
            T.Normalize((0.5,), (0.5,)),
        ])
        train_set = torchvision.datasets.FashionMNIST(
            root='./data', train=True, download=True, transform=transform
        )
        test_set = torchvision.datasets.FashionMNIST(
            root='./data', train=False, download=True, transform=transform
        )
        input_size = 28 * 28 * 1
        classes = ('T-shirt', 'Trouser', 'Pullover', 'Dress', 'Coat',
                   'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot')

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=2)

    print(f"\nDataset: {dataset_name}")
    print(f"  Train samples: {len(train_set)}")
    print(f"  Test samples:  {len(test_set)}")
    print(f"  Input size (flattened): {input_size}")
    print(f"  Classes: {classes}")

    return train_loader, test_loader, input_size, classes


# =============================================================================
# PART 2: MODEL — 2-LAYER MLP (no shortcuts!)
# =============================================================================

class SimpleMLP(nn.Module):
    """
    A simple 2-layer MLP (Multi-Layer Perceptron).

    Architecture:
        Input (flattened image) → Linear → ReLU → Linear → Output (10 classes)

    This is intentionally simple. On Day 4, you'll replace it with a CNN.
    """

    def __init__(self, input_size, hidden_size=512, num_classes=10):
        super().__init__()

        # Layer 1: input → hidden
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()

        # Layer 2: hidden → output
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # Flatten the image: (B, C, H, W) → (B, C*H*W)
        x = x.view(x.size(0), -1)

        # Layer 1
        x = self.fc1(x)
        x = self.relu(x)

        # Layer 2 (no activation — softmax is inside CrossEntropyLoss)
        x = self.fc2(x)

        return x


# =============================================================================
# PART 3: THE TRAINING LOOP — THE HEART OF ML
# =============================================================================

def train_one_epoch(model, train_loader, criterion, optimizer, epoch):
    """
    One epoch of training.

    THE TRAINING LOOP — understand every line:
    1. FORWARD PASS:  predictions = model(inputs)
    2. COMPUTE LOSS:  loss = criterion(predictions, labels)
    3. BACKWARD PASS: loss.backward()  — compute gradients
    4. OPTIMIZER STEP: optimizer.step() — update weights
    5. ZERO GRADIENTS: optimizer.zero_grad() — reset for next batch
    """
    model.train()  # Set model to training mode (enables dropout, batchnorm, etc.)
    running_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (inputs, labels) in enumerate(train_loader):
        # Move data to device (CPU or GPU)
        inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)

        # ─── Step 0: Zero the gradients ───
        # WHY: Gradients accumulate by default in PyTorch.
        # If we don't zero them, gradients from the previous batch
        # will be ADDED to the current batch's gradients.
        optimizer.zero_grad()

        # ─── Step 1: Forward pass ───
        # The model computes predictions for this batch.
        # inputs: (B, C, H, W) → outputs: (B, num_classes)
        outputs = model(inputs)

        # ─── Step 2: Compute loss ───
        # CrossEntropyLoss combines LogSoftmax + NLLLoss.
        # It measures how far our predictions are from the truth.
        loss = criterion(outputs, labels)

        # ─── Step 3: Backward pass (backpropagation) ───
        # Compute the gradient of the loss w.r.t. every parameter
        # that has requires_grad=True. This is the chain rule at work.
        loss.backward()

        # ─── Step 4: Optimizer step ───
        # Update each parameter using its gradient:
        #   param = param - lr * param.grad
        # (SGD does exactly this; Adam is smarter but the idea is the same)
        optimizer.step()

        # Track metrics
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        if (batch_idx + 1) % 200 == 0:
            print(f"    Batch {batch_idx+1}/{len(train_loader)} | "
                  f"Loss: {running_loss / (batch_idx + 1):.4f} | "
                  f"Acc: {100. * correct / total:.1f}%")

    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100. * correct / total
    return epoch_loss, epoch_acc


def evaluate(model, test_loader, criterion):
    """Evaluate the model on the test/validation set."""
    model.eval()  # Set model to evaluation mode (disables dropout, etc.)
    test_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():  # No need to track gradients during evaluation
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    avg_loss = test_loss / len(test_loader)
    accuracy = 100. * correct / total
    return avg_loss, accuracy


# =============================================================================
# PART 4: PUTTING IT ALL TOGETHER
# =============================================================================

def train_mlp(dataset_name="CIFAR10", epochs=20, lr=0.01, batch_size=64, hidden_size=512):
    """Full training pipeline for the MLP."""
    print("=" * 60)
    print(f"Training MLP on {dataset_name}")
    print(f"  Epochs: {epochs}, LR: {lr}, Batch size: {batch_size}")
    print(f"  Hidden size: {hidden_size}")
    print("=" * 60)

    # Data
    train_loader, test_loader, input_size, classes = get_data(dataset_name, batch_size)

    # Model
    model = SimpleMLP(input_size, hidden_size, num_classes=len(classes)).to(DEVICE)
    print(f"\nModel:\n{model}")
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")

    # Loss function
    criterion = nn.CrossEntropyLoss()

    # Optimizer
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)

    # Training history
    history = {
        'train_loss': [], 'train_acc': [],
        'test_loss': [], 'test_acc': [],
    }

    # Training loop
    start_time = time.time()
    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")
        print("-" * 40)

        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, epoch)
        test_loss, test_acc = evaluate(model, test_loader, criterion)

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['test_loss'].append(test_loss)
        history['test_acc'].append(test_acc)

        print(f"  Train — Loss: {train_loss:.4f}, Acc: {train_acc:.1f}%")
        print(f"  Test  — Loss: {test_loss:.4f}, Acc: {test_acc:.1f}%")

    elapsed = time.time() - start_time
    print(f"\nTraining complete in {elapsed:.1f}s")
    print(f"Best test accuracy: {max(history['test_acc']):.1f}%")

    # Plot learning curves
    plot_learning_curves(history, model_name=f"MLP_{dataset_name}")

    return model, history


def plot_learning_curves(history, model_name="model"):
    """Plot loss and accuracy curves."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    epochs = range(1, len(history['train_loss']) + 1)

    # Loss
    ax1.plot(epochs, history['train_loss'], 'b-', label='Train')
    ax1.plot(epochs, history['test_loss'], 'r-', label='Test')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title(f'{model_name} — Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Accuracy
    ax2.plot(epochs, history['train_acc'], 'b-', label='Train')
    ax2.plot(epochs, history['test_acc'], 'r-', label='Test')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title(f'{model_name} — Accuracy')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"day3_{model_name}_curves.png", dpi=150)
    plt.close()
    print(f"Saved: {OUTPUT_DIR / f'day3_{model_name}_curves.png'}")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    # Train the MLP
    # Start with FashionMNIST (easier/faster), then try CIFAR-10
    model, history = train_mlp(
        dataset_name="CIFAR10",  # or "FashionMNIST"
        epochs=20,
        lr=0.01,
        batch_size=64,
        hidden_size=512,
    )

    print("\n" + "=" * 60)
    print("DAY 3 COMPLETE!")
    print("=" * 60)
    print("\nKey takeaways:")
    print("  1. The training loop: forward → loss → backward → step → zero_grad")
    print("  2. model.train() vs model.eval() — important for dropout/batchnorm")
    print("  3. torch.no_grad() during evaluation saves memory")
    print("  4. MLP flattens spatial structure — CNNs will fix this (Day 4)")
    print(f"\n  MLP test accuracy: {max(history['test_acc']):.1f}%")
    print("  (An MLP typically gets ~50-55% on CIFAR-10, showing why CNNs matter)")
    print("\nNext: Day 4 — Convert this MLP into a CNN with convolutions")

    # TODO: Experiment with different hyperparameters:
    #   - Learning rate: 0.1, 0.01, 0.001
    #   - Hidden size: 256, 512, 1024
    #   - Batch size: 32, 64, 128
    #   - Add a third layer — does it help?
