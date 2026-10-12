"""
Day 4: Convolutions — Filters, Feature Maps, Stride, Padding, Pooling
======================================================================

Objectives:
- Understand what a convolution does to an image
- Learn about filters/kernels, feature maps, stride, padding, pooling
- Convert the Day 3 MLP into a CNN
- Compare CNN vs MLP performance

Materials:
- Stanford CS231n — Convolutional Networks
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as T
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import time

OUTPUT_DIR = Path(__file__).parent.parent / "results"
OUTPUT_DIR.mkdir(exist_ok=True)

torch.manual_seed(42)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")


# =============================================================================
# PART 1: UNDERSTANDING CONVOLUTIONS VISUALLY
# =============================================================================

def visualize_convolution():
    """See what convolutional filters do to an image."""
    print("=" * 60)
    print("PART 1: Visualizing Convolutions")
    print("=" * 60)

    # Create a simple test image
    img = torch.zeros(1, 1, 32, 32)
    # Add some patterns
    img[0, 0, 5:27, 5:27] = 1.0      # White square
    img[0, 0, 10:22, 10:22] = 0.5    # Gray inner square
    img[0, 0, 14:18, 14:18] = 1.0    # White center

    # Define some hand-crafted kernels
    kernels = {
        'identity': torch.tensor([[0, 0, 0], [0, 1, 0], [0, 0, 0]], dtype=torch.float32),
        'edge_h': torch.tensor([[-1, -1, -1], [0, 0, 0], [1, 1, 1]], dtype=torch.float32),
        'edge_v': torch.tensor([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], dtype=torch.float32),
        'sharpen': torch.tensor([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=torch.float32),
        'blur': torch.ones(3, 3, dtype=torch.float32) / 9,
    }

    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    axes[0, 0].imshow(img[0, 0], cmap='gray')
    axes[0, 0].set_title("Original")

    for idx, (name, kernel) in enumerate(kernels.items()):
        # Reshape kernel for conv2d: (out_channels, in_channels, H, W)
        k = kernel.reshape(1, 1, 3, 3)
        # Apply convolution
        conv = nn.Conv2d(1, 1, 3, padding=1, bias=False)
        conv.weight.data = k
        with torch.no_grad():
            result = conv(img)

        row, col = divmod(idx + 1, 3)
        axes[row, col].imshow(result[0, 0].numpy(), cmap='gray')
        axes[row, col].set_title(f"{name} filter")

    for ax in axes.flat:
        ax.axis('off')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "day4_convolution_effects.png", dpi=150)
    plt.close()
    print(f"Saved: {OUTPUT_DIR / 'day4_convolution_effects.png'}")

    # TODO: Create your own kernels. What does a diagonal edge detector look like?


# =============================================================================
# PART 2: CONV2D MECHANICS
# =============================================================================

def conv_mechanics():
    """Understand stride, padding, and output size calculation."""
    print("\n" + "=" * 60)
    print("PART 2: Conv2d Mechanics")
    print("=" * 60)

    # Input: (batch=1, channels=3, height=32, width=32)
    x = torch.randn(1, 3, 32, 32)

    # Output size formula:
    # out_size = floor((in_size + 2*padding - kernel_size) / stride) + 1

    configs = [
        {"kernel_size": 3, "stride": 1, "padding": 0},  # 32 → 30
        {"kernel_size": 3, "stride": 1, "padding": 1},  # 32 → 32 (same)
        {"kernel_size": 5, "stride": 1, "padding": 2},  # 32 → 32 (same)
        {"kernel_size": 3, "stride": 2, "padding": 1},  # 32 → 16 (downsample)
        {"kernel_size": 5, "stride": 2, "padding": 0},  # 32 → 14
    ]

    print(f"Input shape: {x.shape}")
    print(f"\n{'Config':<35} {'Output Shape':<20} {'Params':>8}")
    print("-" * 65)

    for cfg in configs:
        conv = nn.Conv2d(in_channels=3, out_channels=16, **cfg)
        out = conv(x)
        params = sum(p.numel() for p in conv.parameters())
        desc = f"k={cfg['kernel_size']}, s={cfg['stride']}, p={cfg['padding']}"
        print(f"{desc:<35} {str(out.shape):<20} {params:>8,}")

    # Pooling layers
    print("\n--- Pooling ---")
    pool = nn.MaxPool2d(kernel_size=2, stride=2)
    pooled = pool(x)
    print(f"MaxPool2d(2, 2): {x.shape} → {pooled.shape}")

    avg_pool = nn.AvgPool2d(kernel_size=2, stride=2)
    avg_pooled = avg_pool(x)
    print(f"AvgPool2d(2, 2): {x.shape} → {avg_pooled.shape}")

    # TODO: Calculate output sizes by hand using the formula before running.
    #       Verify your calculations match.


# =============================================================================
# PART 3: THE CNN — CONVERTING THE MLP
# =============================================================================

class SimpleCNN(nn.Module):
    """
    A simple CNN for CIFAR-10.

    Architecture:
        Conv(3→32, 3x3) → ReLU → MaxPool
        Conv(32→64, 3x3) → ReLU → MaxPool
        Flatten → Linear(64*8*8 → 512) → ReLU → Linear(512 → 10)

    Key differences from the MLP:
    1. Preserves spatial structure (doesn't flatten the image immediately)
    2. Parameter sharing (same filter slides across the whole image)
    3. Translation invariance (can detect features anywhere in the image)
    4. Hierarchical features (early layers → edges, later → shapes, objects)
    """

    def __init__(self, num_classes=10):
        super().__init__()

        # Convolutional layers (feature extraction)
        self.features = nn.Sequential(
            # Conv Block 1: 3 → 32 channels, 32x32 → 16x16
            nn.Conv2d(3, 32, kernel_size=3, padding=1),  # (B,3,32,32) → (B,32,32,32)
            nn.ReLU(),
            nn.MaxPool2d(2, 2),                            # (B,32,32,32) → (B,32,16,16)

            # Conv Block 2: 32 → 64 channels, 16x16 → 8x8
            nn.Conv2d(32, 64, kernel_size=3, padding=1),  # (B,32,16,16) → (B,64,16,16)
            nn.ReLU(),
            nn.MaxPool2d(2, 2),                            # (B,64,16,16) → (B,64,8,8)
        )

        # Fully connected layers (classification)
        self.classifier = nn.Sequential(
            nn.Linear(64 * 8 * 8, 512),
            nn.ReLU(),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)  # Flatten: (B,64,8,8) → (B,4096)
        x = self.classifier(x)
        return x


# =============================================================================
# PART 4: TRAINING THE CNN
# =============================================================================

def get_cifar10(batch_size=64):
    """Load CIFAR-10."""
    transform = T.Compose([
        T.ToTensor(),
        T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    train_set = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform
    )
    test_set = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform
    )
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=2)
    return train_loader, test_loader


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

        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    return total_loss / len(loader), 100. * correct / total


def evaluate(model, loader, criterion):
    """Evaluate the model."""
    model.eval()
    total_loss, correct, total = 0.0, 0, 0

    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    return total_loss / len(loader), 100. * correct / total


def train_cnn(epochs=20, lr=0.01, batch_size=64):
    """Full CNN training pipeline."""
    print("\n" + "=" * 60)
    print("PART 4: Training the CNN on CIFAR-10")
    print("=" * 60)

    train_loader, test_loader = get_cifar10(batch_size)

    model = SimpleCNN(num_classes=10).to(DEVICE)
    print(f"\nModel:\n{model}")
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)

    history = {'train_loss': [], 'train_acc': [], 'test_loss': [], 'test_acc': []}

    start = time.time()
    for epoch in range(epochs):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer)
        test_loss, test_acc = evaluate(model, test_loader, criterion)

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['test_loss'].append(test_loss)
        history['test_acc'].append(test_acc)

        print(f"Epoch {epoch+1:2d}/{epochs} | "
              f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.1f}% | "
              f"Test Loss: {test_loss:.4f}, Acc: {test_acc:.1f}%")

    elapsed = time.time() - start
    print(f"\nTraining complete in {elapsed:.1f}s")
    print(f"Best test accuracy: {max(history['test_acc']):.1f}%")

    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    eps = range(1, epochs + 1)

    ax1.plot(eps, history['train_loss'], 'b-', label='Train')
    ax1.plot(eps, history['test_loss'], 'r-', label='Test')
    ax1.set_xlabel('Epoch'); ax1.set_ylabel('Loss')
    ax1.set_title('CNN — Loss'); ax1.legend(); ax1.grid(True, alpha=0.3)

    ax2.plot(eps, history['train_acc'], 'b-', label='Train')
    ax2.plot(eps, history['test_acc'], 'r-', label='Test')
    ax2.set_xlabel('Epoch'); ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('CNN — Accuracy'); ax2.legend(); ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "day4_CNN_CIFAR10_curves.png", dpi=150)
    plt.close()
    print(f"Saved: {OUTPUT_DIR / 'day4_CNN_CIFAR10_curves.png'}")

    return model, history


# =============================================================================
# PART 5: VISUALIZE LEARNED FILTERS
# =============================================================================

def visualize_filters(model):
    """Visualize the first conv layer's learned filters."""
    print("\n" + "=" * 60)
    print("PART 5: Visualize Learned Filters")
    print("=" * 60)

    # Get first conv layer weights: (32, 3, 3, 3)
    filters = model.features[0].weight.data.cpu()
    print(f"First conv layer filters shape: {filters.shape}")
    print(f"  (out_channels=32, in_channels=3, kernel_h=3, kernel_w=3)")

    # Normalize filters to [0, 1] for visualization
    f_min, f_max = filters.min(), filters.max()
    filters_norm = (filters - f_min) / (f_max - f_min)

    # Plot first 16 filters
    fig, axes = plt.subplots(4, 4, figsize=(8, 8))
    for i, ax in enumerate(axes.flat):
        if i < 16:
            # Show as RGB (3 channels)
            ax.imshow(filters_norm[i].permute(1, 2, 0).numpy())
            ax.set_title(f"Filter {i}")
        ax.axis('off')
    plt.suptitle("First Conv Layer — Learned 3×3 Filters", fontsize=14)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "day4_learned_filters.png", dpi=150)
    plt.close()
    print(f"Saved: {OUTPUT_DIR / 'day4_learned_filters.png'}")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    visualize_convolution()
    conv_mechanics()
    model, history = train_cnn(epochs=20, lr=0.01, batch_size=64)
    visualize_filters(model)

    print("\n" + "=" * 60)
    print("DAY 4 COMPLETE!")
    print("=" * 60)
    print("\nKey takeaways:")
    print("  1. Convolutions preserve spatial structure — MLP loses it by flattening")
    print("  2. Parameter sharing makes CNNs efficient (one filter scans the whole image)")
    print("  3. Pooling reduces spatial size and adds some translation invariance")
    print("  4. Early layers learn edges/textures, later layers learn shapes/objects")
    print(f"\n  CNN test accuracy: {max(history['test_acc']):.1f}%")
    print("  (Should be significantly better than the MLP from Day 3!)")
    print("\nNext: Day 5 — Polish the CNN, add validation, tune hyperparameters, log to W&B")
