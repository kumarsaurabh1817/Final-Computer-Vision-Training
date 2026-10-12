"""
Utility functions for Week 1 — plotting, metrics, etc.
Reusable across Day 3, Day 4, and Day 5 scripts.
"""

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import torch


def count_parameters(model):
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def plot_learning_curves(history, save_path=None, title_prefix="CNN"):
    """
    Plot training and validation loss/accuracy curves.

    Args:
        history: dict with keys 'train_loss', 'train_acc', 'val_loss', 'val_acc'
        save_path: Path to save the figure (optional)
        title_prefix: Prefix for plot titles
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    epochs = range(1, len(history['train_loss']) + 1)

    # Determine val key name (some scripts use 'val', others 'test')
    val_loss_key = 'val_loss' if 'val_loss' in history else 'test_loss'
    val_acc_key = 'val_acc' if 'val_acc' in history else 'test_acc'

    # Loss
    ax1.plot(epochs, history['train_loss'], 'b-o', markersize=3, label='Train')
    ax1.plot(epochs, history[val_loss_key], 'r-o', markersize=3, label='Val')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title(f'{title_prefix} — Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Accuracy
    ax2.plot(epochs, history['train_acc'], 'b-o', markersize=3, label='Train')
    ax2.plot(epochs, history[val_acc_key], 'r-o', markersize=3, label='Val')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title(f'{title_prefix} — Accuracy')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=70, color='g', linestyle='--', alpha=0.5, label='70% target')
    ax2.legend()

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    plt.close()


def plot_confusion_matrix(y_true, y_pred, class_names, save_path=None):
    """
    Plot a confusion matrix with class names.

    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        class_names: List of class name strings
        save_path: Path to save the figure (optional)
    """
    cm = confusion_matrix(y_true, y_pred)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Raw counts
    im1 = ax1.imshow(cm, interpolation='nearest', cmap='Blues')
    ax1.set_title('Confusion Matrix (Counts)')
    ax1.set_xticks(range(len(class_names)))
    ax1.set_yticks(range(len(class_names)))
    ax1.set_xticklabels(class_names, rotation=45, ha='right')
    ax1.set_yticklabels(class_names)
    ax1.set_xlabel('Predicted')
    ax1.set_ylabel('True')
    plt.colorbar(im1, ax=ax1)

    # Normalized
    im2 = ax2.imshow(cm_normalized, interpolation='nearest', cmap='Blues',
                     vmin=0, vmax=1)
    ax2.set_title('Confusion Matrix (Normalized)')
    ax2.set_xticks(range(len(class_names)))
    ax2.set_yticks(range(len(class_names)))
    ax2.set_xticklabels(class_names, rotation=45, ha='right')
    ax2.set_yticklabels(class_names)
    ax2.set_xlabel('Predicted')
    ax2.set_ylabel('True')
    plt.colorbar(im2, ax=ax2)

    # Add text annotations to normalized matrix
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            color = 'white' if cm_normalized[i, j] > 0.5 else 'black'
            ax2.text(j, i, f'{cm_normalized[i, j]:.2f}',
                     ha='center', va='center', color=color, fontsize=8)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    plt.close()

    # Print classification report
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=class_names))


def visualize_predictions(model, test_loader, class_names, device, n=16, save_path=None):
    """Show predictions on random test images."""
    model.eval()
    images, labels = next(iter(test_loader))
    images, labels = images[:n].to(device), labels[:n]

    with torch.no_grad():
        outputs = model(images)
        _, preds = outputs.max(1)

    # Denormalize for display
    mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(3, 1, 1)
    std = torch.tensor([0.2470, 0.2435, 0.2616]).view(3, 1, 1)

    cols = 4
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(12, 3 * rows))

    for i, ax in enumerate(axes.flat):
        if i < n:
            img = images[i].cpu() * std + mean
            img = img.clamp(0, 1).permute(1, 2, 0).numpy()
            ax.imshow(img)

            pred_label = class_names[preds[i]]
            true_label = class_names[labels[i]]
            color = 'green' if preds[i] == labels[i] else 'red'
            ax.set_title(f"P: {pred_label}\nT: {true_label}", color=color, fontsize=10)
        ax.axis('off')

    plt.suptitle("Model Predictions (Green=Correct, Red=Wrong)", fontsize=14)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    plt.close()
