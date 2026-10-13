"""
Day 1: Modern CNN Architectures — ResNet, EfficientNet & Pretrained Backbones
=============================================================================

Objectives:
- Understand ResNet residual connections and why depth works
- Understand EfficientNet compound scaling
- Load pretrained models from torchvision with the new multi-weight API
- Inspect architecture: layer names, parameter counts, output shapes
- Benchmark inference speed of different backbone sizes
- Visualise first-layer conv filters and feature maps

Materials:
- torchvision Models & Pretrained Weights:
    https://pytorch.org/vision/stable/models.html
- ResNet paper (arXiv:1512.03385):
    https://arxiv.org/abs/1512.03385

Run:
    python day1_modern_architectures.py
    python day1_modern_architectures.py --backbone efficientnet_b3
"""

import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import (
    ResNet18_Weights, ResNet50_Weights, ResNet101_Weights,
    EfficientNet_B0_Weights, EfficientNet_B3_Weights,
)
import torchvision.transforms as T
from torchvision import datasets
from torch.utils.data import DataLoader
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from pathlib import Path
import time
import argparse
import urllib.request

# ── Output directory ──────────────────────────────────────────────────────────
RESULTS_DIR = Path(__file__).parent.parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =============================================================================
# 1. ARCHITECTURE OVERVIEW
# =============================================================================

BACKBONES = {
    "resnet18":       (models.resnet18,       ResNet18_Weights.DEFAULT),
    "resnet50":       (models.resnet50,       ResNet50_Weights.DEFAULT),
    "resnet101":      (models.resnet101,      ResNet101_Weights.DEFAULT),
    "efficientnet_b0":(models.efficientnet_b0, EfficientNet_B0_Weights.DEFAULT),
    "efficientnet_b3":(models.efficientnet_b3, EfficientNet_B3_Weights.DEFAULT),
}


def count_parameters(model: nn.Module) -> int:
    """Return total number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def load_backbone(name: str, pretrained: bool = True) -> nn.Module:
    """Load a torchvision backbone with (optional) pretrained weights."""
    builder, weights = BACKBONES[name]
    model = builder(weights=weights if pretrained else None)
    return model.eval().to(DEVICE)


def print_architecture_summary(model: nn.Module, name: str, input_size=(1, 3, 224, 224)):
    """
    Print a concise architecture summary — parameter count per named child
    and a single forward-pass to capture intermediate shapes.
    """
    print(f"\n{'='*60}")
    print(f"Architecture: {name.upper()}")
    print(f"{'='*60}")
    print(f"{'Module':<30} {'Parameters':>12}")
    print("-" * 44)

    total = 0
    for child_name, child in model.named_children():
        params = sum(p.numel() for p in child.parameters())
        total += params
        print(f"  {child_name:<28} {params:>12,}")

    print("-" * 44)
    print(f"  {'TOTAL':<28} {total:>12,}")

    # Shape trace via hooks
    shapes = {}

    def make_hook(n):
        def hook(module, inp, out):
            if isinstance(out, torch.Tensor):
                shapes[n] = tuple(out.shape)
        return hook

    handles = []
    for n, m in model.named_modules():
        handles.append(m.register_forward_hook(make_hook(n)))

    dummy = torch.zeros(input_size, device=DEVICE)
    with torch.no_grad():
        model(dummy)
    for h in handles:
        h.remove()

    # Print first-level child output shapes
    print(f"\n{'Layer':<30} {'Output Shape'}")
    print("-" * 55)
    for n in [cn for cn, _ in model.named_children()]:
        if n in shapes:
            print(f"  {n:<28} {str(shapes[n])}")
    print()


# =============================================================================
# 2. RESIDUAL BLOCKS — ANATOMY
# =============================================================================

def explain_residual_block():
    """
    Print a plain-English explanation of the residual connection and load
    the actual BasicBlock / Bottleneck classes to inspect their code.
    """
    print("\n" + "="*60)
    print("RESIDUAL CONNECTIONS — WHY THEY WORK")
    print("="*60)
    explanation = """
Key problem (pre-ResNet): very deep networks degraded in training accuracy
because gradients vanished by the time they reached early layers.

ResNet solution — the shortcut (skip) connection:
    output = F(x) + x
where F(x) is a stack of conv→BN→ReLU layers.

Benefits:
  1. Gradient highway: gradients flow back directly through the '+',
     bypassing the conv stack → no vanishing gradient for the identity path.
  2. Residual learning: layers only need to learn the *difference* (residual)
     from the identity, which is easier to optimise near zero.
  3. Ensemble effect: each block can be 'skipped' → the network behaves like
     an ensemble of many shallower sub-networks.

Two block variants:
  • BasicBlock  (ResNet-18/34) — two 3×3 convs, fewer FLOPs
  • Bottleneck  (ResNet-50/101/152) — 1×1 → 3×3 → 1×1, more parameters
    compressed through the '3×3' bottleneck → more efficient at depth.

Dimension mismatch: when spatial size or channel count changes,
the shortcut uses a 1×1 conv (projection) to match dimensions.
"""
    print(explanation)

    # Show the actual PyTorch source of BasicBlock
    from torchvision.models.resnet import BasicBlock, Bottleneck
    print("BasicBlock parameter count (3, 64→64):",
          count_parameters(BasicBlock(64, 64)))
    print("Bottleneck  parameter count (64, 64→256):",
          count_parameters(Bottleneck(256, 64)))


# =============================================================================
# 3. EFFICIENTNET — COMPOUND SCALING
# =============================================================================

def explain_efficientnet():
    """Print a concise explanation of compound scaling."""
    print("\n" + "="*60)
    print("EFFICIENTNET — COMPOUND SCALING")
    print("="*60)
    explanation = """
EfficientNet (Tan & Le, 2019) starts from a small baseline (B0) found by
Neural Architecture Search, then scales three dimensions together using
a single compound coefficient φ:

    depth    d = α^φ
    width    w = β^φ
    resolution r = γ^φ

where α·β²·γ² ≈ 2 (double FLOPs per φ step).

Why balance all three?
  • Width alone → wider features, limited depth for complex patterns.
  • Depth alone → richer features, limited resolution input.
  • Resolution alone → finer detail, no extra capacity to exploit it.
  Compound scaling keeps them balanced → better accuracy/efficiency trade-off.

B0 (baseline) → B7 (largest):
  B0: 5.3 M params,  224 px, ~0.39 GFLOPs
  B3: 12 M params,   300 px, ~1.8  GFLOPs
  B7: 66 M params,   600 px, ~37   GFLOPs
"""
    print(explanation)


# =============================================================================
# 4. INFERENCE BENCHMARK
# =============================================================================

def benchmark_inference(backbones=("resnet18", "resnet50", "efficientnet_b0", "efficientnet_b3"),
                         n_runs=50, batch_size=16):
    """
    Run a timing benchmark: how many ms per-image does each backbone take?
    """
    print("\n" + "="*60)
    print(f"INFERENCE BENCHMARK  (batch={batch_size}, runs={n_runs}, device={DEVICE})")
    print("="*60)
    print(f"  {'Model':<20} {'Params (M)':>12} {'Latency (ms/img)':>18} {'Throughput (img/s)':>20}")
    print("  " + "-"*72)

    results = {}
    for name in backbones:
        model = load_backbone(name)
        # Determine expected input size
        if "efficientnet_b3" in name:
            img_size = 300
        else:
            img_size = 224

        dummy = torch.randn(batch_size, 3, img_size, img_size, device=DEVICE)

        # Warm-up
        with torch.no_grad():
            for _ in range(5):
                _ = model(dummy)

        if DEVICE.type == "cuda":
            torch.cuda.synchronize()

        t0 = time.perf_counter()
        with torch.no_grad():
            for _ in range(n_runs):
                _ = model(dummy)
        if DEVICE.type == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0

        total_images = n_runs * batch_size
        ms_per_img = elapsed / total_images * 1000
        throughput = total_images / elapsed
        params_m = count_parameters(model) / 1e6

        results[name] = {
            "params_m": params_m,
            "ms_per_img": ms_per_img,
            "throughput": throughput,
        }
        print(f"  {name:<20} {params_m:>12.1f} {ms_per_img:>18.2f} {throughput:>20.0f}")
        del model

    return results


# =============================================================================
# 5. VISUALISE FILTERS & FEATURE MAPS
# =============================================================================

def get_sample_image() -> torch.Tensor:
    """
    Download a single CIFAR-10 image resized to 224×224 for visualisation.
    Returns a (1, 3, 224, 224) tensor.
    """
    transform = T.Compose([
        T.Resize(224),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    # Use CIFAR-10 test set — just grab the first image
    dataset = datasets.CIFAR10(
        root=str(Path(__file__).parent.parent / "data"),
        train=False, download=True, transform=transform,
    )
    img, label = dataset[0]
    print(f"Sample image label: {dataset.classes[label]}")
    return img.unsqueeze(0).to(DEVICE)


def visualise_first_layer_filters(model: nn.Module, model_name: str):
    """
    Plot the first convolutional layer's filters as RGB images.
    Works for both ResNet (model.conv1) and EfficientNet (first conv).
    """
    # Get first conv weights
    if hasattr(model, "conv1"):              # ResNet
        filters = model.conv1.weight.data.cpu()
    else:                                    # EfficientNet — features[0][0]
        try:
            filters = model.features[0][0].weight.data.cpu()
        except (IndexError, TypeError, AttributeError):
            print("  Could not locate first conv layer for filter viz.")
            return

    # Normalise to [0, 1]
    f_min, f_max = filters.min(), filters.max()
    filters = (filters - f_min) / (f_max - f_min + 1e-8)

    n_filters = min(filters.shape[0], 32)
    cols = 8
    rows = (n_filters + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.5, rows * 1.5))
    fig.suptitle(f"{model_name} — First-Layer Conv Filters", fontsize=12)

    for i, ax in enumerate(axes.flat):
        if i < n_filters:
            f = filters[i].permute(1, 2, 0).numpy()
            ax.imshow(np.clip(f, 0, 1))
        ax.axis("off")

    plt.tight_layout()
    save_path = RESULTS_DIR / f"day1_{model_name}_filters.png"
    plt.savefig(save_path, dpi=120)
    plt.close()
    print(f"  Filter visualisation saved → {save_path}")


def visualise_feature_maps(model: nn.Module, model_name: str, img_tensor: torch.Tensor,
                            n_maps: int = 16):
    """
    Forward-hook to capture and plot intermediate feature maps from the
    first conv block output.
    """
    captured = {}

    # Choose a layer to tap — first block output
    if hasattr(model, "layer1"):            # ResNet
        target_layer = model.layer1
    else:                                   # EfficientNet
        try:
            target_layer = model.features[1]
        except (IndexError, AttributeError):
            print("  Could not locate feature map layer.")
            return

    def hook(module, inp, out):
        captured["feat"] = out.detach().cpu()

    handle = target_layer.register_forward_hook(hook)
    with torch.no_grad():
        model(img_tensor)
    handle.remove()

    feat = captured["feat"][0]              # (C, H, W)
    n_maps = min(n_maps, feat.shape[0])

    cols = 8
    rows = (n_maps + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.5, rows * 1.5))
    fig.suptitle(f"{model_name} — Feature Maps (first block)", fontsize=12)

    for i, ax in enumerate(axes.flat):
        if i < n_maps:
            fmap = feat[i].numpy()
            ax.imshow(fmap, cmap="viridis")
        ax.axis("off")

    plt.tight_layout()
    save_path = RESULTS_DIR / f"day1_{model_name}_feature_maps.png"
    plt.savefig(save_path, dpi=120)
    plt.close()
    print(f"  Feature map visualisation saved → {save_path}")


# =============================================================================
# 6. IMAGENET INFERENCE DEMO
# =============================================================================

IMAGENET_LABELS_URL = (
    "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
)


def load_imagenet_labels() -> list[str]:
    """Download (or read cached) the 1000 ImageNet class labels."""
    cache = RESULTS_DIR / "imagenet_classes.txt"
    if not cache.exists():
        print("  Downloading ImageNet class labels …")
        urllib.request.urlretrieve(IMAGENET_LABELS_URL, cache)
    with open(cache) as f:
        return [line.strip() for line in f.readlines()]


def run_imagenet_inference(model: nn.Module, img_tensor: torch.Tensor,
                           model_name: str, top_k: int = 5):
    """Run inference on a single image, print top-k predictions."""
    labels = load_imagenet_labels()
    with torch.no_grad():
        logits = model(img_tensor)
    probs = logits.softmax(dim=-1)[0]
    top_probs, top_ids = probs.topk(top_k)

    print(f"\n  Top-{top_k} ImageNet predictions for {model_name}:")
    for prob, idx in zip(top_probs, top_ids):
        print(f"    {labels[idx.item()]:<40} {prob.item()*100:5.1f}%")


# =============================================================================
# MAIN
# =============================================================================

def main(args):
    print("\n" + "█"*60)
    print("  Week 2 · Day 1 — Modern CNN Architectures")
    print("█"*60)
    print(f"  Device: {DEVICE}\n")

    # ── 1. Explain residual connections & EfficientNet ─────────────────────
    explain_residual_block()
    explain_efficientnet()

    # ── 2. Architecture summaries ─────────────────────────────────────────
    for name in args.backbones:
        model = load_backbone(name)
        print_architecture_summary(model, name)

    # ── 3. Benchmark ──────────────────────────────────────────────────────
    bench_results = benchmark_inference(
        backbones=args.backbones,
        n_runs=args.bench_runs,
        batch_size=args.bench_batch,
    )

    # ── 4. Visualise filters and feature maps ─────────────────────────────
    print("\n" + "="*60)
    print("FILTER & FEATURE MAP VISUALISATION")
    print("="*60)
    img_tensor = get_sample_image()

    for name in args.backbones[:2]:   # visualise first two to keep it manageable
        print(f"\n  → {name}")
        model = load_backbone(name)
        visualise_first_layer_filters(model, name)
        visualise_feature_maps(model, name, img_tensor)
        run_imagenet_inference(model, img_tensor, name)
        del model

    # ── 5. Summary table ─────────────────────────────────────────────────
    print("\n" + "="*60)
    print("ARCHITECTURE COMPARISON SUMMARY")
    print("="*60)
    print(f"  {'Model':<20} {'Params (M)':>12} {'ms/img':>10} {'img/s':>10}  Notes")
    print("  " + "-"*70)
    notes = {
        "resnet18":       "Lightweight, great baseline",
        "resnet50":       "Strong general-purpose backbone",
        "resnet101":      "High accuracy, heavier",
        "efficientnet_b0":"Efficient baseline",
        "efficientnet_b3":"Best acc/param balance (Day 3 pick)",
    }
    for name, r in bench_results.items():
        note = notes.get(name, "")
        print(f"  {name:<20} {r['params_m']:>12.1f} {r['ms_per_img']:>10.2f}"
              f" {r['throughput']:>10.0f}  {note}")

    print(f"\n✅ Day 1 complete.  Results saved to: {RESULTS_DIR}")
    print("   Next: Day 2 — Transfer Learning (freeze vs fine-tune)")


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Week 2 Day 1 — Modern CNN Architectures")
    parser.add_argument(
        "--backbones", nargs="+",
        default=["resnet18", "resnet50", "efficientnet_b0", "efficientnet_b3"],
        choices=list(BACKBONES.keys()),
        help="Which backbones to compare",
    )
    parser.add_argument("--backbone", type=str, default=None,
                        help="Single backbone shortcut (overrides --backbones)")
    parser.add_argument("--bench-runs", type=int, default=50,
                        help="Number of timing runs per backbone")
    parser.add_argument("--bench-batch", type=int, default=16,
                        help="Batch size used in the benchmark")
    args = parser.parse_args()

    if args.backbone:
        args.backbones = [args.backbone]

    main(args)
