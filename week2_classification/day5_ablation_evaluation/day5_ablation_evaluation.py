"""Week 2 Day 5: final classification ablation and error analysis.

Runs three ResNet-18 experiments on Intel Image Classification:

1. scratch_no_aug: random initialization, deterministic training transform.
2. pretrained_no_aug: ImageNet initialization, deterministic training transform.
3. pretrained_aug: ImageNet initialization, augmented training transform.

The controlled augmentation comparison is (3) versus (2). The transfer-learning
comparison is (2) versus (1). Results include accuracy, macro/weighted F1,
per-class metrics, normalized confusion matrices, curves, checkpoints, JSON,
CSV, and a generated Markdown interpretation.

Examples:
    # Use an existing Intel dataset directory.
    python day5_ablation_evaluation.py --data-dir /path/to/intel-image-classification

    # Download the public dataset automatically with KaggleHub.
    python day5_ablation_evaluation.py --auto-download

    # Fast pipeline check only; do not report these as final metrics.
    python day5_ablation_evaluation.py --auto-download --quick
"""

from __future__ import annotations

import argparse
import gc
import json
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import albumentations as A
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from albumentations.pytorch import ToTensorV2
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import ImageFolder
from torchvision.models import ResNet18_Weights, resnet18


SEED = 42
EXPECTED_CLASSES = {
    "buildings", "forest", "glacier", "mountain", "sea", "street"
}
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@dataclass(frozen=True)
class Experiment:
    """One training treatment in the final comparison."""

    name: str
    pretrained: bool
    augmented: bool
    head_lr: float
    backbone_lr: float


EXPERIMENTS = (
    Experiment("scratch_no_aug", False, False, 1e-3, 1e-3),
    Experiment("pretrained_no_aug", True, False, 1e-3, 1e-5),
    Experiment("pretrained_aug", True, True, 1e-3, 1e-5),
)


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch for repeatable comparisons."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def seed_worker(worker_id: int) -> None:
    """Give every DataLoader worker a deterministic NumPy/Python seed."""

    del worker_id
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def locate_split(root: Path, split_name: str) -> Path:
    """Find seg_train or seg_test across common Kaggle extraction layouts."""

    candidates = [root / split_name / split_name, root / split_name]
    candidates.extend(root.rglob(split_name))
    seen: set[Path] = set()
    for candidate in candidates:
        candidate = candidate.resolve()
        if candidate in seen or not candidate.is_dir():
            continue
        seen.add(candidate)
        child_dirs = {item.name for item in candidate.iterdir() if item.is_dir()}
        if EXPECTED_CLASSES.issubset(child_dirs):
            return candidate
    raise FileNotFoundError(
        f"Could not locate {split_name!r} class folders below {root}. "
        "Expected buildings, forest, glacier, mountain, sea, and street."
    )


def resolve_dataset(data_dir: str | None, auto_download: bool) -> tuple[Path, Path]:
    """Resolve a local dataset or download the public Kaggle dataset."""

    if data_dir:
        root = Path(data_dir).expanduser().resolve()
        if not root.exists():
            raise FileNotFoundError(f"Dataset directory does not exist: {root}")
    elif auto_download:
        try:
            import kagglehub
        except ImportError as exc:
            raise RuntimeError(
                "Automatic download needs kagglehub: pip install kagglehub"
            ) from exc
        root = Path(
            kagglehub.dataset_download("puneet6060/intel-image-classification")
        )
    else:
        default_root = Path(__file__).resolve().parent.parent / "data" / (
            "intel-image-classification"
        )
        if not default_root.exists():
            raise FileNotFoundError(
                "Dataset not found. Pass --data-dir PATH or --auto-download. "
                f"Checked default path: {default_root}"
            )
        root = default_root

    train_dir = locate_split(root, "seg_train")
    test_dir = locate_split(root, "seg_test")
    return train_dir, test_dir


def build_transforms(image_size: int) -> tuple[A.Compose, A.Compose]:
    """Return deterministic control/eval and augmented training pipelines."""

    no_aug = A.Compose(
        [
            A.Resize(256, 256),
            A.CenterCrop(image_size, image_size),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2(),
        ]
    )
    augmented = A.Compose(
        [
            A.Resize(256, 256),
            A.RandomResizedCrop(
                size=(image_size, image_size),
                scale=(0.75, 1.0),
                ratio=(0.9, 1.1),
                p=1.0,
            ),
            A.HorizontalFlip(p=0.5),
            A.Affine(
                scale=(0.9, 1.1),
                translate_percent=(-0.05, 0.05),
                rotate=(-15, 15),
                shear=(-5, 5),
                p=0.5,
            ),
            A.OneOf(
                [
                    A.RandomBrightnessContrast(0.2, 0.2),
                    A.HueSaturationValue(10, 15, 10),
                ],
                p=0.5,
            ),
            A.OneOf(
                [
                    A.GaussianBlur(blur_limit=(3, 5)),
                    A.GaussNoise(std_range=(0.01, 0.04)),
                ],
                p=0.2,
            ),
            A.CoarseDropout(
                num_holes_range=(1, 4),
                hole_height_range=(0.05, 0.15),
                hole_width_range=(0.05, 0.15),
                fill="random_uniform",
                p=0.25,
            ),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2(),
        ]
    )
    return no_aug, augmented


class IndexedAlbumentationsDataset(Dataset):
    """Apply Albumentations to selected ImageFolder samples."""

    def __init__(
        self,
        folder_dataset: ImageFolder,
        indices: np.ndarray,
        transform: A.Compose,
    ) -> None:
        self.samples = [folder_dataset.samples[int(index)] for index in indices]
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        path, label = self.samples[index]
        with Image.open(path) as image:
            image_array = np.asarray(image.convert("RGB"))
        tensor = self.transform(image=image_array)["image"]
        return tensor, label


def stratified_subset(
    indices: np.ndarray,
    targets: np.ndarray,
    num_classes: int,
    per_class: int,
    seed: int,
) -> np.ndarray:
    """Create a small balanced subset for --quick smoke tests."""

    rng = np.random.default_rng(seed)
    selected: list[int] = []
    for class_id in range(num_classes):
        choices = indices[targets[indices] == class_id]
        count = min(per_class, len(choices))
        selected.extend(rng.choice(choices, count, replace=False).tolist())
    return np.asarray(selected, dtype=np.int64)


def prepare_data(
    train_dir: Path,
    test_dir: Path,
    validation_fraction: float,
    quick: bool,
    seed: int,
) -> tuple[ImageFolder, ImageFolder, np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """Build a fixed stratified split while preserving the supplied test set."""

    raw_train = ImageFolder(str(train_dir))
    raw_test = ImageFolder(str(test_dir))
    if raw_train.classes != raw_test.classes:
        raise ValueError("Train and test class mappings differ.")

    all_indices = np.arange(len(raw_train))
    train_targets = np.asarray(raw_train.targets)
    train_indices, validation_indices = train_test_split(
        all_indices,
        test_size=validation_fraction,
        random_state=seed,
        stratify=train_targets,
    )
    test_indices = np.arange(len(raw_test))

    if quick:
        num_classes = len(raw_train.classes)
        train_indices = stratified_subset(
            train_indices, train_targets, num_classes, 80, seed
        )
        validation_indices = stratified_subset(
            validation_indices, train_targets, num_classes, 30, seed
        )
        test_indices = stratified_subset(
            test_indices,
            np.asarray(raw_test.targets),
            num_classes,
            30,
            seed,
        )

    return (
        raw_train,
        raw_test,
        train_indices,
        validation_indices,
        test_indices,
        raw_train.classes,
    )


def make_loaders(
    raw_train: ImageFolder,
    raw_test: ImageFolder,
    train_indices: np.ndarray,
    validation_indices: np.ndarray,
    test_indices: np.ndarray,
    train_transform: A.Compose,
    eval_transform: A.Compose,
    batch_size: int,
    num_workers: int,
    seed: int,
) -> dict[str, DataLoader]:
    """Create fresh deterministic loaders for one experiment."""

    train_transform.set_random_seed(seed)
    eval_transform.set_random_seed(seed)
    datasets = {
        "train": IndexedAlbumentationsDataset(
            raw_train, train_indices, train_transform
        ),
        "val": IndexedAlbumentationsDataset(
            raw_train, validation_indices, eval_transform
        ),
        "test": IndexedAlbumentationsDataset(raw_test, test_indices, eval_transform),
    }
    generator = torch.Generator().manual_seed(seed)
    common: dict[str, Any] = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": torch.cuda.is_available(),
        "worker_init_fn": seed_worker,
        "persistent_workers": num_workers > 0,
    }
    return {
        "train": DataLoader(
            datasets["train"], shuffle=True, generator=generator, **common
        ),
        "val": DataLoader(datasets["val"], shuffle=False, **common),
        "test": DataLoader(datasets["test"], shuffle=False, **common),
    }


def build_model(pretrained: bool, num_classes: int) -> nn.Module:
    """Create ResNet-18 with a new task-specific classifier."""

    weights = ResNet18_Weights.DEFAULT if pretrained else None
    model = resnet18(weights=weights)
    input_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(p=0.3), nn.Linear(input_features, num_classes)
    )
    return model.to(DEVICE)


def make_optimizer(
    model: nn.Module,
    experiment: Experiment,
    weight_decay: float,
) -> optim.Optimizer:
    """Use differential learning rates only for pretrained weights."""

    if not experiment.pretrained:
        return optim.AdamW(
            model.parameters(), lr=experiment.head_lr, weight_decay=weight_decay
        )

    head_ids = {id(parameter) for parameter in model.fc.parameters()}
    backbone_parameters = [
        parameter for parameter in model.parameters() if id(parameter) not in head_ids
    ]
    return optim.AdamW(
        [
            {"params": backbone_parameters, "lr": experiment.backbone_lr},
            {"params": model.fc.parameters(), "lr": experiment.head_lr},
        ],
        weight_decay=weight_decay,
    )


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer | None = None,
) -> dict[str, Any]:
    """Train for one epoch or evaluate when optimizer is None."""

    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total_correct = 0
    total = 0
    predictions: list[int] = []
    targets: list[int] = []
    context = torch.enable_grad() if training else torch.inference_mode()

    with context:
        for images, batch_targets in loader:
            images = images.to(DEVICE, non_blocking=True)
            batch_targets = batch_targets.to(DEVICE, non_blocking=True)
            if training:
                optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, batch_targets)
            if training:
                loss.backward()
                optimizer.step()

            batch_predictions = logits.argmax(dim=1)
            total_loss += loss.item() * len(batch_targets)
            total_correct += (batch_predictions == batch_targets).sum().item()
            total += len(batch_targets)
            if not training:
                predictions.extend(batch_predictions.cpu().tolist())
                targets.extend(batch_targets.cpu().tolist())

    return {
        "loss": total_loss / total,
        "accuracy": total_correct / total,
        "predictions": predictions,
        "targets": targets,
    }


def train_experiment(
    experiment: Experiment,
    data: tuple[ImageFolder, ImageFolder, np.ndarray, np.ndarray, np.ndarray],
    class_names: list[str],
    no_aug_transform: A.Compose,
    aug_transform: A.Compose,
    output_dir: Path,
    epochs: int,
    patience: int,
    batch_size: int,
    num_workers: int,
    weight_decay: float,
    seed: int,
) -> dict[str, Any]:
    """Train, select by validation accuracy, then evaluate the test set once."""

    raw_train, raw_test, train_indices, validation_indices, test_indices = data
    seed_everything(seed)
    loaders = make_loaders(
        raw_train=raw_train,
        raw_test=raw_test,
        train_indices=train_indices,
        validation_indices=validation_indices,
        test_indices=test_indices,
        train_transform=aug_transform if experiment.augmented else no_aug_transform,
        eval_transform=no_aug_transform,
        batch_size=batch_size,
        num_workers=num_workers,
        seed=seed,
    )
    model = build_model(experiment.pretrained, len(class_names))
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = make_optimizer(model, experiment, weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=1e-6
    )
    history: dict[str, list[float]] = {
        "train_loss": [],
        "train_accuracy": [],
        "val_loss": [],
        "val_accuracy": [],
    }
    best_validation_accuracy = -1.0
    stale_epochs = 0
    model_dir = output_dir / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = model_dir / f"{experiment.name}.pt"
    started = time.perf_counter()

    for epoch in range(1, epochs + 1):
        train_metrics = run_epoch(model, loaders["train"], criterion, optimizer)
        validation_metrics = run_epoch(model, loaders["val"], criterion)
        scheduler.step()
        for split, metrics in (
            ("train", train_metrics),
            ("val", validation_metrics),
        ):
            history[f"{split}_loss"].append(metrics["loss"])
            history[f"{split}_accuracy"].append(metrics["accuracy"])

        print(
            f"{experiment.name:20s} epoch {epoch:02d}/{epochs} | "
            f"train={train_metrics['accuracy']:.3f} | "
            f"val={validation_metrics['accuracy']:.3f}"
        )
        if validation_metrics["accuracy"] > best_validation_accuracy + 1e-6:
            best_validation_accuracy = validation_metrics["accuracy"]
            stale_epochs = 0
            torch.save(model.state_dict(), checkpoint)
        else:
            stale_epochs += 1
            if stale_epochs >= patience:
                print(f"Early stopping {experiment.name} at epoch {epoch}.")
                break

    model.load_state_dict(
        torch.load(checkpoint, map_location=DEVICE, weights_only=True)
    )
    test_metrics = run_epoch(model, loaders["test"], criterion)
    report = classification_report(
        test_metrics["targets"],
        test_metrics["predictions"],
        labels=list(range(len(class_names))),
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )
    result = {
        **asdict(experiment),
        "epochs_completed": len(history["val_accuracy"]),
        "best_val_accuracy": best_validation_accuracy,
        "test_accuracy": test_metrics["accuracy"],
        "macro_f1": report["macro avg"]["f1-score"],
        "weighted_f1": report["weighted avg"]["f1-score"],
        "training_seconds": time.perf_counter() - started,
        "history": history,
        "classification_report": report,
        "confusion_matrix": confusion_matrix(
            test_metrics["targets"],
            test_metrics["predictions"],
            labels=list(range(len(class_names))),
        ).tolist(),
    }

    del model, optimizer, loaders
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result


def save_summary(results: list[dict[str, Any]], output_dir: Path) -> pd.DataFrame:
    """Save the experiment-level ablation table."""

    rows = [
        {
            "experiment": result["name"],
            "pretrained": result["pretrained"],
            "augmented": result["augmented"],
            "epochs": result["epochs_completed"],
            "best_val_accuracy": result["best_val_accuracy"],
            "test_accuracy": result["test_accuracy"],
            "macro_f1": result["macro_f1"],
            "weighted_f1": result["weighted_f1"],
            "training_minutes": result["training_seconds"] / 60,
        }
        for result in results
    ]
    frame = pd.DataFrame(rows)
    control_accuracy = frame.set_index("experiment").loc[
        "pretrained_no_aug", "test_accuracy"
    ]
    frame["delta_vs_pretrained_no_aug"] = (
        frame["test_accuracy"] - control_accuracy
    )
    frame.to_csv(output_dir / "ablation_results.csv", index=False)
    return frame


def plot_curves(results: list[dict[str, Any]], output_dir: Path) -> None:
    """Plot training and validation accuracy/loss."""

    figure, axes = plt.subplots(1, 2, figsize=(14, 5))
    for result in results:
        epochs = range(1, len(result["history"]["val_accuracy"]) + 1)
        axes[0].plot(
            epochs, result["history"]["train_accuracy"], "--", alpha=0.55
        )
        axes[0].plot(
            epochs,
            result["history"]["val_accuracy"],
            label=result["name"],
        )
        axes[1].plot(
            epochs, result["history"]["train_loss"], "--", alpha=0.55
        )
        axes[1].plot(
            epochs, result["history"]["val_loss"], label=result["name"]
        )
    axes[0].set(
        title="Accuracy: validation solid / train dashed",
        xlabel="Epoch",
        ylabel="Accuracy",
    )
    axes[1].set(
        title="Loss: validation solid / train dashed",
        xlabel="Epoch",
        ylabel="Cross-entropy",
    )
    for axis in axes:
        axis.grid(alpha=0.25)
        axis.legend()
    figure.tight_layout()
    figure.savefig(output_dir / "learning_curves.png", dpi=160, bbox_inches="tight")
    plt.close(figure)


def save_class_analysis(
    results: list[dict[str, Any]],
    class_names: list[str],
    output_dir: Path,
) -> pd.DataFrame:
    """Save normalized confusion matrices and per-class metrics."""

    figure, axes = plt.subplots(1, len(results), figsize=(20, 5.5))
    metric_frames: list[pd.DataFrame] = []
    for axis, result in zip(axes, results):
        matrix = np.asarray(result["confusion_matrix"])
        normalized = matrix / np.maximum(matrix.sum(axis=1, keepdims=True), 1)
        sns.heatmap(
            normalized,
            annot=True,
            fmt=".2f",
            cmap="Blues",
            vmin=0,
            vmax=1,
            xticklabels=class_names,
            yticklabels=class_names,
            ax=axis,
            cbar=False,
        )
        axis.set(title=result["name"], xlabel="Predicted", ylabel="True")
        report_frame = pd.DataFrame(result["classification_report"]).T.loc[
            class_names, ["precision", "recall", "f1-score", "support"]
        ]
        report_frame.insert(0, "class", report_frame.index)
        report_frame.insert(0, "experiment", result["name"])
        metric_frames.append(report_frame.reset_index(drop=True))

    figure.tight_layout()
    figure.savefig(
        output_dir / "confusion_matrices_normalized.png",
        dpi=180,
        bbox_inches="tight",
    )
    plt.close(figure)
    metrics = pd.concat(metric_frames, ignore_index=True)
    metrics.to_csv(output_dir / "per_class_metrics.csv", index=False)
    return metrics


def write_report(
    results: list[dict[str, Any]],
    class_names: list[str],
    output_dir: Path,
    seed: int,
) -> str:
    """Generate conclusions directly from measured metrics."""

    by_name = {result["name"]: result for result in results}
    transfer_delta = (
        by_name["pretrained_no_aug"]["test_accuracy"]
        - by_name["scratch_no_aug"]["test_accuracy"]
    )
    augmentation_delta = (
        by_name["pretrained_aug"]["test_accuracy"]
        - by_name["pretrained_no_aug"]["test_accuracy"]
    )
    if augmentation_delta > 0:
        direction = "improved"
    elif augmentation_delta < 0:
        direction = "reduced"
    else:
        direction = "did not change"
    best = max(results, key=lambda result: result["test_accuracy"])
    augmented_report = pd.DataFrame(
        by_name["pretrained_aug"]["classification_report"]
    ).T.loc[class_names]
    weakest_class = augmented_report["f1-score"].idxmin()
    strongest_class = augmented_report["f1-score"].idxmax()

    report = f"""# Week 2 Final Results — Image Classification

## Experimental setup

- Dataset: Intel Image Classification ({len(class_names)} classes)
- Backbone: ResNet-18
- Selection: best validation checkpoint; test evaluated after selection
- Seed: {seed}

## Main findings

- Transfer learning changed test accuracy by {transfer_delta:+.2%} versus scratch.
- Augmentation {direction} test accuracy by {augmentation_delta:+.2%} versus the
  pretrained no-augmentation control.
- Best configuration: `{best['name']}` at {best['test_accuracy']:.2%} test
  accuracy and {best['macro_f1']:.3f} macro F1.
- For the augmented model, `{strongest_class}` had the strongest F1 and
  `{weakest_class}` had the weakest F1.

## Interpretation

The transfer-learning and augmentation deltas answer different questions. The
controlled augmentation comparison keeps the initialization type, split, model,
optimizer, learning rates, epoch budget, checkpoint rule, and evaluation pipeline
fixed. A negative augmentation result is valid and may indicate an overly strong
policy, domain mismatch, or an insufficient training budget.
"""
    (output_dir / "final_results.md").write_text(report, encoding="utf-8")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Week 2 Day 5 — final augmentation ablation and evaluation"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Intel dataset root; pass the directory containing seg_train/seg_test",
    )
    parser.add_argument(
        "--auto-download",
        action="store_true",
        help="Download puneet6060/intel-image-classification with KaggleHub",
    )
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "outputs",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="One-epoch balanced smoke test; not valid for final metrics",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.data_dir and args.auto_download:
        raise ValueError("Use either --data-dir or --auto-download, not both.")
    if args.epochs < 1 or args.batch_size < 1 or args.num_workers < 0:
        raise ValueError("epochs/batch-size must be positive; workers cannot be negative.")
    if not 0 < args.val_fraction < 1:
        raise ValueError("--val-fraction must be strictly between 0 and 1.")
    if args.patience < 1:
        raise ValueError("--patience must be at least 1.")


def main() -> None:
    args = parse_args()
    validate_args(args)
    if args.quick:
        args.epochs = 1
        args.patience = 1
        print("QUICK MODE: pipeline verification only; do not report these metrics.")
    if DEVICE.type != "cuda":
        print("WARNING: CUDA is unavailable; full training will be slow.")

    seed_everything(args.seed)
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    train_dir, test_dir = resolve_dataset(args.data_dir, args.auto_download)
    prepared = prepare_data(
        train_dir,
        test_dir,
        validation_fraction=args.val_fraction,
        quick=args.quick,
        seed=args.seed,
    )
    (
        raw_train,
        raw_test,
        train_indices,
        validation_indices,
        test_indices,
        class_names,
    ) = prepared
    print(f"Device: {DEVICE}")
    print(f"Classes: {class_names}")
    print(
        f"Samples — train: {len(train_indices):,}, "
        f"validation: {len(validation_indices):,}, test: {len(test_indices):,}"
    )

    no_aug_transform, aug_transform = build_transforms(args.image_size)
    data = (
        raw_train,
        raw_test,
        train_indices,
        validation_indices,
        test_indices,
    )
    results: list[dict[str, Any]] = []
    for experiment in EXPERIMENTS:
        print("\n" + "=" * 80)
        print(f"Starting {experiment.name}")
        results.append(
            train_experiment(
                experiment=experiment,
                data=data,
                class_names=class_names,
                no_aug_transform=no_aug_transform,
                aug_transform=aug_transform,
                output_dir=output_dir,
                epochs=args.epochs,
                patience=args.patience,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                weight_decay=args.weight_decay,
                seed=args.seed,
            )
        )

    summary = save_summary(results, output_dir)
    plot_curves(results, output_dir)
    save_class_analysis(results, class_names, output_dir)
    with (output_dir / "full_results.json").open("w", encoding="utf-8") as file:
        json.dump(
            {
                "seed": args.seed,
                "quick": args.quick,
                "epochs_requested": args.epochs,
                "class_names": class_names,
                "experiments": results,
            },
            file,
            indent=2,
        )
    report = write_report(results, class_names, output_dir, args.seed)

    print("\n" + summary.to_string(index=False))
    print("\n" + report)
    print(f"Artifacts saved to: {output_dir}")


if __name__ == "__main__":
    main()
