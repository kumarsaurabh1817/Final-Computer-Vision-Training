# Day 4 Notes — Data Augmentation with Albumentations

## Core Concept

Data augmentation **artificially expands** the effective training set by applying random, plausible transformations to each image at training time. The model never sees the exact same image twice, which acts as a strong **regulariser** and improves generalisation — especially when data is limited.

> ⚠️ **Critical mindset**: Augmentation is NOT free accuracy. Some augmentations help, some hurt, some do nothing. The only way to know is to run a **controlled ablation study** with a no-augmentation baseline.

---

## Why Augmentation Works

```
Without augmentation:          With augmentation:
┌─────────────┐                ┌─────────────┐
│ Same image  │                │ Flipped     │ ← epoch 1
│ every epoch │                │ Rotated 15° │ ← epoch 2
│             │                │ Cropped     │ ← epoch 3
│             │                │ Color shift │ ← epoch 4
└─────────────┘                └─────────────┘
→ Overfits quickly             → Sees "new" images every epoch
→ Memorises textures           → Learns invariant features
```

**Key insight**: Augmentation teaches the model *what shouldn't matter* — if a cat flipped horizontally is still a cat, the model learns orientation invariance.

---

## torchvision Transforms vs Albumentations

| Feature | torchvision.transforms | Albumentations |
|---------|----------------------|----------------|
| **Speed** | Moderate (PIL-based) | Fast (OpenCV/NumPy-based) |
| **API** | Works on PIL Images | Works on NumPy arrays |
| **Augmentation variety** | ~30 transforms | ~70+ transforms |
| **Pixel-level + spatial** | Limited | Full support |
| **Bounding box / mask support** | No | Yes (crucial for detection/segmentation) |
| **Reproducibility** | `torch.manual_seed` | `ReplayCompose`, `random_state` |
| **Integration** | Native PyTorch | Needs wrapper (simple) |
| **When to use** | Quick prototyping, simple pipelines | Production, ablation studies, detection/segmentation |

### Why we switch to Albumentations for Day 4:
1. **Faster** — 2–5× vs torchvision for complex pipelines
2. **More transforms** — e.g. `CLAHE`, `GridDistortion`, weather effects
3. **Detection-ready** — bounding box & mask transforms (needed in Week 3)
4. **Industry standard** — Kaggle competitions, production pipelines

---

## Albumentations — Key API Patterns

### Basic Pipeline
```python
import albumentations as A
from albumentations.pytorch import ToTensorV2

transform = A.Compose([
    A.Resize(256, 256),
    A.RandomCrop(224, 224),
    A.HorizontalFlip(p=0.5),
    A.Normalize(mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225)),
    ToTensorV2(),  # NumPy → PyTorch tensor
])

# Apply to a NumPy array (HWC, uint8)
augmented = transform(image=np.array(pil_image))
tensor_image = augmented['image']  # (C, H, W) float32
```

### Important: Albumentations expects NumPy arrays!
```python
# ✅ Correct
image = np.array(Image.open(path).convert("RGB"))  # (H, W, C) uint8
augmented = transform(image=image)

# ❌ Wrong — passing a PIL Image directly
augmented = transform(image=pil_image)  # TypeError!
```

---

## Augmentation Categories

### Spatial Transforms (change geometry)
| Transform | What it does | Typical p |
|-----------|-------------|-----------|
| `HorizontalFlip` | Mirror left↔right | 0.5 |
| `VerticalFlip` | Mirror top↔bottom | 0.5 (only if vertically symmetric data) |
| `Rotate` / `RandomRotate90` | Rotate by angle | 0.5 |
| `ShiftScaleRotate` | Combined shift + scale + rotate | 0.5 |
| `RandomCrop` | Random spatial crop | Always (after resize) |
| `CenterCrop` | Deterministic center crop | Eval only |
| `ElasticTransform` | Warps the image elastically | 0.2–0.3 |
| `GridDistortion` | Grid-based distortion | 0.2–0.3 |
| `Perspective` | Random perspective change | 0.3 |

### Pixel-Level Transforms (change appearance)
| Transform | What it does | Typical p |
|-----------|-------------|-----------|
| `ColorJitter` | Brightness, contrast, saturation, hue | 0.3–0.5 |
| `RGBShift` | Random per-channel shift | 0.3 |
| `HueSaturationValue` | HSV-space jitter | 0.3 |
| `CLAHE` | Contrast-limited adaptive histogram eq. | 0.3 |
| `GaussNoise` | Additive Gaussian noise | 0.2 |
| `GaussianBlur` | Random Gaussian blur | 0.2 |
| `CoarseDropout` | Random rectangular erasure (like Cutout) | 0.3 |
| `RandomBrightnessContrast` | Random brightness + contrast | 0.3–0.5 |
| `RandomGamma` | Gamma correction | 0.3 |

### Composition Helpers
```python
# Apply exactly one of the listed transforms
A.OneOf([
    A.GaussNoise(var_limit=(10, 50)),
    A.GaussianBlur(blur_limit=7),
    A.MedianBlur(blur_limit=7),
], p=0.3)

# Apply all with individual probabilities
A.Compose([
    A.HorizontalFlip(p=0.5),  # 50% chance
    A.ColorJitter(p=0.3),     # 30% chance
])
```

---

## The Ablation Study — Methodology

### What is an ablation study?
An **ablation study** systematically removes or changes one component at a time to measure its individual effect. For augmentation:

```
Experiment Design:
─────────────────────────────────────────────────────
  Baseline     → No augmentation (just resize + normalize)
  + Spatial    → Add flips + rotations + crops
  + Color      → Add color jitter + brightness/contrast
  + Advanced   → Add CoarseDropout + noise + blur
  Full Aug     → All augmentations combined
─────────────────────────────────────────────────────

Everything else stays the SAME:
  ✓ Same backbone (ResNet-18)
  ✓ Same learning rate schedule
  ✓ Same number of epochs
  ✓ Same train/val/test split (same seed!)
  ✓ Same evaluation transforms (deterministic)
```

### Why it matters:
- You can't claim "augmentation helped" if you also changed the LR
- Each row isolates exactly one variable
- Negative results (augmentation hurts!) are equally valuable

---

## Integrating Albumentations with PyTorch

### Custom Dataset Wrapper
```python
class AlbumentationsDataset(Dataset):
    def __init__(self, image_folder_dataset, transform=None):
        self.dataset = image_folder_dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        # ImageFolder returns (PIL Image, label)
        pil_image, label = self.dataset[idx]

        # Convert PIL → NumPy for Albumentations
        image = np.array(pil_image)

        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']

        return image, label
```

> **Key**: The ImageFolder dataset must NOT have transforms itself (no `ToTensor`, no `Normalize`). Those go into the Albumentations pipeline.

---

## Evaluation Transforms — ALWAYS Deterministic

```python
eval_transform = A.Compose([
    A.Resize(256, 256),
    A.CenterCrop(224, 224),
    A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ToTensorV2(),
])

# ❌ NEVER use random augmentation during evaluation
# It makes metrics unreliable and non-reproducible
```

---

## Which Augmentations for Which Data?

| Data Domain | Good Augmentations | Bad Augmentations |
|-------------|-------------------|-------------------|
| **Natural scenes** (our dataset) | HFlip, Rotate, ColorJitter, Crop | VerticalFlip (sky doesn't go below) |
| **Medical images** | Rotate, ElasticTransform, CLAHE | Heavy color jitter (diagnostic info in color) |
| **Satellite/aerial** | Rotate90, Flip (both), Scale | Perspective (always top-down) |
| **Documents/OCR** | Slight rotate, noise, blur | Flip, heavy distortion (text becomes unreadable) |
| **Faces** | HFlip, ColorJitter, slight rotate | VerticalFlip, heavy distortion |

For **Intel Image Classification** (outdoor scenes):
- ✅ HorizontalFlip — a mountain flipped is still a mountain
- ✅ Rotation (±15°) — slight tilt is natural
- ✅ Color jitter — lighting varies in real photos
- ✅ RandomCrop — forces model to use different regions
- ⚠️ VerticalFlip — debatable for outdoor scenes (sky below?)
- ✅ CoarseDropout — forces model not to rely on single regions

---

## Observations & Takeaways
_Fill in with your actual numbers after running the ablation study_

Expected patterns based on the Intel Image Classification dataset (6 outdoor scene classes):

| Observation | Likely Result | Why |
|-------------|--------------|-----|
| Baseline acc | ~88–92% | Dataset is large (14k train), pretrained backbone |
| Spatial aug | +0.5–2% | HFlip + crop = natural outdoor scene variance |
| Color aug | +0.5–1.5% | Lighting variance in outdoor photos |
| Advanced aug | ±0.5% | Dropout/noise effect is dataset-size-dependent |
| Full aug | Best or near-best | Combined effect, may need more epochs to converge |

**What to record in your ablation table:**
1. Which experiment got the highest test accuracy?
2. Did augmentation help? (Most likely yes on this dataset)
3. Which classes benefited most? (Buildings and streets often show biggest gains)
4. Was there overfitting without augmentation? (Compare train vs val gap)

---

## albumentations 2.x API Breaking Changes (IMPORTANT)

The code uses **albumentations ≥ 2.0** which introduced breaking API renames:

### CoarseDropout (Cutout)
```python
# OLD (albumentations 1.x) — will throw TypeError in 2.x
A.CoarseDropout(
    max_holes=8,
    max_height=28,       # pixels
    max_width=28,        # pixels
    fill_value=0,        # pixel fill value
    p=0.3,
)

# NEW (albumentations 2.x) — correct
A.CoarseDropout(
    num_holes_range=(1, 8),           # range tuple
    hole_height_range=(14, 28),       # (min, max) pixels — or (0.0, 1.0) for fractions
    hole_width_range=(14, 28),
    fill=0,                           # renamed from fill_value
    p=0.3,
)
```

### GaussNoise
```python
# OLD (albumentations 1.x) — var_limit takes variance values
A.GaussNoise(var_limit=(10.0, 50.0), p=0.3)   # variance

# NEW (albumentations 2.x) — std_range takes STD DEV values
A.GaussNoise(std_range=(3.0, 7.0), p=0.3)     # std dev = sqrt(variance)
# sqrt(10) ≈ 3.16,  sqrt(50) ≈ 7.07
```

> **Remember:** When migrating old code or copying from tutorials, always check the albumentations version. The 2.x migration guide is at https://albumentations.ai/docs/migration/

---

## Questions to Investigate
- [ ] What happens with very aggressive augmentation (large rotations, heavy distortion)?
- [ ] Does CoarseDropout (Cutout) help more than other augmentations?
- [ ] How does augmentation interact with frozen vs unfrozen backbones?
- [ ] Would MixUp or CutMix (mixing two images) help further?
- [ ] At what dataset size does augmentation become unnecessary?

---

## Resources
- [Albumentations — Documentation](https://albumentations.ai/docs/)
- [Albumentations 2.0 Migration Guide](https://albumentations.ai/docs/migration/)
- [Albumentations — PyTorch Integration](https://albumentations.ai/docs/examples/pytorch_classification/)
- [Albumentations — Full List of Transforms](https://albumentations.ai/docs/getting_started/transforms_and_targets/)
- [A Survey on Image Data Augmentation (Shorten & Khoshgoftaar)](https://journalofbigdata.springeropen.com/articles/10.1186/s40537-019-0197-0)

