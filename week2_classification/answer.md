# Week 2 — Image Classification & Transfer Learning
## Day-by-Day Objectives & How We Reached Them

---

## 📅 Day 1 — Modern CNN Architectures

### 🎯 Objectives
1. Understand **why residual connections (ResNet)** allow training very deep networks without vanishing gradients.
2. Understand **EfficientNet's compound scaling** — scaling depth, width, and resolution together.
3. Load and inspect pretrained models from `torchvision` using the **new multi-weight API**.
4. **Benchmark inference speed** of different backbone sizes.
5. **Visualise first-layer conv filters and intermediate feature maps**.

---

### 🪜 How We Reached There — Step by Step

#### Step 1 — Understood the Core Problem (Why Plain Deep Nets Fail)
- Very deep networks (e.g., 56-layer plain CNNs) performed *worse* than shallower ones due to vanishing gradients.
- Gradients shrank to near-zero as they flowed back through many layers, causing parameters to stop learning.

#### Step 2 — Learned the ResNet Solution: Skip (Residual) Connections
- The key formula: `output = F(x) + x`
- The shortcut adds the input `x` directly to the output, creating a **gradient highway** that bypasses the conv stack.
- Layers only need to learn the *residual* (the difference from identity), which is much easier to optimise.
- **BasicBlock** (ResNet-18/34): two 3×3 convs — fewer params, fast on CPU.
- **Bottleneck** (ResNet-50/101/152): 1×1 → 3×3 → 1×1 — better GPU utilisation.

#### Step 3 — Understood EfficientNet's Compound Scaling
- Instead of scaling only one dimension (depth OR width OR resolution), EfficientNet scales all three together using a **single compound coefficient φ**.
- This avoids diminishing returns from scaling a single dimension.
- B0 = NAS-found baseline; B3 = best param-efficiency for most custom tasks.

#### Step 4 — Loaded Pretrained Backbones with the New torchvision API
```python
from torchvision.models import resnet50, ResNet50_Weights

# New (>= 0.13) multi-weight API — always use this
model = resnet50(weights=ResNet50_Weights.DEFAULT)
```
Loaded: ResNet-18, ResNet-50, ResNet-101, EfficientNet-B0, EfficientNet-B3.

| Model           | Params  | ImageNet Top-1 |
|-----------------|---------|----------------|
| ResNet-18       | 11.7 M  | 69.8%          |
| ResNet-50       | 25.6 M  | 76.1%          |
| ResNet-101      | 44.5 M  | 77.4%          |
| EfficientNet-B0 | 5.3 M   | 77.7%          |
| EfficientNet-B3 | 12.2 M  | 82.0%          |

#### Step 5 — Inspected Architectures
- Printed parameter counts per named child layer.
- Ran dummy forward passes to capture intermediate tensor shapes at each stage.
- Identified where the final classifier head lives:
  - ResNet: `model.fc` → `Linear(2048 → 1000)`
  - EfficientNet: `model.classifier[-1]` → `Linear(1536 → 1000)`

#### Step 6 — Benchmarked Inference Speed
- Timed a batch of 32 images through each backbone on CPU/GPU.
- Saved timing results to `results/` for comparison.

#### Step 7 — Visualised Conv Filters and Feature Maps
- Extracted and plotted the 64 first-layer filters (3×3 kernels) of ResNet to see low-level patterns (edges, colour blobs, gradients).
- Passed a sample image through each layer and visualised the resulting feature maps.

---

### Files
- `day1_modern_architectures/day1_modern_architectures.py`
- `day1_modern_architectures/notes.md`

---

## 📅 Day 2 — Transfer Learning: Feature Extraction vs Fine-Tuning

### 🎯 Objectives
1. Understand **transfer learning** — reusing ImageNet-pretrained backbones on smaller datasets.
2. Implement and compare **two strategies**:
   - **Strategy A**: Feature Extraction — freeze the backbone, only train the new head.
   - **Strategy B**: Full Fine-Tuning — unfreeze the backbone with a very small LR.
3. Understand the **two-LR setup** and why it prevents catastrophic forgetting.
4. Learn how to **replace the classifier head** for an arbitrary N-class problem.
5. Compare results side-by-side on **CIFAR-10**.

---

### 🪜 How We Reached There — Step by Step

#### Step 1 — Understood What Transfer Learning Is
- ImageNet-pretrained backbones already encode rich visual hierarchies:
  - **Early layers**: edges, gradients, colour blobs (universal, always transfer well).
  - **Mid layers**: textures, shapes, object parts (usually transferable).
  - **Late layers**: high-level ImageNet semantics (may need adaptation).
- Transfer learning = reuse those layers, only re-train or adapt the final head.

#### Step 2 — Implemented Strategy A: Feature Extraction
```python
# Freeze everything
for param in model.parameters():
    param.requires_grad = False

# Re-enable only the new head
for param in model.fc.parameters():
    param.requires_grad = True

optimizer = optim.AdamW(model.fc.parameters(), lr=1e-3)
```
- **When to use**: small datasets (< 5k), data similar to ImageNet, need fast training.
- **Pros**: Fast, simple, hard to overfit.
- **Cons**: Backbone features may not be optimal for your domain.

#### Step 3 — Implemented Strategy B: Full Fine-Tuning with Two LRs
```python
optimizer = optim.AdamW([
    {"params": backbone_params, "lr": 1e-5},   # gentle nudge on backbone
    {"params": head_params,     "lr": 1e-3},   # learn from scratch on head
])
```
- **Why two LRs?** Using the same large LR on the backbone overwrites rich ImageNet features — this is **catastrophic forgetting**. Backbone LR must be 100x smaller than the head LR.
- **When to use**: medium/large datasets (> 10k), domain differs from ImageNet.

#### Step 4 — Replaced the Classifier Head
```python
# ResNet
in_features = model.fc.in_features        # 512 for ResNet-18, 2048 for ResNet-50
model.fc = nn.Linear(in_features, N_CLASSES)

# EfficientNet
in_features = model.classifier[-1].in_features
model.classifier[-1] = nn.Linear(in_features, N_CLASSES)
```
- **Critical rule**: Always read `in_features` from the existing layer — never hardcode it.

#### Step 5 — Applied Correct ImageNet Normalisation
```python
T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
```
- Pretrained conv filters were optimised for this distribution; using wrong stats degrades performance.

#### Step 6 — Trained and Compared on CIFAR-10
- Used CIFAR-10 (available out-of-the-box, no download needed).
- Ran 5-epoch smoke tests and 20-epoch full comparisons.
- Saved training curves and results JSON to `results/`.

---

### Files
- `day2_transfer_learning/day2_transfer_learning.py`
- `day2_transfer_learning/notes.md`

---

## 📅 Day 3 — Custom Dataset Pipeline

### 🎯 Objectives
1. Move from toy datasets (CIFAR-10) to a **real-world custom dataset** (Intel Image Classification — Kaggle).
2. Build the full **data pipeline from scratch**: `Dataset` → Transforms → `DataLoader`.
3. Understand `ImageFolder` (easy way) vs writing a **custom `Dataset` class** (flexible way).
4. Perform **dataset sanity checks**: visualise batches, inspect class distribution, verify transforms.
5. Handle **class imbalance** with `WeightedRandomSampler`.
6. Plug the custom pipeline into Day 2's transfer-learning code and train on 6 real-world classes.

---

### 🪜 How We Reached There — Step by Step

#### Step 1 — Chose the Dataset: Intel Image Classification
- **Source**: Kaggle (`puneet6060/intel-image-classification`)
- **6 Classes**: buildings, forest, glacier, mountain, sea, street
- **Size**: ~25k images (150x150 px), ~14k train / ~3k test / ~7k prediction
- **Why this dataset**: Clean folder structure, enough data, domain different from ImageNet (outdoor scenes) to make fine-tuning meaningful.

#### Step 2 — Understood the PyTorch Pipeline Architecture
```
Raw Images on Disk
       |
       v
Dataset (ImageFolder / Custom)    <-- maps idx to (image, label)
       |
       v
Transforms (Compose pipeline)     <-- Resize, Crop, Flip, Normalize
       |
       v
DataLoader (batching + shuffling) <-- yields (batch_images, batch_labels)
       |
       v
Model Training Loop
```

#### Step 3 — Used `ImageFolder` for Clean Directory Structure
```python
from torchvision.datasets import ImageFolder

dataset = ImageFolder(
    root="data/seg_train/seg_train/",   # root/class1/img.jpg, root/class2/img.jpg
    transform=my_transforms,
)
# Automatically discovers classes from folder names
# dataset.classes = ['buildings', 'forest', 'glacier', 'mountain', 'sea', 'street']
```

#### Step 4 — Wrote a Custom `Dataset` Class for Flexibility
```python
class MyDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert("RGB")
        label = self.labels[idx]
        if self.transform:
            img = self.transform(img)
        return img, label
```
- **Use when**: Data is in CSV/JSON, multi-label tasks, non-standard directory structures.

#### Step 5 — Built Separate Train and Eval Transforms
- **Training** (with augmentation):
```python
T.Compose([T.Resize(256), T.RandomCrop(224), T.RandomHorizontalFlip(),
           T.ColorJitter(...), T.ToTensor(), T.Normalize(mean, std)])
```
- **Evaluation** (deterministic — no randomness):
```python
T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor(), T.Normalize(mean, std)])
```
> Critical rule: NEVER use random augmentation during evaluation — it makes metrics unreliable.

#### Step 6 — Configured DataLoader Parameters
| Parameter | Training | Evaluation |
|-----------|----------|------------|
| `shuffle` | `True` | `False` |
| `batch_size` | 32–64 | 32–64 |
| `num_workers` | 2–4 | 2–4 |
| `pin_memory` | `True` (GPU) | `True` (GPU) |

#### Step 7 — Ran Dataset Sanity Checks
1. Visualised a grid of augmented training images — confirmed labels match visuals.
2. Plotted class distribution bar chart — checked for class imbalance.
3. Verified all images resize to 224×224 correctly.
4. Checked pixel value ranges after `ToTensor()` and `Normalize()`.

#### Step 8 — Handled Class Imbalance with `WeightedRandomSampler`
```python
from torch.utils.data import WeightedRandomSampler

weights = 1.0 / torch.tensor(class_counts, dtype=torch.float)
sample_weights = [weights[label] for _, label in dataset]
sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights))
loader = DataLoader(dataset, batch_size=32, sampler=sampler)
```

#### Step 9 — Plugged into Day 2's Training Code
- Swapped in the Intel dataset pipeline instead of CIFAR-10.
- Ran Feature Extraction and Full Fine-Tuning on the 6-class outdoor dataset.
- Used ImageNet normalisation stats (not dataset-specific) since the backbone is pretrained on ImageNet.

---

### Files
- `day3_custom_dataset/day3_custom_dataset.py`
- `day3_custom_dataset/notes.md`

---

## 📅 Day 4 — Data Augmentation with Albumentations

### 🎯 Objectives
1. Introduce **Albumentations** as the augmentation library — faster and more flexible than `torchvision.transforms`, and detection/segmentation-ready for Week 3.
2. Establish a **no-augmentation baseline** (resize + center crop + normalize only).
3. Build **5 incremental augmentation pipelines** to isolate each category's effect:
   - Baseline (no augmentation)
   - Spatial (flips, rotations, crops)
   - Color (brightness, contrast, hue, saturation)
   - Advanced (CoarseDropout, noise, blur)
   - Full (all combined)
4. Run a **controlled ablation study** — same backbone, same LR, same epochs, same split.
5. Compare accuracy, build an ablation table, and visualise augmented samples.

---

### 🪜 How We Reached There — Step by Step

#### Step 1 — Understood Why Augmentation is Needed
- Without augmentation: the model sees the **same image every epoch** and overfits.
- With augmentation: the model sees **new variations every epoch** and learns invariant features.
- **Key insight**: Augmentation teaches the model *what shouldn't matter* — a flipped mountain is still a mountain, so the model learns orientation invariance.

> Important mindset: Augmentation is NOT free accuracy. Some augmentations help, some hurt, some do nothing. The only way to know is to run a controlled ablation with a no-augmentation baseline.

#### Step 2 — Why Switch from torchvision to Albumentations?
| Feature | torchvision | Albumentations |
|---------|-------------|----------------|
| Speed | Moderate (PIL-based) | 2–5x faster (OpenCV/NumPy) |
| Number of transforms | ~30 | 70+ |
| Detection / Segmentation support | No | Yes (bboxes, masks) |
| Industry usage | Tutorials | Kaggle competitions & production |

#### Step 3 — Learned the Albumentations Core API
- torchvision works on **PIL Images**; Albumentations works on **NumPy arrays**.
```python
import albumentations as A
from albumentations.pytorch import ToTensorV2

transform = A.Compose([
    A.Resize(256, 256),
    A.RandomCrop(224, 224),
    A.HorizontalFlip(p=0.5),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2(),   # NumPy HWC uint8 -> PyTorch CHW float32
])

# Must pass a NumPy array (not PIL Image!)
image = np.array(Image.open(path).convert("RGB"))
augmented = transform(image=image)
tensor = augmented['image']
```

#### Step 4 — Built the 5 Augmentation Pipelines

**Pipeline 1 — Baseline** (control group, no augmentation):
- `Resize(256, 256)` → `CenterCrop(224, 224)` → `Normalize` → `ToTensorV2`

**Pipeline 2 — Spatial** (geometry changes):
- Adds: `HorizontalFlip(p=0.5)`, `ShiftScaleRotate(p=0.5)`, `RandomCrop(224, 224)`

**Pipeline 3 — Color** (appearance changes):
- Adds: `RandomBrightnessContrast(p=0.4)`, `HueSaturationValue(p=0.3)`, `RGBShift(p=0.3)`

**Pipeline 4 — Advanced** (information destruction):
- Adds: `CoarseDropout(p=0.3)`, `GaussNoise(p=0.3)`, `GaussianBlur(p=0.2)`

**Pipeline 5 — Full** (everything combined):
- All spatial + color + advanced transforms.

#### Step 5 — Wrote the Dataset Wrapper for Albumentations Integration
```python
class AlbumentationsDataset(Dataset):
    def __init__(self, image_folder_dataset, transform=None):
        self.dataset = image_folder_dataset
        self.transform = transform

    def __getitem__(self, idx):
        pil_image, label = self.dataset[idx]
        image = np.array(pil_image)   # PIL -> NumPy for Albumentations
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']
        return image, label
```
> Key: The underlying `ImageFolder` must have NO transforms — all transforms go into the Albumentations pipeline.

#### Step 6 — Designed the Ablation Study (Controlled Experiment)
```
Controlled variables (kept identical across all 5 experiments):
  - Same backbone: ResNet-18
  - Same learning rate schedule
  - Same number of epochs
  - Same train/val/test split (same random seed!)
  - Same evaluation transforms (deterministic CenterCrop)

Changed variable (one thing different per experiment):
  - Only the training augmentation pipeline
```
This design ensures that any accuracy difference is attributable **only to augmentation**.

#### Step 7 — Navigated Albumentations 2.x API Breaking Changes
CoarseDropout parameter renames:
```python
# Old (1.x) - throws TypeError in 2.x
A.CoarseDropout(max_holes=8, max_height=28, max_width=28, fill_value=0)

# New (2.x) - correct
A.CoarseDropout(num_holes_range=(1, 8), hole_height_range=(14, 28),
                hole_width_range=(14, 28), fill=0)
```
GaussNoise changed from variance to std dev:
```python
# Old (1.x)
A.GaussNoise(var_limit=(10.0, 50.0))

# New (2.x)  — sqrt(10) ≈ 3.16, sqrt(50) ≈ 7.07
A.GaussNoise(std_range=(3.0, 7.0))
```

#### Step 8 — Selected Domain-Appropriate Augmentations
| Transform | Applied? | Reason |
|-----------|----------|--------|
| `HorizontalFlip` | Yes | A flipped mountain is still a mountain |
| `Rotation (±15°)` | Yes | Slight tilt is natural in real photos |
| `ColorJitter` | Yes | Lighting varies in outdoor photography |
| `RandomCrop` | Yes | Forces model to use different image regions |
| `CoarseDropout` | Yes | Prevents reliance on any single region |
| `VerticalFlip` | No | Sky-below-ground looks unnatural for outdoor scenes |

#### Step 9 — Ran the Full Ablation and Compared Results
- Trained all 5 experiments on Intel Image Classification.
- Collected val accuracy and test accuracy per pipeline.
- Built the ablation table:

| Experiment | Augmentation | Expected Effect |
|------------|-------------|-----------------|
| Baseline   | None | ~88–92% val acc (large pretrained model) |
| + Spatial  | Flip + Rotate + Crop | +0.5–2% |
| + Color    | Brightness + HSV + RGBShift | +0.5–1.5% |
| + Advanced | Dropout + Noise + Blur | ±0.5% (depends on dataset size) |
| Full       | All combined | Best or near-best overall |

#### Step 10 — Visualised Augmented Samples and Saved Results
- Plotted a grid showing the same image through each augmentation pipeline.
- Confirmed all augmentations produce visually plausible, label-preserving transforms.
- Saved all plots and JSON results to `results/`.

---

### Files
- `day4_Data_Augmentation/day4_augmentation.py`
- `day4_Data_Augmentation/notes 1.md`

---

## 🗺️ Week 2 — The Big Picture

```
Day 1: Understand the architectures (ResNet, EfficientNet)
          |
          v
Day 2: Apply transfer learning on a known dataset (CIFAR-10)
          |
          v
Day 3: Swap in a real custom dataset (Intel — 6 outdoor classes)
          |
          v
Day 4: Improve generalisation with data augmentation (ablation study)
          |
          v
Day 5 (next): Ablation table + confusion matrix + final results write-up
```

> Each day built directly on the previous one: we went from understanding architectures, to applying them, to using real data, to making that real data work harder through augmentation.

---

## Key Libraries Used

| Library | Purpose |
|---------|---------|
| `torch` | Core deep learning framework |
| `torchvision` | Pretrained models, datasets, base transforms |
| `albumentations` | Advanced augmentation library (Day 4 onwards) |
| `PIL / Pillow` | Image loading and conversion |
| `matplotlib` | Visualisations, filter plots, feature maps |
| `wandb` | Experiment tracking (optional) |
| `numpy` | Array manipulation, image conversion |
