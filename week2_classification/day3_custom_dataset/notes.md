# Day 3 Notes — Custom Dataset Pipeline

## Core Concept

Building a custom data pipeline is the bridge between "I can train on CIFAR-10" (a toy dataset) and "I can train on **my own** data" (real-world skill). Every production CV project starts here.

---

## The Data Pipeline in PyTorch

```
Raw Images on Disk
       │
       ▼
Dataset (ImageFolder / Custom)       ← maps idx → (image, label)
       │
       ▼
Transforms (Compose pipeline)        ← Resize, Crop, Flip, Normalize
       │
       ▼
DataLoader (batching + shuffling)    ← yields (batch_images, batch_labels)
       │
       ▼
Model Training Loop
```

---

## `ImageFolder` vs Custom `Dataset`

### ImageFolder (the easy way)
```python
from torchvision.datasets import ImageFolder

dataset = ImageFolder(
    root="data/train/",   # root/class1/img.jpg, root/class2/img.jpg, ...
    transform=my_transforms,
)
# Automatically discovers classes from folder names
# dataset.classes = ['buildings', 'forest', ...]
# dataset.class_to_idx = {'buildings': 0, 'forest': 1, ...}
```

**Use when:** Your data is in a folder-per-class directory structure.

### Custom Dataset (the flexible way)
```python
from torch.utils.data import Dataset

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

**Use when:**
- Data is in a CSV/JSON with paths and labels
- Multi-label classification
- Non-standard directory structures
- Custom on-the-fly preprocessing

---

## Transforms — Training vs Evaluation

### Training (with augmentation)
```python
T.Compose([
    T.Resize(256),           # Over-resize slightly
    T.RandomCrop(224),       # Random spatial crop → forces the model to use different regions
    T.RandomHorizontalFlip(),# 50% chance of flipping → doubles effective dataset size
    T.ColorJitter(...),      # Slight colour variation → robustness to lighting
    T.ToTensor(),            # → (C, H, W) float [0, 1]
    T.Normalize(mean, std),  # → match pretrained model's expected distribution
])
```

### Evaluation (deterministic, NO augmentation)
```python
T.Compose([
    T.Resize(256),
    T.CenterCrop(224),       # Deterministic — same crop every time
    T.ToTensor(),
    T.Normalize(mean, std),
])
```

> ⚠️ **Critical**: Evaluation transforms must be deterministic! Random augmentation during evaluation makes metrics unreliable.

---

## Which Normalisation Stats?

| Scenario | Stats to Use |
|----------|-------------|
| **Transfer learning** (pretrained backbone) | ImageNet: mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225) |
| **Training from scratch** | Compute your dataset's own mean/std |

**Why ImageNet stats for transfer learning?** The backbone's conv filters were trained expecting ImageNet-normalised inputs. Using different stats shifts the activation distributions, degrading performance.

---

## DataLoader Key Parameters

| Parameter | What | Recommended |
|-----------|------|-------------|
| `batch_size` | Samples per batch | 32–64 (limited by GPU memory) |
| `shuffle` | Randomise order each epoch | `True` for train, `False` for eval |
| `num_workers` | Parallel data loading processes | 2–4 (CPU cores / 2) |
| `pin_memory` | Page-lock CPU memory | `True` when using GPU |
| `drop_last` | Drop incomplete last batch | Sometimes `True` for train (BatchNorm stability) |

---

## Dataset Sanity Checks (ALWAYS DO THESE)

1. **Visualise a batch** — are the images correct? Labels matching?
2. **Check class distribution** — balanced or skewed?
3. **Check image sizes** — all same size? Need resizing?
4. **Verify transforms** — does augmented data look reasonable?
5. **Count total images** — matches what you expect?

---

## Handling Class Imbalance

If your dataset has unequal class counts:

```python
from torch.utils.data import WeightedRandomSampler

# Compute weights inversely proportional to class frequency
class_counts = [count_per_class[i] for i in range(num_classes)]
weights = 1.0 / torch.tensor(class_counts, dtype=torch.float)
sample_weights = [weights[label] for _, label in dataset]

sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights))
loader = DataLoader(dataset, batch_size=32, sampler=sampler)
# Note: Can't use shuffle=True with a sampler
```

---

## Intel Image Classification Dataset

- **Source**: [Kaggle](https://www.kaggle.com/datasets/puneet6060/intel-image-classification)
- **Classes (6)**: buildings, forest, glacier, mountain, sea, street
- **Size**: ~25k images (150×150 pixels)
- **Split**: ~14k train, ~3k test, ~7k prediction (unlabelled)
- **Why this dataset**: Clean folder structure, enough data, domain different from ImageNet (outdoor scenes)

---

## Observations & Takeaways
_Fill in after running the script_

- Which strategy won (feature extraction vs fine-tuning)?
- Which classes are hardest? Any confusion patterns?
- How does this compare to Day 2's CIFAR-10 results?
- Does the domain shift from ImageNet affect fine-tuning?

---

## Questions to Investigate
- [ ] What happens if you use nearest-neighbour resize instead of bilinear?
- [ ] Does WeightedRandomSampler improve accuracy on imbalanced classes?
- [ ] How does val_split size affect the result? Try 0.1 vs 0.2 vs 0.3.
- [ ] What if you accidentally use CIFAR-10 mean/std instead of ImageNet?

---

## Resources
- [PyTorch — Datasets & DataLoaders](https://pytorch.org/tutorials/beginner/basics/data_tutorial.html)
- [PyTorch — Writing Custom Datasets](https://pytorch.org/tutorials/beginner/data_loading_tutorial.html)
- [Intel Image Classification (Kaggle)](https://www.kaggle.com/datasets/puneet6060/intel-image-classification)
