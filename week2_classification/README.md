# Week 2 — Image Classification & Transfer Learning

| Attribute | Detail |
|-----------|--------|
| **Pillar** | Vision Intelligence Core |
| **Goal** | Move from training-from-scratch to fine-tuning pretrained backbones, and use data augmentation to improve performance on a custom dataset. |
| **Effort** | 32–40 hours (full-time) |

---

## Day-by-Day Plan

### Day 1: Modern CNN Architectures
- ResNet residual connections — why depth works
- EfficientNet compound scaling
- Load and inspect pretrained models from `torchvision`
- Benchmark inference speed; visualise filters & feature maps

**Materials:**
- [torchvision — Models & Pretrained Weights](https://pytorch.org/vision/stable/models.html)
- [ResNet paper (arXiv:1512.03385)](https://arxiv.org/abs/1512.03385)

**Files:** [`day1_modern_architectures.py`](day1_modern_architectures/day1_modern_architectures.py) · [`notes.md`](day1_modern_architectures/notes.md)

**Run:**
```bash
cd day1_modern_architectures
python day1_modern_architectures.py
```

---

### Day 2: Transfer Learning Mechanics
- Feature extraction (freeze backbone) vs full fine-tuning (unfreeze + low LR)
- Replace the classifier head for your own classes
- Two-LR strategy to prevent catastrophic forgetting
- Side-by-side comparison on CIFAR-10

**Materials:**
- [PyTorch — Transfer Learning Tutorial](https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html)
- [CS231n — Transfer Learning Notes](https://cs231n.github.io/transfer-learning/)

**Files:** [`day2_transfer_learning.py`](day2_transfer_learning/day2_transfer_learning.py) · [`notes.md`](day2_transfer_learning/notes.md)

**Run:**
```bash
cd day2_transfer_learning

# Quick 5-epoch smoke test (no W&B)
python day2_transfer_learning.py --epochs 5 --no-wandb

# Full 20-epoch comparison with W&B
python day2_transfer_learning.py --epochs 20

# Try EfficientNet-B0
python day2_transfer_learning.py --backbone efficientnet_b0 --epochs 15
```

---

### Day 3: Custom Dataset Pipeline _(coming next)_
- Pick a Kaggle image-classification dataset
- Build `Dataset`, `DataLoader`, and transforms from scratch

---

### Day 4: Data Augmentation with Albumentations _(coming next)_
- Rotation, crop, flip, colour jitter
- Establish a no-augmentation baseline first

---

### Day 5: Ablation Study & Final Results
- Run a controlled no-augmentation vs augmentation comparison
- Compare training from scratch with ImageNet transfer learning
- Generate the ablation table, learning curves, normalized confusion matrices,
  and per-class precision/recall/F1
- Save CSV, JSON, checkpoints, plots, and a metric-driven final write-up

Colab notebook: `day5_ablation_evaluation/Week2_Day5_Colab.ipynb`

Standalone CLI:
`python day5_ablation_evaluation/day5_ablation_evaluation.py --auto-download`

---

## Folder Structure

```
week2_classification/
├── README.md                               ← This file
├── requirements.txt                        ← Week 2 specific deps
│
├── day1_modern_architectures/
│   ├── day1_modern_architectures.py        ← Architecture explorer
│   └── notes.md
│
├── day2_transfer_learning/
│   ├── day2_transfer_learning.py           ← Feature extract vs fine-tune
│   └── notes.md
│
├── day3_custom_dataset/                    ← Coming: Day 3
├── day4_augmentation/                      ← Coming: Day 4
├── day5_ablation_evaluation/
│   ├── Week2_Day5_Colab.ipynb              ← Self-contained Colab workflow
│   ├── day5_ablation_evaluation.py          ← Standalone CLI workflow
│   └── notes.md                             ← Design and interpretation guide
│
├── models/                                 ← Saved checkpoints
│   └── .gitkeep
│
└── results/                                ← Plots, JSON results
    └── .gitkeep
```

---

## Deliverable (End of Week)

A fine-tuned ResNet/EfficientNet on a custom dataset, with:
- [ ] Ablation table showing augmentation effect (positive or negative)
- [ ] Confusion matrix + per-class accuracy
- [ ] Short results write-up explaining what moved the metric most

## Success Gate (Self-Check)

- [ ] Fine-tuned model **beats from-scratch baseline** by a clear margin
- [ ] Ablation table shows **measurable augmentation effect** with correct interpretation
- [ ] Two-LR setup demonstrated and explained in notes

> ⚠️ **Watch out:** A frozen backbone needs a **normal LR** on the new head; full fine-tuning needs a **much smaller LR** on the backbone — or you'll destroy the pretrained weights.

---

## Results (fill in after completing the week)

| Strategy | Backbone | Dataset | Val Acc | Test Acc |
|----------|----------|---------|---------|----------|
| Feature Extraction | ResNet-18 | CIFAR-10 | __% | __% |
| Full Fine-Tuning | ResNet-18 | CIFAR-10 | __% | __% |
| Fine-Tuned + Aug | ResNet-18 | Custom | __% | __% |
