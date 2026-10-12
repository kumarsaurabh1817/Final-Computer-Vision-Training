# Week 1 — ML & CV Foundations

| Attribute | Detail |
|-----------|--------|
| **Pillar** | Foundations (prerequisite) |
| **Goal** | Understand how a neural network trains — tensors, autograd, the forward/backward/optimizer loop — and the fundamentals of how images are represented. |
| **Effort** | 32–40 hours (full-time) |

---

## Day-by-Day Plan

### Day 1: Environment Setup + PyTorch Basics
- Set up Python, PyTorch, GPU/Colab, Git repo, W&B
- Work through PyTorch "Learn the Basics" + "60-Minute Blitz"
- Understand tensors and autograd

**Materials:**
- [PyTorch — Learn the Basics](https://pytorch.org/tutorials/beginner/basics/intro.html)
- [PyTorch — 60-Minute Blitz](https://pytorch.org/tutorials/beginner/deep_learning_60min_blitz.html)

**Files:** [`day1_pytorch_basics.py`](day1_pytorch_basics/day1_pytorch_basics.py)

---

### Day 2: Image Fundamentals
- Pixels, channels, color spaces, resizing
- Load/manipulate images with OpenCV + Pillow

**Materials:**
- [OpenCV — Getting Started with Images](https://docs.opencv.org/4.x/db/deb/tutorial_display_image.html)
- [Image Processing with OpenCV & Pillow](https://realpython.com/image-processing-with-the-python-pillow-library/)

**Files:** [`day2_image_fundamentals.py`](day2_image_fundamentals/day2_image_fundamentals.py)

---

### Day 3: The Training Loop
- Forward pass, loss, backprop, optimizer step — by hand
- Build a 2-layer MLP on CIFAR-10/FashionMNIST (no high-level wrappers)

**Materials:**
- [3Blue1Brown — Neural Networks](https://www.youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi)
- [PyTorch — Optimizing Model Parameters](https://pytorch.org/tutorials/beginner/basics/optimization_tutorial.html)

**Files:** [`day3_training_loop.py`](day3_training_loop/day3_training_loop.py)

---

### Day 4: Convolutions
- Filters, feature maps, stride, padding, pooling
- Convert the MLP into a small CNN

**Materials:**
- [Stanford CS231n — Convolutional Networks](https://cs231n.github.io/convolutional-networks/)

**Files:** [`day4_convolutions.py`](day4_convolutions/day4_convolutions.py)

---

### Day 5: Train CNN on CIFAR-10
- Add validation split, plot loss/accuracy curves
- Tune learning rate and batch size
- Log runs to W&B

**Materials:**
- [PyTorch — Training a Classifier (CIFAR-10)](https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html)
- [Weights & Biases — PyTorch Integration](https://docs.wandb.ai/guides/integrations/pytorch)

**Files:** [`day5_cnn_cifar10.py`](day5_cnn_cifar10/day5_cnn_cifar10.py)

---

### Weekend (Optional)
- fast.ai Lesson 1 for top-down reinforcement
- Andrew Ng — Deep Learning Specialization (Course 1 + CNN course)
- Catch up on any incomplete days

---

## Deliverable

A GitHub folder containing:
1. ✅ A CNN trained on CIFAR-10 with logged loss/accuracy curves
2. ✅ A one-page markdown "Anatomy of the Training Loop" explainer

## Success Gate (Self-Check)

- [ ] CNN reaches **≥ 70% test accuracy** on CIFAR-10
- [ ] Explainer correctly describes **backprop and the optimizer step**
- [ ] All runs logged to **W&B** with visible curves
- [ ] Code committed with **clean, working state**

> ⚠️ **Note:** Don't use pretrained models — the point of this week is understanding the mechanics. Use custom MLP/CNN implementations only.

---

## Folder Structure

```
week1_foundations/
├── README.md                          # This file
├── requirements.txt                   # Week 1 specific dependencies
│
├── day1_pytorch_basics/
│   ├── day1_pytorch_basics.py         # Tensors, autograd, basic ops
│   └── notes.md                       # Learning notes
│
├── day2_image_fundamentals/
│   ├── day2_image_fundamentals.py     # OpenCV + Pillow image manipulation
│   ├── sample_images/                 # Test images for manipulation
│   └── notes.md
│
├── day3_training_loop/
│   ├── day3_training_loop.py          # MLP on CIFAR-10 / FashionMNIST
│   └── notes.md
│
├── day4_convolutions/
│   ├── day4_convolutions.py           # CNN architecture from the MLP
│   └── notes.md
│
├── day5_cnn_cifar10/
│   ├── day5_cnn_cifar10.py            # Full CNN training pipeline with W&B
│   ├── utils.py                       # Plotting, metrics helpers
│   └── notes.md
│
├── models/                            # Saved model checkpoints (.gitkeep)
│   └── .gitkeep
│
├── results/                           # Plots, curves, exported figures
│   └── .gitkeep
│
└── deliverables/
    └── training_loop_explainer.md     # The "Anatomy of the Training Loop"
```

---

## Results

_Fill in after completing the week:_

| Metric | Value |
|--------|-------|
| CNN Test Accuracy | __%  |
| Best Learning Rate | __ |
| Best Batch Size | __ |
| Training Epochs | __ |
| W&B Run Link | [link]() |
