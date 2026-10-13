# Day 2 Notes — Transfer Learning: Feature Extraction vs Fine-Tuning

## Core Concept

Transfer learning = reuse a model trained on a large dataset (ImageNet, 1.2 M images)
on your own smaller, domain-specific dataset.

The pretrained backbone has already learned:
- **Early layers**: edges, gradients, colour blobs (universal, always transferable)
- **Mid layers**: textures, shapes, object parts (usually transferable)
- **Late layers**: high-level semantics specific to ImageNet categories (may need adaptation)

---

## The Two Strategies

### Strategy A — Feature Extraction (freeze backbone)
```python
for param in model.parameters():
    param.requires_grad = False      # freeze everything

# Re-enable new head
for param in model.fc.parameters():
    param.requires_grad = True

optimizer = optim.AdamW(model.fc.parameters(), lr=1e-3)
```

**When to use:**
- Very small dataset (< 5 k samples)
- Domain very similar to ImageNet (natural photos)
- Need fast training / low compute

**Pros:** Fast, simple, hard to overfit  
**Cons:** Backbone features may not be optimal for your domain

---

### Strategy B — Full Fine-Tuning (unfreeze backbone)
```python
# Separate learning rates — CRITICAL
optimizer = optim.AdamW([
    {"params": backbone_params, "lr": 1e-5},   # gentle nudge
    {"params": head_params,     "lr": 1e-3},   # learning from scratch
])
```

**When to use:**
- Medium/large dataset (> 10 k samples)
- Domain is different from ImageNet (medical, satellite, industrial)
- Compute is available

**Pros:** Higher accuracy potential  
**Cons:** Slower, risk of catastrophic forgetting at high LR, needs more data

---

## Why the Two-LR Setup?

If you use the same large LR for the backbone:
- The gradients from your small task will **overwrite** the rich ImageNet features
- This is **catastrophic forgetting** — accuracy collapses in early epochs

The fix: backbone LR ≈ 1e-5 to 1e-6 (100× smaller than head LR).  
The backbone is gently nudged, not rewritten.

---

## Replacing the Classifier Head

### ResNet (model.fc)
```python
model = models.resnet50(weights=ResNet50_Weights.DEFAULT)
in_features = model.fc.in_features      # 2048 for ResNet-50
model.fc = nn.Linear(in_features, N)   # N = your classes
```

### EfficientNet (model.classifier[-1])
```python
model = models.efficientnet_b3(weights=EfficientNet_B3_Weights.DEFAULT)
in_features = model.classifier[-1].in_features   # 1536 for B3
model.classifier[-1] = nn.Linear(in_features, N)
```

> ⚠️ **Always check `in_features` from the existing layer** — don't hardcode it.  
> Different backbone sizes have different feature dimensions.

---

## ImageNet Normalisation — Required!

Pretrained models expect inputs normalised with **ImageNet stats**, not raw pixel values:

```python
T.Normalize(mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225])
```

Using CIFAR-10 stats with an ImageNet-pretrained model gives much worse results.

---

## Results Table (fill in after running)

| Strategy           | Val Acc | Test Acc | Training Time |
|--------------------|---------|----------|---------------|
| Feature Extraction | __%     | __%      | __s           |
| Full Fine-Tuning   | __%     | __%      | __s           |

---

## Observations & Takeaways
_Fill in after running the script_

- Which strategy won?
- How many epochs did each need to converge?
- Did fine-tuning show early instability (sign of LR too high)?

---

## Questions to Investigate
- [ ] What happens if you use `lr=1e-3` for the backbone too? (run and observe)
- [ ] Does early stopping help feature extraction more or fine-tuning more?
- [ ] How does a larger backbone (ResNet-50) change the comparison?

---

## Resources
- [PyTorch Transfer Learning Tutorial](https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html)
- [CS231n Transfer Learning Notes](https://cs231n.github.io/transfer-learning/)
- [Universal Language of Convolutional Neural Networks (Distill)](https://distill.pub/2017/feature-visualization/)
