# Day 1 Notes — Modern CNN Architectures

## Key Concepts Learned

### ResNet — Residual Connections
- **Problem solved**: Vanishing gradients in very deep networks caused accuracy to *degrade* with more layers.
- **Solution**: The skip (shortcut) connection `output = F(x) + x` creates a gradient highway that bypasses the conv stack.
- **Intuition**: Layers only need to learn the *residual* (difference from identity), which is easy to push to zero — so worst case the layer is a no-op.
- **BasicBlock vs Bottleneck**:
  - BasicBlock (18/34): two 3×3 convs → fewer params, fast on CPU
  - Bottleneck (50/101/152): 1×1→3×3→1×1 → more params, better GPU utilisation

### EfficientNet — Compound Scaling
- Scales depth, width, and resolution together with a single compound coefficient φ.
- Avoids the diminishing-returns problem of scaling only one dimension.
- B0 = NAS baseline; B3 is the sweet spot for most custom tasks (good accuracy, < 5 GB VRAM).

---

## Architecture Quick-Reference

| Model           | Params  | ImageNet Top-1 | Notes                        |
|-----------------|---------|----------------|------------------------------|
| ResNet-18       | 11.7 M  | 69.8%          | Lightest, great baseline     |
| ResNet-50       | 25.6 M  | 76.1%          | Standard workhorse           |
| ResNet-101      | 44.5 M  | 77.4%          | Heavy; use when you have GPU |
| EfficientNet-B0 | 5.3 M   | 77.7%          | Tiny but strong              |
| EfficientNet-B3 | 12.2 M  | 82.0%          | Best param-efficiency here   |

---

## Key torchvision API Notes

```python
from torchvision.models import resnet50, ResNet50_Weights

# New (≥ 0.13) multi-weight API — ALWAYS use this
model = resnet50(weights=ResNet50_Weights.DEFAULT)

# Inspect the final classifier head
print(model.fc)           # ResNet: Linear(2048 → 1000)
print(model.classifier)   # EfficientNet: Sequential → Linear

# Freeze all params
for p in model.parameters():
    p.requires_grad = False

# Replace head for N-class task (done in Day 2)
model.fc = nn.Linear(model.fc.in_features, N_CLASSES)
```

---

## Questions / Things to Investigate Further
- [ ] How does the projection shortcut (1×1 conv) in Bottleneck handle dim mismatch?
- [ ] What does EfficientNet's MBConv block look like internally?
- [ ] Why does BatchNorm alone not fix the vanishing gradient problem?

---

## Resources Used
- [torchvision Models & Pretrained Weights](https://pytorch.org/vision/stable/models.html)
- [ResNet paper (arXiv:1512.03385)](https://arxiv.org/abs/1512.03385)
- [EfficientNet paper (arXiv:1905.11946)](https://arxiv.org/abs/1905.11946)
