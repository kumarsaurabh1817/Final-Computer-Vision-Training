# Day 5 Notes — Drift Detection & Explainability

## Core ideas

**Kinds of drift:**

| Kind | What changes | Example here | Detected by |
|---|---|---|---|
| Covariate / data drift | P(x): the inputs | blur, brightness, contrast, JPEG, noise (ImageNet-C style) | input-statistic KS, embedding KS, confidence PSI |
| Label / prior shift | P(y): the class mix | a window of only sea + glacier | χ² on predicted-class counts |
| Concept drift | P(y \| x): the meaning | (not simulated) | needs labels: accuracy on a labelled sample |

**What to monitor without labels:** input statistics (brightness, contrast, sharpness, colourfulness), which catch camera and pipeline changes before any model is involved; the model's **confidence** and **entropy**; its **embeddings** (penultimate layer, reduced with PCA); and the **predicted-class distribution**.

**Tests:**
- **KS two-sample test**: are two samples from the same distribution? (p-value). With several features, apply a **Bonferroni** correction (p × number of tests).
- **PSI** = Σ (cur − ref)·ln(cur/ref) over reference-quantile bins. Rule of thumb: 0.1 is moderate, 0.25 is large.
- **χ²**: observed vs expected class counts.

**Threshold tuning is the experiment.** Bootstrap many clean windows, then set thresholds for a ≈5 % false-alarm rate, *then* look at the drift windows. A monitor that never fires, or always fires, is useless. The clean-vs-drift timeline chart is the proof.

**Grad-CAM:** gradients of the predicted class score with respect to the last conv block's feature maps, averaged per channel to give channel weights. The weighted sum of the maps goes through ReLU and is upsampled into a heatmap. It shows *where* the evidence is, coarsely (7×7 for ResNet-18). Use it to spot shortcuts (a "sea" prediction driven by the sky).

## Metrics
False-alarm rate on clean bootstrap windows; detected drift windows / total; which detector fired per window; accuracy and mean confidence per window; per-class accuracy under each corruption.

## What to check in your run
- Thresholds and the false-alarm rate.
- The timeline: do clean windows stay below and all drift windows go above? Which corruption is hardest to detect at low severity?
- Does label shift get caught only by the χ² test (as it should)?
- Grad-CAM on wrong predictions: is the model looking at the right region?
- The API: `/drift` says `ok` after clean traffic and `DRIFT` after blurred traffic.

## Pitfalls
- Windows too small mean noisy statistics and false alarms. Too large means slow detection.
- KS on raw pixels or high-dimensional embeddings without reduction is either useless or always significant.
- Confidence can stay high under drift (a confidently wrong model). That is why input statistics are monitored too.
- The `imagecorruptions` package pins old scikit-image. The corruptions here are re-implemented with Pillow/NumPy.

## Materials
- [Evidently AI](https://docs.evidentlyai.com/) · [ImageNet-C / CIFAR-10-C](https://github.com/hendrycks/robustness) · [imagecorruptions](https://github.com/bethgelab/imagecorruptions)
- [pytorch-grad-cam](https://github.com/jacobgil/pytorch-grad-cam) · [Grad-CAM paper](https://arxiv.org/abs/1610.02391) · [Failing Loudly (dataset shift detection)](https://arxiv.org/abs/1810.11953)
