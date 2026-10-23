# Day 5 Notes — Anomaly Detection for Defects

## Kaggle dataset
[ipythonx/mvtec-ad](https://www.kaggle.com/datasets/ipythonx/mvtec-ad). The notebook downloads only the `CATEGORY` folder (default `bottle`) via `kagglehub`'s directory download, and falls back to the full ~5 GB if that fails.

```
bottle/
├── train/good/*.png                 # only normal images
├── test/{good,broken_large,broken_small,contamination}/*.png
└── ground_truth/<defect>/*_mask.png # pixel masks for defects
```

## Why "anomaly" detection instead of classification
Defects are rare, varied, and often unknown in advance. One-class methods learn only what "normal" looks like and flag deviations, so they need **no defect labels** for training.

## The two families

**Reconstruction (autoencoder).** Train an encoder–decoder to rebuild good images. At test time, the anomaly map is the per-pixel reconstruction error. It is simple and needs no pretrained model, but a strong autoencoder also reconstructs defects ("identity shortcut"), so scores overlap. The bottleneck size controls that trade-off.

**Embedding distance (PatchCore, PaDiM).** Use a frozen ImageNet CNN as the feature extractor.
- *PatchCore:* store mid-level patch embeddings from all good images in a memory bank, reduced by a greedy coreset. A test patch's score is the distance to its nearest neighbour; the image score is the maximum patch score.
- *PaDiM:* fit a multivariate Gaussian per patch position and score with the Mahalanobis distance.
- These work well because pretrained features already separate textures like "scratch" from "glass"; nothing has to be learned from ~200 images.

## Metrics
- **Image AUROC:** threshold-free ranking of good vs defective images (the headline number).
- **Pixel AUROC:** how well the anomaly map localizes defects (uses the GT masks).
- **Precision / recall / F1 at a threshold.** The notebook calibrates the threshold at the 99th percentile of scores on **held-out good images**, which is what you can do in production. The F1-optimal "oracle" threshold uses test labels and is an optimistic upper bound only.

## What to check in your run
- The comparison table: autoencoder vs PatchCore (ours) vs anomalib PatchCore / PaDiM.
- Score histograms: how much do good and defective scores overlap?
- Flag rate per defect type: which defects get missed (usually the smallest ones)?
- Flagged-sample gallery: does the heatmap sit on the actual defect?

## Pitfalls
- No flip/rotation augmentation for object categories: an orientation change can itself be the defect (e.g. `transistor/misplaced`).
- MVTec test sets are small (bottle has 83 images), so one image moves recall by several points. Don't over-read small differences.
- anomalib's API changed a lot between 0.x, 1.x, and 2.x. The notebook pins `anomalib==2.7.0`; old tutorials using `from anomalib.data import MVTec` are for 1.x.

## Materials
- [anomalib](https://github.com/open-edge-platform/anomalib)
- [MVTec AD dataset](https://www.mvtec.com/company/research/datasets/mvtec-ad)
- [PatchCore paper (arXiv:2106.08265)](https://arxiv.org/abs/2106.08265)
