# Week 2 Day 5 — Ablation and Error Analysis

## Goal

Complete the Week 2 classification deliverable with measured comparisons, not assumed conclusions:

- compare a from-scratch ResNet-18 against ImageNet transfer learning;
- isolate the effect of training-time augmentation;
- report overall accuracy and macro/weighted F1;
- inspect normalized confusion matrices and per-class precision, recall, and F1;
- save reproducible tables, plots, checkpoints, and a results write-up.

## Experiment design

| Experiment | Initialization | Training augmentation | Purpose |
|---|---|---|---|
| `scratch_no_aug` | Random | No | Training-from-scratch reference |
| `pretrained_no_aug` | ImageNet | No | Transfer-learning and augmentation control |
| `pretrained_aug` | ImageNet | Yes | Augmentation treatment |

The valid augmentation ablation is `pretrained_aug` versus `pretrained_no_aug`. Both runs use the same split, architecture, initialization seed, optimizer family, learning rates, epoch budget, checkpoint rule, and deterministic evaluation pipeline. Comparing `scratch_no_aug` with a pretrained run answers a different question—the value of transfer learning—so the notebook reports that delta separately.

## Why these metrics

- **Accuracy** gives the overall fraction of correct predictions.
- **Macro F1** weights every class equally and exposes weak minority or difficult classes.
- **Weighted F1** summarizes class performance while respecting test-set support.
- **Per-class precision** shows how trustworthy a predicted label is.
- **Per-class recall** shows how many real examples of a class were found.
- **Confusion matrix** reveals which class pairs the model systematically mixes up.

## Interpretation rules

Augmentation is not automatically beneficial. A positive test delta supports the chosen policy; a negative delta can indicate overly strong transformations, domain mismatch, insufficient training, or ordinary sampling variation. Conclusions should include the measured delta and the error patterns rather than claiming that augmentation must improve accuracy.

Do not tune transformations on the final test set. Use validation behavior to choose a policy, then report the held-out test result once.

## Colab and dataset

Open `Week2_Day5_Colab.ipynb`, choose a T4 GPU, and run all cells. The notebook uses `kagglehub` to download the public `puneet6060/intel-image-classification` dataset automatically. A separate manual download is unnecessary under normal Colab conditions. Internet access is still required for the dataset and the ImageNet pretrained weights. Download the generated ZIP before the runtime disconnects.

Use `FULL_RUN=False` only to confirm that the pipeline works. Final metrics must come from `FULL_RUN=True`.

## Standalone Python entry point

`day5_ablation_evaluation.py` provides the same three experiments and evaluation
artifacts from the command line:

```bash
# Automatically download the public dataset.
python day5_ablation_evaluation.py --auto-download

# Or use an already downloaded dataset.
python day5_ablation_evaluation.py --data-dir /path/to/intel-image-classification

# One-epoch balanced smoke test only (not final metrics).
python day5_ablation_evaluation.py --auto-download --quick
```

By default, artifacts are written to `day5_ablation_evaluation/outputs/`. Use
`--output-dir PATH` to change this location.
