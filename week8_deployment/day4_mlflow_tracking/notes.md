# Day 4 Notes — Experiment Tracking & Versioning

## Core ideas

**What makes a result reproducible:** the code (git commit), the parameters, the data version, the environment (library versions) and the randomness (seeds, deterministic kernels). MLflow records the first four for each run. Determinism settings handle the fifth.

**MLflow Tracking:**

| Concept | Example here |
|---|---|
| Experiment | `week8-intel-resnet18` |
| Run | one training of one configuration |
| Params | `backbone_lr=1e-4`, `augment=True`, `epochs=2`, `seed=42` |
| Metrics (with steps) | `val_acc` per epoch; `test_acc`, `test_worst_class_acc` |
| Tags | `data_hash_train`, `dvc_md5`, `torch` version |
| Artifacts | confusion matrix, per-class CSV, the model, ONNX files |

`mlflow.search_runs()` returns a DataFrame to compare and plot runs. `mlflow ui` shows the same thing in a browser.

**Model Registry:** a registered model has a **name**, numbered **versions** (each linked to the run that produced it) and movable **aliases** (`champion`, `challenger`). Serving loads `models:/intel-scene-classifier@champion`. Promoting a model means moving the alias, with no code change. It needs a database-backed tracking store (SQLite here).

**Data versioning:**
- A **content hash** of the exact files is cheap and logged with every run.
- **DVC** keeps the data in a cache (and optionally a remote: S3/GCS/Drive/SSH) and a small `.dvc` pointer (md5, size, nfiles) in Git. `dvc status` detects changes, `dvc checkout` restores the pointed version, and `dvc push/pull` share it.

**Reproducing a run:** read its params, check the data hash, set seeds and `torch.use_deterministic_algorithms(True)` (plus `CUBLAS_WORKSPACE_CONFIG` before CUDA starts), retrain and compare. Small differences can remain (non-deterministic GPU ops, different hardware), so report them.

## What to check in your run
- The runs table: which configuration wins, and by how much?
- Registered version, alias, and the reloaded champion's accuracy equal to the logged one.
- The `.dvc` file, the `dvc status` output after deleting an image, and its restoration.
- Reproduction: original vs reproduced test accuracy.

## Pitfalls
- A file-based store (`./mlruns`) has no registry. Use a database URI.
- Logging a model without an input example or signature means the serving side has to guess the input format.
- Colab wipes `/content`. Copy the tracking store to Drive (the notebook does this).
- Data hashes must be computed on the same files in the same order. Sort them.

## Materials
- [MLflow Tracking](https://mlflow.org/docs/latest/tracking.html) · [MLflow Model Registry](https://mlflow.org/docs/latest/model-registry.html)
- [DVC documentation](https://dvc.org/doc) · [PyTorch reproducibility](https://pytorch.org/docs/stable/notes/randomness.html)
