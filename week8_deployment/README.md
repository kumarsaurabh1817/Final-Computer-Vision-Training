# Week 8 — ML Model Deployment and Tracking

| Attribute | Detail |
|-----------|--------|
| **Pillar** | Industrial-Grade Deployment |
| **Goal** | Productionize the Week 2 classifier end to end (optimize, export, serve, track, monitor, explain), then serve the Week 7 RAG system with the same pattern. |
| **Effort** | 32–40 hours (full-time) |
| **Runtime** | Google Colab, T4 GPU. Docker images are built on your machine (WSL2); the files are in [`serving/`](serving/). |

A list of every topic used this week, with a short explanation, is in [`week8notes.md`](week8notes.md).

---

## Data and model

| What | Where | Notes |
|---|---|---|
| **Intel Image Classification** | Your Drive folder **`My Drive/datasets/intel-image-classification/`** (already there). Fallbacks: `intel-image-classification.zip` in the same Drive folder, or the Kaggle download [`puneet6060/intel-image-classification`](https://www.kaggle.com/datasets/puneet6060/intel-image-classification) | 6 classes. A fixed subset is copied to local disk each session |
| **Week 2 checkpoint** (optional) | Put `pretrained_aug.pt` (from Week 2 Day 5) in **`My Drive/checkpoints/week2/`** | Without it, Day 1 trains the same ResNet-18 recipe for 3 epochs (~4 min) and saves it to `My Drive/cv_course_artifacts/week8/resnet18_week2.pt` |
| **Week 6 index** (for the RAG API) | `My Drive/cv_course_artifacts/week6_rag/` | Created by Week 6. Day 3 skips the RAG API if it is missing |

Everything Week 8 produces for later days is kept in **`My Drive/cv_course_artifacts/week8/`**: checkpoints, ONNX models, `reference.json` (drift), `serving/` (the service folder with the model), `mlflow/` (tracking store).

## How to run on Colab

1. **File → Upload notebook** → `dayN_*/Week8_DayN_Colab.ipynb`, in order (Day 1 → 5).
2. **Runtime → Change runtime type → T4 GPU**.
3. **Runtime → Run all**, and **Connect to Google Drive** when asked.

---

## Day-by-day

### Day 1 — Optimization
FP32 baseline → INT8 dynamic → INT8 static PTQ (FX, calibrated) → unstructured pruning 30/50/70 % (+1 epoch) → structured channel pruning 30 % (torch-pruning, +2 epochs) → distillation into MobileNetV3-Small (KD vs CE-only). One table with accuracy, worst-class accuracy, params, size, CPU/GPU latency; per-class deltas after quantization.
**Files:** [`Week8_Day1_Colab.ipynb`](day1_optimization/Week8_Day1_Colab.ipynb) · [`day1_optimization.py`](day1_optimization/day1_optimization.py) · [`notes.md`](day1_optimization/notes.md) · ~25 min

### Day 2 — ONNX export & acceleration
Export with a dynamic batch axis, checker + parity test, ORT graph optimizations, static INT8 (QDQ) with a calibration reader, and a benchmark of PyTorch CPU/CUDA/FP16 vs ORT CPU/CUDA/TensorRT on a fixed batch (batch-1 latency, batch-32 throughput, accuracy). The distilled student exported too.
**Files:** [`Week8_Day2_Colab.ipynb`](day2_onnx_export/Week8_Day2_Colab.ipynb) · [`day2_onnx_export.py`](day2_onnx_export/day2_onnx_export.py) · [`notes.md`](day2_onnx_export/notes.md) · ~10–15 min

### Day 3 — Serving (FastAPI + Docker)
`classifier_api` (`/health`, `/predict`, `/gradcam`, `/drift`, `/metrics`) on ONNX Runtime, tested over HTTP (accuracy, preprocessing parity, errors, concurrency); Dockerfiles; `rag_api` (`/ask` with query + optional image, agent or pipeline mode) running the Week 6–7 system.
**Files:** [`Week8_Day3_Colab.ipynb`](day3_serving/Week8_Day3_Colab.ipynb) · [`day3_serving.py`](day3_serving/day3_serving.py) · [`notes.md`](day3_serving/notes.md) · [`serving/`](serving/) · ~10–15 min

### Day 4 — Experiment tracking & versioning
MLflow runs for three training configurations (params, per-epoch metrics, data-hash tags, artifacts, model), run comparison, Model Registry with the `champion` alias, DVC `add` / `status` / `checkout`, and a run reproduced from its record.
**Files:** [`Week8_Day4_Colab.ipynb`](day4_mlflow_tracking/Week8_Day4_Colab.ipynb) · [`day4_mlflow_tracking.py`](day4_mlflow_tracking/day4_mlflow_tracking.py) · [`notes.md`](day4_mlflow_tracking/notes.md) · ~15 min

### Day 5 — Drift experiment & explainability
Reference statistics, corruption streams (5 types × 5 severities) and a label-shift window; KS / PSI / embedding / χ² detectors with thresholds tuned on clean bootstrap windows; the clean-vs-drift timeline chart; per-class accuracy under corruption; Evidently report; Grad-CAM heatmaps; `/drift` and `/gradcam` exercised over HTTP.
**Files:** [`Week8_Day5_Colab.ipynb`](day5_drift_explainability/Week8_Day5_Colab.ipynb) · [`day5_drift_explainability.py`](day5_drift_explainability/day5_drift_explainability.py) · [`notes.md`](day5_drift_explainability/notes.md) · ~15 min

### Weekend
Write-up: optimization table (Day 1), runtime benchmark (Day 2), serving architecture (Day 3), tracking (Day 4), drift chart and Grad-CAM examples (Day 5). Tag a release.

---

## Serving architecture

```
                     ┌──────────── classifier_api (Docker, CPU) ─────────────┐
 client ─ image ───► │ /predict ─► preprocess (NumPy) ─► ONNX Runtime ─► JSON │──► DriftMonitor window ─► /drift
                     │ /gradcam ─► PyTorch ResNet-18 + Grad-CAM ─► PNG        │
                     └───────────────────────────────────────────────────────┘
                     ┌──────────── rag_api (Docker, GPU or OpenRouter) ──────┐
 client ─ query ───► │ /ask ─► Week 7 agent / Week 6 pipeline ─► answer +    │◄── mounted: cv_course_artifacts/week6_rag
        (+ image)    │         citations (doc, page, bbox)                  │
                     └───────────────────────────────────────────────────────┘
 MLflow registry: models:/intel-scene-classifier@champion · DVC: data/intel_subset.dvc
```

To build the images locally, see [`serving/README.md`](serving/README.md).

## Deliverable & success gate

| Requirement | Where |
|---|---|
| Quantized / pruned / distilled with an accuracy-size-latency table | Day 1 `day1_optimization_table.md` |
| ONNX-exported, ORT-accelerated, benchmarked | Day 2 `day2_runtime_benchmark.csv` |
| Containerized FastAPI endpoint, lower latency at acceptable accuracy | Day 3 + `serving/` |
| Tracked and versioned in MLflow (+ DVC), one run reproduced | Day 4 |
| Drift experiment distinguishing clean from injected drift | Day 5 `day5_drift_timeline.png` |
| Grad-CAM explainability endpoint | Day 3 `/gradcam`, Day 5 examples |
| RAG system behind the same API pattern | Day 3 `rag_api` |

> Quantization can quietly drop accuracy on one class: the tables report the worst class. A drift monitor that never fires (or always fires) is useless: thresholds are tuned on clean data and checked on drifted data.

## Folder structure

```
week8_deployment/
├── README.md
├── week8notes.md
├── requirements.txt
├── serving/                         ← generated from Day 3 by tools/export_serving.py; docker build here
│   ├── classifier_api/ (app/, Dockerfile, requirements.txt)   + model/ (copy from Drive)
│   ├── rag_api/ (app/main.py, app/rag_core.py, Dockerfile, requirements.txt)
│   ├── docker-compose.yml
│   └── README.md
├── tools/
│   ├── shared/classifier.py, serving_header.py
│   ├── inline_shared.py · export_serving.py · py_to_ipynb.py
├── day1_optimization/ … day5_drift_explainability/   ← Week8_DayN_Colab.ipynb, .py, notes.md
├── models/
└── results/
```
