# Week 8 — All Topics at a Glance

Every concept, tool and metric used in Week 8 (ML model deployment and tracking), grouped by theme, each with a short explanation. The **Day** column says where it is used.

---

## 1. The specimen

| Topic | Short explanation | Day |
|---|---|---|
| Week 2 classifier | ResNet-18 fine-tuned on Intel scenes (6 classes), fc = Dropout + Linear, 224 px, ImageNet normalization. | All |
| Intel Image Classification | Kaggle scene dataset (buildings, forest, glacier, mountain, sea, street); read from your Drive folder. | All |
| Fixed evaluation subset | Same seeded per-class sample every day, so numbers are comparable. | All |
| Worst-class accuracy | The lowest per-class accuracy; catches damage that the average hides. | 1, 2, 4 |

## 2. Optimization

| Topic | Short explanation | Day |
|---|---|---|
| Quantization | Store and compute with fewer bits (FP32 → INT8): ~4× smaller, faster integer kernels. | 1, 2 |
| Dynamic quantization | INT8 weights; activations quantized on the fly. Good for Linear/LSTM layers. | 1 |
| Static PTQ | INT8 weights and activations, activation ranges from calibration data. | 1, 2 |
| Calibration | Running representative inputs through observers to learn activation ranges. | 1, 2 |
| FX graph mode | Traces the model, fuses Conv+BN+ReLU, inserts observers, converts to INT8. | 1 |
| QAT | Quantization-aware training: simulate INT8 during training for the best accuracy. | 1 |
| fbgemm / x86 backend | CPU INT8 kernels used by PyTorch quantized models. | 1 |
| Unstructured pruning | Zero individual weights (global L1); sparse but not faster on dense kernels. | 1 |
| Structured pruning | Remove whole channels (torch-pruning); fewer MACs, genuinely faster. | 1 |
| MACs / FLOPs | Multiply-accumulate operations per inference; a hardware-independent cost. | 1 |
| Knowledge distillation | Student learns from the teacher's softened probabilities (temperature T, weight α). | 1 |
| MobileNetV3-Small | Small, fast student backbone (~1.5M params). | 1, 2 |

## 3. Export and acceleration

| Topic | Short explanation | Day |
|---|---|---|
| ONNX | Framework-neutral model graph format (operators + weights). | 2, 3 |
| `torch.onnx.export` | Traces the model into ONNX; opset version, input/output names, dynamic axes. | 2 |
| Dynamic axes | Free batch dimension so one file serves any batch size. | 2 |
| ONNX checker / parity test | Validate the graph; compare logits with PyTorch on the same batch. | 2 |
| ONNX Runtime (ORT) | Inference engine with graph optimizations and execution providers. | 2, 3 |
| Graph optimizations | Constant folding, Conv+BN fusion, node elimination (`ORT_ENABLE_ALL`). | 2 |
| Execution providers | CPU, CUDA, TensorRT backends behind one ORT API. | 2 |
| TensorRT | NVIDIA's optimizing compiler; FP16/INT8 engines, cached per GPU. | 2 |
| QDQ quantization | ONNX INT8 with QuantizeLinear/DequantizeLinear pairs and a calibration reader. | 2 |
| Per-channel quantization | One scale per conv filter; less accuracy loss. | 2 |
| Latency vs throughput | Batch-1 wait time vs batch-32 images per second. | 1, 2, 3 |

## 4. Serving

| Topic | Short explanation | Day |
|---|---|---|
| FastAPI | Python web framework with typed endpoints and automatic docs. | 3, 5 |
| Uvicorn | ASGI server that runs the FastAPI app (workers, host, port). | 3, 5 |
| `/health`, `/predict`, `/metrics` | Liveness, inference and service statistics endpoints. | 3 |
| Multipart upload | Sending an image file in a form (`python-multipart`). | 3, 5 |
| Load once at start-up | Model/session created when the process starts (FastAPI `lifespan`). | 3 |
| Preprocessing parity | Service preprocessing must equal training preprocessing; tested numerically. | 3 |
| Error codes | 400 bad input, 413 too large, 503 loading or missing dependency. | 3 |
| Concurrency | Thread pool / workers for the classifier; a lock for one GPU LLM. | 3 |
| Docker / Dockerfile | Reproducible image: base, dependencies, app, model, command, healthcheck. | 3 |
| docker-compose | Runs both services together with ports, env vars and volumes. | 3 |
| Volumes and env vars | Mount large artifacts; pass secrets at run time, never bake them in. | 3 |
| RAG API | Week 6 pipeline / Week 7 agent behind `POST /ask` (query + optional image). | 3 |

## 5. Tracking and versioning

| Topic | Short explanation | Day |
|---|---|---|
| MLflow Tracking | Runs with params, metrics (per step), tags and artifacts. | 4 |
| Experiment / run | A named group of runs / one execution of training. | 4 |
| `search_runs` | Query runs into a DataFrame for comparison. | 4 |
| Model logging + signature | `mlflow.pytorch.log_model` with an input example. | 4 |
| Model Registry | Named models with versions and aliases (`champion`). | 4 |
| `models:/name@alias` | Load a model by alias; promotion = moving the alias. | 4 |
| SQLite backend | Database tracking store, needed for the registry. | 4 |
| Data hash | Content hash of the exact data files, logged as a tag. | 4 |
| DVC | Data version control: cache + `.dvc` pointer in Git; `add`, `status`, `checkout`, `push/pull`. | 4 |
| Reproducibility | Same code, params, data, environment and seeds → same result. | 4 |
| Deterministic algorithms | `torch.use_deterministic_algorithms`, `CUBLAS_WORKSPACE_CONFIG`, cudnn settings. | 4 |

## 6. Monitoring: drift

| Topic | Short explanation | Day |
|---|---|---|
| Data / covariate drift | The input distribution changes (camera, lighting, compression). | 5 |
| Label / prior shift | The class mix changes. | 5 |
| Concept drift | The input→label relationship changes; needs labels to see. | 5 |
| Reference distribution | Statistics of clean data the model was validated on. | 5 |
| ImageNet-C / CIFAR-10-C | Standard corruption families and severities for robustness and drift tests. | 5 |
| Input statistics | Brightness, contrast, sharpness (Laplacian variance), colourfulness. | 5 |
| Confidence / entropy | Max softmax probability and prediction entropy. | 5 |
| Embedding drift | KS on PCA components of penultimate-layer features. | 5 |
| KS test | Two-sample Kolmogorov–Smirnov test of equal distributions. | 5 |
| Bonferroni correction | Multiply p-values by the number of tests. | 5 |
| PSI | Population Stability Index between binned distributions. | 5 |
| χ² test | Observed vs expected class counts (label shift). | 5 |
| Windowing | Compare fixed-size windows of recent traffic with the reference. | 5 |
| Threshold tuning | Set thresholds from clean bootstrap windows (false-alarm rate), then check detection. | 5 |
| Evidently | Library that runs drift tests and renders an HTML report. | 5 |
| `/drift` endpoint | Rolling-window monitor in the API using `reference.json`. | 3, 5 |

## 7. Explainability

| Topic | Short explanation | Day |
|---|---|---|
| Grad-CAM | Heatmap from gradient-weighted last-conv feature maps for the predicted class. | 3, 5 |
| Target layer | `layer4[-1]` of ResNet-18 (7×7 spatial maps). | 3, 5 |
| `/gradcam` endpoint | Returns the heatmap overlay PNG for an uploaded image. | 3, 5 |
| Shortcut detection | Heatmaps on wrong predictions reveal what the model really used. | 5 |

## 8. Libraries and tools

| Topic | Short explanation | Day |
|---|---|---|
| torch.ao.quantization / torchao | PyTorch quantization APIs (the former is being replaced by the latter). | 1 |
| torch-pruning | Dependency-aware structured pruning. | 1 |
| onnx / onnxruntime-gpu | Model format and inference runtime (CPU, CUDA, TensorRT). | 2, 3 |
| fastapi, uvicorn, python-multipart | Web API stack. | 3, 5 |
| Docker | Containers for the services (built in WSL2). | 3 |
| mlflow, dvc | Tracking/registry and data versioning. | 4 |
| scipy.stats | KS and χ² tests. | 3, 5 |
| evidently | Drift reports. | 5 |
| pytorch-grad-cam (`grad-cam`) | Grad-CAM implementation. | 3, 5 |
| Google Drive artifacts | `My Drive/cv_course_artifacts/week8/` keeps checkpoints, ONNX, serving folder, MLflow store, drift reference. | All |
