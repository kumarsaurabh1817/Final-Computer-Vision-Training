# Computer Vision Specialist Track — Topic Lists (Week 1 → Week 8)

Source: [Computer_vision_training_plan_v3.docx](Computer_vision_training_plan_v3.docx) + the code in each `weekN_*/dayN_*/` folder.

This file has three parts:

1. **[Part A](#part-a--important-topics-in-prerequisite-order)**: the important topics for each week, in prerequisite order. Each topic needs only the topics above it.
2. **[Part B](#part-b--topics-used-in-the-code-week--day)**: the topics, libraries, models and datasets that the code actually uses, day by day.
3. **[Part C](#part-c--services-used-aws-azure-gcp-and-others)**: which services are used (AWS, Azure, GCP and others), with the week and day where each one appears.

---

## Prerequisite chain (big picture)

```
Week 1  Foundations: tensors → autograd → training loop → CNN
   │
Week 2  Classification: pretrained backbones → transfer learning → data pipeline → augmentation → evaluation
   │
Week 3  Object identification: boxes/IoU/mAP → YOLO training → segmentation → tracking → anomaly detection
   │
Week 4  Documents: OCR + preprocessing → layout models → managed cloud services → KV/tables → evaluation
   │
Week 5  VLMs: architecture → VQA → structured JSON → captioning → CLIP embeddings          (needs W2–W4)
   │
Week 6  Multimodal RAG: parse → represent (caption + embed) → index → retrieve → grounded answer   (needs W4–W5)
   │
Week 7  Agentic RAG: tools + ReAct → planning/routing → self-correction → text+image agent → eval   (needs W6)
   │
Week 8  Deployment: optimize → ONNX → FastAPI/Docker → MLflow/DVC → drift + Grad-CAM     (W2 classifier + W7 RAG)
```

---

## Part A — Important topics in prerequisite order

### Week 1 — ML & CV Foundations (pillar: Foundations)

1. Environment: Python 3.12, PyTorch (CUDA), GPU/Colab, Git, W&B
2. Tensors: creation, shapes, dtypes, broadcasting, CPU ↔ GPU
3. Autograd: computational graph, `requires_grad`, `.backward()`, gradients
4. Gradient descent: linear regression from scratch
5. Image fundamentals: pixels, channels, color spaces (RGB/BGR/HSV/Gray), resizing, filtering, image ↔ tensor
6. Neural network basics: layers, activations, MLP
7. The training loop: forward pass → loss → backprop → optimizer step; train vs eval mode
8. Datasets and DataLoaders (CIFAR-10, FashionMNIST), normalization and why it helps
9. Convolutions: filters, feature maps, stride, padding, pooling, receptive field
10. CNN architecture: building a CNN from an MLP
11. Training practice: validation split, learning-curve plots, tuning LR and batch size, overfitting
12. Experiment logging (Weights & Biases), confusion matrix

**Success gate:** custom CNN ≥ 70% on CIFAR-10, plus a written explainer of the training loop. No pretrained models this week.

### Week 2 — Image Classification & Transfer Learning (pillar: Vision Intelligence Core)

1. Modern CNN architectures: ResNet (residual connections, BasicBlock/Bottleneck) and EfficientNet (compound scaling)
2. Pretrained weights (ImageNet), torchvision model zoo, inference benchmarking
3. Feature visualization: first-layer filters, feature maps
4. Transfer learning: feature extraction (frozen backbone) vs fine-tuning (unfrozen, low LR), replacing the classifier head
5. Custom datasets: `Dataset`/`ImageFolder`, transforms, computing dataset mean/std, class imbalance (`WeightedRandomSampler`)
6. Data augmentation with Albumentations: flip, rotate, crop, color jitter; baseline without augmentation first
7. Ablation studies: controlled experiments, seeding, reproducibility
8. Evaluation: confusion matrix, per-class precision/recall/F1, interpreting results

**Success gate:** the fine-tuned model clearly beats a from-scratch baseline, and the ablation table shows the augmentation effect.

### Week 3 — Object identification: Detection, Segmentation, Tracking, Anomaly (pillar: Vision Intelligence Core)

1. Bounding-box formats (xyxy, xywh, YOLO cx-cy-w-h)
2. IoU, Non-Maximum Suppression (NMS)
3. Detection metrics: precision/recall, AP, mAP@0.5 and mAP@0.5:0.95
4. Detector families: one-stage (YOLO, anchor-free) vs two-stage (Faster R-CNN, anchors + RPN)
5. Annotation formats (Pascal VOC XML → YOLO txt), label-quality audit, train/val/test split
6. Training a custom YOLO (Ultralytics) and evaluating it
7. Segmentation: semantic (U-Net, DeepLabV3) vs instance (Mask R-CNN, YOLO-seg), Dice loss, mIoU
8. Multi-object tracking: Hungarian matching, Kalman filter, ByteTrack, BoT-SORT / DeepSORT idea, ID switches
9. Anomaly detection: reconstruction (autoencoder) vs embedding distance (PatchCore, PaDiM), AUROC, MVTec AD

**Success gate:** a sensible mAP on the custom dataset and a tracked video with stable IDs.

### Week 4 — Document Understanding (pillar: Document Intelligence)

1. OCR engines: Tesseract vs EasyOCR vs PaddleOCR
2. OCR metrics: word F1, character error rate (CER)
3. Image preprocessing for OCR: grayscale, denoise, deskew, CLAHE, Otsu/adaptive threshold
4. Why layout matters: text-only (RoBERTa) vs layout-aware (LayoutLMv3)
5. Token classification (BIO tags), key-value linking into JSON
6. OCR-free document understanding (Donut)
7. Managed services: Azure AI Document Intelligence and AWS Textract (Google Document AI is optional)
8. Cost control and caching for paid APIs; data privacy and residency
9. A common JSON schema (pydantic) that every extractor's output is normalized into
10. Hard cases: rotated scans, varied layouts, merged cells, multi-page tables; Table Transformer (TATR)
11. Evaluation: field accuracy, ANLS, line-item F1, TEDS / TEDS-S, cell-adjacency F1, latency, cost per page
12. Choosing an approach: accuracy vs cost vs setup vs privacy

**Success gate:** both a self-hosted and a managed approach produce valid JSON, and a comparison matrix backs a recommendation.

### Week 5 — Vision-Language Models (pillar: Vision + Generative AI)

1. VLM architecture: vision encoder → projector → LLM; visual tokens; chat templates
2. Visual instruction tuning (LLaVA-style), hallucination (POPE probe)
3. Visual Question Answering: classic classifier VQA (ViLT) vs generative VQA (Qwen3-VL)
4. VQA on documents (receipts and invoices) compared with the Week 4 pipelines; validating numbers and grounding
5. Structured output: prompt-only vs schema + repair vs constrained decoding (Outlines), pydantic validation
6. Natural-language explanations of CV output (detector + classifier results)
7. Image captioning: BLIP → BLIP-2 → Qwen3-VL; BLEU/ROUGE/CIDEr; CLIPScore
8. Contrastive learning: CLIP (InfoNCE) vs SigLIP (sigmoid loss), shared embedding space, modality gap
9. Multimodal retrieval: text↔image, Recall@K, zero-shot classification
10. Indexing a caption as text vs a raw embedding vs a hybrid (BM25, dense, RRF); FAISS index

**Success gate:** schema-valid VLM answers, plus caption, embed and text→image retrieval working.

### Week 6 — Multimodal RAG (pillar: Vision + Generative AI)

1. Document ingestion: digital vs scanned PDFs, element schema with metadata (page, bbox, type, source)
2. Layout parsing: PyMuPDF, Table Transformer, OpenCV figure regions, Unstructured.io
3. Visual representation: figure captioning and chart reading, CLIP embeddings of figures
4. Chunking strategies for text and tables (LangChain splitters, markdown vs linearized tables)
5. Text embeddings (BGE) and image embeddings (CLIP)
6. Vector indexing: FAISS (flat vs HNSW), Qdrant (payloads, filtered search), provenance
7. Retrieval: text-only, text+image, cross-modal; hybrid with reciprocal-rank fusion; cross-encoder reranking
8. Retrieval metrics: recall@k, hit@1, MRR
9. Grounded generation: prompt rules, citations, abstention, closed-book baseline
10. An end-to-end pipeline (`MultimodalRAG.ask`) kept as the baseline for Week 7

**Success gate:** text and text+image queries retrieve the right context and get grounded, cited answers.

### Week 7 — Agentic Multimodal RAG (pillar: Vision + Generative AI)

1. Agent fundamentals: agent loop, tool design, tool descriptions, typed arguments
2. The ReAct pattern (Thought → Action → Observation), native tool calling
3. Guardrails: step caps, error observations, logging every tool call (JSONL)
4. Query routing (heuristic vs LLM) and planning (sub-questions per modality and document)
5. LangGraph: state, nodes, conditional edges, recursion limit
6. When *not* to use an agent (simple vs multi-hop questions)
7. Self-correction: grading the evidence, rewriting the query, re-retrieving, verifying the answer
8. Text+image agentic queries: identify, compare, cross-reference
9. Agent evaluation: correctness, evidence recall, citation accuracy, faithfulness (LLM as judge, Ragas idea), cost and efficiency

**Success gate:** the agent passes the eval suite, beats the Week 6 baseline, and self-corrects on at least one case.

### Week 8 — Deployment & Tracking (pillar: Industrial-Grade Deployment)

1. Model optimization: dynamic and static INT8 quantization (PTQ), unstructured and structured pruning, knowledge distillation
2. Measuring trade-offs: accuracy (per class), size, latency
3. ONNX export, ONNX Runtime, execution providers (CPU/CUDA/TensorRT), ONNX INT8 quantization
4. Serving: FastAPI (`/health`, `/predict`, `/gradcam`, `/drift`, `/ask`), Uvicorn, Docker, docker-compose
5. Experiment tracking: MLflow (params, metrics, artifacts, comparing runs)
6. Model Registry (aliases such as `champion`), data versioning with DVC, reproducibility
7. Drift detection: reference distribution, corruption streams (CIFAR-10-C style), KS test, PSI, χ², threshold tuning, Evidently
8. Explainability: Grad-CAM heatmaps

**Success gate:** an HTTP-served optimized model, a drift monitor that flags the injected drift but stays quiet on clean data, and sensible Grad-CAM heatmaps.

---

## Part B — Topics used in the code (week → day)

### Week 1 — [week1_foundations/](week1_foundations/)

| Day | File | Topics in code | Libraries / models / data |
|---|---|---|---|
| 1 | [day1_pytorch_basics.py](week1_foundations/day1_pytorch_basics/day1_pytorch_basics.py) | Tensor creation and ops, GPU ops, autograd basics, autograd with tensors, linear regression from scratch | `torch`, `numpy` |
| 2 | [day2_image_fundamentals.py](week1_foundations/day2_image_fundamentals/day2_image_fundamentals.py) | Synthetic images, loading images, color spaces, transformations, filtering, image as tensor | OpenCV, Pillow, torchvision transforms, matplotlib |
| 3 | [day3_training_loop.py](week1_foundations/day3_training_loop/day3_training_loop.py) | `SimpleMLP`, manual train/eval loop, learning curves | torch.nn/optim, torchvision datasets (FashionMNIST / CIFAR-10) |
| 4 | [day4_convolutions.py](week1_foundations/day4_convolutions/day4_convolutions.py) | Visualizing convolution, conv mechanics (stride/padding/pooling), `SimpleCNN`, filter visualization | torch, CIFAR-10 |
| 5 | [day5_cnn_cifar10.py](week1_foundations/day5_cnn_cifar10/day5_cnn_cifar10.py), [utils.py](week1_foundations/day5_cnn_cifar10/utils.py) | `CIFAR10CNN`, train/val split (`random_split`), curves, confusion matrix, parameter count, CLI args | torch, scikit-learn, **W&B (optional)**, CIFAR-10 |

### Week 2 — [week2_classification/](week2_classification/)

| Day | File | Topics in code | Libraries / models / data |
|---|---|---|---|
| 1 | [day1_modern_architectures.py](week2_classification/day1_modern_architectures/day1_modern_architectures.py) | Loading backbones, architecture summary, residual block explained, EfficientNet explained, inference benchmark, first-layer filters, feature maps, ImageNet inference | torchvision ResNet (18/50), EfficientNet-B0, ImageNet labels |
| 2 | [day2_transfer_learning.py](week2_classification/day2_transfer_learning/day2_transfer_learning.py) | Feature extraction vs fine-tuning, trainable-parameter counting, comparison plots | torchvision pretrained models, CIFAR-10, **W&B (optional)** |
| 3 | [day3_custom_dataset.py](week2_classification/day3_custom_dataset/day3_custom_dataset.py) | Dataset discovery and analysis, mean/std computation, `CustomImageDataset` vs `ImageFolder`, `WeightedRandomSampler`, batch visualization | Intel Image Classification (Kaggle), W&B (optional) |
| 4 | [day4_augmentation.py](week2_classification/day4_Data_Augmentation/day4_augmentation.py) | Albumentations pipelines, augmentation visualization, ablation experiment, ablation table, per-class comparison | Albumentations, `ToTensorV2`, W&B (optional) |
| 5 | [day5_ablation_evaluation.py](week2_classification/day5_ablation_evaluation/day5_ablation_evaluation.py) | Seeding/reproducibility, stratified subset, experiment dataclass, confusion matrix, classification report, markdown report | ResNet-18, Albumentations, scikit-learn, seaborn, pandas, kagglehub |

### Week 3 — [week3_detection/](week3_detection/)

| Day | File | Topics in code | Libraries / models / data |
|---|---|---|---|
| 1 | [day1_detection_concepts.py](week3_detection/day1_detection_concepts/day1_detection_concepts.py) | Box format converters, pairwise IoU, NMS from scratch, COCO-style mAP evaluator, YOLO vs Faster R-CNN | Ultralytics YOLO26n (fallback YOLO11n), torchvision Faster R-CNN, COCO128 (Kaggle) |
| 2 | [day2_yolo_custom_training.py](week3_detection/day2_yolo_custom_training/day2_yolo_custom_training.py) | Parsing VOC XML, label audit, VOC → YOLO conversion, 70/15/15 split, `data.yaml`, fine-tuning, mAP and per-class P/R, recall by object size | Ultralytics YOLO26s, Hard-Hat Detection dataset (Kaggle), Google Drive |
| 3 | [day3_segmentation.py](week3_detection/day3_segmentation/day3_segmentation.py) | Pretrained semantic segmentation (DeepLabV3), instance segmentation (Mask R-CNN, YOLO26-seg), U-Net from scratch, Dice loss, seg metrics | torchvision, Ultralytics, Oxford-IIIT Pet (Kaggle), COCO128 |
| 4 | [day4_object_tracking.py](week3_detection/day4_object_tracking/day4_object_tracking.py) | Naive IoU tracker (Hungarian / `linear_sum_assignment`), ByteTrack, BoT-SORT, tracker config, MOT evaluation | Ultralytics tracking, SciPy, OpenCV, MOT17 (Kaggle) |
| 5 | [day5_anomaly_detection.py](week3_detection/day5_anomaly_detection/day5_anomaly_detection.py) | Convolutional autoencoder, PatchCore from scratch (patch embeddings, greedy coreset), anomalib PatchCore + PaDiM, AUROC/F1 | anomalib, Wide-ResNet50-2, MVTec AD (Kaggle), scikit-learn |

### Week 4 — [week4_documents/](week4_documents/)

| Day | File | Topics in code | Libraries / models / data |
|---|---|---|---|
| 1 | [day1_ocr_preprocessing.py](week4_documents/day1_ocr_preprocessing/day1_ocr_preprocessing.py) | Preprocessing (gray, denoise, CLAHE, binarize, deskew, upscale), synthetic invoice generator + scan degradation, word F1, CER, engine comparison, degradation × preprocessing ablation | Tesseract, EasyOCR, PaddleOCR, rapidfuzz, FUNSD |
| 2 | [day2_layout_extraction.py](week4_documents/day2_layout_extraction/day2_layout_extraction.py) | LayoutLMv3 token classification (BIO), RoBERTa text-only baseline, KV linking → JSON, Donut OCR-free parsing, seqeval F1 | `microsoft/layoutlmv3-base`, `roberta-base`, `naver-clova-ix/donut-base-finetuned-cord-v2`, FUNSD, CORD |
| 3 | [day3_managed_services.py](week4_documents/day3_managed_services/day3_managed_services.py) | **Azure DI** (`prebuilt-invoice`, `prebuilt-layout` + KV, receipt/read), **AWS Textract** (`AnalyzeDocument` FORMS+TABLES, `AnalyzeExpense`), **Google Document AI (optional)**, response parsing tested on mocks, `CostMeter` budget cap, on-disk cache, common pydantic schema | `azure-ai-documentintelligence`, `boto3`, `google-cloud-documentai`, pydantic |
| 4 | [day4_kv_tables.py](week4_documents/day4_kv_tables/day4_kv_tables.py) | Hard cases (rotated, varied layout, merged cells, multi-page), label-rule KV extraction, rule-based tables vs Table Transformer, table stitching, spanning cells; Azure + Textract on the same docs | Tesseract, `TableTransformerForObjectDetection` (TATR), Azure DI, AWS Textract |
| 5 | [day5_evaluation.py](week4_documents/day5_evaluation/day5_evaluation.py) | Field accuracy, ANLS, line-item F1, TEDS / TEDS-S (APTED), cell-adjacency F1, latency, cost per page, `recommend()` comparison matrix | apted, rapidfuzz, Azure DI, AWS Textract, TATR |

### Week 5 — [week5_vlms/](week5_vlms/)

| Day | File | Topics in code | Libraries / models / data |
|---|---|---|---|
| 1 | [day1_vlm_architecture.py](week5_vlms/day1_vlm_architecture/day1_vlm_architecture.py) | `VLM` wrapper, parameter breakdown per component, visual tokens vs resolution, projector forward hook, first runs (caption/VQA/counting/OCR), POPE hallucination probe | Qwen3-VL-2B/4B-Instruct, LLaVA-OneVision-0.5B, Flickr8k (Kaggle) |
| 2 | [day2_visual_qa.py](week5_vlms/day2_visual_qa/day2_visual_qa.py) | ViLT vs Qwen3-VL VQA, receipt VQA (exact match + ANLS), invoice VQA vs Week 4 rules, arithmetic and OCR grounding checks, resolution sweep | `dandelin/vilt-b32-finetuned-vqa`, Qwen3-VL, Tesseract, SROIE (Kaggle) |
| 3 | [day3_structured_explanations.py](week5_vlms/day3_structured_explanations/day3_structured_explanations.py) | Explaining detector + classifier output, JSON modes (`prompt_only`, `schema_repair`, `constrained`), `ConstrainedRunner`, consistency checks, single-call invoice extraction | Outlines, pydantic, Ultralytics YOLO, torchvision classifier, Qwen3-VL |
| 4 | [day4_image_captioning.py](week5_vlms/day4_image_captioning/day4_image_captioning.py) | BLIP / BLIP-2 / Qwen3-VL captioning, BLEU-4 / ROUGE-L / CIDEr, CLIPScore, chart captions, retrieval via BM25 vs dense vs CLIP vs hybrid, reusable `caption_image()` | `Salesforce/blip-image-captioning-large`, `Salesforce/blip2-opt-2.7b`, open_clip, sentence-transformers, rank_bm25, pycocoevalcap |
| 5 | [day5_multimodal_embeddings.py](week5_vlms/day5_multimodal_embeddings/day5_multimodal_embeddings.py) | CLIP vs SigLIP losses, Recall@1/5/10, modality gap, text/image/composed retrieval, zero-shot CIFAR-10, RRF hybrid, reusable FAISS `MultimodalIndex` | open_clip (ViT-B-32/B-16/L-14, ViT-B-16-SigLIP), FAISS, BLIP, Flickr8k |

### Week 6 — [week6_multimodal_rag/](week6_multimodal_rag/)

Shared code lives in [tools/shared/](week6_multimodal_rag/tools/shared/): `corpus.py`, `parse.py`, `represent.py`, `index.py`, `llm.py`, `generate.py`, `pipeline.py`, `evaluate.py`, `storage.py`.

| Day | File | Topics in code | Libraries / models / data |
|---|---|---|---|
| 1 | [day1_layout_parsing.py](week6_multimodal_rag/day1_layout_parsing/day1_layout_parsing.py) | Synthetic annual-report PDFs (ReportLab) + arXiv papers, `Element` schema, digital parsing (text blocks, `find_tables()`, vector figures), scanned parsing (Tesseract + TATR + OpenCV), reading order, parse evaluation, Unstructured.io hi_res | PyMuPDF, ReportLab, Table Transformer, Unstructured, Tesseract |
| 2 | [day2_visual_representation.py](week6_multimodal_rag/day2_visual_representation/day2_visual_representation.py) | Figure captions and chart reading (fact recall, invented numbers), CLIP figure embeddings, caption vs embedding retrieval, chunking comparison, table representations | Qwen3-VL (`LLM` local/OpenRouter), open_clip, sentence-transformers (BGE-small), LangChain text splitters |
| 3 | [day3_multimodal_indexing.py](week6_multimodal_rag/day3_multimodal_indexing/day3_multimodal_indexing.py) | FAISS `IndexFlatIP` per space, save/load, Qdrant collections + payload filters, FAISS↔Qdrant parity, provenance rendering, flat vs HNSW at 100k vectors | FAISS, `qdrant-client` (in-memory/local), CrossEncoder |
| 4 | [day4_multimodal_retrieval.py](week6_multimodal_rag/day4_multimodal_retrieval/day4_multimodal_retrieval.py) | Text-only vs image-space vs hybrid (RRF) vs reranked retrieval, text+image queries, cross-modal page-neighbour expansion, recall@k / hit@1 / MRR, failure analysis | `cross-encoder/ms-marco-MiniLM-L-6-v2`, FAISS, open_clip |
| 5 | [day5_grounded_generation.py](week6_multimodal_rag/day5_grounded_generation/day5_grounded_generation.py) | Grounded prompt + JSON schema + repair, citation accuracy, grounded numbers, abstention, closed-book baseline, captions vs actual images in context, `MultimodalRAG.ask()` | Qwen3-VL local or via OpenRouter, pydantic |

### Week 7 — [week7_agentic_rag/](week7_agentic_rag/)

Shared code: [tools/shared/agent.py](week7_agentic_rag/tools/shared/agent.py) (Toolbox, ReAct, Router, Planner, Grader, Rewriter, LangGraph `Agent`) and [agent_eval.py](week7_agentic_rag/tools/shared/agent_eval.py).

| Day | File | Topics in code | Libraries / models / data |
|---|---|---|---|
| 1 | [day1_agent_tools.py](week7_agentic_rag/day1_agent_tools/day1_agent_tools.py) | Tools (`search_metadata`, `parse_document`, `retrieve_multimodal`, `caption_image`, `answer_vqa`), tool descriptions, ReAct loop validated with pydantic, step cap, JSONL logging, vs Week 6 baseline, native tool calling | Week 6 stack + pydantic, Qwen3-VL (local/OpenRouter) |
| 2 | [day2_planning_routing.py](week7_agentic_rag/day2_planning_routing/day2_planning_routing.py) | Heuristic vs LLM router, planner (1–4 sub-questions), LangGraph `route → plan → execute ⇄ rewrite → synthesize → verify`, traces, agent vs direct path | LangGraph `StateGraph` |
| 3 | [day3_self_correction.py](week7_agentic_rag/day3_self_correction/day3_self_correction.py) | Entity-resolution questions, LLM grader, query rewriting with fallback, self-correction ablation, forced failure + recovery, numeric verification, tool-call budget | LangGraph, difflib |
| 4 | [day4_text_image_agent.py](week7_agentic_rag/day4_text_image_agent/day4_text_image_agent.py) | Image+text queries: identify, compare (altered charts), cross-reference (image → figure → document → table) | open_clip, Qwen3-VL |
| 5 | [day5_evaluation.py](week7_agentic_rag/day5_evaluation/day5_evaluation.py) | Eval suite: correctness, evidence recall/precision, citation accuracy, LLM-as-judge faithfulness (Ragas idea), tool/LLM calls, latency, retries vs baseline | pydantic `Judgement`, pandas |

### Week 8 — [week8_deployment/](week8_deployment/)

Shared code: [tools/shared/classifier.py](week8_deployment/tools/shared/classifier.py) (Intel-image ResNet-18 from Week 2). Serving apps: [serving/](week8_deployment/serving/).

| Day | File | Topics in code | Libraries / models / data |
|---|---|---|---|
| 1 | [day1_optimization.py](week8_deployment/day1_optimization/day1_optimization.py) | FP32 baseline, dynamic INT8, static INT8 PTQ (FX, conv-bn-relu fusion, calibration), global L1 unstructured pruning, structured channel pruning, distillation to MobileNetV3-Small, size and latency | `torch.ao.quantization`, `torch.nn.utils.prune`, torch-pruning, ResNet-18, Intel Image Classification (Kaggle) |
| 2 | [day2_onnx_export.py](week8_deployment/day2_onnx_export/day2_onnx_export.py) | `torch.onnx.export` with dynamic batch, `onnx.checker`, parity check, ORT graph optimizations, CPU/CUDA/TensorRT providers, ONNX static QDQ INT8 with a calibration reader, benchmarks | onnx, onnxruntime(-gpu), TensorRT EP |
| 3 | [day3_serving.py](week8_deployment/day3_serving/day3_serving.py), [classifier_api](week8_deployment/serving/classifier_api/), [rag_api](week8_deployment/serving/rag_api/) | FastAPI classifier (`/health`, `/predict`, `/gradcam`, `/drift`, `/metrics`) on ONNX Runtime; RAG API (`/health`, `/ask` with `mode=agent|pipeline`); Dockerfiles, docker-compose, health checks | FastAPI, Uvicorn, Docker, ONNX Runtime, Week 6/7 RAG core |
| 4 | [day4_mlflow_tracking.py](week8_deployment/day4_mlflow_tracking/day4_mlflow_tracking.py) | MLflow params/metrics/tags/artifacts, `search_runs` comparison, Model Registry + `champion` alias, DVC `add/status/checkout`, dataset hash, deterministic re-run | MLflow (local SQLite backend), DVC (local), Git |
| 5 | [day5_drift_explainability.py](week8_deployment/day5_drift_explainability/day5_drift_explainability.py) | Reference stats (brightness, contrast, sharpness, confidence, entropy, embeddings), corruption streams at severities 1–5, label shift, KS / PSI / χ² detectors, bootstrap threshold tuning, drift timeline chart, Evidently HTML report, Grad-CAM | SciPy, Evidently, pytorch-grad-cam, PIL corruptions |

---

## Part C — Services used (AWS, Azure, GCP and others)

### Cloud AI services (paid, need accounts and keys)

| Service | Provider | Week / Day | What it is used for | Credentials (Colab secrets / env) |
|---|---|---|---|---|
| **Azure AI Document Intelligence** | **Azure** | W4 D3, D4, D5 | `prebuilt-invoice`, `prebuilt-layout` (+ key-value add-on), `prebuilt-receipt`, `prebuilt-read`; optional custom model | `AZURE_DI_ENDPOINT`, `AZURE_DI_KEY`, optional `AZURE_CUSTOM_MODEL_ID` |
| **AWS Textract** | **AWS** | W4 D3, D4, D5 | `AnalyzeDocument` (FORMS + TABLES), `AnalyzeExpense` (invoice/receipt fields + line items) via `boto3` | `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_DEFAULT_REGION` |
| **Google Document AI** | **GCP** | W4 D3 only (optional) | A third managed provider for comparison | GCP project, location and processor settings |

Notes:
- Week 4 is the **only** week that calls Azure, AWS or GCP. Days 3–5 include mock responses and an on-disk cache, so the parsing code runs without keys, and a `CostMeter` page/USD cap refuses calls over budget.
- **No AWS or Azure is used for deployment.** Week 8 serves locally with Docker; MLflow uses a local SQLite file and DVC uses a local cache.

### LLM / model APIs

| Service | Week / Day | Use | Notes |
|---|---|---|---|
| **OpenRouter** (`openrouter.ai`, OpenAI-compatible API) | W6 D2–D5, W7 D1–D5, W8 D3 (`rag_api`, docker-compose) | Optional LLM/VLM backend (default `qwen/qwen3-vl-8b-instruct`) | `OPENROUTER_API_KEY`. The default backend is `local` (Qwen3-VL-4B on the Colab GPU); docker-compose sets `LLM_BACKEND=openrouter` |
| **Hugging Face Hub** | W4–W8 | Downloading model weights (Qwen3-VL, LayoutLMv3, Donut, BLIP/BLIP-2, ViLT, TATR, BGE, cross-encoder) | Free; inference runs locally, not through HF APIs |

### Platforms and tooling (free tier or local)

| Service / tool | Type | Where used |
|---|---|---|
| **Google Colab** (T4 GPU) + Colab Secrets | Compute | W2–W8 notebooks (`WeekN_DayN_Colab.ipynb`) |
| **Google Drive** | Storage | W3 D2, W5, W6–W8 (datasets, artifacts, indexes, checkpoints) |
| **Kaggle** (`kagglehub`) | Datasets | W2 (Intel images), W3 (COCO128, Hard-Hat, Oxford Pets, MOT17, MVTec AD), W5 (Flickr8k, SROIE), W8 (Intel images) |
| **Weights & Biases** | Experiment tracking (optional) | W1 D5, W2 D2–D4 |
| **MLflow** | Tracking + Model Registry (local) | W8 D4 |
| **DVC** | Data versioning (local) | W8 D4 |
| **Docker / docker-compose** | Containers (local) | W8 D3 |
| **Qdrant** | Vector DB (local/in-memory client) | W6 D3 |
| **FAISS** | Vector index (library) | W5 D5, W6 D3–D5, W7, W8 D3 |
| **Evidently** | Drift reports (library) | W8 D5 |

### Service summary by week

| Week | AWS | Azure | GCP | Other external services |
|---|---|---|---|---|
| 1 | — | — | — | W&B (optional) |
| 2 | — | — | — | W&B (optional), Kaggle, Colab |
| 3 | — | — | — | Kaggle, Colab, Google Drive |
| 4 | **Textract** | **Document Intelligence** | Document AI (optional) | Colab, Hugging Face |
| 5 | — | — | — | Kaggle, Colab, Drive, Hugging Face |
| 6 | — | — | — | OpenRouter (optional), Colab, Drive, Hugging Face |
| 7 | — | — | — | OpenRouter (optional), Colab, Drive |
| 8 | — | — | — | OpenRouter (RAG API), Kaggle, Colab, Drive; MLflow/DVC/Docker run locally |
