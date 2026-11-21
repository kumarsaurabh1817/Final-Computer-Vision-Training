# Week 5 — All Topics at a Glance

Every concept, model, tool, dataset and metric used in Week 5 (VLMs: VQA, captioning and multimodal embeddings), grouped by theme, each with a short explanation. The **Day** column says where it is used.

---

## 1. Vision-language model architecture

| Topic | Short explanation | Day |
|---|---|---|
| Vision-Language Model (VLM) | A model that takes images + text and generates text. Built from a vision encoder, a projector and an LLM. | All |
| Vision encoder | Usually a ViT. Cuts the image into patches and outputs one feature vector per patch. | 1, 4 |
| Vision Transformer (ViT) | A transformer applied to a sequence of image patches instead of words. | 1, 4, 5 |
| Patch / patch size | The square of pixels one ViT token covers (Qwen3-VL: 16×16; CLIP B/32: 32×32). | 1, 5 |
| Projector | Small network (LLaVA: 2-layer MLP) that maps vision features into the LLM's embedding size. | 1 |
| Patch merger | Qwen-VL's projector: joins 2×2 neighbouring patches into one token, so one visual token ≈ 32×32 px. | 1 |
| Q-Former | BLIP-2's bridge: 32 learned queries cross-attend to image features and become a fixed 32-token prompt. | 4 |
| Visual tokens | Image features after projection, inserted into the LLM's input. They cost context length and latency like words. | 1, 2 |
| Image placeholder token | `<|image_pad|>` / `<image>` in the chat template. The processor expands it into one token per visual token. | 1 |
| Chat template | The model-specific text format (roles, special tokens, image slots) that turns a message list into a prompt. | 1–4 |
| Processor | Hugging Face object that combines the image processor (resize, normalize, patchify) and the tokenizer. | 1–5 |
| Dynamic / native resolution | Encoding an image at (nearly) its own size and aspect ratio instead of a fixed square. | 1 |
| AnyRes tiling | LLaVA-OneVision's high-resolution trick: several crops plus a downscaled overview. | 1 |
| Prefill vs decode | Prefill = reading the whole prompt (scales with visual tokens); decode = generating output tokens one at a time. | 1 |
| Greedy decoding | `do_sample=False`: always pick the most likely token. Reproducible, preferred for extraction. | All |
| fp16 / bf16 | Half-precision weights. T4 GPUs have no native bf16, so the wrapper uses fp16 there. | All |
| SDPA attention | PyTorch's fused scaled-dot-product attention; the fast path on GPUs without FlashAttention-2 (e.g. T4). | 1–4 |

## 2. Training VLMs

| Topic | Short explanation | Day |
|---|---|---|
| LLaVA | The canonical open VLM: CLIP encoder + MLP projector + Vicuna/Llama LLM, trained in two stages. | 1 |
| Stage 1: feature alignment | Train only the projector on image-caption pairs; encoder and LLM frozen. | 1 |
| Stage 2: visual instruction tuning | Train projector + LLM on image conversations so the model follows instructions about images. | 1 |
| Loss masking (`-100`) | Only assistant tokens are supervised; image and user tokens are context with label `-100`. | 1 |
| Contrastive pretraining | Learn paired encoders by pulling matched image-text pairs together and pushing others apart. | 5 |
| InfoNCE (CLIP loss) | Symmetric cross-entropy over the N×N similarity matrix of a batch (rows = images, columns = texts). | 5 |
| Temperature | Learned scalar that sharpens the softmax over similarities in the CLIP loss. | 5 |
| SigLIP loss | Sigmoid loss per (image, text) pair instead of a batch softmax; works with smaller batches. | 5 |
| CapFilt | BLIP's data bootstrapping: a captioner writes synthetic captions, a filter removes noisy pairs. | 4 |

## 3. Models used

| Model | Short explanation | Day |
|---|---|---|
| Qwen3-VL-4B-Instruct | Current open-weight VLM (Apache 2.0). Strong OCR / DocVQA; fits a T4 in fp16 (~9 GB). The 2B variant is ~2× faster. | 1–4 |
| LLaVA-OneVision-0.5B | Small LLaVA-architecture model (SigLIP + MLP + Qwen2-0.5B) for architecture and hallucination comparison. | 1 |
| ViLT | Vision-and-Language Transformer fine-tuned on VQAv2; classifies over 3,129 fixed answers. | 2 |
| YOLO11n | Ultralytics detector (COCO classes) whose output the VLM explains. | 3 |
| ResNet-50 (ImageNet) | torchvision classifier whose top-3 labels the VLM checks. | 3 |
| BLIP-large | Encoder-decoder captioning model (2022). Fast, short generic captions. | 4, 5 |
| BLIP-2 OPT-2.7B | Frozen ViT + Q-Former + frozen OPT LLM; captioning and simple VQA. | 4 |
| CLIP (ViT-B/32, B/16, L/14) | OpenAI's contrastive image-text encoders; shared embedding space. | 4, 5 |
| SigLIP ViT-B/16 | Google's sigmoid-loss variant of CLIP. | 5 |
| all-MiniLM-L6-v2 | Small sentence-transformer used to embed captions for text-side retrieval. | 4, 5 |

## 4. Visual Question Answering (VQA)

| Topic | Short explanation | Day |
|---|---|---|
| VQA | Answer a natural-language question about an image. | 1, 2 |
| Classification VQA | Pick an answer from a fixed vocabulary (ViLT/VQAv2). Fast but closed. | 2 |
| Generative VQA | Generate the answer as free text (VLMs). Open-ended, can read text. | 2 |
| Document VQA (DocVQA) | VQA on documents: requires reading, locating and formatting values. | 2 |
| Per-field VQA | One question per field. Simple but repeats the image prefill for every field. | 2 |
| Multi-image prompts | Several images in one message (multi-page invoices, image comparison). | 1, 2, 3 |
| Resolution vs accuracy | Small print needs pixels; fewer pixels mean fewer tokens and faster, until digits blur. | 1, 2 |
| SROIE | ICDAR 2019 receipt dataset with company/date/address/total ground truth. | 2 |

## 5. Hallucination and validation

| Topic | Short explanation | Day |
|---|---|---|
| Hallucination | Fluent output that is not supported by the image (invented objects, wrong digits). | 1–3 |
| POPE | Balanced yes/no "Is there a X in the image?" probe for object hallucination. | 1 |
| Yes-bias | Tendency to answer "yes"; a yes-ratio far above 0.5 indicates hallucination. | 1 |
| Arithmetic check | Subtotal + tax must equal total; catches misread digits without ground truth. | 2, 3 |
| OCR grounding check | An independent OCR pass must find the value on the page, otherwise it is flagged as possibly invented. | 2, 3 |
| `needs_review` routing | Send documents that fail a check to a human; measure catch rate and false alarms. | 3 |
| Injected false positive | Add a fake detection to test whether the VLM blindly trusts CV output. | 3 |
| Consistency with CV output | The explanation's objects and counts must match what the detector actually produced. | 3 |

## 6. Structured output (JSON)

| Topic | Short explanation | Day |
|---|---|---|
| Schema-valid JSON | Output that parses and validates against a declared schema (keys, types, enums, patterns). | 3 |
| Pydantic model | Python class defining the schema; `model_validate` checks output, `model_json_schema` exports it. | 3 |
| JSON Schema | Language-neutral description of the expected JSON, put in the prompt or used for constrained decoding. | 3 |
| `Literal` enums | Restrict a field to fixed values (`agree` / `partially_agree` / `disagree`) so programs can branch on it. | 3 |
| Required-but-nullable | `str | None` with no default: the key must appear, `null` means "absent". Prevents silently dropped keys. | 3 |
| Prompt-only JSON | Describe the keys in words and parse; no guarantee. | 3 |
| Validate-and-repair | Validate, send the error back, retry (bounded attempts). | 3 |
| Constrained decoding | Mask logits so only schema-valid tokens can be generated (Outlines). Valid by construction, not necessarily correct. | 3 |
| Outlines | Library for structured generation with Hugging Face models (`outlines.from_transformers`). | 3 |
| JSON parse / valid rate | Compliance metrics: share of replies that parse, and share that validate. | 3 |

## 7. Image captioning

| Topic | Short explanation | Day |
|---|---|---|
| Image captioning | Generate a sentence describing an image. | 1, 4, 5 |
| Short vs detailed captions | Short = human-like; detailed = search-oriented (subjects, setting, colours, visible text). | 4 |
| Figure captioning | Captioning charts and diagrams; useful only if the caption keeps titles and numbers. | 4 |
| Fact recall | Share of expected facts (title, series, key value) present in a figure caption. | 4 |
| `caption_image(img, style)` | Reusable captioning utility for Week 6 indexing. | 4 |
| Batched generation | Several images per `generate` call with left padding; much higher throughput. | 4 |

## 8. Caption metrics

| Topic | Short explanation | Day |
|---|---|---|
| BLEU-4 | Geometric mean of 1–4-gram precision with a brevity penalty. | 4 |
| ROUGE-L | F-score of the longest common subsequence with the references. | 4 |
| CIDEr | TF-IDF-weighted n-gram similarity to the consensus of references; the standard captioning metric. | 4 |
| Human baseline | One human caption scored against the other four; calibrates what a "good" score is. | 4 |
| CLIPScore | 2.5 × max(cos(image, caption), 0); reference-free image-text agreement. | 4 |
| pycocoevalcap | Reference implementation of the COCO caption metrics. | 4 |

## 9. Multimodal embeddings and retrieval

| Topic | Short explanation | Day |
|---|---|---|
| Shared embedding space | Images and texts mapped to vectors where similar meaning means nearby. | 4, 5 |
| L2 normalization + cosine | Unit-length vectors make inner product = cosine similarity. | 4, 5 |
| Text → image retrieval | Embed the query text, find the nearest image vectors. | 4, 5 |
| Image → text retrieval | Embed the image, find the nearest captions. | 5 |
| Image → image retrieval | Same encoder on both sides; similar scenes and near-duplicates. | 5 |
| Text + image (composed) query | `normalize(img + α·text)`: "like this image, but …". The Week 6 multimodal-query primitive. | 5 |
| Modality gap | Image and text vectors form separate clusters; matched pairs have low absolute cosine (~0.3). | 5 |
| Zero-shot classification | Classify by nearest class-name prompt, no training. | 5 |
| Prompt templates / ensembling | Wording like "a photo of a {}." and averaging many templates' embeddings. | 5 |
| Caption-as-text index | Index the image's caption with BM25 or a text embedder; readable and good for text-in-image. | 4, 5 |
| Raw-embedding index | Index the CLIP image vector; keeps visual detail no caption wrote down. | 4, 5 |
| BM25 | Classic keyword ranking (term frequency × inverse document frequency with saturation). | 4 |
| Dense text retrieval | Sentence-embedding similarity between query and caption. | 4, 5 |
| Hybrid retrieval | Combine indexes; z-score sum (Day 4) or reciprocal-rank fusion (Day 5). | 4, 5 |
| Reciprocal-rank fusion (RRF) | Score = Σ 1/(k + rank) over rankings; robust to different score scales. | 5 |
| FAISS | Facebook's vector-search library; `IndexFlatIP` = exact cosine search on normalized vectors. | 5 |
| HNSW / IVF-PQ | Approximate indexes for millions of vectors (graph-based / clustered + compressed). | 5 |
| Metadata / provenance | id, path, caption and source stored with each vector, so every hit is traceable. | 5 |
| `MultimodalIndex` | Reusable add / search_text / search_image / search_text_image / save / load class for Week 6. | 5 |

## 10. Retrieval and VQA metrics

| Topic | Short explanation | Day |
|---|---|---|
| Recall@K | The correct item is among the top K results. | 4, 5 |
| MRR | Mean of 1/rank of the first correct result. | 4 |
| Median rank (MedR) | Median position of the correct item; lower is better. | 5 |
| Exact match | Normalized prediction equals normalized truth. | 2, 3 |
| ANLS | Normalized Levenshtein similarity, 0 below 0.5; partial credit for near-misses (DocVQA standard). | 2, 3 |
| Field accuracy (Week 4) | Per-field exact match with money ±0.01, ISO dates, case-insensitive text. | 2, 3 |
| Accuracy / precision / recall | Used for the POPE probe and for the validation flags. | 1–3 |
| Latency and cost per page | Wall-clock time; self-hosted cost = GPU $/hour × seconds. | 1–4 |

## 11. Comparison with Week 4 (the weekend write-up)

| Topic | Short explanation | Day |
|---|---|---|
| Week 4 held-out set | The same synthetic invoices (`d5_*`, seed 2026, incl. unseen `layout_c`), so Week 4 and Week 5 numbers line up per document. | 2, 3 |
| Self-hosted OCR | Week 4 Tesseract + label rules (+ Table Transformer): on-prem, fast, breaks on unseen labels. | 2, 3 |
| Managed service | Azure Document Intelligence / AWS Textract from Week 4: accurate, per-page cost, data leaves premises. | 2, 3 |
| VLM extraction | One prompt, zero-shot on new layouts, on-prem, but slower and prone to confident wrong digits. | 2, 3 |
| Three-way comparison | Accuracy, unseen-layout accuracy, schema validity, latency, cost, setup, privacy, failure modes. | 3 |

## 12. Datasets and tools

| Topic | Short explanation | Day |
|---|---|---|
| Flickr8k (Kaggle `adityajn105/flickr8k`) | 8,091 photos × 5 captions; used for VQA, explanations, captioning and retrieval. | 1–5 |
| SROIE v2 (Kaggle `urbikn/sroie-datasetv2`) | Scanned receipts with key-field JSON ground truth. | 2 |
| CIFAR-10 (torchvision) | 10-class 32×32 images for zero-shot classification. | 5 |
| kagglehub | Python client that downloads and caches Kaggle datasets (auth via `KAGGLE_API_TOKEN` or `KAGGLE_USERNAME`/`KAGGLE_KEY`). | 1–5 |
| Colab Secrets | Key icon in Colab; stores credentials that `google.colab.userdata` reads. | 1–5 |
| Hugging Face `transformers` | `AutoModelForImageTextToText`, `AutoProcessor`, `pipeline`, BLIP/BLIP-2 classes. | 1–5 |
| OpenCLIP | Library with many CLIP/SigLIP checkpoints behind one API. | 4, 5 |
| Sentence-Transformers | Text (and CLIP) embedding models with a simple `encode` API. | 4, 5 |
| Ultralytics | YOLO training/inference library (Week 3), used here for detections. | 3 |
| vLLM (weekend, optional) | High-throughput serving engine for VLMs when you scale beyond notebook batch sizes. | — |
