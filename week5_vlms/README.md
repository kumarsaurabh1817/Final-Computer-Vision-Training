# Week 5 — VLMs (VQA, Captioning & Multimodal Embeddings)

| Attribute | Detail |
|-----------|--------|
| **Pillar** | Vision + Generative AI |
| **Goal** | Use a modern open VLM for VQA and for natural-language explanation of CV output, enforce schema-valid JSON, caption images and figures, and embed images and text in one space with CLIP for text↔image retrieval. |
| **Effort** | 32–40 hours (full-time) |
| **Runtime** | Google Colab, T4 GPU. Every notebook is self-contained: helper code lives inside the day's own file, and each notebook downloads its own data from Kaggle (or renders it). |

A one-page list of every topic used this week, each with a short explanation, is in [`week5notes.md`](week5notes.md).

---

## Datasets to download (Kaggle)

| Dataset | Kaggle link | Size | Used on | Folder if you upload it manually |
|---|---|---|---|---|
| **Flickr8k** (8,091 photos × 5 captions) | [kaggle.com/datasets/adityajn105/flickr8k](https://www.kaggle.com/datasets/adityajn105/flickr8k) | ~1 GB | Days 1, 2, 3, 4, 5 | `/content/datasets/flickr8k/` |
| **SROIE v2** (scanned receipts + key-field JSON) | [kaggle.com/datasets/urbikn/sroie-datasetv2](https://www.kaggle.com/datasets/urbikn/sroie-datasetv2) | ~0.5 GB | Day 2 | `/content/datasets/sroie/` |

No download is needed for the rest: the **Week 4 held-out invoices** are rendered in code (same generator, seed 2026 and ids as Week 4 Day 5), the charts on Days 4–5 are drawn with matplotlib, and **CIFAR-10** (Day 5) is downloaded by torchvision.

### Option 1 (recommended): keep the zips in Google Drive

Put `flickr8k.zip` and `sroie.zip` (renamed from Kaggle's `archive.zip`) in `My Drive/datasets/`. Each notebook mounts Drive, unzips them to the fast local disk and skips the download. CIFAR-10 is read from `My Drive/datasets/cifar-10-batches-py/` if it is there. To change the folder, edit `DRIVE_DATASETS` in the helper cell; set `USE_DRIVE = False` to turn this off.

### Option 2: Kaggle access from Colab (do this once)

1. On kaggle.com: **Settings → API → Create New Token**. You get either an API token (`KGAT_…`) or a `kaggle.json` with `username` and `key`.
2. In Colab: click the **key icon (Secrets)** in the left sidebar and add either
   - `KAGGLE_API_TOKEN` = your token, **or**
   - `KAGGLE_USERNAME` and `KAGGLE_KEY` (values from `kaggle.json`),

   and switch on **Notebook access** for each.
3. The notebooks call `kagglehub.dataset_download(...)`. It caches, so later notebooks in the same Colab session do not download again.

**Without an account/token:** download the zip from the Kaggle page, upload it to Colab, and unzip it into the folder from the table, e.g. `!unzip -q archive.zip -d /content/datasets/flickr8k`. The loader finds `captions.txt` / `entities/*.txt` anywhere inside that folder.

---

## How to run on Colab

1. Open [colab.research.google.com](https://colab.research.google.com) → **File → Upload notebook** → pick `dayN_*/Week5_DayN_Colab.ipynb`.
2. **Runtime → Change runtime type → T4 GPU**.
3. Add the Kaggle secrets (above).
4. **Runtime → Run all**. Outputs go to `/content/week5_outputs/dayN/`. Download what you want to commit into `week5_vlms/results/dayN/`, because Colab wipes `/content` when the session ends.

Every notebook has a `FULL_RUN` flag. `False` gives a quick smoke test; only report `FULL_RUN=True` numbers. `VLM_ID` can be switched to `Qwen/Qwen3-VL-2B-Instruct` for ~2× speed.

**Linking the days to Week 4:** for Day 2 and Day 3's three-way comparison, upload Week 4 Day 5's `day5_scores_per_doc.csv` (and, on Day 3, Day 2's `day2_invoice_scores.csv`) to `/content/`. Rows are matched on document id, so the comparison is per document. Without these files the notebooks still run and show their own approaches.

---

## Day-by-day

### Day 1 — How VLMs work (Qwen3-VL & LLaVA)
Vision encoder + projector + LLM: parameter breakdown per component, a forward hook on the projector, the chat template with image placeholders, visual tokens vs resolution, latency vs tokens. First runs (caption, VQA, counting, reading, two-image comparison), a LLaVA-OneVision comparison, a POPE-style hallucination probe, and what visual-instruction-tuning data and its loss mask look like.
**Files:** [`Week5_Day1_Colab.ipynb`](day1_vlm_architecture/Week5_Day1_Colab.ipynb) · [`day1_vlm_architecture.py`](day1_vlm_architecture/day1_vlm_architecture.py) · [`notes.md`](day1_vlm_architecture/notes.md) · ~15–20 min

### Day 2 — Visual Question Answering
ViLT (closed-vocabulary classifier) vs Qwen3-VL (generative) on photos and a receipt. Receipt VQA on SROIE (exact match + ANLS vs Tesseract + rules). Per-field VQA on the Week 4 held-out invoices, scored with Week 4's `field_scores` next to Week 4's rules. Arithmetic and OCR-grounding checks for invented numbers, and a resolution sweep.
**Files:** [`Week5_Day2_Colab.ipynb`](day2_visual_qa/Week5_Day2_Colab.ipynb) · [`day2_visual_qa.py`](day2_visual_qa/day2_visual_qa.py) · [`notes.md`](day2_visual_qa/notes.md) · ~20–30 min

### Day 3 — NL explanations of CV output & schema-valid JSON
YOLO11n + ResNet-50 output → VLM explanation in a pydantic schema. Three modes (`prompt_only`, `schema_repair`, Outlines `constrained`) compared on parse rate, schema-valid rate, consistency with the detector, and latency. An injected fake detection tests whether the VLM blindly trusts CV output. One-call invoice extraction into the Week 4 `Fields` schema, `needs_review` validation, and the **three-way comparison** (self-hosted OCR vs managed vs VLM) saved as `three_way_comparison.md/.csv`.
**Files:** [`Week5_Day3_Colab.ipynb`](day3_structured_explanations/Week5_Day3_Colab.ipynb) · [`day3_structured_explanations.py`](day3_structured_explanations/day3_structured_explanations.py) · [`notes.md`](day3_structured_explanations/notes.md) · ~25–35 min

### Day 4 — Image captioning
BLIP, BLIP-2 (Q-Former) and Qwen3-VL (short and detailed prompts) on 150 Flickr8k images. BLEU-4 / ROUGE-L / CIDEr against a human baseline, plus CLIPScore. Figure captioning with fact recall on charts. **Caption-as-text vs raw embedding** retrieval: BM25, dense caption, CLIP image and hybrid indexes on photos and on figures. Reusable `caption_image()`.
**Files:** [`Week5_Day4_Colab.ipynb`](day4_image_captioning/Week5_Day4_Colab.ipynb) · [`day4_image_captioning.py`](day4_image_captioning/day4_image_captioning.py) · [`notes.md`](day4_image_captioning/notes.md) · ~25–35 min

### Day 5 — Multimodal embeddings with CLIP
CLIP vs SigLIP loss on a toy batch. Text→image / image→text Recall@K for CLIP B/32, B/16, L/14 and SigLIP on 1,000 images × 5,000 captions. The modality gap and similarity distributions; text→image, image→image, image→text and text+image composed queries. Zero-shot CIFAR-10 with prompt ensembling, caption vs embedding vs RRF hybrid on the full gallery, and a reusable FAISS `MultimodalIndex` (save/load) that passes the success gate: caption + embed a new image and find it with a text query.
**Files:** [`Week5_Day5_Colab.ipynb`](day5_multimodal_embeddings/Week5_Day5_Colab.ipynb) · [`day5_multimodal_embeddings.py`](day5_multimodal_embeddings/day5_multimodal_embeddings.py) · [`notes.md`](day5_multimodal_embeddings/notes.md) · ~15–20 min

### Weekend — Write-up
- Three-way comparison (self-hosted OCR vs managed vs VLM) on the same invoices: start from Day 3's `three_way_comparison.md` and add effort, cost and failure modes in your own words.
- Consolidate `caption_image()` (Day 4) and `MultimodalIndex` (Day 5) as the utilities for Week 6.
- Optional: serve Qwen3-VL with [vLLM](https://docs.vllm.ai/en/latest/features/multimodal_inputs.html) and compare throughput with the notebook's `generate`.

---

## Deliverable & success gate

| Requirement | Where |
|---|---|
| VLM VQA | Day 2 (SROIE + invoices) |
| NL explanations of CV output | Day 3 Part A |
| Schema-constrained JSON extraction, benchmarked three ways against Week 4 | Day 3 Part B → `three_way_comparison.md` |
| Caption utility | Day 4 `caption_image()` |
| CLIP embeddings + text↔image retrieval | Day 5 benchmark + `MultimodalIndex` |
| **Gate:** schema-valid answers on test images | Day 3 `schema_valid_rate` (scenes + invoices) |
| **Gate:** caption + embed an arbitrary image, retrieve it with a text query | Day 5 section 11 (`GATE PASSED`) |

> ⚠️ VLMs hallucinate confident-but-wrong values, especially digits on invoices. Days 2–3 check every extracted figure (arithmetic + OCR grounding) instead of trusting it. Caption-as-text and raw-embedding retrieval behave differently; Days 4–5 measure both.

---

## Folder structure

```
week5_vlms/
├── README.md
├── week5notes.md                      ← every topic of the week with a short explanation
├── requirements.txt                   ← Week 5 deps (notebooks pip-install their own)
├── tools/py_to_ipynb.py               ← regenerates notebooks from the .py sources
├── day1_vlm_architecture/             ← Week5_Day1_Colab.ipynb, .py, notes.md
├── day2_visual_qa/                    ← Week5_Day2_Colab.ipynb, .py, notes.md
├── day3_structured_explanations/      ← Week5_Day3_Colab.ipynb, .py, notes.md
├── day4_image_captioning/             ← Week5_Day4_Colab.ipynb, .py, notes.md
├── day5_multimodal_embeddings/        ← Week5_Day5_Colab.ipynb, .py, notes.md
├── models/                            ← optional: saved indexes / weights
└── results/                           ← put downloaded plots, CSV/JSON, comparison tables here
```

Edit the `.py` files, then rebuild the notebooks with `python tools/py_to_ipynb.py`.
