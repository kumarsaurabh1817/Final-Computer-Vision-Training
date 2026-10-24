# Week 4 — Document Understanding

| Attribute | Detail |
|-----------|--------|
| **Pillar** | Document Intelligence |
| **Goal** | Turn documents into structured data two ways — self-hosted OCR/layout models and managed cloud services — extract key-value pairs and tables into one JSON schema, and evaluate the trade-offs. |
| **Effort** | 32–40 hours (full-time) |
| **Runtime** | Google Colab, T4 GPU (Day 3 runs on CPU). Every notebook is self-contained: all helper code is inside the day's own files, and each notebook downloads or renders its own data. |

A one-page glossary of every topic used this week is in [`WEEK4_TOPICS.md`](WEEK4_TOPICS.md).

---

## Data

| Day | Data | Size | Used for |
|-----|------|------|----------|
| 1 | [FUNSD](https://guillaumejaume.github.io/FUNSD/) test split + synthetic invoices | ~16 MB | OCR engine comparison, preprocessing ablation |
| 2 | FUNSD (train/test) + [CORD-v2](https://huggingface.co/datasets/naver-clova-ix/cord-v2) (streamed) | ~16 MB + ~100 receipts | LayoutLMv3 fine-tuning, KV linking, Donut |
| 3–4 | 5 synthetic invoices (one per hard case), seed 7 | rendered | Azure / Textract, hard cases, normalization |
| 5 | Held-out synthetic invoices, seed 2026, incl. unseen `layout_c` | rendered | Final evaluation |

The synthetic invoices are generated in code with exact ground truth in the common schema. Layouts: `standard`, `layout_b` (different labels, borderless table, EUR), `merged` (two-level header + section rows), `multipage` (table continues on page 2), `rotated` (skewed, noisy scan), and `layout_c` (held out for Day 5).

---

## How to run on Colab

1. Open [colab.research.google.com](https://colab.research.google.com) → **File → Upload notebook** → pick `dayN_*/Week4_DayN_Colab.ipynb`.
2. **Runtime → Change runtime type → T4 GPU**.
3. Days 3–5, optional: add cloud credentials under the **key icon (Secrets)**: `AZURE_DI_ENDPOINT`, `AZURE_DI_KEY`, `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_DEFAULT_REGION`. Without them the managed-service cells skip and everything else still runs.
4. **Runtime → Run all**. Outputs go to `/content/week4_outputs/dayN/`. Download what you want to commit; Colab wipes `/content` when the session ends.

Days 1, 2 and 5 have a `FULL_RUN` flag. `False` gives a quick smoke test; only report `FULL_RUN=True` numbers.

> 💸 Managed services bill per page. Every call goes through a cost cap (`MAX_PAGES`, `MAX_USD`) and every response is cached to disk, so reruns are free. Days 3 and 4 use the same documents: run them in the same Colab session (or upload `week4_outputs/day3/raw`) and Day 4 costs nothing.

---

## Day-by-day

### Day 1 — OCR landscape & preprocessing
Tesseract vs EasyOCR vs PaddleOCR behind one interface, scored with word F1 and CER on real FUNSD scans and clean invoices. Preprocessing (denoise, projection-profile deskew, CLAHE, Otsu/adaptive threshold) step by step, a degradation × pipeline ablation heatmap, Tesseract PSM modes, and confidence calibration.
**Files:** [`Week4_Day1_Colab.ipynb`](day1_ocr_preprocessing/Week4_Day1_Colab.ipynb) · [`day1_ocr_preprocessing.py`](day1_ocr_preprocessing/day1_ocr_preprocessing.py) · [`notes.md`](day1_ocr_preprocessing/notes.md) · ~10–15 min

### Day 2 — Layout-aware extraction (LayoutLMv3 & Donut)
Fine-tune LayoutLMv3 on FUNSD (BIO token classification), compare with a text-only RoBERTa and with boxes zeroed, link questions to answers into key-value JSON (scored against FUNSD links), then run OCR-free Donut on CORD receipts (field-level F1) and on invoices to see the domain gap.
**Files:** [`Week4_Day2_Colab.ipynb`](day2_layout_extraction/Week4_Day2_Colab.ipynb) · [`day2_layout_extraction.py`](day2_layout_extraction/day2_layout_extraction.py) · [`notes.md`](day2_layout_extraction/notes.md) · ~20–25 min

### Day 3 — Managed services (Azure Document Intelligence & AWS Textract)
Azure `prebuilt-invoice` and `prebuilt-layout` (+ key-value pairs), custom-model notes, Textract `AnalyzeDocument` (FORMS + TABLES with merged cells) and `AnalyzeExpense`, optional Google Document AI. Parsers are tested offline on mock responses; calls are capped and cached; both outputs are mapped into the common schema and checked against ground truth.
**Files:** [`Week4_Day3_Colab.ipynb`](day3_managed_services/Week4_Day3_Colab.ipynb) · [`day3_managed_services.py`](day3_managed_services/day3_managed_services.py) · [`notes.md`](day3_managed_services/notes.md) · ~5 min

### Day 4 — KV & table deep dive, one common schema
Hard cases one by one: deskew on a rotated scan (plus an angle sweep), Table Transformer spanning cells on a two-level header, stitching a table across pages, a new label vocabulary. Four approaches (rules only, rules + Table Transformer, Azure, Textract) normalized into the pydantic `InvoiceDoc` schema (exported as JSON Schema) and compared per hard case.
**Files:** [`Week4_Day4_Colab.ipynb`](day4_kv_tables/Week4_Day4_Colab.ipynb) · [`day4_kv_tables.py`](day4_kv_tables/day4_kv_tables.py) · [`notes.md`](day4_kv_tables/notes.md) · ~5–10 min

### Day 5 — Structured evaluation & comparison matrix
Held-out set with an unseen layout. Field accuracy and ANLS, line-item F1, TEDS / TEDS-S (APTED tree edit distance) and adjacency F1, latency and cost per page. Breakdowns by layout and field, accuracy-vs-cost plots, a comparison matrix (`comparison_matrix.md/.csv`) and a recommendation generated from the numbers, including break-even volume.
**Files:** [`Week4_Day5_Colab.ipynb`](day5_evaluation/Week4_Day5_Colab.ipynb) · [`day5_evaluation.py`](day5_evaluation/day5_evaluation.py) · [`notes.md`](day5_evaluation/notes.md) · ~10–15 min

---

## Folder structure

```
week4_documents/
├── README.md
├── WEEK4_TOPICS.md                  ← every topic of the week with a short explanation
├── requirements.txt                 ← Week 4 deps (notebooks pip-install their own)
├── tools/py_to_ipynb.py             ← regenerates notebooks from the .py sources
├── day1_ocr_preprocessing/          ← Week4_Day1_Colab.ipynb, .py, notes.md
├── day2_layout_extraction/          ← Week4_Day2_Colab.ipynb, .py, notes.md
├── day3_managed_services/           ← Week4_Day3_Colab.ipynb, .py, notes.md
├── day4_kv_tables/                  ← Week4_Day4_Colab.ipynb, .py, notes.md
├── day5_evaluation/                 ← Week4_Day5_Colab.ipynb, .py, notes.md
├── models/                          ← optional: saved LayoutLMv3 weights
└── results/                         ← put downloaded plots, CSV/JSON, comparison matrix here
```

Each day folder is self-contained: the `.py` holds every helper it needs (preprocessing, invoice generator, schema, extractors), so nothing is imported from outside the folder. The `.py` is the source of truth: percent-format cells (`# %%`) that also run as plain scripts. After editing one, rebuild its notebook:

```bash
python tools/py_to_ipynb.py            # all days
python tools/py_to_ipynb.py day4_kv_tables/day4_kv_tables.py
```

Local/WSL2 runs need the Tesseract binary: `sudo apt-get install tesseract-ocr`.

---

## Deliverable
- [ ] Invoice/receipt pipeline extracting KV pairs and tables to a common JSON schema via (a) a self-hosted OCR/layout approach (Days 1, 2, 4) and (b) a managed service — Azure Document Intelligence or AWS Textract (Days 3, 4)
- [ ] Comparison matrix scoring them on accuracy, table extraction, latency, and cost on a held-out set (Day 5)

## Success gate
- [ ] Both approaches produce valid structured JSON on unseen invoices (schema round-trip in Days 4–5; `schema_failures` column = 0)
- [ ] The comparison matrix gives a defensible "use X when…, use Y when…" recommendation backed by numbers

> ⚠️ Cloud services incur per-page cost: keep the caps, stay in free tiers. Managed services send documents off-premise; note data-residency and privacy implications before using real documents.

---

## Results (fill in after your Colab runs)

| Day | Metric | Value |
|-----|--------|-------|
| 1 | FUNSD word F1 Tesseract / EasyOCR / PaddleOCR | __ / __ / __ |
| 1 | Best preprocessing gain for Tesseract (degradation, pipeline, ΔF1) | __ |
| 2 | FUNSD entity F1 LayoutLMv3 / RoBERTa text-only | __ / __ |
| 2 | KV pair F1 (predicted entities) / Donut CORD field F1 | __ / __ |
| 3 | Fields correct Azure / Textract (standard layout, /9) | __ / __ |
| 4 | Merged-header line items recovered: rules / TATR / Azure / Textract | __ / __ / __ / __ |
| 5 | Field accuracy self-hosted / Azure / Textract (held-out) | __ / __ / __ |
| 5 | TEDS self-hosted / Azure / Textract | __ / __ / __ |
| 5 | Unseen `layout_c` field accuracy self-hosted / best managed | __ / __ |
| 5 | Cost per 1k pages self-hosted / Azure / Textract | __ / __ / __ |

## Weekend write-up
Trade-offs (accuracy vs cost vs setup vs data privacy/on-prem) and where each approach fails — Day 5 ends with a template. This sets up the VLM comparison in Week 5 and the RAG ingestion layer in Week 6.
