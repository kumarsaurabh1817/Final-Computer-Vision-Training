# Week 6 — Multimodal RAG (Parsing, Indexing & Retrieval)

| Attribute | Detail |
|-----------|--------|
| **Pillar** | Vision + Generative AI |
| **Goal** | Build the retrieval backbone of a multimodal RAG system: parse documents with layout understanding, extract text + tables + figures with metadata, caption and embed the visual content, index everything, retrieve on text or text+image queries, and answer with citations. |
| **Effort** | 32–40 hours (full-time) |
| **Runtime** | Google Colab, T4 GPU. Each notebook is self-contained (shared helper code is copied in). Results are cached on Google Drive so later days and weeks reuse them. |

A list of every topic used this week, with a short explanation, is in [`week6notes.md`](week6notes.md).

---

## Data — nothing to download

| Corpus part | Where it comes from | Why |
|---|---|---|
| 6 synthetic annual reports (1 scanned) | rendered in the notebook with reportlab, seed 7 | exact ground truth: which page holds which fact; charts contain facts that are **not** in the text |
| First 4 pages of 5 arXiv papers (ResNet, U-Net, CLIP, LLaVA, ReAct) | downloaded from arxiv.org by the notebook | real two-column layouts, vector figures, real tables |
| Your own PDFs (optional) | put up to 10 PDFs in **My Drive/datasets/rag_corpus/** | try the pipeline on your documents |

A question set (~90 questions: text, table, figure, text+image, unanswerable and multi-hop) is generated with the corpus, each with the expected answer and expected sources `(document, page, element type)`.

## Where results are stored

The notebooks mount Google Drive and keep everything later days need in **`My Drive/cv_course_artifacts/week6_rag/`**:

```
week6_rag/
├── pdfs/                     ← the corpus
├── query_images/             ← images for text+image questions
├── corpus.json               ← documents, facts, page map, questions
├── crops/                    ← figure crops
├── elements.jsonl            ← parsed elements (Day 1), with VLM captions (Day 2)
├── index/                    ← FAISS text + image index + item metadata (Day 3)
└── week6_baseline_results.csv← Day 5 answers, the baseline for Week 7
```

Run the days in order. Each day also works alone, but rebuilds what is missing (Day 4 or 5 alone takes ~10 extra minutes the first time). Set `USE_DRIVE = False` in the helper cell to keep everything in `/content` instead.

---

## How to run on Colab

1. **File → Upload notebook** → `dayN_*/Week6_DayN_Colab.ipynb`.
2. **Runtime → Change runtime type → T4 GPU**.
3. **Runtime → Run all**. Click **Connect to Google Drive** when asked.
4. Optional: to use a hosted model instead of the local Qwen3-VL-4B, add `OPENROUTER_API_KEY` in Colab **Secrets** and set `LLM_BACKEND = "openrouter"`.

Per-day outputs (plots, CSV, JSON) go to `/content/week6_outputs/dayN/`. Download what you want to commit into `week6_multimodal_rag/results/dayN/`.

---

## Day-by-day

### Day 1 — Ingestion & layout parsing
Corpus generation; PyMuPDF parsing (text blocks, font-size headings, `find_tables()`, raster images, vector figures from clustered drawings); the scanned-page route (Tesseract + Table Transformer + OpenCV); per-element metadata; parse evaluation against truth (digital vs scanned); Table Transformer vs `find_tables()`; Unstructured `hi_res` comparison.
**Files:** [`Week6_Day1_Colab.ipynb`](day1_layout_parsing/Week6_Day1_Colab.ipynb) · [`day1_layout_parsing.py`](day1_layout_parsing/day1_layout_parsing.py) · [`notes.md`](day1_layout_parsing/notes.md) · ~10–15 min

### Day 2 — Visual content representation
Chart-reading VLM captions vs generic captions (fact recall, invented numbers); CLIP embeddings of figures (CLIP groups charts by type, not content); caption-as-text vs CLIP for figure questions; chunking strategies (element / fixed / section-aware) and table representations scored by retrieval.
**Files:** [`Week6_Day2_Colab.ipynb`](day2_visual_representation/Week6_Day2_Colab.ipynb) · [`day2_visual_representation.py`](day2_visual_representation/day2_visual_representation.py) · [`notes.md`](day2_visual_representation/notes.md) · ~10–15 min

### Day 3 — Multimodal indexing
FAISS text space (BGE) + image space (CLIP) + metadata; save/load from Drive; provenance drawn on the page; the same data in Qdrant with payload filters and a parity check; exact vs HNSW search at 100k vectors.
**Files:** [`Week6_Day3_Colab.ipynb`](day3_multimodal_indexing/Week6_Day3_Colab.ipynb) · [`day3_multimodal_indexing.py`](day3_multimodal_indexing/day3_multimodal_indexing.py) · [`notes.md`](day3_multimodal_indexing/notes.md) · ~5 min

### Day 4 — Multimodal retrieval
Text-only vs image-only vs hybrid (RRF) vs hybrid + cross-encoder rerank, scored by source recall@5, hit@1 and MRR per question type; text+image queries (image→image, composed, text-only); cross-modal retrieval in both directions; recall vs k; failure analysis.
**Files:** [`Week6_Day4_Colab.ipynb`](day4_multimodal_retrieval/Week6_Day4_Colab.ipynb) · [`day4_multimodal_retrieval.py`](day4_multimodal_retrieval/day4_multimodal_retrieval.py) · [`notes.md`](day4_multimodal_retrieval/notes.md) · ~5 min

### Day 5 — Grounded generation & the full pipeline
Numbered sources with figure images → Qwen3-VL → JSON answer with `[S#]` citations mapped to doc/page/bbox. Correctness, citation accuracy, grounded numbers and abstention per question type; closed-book comparison; captions-only vs captions + images; retrieval vs generation misses. Saves `week6_baseline_results.csv` for Week 7.
**Files:** [`Week6_Day5_Colab.ipynb`](day5_grounded_generation/Week6_Day5_Colab.ipynb) · [`day5_grounded_generation.py`](day5_grounded_generation/day5_grounded_generation.py) · [`notes.md`](day5_grounded_generation/notes.md) · ~20–25 min

### Weekend
Document the data flow (diagram in Day 5), write the results section of this README, and keep the baseline: Week 7's agent is scored against it.

---

## Deliverable & success gate

| Requirement | Where |
|---|---|
| Ingest → parse with layout understanding → text/tables/images with metadata | Day 1 |
| Caption + embed visual content | Day 2 |
| Index with provenance | Day 3 |
| Retrieve on text and text+image queries | Day 4 |
| Grounded, cited answer | Day 5 |
| **Gate:** a text query AND a text+image query retrieve the right multimodal context and produce a cited answer without hallucinated content | Day 4 (recall by type), Day 5 (`correct`, `citation_ok`, `grounded_numbers`, unanswerable → abstain) |

> Retrieval quality lives or dies on the representation choice. Day 2 and Day 4 measure caption-as-text vs raw embedding instead of assuming.

---

## Folder structure

```
week6_multimodal_rag/
├── README.md
├── week6notes.md                    ← every topic of the week with a short explanation
├── requirements.txt
├── tools/
│   ├── shared/                      ← canonical helper modules (corpus, parse, llm, represent, index, generate, pipeline, evaluate)
│   ├── inline_shared.py             ← copies tools/shared/*.py into each day's script
│   └── py_to_ipynb.py               ← regenerates the notebooks
├── day1_layout_parsing/             ← Week6_Day1_Colab.ipynb, .py, notes.md
├── day2_visual_representation/
├── day3_multimodal_indexing/
├── day4_multimodal_retrieval/
├── day5_grounded_generation/
├── models/
└── results/
```

To change a helper, edit `tools/shared/<module>.py`, then run `python tools/inline_shared.py && python tools/py_to_ipynb.py`.
