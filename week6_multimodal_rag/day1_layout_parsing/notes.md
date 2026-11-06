# Day 1 Notes — Ingestion & Layout Parsing

## Data
- **Six synthetic annual reports** rendered with reportlab (seed 7). Each has a company overview, a quarterly results table, a revenue-by-region bar chart and a segment pie chart (plus a headcount line chart on half of them). Every number is recorded, so parsing and later answers can be scored exactly.
- **One scanned report** (`proseware_foods`): pages rasterized with slight skew, blur and noise, and no text layer.
- **arXiv papers** (first 4 pages): ResNet, U-Net, CLIP, LLaVA, ReAct. Real two-column layouts, with vector figures and real tables.
- **Your PDFs** (optional): up to 10 files in `My Drive/datasets/rag_corpus/`.

## Core ideas

**An element is the unit of a multimodal RAG system.** Everything downstream needs the same record:

| Field | Why |
|---|---|
| `doc_id`, `page`, `bbox` | Provenance: the answer can cite and highlight the exact region |
| `type` (title / text / table / figure) | Different representation per type (Day 2), filters at query time (Day 3) |
| `section` | Context for chunks ("Financial results > …") |
| `rows` / `image_path` | Tables keep their cells; figures keep their pixels for CLIP and the VLM |
| `parser` | Debugging: which route produced it |

**Two parsing routes:**

| | Digital page (text layer) | Scanned page (pixels only) |
|---|---|---|
| Text | PyMuPDF blocks, exact | Tesseract paragraphs (block/par grouping) |
| Headings | font size ≥ 1.2 × body size | word height ≥ 1.35 × median |
| Tables | `page.find_tables()` (ruling lines + text positions); cells read word by word | Table Transformer detection → OCR rows |
| Figures | embedded images + **vector drawings** clustered on a mask | non-text ink → dilate → connected components |
| Errors | merged/split regions | OCR mistakes + all of the left column |

**Vector figures** (most arXiv plots) are not images in the PDF. They are drawing commands. `get_images()` misses them. Painting each drawing's rectangle on a mask, dilating and taking connected components turns them into regions. Then nearby short text (axis ticks, legends) is absorbed into the box.

**CV layout detection** (Table Transformer, Unstructured's YOLOX) only sees pixels. It is the only option for scans and photos, and a cross-check on digital PDFs.

**Unstructured.io `hi_res`** does the whole job in one call: layout model, OCR and typed elements with coordinates. The cost is a heavy install and slower parsing.

## What to check in your run
- Parse evaluation per report: table found and its numbers recovered; figures found on the right page with captions attached; CEO, HQ and headcount in the text; headings detected. Compare digital vs scanned.
- The layout overlays: merged figures, axis labels leaking into text, two-column reading order on arXiv pages.
- Table Transformer vs `find_tables()` IoU on digital pages.
- Unstructured's element counts and time per document vs ours.

## Pitfalls
- Reading order in two-column papers: sort by column, then y. Full-width elements break simple rules.
- Text inside charts (tick labels) must not become text chunks; it pollutes retrieval with numbers out of context.
- `Table.extract()` can scramble right-aligned numbers. Reading the words inside each cell box is more robust.
- Scans: everything depends on OCR quality. Week 4's preprocessing (deskew, denoise) applies here too.

## Materials
- [PyMuPDF — text, tables, drawings](https://pymupdf.readthedocs.io/en/latest/)
- [Unstructured.io — partitioning PDFs](https://docs.unstructured.io/open-source/core-functionality/partitioning)
- [Table Transformer (Hugging Face)](https://huggingface.co/docs/transformers/model_doc/table-transformer)
