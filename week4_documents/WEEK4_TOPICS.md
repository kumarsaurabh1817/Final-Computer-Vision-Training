# Week 4 — All Topics at a Glance

A short explanation of every concept, tool, model and metric used in Week 4 (Document Understanding), grouped by theme. The **Day** column shows where it is used.

---

## 1. Document understanding basics

| Topic | Short explanation | Day |
|---|---|---|
| Document understanding | Turning a document image/PDF into structured data (fields, tables, JSON) instead of just text. | All |
| Key-value pair (KV) | A label and its value, e.g. `Invoice Date:` → `2026-03-14`. The core unit of form/invoice extraction. | 2–5 |
| Table extraction | Recovering rows, columns, headers and merged cells from a table region, not just its text. | 3–5 |
| Line items | The rows of an invoice table (description, quantity, unit price, amount). | 3–5 |
| Self-hosted vs managed | Self-hosted = you run open-source models on your hardware; managed = a cloud API does it and bills per page. | 3–5 |
| Synthetic documents | Invoices rendered in code with exact ground truth. Free labels, controllable hard cases, no confidential data. | All |
| Ground truth | The correct answer used for scoring (text, fields, table cells). | All |

## 2. OCR (Optical Character Recognition)

| Topic | Short explanation | Day |
|---|---|---|
| OCR | Converting images of text into machine-readable text. Two stages: text **detection** (where) and **recognition** (what). | 1 |
| Tesseract | Classic open-source OCR engine (LSTM recognizer since v4). Fast on CPU, best on clean printed pages. | 1, 2, 4, 5 |
| `pytesseract` | Python wrapper that calls the Tesseract binary; `image_to_data` returns words with boxes and confidences. | 1, 2, 4, 5 |
| Page segmentation mode (PSM) | Tells Tesseract what layout to expect: 3 auto, 4 single column, 6 one block, 11 sparse text. | 1 |
| EasyOCR | PyTorch OCR: CRAFT text detector + CRNN recognizer. Easy setup, 80+ languages, GPU. | 1 |
| PaddleOCR (PP-OCRv5) | Baidu's OCR toolkit: DB detector + SVTR recognizer. Often most accurate on dense/rotated text. | 1 |
| CRAFT / DB | Text detectors: CRAFT predicts character regions and links; DB (Differentiable Binarization) segments text regions. | 1 |
| CRNN / SVTR | Text recognizers: CRNN = CNN + RNN + CTC loss; SVTR = vision-transformer recognizer. | 1 |
| Confidence score | The engine's certainty per word. Useful for routing low-confidence documents to human review if it is calibrated. | 1 |
| Calibration | Whether a confidence of 0.8 really means "right 80% of the time". | 1 |
| Reading order | The order a human reads text (lines top-down, words left-right). OCR output order can differ. | 1 |

## 3. Image preprocessing for OCR

| Topic | Short explanation | Day |
|---|---|---|
| Grayscale | Drop colour; text information is in intensity. | 1 |
| Denoising | Remove speckle that OCR reads as fake characters: median blur (fast), non-local means (strong), Gaussian. | 1 |
| Skew / deskew | A page scanned at an angle; deskew rotates it back so text lines are horizontal. | 1, 4 |
| Projection profile | Deskew method: sum ink per row at many angles; the row profile is "spikiest" (max variance) at the correct angle. | 1, 4 |
| CLAHE | Contrast Limited Adaptive Histogram Equalization: local contrast boost for faded/uneven scans. | 1 |
| Binarization / thresholding | Convert to black text on white. **Otsu** picks one global threshold; **adaptive** thresholds per neighbourhood (handles shadows). | 1 |
| Upscaling | OCR engines prefer ~25–35 px text height; low-res scans are resized up first. | 1 |
| Scan degradation | Simulated skew, blur, noise, low contrast and JPEG artifacts to test robustness. | 1, 4 |
| Ablation | Switching one step on/off at a time to measure its effect. | 1 |

## 4. Layout-aware models

| Topic | Short explanation | Day |
|---|---|---|
| Why layout matters | Meaning depends on position: a value sits right of / below its label. Text-only models lose this. | 2 |
| LayoutLMv3 | Microsoft transformer taking text tokens + each word's 2D box + image patches. Fine-tuned for token classification. Licence CC BY-NC-SA (non-commercial). | 2 |
| Bounding-box normalization (0–1000) | LayoutLM models expect word boxes scaled to a 0–1000 grid regardless of page size. | 2 |
| Token classification | Predicting a label for every token/word (like NER). | 2 |
| BIO tagging | `B-` = first word of an entity, `I-` = inside, `O` = outside. Lets word labels encode multi-word entities. | 2 |
| Sub-word tokens & `-100` | Words split into sub-words; only the first gets the label, the rest get `-100` so the loss ignores them. | 2 |
| FUNSD | Dataset of 199 noisy scanned forms with entity labels (header/question/answer/other) and question→answer links. | 1, 2 |
| Entity linking (relation extraction) | Connecting each answer to its question. Done here with a geometric nearest-left/above heuristic. | 2 |
| RoBERTa text-only baseline | Same task without boxes or image, to measure how much layout adds. | 2 |
| Donut | OCR-free model: Swin image encoder + BART decoder that *generates* JSON directly from pixels. MIT licence. | 2 |
| `token2json` | Donut processor helper converting the generated `<s_field>value</s_field>` sequence into JSON. | 2 |
| CORD | Receipt dataset with JSON ground truth (menu items, subtotal, total); Donut's fine-tuning target. | 2 |
| Domain gap | A model trained on one document type (receipts, forms) performing poorly on another (invoices). | 2 |
| Fine-tuning loop | AdamW, linear warm-up/decay schedule, mixed precision (`torch.autocast` + `GradScaler`), gradient clipping. | 2 |

## 5. Managed cloud services

| Topic | Short explanation | Day |
|---|---|---|
| Azure AI Document Intelligence | Microsoft's managed service (formerly Form Recognizer). Models: read, layout, prebuilt invoice/receipt, custom. | 3–5 |
| `prebuilt-layout` | Azure model returning text, tables (with spans) and, with the `keyValuePairs` feature, generic KV pairs. | 3 |
| `prebuilt-invoice` | Azure model returning typed invoice fields (`InvoiceId`, `InvoiceTotal`, `Items[]`) plus tables, no training. | 3–5 |
| Azure custom models | Your own fields trained from ~5+ labeled samples in Studio: *template* (fixed layout) or *neural* (varied layouts). | 3 |
| Long-running operation / poller | Azure analysis is asynchronous: `begin_analyze_document` returns a poller; `.result()` waits. | 3 |
| AWS Textract | Amazon's managed OCR + forms + tables service. | 3–5 |
| `AnalyzeDocument` (FORMS, TABLES) | Textract API returning KEY_VALUE_SET blocks (forms) and TABLE/CELL/MERGED_CELL blocks (tables). | 3–5 |
| `AnalyzeExpense` | Textract API for invoices/receipts: normalized summary fields (`TOTAL`, `INVOICE_RECEIPT_ID`) and line items. | 3–5 |
| Block graph | Textract's flat list of blocks linked by IDs (`CHILD`, `VALUE`, `MERGED_CELL` relationships) that you traverse to rebuild structure. | 3 |
| Google Document AI | Optional third provider: processors (Invoice Parser, Form Parser) returning `entities`. | 3 |
| Free tiers | Azure F0 (500 pages/month); Textract limited free pages for 3 months. Keep test sets inside them. | 3 |
| Cost cap (`CostMeter`) | Refuses an API call before it would exceed a page/USD limit; refunds failed calls. Prevents runaway bills. | 3–5 |
| Response caching | Saving raw API responses to disk so reruns and later days cost nothing. | 3–5 |
| Credentials / secrets | Keys from environment variables or Colab Secrets; never hard-coded in a notebook. | 3 |
| Data privacy & residency | Managed services send documents off-premise to a cloud region. PII, contracts and regulations may forbid it; on-prem containers exist for Azure. | 3, 5 |

## 6. Tables and the common schema

| Topic | Short explanation | Day |
|---|---|---|
| Table Transformer (TATR) | Microsoft DETR-based models: one **detects** tables on a page, one recognizes **structure** (rows, columns, column headers, spanning cells). | 4, 5 |
| DETR | Detection Transformer: predicts a fixed set of boxes + classes with a "no object" class; no anchors or NMS. | 4 |
| Cell grid | Cells = intersections of detected rows and columns; OCR words assigned to the cell they overlap most. | 4 |
| Merged / spanning cells | A cell covering several rows or columns (`row_span`, `col_span`), e.g. a "Pricing" header over two columns. | 3–5 |
| Two-level header | Header with a group row ("Pricing") over sub-columns ("Unit Price", "Amount"). Column names are joined: "Pricing Unit Price". | 4 |
| Borderless table | Table without ruling lines; structure must be inferred from alignment. | 4 |
| Multi-page table stitching | Joining a table that continues on the next page (same columns, repeated or absent header) into one logical table. | 4, 5 |
| Rule-based extraction | Label regexes + geometry ("value right of / below label"). Transparent and cheap but brittle on new layouts. | 4, 5 |
| Phrases | Words on a line grouped until a large horizontal gap, separating a label from its value. | 4 |
| Value normalization | Converting raw strings to canonical values: money → float, dates → ISO `YYYY-MM-DD`, symbols → ISO currency codes. | 3–5 |
| Date ambiguity | `03/04/2026` is March 4 (US) or 3 April (EU); must be a per-source setting. | 4 |
| Common JSON schema (`InvoiceDoc`) | One pydantic model every approach's output is mapped into, so results are directly comparable. | 3–5 |
| pydantic | Python library for typed data models with validation; also exports JSON Schema. | 3–5 |
| JSON Schema | Standard language-independent description of a JSON document's shape (`invoice_schema.json`). | 4 |
| Normalizer / adapter | A function mapping one tool's raw output (Azure, Textract, self-hosted) into the common schema. | 3–5 |

## 7. Evaluation

| Topic | Short explanation | Day |
|---|---|---|
| Word F1 | Order-free multiset overlap between predicted and true words. | 1 |
| CER | Character Error Rate = edit distance / reference length (after aligning reading order). | 1 |
| Levenshtein (edit) distance | Minimum insertions, deletions, substitutions to turn one string into another. | 1, 5 |
| Precision / recall / F1 | Correct predictions over all predictions / over all truths / their harmonic mean. | All |
| seqeval entity F1 | Entity-level F1 for BIO-tagged sequences: span and type must both match. | 2 |
| Field-level F1 (Donut) | Flatten JSON to `(path, value)` leaves and compare as multisets. | 2 |
| Field accuracy | Fraction of key fields exactly right after normalization. | 3–5 |
| ANLS | Average Normalized Levenshtein Similarity; partial credit, 0 below 0.5 similarity. | 5 |
| Line-item F1 | Items matched by amount and fuzzy description. | 4, 5 |
| TEDS | Tree-Edit-Distance-based Similarity: tables as HTML trees, `1 − edit distance / max nodes`. Scores structure and content. | 5 |
| TEDS-S | TEDS on structure only (ignores cell text), separating structure errors from OCR errors. | 5 |
| APTED | Algorithm/library computing tree edit distance, used for TEDS. | 5 |
| GriTS | Grid Table Similarity: alternative table metric comparing tables as 2D grids. | 5 |
| Adjacency F1 | ICDAR-2013 table metric over pairs of adjacent non-empty cells. | 5 |
| Held-out set | Documents not used while building the pipeline, so scores estimate real performance. | 5 |
| Unseen layout (`layout_c`) | A layout with labels the rules never saw; measures generalization. | 5 |
| Latency per page | Wall-clock processing time per page (after model warm-up). | 1–5 |
| Cost per page | Managed: list price. Self-hosted: GPU $/hour × processing time. | 3–5 |
| Break-even volume | Pages/month where an always-on GPU costs the same as the managed per-page price. | 5 |
| Comparison matrix | Table of all approaches × metrics + qualitative factors (setup, privacy, new layouts) that backs a "use X when…, use Y when…" recommendation. | 5 |
| Failure analysis | Inspecting the worst examples per approach to see *why* a metric is low. | 5 |

## 8. Tools and libraries used

| Library | Used for |
|---|---|
| OpenCV (`cv2`) | Preprocessing: denoise, rotate, CLAHE, thresholds |
| Pillow (PIL) | Rendering synthetic invoices, image I/O, PDF writing |
| NumPy / pandas / matplotlib | Arrays, result tables, plots |
| PyTorch | Model training/inference |
| Hugging Face `transformers` | LayoutLMv3, RoBERTa, Donut, Table Transformer |
| `datasets` | Streaming CORD-v2 from the Hub |
| `seqeval` | Entity-level F1 |
| `rapidfuzz` | Fast Levenshtein distance and fuzzy string matching |
| `python-dateutil` | Parsing dates in any format |
| `pydantic` | Common schema and validation |
| `apted` | Tree edit distance for TEDS |
| `azure-ai-documentintelligence` | Azure SDK |
| `boto3` | AWS SDK for Textract |
| `google-cloud-documentai` | Optional Google SDK |
