# Week 6 — All Topics at a Glance

Every concept, tool, model and metric used in Week 6 (Multimodal RAG: parsing, indexing and retrieval), grouped by theme, each with a short explanation. The **Day** column says where it is used.

---

## 1. Multimodal RAG basics

| Topic | Short explanation | Day |
|---|---|---|
| RAG (retrieval-augmented generation) | Retrieve relevant pieces of a corpus, then let a model answer from them instead of from memory. | All |
| Multimodal RAG | The corpus and the queries contain text, tables *and* images; retrieval and generation handle all three. | All |
| Element | One parsed unit of a document (title, text block, table, figure) with metadata. | 1–5 |
| Item | What is indexed: a text chunk, a table (as text) or a figure (caption + crop). | 2–5 |
| Provenance | Document, page, bounding box, type and element ids carried from parsing to the answer. | 1, 3, 5 |
| Synthetic corpus with ground truth | Reports rendered in code so every fact's location is known and scoring is exact. | All |
| Question types | text, table, figure, text+image, unanswerable, multi-hop (Week 7). | All |

## 2. Ingestion and layout parsing

| Topic | Short explanation | Day |
|---|---|---|
| Digital vs scanned PDF | A digital page has a text layer; a scanned page is only pixels and needs OCR. | 1 |
| PyMuPDF | Fast PDF library: text blocks with fonts and positions, tables, images, vector drawings, page rendering. | 1–5 |
| Text blocks and font size | Headings detected as blocks whose font is ≥ 1.2 × the body size (body = median over characters). | 1 |
| `find_tables()` | PyMuPDF table finder from ruling lines and word positions; cells read word by word. | 1 |
| Raster vs vector figures | Raster = embedded image; vector = drawing commands (most paper plots). | 1 |
| Drawing clustering | Paint drawing rectangles on a mask, dilate, connected components → vector figure boxes. | 1 |
| Label absorption | Grow a figure box over nearby short text (axis ticks, legends) so it is not indexed as text. | 1 |
| Figure caption linking | Attach the "Figure N: …" text below/above a figure to that figure. | 1 |
| Reading order | Sort elements by column, then top-to-bottom (two-column papers). | 1 |
| Table Transformer (TATR) | DETR model that detects tables in page images; the CV route for scans. | 1 |
| OpenCV figure regions | On scans: ink that is not text or table, dilated into regions. | 1 |
| Tesseract paragraphs | OCR words grouped by block/paragraph into text elements. | 1 |
| Unstructured.io `hi_res` | Layout model (YOLOX) + OCR → typed elements with coordinates in one call. | 1 |
| Parse evaluation | Tables, figures, text facts and headings recovered per page vs truth. | 1 |

## 3. Representing content

| Topic | Short explanation | Day |
|---|---|---|
| Chart-reading caption | VLM prompt asking for chart type, title, categories, every value and the takeaway. | 2 |
| Fact recall | Share of a chart's true labels/values found in its caption. | 2 |
| Invented numbers | Numbers in a caption that are not in the chart (caption hallucination). | 2 |
| Caption-as-text | Index a figure by its caption with a text retriever. | 2, 4 |
| Raw embedding | Index a figure by its CLIP image vector. | 2, 4 |
| Chunking | Splitting text into retrievable pieces: element-level, fixed windows, section-aware. | 2 |
| `RecursiveCharacterTextSplitter` | LangChain splitter that cuts on paragraph, then sentence, then word boundaries. | 2 |
| Chunk overlap | Characters repeated between neighbouring chunks so facts on a boundary are not cut. | 2 |
| Section-aware chunks | Split each section separately and prefix "Title > Section" for context. | 2 |
| Table as markdown / linearized | Grid text vs one sentence per row; plus title/caption context. | 2 |
| BGE-small | 384-d text embedder; queries get an instruction prefix. | 2–5 |
| CLIP ViT-B/32 | Shared image-text space (Week 5); used for figure crops and text→image search. | 2–5 |

## 4. Indexing

| Topic | Short explanation | Day |
|---|---|---|
| Text space / image space | Two vector indexes: BGE over all item text, CLIP over figure crops. | 3–5 |
| FAISS `IndexFlatIP` | Exact inner-product search; = cosine on unit vectors. | 3–5 |
| Metadata store | Item records (JSON) aligned with vector rows. | 3–5 |
| Post-filtering | Over-fetch from FAISS, then keep items matching type/document filters. | 3, 4 |
| Qdrant | Vector database: collections, payloads, filtered search, upserts; in-memory or server. | 3 |
| Payload filter | Condition on metadata applied inside the vector search (`FieldCondition`, `MatchAny`). | 3 |
| Index persistence | Save/load index + metadata to Drive; refuse to load with a different embedding model. | 3 |
| HNSW | Graph-based approximate nearest-neighbour index; `M` and `efSearch` trade recall for speed. | 3 |
| Recall@10 (ANN) | Overlap of approximate and exact top-10. | 3 |

## 5. Retrieval

| Topic | Short explanation | Day |
|---|---|---|
| Text-only retrieval | Query → text space (+ CLIP text → image space). | 4, 5 |
| Text+image retrieval | Query image → CLIP image space, optionally composed with the text vector. | 4, 5 |
| Composed query | `normalize(image_vec + α · text_vec)`. | 4 |
| Cross-modal retrieval | Text query → figure; image query → related text. | 4 |
| Page-neighbour expansion | After an image hit, add the text and tables on the same page. | 4 |
| Reciprocal-rank fusion (RRF) | Σ 1/(60 + rank) across ranked lists; merges spaces with different score scales. | 4, 5 |
| Cross-encoder reranker | Model reading (query, passage) together to re-score the top candidates. | 4, 5 |
| Bi-encoder vs cross-encoder | Separate embeddings (fast, coarse) vs joint reading (slow, sharp). | 4 |
| Choosing k | Recall rises with k; so does the context the generator must read. | 4, 5 |

## 6. Grounded generation

| Topic | Short explanation | Day |
|---|---|---|
| Numbered sources `[S#]` | Retrieved items listed with type, document and page; figures attached as images. | 5 |
| Grounding rules | Use only the sources, cite them, copy numbers exactly, abstain when absent. | 5 |
| `GroundedAnswer` schema | `{answer, citations, insufficient_context}` validated with pydantic (+1 repair). | 5 |
| Citation mapping | Labels → items → doc/page/bbox for highlighting. | 5 |
| Abstention | "I don't know based on the documents" for unanswerable questions. | 5 |
| Closed-book baseline | Same model without retrieval; measures invention. | 5 |
| Images in context | Attach the actual figure crops vs captions only. | 5 |
| Retrieval miss vs generation miss | Wrong because evidence was not retrieved vs retrieved but misread. | 5 |
| Baseline pipeline | `MultimodalRAG.ask()`: one retrieve + one generate; Week 7's agent must beat it. | 5 |

## 7. Metrics

| Topic | Short explanation | Day |
|---|---|---|
| Source recall@k | Fraction of expected `(doc, page, type)` sources in the top k. | 2, 4, 5 |
| hit@1, R@1, R@3 | Expected source at rank 1 / within the top 1 or 3. | 2, 4 |
| MRR | Mean reciprocal rank of the first expected source. | 4 |
| Answer correctness | Numbers within 1 %, names as substrings, abstention for unanswerable. | 5 |
| Citation accuracy | A cited item is an expected source. | 5 |
| Grounded numbers | Every number in the answer appears in the cited text. | 5 |
| Valid JSON rate | Answers that validate against the schema. | 5 |
| IoU | Overlap of two boxes (Table Transformer vs `find_tables()`). | 1 |

## 8. Models, libraries and services

| Topic | Short explanation | Day |
|---|---|---|
| Qwen3-VL-4B-Instruct | Local VLM for captions and grounded answers (fp16 on T4). | 2–5 |
| OpenRouter (optional) | Hosted OpenAI-compatible API for larger vision models; `OPENROUTER_API_KEY`. | 2–5 |
| reportlab | Generates the synthetic PDFs (paragraphs, tables, images). | All |
| pytesseract | Tesseract OCR for the scanned report. | 1 |
| sentence-transformers | BGE embeddings and the cross-encoder reranker. | 2–5 |
| OpenCLIP | CLIP image/text encoders. | 2–5 |
| LangChain text splitters | Chunking utilities. | 2–5 |
| faiss-cpu / qdrant-client | Vector search library / vector database client. | 3–5 |
| Google Drive artifacts | `My Drive/cv_course_artifacts/week6_rag/` keeps corpus, elements, index and baseline across sessions. | All |
