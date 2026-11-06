# Day 3 Notes — Multimodal Indexing

## Data
- Captioned elements from Day 2 → items (section-aware text chunks, tables with context, figures with printed + VLM captions).
- 100k random 384-d vectors for the exact-vs-approximate experiment.

## Core ideas

**Two spaces, one metadata store.**
```
text space  (BGE, 384-d)  ← every item's text: chunks, table markdown, figure captions
image space (CLIP, 512-d) ← every figure crop
metadata    (JSON)        ← id, doc_id, doc_title, type, page(s), bbox, text, image_path, element_ids
```
A vector row points to its metadata record, so any hit, from either space, carries provenance.

**Why not one space?** BGE is a much better *text* retriever than CLIP's text encoder, and CLIP is the only one that sees pixels. Each modality gets its best model. Day 4 fuses the ranked lists.

**FAISS vs a vector database:**

| | FAISS (library) | Qdrant (database) |
|---|---|---|
| Storage | in-process arrays, write/read files | collections with payloads, persistent |
| Filters | after search (over-fetch, then filter) | inside the search (`query_filter` on payload fields) |
| Updates | rebuild or `add` only | upsert / delete by id |
| Deployment | embed in your app | in-process, server or cloud, same API |

**Exact vs approximate.** `IndexFlatIP` compares the query with every vector. That is microseconds at 1k vectors and too slow at 100M. HNSW builds a navigable small-world graph: `M` sets the links per node (memory, recall) and `efSearch` the candidates explored per query (latency, recall).

**Provenance is a feature, not a log line.** Drawing the cited bbox on the page lets the user verify an answer in one glance. That is the cheapest hallucination defence there is.

## What to check in your run
- Index sizes: how many text vs image vectors, and MB on disk?
- Reload gives identical results (same models, same files).
- FAISS vs Qdrant parity (top-1 agreement, Jaccard@5).
- Filtered search: tables only, one document only. Note how different the top hits are.
- HNSW recall@10 and latency vs exact, at your scale.

## Pitfalls
- Never mix embedding models in one index. The saved index records its model names and refuses a mismatch on load.
- Normalize vectors before inner-product search (cosine), or scores are meaningless.
- Post-filtering with FAISS can return fewer than k hits if the filter is strict. Over-fetch enough (the code fetches 5k).

## Materials
- [FAISS wiki](https://github.com/facebookresearch/faiss/wiki) · [Guidelines to choose an index](https://github.com/facebookresearch/faiss/wiki/Guidelines-to-choose-an-index)
- [Qdrant — filtering](https://qdrant.tech/documentation/concepts/filtering/) · [HNSW paper](https://arxiv.org/abs/1603.09320)
