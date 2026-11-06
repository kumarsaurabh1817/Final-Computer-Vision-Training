# Day 4 Notes — Multimodal Retrieval

## Data
- The Week 6 index and question set: text, table, figure and text+image questions with expected sources `(doc_id, page, type)`.
- Text+image questions use a **restyled** copy of a report chart (new colours, size and title) as the query image.

## Core ideas

**Retrieval paths:**

| Query | Path |
|---|---|
| text only | BGE(query) → text space (chunks, tables, captions) **+** CLIP-text(query) → image space |
| text + image | the above **+** CLIP-image(query image) [+ α·CLIP-text(query)] → image space |
| image only | CLIP-image → image space; then page-neighbour expansion → text and tables on the same page |

**Reciprocal-rank fusion (RRF):** score(item) = Σ 1 / (60 + rank in each list). It ignores raw scores, which live on different scales in different spaces, and rewards items found by several paths.

**Cross-encoder reranking:** the bi-encoder (BGE) embeds query and passage separately, which is fast and approximate. A cross-encoder reads them *together* and scores relevance much more sharply. It is too slow for the whole corpus, so use it on the top ~15.

**Cross-modal retrieval** means the query and the answer are in different modalities. Text→figure works through captions (BGE) and through CLIP. Image→text works through the matched figure's metadata (doc, page) plus its neighbours.

**The question text does not have to carry the answer key.** "Which report has a chart like this?" names nothing, and only the image can find it. The company name in the answer comes from the matched item's *metadata*, not from the pixels.

## Metrics
- **Source recall@k**: fraction of expected sources in the top k (multi-source questions need several).
- **hit@1**: the top result is an expected source.
- **MRR**: mean of 1 / rank of the first expected source.
- Latency per strategy (ms), recall vs k.

## What to check in your run
- The recall-by-type matrix: text-only fails figure-only facts, CLIP-only fails text facts, hybrid covers both. Does reranking help or hurt?
- Text+image: image→image vs composed (image + text) vs text alone.
- Do figure questions put a *figure* at rank 1?
- The misses list: parse error, caption error, or ranking error?

## Pitfalls
- A "hit" on the right page but the wrong element type (the paragraph that *mentions* Figure 1) is not the evidence. The metric checks type too.
- Reranking text-less items: the cross-encoder scores the caption text of figures. If the caption is poor, reranking can push the right figure down.
- Bigger k raises recall but also the generator's distraction. Choose k with Day 5 in mind.

## Materials
- [RRF paper](https://plg.uwaterloo.ca/~gvcormac/cormacksigir09-rrf.pdf) · [Sentence-Transformers cross-encoders](https://www.sbert.net/docs/cross_encoder/usage/usage.html)
- [LangChain — multimodal inputs](https://python.langchain.com/docs/how_to/multimodal_inputs/)
