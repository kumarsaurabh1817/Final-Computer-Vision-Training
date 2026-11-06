# Day 2 Notes — Representing Visual Content & Choosing Chunks

## Data
- Week 6 corpus and Day 1 elements, loaded from `My Drive/cv_course_artifacts/week6_rag/elements.jsonl` or parsed again.
- Truth for every synthetic chart: region values, segment shares, headcount by year.

## Core ideas

**Two representations per figure, because they fail differently:**

| | Caption as text (VLM) | Raw embedding (CLIP) |
|---|---|---|
| Captures | labels, numbers, titles, *whose* chart it is (from context) | visual shape: chart type, colours, layout |
| Searchable by | any text retriever (BGE, BM25), keyword filters | CLIP text encoder or a query image |
| Fails on | anything the caption omitted; invented numbers | reading text; distinguishing two pie charts |
| Cost | one VLM call per figure at index time | one cheap encoder pass |

**Chart-reading captions.** A generic caption ("a pie chart with four segments") is useless for "what share came from Cloud?". The prompt asks for chart type, title, categories, every value, and the takeaway. It also gives the VLM the document title and printed caption as context.

**Fact recall and invented numbers.** Score captions against chart truth: how many labels/values appear, and how many numbers appear that are *not* in the chart. Invented numbers become confident wrong answers on Day 5.

**Chunking text:**

| Strategy | Idea | Risk |
|---|---|---|
| element | one paragraph = one chunk | missing context ("it", "the company") |
| fixed N chars | sliding window over the document | splits mid-sentence, crosses sections |
| section-aware (default) | split each section separately with `RecursiveCharacterTextSplitter`, prefix "Title > Section" | none of the above, slightly more chunks |

**Tables as text:** markdown keeps the grid. Linearized rows ("Q3: Revenue 412.3; Operating profit 55.1") read like sentences. Markdown + context adds the document title, section and table caption. Context helps the embedder match "X's revenue".

## Metrics
- Caption **fact recall**, **invented numbers**, words, seconds per figure.
- CLIP similarity of same-type vs different-type charts.
- **R@1 / R@3** of the expected source for text questions (chunking) and table questions (table representation), and for figure questions (caption vs CLIP).

## What to check in your run
- Generic vs chart-reading captions: fact recall gap, and invented numbers. Scanned figures are blurrier; are they worse?
- The CLIP similarity matrix: are blocks by chart type, not by company?
- Figure questions: caption-text R@1 vs CLIP R@1. Which one would you use alone? Why not alone?
- Which chunk size wins, and how many chunks does it create?

## Pitfalls
- Captions are only as good as the crop. A figure cut in half by the parser gets half a caption.
- Long captions are truncated by CLIP's 77-token text encoder, but not by BGE. Another reason to index captions as text.
- BGE wants its query instruction prefix on queries, not on passages.

## Materials
- [LangChain — text splitters](https://python.langchain.com/docs/concepts/text_splitters/)
- [OpenCLIP](https://github.com/mlfoundations/open_clip) · [BGE embeddings](https://huggingface.co/BAAI/bge-small-en-v1.5)
- Week 5 Days 4–5 (captioning, CLIP, caption vs embedding)
