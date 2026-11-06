# Day 5 Notes — Grounded Generation & the Full Pipeline

## Data
- The full question set except multi-hop (Week 7): text, table, figure, text+image and unanswerable questions.

## Core ideas

**Grounded prompt = sources + rules + schema.**
```
[S1] (table, Northwind Energy Annual Report 2025, page 2)  | Quarter | Revenue | ...
[S2] (figure, ..., page 2) [shown as attached image #1]    Figure 1: Revenue by region ... <caption>
...
Rules: only these sources · cite [S#] · copy numbers exactly · if absent → insufficient_context=true, "I don't know"
Output: {"answer": ..., "citations": ["S1"], "insufficient_context": false}
```
Citations map back to items, and items carry doc, page and bbox. The user can check every claim.

**Images in the context.** For figures, the VLM gets the *actual* crop, not just the caption. It can read a value the caption missed. The images-in-context ablation measures what that is worth.

**Abstention is a feature.** Unanswerable questions (a year not in the report, a CFO never mentioned) must produce "I don't know". A RAG system that always answers is a confident hallucination machine.

**Error decomposition:**
- *Retrieval miss*: the expected source was not in the top k. Fix parsing, representation, index or retrieval (Days 1–4).
- *Generation miss*: the source was retrieved but the answer is wrong. Fix the prompt, the model, image resolution or context size.

**The baseline.** `MultimodalRAG.ask(question, image=None)` is one retrieve → one generate. No planning, no retries. Week 7's agent has to beat it on the same questions, especially multi-hop ones.

## Metrics
- **correct**: numbers within 1 %, names as substrings, abstention for unanswerable questions.
- **citation_ok**: a cited item is an expected source.
- **grounded_numbers**: every number in the answer appears in the cited text (not checkable for figure-only values).
- **retrieval_recall**: the upper bound for the generator.
- **valid_json**: schema compliance after one repair.
- **Closed-book baseline**: the same model with no sources.

## What to check in your run
- Correctness per question type, and where it trails retrieval recall (generation misses).
- Abstention rate on unanswerable questions, and false abstentions on answerable ones.
- Closed-book: how often does the model invent a number instead of saying "I don't know"?
- Captions only vs captions + images on figure questions.

## Pitfalls
- Citing everything ("[S1][S2][S3][S4][S5]") passes a naive citation check. The metric only asks for one correct citation, so read some answers.
- Numbers formatted differently (`2,295.4` vs `2295.40`) need number-aware matching, not string equality.
- More images = more visual tokens = slower and sometimes worse (Week 5 Day 1). The default attaches at most 3.

## Materials
- [LlamaIndex — multimodal RAG](https://docs.llamaindex.ai/en/stable/use_cases/multimodal/)
- Week 5 Day 3 (schema-valid JSON), Week 5 Day 2 (validating numbers)
