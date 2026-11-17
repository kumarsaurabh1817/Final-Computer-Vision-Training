# Day 3 Notes — Multi-Step Retrieval & Self-Correction

## Data
- **Entity-resolution questions** generated from the corpus facts: "revenue in Q2 2025 of the company headquartered in Lisbon", "largest segment at the company led by X". The question names a *property*, not the company, so a one-shot search lands on the wrong document.
- Multi-hop region + segment questions from Week 6.

## Core ideas

**The loop inside a step:**
```
retrieve(query, modality, document) → [VQA on figures] → grade
   sufficient → record finding + evidence, next step
   not sufficient & retries left → rewrite (new query / modality / document) → retrieve again
```

**Grader:** an LLM call with a strict schema `{sufficient, answer, missing}` and an instruction to be strict ("right document *and* right value"). `missing` feeds the rewriter.

**Rewriter:** reformulates the query, **switches modality** (quarterly numbers live in tables, regions/segments/headcount in figures) or names the document. If the LLM repeats the same query, a deterministic fallback cycles text → table → figure, so a retry is never wasted.

**Verifier** (rule-based, after synthesis):
- every number in the answer (years excluded) must appear in the evidence or the step findings;
- abstaining while findings contain answers is an error.

A failed check sends the answer back to `synthesize` once, with the problem stated.

**Forced failure test:** inject a bad plan (`plan_override`: all steps search prose) to prove that recovery works. Do not wait for a lucky failure.

**Runaway guardrails:** a tool-call budget checked before each step or retry, LangGraph's recursion limit, and `max_retries` per step. Each failure mode ends in an answer, never in an endless loop.

## Metrics
- Correct, evidence recall, retries, tool and LLM calls, seconds: self-correction on vs off.
- **Recovered**: first attempt graded insufficient, final answer correct.

## What to check in your run
- How many hard questions fail their first retrieval, and how many are recovered?
- Cost of self-correction: extra calls and seconds per question.
- The forced-failure trace: grader rejects, rewriter switches modality, answer becomes correct.
- The verifier catching an invented number.

## Pitfalls
- A lenient grader ("looks relevant") never triggers a retry. A paranoid one retries forever, so cap it.
- Graders inherit retrieval errors: if the right document is never retrieved, the only fix is a better query or a document filter. That is why the rewriter can name a document.
- Self-correction cannot fix a wrong parse or a wrong caption. Those errors come from Week 6 and get amplified here.

## Materials
- [LlamaIndex — agents](https://docs.llamaindex.ai/en/stable/use_cases/agents/) · [Self-RAG](https://arxiv.org/abs/2310.11511) · [Corrective RAG (CRAG)](https://arxiv.org/abs/2401.15884)
