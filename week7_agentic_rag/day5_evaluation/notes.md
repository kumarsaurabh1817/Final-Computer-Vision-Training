# Day 5 Notes — Evaluation Harness

## Data
- A fixed, saved suite: a seeded sample per Week 6 question type (text, table, figure, text+image, unanswerable), every multi-hop question, and the entity-resolution questions from Day 3. Each case has expected answers and expected sources.

## Core ideas

**Evaluate systems, not demos.** The same suite runs through the Week 6 baseline and the agent, with the same metrics. A change is an improvement only if the suite says so.

**Metric families** (the same idea as [Ragas](https://docs.ragas.io/)):

| Family | Metric here | Ragas analogue |
|---|---|---|
| Retrieval | evidence recall (expected sources found), precision | context recall / precision |
| Answer | correct (numbers ±1 %, names, all parts, abstention) | answer correctness |
| Grounding | citation_ok, faithful (LLM judge: claims supported by cited sources) | faithfulness |
| Efficiency | tool calls, LLM calls, seconds, retries | — |

**LLM-as-judge:** a second LLM call checks whether the answer's claims are supported by what it cites. It is cheap and scalable, but biased (it sometimes agrees with itself). Spot-check its verdicts by hand.

**Wins / losses / self-corrections:** read the cases where the systems disagree. Losses become the hardening to-do list. A self-corrected case (first retrieval failed, final answer correct) is part of the success gate.

**Guardrail statistics:** maximum tool calls and seconds per question, budget hits, route distribution. A runaway loop shows up here before it shows up on an invoice.

## Success gate
Majority correct in each category (text-only, text+image, multi-hop), sources cited, and at least one self-correction. The notebook prints `PASSED` / `NOT PASSED` and writes `week7_eval_report.md`.

## What to check in your run
- Where does the agent beat the baseline (multi-hop, cross-reference) and where does it lose (simple facts it over-thinks)?
- Cost ratio: how many times slower is the agent, and is it worth it per category?
- Faithfulness vs correctness: correct answers citing the wrong source, and faithful answers that are still wrong.

## Pitfalls
- A suite that changes between runs makes numbers incomparable. It is seeded and saved to `eval_suite.json`.
- Small suites have large error bars: 6 cases per type means one case = 17 %. Report counts, not just rates.
- The judge uses the same model as the agent here. Use a stronger judge (OpenRouter) for a fairer verdict.

## Materials
- [Ragas — RAG evaluation](https://docs.ragas.io/) · [LLM-as-a-judge](https://arxiv.org/abs/2306.05685)
