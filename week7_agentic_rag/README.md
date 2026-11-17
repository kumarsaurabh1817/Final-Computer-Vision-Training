# Week 7 — Agentic Multimodal RAG

| Attribute | Detail |
|-----------|--------|
| **Pillar** | Vision + Generative AI |
| **Goal** | Turn Week 6's static pipeline into an agent that plans, chooses tools (parse, retrieve, caption, VQA, re-retrieve), reasons over several steps and corrects itself, for complex text and text+image queries. |
| **Effort** | 32–40 hours (full-time) |
| **Runtime** | Google Colab, T4 GPU. Each notebook is self-contained and loads the Week 6 index from Google Drive (or rebuilds it). |

A list of every topic used this week, with a short explanation, is in [`week7notes.md`](week7notes.md).

---

## Data — nothing to download

Week 7 uses the Week 6 corpus, index and question set from **`My Drive/cv_course_artifacts/week6_rag/`**. If Week 6 has not been run, the first notebook builds everything (~10 min). New questions are generated from the corpus facts: entity-resolution questions (Day 3) and query images with known or altered chart data (Day 4).

Agent outputs (eval report, per-case CSV) are also saved to **`My Drive/cv_course_artifacts/week7_agent/`**.

## How to run on Colab

1. **File → Upload notebook** → `dayN_*/Week7_DayN_Colab.ipynb`.
2. **Runtime → Change runtime type → T4 GPU**.
3. **Runtime → Run all**, and click **Connect to Google Drive** when asked.
4. Optional but recommended: agents plan much better with a larger model. Add `OPENROUTER_API_KEY` in Colab **Secrets** and set `LLM_BACKEND = "openrouter"` (any vision model id on openrouter.ai in `LLM(..., model_id=...)`).

---

## Day-by-day

### Day 1 — Agent fundamentals & tool design
Week 6 capabilities as five logged tools (`search_metadata`, `parse_document`, `retrieve_multimodal`, `caption_image`, `answer_vqa`); a hand-written ReAct loop with schema-validated actions; step cap with a forced answer; comparison with the baseline; what native tool calling looks like in the chat template.
**Files:** [`Week7_Day1_Colab.ipynb`](day1_agent_tools/Week7_Day1_Colab.ipynb) · [`day1_agent_tools.py`](day1_agent_tools/day1_agent_tools.py) · [`notes.md`](day1_agent_tools/notes.md) · ~10–15 min

### Day 2 — Query planning & routing
Heuristic vs LLM router scored against question types; planner decomposition with modality and document per step (plan coverage); the LangGraph agent and its graph; multi-hop traces; direct vs agentic cost on simple questions ("when not to add an agent").
**Files:** [`Week7_Day2_Colab.ipynb`](day2_planning_routing/Week7_Day2_Colab.ipynb) · [`day2_planning_routing.py`](day2_planning_routing/day2_planning_routing.py) · [`notes.md`](day2_planning_routing/notes.md) · ~20 min

### Day 3 — Multi-step retrieval & self-correction
Entity-resolution questions; grader → rewriter (new query / modality / document) → re-retrieve; self-correction on vs off; a forced bad plan and its recovery; the answer verifier; tool-budget and recursion guardrails.
**Files:** [`Week7_Day3_Colab.ipynb`](day3_self_correction/Week7_Day3_Colab.ipynb) · [`day3_self_correction.py`](day3_self_correction/day3_self_correction.py) · [`notes.md`](day3_self_correction/notes.md) · ~25–30 min

### Day 4 — Text+image agentic queries
Three families with automatic scoring: identify (image → document → fact), compare (altered chart → describe differences with multi-image VQA), cross-reference (image → figure → table). Baseline vs agent; image-only vs composed retrieval.
**Files:** [`Week7_Day4_Colab.ipynb`](day4_text_image_agent/Week7_Day4_Colab.ipynb) · [`day4_text_image_agent.py`](day4_text_image_agent/day4_text_image_agent.py) · [`notes.md`](day4_text_image_agent/notes.md) · ~20 min

### Day 5 — Evaluation harness
Fixed suite (text-only, text+image, multi-hop, unanswerable); agent vs Week 6 baseline on correctness, evidence recall, citations, LLM-judge faithfulness, tool/LLM calls and latency; wins, losses, self-corrections; the success gate; `week7_eval_report.md`.
**Files:** [`Week7_Day5_Colab.ipynb`](day5_evaluation/Week7_Day5_Colab.ipynb) · [`day5_evaluation.py`](day5_evaluation/day5_evaluation.py) · [`notes.md`](day5_evaluation/notes.md) · ~45–60 min

### Weekend
Harden using the losses list, add guardrails where the traces show waste, record a demo, and document the architecture (graph from Day 2, tools from Day 1, guardrails from Day 3).

---

## Architecture

```
                         ┌──────────────── Toolbox (logged) ────────────────┐
question (+ image) ─► route ─direct─► retrieve ───────────────────────► synthesize ─► verify ─► answer + citations
                         │                                                  ▲            │
                         └agentic─► plan ─► execute ⇄ rewrite ──────────────┘   ◄─retry──┘
                                       retrieve_multimodal · answer_vqa · grade
```

| Guardrail | Where |
|---|---|
| Step cap (ReAct) | `react_agent(max_steps=6)` → forced answer |
| Retries per step | `Agent(max_retries=2)` |
| Tool-call budget | `Agent(max_tool_calls=14)` |
| Node-visit limit | LangGraph `recursion_limit=40` |
| Answer checks | verifier: numbers in evidence, no false abstention |
| Citations | only items a tool returned can be cited |

## Deliverable & success gate

| Requirement | Where |
|---|---|
| Plans and executes multi-step retrieval | Days 2–3 |
| Text-only and text+image queries | Days 2, 4 |
| Self-corrects | Day 3 (and the self-corrected cases in Day 5) |
| Grounded, cited answers | every day (`cited_items`) |
| Eval suite with the Week 6 baseline | Day 5 |
| **Gate:** majority correct per category + at least one self-correction | Day 5 → `GATE PASSED` in `week7_eval_report.md` |

> Agents amplify upstream errors: a bad parse or wrong caption becomes a confident wrong answer. Every tool call's I/O is logged (`tool_log` in each result, `day5_agent_traces.jsonl`).

## Folder structure

```
week7_agentic_rag/
├── README.md
├── week7notes.md
├── requirements.txt
├── tools/
│   ├── shared/agent.py, agent_eval.py   ← Toolbox, ReAct loop, LangGraph Agent, eval helpers
│   ├── inline_shared.py                 ← copies Week 6 + Week 7 shared modules into each day
│   └── py_to_ipynb.py
├── day1_agent_tools/ … day5_evaluation/ ← Week7_DayN_Colab.ipynb, .py, notes.md
├── models/
└── results/
```
