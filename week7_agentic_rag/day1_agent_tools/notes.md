# Day 1 Notes — Agent Fundamentals & Tool Design

## Data
- The Week 6 corpus, index and question set, loaded from `My Drive/cv_course_artifacts/week6_rag/` (rebuilt if missing).

## Core ideas

**Pipeline vs agent.** A pipeline runs a fixed sequence (retrieve → answer). An agent runs a loop in which the *model* chooses the next action from a set of tools, observes the result, and decides again, until it answers.

**ReAct** (Reason + Act): each turn = a short *thought* + one *action*; the tool's output is the *observation*.
```
system + tool descriptions + question
  → {"thought": "...", "action": "retrieve_multimodal", "args": {...}}
  ← Observation: [{"id": "...", "type": "table", ...}]
  → {"thought": "...", "action": "answer_vqa", "args": {...}}
  ← Observation: "Europe, 412.3"
  → {"thought": "...", "action": "final_answer", "args": {"answer": "...", "citations": [...]}}
```

**Tool design rules:**

| Rule | Why |
|---|---|
| One job per tool | Easier for the model to choose; easier to test |
| Typed, few arguments | Fewer malformed calls; errors are explicit |
| Compact observations (ids + snippets) | The model pays (tokens, latency, attention) for every character it reads |
| Full records kept in a registry | Citations point to real items, not to what the model remembers |
| Errors returned as observations | A bad argument should lead to a better next call, not a crash |
| Every call logged (args, output, seconds) | Agents amplify upstream errors; the log is how you find them |

**Our tools** wrap Week 6: `search_metadata` (cheap document lookup), `parse_document` (structure), `retrieve_multimodal` (text + image spaces, filters, query image), `caption_image` (Day 2 captions), `answer_vqa` (the VLM reading figure pixels).

**Guardrails:** a step cap (`max_steps`) that ends in a forced answer from the evidence seen so far; schema validation of every action with one repair round.

**Native tool calling:** models trained for tools take JSON-Schema tool definitions in the chat template and emit structured calls. The JSON-in-text approach used here works with any backend, which makes it easier to learn and debug.

## What to check in your run
- The traces: did the model call the cheapest useful tool first? Did it use `answer_vqa` for figure facts?
- How many LLM calls and seconds per question, vs the Week 6 pipeline?
- The step-capped run: what does a forced answer look like?

## Pitfalls
- Small local models (4B) produce weaker plans and JSON than large hosted ones. Try `LLM_BACKEND="openrouter"` to see the difference.
- Observations that are too long push the question out of the model's attention. Truncate.
- An agent that can cite anything will cite things it never saw. Restrict citations to the registry.

## Materials
- [ReAct paper (arXiv:2210.03629)](https://arxiv.org/abs/2210.03629)
- [LangGraph](https://langchain-ai.github.io/langgraph/) · [Anthropic — Building Effective Agents](https://www.anthropic.com/engineering/building-effective-agents)
