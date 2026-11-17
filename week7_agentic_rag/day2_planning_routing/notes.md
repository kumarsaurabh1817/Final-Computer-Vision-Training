# Day 2 Notes — Query Planning & Routing

## Core ideas

**Routing: does this question need an agent?** Most questions are single facts that one retrieval answers. Sending them through planning, grading and verification multiplies cost for no gain. The router picks:
- **direct**: retrieve → synthesize → verify (≈ the Week 6 pipeline);
- **agentic**: plan → execute steps (with self-correction) → synthesize → verify.

Two routers: a **heuristic** (comparison words, attached image) that is free and predictable, and an **LLM router** that is more flexible and costs one call. Measure both against labelled question types.

**Planning = decomposition + modality routing.** "Which company had higher Q4 revenue, A or B?" becomes:
1. A's Q4 revenue → `table` @ A
2. B's Q4 revenue → `table` @ B

Each step names *where the answer lives* (text, table, figure, the user's image) and *which document*. That turns into a filtered search (`types`, `doc_ids`). **Plan coverage** = how many of the expected sources the plan targets.

**LangGraph** models the agent as a state machine:
```
START → route ─direct──► retrieve_direct ───────────────► synthesize → verify → END
             └agentic─► plan → execute ⇄ rewrite                ▲          │
                                 └── all steps done ────────────┘   ◄─retry┘
```
- **State**: a typed dict (question, image, plan, current step, evidence, findings, answer, trace).
- **Nodes**: functions that return state updates.
- **Conditional edges**: Python functions that choose the next node.
- **Recursion limit**: caps node visits, a guardrail against loops.

**When NOT to add agentic complexity** ([Building Effective Agents](https://www.anthropic.com/engineering/building-effective-agents)): start with the simplest system that works, and add steps only where measurements show they help. The direct-vs-agentic comparison on simple questions is that measurement.

## What to check in your run
- Router accuracy and its confusion matrix. Which question types does the heuristic get wrong?
- Plans for multi-hop questions: right number of steps, right modalities, documents resolved?
- Multi-hop traces: one step per document, evidence collected, findings used in the final answer.
- Simple questions: does the agentic path improve accuracy enough to justify its calls?

## Pitfalls
- Planners invent document names. Resolve them fuzzily against the real titles and drop unresolved ones.
- Over-decomposition: four steps for a one-fact question. Cap the steps (1–4) and route simple questions direct.
- Mutating state in place inside a node: return new objects so traces stay correct.

## Materials
- [LangGraph — concepts](https://langchain-ai.github.io/langgraph/concepts/low_level/)
- [Anthropic — Building Effective Agents](https://www.anthropic.com/engineering/building-effective-agents) (routing, orchestrator-workers, evaluator-optimizer patterns)
