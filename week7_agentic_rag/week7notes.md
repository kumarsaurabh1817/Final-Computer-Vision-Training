# Week 7 — All Topics at a Glance

Every concept, tool, model and metric used in Week 7 (Agentic Multimodal RAG), grouped by theme, each with a short explanation. The **Day** column says where it is used.

---

## 1. Agent fundamentals

| Topic | Short explanation | Day |
|---|---|---|
| Agent | A loop in which a model chooses actions (tool calls), observes results and decides again until it answers. | All |
| Pipeline vs agent | Fixed sequence of steps vs model-chosen steps. | 1, 2 |
| Tool | A function the agent can call, with a name, typed arguments and a description. | 1–5 |
| Tool description / spec | The text the model reads to decide which tool to use. | 1 |
| Observation | A tool's output fed back to the model (kept compact). | 1 |
| ReAct | Reason + Act: thought → action → observation → … → final answer. | 1 |
| JSON action protocol | The model replies `{thought, action, args}`; validated with pydantic. | 1 |
| Native tool calling | Tool JSON schemas in the chat template; the model emits structured `<tool_call>` blocks. | 1 |
| Tool registry | Full records of every item a tool returned; citations must point into it. | 1–5 |
| Tool-call log | Inputs, output and duration of every call, for debugging and evaluation. | 1–5 |

## 2. The tools

| Tool | Short explanation | Day |
|---|---|---|
| `search_metadata` | Find documents by title words, or list one document's elements. | 1 |
| `parse_document` | Structure of a document: sections, tables, figures with captions. | 1 |
| `retrieve_multimodal` | Week 6 hybrid retrieval with type/document filters and an optional query image. | 1–5 |
| `caption_image` | Description of a figure item or of the user's image. | 1 |
| `answer_vqa` | Ask the VLM a question while it looks at figure crops (and the user's image). | 1–5 |

## 3. Planning and routing

| Topic | Short explanation | Day |
|---|---|---|
| Router | Decides direct (one retrieval) vs agentic (multi-step) per question. | 2–5 |
| Heuristic router | Regex on comparison words + "image attached"; free and predictable. | 2 |
| LLM router | The model classifies the question (`RouteDecision` schema). | 2 |
| Query decomposition | Splitting a question into 1–4 sub-questions answerable from one place. | 2 |
| Modality routing | Each sub-question says where the answer lives: text, table, figure, image. | 2–4 |
| Document resolution | Fuzzy-matching a named company/paper to a `doc_id` for filtering. | 2, 3 |
| Plan coverage | Fraction of expected sources targeted by the plan's steps. | 2 |
| When not to use an agent | Simple questions: the agentic path adds cost without accuracy. | 2 |

## 4. LangGraph

| Topic | Short explanation | Day |
|---|---|---|
| StateGraph | Graph whose nodes read and update a shared typed state. | 2–5 |
| State (TypedDict) | question, image, plan, current step, evidence, findings, answer, trace. | 2–5 |
| Node | A function returning state updates (route, plan, execute, rewrite, synthesize, verify). | 2–5 |
| Conditional edge | A function choosing the next node from the state. | 2–5 |
| Recursion limit | Maximum node visits per run; a loop guardrail. | 2, 3 |
| Mermaid graph | Text diagram of the compiled graph. | 2 |

## 5. Self-correction and verification

| Topic | Short explanation | Day |
|---|---|---|
| Iterative retrieval | Retrieve → judge → retrieve again with a better query. | 3 |
| Grader | LLM check that the evidence answers the sub-question (`sufficient`, `answer`, `missing`). | 3–5 |
| Query rewriting | New query, modality or document after a failed step. | 3 |
| Modality switching | Text → table → figure when the fact is not in prose. | 3 |
| Deterministic fallback | Rule-based rewrite when the LLM repeats itself. | 3 |
| Verifier | Numbers in the answer must be in the evidence; no abstaining when findings have answers. | 3–5 |
| Forced failure test | Injecting a bad plan (`plan_override`) to prove recovery works. | 3 |
| Entity-resolution question | Names a property (HQ city, CEO) instead of the company: needs an extra hop. | 3, 5 |
| Self-RAG / CRAG | Research versions of retrieval with self-reflection / corrective retrieval. | 3 |

## 6. Text + image agentic queries

| Topic | Short explanation | Day |
|---|---|---|
| Image-led retrieval | Query image → CLIP → similar figures → their documents. | 4 |
| Multi-image VQA | User's image and a retrieved figure in one prompt, each named. | 4 |
| Identify / compare / cross-reference | Image → document fact; image vs figure differences; image → figure → table. | 4 |
| Altered query image | Chart rendered from changed data, so the true difference is known. | 4 |
| Image-only vs composed query | CLIP image vector alone vs image + α·text. | 4 |

## 7. Guardrails

| Topic | Short explanation | Day |
|---|---|---|
| Step cap | Maximum ReAct turns; then a forced answer from the evidence seen. | 1 |
| Retry cap | Maximum rewrites per step. | 3 |
| Tool-call budget | Maximum tool calls per question. | 3, 5 |
| Citation restriction | Only registry items can be cited. | 1–5 |
| Logging every tool call | Agents amplify upstream errors; logs show where. | All |

## 8. Evaluation

| Topic | Short explanation | Day |
|---|---|---|
| Evaluation suite | Fixed, seeded, saved set of cases with expected answers and sources. | 5 |
| Baseline comparison | The Week 6 pipeline on the same suite. | 1, 2, 4, 5 |
| Correctness | Numbers ±1 %, names, all parts, abstention. | All |
| Evidence recall | Expected sources among everything retrieved. | 3–5 |
| Citation accuracy | A cited item is an expected source. | 5 |
| Faithfulness (LLM-as-judge) | Are the answer's claims supported by the cited sources? | 5 |
| Ragas | Library of RAG metrics (faithfulness, context precision/recall, answer correctness). | 5 |
| Tool-use efficiency | Tool calls, LLM calls, seconds, retries per question. | 1–5 |
| Wins / losses / self-corrections | Case-level comparison that drives hardening. | 5 |
| Success gate | Majority correct per category + at least one self-correction. | 5 |

## 9. Models and libraries

| Topic | Short explanation | Day |
|---|---|---|
| Qwen3-VL-4B-Instruct | Local VLM: router, planner, grader, VQA, synthesis. | All |
| OpenRouter | Hosted larger models via `OPENROUTER_API_KEY`; better planning. | All |
| LangGraph | Stateful agent orchestration library. | 2–5 |
| pydantic | Schemas for actions, routes, plans, grades, rewrites, judgements. | All |
| Week 6 pipeline | Corpus, parser, index, retrieval and grounded generation reused as tools. | All |
