# ---- agent.py: shared helper code (canonical copy in week7_agentic_rag/tools/shared/) ----
# Agentic multimodal RAG on top of the Week 6 pipeline (Week 7, reused by the Week 8 RAG API).
#
#   Toolbox      parse_document, search_metadata, retrieve_multimodal, caption_image, answer_vqa  (every call logged)
#   react_agent  a hand-written ReAct loop: the LLM picks one tool per turn as JSON (Day 1)
#   Agent        a LangGraph state machine (Days 2-5):
#
#       START -> route --direct--> retrieve_direct ------------------------------> synthesize -> verify -> END
#                    \--agentic--> plan -> execute <-> rewrite (self-correction)      ^   \--retry--/
#                                              \-- all steps done --------------------/
import difflib
import json
import re
import time
from typing import Literal, TypedDict

from langgraph.graph import END, START, StateGraph
from PIL import Image
from pydantic import BaseModel, Field

TYPES = {"text": ["text"], "table": ["table"], "figure": ["figure"], "image": None}


def compact(h: dict, chars: int = 220) -> dict:
    """What a tool shows the LLM about a hit (full records stay in the toolbox registry)."""
    return {"id": h["id"], "type": h["type"], "doc": h["doc_title"], "page": h["page"], "text": h["text"][:chars]}


class Toolbox:
    """The agent's tools over a loaded MultimodalRAG. Every call is appended to self.log with inputs, output and time."""

    def __init__(self, rag: "MultimodalRAG", query_image: Image.Image | None = None):
        self.rag, self.query_image = rag, query_image
        self.items = {it["id"]: it for it in rag.index.items}
        self.seen: dict[str, dict] = {}  # every item a tool returned: the agent may cite only these
        self.log: list[dict] = []

    def spec(self) -> str:
        docs = "; ".join(f"{d['doc_id']} = {d['title']}" for d in self.rag.docs)
        return (
            "TOOLS\n"
            "- search_metadata(query: str, doc_id: str|null, type: 'text'|'table'|'figure'|null): list documents whose title matches, "
            "or the elements of one document (ids, types, pages). Cheap; use it to find the right document.\n"
            "- parse_document(doc_id: str): structure of one document: pages, sections, tables and figures with their captions.\n"
            "- retrieve_multimodal(query: str, k: int=5, types: list|null, doc_ids: list|null, use_query_image: bool=false): "
            "semantic search over text chunks, tables and figure captions (+ CLIP over figure images). Returns item ids and snippets.\n"
            "- caption_image(item_id: str): detailed description of a figure item, or of the user's image if item_id is 'query_image'.\n"
            "- answer_vqa(question: str, item_ids: list[str], include_query_image: bool=false): ask the vision model a question "
            "while it looks at those figures (and the user's image). Use it to read values from charts.\n"
            f"DOCUMENTS: {docs}")

    def call(self, name: str, args: dict) -> object:
        t0 = time.perf_counter()
        fn = {"search_metadata": self.search_metadata, "parse_document": self.parse_document, "retrieve_multimodal": self.retrieve_multimodal,
              "caption_image": self.caption_image, "answer_vqa": self.answer_vqa}.get(name)
        try:
            out = fn(**args) if fn else {"error": f"unknown tool {name!r}"}
        except TypeError as exc:
            out = {"error": f"bad arguments for {name}: {exc}"}
        self.log.append({"tool": name, "args": {k: v for k, v in args.items() if k != "image"}, "seconds": round(time.perf_counter() - t0, 2),
                         "output": json.dumps(out, default=str)[:600]})
        return out

    def search_metadata(self, query: str = "", doc_id: str | None = None, type: str | None = None) -> list[dict]:
        if doc_id:
            return [{"id": it["id"], "type": it["type"], "page": it["page"], "text": it["text"][:100]}
                    for it in self.items.values() if it["doc_id"] == doc_id and (type is None or it["type"] == type)][:20]
        words = set(re.findall(r"[a-z0-9]+", query.lower()))
        scored = [(len(words & set(re.findall(r"[a-z0-9]+", (d["title"] + " " + d["doc_id"]).lower()))), d) for d in self.rag.docs]
        return [{"doc_id": d["doc_id"], "title": d["title"], "kind": d["kind"]} for s, d in sorted(scored, key=lambda x: -x[0]) if s > 0][:5] \
            or [{"doc_id": d["doc_id"], "title": d["title"]} for d in self.rag.docs]

    def parse_document(self, doc_id: str) -> dict:
        els = [e for e in self.rag.elements if e.doc_id == doc_id]
        if not els:
            return {"error": f"no document {doc_id!r}"}
        return {"doc_id": doc_id, "pages": max(e.page for e in els),
                "sections": list(dict.fromkeys(e.text for e in els if e.type == "title"))[:12],
                "tables": [{"page": e.page, "first_row": (e.rows or [[""]])[0]} for e in els if e.type == "table"],
                "figures": [{"page": e.page, "printed_caption": e.text[:120]} for e in els if e.type == "figure"]}

    def retrieve_multimodal(self, query: str, k: int = 5, types: list | None = None, doc_ids: list | None = None,
                            use_query_image: bool = False, mode: str = "hybrid") -> list[dict]:
        img = self.query_image if use_query_image else None
        hits = retrieve(self.rag.index, query, img, k=k, mode=mode, types=types, doc_ids=doc_ids)
        for h in hits:
            self.seen[h["id"]] = h
        return [compact(h) for h in hits]

    def caption_image(self, item_id: str) -> str:
        if item_id == "query_image" and self.query_image is not None:
            return self.rag.llm.ask("Describe this chart: type, title, categories and every value you can read.", [self.query_image],
                                    max_new_tokens=200)
        it = self.items.get(item_id)
        if not it or not it.get("image_path"):
            return f"{item_id} is not a figure"
        self.seen[item_id] = {**it, "score": 0.0, "space": "caption_image"}
        return it["text"][-600:]

    def answer_vqa(self, question: str, item_ids: list, include_query_image: bool = False) -> str:
        imgs, names = [], []
        if include_query_image and self.query_image is not None:
            imgs.append(self.query_image)
            names.append("image 1 is the USER'S image")
        for iid in item_ids[:3]:
            it = self.items.get(iid)
            if it and it.get("image_path"):
                imgs.append(Image.open(it["image_path"]))
                names.append(f"image {len(imgs)} is {iid} ({it['doc_title']}, page {it['page']})")
                self.seen[iid] = {**it, "score": 0.0, "space": "answer_vqa"}
        if not imgs:
            return "no figure images among those item ids"
        return self.rag.llm.ask(f"{'; '.join(names)}.\n{question}\nRead values exactly from the images. Answer in 1-3 sentences.",
                                imgs, max_new_tokens=160)


# ---------------------------------------------------------------------------------------------
# Day 1: a hand-written ReAct loop
# ---------------------------------------------------------------------------------------------
class ReActStep(BaseModel):
    thought: str = Field(description="one or two sentences of reasoning")
    action: Literal["search_metadata", "parse_document", "retrieve_multimodal", "caption_image", "answer_vqa", "final_answer"]
    args: dict = Field(default_factory=dict, description="tool arguments; for final_answer: {answer: str, citations: [item ids]}")


REACT_SYSTEM = ("You answer questions about a document collection by calling tools, one per turn (ReAct: think, act, observe). "
                "Use retrieve_multimodal to find evidence and answer_vqa to read values from figures. Cite item ids you actually saw. "
                "When you have the answer, use action final_answer with args {\"answer\": ..., \"citations\": [item ids]}. "
                "If the documents do not contain the answer, give final_answer \"I don't know based on the documents.\"")


def react_agent(question: str, rag: "MultimodalRAG", image=None, max_steps: int = 6, verbose: bool = True) -> dict:
    """Plain loop, no framework: prompt -> JSON action -> run tool -> observation -> repeat (capped at max_steps)."""
    t0, calls0 = time.perf_counter(), rag.llm.calls
    tb = Toolbox(rag, image)
    user = [{"type": "text", "text": f"{tb.spec()}\n\nQUESTION: {question}"}]
    if image is not None:
        user = [{"type": "text", "text": "The user attached this image (refer to it as 'query_image'):"}, {"type": "image", "image": image}] + user
    msgs = [{"role": "system", "content": REACT_SYSTEM}, {"role": "user", "content": user}]
    trace, final = [], None
    for step in range(1, max_steps + 1):
        obj, raw, _ = ask_json(rag.llm, msgs, ReActStep, max_new_tokens=300)
        if obj is None:
            trace.append({"step": step, "error": "invalid action JSON", "raw": raw[:200]})
            msgs += [{"role": "assistant", "content": raw}, {"role": "user", "content": "Reply with one valid JSON action."}]
            continue
        if verbose:
            print(f"  [{step}] {obj.thought[:110]}\n      -> {obj.action}({json.dumps(obj.args)[:120]})")
        if obj.action == "final_answer":
            final = obj.args
            trace.append({"step": step, "thought": obj.thought, "action": "final_answer", "args": obj.args})
            break
        args = dict(obj.args)
        if obj.action == "retrieve_multimodal":
            args.setdefault("query", question)
        obs = tb.call(obj.action, args)
        trace.append({"step": step, "thought": obj.thought, "action": obj.action, "args": obj.args, "observation": json.dumps(obs, default=str)[:400]})
        msgs += [{"role": "assistant", "content": obj.model_dump_json()},
                 {"role": "user", "content": f"Observation: {json.dumps(obs, default=str)[:1800]}\nNext action (JSON)."}]
    if final is None:  # step budget exhausted: answer from whatever was seen (a guardrail, not a crash)
        out = generate_answer(rag.llm, question, list(tb.seen.values())[:6], query_image=image)
        final = {"answer": out["answer"], "citations": [h["id"] for h in out["cited_items"]], "forced": True}
    cited = [tb.seen.get(c) or tb.items.get(c) for c in final.get("citations", []) or []]
    return {"answer": str(final.get("answer", "")), "cited_items": [c for c in cited if c], "trace": trace, "tool_log": tb.log,
            "steps": len(trace), "tool_calls": len(tb.log), "llm_calls": rag.llm.calls - calls0,
            "seconds": time.perf_counter() - t0, "forced": final.get("forced", False)}


# ---------------------------------------------------------------------------------------------
# Days 2-5: a LangGraph agent with routing, planning, self-correction and verification
# ---------------------------------------------------------------------------------------------
class RouteDecision(BaseModel):
    route: Literal["direct", "agentic"]
    reason: str = Field(max_length=300)


class PlanStep(BaseModel):
    subquestion: str
    modality: Literal["text", "table", "figure", "image"] = Field(description="where the answer lives; 'image' = match the user's image")
    document: str | None = Field(default=None, description="company or paper name if the step is about one document")


class Plan(BaseModel):
    steps: list[PlanStep] = Field(min_length=1, max_length=4)


class Grade(BaseModel):
    sufficient: bool = Field(description="does the evidence contain the answer to the sub-question?")
    answer: str | None = Field(description="the answer to the sub-question if sufficient, copied exactly")
    missing: str | None = Field(description="what is missing if not sufficient")


class Rewrite(BaseModel):
    query: str
    modality: Literal["text", "table", "figure", "image"]
    document: str | None = None


MULTI_HOP = re.compile(r"\b(compare|versus|vs\.?|higher|lower|more|fewer|both|and who|among|difference|differ|like this|same data)\b", re.I)


def heuristic_route(question: str, has_image: bool) -> str:
    return "agentic" if has_image or MULTI_HOP.search(question) else "direct"


class AgentState(TypedDict, total=False):
    question: str
    image: object
    route: str
    plan: list
    current: int
    attempts: int
    need_retry: bool
    missing: str
    evidence: list
    findings: list
    answer: str
    citations: list
    cited_items: list
    insufficient: bool
    verified: bool
    feedback: str
    synth_rounds: int
    trace: list


class Agent:
    def __init__(self, rag: "MultimodalRAG", k: int = 4, max_retries: int = 2, self_correct: bool = True, verify: bool = True,
                 llm_router: bool = True, max_tool_calls: int = 14, verbose: bool = True, force_route: str | None = None,
                 plan_override=None):
        self.rag, self.k, self.max_retries, self.self_correct, self.do_verify = rag, k, max_retries, self_correct, verify
        self.llm_router, self.max_tool_calls, self.verbose = llm_router, max_tool_calls, verbose
        self.force_route = force_route      # "direct" / "agentic": skip the router (ablations)
        self.plan_override = plan_override  # fn(steps, question) -> steps: inject a bad plan to test self-correction
        self.tb: Toolbox | None = None
        g = StateGraph(AgentState)
        for name in ("route", "retrieve_direct", "plan", "execute", "rewrite", "synthesize", "verify"):
            g.add_node(name, getattr(self, f"_{name}"))
        g.add_edge(START, "route")
        g.add_conditional_edges("route", lambda s: s["route"], {"direct": "retrieve_direct", "agentic": "plan"})
        g.add_edge("retrieve_direct", "synthesize")
        g.add_edge("plan", "execute")
        g.add_conditional_edges("execute", self._after_execute, {"rewrite": "rewrite", "execute": "execute", "synthesize": "synthesize"})
        g.add_edge("rewrite", "execute")
        g.add_edge("synthesize", "verify")
        g.add_conditional_edges("verify", lambda s: "done" if s.get("verified", True) or s.get("synth_rounds", 0) >= 2 else "retry",
                                {"done": END, "retry": "synthesize"})
        self.graph = g.compile()

    # -- helpers ------------------------------------------------------------------------------
    def _log(self, s: AgentState, node: str, **info) -> list:
        if self.verbose:
            print(f"  [{node}] " + " | ".join(f"{k}={str(v)[:110]}" for k, v in info.items()))
        return s.get("trace", []) + [{"node": node, **info}]

    def resolve_doc(self, name: str | None) -> str | None:
        if not name:
            return None
        name = name.lower()
        best, best_r = None, 0.0
        for d in self.rag.docs:
            title = d["title"].lower()
            r = 1.0 if (name in title or d["doc_id"].replace("_", " ") in name) else difflib.SequenceMatcher(None, name, title).ratio()
            if r > best_r:
                best, best_r = d["doc_id"], r
        return best if best_r >= 0.5 else None

    def _budget_left(self) -> bool:
        return len(self.tb.log) < self.max_tool_calls

    # -- nodes ----------------------------------------------------------------------------------
    def _route(self, s: AgentState) -> dict:
        has_img = s.get("image") is not None
        route, reason = heuristic_route(s["question"], has_img), "heuristic"
        if self.force_route:
            route, reason = self.force_route, "forced"
        elif self.llm_router and not has_img:
            obj, _, _ = ask_json(self.rag.llm, [{"role": "user", "content":
                "Decide how to answer this question over a collection of company annual reports and research papers.\n"
                "'direct' = one search finds the answer (a single fact about one document).\n"
                "'agentic' = needs several searches or steps (comparing documents, combining facts, finding an entity first).\n"
                f"QUESTION: {s['question']}"}], RouteDecision, max_new_tokens=120)
            if obj is not None:
                route, reason = obj.route, obj.reason
        return {"route": route, "trace": self._log(s, "route", route=route, reason=reason), "evidence": [], "findings": [],
                "synth_rounds": 0, "feedback": ""}

    def _retrieve_direct(self, s: AgentState) -> dict:
        hits = retrieve(self.rag.index, s["question"], s.get("image"), k=5)
        self.tb.log.append({"tool": "retrieve_multimodal", "args": {"query": s["question"]}, "seconds": 0, "output": str([h["id"] for h in hits])})
        return {"evidence": hits, "trace": self._log(s, "retrieve_direct", hits=[h["id"] for h in hits])}

    def _plan(self, s: AgentState) -> dict:
        img_note = "The user attached an image; use modality 'image' for the step that must match it against the documents.\n" \
            if s.get("image") is not None else ""
        docs = "; ".join(d["title"] for d in self.rag.docs)
        obj, _, _ = ask_json(self.rag.llm, [{"role": "user", "content":
            f"Break the question into 1-4 simple sub-questions, each answerable from ONE place in ONE document.\n"
            f"For each, say where the answer lives: 'text' (prose), 'table' (numbers by quarter), 'figure' (charts: regions, "
            f"segments, headcount) or 'image'. Name the document when known.\n{img_note}DOCUMENTS: {docs}\nQUESTION: {s['question']}"}],
            Plan, max_new_tokens=400)
        steps = [st.model_dump() for st in obj.steps] if obj else [{"subquestion": s["question"], "modality": "image" if s.get("image") is not None else "text", "document": None}]
        for st in steps:
            st["query"] = st["subquestion"]
        if self.plan_override is not None:
            steps = self.plan_override(steps, s["question"])
        return {"plan": steps, "current": 0, "attempts": 0, "need_retry": False,
                "trace": self._log(s, "plan", steps=[(st["subquestion"], st["modality"], st["document"]) for st in steps])}

    def _grade(self, subq: str, hits: list[dict], note: str | None) -> Grade:
        ev = "\n".join(f"[{h['id']}] ({h['type']}, {h['doc_title']}, p{h['page']}) {h['text'][:500]}" for h in hits)
        if note:
            ev += f"\n[vision model reading the figures] {note}"
        obj, _, _ = ask_json(self.rag.llm, [{"role": "user", "content":
            f"SUB-QUESTION: {subq}\nEVIDENCE:\n{ev}\nDoes the evidence contain the answer? Be strict: the right document and the right "
            "value must both be there."}], Grade, max_new_tokens=200)
        return obj or Grade(sufficient=bool(hits), answer=note, missing=None)

    def _execute(self, s: AgentState) -> dict:
        i, plan = s["current"], [dict(p) for p in s["plan"]]
        step = plan[i]
        doc = self.resolve_doc(step.get("document"))
        if step["modality"] == "image" and s.get("image") is not None:
            raw = self.tb.call("retrieve_multimodal", {"query": step["query"], "k": self.k, "use_query_image": True, "mode": "image"})
        else:
            raw = self.tb.call("retrieve_multimodal", {"query": step["query"], "k": self.k, "types": TYPES[step["modality"]],
                                                       "doc_ids": [doc] if doc else None})
        hits = [self.tb.seen[h["id"]] for h in raw if isinstance(h, dict) and h.get("id") in self.tb.seen]
        note = None
        figs = [h["id"] for h in hits if h.get("image_path")][:2]
        if figs and step["modality"] in ("figure", "image") and self._budget_left():
            note = self.tb.call("answer_vqa", {"question": step["subquestion"], "item_ids": figs,
                                               "include_query_image": step["modality"] == "image"})
        grade = self._grade(step["subquestion"], hits, note) if self.self_correct else Grade(sufficient=True, answer=note, missing=None)
        attempts = s.get("attempts", 0)
        trace = self._log(s, "execute", step=i, query=step["query"], modality=step["modality"], doc=doc,
                          hits=[h["id"] for h in hits], vqa=note, sufficient=grade.sufficient, answer=grade.answer)
        if not grade.sufficient and attempts < self.max_retries and self._budget_left():
            return {"need_retry": True, "missing": grade.missing or "", "trace": trace, "plan": plan}
        evidence = s.get("evidence", []) + [h for h in hits if h["id"] not in {e["id"] for e in s.get("evidence", [])}]
        finding = {"step": i, "subquestion": step["subquestion"], "answer": grade.answer if grade.sufficient else None,
                   "vqa": note, "sources": [h["id"] for h in hits], "attempts": attempts}
        return {"need_retry": False, "evidence": evidence, "findings": s.get("findings", []) + [finding], "current": i + 1,
                "attempts": 0, "trace": trace, "plan": plan}

    def _after_execute(self, s: AgentState) -> str:
        if s.get("need_retry"):
            return "rewrite"
        return "execute" if s["current"] < len(s["plan"]) and self._budget_left() else "synthesize"

    def _rewrite(self, s: AgentState) -> dict:
        plan = [dict(p) for p in s["plan"]]
        step = plan[s["current"]]
        obj, _, _ = ask_json(self.rag.llm, [{"role": "user", "content":
            f"A search did not find the answer.\nSUB-QUESTION: {step['subquestion']}\nLAST QUERY: {step['query']} "
            f"(modality {step['modality']}, document {step.get('document')})\nMISSING: {s.get('missing')}\n"
            "Write a better search query. Switch modality if the fact is more likely in a table (quarterly numbers) or a figure "
            "(regions, segment shares, headcount), and name the document if you can.\n"
            f"DOCUMENTS: {'; '.join(d['title'] for d in self.rag.docs)}"}], Rewrite, max_new_tokens=150)
        order = ["text", "table", "figure"]
        if obj is None or (obj.query == step["query"] and obj.modality == step["modality"]):
            nxt = order[(order.index(step["modality"]) + 1) % 3] if step["modality"] in order else "text"
            obj = Rewrite(query=step["subquestion"], modality=nxt, document=step.get("document"))  # deterministic fallback
        step.update(query=obj.query, modality=obj.modality, document=obj.document or step.get("document"))
        return {"plan": plan, "attempts": s.get("attempts", 0) + 1, "need_retry": False,
                "trace": self._log(s, "rewrite", query=obj.query, modality=obj.modality, document=step["document"])}

    def _synthesize(self, s: AgentState) -> dict:
        ev = s.get("evidence", [])[:8]
        findings = "\n".join(f"- {f['subquestion']} -> {f['answer'] or f['vqa'] or 'not found'} (from {', '.join(f['sources'][:3])})"
                             for f in s.get("findings", []))
        question = s["question"]
        if findings:
            question += f"\n\nFINDINGS FROM EARLIER STEPS (each from the listed source ids):\n{findings}"
        if s.get("feedback"):
            question += f"\n\nA CHECK FAILED ON YOUR PREVIOUS ANSWER: {s['feedback']} Fix it."
        out = generate_answer(self.rag.llm, question, ev, query_image=s.get("image"))
        return {"answer": out["answer"], "citations": out["citations"], "cited_items": out["cited_items"], "insufficient": out["insufficient"],
                "synth_rounds": s.get("synth_rounds", 0) + 1,
                "trace": self._log(s, "synthesize", answer=out["answer"], citations=out["citations"])}

    def _verify(self, s: AgentState) -> dict:
        if not self.do_verify:
            return {"verified": True}
        problems = []
        found = [f for f in s.get("findings", []) if f["answer"]]
        if is_abstention(s["answer"]) and found:
            problems.append(f"you abstained but the findings contain answers: {[f['answer'] for f in found]}")
        pool = " ".join(h["text"] for h in s.get("evidence", [])) + " " + " ".join(str(f["answer"] or "") + " " + str(f["vqa"] or "")
                                                                                  for f in s.get("findings", []))
        nums = [n for n in numbers_in(s["answer"]) if not 1900 <= n <= 2100]
        unsupported = [n for n in nums if not any(abs(n - p) <= max(0.01 * abs(n), 0.05) for p in numbers_in(pool))]
        if unsupported:
            problems.append(f"numbers {unsupported} do not appear in the evidence")
        verified = not problems
        return {"verified": verified, "feedback": "; ".join(problems),
                "trace": self._log(s, "verify", verified=verified, problems=problems)}

    # -- entry point ------------------------------------------------------------------------------
    def run(self, question: str, image=None) -> dict:
        if isinstance(image, (str, Path)):
            image = Image.open(image)
        t0, calls0 = time.perf_counter(), self.rag.llm.calls
        self.tb = Toolbox(self.rag, image)
        if self.verbose:
            print(f"Q: {question}")
        s = self.graph.invoke({"question": question, "image": image, "trace": []}, config={"recursion_limit": 40})
        retries = sum(1 for t in s.get("trace", []) if t["node"] == "rewrite")
        return {"answer": s.get("answer", ""), "cited_items": s.get("cited_items", []), "route": s.get("route"), "plan": s.get("plan", []),
                "findings": s.get("findings", []), "evidence": s.get("evidence", []), "verified": s.get("verified"),
                "retries": retries, "synth_rounds": s.get("synth_rounds", 0), "trace": s.get("trace", []), "tool_log": self.tb.log,
                "tool_calls": len(self.tb.log), "llm_calls": self.rag.llm.calls - calls0, "seconds": time.perf_counter() - t0}
