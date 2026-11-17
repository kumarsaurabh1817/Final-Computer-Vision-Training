# ---- agent_eval.py: shared helper code (canonical copy in week7_agentic_rag/tools/shared/) ----
# Evaluation helpers for Week 7: harder entity-resolution questions, a common score row for the baseline and the
# agent, and an LLM-as-judge faithfulness check (the idea behind Ragas' "faithfulness").
import json
import random

from pydantic import BaseModel, Field


def load_facts(work: Path) -> tuple[dict, dict]:
    corpus = json.loads((work / "corpus.json").read_text(encoding="utf-8"))
    return corpus["facts"], corpus["pages"]


def entity_questions(work: Path, n_docs: int = 4, seed: int = 1) -> list[dict]:
    """Questions that name a property instead of the company: the agent must first find WHICH document, then the fact."""
    facts, pages = load_facts(work)
    rng = random.Random(seed)
    out = []
    for d, f in rng.sample(sorted(facts.items()), min(n_docs, len(facts))):
        q, pg = rng.randrange(4), pages[d]
        out.append(_qa(f"ent-hq-{d}", f"What was the revenue in {QUARTERS[q]} 2025 of the company headquartered in {f['hq']}?",
                       f"{f['revenue'][q]:,.1f}", "multi-hop", [(d, pg["overview"], "text"), (d, pg["table1"], "table")], f["revenue"][q]))
        seg = max(f["segments"], key=f["segments"].get)
        out.append(_qa(f"ent-ceo-{d}", f"Which business segment had the largest revenue share at the company led by {f['ceo']}?", seg,
                       "multi-hop", [(d, pg["overview"], "text"), (d, pg["fig2"], "figure")]))
    return out


class Judgement(BaseModel):
    supported: bool = Field(description="is every claim in the answer supported by the cited sources?")
    reason: str = Field(max_length=300)


def judge_faithful(llm: "LLM", question: str, answer: str, cited: list[dict]) -> bool | None:
    """LLM-as-judge: is the answer supported by what it cites? None for abstentions (nothing to check)."""
    if is_abstention(answer):
        return None
    if not cited:
        return False
    src = "\n".join(f"- ({h['type']}, {h['doc_title']}, p{h['page']}) {h['text'][:700]}" for h in cited[:5])
    imgs = [Image.open(h["image_path"]) for h in cited if h.get("image_path")][:2]
    obj, _, _ = ask_json(llm, [{"role": "user", "content": [{"type": "image", "image": im} for im in imgs] + [{"type": "text", "text":
        f"QUESTION: {question}\nANSWER: {answer}\nCITED SOURCES (figures attached as images):\n{src}\n"
        "Judge strictly: is every factual claim in the answer supported by the cited sources?"}]}], Judgement, max_new_tokens=150)
    return None if obj is None else obj.supported


def score_case(q: dict, out: dict, system: str, judge: "LLM | None" = None) -> dict:
    evidence = out.get("evidence") or out.get("hits") or []
    return {"system": system, "qid": q["id"], "qtype": q["qtype"], "answer": out["answer"][:300], "truth": q["answer"],
            "correct": answer_correct(out["answer"], q), "citation_ok": citation_ok(out.get("cited_items", []), q),
            "evidence_recall": source_recall(evidence, q),
            "faithful": judge_faithful(judge, q["question"], out["answer"], out.get("cited_items", [])) if judge else None,
            "tool_calls": out.get("tool_calls", 1), "llm_calls": out.get("llm_calls", 1), "retries": out.get("retries", 0),
            "seconds": out.get("seconds", out.get("total_latency_s", 0.0)), "route": out.get("route", "baseline")}
