# ---- generate.py: shared helper code (canonical copy in week6_multimodal_rag/tools/shared/) ----
# Grounded generation (Week 6 Day 5): retrieved items -> numbered sources [S1..Sn] (text + actual figure images)
# -> VLM answer in a JSON schema with citations -> citations mapped back to item ids, docs, pages.
import time

from PIL import Image
from pydantic import BaseModel, Field


class GroundedAnswer(BaseModel):
    answer: str = Field(description="the answer, using only the sources; numbers copied exactly")
    citations: list[str] = Field(description="source labels used, e.g. ['S1', 'S3']")
    insufficient_context: bool = Field(description="true if the sources do not contain the answer")


ANSWER_RULES = ("Answer the question using ONLY the numbered sources below. Cite every source you use as [S#]. "
                "Copy numbers exactly as written in the sources. Figures are attached as images; read values from them. "
                "If the sources do not contain the answer, set insufficient_context to true and answer "
                "\"I don't know based on the documents.\"")


def build_context(hits: list[dict], max_images: int = 3, max_chars: int = 1200) -> tuple[str, list, dict]:
    """Numbered source block, the images to attach, and label -> hit."""
    lines, images, labels = [], [], {}
    for i, h in enumerate(hits, 1):
        label = f"S{i}"
        labels[label] = h
        note = ""
        if h.get("image_path") and len(images) < max_images:
            images.append(Image.open(h["image_path"]))
            note = f" [shown as attached image #{len(images)}]"
        lines.append(f"[{label}] ({h['type']}, {h['doc_title']}, page {h['page']}){note}\n{h['text'][:max_chars]}")
    return "\n\n".join(lines), images, labels


def generate_answer(llm: "LLM", question: str, hits: list[dict], query_image=None, max_images: int = 3,
                    use_images: bool = True, max_new_tokens: int = 300) -> dict:
    """One grounded answer. Returns answer, citations (labels and item ids), validity, latency."""
    t0 = time.perf_counter()
    context, images, labels = build_context(hits, max_images if use_images else 0)
    content = []
    if query_image is not None:
        content += [{"type": "text", "text": "The user's query image:"}, {"type": "image", "image": query_image}]
    content += [{"type": "image", "image": im} for im in images]
    content.append({"type": "text", "text": f"{ANSWER_RULES}\n\nSOURCES:\n{context}\n\nQUESTION: {question}"})
    obj, raw, attempts = ask_json(llm, [{"role": "user", "content": content}], GroundedAnswer, max_new_tokens)
    if obj is None:
        return {"answer": raw, "citations": [], "cited_items": [], "insufficient": False, "valid": False,
                "attempts": attempts, "latency_s": time.perf_counter() - t0, "sources": labels}
    cited = [c.strip("[] ") for c in obj.citations]
    return {"answer": obj.answer, "citations": cited, "cited_items": [labels[c] for c in cited if c in labels],
            "insufficient": obj.insufficient_context, "valid": True, "attempts": attempts,
            "latency_s": time.perf_counter() - t0, "sources": labels}
