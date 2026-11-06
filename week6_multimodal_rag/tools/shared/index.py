# ---- index.py: shared helper code (canonical copy in week6_multimodal_rag/tools/shared/) ----
# Multimodal index (Week 6 Day 3) + retrieval (Day 4).
#
# Two vector spaces, one metadata store:
#     text space  (BGE)  : every item's text  -> text chunks, table markdown, figure captions ("caption-as-text")
#     image space (CLIP) : every figure crop  -> reached by a CLIP text query or a query image ("raw embedding")
# Each hit carries its provenance (doc, page, bbox, type, element ids, image path) all the way to the answer.
import json

import faiss
import numpy as np
from PIL import Image


class MultimodalIndex:
    def __init__(self, text_emb: "TextEmbedder", clip: "ClipEmbedder"):
        self.text_emb, self.clip = text_emb, clip
        self.items: list[dict] = []
        self.text_index = self.image_index = None
        self.image_rows: list[int] = []  # image-space row -> item index

    def build(self, items: list[dict]) -> "MultimodalIndex":
        self.items = items
        tv = self.text_emb.passages([it["text"] for it in items])
        self.text_index = faiss.IndexFlatIP(tv.shape[1])
        self.text_index.add(tv)
        self.image_rows = [i for i, it in enumerate(items) if it.get("image_path")]
        iv = self.clip.images([items[i]["image_path"] for i in self.image_rows])
        self.image_index = faiss.IndexFlatIP(iv.shape[1] if len(iv) else 512)
        if len(iv):
            self.image_index.add(iv)
        return self

    def _hits(self, index, qv: np.ndarray, k: int, rows: list[int] | None, space: str, types=None, doc_ids=None) -> list[dict]:
        if index is None or index.ntotal == 0:
            return []
        scores, ids = index.search(qv.reshape(1, -1).astype(np.float32), min(index.ntotal, max(k * 5, 50)))
        out = []
        for s, i in zip(scores[0], ids[0]):
            if i < 0:
                continue
            it = self.items[rows[i] if rows is not None else i]
            if (types and it["type"] not in types) or (doc_ids and it["doc_id"] not in doc_ids):
                continue  # filters are applied after an over-fetch (FAISS flat indexes have no payload filters)
            out.append({**it, "score": float(s), "space": space})
            if len(out) >= k:
                break
        return out

    def search_text(self, query: str, k: int = 5, **filters) -> list[dict]:
        return self._hits(self.text_index, self.text_emb.queries([query])[0], k, None, "text", **filters)

    def search_image_by_text(self, query: str, k: int = 5, **filters) -> list[dict]:
        return self._hits(self.image_index, self.clip.texts([query])[0], k, self.image_rows, "clip_text->image", **filters)

    def search_by_image(self, image, query: str | None = None, alpha: float = 0.5, k: int = 5, **filters) -> list[dict]:
        v = self.clip.images([image])[0]
        if query:
            v = v + alpha * self.clip.texts([query])[0]
            v = v / np.linalg.norm(v)
        return self._hits(self.image_index, v, k, self.image_rows, "clip_image->image", **filters)

    def save(self, folder: Path) -> None:
        folder.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.text_index, str(folder / "text.faiss"))
        faiss.write_index(self.image_index, str(folder / "image.faiss"))
        (folder / "items.json").write_text(json.dumps({"text_model": self.text_emb.model_id, "clip_model": self.clip.name,
                                                       "image_rows": self.image_rows, "items": self.items}, indent=1), encoding="utf-8")

    @classmethod
    def load(cls, folder: Path, text_emb: "TextEmbedder", clip: "ClipEmbedder") -> "MultimodalIndex":
        meta = json.loads((folder / "items.json").read_text(encoding="utf-8"))
        if meta["text_model"] != text_emb.model_id or meta["clip_model"] != clip.name:
            raise ValueError(f"index built with {meta['text_model']} / {meta['clip_model']}")
        obj = cls(text_emb, clip)
        obj.items, obj.image_rows = meta["items"], meta["image_rows"]
        obj.text_index = faiss.read_index(str(folder / "text.faiss"))
        obj.image_index = faiss.read_index(str(folder / "image.faiss"))
        return obj


def rrf_fuse(result_lists: list[list[dict]], k: int, c: int = 60) -> list[dict]:
    """Reciprocal-rank fusion over item ids; a fused hit remembers every space that found it."""
    scores, best = {}, {}
    for hits in result_lists:
        for rank, h in enumerate(hits, 1):
            scores[h["id"]] = scores.get(h["id"], 0.0) + 1.0 / (c + rank)
            if h["id"] not in best:
                best[h["id"]] = {**h, "spaces": [h["space"]]}
            elif h["space"] not in best[h["id"]]["spaces"]:
                best[h["id"]]["spaces"].append(h["space"])
    order = sorted(scores, key=scores.get, reverse=True)[:k]
    return [{**best[i], "score": scores[i]} for i in order]


class Reranker:
    """Cross-encoder: reads (query, passage) together. Slower than embeddings, sharper ranking of the top candidates."""

    def __init__(self, model_id: str = "cross-encoder/ms-marco-MiniLM-L-6-v2", device: str = "cpu"):
        from sentence_transformers import CrossEncoder
        self.model = CrossEncoder(model_id, device=device)

    def rerank(self, query: str, hits: list[dict], k: int) -> list[dict]:
        if not hits:
            return hits
        scores = self.model.predict([(query, h["text"][:1500]) for h in hits])
        return [{**h, "rerank_score": float(s)} for s, h in sorted(zip(scores, hits), key=lambda x: -x[0])][:k]


def retrieve(index: MultimodalIndex, query: str | None = None, image=None, k: int = 5, mode: str = "hybrid",
             reranker: "Reranker | None" = None, types=None, doc_ids=None) -> list[dict]:
    """
    mode "text"   : text space only (chunks, tables, figure captions-as-text)
         "image"  : CLIP image space only (text->image, or image->image when an image is given)
         "hybrid" : both, fused with RRF (default)
    A query image adds image->image search (composed with the text when both are given).
    """
    filters = {"types": types, "doc_ids": doc_ids}
    lists = []
    if query and mode in ("text", "hybrid"):
        lists.append(index.search_text(query, k * 3, **filters))
    if query and mode in ("image", "hybrid") and image is None:
        lists.append(index.search_image_by_text(query, k * 3, **filters))
    if image is not None:
        lists.append(index.search_by_image(image, query if mode != "image" else None, k=k * 3, **filters))
    fused = rrf_fuse(lists, k * 3 if reranker else k)
    return reranker.rerank(query, fused, k) if reranker and query else fused[:k]
