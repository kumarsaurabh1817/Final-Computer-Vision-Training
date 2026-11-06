# ---- pipeline.py: shared helper code (canonical copy in week6_multimodal_rag/tools/shared/) ----
# The whole Week 6 pipeline behind one class, cached on Drive:
#     corpus -> parse -> caption figures -> items -> index        (built once, ~5 min on a T4)
#     query (+ image) -> retrieve -> grounded answer with citations
# Week 7's agent and Week 8's RAG API load the same folder: My Drive/cv_course_artifacts/week6_rag/
import json
import time

from PIL import Image


class MultimodalRAG:
    def __init__(self, work_dir: Path, llm: "LLM", text_emb: "TextEmbedder", clip: "ClipEmbedder"):
        self.work_dir, self.llm, self.text_emb, self.clip = work_dir, llm, text_emb, clip
        self.index: MultimodalIndex | None = None
        self.elements: list = []
        self.docs: list[dict] = []
        self.qa: list[dict] = []

    @property
    def ready(self) -> bool:
        return (self.work_dir / "index" / "items.json").exists() and (self.work_dir / "elements.jsonl").exists()

    def build(self, seed: int = 7, real_papers: bool = True, detector: "TableDetector | None" = None,
              reuse_elements: bool = True) -> "MultimodalRAG":
        """Corpus -> elements -> captions -> items -> index. Reuses parsed/captioned elements saved by an earlier day."""
        t0 = time.perf_counter()
        self.docs, _, self.qa = build_corpus(self.work_dir, seed=seed, real_papers=real_papers)
        cached = self.work_dir / "elements.jsonl"
        old = [Element(**json.loads(line)) for line in open(cached, encoding="utf-8")] if reuse_elements and cached.exists() else []
        have = {e.doc_id for e in old}
        self.elements = [e for e in old if e.doc_id in {d["doc_id"] for d in self.docs}]
        for d in self.docs:
            if d["doc_id"] not in have:
                self.elements += parse_pdf(d["path"], d["doc_id"], self.work_dir / "crops", detector=detector)
        titles = doc_titles(self.elements, self.docs)
        caption_figures(self.elements, self.llm, titles)  # only figures without a caption yet
        self.index = MultimodalIndex(self.text_emb, self.clip).build(make_items(self.elements, titles))
        self.save()
        print(f"built in {time.perf_counter() - t0:.0f}s: {len(self.docs)} docs, {len(self.elements)} elements, "
              f"{len(self.index.items)} items -> {self.work_dir}")
        return self

    def save(self) -> None:
        with open(self.work_dir / "elements.jsonl", "w", encoding="utf-8") as fh:
            for e in self.elements:
                fh.write(json.dumps(e.to_dict()) + "\n")
        self.index.save(self.work_dir / "index")

    def load(self) -> "MultimodalRAG":
        corpus = json.loads((self.work_dir / "corpus.json").read_text(encoding="utf-8"))
        self.docs, self.qa = corpus["docs"], corpus["qa"]
        self.elements = [Element(**json.loads(line)) for line in open(self.work_dir / "elements.jsonl", encoding="utf-8")]
        self.index = MultimodalIndex.load(self.work_dir / "index", self.text_emb, self.clip)
        print(f"loaded {len(self.docs)} docs, {len(self.elements)} elements, {len(self.index.items)} items from {self.work_dir}")
        return self

    def build_or_load(self, rebuild: bool = False, **kw) -> "MultimodalRAG":
        return self.load() if self.ready and not rebuild else self.build(**kw)

    def ask(self, question: str, image=None, k: int = 5, mode: str = "hybrid", reranker=None, use_images: bool = True) -> dict:
        t0 = time.perf_counter()
        img = Image.open(image) if isinstance(image, (str, Path)) else image
        hits = retrieve(self.index, question, img, k=k, mode=mode, reranker=reranker)
        out = generate_answer(self.llm, question, hits, query_image=img, use_images=use_images)
        out.update({"hits": hits, "total_latency_s": time.perf_counter() - t0})
        return out
