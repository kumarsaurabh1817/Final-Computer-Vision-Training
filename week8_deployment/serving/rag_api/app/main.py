"""Multimodal RAG API (Week 8): the Week 6 pipeline and the Week 7 agent behind HTTP.

GET  /health   index and model info
POST /ask      form: query (str), mode ("agent" | "pipeline"), k (int), image (optional file) -> grounded answer + citations
"""
import io
import os
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path

import torch
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from PIL import Image, UnidentifiedImageError

from .rag_core import Agent, ClipEmbedder, LLM, MultimodalRAG, TextEmbedder

ARTIFACTS = Path(os.environ.get("RAG_ARTIFACTS", "/artifacts/week6_rag"))
BACKEND = os.environ.get("LLM_BACKEND", "openrouter")
MODEL_ID = os.environ.get("LLM_MODEL") or None
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
STATE: dict = {}
LOCK = threading.Lock()


@asynccontextmanager
async def lifespan(app: FastAPI):
    llm = LLM(BACKEND, MODEL_ID)
    rag = MultimodalRAG(ARTIFACTS, llm, TextEmbedder(device=DEVICE), ClipEmbedder(device=DEVICE)).load()
    STATE.update(rag=rag, agent=Agent(rag, verbose=False))
    yield
    STATE.clear()


app = FastAPI(title="Multimodal RAG", version="1.0", lifespan=lifespan)


def cite(h: dict) -> dict:
    return {k: h.get(k) for k in ("id", "doc_id", "doc_title", "type", "page", "bbox")}


@app.get("/health")
def health() -> dict:
    rag = STATE.get("rag")
    return {"status": "ok" if rag else "loading", "llm": f"{BACKEND}:{rag.llm.model_id}" if rag else None,
            "documents": len(rag.docs) if rag else 0, "items": len(rag.index.items) if rag else 0}


@app.post("/ask")
def ask(query: str = Form(...), mode: str = Form("agent"), k: int = Form(5), image: UploadFile | None = File(None)) -> dict:
    if "rag" not in STATE:
        raise HTTPException(503, "still loading")
    if mode not in ("agent", "pipeline"):
        raise HTTPException(400, "mode must be 'agent' or 'pipeline'")
    img = None
    if image is not None:
        try:
            img = Image.open(io.BytesIO(image.file.read())).convert("RGB")
        except (UnidentifiedImageError, OSError):
            raise HTTPException(400, "image is not readable")
    t0 = time.perf_counter()
    with LOCK:  # one model on one GPU: serialize generation
        out = STATE["agent"].run(query, img) if mode == "agent" else STATE["rag"].ask(query, img, k=k)
    return {"answer": out["answer"], "citations": [cite(h) for h in out.get("cited_items", [])], "mode": mode,
            "route": out.get("route"), "tool_calls": out.get("tool_calls"), "latency_s": round(time.perf_counter() - t0, 2)}
