# Day 3 Notes — Serving with FastAPI and Docker

## Core ideas

**A model service** = load once at start-up, validate input, run inference, return JSON, report health. Two services, one pattern:

| | `classifier_api` | `rag_api` |
|---|---|---|
| Load at start-up | ONNX session, labels, drift reference | index, embedders, LLM, agent (FastAPI `lifespan`) |
| Main endpoint | `POST /predict` (multipart image) | `POST /ask` (form `query`, `mode`, optional `image`) |
| Extras | `/gradcam` (PNG), `/drift`, `/metrics` | citations with doc/page/bbox, route, tool calls |
| Concurrency | ORT session is thread-safe; `--workers 2` | one GPU model → a lock around generation |
| Image | `python:3.12-slim` + CPU torch (Grad-CAM only) | same; artifacts mounted as a volume |

**Preprocessing parity.** Training used torchvision, while the service re-implements resize → centre crop → normalize in NumPy/Pillow. A mismatch (RGB vs BGR, missing normalization, different resize) silently costs accuracy. Test it: compare probabilities from both paths on the same images.

**Errors are part of the API:** 400 for unreadable images, 413 for oversized uploads, 503 while loading or when an optional dependency is missing.

**Docker:**
```bash
docker build -t intel-classifier week8_deployment/serving/classifier_api
docker run --rm -p 8000:8000 intel-classifier
curl -F "file=@photo.jpg" http://localhost:8000/predict
```
- `HEALTHCHECK` lets orchestrators restart a dead container.
- Large models and indexes are **mounted**, not baked into the image (`-v …/week6_rag:/artifacts/week6_rag`).
- Secrets come in through environment variables (`-e OPENROUTER_API_KEY=…`), never through the image.

**docker-compose** runs both services together (`docker compose up --build` in `serving/`).

## Metrics
Accuracy over HTTP (should equal offline), server latency p50, round-trip p50/p95, requests/s with 1/4/8 concurrent clients, the preprocessing probability difference.

## What to check in your run
- `/health` reports the providers (CPU vs CUDA) and whether a drift reference is loaded.
- Accuracy over HTTP vs offline, and the parity number.
- Throughput as concurrency rises: where does it stop scaling?
- RAG API: pipeline vs agent latency for the same question.

## Pitfalls
- Loading the model per request (seconds) instead of once at start-up.
- `async def` endpoints that do blocking CPU work stall the event loop. Use `def` (thread pool) or a worker process.
- Unpinned dependency versions in the image: a rebuild next month behaves differently.
- Colab has no Docker daemon. Build and run the image in WSL2 or on a VM.

## Materials
- [FastAPI](https://fastapi.tiangolo.com/) · [Uvicorn deployment](https://www.uvicorn.org/deployment/)
- [Docker — Python images](https://docs.docker.com/language/python/) · [Deploy PyTorch with FastAPI + Docker](https://testdriven.io/blog/fastapi-docker-traefik/)
