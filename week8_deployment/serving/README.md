# Serving — build and run the Week 8 services with Docker (WSL2)

These files are generated from `day3_serving/day3_serving.py` (`python tools/export_serving.py`). Do not edit them here.

## 1. Classifier API

The image needs the model folder, which is not in Git. Download it from Google Drive, `My Drive/cv_course_artifacts/week8/serving/classifier_api/model/` (created by Day 3), or copy these files into `classifier_api/model/`:

| File | From |
|---|---|
| `model.onnx` | `cv_course_artifacts/week8/resnet18_fp32.onnx` (Day 2) |
| `model.pt` | `cv_course_artifacts/week8/resnet18_week2.pt` (only for `/gradcam`) |
| `labels.json` | `cv_course_artifacts/week8/labels.json` |
| `reference.json` | `cv_course_artifacts/week8/reference.json` (Day 5, for `/drift`) |

```bash
cd week8_deployment/serving/classifier_api
docker build -t intel-classifier .
docker run --rm -p 8000:8000 intel-classifier

curl http://localhost:8000/health
curl -F "file=@some_photo.jpg" http://localhost:8000/predict
curl -F "file=@some_photo.jpg" http://localhost:8000/gradcam -o heatmap.png
curl http://localhost:8000/drift
```

## 2. RAG API

The Week 6 artifacts are mounted, not copied into the image. Download `My Drive/cv_course_artifacts/week6_rag/` first.

```bash
cd week8_deployment/serving/rag_api
docker build -t multimodal-rag .
# hosted LLM (no GPU needed)
docker run --rm -p 8001:8000 -e LLM_BACKEND=openrouter -e OPENROUTER_API_KEY=$OPENROUTER_API_KEY \
  -v /path/to/week6_rag:/artifacts/week6_rag multimodal-rag
# local Qwen3-VL on an NVIDIA GPU: add --gpus all -e LLM_BACKEND=local (the image installs CPU torch;
# switch the torch install line in the Dockerfile to a CUDA wheel for GPU inference)

curl -F "query=Who is the CEO of Tailspin Air?" -F "mode=pipeline" http://localhost:8001/ask
curl -F "query=Which company's report has a chart like this?" -F "image=@chart.png" http://localhost:8001/ask
```

The RAG artifacts store absolute paths (figure crops, query images) from Colab, `/content/drive/MyDrive/...`. If you mount them elsewhere, keep the same path inside the container (`-v /path/to/week6_rag:/content/drive/MyDrive/cv_course_artifacts/week6_rag` with `-e RAG_ARTIFACTS=` set to that path), or rebuild the index locally.

## 3. Both at once

```bash
cd week8_deployment/serving
OPENROUTER_API_KEY=... RAG_ARTIFACTS=/path/to/week6_rag docker compose up --build
```
