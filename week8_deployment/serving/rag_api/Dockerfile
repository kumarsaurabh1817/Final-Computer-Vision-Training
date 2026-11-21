# Image for the multimodal RAG API. The Week 6 artifacts (index, crops, corpus.json) are mounted, not baked in.
#   docker build -t multimodal-rag .
#   docker run --rm -p 8001:8000 -e LLM_BACKEND=openrouter -e OPENROUTER_API_KEY=... \
#       -v /path/to/cv_course_artifacts/week6_rag:/artifacts/week6_rag multimodal-rag
# With an NVIDIA GPU: add --gpus all and -e LLM_BACKEND=local (Qwen3-VL-4B, ~9 GB VRAM).
FROM python:3.12-slim
WORKDIR /srv
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY app/ app/
ENV RAG_ARTIFACTS=/artifacts/week6_rag LLM_BACKEND=openrouter
EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
