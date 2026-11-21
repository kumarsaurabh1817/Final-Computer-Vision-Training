# CPU image for the Intel scene classifier API.
#   docker build -t intel-classifier .
#   docker run --rm -p 8000:8000 intel-classifier
FROM python:3.12-slim
WORKDIR /srv
# CPU-only PyTorch (for /gradcam); /predict uses ONNX Runtime only
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY app/ app/
COPY model/ model/
ENV MODEL_DIR=/srv/model
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
