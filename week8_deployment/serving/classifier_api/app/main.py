"""Intel scene classifier API (Week 8).

GET  /health    liveness + model info
POST /predict   image file -> {label, confidence, probabilities, latency_ms}
POST /gradcam   image file -> PNG heatmap of the pixels behind the prediction (needs torch + model.pt)
GET  /drift     rolling-window drift report vs reference.json (KS test on confidence, PSI)
GET  /metrics   request counts and latency
"""
import io
import json
import os
import threading
import time
from collections import deque
from pathlib import Path

import numpy as np
import onnxruntime as ort
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import Response
from PIL import Image, UnidentifiedImageError

MODEL_DIR = Path(os.environ.get("MODEL_DIR", Path(__file__).resolve().parent.parent / "model"))
ONNX_FILE = os.environ.get("ONNX_FILE", "model.onnx")
MAX_BYTES = int(os.environ.get("MAX_UPLOAD_MB", "10")) * 1024 * 1024
LABELS = json.loads((MODEL_DIR / "labels.json").read_text())
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)
PROVIDERS = [p for p in ("CUDAExecutionProvider", "CPUExecutionProvider") if p in ort.get_available_providers()]
SESSION = ort.InferenceSession(str(MODEL_DIR / ONNX_FILE), providers=PROVIDERS)
INPUT = SESSION.get_inputs()[0].name

app = FastAPI(title="Intel scene classifier", version="1.0")
STATS = {"requests": 0, "errors": 0, "latency_ms": deque(maxlen=1000)}


def preprocess(img: Image.Image) -> np.ndarray:
    """Same as torchvision Resize(256) + CenterCrop(224) + ToTensor + Normalize."""
    img = img.convert("RGB")
    w, h = img.size
    s = 256 / min(w, h)
    img = img.resize((max(256, int(w * s)), max(256, int(h * s))), Image.BILINEAR)
    w, h = img.size
    left, top = int(round((w - 224) / 2.0)), int(round((h - 224) / 2.0))
    x = np.asarray(img.crop((left, top, left + 224, top + 224)), dtype=np.float32).transpose(2, 0, 1) / 255.0
    return ((x - MEAN) / STD)[None].astype(np.float32)


def softmax(z: np.ndarray) -> np.ndarray:
    e = np.exp(z - z.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


async def read_image(file: UploadFile) -> Image.Image:
    data = await file.read()
    if len(data) > MAX_BYTES:
        raise HTTPException(413, f"file larger than {MAX_BYTES // 1024 // 1024} MB")
    try:
        img = Image.open(io.BytesIO(data))
        img.load()
        return img.convert("RGB")
    except (UnidentifiedImageError, OSError):
        raise HTTPException(400, "not a readable image")


class DriftMonitor:
    """Rolling window of prediction confidence and image brightness, compared with the reference set (Day 5)."""

    def __init__(self, window: int = 200):
        ref_path = MODEL_DIR / "reference.json"
        self.ref = json.loads(ref_path.read_text()) if ref_path.exists() else None
        self.conf, self.bright = deque(maxlen=window), deque(maxlen=window)
        self.lock = threading.Lock()

    def add(self, conf: float, brightness: float) -> None:
        with self.lock:
            self.conf.append(conf)
            self.bright.append(brightness)

    def report(self) -> dict:
        if self.ref is None:
            return {"status": "no reference.json in the model folder (created on Week 8 Day 5)"}
        with self.lock:
            conf, bright = np.array(self.conf), np.array(self.bright)
        if len(conf) < 30:
            return {"status": "collecting", "window": len(conf)}
        from scipy.stats import ks_2samp
        edges = np.array(self.ref["confidence_bin_edges"])
        ref_frac = np.array(self.ref["confidence_bin_frac"])
        cur = np.histogram(np.clip(conf, edges[0], edges[-1]), bins=edges)[0] / len(conf)
        psi = float(np.sum((cur - ref_frac) * np.log((cur + 1e-4) / (ref_frac + 1e-4))))
        p_conf = float(ks_2samp(conf, self.ref["confidence_sample"]).pvalue)
        p_bright = float(ks_2samp(bright, self.ref["brightness_sample"]).pvalue)
        alert = psi > self.ref.get("psi_threshold", 0.2) or min(p_conf, p_bright) < self.ref.get("p_threshold", 0.01)
        return {"status": "DRIFT" if alert else "ok", "window": len(conf), "psi_confidence": psi, "ks_p_confidence": p_conf,
                "ks_p_brightness": p_bright, "mean_confidence": float(conf.mean()), "reference_mean_confidence": self.ref["mean_confidence"]}


DRIFT = DriftMonitor()
CAM = {}


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "model": ONNX_FILE, "providers": SESSION.get_providers(), "classes": LABELS,
            "drift_reference": DRIFT.ref is not None}


@app.post("/predict")
async def predict(file: UploadFile = File(...)) -> dict:
    t0 = time.perf_counter()
    STATS["requests"] += 1
    try:
        img = await read_image(file)
    except HTTPException:
        STATS["errors"] += 1
        raise
    probs = softmax(SESSION.run(None, {INPUT: preprocess(img)})[0])[0]
    i = int(probs.argmax())
    DRIFT.add(float(probs[i]), float(np.asarray(img.convert("L"), dtype=np.float32).mean() / 255.0))
    ms = (time.perf_counter() - t0) * 1000
    STATS["latency_ms"].append(ms)
    return {"label": LABELS[i], "confidence": float(probs[i]), "probabilities": {c: float(p) for c, p in zip(LABELS, probs)},
            "latency_ms": round(ms, 2)}


def _cam_model():
    if "model" not in CAM:
        import torch
        import torch.nn as nn
        from torchvision.models import resnet18
        path = MODEL_DIR / "model.pt"
        if not path.exists():
            raise HTTPException(503, "Grad-CAM needs model.pt (the PyTorch weights) in the model folder")
        m = resnet18()
        m.fc = nn.Sequential(nn.Dropout(0.3), nn.Linear(m.fc.in_features, len(LABELS)))
        m.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
        CAM["model"] = m.eval()
    return CAM["model"]


@app.post("/gradcam")
async def gradcam(file: UploadFile = File(...)) -> Response:
    """Heatmap over the image: which pixels pushed the score of the predicted class (Grad-CAM on layer4)."""
    try:
        import torch
        from pytorch_grad_cam import GradCAM
        from pytorch_grad_cam.utils.image import show_cam_on_image
        from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
    except ImportError:
        raise HTTPException(503, "Grad-CAM needs torch, torchvision and grad-cam installed in this image")
    img = await read_image(file)
    model = _cam_model()
    x = torch.from_numpy(preprocess(img))
    with torch.no_grad():
        cls = int(model(x).argmax(1))
    with GradCAM(model=model, target_layers=[model.layer4[-1]]) as cam:
        mask = cam(input_tensor=x, targets=[ClassifierOutputTarget(cls)])[0]
    rgb = ((x[0].numpy() * STD + MEAN).transpose(1, 2, 0)).clip(0, 1)
    overlay = Image.fromarray(show_cam_on_image(rgb.astype(np.float32), mask, use_rgb=True))
    buf = io.BytesIO()
    overlay.save(buf, format="PNG")
    return Response(buf.getvalue(), media_type="image/png", headers={"X-Predicted-Label": LABELS[cls]})


@app.get("/drift")
def drift() -> dict:
    return DRIFT.report()


@app.get("/metrics")
def metrics() -> dict:
    lat = np.array(STATS["latency_ms"]) if STATS["latency_ms"] else np.array([0.0])
    return {"requests": STATS["requests"], "errors": STATS["errors"], "latency_ms_p50": float(np.percentile(lat, 50)),
            "latency_ms_p95": float(np.percentile(lat, 95))}
