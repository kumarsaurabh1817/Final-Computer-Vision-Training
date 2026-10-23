# Week 3 — Object Identification (Detection, Segmentation, Tracking & Anomaly Detection)

| Attribute | Detail |
|-----------|--------|
| **Pillar** | Vision Intelligence Core |
| **Goal** | Locate objects (detection), delineate them (segmentation), follow them across frames (tracking), and flag defects (anomaly detection). |
| **Effort** | 32–40 hours (full-time) |
| **Runtime** | Google Colab, T4 GPU. Every notebook is self-contained and downloads its own data. |

---

## Kaggle datasets (all downloaded automatically by `kagglehub`, no API token needed)

| Day | Dataset | Size | Used for |
|-----|---------|------|----------|
| 1 | [ultralytics/coco128](https://www.kaggle.com/datasets/ultralytics/coco128) | ~7 MB | IoU / NMS / mAP from scratch, YOLO26 vs Faster R-CNN |
| 2 | [andrewmvd/hard-hat-detection](https://www.kaggle.com/datasets/andrewmvd/hard-hat-detection) | ~1.2 GB | VOC→YOLO conversion, YOLO26 fine-tuning, test mAP |
| 3 | [julinmaloof/the-oxfordiiit-pet-dataset](https://www.kaggle.com/datasets/julinmaloof/the-oxfordiiit-pet-dataset) + coco128 | ~780 MB | U-Net training; DeepLabV3 / Mask R-CNN / YOLO26-seg demos |
| 4 | [mnhlfuch/mot17-02-frcnn](https://www.kaggle.com/datasets/mnhlfuch/mot17-02-frcnn) (fallback [wenhoujinjust/mot-17](https://www.kaggle.com/datasets/wenhoujinjust/mot-17)) | ~96 MB | Tracked video + MOTA / IDF1 / ID switches vs ground truth |
| 5 | [ipythonx/mvtec-ad](https://www.kaggle.com/datasets/ipythonx/mvtec-ad) (only the chosen category is fetched) | ~150 MB for `bottle` | Autoencoder vs PatchCore vs anomalib |

If you prefer manual downloads, upload the extracted folder to Colab and point the download cell at it. Each notebook locates files by searching below the returned path.

---

## How to run on Colab

1. Open [colab.research.google.com](https://colab.research.google.com) → **File → Upload notebook** → pick `dayN_*/Week3_DayN_Colab.ipynb`.
2. **Runtime → Change runtime type → T4 GPU**.
3. **Runtime → Run all**. Outputs (plots, CSV/JSON, weights, video) are written to `/content/week3_outputs/dayN/`.
4. Download what you want to commit from the Files panel. Colab wipes `/content` when the session ends.

Each notebook has a `FULL_RUN` flag. `False` gives a quick smoke test; only report `FULL_RUN=True` numbers.

---

## Day-by-day

### Day 1 — Detection concepts
Box formats, IoU, greedy NMS, and a COCO-style mAP evaluator, all written from scratch and verified against torchvision and Ultralytics. Pretrained YOLO26n (one-stage, anchor-free, NMS-free) vs Faster R-CNN (two-stage, anchors) on the same images: mAP and latency.
**Files:** [`Week3_Day1_Colab.ipynb`](day1_detection_concepts/Week3_Day1_Colab.ipynb) · [`day1_detection_concepts.py`](day1_detection_concepts/day1_detection_concepts.py) · [`notes.md`](day1_detection_concepts/notes.md) · ~5 min

### Day 2 — Train YOLO on a custom dataset
Parse Pascal VOC XML, audit label quality, convert to YOLO format, split 70/15/15, redraw labels to validate, fine-tune YOLO26s, and report per-class P/R/mAP on the held-out test split plus recall by object size.
**Files:** [`Week3_Day2_Colab.ipynb`](day2_yolo_custom_training/Week3_Day2_Colab.ipynb) · [`day2_yolo_custom_training.py`](day2_yolo_custom_training/day2_yolo_custom_training.py) · [`notes.md`](day2_yolo_custom_training/notes.md) · ~30–45 min

### Day 3 — Segmentation
Inspect DeepLabV3's `(C,H,W)` logits (semantic) vs Mask R-CNN's per-object soft masks (instance), try YOLO26-seg, then train a U-Net from scratch on Oxford-IIIT Pet trimaps (CE + Dice, mIoU/Dice) and compare it with zero-shot DeepLabV3.
**Files:** [`Week3_Day3_Colab.ipynb`](day3_segmentation/Week3_Day3_Colab.ipynb) · [`day3_segmentation.py`](day3_segmentation/day3_segmentation.py) · [`notes.md`](day3_segmentation/notes.md) · ~20–30 min

### Day 4 — Multi-object tracking
A naive IoU tracker (built here) vs ByteTrack vs BoT-SORT on top of YOLO26s on MOT17-02. MOTA, IDF1, ID switches, and fragmentations are computed from their definitions. The run also produces a track-lifetime chart and renders an H.264 tracked video with IDs and trails.
**Files:** [`Week3_Day4_Colab.ipynb`](day4_object_tracking/Week3_Day4_Colab.ipynb) · [`day4_object_tracking.py`](day4_object_tracking/day4_object_tracking.py) · [`notes.md`](day4_object_tracking/notes.md) · ~10 min

### Day 5 — Anomaly detection
One-class defect detection on MVTec AD: a convolutional autoencoder (reconstruction), PatchCore written from scratch (embedding distance + greedy coreset), and anomalib 2.7 PatchCore / PaDiM. Reports image and pixel AUROC, and flags defective samples at a threshold calibrated on good images only.
**Files:** [`Week3_Day5_Colab.ipynb`](day5_anomaly_detection/Week3_Day5_Colab.ipynb) · [`day5_anomaly_detection.py`](day5_anomaly_detection/day5_anomaly_detection.py) · [`notes.md`](day5_anomaly_detection/notes.md) · ~10–15 min

---

## Folder structure

```
week3_detection/
├── README.md
├── requirements.txt                 ← Week 3 deps (notebooks pip-install their own)
├── tools/py_to_ipynb.py             ← regenerates notebooks from the .py sources
├── day1_detection_concepts/         ← Week3_Day1_Colab.ipynb, .py, notes.md
├── day2_yolo_custom_training/       ← Week3_Day2_Colab.ipynb, .py, notes.md
├── day3_segmentation/               ← Week3_Day3_Colab.ipynb, .py, notes.md
├── day4_object_tracking/            ← Week3_Day4_Colab.ipynb, .py, notes.md
├── day5_anomaly_detection/          ← Week3_Day5_Colab.ipynb, .py, notes.md
├── models/                          ← put downloaded best.pt / unet weights here
└── results/                         ← put downloaded plots, CSV/JSON, video here
```

The `.py` file is the source of truth for each day: percent-format cells (`# %%`) that also run as plain scripts. After editing one, rebuild the notebook:

```bash
python tools/py_to_ipynb.py            # all days
python tools/py_to_ipynb.py day2_yolo_custom_training/day2_yolo_custom_training.py
```

---

## Deliverable
- [ ] YOLO detector trained on a custom dataset with test mAP reported (Day 2)
- [ ] Tracked output video with stable IDs, backed by IDF1 / IDSW (Day 4)
- [ ] Anomaly-detection demo flagging defective samples (Day 5)
- [ ] Detection, segmentation, tracking, and anomaly demos each documented (this README + notes)

## Success gate
- [ ] Detector reaches a sensible mAP for the dataset size (report per-class AP with instance counts)
- [ ] Video shows stable track IDs across frames (ByteTrack/BoT-SORT IDF1 clearly above the naive tracker)

> ⚠️ Detection datasets live or die on annotation quality and format. Days 1, 2, and 4 draw ground-truth labels before using them. Look at those plots.

---

## Results (fill in after your Colab runs)

| Day | Metric | Value |
|-----|--------|-------|
| 1 | YOLO26n mAP@0.5 / @0.5:0.95 on COCO128 (ours / Ultralytics) | __ / __ |
| 1 | Faster R-CNN mAP@0.5 / latency vs YOLO26n latency | __ / __ ms vs __ ms |
| 2 | Hard-hat test mAP@0.5 / mAP@0.5:0.95 | __ / __ |
| 2 | Test AP@0.5 helmet / head / person | __ / __ / __ |
| 3 | U-Net test mIoU / pet IoU vs DeepLabV3 pet IoU | __ / __ vs __ |
| 4 | IDF1 naive / ByteTrack / BoT-SORT | __ / __ / __ |
| 4 | ID switches naive / ByteTrack / BoT-SORT | __ / __ / __ |
| 5 | Image AUROC AE / PatchCore (ours) / anomalib PatchCore / PaDiM | __ / __ / __ / __ |
