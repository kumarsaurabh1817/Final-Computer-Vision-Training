# Day 1 Notes — Detection Concepts

## Kaggle dataset
[ultralytics/coco128](https://www.kaggle.com/datasets/ultralytics/coco128) (~7 MB). The notebook downloads it with `kagglehub`; no manual step needed.

## Core ideas

**Box formats.** `xyxy` (corners, pixels), `xywh` (COCO JSON), YOLO `cx cy w h` normalized to 0–1. Convert once, in one tested function, and never by hand.

**IoU** = intersection area / union area. It decides whether a prediction is a hit. 0.5 is a loose hit; 0.75+ means tight localization.

**Precision / recall at a threshold.** Precision = TP / (TP + FP), recall = TP / #GT. Raising the confidence threshold trades recall for precision.

**AP and mAP.**
1. Sort predictions of one class by confidence.
2. Mark each TP/FP by matching to the best unmatched GT with IoU ≥ t.
3. Build cumulative precision/recall, take the monotone envelope, and average precision at 101 recall points.

- `mAP@0.5`: mean AP over classes at IoU 0.5 (Pascal VOC-style headline).
- `mAP@0.5:0.95`: also averaged over IoU 0.50, 0.55, …, 0.95 (COCO headline). It punishes sloppy boxes.

**NMS.** Keep the top-scoring box, drop neighbours with IoU > threshold, repeat. Needed when a head outputs duplicates.

**One-stage vs two-stage.**

| | One-stage (YOLO) | Two-stage (Faster R-CNN) |
|---|---|---|
| Pipeline | Dense predictions in one pass | RPN proposals → RoIAlign → classify/refine |
| Speed | Real-time | Slower, more compute per object |
| Anchors | YOLOv8+/YOLO11/YOLO26 are anchor-free | Anchor-based (sizes × ratios per FPN level) |
| Post-processing | YOLO26: NMS-free one-to-one head | NMS in RPN and final head |

**Anchor vs anchor-free.** Anchor-based models regress offsets from preset template boxes, so anchor sizes must suit your objects. Anchor-free models predict the box directly from each location, which means fewer hyperparameters.

## What to check in your run
- Our evaluator and `model.val()` agree to within a few points. A big gap means a format bug.
- Faster R-CNN vs YOLO26n: compare `ms/img` with the mAP difference.
- The confidence sweep: which threshold maximizes F1?

## Pitfalls
- COCO128 images come from COCO *train2017*, so both pretrained models saw them during training. Treat the absolute mAP as optimistic.
- `conf=0.001` is right for computing mAP and wrong for deployment. Choose a deployment threshold from the sweep.
- Ultralytics `Results.plot()` returns **BGR**. Reverse the channels before `imshow`.

## Materials
- [Ultralytics YOLO docs](https://docs.ultralytics.com/)
- [Roboflow — mAP explained](https://blog.roboflow.com/mean-average-precision/)
