# Day 2 Notes — Training YOLO on a Custom Dataset

## Kaggle dataset
[andrewmvd/hard-hat-detection](https://www.kaggle.com/datasets/andrewmvd/hard-hat-detection) (~1.2 GB): 5,000 PNGs at 416×416 plus Pascal VOC XML files. Classes are `helmet`, `head`, and `person`. The notebook downloads it with `kagglehub`.

## Annotation formats

| Format | File layout | Box encoding |
|---|---|---|
| Pascal VOC | one `.xml` per image | `xmin ymin xmax ymax`, absolute pixels |
| COCO | one `.json` for the split | `x y w h`, absolute pixels |
| YOLO | one `.txt` per image, same stem | `class cx cy w h`, normalized 0–1 |

Ultralytics finds a label by swapping `/images/` for `/labels/` in the image path, so the two trees must mirror each other:

```
hardhat_yolo/
├── data.yaml
├── images/{train,val,test}/*.png
└── labels/{train,val,test}/*.txt
```

## Workflow in the notebook
1. Audit the labels: unknown classes, boxes outside the image, degenerate boxes, XML size vs real size.
2. Convert VOC → YOLO, then do a seeded 70/15/15 split.
3. **Round-trip check:** redraw boxes from the generated `.txt` files. Don't train until these look right.
4. Fine-tune `yolo26s.pt` (COCO-pretrained) for 40 epochs at 640 px with early stopping (`patience=15`).
5. Evaluate on the **test** split, which training and model selection never touch.
6. Measure recall by object size (COCO buckets: small < 32², medium < 96², large ≥ 96² px).

## Reading the results
- `metrics/mAP50(B)` rising then flattening is normal. If val loss climbs while train loss falls, the model is overfitting.
- `person` is far rarer than `helmet` and `head`, so expect it to have the lowest AP. Per-class AP plus instance counts tells you whether a weak class is a data problem.
- The normalized confusion matrix's `background` row and column show missed objects and false alarms, separate from class confusion.
- The success gate asks for a sensible mAP for the dataset size. Report test mAP@0.5 and mAP@0.5:0.95 with the image counts, and explain the weakest class.

## Levers if mAP is low
- Fix the labels first (see the audit).
- Higher `imgsz` or tiling (SAHI) for small objects.
- Bigger model (`yolo26m.pt`) or more epochs.
- Class imbalance: collect more `person` examples. Mosaic augmentation already helps somewhat.

## Pitfalls
- Never tune thresholds or pick checkpoints on the test split.
- Colab deletes `/content` when the session ends. Set `SAVE_TO_DRIVE=True` to keep `hardhat_best.pt`.

## Materials
- [Ultralytics — Train custom data](https://docs.ultralytics.com/modes/train/)
- [Roboflow blog & datasets](https://blog.roboflow.com/)
