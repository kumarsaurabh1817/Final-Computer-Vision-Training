# Day 3 Notes — Semantic vs Instance Segmentation

## Kaggle datasets
- [julinmaloof/the-oxfordiiit-pet-dataset](https://www.kaggle.com/datasets/julinmaloof/the-oxfordiiit-pet-dataset) (~780 MB): 7,390 images plus `annotations/trimaps/*.png`, used for U-Net training. If Kaggle fails, the notebook falls back to torchvision's `OxfordIIITPet` downloader.
- [ultralytics/coco128](https://www.kaggle.com/datasets/ultralytics/coco128) (~7 MB): busy multi-object scenes for the instance-segmentation demo.

## Semantic vs instance

| | Semantic | Instance |
|---|---|---|
| Question | What class is each pixel? | Which object does each pixel belong to? |
| Output | `(C, H, W)` logits → `argmax` → `(H, W)` label map | List of `(mask H×W, box, class, score)` |
| Two touching people | One merged "person" region | Two separate masks |
| Models | U-Net, DeepLabV3, SegFormer | Mask R-CNN, YOLO-seg |

Panoptic segmentation combines the two: instances for "things" (people, cars) and semantic labels for "stuff" (sky, road).

## U-Net in one paragraph
The encoder (conv blocks + max-pool) halves resolution and doubles channels at each level. The decoder upsamples back with transposed convolutions. At each scale, a **skip connection** concatenates the matching encoder feature map, so the decoder gets semantic context from the bottleneck *and* precise edges from early layers. A 1×1 conv at the end produces per-pixel class logits.

## Mask R-CNN in one paragraph
Mask R-CNN is Faster R-CNN plus a small FCN mask head that predicts a 28×28 soft mask per detected box, using RoIAlign so the features line up with pixels. Masks are pasted back into the box at full resolution; threshold them at 0.5. torchvision's `maskrcnn_resnet50_fpn_v2` follows the same design as Detectron2's reference model and needs no compilation on Colab.

## YOLO-seg
YOLO-seg predicts shared prototype masks plus per-detection coefficients, so each instance mask is a weighted sum of prototypes. It is much faster than Mask R-CNN, with coarser boundaries. Its labels are polygons (`class x1 y1 x2 y2 …`, normalized).

## Losses and metrics
- **Cross-entropy:** per-pixel classification. Dominated by large classes.
- **Dice loss:** overlap-based and robust to imbalance. CE + Dice is the standard combination.
- **IoU per class**, **mIoU** (mean over classes), **Dice/F1**. Pixel accuracy alone hides a failed small class.
- Pet trimap ids: 1 pet, 2 background, 3 border. The border is ambiguous, so it is ignored in the binary pet-IoU comparison.

## What to check in your run
- U-Net test mIoU and per-class IoU (`border` will be lowest).
- Pet IoU: U-Net from scratch vs zero-shot DeepLabV3. Which wins, and why? (Think pretraining data.)
- Look at the error maps: are errors concentrated at boundaries or whole missed regions?

## Pitfalls
- Resize masks with **nearest**, never bilinear.
- Apply geometric augmentation to image and mask together; photometric augmentation to the image only.
- Some Oxford-IIIT Pet files are unreadable; the loader skips them and reports how many.

## Materials
- [U-Net paper (arXiv:1505.04597)](https://arxiv.org/abs/1505.04597)
- [Detectron2](https://github.com/facebookresearch/detectron2)
