# Day 1 Notes — OCR Landscape & Preprocessing

## Data
- [FUNSD](https://guillaumejaume.github.io/FUNSD/) (~16 MB): 199 real scanned forms with word boxes and text. The notebook downloads the official zip and uses the 50 test forms.
- Synthetic invoices rendered inside the notebook with exact text ground truth, then degraded on purpose (skew, noise, blur, low contrast, JPEG).

## Core ideas

**OCR = detection + recognition.** Find where text is (boxes), then read each box. Engines differ mainly in how they do these two steps.

| Engine | Pipeline | Wins when | Loses when |
|---|---|---|---|
| Tesseract 5 | Layout analysis → LSTM line recognizer | Clean printed pages, CPU only, high volume | Noise, skew, low contrast, handwriting |
| EasyOCR | CRAFT detector → CRNN recognizer | Quick multilingual prototype on GPU | CPU speed; returns phrases, not words |
| PaddleOCR (PP-OCRv5) | DB detector → SVTR recognizer | Dense, small or rotated text; best accuracy overall | Heavy install, API changed in 3.x |

**Measuring OCR.**
- *Word F1* (order-free): multiset overlap of normalized words. Robust to reading-order differences.
- *CER* = edit distance / GT characters, after putting both outputs in the same reading order. Catches `0`↔`O`, `1`↔`l`, which ruin invoice numbers and amounts.

**Preprocessing steps.**
- Grayscale → denoise (median / non-local means) → **deskew** → CLAHE (contrast) → threshold (Otsu global / adaptive local).
- Deskew via projection profile: rotate the page over candidate angles; when lines are horizontal, the row sums of ink are spiky, so their variance peaks at the true angle.
- Page segmentation modes (PSM) tell Tesseract what layout to expect: 3 = auto, 4 = single column, 6 = one block, 11 = sparse text. Forms are often sparse.

## What to check in your run
- Engine ranking on FUNSD (real scans) vs synthetic (clean): clean renders hide differences.
- The ablation heatmap: which degradation does each step fix, and which step *hurts* on clean input?
- Confidence calibration: is there a confidence below which most words are wrong? That is your human-review trigger.

## Pitfalls
- Preprocessing is not always good. Thresholding a clean image can break thin strokes.
- Deep-learning engines gain less from preprocessing than Tesseract does; tune per engine.
- CER can exceed 1.0 when the engine invents text (noise read as characters).
- Install: Tesseract is a system binary (`apt-get install tesseract-ocr`); `pytesseract` is only the wrapper.

## Materials
- [Tesseract OCR](https://github.com/tesseract-ocr/tesseract) · [Improving quality](https://tesseract-ocr.github.io/tessdoc/ImproveQuality.html)
- [EasyOCR](https://github.com/JaidedAI/EasyOCR)
- [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR)
