# Day 2 Notes — Visual Question Answering

## Data
- [SROIE v2 (Kaggle: `urbikn/sroie-datasetv2`)](https://www.kaggle.com/datasets/urbikn/sroie-datasetv2): scanned Malaysian receipts from the ICDAR 2019 SROIE challenge. Each receipt has `img/<id>.jpg` and `entities/<id>.txt` (JSON with `company`, `date`, `address`, `total`, written as printed). The notebook uses 60 test receipts.
- [Flickr8k (Kaggle: `adityajn105/flickr8k`)](https://www.kaggle.com/datasets/adityajn105/flickr8k): photos for the ViLT vs VLM comparison.
- **Week 4 held-out invoices**: rendered in code with the same generator, seed (2026) and ids (`d5_<layout>_00/01`) as Week 4 Day 5, so the rows match your Week 4 comparison matrix one-to-one.

## Core ideas

**Two kinds of VQA model.**

| | ViLT (classic) | Qwen3-VL (generative) |
|---|---|---|
| Output | classifier over 3,129 frequent VQAv2 answers | free text, token by token |
| Can read text | no | yes (OCR is in its pretraining) |
| Open-ended answers | no | yes |
| Speed | very fast, one forward pass | slower, autoregressive |
| Typical use | closed-domain yes/no, colours, counts | documents, explanations, anything open |

**Document VQA = reading + locating + formatting.** "What is the total?" requires reading the digits, finding the right number among subtotal, tax and total, and returning it in a usable format. OCR + rules separate these steps. A VLM does all three in one pass, which makes it flexible but harder to debug.

**Per-field VQA is expensive.** Nine questions per invoice means nine prefills of the same ~2,000 image tokens. Day 3 replaces this with one structured call.

**Validate every number.** VLMs output confident, well-formatted, wrong values. Two checks need no ground truth:
1. *Arithmetic*: subtotal + tax = total.
2. *OCR grounding*: an independent OCR pass must find the value on the page. A value no OCR engine can see is a candidate hallucination.

## Metrics
- **Exact match** after normalization: uppercase, punctuation stripped, dates parsed day-first for SROIE, money as 2-decimal numbers.
- **ANLS** (Average Normalized Levenshtein Similarity): similarity = 1 − edit distance / max length, set to 0 below 0.5. It gives partial credit for small misreads and is the DocVQA standard.
- **Week 4 `field_scores`** on invoices: money within ±0.01, ISO dates, case/whitespace-insensitive text. Reused unchanged.
- **Flag precision/recall**: how many real VLM errors the arithmetic/grounding checks catch, and how many flags are false alarms.

## What to check in your run
- SROIE: which field is hardest for each approach? (Usually address for both. Company is hard for rules when the logo line is not text.)
- Invoices: does the VLM hold up on `layout_c`, the layout your Week 4 rules were never written for?
- Rotated scans: the VLM gets the crooked page as is, with no deskew. Does it still read it?
- Which wrong values are caught by grounding vs arithmetic?
- Resolution: the smallest image size where accuracy stops improving.

## Pitfalls
- SROIE ground truth is as printed (`25/12/2018`, `RM 9.00`), so normalize before comparing or exact match will mislead you.
- Ask for "the value only". Otherwise the VLM answers "The total amount is $1,234.50." and the parsing step becomes the bottleneck.
- Multi-page invoices: pass every page as a separate image in the same prompt. The total is on the last page.
- Day 2 reloads Flickr8k from the kagglehub cache. If you run in a fresh Colab VM, it downloads again.

## Materials
- [Hugging Face — Visual Question Answering task guide](https://huggingface.co/docs/transformers/tasks/visual_question_answering)
- [ViLT paper](https://arxiv.org/abs/2102.03334) · [DocVQA & ANLS](https://arxiv.org/abs/2007.00398) · [SROIE challenge](https://rrc.cvc.uab.es/?ch=13)
