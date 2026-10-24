# Day 2 Notes — Layout-Aware Extraction (LayoutLMv3 & Donut)

## Data
- [FUNSD](https://guillaumejaume.github.io/FUNSD/): 149 train / 50 test forms. Each entity is `header`, `question`, `answer` or `other`; `linking` pairs connect questions to answers (= key-value pairs).
- [CORD-v2](https://huggingface.co/datasets/naver-clova-ix/cord-v2): receipts with JSON ground truth (`menu`, `sub_total`, `total`). Streamed from the Hugging Face Hub.

## Core ideas

**Why layout matters.** "12/10/98" is a date *answer* because it sits to the right of "Date:". Plain text loses that. Layout-aware models take each word's box (0–1000 grid) and often the page image too.

**LayoutLMv3** = one transformer over three inputs: text tokens, a 2D position embedding per token (from its word box), and image patches. Pretrained with masked language modeling, masked image modeling and word-patch alignment. Fine-tuned here for **token classification** with BIO tags (`B-QUESTION`, `I-QUESTION`, …).

**From labels to key-value JSON.** The token classifier says *what* each word is, not *which* answer belongs to which question. A geometric linker (nearest question to the left on the same line, or directly above) turns entities into `{key: value}`.

**Donut** (Document Understanding Transformer) = Swin image encoder + BART decoder. Reads pixels, *generates* a tag sequence → JSON. No OCR engine.

| | LayoutLMv3 | Donut |
|---|---|---|
| Needs OCR | Yes (words + boxes) | No |
| Output | Label per word → you build JSON | JSON directly |
| Provenance | Word boxes for every value | None |
| New document type | Same labels often transfer | New fine-tune with labeled JSON |
| Failure mode | OCR errors propagate | Hallucinated values |
| Licence | CC BY-NC-SA (non-commercial) | MIT |

## Metrics
- **Entity F1** (seqeval): an entity counts only if span and type both match.
- **KV pair F1**: (question text, answer text) pairs vs FUNSD links. Run with oracle entities to isolate linking errors.
- **Field-level F1** (Donut paper): flatten JSON to `(path, value)` leaves and compare as multisets.

## What to check in your run
- LayoutLMv3 vs RoBERTa (text only): how many F1 points do layout + image add?
- Zeroed boxes at inference: how much does the trained model rely on position?
- Oracle vs predicted KV F1: is the bottleneck labeling or linking?
- Donut on an invoice: it can only answer in CORD's schema.

## Pitfalls
- FUNSD forms can exceed 512 tokens; the notebook splits each form into word windows.
- Labels go on the **first sub-token** only; the others get `-100` (ignored by the loss).
- Boxes must be normalized to 0–1000 regardless of image size.
- Donut prices on CORD are Indonesian format (`60.000`): normalize before comparing.

## Materials
- [LayoutLMv3 (Hugging Face)](https://huggingface.co/docs/transformers/model_doc/layoutlmv3) · [paper](https://arxiv.org/abs/2204.08387)
- [Donut](https://github.com/clovaai/donut) · [HF docs](https://huggingface.co/docs/transformers/model_doc/donut)
