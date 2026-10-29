# Day 3 Notes — Explaining CV Output & Schema-Valid JSON

## Data
- [Flickr8k (Kaggle: `adityajn105/flickr8k`)](https://www.kaggle.com/datasets/adityajn105/flickr8k): scenes for the explanation task.
- Week 4 held-out invoices (`d5_*`, seed 2026), rendered in code. These are the same documents as Week 4 Day 5 and Day 2.
- Optional uploads for the three-way table: Week 4 `day5_scores_per_doc.csv` (Azure / Textract rows) and Day 2 `day2_invoice_scores.csv`.

## Core ideas

**Explaining CV output.** The VLM gets the image plus the detector's and classifier's output as JSON. It returns an explanation a person can read and a program can check: confirmed objects with counts, likely false positives, missed objects, and agreement with the classifier. A useful explainer has to be able to *disagree* with the model it explains. The injected fake detection tests exactly that.

**Three ways to get JSON:**

| Mode | Mechanism | Valid output | Cost |
|---|---|---|---|
| `prompt_only` | describe the keys in words, parse the reply | often, not always (extra text, wrong enums, wrong date format) | 1 call |
| `schema_repair` | JSON Schema in the prompt → pydantic validation → error message back → retry | high, if a retry succeeds | 1–3 calls |
| `constrained` | Outlines compiles the schema to a token-level automaton and masks invalid tokens at every step | by construction (structure, types, enums, regex patterns) | 1 call, some overhead |

**Valid ≠ correct.** Constrained decoding guarantees the *shape* only. The model can still write `1284.50` when the page says `1248.50`. You still need semantic checks: arithmetic, OCR grounding, and consistency with the detector output.

**Schema design rules:**
- Make keys **required but nullable** (`str | None` with no default). The model then has to say `null` instead of silently dropping a key.
- Use `Literal[...]` enums for anything a program branches on, and `pattern=` for dates and currency codes.
- Keep strings bounded (`max_length`). Long unbounded strings are where constrained decoding runs out of `max_new_tokens`.

## Metrics
- **JSON parse rate**: a `{...}` object was found and parsed.
- **Schema-valid rate**: pydantic `model_validate` succeeds.
- **Mean attempts**: repair rounds needed.
- **Consistency with CV output**: fraction of confirmed objects whose label the detector produced (`labels_consistent`) and whose count matches (`counts_consistent`).
- **Injected false positive**: `flagged_fake` (good) vs `confirmed_fake` (blindly trusted).
- **Invoices**: Week 4 field accuracy / ANLS, latency and cost per page (GPU $/h × time), and the `needs_review` flag's catch rate.

## What to check in your run
- Which validation errors does `prompt_only` make? (Typical: prose around the JSON, `"Agree"` instead of `"agree"`, dates as `March 3, 2026`.)
- Does `constrained` reach 100% valid? If not, the cause is almost always output truncated by `max_new_tokens`.
- Does the VLM flag the fake giraffe, or repeat it?
- One structured call vs Day 2's nine questions: accuracy and latency per page.
- Three-way table: where does the VLM beat Week 4 (unseen layout?) and where does it lose (latency, digits)?

## Pitfalls
- The Outlines API changed in v1.0 (`outlines.from_transformers`, `outlines.inputs.Image`). If your installed version does not match, the notebook keeps running without the constrained mode and prints why.
- An Outlines `Image` needs a PIL image with `.format` set. The helper round-trips through PNG.
- Detection labels are COCO class names. Compare singular forms (`person` vs `persons`).
- Cost for a self-hosted VLM = GPU price × seconds. It is not free; it is cheap at scale only if the GPU is kept busy.

## Materials
- [Outlines — structured generation](https://dottxt-ai.github.io/outlines/) · [Efficient Guided Generation paper](https://arxiv.org/abs/2307.09702)
- [Pydantic — JSON Schema](https://docs.pydantic.dev/latest/concepts/json_schema/)
- [Ultralytics YOLO](https://docs.ultralytics.com/) · [torchvision ResNet-50 weights](https://pytorch.org/vision/stable/models.html)
