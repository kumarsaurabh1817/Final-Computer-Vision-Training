# Day 4 Notes — Image Captioning

## Data
- [Flickr8k (Kaggle: `adityajn105/flickr8k`)](https://www.kaggle.com/datasets/adityajn105/flickr8k): 150 test images (same split as Days 1–3), each with 5 human captions.
- Three charts rendered in code (bar, line, pie) with known titles and values, to test figure captioning.

## Core ideas

**Three generations of captioner:**

| Model | Architecture | Trained for | Style |
|---|---|---|---|
| BLIP-large (2022) | ViT encoder + BERT-style decoder with cross-attention; CapFilt data cleaning | captioning (and VQA/retrieval heads) | short, generic, COCO-like |
| BLIP-2 OPT-2.7B (2023) | frozen ViT-g → **Q-Former** (32 learned queries) → frozen OPT LLM | image-text alignment, then generation | short; answers `Question: … Answer:` prompts |
| Qwen3-VL-4B (2025) | ViT → patch merger → LLM, instruction-tuned | following instructions about images | any style you prompt: short, detailed, chart-aware |

**The Q-Former** solves the same problem as Day 1's projector. Instead of one token per patch, 32 learned queries cross-attend to the image features and become 32 soft-prompt tokens. The token cost is fixed regardless of resolution, so it is cheap, but fine detail and text are lost.

**Caption-as-text vs raw embedding: which index?**

| | Caption (text) index | Raw image embedding (CLIP) |
|---|---|---|
| Search with | BM25 / text embeddings | CLIP text encoder |
| Strong on | text in the image, chart titles, numbers, names; exact keywords | visual similarity, style, things nobody wrote down |
| Weak on | anything the caption omitted (it is lossy) | reading text; long or compositional queries (77-token limit) |
| Explainable / editable | yes, you can read and fix the caption | no |
| Cost | one VLM call per image at index time | one cheap encoder pass |

In practice you index **both** and fuse the scores (hybrid). That is the Week 6 design.

## Metrics
- **BLEU-4**: n-gram precision up to 4-grams with a brevity penalty. Rewards copying reference wording.
- **ROUGE-L**: F-score of the longest common subsequence.
- **CIDEr**: TF-IDF-weighted n-gram similarity to the consensus of all references. The standard captioning metric.
- **Human baseline**: one human caption scored against the other four. Models are scored against the same four, so the comparison is fair. Models can beat humans on CIDEr by producing the most average caption.
- **CLIPScore** = 2.5 × max(cos(image, caption), 0). Reference-free; checks image-text agreement instead of wording.
- **Fact recall (figures)**: share of expected facts (title words, series names, a key number) found in the caption.
- **Retrieval**: Recall@1/5/10 and MRR for text → image search.

## What to check in your run
- Detailed Qwen captions: low CIDEr but high CLIPScore and better retrieval. Can you explain why?
- Figures: which models mention the title and numbers, and which say "a bar graph with blue bars"?
- Retrieval: CLIP vs caption indexes on photos vs on figures. Does the hybrid win both?
- Latency per image for each captioner. At 100k images, which one can you afford?

## Pitfalls
- Reference metrics depend on tokenization. Use the same tokenizer for every model (the notebook lowercases and keeps alphanumerics).
- CLIP's text encoder truncates at 77 tokens, so long detailed captions are partly ignored by CLIPScore.
- BLIP-2 OPT-2.7B is ~7.5 GB in fp16. Free each model before loading the next on a T4.
- Captions inherit VLM hallucinations. An invented object in a caption becomes a false retrieval hit in Week 6.

## Materials
- [BLIP-2 (Hugging Face docs)](https://huggingface.co/docs/transformers/model_doc/blip-2) · [BLIP-2 paper](https://arxiv.org/abs/2301.12597) · [BLIP paper](https://arxiv.org/abs/2201.12086)
- [CIDEr](https://arxiv.org/abs/1411.5726) · [CLIPScore](https://arxiv.org/abs/2104.08718) · [pycocoevalcap](https://github.com/salaniz/pycocoevalcap)
