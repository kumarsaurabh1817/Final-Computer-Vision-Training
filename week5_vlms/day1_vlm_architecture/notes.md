# Day 1 Notes — How Vision-Language Models Work

## Data
- [Flickr8k (Kaggle: `adityajn105/flickr8k`)](https://www.kaggle.com/datasets/adityajn105/flickr8k): 8,091 photos, 5 human captions each, ~1 GB. Layout: `Images/*.jpg` + `captions.txt` (CSV `image,caption`).
- No official split in the Kaggle copy, so every Week 5 notebook uses the same seeded 1,000-image test split.

## Core ideas

**A VLM = vision encoder + projector + LLM.**
```
image ──► vision encoder (ViT) ──► patch features ──► projector ──► visual tokens ─┐
                                                                                    ├─► LLM ──► text
text  ──► tokenizer ──────────────────────────────► text token embeddings ─────────┘
```
- **Vision encoder**: a ViT cuts the image into patches (Qwen3-VL: 16×16 px) and outputs one feature vector per patch.
- **Projector**: maps vision features into the LLM's embedding size. LLaVA uses a 2-layer MLP. Qwen3-VL's "merger" also joins 2×2 neighbouring patches into one token, so one visual token ≈ 32×32 px. BLIP-2 uses a Q-Former with a fixed 32 tokens (Day 4).
- **LLM**: an ordinary decoder-only language model. It reads visual tokens as if they were words. It holds most of the parameters (look at the pie chart).

**Visual tokens cost context.** The chat template has an image placeholder. The processor expands it into N `<|image_pad|>` tokens, with N ≈ pixels / 1024 for Qwen3-VL. A 1-megapixel page is ~1,000 tokens before you ask anything. More pixels mean better small-text reading, but slower prefill and more memory.

**Dynamic resolution.** Qwen3-VL keeps the aspect ratio and snaps sizes to multiples of 32, with a minimum and maximum pixel budget. LLaVA-OneVision uses "AnyRes": it tiles a high-resolution image into crops plus a downscaled overview.

**Training in two stages (LLaVA recipe).**
1. *Alignment*: train only the projector on image-caption pairs. Encoder and LLM are frozen.
2. *Visual instruction tuning*: train projector + LLM on image conversations (LLaVA's were GPT-generated from captions and boxes). The loss is computed **only on assistant tokens**; image and question tokens get label `-100`.

**Hallucination.** A VLM inherits the LLM's habit of producing fluent text, including about objects that aren't in the image. POPE measures this with balanced yes/no "Is there a X?" questions. A yes-ratio well above 0.5 together with lower precision means the model is inventing objects.

## Qwen3-VL vs LLaVA-OneVision (what you measured)

| | Qwen3-VL-4B | LLaVA-OV-0.5B |
|---|---|---|
| Vision encoder | own ViT, native dynamic resolution | SigLIP-SO400M |
| Projector | patch merger (2×2 → 1 token) | 2-layer MLP |
| LLM | Qwen3 4B | Qwen2 0.5B |
| High resolution | dynamic, any aspect ratio | AnyRes tiling |
| Licence | Apache 2.0 | Apache 2.0 |

## What to check in your run
- Tokens vs resolution: is the line straight, and where does the minimum-pixel floor kick in?
- The projector hook: does the output shape equal (visual tokens, LLM hidden)?
- Latency vs visual tokens: how much does going from 448 px to 1344 px cost?
- POPE: which model has the stronger "yes" bias?
- Instruction-tuning sample: what fraction of the tokens is supervised?

## Pitfalls
- **T4 has no bf16.** The wrapper picks fp16 on T4. If outputs are empty or garbage, try `Qwen3-VL-2B-Instruct` or fp32.
- `transformers` must be ≥ 4.57 for Qwen3-VL.
- Greedy decoding (`do_sample=False`) keeps runs reproducible. Sampling makes numbers move between runs.
- The POPE labels come from captions, which miss things, so a few "absent" objects are actually present.

## Materials
- [Qwen3-VL (Hugging Face)](https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct) · [Qwen-VL repo](https://github.com/QwenLM/Qwen3-VL)
- [LLaVA project](https://llava-vl.github.io/) · [Visual Instruction Tuning paper](https://arxiv.org/abs/2304.08485) · [LLaVA-OneVision](https://arxiv.org/abs/2408.03326)
- [POPE: object hallucination evaluation](https://arxiv.org/abs/2305.10355)
