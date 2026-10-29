# Day 5 Notes — Multimodal Embeddings (CLIP)

## Data
- [Flickr8k (Kaggle: `adityajn105/flickr8k`)](https://www.kaggle.com/datasets/adityajn105/flickr8k): 1,000 test images × 5 captions = 5,000 text queries (the standard "1K test" protocol size).
- CIFAR-10 test set: 2,000 images, downloaded automatically by torchvision, for zero-shot classification.

## Core ideas

**Contrastive learning builds the shared space.** CLIP trains an image encoder and a text encoder on 400M (image, caption) pairs. In each batch of N pairs, it computes the N×N cosine-similarity matrix (scaled by a learned temperature) and applies cross-entropy over rows *and* columns (symmetric InfoNCE). Matched pairs are pulled together and all others pushed apart. Afterwards, "close in vector space" means "about the same thing", in either modality.

**SigLIP** replaces the batch softmax with an independent sigmoid per pair (is this pair matched, yes or no?). It needs no all-to-all normalization, works with smaller batches, and is often stronger at the same size.

**Retrieval directions:**
- text → image: embed the query with the text encoder, search the image vectors.
- image → text: embed the image, search the caption/text vectors.
- image → image: same encoder on both sides. Finds near-duplicates and similar scenes.
- **text + image**: `normalize(img_vec + α · text_vec)`, e.g. "this picture, but in the snow". α trades image similarity for the text modification.

**The modality gap.** Image vectors and text vectors occupy two separate regions of the sphere. A *matched* image-caption pair has cosine ≈ 0.3, lower than two random images. So ranking works, but absolute thresholds ("only answer if similarity > 0.8") do not transfer from text-only RAG. Tune thresholds per model and per modality pair.

**Zero-shot classification** is retrieval over class prompts: embed "a photo of a {class}." for each class and assign each image to the nearest prompt. Prompt wording matters, and averaging many templates (prompt ensembling) adds a few points.

**Caption-as-text vs raw embedding vs hybrid** (repeated from Day 4 at full gallery size): captions are lossy but readable, embeddings keep visual detail but cannot read well, and fusion (reciprocal-rank fusion) is the robust default.

**FAISS.** `IndexFlatIP` on L2-normalized vectors gives exact cosine search, fine up to ~100k vectors. Beyond that, use HNSW (graph) or IVF-PQ (clustered and compressed). Always store metadata (id, path, caption, source) next to the vector so every hit has provenance.

## Metrics
- **Recall@K**: the correct item is in the top K. For image→text, a hit is any of the 5 captions.
- **Median rank (MedR)**: lower is better.
- **Zero-shot accuracy** on CIFAR-10, with one prompt vs an 8-template ensemble.
- **Cost**: ms per image/text and storage per 1M vectors (dim × 4 bytes for fp32).

## What to check in your run
- Which model is best for t2i, and what does it cost in latency and dimension? (ViT-L/14 is better and ~3–4× slower.)
- Matched vs unmatched cosine histograms: where would you put a threshold?
- The PCA plot: are images and captions two separate clouds?
- Composed queries: at which α does the "snow" modification show up without losing the dog?
- Zero-shot CIFAR-10 vs your Week 1 CNN (≥70% after training on 50k labeled images).
- The success gate: a brand-new image is captioned, embedded, added and found by a text query.

## Pitfalls
- Always L2-normalize before inner-product search, or FAISS's `IP` is not cosine.
- Never mix vectors from different models in one index. The saved index records its model name and refuses a mismatch.
- CIFAR images are 32×32 and upscaled to 224, so CLIP sees blurry inputs. That's what the "low resolution photo" templates are for.
- The Kaggle Flickr8k test split is not the Karpathy split, so numbers are close to published 1K-test results but not identical.

## Materials
- [OpenAI CLIP](https://github.com/openai/CLIP) · [CLIP paper](https://arxiv.org/abs/2103.00020) · [SigLIP paper](https://arxiv.org/abs/2303.15343)
- [OpenCLIP](https://github.com/mlfoundations/open_clip) · [Sentence-Transformers — Image Search](https://www.sbert.net/examples/applications/image-search/README.html)
- [Modality gap paper](https://arxiv.org/abs/2203.02053) · [FAISS wiki](https://github.com/facebookresearch/faiss/wiki)
