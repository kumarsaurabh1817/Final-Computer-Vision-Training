# Day 4 Notes — Text + Image Agentic Queries

## Data
- Query images rendered from the true chart data in a different style (colours, size, title), so the right answer is known.
- **Compare** images have altered data: the largest and smallest pie slices are swapped.

## Core ideas

**The image-led path:**
```
user image ──CLIP──► similar figures (image space) ──► multi-image VQA: [user image, candidate figure]
                                │                                │
                                └── matched document ──► table / text steps ──► synthesize with citations
```

**Three query families:**

| Family | Hops | What makes it hard |
|---|---|---|
| identify | image → figure → document → its largest segment | CLIP matches the *shape*; the company name comes from metadata |
| compare | image → figure → look at both → describe differences | requires seeing two images at once (multi-image VQA) |
| cross-reference | image → figure → document → table total | crosses modality twice (image → figure → table) |

**Multi-image VQA:** the VLM gets the user's image and the retrieved figure in one prompt, with names for each image ("image 1 is the USER'S image; image 2 is …"). Without the names, models mix them up.

**Image-only vs composed retrieval:** the question text ("a chart like this") adds little. Composing image + text can dilute the image signal, so measure both.

## Metrics
- Per family: the company named (doc found), all expected terms (segment names for compare), the number within 1 % (cross-reference), evidence recall.
- Baseline vs agent per family.

## What to check in your run
- Which family does the baseline already solve? (Identify, often: one image search + the answer.)
- Compare: does the agent name the two swapped segments, or just say "similar"?
- Cross-reference: does the agent go on to the table after matching the figure?

## Pitfalls
- Similar charts (two pie charts with similar slices) confuse CLIP. Check the top-3, not just the top-1.
- The VLM may describe the user's image correctly but compare it with the wrong retrieved figure. Cite which figure was compared.
- Do not let the model "read" the company name off a chart that does not contain it: the name must come from the retrieved item's metadata.

## Materials
- [Qwen3-VL — multi-image input](https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct)
- Week 6 Day 4 (image queries), Week 5 Day 5 (composed CLIP queries)
