# Day 5 Notes — Structured Evaluation & Comparison Matrix

## Data
A held-out set rendered with a new seed: `N_PER_LAYOUT` invoices for each of the five Day 3–4 layouts, plus **`layout_c`**, a layout whose labels ("Document No:", "Issued:", "Balance:", GBP) none of the rules were written for. Managed services score the first `MANAGED_PER_LAYOUT` documents per layout (cost cap).

## Metrics

| Metric | Definition | Why |
|---|---|---|
| Field accuracy | Exact match per field after normalization (money ±0.01) | The number a business user cares about |
| Field ANLS | Mean normalized Levenshtein similarity, 0 below 0.5 | Partial credit for near-misses |
| Line-item F1 | Items matched on amount + fuzzy description | Tables matter only if the rows come out right |
| **TEDS** | 1 − tree edit distance / max nodes, tables as HTML trees | Structure *and* content in one number |
| TEDS-S | TEDS ignoring cell text | Separates structure errors from OCR errors |
| Adjacency F1 | Pairs of horizontally/vertically adjacent cells (ICDAR 2013) | Robust to one inserted row shifting everything |
| Latency / cost per page | Wall-clock; list price or GPU $/h × time | Operating cost |

**TEDS in practice:** a missing row costs its whole subtree; a merged cell split in two costs a rename + an insertion; a misread cell costs only its text edit distance. **GriTS** (from the Table Transformer authors) is an alternative that scores the grid directly as 2D matrices.

## Building the recommendation
- Compare on the **same documents** (the shared subset) as well as overall.
- Report the unseen layout separately: that is where rules and trained models differ.
- Break-even volume: an always-on GPU costs `GPU $/h × 730` per month; divide by the managed price per page to find the volume where self-hosting becomes cheaper.
- Add the non-numeric columns: setup effort, data privacy / on-prem, cost of supporting a new layout, merged-cell support.

## What to check in your run
- Does the ranking change between seen layouts and `layout_c`?
- Is table structure (TEDS-S) the differentiator, or key-value generalization?
- Look at the worst-TEDS example per approach before trusting the averages.

## Pitfalls
- A few dozen synthetic documents give wide confidence intervals; say so.
- Self-hosted rules were written while looking at Days 3–4 layouts; only `layout_c` is truly unseen.
- A crashed or invalid output must count as a failure (score 0), not be silently dropped.

## Materials
- [TEDS (PubTabNet paper)](https://arxiv.org/abs/1911.10683) · [GriTS](https://arxiv.org/abs/2203.12555)
- [APTED tree edit distance](https://pypi.org/project/apted/)
