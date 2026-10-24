# Day 3 Notes — Managed Services (Azure Document Intelligence & AWS Textract)

## Data
Five synthetic invoices rendered in the notebook, one per hard case (`standard`, `layout_b`, `merged`, `multipage`, `rotated`). They are saved as PNG + PDF + `truth.json` under `day3/docs/` so you can also try them in the web consoles.

## Setup
| Provider | Create | Secrets (Colab key icon or env vars) |
|---|---|---|
| Azure | Portal → Document Intelligence → tier **F0** (free, 500 pages/month) | `AZURE_DI_ENDPOINT`, `AZURE_DI_KEY` |
| AWS | IAM user with `AmazonTextractFullAccess` → access key | `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_DEFAULT_REGION` |
| Google (optional) | Document AI processor (Invoice / Form Parser) | `GCP_PROJECT_ID`, `GCP_LOCATION`, `GCP_PROCESSOR_ID` |

Without credentials every cell still runs: parsers are tested on mock responses and API cells skip.

## Core ideas

**Azure AI Document Intelligence** — returns a *document model*.
- `prebuilt-read` (OCR) → `prebuilt-layout` (+ tables, KV with the `keyValuePairs` feature) → `prebuilt-invoice` / `prebuilt-receipt` (typed fields like `InvoiceTotal.valueCurrency.amount`, `Items[]`).
- Custom models: label ~5+ samples in Studio; *template* for fixed layouts, *neural* for varied ones.
- Tables come with explicit `rowSpan`, `columnSpan`, `kind="columnHeader"`. Multi-page PDF in one call (long-running operation → poller).

**AWS Textract** — returns a flat *block graph*.
- `PAGE → LINE → WORD`; `KEY_VALUE_SET` (KEY → VALUE relationship); `TABLE → CELL → WORD`; `MERGED_CELL` groups the cells it covers.
- `AnalyzeExpense` gives normalized summary fields (`TOTAL`, `INVOICE_RECEIPT_ID`, …) + line items.
- Synchronous APIs take one page; multi-page PDFs need the async `Start…`/`Get…` APIs via S3.

**Cost control (built into the helpers).**
1. A `CostMeter` refuses a call *before* it would exceed the page or USD cap.
2. Every raw response is cached as JSON; re-runs cost nothing; Days 4–5 reuse the cache.
3. A failed call is refunded in the meter (rejected requests are not billed).

## What to check in your run
- Fields right per layout, per provider. Which layout hurt each one?
- Did Textract emit a `MERGED_CELL` for "Pricing"? Did Azure report `columnSpan=2`?
- Latency per document and total spend (`day3_summary.json`).

## Pitfalls
- Azure polygons are in **inches** for PDF input and pixels for images: scale by page width/height before drawing.
- Textract `RowIndex`/`ColumnIndex` are 1-based; the schema is 0-based.
- Prices change and differ by region: check the pricing pages; the constants in the helper are list prices for orientation.
- **Privacy:** every page leaves your network. Check data residency rules, PII and contracts before sending real documents. Azure offers containers for on-prem use.

## Materials
- [Azure AI Document Intelligence](https://learn.microsoft.com/azure/ai-services/document-intelligence/)
- [AWS Textract](https://docs.aws.amazon.com/textract/) · [AnalyzeExpense](https://docs.aws.amazon.com/textract/latest/dg/analyzing-document-expense.html)
- [Google Document AI](https://cloud.google.com/document-ai/docs) (optional)
