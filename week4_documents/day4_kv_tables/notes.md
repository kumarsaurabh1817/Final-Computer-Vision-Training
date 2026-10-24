# Day 4 Notes — KV & Table Deep Dive + One Common Schema

## Data
The same five invoices as Day 3 (same seed), so cached Azure/Textract responses are reused at no cost. Day 4 reads them from the Day 3 output folder of the same Colab session (`/content/week4_outputs/day3/raw`); upload that folder if you start a new session.

## Hard cases

| Case | What breaks | Fix |
|---|---|---|
| Rotated scan | OCR lines turn diagonal; rows mix | Projection-profile deskew before OCR |
| Different layout | New labels ("Bill No.", "Amount Due"); borderless table | Label synonyms; Table Transformer handles borderless tables |
| Merged cells | Two-level header, section rows spanning the table | Table Transformer spanning cells → `row_span` / `col_span` |
| Multi-page table | Table split, header repeated | `stitch_tables`: same column count + repeated header → one table |

## The common JSON schema (`InvoiceDoc`, pydantic)
- `fields`: `invoice_number`, `invoice_date` / `due_date` (ISO `YYYY-MM-DD`), `vendor_name`, `bill_to`, `subtotal` / `tax` / `total` (float), `currency` (ISO 4217 code).
- `line_items`: description, quantity, unit_price, amount.
- `tables`: explicit cells with `row`, `col`, `row_span`, `col_span`, `text`, `is_header`. A validator rejects cells outside the grid.
- `meta`: source, pages, latency, cost (extra keys allowed).
- Exported as JSON Schema to `invoice_schema.json`. Every approach's output is round-tripped through it, so "valid JSON" is checked, not assumed.

## Self-hosted pipeline
1. Deskew → Tesseract words → lines (cluster by y) → **phrases** (split at large x-gaps, so "Invoice Number:" and its value are separate).
2. **KV rules:** label regexes anchored at phrase start; value = rest of the phrase, else the next phrase on the line, else the phrase below. Header fields keep the first match, totals keep the last. `Tax (8%)` → the percent is stripped so 8 is not read as the amount.
3. **Tables:** Table Transformer detection → crop → structure recognition (rows, columns, header, spanning cells) → cells = row ∩ column → OCR words to cells. Fallback: keyword region + geometric rules.
4. Tables stitched across pages; line items read from the table whose header matches description/qty/price/amount synonyms.

## Value normalization
- Money: `$1,234.50`, `1.234,50 €` → 1234.5 (the last separator is the decimal point).
- Dates: any format → ISO; `03/04/2026` is ambiguous (US month-first by default). Make `dayfirst` a per-source setting.
- Currency: symbol or ISO code → `USD`/`EUR`/`GBP`.

## Pitfalls
- Rules are brittle: they only know the labels you listed. Day 5's unseen layout measures that.
- Table Transformer was trained mostly on papers and financial reports; check its detections on your documents.
- PIL `crop` outside the image pads with black: clamp the crop box.

## Materials
- [Table Transformer (HF)](https://huggingface.co/docs/transformers/model_doc/table-transformer) · [PubTables-1M paper](https://arxiv.org/abs/2110.00061)
- [pydantic JSON Schema](https://docs.pydantic.dev/latest/concepts/json_schema/)
