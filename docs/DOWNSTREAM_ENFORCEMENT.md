# Downstream enforcement

Two paths let raw uploaded content out of the pipeline without ever touching
the sanitizer. Both are now gated by `services/release_gate.py`.

## The index bypass

`/store` and `/store_enhanced` accepted a client-supplied `extracted_content`
field and wrote it straight into the ChromaDB vector store, which `/search`,
`/search_enhanced`, `/filter`, and `/documents/{id}` read back. `summary` and
`dataset_title` went in the same way.

Nothing checked any of it. A caller whose upload `/api/v1/ingest` blocked could
post the same text to `/store` and have it become searchable — the sanitizer
was simply not in that path. Rule 2 names *indexable* bytes explicitly, so this
was the largest remaining hole.

`sanitize_for_index` now runs before anything is written:

- `dataset_title`, `summary`, and `extracted_content` are redacted through the
  typed pipeline and re-scanned with `residual_phi_categories`. Anything
  surviving blocks the write with 422.
- The **redacted** text is what gets indexed and chunked. The raw text never
  reaches the store.
- Metadata string values are **scanned but never rewritten** — the platform
  filters on them, so a silent rewrite would corrupt them. PHI found in a
  metadata value blocks the write.
- Structural keys (`owner_address`, `cid`, `file_type`, chunk bookkeeping,
  timestamps) are excluded from the scan. A wallet address is not PHI, and
  blocking on one would break filtering without protecting anything.
- `research` is never indexable: the gate returns
  `expert_determination_required` and nothing is written.

`PUT /documents/{id}` writes into the same index and goes through the same
gate, re-sanitizing only the caller-supplied fields.

Store responses carry a `sanitization` block with the status, detected
categories and counts, blocked field names, and reason codes — never a value.

## The preview bypass

`ImagePreviewGenerator` documented its own behaviour as *"Returns images
directly without modification (bypass behavior)"*. `/simple_preview` streamed
the uploaded bytes straight back. `/preview_dicom` decoded the raw pixel array
and rendered it to PNG with any burned-in text intact.

`sanitized_preview` now produces preview bytes only from verified sanitized
pixels:

| Modality | Handling |
|---|---|
| JPEG / PNG | Verified redaction path (fill every detected text region, structural + residual-OCR verification, metadata stripped) |
| DICOM | Metadata scrub → pixel redaction → final-byte validation → render → residual OCR scan on the rendered PNG |
| NIfTI | **Blocked** — `facial_reconstruction_not_mitigated`. A slice through a head volume is itself a comparable image, and no defacing step exists. |
| WSI | **Blocked** — `validated_writer_unavailable`. The slide label is exactly where identifiers live. |
| Anything else | **Blocked** — `preview_modality_not_sanitizable` |
| `research` profile | **Blocked** — `expert_determination_required` |

A blocked preview answers 422 with a JSON summary; it never falls back to
returning the input. For DICOM specifically, if `pixel_validation_status` is
not `verified`, nothing is rendered — rendering the input there is exactly the
bypass this gate exists to close.

The generators in `services/preview/` are no longer wired to any endpoint. They
are left in place with a module-level warning rather than silently retained as
a callable raw-bytes path.

## Error hygiene

`/store`, `/store_enhanced`, and both preview endpoints previously interpolated
`str(e)` into their 500 responses, which could echo stored or uploaded content.
They now return fixed messages.
