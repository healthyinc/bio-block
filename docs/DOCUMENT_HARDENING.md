# TXT and PDF hardening

## TXT

The text path already rejected oversized uploads rather than truncating them.
Two gaps are now closed:

- **Undecodable and NUL-bearing input.** A NUL can split a value apart so the
  detector never matches it; such an upload is rejected rather than scanned and
  reported clean.
- **Post-redaction validation.** `residual_phi_categories` re-runs the typed
  detector over the redacted output before a release is issued. Our own
  replacement tokens (`<REDACTED_*>`) and research-profile surrogates are
  masked out first — offsets preserved — so the validator measures what
  survived redaction rather than re-flagging the redaction itself.

A text release now carries three validators: `typed_phi_detection`,
`deterministic_redaction`, and `residual_phi_rescan`. If the rescan finds
anything, the release is refused with reason code
`privacy_requirements_not_met` plus the surviving categories. Counts and
category names only — never a surviving value.

## PDF

A PDF carries PHI on many more surfaces than the visible text layer. Scanning
only `page.get_text()` and reporting zero entities is a fail-open answer,
because an image-only page is indistinguishable from a clean one.

`services/document_sanitization.py` inventories these surfaces:

| Surface | Handling |
|---|---|
| Page text layer | Extracted and scanned |
| Document info metadata (title, author, subject, keywords, creator, producer) | Extracted and scanned |
| XMP metadata | Tags stripped, remaining text scanned |
| Annotations (contents, title, subject) | Extracted and scanned |
| Form-field widgets (name and value) | Extracted and scanned |
| Link targets (URI and file) | Extracted and scanned |
| Bookmarks / outline titles | Extracted and scanned |
| Raster images | **Not** scanned - marked `pdf_raster_requires_ocr` |
| Embedded attachments | **Not** scanned - marked `pdf_embedded_files_unscannable` |
| JavaScript, Launch, RichMedia, Movie markers | **Not** scanned - marked `pdf_active_content_present` |

Short markers such as `/JS` and `/AA` are deliberately not treated as
active-content indicators: they collide with ordinary bytes inside compressed
streams often enough to be meaningless, and treating them as signals would
block nearly every document.

### Two independent conclusions

- `text_layer_complete` - every text surface was read and scanned, and the
  redacted result carried no residual PHI. Redacted page text is returned only
  when this holds; otherwise it is withheld entirely, so a partial scan cannot
  be mistaken for a complete one.
- `scannable` - the above, and no unreadable surface exists.

### Release posture

A PDF is **never** automatically releasable. There is no validated PDF writer
that can produce sanitized bytes, so the best available outcome is
`manual_review_required` with reason `pdf_validated_writer_unavailable`; every
other condition yields `unsupported_or_unscannable`. Original bytes are never
returned, and the scan happens entirely in memory — PHI is no longer written to
a temporary file on disk, as the previous implementation did.

### Limits

| Limit | Value |
|---|---|
| `MAX_PDF_BYTES` | 32 MiB |
| `MAX_PDF_PAGES` | 500 |
| `MAX_SURFACE_TEXT_BYTES` | 128 KiB per surface |

Exceeding any of these blocks rather than truncating.

### Error hygiene

`/anonymize_pdf` returns a fixed `"Failed to scan PDF"` message on unexpected
errors. The previous implementation interpolated `str(e)`, which could echo
document content into an HTTP response.
