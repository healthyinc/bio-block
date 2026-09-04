# CSV and workbook routing

## The policy gate on `/anonymize_csv`

`/anonymize_csv` streamed a downloadable anonymized CSV with **no privacy
profile parameter at all**. Every other release path resolves a policy first,
so this route was the one place a caller could obtain rows without one — and a
research-intent caller could simply use it instead of `/api/v1/ingest`.

It now takes a `profile` form field (default `strict`) and routes through
`resolve_privacy_policy` like everything else:

| Profile | Result |
|---|---|
| `strict` / `safe_harbor_v1` | CSV streamed, with a release decision and artifact digest |
| `research` | 200 JSON, `expert_determination_required`, no rows |
| anything else | 400 |

A run whose `anonymization_status` is `failed_privacy_validation` now answers
422 with a blocked summary instead of streaming the rows. Previously those rows
were streamed regardless.

Response headers gained `X-BioBlock-Serialized-Output-Validation`,
`X-BioBlock-Artifact-Sha256`, and `X-BioBlock-Privacy-Policy`.

## Serialized-output validation

The tabular pipeline reported which columns it removed and how many rows it
kept, but nothing ever re-read the CSV it serialized. A column that survived
serialization, or a removed identifier value reappearing inside a retained
free-text cell, would have been reported as clean.

`services/tabular_validation.py` re-parses the emitted CSV and checks it
against both the plan and the original input:

| Check | Failure code |
|---|---|
| Output parses as CSV | `serialized_output_unparseable` |
| Header equals the retained-column plan | `serialized_header_does_not_match_plan` |
| Row count equals `rows_out` | `serialized_row_count_does_not_match_plan` |
| No removed column name in the header | `removed_column_present_in_output` |
| No removed identifier value anywhere in the output | `removed_identifier_value_present_in_output` |

The value check compares distinct values from the removed identifier columns
against the whole serialized output, so an identifier echoed into a kept column
is caught. Values shorter than 5 characters are excluded — `F`, `31`, and `NY`
would otherwise match half a dataset — and the candidate set is capped at 5000
so the comparison cannot go quadratic on a wide input.

`anonymize_tabular_csv` now always serializes, so the bytes that get validated
are the bytes a download would produce. A validation failure downgrades
`anonymization_status` to `failed_privacy_validation`.

Failure reports carry column names and counts. Column names are schema, not
patient values; no leaked value is ever included.

### What did **not** change

The `/api/v1/ingest` release posture for CSV is **unchanged**: still
`manual_review_required`. The old reason code `serialized_output_validation_pending`
is now split into `serialized_output_validation_passed` plus
`tabular_release_policy_review_pending`, so the completed work is visible while
the release decision stays blocked.

Whether generalized quasi-identifiers clear the release bar is a
re-identification-risk judgment rather than a Safe Harbor determination, and
that policy decision has not been made. Flipping CSV to releasable is a
deliberate call for review, not something this work assumed.

## Workbooks (.xlsx / .xlsm)

An .xlsx upload was previously rejected as an unsupported modality. That is
safe but uninformative, and it pushes callers into converting workbooks to CSV
by hand — which discards every surface a CSV cannot carry.

`services/workbook_sanitization.py` inventories and scans:

| Surface | Handling |
|---|---|
| Every cell of every sheet, hidden sheets included | Scanned, batched under the text limit |
| Sheet names | Scanned |
| Cell comments and notes | Scanned |
| Defined names | Scanned |
| Document properties (creator, title, subject, keywords, lastModifiedBy, …) | Scanned |
| Formulas | Kept visible (`data_only=False`) and scanned — a formula string can carry an identifier or a path to one |
| Macros (`vbaProject.bin`, macrosheets) | **Not** scanned — `workbook_macros_present` |
| Embedded objects and media | **Not** scanned — `workbook_embedded_objects_present` |
| External links | **Not** scanned — `workbook_external_links_present` |

Hidden sheets are read and scanned rather than skipped, and reported separately
because a user may not know they are there.

A workbook is **never** releasable — there is no validated workbook writer — so
the best outcome is `manual_review_required` with the inventory attached.
Original bytes are never returned.

| Limit | Value |
|---|---|
| `MAX_WORKBOOK_BYTES` | 32 MiB |
| `MAX_WORKBOOK_SHEETS` | 100 |
| `MAX_WORKBOOK_CELLS` | 200,000 |

Legacy `.xls` (compound binary, not OOXML) is rejected rather than
half-supported.
