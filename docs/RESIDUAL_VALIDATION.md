# Residual validation and the layered evidence model

Phase 10 left two defects that look unrelated and are not. Sixty per cent of
documents were held for manual review with no expected synthetic PHI
surviving, and the protection for clinical language depended on a list of ten
words. Both are the same mistake in different places: a decision made without
looking at the evidence in front of it.

## The second pass was scanning text it had destroyed

The residual validator re-read the redacted output to check that nothing had
survived. Before scanning, it replaced every token the sanitizer had written
with spaces of the same length, so the validator would not re-detect its own
surrogates as PHI.

That masking rewrote the sentence it was checking:

```
Care was provided by Dr. PROVIDER_001 at FACILITY_001.
Care was provided by Dr.              at              .
```

Both models then predicted, correctly by their own lights, that a name
belonged in the hole — and the validator counted those predictions as
surviving PHI. The blocks were artifacts of the masking. Nothing had leaked;
the check had broken the evidence and then read the wreckage.

### Provenance instead of masking

`services/transformation_provenance.py` records every range the sanitizer
wrote, in **final output** coordinates, as the replacement is applied. The
validator now scans the exact serialized bytes — no masking, no distortion —
and attributes each finding against that map.

A finding is discounted only when its span lies **wholly** inside a recorded
region. Partial overlap is never enough: the uncovered part is text the
sanitizer did not write, and a missed identifier sitting immediately beside a
surrogate is precisely the case that must not be waved through.

Getting the offsets right required rebuilding the replacement loop
left-to-right. The old loop replaced from the end of the document backwards,
which keeps *input* offsets valid but makes every recorded *output* offset
wrong as soon as an earlier entity changes length.

### Seven classifications, three of which block

Every finding is recorded as six fields — detector, normalized category,
evidence type, location type, whether it overlaps a generated region, and the
classification — and nothing else. No value, no offsets, no sentence.

| Classification | Blocks | Meaning |
| --- | --- | --- |
| `genuine_surviving_phi` | yes | a deterministic identifier match in released text |
| `additional_plausible_phi` | yes | a model or context finding outside every generated region |
| `malformed_output` | yes | the sanitizer's own output is structurally unsound |
| `exact_generated_surrogate` | no | wholly inside a surrogate this pass wrote |
| `anonymizer_placeholder` | no | wholly inside a `<REDACTED_*>` token |
| `detector_artefact_modified_context` | no | a heuristic finding straddling a replacement boundary |
| `useful_clinical_content` | no | a clinical reading with no naming evidence |

`malformed_output` is the fail-closed backstop. A truncated placeholder or a
dangling surrogate stem means the replacement loop produced something the
validator cannot reason about, and reasoning about it anyway is how a
partially redacted value gets released.

## The closed list was the decision-maker

The clinical vocabulary protected known words and destroyed unseen ones. Worse,
it protected them unconditionally: a patient genuinely surnamed Wilson was
shielded by a rule written for Wilson disease. And the filter ran *inside the
detector*, so a model finding was deleted on dictionary grounds before any
context could be weighed — which is why `Blood Pressure Diagnostics Ltd`
vanished from the findings entirely. It contains a vital sign.

`services/detection_evidence.py` replaces membership with five layers:

1. **Exact structured PHI always wins.** A regex match on an SSN, an email or
   an MRN is never overridden.
2. **Multiple agreeing detectors normally win.** Agreement means two detectors
   reading the same span the *same way* — not bare overlap. A structured hit
   on `MRN: 123456` overlaps a model calling the label `MRN` an organisation;
   counting that as corroboration escalated every field label to review.
3. **A clinical reading blocks a weak, heuristic-only redaction.** This is
   where the vocabulary is consulted — as one source among several.
4. **Naming context beats the dictionary.** `Dr Parkinson` is a person even
   though Parkinson is a recorded eponym. An organisational designator (`Ltd`,
   `Diagnostics`, `Hospital`) is naming evidence in the same way.
5. **Unexplained name-shaped spans fail closed.** They are replaced with a
   surrogate rather than left in place pending review: the proper-noun
   fallback exists to catch names the models miss, and leaving one in released
   text to await a human would be a privacy regression.

Two grammatical signals do work the list used to do, and they generalise to
terms nobody has recorded:

- `<Word>'s <head noun>` reads as a condition — `Verrando syndrome` survives.
- `<Word> <number> <unit>` reads as prescribing — `Ranolazatib 250 mg`
  survives.

### The case-artifact probe

English capitalises the first word of every sentence and of most document
labels, so a tagger seeing `Care was provided by ...` or a PDF title `Chart
for ...` has no case evidence and frequently guesses PROPN. The proper-noun
fallback then proposed ordinary nouns as names.

Rather than exempting those words by listing them — which would only ever
protect the words somebody thought of — the detector asks the tagger a second
question: with the capitalisation removed, is it still a proper noun?
`care` and `chart` fall back to NOUN. `kartik` and `jordan` stay PROPN,
because their proper-noun reading comes from the word itself. An unseen
surname is still caught.

## What changed, measured

Manual-review rate on the calibration partition fell from 0.60 to 0.195 with
no relaxation of any privacy assertion: deterministic findings still always
block, and every previously blocked pattern has a regression test in
`tests/test_residual_validator.py`.

`evaluations/residual_diagnostics.py` opens every remaining block and reports
the six fields per finding. It reads the pipeline's own verdict rather than
re-scanning, because a re-scan without the provenance map reproduces the
original defect exactly — which it did, on first writing, and reported 52.5%
until it was corrected.

## What this is not

These are project engineering targets, not legal HIPAA thresholds. A low
manual-review rate on a synthetic corpus is evidence that the pipeline stopped
blocking documents for no reason. It is not evidence of compliance, and zero
residual canaries on invented data is not evidence of zero real-world leakage.
