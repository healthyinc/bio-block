# Utility-preserving anonymization

"Remove identity, preserve medical meaning, and block unsupported cases."

Phase 9 achieved the first of those and failed the second: zero expected
leakage with useful-text preservation of 0.214, and nine of ten documents sent
to manual review. A file with every clinical word removed passes every privacy
check ever written and is worthless.

## Root cause of the 0.214

Three distinct defects, all in the detection layer rather than in either
pinned model:

1. **The possessive broke the eponym guard.** The code asked whether the token
   after an eponym was `disease`. In the standard form *Parkinson's disease*
   that token is `'s`, so every possessive eponym failed the very test that
   existed to protect it. Parkinson, Alzheimer, Crohn and Paget were redacted
   as person names.
2. **The eponym list held five terms and was consulted for `PERSON` only.**
   spaCy tags Addison, Bell, Cushing, Wilson and Graves as `ORG`, which never
   reached the check at all.
3. **The models were never filtered.** Stanford and GLiNER label "Parkinson" a
   PERSON as readily as spaCy does, and nothing filtered their output, so
   clinical terms were destroyed by a second route even after the first was
   fixed.

Two further defects surfaced while fixing those: `"Blood pressure"` was
labelled an ORGANIZATION, and `"Atorvastatin 40 mg nightly"` was labelled a
PERSON, because the vocabulary check compared whole multi-word spans against a
single-term list.

## The vocabulary

`services/clinical_vocabulary.py` is the single source of truth for "this word
is medicine, not a person", consulted by **every** detector.

It is a closed vocabulary, never a pattern that could admit an arbitrary name.
An eponym earns protection only **in a clinical construction**:

| Text | Protected | Why |
|---|---|---|
| `Wilson's disease was excluded` | yes | eponym + head noun |
| `Wilson attended the clinic` | **no** | a patient really can be called Wilson |
| `Bell's palsy resolved` | yes | eponym + head noun |
| `Ollivander's disease` | **no** | not a recorded eponym |
| `CT of the abdomen` | yes | a modality is never a person |
| `Atorvastatin 40 mg` | yes | contains a distinctive drug token |

The lookahead skips the possessive marker, which is what the original check
got wrong.

## Surrogates

Identifiers are replaced with consistent study-local surrogates so coreference
survives:

```
Patient PATIENT_001 (RECORD_001) was seen by Dr. PROVIDER_001 at FACILITY_001.
PATIENT_001 is 90+ years old.
Diagnosis: Parkinson's disease with Alzheimer's dementia.
Metformin 500 mg twice daily. HbA1c 7.2 percent. Creatinine 1.1 mg/dL.
```

Three constraints, each deliberate:

- **Study-local.** One allocator per upload bundle, discarded with the
  request. Two studies containing the same person produce unrelated
  surrogates, so no cross-study linkage is created.
- **Not derived from the value.** Assignment is by order of first appearance,
  never computed from the text. A hash - salted or not - is a function of the
  identifier, so an attacker with a candidate list can confirm a guess by
  recomputing it. Order of appearance carries no such information.
- **Never persisted.** The mapping lives in memory for one request.
  `SurrogateAllocator` exposes counts and nothing else; there is no accessor
  that returns the originals, and the transformation manifest has no field for
  them.

Patient and provider are distinguished by deterministic textual evidence only:
a title actually present (`Dr.`, `Professor`, `attending`) or an explicit
"seen by" / "referred by" construction. Absent that evidence the person is
treated as the patient.

## Ages

Safe Harbor requires ages over 89 to be aggregated, not removed.

| Input | Output |
|---|---|
| `94 years old` | `90+ years old` |
| `aged 97` | `aged 90+` |
| `89 years old` | unchanged - not an identifier |
| `received 94 mg` | unchanged - no age cue |
| `a nonagenarian` | removed, and the document goes to review |

A bare number is never treated as an age. An age reference that cannot be
resolved to a number cannot be judged against the threshold, so it escalates
rather than being guessed at in either direction.

## Release rule

Three-valued, and the order matters:

1. **Privacy cannot be established** - block. Never overridden by utility.
2. **Privacy passes, utility fails** - `manual_review_required` with
   `utility_validation_failed`. A technically safe but medically useless file
   is not reported as a successful research artifact.
3. **Both pass** - release only if the policy permits it.

An unmeasured contract metric counts as a failure, not a pass: an unmeasured
term is indistinguishable from an unmet one, and treating it as satisfied is
how a utility gate quietly stops gating anything.

## Results

| Metric | Phase 9 held-out | Phase 10 held-out |
|---|---|---|
| PHI recall | 1.0000 | **1.0000** |
| False negatives | 0 | **0** |
| Residual canaries | 0 | **0** |
| PHI precision | 0.5315 | **0.7284** |
| Useful-text preservation | 0.2143 | **0.9286** |
| Clinical-term preservation | not measured | **1.0000** |
| Manual-review rate | 0.90 | **0.60** |

Full report: [reports/PHASE10_EVALUATION.md](reports/PHASE10_EVALUATION.md).

## Safe Harbor and high-utility research stay separate

`safe_harbor_v1` remains the only automatically releasable policy. Under it,
dates are handled at year level, ages above 89 become `90+`, and required
identifiers are removed or replaced. Consistent date shifting is **not**
claimed to satisfy Safe Harbor.

A high-utility research transformation that shifts dates consistently,
preserves intervals and ordering, and retains more granular quasi-identifiers
must always return `expert_determination_required`. It is never downloadable,
previewable, indexed, uploaded to IPFS or submitted to blockchain without an
external approved release decision. It is not called HIPAA compliant.
