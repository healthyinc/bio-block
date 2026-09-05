# Privacy and utility evaluation, version 2

Two separate groups of metrics, reported separately because they answer
different questions. A good privacy number does not excuse a bad utility
number, and the release rule enforces that ordering: privacy failure blocks,
utility failure sends to manual review, and only both passing can release.

- Corpus version: `canary-v3.0` (four partitions, disjoint value pools)
- Contract version: `utility-contract-v1`
- Model mode: `offline` - real pinned weights, loaded offline
- Configuration frozen before the held-out partition was run once.

## Privacy

| Metric | Phase 9 held-out | Phase 10 calibration | Phase 10 held-out |
|---|---|---|---|
| PHI recall | 1.0000 | 1.0000 | **1.0000** |
| PHI precision | 0.5315 | 0.7468 | **0.7284** |
| F1 | 0.6941 | 0.8551 | **0.8429** |
| False negatives | 0 | 0 | **0** |
| False positives | 52 | 20 | **22** |
| Document leakage | 0.00 | 0.00 | **0.00** |
| Residual canaries | 0 | 0 | **0** |
| Manual-review rate | 0.90 | 0.60 | **0.60** |
| Missed categories | none | none | **none** |

## Utility

| Metric | Phase 9 held-out | Phase 10 calibration | Phase 10 held-out |
|---|---|---|---|
| Useful-text preservation | 0.2143 | 0.9286 | **0.9286** |
| Clinical-term preservation | not measured | 1.0000 | **1.0000** |
| Content-token preservation | not measured | 0.9625 | **0.9596** |
| Numeric preservation | not measured | 1.0000 | **1.0000** |
| Non-PHI terms destroyed | 11 of 14 | 1 of 14 | **1 of 14** |

Privacy did not move: recall stays at 1.0000 with zero false negatives, zero
document leakage and zero surviving canaries on a partition that was untouched
until the configuration was frozen. What changed is that the output is now
usable.

## Release decisions

| Partition | blocked_privacy | releasable |
|---|---|---|
| calibration | 6 | 4 |
| held-out | 6 | 4 |

The remaining manual review is the residual validator, not a detection
failure: zero gold values actually survive in any document. Six of ten
documents still carry a residual finding on the masked text. That is
fail-closed and safe, and it is the next thing to reduce.

## Transformation manifests

Every evaluated artifact carries one, recording modality, policy version,
model versions, categories transformed, transformation types, counts of
removed/replaced/generalized fields, regions modified, both check groups and
the release decision. Manifests carry counts and categories only; there is
deliberately no field for the surrogate mapping.

## Claim boundary

Synthetic corpus, ten documents per partition, invented values throughout.
Zero surviving canaries is an acceptance condition for this suite. It is
**not** proof of zero real-world PHI leakage and **not** a statement about
HIPAA compliance.
