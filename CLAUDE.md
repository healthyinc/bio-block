# CLAUDE.md — Bio-Block AI Multimodal PHI Hardening

Repo-level operating constraints for this worktree
(`healthy-ai-multimodal-phi-hardening`, branch `gsoc-ai-multimodal-phi-hardening`).

## Working context

- Work only in this worktree. Do not operate on `healthy`, `healthy-text-ner-pr`,
  `healthy-week3-pr`, or `week4-pr-fix`.
- Week 8 commit `bba968b` is deliberately excluded from this branch's ancestry.
- Week 7 / t-closeness is on mentor hold. Do not implement it.
- Tests: `py -3.11 -m pytest tests -q` from `python_backend/`.

## NON-NEGOTIABLE RULES

1. Week 6 Safe Harbor logic is authoritative for privacy policy, transformation,
   validation, and API compatibility. AI models (Stanford deidentifier, GLiNER) are
   detectors only — they propose PHI spans, they never make the release decision.
2. safe_harbor_v1 is the only policy that can produce a releasable/downloadable
   artifact. strict is a compatibility alias of it. research always returns
   expert_determination_required and must never expose downloadable/previewable/
   indexable bytes through any public endpoint, even after Phase 3 changes.
3. No unsalted hashes, ever, as a releasable Safe Harbor transformation. No
   identifier-derived hashing may satisfy Safe Harbor even if it happens to pass a
   test — if you find a test asserting hashed output as "safe", fix the test to
   assert the correct Safe Harbor transformation instead, don't restore the hash path.
4. Fail-closed everywhere: any model load failure, inference failure, checksum
   mismatch, or timeout must never fall back to returning original/raw content. It
   must return manual_review_required or an equivalent blocked state.
5. Do not download model weights during ordinary `pytest` runs. Ordinary tests use
   mocked detectors / fixtures. Real-model inference is a separate, explicitly
   opt-in evaluation path (e.g. a marked/skippable test suite or a standalone script).
6. Never log, report, or put into exceptions/telemetry/test snapshots any raw
   detected PHI value — only categories, counts, detector names/versions, and status.
7. Do not claim "HIPAA compliant" anywhere in code, docs, comments, or status
   strings. Allowed status vocabulary: safe_harbor_technical_checks_passed,
   residual_phi_not_detected, privacy_requirements_not_met, manual_review_required,
   expert_determination_required, unsupported_or_unscannable.
8. Pin dependency/model revisions, keep checksums recorded, and do not commit model
   weights to Git.
9. Never touch Week 8, Week 7/t-closeness, the dirty NER checkout, or unrelated
   frontend/IPFS/blockchain code while doing this phase.
10. Do not push or force-push anything — local commits only.
