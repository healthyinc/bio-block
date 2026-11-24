"""Phase 12D: shadow evaluation of residual-validator candidates.

The second pass costs about as much as the first, and GLiNER is most of it,
so the question is whether the second GLiNER run earns its cost. These tests
cover the machinery that answers it: a corpus that injects synthetic PHI
*after* transformation, a cache that refuses anything not produced under the
exact configuration being asked about, and candidate composition.

The load-bearing test is the last one. A shadow candidate exists to be
compared, never to decide anything, so nothing here may reach a release
decision or alter the production validator.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evaluations import residual_corpus as corpus  # noqa: E402
from evaluations import residual_shadow as shadow  # noqa: E402

# -- the corpus tests the validator, not the first pass ---------------------


def test_documents_look_like_finished_output():
    """Injection happens after transformation, so surrogates are present."""
    documents = corpus.partition_documents(corpus.VALIDATION, count=20)

    assert any(
        "PATIENT_001" in d.text or "<REDACTED_" in d.text for d in documents
    ), "no document resembles transformed output"


def test_every_injected_span_maps_to_its_text():
    for partition in (corpus.CALIBRATION, corpus.VALIDATION):
        for document in corpus.partition_documents(partition, count=40):
            for span in document.spans:
                assert document.text[
                    span.start : span.end
                ].strip(), f"{document.doc_id}: span offsets do not map to text"


def test_every_safe_harbor_category_is_represented():
    coverage = corpus.category_coverage(corpus.VALIDATION)

    missing = [name for name, count in coverage.items() if count == 0]
    assert not missing, f"categories with no coverage: {missing}"
    assert set(coverage) == set(corpus.CATEGORIES)


def test_unrepresentable_categories_are_named_not_hidden():
    """Full-face photographs have no textual form; say so rather than omit."""
    assert corpus.UNREPRESENTABLE_CATEGORIES
    assert "FULL_FACE_PHOTOGRAPH" in corpus.UNREPRESENTABLE_CATEGORIES


def test_safe_corpus_contains_no_injected_identifier():
    for document in corpus.partition_documents(corpus.SAFE_CLINICAL, count=40):
        assert document.spans == ()
        assert document.has_residual is False


def test_partitions_do_not_share_documents():
    """Validation must not be tunable through the calibration partition."""
    calibration = {d.text for d in corpus.partition_documents(corpus.CALIBRATION)}
    validation = {d.text for d in corpus.partition_documents(corpus.VALIDATION)}

    assert calibration.isdisjoint(validation)


def test_corpus_is_deterministic():
    first = corpus.partition_digest(corpus.VALIDATION)
    second = corpus.partition_digest(corpus.VALIDATION)

    assert first == second


def test_difficult_shapes_are_all_exercised():
    shapes = {
        span.shape
        for document in corpus.partition_documents(corpus.VALIDATION)
        for span in document.spans
    }

    for required in (
        "start_of_text",
        "end_of_text",
        "after_crlf",
        "table_row",
        "unicode_context",
        "long_line",
    ):
        assert required in shapes, f"shape not exercised: {required}"


# -- cache validation -------------------------------------------------------


def _cache_entry(documents=2, detectors=None):
    detectors = detectors or shadow.ALL_DETECTORS
    return {
        "cache_key": "abc",
        "count": documents,
        "documents": [
            {"doc_id": f"d{i}", "findings": {name: [] for name in detectors}}
            for i in range(documents)
        ],
    }


def test_cache_is_used_when_everything_matches():
    assert shadow._cache_is_usable(_cache_entry(), "abc", 2) is True


def test_cache_with_a_different_key_is_rejected():
    """A different model revision or threshold invalidates the entry."""
    assert shadow._cache_is_usable(_cache_entry(), "different", 2) is False


def test_cache_with_the_wrong_document_count_is_rejected():
    assert shadow._cache_is_usable(_cache_entry(documents=2), "abc", 5) is False


def test_cache_missing_a_detector_is_rejected():
    """A partial record would understate whichever candidate needs it."""
    partial = _cache_entry(detectors=(shadow.D_STRUCTURED, shadow.D_STANFORD))

    assert shadow._cache_is_usable(partial, "abc", 2) is False


def test_cache_with_a_malformed_document_is_rejected():
    broken = _cache_entry()
    broken["documents"][0]["findings"] = None

    assert shadow._cache_is_usable(broken, "abc", 2) is False


def test_fingerprint_changes_with_the_corpus():
    fingerprint = {"cache_version": 1, "models": {}, "detectors": {}}

    assert shadow._fingerprint_digest(fingerprint, "corpus-a") != (
        shadow._fingerprint_digest(fingerprint, "corpus-b")
    )


def test_fingerprint_changes_with_the_configuration():
    assert shadow._fingerprint_digest({"a": 1}, "c") != shadow._fingerprint_digest(
        {"a": 2}, "c"
    )


# -- candidate composition --------------------------------------------------


def _entry(**findings):
    rows = {name: [] for name in shadow.ALL_DETECTORS}
    rows.update(findings)
    return {"doc_id": "d0", "findings": rows}


def test_control_uses_every_detector():
    assert set(shadow.CANDIDATES["control"]) == set(shadow.ALL_DETECTORS)


def test_candidate_a_excludes_gliner():
    assert shadow.D_GLINER not in shadow.CANDIDATES["A_no_gliner"]
    assert shadow.D_STANFORD in shadow.CANDIDATES["A_no_gliner"]


def test_composition_includes_only_the_candidate_detectors():
    entry = _entry(
        gliner=[[0, 5, "PERSON", 0.9]],
        stanford=[[10, 15, "PERSON", 0.9]],
    )

    control = shadow.compose(entry, "control")
    without = shadow.compose(entry, "A_no_gliner")

    assert {f.detector for f in control} == {"gliner", "stanford"}
    assert {f.detector for f in without} == {"stanford"}


def test_composition_is_sorted_deterministically():
    entry = _entry(
        stanford=[[30, 35, "PERSON", 0.9], [10, 15, "PERSON", 0.9]],
        spacy=[[20, 25, "PERSON", None]],
    )

    findings = shadow.compose(entry, "A_no_gliner")

    assert [f.start for f in findings] == sorted(f.start for f in findings)


def test_cascade_runs_gliner_only_inside_a_nominated_region():
    """A GLiNER span far from any cheap signal is not consulted."""
    entry = _entry(
        structured=[[0, 5, "US_SSN", None]],
        gliner=[[10_000, 10_010, "PERSON", 0.9]],
    )

    findings = shadow.compose(entry, "D_cascade")

    assert all(f.detector != "gliner" for f in findings)


def test_cascade_keeps_a_gliner_span_near_a_cheap_signal():
    entry = _entry(
        structured=[[100, 110, "US_SSN", None]],
        gliner=[[120, 130, "PERSON", 0.9]],
    )

    findings = shadow.compose(entry, "D_cascade")

    assert any(f.detector == "gliner" for f in findings)


def test_cascade_nominates_nothing_when_no_cheap_signal_fires():
    assert shadow.high_risk_regions(_entry()) == []


# -- scoring ----------------------------------------------------------------


def test_a_missed_span_is_counted_as_a_false_negative():
    document = corpus.build_document(corpus.VALIDATION, 0)
    cached = {
        "documents": [
            {
                "doc_id": document.doc_id,
                "findings": {n: [] for n in shadow.ALL_DETECTORS},
            }
        ],
        "detector_ms": {n: 1.0 for n in shadow.ALL_DETECTORS},
    }

    result = shadow.evaluate_candidate(cached, "control", corpus.VALIDATION, 1)

    assert result["false_negatives"] == len(document.spans)
    assert result["recall"] == 0.0
    assert result["document_leakage_rate"] == 1.0
    assert result["missed_categories"]


def test_a_caught_span_is_credited_to_its_only_detector():
    document = corpus.build_document(corpus.VALIDATION, 0)
    span = document.spans[0]
    cached = {
        "documents": [
            {
                "doc_id": document.doc_id,
                "findings": {
                    **{n: [] for n in shadow.ALL_DETECTORS},
                    shadow.D_STANFORD: [[span.start, span.end, "PERSON", 0.9]],
                },
            }
        ],
        "detector_ms": {n: 1.0 for n in shadow.ALL_DETECTORS},
    }

    result = shadow.evaluate_candidate(cached, "control", corpus.VALIDATION, 1)

    assert result["unique_detector_credit"][shadow.D_STANFORD] >= 1
    assert result["unique_detector_credit"][shadow.D_GLINER] == 0


def test_category_recall_is_reported_per_category():
    document = corpus.build_document(corpus.VALIDATION, 0)
    cached = {
        "documents": [
            {
                "doc_id": document.doc_id,
                "findings": {n: [] for n in shadow.ALL_DETECTORS},
            }
        ],
        "detector_ms": {n: 1.0 for n in shadow.ALL_DETECTORS},
    }

    result = shadow.evaluate_candidate(cached, "control", corpus.VALIDATION, 1)

    assert set(result["missed_categories"]) <= set(corpus.CATEGORIES)


# -- the boundary that matters ---------------------------------------------


def test_no_shadow_candidate_can_authorize_a_release():
    """A shadow candidate compares; it never decides.

    The evaluation module must not import, construct or reach the release
    gate. If a future change wires one in, this fails.
    """
    import inspect

    source = inspect.getsource(shadow)

    for forbidden in (
        "release_gate",
        "release_decision",
        "issue_release",
        "sanitize_for_index",
        "anonymize_clinical_text",
    ):
        assert (
            forbidden not in source
        ), f"shadow evaluation reached production release machinery: {forbidden}"


def test_shadow_evaluation_does_not_touch_the_production_validator():
    """residual_findings is production's second pass and stays untouched."""
    import inspect

    assert "residual_findings" not in inspect.getsource(shadow)
    assert "residual_phi_categories" not in inspect.getsource(shadow)


def test_production_validator_still_runs_every_detector():
    """The control candidate must still describe what production does."""
    from services.text_anonymization import residual_findings

    assert callable(residual_findings)
    assert set(shadow.CANDIDATES["control"]) == set(shadow.ALL_DETECTORS)


@pytest.mark.parametrize("candidate", sorted(shadow.CANDIDATES))
def test_no_candidate_omits_all_model_validation(candidate):
    """Rules alone are not a validator; at least one model must remain."""
    detectors = set(shadow.CANDIDATES[candidate])

    assert detectors & {
        shadow.D_SPACY,
        shadow.D_STANFORD,
        shadow.D_GLINER,
    }, f"{candidate} would leave no model checking the output"
