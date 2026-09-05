"""Integrity tests for the labelled evaluation corpus (Phase 9).

These run in the ordinary suite and load no models. The guarantee they protect
is the one every calibration number depends on: a value used to choose a
threshold must never reappear in the held-out measurement.
"""

import json
import os
import sys

import pytest

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evaluations.labelled_corpus import (  # noqa: E402
    CATEGORY_ALIASES,
    CORPUS_VERSION,
    NON_PHI_CLINICAL_TERMS,
    PARTITION_CALIB,
    PARTITION_DEV,
    PARTITION_TEST,
    PARTITIONS,
    REQUIRED_CATEGORIES,
    all_values,
    build_corpus,
    corpus_statistics,
    partition_documents,
)
from evaluations.metrics import (  # noqa: E402
    Detection,
    aggregate,
    missed_categories,
    percentile,
    score_document,
)


# ---------------------------------------------------------------------------
# Structure
# ---------------------------------------------------------------------------


def test_three_partitions_exist():
    assert PARTITIONS == (PARTITION_DEV, PARTITION_CALIB, PARTITION_TEST)
    assert set(build_corpus()) == set(PARTITIONS)


def test_gold_offsets_are_exact_and_in_range():
    for documents in build_corpus().values():
        for document in documents:
            for span in document.spans:
                assert 0 <= span.start < span.end <= len(document.text)
                assert document.value(span).strip(), "gold span is blank"


def test_negative_control_offsets_are_exact():
    for documents in build_corpus().values():
        for document in documents:
            for start, end, _label in document.negatives:
                assert 0 <= start < end <= len(document.text)
                assert document.text[start:end].strip()


def test_partitions_share_no_gold_value():
    """The guarantee the whole calibration rests on."""
    pools = {partition: set(all_values(partition)) for partition in PARTITIONS}

    for left in PARTITIONS:
        for right in PARTITIONS:
            if left < right:
                shared = pools[left] & pools[right]
                assert shared == set(), (
                    f"{left} and {right} share {len(shared)} values; a "
                    "threshold chosen on one would leak into the other"
                )


def test_partitions_share_no_document_text():
    texts = {
        partition: {d.text for d in partition_documents(partition)}
        for partition in PARTITIONS
    }
    assert texts[PARTITION_CALIB].isdisjoint(texts[PARTITION_TEST])
    assert texts[PARTITION_DEV].isdisjoint(texts[PARTITION_TEST])


def test_every_required_category_appears_in_every_partition():
    for partition in PARTITIONS:
        present = {
            span.category
            for document in partition_documents(partition)
            for span in document.spans
        }
        missing = set(REQUIRED_CATEGORIES) - present
        assert missing == set(), f"{partition} is missing {sorted(missing)}"


@pytest.mark.parametrize(
    "tag",
    [
        "indian_name",
        "international_name",
        "unicode",
        "transliteration",
        "mixed_case",
        "misspelt_label",
        "misspelt_value",
        "abbreviation_context",
        "multiline",
        "overlapping",
        "chunk_boundary",
        "age_over_89",
        "biometric",
        "vehicle",
        "account",
        "certificate",
        "license",
        "unusual_format",
        "phone_international",
        "ip_end_of_sentence",
    ],
)
def test_required_stress_features_are_present(tag):
    tags = {
        t
        for partition in PARTITIONS
        for document in partition_documents(partition)
        for t in document.tags
    }
    assert tag in tags


def test_clinical_eponyms_are_recorded_as_negative_controls():
    negatives = {
        document.text[start:end]
        for document in partition_documents(PARTITION_TEST)
        for start, end, _label in document.negatives
    }
    for term in ("Parkinson", "Alzheimer", "Crohn", "Hodgkin"):
        assert term in negatives
    assert set(NON_PHI_CLINICAL_TERMS)


def test_chunk_boundary_document_actually_straddles_a_window():
    from evaluations.labelled_corpus import partition_documents as docs

    document = next(
        d for d in docs(PARTITION_TEST) if "chunk_boundary" in d.tags
    )
    boundary_spans = [s for s in document.spans if "chunk_boundary" in s.tags]

    assert boundary_spans, "no span is tagged as straddling a boundary"
    # The default window is 2000 characters; the tagged spans must sit near it.
    assert any(1900 < span.start < 2100 for span in boundary_spans)


def test_corpus_statistics_report_counts_only():
    stats = corpus_statistics()
    serialized = json.dumps(stats)

    for partition in PARTITIONS:
        for value in all_values(partition):
            assert value not in serialized


def test_corpus_version_is_declared():
    assert CORPUS_VERSION
    assert CORPUS_VERSION.startswith("canary-v")


def test_every_category_has_an_alias_mapping():
    for category in REQUIRED_CATEGORIES:
        assert CATEGORY_ALIASES[category]


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


def test_overlapping_detection_counts_as_a_span_hit():
    document = partition_documents(PARTITION_DEV)[0]
    span = document.spans[0]
    # Partial overlap only: the value would still be redacted.
    detection = Detection(
        start=span.start + 1, end=span.end + 5, category="PERSON", source="test"
    )

    score = score_document(document, [detection])

    assert score.span_hits == 1


def test_a_missed_span_is_a_false_negative_and_a_leaking_document():
    document = partition_documents(PARTITION_DEV)[0]

    report = aggregate([document], {document.doc_id: []})

    assert report["false_negatives"] == len(document.spans)
    assert report["span_recall"] == 0.0
    assert report["document_leakage_rate"] == 1.0
    assert document.doc_id in report["leaking_documents"]


def test_typed_recall_separates_wrong_label_from_missed_span():
    document = partition_documents(PARTITION_DEV)[0]
    span = document.spans[0]
    wrong_label = Detection(
        start=span.start, end=span.end, category="NOT_A_CATEGORY", source="test"
    )

    report = aggregate([document], {document.doc_id: [wrong_label]})

    # Redacted, so no false negative, but not correctly categorised.
    assert report["true_positives"] >= 1
    assert report["typed_recall"] < report["span_recall"]


def test_redacting_a_negative_control_lowers_useful_text_preservation():
    document = next(
        d for d in partition_documents(PARTITION_DEV) if d.negatives
    )
    start, end, _ = document.negatives[0]
    detection = Detection(start=start, end=end, category="PERSON", source="test")

    report = aggregate([document], {document.doc_id: [detection]})

    assert report["useful_text_preservation"] < 1.0


def test_missed_categories_lists_only_zero_recall_categories():
    document = partition_documents(PARTITION_DEV)[0]
    report = aggregate([document], {document.doc_id: []})

    missed = missed_categories(report)

    assert missed
    assert all(report["by_category"][name]["span_recall"] == 0.0 for name in missed)


def test_percentile_is_nearest_rank():
    assert percentile([1, 2, 3, 4, 5], 0.5) == 3
    assert percentile([1, 2, 3, 4, 5], 1.0) == 5
    assert percentile([], 0.9) == 0.0
