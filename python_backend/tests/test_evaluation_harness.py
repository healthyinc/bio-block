"""The canary evaluation runs as a regression gate (Phase 8).

This runs the whole harness in the default `legacy_test` model mode, so it uses
the rule-based and spaCy detectors and downloads nothing. The real-model run is
the separate opt-in path.

Zero residual canaries is an acceptance condition here. It is not proof that an
artifact, model, modality, or deployment has zero PHI leakage.
"""

import json
import os
import sys

import pytest

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evaluations import canary_corpus  # noqa: E402
from evaluations.run_evaluation import run_evaluation  # noqa: E402
from services.phi_detection import StructuredPatternDetector  # noqa: E402


@pytest.fixture(scope="module")
def report():
    os.environ.setdefault("BIOBLOCK_STUDY_SALT", "evaluation-salt")
    return run_evaluation()


def test_every_case_runs(report):
    assert report["cases_run"] > 0
    assert report["cases_skipped"] == 0, "an optional dependency is missing"


def test_no_canary_survives_anything_the_pipeline_released(report):
    offenders = [
        (case["case_id"], case["residual_canaries"])
        for case in report["cases"]
        if case.get("residual_canary_count")
    ]

    assert offenders == [], f"canaries survived: {offenders}"
    assert report["acceptance"]["zero_residual_canaries"] is True


def test_research_never_releases_on_any_modality(report):
    assert report["research_cases_released"] == []
    assert report["acceptance"]["research_never_releases"] is True


def test_release_posture_matches_the_documented_contract(report):
    assert report["release_posture_mismatches"] == []


def test_only_text_can_release_automatically(report):
    released = {
        case["case_id"]
        for case in report["cases"]
        if case.get("releasable") and not case["case_id"].endswith("__research")
    }
    non_text = {
        case_id
        for case_id in released
        if not case_id.startswith(("text_", "raster_", "index_gate"))
    }

    assert non_text == set(), f"unexpected releasable modality: {non_text}"


def test_image_only_pdf_is_not_reported_as_clean(report):
    case = next(
        item for item in report["cases"] if item["case_id"] == "pdf_image_only_page"
    )

    assert case["releasable"] is False
    assert case["anonymization_status"] == "unsupported_or_unscannable"


def test_report_never_carries_a_canary_value(report):
    serialized = json.dumps(report, default=str)

    for canary in canary_corpus.ALL_CANARIES:
        assert canary not in serialized, "the report leaked a canary value"


def test_report_names_canaries_by_index_not_value():
    labels = canary_corpus.canary_labels([canary_corpus.CANARY_SSN])

    assert labels == ["canary_06"]
    assert canary_corpus.CANARY_SSN not in labels[0]


# ---------------------------------------------------------------------------
# Regressions the harness surfaced
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text",
    [
        "accessed from 203.0.113.42.",
        "accessed from 203.0.113.42",
        "host 203.0.113.42, port 80",
        "(203.0.113.42)",
        "203.0.113.42 was the source",
    ],
)
def test_ip_address_is_detected_including_at_end_of_sentence(text):
    # The old boundary was (?![\d.]), so a trailing sentence period defeated
    # it and an IP ending a sentence was never detected.
    found = [
        entity
        for entity in StructuredPatternDetector().detect(text)
        if entity.entity_type == "IP_ADDRESS"
    ]

    assert len(found) == 1
    assert text[found[0].start : found[0].end] == "203.0.113.42"


@pytest.mark.parametrize(
    "text",
    [
        "version 1.2.3.4.5 released",
        "ratio 10.5 and 2.3 measured",
        "in 2019 the value 42 was 3.14",
        "dose 1.5.2.3.9.1 schedule",
    ],
)
def test_ip_boundary_still_rejects_longer_dotted_numbers(text):
    found = [
        entity
        for entity in StructuredPatternDetector().detect(text)
        if entity.entity_type == "IP_ADDRESS"
    ]

    assert found == []


def test_residual_scan_ignores_the_high_recall_proper_noun_heuristic():
    # Masking placeholders changes sentence shape, which exposes ordinary
    # capitalized words to the high-recall rule. Blocking on those would block
    # releases on our own leftovers rather than on surviving PHI.
    from services.text_anonymization import residual_phi_categories

    assert residual_phi_categories("Portal <REDACTED_URL> was accessed.") == {}


def test_residual_scan_still_catches_evidence_based_findings():
    from services.text_anonymization import residual_phi_categories

    residual = residual_phi_categories(
        "Contact <REDACTED_NAME> at jordan.fictional@example.invalid."
    )

    assert residual.get("EMAIL_ADDRESS") == 1
