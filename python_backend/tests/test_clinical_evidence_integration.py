"""Invented, bounded examples of the shared text policy and written ledger."""

import json

import pytest
from services import text_anonymization as text
from services.local_model_detectors import LocalModelError
from services.phi_detection import DetectedEntity, StructuredPatternDetector
from services.surrogates import SurrogateAllocator
from services.transformation_provenance import generated_token_provenance


@pytest.fixture
def structured_only(monkeypatch):
    monkeypatch.setattr(
        text, "_detectors", lambda *args: (StructuredPatternDetector(),)
    )


def test_contact_and_measurement_with_identical_digits_keep_separate_decisions(
    structured_only, simulated_production_detectors
):
    from services.ingestion import route_for_ingestion

    original = b"Contact pager 6204.\nMRI TR 6204 ms.\nALP 6204 U/L."
    result = route_for_ingestion(
        "synthetic.txt", "text/plain", original, "strict", text_content=original
    )
    written = result["anonymized_text"]
    assert "pager 6204" not in written
    assert "TR 6204 ms" in written
    assert "ALP 6204 U/L" in written
    assert result["release_decision"]["releasable"]


def test_written_offsets_and_emitted_ledger_survive_a_text_roundtrip(
    tmp_path, structured_only
):
    original = "Email: synthetic.person@example.invalid; pager 6204."
    email_start = original.index("synthetic.person")
    email_end = original.index(";")
    phone_start = original.index("6204")
    entities = [
        DetectedEntity("EMAIL_ADDRESS", email_start, email_end, "structured_pattern"),
        DetectedEntity(
            "PHONE_NUMBER", phone_start, phone_start + 4, "structured_pattern"
        ),
    ]
    _, settings = text._profile_settings("strict")
    written, ledger = text._replace_entities(
        original, entities, "", settings, SurrogateAllocator()
    )
    assert ledger.regions[1].start != phone_start
    output = tmp_path / "written.txt"
    output.write_text(written, encoding="utf-8")
    reopened = output.read_text(encoding="utf-8")
    recovered = generated_token_provenance(reopened, ledger.emitted)
    assert len(recovered.regions) == 2
    assert not any(
        finding["blocking"] for finding in text.residual_findings(reopened, recovered)
    )

    # A token-looking input and a surviving identifier receive no trust from
    # the genuine replacement immediately beside them.
    changed = reopened + " PERSON_DEADBEEF MRN: SYN-92836"
    output.write_text(changed, encoding="utf-8")
    reopened = output.read_text(encoding="utf-8")
    recovered = generated_token_provenance(reopened, ledger.emitted)
    assert len(recovered.regions) == 2
    findings = text.residual_findings(reopened, recovered)
    assert any(finding["blocking"] for finding in findings)
    assert "SYN-92836" not in json.dumps(findings)
    assert "PERSON_DEADBEEF" not in json.dumps(findings)


def test_detector_failure_cannot_become_a_clean_release(monkeypatch):
    from services.ingestion import IngestionError, route_for_ingestion

    class Unavailable:
        source = "stanford_deidentifier"

        def detect(self, value):
            raise LocalModelError("stanford_model_unavailable", 503)

    monkeypatch.setattr(text, "_detectors", lambda *args: (Unavailable(),))
    original = b"Routine observation."
    with pytest.raises(IngestionError) as raised:
        route_for_ingestion(
            "synthetic.txt", "text/plain", original, "strict", text_content=original
        )
    assert raised.value.status_code == 503


@pytest.mark.parametrize(
    "token", ["<PERSON_999>", "<REDACTED_NAME>", "PERSON_DEADBEEF"]
)
def test_unemitted_current_and_legacy_token_shapes_still_block(token, structured_only):
    written = "Copied: " + token + "."
    recovered = generated_token_provenance(written, frozenset())
    assert recovered.regions == ()
    assert any(
        finding["blocking"] for finding in text.residual_findings(written, recovered)
    )
