"""PDF surface-inventory and TXT hardening tests (Phase 4).

Every fixture PDF is built in memory from synthetic values. No real patient
information appears in this file.
"""

import os
import sys
from io import BytesIO

import pytest

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services import document_sanitization as docs  # noqa: E402
from services.ingestion import IngestionError, detect_modality, route_for_ingestion  # noqa: E402
from services.text_anonymization import (  # noqa: E402
    mask_release_placeholders,
    residual_phi_categories,
)

fitz = pytest.importorskip("fitz", reason="PyMuPDF is required for PDF tests")

SYNTHETIC_NAME = "Jordan Fictional"
SYNTHETIC_SSN = "123-45-6789"
SYNTHETIC_EMAIL = "jordan.fictional@example.invalid"


def _pdf(
    page_texts=("Patient Name: Jordan Fictional\nSSN: 123-45-6789",),
    metadata=None,
    annotation_text=None,
    with_image=False,
    embedded=None,
) -> bytes:
    document = fitz.open()
    for text in page_texts:
        page = document.new_page()
        if text:
            page.insert_text((72, 720), text, fontsize=11)
        if annotation_text:
            annot = page.add_text_annot((200, 200), annotation_text)
            annot.update()
        if with_image:
            pixmap = fitz.Pixmap(fitz.csRGB, fitz.IRect(0, 0, 8, 8))
            pixmap.clear_with(200)
            page.insert_image(fitz.Rect(300, 300, 340, 340), pixmap=pixmap)
    if metadata:
        document.set_metadata(metadata)
    if embedded:
        document.embfile_add(embedded[0], embedded[1])
    payload = document.tobytes()
    document.close()
    return payload


# ---------------------------------------------------------------------------
# Modality routing
# ---------------------------------------------------------------------------


def test_pdf_extension_routes_to_the_pdf_modality():
    assert detect_modality("note.pdf", "application/pdf", b"%PDF-1.7\n") == "pdf"


def test_pdf_magic_wins_over_a_misleading_extension():
    assert detect_modality("note.txt", "text/plain", b"%PDF-1.7\n") == "pdf"


def test_pdf_mime_alone_is_enough():
    assert detect_modality("note", "application/pdf", b"\x00\x01") == "pdf"


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


def test_non_pdf_bytes_are_rejected():
    with pytest.raises(docs.DocumentSanitizationError):
        docs.scan_pdf_bytes(b"This is not a PDF")


def test_empty_upload_is_rejected():
    with pytest.raises(docs.DocumentSanitizationError):
        docs.scan_pdf_bytes(b"")


def test_oversized_upload_is_rejected():
    payload = b"%PDF-1.7\n" + b"0" * (docs.MAX_PDF_BYTES + 1)

    with pytest.raises(docs.DocumentSanitizationError) as exc:
        docs.scan_pdf_bytes(payload)

    assert exc.value.status_code == 413


def test_missing_reader_blocks_instead_of_reporting_clean(monkeypatch):
    monkeypatch.setattr(docs, "_load_pymupdf", lambda: None)

    result = docs.scan_pdf_bytes(b"%PDF-1.7\ntrailer")

    assert result["anonymization_status"] == docs.STATUS_UNSCANNABLE
    assert docs.REASON_READER_UNAVAILABLE in result["unscannable_reasons"]
    assert result["pages"] == []
    assert result["entity_count"] == 0


def test_unparseable_pdf_blocks():
    result = docs.scan_pdf_bytes(b"%PDF-1.7\nthis is not really a pdf body")

    assert result["anonymization_status"] == docs.STATUS_UNSCANNABLE
    assert docs.REASON_UNPARSEABLE in result["unscannable_reasons"]


def test_encrypted_pdf_blocks():
    document = fitz.open()
    page = document.new_page()
    page.insert_text((72, 720), "Patient Name: Jordan Fictional")
    payload = document.tobytes(
        encryption=fitz.PDF_ENCRYPT_AES_256,
        owner_pw="owner-secret",
        user_pw="user-secret",
    )
    document.close()

    result = docs.scan_pdf_bytes(payload)

    assert result["anonymization_status"] == docs.STATUS_UNSCANNABLE
    assert docs.REASON_ENCRYPTED in result["unscannable_reasons"]
    assert result["pages"] == []


def test_page_limit_blocks(monkeypatch):
    monkeypatch.setattr(docs, "MAX_PDF_PAGES", 1)

    result = docs.scan_pdf_bytes(_pdf(page_texts=("one", "two")))

    assert result["anonymization_status"] == docs.STATUS_UNSCANNABLE
    assert docs.REASON_PAGE_LIMIT in result["unscannable_reasons"]


# ---------------------------------------------------------------------------
# Surface coverage
# ---------------------------------------------------------------------------


def test_text_layer_is_scanned_and_redacted():
    result = docs.scan_pdf_bytes(_pdf())

    assert result["text_layer_complete"] is True
    page = result["pages"][0]
    assert SYNTHETIC_NAME not in page["anonymized_text"]
    assert SYNTHETIC_SSN not in page["anonymized_text"]
    assert "<REDACTED_SSN>" in page["anonymized_text"]
    assert result["detected_entities"].get("US_SSN") == 1


def test_document_metadata_is_scanned_not_ignored():
    payload = _pdf(
        page_texts=("Routine follow-up visit.",),
        metadata={"title": f"Chart for {SYNTHETIC_NAME}", "author": SYNTHETIC_NAME},
    )

    result = docs.scan_pdf_bytes(payload)

    assert "title" in result["pdf_summary"]["metadata_fields_present"]
    assert result["entity_count"] > 0
    assert result["detected_entities"].get("PERSON")


def test_annotation_text_is_scanned():
    payload = _pdf(
        page_texts=("Routine follow-up visit.",),
        annotation_text=f"Call {SYNTHETIC_EMAIL} about results",
    )

    result = docs.scan_pdf_bytes(payload)

    assert result["pdf_summary"]["annotation_surfaces"] > 0
    assert result["detected_entities"].get("EMAIL_ADDRESS")


def test_raster_page_is_marked_unscannable():
    result = docs.scan_pdf_bytes(_pdf(with_image=True))

    assert result["pdf_summary"]["raster_pages"] == 1
    assert docs.REASON_RASTER_REQUIRES_OCR in result["unscannable_reasons"]
    assert result["anonymization_status"] == docs.STATUS_UNSCANNABLE
    assert result["scannable"] is False


def test_image_only_page_is_never_reported_as_clean():
    result = docs.scan_pdf_bytes(_pdf(page_texts=("",), with_image=True))

    page = result["pages"][0]
    assert page["has_text_layer"] is False
    assert page["image_count"] == 1
    assert result["pdf_summary"]["image_only_pages"] == 1
    assert result["anonymization_status"] == docs.STATUS_UNSCANNABLE
    assert docs.REASON_RASTER_REQUIRES_OCR in result["unscannable_reasons"]


def test_embedded_file_blocks():
    payload = _pdf(embedded=("attachment.txt", b"Patient Jordan Fictional"))

    result = docs.scan_pdf_bytes(payload)

    assert result["pdf_summary"]["embedded_file_count"] >= 1
    assert docs.REASON_EMBEDDED_FILES in result["unscannable_reasons"]
    assert result["anonymization_status"] == docs.STATUS_UNSCANNABLE


def test_active_content_marker_blocks():
    payload = _pdf() + b"\n% /JavaScript app.alert(1)\n"

    result = docs.scan_pdf_bytes(payload)

    assert "JavaScript" in result["pdf_summary"]["blocking_content_indicators"]
    assert docs.REASON_ACTIVE_CONTENT in result["unscannable_reasons"]
    assert result["anonymization_status"] == docs.STATUS_UNSCANNABLE


def test_short_ambiguous_tokens_do_not_trigger_active_content():
    # "/JS" and "/AA" collide with ordinary compressed bytes and must not be
    # treated as active-content markers.
    assert b"/JS" not in b"".join(docs._BLOCKING_CONTENT_TOKENS).replace(
        b"/JavaScript", b""
    )
    assert b"/AA" not in b"".join(docs._BLOCKING_CONTENT_TOKENS)


def test_surface_over_the_scan_limit_blocks(monkeypatch):
    monkeypatch.setattr(docs, "MAX_SURFACE_TEXT_BYTES", 8)

    result = docs.scan_pdf_bytes(_pdf())

    assert docs.REASON_SURFACE_TOO_LARGE in result["unscannable_reasons"]
    assert result["text_layer_complete"] is False
    assert all(page["anonymized_text"] is None for page in result["pages"])


# ---------------------------------------------------------------------------
# Release posture
# ---------------------------------------------------------------------------


def test_a_clean_pdf_is_still_never_automatically_releasable():
    result = docs.scan_pdf_bytes(_pdf(page_texts=("Routine follow-up visit.",)))

    assert result["anonymization_status"] == docs.STATUS_MANUAL_REVIEW
    assert docs.REASON_NO_VALIDATED_WRITER in result["unscannable_reasons"]


def test_blocked_scan_withholds_every_page_text():
    result = docs.scan_pdf_bytes(_pdf(embedded=("a.txt", b"x")))

    # An unreadable non-text surface does not by itself hide the text layer,
    # but the document is not releasable.
    assert result["scannable"] is False
    assert result["anonymization_status"] == docs.STATUS_UNSCANNABLE


def test_original_bytes_are_never_returned():
    payload = _pdf()

    result = docs.scan_pdf_bytes(payload)

    serialized = repr(result)
    assert "%PDF-" not in serialized
    assert SYNTHETIC_NAME not in serialized
    assert SYNTHETIC_SSN not in serialized


def test_ingestion_routes_pdf_to_manual_review():
    payload = _pdf()

    response = route_for_ingestion(
        filename="note.pdf",
        content_type="application/pdf",
        header=payload[:4096],
        profile="strict",
        file_content=payload,
    )

    assert response["detected_modality"] == "pdf"
    assert response["release_decision"]["releasable"] is False
    assert response["release_decision"]["artifact_sha256"] is None
    assert response["downstream"] == {
        "ipfs_chunking": "blocked",
        "cid_encryption": "blocked",
        "metadata_indexing": "blocked",
        "blockchain_transaction": "blocked",
    }


def test_research_profile_pdf_returns_expert_determination():
    payload = _pdf()

    response = route_for_ingestion(
        filename="note.pdf",
        content_type="application/pdf",
        header=payload[:4096],
        profile="research",
        file_content=payload,
    )

    assert response["anonymization_status"] == "expert_determination_required"
    assert response["release_decision"]["releasable"] is False
    assert "pages" not in response
    assert SYNTHETIC_NAME not in repr(response)


# ---------------------------------------------------------------------------
# TXT hardening
# ---------------------------------------------------------------------------


def test_non_utf8_text_upload_is_rejected():
    with pytest.raises(IngestionError) as exc:
        route_for_ingestion(
            filename="note.txt",
            content_type="text/plain",
            header=b"note",
            profile="strict",
            text_content=b"\xff\xfe\x00bad",
        )

    assert exc.value.detail == "Text uploads must be UTF-8 encoded"


def test_nul_byte_text_upload_is_rejected():
    # A NUL can split a value apart so the detector never matches it.
    with pytest.raises(IngestionError) as exc:
        route_for_ingestion(
            filename="note.txt",
            content_type="text/plain",
            header=b"note",
            profile="strict",
            text_content=b"MRN: 123456\x00 hidden",
        )

    assert exc.value.detail == "Text uploads must not contain NUL bytes"


def test_release_placeholders_are_masked_without_shifting_offsets():
    text = "Patient <REDACTED_NAME> called."

    masked = mask_release_placeholders(text)

    assert len(masked) == len(text)
    assert "REDACTED" not in masked


def test_clean_redacted_text_has_no_residual_findings():
    assert residual_phi_categories("Patient <REDACTED_NAME> called <REDACTED_PHONE>.") == {}


def test_residual_phi_blocks_the_text_release(monkeypatch):
    from services import ingestion

    def leaky_handler(text_content, profile, study_salt=None):
        return {
            "handler": "anonymize_text",
            "routing_status": "handler_selected",
            "anonymization_status": "completed",
            "message": "Text anonymization completed.",
            "anonymized_text": "Contact SSN 123-45-6789 for details.",
            "residual_phi_categories": {"US_SSN": 1},
            "date_strategy": "redact",
            "text_identifier_strategy": "redact",
            "detected_entities": {},
            "entity_count": 0,
            "detection_sources": {},
            "ner_model": "en_core_web_sm",
            "trained_ner_active": True,
        }

    monkeypatch.setitem(ingestion.HANDLER_REGISTRY, "text", leaky_handler)

    response = route_for_ingestion(
        filename="note.txt",
        content_type="text/plain",
        header=b"note",
        profile="strict",
        text_content=b"Contact SSN 123-45-6789 for details.",
    )

    decision = response["release_decision"]
    assert decision["releasable"] is False
    assert decision["disposition"] == "manual_review_required"
    assert "privacy_requirements_not_met" in decision["reason_codes"]
    assert decision["artifact_sha256"] is None
    assert "123-45-6789" not in repr(decision)


def test_clean_text_still_releases_with_the_residual_validator():
    response = route_for_ingestion(
        filename="note.txt",
        content_type="text/plain",
        header=b"note",
        profile="strict",
        text_content=b"Patient has MRN: 123456 and no other identifiers.",
        study_salt="study-a",
    )

    decision = response["release_decision"]
    assert decision["releasable"] is True
    assert "safe_harbor_technical_checks_passed" in decision["reason_codes"]
    assert response["residual_phi_categories"] == {}


# ---------------------------------------------------------------------------
# /anonymize_pdf endpoint posture
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def api_client():
    from fastapi.testclient import TestClient

    from main import app

    return TestClient(app)


def test_endpoint_reports_a_blocked_release_for_a_clean_pdf(api_client):
    payload = _pdf(page_texts=("Routine follow-up visit.",))

    response = api_client.post(
        "/anonymize_pdf",
        files={"file": ("note.pdf", BytesIO(payload), "application/pdf")},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["anonymization_status"] == docs.STATUS_MANUAL_REVIEW
    assert body["release_decision"]["releasable"] is False
    assert body["release_decision"]["artifact_sha256"] is None
    assert docs.REASON_NO_VALIDATED_WRITER in body["unscannable_reasons"]


def test_endpoint_withholds_page_text_for_an_image_only_pdf(api_client):
    payload = _pdf(page_texts=("",), with_image=True)

    response = api_client.post(
        "/anonymize_pdf",
        files={"file": ("scan.pdf", BytesIO(payload), "application/pdf")},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["anonymization_status"] == docs.STATUS_UNSCANNABLE
    assert docs.REASON_RASTER_REQUIRES_OCR in body["unscannable_reasons"]
    assert body["pages"][0]["image_count"] == 1
    # An unscanned raster page must never be reported as clean text.
    assert body["total_entities"] == 0
    assert body["pdf_summary"]["image_only_pages"] == 1


def test_endpoint_scans_metadata_the_old_text_only_path_ignored(api_client):
    payload = _pdf(
        page_texts=("Routine follow-up visit.",),
        metadata={"title": f"Chart for {SYNTHETIC_NAME}"},
    )

    response = api_client.post(
        "/anonymize_pdf",
        files={"file": ("note.pdf", BytesIO(payload), "application/pdf")},
    )

    body = response.json()
    assert body["total_entities"] > 0
    assert SYNTHETIC_NAME not in response.text


def test_endpoint_never_echoes_document_content_in_errors(api_client):
    payload = b"%PDF-1.7\n" + SYNTHETIC_SSN.encode() + b" corrupted body"

    response = api_client.post(
        "/anonymize_pdf",
        files={"file": ("broken.pdf", BytesIO(payload), "application/pdf")},
    )

    assert SYNTHETIC_SSN not in response.text


def test_endpoint_rejects_a_non_pdf_filename(api_client):
    response = api_client.post(
        "/anonymize_pdf",
        files={"file": ("note.txt", BytesIO(b"not a pdf"), "text/plain")},
    )

    assert response.status_code == 400


def test_every_blocked_path_returns_the_full_result_shape(monkeypatch):
    # A blocked scan must be shaped exactly like a completed one, or callers
    # that read the summary keys crash instead of blocking.
    monkeypatch.setattr(docs, "_load_pymupdf", lambda: None)
    blocked = docs.scan_pdf_bytes(b"%PDF-1.7\ntrailer")

    monkeypatch.undo()
    complete = docs.scan_pdf_bytes(_pdf())

    assert set(blocked) == set(complete)


def test_ingestion_handles_an_unparseable_pdf_without_crashing():
    response = route_for_ingestion(
        filename="broken.pdf",
        content_type="application/pdf",
        header=b"%PDF-1.7\nnot a real body",
        profile="strict",
        file_content=b"%PDF-1.7\nnot a real body",
    )

    assert response["anonymization_status"] == docs.STATUS_UNSCANNABLE
    assert response["release_decision"]["releasable"] is False
    assert docs.REASON_UNPARSEABLE in response["unscannable_reasons"]
