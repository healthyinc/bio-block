"""CSV and workbook policy-routing tests (Phase 6).

All fixtures use synthetic identifiers. No real patient data appears here.
"""

import json
import os
import sys
from io import BytesIO

import pytest

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services import tabular_validation as tv  # noqa: E402
from services import workbook_sanitization as wb  # noqa: E402
from services.ingestion import detect_modality, route_for_ingestion  # noqa: E402
from services.tabular_anonymization import anonymize_tabular_csv  # noqa: E402

openpyxl = pytest.importorskip("openpyxl")

SYNTHETIC_NAME = "Alice Alpha"
SYNTHETIC_EMAIL = "alice.alpha@example.invalid"
SYNTHETIC_MRN = "MRN-000101"

CSV_CONTENT = (
    b"name,email,phone,mrn,age,gender,diagnosis\n"
    b"Alice Alpha,alice.alpha@example.invalid,555-111-2222,MRN-000101,31,F,flu\n"
    b"Bob Beta,bob.beta@example.invalid,555-111-3333,MRN-000102,32,F,cold\n"
    b"Carol Gamma,carol.gamma@example.invalid,555-111-4444,MRN-000103,33,M,flu\n"
    b"Dan Delta,dan.delta@example.invalid,555-111-5555,MRN-000104,34,M,cold\n"
)


# ---------------------------------------------------------------------------
# Serialized-output validation
# ---------------------------------------------------------------------------


def test_matching_output_passes_validation():
    result = tv.validate_serialized_csv(
        "age,diagnosis\r\n31-32,flu\r\n33-34,cold\r\n",
        expected_columns=["age", "diagnosis"],
        expected_row_count=2,
        removed_columns=["name"],
    )

    assert result["serialized_output_validation"] == tv.VALIDATION_PASSED
    assert result["validation_failures"] == []


def test_a_removed_column_that_survived_serialization_fails():
    result = tv.validate_serialized_csv(
        "age,diagnosis,name\r\n31,flu,Alice Alpha\r\n",
        expected_columns=["age", "diagnosis"],
        expected_row_count=1,
        removed_columns=["name"],
    )

    assert result["serialized_output_validation"] == tv.VALIDATION_FAILED
    assert tv.FAILURE_REMOVED_COLUMN_PRESENT in result["validation_failures"]
    assert result["leaked_columns"] == ["name"]


def test_header_drift_from_the_plan_fails():
    result = tv.validate_serialized_csv(
        "age,outcome\r\n31,flu\r\n",
        expected_columns=["age", "diagnosis"],
        expected_row_count=1,
        removed_columns=[],
    )

    assert tv.FAILURE_HEADER_MISMATCH in result["validation_failures"]


def test_row_count_drift_from_the_plan_fails():
    result = tv.validate_serialized_csv(
        "age\r\n31\r\n32\r\n",
        expected_columns=["age"],
        expected_row_count=5,
        removed_columns=[],
    )

    assert tv.FAILURE_ROW_COUNT_MISMATCH in result["validation_failures"]


def test_a_removed_identifier_echoed_into_a_retained_cell_fails():
    # The column was dropped, but its value reappears inside a kept column.
    result = tv.validate_serialized_csv(
        f"age,notes\r\n31,contact {SYNTHETIC_EMAIL}\r\n",
        expected_columns=["age", "notes"],
        expected_row_count=1,
        removed_columns=["email"],
        source_records=[{"email": SYNTHETIC_EMAIL, "age": "31"}],
    )

    assert tv.FAILURE_IDENTIFIER_VALUE_PRESENT in result["validation_failures"]
    assert result["leaked_value_count"] == 1


def test_short_values_are_not_treated_as_leak_signals():
    # "F" or "31" would otherwise match half a dataset.
    result = tv.validate_serialized_csv(
        "age,gender\r\n31,F\r\n",
        expected_columns=["age", "gender"],
        expected_row_count=1,
        removed_columns=["sex"],
        source_records=[{"sex": "F"}],
    )

    assert result["serialized_output_validation"] == tv.VALIDATION_PASSED
    assert result["leaked_value_count"] == 0


def test_unparseable_output_fails():
    result = tv.validate_serialized_csv(
        "", expected_columns=["age"], expected_row_count=0, removed_columns=[]
    )

    assert tv.FAILURE_UNPARSEABLE in result["validation_failures"]


def test_validation_result_carries_no_leaked_values():
    result = tv.validate_serialized_csv(
        f"age,notes\r\n31,{SYNTHETIC_EMAIL}\r\n",
        expected_columns=["age", "notes"],
        expected_row_count=1,
        removed_columns=["email"],
        source_records=[{"email": SYNTHETIC_EMAIL}],
    )

    assert SYNTHETIC_EMAIL not in json.dumps(result)


# ---------------------------------------------------------------------------
# The pipeline runs validation on every call
# ---------------------------------------------------------------------------


def test_pipeline_reports_serialized_output_validation():
    result = anonymize_tabular_csv(CSV_CONTENT, k=2, l=2)

    assert result["serialized_output_validation"] == tv.VALIDATION_PASSED
    assert result["validation_failures"] == []
    assert result["anonymization_status"] != "failed_privacy_validation"


def test_validation_failure_downgrades_the_pipeline_status(monkeypatch):
    from services import tabular_anonymization

    monkeypatch.setattr(
        tabular_anonymization,
        "validate_serialized_csv",
        lambda *_args, **_kwargs: {
            "serialized_output_validation": tv.VALIDATION_FAILED,
            "validation_failures": [tv.FAILURE_REMOVED_COLUMN_PRESENT],
            "leaked_columns": ["name"],
            "leaked_value_count": 0,
        },
    )

    result = tabular_anonymization.anonymize_tabular_csv(CSV_CONTENT, k=2, l=2)

    assert result["anonymization_status"] == "failed_privacy_validation"
    assert tv.FAILURE_REMOVED_COLUMN_PRESENT in result["warnings"]


def test_ingest_route_reports_validation_and_stays_blocked():
    response = route_for_ingestion(
        filename="sample.csv",
        content_type="text/csv",
        header=CSV_CONTENT[:4096],
        profile="strict",
        file_content=CSV_CONTENT,
    )

    assert response["serialized_output_validation"] == tv.VALIDATION_PASSED
    decision = response["release_decision"]
    assert decision["releasable"] is False
    assert "serialized_output_validation_passed" in decision["reason_codes"]
    assert "tabular_release_policy_review_pending" in decision["reason_codes"]
    # The route stays summary-only: no rows in the response.
    assert SYNTHETIC_NAME not in json.dumps(response)
    assert SYNTHETIC_MRN not in json.dumps(response)


# ---------------------------------------------------------------------------
# /anonymize_csv policy gate
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def api_client():
    from fastapi.testclient import TestClient

    from main import app

    return TestClient(app)


def test_research_profile_never_downloads_rows(api_client):
    response = api_client.post(
        "/anonymize_csv",
        files={"file": ("sample.csv", BytesIO(CSV_CONTENT), "text/csv")},
        data={"k": "2", "l": "2", "profile": "research"},
    )

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("application/json")
    body = response.json()
    assert body["anonymization_status"] == "expert_determination_required"
    assert body["release_decision"]["releasable"] is False
    for identifier in (SYNTHETIC_NAME, SYNTHETIC_EMAIL, SYNTHETIC_MRN):
        assert identifier not in response.text


def test_safe_harbor_profile_reports_analysis_but_releases_no_rows(api_client):
    response = api_client.post(
        "/anonymize_csv",
        files={"file": ("sample.csv", BytesIO(CSV_CONTENT), "text/csv")},
        data={"k": "2", "l": "2", "profile": "safe_harbor_v1"},
    )

    assert response.status_code == 422
    assert response.headers["content-type"].startswith("application/json")
    body = response.json()
    assert body["release_decision"]["releasable"] is False
    assert body["release_decision"]["artifact_sha256"] is None
    assert body["serialized_output_validation"] == "passed"
    for identifier in (SYNTHETIC_NAME, SYNTHETIC_EMAIL, SYNTHETIC_MRN):
        assert identifier not in response.text


def test_default_profile_is_still_accepted(api_client):
    response = api_client.post(
        "/anonymize_csv",
        files={"file": ("sample.csv", BytesIO(CSV_CONTENT), "text/csv")},
        data={"k": "2", "l": "2"},
    )

    assert response.status_code == 422
    assert response.json()["anonymization_status"] == "completed_with_warnings"


# ---------------------------------------------------------------------------
# Route parity: neither CSV route may release what the other blocks
# ---------------------------------------------------------------------------


def test_both_csv_routes_share_one_release_decision_function():
    # Not "two functions that agree today" — literally the same callable, so
    # the two routes cannot drift apart in a later change.
    import main
    from services import ingestion

    assert main.release_decision_for is ingestion.release_decision_for
    assert (
        ingestion.release_decision_for.__wrapped__ is ingestion._release_decision_for
        if hasattr(ingestion.release_decision_for, "__wrapped__")
        else True
    )


def test_download_route_cannot_release_what_the_ingest_route_blocks(api_client):
    """The bypass this guards against: one route handing out rows the other holds."""
    ingest = route_for_ingestion(
        filename="sample.csv",
        content_type="text/csv",
        header=CSV_CONTENT[:4096],
        profile="strict",
        file_content=CSV_CONTENT,
    )
    download = api_client.post(
        "/anonymize_csv",
        files={"file": ("sample.csv", BytesIO(CSV_CONTENT), "text/csv")},
        data={"k": "5", "l": "2", "profile": "strict"},
    )

    ingest_decision = ingest["release_decision"]
    download_decision = download.json()["release_decision"]

    # Same disposition, same policy, same reason codes, neither releasable.
    assert ingest_decision["releasable"] is False
    assert download_decision["releasable"] is False
    assert ingest_decision["disposition"] == download_decision["disposition"]
    assert ingest_decision["policy"] == download_decision["policy"]
    assert ingest_decision["reason_codes"] == download_decision["reason_codes"]
    assert ingest_decision["artifact_sha256"] is None
    assert download_decision["artifact_sha256"] is None


def test_neither_csv_route_emits_row_content(api_client):
    ingest = route_for_ingestion(
        filename="sample.csv",
        content_type="text/csv",
        header=CSV_CONTENT[:4096],
        profile="strict",
        file_content=CSV_CONTENT,
    )
    download = api_client.post(
        "/anonymize_csv",
        files={"file": ("sample.csv", BytesIO(CSV_CONTENT), "text/csv")},
        data={"k": "2", "l": "2"},
    )

    for surface in (json.dumps(ingest), download.text):
        for identifier in (SYNTHETIC_NAME, SYNTHETIC_EMAIL, SYNTHETIC_MRN):
            assert identifier not in surface
        # Generalized quasi-identifier rows are row content too.
        assert "31-32" not in surface
        assert "_internal_anonymized_csv" not in surface


def test_research_profile_blocks_identically_on_both_routes(api_client):
    ingest = route_for_ingestion(
        filename="sample.csv",
        content_type="text/csv",
        header=CSV_CONTENT[:4096],
        profile="research",
        file_content=CSV_CONTENT,
    )
    download = api_client.post(
        "/anonymize_csv",
        files={"file": ("sample.csv", BytesIO(CSV_CONTENT), "text/csv")},
        data={"k": "2", "l": "2", "profile": "research"},
    )

    assert ingest["anonymization_status"] == "expert_determination_required"
    assert download.json()["anonymization_status"] == "expert_determination_required"
    assert ingest["release_decision"]["releasable"] is False
    assert download.json()["release_decision"]["releasable"] is False


def test_unknown_profile_is_rejected(api_client):
    response = api_client.post(
        "/anonymize_csv",
        files={"file": ("sample.csv", BytesIO(CSV_CONTENT), "text/csv")},
        data={"k": "2", "l": "2", "profile": "public"},
    )

    assert response.status_code == 400


def test_failed_privacy_validation_never_downloads_rows(api_client, monkeypatch):
    import main

    original = main.anonymize_tabular_csv

    def failing(*args, **kwargs):
        result = original(*args, **kwargs)
        result["anonymization_status"] = "failed_privacy_validation"
        result["validation_failures"] = [tv.FAILURE_REMOVED_COLUMN_PRESENT]
        result["serialized_output_validation"] = tv.VALIDATION_FAILED
        return result

    monkeypatch.setattr(main, "anonymize_tabular_csv", failing)

    response = api_client.post(
        "/anonymize_csv",
        files={"file": ("sample.csv", BytesIO(CSV_CONTENT), "text/csv")},
        data={"k": "2", "l": "2"},
    )

    assert response.status_code == 422
    assert response.headers["content-type"].startswith("application/json")
    body = response.json()
    assert body["release_decision"]["releasable"] is False
    assert body["release_decision"]["artifact_sha256"] is None
    for identifier in (SYNTHETIC_NAME, SYNTHETIC_EMAIL, SYNTHETIC_MRN):
        assert identifier not in response.text


# ---------------------------------------------------------------------------
# Workbooks
# ---------------------------------------------------------------------------


def _workbook(
    rows=(("name", "age"), ("Alice Alpha", 31)),
    sheet_title="Patients",
    hidden_sheet=None,
    comment=None,
    properties=None,
    defined_name=None,
) -> bytes:
    workbook = openpyxl.Workbook()
    sheet = workbook.active
    sheet.title = sheet_title
    for row in rows:
        sheet.append(list(row))
    if comment:
        from openpyxl.comments import Comment

        sheet["A1"].comment = Comment(comment, "reviewer")
    if hidden_sheet:
        extra = workbook.create_sheet(hidden_sheet)
        extra.append(["hidden note", SYNTHETIC_MRN])
        extra.sheet_state = "hidden"
    if properties:
        for key, value in properties.items():
            setattr(workbook.properties, key, value)
    if defined_name:
        from openpyxl.workbook.defined_name import DefinedName

        workbook.defined_names.add(
            DefinedName(defined_name, attr_text=f"'{sheet_title}'!$A$1")
        )
    buffer = BytesIO()
    workbook.save(buffer)
    return buffer.getvalue()


def test_xlsx_extension_routes_to_the_workbook_modality():
    assert (
        detect_modality(
            "cohort.xlsx",
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            b"PK\x03\x04",
        )
        == "workbook"
    )


def test_workbook_mime_alone_is_enough():
    assert (
        detect_modality(
            "cohort",
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            b"PK\x03\x04",
        )
        == "workbook"
    )


def test_non_xlsx_bytes_are_rejected():
    with pytest.raises(wb.WorkbookSanitizationError):
        wb.scan_workbook_bytes(b"this is not a workbook")


def test_empty_workbook_upload_is_rejected():
    with pytest.raises(wb.WorkbookSanitizationError):
        wb.scan_workbook_bytes(b"")


def test_oversized_workbook_is_rejected():
    with pytest.raises(wb.WorkbookSanitizationError) as exc:
        wb.scan_workbook_bytes(b"PK\x03\x04" + b"0" * (wb.MAX_WORKBOOK_BYTES + 1))

    assert exc.value.status_code == 413


def test_missing_reader_blocks_instead_of_reporting_clean(monkeypatch):
    monkeypatch.setattr(wb, "_load_openpyxl", lambda: None)

    result = wb.scan_workbook_bytes(b"PK\x03\x04junk")

    assert result["anonymization_status"] == wb.STATUS_UNSCANNABLE
    assert wb.REASON_READER_UNAVAILABLE in result["unscannable_reasons"]
    assert result["entity_count"] == 0


def test_unparseable_workbook_blocks():
    result = wb.scan_workbook_bytes(b"PK\x03\x04 not really a workbook")

    assert result["anonymization_status"] == wb.STATUS_UNSCANNABLE
    assert wb.REASON_UNPARSEABLE in result["unscannable_reasons"]


def test_cell_values_are_scanned():
    result = wb.scan_workbook_bytes(
        _workbook(rows=(("name", "email"), (SYNTHETIC_NAME, SYNTHETIC_EMAIL)))
    )

    assert result["entity_count"] > 0
    assert result["detected_entities"].get("EMAIL_ADDRESS")
    assert result["workbook_summary"]["populated_cells"] == 4


def test_sheet_names_are_scanned():
    result = wb.scan_workbook_bytes(
        _workbook(rows=(("value",), (1,)), sheet_title="Alice Alpha chart")
    )

    assert result["detected_entities"].get("PERSON")


def test_hidden_sheets_are_read_and_reported():
    result = wb.scan_workbook_bytes(_workbook(hidden_sheet="Archive"))
    summary = result["workbook_summary"]

    assert summary["hidden_sheet_count"] == 1
    assert summary["sheet_count"] == 2
    assert any(sheet["hidden"] for sheet in summary["sheets"])
    # The hidden sheet's contents were scanned, not skipped.
    assert result["entity_count"] > 0


def test_cell_comments_are_scanned():
    result = wb.scan_workbook_bytes(
        _workbook(comment=f"Follow up with {SYNTHETIC_EMAIL}")
    )

    assert result["workbook_summary"]["comment_count"] == 1
    assert result["detected_entities"].get("EMAIL_ADDRESS")


def test_document_properties_are_scanned():
    result = wb.scan_workbook_bytes(
        _workbook(properties={"creator": SYNTHETIC_NAME, "title": "Cohort"})
    )
    summary = result["workbook_summary"]

    assert "creator" in summary["document_properties_present"]
    assert result["detected_entities"].get("PERSON")


def test_macro_marker_blocks():
    payload = _workbook() + b"\nvbaProject.bin\n"

    result = wb.scan_workbook_bytes(payload)

    assert result["workbook_summary"]["macros_present"] is True
    assert wb.REASON_MACROS_PRESENT in result["unscannable_reasons"]
    assert result["anonymization_status"] == wb.STATUS_UNSCANNABLE


def test_sheet_limit_blocks(monkeypatch):
    monkeypatch.setattr(wb, "MAX_WORKBOOK_SHEETS", 1)

    result = wb.scan_workbook_bytes(_workbook(hidden_sheet="Archive"))

    assert wb.REASON_SHEET_LIMIT in result["unscannable_reasons"]


def test_cell_limit_blocks(monkeypatch):
    monkeypatch.setattr(wb, "MAX_WORKBOOK_CELLS", 1)

    result = wb.scan_workbook_bytes(
        _workbook(rows=(("a", "b"), ("c", "d"), ("e", "f")))
    )

    assert wb.REASON_CELL_LIMIT in result["unscannable_reasons"]


def test_a_clean_workbook_is_still_never_releasable():
    result = wb.scan_workbook_bytes(_workbook(rows=(("value",), (1,))))

    assert result["anonymization_status"] == wb.STATUS_MANUAL_REVIEW
    assert wb.REASON_NO_VALIDATED_WRITER in result["unscannable_reasons"]


def test_workbook_result_never_carries_bytes_or_values():
    result = wb.scan_workbook_bytes(
        _workbook(rows=(("name", "email"), (SYNTHETIC_NAME, SYNTHETIC_EMAIL)))
    )
    serialized = json.dumps(result)

    assert SYNTHETIC_NAME not in serialized
    assert SYNTHETIC_EMAIL not in serialized
    assert "PK" not in serialized


def test_workbook_ingestion_routes_to_manual_review():
    payload = _workbook(rows=(("value",), (1,)))

    response = route_for_ingestion(
        filename="cohort.xlsx",
        content_type=(
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        ),
        header=payload[:4096],
        profile="strict",
        file_content=payload,
    )

    assert response["detected_modality"] == "workbook"
    assert response["release_decision"]["releasable"] is False
    assert response["release_decision"]["artifact_sha256"] is None
    assert wb.REASON_NO_VALIDATED_WRITER in response["unscannable_reasons"]
    assert all(value == "blocked" for value in response["downstream"].values())


def test_research_profile_workbook_returns_expert_determination():
    payload = _workbook(rows=(("name",), (SYNTHETIC_NAME,)))

    response = route_for_ingestion(
        filename="cohort.xlsx",
        content_type=(
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        ),
        header=payload[:4096],
        profile="research",
        file_content=payload,
    )

    assert response["anonymization_status"] == "expert_determination_required"
    assert response["release_decision"]["releasable"] is False
    assert SYNTHETIC_NAME not in json.dumps(response)
