import csv
import io
import json
import os
import sys
from io import BytesIO
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from main import app  # noqa: E402
from services.tabular_anonymization import anonymize_tabular_csv  # noqa: E402

client = TestClient(app)


REPORT_KEYS = {
    "safe_harbor_validation_status",
    "identifier_categories_detected",
    "identifier_categories_removed",
    "unresolved_identifier_categories",
    "date_compliance_status",
    "age_over_89_status",
    "geographic_compliance_status",
    "free_text_scan_status",
    "unique_code_scan_status",
    "actual_knowledge_review_required",
    "warnings",
}

DIRECT_CATEGORY_CASES = (
    ("relative_name", ["Alice Alpha", "Bob Beta", "Carol Gamma", "Dan Delta"], "1_names"),
    ("city", ["Boston", "Cambridge", "Quincy", "Salem"], "2_geographic_subdivisions"),
    ("telephone_number", ["617-555-0101", "617-555-0102", "617-555-0103", "617-555-0104"], "5_telephone_numbers"),
    ("fax_number", ["617-555-0201", "617-555-0202", "617-555-0203", "617-555-0204"], "6_fax_numbers"),
    ("email_address", ["a@example.test", "b@example.test", "c@example.test", "d@example.test"], "7_email_addresses"),
    ("ssn", ["111-22-3333", "222-33-4444", "333-44-5555", "444-55-6666"], "8_social_security_numbers"),
    ("mrn", ["MRN-A001", "MRN-B002", "MRN-C003", "MRN-D004"], "9_medical_record_numbers"),
    ("beneficiary_number", ["PLAN-A001", "PLAN-B002", "PLAN-C003", "PLAN-D004"], "10_health_plan_beneficiary_numbers"),
    ("account_number", ["ACCT-A001", "ACCT-B002", "ACCT-C003", "ACCT-D004"], "11_account_numbers"),
    ("license_number", ["LIC-A001", "LIC-B002", "LIC-C003", "LIC-D004"], "12_certificate_and_licence_numbers"),
    ("vin", ["1HGCM82633A004351", "1HGCM82633A004352", "1HGCM82633A004353", "1HGCM82633A004354"], "13_vehicle_identifiers"),
    ("device_serial_number", ["DEV-A001", "DEV-B002", "DEV-C003", "DEV-D004"], "14_device_identifiers"),
    ("url", ["https://one.test/a", "https://two.test/b", "https://three.test/c", "https://four.test/d"], "15_urls"),
    ("ip_address", ["192.0.2.1", "192.0.2.2", "192.0.2.3", "192.0.2.4"], "16_ip_addresses"),
    ("fingerprint_id", ["BIO-A001", "BIO-B002", "BIO-C003", "BIO-D004"], "17_biometric_identifiers"),
    ("unique_id", ["550e8400-e29b-41d4-a716-446655440001", "550e8400-e29b-41d4-a716-446655440002", "550e8400-e29b-41d4-a716-446655440003", "550e8400-e29b-41d4-a716-446655440004"], "18_photographs_and_unique_identifiers"),
)


def make_csv(extra_column, values, include_sensitive=True):
    buffer = io.StringIO()
    columns = [extra_column]
    if extra_column != "age":
        columns.append("age")
    columns.append("gender")
    if include_sensitive:
        columns.append("diagnosis")
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(columns)
    for index, value in enumerate(values):
        row = [value]
        if extra_column != "age":
            row.append(30 + index)
        row.append("F" if index % 2 == 0 else "M")
        if include_sensitive:
            row.append("alpha" if index % 2 == 0 else "beta")
        writer.writerow(row)
    return buffer.getvalue().encode("utf-8")


def output_rows(result):
    return list(csv.DictReader(io.StringIO(result["_internal_anonymized_csv"])))


@pytest.mark.parametrize("column,values,category", DIRECT_CATEGORY_CASES)
def test_each_direct_safe_harbor_category_is_removed(column, values, category):
    result = anonymize_tabular_csv(
        make_csv(column, values),
        k=2,
        l=2,
        include_anonymized_csv=True,
    )
    report = result["safe_harbor_report"]
    serialized = json.dumps(result)

    assert column not in result["output_columns"]
    assert category in report["identifier_categories_detected"]
    assert category in report["identifier_categories_removed"]
    assert report["unresolved_identifier_categories"] == []
    assert result["anonymization_status"] == "completed_with_warnings"
    for value in values:
        assert value not in serialized


def test_safe_harbor_column_matching_is_case_insensitive():
    values = ["a@example.test", "b@example.test", "c@example.test", "d@example.test"]
    result = anonymize_tabular_csv(
        make_csv("EMAIL_ADDRESS", values),
        k=2,
        include_anonymized_csv=True,
    )

    assert "EMAIL_ADDRESS" not in result["output_columns"]
    assert "7_email_addresses" in result["safe_harbor_report"][
        "identifier_categories_removed"
    ]


@pytest.mark.parametrize(
    "column",
    [
        "patient_name",
        "relative_name",
        "employer_name",
        "provider_name",
        "household_member_name",
    ],
)
def test_all_subject_identifying_name_aliases_are_removed(column):
    values = ["Alice Alpha", "Bob Beta", "Carol Gamma", "Dan Delta"]
    result = anonymize_tabular_csv(
        make_csv(column, values),
        k=2,
        include_anonymized_csv=True,
    )

    assert column not in result["output_columns"]
    assert "1_names" in result["safe_harbor_report"][
        "identifier_categories_removed"
    ]


@pytest.mark.parametrize(
    "column,values",
    [
        ("address", ["1 Main Street", "2 Main Street", "3 Main Street", "4 Main Street"]),
        ("city", ["Boston", "Quincy", "Salem", "Cambridge"]),
        ("county", ["Suffolk", "Norfolk", "Essex", "Middlesex"]),
        ("precinct", ["P-01", "P-02", "P-03", "P-04"]),
        ("zip", ["02139", "02140", "94107", "94110"]),
        ("latitude", ["42.1", "42.2", "42.3", "42.4"]),
        ("longitude", ["-71.1", "-71.2", "-71.3", "-71.4"]),
        ("geocode", ["GEO-A", "GEO-B", "GEO-C", "GEO-D"]),
        ("birthplace", ["Boston MA", "Quincy MA", "Salem MA", "Cambridge MA"]),
    ],
)
def test_prohibited_geographic_detail_is_removed(column, values):
    result = anonymize_tabular_csv(
        make_csv(column, values),
        k=2,
        include_anonymized_csv=True,
    )

    assert column not in result["output_columns"]
    assert result["safe_harbor_report"]["geographic_compliance_status"] == "passed"
    for value in values:
        assert value not in result["_internal_anonymized_csv"]


def test_state_may_be_retained_but_is_still_generalized_for_k_anonymity():
    result = anonymize_tabular_csv(
        make_csv("state", ["MA", "MA", "CA", "CA"]),
        k=2,
        include_anonymized_csv=True,
    )

    assert "state" in result["output_columns"]
    assert set(row["state"] for row in output_rows(result)) == {"*"}
    assert result["safe_harbor_report"]["geographic_compliance_status"] == "passed"


@pytest.mark.parametrize(
    "column",
    [
        "birth_date",
        "admission_date",
        "discharge_date",
        "service_date",
        "procedure_date",
        "death_date",
    ],
)
def test_individual_related_dates_retain_year_only(column):
    exact_dates = ["1970-01-02", "1980-03-04", "1990-05-06", "2000-07-08"]
    result = anonymize_tabular_csv(
        make_csv(column, exact_dates),
        k=2,
        include_anonymized_csv=True,
    )
    output = result["_internal_anonymized_csv"]
    report = result["safe_harbor_report"]

    assert column in result["output_columns"]
    assert report["date_compliance_status"] == "passed"
    assert "3_individual_related_dates" in report["identifier_categories_detected"]
    for exact_date in exact_dates:
        assert exact_date not in output
    assert all(
        value == "UNKNOWN" or not value.count("-") == 2
        for value in (row[column] for row in output_rows(result))
    )


def test_ages_over_89_are_grouped_as_90_plus():
    values = ["45", "88", "91", "103"]
    result = anonymize_tabular_csv(
        make_csv("age", values),
        k=2,
        include_anonymized_csv=True,
    )
    output = result["_internal_anonymized_csv"]
    report = result["safe_harbor_report"]

    assert "90+" in output
    assert "91" not in output
    assert "103" not in output
    assert report["age_over_89_status"] == "passed"
    assert "4_ages_over_89" in report["identifier_categories_detected"]


def test_birth_year_revealing_age_over_89_is_grouped_as_90_plus():
    csv_bytes = (
        b"birth_date,age,gender\n"
        b"1920-01-02,95,F\n"
        b"1925-03-04,101,M\n"
        b"1980-05-06,45,F\n"
        b"1981-07-08,44,M\n"
    )
    result = anonymize_tabular_csv(
        csv_bytes,
        k=2,
        include_anonymized_csv=True,
    )
    output = result["_internal_anonymized_csv"]

    assert "90+" in output
    assert "1920" not in output
    assert "1925" not in output
    assert "95" not in output
    assert "101" not in output
    assert result["safe_harbor_report"]["age_over_89_status"] == "passed"


@pytest.mark.parametrize(
    "values,category",
    [
        (["a@one.test", "b@two.test", "c@three.test", "d@four.test"], "7_email_addresses"),
        (["617-555-0101", "617-555-0102", "617-555-0103", "617-555-0104"], "5_telephone_numbers"),
        (["111-22-3333", "222-33-4444", "333-44-5555", "444-55-6666"], "8_social_security_numbers"),
        (["https://one.test", "https://two.test", "https://three.test", "https://four.test"], "15_urls"),
        (["192.0.2.1", "192.0.2.2", "192.0.2.3", "192.0.2.4"], "16_ip_addresses"),
        (["550e8400-e29b-41d4-a716-446655440001", "550e8400-e29b-41d4-a716-446655440002", "550e8400-e29b-41d4-a716-446655440003", "550e8400-e29b-41d4-a716-446655440004"], "18_photographs_and_unique_identifiers"),
        (["1 Main Street", "2 Main Street", "3 Main Street", "4 Main Street"], "2_geographic_subdivisions"),
        (["1970-01-02", "1980-03-04", "1990-05-06", "2000-07-08"], "3_individual_related_dates"),
    ],
)
def test_data_patterns_are_detected_with_unfamiliar_column_names(values, category):
    result = anonymize_tabular_csv(
        make_csv("unfamiliar_field", values),
        k=2,
        include_anonymized_csv=True,
    )
    serialized = json.dumps(result)

    assert category in result["safe_harbor_report"][
        "identifier_categories_detected"
    ]
    for value in values:
        assert value not in serialized


def test_unfamiliar_title_case_name_column_is_removed_by_value_pattern():
    values = ["Alice Alpha", "Bob Beta", "Carol Gamma", "Dan Delta"]
    result = anonymize_tabular_csv(
        make_csv("unfamiliar_field", values),
        k=2,
        include_anonymized_csv=True,
    )

    assert "unfamiliar_field" not in result["output_columns"]
    assert "1_names" in result["safe_harbor_report"][
        "identifier_categories_removed"
    ]


def test_configured_mapping_removes_custom_identifier_column():
    values = ["REF-A", "REF-B", "REF-C", "REF-D"]
    result = anonymize_tabular_csv(
        make_csv("custom_reference", values),
        k=2,
        safe_harbor_mappings={"11_account_numbers": ["custom_reference"]},
        include_anonymized_csv=True,
    )

    assert "custom_reference" not in result["output_columns"]
    assert "11_account_numbers" in result["safe_harbor_report"][
        "identifier_categories_removed"
    ]


def test_free_text_with_phi_is_removed_after_existing_text_service_scan():
    values = [
        "Patient Alice Alpha phone 617-555-0101",
        "Patient Bob Beta phone 617-555-0102",
        "Patient Carol Gamma phone 617-555-0103",
        "Patient Dan Delta phone 617-555-0104",
    ]
    result = anonymize_tabular_csv(
        make_csv("notes", values),
        k=2,
        include_anonymized_csv=True,
    )
    serialized = json.dumps(result)

    assert "notes" not in result["output_columns"]
    assert result["safe_harbor_report"]["free_text_scan_status"] == "passed"
    for value in values:
        assert value not in serialized


def test_scanned_free_text_without_detected_phi_may_be_preserved():
    values = [
        "routine follow up stable",
        "routine follow up improving",
        "routine follow up unchanged",
        "routine follow up complete",
    ]
    result = anonymize_tabular_csv(
        make_csv("notes", values),
        k=2,
        include_anonymized_csv=True,
    )

    assert "notes" in result["output_columns"]
    assert result["safe_harbor_report"]["free_text_scan_status"] == "passed"


def test_unknown_high_cardinality_code_is_removed_for_review():
    values = ["ZXA-001", "ZXA-002", "ZXA-003", "ZXA-004"]
    result = anonymize_tabular_csv(
        make_csv("unfamiliar_code_field", values),
        k=2,
        include_anonymized_csv=True,
    )
    report = result["safe_harbor_report"]

    assert "unfamiliar_code_field" not in result["output_columns"]
    assert report["unique_code_scan_status"] == "passed_with_removals"
    assert report["actual_knowledge_review_required"] is True
    assert "18_photographs_and_unique_identifiers" in report[
        "identifier_categories_removed"
    ]


def test_unknown_high_cardinality_numeric_identifier_is_removed():
    result = anonymize_tabular_csv(
        make_csv("unfamiliar_numeric_field", ["101", "202", "303", "404"]),
        k=2,
        include_anonymized_csv=True,
    )

    assert "unfamiliar_numeric_field" not in result["output_columns"]
    assert result["safe_harbor_report"][
        "unique_code_scan_status"
    ] == "passed_with_removals"


@pytest.mark.parametrize(
    "column,values,category",
    [
        ("binary", ["opaque-a", "opaque-b", "opaque-c", "opaque-d"], "18_photographs_and_unique_identifiers"),
        ("voiceprint_id", ["VOICE-A", "VOICE-B", "VOICE-C", "VOICE-D"], "17_biometric_identifiers"),
        ("unknown_blob", ["data:image/png;base64,AAAA", "data:image/png;base64,BBBB", "data:image/png;base64,CCCC", "data:image/png;base64,DDDD"], "18_photographs_and_unique_identifiers"),
    ],
)
def test_binary_image_and_biometric_columns_are_quarantined(column, values, category):
    result = anonymize_tabular_csv(
        make_csv(column, values),
        k=2,
        include_anonymized_csv=True,
    )

    assert column not in result["output_columns"]
    assert category in result["safe_harbor_report"][
        "identifier_categories_removed"
    ]


def test_unavailable_free_text_scanner_removes_column_instead_of_preserving_it():
    from services.text_anonymization import TextAnonymizationError

    with patch(
        "services.safe_harbor.detect_clinical_phi",
        side_effect=TextAnonymizationError("scanner unavailable", status_code=503),
    ):
        result = anonymize_tabular_csv(
            make_csv("notes", ["safe a", "safe b", "safe c", "safe d"]),
            k=2,
            include_anonymized_csv=True,
        )

    assert "notes" not in result["output_columns"]
    assert result["safe_harbor_report"][
        "free_text_scan_status"
    ] == "passed_with_quarantine"


def test_safe_harbor_report_contains_only_safe_metadata_fields():
    raw_values = ["111-22-3333", "222-33-4444", "333-44-5555", "444-55-6666"]
    result = anonymize_tabular_csv(make_csv("ssn", raw_values), k=2)
    report = result["safe_harbor_report"]
    serialized = json.dumps(report)

    assert set(report) == REPORT_KEYS
    assert report["safe_harbor_validation_status"] == "passed_with_warnings"
    assert report["actual_knowledge_review_required"] is True
    assert any("legal review" in warning for warning in report["warnings"])
    for value in raw_values:
        assert value not in serialized


def test_failed_k_or_l_never_reports_completed():
    k_failure = anonymize_tabular_csv(
        b"age,gender\n30,F\n31,M\n",
        k=5,
    )
    l_failure = anonymize_tabular_csv(
        b"age,gender,diagnosis\n30,F,alpha\n31,F,alpha\n40,M,alpha\n41,M,alpha\n",
        k=2,
        l=2,
    )

    assert k_failure["anonymization_status"] == "failed_privacy_validation"
    assert l_failure["anonymization_status"] == "failed_privacy_validation"


def test_unresolved_safe_harbor_category_forces_failed_privacy_validation():
    forced_report = {
        "safe_harbor_validation_status": "failed",
        "identifier_categories_detected": ["18_photographs_and_unique_identifiers"],
        "identifier_categories_removed": [],
        "unresolved_identifier_categories": ["18_photographs_and_unique_identifiers"],
        "date_compliance_status": "passed",
        "age_over_89_status": "passed",
        "geographic_compliance_status": "passed",
        "free_text_scan_status": "passed",
        "unique_code_scan_status": "failed",
        "actual_knowledge_review_required": True,
        "warnings": ["Technical review required."],
    }
    with patch(
        "services.tabular_anonymization.build_safe_harbor_report",
        return_value=forced_report,
    ):
        result = anonymize_tabular_csv(
            b"age,gender\n30,F\n31,M\n",
            k=2,
        )

    assert result["k_anonymity_satisfied"] is True
    assert result["anonymization_status"] == "failed_privacy_validation"


def test_api_does_not_leak_identifier_in_body_headers_filename_log_or_error():
    csv_content = (
        b"age,gender,notes\n"
        b"30,F,Patient Alice Alpha SSN 111-22-3333\n"
        b"31,M,Patient Bob Beta SSN 222-33-4444\n"
        b"32,F,Patient Carol Gamma SSN 333-44-5555\n"
        b"33,M,Patient Dan Delta SSN 444-55-6666\n"
    )
    logged_details = []

    with patch(
        "main.audit_logger.log_operation",
        side_effect=lambda operation, details: logged_details.append(details),
    ):
        response = client.post(
            "/anonymize_csv",
            files={
                "file": (
                    "111-22-3333.csv",
                    BytesIO(csv_content),
                    "text/csv",
                )
            },
            data={"k": "2", "l": "2"},
        )

    response_surface = (
        response.content.decode("utf-8")
        + json.dumps(dict(response.headers))
        + json.dumps(logged_details)
    )
    assert response.status_code == 200
    assert "anonymized_dataset.csv" in response.headers["content-disposition"]
    assert response.headers["x-bioblock-safe-harbor-status"] == "passed_with_warnings"
    for identifier in (
        "111-22-3333",
        "222-33-4444",
        "Alice Alpha",
        "Bob Beta",
    ):
        assert identifier not in response_surface

    error_response = client.post(
        "/anonymize_csv",
        files={"file": ("dataset.csv", BytesIO(csv_content), "text/csv")},
        data={
            "k": "2",
            "safe_harbor_mappings": json.dumps(
                {"11_account_numbers": ["missing_column"]}
            ),
        },
    )
    assert error_response.status_code == 400
    assert "111-22-3333" not in error_response.text
