import csv
import io
import json
import os
import re
import sys
from collections import Counter

import pytest

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services.tabular_anonymization import (  # noqa: E402
    TabularAnonymizationError,
    anonymize_tabular_csv,
    classify_tabular_columns,
)


DIRECT_COLUMNS = {
    "Id",
    "SSN",
    "DRIVERS",
    "PASSPORT",
    "PREFIX",
    "FIRST",
    "LAST",
    "SUFFIX",
    "MAIDEN",
    "ADDRESS",
}

QUASI_COLUMNS = {
    "BIRTHDATE",
    "DEATHDATE",
    "GENDER",
    "RACE",
    "ETHNICITY",
    "MARITAL",
    "BIRTHPLACE",
    "CITY",
    "STATE",
    "COUNTY",
    "ZIP",
    "LAT",
    "LON",
}

SYNTHETIC_SYNTHEA_CSV = """Id,BIRTHDATE,DEATHDATE,SSN,DRIVERS,PASSPORT,PREFIX,FIRST,LAST,SUFFIX,MAIDEN,MARITAL,RACE,ETHNICITY,GENDER,BIRTHPLACE,ADDRESS,CITY,STATE,COUNTY,ZIP,LAT,LON,HEALTHCARE_EXPENSES,HEALTHCARE_COVERAGE
fake-uuid-001,1980-01-04,,111-22-3333,DL-FAKE-01,P-FAKE-01,Ms.,Alicia,Alpha,Jr.,MaidenA,M,white,nonhispanic,F,Boston MA,101 Fake Street,Boston,MA,Suffolk,02139,42.3601,-71.0589,1000.00,600.00
fake-uuid-002,1981-01-05,,222-33-4444,DL-FAKE-02,P-FAKE-02,Mr.,Boris,Beta,Sr.,MaidenB,M,white,nonhispanic,M,Boston MA,202 Fake Avenue,Boston,MA,Suffolk,02140,42.3611,-71.0599,1100.00,650.00
fake-uuid-003,1990-06-14,2024-02-01,333-44-5555,DL-FAKE-03,P-FAKE-03,Dr.,Carla,Gamma,III,MaidenC,S,black,hispanic,F,San Francisco CA,303 Fake Road,San Francisco,CA,San Francisco,94107,37.7749,-122.4194,1200.00,700.00
fake-uuid-004,1991-06-15,2024-02-02,444-55-6666,DL-FAKE-04,P-FAKE-04,Mx.,Dario,Delta,IV,MaidenD,S,black,hispanic,M,San Francisco CA,404 Fake Lane,San Francisco,CA,San Francisco,94110,37.7759,-122.4184,1300.00,750.00
""".encode("utf-8")


def internal_rows(result):
    return list(csv.DictReader(io.StringIO(result["_internal_anonymized_csv"])))


def independently_group(rows, quasi_identifiers):
    return Counter(
        tuple(row[column] for column in quasi_identifiers)
        for row in rows
    )


def test_synthea_column_classification_is_case_insensitive():
    upper = classify_tabular_columns([*DIRECT_COLUMNS, *QUASI_COLUMNS])
    lower_header = [column.lower() for column in [*DIRECT_COLUMNS, *QUASI_COLUMNS]]
    lower = classify_tabular_columns(lower_header)

    assert set(upper["direct_identifiers"]) == DIRECT_COLUMNS
    assert set(upper["quasi_identifiers"]) == QUASI_COLUMNS
    assert set(lower["direct_identifiers"]) == {
        column.lower() for column in DIRECT_COLUMNS
    }
    assert set(lower["quasi_identifiers"]) == {
        column.lower() for column in QUASI_COLUMNS
    }


def test_column_classification_supports_common_aliases():
    header = [
        "patient_id",
        "first_name",
        "family_name",
        "driver_license",
        "passport_number",
        "street_address",
        "birth_date",
        "death_date",
        "marital_status",
        "birth_place",
        "postal_code",
        "latitude",
        "longitude",
    ]

    classified = classify_tabular_columns(header)

    assert set(classified["direct_identifiers"]) == set(header[:6])
    assert set(classified["quasi_identifiers"]) == set(header[6:])
    assert classified["precise_geography"] == ["latitude", "longitude"]


def test_direct_identifiers_and_exact_coordinates_are_removed_without_leakage():
    result = anonymize_tabular_csv(
        SYNTHETIC_SYNTHEA_CSV,
        k=2,
        include_anonymized_csv=True,
    )
    output = result["_internal_anonymized_csv"]
    rows = internal_rows(result)

    assert DIRECT_COLUMNS.isdisjoint(rows[0])
    assert {"LAT", "LON"}.isdisjoint(rows[0])
    assert set(result["direct_identifiers_removed"]) == DIRECT_COLUMNS
    assert result["precise_geography_columns_removed"] == ["LAT", "LON"]
    for fake_value in (
        "fake-uuid-001",
        "111-22-3333",
        "DL-FAKE-01",
        "P-FAKE-01",
        "Alicia",
        "Alpha",
        "MaidenA",
        "101 Fake Street",
        "42.3601",
        "-71.0589",
    ):
        assert fake_value not in output


def test_dates_are_generalized_and_missing_deathdate_is_safe():
    result = anonymize_tabular_csv(
        SYNTHETIC_SYNTHEA_CSV,
        k=2,
        include_anonymized_csv=True,
    )
    rows = internal_rows(result)
    output = result["_internal_anonymized_csv"]

    assert all(not re.fullmatch(r"\d{4}-\d{2}-\d{2}", row["BIRTHDATE"]) for row in rows)
    assert all(not re.fullmatch(r"\d{4}-\d{2}-\d{2}", row["DEATHDATE"]) for row in rows)
    assert "UNKNOWN" in {row["DEATHDATE"] for row in rows}
    for exact_date in ("1980-01-04", "1981-01-05", "2024-02-01", "2024-02-02"):
        assert exact_date not in output


def test_invalid_date_becomes_unknown_with_safe_warning():
    csv_bytes = b"Id,BIRTHDATE,GENDER\na,not-a-date,F\nb,2001-02-03,M\n"

    result = anonymize_tabular_csv(
        csv_bytes,
        k=2,
        include_anonymized_csv=True,
    )

    assert "not-a-date" not in result["_internal_anonymized_csv"]
    assert all(row["BIRTHDATE"] == "UNKNOWN" for row in internal_rows(result))
    assert any("1 invalid date" in warning for warning in result["warnings"])


def test_geography_is_removed_generalized_or_suppressed_consistently():
    result = anonymize_tabular_csv(
        SYNTHETIC_SYNTHEA_CSV,
        k=2,
        include_anonymized_csv=True,
    )
    rows = internal_rows(result)
    output = result["_internal_anonymized_csv"]

    assert {"ADDRESS", "CITY", "COUNTY", "BIRTHPLACE", "ZIP", "LAT", "LON"}.isdisjoint(rows[0])
    assert set(row["STATE"] for row in rows) <= {"*", "UNKNOWN"}
    for raw_value in (
        "02139",
        "94107",
        "Boston",
        "Suffolk",
        "101 Fake Street",
        "42.3601",
        "-122.4194",
    ):
        assert raw_value not in output


def test_reported_k_metrics_match_independent_csv_calculation():
    result = anonymize_tabular_csv(
        SYNTHETIC_SYNTHEA_CSV,
        k=2,
        include_anonymized_csv=True,
    )
    rows = internal_rows(result)
    groups = independently_group(rows, result["quasi_identifiers_used"])
    independent_minimum = min(groups.values())

    assert independent_minimum >= 2
    assert result["min_group_size"] == independent_minimum
    assert result["equivalence_classes"] == len(groups)
    assert result["k_anonymity_satisfied"] is True
    assert result["anonymization_status"] == "completed_with_warnings"


def test_k_failure_cannot_report_completed_or_satisfied():
    csv_bytes = b"Id,age,gender\na,30,F\nb,31,M\nc,32,F\n"

    result = anonymize_tabular_csv(
        csv_bytes,
        k=5,
        include_anonymized_csv=True,
    )
    rows = internal_rows(result)
    groups = independently_group(rows, result["quasi_identifiers_used"])

    assert min(groups.values()) == 3
    assert result["min_group_size"] == 3
    assert result["k_anonymity_satisfied"] is False
    assert result["anonymization_status"] == "failed_privacy_validation"
    assert any("k-anonymity" in warning for warning in result["warnings"])


def test_l_diversity_passes_when_sensitive_column_is_diverse():
    csv_bytes = (
        b"Id,age,gender,diagnosis\n"
        b"a,30,F,alpha\n"
        b"b,31,F,beta\n"
        b"c,40,M,alpha\n"
        b"d,41,M,beta\n"
    )

    result = anonymize_tabular_csv(csv_bytes, k=2, l=2)

    assert result["sensitive_column"] == "diagnosis"
    assert result["l_diversity_satisfied"] is True
    assert result["anonymization_status"] == "completed_with_warnings"


def test_l_diversity_failure_is_truthful_and_not_completed():
    csv_bytes = (
        b"Id,age,gender,diagnosis\n"
        b"a,30,F,alpha\n"
        b"b,31,F,alpha\n"
        b"c,40,M,alpha\n"
        b"d,41,M,alpha\n"
    )

    result = anonymize_tabular_csv(csv_bytes, k=2, l=2)

    assert result["k_anonymity_satisfied"] is True
    assert result["l_diversity_satisfied"] is False
    assert result["anonymization_status"] == "failed_privacy_validation"
    assert any("l-diversity" in warning for warning in result["warnings"])


def test_l_diversity_is_not_applicable_without_sensitive_column():
    csv_bytes = b"Id,age,gender\na,30,F\nb,31,F\nc,32,M\nd,33,M\n"

    result = anonymize_tabular_csv(csv_bytes, k=2, l=2)

    assert result["sensitive_column"] is None
    assert result["l_diversity_satisfied"] == "not_applicable"
    assert result["l_diversity_satisfied"] is not True
    assert any("not evaluated" in warning for warning in result["warnings"])


@pytest.mark.parametrize("sensitive_column", ["HEALTHCARE_EXPENSES", "HEALTHCARE_COVERAGE"])
def test_raw_numerical_sensitive_columns_require_bucketing(sensitive_column):
    with pytest.raises(TabularAnonymizationError) as exc:
        anonymize_tabular_csv(
            SYNTHETIC_SYNTHEA_CSV,
            k=2,
            sensitive_column=sensitive_column,
        )

    assert "requires bucketing" in exc.value.detail
    assert "pre-bucketed categorical" in exc.value.detail


def test_safe_summary_does_not_expose_rows_or_raw_sensitive_values():
    csv_bytes = (
        b"Id,SSN,FIRST,ADDRESS,age,gender,diagnosis\n"
        b"uuid-secret-1,111-22-3333,NameSecret,1 Secret Road,30,F,RareAlpha\n"
        b"uuid-secret-2,222-33-4444,OtherSecret,2 Secret Road,31,M,RareBeta\n"
    )

    result = anonymize_tabular_csv(csv_bytes, k=2, l=2)
    response_text = json.dumps(result)

    assert "_internal_anonymized_csv" not in result
    for raw_value in (
        "uuid-secret-1",
        "111-22-3333",
        "NameSecret",
        "1 Secret Road",
        "RareAlpha",
        "RareBeta",
    ):
        assert raw_value not in response_text


def test_rows_columns_missing_values_and_analytical_data_are_preserved():
    csv_bytes = (
        b"Id,score,gender,lab_result,notes\n"
        b"a,-10,F,7.1,cohort-a\n"
        b"b,-5,F,,cohort-b\n"
        b"c,10,M,8.2,cohort-c\n"
        b"d,20,M,9.3,cohort-d\n"
    )

    result = anonymize_tabular_csv(
        csv_bytes,
        k=2,
        quasi_identifiers=["score", "gender"],
        include_anonymized_csv=True,
    )
    rows = internal_rows(result)

    assert result["rows_in"] == result["rows_out"] == 4
    assert result["output_columns"] == ["score", "gender", "lab_result", "notes"]
    assert list(rows[0]) == result["output_columns"]
    assert [row["notes"] for row in rows] == [
        "cohort-a",
        "cohort-b",
        "cohort-c",
        "cohort-d",
    ]
    assert rows[1]["lab_result"] == "UNKNOWN"
    assert all("--" not in row["score"] for row in rows)
    assert any(" to " in row["score"] for row in rows)
    assert 0.0 <= result["generalization_rate"] <= 1.0
    assert 0.0 <= result["suppression_rate"] <= 1.0


def test_empty_and_invalid_csv_are_rejected_clearly():
    with pytest.raises(TabularAnonymizationError, match="empty"):
        anonymize_tabular_csv(b"  \n")
    with pytest.raises(TabularAnonymizationError, match="Invalid CSV"):
        anonymize_tabular_csv(b'age,diagnosis\n"42,flu\n')


def test_missing_required_columns_are_rejected_clearly():
    with pytest.raises(TabularAnonymizationError, match="Missing quasi-identifier"):
        anonymize_tabular_csv(
            b"diagnosis\nflu\n",
            quasi_identifiers=["age"],
        )
    with pytest.raises(TabularAnonymizationError, match="Missing sensitive column"):
        anonymize_tabular_csv(
            b"age,diagnosis\n40,flu\n41,cold\n",
            sensitive_column="outcome",
        )
