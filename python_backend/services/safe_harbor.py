from __future__ import annotations

import math
import re
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Mapping, Optional, Sequence, Set

from services.text_anonymization import (
    TextAnonymizationError,
    detect_clinical_phi,
)


UNKNOWN_VALUE = "UNKNOWN"

CATEGORY_LABELS = {
    1: "1_names",
    2: "2_geographic_subdivisions",
    3: "3_individual_related_dates",
    4: "4_ages_over_89",
    5: "5_telephone_numbers",
    6: "6_fax_numbers",
    7: "7_email_addresses",
    8: "8_social_security_numbers",
    9: "9_medical_record_numbers",
    10: "10_health_plan_beneficiary_numbers",
    11: "11_account_numbers",
    12: "12_certificate_and_licence_numbers",
    13: "13_vehicle_identifiers",
    14: "14_device_identifiers",
    15: "15_urls",
    16: "16_ip_addresses",
    17: "17_biometric_identifiers",
    18: "18_photographs_and_unique_identifiers",
}

CATEGORY_ALIASES = {
    1: {
        "name", "full_name", "patient_name", "first", "first_name",
        "last", "last_name", "maiden", "maiden_name", "prefix", "suffix",
        "relative_name", "employer_name", "provider_name", "physician_name",
        "household_member_name", "guardian_name",
    },
    2: {
        "address", "street", "street_address", "city", "county", "precinct",
        "zip", "zip_code", "postal_code", "lat", "latitude", "lon",
        "longitude", "geocode", "geo_code", "birthplace", "birth_place",
    },
    3: {
        "date", "birthdate", "birth_date", "dob", "admission_date",
        "admit_date", "discharge_date", "service_date", "procedure_date",
        "diagnosis_date", "deathdate", "death_date", "dod",
    },
    4: {"age", "patient_age", "age_years"},
    5: {"phone", "phone_number", "telephone", "telephone_number", "mobile"},
    6: {"fax", "fax_number", "facsimile"},
    7: {"email", "email_address"},
    8: {"ssn", "social_security_number"},
    9: {"mrn", "medical_record_number", "chart_number", "hospital_number"},
    10: {
        "health_plan_id", "health_plan_number", "beneficiary_number",
        "beneficiary_id", "insurance_id", "member_id", "subscriber_id",
        "policy_number",
    },
    11: {"account", "account_number", "billing_account", "claim_account"},
    12: {
        "certificate_number", "licence_number", "license_number", "drivers",
        "driver_license", "drivers_license", "professional_license",
    },
    13: {
        "vehicle_id", "vehicle_identifier", "vin", "vehicle_serial_number",
        "license_plate", "licence_plate", "plate_number",
    },
    14: {
        "device_id", "device_identifier", "device_serial_number",
        "serial_number", "implant_id", "udi",
    },
    15: {"url", "website", "web_address"},
    16: {"ip", "ip_address", "ipv4", "ipv6"},
    17: {
        "biometric", "biometric_id", "fingerprint", "fingerprint_id",
        "voiceprint", "voiceprint_id", "retina_scan", "iris_scan",
    },
    18: {
        "id", "patient_id", "unique_id", "uuid", "hash", "patient_hash",
        "source_system_id", "encounter_id", "claim_id", "accession_id",
        "token", "photo", "photograph", "face_photo", "full_face_photo",
        "image", "image_reference", "binary", "blob",
    },
}

FREE_TEXT_COLUMN_NAMES = {
    "note", "notes", "comment", "comments", "description", "narrative",
    "text", "free_text", "reason", "summary",
}

BIRTH_DATE_COLUMN_NAMES = {"birthdate", "birth_date", "dob"}

KNOWN_ANALYTICAL_COLUMNS = {
    "gender", "sex", "race", "ethnicity", "marital", "marital_status",
    "state", "diagnosis", "disease", "condition", "outcome", "lab_result",
    "healthcare_expenses", "healthcare_coverage",
}

TEXT_ENTITY_CATEGORIES = {
    "PERSON": 1,
    "MEDICAL_RECORD_NUMBER": 9,
    "PATIENT_ID": 18,
    "HEALTH_PLAN_ID": 10,
    "INSURANCE_ID": 10,
    "ACCESSION_NUMBER": 18,
    "DEVICE_ID": 14,
    "EMAIL_ADDRESS": 7,
    "PHONE_NUMBER": 5,
    "US_SSN": 8,
    "SSN": 8,
    "DATE_TIME": 3,
}


class SafeHarborValidationError(ValueError):
    def __init__(self, detail: str, status_code: int = 400):
        super().__init__(detail)
        self.detail = detail
        self.status_code = status_code


@dataclass
class SafeHarborPreparation:
    rows: List[Dict[str, str]]
    columns_to_remove: List[str]
    date_columns: List[str]
    age_columns: List[str]
    detected_categories: Set[int]
    removed_categories: Set[int]
    unresolved_categories: Set[int]
    column_categories: Dict[str, Set[int]]
    free_text_scan_status: str
    unique_code_scan_status: str
    warnings: List[str]


DATE_FORMATS = (
    "%Y-%m-%d", "%Y/%m/%d", "%m/%d/%Y", "%m/%d/%y", "%d-%m-%Y",
    "%Y%m%d", "%Y-%m", "%Y",
)

PATTERN_CATEGORY_RULES = (
    (6, re.compile(r"\bfax\s*[:#-]?\s*(?:\+?1[\s.-]?)?(?:\(?\d{3}\)?[\s.-]?)\d{3}[\s.-]?\d{4}\b", re.I)),
    (7, re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.I)),
    (8, re.compile(r"\b\d{3}-\d{2}-\d{4}\b")),
    (15, re.compile(r"\b(?:https?://|www\.)\S+", re.I)),
    (16, re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b|\b[0-9A-F]{1,4}(?::[0-9A-F]{0,4}){2,7}\b", re.I)),
    (5, re.compile(r"(?<!\w)(?:\+?1[\s.-]?)?(?:\(?\d{3}\)?[\s.-]?)\d{3}[\s.-]?\d{4}(?!\w)")),
    (9, re.compile(r"\b(?:MRN|chart|hospital)\s*[:#-]?\s*[A-Z0-9-]{4,}\b", re.I)),
    (10, re.compile(r"\b(?:member|beneficiary|subscriber|policy|insurance)\s*(?:id|number)?\s*[:#-]?\s*[A-Z0-9-]{5,}\b", re.I)),
    (11, re.compile(r"\baccount\s*(?:id|number)?\s*[:#-]?\s*[A-Z0-9-]{4,}\b", re.I)),
    (12, re.compile(r"\b(?:licen[cs]e|certificate)\s*(?:id|number)?\s*[:#-]?\s*[A-Z0-9-]{4,}\b", re.I)),
    (13, re.compile(r"\b(?:VIN|vehicle|plate)\s*[:#-]?\s*[A-HJ-NPR-Z0-9-]{5,17}\b", re.I)),
    (14, re.compile(r"\b(?:device|implant|serial|UDI)\s*[:#-]?\s*[A-Z0-9-]{5,}\b", re.I)),
    (17, re.compile(r"\b(?:fingerprint|voiceprint|retina|iris|biometric)\s*[:#-]?\s*[A-Z0-9+/=-]{4,}\b", re.I)),
    (18, re.compile(r"\b[0-9A-F]{8}-[0-9A-F]{4}-[1-5][0-9A-F]{3}-[89AB][0-9A-F]{3}-[0-9A-F]{12}\b", re.I)),
    (18, re.compile(r"\b[0-9A-F]{16,128}\b", re.I)),
    (18, re.compile(r"(?:data:image/|\.(?:jpe?g|png|gif|bmp|tiff?)\b)", re.I)),
    (3, re.compile(r"\b(?:\d{4}-\d{2}-\d{2}|\d{1,2}/\d{1,2}/\d{2,4})\b")),
    (2, re.compile(r"\b\d{1,6}\s+[A-Z0-9.' -]+\s(?:street|st|avenue|ave|road|rd|lane|ln|drive|dr|boulevard|blvd)\b", re.I)),
    (2, re.compile(r"\b\d{5}(?:-\d{4})?\b")),
)


def normalize_column_name(column_name: str) -> str:
    normalized = re.sub(r"[^a-z0-9]+", "_", column_name.strip().lower())
    return normalized.strip("_")


def _category_from_mapping_key(key: Any) -> int:
    text = str(key).strip().lower()
    if text.isdigit() and int(text) in CATEGORY_LABELS:
        return int(text)
    for category, label in CATEGORY_LABELS.items():
        if text in {label, label.split("_", 1)[1]}:
            return category
    raise SafeHarborValidationError(
        "Safe Harbor mapping contains an unsupported identifier category"
    )


def _parse_date(value: str):
    for date_format in DATE_FORMATS:
        try:
            return datetime.strptime(value, date_format).date()
        except (TypeError, ValueError):
            continue
    return None


def _parse_number(value: str) -> Optional[float]:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _pattern_categories(value: str) -> Set[int]:
    if not value or value == UNKNOWN_VALUE:
        return set()
    return {
        category
        for category, pattern in PATTERN_CATEGORY_RULES
        if pattern.search(value)
    }


def _is_free_text_column(column: str, values: Sequence[str]) -> bool:
    if normalize_column_name(column) in FREE_TEXT_COLUMN_NAMES:
        return True
    return any(
        len(value) >= 40 or len(value.split()) >= 6
        for value in values
        if value != UNKNOWN_VALUE
    )


def _looks_like_unknown_unique_code(column: str, values: Sequence[str]) -> bool:
    normalized = normalize_column_name(column)
    if normalized in KNOWN_ANALYTICAL_COLUMNS:
        return False
    known = [value for value in values if value != UNKNOWN_VALUE]
    if len(known) < 3:
        return False
    if re.search(r"(?:^|_)(?:id|identifier|code|hash|uuid|token|key|number)$", normalized):
        return True
    unique_ratio = len(set(known)) / len(known)
    if all(_parse_number(value) is not None for value in known):
        return unique_ratio >= 0.9
    code_like = all(
        len(value) <= 128 and not re.search(r"\s", value)
        for value in known
    )
    return unique_ratio >= 0.9 and code_like


def _looks_like_person_name_values(values: Sequence[str]) -> bool:
    known = [value for value in values if value != UNKNOWN_VALUE]
    if len(known) < 2:
        return False
    name_pattern = re.compile(
        r"^[A-Z][A-Za-z'-]+(?:\s+[A-Z][A-Za-z'-]+){1,3}$"
    )
    return (
        len(set(known)) / len(known) >= 0.8
        and all(name_pattern.fullmatch(value) for value in known)
    )


def prepare_safe_harbor_rows(
    header: Sequence[str],
    records: Sequence[Dict[str, str]],
    configured_mappings: Optional[Mapping[str, Sequence[str]]] = None,
    configured_direct_columns: Optional[Sequence[str]] = None,
    reviewed_columns: Optional[Sequence[str]] = None,
) -> SafeHarborPreparation:
    rows = [dict(record) for record in records]
    normalized_columns = {
        normalize_column_name(column): column
        for column in header
    }
    column_categories: Dict[str, Set[int]] = {
        column: set()
        for column in header
    }

    for column in header:
        normalized = normalize_column_name(column)
        for category, aliases in CATEGORY_ALIASES.items():
            if normalized in aliases:
                column_categories[column].add(category)

    if configured_mappings:
        for raw_category, configured_columns in configured_mappings.items():
            category = _category_from_mapping_key(raw_category)
            if isinstance(configured_columns, str):
                configured_columns = [configured_columns]
            for configured_column in configured_columns:
                actual = normalized_columns.get(
                    normalize_column_name(configured_column)
                )
                if actual is None:
                    raise SafeHarborValidationError(
                        "Configured Safe Harbor mapping references a missing column"
                    )
                column_categories[actual].add(category)

    for configured_column in configured_direct_columns or []:
        actual = normalized_columns.get(normalize_column_name(configured_column))
        if actual is None:
            continue
        if not column_categories[actual]:
            column_categories[actual].add(18)

    values_by_column = {
        column: [row.get(column, UNKNOWN_VALUE) for row in rows]
        for column in header
    }
    reviewed_column_set = {
        actual
        for column in reviewed_columns or []
        if (actual := normalized_columns.get(normalize_column_name(column)))
        is not None
    }
    for column, values in values_by_column.items():
        for value in values:
            column_categories[column].update(_pattern_categories(value))

        if not column_categories[column] and _looks_like_person_name_values(values):
            column_categories[column].add(1)

    date_columns = [
        column for column in header if 3 in column_categories[column]
    ]
    current_year = datetime.utcnow().year
    related_birth_date_columns = [
        column
        for column in date_columns
        if normalize_column_name(column) in BIRTH_DATE_COLUMN_NAMES
    ]
    for column in related_birth_date_columns:
        if any(
            (parsed := _parse_date(value)) is not None
            and parsed.year <= current_year - 90
            for value in values_by_column[column]
            if value != UNKNOWN_VALUE
        ):
            column_categories[column].add(4)
    age_columns = [
        column
        for column in header
        if normalize_column_name(column) in CATEGORY_ALIASES[4]
        or (
            4 in column_categories[column]
            and 3 not in column_categories[column]
        )
    ]
    for column in age_columns:
        has_age_over_89 = any(
            (number := _parse_number(value)) is not None and number > 89
            for value in values_by_column[column]
            if value != UNKNOWN_VALUE
        )
        if not has_age_over_89:
            column_categories[column].discard(4)
    columns_to_remove: Set[str] = {
        column
        for column in header
        if column_categories[column].difference({3, 4})
    }

    free_text_status = "passed"
    free_text_columns: Set[str] = set()
    quarantined_text_columns = 0
    for column in header:
        if column in columns_to_remove:
            continue
        values = values_by_column[column]
        if not _is_free_text_column(column, values):
            continue
        free_text_columns.add(column)

        detected_in_text: Set[int] = set()
        try:
            for value in values:
                if value == UNKNOWN_VALUE:
                    continue
                detected_in_text.update(_pattern_categories(value))
                entities = detect_clinical_phi(value)
                detected_in_text.update(
                    TEXT_ENTITY_CATEGORIES[entity]
                    for entity in entities
                    if entity in TEXT_ENTITY_CATEGORIES
                )
        except TextAnonymizationError:
            columns_to_remove.add(column)
            column_categories[column].add(18)
            quarantined_text_columns += 1
            free_text_status = "passed_with_quarantine"
            continue

        if detected_in_text:
            column_categories[column].update(detected_in_text)
            columns_to_remove.add(column)

    unknown_unique_columns = 0
    for column in header:
        if (
            column in columns_to_remove
            or column in reviewed_column_set
            or column in free_text_columns
            or column in date_columns
            or column in age_columns
        ):
            continue
        if _looks_like_unknown_unique_code(column, values_by_column[column]):
            column_categories[column].add(18)
            columns_to_remove.add(column)
            unknown_unique_columns += 1

    invalid_date_count = 0
    ages_capped = 0
    for row in rows:
        for column in date_columns:
            if column in columns_to_remove:
                continue
            value = row.get(column, UNKNOWN_VALUE)
            if value == UNKNOWN_VALUE:
                continue
            parsed = _parse_date(value)
            if parsed is None:
                row[column] = UNKNOWN_VALUE
                invalid_date_count += 1
            elif (
                column in related_birth_date_columns
                and parsed.year <= current_year - 90
            ):
                row[column] = "90+"
            else:
                row[column] = f"{parsed.year:04d}"

        row_has_age_over_89 = False
        for column in age_columns:
            if column in columns_to_remove:
                continue
            value = row.get(column, UNKNOWN_VALUE)
            if value == UNKNOWN_VALUE:
                continue
            parsed_age = _parse_number(value)
            if parsed_age is None or parsed_age < 0:
                row[column] = UNKNOWN_VALUE
            elif parsed_age > 89:
                row[column] = "90+"
                ages_capped += 1
                row_has_age_over_89 = True

        if row_has_age_over_89:
            for column in related_birth_date_columns:
                if column not in columns_to_remove:
                    row[column] = "90+"

    ordered_columns_to_remove = [
        column for column in header if column in columns_to_remove
    ]
    detected_categories = {
        category
        for categories in column_categories.values()
        for category in categories
    }
    removed_categories = {
        category
        for column in ordered_columns_to_remove
        for category in column_categories[column]
    }
    warnings = [
        "Organizational and legal review remains required; these are technical checks only."
    ]
    if invalid_date_count:
        warnings.append(
            f"{invalid_date_count} invalid date value(s) were replaced with UNKNOWN."
        )
    if ages_capped:
        warnings.append(f"{ages_capped} age value(s) were grouped into 90+.")
    if quarantined_text_columns:
        warnings.append(
            f"{quarantined_text_columns} unscannable free-text column(s) were removed."
        )
    if unknown_unique_columns:
        warnings.append(
            f"{unknown_unique_columns} unknown high-cardinality code column(s) were removed."
        )

    unique_code_status = (
        "passed_with_removals" if unknown_unique_columns else "passed"
    )
    return SafeHarborPreparation(
        rows=rows,
        columns_to_remove=ordered_columns_to_remove,
        date_columns=date_columns,
        age_columns=age_columns,
        detected_categories=detected_categories,
        removed_categories=removed_categories,
        unresolved_categories=set(),
        column_categories=column_categories,
        free_text_scan_status=free_text_status,
        unique_code_scan_status=unique_code_status,
        warnings=warnings,
    )


def build_safe_harbor_report(
    preparation: SafeHarborPreparation,
    anonymized_rows: Sequence[Dict[str, str]],
    output_columns: Sequence[str],
) -> Dict[str, Any]:
    unresolved = set(preparation.unresolved_categories)
    output_column_set = set(output_columns)

    for column, categories in preparation.column_categories.items():
        if column not in output_column_set:
            continue
        unresolved.update(categories.difference({3, 4}))

    exact_date_found = False
    for row in anonymized_rows:
        for value in row.values():
            categories = _pattern_categories(value)
            direct_patterns = categories.difference({2})
            if 3 in direct_patterns:
                exact_date_found = True
            unresolved.update(direct_patterns.difference({3}))

    exact_age_over_89_found = False
    for row in anonymized_rows:
        for column in preparation.age_columns:
            if column not in output_column_set:
                continue
            value = row.get(column, UNKNOWN_VALUE)
            number = _parse_number(value)
            if number is not None and number > 89:
                exact_age_over_89_found = True

    if exact_date_found:
        unresolved.add(3)
    if exact_age_over_89_found:
        unresolved.add(4)

    prohibited_geography_remains = any(
        2 in preparation.column_categories.get(column, set())
        for column in output_columns
    )
    if prohibited_geography_remains:
        unresolved.add(2)

    free_text_status = preparation.free_text_scan_status
    unique_code_status = preparation.unique_code_scan_status
    if free_text_status == "failed":
        unresolved.add(18)
    if unique_code_status == "failed":
        unresolved.add(18)

    warnings = list(preparation.warnings)
    validation_status = (
        "failed"
        if unresolved
        else "passed_with_warnings" if warnings else "passed"
    )

    return {
        "safe_harbor_validation_status": validation_status,
        "identifier_categories_detected": _category_labels(
            preparation.detected_categories
        ),
        "identifier_categories_removed": _category_labels(
            preparation.removed_categories
        ),
        "unresolved_identifier_categories": _category_labels(unresolved),
        "date_compliance_status": "failed" if exact_date_found else "passed",
        "age_over_89_status": (
            "failed" if exact_age_over_89_found else "passed"
        ),
        "geographic_compliance_status": (
            "failed" if prohibited_geography_remains else "passed"
        ),
        "free_text_scan_status": free_text_status,
        "unique_code_scan_status": unique_code_status,
        "actual_knowledge_review_required": True,
        "warnings": warnings,
    }


def _category_labels(categories: Set[int]) -> List[str]:
    return [
        CATEGORY_LABELS[category]
        for category in sorted(categories)
    ]
