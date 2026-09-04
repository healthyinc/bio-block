from __future__ import annotations

import csv
import io
import math
import re
from collections import Counter, defaultdict
from datetime import datetime
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from services.safe_harbor import (
    SafeHarborValidationError,
    build_safe_harbor_report,
    prepare_safe_harbor_rows,
)
from services.tabular_validation import VALIDATION_PASSED, validate_serialized_csv


UNKNOWN_VALUE = "UNKNOWN"
SUPPRESSED_VALUE = "*"

DEFAULT_DIRECT_IDENTIFIER_COLUMNS = (
    "name",
    "full_name",
    "email",
    "phone",
    "mobile",
    "mrn",
    "medical_record_number",
    "patient_id",
    "id",
    "ssn",
    "social_security_number",
    "drivers",
    "driver_license",
    "drivers_license",
    "licence_number",
    "license_number",
    "passport",
    "passport_number",
    "prefix",
    "first",
    "first_name",
    "given_name",
    "last",
    "last_name",
    "family_name",
    "suffix",
    "maiden",
    "maiden_name",
    "address",
    "street_address",
)

DEFAULT_QUASI_IDENTIFIER_COLUMNS = (
    "age",
    "gender",
    "sex",
    "zip",
    "zip_code",
    "postal_code",
    "city",
    "state",
    "date",
    "admission_date",
    "discharge_date",
    "diagnosis_date",
    "birthdate",
    "birth_date",
    "deathdate",
    "death_date",
    "race",
    "ethnicity",
    "marital",
    "marital_status",
    "birthplace",
    "birth_place",
    "county",
    "lat",
    "latitude",
    "lon",
    "longitude",
)

DEFAULT_SENSITIVE_COLUMNS = (
    "diagnosis",
    "disease",
    "condition",
    "outcome",
)

DATE_FORMATS = (
    "%Y-%m-%d",
    "%Y/%m/%d",
    "%m/%d/%Y",
    "%d-%m-%Y",
    "%Y%m%d",
    "%Y-%m",
    "%Y",
)

DATE_COLUMN_NAMES = {
    "date",
    "admission_date",
    "discharge_date",
    "diagnosis_date",
    "birthdate",
    "birth_date",
    "deathdate",
    "death_date",
}

ZIP_COLUMN_NAMES = {"zip", "zip_code", "postal_code"}

AGE_COLUMN_NAMES = {"age", "patient_age", "age_years"}

PRECISE_GEOGRAPHY_COLUMN_NAMES = {"lat", "latitude", "lon", "longitude"}

NUMERICAL_SENSITIVE_COLUMN_NAMES = {
    "healthcare_expenses",
    "healthcare_coverage",
}


class TabularAnonymizationError(ValueError):
    def __init__(self, detail: str, status_code: int = 400):
        super().__init__(detail)
        self.detail = detail
        self.status_code = status_code


def anonymize_tabular_csv(
    file_bytes: bytes,
    k: int = 5,
    l: int = 2,
    direct_identifiers: Optional[List[str]] = None,
    quasi_identifiers: Optional[List[str]] = None,
    sensitive_column: Optional[str] = None,
    safe_harbor_mappings: Optional[Mapping[str, Sequence[str]]] = None,
    include_anonymized_csv: bool = False,
) -> Dict[str, Any]:
    _validate_thresholds(k, l)
    header, records = _read_csv(file_bytes)
    normalized_columns = _build_normalized_column_map(header)

    direct_identifier_columns = _resolve_column_list(
        available_columns=normalized_columns,
        requested=direct_identifiers,
        defaults=DEFAULT_DIRECT_IDENTIFIER_COLUMNS,
        column_kind="direct identifier",
    )
    sensitive_column_name = _resolve_sensitive_column(
        available_columns=normalized_columns,
        requested=sensitive_column,
    )
    quasi_identifier_columns = _resolve_column_list(
        available_columns=normalized_columns,
        requested=quasi_identifiers,
        defaults=DEFAULT_QUASI_IDENTIFIER_COLUMNS,
        column_kind="quasi-identifier",
    )

    try:
        safe_harbor = prepare_safe_harbor_rows(
            header=header,
            records=records,
            configured_mappings=safe_harbor_mappings,
            configured_direct_columns=direct_identifier_columns,
            reviewed_columns=quasi_identifiers,
        )
    except SafeHarborValidationError as exc:
        raise TabularAnonymizationError(
            exc.detail,
            status_code=exc.status_code,
        ) from exc
    records = safe_harbor.rows
    precise_geography_columns = [
        column
        for column in header
        if _normalize_column_name(column) in PRECISE_GEOGRAPHY_COLUMN_NAMES
    ]

    overlap = set(direct_identifier_columns).intersection(quasi_identifier_columns)
    if overlap:
        raise TabularAnonymizationError(
            "Columns cannot be both direct identifiers and quasi-identifiers: "
            + ", ".join(sorted(overlap))
        )
    if sensitive_column_name and sensitive_column_name in direct_identifier_columns:
        raise TabularAnonymizationError(
            "Sensitive column cannot also be removed as a direct identifier"
        )
    if sensitive_column_name and sensitive_column_name in safe_harbor.columns_to_remove:
        raise TabularAnonymizationError(
            "Sensitive column is a Safe Harbor identifier and cannot be retained"
        )

    safe_harbor_removed_set = set(safe_harbor.columns_to_remove)
    quasi_identifier_columns = [
        column
        for column in quasi_identifier_columns
        if column not in safe_harbor_removed_set
    ]
    if not quasi_identifier_columns:
        raise TabularAnonymizationError(
            "At least one quasi-identifier column must remain after Safe "
            "Harbor identifier removal"
        )
    if sensitive_column_name and sensitive_column_name in quasi_identifier_columns:
        raise TabularAnonymizationError(
            "Sensitive column cannot also be a quasi-identifier"
        )

    removed_column_set = set(direct_identifier_columns).union(
        safe_harbor.columns_to_remove
    )
    retained_columns = [
        column for column in header if column not in removed_column_set
    ]
    retained_column_set = set(retained_columns)
    missing_quasi_after_removal = [
        column for column in quasi_identifier_columns if column not in retained_column_set
    ]
    if missing_quasi_after_removal:
        raise TabularAnonymizationError(
            "Missing quasi-identifier column(s): "
            + ", ".join(missing_quasi_after_removal)
        )

    working_rows = _remove_direct_identifiers(
        records=records,
        retained_columns=retained_columns,
    )
    quasi_identifier_types = {
        column: _infer_column_type(working_rows, column)
        for column in quasi_identifier_columns
    }
    processing_warnings = _normalize_invalid_date_values(
        rows=working_rows,
        quasi_identifier_types=quasi_identifier_types,
    )
    _validate_sensitive_column_for_l_diversity(
        rows=working_rows,
        sensitive_column=sensitive_column_name,
    )

    partitions = _mondrian_partitions(
        rows=working_rows,
        quasi_identifier_types=quasi_identifier_types,
        k=k,
        l=l,
        sensitive_column=sensitive_column_name,
    )
    anonymized_rows, generalized_cells, suppressed_cells = _generalize_partitions(
        rows=working_rows,
        partitions=partitions,
        quasi_identifier_types=quasi_identifier_types,
    )

    equivalence_groups = _equivalence_groups(
        rows=anonymized_rows,
        quasi_identifiers=quasi_identifier_columns,
    )
    group_sizes = [len(indices) for indices in equivalence_groups.values()]
    min_group_size = min(group_sizes) if group_sizes else 0
    k_anonymity_satisfied = bool(group_sizes) and min_group_size >= k

    if not k_anonymity_satisfied:
        processing_warnings.append(
            "k-anonymity could not be satisfied; minimum equivalence class "
            f"size is {min_group_size}."
        )

    if sensitive_column_name is None:
        l_diversity_satisfied: Any = "not_applicable"
        processing_warnings.append(
            "l-diversity was not evaluated because no sensitive column was "
            "provided or detected."
        )
    else:
        l_diversity_satisfied = _l_diversity_satisfied(
            rows=anonymized_rows,
            equivalence_groups=equivalence_groups,
            sensitive_column=sensitive_column_name,
            l=l,
        )
        if not l_diversity_satisfied:
            processing_warnings.append(
                "l-diversity could not be satisfied for at least one "
                "equivalence class."
            )

    safe_harbor_report = build_safe_harbor_report(
        preparation=safe_harbor,
        anonymized_rows=anonymized_rows,
        output_columns=retained_columns,
    )
    warnings = [
        *safe_harbor_report["warnings"],
        *processing_warnings,
    ]
    privacy_requirements_satisfied = (
        safe_harbor_report["safe_harbor_validation_status"] != "failed"
        and not safe_harbor_report["unresolved_identifier_categories"]
        and k_anonymity_satisfied
        and l_diversity_satisfied in {True, "not_applicable"}
    )
    if not privacy_requirements_satisfied:
        anonymization_status = "failed_privacy_validation"
    elif warnings:
        anonymization_status = "completed_with_warnings"
    else:
        anonymization_status = "completed"
    total_quasi_identifier_cells = (
        len(anonymized_rows) * len(quasi_identifier_columns)
    )
    columns_removed = [
        column for column in header if column in removed_column_set
    ]

    result: Dict[str, Any] = {
        "anonymization_status": anonymization_status,
        "rows_in": len(records),
        "rows_out": len(anonymized_rows),
        "k": k,
        "l": l,
        "direct_identifiers_removed": direct_identifier_columns,
        "precise_geography_columns_removed": precise_geography_columns,
        "columns_removed": columns_removed,
        "quasi_identifiers_used": quasi_identifier_columns,
        "sensitive_column": sensitive_column_name,
        "output_columns": retained_columns,
        "safe_harbor_report": safe_harbor_report,
        "equivalence_classes": len(equivalence_groups),
        "min_group_size": min_group_size,
        "k_anonymity_satisfied": k_anonymity_satisfied,
        "l_diversity_satisfied": l_diversity_satisfied,
        "generalized_cells_count": generalized_cells,
        "suppressed_cells_count": suppressed_cells,
        "generalization_rate": _safe_rate(
            generalized_cells,
            total_quasi_identifier_cells,
        ),
        "suppression_rate": _safe_rate(
            suppressed_cells,
            total_quasi_identifier_cells,
        ),
        "warnings": warnings,
    }

    # Always serialize, so the bytes a caller could download are the bytes that
    # get validated. Reporting a removal plan without re-reading the output is
    # how a column that survived serialization gets called clean.
    serialized_csv = _write_csv(retained_columns, anonymized_rows)
    validation = validate_serialized_csv(
        serialized_csv,
        expected_columns=retained_columns,
        expected_row_count=len(anonymized_rows),
        removed_columns=columns_removed,
        source_records=records,
    )
    result.update(validation)
    if validation["serialized_output_validation"] != VALIDATION_PASSED:
        result["anonymization_status"] = "failed_privacy_validation"
        result["warnings"] = [*warnings, *validation["validation_failures"]]

    if include_anonymized_csv:
        result["_internal_anonymized_csv"] = serialized_csv

    return result


def classify_tabular_columns(header: Sequence[str]) -> Dict[str, List[str]]:
    normalized_columns = _build_normalized_column_map(header)
    direct_defaults = set(DEFAULT_DIRECT_IDENTIFIER_COLUMNS)
    quasi_defaults = set(DEFAULT_QUASI_IDENTIFIER_COLUMNS)

    return {
        "direct_identifiers": [
            actual
            for normalized, actual in normalized_columns.items()
            if normalized in direct_defaults
        ],
        "quasi_identifiers": [
            actual
            for normalized, actual in normalized_columns.items()
            if normalized in quasi_defaults
        ],
        "precise_geography": [
            actual
            for normalized, actual in normalized_columns.items()
            if normalized in PRECISE_GEOGRAPHY_COLUMN_NAMES
        ],
    }


def _validate_thresholds(k: int, l: int) -> None:
    if not isinstance(k, int) or k < 1:
        raise TabularAnonymizationError("k must be a positive integer")
    if not isinstance(l, int) or l < 1:
        raise TabularAnonymizationError("l must be a positive integer")


def _read_csv(file_bytes: bytes) -> Tuple[List[str], List[Dict[str, str]]]:
    if not file_bytes:
        raise TabularAnonymizationError("CSV input is empty")

    try:
        csv_text = file_bytes.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise TabularAnonymizationError("CSV uploads must be UTF-8 encoded") from exc

    if not csv_text.strip():
        raise TabularAnonymizationError("CSV input is empty")

    try:
        reader = csv.reader(io.StringIO(csv_text), strict=True)
        raw_rows = [row for row in reader if any(cell.strip() for cell in row)]
    except csv.Error as exc:
        raise TabularAnonymizationError("Invalid CSV format") from exc

    if not raw_rows:
        raise TabularAnonymizationError("CSV input is empty")

    header = [cell.strip() for cell in raw_rows[0]]
    if not header or not any(header):
        raise TabularAnonymizationError("CSV header row is missing")
    if any(not column for column in header):
        raise TabularAnonymizationError("CSV header row contains empty column names")

    normalized_header = [_normalize_column_name(column) for column in header]
    if len(set(normalized_header)) != len(normalized_header):
        raise TabularAnonymizationError("CSV column names must be unique")

    if len(raw_rows) == 1:
        raise TabularAnonymizationError("CSV input must include at least one data row")

    records: List[Dict[str, str]] = []
    expected_columns = len(header)
    for row_number, row in enumerate(raw_rows[1:], start=2):
        if len(row) != expected_columns:
            raise TabularAnonymizationError(
                f"CSV row {row_number} has {len(row)} fields; "
                f"expected {expected_columns}"
            )
        records.append(
            {
                column: _clean_cell(value)
                for column, value in zip(header, row)
            }
        )

    if not records:
        raise TabularAnonymizationError("CSV input must include at least one data row")

    return header, records


def _clean_cell(value: Any) -> str:
    cleaned = str(value).strip()
    return cleaned if cleaned else UNKNOWN_VALUE


def _normalize_column_name(column_name: str) -> str:
    normalized = re.sub(r"[^a-z0-9]+", "_", column_name.strip().lower())
    return normalized.strip("_")


def _build_normalized_column_map(header: Sequence[str]) -> Dict[str, str]:
    return {_normalize_column_name(column): column for column in header}


def _resolve_column_list(
    available_columns: Dict[str, str],
    requested: Optional[List[str]],
    defaults: Sequence[str],
    column_kind: str,
) -> List[str]:
    if requested is not None:
        resolved: List[str] = []
        missing: List[str] = []
        seen = set()
        for column in requested:
            normalized = _normalize_column_name(column)
            actual_column = available_columns.get(normalized)
            if actual_column is None:
                missing.append(column)
                continue
            if actual_column not in seen:
                resolved.append(actual_column)
                seen.add(actual_column)

        if missing:
            raise TabularAnonymizationError(
                f"Missing {column_kind} column(s): " + ", ".join(missing)
            )
        if column_kind == "quasi-identifier" and not resolved:
            raise TabularAnonymizationError(
                "At least one quasi-identifier column is required"
            )
        return resolved

    normalized_defaults = set(defaults)
    resolved_defaults = [
        actual_column
        for normalized, actual_column in available_columns.items()
        if normalized in normalized_defaults
    ]
    if column_kind == "quasi-identifier" and not resolved_defaults:
        raise TabularAnonymizationError(
            "No quasi-identifier columns were detected; provide quasi_identifiers"
        )
    return resolved_defaults


def _resolve_sensitive_column(
    available_columns: Dict[str, str],
    requested: Optional[str],
) -> Optional[str]:
    if requested:
        actual_column = available_columns.get(_normalize_column_name(requested))
        if actual_column is None:
            raise TabularAnonymizationError(
                f"Missing sensitive column: {requested}"
            )
        return actual_column

    for candidate in DEFAULT_SENSITIVE_COLUMNS:
        actual_column = available_columns.get(candidate)
        if actual_column is not None:
            return actual_column
    return None


def _remove_direct_identifiers(
    records: Sequence[Dict[str, str]],
    retained_columns: Sequence[str],
) -> List[Dict[str, str]]:
    return [
        {
            column: _clean_cell(record.get(column, UNKNOWN_VALUE))
            for column in retained_columns
        }
        for record in records
    ]


def _infer_column_type(rows: Sequence[Dict[str, str]], column: str) -> str:
    normalized_column = _normalize_column_name(column)
    if (
        normalized_column in DATE_COLUMN_NAMES
        or normalized_column.endswith("_date")
    ):
        return "date"
    if normalized_column in ZIP_COLUMN_NAMES:
        return "zip"
    if normalized_column in AGE_COLUMN_NAMES:
        return "age"

    values = [
        row[column]
        for row in rows
        if row.get(column, UNKNOWN_VALUE) != UNKNOWN_VALUE
    ]
    if not values:
        return "categorical"

    if all(_parse_number(value) is not None for value in values):
        return "numeric"
    if all(_parse_date(value) is not None for value in values):
        return "date"
    return "categorical"


def _normalize_invalid_date_values(
    rows: Sequence[Dict[str, str]],
    quasi_identifier_types: Dict[str, str],
) -> List[str]:
    warnings: List[str] = []
    for column, column_type in quasi_identifier_types.items():
        if column_type != "date":
            continue

        invalid_count = 0
        for row in rows:
            value = row.get(column, UNKNOWN_VALUE)
            if (
                value in {UNKNOWN_VALUE, "90+"}
                or _parse_date(value) is not None
            ):
                continue
            row[column] = UNKNOWN_VALUE
            invalid_count += 1

        if invalid_count:
            warnings.append(
                f"Column {column} contained {invalid_count} invalid date "
                "value(s); replaced with UNKNOWN before generalization."
            )
    return warnings


def _validate_sensitive_column_for_l_diversity(
    rows: Sequence[Dict[str, str]],
    sensitive_column: Optional[str],
) -> None:
    if sensitive_column is None:
        return
    if (
        _normalize_column_name(sensitive_column)
        not in NUMERICAL_SENSITIVE_COLUMN_NAMES
    ):
        return

    values = [
        row.get(sensitive_column, UNKNOWN_VALUE)
        for row in rows
        if row.get(sensitive_column, UNKNOWN_VALUE) != UNKNOWN_VALUE
    ]
    if values and all(_parse_number(value) is not None for value in values):
        raise TabularAnonymizationError(
            f"Numerical sensitive column {sensitive_column} requires bucketing "
            "before l-diversity evaluation; provide a pre-bucketed categorical "
            "sensitive column."
        )


def _parse_number(value: str) -> Optional[float]:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return number


def _parse_age(value: str) -> Optional[float]:
    if value == "90+":
        return 90.0
    return _parse_number(value)


def _parse_date(value: str):
    for date_format in DATE_FORMATS:
        try:
            return datetime.strptime(value, date_format).date()
        except (TypeError, ValueError):
            continue
    return None


def _mondrian_partitions(
    rows: Sequence[Dict[str, str]],
    quasi_identifier_types: Dict[str, str],
    k: int,
    l: int,
    sensitive_column: Optional[str],
) -> List[List[int]]:
    final_partitions: List[List[int]] = []
    pending_partitions: List[List[int]] = [list(range(len(rows)))]

    while pending_partitions:
        indices = pending_partitions.pop()
        if len(indices) < 2 * k:
            final_partitions.append(indices)
            continue

        split = _best_split(
            rows=rows,
            indices=indices,
            quasi_identifier_types=quasi_identifier_types,
            k=k,
            l=l,
            sensitive_column=sensitive_column,
        )
        if split is None:
            final_partitions.append(indices)
            continue

        left_indices, right_indices = split
        pending_partitions.append(left_indices)
        pending_partitions.append(right_indices)

    return final_partitions


def _best_split(
    rows: Sequence[Dict[str, str]],
    indices: Sequence[int],
    quasi_identifier_types: Dict[str, str],
    k: int,
    l: int,
    sensitive_column: Optional[str],
) -> Optional[Tuple[List[int], List[int]]]:
    best: Optional[Tuple[Tuple[int, float], List[int], List[int]]] = None

    for column, column_type in quasi_identifier_types.items():
        split = _split_partition(rows, indices, column, column_type)
        if split is None:
            continue

        left_indices, right_indices = split
        if len(left_indices) < k or len(right_indices) < k:
            continue

        if sensitive_column is not None:
            if not _partition_l_diverse(rows, left_indices, sensitive_column, l):
                continue
            if not _partition_l_diverse(rows, right_indices, sensitive_column, l):
                continue

        score = _split_score(rows, indices, column, column_type)
        if best is None or score > best[0]:
            best = (score, left_indices, right_indices)

    if best is None:
        return None
    return best[1], best[2]


def _split_partition(
    rows: Sequence[Dict[str, str]],
    indices: Sequence[int],
    column: str,
    column_type: str,
) -> Optional[Tuple[List[int], List[int]]]:
    if column_type in {"numeric", "date", "age"}:
        return _split_ordered(rows, indices, column, column_type)
    return _split_categorical(rows, indices, column)


def _split_ordered(
    rows: Sequence[Dict[str, str]],
    indices: Sequence[int],
    column: str,
    column_type: str,
) -> Optional[Tuple[List[int], List[int]]]:
    sortable: List[Tuple[bool, float, int]] = []
    distinct_values = set()
    for row_index in indices:
        raw_value = rows[row_index][column]
        parsed_value: Optional[float]
        if column_type == "numeric":
            parsed_value = _parse_number(raw_value)
        elif column_type == "age":
            parsed_value = _parse_age(raw_value)
        else:
            if raw_value == "90+":
                parsed_value = -1.0
            else:
                parsed_date = _parse_date(raw_value)
                parsed_value = (
                    float(parsed_date.toordinal()) if parsed_date else None
                )

        if parsed_value is None:
            sortable.append((True, 0.0, row_index))
        else:
            sortable.append((False, parsed_value, row_index))
            distinct_values.add(parsed_value)

    if len(distinct_values) < 2:
        return None

    sortable.sort(key=lambda item: (item[0], item[1], item[2]))
    midpoint = len(sortable) // 2
    left_indices = [item[2] for item in sortable[:midpoint]]
    right_indices = [item[2] for item in sortable[midpoint:]]
    return left_indices, right_indices


def _split_categorical(
    rows: Sequence[Dict[str, str]],
    indices: Sequence[int],
    column: str,
) -> Optional[Tuple[List[int], List[int]]]:
    counts = Counter(rows[row_index][column] for row_index in indices)
    if len(counts) < 2:
        return None

    left_categories = set()
    right_categories = set()
    left_size = 0
    right_size = 0
    for category, count in sorted(counts.items(), key=lambda item: (-item[1], item[0])):
        if left_size <= right_size:
            left_categories.add(category)
            left_size += count
        else:
            right_categories.add(category)
            right_size += count

    left_indices = [
        row_index
        for row_index in indices
        if rows[row_index][column] in left_categories
    ]
    right_indices = [
        row_index
        for row_index in indices
        if rows[row_index][column] in right_categories
    ]
    if not left_indices or not right_indices:
        return None
    return left_indices, right_indices


def _split_score(
    rows: Sequence[Dict[str, str]],
    indices: Sequence[int],
    column: str,
    column_type: str,
) -> Tuple[int, float]:
    if column_type in {"numeric", "age"}:
        numbers = [
            (
                _parse_age(rows[row_index][column])
                if column_type == "age"
                else _parse_number(rows[row_index][column])
            )
            for row_index in indices
        ]
        known_numbers = [number for number in numbers if number is not None]
        if len(known_numbers) < 2:
            return (0, 0.0)
        return (2, max(known_numbers) - min(known_numbers))

    if column_type == "date":
        known_ordinals = [
            (
                -1.0
                if rows[row_index][column] == "90+"
                else float(parsed_date.toordinal())
            )
            for row_index in indices
            if (
                (parsed_date := _parse_date(rows[row_index][column]))
                is not None
                or rows[row_index][column] == "90+"
            )
        ]
        if len(known_ordinals) < 2:
            return (0, 0.0)
        return (2, max(known_ordinals) - min(known_ordinals))

    unique_values = {rows[row_index][column] for row_index in indices}
    return (1, float(len(unique_values)))


def _partition_l_diverse(
    rows: Sequence[Dict[str, str]],
    indices: Sequence[int],
    sensitive_column: str,
    l: int,
) -> bool:
    sensitive_values = {
        rows[row_index].get(sensitive_column, UNKNOWN_VALUE)
        for row_index in indices
    }
    return len(sensitive_values) >= l


def _generalize_partitions(
    rows: Sequence[Dict[str, str]],
    partitions: Sequence[Sequence[int]],
    quasi_identifier_types: Dict[str, str],
) -> Tuple[List[Dict[str, str]], int, int]:
    anonymized_rows = [dict(row) for row in rows]
    generalized_cells = 0
    suppressed_cells = 0

    for partition in partitions:
        for column, column_type in quasi_identifier_types.items():
            generalized_value, is_suppressed = _generalized_value(
                rows=rows,
                partition=partition,
                column=column,
                column_type=column_type,
            )
            for row_index in partition:
                original_value = anonymized_rows[row_index][column]
                anonymized_rows[row_index][column] = generalized_value
                if original_value != generalized_value:
                    generalized_cells += 1
                if is_suppressed:
                    suppressed_cells += 1

    return anonymized_rows, generalized_cells, suppressed_cells


def _generalized_value(
    rows: Sequence[Dict[str, str]],
    partition: Sequence[int],
    column: str,
    column_type: str,
) -> Tuple[str, bool]:
    values = [rows[row_index][column] for row_index in partition]
    if any(value == UNKNOWN_VALUE for value in values):
        return UNKNOWN_VALUE, True

    if column_type == "numeric":
        numbers = [_parse_number(value) for value in values]
        known_numbers = [number for number in numbers if number is not None]
        if not known_numbers:
            return UNKNOWN_VALUE, True
        return (_format_numeric_range(known_numbers), False)

    if column_type == "age":
        ages = [_parse_age(value) for value in values]
        known_ages = [age for age in ages if age is not None]
        if not known_ages:
            return UNKNOWN_VALUE, True
        minimum = min(known_ages)
        maximum = max(known_ages)
        if minimum >= 90:
            return "90+", False
        if maximum >= 90:
            return f"{_format_number(minimum)}-90+", False
        return (_format_numeric_range(known_ages), False)

    if column_type == "date":
        if "90+" in values:
            return "90+", False
        dates = [_parse_date(value) for value in values]
        known_dates = [parsed_date for parsed_date in dates if parsed_date is not None]
        if not known_dates:
            return UNKNOWN_VALUE, True
        min_date = min(known_dates)
        max_date = max(known_dates)
        if min_date.year == max_date.year and min_date.month == max_date.month:
            return f"{min_date.year:04d}-{min_date.month:02d}", False
        if min_date.year == max_date.year:
            return f"{min_date.year:04d}", False
        return f"{min_date.year:04d}-{max_date.year:04d}", False

    if column_type == "zip":
        return _generalize_zip(values)

    return SUPPRESSED_VALUE, True


def _generalize_zip(values: Sequence[str]) -> Tuple[str, bool]:
    if any(value == UNKNOWN_VALUE for value in values):
        return UNKNOWN_VALUE, True

    normalized_values = [re.sub(r"[^0-9]", "", value) for value in values]
    if not normalized_values or any(len(value) < 5 for value in normalized_values):
        return SUPPRESSED_VALUE, True

    prefixes = {value[:3] for value in normalized_values}
    if len(prefixes) == 1:
        return f"{next(iter(prefixes))}**", False
    return SUPPRESSED_VALUE, True


def _format_number(number: float) -> str:
    if number.is_integer():
        return str(int(number))
    return f"{number:.2f}".rstrip("0").rstrip(".")


def _format_numeric_range(numbers: Sequence[float]) -> str:
    minimum = min(numbers)
    maximum = max(numbers)
    separator = " to " if minimum < 0 or maximum < 0 else "-"
    return f"{_format_number(minimum)}{separator}{_format_number(maximum)}"


def _safe_rate(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return round(numerator / denominator, 6)


def _equivalence_groups(
    rows: Sequence[Dict[str, str]],
    quasi_identifiers: Sequence[str],
) -> Dict[Tuple[str, ...], List[int]]:
    groups: Dict[Tuple[str, ...], List[int]] = defaultdict(list)
    for row_index, row in enumerate(rows):
        key = tuple(row[column] for column in quasi_identifiers)
        groups[key].append(row_index)
    return dict(groups)


def _l_diversity_satisfied(
    rows: Sequence[Dict[str, str]],
    equivalence_groups: Dict[Tuple[str, ...], List[int]],
    sensitive_column: str,
    l: int,
) -> bool:
    if not equivalence_groups:
        return False

    for row_indices in equivalence_groups.values():
        sensitive_values = {
            rows[row_index].get(sensitive_column, UNKNOWN_VALUE)
            for row_index in row_indices
        }
        if len(sensitive_values) < l:
            return False
    return True


def _write_csv(
    retained_columns: Sequence[str],
    anonymized_rows: Sequence[Dict[str, str]],
) -> str:
    buffer = io.StringIO()
    writer = csv.DictWriter(
        buffer,
        fieldnames=list(retained_columns),
        lineterminator="\n",
    )
    writer.writeheader()
    for row in anonymized_rows:
        writer.writerow(row)
    return buffer.getvalue()
