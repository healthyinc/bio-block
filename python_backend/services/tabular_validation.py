"""Serialized-output validation for anonymized tabular data.

The tabular pipeline reports which columns it removed and which rows it kept,
but until now nothing re-read the CSV it actually serialized. A column that
survived serialization, or a removed identifier value that reappeared inside a
retained free-text cell, would have been reported as a clean result.

This module re-parses the emitted CSV and checks it against both the reported
plan and the original input. It returns categories, counts, and column names -
never a leaked value.
"""

from __future__ import annotations

import csv
import io
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Set

# Values shorter than this are too collision-prone to treat as a leak signal
# ("F", "31", "NY" would match half a dataset).
MIN_LEAK_CANDIDATE_LENGTH = 5
# Bound the comparison so a wide input cannot make validation quadratic.
MAX_LEAK_CANDIDATES = 5000

VALIDATION_PASSED = "passed"
VALIDATION_FAILED = "failed"

FAILURE_UNPARSEABLE = "serialized_output_unparseable"
FAILURE_HEADER_MISMATCH = "serialized_header_does_not_match_plan"
FAILURE_ROW_COUNT_MISMATCH = "serialized_row_count_does_not_match_plan"
FAILURE_REMOVED_COLUMN_PRESENT = "removed_column_present_in_output"
FAILURE_IDENTIFIER_VALUE_PRESENT = "removed_identifier_value_present_in_output"


def _distinct_values(
    records: Iterable[Mapping[str, Any]],
    columns: Sequence[str],
) -> Set[str]:
    """Collect distinct, long-enough values from the removed identifier columns."""
    candidates: Set[str] = set()
    for record in records:
        for column in columns:
            value = str(record.get(column, "") or "").strip()
            if len(value) >= MIN_LEAK_CANDIDATE_LENGTH:
                candidates.add(value)
            if len(candidates) >= MAX_LEAK_CANDIDATES:
                return candidates
    return candidates


def validate_serialized_csv(
    serialized_csv: str,
    expected_columns: Sequence[str],
    expected_row_count: int,
    removed_columns: Sequence[str],
    source_records: Sequence[Mapping[str, Any]] = (),
) -> Dict[str, Any]:
    """Re-read the emitted CSV and confirm it matches the reported plan.

    ``source_records`` are the parsed input rows. When supplied, the values of
    every removed identifier column are checked for reappearance anywhere in
    the output, which catches an identifier echoed inside a retained free-text
    column.
    """
    failures: List[str] = []
    leaked_columns: List[str] = []

    try:
        reader = csv.reader(io.StringIO(serialized_csv), strict=True)
        rows = list(reader)
    except Exception:
        return _result([FAILURE_UNPARSEABLE], [], 0)

    if not rows:
        return _result([FAILURE_UNPARSEABLE], [], 0)

    header = [str(column) for column in rows[0]]
    body = rows[1:]

    if header != [str(column) for column in expected_columns]:
        failures.append(FAILURE_HEADER_MISMATCH)
    if len(body) != int(expected_row_count):
        failures.append(FAILURE_ROW_COUNT_MISMATCH)

    normalized_header = {column.strip().casefold() for column in header}
    for column in removed_columns:
        if str(column).strip().casefold() in normalized_header:
            leaked_columns.append(str(column))
    if leaked_columns:
        failures.append(FAILURE_REMOVED_COLUMN_PRESENT)

    leaked_value_count = 0
    if source_records and removed_columns:
        candidates = _distinct_values(source_records, removed_columns)
        if candidates:
            haystack = serialized_csv
            leaked_value_count = sum(
                1 for candidate in candidates if candidate in haystack
            )
        if leaked_value_count:
            failures.append(FAILURE_IDENTIFIER_VALUE_PRESENT)

    return _result(failures, leaked_columns, leaked_value_count)


def _result(
    failures: Sequence[str],
    leaked_columns: Sequence[str],
    leaked_value_count: int,
) -> Dict[str, Any]:
    return {
        "serialized_output_validation": (
            VALIDATION_PASSED if not failures else VALIDATION_FAILED
        ),
        "validation_failures": sorted(set(failures)),
        # Column names are part of the schema, not patient values.
        "leaked_columns": sorted(set(leaked_columns)),
        "leaked_value_count": int(leaked_value_count),
    }
