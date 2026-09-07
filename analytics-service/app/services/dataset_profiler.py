"""Dataset profiler for Hypothesis Lab."""

from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd
from scipy import stats as sp_stats

from app.models.demo_schemas import (
    ColumnProfile,
    ColumnType,
    DatasetProfile,
    VariableRole,
)


# ---------------------------------------------------------------------------
# Thresholds
# ---------------------------------------------------------------------------

_IDENTIFIER_UNIQUE_RATIO = 0.95  # ≥95 % unique → likely identifier
_LOW_CARDINALITY_MAX = 20       # ≤20 unique values → treat as categorical/group
_TIME_KEYWORDS = {"date", "time", "timestamp", "year", "month", "day", "week", "period"}
_SUBJECT_KEYWORDS = {"id", "subject", "participant", "patient", "sample", "user", "respondent"}
_HIGH_MISSING_PCT = 50.0        # warn when >50 % missing


def _is_likely_identifier(name: str, series: pd.Series) -> bool:
    """Heuristic: likely an ID column (patient #, names, sequential ints, etc)."""
    if len(series.dropna()) < 3:
        return False

    clean = series.dropna()
    nunique = clean.nunique()
    unique_ratio = nunique / max(len(clean), 1)
    lower_name = name.lower().strip().replace("_", " ").replace("-", " ")

    # Explicit name matches (e.g. "patient #", "patient id", "first name", "last name")
    id_name_patterns = [
        "patient #", "patient id", "patient_id", "pt #", "pt id",
        "subject id", "subject_id", "participant id", "sample id",
        "first name", "last name", "full name", "mrn", "ssn", "uuid"
    ]
    if any(pat in lower_name for pat in id_name_patterns):
        return True

    if lower_name in {"id", "row", "index", "seq", "#", "number", "patient"}:
        return True

    if lower_name.endswith(" id") or lower_name.endswith(" #") or lower_name.endswith("_id"):
        return True

    # High-cardinality checks for numeric sequence IDs or string identifiers
    if unique_ratio >= _IDENTIFIER_UNIQUE_RATIO and nunique >= 5:
        if any(kw in lower_name for kw in _SUBJECT_KEYWORDS) or "name" in lower_name or "#" in lower_name or "num" in lower_name:
            return True
        # Check if numeric values are sequential integers (e.g. 1..N)
        if pd.api.types.is_numeric_dtype(series):
            vals = clean.values
            if np.issubdtype(vals.dtype, np.integer) or (np.all(np.mod(vals, 1) == 0)):
                if np.all(np.diff(np.sort(vals)) == 1):
                    return True

    return False


def _is_likely_time_column(name: str, series: pd.Series) -> bool:
    lower = name.lower().replace("_", " ").replace("-", " ")
    if any(kw in lower for kw in _TIME_KEYWORDS):
        return True
    if pd.api.types.is_datetime64_any_dtype(series):
        return True
    return False


def _is_likely_subject_column(name: str) -> bool:
    lower = name.lower().replace("_", " ").replace("-", " ")
    return any(kw in lower for kw in _SUBJECT_KEYWORDS)


def _infer_column_type(series: pd.Series, name: str = "") -> ColumnType:
    """Infer column type from dtype, name, and data shape."""
    if _is_likely_identifier(name, series):
        return ColumnType.IDENTIFIER

    if pd.api.types.is_datetime64_any_dtype(series):
        return ColumnType.DATETIME

    if pd.api.types.is_bool_dtype(series):
        return ColumnType.BOOLEAN

    if pd.api.types.is_numeric_dtype(series):
        # A numeric column with very few unique values may be categorical
        nunique = series.nunique()
        if nunique <= 2 and set(series.dropna().unique()).issubset({0, 1, 0.0, 1.0}):
            return ColumnType.BOOLEAN
        return ColumnType.NUMERIC

    nunique = series.nunique()
    if nunique <= _LOW_CARDINALITY_MAX:
        return ColumnType.CATEGORICAL

    # Try parsing as dates
    try:
        pd.to_datetime(series.dropna().head(20))
        return ColumnType.DATETIME
    except (ValueError, TypeError):
        pass

    return ColumnType.TEXT


def _profile_numeric(series: pd.Series) -> dict:
    clean = series.dropna().astype(float)
    if len(clean) == 0:
        return {}

    result = {
        "mean": round(float(clean.mean()), 4),
        "median": round(float(clean.median()), 4),
        "std": round(float(clean.std()), 4),
        "min_val": round(float(clean.min()), 4),
        "max_val": round(float(clean.max()), 4),
    }

    if len(clean) >= 3:
        result["skewness"] = round(float(clean.skew()), 4)
        result["kurtosis"] = round(float(clean.kurtosis()), 4)

    # Normality hint — Shapiro-Wilk for small/moderate samples
    if 8 <= len(clean) <= 500:
        _, p = sp_stats.shapiro(clean.values)
        if p > 0.05:
            result["normality_hint"] = "appears_normal"
        else:
            result["normality_hint"] = "likely_non_normal"
    elif len(clean) > 500:
        # large n: SW rejects everything, use skew/kurtosis heuristic instead
        skew = abs(float(clean.skew())) if len(clean) >= 3 else 0.0
        kurt = abs(float(clean.kurtosis())) if len(clean) >= 3 else 0.0
        if skew > 1.5 or kurt > 5.0:
            result["normality_hint"] = "likely_non_normal"
        else:
            result["normality_hint"] = "appears_normal"

    return result


def _profile_categorical(series: pd.Series) -> dict:
    counts = series.value_counts()
    top = [
        {"value": str(v), "count": int(c)}
        for v, c in counts.head(10).items()
    ]
    return {
        "top_values": top,
        "cardinality": int(series.nunique()),
    }


def _profile_datetime(series: pd.Series) -> dict:
    try:
        parsed = pd.to_datetime(series.dropna())
        if len(parsed) == 0:
            return {}
        return {
            "date_min": str(parsed.min()),
            "date_max": str(parsed.max()),
        }
    except Exception:
        return {}


def _suggest_roles(name: str, dtype: ColumnType, series: pd.Series) -> List[VariableRole]:
    """Suggest variable roles from column name and type."""
    roles: List[VariableRole] = []

    if dtype == ColumnType.NUMERIC:
        roles.append(VariableRole.OUTCOME)
        roles.append(VariableRole.COVARIATE)

    if dtype == ColumnType.CATEGORICAL:
        nunique = series.nunique()
        if 2 <= nunique <= 10:
            roles.append(VariableRole.GROUP)
        if nunique == 2:
            roles.append(VariableRole.PREDICTOR)

    if _is_likely_time_column(name, series):
        roles.append(VariableRole.TIME)

    if _is_likely_subject_column(name):
        roles.append(VariableRole.SUBJECT)

    if dtype == ColumnType.IDENTIFIER:
        roles.append(VariableRole.SUBJECT)

    return roles


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def profile_dataset(df: pd.DataFrame) -> DatasetProfile:
    """Profile every column, return DatasetProfile."""
    columns: List[ColumnProfile] = []
    warnings: List[str] = []
    suggested_group: List[str] = []
    suggested_time: List[str] = []
    suggested_subject: List[str] = []
    suggested_outcome: List[str] = []

    for col_name in df.columns:
        series = df[col_name]
        dtype = _infer_column_type(series, col_name)

        missing_count = int(series.isna().sum())
        total_count = len(series)
        missing_pct = round(100.0 * missing_count / max(total_count, 1), 2)

        profile_data = {
            "name": col_name,
            "dtype": dtype,
            "missing_count": missing_count,
            "missing_pct": missing_pct,
            "unique_count": int(series.nunique()),
            "total_count": total_count,
        }

        if dtype == ColumnType.NUMERIC:
            profile_data.update(_profile_numeric(series))
        elif dtype in (ColumnType.CATEGORICAL, ColumnType.TEXT):
            profile_data.update(_profile_categorical(series))
        elif dtype == ColumnType.DATETIME:
            profile_data.update(_profile_datetime(series))
        elif dtype == ColumnType.BOOLEAN:
            vc = series.value_counts()
            profile_data["top_values"] = [
                {"value": str(v), "count": int(c)} for v, c in vc.items()
            ]
            profile_data["cardinality"] = int(series.nunique())

        roles = _suggest_roles(col_name, dtype, series)
        profile_data["suggested_roles"] = roles

        col_profile = ColumnProfile(**profile_data)
        columns.append(col_profile)

        # Aggregate suggestions
        if VariableRole.GROUP in roles:
            suggested_group.append(col_name)
        if VariableRole.TIME in roles:
            suggested_time.append(col_name)
        if VariableRole.SUBJECT in roles:
            suggested_subject.append(col_name)
        if VariableRole.OUTCOME in roles:
            suggested_outcome.append(col_name)

        # Warnings
        if missing_pct > _HIGH_MISSING_PCT:
            warnings.append(
                f"Column '{col_name}' has {missing_pct}% missing values."
            )

        if dtype == ColumnType.NUMERIC and series.nunique() == 1:
            warnings.append(
                f"Column '{col_name}' is constant (single value). "
                "It cannot be used for statistical comparisons."
            )

    # Global warnings
    dup_count = df.duplicated().sum()
    if dup_count > 0:
        warnings.append(
            f"Dataset contains {int(dup_count)} duplicate row(s)."
        )

    if len(df) < 10:
        warnings.append(
            f"Very small dataset ({len(df)} rows). "
            "Statistical tests may have low power."
        )

    return DatasetProfile(
        row_count=len(df),
        column_count=len(df.columns),
        columns=columns,
        warnings=warnings,
        suggested_group_columns=suggested_group,
        suggested_time_columns=suggested_time,
        suggested_subject_columns=suggested_subject,
        suggested_outcome_columns=suggested_outcome,
    )
