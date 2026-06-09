

from __future__ import annotations

from typing import Any, Optional

import numpy as np
import pandas as pd


def compute_column_stats(series: pd.Series) -> dict:
    clean = series.dropna()

    # Guard: return None instead of NaN when column has no data
    if clean.empty:
        return {
            "count": 0,
            "mean": None,
            "std": None,
            "min": None,
            "q1": None,
            "median": None,
            "q3": None,
            "iqr": None,
            "max": None,
            "missing_pct": 100.0,
        }

    q1 = float(clean.quantile(0.25))
    q3 = float(clean.quantile(0.75))
    return {
        "count": int(clean.count()),
        "mean": round(float(clean.mean()), 4),
        "std": round(float(clean.std()), 4),
        "min": float(clean.min()),
        "q1": round(q1, 4),
        "median": float(clean.median()),
        "q3": round(q3, 4),
        "iqr": round(q3 - q1, 4),
        "max": float(clean.max()),
        "missing_pct": round(float(series.isna().mean() * 100), 2),
    }


def classify_columns(df: pd.DataFrame) -> dict:
    result = {}
    for col in df.columns:
        if np.issubdtype(df[col].dtype, np.number):
            result[col] = "numeric"
        elif pd.api.types.is_datetime64_any_dtype(df[col]):
            result[col] = "datetime"
        else:
            result[col] = "categorical"
    return result


def compute_correlation_matrix(df: pd.DataFrame, columns: list) -> Optional[dict]:
    """Pearson + Spearman correlation matrices. None if < 2 columns."""
    if len(columns) < 2:
        return None
    return {
        "pearson": df[columns].corr(method="pearson").round(4).to_dict(),
        "spearman": df[columns].corr(method="spearman").round(4).to_dict(),
    }


def run_descriptive_analysis(df: pd.DataFrame, columns: Optional[list] = None) -> dict:
    col_types = classify_columns(df)

    if columns:
        target = [c for c in columns if c in df.columns]
    else:
        target = [c for c, t in col_types.items() if t == "numeric"]

    stats = {}
    numeric_found = []
    for col in target:
        if col_types.get(col) != "numeric":
            continue
        stats[col] = compute_column_stats(df[col])
        numeric_found.append(col)

    corr = compute_correlation_matrix(df, numeric_found)
    if corr:
        stats["_correlations"] = corr

    return {
        "results": stats,
        "column_types": col_types,
        "columns_analyzed": numeric_found,
    }
