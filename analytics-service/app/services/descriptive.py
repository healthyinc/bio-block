"""Descriptive statistics engine — column stats, type classification, correlations."""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
import pandas as pd


def compute_column_stats(series: pd.Series) -> dict:
    """Compute summary stats for a single numeric column."""
    clean = series.dropna()
    return {
        "count": int(clean.count()),
        "mean": round(float(clean.mean()), 4),
        "std": round(float(clean.std()), 4),
        "min": float(clean.min()),
        "max": float(clean.max()),
        "median": float(clean.median()),
        "missing_pct": round(float(series.isna().mean() * 100), 2),
    }


def classify_columns(df: pd.DataFrame) -> dict:
    """Classify each column as numeric, categorical, or datetime."""
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
    """Pearson correlation matrix. Returns None if < 2 columns."""
    if len(columns) < 2:
        return None
    return df[columns].corr().round(4).to_dict()


def run_descriptive_analysis(df: pd.DataFrame, columns: Optional[list] = None) -> dict:
    """Run full descriptive analysis. Auto-selects numeric columns if none specified."""
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
