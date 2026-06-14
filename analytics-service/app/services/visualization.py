
from __future__ import annotations

import base64
import io
from contextlib import contextmanager
from typing import Any, Dict, List, Optional

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for server use
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency
import matplotlib.dates as mdates


def _is_datetime(series: pd.Series) -> bool:
    """Return True if *series* has a datetime dtype."""
    return pd.api.types.is_datetime64_any_dtype(series)


def _is_categorical(series: pd.Series) -> bool:
    """Return True if *series* should be treated as categorical.

    Numeric and datetime columns are NOT categorical.
    """
    if np.issubdtype(series.dtype, np.number):
        return False
    if _is_datetime(series):
        return False
    return True


def _coerce_datetime(series: pd.Series) -> pd.Series:
    """Attempt to parse a string series as datetime.

    Returns the parsed series if successful, otherwise the original.
    """
    if _is_datetime(series):
        return series
    if series.dtype == object:
        try:
            parsed = pd.to_datetime(series, errors='coerce')
            # Only accept if >50% of non-null values parsed successfully
            if parsed.notna().sum() > 0.5 * series.notna().sum():
                return parsed
        except Exception:
            pass
    return series


@contextmanager
def _safe_figure(figsize=(8, 5)):
    """Context manager that guarantees ``plt.close(fig)``.

    Usage::

        with _safe_figure() as (fig, ax):
            ax.plot(...)
            image_b64 = _fig_to_base64(fig)

    If any exception is raised inside the block, the figure is still
    closed so that matplotlib does not leak memory.
    """
    fig, ax = plt.subplots(figsize=figsize)
    try:
        yield fig, ax
    finally:
        plt.close(fig)


def _build_histogram_config(
    df: pd.DataFrame, column: str, bins: int = 20
) -> Dict[str, Any]:
    """Build histogram chart config and render PNG.

    For numeric columns a standard histogram is drawn.  For categorical
    (qualitative) columns a **frequency bar chart** is rendered instead.
    For datetime columns a **time-binned histogram** is rendered.
    """
    series = df[column].dropna()

    # --- Datetime: time-binned histogram ---
    if _is_datetime(series):
        with _safe_figure() as (fig, ax):
            ax.hist(series, bins=bins, edgecolor="black", alpha=0.75, color="#4C9AFF")
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
            ax.xaxis.set_major_locator(mdates.AutoDateLocator())
            ax.set_xlabel(column)
            ax.set_ylabel("Frequency")
            ax.set_title(f"Distribution of {column}")
            plt.xticks(rotation=45, ha="right")
            plt.tight_layout()
            image_b64 = _fig_to_base64(fig)

        chart_config = {
            "type": "histogram",
            "labels": [],
            "datasets": [{"label": column, "data": []}],
            "options": {"xLabel": column, "yLabel": "Frequency", "datetime": True, "bins": bins},
        }
        return {"chart_config": chart_config, "image": image_b64}

    # --- Categorical: frequency bar chart ---
    if _is_categorical(series):
        counts = series.value_counts().sort_index()
        labels = [str(x) for x in counts.index.tolist()]
        values = counts.values.tolist()

        chart_config = {
            "type": "histogram",
            "labels": labels,
            "datasets": [{"label": column, "data": values}],
            "options": {"xLabel": column, "yLabel": "Frequency", "categorical": True},
        }

        with _safe_figure() as (fig, ax):
            colors = plt.cm.Set2(np.linspace(0, 1, len(labels)))
            ax.bar(labels, values, edgecolor="black", alpha=0.8, color=colors)
            ax.set_xlabel(column)
            ax.set_ylabel("Frequency")
            ax.set_title(f"Distribution of {column}")
            plt.xticks(rotation=45, ha="right")
            plt.tight_layout()
            image_b64 = _fig_to_base64(fig)
        return {"chart_config": chart_config, "image": image_b64}

    # --- Numeric: standard histogram ---
    counts, edges = np.histogram(series, bins=bins)
    labels = [f"{edges[i]:.1f}-{edges[i+1]:.1f}" for i in range(len(counts))]

    chart_config = {
        "type": "histogram",
        "labels": labels,
        "datasets": [
            {
                "label": column,
                "data": counts.tolist(),
            }
        ],
        "options": {"xLabel": column, "yLabel": "Frequency", "bins": bins},
    }

    with _safe_figure() as (fig, ax):
        ax.hist(series, bins=bins, edgecolor="black", alpha=0.75, color="#4C9AFF")
        ax.set_xlabel(column)
        ax.set_ylabel("Frequency")
        ax.set_title(f"Distribution of {column}")
        plt.tight_layout()
        image_b64 = _fig_to_base64(fig)

    return {"chart_config": chart_config, "image": image_b64}


def _encode_categorical_axis(series: pd.Series):
    """Map a categorical series to integer positions and return (codes, labels)."""
    categories = series.dropna().unique()
    cat_map = {cat: idx for idx, cat in enumerate(categories)}
    codes = series.map(cat_map)
    return codes, categories.tolist(), cat_map


def _serialize_val(val):
    """Convert a value to a JSON-serializable type.

    Handles Timestamps, numpy types, and regular Python types.
    """
    if hasattr(val, 'isoformat'):
        return val.isoformat()
    try:
        return float(val)
    except (TypeError, ValueError):
        return str(val)


def _build_scatter_config(
    df: pd.DataFrame,
    x_column: str,
    y_column: str,
    group_column: Optional[str] = None,
) -> Dict[str, Any]:
    """Build scatter plot config and render PNG.

    Categorical (qualitative) axes are automatically label-encoded to
    integer positions so the scatter plot can be rendered with readable
    tick labels.  Datetime axes are plotted on a continuous timeline.
    """
    work = df[[x_column, y_column]].dropna().copy()

    x_is_cat = _is_categorical(work[x_column])
    y_is_cat = _is_categorical(work[y_column])
    x_is_dt = _is_datetime(work[x_column])
    y_is_dt = _is_datetime(work[y_column])

    x_tick_labels: Optional[list] = None
    y_tick_labels: Optional[list] = None

    if x_is_cat:
        codes, x_tick_labels, _ = _encode_categorical_axis(work[x_column])
        work[x_column] = codes
    if y_is_cat:
        codes, y_tick_labels, _ = _encode_categorical_axis(work[y_column])
        work[y_column] = codes

    # Add jitter for categorical axes to avoid overlapping points
    # Datetime and numeric axes do NOT get jitter
    jitter_x = np.random.uniform(-0.15, 0.15, len(work)) if x_is_cat else 0
    jitter_y = np.random.uniform(-0.15, 0.15, len(work)) if y_is_cat else 0

    datasets: List[Dict[str, Any]] = []

    with _safe_figure() as (fig, ax):
        if group_column and group_column in df.columns:
            groups = df.loc[work.index, group_column].dropna().unique()
            colors = plt.cm.tab10(np.linspace(0, 1, min(len(groups), 10)))
            for idx, group in enumerate(groups):
                mask = df.loc[work.index, group_column] == group
                subset = work.loc[mask]
                color = colors[idx % len(colors)]
                sx = subset[x_column].values + (jitter_x[mask.values] if x_is_cat else 0)
                sy = subset[y_column].values + (jitter_y[mask.values] if y_is_cat else 0)
                ax.scatter(sx, sy, label=str(group), alpha=0.7, color=color)
                datasets.append({
                    "label": str(group),
                    "data": [
                        {"x": _serialize_val(x), "y": _serialize_val(y)}
                        for x, y in zip(subset[x_column], subset[y_column])
                    ],
                })
            ax.legend()
        else:
            sx = work[x_column].values + jitter_x
            sy = work[y_column].values + jitter_y
            ax.scatter(sx, sy, alpha=0.7, color="#4C9AFF")
            datasets.append({
                "label": f"{x_column} vs {y_column}",
                "data": [
                    {"x": _serialize_val(x), "y": _serialize_val(y)}
                    for x, y in zip(work[x_column], work[y_column])
                ],
            })

        if x_tick_labels is not None:
            ax.set_xticks(range(len(x_tick_labels)))
            ax.set_xticklabels([str(l) for l in x_tick_labels], rotation=45, ha="right")
        if y_tick_labels is not None:
            ax.set_yticks(range(len(y_tick_labels)))
            ax.set_yticklabels([str(l) for l in y_tick_labels])

        # Format datetime axes
        if x_is_dt:
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
            ax.xaxis.set_major_locator(mdates.AutoDateLocator())
            plt.xticks(rotation=45, ha="right")
        if y_is_dt:
            ax.yaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))

        ax.set_xlabel(x_column)
        ax.set_ylabel(y_column)
        ax.set_title(f"{x_column} vs {y_column}")
        plt.tight_layout()
        image_b64 = _fig_to_base64(fig)

    options: Dict[str, Any] = {"xLabel": x_column, "yLabel": y_column}
    if x_tick_labels is not None:
        options["xCategories"] = [str(l) for l in x_tick_labels]
    if y_tick_labels is not None:
        options["yCategories"] = [str(l) for l in y_tick_labels]
    if x_is_dt:
        options["xDatetime"] = True
    if y_is_dt:
        options["yDatetime"] = True

    chart_config = {
        "type": "scatter",
        "datasets": datasets,
        "options": options,
    }
    return {"chart_config": chart_config, "image": image_b64}


def _build_box_config(
    df: pd.DataFrame,
    columns: List[str],
    group_column: Optional[str] = None,
) -> Dict[str, Any]:
    """Build box plot config and render PNG.

    Categorical columns are silently skipped.  If *no* numeric columns
    remain, a clear ``ValueError`` is raised instead of crashing.
    """
    # Validate before creating any figure
    if group_column and group_column in df.columns and len(columns) == 1:
        col = columns[0]
        if _is_categorical(df[col]):
            raise ValueError(
                f"Column '{col}' is categorical. "
                "box plot requires numeric columns. "
                "Use 'bar' or 'pie' chart for categorical data."
            )
    else:
        col = None
        numeric_cols = [c for c in columns if c in df.columns and not _is_categorical(df[c])]
        skipped = [c for c in columns if c in df.columns and _is_categorical(df[c])]

        if not numeric_cols:
            raise ValueError(
                f"All requested columns are categorical ({', '.join(skipped)}). "
                "box plot requires at least one numeric column. "
                "Use 'bar' or 'pie' chart for categorical data, "
                "or 'heatmap' for a categorical association matrix."
            )

    with _safe_figure() as (fig, ax):
        if col is not None:
            # Single column grouped by group_column
            groups = df[group_column].dropna().unique().tolist()
            data_per_group = [
                df.loc[df[group_column] == g, col].dropna().tolist() for g in groups
            ]
            ax.boxplot(data_per_group, tick_labels=[str(g) for g in groups])
            ax.set_ylabel(col)
            ax.set_title(f"{col} by {group_column}")

            datasets = []
            for g, d in zip(groups, data_per_group):
                s = pd.Series(d)
                if s.empty:
                    datasets.append({"label": str(g), "min": None, "q1": None, "median": None, "q3": None, "max": None})
                else:
                    datasets.append({
                        "label": str(g),
                        "min": float(s.min()),
                        "q1": float(s.quantile(0.25)),
                        "median": float(s.median()),
                        "q3": float(s.quantile(0.75)),
                        "max": float(s.max()),
                    })
            chart_config = {"type": "box", "labels": [str(g) for g in groups], "datasets": datasets}
        else:
            # Multiple numeric columns side by side
            data = [df[c].dropna().tolist() for c in numeric_cols]
            ax.boxplot(data, tick_labels=numeric_cols)
            ax.set_title("Box Plot")
            ax.set_ylabel("Value")

            datasets = []
            for c in numeric_cols:
                s = df[c].dropna()
                datasets.append({
                    "label": c,
                    "min": float(s.min()),
                    "q1": float(s.quantile(0.25)),
                    "median": float(s.median()),
                    "q3": float(s.quantile(0.75)),
                    "max": float(s.max()),
                })
            chart_config = {
                "type": "box",
                "labels": numeric_cols,
                "datasets": datasets,
            }
            if skipped:
                chart_config["skipped_categorical"] = skipped

        plt.tight_layout()
        image_b64 = _fig_to_base64(fig)

    return {"chart_config": chart_config, "image": image_b64}


def _cramers_v(x: pd.Series, y: pd.Series) -> float:
    """Compute Cramér's V statistic for two categorical series."""
    ct = pd.crosstab(x, y)
    n = ct.sum().sum()
    if n == 0:
        return 0.0
    chi2 = chi2_contingency(ct)[0]
    min_dim = min(ct.shape[0], ct.shape[1]) - 1
    if min_dim == 0:
        return 0.0
    return float(np.sqrt(chi2 / (n * min_dim)))


def _build_heatmap_config(
    df: pd.DataFrame, columns: List[str], method: str = "pearson"
) -> Dict[str, Any]:
    """Build heatmap config and render PNG.

    For numeric columns a Pearson/Spearman correlation matrix is drawn.
    For categorical (qualitative) columns a **Cramér's V** association
    matrix is drawn instead.  If a mix is provided, numeric columns get
    a correlation matrix while categorical columns are noted separately.
    """
    numeric_cols = [c for c in columns if c in df.columns and not _is_categorical(df[c])]
    cat_cols = [c for c in columns if c in df.columns and _is_categorical(df[c])]

    # --- Pure categorical: Cramér's V association matrix ---
    if len(cat_cols) >= 2 and len(numeric_cols) < 2:
        n = len(cat_cols)
        assoc = pd.DataFrame(np.ones((n, n)), index=cat_cols, columns=cat_cols)
        for i in range(n):
            for j in range(i + 1, n):
                v = _cramers_v(df[cat_cols[i]].dropna(), df[cat_cols[j]].dropna())
                assoc.iloc[i, j] = round(v, 4)
                assoc.iloc[j, i] = round(v, 4)

        with _safe_figure(figsize=(8, 6)) as (fig, ax):
            im = ax.imshow(assoc.values, cmap="YlOrRd", vmin=0, vmax=1, aspect="auto")
            ax.set_xticks(range(n))
            ax.set_yticks(range(n))
            ax.set_xticklabels(cat_cols, rotation=45, ha="right")
            ax.set_yticklabels(cat_cols)

            for i in range(n):
                for j in range(n):
                    val = assoc.values[i, j]
                    ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                            color="white" if val > 0.5 else "black", fontsize=9)

            fig.colorbar(im)
            ax.set_title("Cramér's V Association Heatmap")
            plt.tight_layout()
            image_b64 = _fig_to_base64(fig)

        chart_config = {
            "type": "heatmap",
            "labels": cat_cols,
            "data": assoc.round(4).to_dict(),
            "options": {"method": "cramers_v", "categorical": True},
        }
        return {"chart_config": chart_config, "image": image_b64}

    # --- Numeric correlation matrix (original path) ---
    if len(numeric_cols) < 2:
        raise ValueError(
            "Heatmap requires at least 2 numeric columns for correlation "
            "or at least 2 categorical columns for association (Cramér's V)."
        )

    corr = df[numeric_cols].corr(method=method).round(4)

    with _safe_figure(figsize=(8, 6)) as (fig, ax):
        im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
        ax.set_xticks(range(len(numeric_cols)))
        ax.set_yticks(range(len(numeric_cols)))
        ax.set_xticklabels(numeric_cols, rotation=45, ha="right")
        ax.set_yticklabels(numeric_cols)

        # Add correlation values as text annotations
        for i in range(len(numeric_cols)):
            for j in range(len(numeric_cols)):
                ax.text(j, i, f"{corr.values[i, j]:.2f}", ha="center", va="center",
                        color="white" if abs(corr.values[i, j]) > 0.5 else "black", fontsize=9)

        fig.colorbar(im)
        ax.set_title(f"{method.capitalize()} Correlation Heatmap")
        plt.tight_layout()
        image_b64 = _fig_to_base64(fig)

    chart_config = {
        "type": "heatmap",
        "labels": numeric_cols,
        "data": corr.to_dict(),
        "options": {"method": method},
    }
    return {"chart_config": chart_config, "image": image_b64}


def _build_bar_config(
    df: pd.DataFrame,
    x_column: str,
    y_column: Optional[str] = None,
    aggregation: str = "count",
) -> Dict[str, Any]:
    """Build bar chart config and render PNG."""
    if y_column and y_column in df.columns and aggregation != "count":
        grouped = df.groupby(x_column)[y_column]
        if aggregation == "mean":
            agg_data = grouped.mean()
        elif aggregation == "sum":
            agg_data = grouped.sum()
        else:
            agg_data = grouped.count()
        y_label = f"{aggregation.capitalize()} of {y_column}"
    else:
        agg_data = df[x_column].value_counts().sort_index()
        y_label = "Count"

    labels = [str(x) for x in agg_data.index.tolist()]
    values = [float(v) for v in agg_data.values.tolist()]

    with _safe_figure() as (fig, ax):
        ax.bar(labels, values, color="#4C9AFF", edgecolor="black", alpha=0.8)
        ax.set_xlabel(x_column)
        ax.set_ylabel(y_label)
        ax.set_title(f"{y_label} by {x_column}")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        image_b64 = _fig_to_base64(fig)

    chart_config = {
        "type": "bar",
        "labels": labels,
        "datasets": [{"label": y_label, "data": values}],
        "options": {"xLabel": x_column, "yLabel": y_label, "aggregation": aggregation},
    }
    return {"chart_config": chart_config, "image": image_b64}


def _build_line_config(
    df: pd.DataFrame,
    x_column: str,
    y_column: str,
    aggregation: str = "mean",
) -> Dict[str, Any]:
    """Build line chart config and render PNG.

    Datetime x-axes are parsed and sorted chronologically.
    Categorical x-axes are aggregated by group.
    """
    work = df[[x_column, y_column]].dropna().copy()

    # Attempt datetime coercion for the x-axis
    work[x_column] = _coerce_datetime(work[x_column])
    x_is_dt = _is_datetime(work[x_column])

    # Sort chronologically or numerically
    work = work.sort_values(x_column)

    options: Dict[str, Any] = {"xLabel": x_column, "yLabel": y_column, "aggregation": aggregation}

    with _safe_figure() as (fig, ax):
        if x_is_dt:
            # --- Datetime x-axis: plot as time series ---
            ax.plot(work[x_column], work[y_column], marker="o", color="#4C9AFF",
                    linewidth=2, markersize=4)
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
            ax.xaxis.set_major_locator(mdates.AutoDateLocator())
            x_vals = [v.isoformat() if hasattr(v, 'isoformat') else str(v)
                      for v in work[x_column].tolist()]
            y_vals = [float(v) for v in work[y_column].tolist()]
            options["datetime"] = True

        elif _is_categorical(work[x_column]):
            # --- Categorical x-axis: aggregate ---
            grouped = work.groupby(x_column)[y_column]
            if aggregation == "mean":
                agg = grouped.mean()
            elif aggregation == "sum":
                agg = grouped.sum()
            else:
                agg = grouped.count()
            x_vals = [str(x) for x in agg.index.tolist()]
            y_vals = [float(v) for v in agg.values.tolist()]
            ax.plot(x_vals, y_vals, marker="o", color="#4C9AFF", linewidth=2, markersize=4)

        else:
            # --- Numeric x-axis ---
            x_vals = [float(v) for v in work[x_column].tolist()]
            y_vals = [float(v) for v in work[y_column].tolist()]
            ax.plot(x_vals, y_vals, marker="o", color="#4C9AFF", linewidth=2, markersize=4)

        ax.set_xlabel(x_column)
        ax.set_ylabel(y_column)
        ax.set_title(f"{y_column} over {x_column}")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        image_b64 = _fig_to_base64(fig)

    chart_config = {
        "type": "line",
        "labels": x_vals if all(isinstance(v, str) for v in x_vals) else None,
        "datasets": [
            {
                "label": y_column,
                "data": (
                    y_vals
                    if all(isinstance(v, str) for v in x_vals)
                    else [{"x": x, "y": y} for x, y in zip(x_vals, y_vals)]
                ),
            }
        ],
        "options": options,
    }
    return {"chart_config": chart_config, "image": image_b64}


def _build_pie_config(
    df: pd.DataFrame,
    column: str,
) -> Dict[str, Any]:
    """Build pie chart config and render PNG."""
    counts = df[column].value_counts()
    labels = [str(x) for x in counts.index.tolist()]
    values = [int(v) for v in counts.values.tolist()]

    with _safe_figure(figsize=(8, 6)) as (fig, ax):
        colors = plt.cm.Set3(np.linspace(0, 1, len(labels)))
        wedges, texts, autotexts = ax.pie(
            values, labels=labels, autopct="%1.1f%%", colors=colors, startangle=90
        )
        ax.set_title(f"Distribution of {column}")
        plt.tight_layout()
        image_b64 = _fig_to_base64(fig)

    chart_config = {
        "type": "pie",
        "labels": labels,
        "datasets": [{"data": values}],
    }
    return {"chart_config": chart_config, "image": image_b64}




def _fig_to_base64(fig: plt.Figure) -> str:
    """Convert a matplotlib figure to a base64-encoded PNG string.

    Note: This function does NOT close the figure.  Callers should use
    ``_safe_figure()`` context manager to guarantee cleanup.
    """
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


def _coerce_numeric_columns(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    """Return a copy of *df* with *columns* coerced to numeric where possible.

    CSV parsers sometimes read numeric data as strings (object dtype).
    This helper applies ``pd.to_numeric(errors='coerce')`` so that
    downstream chart builders (histogram, box, heatmap, scatter, line)
    receive proper numeric arrays.
    """
    df = df.copy()
    for col in columns:
        if col not in df.columns:
            continue
        # Skip columns that are already numeric or datetime
        if np.issubdtype(df[col].dtype, np.number):
            continue
        if _is_datetime(df[col]):
            continue
        converted = pd.to_numeric(df[col], errors="coerce")
        # Only replace if at least one value converted successfully
        if converted.notna().any():
            df[col] = converted
    return df


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

CHART_BUILDERS = {
    "histogram": _build_histogram_config,
    "scatter": _build_scatter_config,
    "box": _build_box_config,
    "heatmap": _build_heatmap_config,
    "bar": _build_bar_config,
    "line": _build_line_config,
    "pie": _build_pie_config,
}

VALID_CHART_TYPES = list(CHART_BUILDERS.keys())


def generate_chart(
    df: pd.DataFrame,
    chart_type: str,
    x_column: Optional[str] = None,
    y_column: Optional[str] = None,
    columns: Optional[List[str]] = None,
    group_column: Optional[str] = None,
    aggregation: str = "count",
    bins: int = 20,
) -> Dict[str, Any]:
    """Generate a chart of the specified type.

    Parameters
    ----------
    df : DataFrame
        The dataset to visualize.
    chart_type : str
        One of: histogram, scatter, box, heatmap, bar, line, pie.
    x_column : str, optional
        X-axis column (required for scatter, bar, line).
    y_column : str, optional
        Y-axis column (required for scatter, line).
    columns : list[str], optional
        Columns for multi-column charts (box, heatmap).
    group_column : str, optional
        Grouping/color column (scatter, box).
    aggregation : str
        Aggregation method for bar/line: count, mean, sum.
    bins : int
        Number of bins for histograms.

    Returns
    -------
    dict with keys: chart_config, image (base64-encoded PNG)
    """
    if chart_type not in CHART_BUILDERS:
        raise ValueError(
            f"Unsupported chart type '{chart_type}'. "
            f"Valid types: {VALID_CHART_TYPES}"
        )

    # --- Coerce columns that should be numeric but were parsed as strings ---
    cols_to_coerce: List[str] = []
    if x_column and x_column in df.columns:
        cols_to_coerce.append(x_column)
    if y_column and y_column in df.columns:
        cols_to_coerce.append(y_column)
    if columns:
        cols_to_coerce.extend(c for c in columns if c in df.columns)
    if cols_to_coerce:
        df = _coerce_numeric_columns(df, cols_to_coerce)

    if chart_type == "histogram":
        if not x_column:
            raise ValueError("histogram requires 'x_column'.")
        if x_column not in df.columns:
            raise ValueError(f"Column '{x_column}' not found in dataset.")
        return _build_histogram_config(df, x_column, bins=bins)

    elif chart_type == "scatter":
        if not x_column or not y_column:
            raise ValueError("scatter requires 'x_column' and 'y_column'.")
        for c in [x_column, y_column]:
            if c not in df.columns:
                raise ValueError(f"Column '{c}' not found in dataset.")
        return _build_scatter_config(df, x_column, y_column, group_column)

    elif chart_type == "box":
        effective_cols = columns if columns else ([x_column] if x_column else [])
        if not effective_cols:
            raise ValueError("box requires 'columns' or 'x_column'.")
        for c in effective_cols:
            if c not in df.columns:
                raise ValueError(f"Column '{c}' not found in dataset.")
        return _build_box_config(df, effective_cols, group_column)

    elif chart_type == "heatmap":
        effective_cols = columns or list(df.select_dtypes(include=[np.number]).columns)
        # When no explicit columns given, re-coerce all auto-detected numeric cols
        if not columns:
            df = _coerce_numeric_columns(df, list(df.columns))
            effective_cols = list(df.select_dtypes(include=[np.number]).columns)
        return _build_heatmap_config(df, effective_cols)

    elif chart_type == "bar":
        if not x_column:
            raise ValueError("bar requires 'x_column'.")
        if x_column not in df.columns:
            raise ValueError(f"Column '{x_column}' not found in dataset.")
        return _build_bar_config(df, x_column, y_column, aggregation)

    elif chart_type == "line":
        if not x_column or not y_column:
            raise ValueError("line requires 'x_column' and 'y_column'.")
        for c in [x_column, y_column]:
            if c not in df.columns:
                raise ValueError(f"Column '{c}' not found in dataset.")
        return _build_line_config(df, x_column, y_column, aggregation)

    elif chart_type == "pie":
        col = x_column or (columns[0] if columns else None)
        if not col:
            raise ValueError("pie requires 'x_column' or 'columns'.")
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found in dataset.")
        return _build_pie_config(df, col)

    # Should never reach here due to the check at the top
    raise ValueError(f"Unhandled chart type: {chart_type}")  # pragma: no cover
