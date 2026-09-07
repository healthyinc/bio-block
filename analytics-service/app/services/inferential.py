

from __future__ import annotations

import math
import itertools
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
from statsmodels.formula.api import ols as sm_ols
from statsmodels.stats.anova import anova_lm

SHAPIRO_MAX_N = 5000

COHENS_D_THRESHOLDS = {"small": 0.2, "medium": 0.5, "large": 0.8}
RANK_BISERIAL_THRESHOLDS = {"small": 0.1, "medium": 0.3, "large": 0.5}
ETA_SQUARED_THRESHOLDS = {"small": 0.01, "medium": 0.06, "large": 0.14}
EPSILON_SQUARED_THRESHOLDS = {"small": 0.01, "medium": 0.06, "large": 0.14}
KENDALLS_W_THRESHOLDS = {"small": 0.1, "medium": 0.3, "large": 0.5}

_TEST_DISPLAY_NAMES = {
    "students_ttest": "Student's t-test",
    "welchs_ttest": "Welch's t-test",
    "mann_whitney_u": "Mann-Whitney U test",
    "paired_ttest": "paired t-test",
    "wilcoxon_signed_rank": "Wilcoxon signed-rank test",
    "one_sample_ttest": "one-sample t-test",
    "one_sample_wilcoxon": "one-sample Wilcoxon signed-rank test",
    "one_way_anova": "one-way ANOVA (F-test)",
    "welchs_anova": "Welch's ANOVA",
    "kruskal_wallis": "Kruskal-Wallis H test",
    "two_way_anova": "two-way ANOVA",
    "repeated_measures_anova": "repeated-measures ANOVA",
    "friedman": "Friedman test",
}


def check_sdc(n: int, min_suppress: int = 5, min_warn: int = 30) -> Dict[str, Any]:
    """Statistical Disclosure Control: suppress n<5, warn n<30."""
    if n < min_suppress:
        return {
            "status": "suppress",
            "message": (
                f"SDC violation: group size n={n} is below the minimum "
                f"threshold of {min_suppress}. Results suppressed to "
                "protect individual privacy."
            ),
        }
    if n < min_warn:
        return {
            "status": "warn",
            "message": (
                f"Small cohort warning: group size n={n} is below {min_warn}. "
                "Results are shown but may not be statistically reliable "
                "and could risk individual identification."
            ),
        }
    return {"status": "ok", "message": None}


def check_normality(data: np.ndarray, alpha: float = 0.05) -> Dict[str, Any]:
    """Shapiro-Wilk for n ≤ 5000, Kolmogorov-Smirnov for larger samples."""
    data = np.asarray(data, dtype=float)
    data = data[~np.isnan(data)]
    n = len(data)

    if n < 3:
        return {
            "test": "insufficient_data",
            "statistic": None,
            "p_value": None,
            "normal": False,
            "note": f"Cannot test normality with n={n} (minimum 3 required).",
        }

    if n <= SHAPIRO_MAX_N:
        stat, p = stats.shapiro(data)
        return {
            "test": "shapiro_wilk",
            "statistic": round(float(stat), 6),
            "p_value": round(float(p), 6),
            "normal": bool(p > alpha),
            "note": None,
        }

    stat, p = stats.kstest(data, "norm", args=(np.mean(data), np.std(data, ddof=1)))
    return {
        "test": "kolmogorov_smirnov",
        "statistic": round(float(stat), 6),
        "p_value": round(float(p), 6),
        "normal": bool(p > alpha),
        "note": (
            f"Switched to Kolmogorov-Smirnov test: n={n} exceeds "
            f"Shapiro-Wilk reliable range (≤{SHAPIRO_MAX_N})."
        ),
    }


def check_equal_variance(
    group1: np.ndarray, group2: np.ndarray, alpha: float = 0.05
) -> Dict[str, Any]:
    """Levene's test for equality of variances."""
    stat, p = stats.levene(group1, group2)
    return {
        "test": "levenes",
        "statistic": round(float(stat), 6),
        "p_value": round(float(p), 6),
        "equal_variance": bool(p > alpha),
    }


# -- Effect sizes --

def compute_cohens_d(group1: np.ndarray, group2: np.ndarray) -> float:
    """Cohen's d with pooled standard deviation."""
    n1, n2 = len(group1), len(group2)
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
    pooled_std = math.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    if pooled_std == 0:
        return 0.0
    return float((np.mean(group1) - np.mean(group2)) / pooled_std)


def compute_cohens_d_paired(diffs: np.ndarray) -> float:
    """Cohen's d for paired samples (mean of diffs / SD of diffs)."""
    std = np.std(diffs, ddof=1)
    return 0.0 if std == 0 else float(np.mean(diffs) / std)


def compute_cohens_d_one_sample(sample: np.ndarray, population_mean: float) -> float:
    """Cohen's d for one-sample test."""
    std = np.std(sample, ddof=1)
    return 0.0 if std == 0 else float((np.mean(sample) - population_mean) / std)


def compute_rank_biserial(U: float, n1: int, n2: int) -> float:
    """Rank-biserial r for Mann-Whitney U: r = 1 - 2U/(n1*n2)."""
    denom = n1 * n2
    return 0.0 if denom == 0 else float(1 - (2 * U) / denom)


def classify_effect_size(value: float, metric: str) -> str:
    """Classify as negligible/small/medium/large using standard thresholds."""
    abs_val = abs(value)
    _threshold_map = {
        "rank_biserial_r": RANK_BISERIAL_THRESHOLDS,
        "eta_squared": ETA_SQUARED_THRESHOLDS,
        "epsilon_squared": EPSILON_SQUARED_THRESHOLDS,
        "kendalls_w": KENDALLS_W_THRESHOLDS,
    }
    thresholds = _threshold_map.get(metric, COHENS_D_THRESHOLDS)
    if abs_val >= thresholds["large"]:
        return "large"
    if abs_val >= thresholds["medium"]:
        return "medium"
    if abs_val >= thresholds["small"]:
        return "small"
    return "negligible"


# -- Individual test runners --

def _make_result(test_name, stat, p, alpha, d_or_r, metric):
    """Build the common result dict returned by every test runner."""
    return {
        "test_used": test_name,
        "statistic": round(float(stat), 6),
        "p_value": round(float(p), 6),
        "significant": bool(p <= alpha),
        "effect_size": {
            "metric": metric,
            "value": round(d_or_r, 4),
            "magnitude": classify_effect_size(d_or_r, metric),
        },
    }


def run_students_ttest(group1, group2, alpha=0.05, alternative="two-sided"):
    """Student's t-test (equal variances assumed)."""
    stat, p = stats.ttest_ind(group1, group2, equal_var=True, alternative=alternative)
    return _make_result("students_ttest", stat, p, alpha,
                        compute_cohens_d(group1, group2), "cohens_d")


def run_welchs_ttest(group1, group2, alpha=0.05, alternative="two-sided"):
    """Welch's t-test (unequal variances)."""
    stat, p = stats.ttest_ind(group1, group2, equal_var=False, alternative=alternative)
    return _make_result("welchs_ttest", stat, p, alpha,
                        compute_cohens_d(group1, group2), "cohens_d")


def run_mann_whitney(group1, group2, alpha=0.05, alternative="two-sided"):
    """Mann-Whitney U (non-parametric)."""
    stat, p = stats.mannwhitneyu(group1, group2, alternative=alternative)
    r = compute_rank_biserial(stat, len(group1), len(group2))
    return _make_result("mann_whitney_u", stat, p, alpha, r, "rank_biserial_r")


def run_paired_ttest(data1, data2, alpha=0.05, alternative="two-sided"):
    """Paired t-test for dependent samples."""
    stat, p = stats.ttest_rel(data1, data2, alternative=alternative)
    d = compute_cohens_d_paired(data1 - data2)
    return _make_result("paired_ttest", stat, p, alpha, d, "cohens_d")


def run_wilcoxon_signed_rank(data1, data2, alpha=0.05, alternative="two-sided"):
    """Wilcoxon signed-rank (non-parametric paired)."""
    diffs = data1 - data2
    non_zero = diffs[diffs != 0]
    if len(non_zero) == 0:
        return _make_result("wilcoxon_signed_rank", 0.0, 1.0, alpha, 0.0, "rank_biserial_r")
    stat, p = stats.wilcoxon(data1, data2, alternative=alternative)
    n = len(non_zero)
    r = 1 - (2 * stat) / (n * (n + 1) / 2) if n > 0 else 0.0
    return _make_result("wilcoxon_signed_rank", stat, p, alpha, float(r), "rank_biserial_r")


def run_one_sample_ttest(sample, population_mean, alpha=0.05, alternative="two-sided"):
    """One-sample t-test against a known population mean."""
    stat, p = stats.ttest_1samp(sample, population_mean, alternative=alternative)
    d = compute_cohens_d_one_sample(sample, population_mean)
    return _make_result("one_sample_ttest", stat, p, alpha, d, "cohens_d")


def run_one_sample_wilcoxon(sample, population_mean, alpha=0.05, alternative="two-sided"):
    """Wilcoxon signed-rank against a known population mean."""
    shifted = sample - population_mean
    non_zero = shifted[shifted != 0]
    if len(non_zero) == 0:
        return _make_result("one_sample_wilcoxon", 0.0, 1.0, alpha, 0.0, "rank_biserial_r")
    stat, p = stats.wilcoxon(shifted, alternative=alternative)
    n = len(non_zero)
    r = 1 - (2 * stat) / (n * (n + 1) / 2) if n > 0 else 0.0
    return _make_result("one_sample_wilcoxon", stat, p, alpha, float(r), "rank_biserial_r")


# -- Interpretation --

def _format_p(p: float) -> str:
    return "p<0.001" if p < 0.001 else f"p={p:.4f}"


def _stat_label_for(test_used: str) -> str:
    """Return the conventional statistic label for a given test."""
    if "mann_whitney" in test_used:
        return "U"
    if "kruskal" in test_used:
        return "H"
    if "friedman" in test_used:
        return "χ²"
    if "anova" in test_used:
        return "F"
    if "wilcoxon" in test_used:
        return "W"
    return "t"


def generate_interpretation(test_used, result, effect_size, group_stats,
                            column_name, alpha):
    """Build a plain-English interpretation string."""
    p_str = _format_p(result["p_value"])
    es_val = effect_size["value"]
    es_mag = effect_size["magnitude"]
    es_metric = effect_size["metric"].replace("_", " ")
    test_label = _TEST_DISPLAY_NAMES.get(test_used, test_used)
    stat_val = result["statistic"]
    stat_label = _stat_label_for(test_used)

    verdict = (
        "There IS a statistically significant difference"
        if result["significant"]
        else "There is NO statistically significant difference"
    )

    # Filter internal keys (prefixed with _) from group listing
    display_groups = [k for k in group_stats.keys() if not k.startswith("_")]

    if len(display_groups) == 2:
        g1, g2 = display_groups[0], display_groups[1]
        m1 = group_stats[g1].get("mean") or group_stats[g1].get("median", "?")
        m2 = group_stats[g2].get("mean") or group_stats[g2].get("median", "?")
        context = (
            f" in {column_name} between groups {g1} and {g2} "
            f"({test_label}, {stat_label}={stat_val}, {p_str}, "
            f"{es_metric}={es_val} [{es_mag} effect]). "
            f"Group {g1} (M={m1}) vs Group {g2} (M={m2})."
        )
    elif len(display_groups) == 1:
        g = display_groups[0]
        mean = group_stats[g].get("mean", "?")
        ref = group_stats.get("_reference", {}).get("value", "?")
        context = (
            f" in {column_name} compared to the reference value of {ref} "
            f"({test_label}, {stat_label}={stat_val}, {p_str}, "
            f"{es_metric}={es_val} [{es_mag} effect]). "
            f"Sample mean={mean}."
        )
    elif len(display_groups) > 2:
        # Multi-group (ANOVA / Kruskal-Wallis / Friedman)
        group_summaries = []
        for g in display_groups:
            gs = group_stats[g]
            m = gs.get("mean") or gs.get("median", "?")
            group_summaries.append(f"{g} (M={m})")
        context = (
            f" in {column_name} across {len(display_groups)} groups "
            f"({test_label}, {stat_label}={stat_val}, {p_str}, "
            f"{es_metric}={es_val} [{es_mag} effect]). "
            f"Groups: {', '.join(group_summaries)}."
        )
    else:
        context = (
            f" in {column_name} "
            f"({test_label}, {stat_label}={stat_val}, {p_str}, "
            f"{es_metric}={es_val} [{es_mag} effect])."
        )

    return verdict + context


# -- Suppressed response helper --

def _suppressed_response(sdc, alpha, alternative, group_stats):
    """Standard response when SDC check forces suppression."""
    return {
        "test_used": "suppressed",
        "reason": sdc["message"],
        "assumptions": {},
        "result": {
            "statistic": None,
            "p_value": None,
            "significant": None,
            "alpha": alpha,
            "alternative": alternative,
        },
        "effect_size": {"metric": None, "value": None, "magnitude": None},
        "group_stats": group_stats,
        "interpretation": sdc["message"],
        "warnings": [sdc["message"]],
    }


def _build_response(test_result, reason, assumptions, group_stats,
                    column_name, alpha, alternative, warnings):
    """Assemble the final response dict from test results."""
    interpretation = generate_interpretation(
        test_used=test_result["test_used"],
        result={
            "statistic": test_result["statistic"],
            "p_value": test_result["p_value"],
            "significant": test_result["significant"],
        },
        effect_size=test_result["effect_size"],
        group_stats=group_stats,
        column_name=column_name,
        alpha=alpha,
    )
    return {
        "test_used": test_result["test_used"],
        "reason": reason,
        "assumptions": assumptions,
        "result": {
            "statistic": test_result["statistic"],
            "p_value": test_result["p_value"],
            "significant": test_result["significant"],
            "alpha": alpha,
            "alternative": alternative,
        },
        "effect_size": test_result["effect_size"],
        "group_stats": group_stats,
        "interpretation": interpretation,
        "warnings": warnings,
    }


# -- Orchestrators --

def _validate_numeric_col(df, col):
    if col not in df.columns:
        raise ValueError(f"Column '{col}' not found in dataset.")
    if not np.issubdtype(df[col].dtype, np.number):
        raise ValueError(f"Column '{col}' is not numeric (dtype: {df[col].dtype}).")


def run_two_group_test(df, numeric_col, group_col, alpha=0.05, alternative="two-sided"):
    """Independent 2-group comparison with automatic test selection.

    Shapiro-Wilk/KS → Levene's → Student's t / Welch's t / Mann-Whitney U.
    """
    _validate_numeric_col(df, numeric_col)
    if group_col not in df.columns:
        raise ValueError(f"Column '{group_col}' not found in dataset.")

    clean = df[[numeric_col, group_col]].dropna()
    groups = clean[group_col].unique()
    if len(groups) != 2:
        raise ValueError(
            f"Column '{group_col}' must have exactly 2 unique values for a "
            f"t-test, found {len(groups)}: {list(groups)}."
        )

    group1 = clean[clean[group_col] == groups[0]][numeric_col].values
    group2 = clean[clean[group_col] == groups[1]][numeric_col].values

    warnings: List[str] = []
    min_n = min(len(group1), len(group2))
    sdc = check_sdc(min_n)

    if sdc["status"] == "suppress":
        return _suppressed_response(sdc, alpha, alternative, {
            str(groups[0]): {"n": len(group1)},
            str(groups[1]): {"n": len(group2)},
        })
    if sdc["status"] == "warn":
        warnings.append(sdc["message"])

    norm1 = check_normality(group1, alpha)
    norm2 = check_normality(group2, alpha)
    both_normal = norm1["normal"] and norm2["normal"]

    assumptions: Dict[str, Any] = {
        "normality": {
            "test": norm1["test"],
            str(groups[0]): {
                "statistic": norm1["statistic"],
                "p_value": norm1["p_value"],
                "normal": norm1["normal"],
            },
            str(groups[1]): {
                "statistic": norm2["statistic"],
                "p_value": norm2["p_value"],
                "normal": norm2["normal"],
            },
        },
    }
    note = norm1.get("note") or norm2.get("note")
    if note:
        assumptions["normality"]["note"] = note

    if not both_normal:
        test_result = run_mann_whitney(group1, group2, alpha, alternative)
        reason = (
            f"Data is NOT normally distributed "
            f"({norm1['test']}: group '{groups[0]}' p={norm1['p_value']}, "
            f"group '{groups[1]}' p={norm2['p_value']}). "
            f"Using non-parametric Mann-Whitney U test instead of t-test."
        )
    else:
        variance_check = check_equal_variance(group1, group2, alpha)
        assumptions["equal_variance"] = variance_check

        if variance_check["equal_variance"]:
            test_result = run_students_ttest(group1, group2, alpha, alternative)
            reason = (
                f"Data is normal ({norm1['test']}: "
                f"group '{groups[0]}' p={norm1['p_value']}, "
                f"group '{groups[1]}' p={norm2['p_value']}) "
                f"and has equal variances (Levene's p={variance_check['p_value']}). "
                f"Using Student's t-test."
            )
        else:
            test_result = run_welchs_ttest(group1, group2, alpha, alternative)
            reason = (
                f"Data is normal ({norm1['test']}: "
                f"group '{groups[0]}' p={norm1['p_value']}, "
                f"group '{groups[1]}' p={norm2['p_value']}) "
                f"but has UNEQUAL variances (Levene's p={variance_check['p_value']}). "
                f"Using Welch's t-test."
            )

    group_stats = {
        str(groups[0]): {
            "n": int(len(group1)),
            "mean": round(float(np.mean(group1)), 4),
            "std": round(float(np.std(group1, ddof=1)), 4),
            "median": round(float(np.median(group1)), 4),
        },
        str(groups[1]): {
            "n": int(len(group2)),
            "mean": round(float(np.mean(group2)), 4),
            "std": round(float(np.std(group2, ddof=1)), 4),
            "median": round(float(np.median(group2)), 4),
        },
    }

    return _build_response(test_result, reason, assumptions, group_stats,
                           numeric_col, alpha, alternative, warnings)


def run_paired_test(df, col1, col2, alpha=0.05, alternative="two-sided"):
    """Paired comparison with automatic test selection.

    Shapiro-Wilk/KS on differences → paired t-test OR Wilcoxon signed-rank.
    """
    for col in (col1, col2):
        _validate_numeric_col(df, col)

    clean = df[[col1, col2]].dropna()
    data1, data2 = clean[col1].values, clean[col2].values
    diffs = data1 - data2
    n_pairs = len(diffs)

    warnings: List[str] = []
    sdc = check_sdc(n_pairs)

    if sdc["status"] == "suppress":
        return _suppressed_response(sdc, alpha, alternative, {"pairs": {"n": n_pairs}})
    if sdc["status"] == "warn":
        warnings.append(sdc["message"])

    norm = check_normality(diffs, alpha)
    assumptions: Dict[str, Any] = {
        "normality": {
            "test": norm["test"],
            "differences": {
                "statistic": norm["statistic"],
                "p_value": norm["p_value"],
                "normal": norm["normal"],
            },
        },
    }
    if norm.get("note"):
        assumptions["normality"]["note"] = norm["note"]

    if norm["normal"]:
        test_result = run_paired_ttest(data1, data2, alpha, alternative)
        reason = (
            f"Differences are normally distributed "
            f"({norm['test']} p={norm['p_value']}). "
            f"Using paired t-test."
        )
    else:
        test_result = run_wilcoxon_signed_rank(data1, data2, alpha, alternative)
        reason = (
            f"Differences are NOT normally distributed "
            f"({norm['test']} p={norm['p_value']}). "
            f"Using non-parametric Wilcoxon signed-rank test."
        )

    group_stats = {
        col1: {
            "n": int(len(data1)),
            "mean": round(float(np.mean(data1)), 4),
            "std": round(float(np.std(data1, ddof=1)), 4),
        },
        col2: {
            "n": int(len(data2)),
            "mean": round(float(np.mean(data2)), 4),
            "std": round(float(np.std(data2, ddof=1)), 4),
        },
    }

    return _build_response(test_result, reason, assumptions, group_stats,
                           f"{col1} vs {col2}", alpha, alternative, warnings)


def run_one_sample_test(df, numeric_col, population_mean, alpha=0.05,
                        alternative="two-sided"):
    """One-sample test with automatic test selection.

    Shapiro-Wilk/KS → one-sample t-test OR Wilcoxon signed-rank.
    """
    _validate_numeric_col(df, numeric_col)

    sample = df[numeric_col].dropna().values
    n = len(sample)

    warnings: List[str] = []
    sdc = check_sdc(n)

    if sdc["status"] == "suppress":
        return _suppressed_response(sdc, alpha, alternative, {"sample": {"n": n}})
    if sdc["status"] == "warn":
        warnings.append(sdc["message"])

    norm = check_normality(sample, alpha)
    assumptions: Dict[str, Any] = {
        "normality": {
            "test": norm["test"],
            "sample": {
                "statistic": norm["statistic"],
                "p_value": norm["p_value"],
                "normal": norm["normal"],
            },
        },
    }
    if norm.get("note"):
        assumptions["normality"]["note"] = norm["note"]

    if norm["normal"]:
        test_result = run_one_sample_ttest(sample, population_mean, alpha, alternative)
        reason = (
            f"Data is normally distributed ({norm['test']} p={norm['p_value']}). "
            f"Using one-sample t-test."
        )
    else:
        test_result = run_one_sample_wilcoxon(sample, population_mean, alpha, alternative)
        reason = (
            f"Data is NOT normally distributed ({norm['test']} p={norm['p_value']}). "
            f"Using non-parametric one-sample Wilcoxon signed-rank test."
        )

    group_stats = {
        "sample": {
            "n": int(n),
            "mean": round(float(np.mean(sample)), 4),
            "std": round(float(np.std(sample, ddof=1)), 4),
            "median": round(float(np.median(sample)), 4),
        },
        "_reference": {"value": population_mean},
    }

    return _build_response(test_result, reason, assumptions, group_stats,
                           numeric_col, alpha, alternative, warnings)


# ============================================================
# ANOVA family + non-parametric alternatives
# ============================================================

def check_equal_variance_multi(*groups, alpha: float = 0.05) -> Dict[str, Any]:
    """Levene's test for equality of variances across k groups."""
    stat, p = stats.levene(*groups)
    return {
        "test": "levenes",
        "statistic": round(float(stat), 6),
        "p_value": round(float(p), 6),
        "equal_variance": bool(p > alpha),
    }


# -- ANOVA effect sizes --

def compute_eta_squared(ss_between: float, ss_total: float) -> float:
    """Eta-squared: proportion of total variance explained by groups."""
    if ss_total == 0:
        return 0.0
    return float(ss_between / ss_total)


def compute_epsilon_squared(H: float, k: int, N: int) -> float:
    """Epsilon-squared effect size for Kruskal-Wallis: (H - k + 1) / (N - k)."""
    denom = N - k
    if denom <= 0:
        return 0.0
    return max(0.0, float((H - k + 1) / denom))


def compute_kendalls_w(chi2: float, n: int, k: int) -> float:
    """Kendall's W concordance coefficient for Friedman: χ² / (n × (k - 1))."""
    denom = n * (k - 1)
    if denom == 0:
        return 0.0
    return float(chi2 / denom)


# -- Post-hoc helpers --

def _bonferroni_dunns_posthoc(
    group_names: List[str],
    group_data: List[np.ndarray],
    alpha: float = 0.05,
) -> Dict[str, Any]:
    """Dunn's post-hoc via pairwise Mann-Whitney U with Bonferroni correction."""
    pairs = list(itertools.combinations(range(len(group_names)), 2))
    n_comparisons = len(pairs)
    adjusted_alpha = alpha / n_comparisons if n_comparisons > 0 else alpha

    comparisons = []
    for i, j in pairs:
        stat, p = stats.mannwhitneyu(
            group_data[i], group_data[j], alternative="two-sided"
        )
        p_adj = min(p * n_comparisons, 1.0)  # Bonferroni
        r = compute_rank_biserial(stat, len(group_data[i]), len(group_data[j]))
        comparisons.append({
            "group_1": group_names[i],
            "group_2": group_names[j],
            "statistic": round(float(stat), 6),
            "p_value": round(float(p), 6),
            "p_value_adjusted": round(float(p_adj), 6),
            "significant": bool(p_adj <= alpha),
            "effect_size": round(float(r), 4),
        })

    return {
        "method": "dunns_bonferroni",
        "n_comparisons": n_comparisons,
        "adjusted_alpha": round(adjusted_alpha, 6),
        "comparisons": comparisons,
    }


def _tukey_hsd_posthoc(
    group_names: List[str],
    group_data: List[np.ndarray],
) -> Dict[str, Any]:
    """Tukey HSD post-hoc test for one-way ANOVA."""
    result = stats.tukey_hsd(*group_data)
    comparisons = []
    for i, j in itertools.combinations(range(len(group_names)), 2):
        p_val = result.pvalue[i][j]
        # Cohen's d between pairs
        d = compute_cohens_d(group_data[i], group_data[j])
        comparisons.append({
            "group_1": group_names[i],
            "group_2": group_names[j],
            "p_value": round(float(p_val), 6),
            "significant": bool(p_val <= 0.05),
            "cohens_d": round(float(d), 4),
        })

    return {
        "method": "tukey_hsd",
        "n_comparisons": len(comparisons),
        "comparisons": comparisons,
    }


def _games_howell_posthoc(
    group_names: List[str],
    group_data: List[np.ndarray],
) -> Dict[str, Any]:
    """Games-Howell post-hoc for Welch's ANOVA (unequal variances).

    Approximation via pairwise Welch's t-tests with Bonferroni correction.
    """
    pairs = list(itertools.combinations(range(len(group_names)), 2))
    n_comparisons = len(pairs)

    comparisons = []
    for i, j in pairs:
        stat, p = stats.ttest_ind(
            group_data[i], group_data[j], equal_var=False
        )
        p_adj = min(p * n_comparisons, 1.0)
        d = compute_cohens_d(group_data[i], group_data[j])
        comparisons.append({
            "group_1": group_names[i],
            "group_2": group_names[j],
            "statistic": round(float(stat), 6),
            "p_value": round(float(p), 6),
            "p_value_adjusted": round(float(p_adj), 6),
            "significant": bool(p_adj <= 0.05),
            "cohens_d": round(float(d), 4),
        })

    return {
        "method": "games_howell_approx",
        "n_comparisons": n_comparisons,
        "comparisons": comparisons,
    }


def _pairwise_wilcoxon_posthoc(
    condition_names: List[str],
    condition_data: List[np.ndarray],
    alpha: float = 0.05,
) -> Dict[str, Any]:
    """Pairwise Wilcoxon signed-rank with Bonferroni correction (Friedman post-hoc)."""
    pairs = list(itertools.combinations(range(len(condition_names)), 2))
    n_comparisons = len(pairs)
    adjusted_alpha = alpha / n_comparisons if n_comparisons > 0 else alpha

    comparisons = []
    for i, j in pairs:
        diffs = condition_data[i] - condition_data[j]
        non_zero = diffs[diffs != 0]
        if len(non_zero) == 0:
            comparisons.append({
                "condition_1": condition_names[i],
                "condition_2": condition_names[j],
                "statistic": 0.0,
                "p_value": 1.0,
                "p_value_adjusted": 1.0,
                "significant": False,
            })
            continue
        stat, p = stats.wilcoxon(condition_data[i], condition_data[j])
        p_adj = min(p * n_comparisons, 1.0)
        comparisons.append({
            "condition_1": condition_names[i],
            "condition_2": condition_names[j],
            "statistic": round(float(stat), 6),
            "p_value": round(float(p), 6),
            "p_value_adjusted": round(float(p_adj), 6),
            "significant": bool(p_adj <= alpha),
        })

    return {
        "method": "pairwise_wilcoxon_bonferroni",
        "n_comparisons": n_comparisons,
        "adjusted_alpha": round(adjusted_alpha, 6),
        "comparisons": comparisons,
    }


# -- ANOVA orchestrator response builder --

def _build_anova_response(
    test_result: Dict,
    reason: str,
    assumptions: Dict,
    group_stats: Dict,
    column_name: str,
    alpha: float,
    warnings: List[str],
    post_hoc: Optional[Dict] = None,
) -> Dict[str, Any]:
    """Build final response dict for ANOVA-family tests."""
    interpretation = generate_interpretation(
        test_used=test_result["test_used"],
        result={
            "statistic": test_result["statistic"],
            "p_value": test_result["p_value"],
            "significant": test_result["significant"],
        },
        effect_size=test_result["effect_size"],
        group_stats=group_stats,
        column_name=column_name,
        alpha=alpha,
    )

    if post_hoc and test_result["significant"]:
        sig_pairs = [
            c for c in post_hoc.get("comparisons", [])
            if c.get("significant")
        ]
        if sig_pairs:
            pair_strs = []
            for c in sig_pairs:
                g1 = c.get("group_1") or c.get("condition_1")
                g2 = c.get("group_2") or c.get("condition_2")
                pair_strs.append(f"{g1} vs {g2}")
            interpretation += (
                f" Post-hoc analysis found significant pairwise differences: "
                f"{'; '.join(pair_strs)}."
            )

    resp = {
        "test_used": test_result["test_used"],
        "reason": reason,
        "assumptions": assumptions,
        "result": {
            "statistic": test_result["statistic"],
            "p_value": test_result["p_value"],
            "significant": test_result["significant"],
            "alpha": alpha,
            "alternative": "two-sided",
        },
        "effect_size": test_result["effect_size"],
        "group_stats": group_stats,
        "interpretation": interpretation,
        "warnings": warnings,
    }
    if post_hoc is not None:
        resp["post_hoc"] = post_hoc
    return resp


# -- One-Way ANOVA / Kruskal-Wallis --

def run_one_way_anova(
    df: pd.DataFrame,
    numeric_col: str,
    group_col: str,
    alpha: float = 0.05,
) -> Dict[str, Any]:
    """One-way ANOVA with automatic test selection.

    check_normality() per group (SW ≤5000, KS >5000) → Levene's →
    ANOVA / Welch's ANOVA / Kruskal-Wallis.
    Post-hoc if significant: Tukey / Games-Howell / Dunn's.
    """
    _validate_numeric_col(df, numeric_col)
    if group_col not in df.columns:
        raise ValueError(f"Column '{group_col}' not found in dataset.")

    clean = df[[numeric_col, group_col]].dropna()
    unique_groups = clean[group_col].unique()
    if len(unique_groups) < 3:
        raise ValueError(
            f"ANOVA requires at least 3 groups, found {len(unique_groups)}: "
            f"{list(unique_groups)}. Use t-test for 2-group comparisons."
        )

    group_names = [str(g) for g in unique_groups]
    group_data = [
        clean[clean[group_col] == g][numeric_col].values for g in unique_groups
    ]

    warnings: List[str] = []
    min_n = min(len(gd) for gd in group_data)
    sdc = check_sdc(min_n)

    if sdc["status"] == "suppress":
        gs = {name: {"n": len(gd)} for name, gd in zip(group_names, group_data)}
        return _suppressed_response(sdc, alpha, "two-sided", gs)
    if sdc["status"] == "warn":
        warnings.append(sdc["message"])

    # Normality per group (SW first, KS fallback for n > 5000)
    normality_results = {}
    all_normal = True
    for name, gd in zip(group_names, group_data):
        norm = check_normality(gd, alpha)
        normality_results[name] = {
            "statistic": norm["statistic"],
            "p_value": norm["p_value"],
            "normal": norm["normal"],
        }
        if not norm["normal"]:
            all_normal = False

    # Pick the test name from the first group (all use the same test)
    first_norm = check_normality(group_data[0], alpha)
    assumptions: Dict[str, Any] = {
        "normality": {
            "test": first_norm["test"],
            **normality_results,
        },
    }
    if first_norm.get("note"):
        assumptions["normality"]["note"] = first_norm["note"]

    post_hoc = None

    if not all_normal:
        # Non-parametric fallback: Kruskal-Wallis
        H_stat, p_val = stats.kruskal(*group_data)
        N = sum(len(gd) for gd in group_data)
        k = len(group_data)
        eps_sq = compute_epsilon_squared(H_stat, k, N)

        test_result = {
            "test_used": "kruskal_wallis",
            "statistic": round(float(H_stat), 6),
            "p_value": round(float(p_val), 6),
            "significant": bool(p_val <= alpha),
            "effect_size": {
                "metric": "epsilon_squared",
                "value": round(eps_sq, 4),
                "magnitude": classify_effect_size(eps_sq, "epsilon_squared"),
            },
        }
        reason = (
            f"Not all groups are normally distributed "
            f"({first_norm['test']}). "
            f"Using non-parametric Kruskal-Wallis H test."
        )
        if test_result["significant"]:
            post_hoc = _bonferroni_dunns_posthoc(group_names, group_data, alpha)
    else:
        # Check equal variances
        variance_check = check_equal_variance_multi(*group_data, alpha=alpha)
        assumptions["equal_variance"] = variance_check

        if variance_check["equal_variance"]:
            # Standard one-way ANOVA
            F_stat, p_val = stats.f_oneway(*group_data)

            # Compute eta-squared
            grand_mean = np.mean(np.concatenate(group_data))
            ss_between = sum(
                len(gd) * (np.mean(gd) - grand_mean) ** 2 for gd in group_data
            )
            ss_total = sum(
                np.sum((gd - grand_mean) ** 2) for gd in group_data
            )
            eta_sq = compute_eta_squared(ss_between, ss_total)

            test_result = {
                "test_used": "one_way_anova",
                "statistic": round(float(F_stat), 6),
                "p_value": round(float(p_val), 6),
                "significant": bool(p_val <= alpha),
                "effect_size": {
                    "metric": "eta_squared",
                    "value": round(eta_sq, 4),
                    "magnitude": classify_effect_size(eta_sq, "eta_squared"),
                },
            }
            reason = (
                f"All groups are normally distributed ({first_norm['test']}) "
                f"and have equal variances "
                f"(Levene's p={variance_check['p_value']}). "
                f"Using one-way ANOVA (F-test)."
            )
            if test_result["significant"]:
                post_hoc = _tukey_hsd_posthoc(group_names, group_data)
        else:
            # Welch's ANOVA (unequal variances)
            # scipy doesn't have Welch's ANOVA directly; use Alexander-Govern
            # or manual Welch F. We use the Welch one-way test from scipy.
            # stats.f_oneway doesn't handle unequal variance, so we use
            # a manual implementation.
            weights = np.array([len(gd) / np.var(gd, ddof=1) for gd in group_data])
            sum_weights = np.sum(weights)
            weighted_mean = np.sum(
                weights * np.array([np.mean(gd) for gd in group_data])
            ) / sum_weights
            k = len(group_data)

            # Welch's F numerator
            msb = np.sum(
                weights * (np.array([np.mean(gd) for gd in group_data]) - weighted_mean) ** 2
            ) / (k - 1)

            # Welch's F denominator (Lambda)
            lambdas = np.array([
                (1 - weights[i] / sum_weights) ** 2 / (len(group_data[i]) - 1)
                for i in range(k)
            ])
            denominator = 1 + (2 * (k - 2) / (k ** 2 - 1)) * np.sum(lambdas)

            F_welch = float(msb / denominator) if denominator != 0 else 0.0

            # Degrees of freedom for p-value
            df1 = k - 1
            df2_inv = (3 / (k ** 2 - 1)) * np.sum(lambdas)
            df2 = 1 / df2_inv if df2_inv != 0 else float('inf')
            p_val = float(1 - stats.f.cdf(abs(F_welch), df1, df2))

            # Eta-squared (same formula, using Welch F)
            grand_mean = np.mean(np.concatenate(group_data))
            ss_between = sum(
                len(gd) * (np.mean(gd) - grand_mean) ** 2 for gd in group_data
            )
            ss_total = sum(
                np.sum((gd - grand_mean) ** 2) for gd in group_data
            )
            eta_sq = compute_eta_squared(ss_between, ss_total)

            test_result = {
                "test_used": "welchs_anova",
                "statistic": round(float(F_welch), 6),
                "p_value": round(float(p_val), 6),
                "significant": bool(p_val <= alpha),
                "effect_size": {
                    "metric": "eta_squared",
                    "value": round(eta_sq, 4),
                    "magnitude": classify_effect_size(eta_sq, "eta_squared"),
                },
            }
            reason = (
                f"All groups are normally distributed ({first_norm['test']}) "
                f"but have UNEQUAL variances "
                f"(Levene's p={variance_check['p_value']}). "
                f"Using Welch's ANOVA."
            )
            if test_result["significant"]:
                post_hoc = _games_howell_posthoc(group_names, group_data)

    group_stats = {}
    for name, gd in zip(group_names, group_data):
        group_stats[name] = {
            "n": int(len(gd)),
            "mean": round(float(np.mean(gd)), 4),
            "std": round(float(np.std(gd, ddof=1)), 4),
            "median": round(float(np.median(gd)), 4),
        }

    return _build_anova_response(
        test_result, reason, assumptions, group_stats,
        numeric_col, alpha, warnings, post_hoc,
    )


# -- Two-Way ANOVA --

def run_two_way_anova(
    df: pd.DataFrame,
    numeric_col: str,
    factor_col_1: str,
    factor_col_2: str,
    alpha: float = 0.05,
) -> Dict[str, Any]:
    """Two-way ANOVA using statsmodels OLS with Type II sums of squares.

    Checks normality of residuals via check_normality() (SW / KS fallback).
    Reports main effects and interaction.
    """
    _validate_numeric_col(df, numeric_col)
    for col in (factor_col_1, factor_col_2):
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found in dataset.")

    clean = df[[numeric_col, factor_col_1, factor_col_2]].dropna()
    n = len(clean)

    warnings: List[str] = []
    sdc = check_sdc(n)
    if sdc["status"] == "suppress":
        return _suppressed_response(sdc, alpha, "two-sided", {"total_n": n})
    if sdc["status"] == "warn":
        warnings.append(sdc["message"])

    # Sanitize column names for formula (replace spaces/special chars)
    _safe = {}
    for col in [numeric_col, factor_col_1, factor_col_2]:
        safe_name = col.replace(" ", "_").replace("-", "_").replace(".", "_")
        _safe[col] = safe_name
    safe_df = clean.rename(columns=_safe)

    # Ensure factors are categorical
    safe_df[_safe[factor_col_1]] = safe_df[_safe[factor_col_1]].astype(str)
    safe_df[_safe[factor_col_2]] = safe_df[_safe[factor_col_2]].astype(str)

    formula = (
        f"Q('{_safe[numeric_col]}') ~ "
        f"C(Q('{_safe[factor_col_1]}')) * C(Q('{_safe[factor_col_2]}'))"
    )

    try:
        model = sm_ols(formula, data=safe_df).fit()
        anova_table = anova_lm(model, typ=2)
    except Exception as exc:
        raise ValueError(f"Two-way ANOVA model failed: {exc}")

    # Extract results
    residuals = model.resid.values
    norm_residuals = check_normality(residuals, alpha)

    assumptions: Dict[str, Any] = {
        "normality": {
            "test": norm_residuals["test"],
            "residuals": {
                "statistic": norm_residuals["statistic"],
                "p_value": norm_residuals["p_value"],
                "normal": norm_residuals["normal"],
            },
        },
    }
    if norm_residuals.get("note"):
        assumptions["normality"]["note"] = norm_residuals["note"]

    if not norm_residuals["normal"]:
        warnings.append(
            "Residuals are NOT normally distributed "
            f"({norm_residuals['test']} p={norm_residuals['p_value']}). "
            "ANOVA results should be interpreted with caution. "
            "Consider a non-parametric alternative for individual factors."
        )

    # Parse ANOVA table rows
    ss_total = anova_table["sum_sq"].sum()
    effects = {}
    for row_name in anova_table.index:
        if row_name == "Residual":
            continue
        row = anova_table.loc[row_name]
        eta_sq = compute_eta_squared(row["sum_sq"], ss_total)
        effects[row_name] = {
            "sum_sq": round(float(row["sum_sq"]), 4),
            "df": int(row["df"]),
            "F": round(float(row["F"]), 6),
            "p_value": round(float(row["PR(>F)"]), 6),
            "significant": bool(row["PR(>F)"] <= alpha),
            "eta_squared": round(eta_sq, 4),
            "eta_squared_magnitude": classify_effect_size(eta_sq, "eta_squared"),
        }

    # Overall F from the model
    overall_F = round(float(model.fvalue), 6) if not np.isnan(model.fvalue) else 0.0
    overall_p = round(float(model.f_pvalue), 6) if not np.isnan(model.f_pvalue) else 1.0

    # Total eta-squared for the model
    ss_resid = anova_table.loc["Residual", "sum_sq"] if "Residual" in anova_table.index else 0
    model_eta_sq = compute_eta_squared(ss_total - ss_resid, ss_total)

    test_result = {
        "test_used": "two_way_anova",
        "statistic": overall_F,
        "p_value": overall_p,
        "significant": bool(overall_p <= alpha),
        "effect_size": {
            "metric": "eta_squared",
            "value": round(model_eta_sq, 4),
            "magnitude": classify_effect_size(model_eta_sq, "eta_squared"),
        },
    }
    reason = (
        f"Two-way ANOVA (Type II SS) with factors '{factor_col_1}' "
        f"and '{factor_col_2}' on '{numeric_col}'."
    )

    # Build group stats per factor combination
    group_stats: Dict[str, Any] = {}
    for (f1, f2), sub_df in clean.groupby([factor_col_1, factor_col_2]):
        key = f"{f1}:{f2}"
        vals = sub_df[numeric_col].values
        group_stats[key] = {
            "n": int(len(vals)),
            "mean": round(float(np.mean(vals)), 4),
            "std": round(float(np.std(vals, ddof=1)), 4) if len(vals) > 1 else 0.0,
        }
    group_stats["_effects"] = effects

    return _build_anova_response(
        test_result, reason, assumptions, group_stats,
        numeric_col, alpha, warnings, None,
    )


# -- Repeated-Measures ANOVA / Friedman --

def run_repeated_measures_anova(
    df: pd.DataFrame,
    condition_cols: List[str],
    alpha: float = 0.05,
) -> Dict[str, Any]:
    """Repeated-measures comparison with automatic test selection.

    check_normality() on pairwise differences → RM-ANOVA or Friedman.
    """
    for col in condition_cols:
        _validate_numeric_col(df, col)
    if len(condition_cols) < 3:
        raise ValueError(
            f"Repeated-measures ANOVA requires at least 3 conditions, "
            f"found {len(condition_cols)}. Use paired test for 2 conditions."
        )

    clean = df[condition_cols].dropna()
    n_subjects = len(clean)
    k = len(condition_cols)

    warnings: List[str] = []
    sdc = check_sdc(n_subjects)
    if sdc["status"] == "suppress":
        return _suppressed_response(sdc, alpha, "two-sided", {"subjects": n_subjects})
    if sdc["status"] == "warn":
        warnings.append(sdc["message"])

    condition_data = [clean[col].values for col in condition_cols]

    # Check normality on all pairwise differences
    all_normal = True
    normality_results = {}
    for i, j in itertools.combinations(range(k), 2):
        diffs = condition_data[i] - condition_data[j]
        norm = check_normality(diffs, alpha)
        pair_key = f"{condition_cols[i]}_vs_{condition_cols[j]}"
        normality_results[pair_key] = {
            "statistic": norm["statistic"],
            "p_value": norm["p_value"],
            "normal": norm["normal"],
        }
        if not norm["normal"]:
            all_normal = False

    first_diff = condition_data[0] - condition_data[1]
    first_norm = check_normality(first_diff, alpha)
    assumptions: Dict[str, Any] = {
        "normality": {
            "test": first_norm["test"],
            "differences": normality_results,
        },
    }
    if first_norm.get("note"):
        assumptions["normality"]["note"] = first_norm["note"]

    post_hoc = None

    if not all_normal:
        # Friedman test (non-parametric repeated-measures)
        chi2_stat, p_val = stats.friedmanchisquare(*condition_data)
        w = compute_kendalls_w(chi2_stat, n_subjects, k)

        test_result = {
            "test_used": "friedman",
            "statistic": round(float(chi2_stat), 6),
            "p_value": round(float(p_val), 6),
            "significant": bool(p_val <= alpha),
            "effect_size": {
                "metric": "kendalls_w",
                "value": round(w, 4),
                "magnitude": classify_effect_size(w, "kendalls_w"),
            },
        }
        reason = (
            f"Not all pairwise differences are normally distributed "
            f"({first_norm['test']}). "
            f"Using non-parametric Friedman test."
        )
        if test_result["significant"]:
            post_hoc = _pairwise_wilcoxon_posthoc(
                condition_cols, condition_data, alpha
            )
    else:
        # Repeated-measures ANOVA (manual F-test)
        # Using standard RM-ANOVA computation
        data_matrix = np.column_stack(condition_data)
        grand_mean = np.mean(data_matrix)

        # SS total
        ss_total = np.sum((data_matrix - grand_mean) ** 2)

        # SS between conditions (treatment)
        condition_means = np.mean(data_matrix, axis=0)
        ss_between = n_subjects * np.sum((condition_means - grand_mean) ** 2)

        # SS between subjects
        subject_means = np.mean(data_matrix, axis=1)
        ss_subjects = k * np.sum((subject_means - grand_mean) ** 2)

        # SS error (residual)
        ss_error = ss_total - ss_between - ss_subjects

        # Degrees of freedom
        df_between = k - 1
        df_error = (n_subjects - 1) * (k - 1)

        # Mean squares
        ms_between = ss_between / df_between if df_between > 0 else 0
        ms_error = ss_error / df_error if df_error > 0 else 0

        # F statistic
        F_stat = ms_between / ms_error if ms_error > 0 else 0.0
        p_val = float(1 - stats.f.cdf(F_stat, df_between, df_error))

        # Eta-squared
        eta_sq = compute_eta_squared(ss_between, ss_total)

        # Sphericity check (Mauchly's approximation)
        # Simplified: compare variances of all pairwise differences
        diff_variances = []
        for i, j in itertools.combinations(range(k), 2):
            diff_variances.append(float(np.var(condition_data[i] - condition_data[j], ddof=1)))
        if len(diff_variances) > 1 and all(v > 0 for v in diff_variances):
            max_var = max(diff_variances)
            min_var = min(diff_variances)
            variance_ratio = max_var / min_var if min_var > 0 else float('inf')
            sphericity_ok = bool(variance_ratio < 2.0)  # Rough heuristic
        else:
            sphericity_ok = True
            variance_ratio = 1.0

        assumptions["sphericity"] = {
            "method": "variance_ratio_heuristic",
            "variance_ratio": round(float(variance_ratio), 4),
            "sphericity_assumed": sphericity_ok,
        }

        if not sphericity_ok:
            # Greenhouse-Geisser correction
            # Epsilon approximation
            diff_vars = np.array(diff_variances)
            mean_var = np.mean(diff_vars)
            var_of_vars = np.var(diff_vars, ddof=1) if len(diff_vars) > 1 else 0
            n_diffs = len(diff_vars)
            if var_of_vars > 0 and mean_var > 0:
                epsilon = min(1.0, (mean_var ** 2 * n_diffs) /
                              (n_diffs * (mean_var ** 2 + var_of_vars)))
            else:
                epsilon = 1.0

            # Adjust degrees of freedom
            df_between_adj = df_between * epsilon
            df_error_adj = df_error * epsilon
            p_val = float(1 - stats.f.cdf(F_stat, df_between_adj, df_error_adj))

            assumptions["sphericity"]["greenhouse_geisser_epsilon"] = round(epsilon, 4)
            warnings.append(
                f"Sphericity violated (variance ratio={variance_ratio:.2f}). "
                f"Applied Greenhouse-Geisser correction (ε={epsilon:.4f})."
            )

        test_result = {
            "test_used": "repeated_measures_anova",
            "statistic": round(float(F_stat), 6),
            "p_value": round(float(p_val), 6),
            "significant": bool(p_val <= alpha),
            "effect_size": {
                "metric": "eta_squared",
                "value": round(eta_sq, 4),
                "magnitude": classify_effect_size(eta_sq, "eta_squared"),
            },
        }
        reason = (
            f"All pairwise differences are normally distributed "
            f"({first_norm['test']}). "
            f"Using repeated-measures ANOVA."
        )
        if test_result["significant"]:
            # Pairwise paired t-tests with Bonferroni for RM-ANOVA post-hoc
            pairs = list(itertools.combinations(range(k), 2))
            n_comp = len(pairs)
            comparisons = []
            for i, j in pairs:
                stat, p = stats.ttest_rel(condition_data[i], condition_data[j])
                p_adj = min(p * n_comp, 1.0)
                d = compute_cohens_d_paired(condition_data[i] - condition_data[j])
                comparisons.append({
                    "condition_1": condition_cols[i],
                    "condition_2": condition_cols[j],
                    "statistic": round(float(stat), 6),
                    "p_value": round(float(p), 6),
                    "p_value_adjusted": round(float(p_adj), 6),
                    "significant": bool(p_adj <= alpha),
                    "cohens_d": round(float(d), 4),
                })
            post_hoc = {
                "method": "pairwise_paired_ttest_bonferroni",
                "n_comparisons": n_comp,
                "adjusted_alpha": round(alpha / n_comp, 6),
                "comparisons": comparisons,
            }

    # Group stats per condition
    group_stats = {}
    for col, gd in zip(condition_cols, condition_data):
        group_stats[col] = {
            "n": int(len(gd)),
            "mean": round(float(np.mean(gd)), 4),
            "std": round(float(np.std(gd, ddof=1)), 4),
            "median": round(float(np.median(gd)), 4),
        }
    group_stats["_design"] = {
        "n_subjects": n_subjects,
        "n_conditions": k,
    }

    return _build_anova_response(
        test_result, reason, assumptions, group_stats,
        " vs ".join(condition_cols), alpha, warnings, post_hoc,
    )


# -- Top-level multi-group orchestrator --

def run_multi_group_test(
    df: pd.DataFrame,
    numeric_col: str,
    group_col: str,
    alpha: float = 0.05,
) -> Dict[str, Any]:
    """Entrypoint for one-way multi-group comparison.

    Delegates to run_one_way_anova which auto-selects
    ANOVA / Welch's ANOVA / Kruskal-Wallis.
    """
    return run_one_way_anova(df, numeric_col, group_col, alpha)


# =====================================================================
# Chi-Square Tests (Week 7)
# =====================================================================

CRAMERS_V_THRESHOLDS = {"small": 0.1, "medium": 0.3, "large": 0.5}


def compute_cramers_v(chi2: float, n: int, k: int, r: int) -> float:
    """Cramér's V effect size for chi-square test of independence.

    V = sqrt(chi2 / (n * min(k-1, r-1)))
    """
    min_dim = min(k - 1, r - 1)
    if min_dim == 0 or n == 0:
        return 0.0
    return float(math.sqrt(chi2 / (n * min_dim)))


def _classify_cramers_v(v: float) -> str:
    """Classify Cramér's V magnitude."""
    abs_v = abs(v)
    if abs_v >= CRAMERS_V_THRESHOLDS["large"]:
        return "large"
    if abs_v >= CRAMERS_V_THRESHOLDS["medium"]:
        return "medium"
    if abs_v >= CRAMERS_V_THRESHOLDS["small"]:
        return "small"
    return "negligible"


def run_chi_square_independence(
    df: pd.DataFrame,
    col1: str,
    col2: str,
    alpha: float = 0.05,
) -> Dict[str, Any]:
    """Chi-square test of independence between two categorical columns.

    Automatically falls back to Fisher's exact test for 2×2 tables
    when any expected frequency is below 5.
    """
    if col1 not in df.columns:
        raise ValueError(f"Column '{col1}' not found in dataset.")
    if col2 not in df.columns:
        raise ValueError(f"Column '{col2}' not found in dataset.")

    clean = df[[col1, col2]].dropna()
    n = len(clean)

    # SDC check on overall sample size
    sdc = check_sdc(n)
    if sdc["status"] == "suppress":
        return {
            "test_used": "suppressed",
            "reason": sdc["message"],
            "assumptions": {},
            "result": {
                "statistic": None, "p_value": None,
                "significant": None, "alpha": alpha,
            },
            "effect_size": {"metric": None, "value": None, "magnitude": None},
            "contingency_table": {},
            "interpretation": sdc["message"],
            "warnings": [sdc["message"]],
        }

    warnings_list: List[str] = []
    if sdc["status"] == "warn":
        warnings_list.append(sdc["message"])

    # Build contingency table
    contingency = pd.crosstab(clean[col1], clean[col2])
    observed = contingency.values
    r, k = observed.shape

    # Check expected frequencies
    chi2_stat, p_value, dof, expected = stats.chi2_contingency(observed)
    min_expected = expected.min()
    use_fisher = False

    if r == 2 and k == 2 and min_expected < 5:
        # Fisher's exact test for 2×2 tables with small expected counts
        use_fisher = True
        odds_ratio, p_value = stats.fisher_exact(observed)
        test_used = "fisher_exact"
        stat_value = odds_ratio
        reason = (
            f"2×2 contingency table has expected frequency {min_expected:.1f} < 5. "
            f"Using Fisher's exact test instead of chi-square."
        )
    else:
        test_used = "chi_square_independence"
        stat_value = chi2_stat
        reason = (
            f"Chi-square test of independence (χ²={chi2_stat:.4f}, df={dof}, "
            f"min expected={min_expected:.1f})."
        )
        if min_expected < 5:
            warnings_list.append(
                f"Some expected frequencies are below 5 (min={min_expected:.1f}). "
                "Chi-square approximation may be unreliable."
            )

    # Effect size
    v = compute_cramers_v(chi2_stat, n, k, r)
    effect_size = {
        "metric": "cramers_v",
        "value": round(v, 4),
        "magnitude": _classify_cramers_v(v),
    }

    # Contingency table for response
    ct_dict = {
        "rows": list(contingency.index.astype(str)),
        "columns": list(contingency.columns.astype(str)),
        "observed": contingency.values.tolist(),
        "expected": expected.round(2).tolist(),
    }

    significant = bool(p_value <= alpha)
    verdict = (
        f"There IS a statistically significant association between "
        f"'{col1}' and '{col2}'"
        if significant
        else f"There is NO statistically significant association between "
        f"'{col1}' and '{col2}'"
    )
    test_label = "Fisher's exact test" if use_fisher else "chi-square test"
    stat_label = "OR" if use_fisher else "χ²"
    p_str = _format_p(p_value)
    interpretation = (
        f"{verdict} ({test_label}, {stat_label}={stat_value:.4f}, {p_str}, "
        f"Cramér's V={v:.4f} [{effect_size['magnitude']} effect])."
    )

    return {
        "test_used": test_used,
        "reason": reason,
        "assumptions": {
            "min_expected_frequency": round(float(min_expected), 2),
            "fisher_fallback": use_fisher,
        },
        "result": {
            "statistic": round(float(stat_value), 6),
            "p_value": round(float(p_value), 6),
            "significant": significant,
            "alpha": alpha,
            "degrees_of_freedom": int(dof) if not use_fisher else None,
        },
        "effect_size": effect_size,
        "contingency_table": ct_dict,
        "interpretation": interpretation,
        "warnings": warnings_list,
    }


def run_chi_square_goodness_of_fit(
    df: pd.DataFrame,
    column: str,
    expected_proportions: Optional[Dict[str, float]] = None,
    alpha: float = 0.05,
) -> Dict[str, Any]:
    """Chi-square goodness-of-fit test for a single categorical column.

    Tests whether observed frequencies match expected proportions.
    If expected_proportions is None, tests against uniform distribution.
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in dataset.")

    clean = df[column].dropna()
    n = len(clean)

    sdc = check_sdc(n)
    if sdc["status"] == "suppress":
        return {
            "test_used": "suppressed",
            "reason": sdc["message"],
            "assumptions": {},
            "result": {
                "statistic": None, "p_value": None,
                "significant": None, "alpha": alpha,
            },
            "effect_size": {"metric": None, "value": None, "magnitude": None},
            "frequency_table": {},
            "interpretation": sdc["message"],
            "warnings": [sdc["message"]],
        }

    warnings_list: List[str] = []
    if sdc["status"] == "warn":
        warnings_list.append(sdc["message"])

    observed_counts = clean.value_counts()
    categories = list(observed_counts.index.astype(str))
    observed = observed_counts.values.astype(float)
    k = len(categories)

    if expected_proportions:
        # Map user-supplied proportions to observed category order
        expected_freq = np.array([
            expected_proportions.get(cat, 1.0 / k) * n
            for cat in categories
        ])
    else:
        # Uniform distribution
        expected_freq = np.full(k, n / k)

    chi2_stat, p_value = stats.chisquare(observed, f_exp=expected_freq)
    dof = k - 1

    # Effect size: w = sqrt(chi2 / n)
    w = math.sqrt(chi2_stat / n) if n > 0 else 0.0
    w_mag = "large" if w >= 0.5 else ("medium" if w >= 0.3 else ("small" if w >= 0.1 else "negligible"))

    significant = bool(p_value <= alpha)
    dist_type = "uniform" if not expected_proportions else "specified"
    verdict = (
        f"The observed distribution of '{column}' significantly differs "
        f"from the {dist_type} distribution"
        if significant
        else f"The observed distribution of '{column}' does NOT significantly "
        f"differ from the {dist_type} distribution"
    )
    p_str = _format_p(p_value)
    interpretation = (
        f"{verdict} (χ²={chi2_stat:.4f}, df={dof}, {p_str}, "
        f"w={w:.4f} [{w_mag} effect])."
    )

    freq_table = {
        "categories": categories,
        "observed": observed.tolist(),
        "expected": expected_freq.round(2).tolist(),
    }

    min_expected = expected_freq.min()
    if min_expected < 5:
        warnings_list.append(
            f"Some expected frequencies are below 5 (min={min_expected:.1f}). "
            "Chi-square approximation may be unreliable."
        )

    return {
        "test_used": "chi_square_goodness_of_fit",
        "reason": f"Goodness-of-fit test against {dist_type} distribution.",
        "assumptions": {
            "min_expected_frequency": round(float(min_expected), 2),
        },
        "result": {
            "statistic": round(float(chi2_stat), 6),
            "p_value": round(float(p_value), 6),
            "significant": significant,
            "alpha": alpha,
            "degrees_of_freedom": dof,
        },
        "effect_size": {
            "metric": "cohens_w",
            "value": round(w, 4),
            "magnitude": w_mag,
        },
        "frequency_table": freq_table,
        "interpretation": interpretation,
        "warnings": warnings_list,
    }


# =====================================================================
# Correlation Analysis (Week 7)
# =====================================================================

CORRELATION_THRESHOLDS = {"small": 0.1, "medium": 0.3, "large": 0.5}


def _classify_correlation(r: float) -> str:
    """Classify correlation magnitude."""
    abs_r = abs(r)
    if abs_r >= CORRELATION_THRESHOLDS["large"]:
        return "large"
    if abs_r >= CORRELATION_THRESHOLDS["medium"]:
        return "medium"
    if abs_r >= CORRELATION_THRESHOLDS["small"]:
        return "small"
    return "negligible"


def _correlation_ci(r: float, n: int, confidence: float = 0.95) -> Dict[str, float]:
    """Fisher z-transformation confidence interval for correlation."""
    if n < 4:
        return {"lower": None, "upper": None, "confidence": confidence}
    z = np.arctanh(r)
    se = 1.0 / math.sqrt(n - 3)
    z_crit = stats.norm.ppf((1 + confidence) / 2)
    lower = math.tanh(z - z_crit * se)
    upper = math.tanh(z + z_crit * se)
    return {
        "lower": round(float(lower), 4),
        "upper": round(float(upper), 4),
        "confidence": confidence,
    }


def run_pearson_correlation(
    df: pd.DataFrame,
    col1: str,
    col2: str,
    alpha: float = 0.05,
) -> Dict[str, Any]:
    """Pearson product-moment correlation with CI and effect size."""
    _validate_numeric_col(df, col1)
    _validate_numeric_col(df, col2)

    clean = df[[col1, col2]].dropna()
    n = len(clean)

    sdc = check_sdc(n)
    warnings_list: List[str] = []
    if sdc["status"] == "suppress":
        return {
            "test_used": "suppressed",
            "reason": sdc["message"],
            "result": {"statistic": None, "p_value": None, "significant": None, "alpha": alpha},
            "effect_size": {"metric": None, "value": None, "magnitude": None},
            "confidence_interval": {},
            "interpretation": sdc["message"],
            "warnings": [sdc["message"]],
        }
    if sdc["status"] == "warn":
        warnings_list.append(sdc["message"])

    r, p = stats.pearsonr(clean[col1], clean[col2])
    ci = _correlation_ci(r, n)

    significant = bool(p <= alpha)
    p_str = _format_p(p)
    direction = "positive" if r > 0 else "negative"
    magnitude = _classify_correlation(r)
    verdict = (
        f"There IS a statistically significant {direction} correlation"
        if significant
        else f"There is NO statistically significant correlation"
    )
    interpretation = (
        f"{verdict} between '{col1}' and '{col2}' "
        f"(Pearson r={r:.4f}, {p_str}, [{magnitude} effect], "
        f"95% CI [{ci['lower']}, {ci['upper']}])."
    )

    return {
        "test_used": "pearson",
        "reason": "Pearson product-moment correlation.",
        "result": {
            "statistic": round(float(r), 6),
            "p_value": round(float(p), 6),
            "significant": significant,
            "alpha": alpha,
        },
        "effect_size": {
            "metric": "pearson_r",
            "value": round(float(r), 4),
            "magnitude": magnitude,
        },
        "confidence_interval": ci,
        "sample_size": n,
        "interpretation": interpretation,
        "warnings": warnings_list,
    }


def run_spearman_correlation(
    df: pd.DataFrame,
    col1: str,
    col2: str,
    alpha: float = 0.05,
) -> Dict[str, Any]:
    """Spearman rank correlation with effect size."""
    _validate_numeric_col(df, col1)
    _validate_numeric_col(df, col2)

    clean = df[[col1, col2]].dropna()
    n = len(clean)

    sdc = check_sdc(n)
    warnings_list: List[str] = []
    if sdc["status"] == "suppress":
        return {
            "test_used": "suppressed",
            "reason": sdc["message"],
            "result": {"statistic": None, "p_value": None, "significant": None, "alpha": alpha},
            "effect_size": {"metric": None, "value": None, "magnitude": None},
            "interpretation": sdc["message"],
            "warnings": [sdc["message"]],
        }
    if sdc["status"] == "warn":
        warnings_list.append(sdc["message"])

    rho, p = stats.spearmanr(clean[col1], clean[col2])
    ci = _correlation_ci(rho, n)

    significant = bool(p <= alpha)
    p_str = _format_p(p)
    direction = "positive" if rho > 0 else "negative"
    magnitude = _classify_correlation(rho)
    verdict = (
        f"There IS a statistically significant {direction} monotonic relationship"
        if significant
        else f"There is NO statistically significant monotonic relationship"
    )
    interpretation = (
        f"{verdict} between '{col1}' and '{col2}' "
        f"(Spearman ρ={rho:.4f}, {p_str}, [{magnitude} effect], "
        f"95% CI [{ci['lower']}, {ci['upper']}])."
    )

    return {
        "test_used": "spearman",
        "reason": "Spearman rank correlation (non-parametric).",
        "result": {
            "statistic": round(float(rho), 6),
            "p_value": round(float(p), 6),
            "significant": significant,
            "alpha": alpha,
        },
        "effect_size": {
            "metric": "spearman_rho",
            "value": round(float(rho), 4),
            "magnitude": magnitude,
        },
        "confidence_interval": ci,
        "sample_size": n,
        "interpretation": interpretation,
        "warnings": warnings_list,
    }


def run_correlation_analysis(
    df: pd.DataFrame,
    col1: str,
    col2: str,
    alpha: float = 0.05,
) -> Dict[str, Any]:
    """Auto-select Pearson or Spearman based on normality testing.

    Uses Pearson if both columns are normally distributed,
    otherwise falls back to Spearman.
    """
    _validate_numeric_col(df, col1)
    _validate_numeric_col(df, col2)

    clean = df[[col1, col2]].dropna()
    norm1 = check_normality(clean[col1].values, alpha)
    norm2 = check_normality(clean[col2].values, alpha)
    both_normal = norm1["normal"] and norm2["normal"]

    if both_normal:
        result = run_pearson_correlation(df, col1, col2, alpha)
        result["reason"] = (
            f"Both columns are normally distributed "
            f"({norm1['test']}: '{col1}' p={norm1['p_value']}, "
            f"'{col2}' p={norm2['p_value']}). Using Pearson correlation."
        )
    else:
        result = run_spearman_correlation(df, col1, col2, alpha)
        result["reason"] = (
            f"Data is NOT normally distributed "
            f"({norm1['test']}: '{col1}' p={norm1['p_value']}, "
            f"'{col2}' p={norm2['p_value']}). Using Spearman correlation."
        )

    result["assumptions"] = {
        "normality": {
            "test": norm1["test"],
            col1: {"p_value": norm1["p_value"], "normal": norm1["normal"]},
            col2: {"p_value": norm2["p_value"], "normal": norm2["normal"]},
        },
        "method_selected": "pearson" if both_normal else "spearman",
    }
    return result


def run_correlation_matrix(
    df: pd.DataFrame,
    columns: Optional[List[str]] = None,
    method: str = "auto",
    alpha: float = 0.05,
) -> Dict[str, Any]:
    """Pairwise correlation matrix across multiple numeric columns.

    method: 'pearson', 'spearman', or 'auto' (auto-selects per pair).
    """
    if columns:
        for col in columns:
            _validate_numeric_col(df, col)
        target_cols = columns
    else:
        target_cols = [c for c in df.columns if np.issubdtype(df[c].dtype, np.number)]

    if len(target_cols) < 2:
        raise ValueError(
            f"Correlation matrix requires at least 2 numeric columns, "
            f"found {len(target_cols)}."
        )

    matrix: Dict[str, Dict[str, Any]] = {}
    warnings_list: List[str] = []

    for i, c1 in enumerate(target_cols):
        matrix[c1] = {}
        for j, c2 in enumerate(target_cols):
            if i == j:
                matrix[c1][c2] = {
                    "r": 1.0, "p_value": 0.0, "method": "identity",
                    "significant": True, "magnitude": "large",
                }
                continue

            if j < i:
                # Symmetric — copy from the other direction
                matrix[c1][c2] = matrix[c2][c1].copy()
                continue

            if method == "pearson":
                result = run_pearson_correlation(df, c1, c2, alpha)
            elif method == "spearman":
                result = run_spearman_correlation(df, c1, c2, alpha)
            else:
                result = run_correlation_analysis(df, c1, c2, alpha)

            matrix[c1][c2] = {
                "r": result["result"]["statistic"],
                "p_value": result["result"]["p_value"],
                "method": result["test_used"],
                "significant": result["result"]["significant"],
                "magnitude": result["effect_size"]["magnitude"],
            }
            if result.get("warnings"):
                warnings_list.extend(result["warnings"])

    return {
        "test_used": "correlation_matrix",
        "reason": f"Pairwise correlation matrix ({method} method) "
                  f"across {len(target_cols)} columns.",
        "columns": target_cols,
        "matrix": matrix,
        "method": method,
        "alpha": alpha,
        "interpretation": (
            f"Correlation matrix computed for {len(target_cols)} columns "
            f"with {len(target_cols) * (len(target_cols) - 1) // 2} "
            f"unique pairs."
        ),
        "warnings": list(set(warnings_list)),
    }
