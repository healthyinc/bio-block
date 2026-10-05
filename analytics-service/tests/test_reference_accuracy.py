import math
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from app.services.inferential import (
    run_students_ttest,
    run_welchs_ttest,
    run_mann_whitney,
    run_one_sample_ttest,
    run_two_group_test,
    run_one_way_anova,
    run_chi_square_independence,
    run_pearson_correlation,
    run_spearman_correlation,
    check_normality,
    check_equal_variance,
    compute_cohens_d,
)

FIXTURES_DIR = Path(__file__).parent / "fixtures"


@pytest.fixture(scope="module")
def uci_heart():
    return pd.read_csv(FIXTURES_DIR / "uci_heart_disease.csv")


@pytest.fixture(scope="module")
def mimic_demo():
    return pd.read_csv(FIXTURES_DIR / "mimic_iv_demo.csv")


# --- Descriptive stats ---


def test_uci_age_descriptives(uci_heart):
    age = uci_heart["age"].dropna()
    assert abs(age.mean() - 52.60) < 0.5
    assert abs(age.median() - 53.0) < 1.0
    assert age.min() == 29
    assert age.max() == 71


def test_uci_cholesterol_descriptives(uci_heart):
    chol = uci_heart["chol"].dropna()
    assert abs(chol.mean() - np.mean(chol.values)) < 1e-10
    assert abs(chol.std(ddof=1) - np.std(chol.values, ddof=1)) < 1e-10


def test_mimic_heart_rate_descriptives(mimic_demo):
    hr = mimic_demo["heart_rate"].dropna()
    assert 60 <= hr.mean() <= 120
    assert hr.min() >= 50
    assert hr.max() <= 150


def test_mimic_creatinine_by_mortality(mimic_demo):
    survivors = mimic_demo[mimic_demo["mortality"] == 0]["creatinine"]
    non_survivors = mimic_demo[mimic_demo["mortality"] == 1]["creatinine"]
    assert survivors.mean() < non_survivors.mean()


# --- Two-sample t-tests ---


def test_welchs_ttest_uci_chol_by_target(uci_heart):
    g0 = uci_heart[uci_heart["target"] == 0]["chol"].dropna().values
    g1 = uci_heart[uci_heart["target"] == 1]["chol"].dropna().values

    scipy_stat, scipy_p = stats.ttest_ind(g0, g1, equal_var=False)
    res = run_welchs_ttest(g0, g1)

    assert abs(res["statistic"] - scipy_stat) < 1e-6
    assert abs(res["p_value"] - scipy_p) < 1e-6


def test_students_ttest_uci_trestbps_by_sex(uci_heart):
    male = uci_heart[uci_heart["sex"] == 1]["trestbps"].dropna().values
    female = uci_heart[uci_heart["sex"] == 0]["trestbps"].dropna().values

    scipy_stat, scipy_p = stats.ttest_ind(male, female, equal_var=True)
    res = run_students_ttest(male, female)

    assert abs(res["statistic"] - scipy_stat) < 1e-6
    assert abs(res["p_value"] - scipy_p) < 1e-6


def test_welchs_ttest_mimic_hr_by_mortality(mimic_demo):
    survivors = mimic_demo[mimic_demo["mortality"] == 0]["heart_rate"].dropna().values
    non_survivors = mimic_demo[mimic_demo["mortality"] == 1]["heart_rate"].dropna().values

    scipy_stat, scipy_p = stats.ttest_ind(survivors, non_survivors, equal_var=False)
    res = run_welchs_ttest(survivors, non_survivors)

    assert abs(res["statistic"] - scipy_stat) < 1e-6
    assert abs(res["p_value"] - scipy_p) < 1e-6


def test_cohens_d_formula(mimic_demo):
    g1 = mimic_demo[mimic_demo["mortality"] == 0]["creatinine"].dropna().values
    g2 = mimic_demo[mimic_demo["mortality"] == 1]["creatinine"].dropna().values

    d = compute_cohens_d(g1, g2)

    n1, n2 = len(g1), len(g2)
    s_pooled = math.sqrt(
        ((n1 - 1) * np.std(g1, ddof=1) ** 2 + (n2 - 1) * np.std(g2, ddof=1) ** 2)
        / (n1 + n2 - 2)
    )
    expected = (np.mean(g1) - np.mean(g2)) / s_pooled
    assert abs(d - expected) < 1e-6


# --- Mann-Whitney U ---


def test_mann_whitney_uci_oldpeak_by_target(uci_heart):
    g0 = uci_heart[uci_heart["target"] == 0]["oldpeak"].dropna().values
    g1 = uci_heart[uci_heart["target"] == 1]["oldpeak"].dropna().values

    scipy_stat, scipy_p = stats.mannwhitneyu(g0, g1, alternative="two-sided")
    res = run_mann_whitney(g0, g1)

    assert abs(res["statistic"] - scipy_stat) < 1e-6
    assert abs(res["p_value"] - scipy_p) < 1e-4


def test_mann_whitney_mimic_lactate_by_mortality(mimic_demo):
    survivors = mimic_demo[mimic_demo["mortality"] == 0]["lactate"].dropna().values
    non_survivors = mimic_demo[mimic_demo["mortality"] == 1]["lactate"].dropna().values

    scipy_stat, scipy_p = stats.mannwhitneyu(survivors, non_survivors, alternative="two-sided")
    res = run_mann_whitney(survivors, non_survivors)

    assert abs(res["statistic"] - scipy_stat) < 1e-6
    assert abs(res["p_value"] - scipy_p) < 1e-4


# --- One-way ANOVA / Kruskal-Wallis ---


def test_anova_uci_chol_by_cp(uci_heart):
    groups = [
        uci_heart[uci_heart["cp"] == cp]["chol"].dropna().values
        for cp in sorted(uci_heart["cp"].unique())
        if len(uci_heart[uci_heart["cp"] == cp]["chol"].dropna()) > 0
    ]

    df = uci_heart[["chol", "cp"]].dropna().copy()
    df["cp"] = df["cp"].astype(str)
    res = run_one_way_anova(df, "chol", "cp")

    if res["test_used"] == "kruskal_wallis":
        scipy_stat, scipy_p = stats.kruskal(*groups)
    else:
        scipy_stat, scipy_p = stats.f_oneway(*groups)

    assert abs(res["result"]["statistic"] - scipy_stat) < 1e-4
    assert abs(res["result"]["p_value"] - scipy_p) < 1e-4


def test_anova_uci_thalach_by_slope(uci_heart):
    groups = [
        uci_heart[uci_heart["slope"] == s]["thalach"].dropna().values
        for s in sorted(uci_heart["slope"].unique())
        if len(uci_heart[uci_heart["slope"] == s]["thalach"].dropna()) > 0
    ]

    df = uci_heart[["thalach", "slope"]].dropna().copy()
    df["slope"] = df["slope"].astype(str)
    res = run_one_way_anova(df, "thalach", "slope")

    if res["test_used"] == "kruskal_wallis":
        scipy_stat, scipy_p = stats.kruskal(*groups)
    else:
        scipy_stat, scipy_p = stats.f_oneway(*groups)

    assert abs(res["result"]["statistic"] - scipy_stat) < 1e-4
    assert abs(res["result"]["p_value"] - scipy_p) < 1e-4


# --- Chi-Square ---


def test_chi_square_uci_sex_target(uci_heart):
    ct = pd.crosstab(uci_heart["sex"], uci_heart["target"])
    scipy_chi2, scipy_p, scipy_dof, _ = stats.chi2_contingency(ct)

    df = uci_heart[["sex", "target"]].dropna().copy()
    df["sex"] = df["sex"].astype(str)
    df["target"] = df["target"].astype(str)
    res = run_chi_square_independence(df, "sex", "target")

    assert abs(res["result"]["statistic"] - scipy_chi2) < 1e-4
    assert abs(res["result"]["p_value"] - scipy_p) < 1e-4
    assert res["result"]["degrees_of_freedom"] == scipy_dof


def test_chi_square_uci_fbs_target(uci_heart):
    df = uci_heart[["fbs", "target"]].dropna().copy()
    df["fbs"] = df["fbs"].astype(str)
    df["target"] = df["target"].astype(str)
    res = run_chi_square_independence(df, "fbs", "target")

    assert res["test_used"] in ["chi_square_independence", "fisher_exact"]
    assert 0 <= res["result"]["p_value"] <= 1


def test_chi_square_mimic_gender_mortality(mimic_demo):
    df = mimic_demo[["gender", "mortality"]].dropna().copy()
    df["mortality"] = df["mortality"].astype(str)
    res = run_chi_square_independence(df, "gender", "mortality")

    assert res["test_used"] in ["chi_square_independence", "fisher_exact"]
    assert 0 <= res["result"]["p_value"] <= 1


# --- Correlation ---


def test_pearson_uci_age_trestbps(uci_heart):
    x = uci_heart["age"].dropna().values.astype(float)
    y = uci_heart["trestbps"].dropna().values.astype(float)
    mask = ~(np.isnan(x) | np.isnan(y))
    x, y = x[mask], y[mask]

    scipy_r, scipy_p = stats.pearsonr(x, y)

    df = uci_heart[["age", "trestbps"]].dropna()
    res = run_pearson_correlation(df, "age", "trestbps")

    assert abs(res["result"]["statistic"] - scipy_r) < 1e-5
    assert abs(res["result"]["p_value"] - scipy_p) < 1e-4


def test_spearman_uci_age_thalach(uci_heart):
    x = uci_heart["age"].dropna().values.astype(float)
    y = uci_heart["thalach"].dropna().values.astype(float)
    mask = ~(np.isnan(x) | np.isnan(y))
    x, y = x[mask], y[mask]

    scipy_rho, scipy_p = stats.spearmanr(x, y)

    df = uci_heart[["age", "thalach"]].dropna()
    res = run_spearman_correlation(df, "age", "thalach")

    assert abs(res["result"]["statistic"] - scipy_rho) < 1e-5
    assert abs(res["result"]["p_value"] - scipy_p) < 1e-4


def test_pearson_mimic_creatinine_lactate(mimic_demo):
    x = mimic_demo["creatinine"].dropna().values.astype(float)
    y = mimic_demo["lactate"].dropna().values.astype(float)

    scipy_r, scipy_p = stats.pearsonr(x, y)

    df = mimic_demo[["creatinine", "lactate"]].dropna()
    res = run_pearson_correlation(df, "creatinine", "lactate")

    assert abs(res["result"]["statistic"] - scipy_r) < 1e-5
    assert abs(res["result"]["p_value"] - scipy_p) < 1e-4
    assert res["result"]["statistic"] > 0


def test_spearman_mimic_wbc_lactate(mimic_demo):
    x = mimic_demo["wbc"].dropna().values.astype(float)
    y = mimic_demo["lactate"].dropna().values.astype(float)

    scipy_rho, scipy_p = stats.spearmanr(x, y)

    df = mimic_demo[["wbc", "lactate"]].dropna()
    res = run_spearman_correlation(df, "wbc", "lactate")

    assert abs(res["result"]["statistic"] - scipy_rho) < 1e-5
    assert abs(res["result"]["p_value"] - scipy_p) < 1e-4


# --- Normality & homoscedasticity ---


def test_shapiro_uci_age(uci_heart):
    data = uci_heart["age"].dropna().values.astype(float)
    scipy_stat, scipy_p = stats.shapiro(data)
    res = check_normality(data)

    assert res["test"] == "shapiro_wilk"
    assert abs(res["statistic"] - scipy_stat) < 1e-6
    assert abs(res["p_value"] - scipy_p) < 1e-4


def test_shapiro_mimic_glucose(mimic_demo):
    data = mimic_demo["glucose_lab"].dropna().values.astype(float)
    scipy_stat, scipy_p = stats.shapiro(data)
    res = check_normality(data)

    assert res["test"] == "shapiro_wilk"
    assert abs(res["statistic"] - scipy_stat) < 1e-6
    assert abs(res["p_value"] - scipy_p) < 1e-4


def test_levene_uci_chol_by_target(uci_heart):
    g0 = uci_heart[uci_heart["target"] == 0]["chol"].dropna().values
    g1 = uci_heart[uci_heart["target"] == 1]["chol"].dropna().values

    scipy_stat, scipy_p = stats.levene(g0, g1)
    res = check_equal_variance(g0, g1)

    assert abs(res["statistic"] - scipy_stat) < 1e-4
    assert abs(res["p_value"] - scipy_p) < 1e-4


# --- One-sample t-test ---


def test_one_sample_uci_trestbps(uci_heart):
    data = uci_heart["trestbps"].dropna().values.astype(float)
    scipy_stat, scipy_p = stats.ttest_1samp(data, 120)
    res = run_one_sample_ttest(data, 120)

    assert abs(res["statistic"] - scipy_stat) < 1e-6
    assert abs(res["p_value"] - scipy_p) < 1e-6


def test_one_sample_mimic_hr(mimic_demo):
    data = mimic_demo["heart_rate"].dropna().values.astype(float)
    scipy_stat, scipy_p = stats.ttest_1samp(data, 80)
    res = run_one_sample_ttest(data, 80)

    assert abs(res["statistic"] - scipy_stat) < 1e-6
    assert abs(res["p_value"] - scipy_p) < 1e-6


# --- Pipeline sanity ---


def test_auto_test_selection_two_group(uci_heart):
    df = uci_heart[["trestbps", "sex"]].dropna().copy()
    df["sex"] = df["sex"].astype(str)
    res = run_two_group_test(df, "trestbps", "sex")

    assert res["test_used"] in ["students_ttest", "welchs_ttest", "mann_whitney_u"]
    assert 0 <= res["result"]["p_value"] <= 1


def test_auto_test_mimic_lactate_mortality(mimic_demo):
    df = mimic_demo[["lactate", "mortality"]].dropna().copy()
    df["mortality"] = df["mortality"].astype(str)
    res = run_two_group_test(df, "lactate", "mortality")

    assert res["test_used"] in ["students_ttest", "welchs_ttest", "mann_whitney_u"]
    assert res["result"]["p_value"] < 0.05
