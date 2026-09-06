import numpy as np
import pandas as pd
import pytest

from app.services.inferential import (
    check_equal_variance_multi,
    check_normality,
    compute_eta_squared,
    compute_epsilon_squared,
    compute_kendalls_w,
    classify_effect_size,
    run_one_way_anova,
    run_two_way_anova,
    run_repeated_measures_anova,
    run_multi_group_test,
)


def _normal_groups(k=3, n=50, base_mean=100, std=10, shift=0, seed=42):
    """Generate k groups of normally distributed data."""
    rng = np.random.default_rng(seed)
    values, groups = [], []
    for i in range(k):
        data = rng.normal(base_mean + i * shift, std, n)
        values.extend(data)
        groups.extend([f"G{i+1}"] * n)
    return pd.DataFrame({"score": values, "group": groups})


def _skewed_groups(k=3, n=50, seed=42):
    """Generate k groups of exponentially distributed (skewed) data."""
    rng = np.random.default_rng(seed)
    values, groups = [], []
    for i in range(k):
        data = rng.exponential(scale=2.0 + i, size=n)
        values.extend(data)
        groups.extend([f"G{i+1}"] * n)
    return pd.DataFrame({"score": values, "group": groups})


def _two_way_df(n_per_cell=30, seed=42):
    """Generate a two-way ANOVA dataset with factors A and B."""
    rng = np.random.default_rng(seed)
    rows = []
    for a in ["Low", "High"]:
        for b in ["Control", "Treatment"]:
            base = 100
            if a == "High":
                base += 10
            if b == "Treatment":
                base += 5
            vals = rng.normal(base, 8, n_per_cell)
            for v in vals:
                rows.append({"score": v, "factor_a": a, "factor_b": b})
    return pd.DataFrame(rows)


def _repeated_measures_df(n_subjects=50, k=3, seed=42):
    """Generate repeated-measures data with k conditions."""
    rng = np.random.default_rng(seed)
    data = {}
    base = rng.normal(100, 10, n_subjects)
    for i in range(k):
        data[f"cond_{i+1}"] = base + rng.normal(i * 5, 3, n_subjects)
    return pd.DataFrame(data)


def _skewed_repeated_measures_df(n_subjects=50, k=3, seed=42):
    """Generate repeated-measures data with non-normal differences."""
    rng = np.random.default_rng(seed)
    data = {}
    for i in range(k):
        data[f"cond_{i+1}"] = rng.exponential(scale=2.0 + i * 2, size=n_subjects)
    return pd.DataFrame(data)


# ==============================================================
# Effect Size Tests
# ==============================================================

class TestAnovaEffectSizes:
    def test_eta_squared_basic(self):
        eta = compute_eta_squared(20.0, 100.0)
        assert eta == 0.2

    def test_eta_squared_zero_total(self):
        assert compute_eta_squared(10.0, 0.0) == 0.0

    def test_epsilon_squared_basic(self):
        eps = compute_epsilon_squared(15.0, 3, 150)
        # (15 - 3 + 1) / (150 - 3) = 13 / 147 ≈ 0.0884
        assert 0.08 < eps < 0.10

    def test_epsilon_squared_small_N(self):
        assert compute_epsilon_squared(5.0, 3, 3) == 0.0

    def test_kendalls_w_basic(self):
        w = compute_kendalls_w(12.0, 10, 3)
        # 12 / (10 * 2) = 0.6
        assert w == 0.6

    def test_kendalls_w_zero_denom(self):
        assert compute_kendalls_w(5.0, 0, 3) == 0.0

    def test_classify_eta_squared(self):
        assert classify_effect_size(0.005, "eta_squared") == "negligible"
        assert classify_effect_size(0.02, "eta_squared") == "small"
        assert classify_effect_size(0.08, "eta_squared") == "medium"
        assert classify_effect_size(0.20, "eta_squared") == "large"

    def test_classify_epsilon_squared(self):
        assert classify_effect_size(0.005, "epsilon_squared") == "negligible"
        assert classify_effect_size(0.03, "epsilon_squared") == "small"
        assert classify_effect_size(0.10, "epsilon_squared") == "medium"
        assert classify_effect_size(0.20, "epsilon_squared") == "large"

    def test_classify_kendalls_w(self):
        assert classify_effect_size(0.05, "kendalls_w") == "negligible"
        assert classify_effect_size(0.15, "kendalls_w") == "small"
        assert classify_effect_size(0.35, "kendalls_w") == "medium"
        assert classify_effect_size(0.6, "kendalls_w") == "large"


# ==============================================================
# Multi-group Variance Test
# ==============================================================

class TestCheckEqualVarianceMulti:
    def test_equal_variances_k_groups(self):
        rng = np.random.default_rng(42)
        g1 = rng.normal(100, 10, 50)
        g2 = rng.normal(100, 10, 50)
        g3 = rng.normal(100, 10, 50)
        result = check_equal_variance_multi(g1, g2, g3)
        assert result["test"] == "levenes"
        assert result["equal_variance"] is True

    def test_unequal_variances_k_groups(self):
        rng = np.random.default_rng(42)
        g1 = rng.normal(100, 5, 50)
        g2 = rng.normal(100, 50, 50)
        g3 = rng.normal(100, 5, 50)
        result = check_equal_variance_multi(g1, g2, g3)
        assert result["test"] == "levenes"
        assert result["equal_variance"] is False


# ==============================================================
# One-Way ANOVA
# ==============================================================

class TestOneWayAnova:
    def test_routes_to_anova_f_test(self):
        df = _normal_groups(k=3, n=50, shift=15)
        result = run_one_way_anova(df, "score", "group")
        assert result["test_used"] == "one_way_anova"
        assert "ANOVA" in result["reason"]
        assert "normality" in result["assumptions"]
        assert result["effect_size"]["metric"] == "eta_squared"

    def test_routes_to_kruskal_wallis(self):
        df = _skewed_groups(k=3, n=50)
        result = run_one_way_anova(df, "score", "group")
        assert result["test_used"] == "kruskal_wallis"
        assert "Kruskal-Wallis" in result["reason"]
        assert result["effect_size"]["metric"] == "epsilon_squared"

    def test_routes_to_welchs_anova(self):
        """Normal data but unequal variances → Welch's ANOVA."""
        rng = np.random.default_rng(42)
        values = np.concatenate([
            rng.normal(100, 5, 50),
            rng.normal(110, 50, 50),
            rng.normal(105, 5, 50),
        ])
        groups = ["A"] * 50 + ["B"] * 50 + ["C"] * 50
        df = pd.DataFrame({"score": values, "group": groups})
        result = run_one_way_anova(df, "score", "group")
        assert result["test_used"] in ("welchs_anova", "kruskal_wallis")

    def test_significant_result_has_post_hoc(self):
        df = _normal_groups(k=3, n=50, shift=20)
        result = run_one_way_anova(df, "score", "group")
        assert result["result"]["significant"] is True
        assert "post_hoc" in result
        assert len(result["post_hoc"]["comparisons"]) > 0

    def test_not_significant_no_post_hoc(self):
        df = _normal_groups(k=3, n=50, shift=0)
        result = run_one_way_anova(df, "score", "group")
        if not result["result"]["significant"]:
            assert result.get("post_hoc") is None

    def test_kruskal_significant_has_dunns_posthoc(self):
        rng = np.random.default_rng(42)
        values = np.concatenate([
            rng.exponential(1.0, 50),
            rng.exponential(5.0, 50),
            rng.exponential(10.0, 50),
        ])
        groups = ["A"] * 50 + ["B"] * 50 + ["C"] * 50
        df = pd.DataFrame({"score": values, "group": groups})
        result = run_one_way_anova(df, "score", "group")
        if result["test_used"] == "kruskal_wallis" and result["result"]["significant"]:
            assert result["post_hoc"]["method"] == "dunns_bonferroni"

    def test_requires_at_least_3_groups(self):
        df = pd.DataFrame({
            "score": [1, 2, 3, 4],
            "group": ["A", "A", "B", "B"],
        })
        with pytest.raises(ValueError, match="at least 3"):
            run_one_way_anova(df, "score", "group")

    def test_column_not_found(self):
        df = _normal_groups()
        with pytest.raises(ValueError, match="not found"):
            run_one_way_anova(df, "nonexistent", "group")

    def test_group_col_not_found(self):
        df = _normal_groups()
        with pytest.raises(ValueError, match="not found"):
            run_one_way_anova(df, "score", "nonexistent")

    def test_sdc_suppression(self):
        df = pd.DataFrame({
            "score": [1, 2, 3, 10, 11, 12, 20, 21, 22],
            "group": ["A", "A", "A", "B", "B", "B", "C", "C", "C"],
        })
        result = run_one_way_anova(df, "score", "group")
        assert result["test_used"] == "suppressed"

    def test_sdc_warning(self):
        df = _normal_groups(k=3, n=15, shift=10)
        result = run_one_way_anova(df, "score", "group")
        assert result["test_used"] != "suppressed"
        assert len(result["warnings"]) > 0

    def test_response_has_all_fields(self):
        df = _normal_groups(k=3, n=50, shift=10)
        result = run_one_way_anova(df, "score", "group")
        expected_keys = {
            "test_used", "reason", "assumptions", "result",
            "effect_size", "group_stats", "interpretation", "warnings",
        }
        assert expected_keys.issubset(result.keys())

    def test_group_stats_has_all_groups(self):
        df = _normal_groups(k=4, n=50, shift=5)
        result = run_one_way_anova(df, "score", "group")
        for g in ["G1", "G2", "G3", "G4"]:
            assert g in result["group_stats"]
            gs = result["group_stats"][g]
            assert "n" in gs and "mean" in gs and "std" in gs

    def test_interpretation_mentions_groups(self):
        df = _normal_groups(k=3, n=50, shift=15)
        result = run_one_way_anova(df, "score", "group")
        interp = result["interpretation"]
        assert "significant" in interp.lower()
        assert "effect" in interp.lower()

    def test_normality_check_uses_sw_for_small_n(self):
        df = _normal_groups(k=3, n=50, shift=10)
        result = run_one_way_anova(df, "score", "group")
        norm_test = result["assumptions"]["normality"]["test"]
        assert norm_test == "shapiro_wilk"

    def test_orchestrator_delegates(self):
        """run_multi_group_test should produce the same output."""
        df = _normal_groups(k=3, n=50, shift=10)
        result = run_multi_group_test(df, "score", "group")
        assert result["test_used"] in (
            "one_way_anova", "welchs_anova", "kruskal_wallis"
        )


# ==============================================================
# Two-Way ANOVA
# ==============================================================

class TestTwoWayAnova:
    def test_basic_two_way(self):
        df = _two_way_df(n_per_cell=30)
        result = run_two_way_anova(df, "score", "factor_a", "factor_b")
        assert result["test_used"] == "two_way_anova"
        assert "effects" not in result  # effects are inside group_stats._effects
        assert "_effects" in result["group_stats"]
        effects = result["group_stats"]["_effects"]
        assert len(effects) > 0

    def test_reports_main_effects_and_interaction(self):
        df = _two_way_df(n_per_cell=30)
        result = run_two_way_anova(df, "score", "factor_a", "factor_b")
        effects = result["group_stats"]["_effects"]
        # Should have entries for both factors and their interaction
        effect_keys = list(effects.keys())
        assert len(effect_keys) >= 2  # At least 2 main effects

    def test_each_effect_has_eta_squared(self):
        df = _two_way_df(n_per_cell=30)
        result = run_two_way_anova(df, "score", "factor_a", "factor_b")
        effects = result["group_stats"]["_effects"]
        for key, eff in effects.items():
            assert "eta_squared" in eff
            assert "F" in eff
            assert "p_value" in eff
            assert "significant" in eff

    def test_residual_normality_checked(self):
        df = _two_way_df(n_per_cell=30)
        result = run_two_way_anova(df, "score", "factor_a", "factor_b")
        assert "normality" in result["assumptions"]
        assert "residuals" in result["assumptions"]["normality"]

    def test_column_not_found(self):
        df = _two_way_df()
        with pytest.raises(ValueError, match="not found"):
            run_two_way_anova(df, "score", "factor_a", "nonexistent")

    def test_numeric_col_not_found(self):
        df = _two_way_df()
        with pytest.raises(ValueError, match="not found"):
            run_two_way_anova(df, "nonexistent", "factor_a", "factor_b")

    def test_group_stats_per_cell(self):
        df = _two_way_df(n_per_cell=30)
        result = run_two_way_anova(df, "score", "factor_a", "factor_b")
        # Should have stats for factor combinations
        non_internal = {
            k: v for k, v in result["group_stats"].items()
            if not k.startswith("_")
        }
        assert len(non_internal) >= 4  # 2x2 = 4 cells

    def test_sdc_suppression(self):
        df = pd.DataFrame({
            "score": [1, 2, 3, 4],
            "factor_a": ["A", "A", "B", "B"],
            "factor_b": ["X", "Y", "X", "Y"],
        })
        result = run_two_way_anova(df, "score", "factor_a", "factor_b")
        assert result["test_used"] == "suppressed"


# ==============================================================
# Repeated-Measures ANOVA / Friedman
# ==============================================================

class TestRepeatedMeasuresAnova:
    def test_routes_to_rm_anova(self):
        df = _repeated_measures_df(n_subjects=50, k=3)
        result = run_repeated_measures_anova(df, ["cond_1", "cond_2", "cond_3"])
        assert result["test_used"] == "repeated_measures_anova"
        assert result["effect_size"]["metric"] == "eta_squared"

    def test_routes_to_friedman(self):
        df = _skewed_repeated_measures_df(n_subjects=50, k=3)
        result = run_repeated_measures_anova(df, ["cond_1", "cond_2", "cond_3"])
        assert result["test_used"] == "friedman"
        assert result["effect_size"]["metric"] == "kendalls_w"

    def test_significant_rm_anova_has_posthoc(self):
        df = _repeated_measures_df(n_subjects=50, k=3)
        result = run_repeated_measures_anova(df, ["cond_1", "cond_2", "cond_3"])
        if result["result"]["significant"]:
            assert "post_hoc" in result
            assert len(result["post_hoc"]["comparisons"]) > 0

    def test_friedman_significant_has_wilcoxon_posthoc(self):
        df = _skewed_repeated_measures_df(n_subjects=50, k=3)
        result = run_repeated_measures_anova(df, ["cond_1", "cond_2", "cond_3"])
        if result["test_used"] == "friedman" and result["result"]["significant"]:
            assert result["post_hoc"]["method"] == "pairwise_wilcoxon_bonferroni"

    def test_requires_at_least_3_conditions(self):
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        with pytest.raises(ValueError, match="at least 3"):
            run_repeated_measures_anova(df, ["a", "b"])

    def test_column_not_found(self):
        df = _repeated_measures_df()
        with pytest.raises(ValueError, match="not found"):
            run_repeated_measures_anova(df, ["cond_1", "cond_2", "nonexistent"])

    def test_column_not_numeric(self):
        df = pd.DataFrame({
            "a": ["x", "y", "z", "w", "v"],
            "b": [1, 2, 3, 4, 5],
            "c": [6, 7, 8, 9, 10],
        })
        with pytest.raises(ValueError, match="not numeric"):
            run_repeated_measures_anova(df, ["a", "b", "c"])

    def test_sdc_suppression(self):
        df = pd.DataFrame({
            "c1": [1.0, 2.0, 3.0],
            "c2": [4.0, 5.0, 6.0],
            "c3": [7.0, 8.0, 9.0],
        })
        result = run_repeated_measures_anova(df, ["c1", "c2", "c3"])
        assert result["test_used"] == "suppressed"

    def test_group_stats_per_condition(self):
        df = _repeated_measures_df(n_subjects=50, k=3)
        result = run_repeated_measures_anova(df, ["cond_1", "cond_2", "cond_3"])
        for col in ["cond_1", "cond_2", "cond_3"]:
            assert col in result["group_stats"]
            gs = result["group_stats"][col]
            assert "n" in gs and "mean" in gs

    def test_design_metadata(self):
        df = _repeated_measures_df(n_subjects=50, k=3)
        result = run_repeated_measures_anova(df, ["cond_1", "cond_2", "cond_3"])
        assert "_design" in result["group_stats"]
        design = result["group_stats"]["_design"]
        assert design["n_subjects"] == 50
        assert design["n_conditions"] == 3

    def test_response_has_all_fields(self):
        df = _repeated_measures_df(n_subjects=50, k=3)
        result = run_repeated_measures_anova(df, ["cond_1", "cond_2", "cond_3"])
        expected_keys = {
            "test_used", "reason", "assumptions", "result",
            "effect_size", "group_stats", "interpretation", "warnings",
        }
        assert expected_keys.issubset(result.keys())

    def test_normality_checks_pairwise_differences(self):
        df = _repeated_measures_df(n_subjects=50, k=3)
        result = run_repeated_measures_anova(df, ["cond_1", "cond_2", "cond_3"])
        norm = result["assumptions"]["normality"]
        assert "differences" in norm
        # Should have C(3,2) = 3 pairwise comparisons
        assert len(norm["differences"]) == 3

    def test_four_conditions(self):
        df = _repeated_measures_df(n_subjects=50, k=4)
        cols = [f"cond_{i+1}" for i in range(4)]
        result = run_repeated_measures_anova(df, cols)
        assert result["test_used"] in ("repeated_measures_anova", "friedman")
        # C(4,2) = 6 pairwise difference checks
        assert len(result["assumptions"]["normality"]["differences"]) == 6
