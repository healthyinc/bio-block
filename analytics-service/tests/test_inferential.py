import numpy as np
import pandas as pd
import pytest

from app.services.inferential import (
    check_normality,
    check_equal_variance,
    check_sdc,
    compute_cohens_d,
    compute_cohens_d_paired,
    compute_cohens_d_one_sample,
    compute_rank_biserial,
    classify_effect_size,
    run_two_group_test,
    run_paired_test,
    run_one_sample_test,
    SHAPIRO_MAX_N,
)


def _normal_data(n=50, mean=100, std=10, seed=42):
    rng = np.random.default_rng(seed)
    return rng.normal(mean, std, n)


def _skewed_data(n=50, seed=42):
    rng = np.random.default_rng(seed)
    return rng.exponential(scale=2.0, size=n)


def _two_group_df(g1_mean=100, g2_mean=100, g1_std=10, g2_std=10,
                  n1=50, n2=50, seed=42):
    rng = np.random.default_rng(seed)
    values = np.concatenate([
        rng.normal(g1_mean, g1_std, n1),
        rng.normal(g2_mean, g2_std, n2),
    ])
    groups = ["A"] * n1 + ["B"] * n2
    return pd.DataFrame({"score": values, "group": groups})


class TestCheckNormality:
    def test_normal_data_passes(self):
        data = _normal_data(n=100)
        result = check_normality(data)
        assert result["test"] == "shapiro_wilk"
        assert result["normal"] is True
        assert result["note"] is None

    def test_skewed_data_fails(self):
        data = _skewed_data(n=100)
        result = check_normality(data)
        assert result["test"] == "shapiro_wilk"
        assert result["normal"] is False

    def test_insufficient_data(self):
        data = np.array([1.0, 2.0])
        result = check_normality(data)
        assert result["test"] == "insufficient_data"
        assert result["normal"] is False
        assert result["statistic"] is None

    def test_ks_fallback_large_n(self):
        data = _normal_data(n=SHAPIRO_MAX_N + 100, mean=50, std=5)
        result = check_normality(data)
        assert result["test"] == "kolmogorov_smirnov"
        assert result["note"] is not None
        assert "Kolmogorov-Smirnov" in result["note"]

    def test_ks_normal_large_sample_passes(self):
        data = _normal_data(n=6000, mean=50, std=5)
        result = check_normality(data)
        assert result["test"] == "kolmogorov_smirnov"
        assert result["normal"] is True

    def test_handles_nan_values(self):
        data = np.array([1.0, 2.0, np.nan, 3.0, 4.0, 5.0, 6.0, 7.0])
        result = check_normality(data)
        assert result["statistic"] is not None


class TestCheckEqualVariance:
    def test_equal_variance(self):
        g1 = _normal_data(n=50, std=10, seed=1)
        g2 = _normal_data(n=50, std=10, seed=2)
        result = check_equal_variance(g1, g2)
        assert result["test"] == "levenes"
        assert result["equal_variance"] is True

    def test_unequal_variance(self):
        g1 = _normal_data(n=50, std=5, seed=1)
        g2 = _normal_data(n=50, std=50, seed=2)
        result = check_equal_variance(g1, g2)
        assert result["test"] == "levenes"
        assert result["equal_variance"] is False


class TestCheckSdc:
    def test_suppress_n_lt_5(self):
        result = check_sdc(3)
        assert result["status"] == "suppress"
        assert "suppressed" in result["message"].lower()

    def test_warn_n_lt_30(self):
        result = check_sdc(15)
        assert result["status"] == "warn"
        assert "warning" in result["message"].lower()

    def test_ok_n_gte_30(self):
        result = check_sdc(50)
        assert result["status"] == "ok"
        assert result["message"] is None

    def test_boundary_n_5(self):
        result = check_sdc(5)
        assert result["status"] == "warn"

    def test_boundary_n_30(self):
        result = check_sdc(30)
        assert result["status"] == "ok"


class TestEffectSizes:
    def test_cohens_d_independent(self):
        g1 = np.array([10.0, 12.0, 14.0, 16.0, 18.0])
        g2 = np.array([20.0, 22.0, 24.0, 26.0, 28.0])
        d = compute_cohens_d(g1, g2)
        assert abs(d) > 2.0

    def test_cohens_d_paired(self):
        diffs = np.array([1.0, 2.0, 1.5, 2.5, 1.0])
        d = compute_cohens_d_paired(diffs)
        assert d > 0

    def test_cohens_d_one_sample(self):
        sample = np.array([105.0, 110.0, 108.0, 112.0, 107.0])
        d = compute_cohens_d_one_sample(sample, population_mean=100.0)
        assert d > 0

    def test_cohens_d_zero_std(self):
        g1 = np.array([5.0, 5.0, 5.0])
        g2 = np.array([5.0, 5.0, 5.0])
        d = compute_cohens_d(g1, g2)
        assert d == 0.0

    def test_rank_biserial(self):
        r = compute_rank_biserial(0.0, 5, 5)
        assert r == 1.0
        r = compute_rank_biserial(12.5, 5, 5)
        assert abs(r) < 0.01

    def test_classify_cohens_d_thresholds(self):
        assert classify_effect_size(0.1, "cohens_d") == "negligible"
        assert classify_effect_size(0.3, "cohens_d") == "small"
        assert classify_effect_size(0.6, "cohens_d") == "medium"
        assert classify_effect_size(1.0, "cohens_d") == "large"

    def test_classify_rank_biserial_thresholds(self):
        assert classify_effect_size(0.05, "rank_biserial_r") == "negligible"
        assert classify_effect_size(0.15, "rank_biserial_r") == "small"
        assert classify_effect_size(0.35, "rank_biserial_r") == "medium"
        assert classify_effect_size(0.6, "rank_biserial_r") == "large"

    def test_negative_effect_size_uses_absolute_value(self):
        assert classify_effect_size(-0.9, "cohens_d") == "large"


class TestRunTwoGroupTest:
    def test_routes_to_students_t(self):
        df = _two_group_df(g1_mean=100, g2_mean=105, g1_std=10, g2_std=10)
        result = run_two_group_test(df, "score", "group")
        assert result["test_used"] == "students_ttest"
        assert "Student" in result["reason"]
        assert "normality" in result["assumptions"]

    def test_routes_to_welchs_t(self):
        df = _two_group_df(g1_mean=100, g2_mean=105, g1_std=5, g2_std=50)
        result = run_two_group_test(df, "score", "group")
        assert result["test_used"] == "welchs_ttest"
        assert "Welch" in result["reason"]
        assert "equal_variance" in result["assumptions"]

    def test_routes_to_mann_whitney(self):
        rng = np.random.default_rng(42)
        values = np.concatenate([
            rng.exponential(2.0, 50),
            rng.exponential(5.0, 50),
        ])
        groups = ["A"] * 50 + ["B"] * 50
        df = pd.DataFrame({"score": values, "group": groups})
        result = run_two_group_test(df, "score", "group")
        assert result["test_used"] == "mann_whitney_u"
        assert "Mann-Whitney" in result["reason"]

    def test_significant_result(self):
        df = _two_group_df(g1_mean=100, g2_mean=120, g1_std=10, g2_std=10)
        result = run_two_group_test(df, "score", "group")
        assert result["result"]["significant"] is True
        assert result["result"]["p_value"] < 0.05

    def test_not_significant_result(self):
        df = _two_group_df(g1_mean=100, g2_mean=100, g1_std=10, g2_std=10)
        result = run_two_group_test(df, "score", "group")
        assert result["result"]["significant"] is False

    def test_group_col_not_two_groups(self):
        df = pd.DataFrame({
            "score": [1, 2, 3, 4, 5, 6],
            "group": ["A", "B", "C", "A", "B", "C"],
        })
        with pytest.raises(ValueError, match="exactly 2"):
            run_two_group_test(df, "score", "group")

    def test_column_not_numeric(self):
        df = pd.DataFrame({
            "name": ["alice", "bob", "charlie", "dave"],
            "group": ["A", "B", "A", "B"],
        })
        with pytest.raises(ValueError, match="not numeric"):
            run_two_group_test(df, "name", "group")

    def test_column_not_found(self):
        df = _two_group_df()
        with pytest.raises(ValueError, match="not found"):
            run_two_group_test(df, "nonexistent", "group")

    def test_sdc_suppression(self):
        df = pd.DataFrame({
            "score": [1.0, 2.0, 3.0, 10.0, 11.0, 12.0],
            "group": ["A", "A", "A", "B", "B", "B"],
        })
        result = run_two_group_test(df, "score", "group")
        assert result["test_used"] == "suppressed"
        assert len(result["warnings"]) > 0

    def test_sdc_warning(self):
        df = _two_group_df(n1=15, n2=15)
        result = run_two_group_test(df, "score", "group")
        assert result["test_used"] != "suppressed"
        assert len(result["warnings"]) > 0
        assert "warning" in result["warnings"][0].lower()

    def test_response_has_all_fields(self):
        df = _two_group_df()
        result = run_two_group_test(df, "score", "group")
        expected_keys = {
            "test_used", "reason", "assumptions", "result",
            "effect_size", "group_stats", "interpretation", "warnings",
        }
        assert expected_keys.issubset(result.keys())

    def test_interpretation_is_human_readable(self):
        df = _two_group_df(g1_mean=100, g2_mean=120)
        result = run_two_group_test(df, "score", "group")
        interp = result["interpretation"]
        assert "significant" in interp.lower()
        assert "score" in interp
        assert "effect" in interp.lower()


class TestRunPairedTest:
    def test_paired_normal_diffs(self):
        rng = np.random.default_rng(42)
        before = rng.normal(100, 10, 50)
        after = before + rng.normal(5, 3, 50)
        df = pd.DataFrame({"before": before, "after": after})
        result = run_paired_test(df, "before", "after")
        assert result["test_used"] == "paired_ttest"

    def test_paired_skewed_diffs(self):
        rng = np.random.default_rng(42)
        before = rng.exponential(10, 50)
        after = before + rng.exponential(5, 50)
        df = pd.DataFrame({"before": before, "after": after})
        result = run_paired_test(df, "before", "after")
        assert result["test_used"] == "wilcoxon_signed_rank"

    def test_paired_column_not_numeric(self):
        df = pd.DataFrame({"a": ["x", "y", "z"], "b": [1, 2, 3]})
        with pytest.raises(ValueError, match="not numeric"):
            run_paired_test(df, "a", "b")

    def test_paired_column_not_found(self):
        df = pd.DataFrame({"a": [1, 2, 3]})
        with pytest.raises(ValueError, match="not found"):
            run_paired_test(df, "a", "nonexistent")

    def test_paired_sdc_suppression(self):
        df = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [4.0, 5.0, 6.0]})
        result = run_paired_test(df, "a", "b")
        assert result["test_used"] == "suppressed"

    def test_paired_response_structure(self):
        rng = np.random.default_rng(42)
        before = rng.normal(100, 10, 50)
        after = before + rng.normal(5, 3, 50)
        df = pd.DataFrame({"before": before, "after": after})
        result = run_paired_test(df, "before", "after")
        assert "normality" in result["assumptions"]
        assert "differences" in result["assumptions"]["normality"]


class TestRunOneSampleTest:
    def test_one_sample_normal(self):
        data = _normal_data(n=50, mean=105, std=10)
        df = pd.DataFrame({"bp": data})
        result = run_one_sample_test(df, "bp", population_mean=100.0)
        assert result["test_used"] == "one_sample_ttest"

    def test_one_sample_skewed(self):
        data = _skewed_data(n=50)
        df = pd.DataFrame({"bp": data})
        result = run_one_sample_test(df, "bp", population_mean=0.5)
        assert result["test_used"] == "one_sample_wilcoxon"

    def test_one_sample_significant(self):
        data = _normal_data(n=50, mean=120, std=10)
        df = pd.DataFrame({"bp": data})
        result = run_one_sample_test(df, "bp", population_mean=100.0)
        assert result["result"]["significant"] is True

    def test_one_sample_not_significant(self):
        data = _normal_data(n=50, mean=100, std=10)
        df = pd.DataFrame({"bp": data})
        result = run_one_sample_test(df, "bp", population_mean=100.0)
        assert result["result"]["significant"] is False

    def test_one_sample_column_not_found(self):
        df = pd.DataFrame({"a": [1, 2, 3]})
        with pytest.raises(ValueError, match="not found"):
            run_one_sample_test(df, "nonexistent", population_mean=0.0)

    def test_one_sample_column_not_numeric(self):
        df = pd.DataFrame({"name": ["a", "b", "c", "d", "e", "f"]})
        with pytest.raises(ValueError, match="not numeric"):
            run_one_sample_test(df, "name", population_mean=0.0)

    def test_one_sample_sdc_suppression(self):
        df = pd.DataFrame({"bp": [120.0, 130.0, 125.0]})
        result = run_one_sample_test(df, "bp", population_mean=120.0)
        assert result["test_used"] == "suppressed"

    def test_one_sample_response_has_reference(self):
        data = _normal_data(n=50, mean=105, std=10)
        df = pd.DataFrame({"bp": data})
        result = run_one_sample_test(df, "bp", population_mean=100.0)
        assert "_reference" in result["group_stats"]
        assert result["group_stats"]["_reference"]["value"] == 100.0

    def test_one_sample_response_structure(self):
        data = _normal_data(n=50, mean=105, std=10)
        df = pd.DataFrame({"bp": data})
        result = run_one_sample_test(df, "bp", population_mean=100.0)
        assert "normality" in result["assumptions"]
        assert "sample" in result["assumptions"]["normality"]
