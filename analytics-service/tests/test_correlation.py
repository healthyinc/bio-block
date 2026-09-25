import numpy as np
import pandas as pd
import pytest

from app.services.inferential import (
    run_pearson_correlation,
    run_spearman_correlation,
    run_correlation_analysis,
    run_correlation_matrix,
)


def _correlation_df(n=50, r=0.7, non_normal=False, seed=42):
    rng = np.random.default_rng(seed)
    x = rng.normal(100, 15, n) if not non_normal else rng.exponential(scale=5.0, size=n)
    # Generate correlated y
    noise_sd = 10 if not non_normal else 1.0
    noise = rng.normal(0, noise_sd, n)
    y = r * x + noise
    return pd.DataFrame({"x": x, "y": y})


class TestCorrelationAnalysis:
    def test_pearson_correlation_significant(self):
        df = _correlation_df(n=100, r=0.8)
        result = run_pearson_correlation(df, "x", "y")
        assert result["test_used"] == "pearson"
        assert result["result"]["significant"] is True
        assert result["result"]["statistic"] > 0.5
        assert "confidence_interval" in result
        assert result["effect_size"]["magnitude"] == "large"

    def test_spearman_correlation_significant(self):
        df = _correlation_df(n=100, r=0.8, non_normal=True)
        result = run_spearman_correlation(df, "x", "y")
        assert result["test_used"] == "spearman"
        assert result["result"]["significant"] is True
        assert result["result"]["statistic"] > 0.5
        assert result["effect_size"]["magnitude"] == "large"

    def test_correlation_analysis_auto_selects_pearson(self):
        # Pearson is used when both columns are normal
        df = _correlation_df(n=100, r=0.6, non_normal=False)
        result = run_correlation_analysis(df, "x", "y")
        assert result["assumptions"]["method_selected"] == "pearson"

    def test_correlation_analysis_auto_selects_spearman(self):
        # Spearman is used when at least one column is non-normal
        df = _correlation_df(n=100, r=0.6, non_normal=True)
        result = run_correlation_analysis(df, "x", "y")
        assert result["assumptions"]["method_selected"] == "spearman"

    def test_correlation_matrix_pearson(self):
        rng = np.random.default_rng(42)
        df = pd.DataFrame({
            "a": rng.normal(10, 2, 50),
            "b": rng.normal(20, 5, 50),
            "c": rng.normal(30, 8, 50),
        })
        result = run_correlation_matrix(df, method="pearson")
        assert result["test_used"] == "correlation_matrix"
        assert "a" in result["matrix"]
        assert "b" in result["matrix"]["a"]
        assert result["matrix"]["a"]["a"]["r"] == 1.0

    def test_correlation_matrix_too_few_numeric_columns(self):
        df = pd.DataFrame({
            "a": [1, 2, 3],
            "b": ["x", "y", "z"],
        })
        with pytest.raises(ValueError, match="at least 2"):
            run_correlation_matrix(df)

    def test_sdc_suppression(self):
        df = pd.DataFrame({
            "x": [1.0, 2.0, 3.0],
            "y": [2.0, 4.0, 6.0],
        })
        result = run_pearson_correlation(df, "x", "y")
        assert result["test_used"] == "suppressed"

    def test_column_not_found(self):
        df = pd.DataFrame({"x": [1, 2], "y": [3, 4]})
        with pytest.raises(ValueError, match="not found"):
            run_pearson_correlation(df, "x", "z")
