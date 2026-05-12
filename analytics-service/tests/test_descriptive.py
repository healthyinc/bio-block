import pandas as pd
import pytest

from app.services.descriptive import (
    classify_columns,
    compute_column_stats,
    compute_correlation_matrix,
    run_descriptive_analysis,
)


class TestComputeColumnStats:

    def test_basic_stats(self):
        stats = compute_column_stats(pd.Series([10, 20, 30, 40, 50]))
        assert stats["count"] == 5
        assert stats["mean"] == 30.0
        assert stats["min"] == 10.0
        assert stats["max"] == 50.0
        assert stats["median"] == 30.0
        assert stats["missing_pct"] == 0.0

    def test_with_missing_values(self):
        stats = compute_column_stats(pd.Series([10, None, 30, None, 50]))
        assert stats["count"] == 3
        assert stats["missing_pct"] == 40.0
        assert stats["mean"] == 30.0

    def test_single_value(self):
        stats = compute_column_stats(pd.Series([42]))
        assert stats["count"] == 1
        assert stats["mean"] == 42.0

    def test_zero_variance(self):
        stats = compute_column_stats(pd.Series([7, 7, 7, 7]))
        assert stats["std"] == 0.0


class TestClassifyColumns:

    def test_numeric(self, sample_dataframe):
        types = classify_columns(sample_dataframe)
        assert all(types[c] == "numeric" for c in ["age", "glucose", "cholesterol"])

    def test_categorical(self):
        df = pd.DataFrame({"name": ["Alice", "Bob"], "age": [25, 30]})
        types = classify_columns(df)
        assert types["name"] == "categorical"
        assert types["age"] == "numeric"


class TestCorrelationMatrix:

    def test_perfect_correlation(self):
        df = pd.DataFrame({"x": [1, 2, 3, 4], "y": [2, 4, 6, 8]})
        corr = compute_correlation_matrix(df, ["x", "y"])
        assert corr["x"]["y"] == 1.0

    def test_single_column_returns_none(self):
        df = pd.DataFrame({"x": [1, 2, 3]})
        assert compute_correlation_matrix(df, ["x"]) is None

    def test_values_in_range(self, sample_dataframe):
        cols = ["age", "glucose", "cholesterol"]
        corr = compute_correlation_matrix(sample_dataframe, cols)
        for a in cols:
            for b in cols:
                assert -1.0 <= corr[a][b] <= 1.0


class TestRunDescriptiveAnalysis:

    def test_auto_selects_numeric(self, sample_dataframe):
        result = run_descriptive_analysis(sample_dataframe)
        assert set(result["columns_analyzed"]) == {"age", "glucose", "cholesterol"}

    def test_column_filter(self, sample_dataframe):
        result = run_descriptive_analysis(sample_dataframe, columns=["age", "glucose"])
        assert "cholesterol" not in result["columns_analyzed"]

    def test_has_correlations(self, sample_dataframe):
        result = run_descriptive_analysis(sample_dataframe)
        assert "_correlations" in result["results"]

    def test_no_correlations_single_col(self, sample_dataframe):
        result = run_descriptive_analysis(sample_dataframe, columns=["age"])
        assert "_correlations" not in result["results"]
