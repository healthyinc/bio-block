import numpy as np
import pandas as pd
import pytest

from app.services.inferential import (
    run_chi_square_independence,
    run_chi_square_goodness_of_fit,
)


def _chi_square_independence_df(n_per_cell=50, seed=42):
    rng = np.random.default_rng(seed)
    # Generate data with association
    # Factor A: Treatment (Yes/No), Factor B: Recovery (Yes/No)
    data = []
    # Yes Treatment -> 80% Recovery
    for _ in range(n_per_cell):
        recovery = "Yes" if rng.random() < 0.8 else "No"
        data.append({"treatment": "Yes", "recovered": recovery})
    # No Treatment -> 30% Recovery
    for _ in range(n_per_cell):
        recovery = "Yes" if rng.random() < 0.3 else "No"
        data.append({"treatment": "No", "recovered": recovery})
    return pd.DataFrame(data)


def _goodness_of_fit_df(n=100, p=[0.5, 0.3, 0.2], seed=42):
    rng = np.random.default_rng(seed)
    categories = rng.choice(["A", "B", "C"], size=n, p=p)
    return pd.DataFrame({"category": categories})


class TestChiSquareIndependence:
    def test_independence_significant(self):
        df = _chi_square_independence_df(n_per_cell=50)
        result = run_chi_square_independence(df, "treatment", "recovered")
        assert result["test_used"] == "chi_square_independence"
        assert result["result"]["significant"] is True
        assert "contingency_table" in result
        assert result["effect_size"]["metric"] == "cramers_v"
        assert result["effect_size"]["value"] > 0.3

    def test_independence_not_significant(self):
        # Generate independent columns
        rng = np.random.default_rng(42)
        treatment = rng.choice(["Yes", "No"], size=100)
        recovered = rng.choice(["Yes", "No"], size=100)
        df = pd.DataFrame({"treatment": treatment, "recovered": recovered})
        result = run_chi_square_independence(df, "treatment", "recovered")
        assert result["result"]["significant"] is False

    def test_fisher_exact_fallback(self):
        # 2x2 with extremely small numbers to trigger Fisher's Exact
        df = pd.DataFrame([
            {"treatment": "Yes", "recovered": "Yes"},
            {"treatment": "Yes", "recovered": "No"},
            {"treatment": "No", "recovered": "Yes"},
            {"treatment": "No", "recovered": "No"},
            {"treatment": "Yes", "recovered": "Yes"},
            {"treatment": "No", "recovered": "No"},
        ])
        result = run_chi_square_independence(df, "treatment", "recovered")
        assert result["test_used"] == "fisher_exact"
        assert "odds_ratio" in result["reason"].lower() or "fisher" in result["test_used"]
        assert result["assumptions"]["fisher_fallback"] is True

    def test_sdc_suppression(self):
        # Sample size < 5
        df = pd.DataFrame([
            {"treatment": "Yes", "recovered": "Yes"},
            {"treatment": "Yes", "recovered": "No"},
            {"treatment": "No", "recovered": "Yes"},
        ])
        result = run_chi_square_independence(df, "treatment", "recovered")
        assert result["test_used"] == "suppressed"

    def test_column_not_found(self):
        df = pd.DataFrame({"A": [1, 2], "B": [3, 4]})
        with pytest.raises(ValueError, match="not found"):
            run_chi_square_independence(df, "A", "C")


class TestChiSquareGoodnessOfFit:
    def test_goodness_of_fit_uniform(self):
        df = _goodness_of_fit_df(n=150, p=[0.33, 0.33, 0.34])
        result = run_chi_square_goodness_of_fit(df, "category")
        assert result["test_used"] == "chi_square_goodness_of_fit"
        # Since it's uniform, it should not be significant
        assert result["result"]["significant"] is False

    def test_goodness_of_fit_non_uniform(self):
        df = _goodness_of_fit_df(n=150, p=[0.7, 0.2, 0.1])
        result = run_chi_square_goodness_of_fit(df, "category")
        assert result["result"]["significant"] is True

    def test_goodness_of_fit_specified_proportions(self):
        df = _goodness_of_fit_df(n=200, p=[0.6, 0.3, 0.1])
        expected = {"A": 0.6, "B": 0.3, "C": 0.1}
        result = run_chi_square_goodness_of_fit(df, "category", expected_proportions=expected)
        # Should fit the specified proportions, hence not significant difference
        assert result["result"]["significant"] is False
