"""Tests for the dataset profiler service."""

import numpy as np
import pandas as pd
import pytest

from app.models.demo_schemas import ColumnType, VariableRole
from app.services.dataset_profiler import profile_dataset




@pytest.fixture
def clinical_df():
    """Simulated clinical trial dataset."""
    np.random.seed(42)
    n = 100
    return pd.DataFrame({
        "patient_id": [f"P{i:03d}" for i in range(n)],
        "treatment": np.random.choice(["Drug", "Placebo"], n),
        "age": np.random.normal(55, 10, n).round(1),
        "blood_pressure": np.random.normal(130, 15, n).round(1),
        "cholesterol": np.random.normal(200, 30, n).round(1),
        "sex": np.random.choice(["M", "F"], n),
        "visit_date": pd.date_range("2024-01-01", periods=n, freq="D"),
    })


@pytest.fixture
def minimal_df():
    """Very small dataset for edge-case testing."""
    return pd.DataFrame({
        "x": [1, 2, 3],
        "group": ["A", "B", "A"],
    })


@pytest.fixture
def missing_df():
    """Dataset with heavy missingness."""
    return pd.DataFrame({
        "complete": [1, 2, 3, 4, 5],
        "half_missing": [1, None, None, None, 5],
        "all_missing": [None, None, None, None, None],
    })




class TestProfileDataset:
    def test_correct_column_count(self, clinical_df):
        profile = profile_dataset(clinical_df)
        assert profile.column_count == 7
        assert profile.row_count == 100
        assert len(profile.columns) == 7

    def test_numeric_columns_have_stats(self, clinical_df):
        profile = profile_dataset(clinical_df)
        age_col = next(c for c in profile.columns if c.name == "age")
        assert age_col.dtype == ColumnType.NUMERIC
        assert age_col.mean is not None
        assert age_col.median is not None
        assert age_col.std is not None
        assert age_col.min_val is not None
        assert age_col.max_val is not None
        assert age_col.skewness is not None

    def test_categorical_columns_detected(self, clinical_df):
        profile = profile_dataset(clinical_df)
        treatment = next(c for c in profile.columns if c.name == "treatment")
        assert treatment.dtype == ColumnType.CATEGORICAL
        assert treatment.top_values is not None
        assert treatment.cardinality == 2

    def test_identifier_detected(self, clinical_df):
        profile = profile_dataset(clinical_df)
        pid = next(c for c in profile.columns if c.name == "patient_id")
        assert pid.dtype == ColumnType.IDENTIFIER

    def test_datetime_detected(self, clinical_df):
        profile = profile_dataset(clinical_df)
        date_col = next(c for c in profile.columns if c.name == "visit_date")
        assert date_col.dtype == ColumnType.DATETIME

    def test_role_suggestions_group(self, clinical_df):
        profile = profile_dataset(clinical_df)
        assert "treatment" in profile.suggested_group_columns
        assert "sex" in profile.suggested_group_columns

    def test_role_suggestions_outcome(self, clinical_df):
        profile = profile_dataset(clinical_df)
        assert "blood_pressure" in profile.suggested_outcome_columns
        assert "age" in profile.suggested_outcome_columns

    def test_role_suggestions_subject(self, clinical_df):
        profile = profile_dataset(clinical_df)
        assert "patient_id" in profile.suggested_subject_columns

    def test_missing_data_warning(self, missing_df):
        profile = profile_dataset(missing_df)
        assert any("half_missing" in w for w in profile.warnings)

    def test_small_dataset_warning(self, minimal_df):
        profile = profile_dataset(minimal_df)
        assert any("small dataset" in w.lower() or "Very small" in w for w in profile.warnings)

    def test_constant_column_warning(self):
        df = pd.DataFrame({"const": [5, 5, 5, 5, 5], "var": [1, 2, 3, 4, 5]})
        profile = profile_dataset(df)
        assert any("constant" in w.lower() for w in profile.warnings)

    def test_duplicate_rows_warning(self):
        df = pd.DataFrame({"a": [1, 1, 2], "b": ["x", "x", "y"]})
        profile = profile_dataset(df)
        assert any("duplicate" in w.lower() for w in profile.warnings)

    def test_normality_hint_present(self, clinical_df):
        profile = profile_dataset(clinical_df)
        bp = next(c for c in profile.columns if c.name == "blood_pressure")
        assert bp.normality_hint is not None
