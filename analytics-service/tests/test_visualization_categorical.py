"""Tests for categorical column handling, datetime support, and Excel parsing.

These tests cover the new features added to the visualization engine:
- Categorical (qualitative) column support in histogram, scatter, box, heatmap
- Datetime/time-series column support in histogram, scatter, line
- Excel (.xlsx) file parsing in csv_parser
- Input sanitisation of Swagger UI placeholder values
"""

import io
import pytest
import numpy as np
import pandas as pd

from app.services.visualization import (
    _is_categorical,
    _is_datetime,
    _coerce_datetime,
    _cramers_v,
    _encode_categorical_axis,
    generate_chart,
)
from app.utils.csv_parser import parse_csv, _is_excel


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def categorical_df():
    """DataFrame with both numeric and categorical columns."""
    return pd.DataFrame({
        "age": [45, 62, 33, 51, 70, 28, 55, 41, 67, 36],
        "weight": [150, 180, 140, 165, 200, 130, 175, 155, 190, 135],
        "gender": ["M", "F", "M", "F", "M", "F", "M", "F", "M", "F"],
        "diagnosis": [
            "Healthy", "Type2", "Healthy", "PreDiab",
            "Type2", "Healthy", "Type2", "Healthy",
            "Type2", "Healthy",
        ],
    })


@pytest.fixture
def datetime_df():
    """DataFrame with a pre-parsed datetime column."""
    return pd.DataFrame({
        "date": pd.to_datetime([
            "2024-11-30", "2025-01-10", "2025-02-15",
            "2025-03-01", "2024-12-01", "2025-04-15",
        ]),
        "temperature": [98.1, 99.5, 97.8, 100.2, 98.6, 99.0],
    })


@pytest.fixture
def string_dates_df():
    """DataFrame with dates stored as strings (US format)."""
    return pd.DataFrame({
        "date": ["12/01/2024", "01/10/2025", "03/01/2025",
                 "11/30/2024", "02/15/2025"],
        "value": [50, 10, 30, 45, 20],
    })


# ---------------------------------------------------------------------------
# Type Detection Tests
# ---------------------------------------------------------------------------

class TestTypeDetection:
    """Tests for _is_categorical, _is_datetime, _coerce_datetime helpers."""

    def test_numeric_is_not_categorical(self):
        s = pd.Series([1, 2, 3, 4, 5])
        assert not _is_categorical(s)
        assert not _is_datetime(s)

    def test_float_is_not_categorical(self):
        s = pd.Series([1.1, 2.2, 3.3])
        assert not _is_categorical(s)

    def test_string_is_categorical(self):
        s = pd.Series(["A", "B", "C"])
        assert _is_categorical(s)
        assert not _is_datetime(s)

    def test_datetime_is_not_categorical(self):
        s = pd.Series(pd.to_datetime(["2025-01-01", "2025-02-01"]))
        assert not _is_categorical(s)
        assert _is_datetime(s)

    def test_datetime_detected(self):
        s = pd.Series(pd.to_datetime(["2025-01-01"]))
        assert _is_datetime(s)

    def test_coerce_datetime_from_string(self):
        s = pd.Series(["2025-01-01", "2025-02-01", "2025-03-01"])
        result = _coerce_datetime(s)
        assert _is_datetime(result)

    def test_coerce_datetime_preserves_existing(self):
        s = pd.Series(pd.to_datetime(["2025-01-01"]))
        result = _coerce_datetime(s)
        assert _is_datetime(result)

    def test_coerce_datetime_returns_original_on_failure(self):
        s = pd.Series(["hello", "world", "foo"])
        result = _coerce_datetime(s)
        assert result.dtype == object  # unchanged

    def test_coerce_datetime_threshold(self):
        """Only coerce if >50% of values parse as dates."""
        s = pd.Series(["2025-01-01", "not-a-date", "also-not"])
        result = _coerce_datetime(s)
        # Only 1/3 parses = 33%, below threshold
        assert result.dtype == object


# ---------------------------------------------------------------------------
# Categorical Column Encoding
# ---------------------------------------------------------------------------

class TestCategoricalEncoding:
    """Tests for _encode_categorical_axis."""

    def test_basic_encoding(self):
        s = pd.Series(["A", "B", "C", "A", "B"])
        codes, labels, cat_map = _encode_categorical_axis(s)
        assert len(labels) == 3
        assert set(labels) == {"A", "B", "C"}
        assert codes.notna().all()

    def test_encoding_preserves_mapping(self):
        s = pd.Series(["X", "Y", "X"])
        codes, labels, cat_map = _encode_categorical_axis(s)
        # Same category should get the same code
        assert codes.iloc[0] == codes.iloc[2]
        assert codes.iloc[0] != codes.iloc[1]


# ---------------------------------------------------------------------------
# Cramér's V
# ---------------------------------------------------------------------------

class TestCramersV:
    """Tests for the Cramér's V association statistic."""

    def test_perfect_association(self):
        """Identical columns should have V = 1.0."""
        x = pd.Series(["A", "B", "C"] * 20)
        v = _cramers_v(x, x)
        assert v == pytest.approx(1.0, abs=0.01)

    def test_v_in_valid_range(self):
        """Cramér's V must be in [0, 1]."""
        x = pd.Series(["A", "B"] * 50)
        y = pd.Series(["X", "Y", "X", "Y"] * 25)
        v = _cramers_v(x, y)
        assert 0 <= v <= 1

    def test_independent_variables_low_v(self):
        """Independent variables should have V close to 0."""
        np.random.seed(42)
        x = pd.Series(np.random.choice(["A", "B", "C"], 200))
        y = pd.Series(np.random.choice(["X", "Y", "Z"], 200))
        v = _cramers_v(x, y)
        assert v < 0.2  # Should be close to 0 for independent vars


# ---------------------------------------------------------------------------
# Categorical Histogram (Frequency Bar Chart)
# ---------------------------------------------------------------------------

class TestCategoricalHistogram:
    """Tests for histogram with qualitative columns."""

    def test_categorical_histogram_succeeds(self, categorical_df):
        result = generate_chart(categorical_df, "histogram", x_column="gender")
        assert result["chart_config"]["type"] == "histogram"
        assert result["chart_config"]["options"].get("categorical") is True
        assert "image" in result

    def test_categorical_histogram_labels(self, categorical_df):
        result = generate_chart(categorical_df, "histogram", x_column="diagnosis")
        labels = result["chart_config"]["labels"]
        data = result["chart_config"]["datasets"][0]["data"]
        # Should have one label per unique value
        assert len(labels) == categorical_df["diagnosis"].nunique()
        assert len(data) == len(labels)
        # Frequencies should sum to total rows
        assert sum(data) == len(categorical_df)

    def test_numeric_histogram_unchanged(self, categorical_df):
        """Numeric columns should still produce standard histograms."""
        result = generate_chart(categorical_df, "histogram", x_column="age")
        assert result["chart_config"]["options"].get("categorical") is None


# ---------------------------------------------------------------------------
# Categorical Scatter Plot
# ---------------------------------------------------------------------------

class TestCategoricalScatter:
    """Tests for scatter plot with qualitative columns."""

    def test_two_categoricals(self, categorical_df):
        result = generate_chart(
            categorical_df, "scatter",
            x_column="gender", y_column="diagnosis",
        )
        assert result["chart_config"]["type"] == "scatter"
        opts = result["chart_config"]["options"]
        assert "xCategories" in opts
        assert "yCategories" in opts

    def test_categorical_x_numeric_y(self, categorical_df):
        result = generate_chart(
            categorical_df, "scatter",
            x_column="diagnosis", y_column="weight",
        )
        opts = result["chart_config"]["options"]
        assert "xCategories" in opts
        assert "yCategories" not in opts

    def test_numeric_scatter_unchanged(self, categorical_df):
        """Numeric-only scatter should not have category metadata."""
        result = generate_chart(
            categorical_df, "scatter",
            x_column="age", y_column="weight",
        )
        opts = result["chart_config"]["options"]
        assert "xCategories" not in opts
        assert "yCategories" not in opts


# ---------------------------------------------------------------------------
# Categorical Box Plot
# ---------------------------------------------------------------------------

class TestCategoricalBox:
    """Tests for box plot with qualitative columns."""

    def test_all_categorical_raises_clear_error(self, categorical_df):
        with pytest.raises(ValueError, match="categorical"):
            generate_chart(
                categorical_df, "box",
                columns=["gender", "diagnosis"],
            )

    def test_single_categorical_raises_clear_error(self, categorical_df):
        with pytest.raises(ValueError, match="categorical"):
            generate_chart(
                categorical_df, "box",
                columns=["diagnosis"],
            )

    def test_mixed_skips_categorical(self, categorical_df):
        """Mixed columns: numeric should work, categorical should be skipped."""
        result = generate_chart(
            categorical_df, "box",
            columns=["age", "gender"],
        )
        config = result["chart_config"]
        # Only the numeric column should appear
        assert "age" in config["labels"]
        assert "gender" not in config["labels"]
        assert config.get("skipped_categorical") == ["gender"]

    def test_categorical_with_group_column(self, categorical_df):
        """Box plot with group_column should work for numeric value column."""
        result = generate_chart(
            categorical_df, "box",
            columns=["weight"],
            group_column="gender",
        )
        config = result["chart_config"]
        assert len(config["datasets"]) == 2  # M and F


# ---------------------------------------------------------------------------
# Categorical Heatmap (Cramér's V)
# ---------------------------------------------------------------------------

class TestCategoricalHeatmap:
    """Tests for heatmap with qualitative columns (Cramér's V)."""

    def test_categorical_heatmap_uses_cramers_v(self, categorical_df):
        result = generate_chart(
            categorical_df, "heatmap",
            columns=["gender", "diagnosis"],
        )
        config = result["chart_config"]
        assert config["options"]["method"] == "cramers_v"
        assert config["options"].get("categorical") is True
        assert "image" in result

    def test_categorical_heatmap_values_in_range(self, categorical_df):
        result = generate_chart(
            categorical_df, "heatmap",
            columns=["gender", "diagnosis"],
        )
        data = result["chart_config"]["data"]
        # All values in Cramér's V matrix should be in [0, 1]
        for col_name, col_vals in data.items():
            for row_name, val in col_vals.items():
                assert 0 <= val <= 1, f"V({col_name},{row_name})={val} out of range"

    def test_numeric_heatmap_unchanged(self, categorical_df):
        """Numeric-only heatmap should still use Pearson correlation."""
        result = generate_chart(
            categorical_df, "heatmap",
            columns=["age", "weight"],
        )
        assert result["chart_config"]["options"]["method"] == "pearson"


# ---------------------------------------------------------------------------
# Datetime Histogram
# ---------------------------------------------------------------------------

class TestDatetimeHistogram:
    """Tests for histogram with datetime columns."""

    def test_datetime_histogram_succeeds(self, datetime_df):
        result = generate_chart(datetime_df, "histogram", x_column="date")
        assert result["chart_config"]["type"] == "histogram"
        assert result["chart_config"]["options"].get("datetime") is True
        assert "image" in result


# ---------------------------------------------------------------------------
# Datetime Scatter Plot
# ---------------------------------------------------------------------------

class TestDatetimeScatter:
    """Tests for scatter plot with datetime columns."""

    def test_datetime_x_axis(self, datetime_df):
        result = generate_chart(
            datetime_df, "scatter",
            x_column="date", y_column="temperature",
        )
        opts = result["chart_config"]["options"]
        assert opts.get("xDatetime") is True
        # Should NOT have category encoding
        assert "xCategories" not in opts

    def test_datetime_not_jittered(self, datetime_df):
        """Datetime axes should not receive jitter (no xCategories)."""
        result = generate_chart(
            datetime_df, "scatter",
            x_column="date", y_column="temperature",
        )
        assert "xCategories" not in result["chart_config"]["options"]


# ---------------------------------------------------------------------------
# Datetime Line Chart (Time Series)
# ---------------------------------------------------------------------------

class TestDatetimeLineChart:
    """Tests for line chart with datetime and string-date columns."""

    def test_datetime_line_chart(self, datetime_df):
        result = generate_chart(
            datetime_df, "line",
            x_column="date", y_column="temperature",
        )
        opts = result["chart_config"]["options"]
        assert opts.get("datetime") is True
        assert "image" in result

    def test_string_dates_coerced_and_sorted(self, string_dates_df):
        """String dates should be auto-parsed and sorted chronologically."""
        result = generate_chart(
            string_dates_df, "line",
            x_column="date", y_column="value",
        )
        opts = result["chart_config"]["options"]
        assert opts.get("datetime") is True

        # Check chronological order: first label should be the earliest date
        labels = result["chart_config"]["labels"]
        assert labels is not None
        # 11/30/2024 is earliest, should come first
        assert "2024-11-30" in labels[0]

    def test_us_format_dates_not_alphabetical(self, string_dates_df):
        """US-format dates should NOT be sorted alphabetically."""
        result = generate_chart(
            string_dates_df, "line",
            x_column="date", y_column="value",
        )
        labels = result["chart_config"]["labels"]
        # Alphabetical would put 01/10/2025 first; chronological puts 2024-11-30 first
        assert "2024" in labels[0]


# ---------------------------------------------------------------------------
# Excel File Detection
# ---------------------------------------------------------------------------

class TestExcelParsing:
    """Tests for Excel file detection and parsing."""

    def test_xlsx_magic_bytes_detected(self):
        """XLSX files start with PK (ZIP format)."""
        assert _is_excel(b"PK\x03\x04rest-of-file")

    def test_xls_magic_bytes_detected(self):
        """XLS files start with OLE2 compound document signature."""
        assert _is_excel(b"\xd0\xcf\x11\xe0rest-of-file")

    def test_csv_not_detected_as_excel(self):
        assert not _is_excel(b"col1,col2\n1,2\n3,4")

    def test_empty_bytes_not_excel(self):
        assert not _is_excel(b"")

    def test_parse_csv_still_works(self):
        """Regular CSV parsing should still work after Excel support was added."""
        csv_bytes = b"name,age\nAlice,30\nBob,25\n"
        df = parse_csv(csv_bytes)
        assert list(df.columns) == ["name", "age"]
        assert len(df) == 2

    def test_parse_csv_strips_whitespace(self):
        """Column names with whitespace should be stripped."""
        csv_bytes = b" name , age \nAlice,30\n"
        df = parse_csv(csv_bytes)
        assert list(df.columns) == ["name", "age"]

    def test_xlsx_roundtrip(self, tmp_path):
        """Create a real .xlsx file and verify parse_csv handles it."""
        df_original = pd.DataFrame({
            "Patient": [1, 2, 3],
            "Blood Pressure Systolic": [120, 140, 130],
            "Sex (m/f)": ["Male", "Female", "Male"],
        })
        xlsx_path = tmp_path / "test.xlsx"
        df_original.to_excel(xlsx_path, index=False)
        content = xlsx_path.read_bytes()

        df_parsed = parse_csv(content)
        assert "Blood Pressure Systolic" in df_parsed.columns
        assert "Sex (m/f)" in df_parsed.columns
        assert len(df_parsed) == 3
        assert list(df_parsed["Blood Pressure Systolic"]) == [120, 140, 130]


# ---------------------------------------------------------------------------
# Input Sanitisation (Swagger "string" placeholder)
# ---------------------------------------------------------------------------

class TestInputSanitisation:
    """Tests for the _clean function behaviour in the API endpoint.

    We can't easily unit-test the FastAPI route's inner function, so we test
    the behaviour through generate_chart with realistic scenarios.
    """

    def test_box_chart_ignores_none_group(self, categorical_df):
        """group_column=None should not crash box plot."""
        result = generate_chart(
            categorical_df, "box",
            columns=["age", "weight"],
            group_column=None,
        )
        assert result["chart_config"]["type"] == "box"

    def test_heatmap_with_no_explicit_columns(self, categorical_df):
        """columns=None should auto-select numeric columns for heatmap."""
        result = generate_chart(
            categorical_df, "heatmap",
        )
        config = result["chart_config"]
        # Should auto-detect age and weight as numeric
        assert "age" in config["labels"] or "weight" in config["labels"]
