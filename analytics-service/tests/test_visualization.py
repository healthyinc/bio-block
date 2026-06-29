

import base64

import numpy as np
import pandas as pd
import pytest

from app.services.visualization import (
    VALID_CHART_TYPES,
    generate_chart,
)


@pytest.fixture
def health_df():
    return pd.DataFrame({
        "patient_id": [f"P{i:03d}" for i in range(1, 21)],
        "age": [45, 62, 33, 51, 70, 28, 55, 41, 67, 36, 58, 44, 73, 30, 49, 65, 38, 52, 71, 25],
        "gender": ["M", "F", "M", "F", "M", "F", "M", "F", "M", "F",
                    "M", "F", "M", "F", "M", "F", "M", "F", "M", "F"],
        "glucose_fasting": [92, 145, 88, 126, 165, 85, 138, 95, 152, 90,
                            142, 98, 170, 82, 118, 148, 87, 130, 160, 80],
        "cholesterol_total": [185, 242, 178, 215, 268, 165, 230, 192, 255, 172,
                              238, 188, 275, 160, 205, 248, 175, 220, 262, 155],
        "diagnosis": ["Healthy", "Type 2 Diabetes", "Healthy", "Pre-Diabetes",
                       "Type 2 Diabetes", "Healthy", "Type 2 Diabetes", "Healthy",
                       "Type 2 Diabetes", "Healthy", "Type 2 Diabetes", "Healthy",
                       "Type 2 Diabetes", "Healthy", "Pre-Diabetes", "Type 2 Diabetes",
                       "Healthy", "Pre-Diabetes", "Type 2 Diabetes", "Healthy"],
    })


@pytest.fixture
def simple_df():
    return pd.DataFrame({
        "x": [1, 2, 3, 4, 5],
        "y": [2, 4, 6, 8, 10],
        "category": ["A", "A", "B", "B", "B"],
    })


def _assert_valid_result(result: dict, chart_type: str):
    assert "chart_config" in result
    assert "image" in result
    assert result["chart_config"]["type"] == chart_type
    decoded = base64.b64decode(result["image"])
    assert decoded[:8] == b"\x89PNG\r\n\x1a\n"


class TestValidChartTypes:
    def test_all_seven_types_registered(self):
        assert len(VALID_CHART_TYPES) == 7
        expected = {"histogram", "scatter", "box", "heatmap", "bar", "line", "pie"}
        assert set(VALID_CHART_TYPES) == expected


class TestHistogram:
    def test_basic_histogram(self, health_df):
        result = generate_chart(health_df, "histogram", x_column="age")
        _assert_valid_result(result, "histogram")
        config = result["chart_config"]
        assert len(config["labels"]) > 0
        assert len(config["datasets"]) == 1
        assert config["datasets"][0]["label"] == "age"

    def test_custom_bins(self, health_df):
        result = generate_chart(health_df, "histogram", x_column="age", bins=5)
        config = result["chart_config"]
        assert len(config["labels"]) == 5
        assert config["options"]["bins"] == 5

    def test_missing_column_raises(self, health_df):
        with pytest.raises(ValueError, match="not found"):
            generate_chart(health_df, "histogram", x_column="nonexistent")

    def test_requires_x_column(self, health_df):
        with pytest.raises(ValueError, match="requires"):
            generate_chart(health_df, "histogram")


class TestScatter:
    def test_basic_scatter(self, health_df):
        result = generate_chart(health_df, "scatter", x_column="age", y_column="glucose_fasting")
        _assert_valid_result(result, "scatter")
        config = result["chart_config"]
        assert config["options"]["xLabel"] == "age"
        assert config["options"]["yLabel"] == "glucose_fasting"

    def test_scatter_with_grouping(self, health_df):
        result = generate_chart(
            health_df, "scatter",
            x_column="age", y_column="glucose_fasting",
            group_column="gender"
        )
        _assert_valid_result(result, "scatter")
        assert len(result["chart_config"]["datasets"]) == 2  # M and F

    def test_requires_both_columns(self, health_df):
        with pytest.raises(ValueError, match="requires"):
            generate_chart(health_df, "scatter", x_column="age")

    def test_missing_y_column(self, health_df):
        with pytest.raises(ValueError, match="not found"):
            generate_chart(health_df, "scatter", x_column="age", y_column="nonexistent")


class TestBox:
    def test_multi_column_box(self, health_df):
        result = generate_chart(
            health_df, "box",
            columns=["age", "glucose_fasting", "cholesterol_total"]
        )
        _assert_valid_result(result, "box")
        assert len(result["chart_config"]["datasets"]) == 3

    def test_box_with_grouping(self, health_df):
        result = generate_chart(
            health_df, "box",
            columns=["glucose_fasting"],
            group_column="gender"
        )
        _assert_valid_result(result, "box")
        assert len(result["chart_config"]["datasets"]) == 2  # M and F

    def test_box_stats_values(self, simple_df):
        result = generate_chart(simple_df, "box", columns=["x"])
        ds = result["chart_config"]["datasets"][0]
        assert ds["min"] == 1.0
        assert ds["max"] == 5.0
        assert ds["median"] == 3.0

    def test_requires_columns(self, health_df):
        with pytest.raises(ValueError, match="requires"):
            generate_chart(health_df, "box")


class TestHeatmap:
    def test_correlation_heatmap(self, health_df):
        result = generate_chart(
            health_df, "heatmap",
            columns=["age", "glucose_fasting", "cholesterol_total"]
        )
        _assert_valid_result(result, "heatmap")
        data = result["chart_config"]["data"]
        # Diagonal should be 1.0
        assert data["age"]["age"] == 1.0
        assert data["glucose_fasting"]["glucose_fasting"] == 1.0

    def test_auto_selects_numeric(self, health_df):
        result = generate_chart(health_df, "heatmap")
        _assert_valid_result(result, "heatmap")
        # Should only include numeric columns
        labels = result["chart_config"]["labels"]
        assert "patient_id" not in labels
        assert "diagnosis" not in labels

    def test_too_few_columns_raises(self):
        df = pd.DataFrame({"x": [1, 2, 3]})
        with pytest.raises(ValueError, match="at least 2"):
            generate_chart(df, "heatmap", columns=["x"])


class TestBar:
    def test_count_bar(self, health_df):
        result = generate_chart(health_df, "bar", x_column="diagnosis")
        _assert_valid_result(result, "bar")
        config = result["chart_config"]
        assert len(config["labels"]) > 0
        assert sum(config["datasets"][0]["data"]) == 20  # Total rows

    def test_mean_aggregation(self, health_df):
        result = generate_chart(
            health_df, "bar",
            x_column="gender", y_column="glucose_fasting",
            aggregation="mean"
        )
        _assert_valid_result(result, "bar")
        assert result["chart_config"]["options"]["aggregation"] == "mean"

    def test_sum_aggregation(self, simple_df):
        result = generate_chart(
            simple_df, "bar",
            x_column="category", y_column="y",
            aggregation="sum"
        )
        _assert_valid_result(result, "bar")

    def test_requires_x_column(self, health_df):
        with pytest.raises(ValueError, match="requires"):
            generate_chart(health_df, "bar")


class TestLine:
    def test_numeric_line(self, simple_df):
        result = generate_chart(simple_df, "line", x_column="x", y_column="y")
        _assert_valid_result(result, "line")
        datasets = result["chart_config"]["datasets"]
        assert len(datasets) == 1
        assert datasets[0]["label"] == "y"

    def test_categorical_x_line(self, health_df):
        result = generate_chart(
            health_df, "line",
            x_column="diagnosis", y_column="glucose_fasting",
            aggregation="mean"
        )
        _assert_valid_result(result, "line")

    def test_requires_both_columns(self, health_df):
        with pytest.raises(ValueError, match="requires"):
            generate_chart(health_df, "line", x_column="age")


class TestPie:
    def test_basic_pie(self, health_df):
        result = generate_chart(health_df, "pie", x_column="diagnosis")
        _assert_valid_result(result, "pie")
        config = result["chart_config"]
        assert sum(config["datasets"][0]["data"]) == 20

    def test_pie_gender(self, health_df):
        result = generate_chart(health_df, "pie", x_column="gender")
        _assert_valid_result(result, "pie")
        assert len(result["chart_config"]["labels"]) == 2

    def test_requires_column(self, health_df):
        with pytest.raises(ValueError, match="requires"):
            generate_chart(health_df, "pie")


class TestInvalidChartType:
    def test_invalid_type_raises(self, simple_df):
        with pytest.raises(ValueError, match="Unsupported chart type"):
            generate_chart(simple_df, "radar", x_column="x")
