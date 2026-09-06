"""Tests for result serializer."""

from datetime import datetime

from app.services.result_serializer import serialize_analytics_result


class TestSerializeAnalyticsResult:

    def test_has_all_required_fields(self):
        doc = serialize_analytics_result(
            analysis_type="descriptive",
            source_cid="QmTestSource123",
            wallet_address="0xabc123",
            results={"age": {"mean": 35.0}},
            row_count=100,
            columns=["age"],
        )
        assert doc["version"] == "1.0"
        assert doc["analysis_type"] == "descriptive"
        assert doc["source_dataset_cid"] == "QmTestSource123"
        assert doc["analyst_wallet"] == "0xabc123"
        assert "timestamp" in doc
        assert doc["parameters"] == {}
        assert doc["results"] == {"age": {"mean": 35.0}}
        assert doc["metadata"]["api_version"] == "0.1.0"
        assert doc["metadata"]["row_count"] == 100
        assert doc["metadata"]["columns_analyzed"] == ["age"]

    def test_timestamp_is_valid_iso(self):
        doc = serialize_analytics_result(
            analysis_type="graphical",
            source_cid="QmTest",
            wallet_address="0x0",
            results={},
            row_count=0,
            columns=[],
        )
        # Should parse without error
        datetime.fromisoformat(doc["timestamp"])

    def test_graphical_type(self):
        doc = serialize_analytics_result(
            analysis_type="graphical",
            source_cid="QmChart",
            wallet_address="0xdef456",
            results={"chart_config": {"type": "histogram"}},
            row_count=50,
            columns=["age", "glucose"],
            parameters={"chart_type": "histogram", "bins": 20},
        )
        assert doc["analysis_type"] == "graphical"
        assert doc["parameters"]["chart_type"] == "histogram"
        assert doc["results"]["chart_config"]["type"] == "histogram"

    def test_inferential_type(self):
        doc = serialize_analytics_result(
            analysis_type="inferential",
            source_cid="QmInfer",
            wallet_address="0xfeed",
            results={"test": "t-test", "p_value": 0.03},
            row_count=200,
            columns=["treatment", "outcome"],
            parameters={"test_type": "independent_t_test", "alpha": 0.05},
        )
        assert doc["analysis_type"] == "inferential"
        assert doc["parameters"]["test_type"] == "independent_t_test"

    def test_wallet_address_included(self):
        doc = serialize_analytics_result(
            analysis_type="descriptive",
            source_cid="QmTest",
            wallet_address="0x1234567890abcdef1234567890abcdef12345678",
            results={},
            row_count=10,
            columns=["a"],
        )
        assert doc["analyst_wallet"] == "0x1234567890abcdef1234567890abcdef12345678"

    def test_parameters_optional_default_empty(self):
        doc = serialize_analytics_result(
            analysis_type="descriptive",
            source_cid="QmTest",
            wallet_address="0x0",
            results={},
            row_count=0,
            columns=[],
        )
        assert doc["parameters"] == {}
