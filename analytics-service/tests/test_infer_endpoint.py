import io
import time

import numpy as np
import pandas as pd
import pytest
import pytest_asyncio
from httpx import AsyncClient, ASGITransport

from app.auth.eip712 import clear_nonces
from app.auth.rate_limiter import rate_limiter
from app.main import app


@pytest.fixture(autouse=True)
def _clean_state():
    clear_nonces()
    rate_limiter.reset()
    yield
    clear_nonces()
    rate_limiter.reset()


def _auth_form(nonce=1):
    return {
        "wallet_address": "0x1234567890abcdef1234567890abcdef12345678",
        "dataset_cid": "QmTestDatasetCID123456789",
        "signature": "0xdeadbeef",
        "timestamp": str(int(time.time())),
        "nonce": str(nonce),
        "request_hash": "sha256:test_hash",
    }


def _two_group_csv(n1=50, n2=50, g1_mean=100, g2_mean=115,
                    g1_std=10, g2_std=10, seed=42):
    rng = np.random.default_rng(seed)
    scores = np.concatenate([
        rng.normal(g1_mean, g1_std, n1),
        rng.normal(g2_mean, g2_std, n2),
    ])
    groups = ["A"] * n1 + ["B"] * n2
    df = pd.DataFrame({"score": scores, "treatment": groups})
    buf = io.BytesIO()
    df.to_csv(buf, index=False)
    return buf.getvalue()


def _paired_csv(n=50, seed=42):
    rng = np.random.default_rng(seed)
    before = rng.normal(100, 10, n)
    after = before + rng.normal(5, 3, n)
    df = pd.DataFrame({"before": before, "after": after})
    buf = io.BytesIO()
    df.to_csv(buf, index=False)
    return buf.getvalue()


def _one_sample_csv(n=50, mean=110, std=10, seed=42):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({"blood_pressure": rng.normal(mean, std, n)})
    buf = io.BytesIO()
    df.to_csv(buf, index=False)
    return buf.getvalue()


def _chi_square_csv():
    df = pd.DataFrame({
        "treatment": ["Yes", "Yes", "No", "No"] * 25,
        "recovered": ["Yes", "No", "Yes", "No"] * 25
    })
    buf = io.BytesIO()
    df.to_csv(buf, index=False)
    return buf.getvalue()


@pytest_asyncio.fixture
async def client():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


class TestIndependentTTest:
    @pytest.mark.asyncio
    async def test_full_flow(self, client):
        resp = await client.post(
            "/analytics/infer",
            data={
                **_auth_form(nonce=1),
                "test_type": "t_test",
                "test_subtype": "independent",
                "numeric_column": "score",
                "group_column": "treatment",
                "alpha": "0.05",
                "alternative": "two-sided",
            },
            files={"file": ("data.csv", _two_group_csv(), "text/csv")},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["analysis_type"] == "inferential"
        assert body["test_category"] == "t_test"
        assert body["test_subtype"] == "independent"
        assert body["test_used"] in (
            "students_ttest", "welchs_ttest", "mann_whitney_u"
        )
        assert "reason" in body
        assert "assumptions" in body
        assert "result" in body
        assert "effect_size" in body
        assert "group_stats" in body
        assert "interpretation" in body

    @pytest.mark.asyncio
    async def test_response_result_shape(self, client):
        resp = await client.post(
            "/analytics/infer",
            data={
                **_auth_form(nonce=2),
                "test_type": "t_test",
                "test_subtype": "independent",
                "numeric_column": "score",
                "group_column": "treatment",
            },
            files={"file": ("data.csv", _two_group_csv(), "text/csv")},
        )
        body = resp.json()
        result = body["result"]
        assert "statistic" in result
        assert "p_value" in result
        assert "significant" in result
        assert "alpha" in result

    @pytest.mark.asyncio
    async def test_missing_group_column(self, client):
        resp = await client.post(
            "/analytics/infer",
            data={
                **_auth_form(nonce=3),
                "test_type": "t_test",
                "test_subtype": "independent",
                "numeric_column": "score",
            },
            files={"file": ("data.csv", _two_group_csv(), "text/csv")},
        )
        assert resp.status_code == 400
        assert "group_column" in resp.json()["detail"].lower()


class TestPairedTTest:
    @pytest.mark.asyncio
    async def test_full_flow(self, client):
        resp = await client.post(
            "/analytics/infer",
            data={
                **_auth_form(nonce=10),
                "test_type": "t_test",
                "test_subtype": "paired",
                "numeric_column": "before",
                "numeric_column_2": "after",
            },
            files={"file": ("data.csv", _paired_csv(), "text/csv")},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["test_subtype"] == "paired"
        assert body["test_used"] in ("paired_ttest", "wilcoxon_signed_rank")

    @pytest.mark.asyncio
    async def test_missing_second_column(self, client):
        resp = await client.post(
            "/analytics/infer",
            data={
                **_auth_form(nonce=11),
                "test_type": "t_test",
                "test_subtype": "paired",
                "numeric_column": "before",
            },
            files={"file": ("data.csv", _paired_csv(), "text/csv")},
        )
        assert resp.status_code == 400
        assert "numeric_column_2" in resp.json()["detail"].lower()


class TestOneSampleTTest:
    @pytest.mark.asyncio
    async def test_full_flow(self, client):
        resp = await client.post(
            "/analytics/infer",
            data={
                **_auth_form(nonce=20),
                "test_type": "t_test",
                "test_subtype": "one_sample",
                "numeric_column": "blood_pressure",
                "population_mean": "120.0",
            },
            files={
                "file": ("data.csv", _one_sample_csv(), "text/csv"),
            },
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["test_subtype"] == "one_sample"
        assert body["test_used"] in ("one_sample_ttest", "one_sample_wilcoxon")

    @pytest.mark.asyncio
    async def test_missing_population_mean(self, client):
        resp = await client.post(
            "/analytics/infer",
            data={
                **_auth_form(nonce=21),
                "test_type": "t_test",
                "test_subtype": "one_sample",
                "numeric_column": "blood_pressure",
            },
            files={
                "file": ("data.csv", _one_sample_csv(), "text/csv"),
            },
        )
        assert resp.status_code == 400
        assert "population_mean" in resp.json()["detail"].lower()


def _multi_group_csv(k=3, n=50, shift=15, seed=42):
    rng = np.random.default_rng(seed)
    scores = np.concatenate([
        rng.normal(100 + i * shift, 10, n) for i in range(k)
    ])
    groups = []
    for i in range(k):
        groups.extend([f"G{i+1}"] * n)
    df = pd.DataFrame({"score": scores, "treatment": groups})
    buf = io.BytesIO()
    df.to_csv(buf, index=False)
    return buf.getvalue()


def _two_way_csv(n_per_cell=30, seed=42):
    rng = np.random.default_rng(seed)
    rows = []
    for a in ["Low", "High"]:
        for b in ["Control", "Treatment"]:
            base = 100 + (10 if a == "High" else 0) + (5 if b == "Treatment" else 0)
            vals = rng.normal(base, 8, n_per_cell)
            for v in vals:
                rows.append({"score": v, "factor_a": a, "factor_b": b})
    df = pd.DataFrame(rows)
    buf = io.BytesIO()
    df.to_csv(buf, index=False)
    return buf.getvalue()


def _repeated_measures_csv(n_subjects=50, k=3, seed=42):
    rng = np.random.default_rng(seed)
    base = rng.normal(100, 10, n_subjects)
    data = {}
    for i in range(k):
        data[f"cond_{i+1}"] = base + rng.normal(i * 5, 3, n_subjects)
    df = pd.DataFrame(data)
    buf = io.BytesIO()
    df.to_csv(buf, index=False)
    return buf.getvalue()


class TestOneWayAnovaEndpoint:
    @pytest.mark.asyncio
    async def test_full_flow(self, client):
        resp = await client.post(
            "/analytics/infer",
            data={
                **_auth_form(nonce=40),
                "test_type": "anova",
                "test_subtype": "one_way",
                "numeric_column": "score",
                "group_column": "treatment",
                "alpha": "0.05",
            },
            files={"file": ("data.csv", _multi_group_csv(), "text/csv")},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["analysis_type"] == "inferential"
        assert body["test_category"] == "anova"
        assert body["test_subtype"] == "one_way"
        assert body["test_used"] in (
            "one_way_anova", "welchs_anova", "kruskal_wallis"
        )
        assert "assumptions" in body
        assert "result" in body
        assert "effect_size" in body
        assert "interpretation" in body

    @pytest.mark.asyncio
    async def test_missing_group_column(self, client):
        resp = await client.post(
            "/analytics/infer",
            data={
                **_auth_form(nonce=41),
                "test_type": "anova",
                "test_subtype": "one_way",
                "numeric_column": "score",
            },
            files={"file": ("data.csv", _multi_group_csv(), "text/csv")},
        )
        assert resp.status_code == 400
        assert "group_column" in resp.json()["detail"].lower()

    @pytest.mark.asyncio
    async def test_significant_has_post_hoc(self, client):
        resp = await client.post(
            "/analytics/infer",
            data={
                **_auth_form(nonce=42),
                "test_type": "anova",
                "test_subtype": "one_way",
                "numeric_column": "score",
                "group_column": "treatment",
            },
            files={"file": ("data.csv", _multi_group_csv(shift=20), "text/csv")},
        )
        assert resp.status_code == 200
        body = resp.json()
        if body["result"]["significant"]:
            assert body.get("post_hoc") is not None


class TestTwoWayAnovaEndpoint:
    @pytest.mark.asyncio
    async def test_full_flow(self, client):
        resp = await client.post(
            "/analytics/infer",
            data={
                **_auth_form(nonce=50),
                "test_type": "anova",
                "test_subtype": "two_way",
                "numeric_column": "score",
                "group_column": "factor_a",
                "factor_column_2": "factor_b",
            },
            files={"file": ("data.csv", _two_way_csv(), "text/csv")},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["test_used"] == "two_way_anova"
        assert body["test_category"] == "anova"

    @pytest.mark.asyncio
    async def test_missing_factor_column_2(self, client):
        resp = await client.post(
            "/analytics/infer",
            data={
                **_auth_form(nonce=51),
                "test_type": "anova",
                "test_subtype": "two_way",
                "numeric_column": "score",
                "group_column": "factor_a",
            },
            files={"file": ("data.csv", _two_way_csv(), "text/csv")},
        )
        assert resp.status_code == 400
        assert "factor_column_2" in resp.json()["detail"].lower()


class TestRepeatedMeasuresEndpoint:
    @pytest.mark.asyncio
    async def test_full_flow(self, client):
        resp = await client.post(
            "/analytics/infer",
            data={
                **_auth_form(nonce=60),
                "test_type": "anova",
                "test_subtype": "repeated_measures",
                "numeric_column": "cond_1",
                "repeated_columns": "cond_1,cond_2,cond_3",
            },
            files={"file": ("data.csv", _repeated_measures_csv(), "text/csv")},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["test_used"] in ("repeated_measures_anova", "friedman")
        assert body["test_category"] == "anova"

    @pytest.mark.asyncio
    async def test_missing_repeated_columns(self, client):
        resp = await client.post(
            "/analytics/infer",
            data={
                **_auth_form(nonce=61),
                "test_type": "anova",
                "test_subtype": "repeated_measures",
                "numeric_column": "cond_1",
            },
            files={"file": ("data.csv", _repeated_measures_csv(), "text/csv")},
        )
        assert resp.status_code == 400
        assert "repeated_columns" in resp.json()["detail"].lower()

    @pytest.mark.asyncio
    async def test_too_few_repeated_columns(self, client):
        resp = await client.post(
            "/analytics/infer",
            data={
                **_auth_form(nonce=62),
                "test_type": "anova",
                "test_subtype": "repeated_measures",
                "numeric_column": "cond_1",
                "repeated_columns": "cond_1,cond_2",
            },
            files={"file": ("data.csv", _repeated_measures_csv(), "text/csv")},
        )
        assert resp.status_code == 400
        assert "at least 3" in resp.json()["detail"].lower()


class TestValidation:
    @pytest.mark.asyncio
    async def test_anova_one_way_works(self, client):
        """ANOVA is now implemented — should return 200."""
        resp = await client.post(
            "/analytics/infer",
            data={
                **_auth_form(nonce=30),
                "test_type": "anova",
                "test_subtype": "one_way",
                "numeric_column": "score",
                "group_column": "treatment",
            },
            files={"file": ("data.csv", _multi_group_csv(), "text/csv")},
        )
        assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_chi_square_endpoint(self, client):
        resp = await client.post(
            "/analytics/infer",
            data={
                **_auth_form(nonce=31),
                "test_type": "chi_square",
                "test_subtype": "independence",
                "column_1": "treatment",
                "column_2": "recovered",
            },
            files={"file": ("data.csv", _chi_square_csv(), "text/csv")},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["test_used"] == "chi_square_independence"
        assert "contingency_table" in body

    @pytest.mark.asyncio
    async def test_correlation_endpoint(self, client):
        resp = await client.post(
            "/analytics/infer",
            data={
                **_auth_form(nonce=32),
                "test_type": "correlation",
                "test_subtype": "pearson",
                "column_1": "before",
                "column_2": "after",
            },
            files={"file": ("data.csv", _paired_csv(), "text/csv")},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["test_used"] == "pearson"
        assert "confidence_interval" in body

    @pytest.mark.asyncio
    async def test_invalid_subtype(self, client):
        resp = await client.post(
            "/analytics/infer",
            data={
                **_auth_form(nonce=33),
                "test_type": "t_test",
                "test_subtype": "three_way",
                "numeric_column": "score",
            },
            files={"file": ("data.csv", _two_group_csv(), "text/csv")},
        )
        assert resp.status_code == 400
        assert "test_subtype" in resp.json()["detail"].lower()

    @pytest.mark.asyncio
    async def test_invalid_alternative(self, client):
        resp = await client.post(
            "/analytics/infer",
            data={
                **_auth_form(nonce=34),
                "test_type": "t_test",
                "numeric_column": "score",
                "group_column": "treatment",
                "alternative": "left",
            },
            files={"file": ("data.csv", _two_group_csv(), "text/csv")},
        )
        assert resp.status_code == 400
        assert "alternative" in resp.json()["detail"].lower()

    @pytest.mark.asyncio
    async def test_custom_alpha(self, client):
        resp = await client.post(
            "/analytics/infer",
            data={
                **_auth_form(nonce=35),
                "test_type": "t_test",
                "test_subtype": "independent",
                "numeric_column": "score",
                "group_column": "treatment",
                "alpha": "0.01",
            },
            files={"file": ("data.csv", _two_group_csv(), "text/csv")},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["result"]["alpha"] == 0.01

    @pytest.mark.asyncio
    async def test_one_sided_alternative(self, client):
        resp = await client.post(
            "/analytics/infer",
            data={
                **_auth_form(nonce=36),
                "test_type": "t_test",
                "test_subtype": "independent",
                "numeric_column": "score",
                "group_column": "treatment",
                "alternative": "less",
            },
            files={"file": ("data.csv", _two_group_csv(), "text/csv")},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["result"]["alternative"] == "less"

    @pytest.mark.asyncio
    async def test_rate_limited(self, client):
        wallet = "0x1234567890abcdef1234567890abcdef12345678"
        for _ in range(30):
            rate_limiter.check(wallet)

        resp = await client.post(
            "/analytics/infer",
            data={
                **_auth_form(nonce=37),
                "test_type": "t_test",
                "numeric_column": "score",
                "group_column": "treatment",
            },
            files={"file": ("data.csv", _two_group_csv(), "text/csv")},
        )
        assert resp.status_code == 429

