"""Tests for the EIP-712 auth FastAPI dependency."""

import time

import pytest
import pytest_asyncio
from httpx import AsyncClient, ASGITransport

from app.auth.eip712 import clear_nonces
from app.auth.rate_limiter import rate_limiter
from app.main import app


@pytest.fixture(autouse=True)
def _clean_state():
    """Reset auth state before each test."""
    clear_nonces()
    rate_limiter.reset()
    yield
    clear_nonces()
    rate_limiter.reset()


@pytest.fixture
def valid_form_data():
    return {
        "wallet_address": "0x1234567890abcdef1234567890abcdef12345678",
        "dataset_cid": "QmTestCID",
        "signature": "0xdeadbeef",
        "timestamp": str(int(time.time())),
        "nonce": "1",
        "request_hash": "sha256:abc",
    }


@pytest.fixture
def sample_csv():
    return b"age,glucose\n25,90\n30,110\n"


@pytest_asyncio.fixture
async def client():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


class TestAuthDependency:
    """Integration tests exercising require_eip712_auth through the /analytics/describe endpoint."""

    @pytest.mark.asyncio
    async def test_valid_auth(self, client, valid_form_data, sample_csv):
        resp = await client.post(
            "/analytics/describe",
            data=valid_form_data,
            files={"file": ("data.csv", sample_csv, "text/csv")},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["source_dataset_cid"] == "QmTestCID"
        assert body["analysis_type"] == "descriptive"

    @pytest.mark.asyncio
    async def test_expired_timestamp_returns_401(self, client, valid_form_data, sample_csv):
        valid_form_data["timestamp"] = str(int(time.time()) - 600)
        resp = await client.post(
            "/analytics/describe",
            data=valid_form_data,
            files={"file": ("data.csv", sample_csv, "text/csv")},
        )
        assert resp.status_code == 401

    @pytest.mark.asyncio
    async def test_replay_nonce_returns_401(self, client, valid_form_data, sample_csv):
        # First request succeeds
        resp1 = await client.post(
            "/analytics/describe",
            data=valid_form_data,
            files={"file": ("data.csv", sample_csv, "text/csv")},
        )
        assert resp1.status_code == 200

        # Same nonce should be rejected
        valid_form_data["timestamp"] = str(int(time.time()))
        resp2 = await client.post(
            "/analytics/describe",
            data=valid_form_data,
            files={"file": ("data.csv", sample_csv, "text/csv")},
        )
        assert resp2.status_code == 401

    @pytest.mark.asyncio
    async def test_missing_wallet_returns_422(self, client, sample_csv):
        resp = await client.post(
            "/analytics/describe",
            data={
                "dataset_cid": "QmCID",
                "signature": "0xsig",
                "timestamp": str(int(time.time())),
                "nonce": "1",
                "request_hash": "hash",
            },
            files={"file": ("data.csv", sample_csv, "text/csv")},
        )
        assert resp.status_code == 422  # validation error


class TestRateLimitIntegration:
    """Tests that the rate limiter triggers 429 through the API."""

    @pytest.mark.asyncio
    async def test_rate_limit_returns_429(self, client, sample_csv):
        """Exhaust the rate limit then verify 429."""
        # Use a tight limiter for testing
        rate_limiter.max_requests = 3
        rate_limiter.window_seconds = 60

        wallet = "0xAAAA" + "a" * 36
        for i in range(3):
            data = {
                "wallet_address": wallet,
                "dataset_cid": "QmCID",
                "signature": "0xsig",
                "timestamp": str(int(time.time())),
                "nonce": str(i + 100),
                "request_hash": "hash",
            }
            resp = await client.post(
                "/analytics/describe",
                data=data,
                files={"file": ("data.csv", sample_csv, "text/csv")},
            )
            assert resp.status_code == 200

        # 4th request should be rate-limited
        data["nonce"] = "999"
        data["timestamp"] = str(int(time.time()))
        resp = await client.post(
            "/analytics/describe",
            data=data,
            files={"file": ("data.csv", sample_csv, "text/csv")},
        )
        assert resp.status_code == 429
