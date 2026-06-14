import os

# Must be set BEFORE importing app modules so eip712's import-time
# guard allows loading without web3.py.
os.environ["APP_ENV"] = "test"

import time

import pytest
import pandas as pd
from httpx import AsyncClient, ASGITransport

from app.main import app


@pytest.fixture
def sample_csv_bytes():
    return b"age,glucose,cholesterol\n25,90,180\n30,110,200\n45,95,220\n60,140,250\n35,88,190\n"


@pytest.fixture
def sample_dataframe():
    return pd.DataFrame({
        "age": [25, 30, 45, 60, 35],
        "glucose": [90, 110, 95, 140, 88],
        "cholesterol": [180, 200, 220, 250, 190],
    })


@pytest.fixture
def sample_csv_with_missing():
    return b"age,glucose,cholesterol\n25,90,180\n30,,200\n45,95,\n60,140,250\n,88,190\n"


@pytest.fixture
def auth_form_data():
    return {
        "wallet_address": "0x1234567890abcdef1234567890abcdef12345678",
        "dataset_cid": "QmTestDatasetCID123456789",
        "signature": "0xabc123",
        "timestamp": str(int(time.time())),
        "nonce": "1",
        "request_hash": "sha256:test_hash",
    }


@pytest.fixture
async def async_client():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        yield client
