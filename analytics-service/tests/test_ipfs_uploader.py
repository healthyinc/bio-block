"""Tests for IPFS uploader."""

import pytest
import httpx

from app.services.ipfs_uploader import upload_result_to_ipfs


@pytest.fixture
def sample_result():
    return {
        "version": "1.0",
        "analysis_type": "descriptive",
        "source_dataset_cid": "QmTestSource",
        "results": {"age": {"mean": 35.0}},
    }


class TestUploadResultToIpfs:

    @pytest.mark.asyncio
    async def test_successful_upload(self, sample_result, monkeypatch):

        async def _mock_post(self_client, url, **kwargs):
            class MockResp:
                status_code = 200

                def raise_for_status(self):
                    pass

                def json(self):
                    return {
                        "success": True,
                        "ipfsHash": "QmMockResult123",
                        "fileName": "analytics-descriptive-QmTestSourc.json",
                        "fileSize": 256,
                    }

            return MockResp()

        monkeypatch.setattr(httpx.AsyncClient, "post", _mock_post)

        cid = await upload_result_to_ipfs(
            result_data=sample_result,
            analysis_type="descriptive",
            source_cid="QmTestSource",
        )
        assert cid == "QmMockResult123"

    @pytest.mark.asyncio
    async def test_connection_error_returns_none(self, sample_result, monkeypatch):

        async def _mock_post(self_client, url, **kwargs):
            raise httpx.ConnectError("Connection refused")

        monkeypatch.setattr(httpx.AsyncClient, "post", _mock_post)

        cid = await upload_result_to_ipfs(
            result_data=sample_result,
            analysis_type="descriptive",
            source_cid="QmTestSource",
        )
        assert cid is None

    @pytest.mark.asyncio
    async def test_http_error_returns_none(self, sample_result, monkeypatch):

        async def _mock_post(self_client, url, **kwargs):
            class MockResp:
                status_code = 500
                text = "Internal Server Error"

                def raise_for_status(self):
                    raise httpx.HTTPStatusError(
                        "Server Error",
                        request=httpx.Request("POST", url),
                        response=self,
                    )

            return MockResp()

        monkeypatch.setattr(httpx.AsyncClient, "post", _mock_post)

        cid = await upload_result_to_ipfs(
            result_data=sample_result,
            analysis_type="descriptive",
            source_cid="QmTestSource",
        )
        assert cid is None

    @pytest.mark.asyncio
    async def test_missing_ipfs_hash_returns_none(self, sample_result, monkeypatch):

        async def _mock_post(self_client, url, **kwargs):
            class MockResp:
                status_code = 200

                def raise_for_status(self):
                    pass

                def json(self):
                    return {"success": True, "fileName": "test.json"}

            return MockResp()

        monkeypatch.setattr(httpx.AsyncClient, "post", _mock_post)

        cid = await upload_result_to_ipfs(
            result_data=sample_result,
            analysis_type="descriptive",
            source_cid="QmTestSource",
        )
        assert cid is None
