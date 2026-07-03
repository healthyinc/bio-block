"""Tests for on-chain registry."""

import pytest

from app.services import chain_registry


class TestRegisterOnChain:

    @pytest.mark.asyncio
    async def test_returns_none_when_config_missing(self, monkeypatch):
        # Reset module-level state
        monkeypatch.setattr(chain_registry, "_w3", None)
        monkeypatch.setattr(chain_registry, "_contract", None)
        monkeypatch.setattr(chain_registry, "_account", None)
        monkeypatch.setattr(chain_registry, "SEPOLIA_RPC_URL", "")
        monkeypatch.setattr(chain_registry, "ANALYTICS_PRIVATE_KEY", "")
        monkeypatch.setattr(chain_registry, "ANALYTICS_REGISTRY_ADDRESS", "")

        result = await chain_registry.register_on_chain(
            source_cid="QmSource",
            result_cid="QmResult",
            analysis_type="descriptive",
        )
        assert result is None

    @pytest.mark.asyncio
    async def test_returns_tx_hash_on_success(self, monkeypatch):
        monkeypatch.setattr(chain_registry, "_ensure_web3", lambda: True)

        # Mock objects
        class MockAccount:
            address = "0xMockAddress"

        class MockFunctions:
            @staticmethod
            def registerAnalytics(source_cid, result_cid, analysis_type):
                class _Tx:
                    @staticmethod
                    def build_transaction(params):
                        return {"mock": "tx"}
                return _Tx()

        class MockContract:
            functions = MockFunctions()

        class MockEthAccount:
            @staticmethod
            def sign_transaction(tx, key):
                class _Signed:
                    raw_transaction = b"\x00"
                return _Signed()

        class MockEth:
            @staticmethod
            def get_transaction_count(addr, *args):
                return 0

            @staticmethod
            def send_raw_transaction(raw_tx):
                return bytes.fromhex("abcdef1234567890" * 4)

            gas_price = 1000000000

            account = MockEthAccount()

        class MockW3:
            eth = MockEth()

        monkeypatch.setattr(chain_registry, "_w3", MockW3())
        monkeypatch.setattr(chain_registry, "_contract", MockContract())
        monkeypatch.setattr(chain_registry, "_account", MockAccount())
        monkeypatch.setattr(chain_registry, "ANALYTICS_PRIVATE_KEY", "0xfakekey")

        result = await chain_registry.register_on_chain(
            source_cid="QmSource",
            result_cid="QmResult",
            analysis_type="descriptive",
        )
        assert result is not None
        assert isinstance(result, str)


class TestGetAnalyticsForDataset:

    def test_returns_empty_when_config_missing(self, monkeypatch):
        monkeypatch.setattr(chain_registry, "_w3", None)
        monkeypatch.setattr(chain_registry, "_contract", None)
        monkeypatch.setattr(chain_registry, "_account", None)
        monkeypatch.setattr(chain_registry, "SEPOLIA_RPC_URL", "")
        monkeypatch.setattr(chain_registry, "ANALYTICS_PRIVATE_KEY", "")
        monkeypatch.setattr(chain_registry, "ANALYTICS_REGISTRY_ADDRESS", "")

        result = chain_registry.get_analytics_for_dataset("QmSource")
        assert result == []

    def test_returns_cids_on_success(self, monkeypatch):
        monkeypatch.setattr(chain_registry, "_ensure_web3", lambda: True)

        class MockFunctions:
            @staticmethod
            def getAnalyticsForDataset(source_cid):
                class _Call:
                    @staticmethod
                    def call():
                        return ["QmResult1", "QmResult2"]
                return _Call()

        class MockContract:
            functions = MockFunctions()

        monkeypatch.setattr(chain_registry, "_contract", MockContract())

        result = chain_registry.get_analytics_for_dataset("QmSource")
        assert result == ["QmResult1", "QmResult2"]
