import os
import importlib
import time
from unittest import mock

import pytest

# conftest.py sets APP_ENV=test before we get here, so this import is safe.
from app.auth.eip712 import clear_nonces, verify_signature

WALLET = "0x1234567890abcdef1234567890abcdef12345678"


class TestVerifySignature:

    def setup_method(self):
        clear_nonces()

    def test_valid_signature(self):
        assert verify_signature(WALLET, "QmCID", "0xsig", int(time.time()), 1, "hash")

    def test_expired(self):
        assert not verify_signature(WALLET, "QmCID", "0xsig", int(time.time()) - 400, 2, "hash")

    def test_replay_blocked(self):
        ts = int(time.time())
        assert verify_signature(WALLET, "QmCID", "0xsig", ts, 42, "hash")
        assert not verify_signature(WALLET, "QmCID", "0xsig", ts, 42, "hash")

    def test_different_wallets_same_nonce(self):
        ts = int(time.time())
        assert verify_signature("0xAAAA" + "a" * 36, "QmCID", "0xsig", ts, 1, "hash")
        assert verify_signature("0xBBBB" + "b" * 36, "QmCID", "0xsig", ts, 1, "hash")

    def test_case_insensitive(self):
        ts = int(time.time())
        assert verify_signature("0xAbCd" + "e" * 36, "QmCID", "0xsig", ts, 99, "hash")
        # Same wallet different case — replay should be caught
        assert not verify_signature("0xabcd" + "e" * 36, "QmCID", "0xsig", ts, 99, "hash")

    def test_clear_nonces(self):
        ts = int(time.time())
        verify_signature(WALLET, "QmCID", "0xsig", ts, 1, "hash")
        clear_nonces(WALLET)
        assert verify_signature(WALLET, "QmCID", "0xsig", ts, 1, "hash")


class TestProductionGuard:
    """Verify that production mode does NOT silently bypass crypto checks."""

    def test_import_fails_without_web3_in_production(self):
        """Importing eip712 in production without web3.py must raise
        RuntimeError at import time (fail-fast)."""
        with mock.patch.dict(os.environ, {"APP_ENV": "production"}):
            # Simulate web3 not being installed
            with mock.patch.dict("sys.modules", {"web3": None, "eth_account.messages": None}):
                with pytest.raises(RuntimeError, match="web3.py is required"):
                    importlib.reload(importlib.import_module("app.auth.eip712"))

        # Restore the module with test env so other tests aren't broken
        with mock.patch.dict(os.environ, {"APP_ENV": "test"}):
            importlib.reload(importlib.import_module("app.auth.eip712"))

    def test_test_env_skips_crypto(self):
        """In test env, verify_signature should succeed without real crypto."""
        import app.auth.eip712 as mod

        assert mod.APP_ENV in mod._SAFE_ENVS
        clear_nonces()
        # This succeeds because APP_ENV=test skips signer recovery
        assert verify_signature(WALLET, "QmCID", "0xsig", int(time.time()), 777, "hash")

