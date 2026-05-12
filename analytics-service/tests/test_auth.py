import time
import pytest

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
