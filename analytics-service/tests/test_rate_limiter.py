"""Tests for the per-wallet sliding-window rate limiter."""

import time

import pytest

from app.auth.rate_limiter import WalletRateLimiter


WALLET_A = "0xAAAA" + "a" * 36
WALLET_B = "0xBBBB" + "b" * 36


class TestWalletRateLimiter:

    def setup_method(self):
        self.limiter = WalletRateLimiter(max_requests=5, window_seconds=10)

    # ---------- basic checks ----------

    def test_allows_under_limit(self):
        for _ in range(5):
            assert self.limiter.check(WALLET_A)

    def test_blocks_over_limit(self):
        for _ in range(5):
            self.limiter.check(WALLET_A)
        assert not self.limiter.check(WALLET_A)

    def test_independent_wallets(self):
        """Different wallets have separate buckets."""
        for _ in range(5):
            self.limiter.check(WALLET_A)
        # wallet_a is blocked but wallet_b is fresh
        assert not self.limiter.check(WALLET_A)
        assert self.limiter.check(WALLET_B)

    def test_case_insensitive_wallet(self):
        """0xAbCd… and 0xabcd… should share the same bucket."""
        limiter = WalletRateLimiter(max_requests=2, window_seconds=10)
        assert limiter.check("0xAbCd" + "e" * 36)
        assert limiter.check("0xabcd" + "e" * 36)
        assert not limiter.check("0xABCD" + "e" * 36)

    # ---------- remaining ----------

    def test_remaining_decreases(self):
        assert self.limiter.remaining(WALLET_A) == 5
        self.limiter.check(WALLET_A)
        assert self.limiter.remaining(WALLET_A) == 4

    def test_remaining_never_negative(self):
        for _ in range(10):
            self.limiter.check(WALLET_A)
        assert self.limiter.remaining(WALLET_A) == 0

    # ---------- reset ----------

    def test_reset_single_wallet(self):
        for _ in range(5):
            self.limiter.check(WALLET_A)
        self.limiter.check(WALLET_B)
        self.limiter.reset(WALLET_A)
        assert self.limiter.remaining(WALLET_A) == 5
        assert self.limiter.remaining(WALLET_B) == 4

    def test_reset_all(self):
        self.limiter.check(WALLET_A)
        self.limiter.check(WALLET_B)
        self.limiter.reset()
        assert self.limiter.remaining(WALLET_A) == 5
        assert self.limiter.remaining(WALLET_B) == 5

    # ---------- sliding window ----------

    def test_window_expiry(self):
        """Timestamps older than the window should be pruned."""
        limiter = WalletRateLimiter(max_requests=2, window_seconds=1)
        assert limiter.check(WALLET_A)
        assert limiter.check(WALLET_A)
        assert not limiter.check(WALLET_A)

        # Fast-forward past the window
        bucket = limiter._buckets[WALLET_A.lower()]
        bucket.timestamps = [t - 2 for t in bucket.timestamps]

        # After expiry, requests should be allowed again
        assert limiter.check(WALLET_A)
