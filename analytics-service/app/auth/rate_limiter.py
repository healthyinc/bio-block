"""Per-wallet rate limiter for the analytics API.

Uses a simple sliding-window counter backed by an in-memory dict.
In production this should be swapped for Redis (INCR + EXPIRE),
but the in-memory version is fine for single-instance dev/staging.
"""

from __future__ import annotations

import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class _WalletBucket:
    """Sliding window of request timestamps for a single wallet."""
    timestamps: List[float] = field(default_factory=list)


class WalletRateLimiter:
    """Sliding-window rate limiter keyed by wallet address.

    Parameters
    ----------
    max_requests : int
        Maximum number of requests allowed in the window.
    window_seconds : int
        Length of the sliding window in seconds.

    Example
    -------
    >>> limiter = WalletRateLimiter(max_requests=30, window_seconds=60)
    >>> limiter.check("0xABC...")  # True  — allowed
    >>> limiter.check("0xABC...")  # True  — still under limit
    """

    def __init__(self, max_requests: int = 30, window_seconds: int = 60) -> None:
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self._buckets: Dict[str, _WalletBucket] = defaultdict(_WalletBucket)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def check(self, wallet_address: str) -> bool:
        """Return True if the request is allowed, False if rate-limited."""
        key = wallet_address.lower()
        bucket = self._buckets[key]
        now = time.time()

        # Prune timestamps outside the window
        cutoff = now - self.window_seconds
        bucket.timestamps = [ts for ts in bucket.timestamps if ts > cutoff]

        if len(bucket.timestamps) >= self.max_requests:
            return False

        bucket.timestamps.append(now)
        return True

    def remaining(self, wallet_address: str) -> int:
        """Return the number of requests the wallet has left in the window."""
        key = wallet_address.lower()
        bucket = self._buckets[key]
        now = time.time()
        cutoff = now - self.window_seconds
        active = sum(1 for ts in bucket.timestamps if ts > cutoff)
        return max(0, self.max_requests - active)

    def reset(self, wallet_address: str | None = None) -> None:
        """Clear rate-limit state.  None clears all wallets."""
        if wallet_address:
            self._buckets.pop(wallet_address.lower(), None)
        else:
            self._buckets.clear()


# Module-level singleton — import and use directly
rate_limiter = WalletRateLimiter(max_requests=30, window_seconds=60)
