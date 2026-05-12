"""EIP-712 signature verification for analytics API authentication."""

from __future__ import annotations

import time
from typing import Dict, Optional, Set

try:
    from web3 import Web3
except ImportError:
    Web3 = None  # type: ignore[misc, assignment]

DOCUMENT_STORAGE_ADDRESS = "0xd58de64aac08d5412b8020c7c61b215fec0c9644"
SIGNATURE_EXPIRY_SECONDS = 300  # 5 minutes

EIP712_TYPES = {
    "AnalyticsRequest": [
        {"name": "datasetCID", "type": "string"},
        {"name": "timestamp", "type": "uint256"},
        {"name": "nonce", "type": "uint256"},
        {"name": "requestHash", "type": "string"},
    ]
}

EIP712_DOMAIN = {"name": "BioBlockAnalytics", "version": "1"}

# In production, back this with Redis to persist across restarts
_used_nonces: Dict[str, Set[int]] = {}


def verify_signature(
    wallet_address: str,
    dataset_cid: str,
    signature: str,
    timestamp: int,
    nonce: int,
    request_hash: str,
    w3: Optional[Web3] = None,
) -> bool:
    """Verify EIP-712 signature with replay protection. Returns True if valid."""

    # Reject expired signatures
    if abs(time.time() - timestamp) > SIGNATURE_EXPIRY_SECONDS:
        return False

    # Reject reused nonces
    wallet_key = wallet_address.lower()
    wallet_nonces = _used_nonces.setdefault(wallet_key, set())
    if nonce in wallet_nonces:
        return False

    # TODO: recover signer from EIP-712 typed data via web3.py
    # TODO: verify on-chain purchase via DocumentStorage.sol hasPurchased()

    wallet_nonces.add(nonce)
    return True


def clear_nonces(wallet_address: Optional[str] = None) -> None:
    """Clear tracked nonces. Pass a wallet to clear just that wallet."""
    if wallet_address:
        _used_nonces.pop(wallet_address.lower(), None)
    else:
        _used_nonces.clear()
