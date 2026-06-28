"""EIP-712 signature verification with environment-aware security guards."""

from __future__ import annotations

import logging
import os
import time
from typing import Dict, Optional, Set

try:
    from web3 import Web3
    from eth_account.messages import encode_structured_data
except ImportError:
    Web3 = None  # type: ignore[misc, assignment]

logger = logging.getLogger(__name__)

APP_ENV = os.getenv("APP_ENV", "production")
_SAFE_ENVS = {"test", "development"}

# ── Fail-fast: production MUST have web3.py installed ──────────────────
if Web3 is None and APP_ENV not in _SAFE_ENVS:
    raise RuntimeError(
        "web3.py is required in production but is not installed. "
        "Set APP_ENV=test or APP_ENV=development to skip crypto checks."
    )

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

# Minimal ABI — only the one read we need.
# Solidity mapping: hasPurchased(string ipfsHash, address buyer)
DOCUMENT_STORAGE_ABI = [
    {
        "inputs": [
            {"internalType": "string", "name": "", "type": "string"},
            {"internalType": "address", "name": "", "type": "address"},
        ],
        "name": "hasPurchased",
        "outputs": [{"internalType": "bool", "name": "", "type": "bool"}],
        "stateMutability": "view",
        "type": "function",
    }
]

SEPOLIA_RPC = "https://rpc.sepolia.org"


def verify_signature(
    wallet_address: str,
    dataset_cid: str,
    signature: str,
    timestamp: int,
    nonce: int,
    request_hash: str,
    w3: Optional["Web3"] = None,
) -> bool:
    """Verify an EIP-712 AnalyticsRequest signature.

    Checks (in order):
      1. Timestamp within the 5-minute expiry window.
      2. Nonce not reused by this wallet.
      3. Recovered signer matches the claimed wallet address.
      4. Wallet has purchased the dataset on-chain (hasPurchased mapping).

    In test/development environments (``APP_ENV``), steps 3-4 are skipped
    so tests can run without ``web3.py``.  In production the full
    verification pipeline is enforced.
    """

    # ── Timestamp expiry (all environments) ────────────────────────────
    if abs(time.time() - timestamp) > SIGNATURE_EXPIRY_SECONDS:
        return False

    # ── Replay protection (all environments) ───────────────────────────
    wallet_key = wallet_address.lower()
    wallet_nonces = _used_nonces.setdefault(wallet_key, set())
    if nonce in wallet_nonces:
        return False

    # ── Environment gate ───────────────────────────────────────────────
    if Web3 is None and APP_ENV in _SAFE_ENVS:
        logger.debug(
            "APP_ENV=%s — skipping crypto verification (non-production)",
            APP_ENV,
        )
        wallet_nonces.add(nonce)
        return True

    if w3 is None:
        if Web3 is None:
            raise RuntimeError("web3.py is required in production.")
        w3 = Web3(Web3.HTTPProvider(SEPOLIA_RPC))

    # --- Signer recovery ---
    typed_data = {
        "types": {
            "EIP712Domain": [
                {"name": "name", "type": "string"},
                {"name": "version", "type": "string"},
                {"name": "verifyingContract", "type": "address"},
            ],
            **EIP712_TYPES,
        },
        "primaryType": "AnalyticsRequest",
        "domain": {**EIP712_DOMAIN, "verifyingContract": DOCUMENT_STORAGE_ADDRESS},
        "message": {
            "datasetCID": dataset_cid,
            "timestamp": timestamp,
            "nonce": nonce,
            "requestHash": request_hash,
        },
    }
    try:
        encoded = encode_structured_data(primitive=typed_data)
        signer = w3.eth.account.recover_message(encoded, signature=signature)
    except Exception:
        return False  # malformed signature

    if signer.lower() != wallet_key:
        return False

    # --- On-chain purchase check ---
    try:
        contract = w3.eth.contract(
            address=Web3.to_checksum_address(DOCUMENT_STORAGE_ADDRESS),
            abi=DOCUMENT_STORAGE_ABI,
        )
        purchased = contract.functions.hasPurchased(
            dataset_cid,
            Web3.to_checksum_address(wallet_address),
        ).call()
    except Exception:
        return False  # RPC failure — fail closed

    if not purchased:
        return False

    wallet_nonces.add(nonce)
    return True


def clear_nonces(wallet_address: Optional[str] = None) -> None:
    """Clear stored nonces for a wallet (or all wallets)."""
    if wallet_address:
        _used_nonces.pop(wallet_address.lower(), None)
    else:
        _used_nonces.clear()

