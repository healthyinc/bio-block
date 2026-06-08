"""FastAPI dependency for EIP-712 wallet authentication.

Extracts the common auth parameters (wallet, signature, nonce, etc.) from
multipart form data and runs ``verify_signature`` so that individual route
handlers do not need to repeat the same boilerplate.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from fastapi import Form, HTTPException

from app.auth.eip712 import verify_signature

logger = logging.getLogger(__name__)


@dataclass
class AuthenticatedWallet:
    """Result returned by a successful EIP-712 auth check."""

    wallet_address: str
    dataset_cid: str
    nonce: int


async def require_eip712_auth(
    wallet_address: str = Form(..., description="Ethereum wallet address (0x…)"),
    dataset_cid: str = Form(..., description="IPFS CID of the target dataset"),
    signature: str = Form(..., description="EIP-712 hex signature"),
    timestamp: int = Form(..., description="Unix epoch seconds when the request was signed"),
    nonce: int = Form(..., description="Single-use nonce for replay protection"),
    request_hash: str = Form(..., description="SHA-256 hash of the request payload"),
) -> AuthenticatedWallet:
    """FastAPI dependency that gates a route behind EIP-712 auth.

    Usage::

        @app.post("/analytics/describe")
        async def describe(
            auth: AuthenticatedWallet = Depends(require_eip712_auth),
            file: UploadFile = File(...),
        ):
            ...

    Raises:
        HTTPException 401 – if any auth check fails.
    """

    if not verify_signature(
        wallet_address=wallet_address,
        dataset_cid=dataset_cid,
        signature=signature,
        timestamp=timestamp,
        nonce=nonce,
        request_hash=request_hash,
    ):
        logger.warning(
            "EIP-712 auth failed for wallet=%s dataset=%s nonce=%d",
            wallet_address,
            dataset_cid,
            nonce,
        )
        raise HTTPException(status_code=401, detail="Invalid or expired signature.")

    logger.info(
        "EIP-712 auth OK for wallet=%s dataset=%s nonce=%d",
        wallet_address,
        dataset_cid,
        nonce,
    )
    return AuthenticatedWallet(
        wallet_address=wallet_address,
        dataset_cid=dataset_cid,
        nonce=nonce,
    )
