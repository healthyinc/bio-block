from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import List, Optional

from app.config import (
    ANALYTICS_PRIVATE_KEY,
    ANALYTICS_REGISTRY_ADDRESS,
    SEPOLIA_RPC_URL,
)

logger = logging.getLogger(__name__)

_ABI_PATH = Path(__file__).resolve().parent.parent / "contracts" / "AnalyticsRegistry.json"

_w3 = None
_contract = None
_account = None


def _ensure_web3():
    global _w3, _contract, _account  # noqa: PLW0603

    if _w3 is not None:
        return True

    if not SEPOLIA_RPC_URL or not ANALYTICS_PRIVATE_KEY or not ANALYTICS_REGISTRY_ADDRESS:
        logger.warning("On-chain config incomplete — chain features disabled.")
        return False

    try:
        from web3 import Web3

        _w3 = Web3(Web3.HTTPProvider(SEPOLIA_RPC_URL))
        if not _w3.is_connected():
            logger.warning("web3 failed to connect to %s", SEPOLIA_RPC_URL)
            _w3 = None
            return False

        with open(_ABI_PATH) as f:
            abi = json.load(f)

        _contract = _w3.eth.contract(
            address=Web3.to_checksum_address(ANALYTICS_REGISTRY_ADDRESS),
            abi=abi,
        )
        _account = _w3.eth.account.from_key(ANALYTICS_PRIVATE_KEY)
        logger.info(
            "web3 connected  chain=%s  registry=%s  wallet=%s",
            _w3.eth.chain_id,
            ANALYTICS_REGISTRY_ADDRESS,
            _account.address,
        )
        return True
    except Exception:
        logger.exception("Failed to initialise web3")
        _w3 = None
        return False


async def register_on_chain(
    source_cid: str,
    result_cid: str,
    analysis_type: str,
    analyst_address: str,
) -> Optional[str]:
    """Call registerAnalytics() on-chain.

    The server wallet (ANALYTICS_PRIVATE_KEY) pays the gas, but the
    ``analyst_address`` — obtained from EIP-712 authentication — is
    forwarded to the contract so the on-chain record is attributed to
    the real analyst, not the relayer.

    Returns tx hash or None.
    """
    if not _ensure_web3():
        return None

    try:
        from web3 import Web3

        analyst_checksum = Web3.to_checksum_address(analyst_address)
        nonce = _w3.eth.get_transaction_count(_account.address)
        tx = _contract.functions.registerAnalytics(
            source_cid, result_cid, analysis_type, analyst_checksum
        ).build_transaction(
            {
                "from": _account.address,
                "nonce": nonce,
                "gas": 300_000,
                "gasPrice": _w3.eth.gas_price,
            }
        )
        signed = _w3.eth.account.sign_transaction(tx, ANALYTICS_PRIVATE_KEY)
        tx_hash = _w3.eth.send_raw_transaction(signed.raw_transaction)
        hex_hash = tx_hash.hex()
        logger.info(
            "On-chain registration sent  tx=%s  analyst=%s  source=%s  result=%s",
            hex_hash,
            analyst_address,
            source_cid[:16],
            result_cid[:16],
        )
        return hex_hash
    except Exception as exc:
        if "nonce" in str(exc).lower():
            logger.warning("Nonce conflict — retrying once: %s", exc)
            try:
                nonce = _w3.eth.get_transaction_count(_account.address, "pending")
                tx = _contract.functions.registerAnalytics(
                    source_cid, result_cid, analysis_type, analyst_checksum
                ).build_transaction(
                    {
                        "from": _account.address,
                        "nonce": nonce,
                        "gas": 300_000,
                        "gasPrice": _w3.eth.gas_price,
                    }
                )
                signed = _w3.eth.account.sign_transaction(tx, ANALYTICS_PRIVATE_KEY)
                tx_hash = _w3.eth.send_raw_transaction(signed.raw_transaction)
                return tx_hash.hex()
            except Exception:
                logger.exception("On-chain registration failed on retry")
                return None
        logger.exception("On-chain registration failed")
        return None


def get_analytics_for_dataset(source_cid: str) -> List[str]:
    """Query all result CIDs linked to a source dataset on-chain."""
    if not _ensure_web3():
        return []

    try:
        return _contract.functions.getAnalyticsForDataset(source_cid).call()
    except Exception:
        logger.exception("getAnalyticsForDataset call failed for %s", source_cid)
        return []

