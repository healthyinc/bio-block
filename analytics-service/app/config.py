"""Centralized configuration for the Bio-Block Analytics API.

Loads environment variables with sensible defaults for local development.
"""

from __future__ import annotations

import os
import logging

from dotenv import load_dotenv

load_dotenv()  

logger = logging.getLogger(__name__)


def _get(key: str, default: str = "") -> str:
    return os.environ.get(key, default)


JS_BACKEND_URL: str = _get("JS_BACKEND_URL", "http://127.0.0.1:3001")

SEPOLIA_RPC_URL: str = _get("SEPOLIA_RPC_URL", "")
ANALYTICS_PRIVATE_KEY: str = _get("ANALYTICS_PRIVATE_KEY", "")
ANALYTICS_REGISTRY_ADDRESS: str = _get(
    "ANALYTICS_REGISTRY_ADDRESS",
    "0x9148Cd47B9c166CC651F57B8BfA77bf66496A90f",
)

PINATA_GATEWAY_URL: str = _get(
    "PINATA_GATEWAY_URL", "https://gateway.pinata.cloud/ipfs"
)
