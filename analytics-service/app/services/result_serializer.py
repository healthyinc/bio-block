"""Serialize analytics results into the canonical JSON format for IPFS."""

from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

APP_VERSION = "0.1.0"


def _sanitize_for_json(obj: Any) -> Any:
    """Replace NaN/Infinity with None for valid JSON output."""
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    if isinstance(obj, dict):
        return {k: _sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize_for_json(v) for v in obj]
    return obj


def serialize_analytics_result(
    analysis_type: str,
    source_cid: str,
    wallet_address: str,
    results: Dict[str, Any],
    row_count: int,
    columns: List[str],
    parameters: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Build the structured result document that gets pinned to IPFS."""
    return _sanitize_for_json({
        "version": "1.0",
        "analysis_type": analysis_type,
        "source_dataset_cid": source_cid,
        "analyst_wallet": wallet_address,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "parameters": parameters or {},
        "results": results,
        "metadata": {
            "api_version": APP_VERSION,
            "row_count": row_count,
            "columns_analyzed": columns,
        },
    })
