from __future__ import annotations

import logging
from typing import Optional

import httpx

from app.config import JS_BACKEND_URL

logger = logging.getLogger(__name__)

_TIMEOUT = httpx.Timeout(30.0, connect=10.0)


async def upload_result_to_ipfs(
    result_data: dict,
    analysis_type: str,
    source_cid: str,
) -> Optional[str]:
    """Upload serialised analytics JSON to IPFS via the JS backend."""
    url = f"{JS_BACKEND_URL}/api/ipfs/upload-analytics-result"
    payload = {
        "resultJson": result_data,
        "fileName": f"analytics-{analysis_type}-{source_cid[:12]}.json",
    }

    try:
        async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
            resp = await client.post(url, json=payload)
            resp.raise_for_status()
            data = resp.json()
            cid = data.get("ipfsHash")
            if cid:
                logger.info("IPFS upload OK  cid=%s  type=%s", cid, analysis_type)
            else:
                logger.warning("IPFS response missing ipfsHash: %s", data)
            return cid
    except httpx.ConnectError:
        logger.warning("JS backend unreachable at %s — skipping IPFS", JS_BACKEND_URL)
        return None
    except httpx.HTTPStatusError as exc:
        logger.error(
            "IPFS upload failed  status=%d  body=%s",
            exc.response.status_code,
            exc.response.text[:500],
        )
        return None
    except Exception:
        logger.exception("Unexpected error during IPFS upload")
        return None

