"""Whether the detectors behind a result can authorise an automatic release.

Release used to depend only on what a handler verified. A handler verifies
against the detectors it was given, so the same checks passed whether the
pinned real models ran or the lightweight detectors the unit suite uses - and
in that mode a native PDF was released with a record number the lightweight
detectors never recognised. Verification cannot vouch for detection it did not
perform.

This module answers one question for the release boundary:

* ``production_models`` - the pinned models are the configured detectors and
  have not failed to load. Release behaves exactly as before.
* ``non_production_detectors`` - the lightweight test detectors are in use.
  Results are still produced, but nothing is ever released.
* ``models_unavailable`` - the model configuration is invalid or loading
  failed. Requests already fail closed; if anything reaches the release
  boundary anyway, it is held.

``privacy_contracts.issue_release``, the shared release decision and the
released-artifact store all consult it, so no format handler has to remember
to. It reports a capability and a reason code, never a detector name or path.
"""

from __future__ import annotations

from typing import Optional

CAPABILITY_PRODUCTION = "production_models"
CAPABILITY_NON_PRODUCTION = "non_production_detectors"
CAPABILITY_UNAVAILABLE = "models_unavailable"

REASON_NON_PRODUCTION = "non_production_detector_mode"
REASON_UNAVAILABLE = "detector_capability_unavailable"

_BLOCKING_REASONS = {
    CAPABILITY_NON_PRODUCTION: REASON_NON_PRODUCTION,
    CAPABILITY_UNAVAILABLE: REASON_UNAVAILABLE,
}


def current_capability() -> str:
    """The capability of the configured detectors, failing closed."""
    from services.local_model_detectors import (
        MODE_OFFLINE,
        LocalModelError,
        resolve_model_mode,
    )
    from services.model_client import current_client
    from services.model_lifecycle import STATE_READY, state

    try:
        mode = resolve_model_mode()
    except LocalModelError:
        return CAPABILITY_UNAVAILABLE
    if mode != MODE_OFFLINE:
        return CAPABILITY_NON_PRODUCTION
    client = current_client()
    if state().state != STATE_READY or client is None or not client.is_ready():
        return CAPABILITY_UNAVAILABLE
    return CAPABILITY_PRODUCTION


def release_blocker() -> Optional[str]:
    """The reason code that forbids release, or None when release may proceed."""
    return _BLOCKING_REASONS.get(current_capability())
