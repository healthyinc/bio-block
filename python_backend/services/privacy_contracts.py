from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Protocol, Tuple


class PrivacyPolicy(str, Enum):
    SAFE_HARBOR_V1 = "safe_harbor_v1"
    EXPERT_DETERMINATION = "expert_determination"


class ReleaseDisposition(str, Enum):
    RELEASABLE = "releasable"
    MANUAL_REVIEW_REQUIRED = "manual_review_required"
    EXPERT_DETERMINATION_REQUIRED = "expert_determination_required"
    PROCESSING_FAILED = "processing_failed"


@dataclass(frozen=True)
class PhiEntity:
    """A normalized PHI finding; offsets refer only to transient input text."""

    entity_type: str
    start: int
    end: int
    source: str
    score: Optional[float] = None
    original_label: Optional[str] = None


class PhiDetector(Protocol):
    def detect(self, text: str) -> List[PhiEntity]:
        """Return findings without persisting or logging matched values."""


@dataclass(frozen=True)
class SanitizedArtifact:
    """Internal sanitized bytes plus immutable validation provenance."""

    content: bytes = field(repr=False)
    media_type: str
    filename: str
    validators: Tuple[str, ...]
    sha256: str = field(init=False)

    def __post_init__(self) -> None:
        if not self.content:
            raise ValueError("sanitized artifact content must not be empty")
        if not self.validators:
            raise ValueError("sanitized artifact requires at least one validator")
        object.__setattr__(self, "sha256", hashlib.sha256(self.content).hexdigest())


@dataclass(frozen=True)
class ReleaseDecision:
    policy: PrivacyPolicy
    disposition: ReleaseDisposition
    reason_codes: Tuple[str, ...]
    artifact_sha256: Optional[str] = None

    def __post_init__(self) -> None:
        releasable = self.disposition is ReleaseDisposition.RELEASABLE
        if releasable and self.policy is not PrivacyPolicy.SAFE_HARBOR_V1:
            raise ValueError("only safe_harbor_v1 can authorize automatic release")
        if releasable and not self.artifact_sha256:
            raise ValueError("releasable decisions require a sanitized artifact digest")
        if not releasable and self.artifact_sha256 is not None:
            raise ValueError("blocked decisions must not bind releasable bytes")
        if not self.reason_codes:
            raise ValueError("release decisions require at least one reason code")

    @property
    def releasable(self) -> bool:
        return self.disposition is ReleaseDisposition.RELEASABLE

    def to_public_dict(self) -> dict:
        return {
            "policy": self.policy.value,
            "disposition": self.disposition.value,
            "releasable": self.releasable,
            "reason_codes": list(self.reason_codes),
            "artifact_sha256": self.artifact_sha256,
        }


def issue_release(
    artifact: SanitizedArtifact,
    reason_code: str = "safe_harbor_validation_passed",
) -> ReleaseDecision:
    return ReleaseDecision(
        policy=PrivacyPolicy.SAFE_HARBOR_V1,
        disposition=ReleaseDisposition.RELEASABLE,
        reason_codes=(reason_code,),
        artifact_sha256=artifact.sha256,
    )


def manual_review_decision(*reason_codes: str) -> ReleaseDecision:
    return ReleaseDecision(
        policy=PrivacyPolicy.SAFE_HARBOR_V1,
        disposition=ReleaseDisposition.MANUAL_REVIEW_REQUIRED,
        reason_codes=tuple(reason_codes) or ("validation_incomplete",),
    )


def expert_determination_decision() -> ReleaseDecision:
    return ReleaseDecision(
        policy=PrivacyPolicy.EXPERT_DETERMINATION,
        disposition=ReleaseDisposition.EXPERT_DETERMINATION_REQUIRED,
        reason_codes=("expert_determination_required",),
    )
