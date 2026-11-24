"""Which detectors the second pass runs, chosen by the server alone.

Phase 12D measured the residual pass across 300 synthetic documents spanning
every text-representable Safe Harbor category. GLiNER caught nothing there
that another detector did not also catch - zero uniquely attributed spans
against Stanford's seventeen - while costing roughly three quarters of the
pass. ``optimized_v1`` drops it from that pass only.

``optimized_v1`` was then evaluated for production and rejected: its locked
acceptance runs failed privacy and utility gates, and no run showed the saving
could be taken safely. ``full_v1`` remains the default and the supported
production profile. ``optimized_v1`` stays selectable only by explicit server
configuration, for measurement.

Two things this deliberately does not do. It does not touch the first pass:
GLiNER is where most of the detection happens and it stays mandatory there in
both profiles. And it does not gate by region or by risk - both profiles scan
the complete final output, because a validator that only looks where it
expects trouble is not a validator.

The profile is server configuration and nothing else. It is read from the
environment, never from a request, and the name is deliberately unlike the
``profile`` form field that selects strict or research, so a request field
cannot be mistaken for this one. An unrecognized value raises instead of
falling back, because the quiet failure mode here - a typo silently selecting
the weaker profile - is exactly the one worth refusing.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, FrozenSet, Tuple

#: Server-side only. Never populated from request data.
PROFILE_ENV_VAR = "PHI_RESIDUAL_VALIDATOR_PROFILE"

PROFILE_FULL = "full_v1"
PROFILE_OPTIMIZED = "optimized_v1"

#: full_v1 is the default and the rollback. Changing this default is a
#: deliberate decision, not a consequence of an unset variable.
DEFAULT_PROFILE = PROFILE_FULL

ALLOWED_PROFILES: FrozenSet[str] = frozenset({PROFILE_FULL, PROFILE_OPTIMIZED})

#: Detector source names, matching what each adapter reports.
SOURCE_STRUCTURED = "structured_pattern"
SOURCE_AGE = "age_rule"
SOURCE_SPACY = "ner"
SOURCE_STANFORD_NAME = "stanford_deidentifier"
SOURCE_GLINER_NAME = "gliner_multi_pii"

#: The deterministic rules run in every profile and every pass.
_ALWAYS = (SOURCE_STRUCTURED, SOURCE_AGE, SOURCE_SPACY)

#: An explicit allowlist per profile. Anything not named here cannot be
#: selected, so a malformed or hostile value cannot widen or narrow the set.
_RESIDUAL_SOURCES: Dict[str, Tuple[str, ...]] = {
    PROFILE_FULL: _ALWAYS + (SOURCE_STANFORD_NAME, SOURCE_GLINER_NAME),
    PROFILE_OPTIMIZED: _ALWAYS + (SOURCE_STANFORD_NAME,),
}

#: The first pass is identical in both profiles.
FIRST_PASS_SOURCES: Tuple[str, ...] = _ALWAYS + (
    SOURCE_STANFORD_NAME,
    SOURCE_GLINER_NAME,
)

#: A residual pass with no model at all is not a validator. Enforced rather
#: than assumed, so a future profile cannot reduce to rules only.
_MODEL_SOURCES = frozenset({SOURCE_SPACY, SOURCE_STANFORD_NAME, SOURCE_GLINER_NAME})

PASS_FIRST = "first"
PASS_RESIDUAL = "residual"


class ResidualProfileError(ValueError):
    """Configuration that cannot be honoured safely."""

    def __init__(self, error_code: str):
        super().__init__(error_code)
        self.error_code = error_code


def configured_profile() -> str:
    """The active profile, from server configuration only.

    Raises on anything unrecognized. Falling back to a default here would
    turn a deployment typo into a silently weaker validator.
    """
    raw = os.getenv(PROFILE_ENV_VAR)
    if raw is None or not raw.strip():
        return DEFAULT_PROFILE
    value = raw.strip()
    if value not in ALLOWED_PROFILES:
        raise ResidualProfileError("invalid_residual_validator_profile")
    return value


def validate_configuration() -> str:
    """Check the configuration at startup. Returns the profile it accepted."""
    profile = configured_profile()
    sources = _RESIDUAL_SOURCES[profile]
    if not set(sources) & _MODEL_SOURCES:
        raise ResidualProfileError("residual_validator_has_no_model")
    if SOURCE_GLINER_NAME not in FIRST_PASS_SOURCES:
        raise ResidualProfileError("first_pass_missing_gliner")
    return profile


def required_sources(pass_kind: str, profile: str) -> Tuple[str, ...]:
    """The detectors that must run and report for this pass to be complete.

    The first pass ignores the profile entirely; only the residual pass
    varies. A pass name that is not one of the two is an error, not a reason
    to guess.
    """
    if pass_kind == PASS_FIRST:
        return FIRST_PASS_SOURCES
    if pass_kind != PASS_RESIDUAL:
        raise ResidualProfileError("unknown_validation_pass")
    if profile not in ALLOWED_PROFILES:
        raise ResidualProfileError("invalid_residual_validator_profile")
    return _RESIDUAL_SOURCES[profile]


def omits_gliner(pass_kind: str, profile: str) -> bool:
    return SOURCE_GLINER_NAME not in required_sources(pass_kind, profile)


@dataclass(frozen=True)
class ValidationEvidence:
    """What actually ran, for the release decision to check.

    Carries names and counts only - no text, no offsets, no model paths, no
    checksums - because this record travels with the artifact's manifest.
    """

    profile: str
    pass_kind: str
    required: Tuple[str, ...]
    completed: Tuple[str, ...]

    @property
    def complete(self) -> bool:
        return set(self.required) <= set(self.completed)

    @property
    def missing(self) -> Tuple[str, ...]:
        return tuple(sorted(set(self.required) - set(self.completed)))

    def to_public_dict(self) -> Dict[str, object]:
        return {
            "validator_profile": self.profile,
            "pass": self.pass_kind,
            "detectors_required": sorted(self.required),
            "detectors_completed": sorted(self.completed),
            "validation_complete": self.complete,
            "detectors_missing": list(self.missing),
        }


def build_evidence(
    pass_kind: str, profile: str, completed: Tuple[str, ...]
) -> ValidationEvidence:
    return ValidationEvidence(
        profile=profile,
        pass_kind=pass_kind,
        required=required_sources(pass_kind, profile),
        completed=tuple(completed),
    )
