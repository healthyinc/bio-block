import json
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict


PROFILE_CONFIG_PATH = (
    Path(__file__).resolve().parent.parent / "config" / "privacy_profiles.json"
)
REQUIRED_SETTINGS = {
    "redact_dates",
    "date_strategy",
    "text_identifier_strategy",
    "remove_dicom_private_tags",
    "preserve_dicom_technical_metadata",
    "remove_nifti_extensions",
    "ocr_confidence_threshold",
    "allow_generalized_demographics",
}
REQUIRED_PROFILES = {"strict", "research"}
DATE_STRATEGIES = {"redact", "shift", "generalize"}
TEXT_IDENTIFIER_STRATEGIES = {"redact", "pseudonymize"}


class PrivacyProfileError(ValueError):
    def __init__(self, detail: str, status_code: int = 400):
        super().__init__(detail)
        self.detail = detail
        self.status_code = status_code


def load_privacy_profiles() -> Dict[str, Dict[str, Any]]:
    return {name: dict(settings) for name, settings in _load_profiles().items()}


def get_privacy_profile(profile: str) -> Dict[str, Any]:
    profiles = _load_profiles()
    normalized = _normalize_profile_name(profile, profiles)
    return dict(profiles[normalized])


def validate_privacy_profile(profile: str) -> str:
    return _normalize_profile_name(profile, _load_profiles())


@lru_cache(maxsize=1)
def _load_profiles() -> Dict[str, Dict[str, Any]]:
    try:
        raw_profiles = json.loads(PROFILE_CONFIG_PATH.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise PrivacyProfileError(
            f"Privacy profile config not found: {PROFILE_CONFIG_PATH}",
            status_code=500,
        ) from exc
    except json.JSONDecodeError as exc:
        raise PrivacyProfileError(
            "Privacy profile config is not valid JSON.",
            status_code=500,
        ) from exc

    if not isinstance(raw_profiles, dict):
        raise PrivacyProfileError(
            "Privacy profile config must be a JSON object.",
            status_code=500,
        )

    missing_profiles = REQUIRED_PROFILES - set(raw_profiles)
    if missing_profiles:
        missing = ", ".join(sorted(missing_profiles))
        raise PrivacyProfileError(
            f"Privacy profile config missing required profiles: {missing}",
            status_code=500,
        )

    profiles: Dict[str, Dict[str, Any]] = {}
    for name, settings in raw_profiles.items():
        normalized = str(name).strip().lower()
        if not normalized:
            raise PrivacyProfileError(
                "Privacy profile names must not be empty.",
                status_code=500,
            )
        if not isinstance(settings, dict):
            raise PrivacyProfileError(
                f"Privacy profile '{normalized}' must be a JSON object.",
                status_code=500,
            )

        _validate_settings(normalized, settings)
        profiles[normalized] = dict(settings)

    return profiles


def _normalize_profile_name(
    profile: str,
    profiles: Dict[str, Dict[str, Any]],
) -> str:
    normalized = (profile or "").strip().lower()
    if normalized not in profiles:
        supported = ", ".join(sorted(profiles))
        raise PrivacyProfileError(
            f"Invalid privacy profile '{profile}'. Supported profiles: {supported}"
        )
    return normalized


def _validate_settings(name: str, settings: Dict[str, Any]) -> None:
    missing_settings = REQUIRED_SETTINGS - set(settings)
    if missing_settings:
        missing = ", ".join(sorted(missing_settings))
        raise PrivacyProfileError(
            f"Privacy profile '{name}' missing settings: {missing}",
            status_code=500,
        )

    for key in (
        "redact_dates",
        "remove_dicom_private_tags",
        "preserve_dicom_technical_metadata",
        "remove_nifti_extensions",
        "allow_generalized_demographics",
    ):
        if not isinstance(settings[key], bool):
            raise PrivacyProfileError(
                f"Privacy profile '{name}' setting '{key}' must be a boolean.",
                status_code=500,
            )

    if settings["date_strategy"] not in DATE_STRATEGIES:
        allowed = ", ".join(sorted(DATE_STRATEGIES))
        raise PrivacyProfileError(
            f"Privacy profile '{name}' date_strategy must be one of: {allowed}",
            status_code=500,
        )

    if settings["text_identifier_strategy"] not in TEXT_IDENTIFIER_STRATEGIES:
        allowed = ", ".join(sorted(TEXT_IDENTIFIER_STRATEGIES))
        raise PrivacyProfileError(
            (
                f"Privacy profile '{name}' text_identifier_strategy must be "
                f"one of: {allowed}"
            ),
            status_code=500,
        )

    threshold = settings["ocr_confidence_threshold"]
    if not isinstance(threshold, (int, float)) or not 0 <= float(threshold) <= 1:
        raise PrivacyProfileError(
            (
                f"Privacy profile '{name}' ocr_confidence_threshold must be "
                "between 0 and 1."
            ),
            status_code=500,
        )

