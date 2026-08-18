import os
import sys

import pytest

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services.privacy_profiles import (  # noqa: E402
    PrivacyProfileError,
    get_privacy_profile,
    load_privacy_profiles,
    validate_privacy_profile,
)


def test_strict_and_research_profiles_load():
    profiles = load_privacy_profiles()

    assert set(profiles) >= {"strict", "research"}
    assert profiles["strict"]["date_strategy"] == "redact"
    assert profiles["strict"]["text_identifier_strategy"] == "redact"
    assert profiles["strict"]["remove_dicom_private_tags"] is True
    assert profiles["research"]["date_strategy"] == "shift"
    assert profiles["research"]["text_identifier_strategy"] == "pseudonymize"
    assert profiles["research"]["preserve_dicom_technical_metadata"] is True
    assert profiles["research"]["remove_nifti_extensions"] is True


def test_profile_name_validation_is_case_insensitive():
    assert validate_privacy_profile(" Strict ") == "strict"
    assert validate_privacy_profile("RESEARCH") == "research"


def test_profile_settings_are_returned_as_a_copy():
    strict_settings = get_privacy_profile("strict")
    strict_settings["redact_dates"] = False

    assert get_privacy_profile("strict")["redact_dates"] is True


def test_invalid_profile_fails_clearly():
    with pytest.raises(PrivacyProfileError) as exc:
        get_privacy_profile("public")

    assert exc.value.status_code == 400
    assert "Invalid privacy profile" in exc.value.detail
    assert "strict" in exc.value.detail
    assert "research" in exc.value.detail


