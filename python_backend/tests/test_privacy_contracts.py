import os
import sys

import pytest

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services.privacy_contracts import (  # noqa: E402
    PrivacyPolicy,
    ReleaseDecision,
    ReleaseDisposition,
    SanitizedArtifact,
    expert_determination_decision,
    issue_release,
)
from services.privacy_policy import resolve_privacy_policy  # noqa: E402
from services.ingestion import route_for_ingestion  # noqa: E402


def test_strict_is_compatibility_alias_for_safe_harbor_v1():
    strict = resolve_privacy_policy("strict")
    canonical = resolve_privacy_policy("safe_harbor_v1")

    assert strict.policy is PrivacyPolicy.SAFE_HARBOR_V1
    assert canonical.policy is PrivacyPolicy.SAFE_HARBOR_V1
    assert strict.config_profile == canonical.config_profile == "strict"


def test_research_policy_requires_expert_determination():
    resolved = resolve_privacy_policy("research")
    decision = expert_determination_decision()

    assert resolved.automatic_release_allowed is False
    assert decision.disposition is ReleaseDisposition.EXPERT_DETERMINATION_REQUIRED
    assert decision.releasable is False
    assert decision.artifact_sha256 is None


def test_only_safe_harbor_policy_can_issue_release():
    artifact = SanitizedArtifact(
        content=b"sanitized",
        media_type="text/plain",
        filename="note.txt",
        validators=("synthetic_validator",),
    )
    decision = issue_release(artifact)

    assert decision.releasable is True
    assert decision.artifact_sha256 == artifact.sha256

    with pytest.raises(ValueError):
        ReleaseDecision(
            policy=PrivacyPolicy.EXPERT_DETERMINATION,
            disposition=ReleaseDisposition.RELEASABLE,
            reason_codes=("invalid",),
            artifact_sha256=artifact.sha256,
        )


def test_blocked_decision_cannot_reference_artifact_bytes():
    with pytest.raises(ValueError):
        ReleaseDecision(
            policy=PrivacyPolicy.SAFE_HARBOR_V1,
            disposition=ReleaseDisposition.MANUAL_REVIEW_REQUIRED,
            reason_codes=("blocked",),
            artifact_sha256="abc",
        )


@pytest.mark.parametrize(
    ("filename", "content_type", "header"),
    [
        ("records.csv", "text/csv", b"name\nSynthetic Person\n"),
        ("note.txt", "text/plain", b"Patient Synthetic Person"),
        ("scan.dcm", "application/dicom", b"synthetic-dicom"),
        ("brain.nii", "application/octet-stream", b"synthetic-nifti"),
        ("slide.svs", "application/octet-stream", b"synthetic-wsi"),
    ],
)
def test_research_routing_never_exposes_or_processes_content(
    monkeypatch,
    filename,
    content_type,
    header,
):
    from services import ingestion

    def must_not_run(*args, **kwargs):
        raise AssertionError("research modality handler must not run")

    modality = ingestion.detect_modality(filename, content_type, header)
    monkeypatch.setitem(ingestion.HANDLER_REGISTRY, modality, must_not_run)

    result = route_for_ingestion(
        filename=filename,
        content_type=content_type,
        header=header,
        profile="research",
        text_content=header,
        file_content=header,
    )

    serialized = repr(result)
    assert result["anonymization_status"] == "expert_determination_required"
    assert result["release_decision"]["releasable"] is False
    assert "anonymized_text" not in result
    assert "Synthetic Person" not in serialized
    assert all(value == "blocked" for value in result["downstream"].values())


def test_sanitized_artifact_repr_hides_content():
    artifact = SanitizedArtifact(
        content=b"synthetic-sensitive-marker",
        media_type="application/octet-stream",
        filename="artifact.bin",
        validators=("synthetic_validator",),
    )

    assert "synthetic-sensitive-marker" not in repr(artifact)
