from dataclasses import dataclass

from services.privacy_contracts import PrivacyPolicy


class PrivacyPolicyError(ValueError):
    pass


@dataclass(frozen=True)
class ResolvedPrivacyPolicy:
    requested_profile: str
    policy: PrivacyPolicy
    config_profile: str

    @property
    def automatic_release_allowed(self) -> bool:
        return self.policy is PrivacyPolicy.SAFE_HARBOR_V1


def resolve_privacy_policy(profile: str) -> ResolvedPrivacyPolicy:
    normalized = (profile or "").strip().lower()
    if normalized in {"strict", "safe_harbor_v1"}:
        return ResolvedPrivacyPolicy(
            requested_profile=normalized,
            policy=PrivacyPolicy.SAFE_HARBOR_V1,
            config_profile="strict",
        )
    if normalized == "research":
        return ResolvedPrivacyPolicy(
            requested_profile=normalized,
            policy=PrivacyPolicy.EXPERT_DETERMINATION,
            config_profile="research",
        )
    raise PrivacyPolicyError(
        "Invalid privacy profile. Supported profiles: safe_harbor_v1, strict, research"
    )
