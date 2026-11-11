"""RSC11 S13 staged CLI bindings; only the atomic cutover can install live v2 services.

Production v1 remains the single serving contract. No implicit fallback to
the shell launcher, no second manifest writer, and no synthetic authorization.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from ..engine_adapters.radiance_inventory import Layout
from ..hardware.gpu_power import Setting
from .manifest_power import ManifestPower
from .supervisor import ServeRefused


@dataclass(frozen=True)
class QualifiedPowerGrant:
    """S10-qualified proof projected onto the exact signed v2 power plan.

    Only a trusted S10 qualification loader may construct this in production.
    The user-facing CLI never accepts one from flags, JSON or environment.
    """
    profile: str
    profile_digest: str
    settings: dict[str, Setting]
    evidence: dict[str, str]
    engine_build_id: str
    helper_build_id: str
    driver_id: str


@dataclass(frozen=True)
class V2Bindings:
    """Real writer and engine pre-weight validator, installed only at S13(c)."""
    source: object
    writer: object
    validator: object
    qualification_gate: Callable[[ManifestPower, Layout], QualifiedPowerGrant] | None = None


def require_power_authorization(
    policy: ManifestPower,
    layout: Layout,
    *,
    explicit_profile: str | None,
    qualification_gate: Callable[[ManifestPower, Layout], QualifiedPowerGrant] | None,
) -> None:
    """Gate all S12 power application on explicit selection and audited S10 proof.

    Observe is never translated into apply, regardless of profile selection.
    """
    if explicit_profile is not None:
        if not isinstance(explicit_profile, str) or not explicit_profile.strip():
            raise ServeRefused("empty explicitly named power profile")
        if explicit_profile != policy.profile:
            raise ServeRefused("v2 manifest power profile differs from explicit CLI selection")
    if policy.mode == "observe":
        if policy.requested:
            raise ServeRefused("observe-only v2 power plan cannot request hardware writes")
        return
    if policy.mode != "apply":
        raise ServeRefused("unknown v2 power mode; refusing hardware writes")
    if not explicit_profile or explicit_profile == "stock" or not policy.requested:
        raise ServeRefused(
            "power apply requires an explicitly named non-stock profile with resolved settings"
        )
    if qualification_gate is None:
        raise ServeRefused(
            "power apply refused: no trusted S10 qualification loader for named profile"
        )
    grant = qualification_gate(policy, layout)
    if not isinstance(grant, QualifiedPowerGrant):
        raise ServeRefused("power apply refused: S10 qualification proof missing or untyped")
    if grant.profile != policy.profile or grant.profile_digest != policy.profile_digest:
        raise ServeRefused("power apply refused: qualified profile identity differs from v2")
    if grant.settings != policy.requested or set(grant.evidence) != set(policy.requested):
        raise ServeRefused("power apply refused: qualified BDF/settings set differs from v2")
    if (
        not grant.engine_build_id or not grant.helper_build_id or not grant.driver_id
        or any(not isinstance(e, str) or not e.strip() for e in grant.evidence.values())
    ):
        raise ServeRefused("power apply refused: qualified build/driver/evidence identity absent")


def production_v2_bindings() -> V2Bindings:
    """Replaced by the ONE atomic S13(c) production writer/reader cutover.

    A dummy writer or v1 manifest adapter would create a second active
    contract, so this explicitly refuses instead.
    """
    raise ServeRefused(
        "S13 v2 activation incomplete: the native P1 source, strict S10 writer "
        "and engine validator are not installed; v1 serving remains the single "
        "production contract, no legacy fallback"
    )
