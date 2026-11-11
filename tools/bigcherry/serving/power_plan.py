"""Inactive PWR08/S10 power-plan resolver: bc-serve BDFs plus probed capabilities.

Does not resolve layouts, read environment inventory, write a manifest, or
write sysfs. The C++ S10 integrator owns the manifest-writer cutover. An
explicit apply request must be qualified for each selected physical device
and for each requested control, then freshly supported by the probe. This
module is deliberately unused by the v1 launcher.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from ..engine_adapters.radiance_inventory import Layout
from ..hardware.power_probe import GpuCapability
from ..hardware.power_profiles import (
    Profile,
    ProfileError,
    resolved_profile_digest,
    validate_settings,
)


class PowerPlanRefused(ValueError):
    """No plan emitted; the caller must not apply settings."""


@dataclass(frozen=True)
class Qualification:
    """Verified external calibration; only S10 should construct these in production.

    The loader which authenticates evidence, build and driver identity is not
    in this inactive slice. Tests inject proof-shaped objects; do not treat
    this dataclass alone as a certification authority.
    """

    bdf: str
    arch: str
    profile: str
    settings: dict[str, int]
    evidence: str
    build_id: str
    driver_id: str
    qualified: bool


@dataclass(frozen=True)
class PowerDevice:
    role: str
    bdf: str
    settings: dict[str, int]


@dataclass(frozen=True)
class PowerPlan:
    mode: str
    profile: str
    profile_digest: str
    devices: tuple[PowerDevice, ...]

    def to_manifest_section(self) -> dict:
        return {
            "mode": self.mode,
            "profile": self.profile,
            "profile_digest": self.profile_digest,
            "devices": [
                {"role": d.role, "bdf": d.bdf, "settings": dict(d.settings)}
                for d in self.devices
            ],
            "cpu": {"settings": {}},
        }


def _selected(layout: Layout) -> list[tuple[str, str, str]]:
    selected = [
        *[
            ("engine", bdf, layout.inventory.by_bdf(bdf).arch)
            for bdf in layout.engine_bdfs
        ],
        *[
            (f"helper{i}", bdf, layout.inventory.by_bdf(bdf).arch)
            for i, bdf in enumerate(layout.helper_bdfs)
        ],
    ]
    bdfs = [bdf for _, bdf, _ in selected]
    if not layout.engine_bdfs or len(bdfs) != len(set(bdfs)):
        raise PowerPlanRefused("bc-serve returned empty/duplicate power-plan BDFs")
    return selected


def _check_probed(
    bdf: str, requested: dict[str, int], probe: GpuCapability | None
) -> None:
    if (
        probe is None
        or probe.bdf != bdf
        or probe.driver.status != "supported"
        or probe.driver.value != "amdgpu"
    ):
        raise PowerPlanRefused(
            f"{bdf}: missing or non-amdgpu live power capability probe"
        )
    for key, value in requested.items():
        if key == "sclk_offset_mhz":
            # The current PowerTransaction/backend cannot apply this control;
            # do not promise it merely because the probe can report it.
            raise PowerPlanRefused(f"{bdf}: sclk_offset_mhz has no transaction backend")
        cap = probe.controls.get(key)
        if (
            cap is None
            or cap.status != "supported"
            or cap.minimum is None
            or cap.maximum is None
        ):
            raise PowerPlanRefused(
                f"{bdf}: {key} not supported by the fresh power probe"
            )
        requested_value = value * 1_000_000 if key == "power_cap_w" else value
        if not cap.minimum <= requested_value <= cap.maximum:
            raise PowerPlanRefused(f"{bdf}: {key} outside probed device range")
        if cap.current.status != "supported":
            raise PowerPlanRefused(
                f"{bdf}: {key} original value not readable for restore"
            )


def resolve_power_plan(
    profile: Profile,
    layout: Layout,
    *,
    mode: str = "observe",
    probes: Mapping[str, GpuCapability] | None = None,
    qualifications: Mapping[str, Qualification] | None = None,
    role_overrides: Mapping[str, dict[str, int]] | None = None,
) -> PowerPlan:
    """Resolve only the bc-serve-selected BDFs. Never apply on profile choice alone.

    Observe is always write-free, even for a candidate profile. Apply requires
    explicit mode, per-card externally verified qualification and live probe
    range/readback support for every nonempty setting. No unvalidated override.
    """
    if mode not in ("observe", "apply"):
        raise PowerPlanRefused(f"unknown power mode {mode!r}")
    selected = _selected(layout)
    roles: dict[str, dict[str, int]] = {}
    overrides = role_overrides or {}
    unknown = set(overrides) - {role for role, _, _ in selected}
    if unknown:
        raise PowerPlanRefused(
            f"power override for non-selected role(s): {sorted(unknown)}"
        )
    try:
        for role, _, arch in selected:
            candidate = profile.settings_for(role, arch)
            if role in roles and roles[role] != candidate:
                raise PowerPlanRefused(
                    f"engine rank architecture yields conflicting role settings for {role}"
                )
            roles[role] = candidate
        # v2 digest is taken before intent overrides; role/BDF selection is
        # exactly the C++ bc-serve resolved inventory, never ordinal guesses.
        digest = resolved_profile_digest(profile.name, roles)
        resolved = []
        for role, bdf, arch in selected:
            settings = dict(roles[role])
            if role in overrides:
                patch = validate_settings(overrides[role], empty_ok=False)
                settings.update(patch)
                settings = validate_settings(settings)
            # Never emit a lone gfx1100 voltage offset: the November A/B
            # found it increased power without a matching SCLK cap.
            if arch == "gfx1100" and "vddgfx_offset_mv" in settings and "sclk_max_mhz" not in settings:
                raise PowerPlanRefused(
                    f"{bdf}: gfx1100 VDDGFX offset requires sclk_max_mhz"
                )
            if mode == "observe":
                settings = {}
            elif settings:
                q = (qualifications or {}).get(bdf)
                if (
                    q is None
                    or not q.qualified
                    or q.bdf != bdf
                    or q.arch != arch
                    or q.profile != profile.name
                    or q.settings != settings
                    or not q.evidence
                    or not q.build_id
                    or not q.driver_id
                ):
                    raise PowerPlanRefused(
                        f"{bdf}: explicit apply needs a matching qualified per-device calibration"
                    )
                _check_probed(bdf, settings, (probes or {}).get(bdf))
            resolved.append(PowerDevice(role, bdf, settings))
    except ProfileError as exc:
        raise PowerPlanRefused(str(exc)) from exc
    return PowerPlan(mode, profile.name, digest, tuple(resolved))
