"""RSC11 S12: read *only* the immutable v2 manifest's resolved power plan.

The supervisor accepts no second BDF->settings carrier. The strict C++ v2
reader/validator (S9/S10/S13) must have authenticated the full manifest before
the injected planning step returns. This boundary rechecks the subset that
controls hardware, so a corrupted or reordered manifest never causes writes.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from ..hardware.gpu_power import Setting
from ..hardware.power_receipt import manifest_binding, read_json

_BDF = re.compile(r"^[0-9a-f]{4}:[0-9a-f]{2}:[0-9a-f]{2}\.[0-7]$")
_NAME = re.compile(r"^[a-z][a-z0-9_-]{0,31}$")
_DIGEST = re.compile(r"^sha256:[0-9a-f]{64}$")
_ALLOWED = {"vddgfx_offset_mv", "sclk_max_mhz", "mclk_max_mhz", "sclk_offset_mhz", "power_cap_w"}
_IJSON_MAX = (1 << 53) - 1



class PowerManifestRefused(ValueError):
    """v2 manifest power section is missing, stale, unsafe or inconsistent."""


@dataclass(frozen=True)
class ManifestPower:
    mode: str
    profile: str
    profile_digest: str
    manifest_digest: str
    devices: tuple[dict, ...]
    requested: dict[str, Setting]

    def receipt_request(self) -> dict:
        return {
            "mode": self.mode,
            "profile": self.profile,
            "profile_digest": self.profile_digest,
            "devices": [dict(d) for d in self.devices],
        }


def _settings(data: object) -> Setting | None:
    if not isinstance(data, dict):
        raise PowerManifestRefused("manifest device settings must be an object")
    if keys := set(data) - _ALLOWED:
        raise PowerManifestRefused(f"unknown power setting(s): {sorted(keys)}")
    for key, value in data.items():
        # S10 uses a fresh PWR07 per-BDF probe for hardware ranges; the
        # S12 adapter enforces only the shared wire type/safe integer bounds.
        if type(value) is not int or abs(value) > _IJSON_MAX or (
            key in ("sclk_max_mhz", "mclk_max_mhz", "power_cap_w") and value <= 0
        ):
            raise PowerManifestRefused(f"{key} is not a typed I-JSON-safe setting")
    if "sclk_max_mhz" in data and "sclk_offset_mhz" in data:
        raise PowerManifestRefused("both sclk controls requested simultaneously")
    if "sclk_offset_mhz" in data:
        raise PowerManifestRefused(
            "sclk_offset_mhz not implemented by PowerTransaction"
        )
    if not data:
        return None
    return Setting(
        vddgfx_offset_mv=data.get("vddgfx_offset_mv"),
        sclk_max_mhz=data.get("sclk_max_mhz"),
        mclk_max_mhz=data.get("mclk_max_mhz"),
        power_cap_w=data.get("power_cap_w"),
    )


def read_manifest_power(
    path: Path, *, engine_bdfs: tuple[str, ...], helper_bdfs: tuple[str, ...]
) -> ManifestPower:
    """Cross-check every manifest role/BDF; return no external power overrides."""
    try:
        doc = read_json(path)
        meta = doc["metadata"]
        intent = doc["spec"]["intent"]["power"]
        plan = doc["spec"]["plan"]["power"]
        binding = manifest_binding(meta["content_digest"])
        if binding["status"] != "resolved":
            raise PowerManifestRefused("manifest has no canonical content digest")
        if not isinstance(intent, dict) or set(intent) != {"profile", "role_overrides"}:
            raise PowerManifestRefused("v2 intent power has invalid keys")
        if not isinstance(intent["role_overrides"], dict):
            raise PowerManifestRefused("v2 power role overrides must be a map")
        if not isinstance(plan, dict) or set(plan) != {
            "mode",
            "profile",
            "profile_digest",
            "devices",
            "cpu",
        }:
            raise PowerManifestRefused("v2 plan power has invalid keys")
        mode, profile, digest = plan["mode"], plan["profile"], plan["profile_digest"]
        if mode not in ("observe", "apply"):
            raise PowerManifestRefused("unknown v2 power mode")
        if (
            not isinstance(profile, str)
            or not _NAME.fullmatch(profile)
            or profile != intent["profile"]
        ):
            raise PowerManifestRefused("v2 power profile disagrees with intent")
        if not isinstance(digest, str) or not _DIGEST.fullmatch(digest):
            raise PowerManifestRefused("v2 power profile digest invalid")
        if plan["cpu"] != {"settings": {}}:
            raise PowerManifestRefused(
                "CPU power controls have no S12 policy controller"
            )
        expected = [("engine", bdf) for bdf in engine_bdfs] + [
            (f"helper{i}", bdf) for i, bdf in enumerate(helper_bdfs)
        ]
        if not engine_bdfs or len({b for _, b in expected}) != len(expected):
            raise PowerManifestRefused("resolver supplied empty/duplicate device BDFs")
        if not isinstance(plan["devices"], list) or len(plan["devices"]) != len(
            expected
        ):
            raise PowerManifestRefused(
                "power devices do not cover exact serving layout"
            )
        wanted = {role for role, _ in expected}
        if set(intent["role_overrides"]) - wanted:
            raise PowerManifestRefused(
                "intent power override targets a nonselected role"
            )
        for patch in intent["role_overrides"].values():
            if not isinstance(patch, dict) or not patch:
                raise PowerManifestRefused("empty or malformed role power patch")
            _settings(patch)
        writes: dict[str, Setting] = {}
        entries = []
        for (role, bdf), raw in zip(expected, plan["devices"]):
            if not isinstance(raw, dict) or set(raw) != {"role", "bdf", "settings"}:
                raise PowerManifestRefused("invalid power device fields")
            if (raw["role"], raw["bdf"]) != (role, bdf) or not _BDF.fullmatch(bdf):
                raise PowerManifestRefused(
                    "power role/BDF does not match bc-serve inventory"
                )
            setting = _settings(raw["settings"])
            if mode == "observe" and setting is not None:
                raise PowerManifestRefused("observe-mode power settings must be empty")
            if mode == "apply" and setting is not None:
                writes[bdf] = setting
            entries.append(
                {"role": role, "bdf": bdf, "settings": dict(raw["settings"])}
            )
        return ManifestPower(
            mode, profile, digest, binding["content_digest"], tuple(entries), writes
        )
    except (OSError, KeyError, TypeError, AttributeError, ValueError) as exc:
        if isinstance(exc, PowerManifestRefused):
            raise
        raise PowerManifestRefused(f"invalid v2 manifest power section: {exc}") from exc
