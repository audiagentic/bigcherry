"""PWR08: card-free named power profile data, with canonical content digests.

A named profile is operator *intent*, not authority to write sysfs. Rules are
architecture and role filters, never PCI locators; calibration is keyed by
live inventory BDF and must be separately qualified before explicit apply.
All built-in profiles default to observe; a profile load never probes a GPU.

The v2 wire digest is over the JCS-compatible, integer-only resolved
{"name": name, "roles": ...} shape AFTER device calibration and BEFORE intent
overrides. The source digest detects edits to the named profile rules.
"""

from __future__ import annotations

import hashlib
import json
import re
import tomllib
from dataclasses import dataclass
from pathlib import Path

_NAME = re.compile(r"^[a-z][a-z0-9_-]{0,31}$")
_ROLE = re.compile(r"^(engine|helper\*|helper[0-9]+)$")
_ARCH = re.compile(r"^gfx[0-9a-f]+$")
_CONTROLS = frozenset(("vddgfx_offset_mv", "sclk_max_mhz", "mclk_max_mhz", "sclk_offset_mhz", "power_cap_w"))
_IJSON_MAX = (1 << 53) - 1



class ProfileError(ValueError):
    """Malformed profile data or unknown named selection."""


def _canonical(document: dict) -> bytes:
    # Only ASCII keys, string values and integer controls are permitted.
    # Thus sorted compact JSON is byte-for-byte JCS for this restricted data.
    return json.dumps(
        document,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _digest(document: dict) -> str:
    return "sha256:" + hashlib.sha256(_canonical(document)).hexdigest()


def validate_settings(settings: object, *, empty_ok: bool = True) -> dict[str, int]:
    if not isinstance(settings, dict):
        raise ProfileError("power settings must be an object")
    if not settings and not empty_ok:
        raise ProfileError("a profile rule needs at least one setting")
    if unknown := set(settings) - _CONTROLS:
        raise ProfileError(f"unknown power setting(s): {sorted(unknown)}")
    for k, v in settings.items():
        # Hardware-specific bounds come ONLY from the fresh PWR07 BDF probe.
        if type(v) is not int or abs(v) > _IJSON_MAX or (
            k in ("sclk_max_mhz", "mclk_max_mhz", "power_cap_w") and v <= 0
        ):
            raise ProfileError(f"{k} must be a typed I-JSON-safe power setting")
    if "sclk_max_mhz" in settings and "sclk_offset_mhz" in settings:
        raise ProfileError("sclk_max_mhz and sclk_offset_mhz are mutually exclusive")
    return dict(settings)


@dataclass(frozen=True)
class Rule:
    role: str
    arch: str
    settings: dict[str, int]
    evidence: str

    def matches(self, role: str, arch: str) -> bool:
        return self.arch == arch and (
            self.role == role
            or self.role == "helper*"
            and bool(re.fullmatch(r"helper[0-9]+", role))
        )


@dataclass(frozen=True)
class Profile:
    name: str
    default_mode: str
    rules: tuple[Rule, ...]
    source_digest: str

    def settings_for(self, role: str, arch: str) -> dict[str, int]:
        if not _ROLE.fullmatch(role) or role == "helper*" or not _ARCH.fullmatch(arch):
            raise ProfileError(f"invalid concrete power target {role!r}/{arch!r}")
        # Wildcard rule is less specific than a matching exact-role rule.
        rules = sorted(
            (r for r in self.rules if r.matches(role, arch)),
            key=lambda r: r.role != "helper*",
        )
        merged: dict[str, int] = {}
        for rule in rules:
            merged.update(rule.settings)
        return validate_settings(merged)


def resolved_profile_digest(name: str, roles: dict[str, dict[str, int]]) -> str:
    """The v2 profile_digest, after BDF calibration and before CLI overrides."""
    if not _NAME.fullmatch(name):
        raise ProfileError(f"invalid profile name {name!r}")
    if not roles:
        raise ProfileError("profile digest needs resolved roles")
    for role, setting in roles.items():
        if not _ROLE.fullmatch(role) or role == "helper*":
            raise ProfileError(f"invalid resolved role {role!r}")
        validate_settings(setting)
    return _digest({"name": name, "roles": roles})


def load_profiles(path: Path) -> dict[str, Profile]:
    source = tomllib.loads(Path(path).read_text(encoding="utf-8"))
    if set(source) != {"profiles"} or not isinstance(source["profiles"], dict):
        raise ProfileError("expected exactly one [profiles] table")
    result = {}
    for name, raw in source["profiles"].items():
        if not _NAME.fullmatch(name) or not isinstance(raw, dict):
            raise ProfileError(f"invalid named power profile {name!r}")
        if extra := set(raw) - {"default_mode", "rules"}:
            raise ProfileError(f"{name}: unknown keys {sorted(extra)}")
        mode = raw.get("default_mode", "observe")
        if mode != "observe":
            raise ProfileError(f"{name}: built-in profile cannot auto-apply power")
        entries = raw.get("rules", [])
        if not isinstance(entries, list):
            raise ProfileError(f"{name}: rules must be a list")
        rules = []
        for i, entry in enumerate(entries):
            if not isinstance(entry, dict) or set(entry) != {
                "role",
                "arch",
                "evidence",
                "settings",
            }:
                raise ProfileError(
                    f"{name} rule {i}: expected role, arch, evidence, settings"
                )
            role, arch, evidence = entry["role"], entry["arch"], entry["evidence"]
            if not isinstance(role, str) or not _ROLE.fullmatch(role):
                raise ProfileError(f"{name} rule {i}: invalid role")
            if not isinstance(arch, str) or not _ARCH.fullmatch(arch):
                raise ProfileError(f"{name} rule {i}: invalid architecture")
            if (
                not isinstance(evidence, str)
                or not evidence
                or evidence.startswith("/")
            ):
                raise ProfileError(
                    f"{name} rule {i}: missing or host-specific evidence"
                )
            settings = validate_settings(entry["settings"], empty_ok=False)
            rules.append(Rule(role, arch, settings, evidence))
        projection = {
            "name": name,
            "default_mode": mode,
            "rules": [
                {
                    "role": r.role,
                    "arch": r.arch,
                    "settings": r.settings,
                    "evidence": r.evidence,
                }
                for r in rules
            ],
        }
        result[name] = Profile(name, mode, tuple(rules), _digest(projection))
    if "stock" not in result or result["stock"].rules:
        raise ProfileError("stock must exist and have no settings rules")
    return result


def get_profile(profiles: dict[str, Profile], name: str) -> Profile:
    try:
        return profiles[name]
    except KeyError as exc:
        raise ProfileError(f"unknown power profile {name!r}") from exc
