"""Profile-evidence tier: the lightweight promotion record (QFP18, PA45).

A patch can be promoted on evidence the lab already produces, without a multi-session HI83 contract campaign:
offline mechanics, an activation marker from the run that produced the numbers, a one-binary A/B, and identical
greedy output, on the current pin. A patch that is on by default additionally needs a no-regression run on a second
model.

The record is ``<patch package>/evidence/promotion.json``. It is bound to the pin and to the patch implementation
(the same subject digest HI83 records use, so the STATE change that promotion itself makes does not void it; any
other edit to the patch does, and the evidence has to be confirmed again).

This module holds the record's shape and its checks. ``evidence.verify_validated_patch`` consults it when no HI83
record qualifies; ``release.patch_promote`` writes it.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

SCHEMA = 1
TIER = "profile-evidence"
RECORD = Path("evidence") / "promotion.json"

# every promotion states these, in words a reviewer can check against the run logs
REQUIRED_CHECKS = ("mechanics", "activation", "identity", "ab")
# a default-on patch reaches every model, so it also names a run on a second one
DEFAULT_ON_CHECK = "no-regression"


@dataclass(frozen=True)
class ProfileEvidence:
    patch_id: str
    pinned: str
    subject_digest: str
    default_on: bool
    models: tuple[str, ...]
    checks: dict[str, str]

    def document(self) -> dict[str, object]:
        return {
            "schema": SCHEMA,
            "tier": TIER,
            "patch-id": self.patch_id,
            "pinned": self.pinned,
            "subject-digest": self.subject_digest,
            "default-on": self.default_on,
            "models": list(self.models),
            "checks": dict(sorted(self.checks.items())),
        }


def problems(evidence: ProfileEvidence) -> tuple[str, ...]:
    """What a record lacks. Empty means it qualifies on its own terms (binding is checked by ``verify``)."""
    found: list[str] = []
    if not evidence.models or not all(isinstance(m, str) and m.strip() for m in evidence.models):
        found.append("at least one named model is required")
    for name in REQUIRED_CHECKS:
        if not str(evidence.checks.get(name, "")).strip():
            found.append(f"check {name!r} is missing")
    if evidence.default_on:
        if not str(evidence.checks.get(DEFAULT_ON_CHECK, "")).strip():
            found.append(f"a default-on patch needs check {DEFAULT_ON_CHECK!r} (a run on a second model)")
        if len({m.strip().casefold() for m in evidence.models if isinstance(m, str)}) < 2:
            found.append("a default-on patch needs two distinct named models")
    return tuple(found)


def record_path(package_dir: Path) -> Path:
    return Path(package_dir) / RECORD


def write(package_dir: Path, evidence: ProfileEvidence) -> Path:
    lacking = problems(evidence)
    if lacking:
        raise ValueError("profile evidence is incomplete: " + "; ".join(lacking))
    path = record_path(package_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(evidence.document(), indent=2) + "\n", encoding="utf-8", newline="\n")
    return path


def load(package_dir: Path) -> ProfileEvidence | None:
    """The package's record, or None when it has none. A malformed record raises ValueError."""
    path = record_path(package_dir)
    if not path.is_file():
        return None
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{path}: not valid JSON: {exc}") from exc
    if not isinstance(raw, dict) or raw.get("schema") != SCHEMA or raw.get("tier") != TIER:
        raise ValueError(f"{path}: not a schema-{SCHEMA} {TIER} record")
    checks = raw.get("checks")
    models = raw.get("models")
    if not isinstance(checks, dict) or not isinstance(models, list) or not isinstance(raw.get("default-on"), bool):
        raise ValueError(f"{path}: checks, models and default-on are required")
    return ProfileEvidence(
        patch_id=str(raw.get("patch-id", "")),
        pinned=str(raw.get("pinned", "")),
        subject_digest=str(raw.get("subject-digest", "")),
        default_on=raw["default-on"],
        models=tuple(str(m) for m in models),
        checks={str(k): str(v) for k, v in checks.items()},
    )


def verify(package_dir: Path, *, patch_id: str, pinned_ref: str, subject_digest: str) -> tuple[str, ...] | None:
    """None when the package has no record; otherwise the reasons it does not qualify (empty = it qualifies)."""
    try:
        evidence = load(package_dir)
    except ValueError as exc:
        return (str(exc),)
    if evidence is None:
        return None
    found = list(problems(evidence))
    if evidence.patch_id != patch_id:
        found.append(f"record is for {evidence.patch_id!r}")
    if evidence.pinned != pinned_ref:
        found.append(f"record is for pin {evidence.pinned!r}, current pin is {pinned_ref!r}")
    if evidence.subject_digest != subject_digest:
        found.append("the patch implementation changed since the evidence was taken; confirm it again")
    return tuple(found)
