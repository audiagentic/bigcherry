"""Audited subject-digest lineage for *mechanical* pin-bump re-anchors.

This is not a blanket waiver for changed patch implementations. A transition
must be recorded with the pin and checksum of the rebase evidence before
build-only carry-forward may consider the predecessor's evidence. All other
validation checks (contracts, activation, architectures, etc.) still apply.
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

SCHEMA = 1
RECORD = Path("evidence") / "lineage.json"
_REASON = "rebase re-anchor"
_HEX64 = re.compile(r"[0-9a-f]{64}\Z")


def _valid_digest(value: object) -> bool:
    return isinstance(value, str) and _HEX64.fullmatch(value) is not None


def load(package_dir: Path) -> tuple[dict[str, str], ...]:
    """Fail closed on missing/malformed evidence; absence means no lineage."""
    path = Path(package_dir) / RECORD
    if not path.is_file():
        return ()
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ValueError(f"{path}: invalid lineage document: {exc}") from exc
    if not isinstance(document, dict) or document.get("schema") != SCHEMA:
        raise ValueError(f"{path}: expected schema {SCHEMA}")
    entries = document.get("transitions")
    if not isinstance(entries, list):
        raise TypeError(f"{path}: transitions must be a list")
    seen: dict[str, str] = {}
    for entry in entries:
        if not isinstance(entry, dict) or set(entry) != {
            "from_digest", "to_digest", "pin", "reason", "rebase_report_sha256",
        }:
            raise ValueError(f"{path}: invalid transition fields")
        before, after = entry["from_digest"], entry["to_digest"]
        if not _valid_digest(before) or not _valid_digest(after) or before == after:
            raise ValueError(f"{path}: invalid or identical digests")
        if not isinstance(entry["pin"], str) or not entry["pin"].strip():
            raise ValueError(f"{path}: pin is required")
        if entry["reason"] != _REASON or not _valid_digest(entry["rebase_report_sha256"]):
            raise ValueError(f"{path}: re-anchor reason and report SHA-256 are required")
        if before in seen:
            raise ValueError(f"{path}: duplicate predecessor digest")
        seen[before] = after
    for before in seen:
        reached = {before}
        cursor = before
        while cursor in seen:
            cursor = seen[cursor]
            if cursor in reached:
                raise ValueError(f"{path}: cyclic lineage")
            reached.add(cursor)
    return tuple(entries)


def is_ancestor(package_dir: Path, *, previous: str, current: str) -> bool:
    """Only an explicitly recorded, complete chain qualifies older subjects."""
    if not _valid_digest(previous) or not _valid_digest(current):
        return False
    transitions = load(package_dir)
    links = {entry["from_digest"]: entry["to_digest"] for entry in transitions}
    cursor = previous
    while cursor in links:
        cursor = links[cursor]
        if cursor == current:
            return True
    return False


def write_reanchor(
    package_dir: Path, *, from_digest: str, to_digest: str,
    pin: str, rebase_report: Path,
) -> Path:
    """Record a reviewed *pure* re-anchor with a real immutable report hash.

    Call only after checking that replacement text and owned files are
    unchanged. This function cannot prove semantic equivalence by itself.
    """
    if not _valid_digest(from_digest) or not _valid_digest(to_digest) or from_digest == to_digest:
        raise ValueError("lineage requires two distinct SHA-256 subject digests")
    raw = Path(rebase_report).read_bytes()
    report = json.loads(raw)
    if not isinstance(report, dict) or not report:
        raise ValueError("rebase report is not an object")
    if not pin.strip():
        raise ValueError("lineage requires the pin")
    entry = {
        "from_digest": from_digest,
        "to_digest": to_digest,
        "pin": pin,
        "reason": _REASON,
        "rebase_report_sha256": hashlib.sha256(raw).hexdigest(),
    }
    existing = list(load(package_dir))
    if any(e["from_digest"] == from_digest for e in existing):
        raise ValueError("predecessor already recorded")
    path = Path(package_dir) / RECORD
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"schema": SCHEMA, "transitions": [*existing, entry]},
                               indent=2, sort_keys=True) + "\n", encoding="utf-8")
    load(package_dir)
    return path
