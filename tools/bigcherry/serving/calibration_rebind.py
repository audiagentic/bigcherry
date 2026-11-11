"""Audited, immutable S8 evidence re-binding across unaffected build bumps.

No wildcard build scope: a rebind explicitly identifies an old record, exact
old/new digests and commits, all changed paths and a reviewed reason. Unknown
or measurement-affecting source changes require a new hardware measurement.
A rebind never extends the original evidence's validity window.
"""
from __future__ import annotations

import hashlib
import subprocess
import time
from pathlib import Path

from .calibration_store import (
    CalibrationInvalid,
    _read,
    canonical,
    select_records,
    validate_record,
    validate_scope,
)

_KIND = "s8_calibration_rebind"
_BUILD_FIELDS = ("engine_build_digest", "helper_build_digest")
_REBIND_FIELDS = frozenset((
    "schema_version", "kind", "source_record_sha256", "key",
    "old_scope", "new_scope", "old_commit", "new_commit",
    "changed_paths", "reason", "created_at_unix_s",
))
# Only changes that cannot change engine/helper code, kernel build,
# measurement instrumentation or selected serving policy are automatically
# known-unaffected. Unknown paths fail closed rather than taking an assertion
# from a rebind submitter as proof.
_SAFE_PREFIXES = ("docs/", "tools/tests/", "tests/", "engines/llamacpp/", ".github/ISSUE_TEMPLATE/")
_SAFE_FILES = frozenset((
    "README.md", "CHANGELOG.md", ".release-please-manifest.json",
    "engines/radiance/CHANGELOG.md", "engines/radiance/README.md",
))


def _sha(data: object) -> str:
    return hashlib.sha256(canonical(data)).hexdigest()


def _unchanged_measurement_scope(old: dict, new: dict) -> None:
    validate_scope(old)
    validate_scope(new)
    if any(old[name] != new[name] for name in old if name not in _BUILD_FIELDS):
        raise CalibrationInvalid(
            "S13 rebind refused: model/layout/inventory/owner-map/"
            "context/retention/KV scope changed; remeasure on hardware"
        )
    if all(old[k] == new[k] for k in _BUILD_FIELDS):
        raise CalibrationInvalid("S13 rebind refused: no build digest changed")


def _safe_path(path: str) -> bool:
    if not isinstance(path, str) or not path or path.startswith("/") or "\\" in path:
        return False
    if any(p in ("", ".", "..") for p in path.split("/")):
        return False
    return path in _SAFE_FILES or any(path.startswith(prefix) for prefix in _SAFE_PREFIXES)


def _git(repo: Path, *args: str) -> bytes:
    try:
        return subprocess.run(
            ["git", "-C", str(repo), *args], check=True,
            capture_output=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError) as exc:
        raise CalibrationInvalid(f"S13 rebind git audit unavailable: {args}") from exc


def audit_unaffected(repo: Path, old_ref: str, new_ref: str):
    """Return resolved source revisions and all changed paths, or refuse."""
    old = _git(repo, "rev-parse", "--verify", old_ref + "^{commit}").decode().strip()
    new = _git(repo, "rev-parse", "--verify", new_ref + "^{commit}").decode().strip()
    if old == new:
        raise CalibrationInvalid("S13 rebind refused: identical source revisions")
    changed = _git(repo, "diff", "--name-only", "-z", old, new)
    paths = tuple(sorted({p.decode("utf-8") for p in changed.split(b"\0") if p}))
    unsafe = [path for path in paths if not _safe_path(path)]
    if unsafe:
        raise CalibrationInvalid(
            "S13 rebind refused: changes may affect S8 measurements: " + ", ".join(unsafe)
        )
    # A build bump with no source change must be independently verified by
    # build provenance; we do not certify changed compiler/flags/pin silently.
    if not paths:
        raise CalibrationInvalid(
            "S13 rebind refused: no auditable non-runtime source changes; "
            "verify build/toolchain/pin separately or remeasure"
        )
    return old, new, paths


def read_native_scope(path: Path) -> dict:
    native = _read(path)
    if (not isinstance(native, dict) or
            native.get("kind") != "bc_serve_v2_measured_inputs" or
            type(native.get("schema_version")) is not int or
            native["schema_version"] != 1 or
            not isinstance(native.get("planning_key"), str) or
            len(native["planning_key"]) != 64 or
            any(ch not in "0123456789abcdef" for ch in native["planning_key"]) or
            not isinstance(native.get("provenance"), str) or
            not native["provenance"].strip() or
            not isinstance(native.get("scope"), dict) or
            not isinstance(native.get("resource_inputs"), dict) or
            native["resource_inputs"].get("owner_map_digest") != native["scope"].get("owner_map_digest") or
            any(native["resource_inputs"].get(name) != native["scope"].get(name)
                for name in ("kv_bytes_per_token", "context_target_tokens", "retention_target_tokens"))):
        raise CalibrationInvalid("S13 rebind requires a fresh native P1 measurement scope")
    return validate_scope(native["scope"])


def create_rebind(
    source_record: Path,
    target_measurement: Path,
    store: Path,
    *,
    repo: Path,
    old_ref: str,
    new_ref: str,
    reason: str,
    now_unix_s: int | None = None,
) -> Path:
    """Generate one immutable rebind proof from Git and the new native P1 record."""
    now = int(time.time()) if now_unix_s is None else now_unix_s
    if not isinstance(reason, str) or len(reason.strip()) < 12:
        raise CalibrationInvalid("S13 rebind reason must explain why evidence is unaffected")
    record = _read(source_record)
    validate_record(record, now_unix_s=now)
    new_scope = read_native_scope(target_measurement)
    _unchanged_measurement_scope(record["scope"], new_scope)
    old, new, paths = audit_unaffected(repo, old_ref, new_ref)
    # Input must be from this store, not an unreviewed arbitrary JSON file.
    resolved_source = Path(source_record).resolve(strict=True)
    resolved_store = Path(store).resolve(strict=True)
    if resolved_source.parent != resolved_store or resolved_source.suffix != ".json":
        raise CalibrationInvalid("S13 rebind source must be a P2 record inside the store")
    entry = {
        "schema_version": 1,
        "kind": _KIND,
        "source_record_sha256": _sha(record),
        "key": record["key"],
        "old_scope": record["scope"],
        "new_scope": new_scope,
        "old_commit": old,
        "new_commit": new,
        "changed_paths": list(paths),
        "reason": reason.strip(),
        "created_at_unix_s": now,
    }
    directory = Path(store) / "rebinds"
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / (_sha(entry) + ".json")
    if target.exists():
        if _read(target) != entry:
            raise CalibrationInvalid("S13 rebind evidence hash collision")
        return target
    with target.open("x", encoding="utf-8") as out:
        out.write(canonical(entry).decode("utf-8") + "\n")
    return target


def validate_rebind(entry: object, *, source_record: dict,
                    new_scope: dict, now_unix_s: int) -> None:
    if not isinstance(entry, dict) or set(entry) != _REBIND_FIELDS:
        raise CalibrationInvalid("S13 malformed rebind entry")
    if (type(entry["schema_version"]) is not int or entry["schema_version"] != 1 or
            entry["kind"] != _KIND or entry["key"] != source_record["key"] or
            entry["source_record_sha256"] != _sha(source_record) or
            entry["old_scope"] != source_record["scope"] or
            entry["new_scope"] != new_scope):
        raise CalibrationInvalid("S13 rebind entry does not bind original record and target")
    _unchanged_measurement_scope(entry["old_scope"], entry["new_scope"])
    for ref in (entry["old_commit"], entry["new_commit"]):
        if not isinstance(ref, str) or len(ref) != 40 or any(ch not in "0123456789abcdef" for ch in ref):
            raise CalibrationInvalid("S13 rebind missing resolved Git commit identity")
    if entry["old_commit"] == entry["new_commit"]:
        raise CalibrationInvalid("S13 rebind source commits identical")
    paths = entry["changed_paths"]
    if not isinstance(paths, list) or not paths or paths != sorted(set(paths)) or not all(_safe_path(p) for p in paths):
        raise CalibrationInvalid("S13 rebind includes unknown/measurement-affecting changes")
    if (not isinstance(entry["reason"], str) or len(entry["reason"].strip()) < 12 or
            type(entry["created_at_unix_s"]) is not int or
            entry["created_at_unix_s"] <= 0 or entry["created_at_unix_s"] > now_unix_s):
        raise CalibrationInvalid("S13 rebind reason/date is invalid")
    validate_record(source_record, now_unix_s=now_unix_s)


def select_with_rebind(
    store: Path, *, scope: dict, required_keys,
    now_unix_s: int,
) -> tuple[dict, ...]:
    """Select true direct records first; otherwise require exact rebind proof.

    An original selected setting is not updated; only its build scope and
    evidence_id are bound to the new build via a named immutable receipt.
    """
    validate_scope(scope)
    store = Path(store)
    output = []
    for key in required_keys:
        try:
            selected = select_records(store, scope=scope, required_keys=(key,),
                                      now_unix_s=now_unix_s)
        except CalibrationInvalid as exc:
            if not str(exc).startswith("missing S8 calibration:"):
                raise
        else:
            output.extend(selected)
            continue
        candidates = []
        for source_path in sorted(store.glob("*.json")):
            record = _read(source_path)
            if not isinstance(record, dict) or record.get("key") != key:
                continue
            if not isinstance(record.get("scope"), dict):
                raise CalibrationInvalid(f"S13 {key}: malformed source scope")
            try:
                _unchanged_measurement_scope(record["scope"], scope)
            except CalibrationInvalid:
                continue
            for proof_path in sorted((store / "rebinds").glob("*.json")):
                proof = _read(proof_path)
                if not isinstance(proof, dict) or proof.get("source_record_sha256") != _sha(record):
                    continue
                if proof.get("key") != key or proof.get("new_scope") != scope:
                    continue
                if proof_path.stem != _sha(proof):
                    raise CalibrationInvalid(f"S13 {key}: tampered immutable rebind receipt")
                validate_rebind(proof, source_record=record, new_scope=scope,
                                now_unix_s=now_unix_s)
                # Preserve source run, original measurement date and validity.
                # Use the rebind receipt as additional explicit evidence.
                bound = {
                    **record, "scope": dict(scope),
                    "evidence_id": record["evidence_id"] + " | rebinds/" + proof_path.name,
                    "method": record["method"] + " | explicit rebind: " + proof["reason"],
                }
                candidates.append(bound)
        if len(candidates) != 1:
            qualifier = "missing" if not candidates else "ambiguous"
            raise CalibrationInvalid(f"S13 {key}: {qualifier} exact-scope rebind evidence")
        output.append(candidates[0])
    return tuple(output)


def main(argv=None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        description="Audited S8 build-only evidence rebind (never a wildcard)"
    )
    parser.add_argument("--source-record", type=Path, required=True)
    parser.add_argument("--target-measurement", type=Path, required=True)
    parser.add_argument("--store", type=Path, required=True)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--old-rev", required=True)
    parser.add_argument("--new-rev", required=True)
    parser.add_argument("--reason", required=True)
    args = parser.parse_args(argv)
    try:
        print(create_rebind(
            args.source_record, args.target_measurement, args.store,
            repo=args.repo, old_ref=args.old_rev, new_ref=args.new_rev,
            reason=args.reason,
        ))
        return 0
    except (CalibrationInvalid, OSError, ValueError) as exc:
        parser.error(str(exc))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
