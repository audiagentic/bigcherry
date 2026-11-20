"""MRC04 platform calibration record contract and offline operator store.

The native engine owns production TOML updates; this package validates and
queries them without importing Python into the engine. New platform-native
JSON records use atomic, locked replace under a content-addressed key.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import re
import tempfile
import tomllib
from pathlib import Path
from typing import Any

from .run_records import RecordError, _check_schema

SCHEMA_DIR = Path(__file__).parent / "schemas"
_SHA = re.compile(r"[0-9a-f]{64}\Z")
_KIND = re.compile(r"[a-z][a-z0-9_.-]+\Z")
_MISSING = object()


class CalibrationError(ValueError):
    """Invalid calibration record, key, provenance or unsafe store operation."""


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def key_declaration(kind: str) -> dict[str, Any]:
    """The versioned per-kind allowlist: never infer the key from provenance."""
    declarations = _load(SCHEMA_DIR / "calibration-kinds-v1.json")["kinds"]
    if kind not in declarations:
        raise CalibrationError(f"unknown calibration kind: {kind}")
    return declarations[kind]


def _bytes(obj: object) -> bytes:
    # This is the nlohmann::json object-dump SHA-256 convention already used by
    # Radiance policy_calibration_key: lexicographically sorted, no spaces.
    try:
        return json.dumps(
            obj,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise CalibrationError("calibration key: unsupported JSON input") from exc


def key_of(kind: str, inputs: dict[str, Any]) -> dict[str, Any]:
    declaration = key_declaration(kind)
    fields = declaration["key_fields"]
    if not isinstance(inputs, dict) or any(f not in inputs for f in fields):
        raise CalibrationError("calibration key: missing declared inputs")
    selected = {field: inputs[field] for field in fields}

    # Floats are not part of the v2 Radiance scope; refuse cross-language
    # ambiguous representations (including negative zero and exponent forms).
    def check(value: Any) -> None:
        if isinstance(value, float):
            raise CalibrationError("calibration key: floating-point inputs unsupported")
        if isinstance(value, list):
            for item in value:
                check(item)
        elif isinstance(value, dict):
            for item in value.values():
                check(item)

    check(selected)
    return {
        "version": declaration["key_version"],
        "fields": list(fields),
        "inputs": selected,
        "sha256": hashlib.sha256(_bytes(selected)).hexdigest(),
    }


def new_record(
    kind: str,
    inputs: dict[str, Any],
    quantities: dict[str, Any],
    provenance: dict[str, Any],
) -> dict[str, Any]:
    record = {
        "schema_version": 1,
        "kind": kind,
        "key": key_of(kind, inputs),
        "quantities": quantities,
        "provenance": provenance,
    }
    return validate_record(record)


def _validate_quantities(kind: str, quantities: dict[str, Any]) -> None:
    """Check the kind's declared measurement/absent partition.

    This preserves Radiance's native parser semantics without importing the
    engine into the platform: the allowed quantity names live in the registry.
    """
    declaration = key_declaration(kind)
    names = set(declaration.get("quantity_fields", []))
    measured = quantities["measurements"]
    absent = quantities["absent"]
    if names:
        used = set(measured) | set(absent)
        if used != names or set(measured) & set(absent):
            raise CalibrationError(
                f"calibration quantities: missing, duplicate or undeclared fields "
                f"{sorted(names ^ used)}"
            )
        zero_allowed = set(declaration.get("zero_allowed", []))
        for key, value in measured.items():
            if key == "effect_window_noise_ladder":
                if not isinstance(value, list) or len(value) < 3:
                    raise CalibrationError(
                        "effect_window_noise_ladder: needs at least three observations"
                    )
                previous = 0
                for pair in value:
                    if (
                        not isinstance(pair, list)
                        or len(pair) != 2
                        or any(
                            type(x) is not int or x < 0 or x > (1 << 53) - 1
                            for x in pair
                        )
                        or pair[0] <= previous
                    ):
                        raise CalibrationError(
                            "effect_window_noise_ladder: invalid rows/noise"
                        )
                    previous = pair[0]
            elif (
                type(value) is not int
                or value < (0 if key in zero_allowed else 1)
                or value > (1 << 53) - 1
            ):
                raise CalibrationError(f"calibration quantity: invalid {key}")
        if measured.get("heat_watermarked", 0) > 1:
            raise CalibrationError(
                "calibration quantity: heat_watermarked must be 0 or 1"
            )
        if (
            measured.get("marginal_error_ppm", 0) >= 1_000_000
            or measured.get("tail_share_gain_ppm", 0) >= 1_000_000
        ):
            raise CalibrationError("calibration quantity: invalid fractional reading")
        if (
            "txn_p50_ns" in measured
            and "txn_p95_ns" in measured
            and measured["txn_p50_ns"] > measured["txn_p95_ns"]
        ):
            raise CalibrationError(
                "calibration quantity: txn_p50_ns exceeds txn_p95_ns"
            )


def validate_record(record: Any) -> dict[str, Any]:
    """Validate structure, declared key scope and the hash, not just JSON shape."""
    schema = _load(SCHEMA_DIR / "calibration-record-v1.schema.json")
    try:
        _check_schema(record, schema, schema, "calibration")
    except (RecordError, KeyError) as exc:
        raise CalibrationError(str(exc)) from exc
    key = record["key"]
    expected = key_of(record["kind"], key["inputs"])
    if key != expected:
        raise CalibrationError(
            "calibration key: declared fields/version/digest mismatch"
        )
    if not key["inputs"].get("policy_key_version") == key["version"]:
        raise CalibrationError("calibration key: policy key version mismatch")
    _validate_quantities(record["kind"], record["quantities"])
    try:
        from datetime import date

        date.fromisoformat(record["provenance"]["recorded_at"])
    except ValueError as exc:
        raise CalibrationError("calibration provenance: invalid date") from exc
    return record


def _nonduplicate(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise CalibrationError(f"duplicate JSON input key: {key}")
        value[key] = item
    return value


def read_native_toml(path: Path) -> list[dict[str, Any]]:
    """Translate the native TOML index to platform records.

    Every row must carry calibration_inputs; a row without them is a format this reader does not know and the read
    is refused (the native ``bc-serve calibration-prune`` removes such rows). Malformed rows that do carry them get an invalid verdict
    rather than activating. No native cache is mutated by this read.
    """
    path = Path(path)
    if not path.is_file() or path.is_symlink():
        raise CalibrationError(f"missing or unsafe calibration file: {path}")
    try:
        raw = tomllib.loads(path.read_text(encoding="utf-8"))
    except (ValueError, UnicodeError, OSError) as exc:
        raise CalibrationError(f"invalid calibration TOML: {exc}") from exc
    if raw.get("schema_version") != 1:
        raise CalibrationError("unsupported native TOML store schema_version")
    entries = raw.get("live_policy_record", [])
    if not isinstance(entries, list):
        raise CalibrationError("live_policy_record must be a list")
    seen: set[str] = set()
    rows: list[dict[str, Any]] = []
    metadata = {
        "plan_cache_key",
        "record_schema_version",
        "calibration_kind",
        "calibration_key_fields",
        "calibration_inputs",
        "source_plan_key",
        "source_engine_build",
        "source_revision",
        "run_id",
        "host_facts_digest",
        "layout",
        "objective",
        "seeded",
        "evidence",
        "absent",
    }
    for index, entry in enumerate(entries):
        if not isinstance(entry, dict):
            raise CalibrationError(f"live_policy_record[{index}]: expected a table")
        digest = entry.get("plan_cache_key")
        if not isinstance(digest, str) or not _SHA.fullmatch(digest):
            raise CalibrationError(
                f"live_policy_record[{index}]: invalid plan_cache_key"
            )
        if digest in seen:
            raise CalibrationError(
                f"live_policy_record[{index}]: duplicate plan_cache_key"
            )
        seen.add(digest)
        if "calibration_inputs" not in entry or "record_schema_version" not in entry:
            raise CalibrationError(
                f"live_policy_record[{index}]: unsupported record format (needs calibration_inputs and "
                "record_schema_version 1); remove it with `bc-serve calibration-prune`"
            )
        info = {
            "kind": "radiance.live_policy",
            "key": digest,
            "index": index,
            "status": "invalid",
            "reason": "unchecked",
            "record": None,
            "native": dict(entry),
        }
        try:
            kind = "radiance.live_policy"
            if type(entry["record_schema_version"]) is not int or entry["record_schema_version"] != 1:
                raise CalibrationError("unsupported record_schema_version")
            if entry["calibration_kind"] != kind:
                raise CalibrationError("unexpected calibration kind")
            if entry["calibration_key_fields"] != key_declaration(kind)["key_fields"]:
                raise CalibrationError("calibration_key_fields differs from declaration")
            inputs = json.loads(
                entry["calibration_inputs"], object_pairs_hook=_nonduplicate
            )
            declared = key_declaration(kind)["key_fields"]
            if not isinstance(inputs, dict) or set(inputs) != set(declared):
                raise CalibrationError(
                    "calibration_inputs differs from declared narrow scope"
                )
            key = key_of(kind, inputs)
            if key["sha256"] != digest:
                raise CalibrationError(
                    "record digest differs from stored plan_cache_key"
                )
            if (
                entry.get("layout") != inputs["layout"]
                or entry.get("objective") != inputs["objective"]
            ):
                raise CalibrationError("layout/objective differs from declared scope")
            source_plan = entry["source_plan_key"]
            source_build = entry["source_engine_build"]
            if (
                not isinstance(source_plan, str)
                or not _SHA.fullmatch(source_plan)
                or not isinstance(source_build, str)
                or not _SHA.fullmatch(source_build)
            ):
                raise CalibrationError("source plan/build identities must be SHA-256")
            quantities = {
                "measurements": {
                    name: value for name, value in entry.items() if name not in metadata
                },
                "absent": entry.get("absent", {}),
            }
            provenance = {
                "build_id": source_build,
                "plan_key": source_plan,
                "recorded_at": entry["seeded"],
            }
            provenance.update(
                source_revision=entry["source_revision"],
                run_id=entry["run_id"],
                host_facts_digest=entry["host_facts_digest"],
            )
            record = new_record(kind, inputs, quantities, provenance)
            record_format = "native_v1"
            info.update(
                status="current", reason=None, record=record, format=record_format
            )
        except (CalibrationError, KeyError, ValueError, TypeError) as exc:
            info.update(status="invalid", reason=str(exc))
        rows.append(info)
    return rows


def _json_records(root: Path) -> list[dict[str, Any]]:
    if root.is_symlink():
        raise CalibrationError("unsafe calibration store directory")
    return CalibrationStore(root).list()


class CalibrationStore:
    """Engine-agnostic JSON index with locked atomic append or explicit replace.

    Source provenance is never part of lookup key. Both keys and selected
    filenames are checked before use; no symlinks are followed for records.
    """

    def __init__(self, root: Path):
        self.root = Path(root)

    def _path(self, kind: str, digest: str) -> Path:
        key_declaration(kind)
        if not _KIND.fullmatch(kind) or not _SHA.fullmatch(digest):
            raise CalibrationError("unsafe calibration store kind/key")
        return self.root / "records" / kind / f"{digest}.json"

    def get(self, kind: str, digest: str) -> dict[str, Any] | None:
        path = self._path(kind, digest)
        if (
            self.root.is_symlink()
            or (self.root / "records").is_symlink()
            or path.parent.is_symlink()
        ):
            raise CalibrationError("unsafe calibration store directory")
        if not path.exists():
            return None
        if path.is_symlink() or not path.is_file():
            raise CalibrationError(f"unsafe calibration file: {path}")
        try:
            data = json.loads(
                path.read_text(encoding="utf-8"), object_pairs_hook=_nonduplicate
            )
            validate_record(data)
        except (OSError, ValueError, TypeError) as exc:
            raise CalibrationError(f"invalid stored calibration {path}: {exc}") from exc
        if data["kind"] != kind or data["key"]["sha256"] != digest:
            raise CalibrationError(
                "stored calibration path does not match record identity"
            )
        return data

    def list(self) -> list[dict[str, Any]]:
        folder = self.root / "records"
        if not folder.exists():
            return []
        if folder.is_symlink() or not folder.is_dir():
            raise CalibrationError("unsafe calibration records directory")
        rows = []
        for kind_dir in sorted(folder.iterdir()):
            if kind_dir.is_symlink() or not kind_dir.is_dir():
                raise CalibrationError(f"unsafe calibration kind directory: {kind_dir}")
            for path in sorted(kind_dir.iterdir()):
                if path.suffix != ".json":
                    raise CalibrationError(f"unexpected calibration entry: {path}")
                row = self.get(kind_dir.name, path.stem)
                if row:
                    rows.append(row)
        return rows

    def put(self, record: dict[str, Any], *, replace: bool = False) -> Path:
        validate_record(record)
        path = self._path(record["kind"], record["key"]["sha256"])
        path.parent.mkdir(parents=True, exist_ok=True)
        if (
            path.parent.is_symlink()
            or self.root.is_symlink()
            or (self.root / "records").is_symlink()
        ):
            raise CalibrationError("unsafe calibration store directory")
        flags = os.O_CREAT | os.O_RDWR | os.O_CLOEXEC | os.O_NOFOLLOW
        lock_fd = os.open(self.root / ".lock", flags, 0o600)
        try:
            fcntl.flock(lock_fd, fcntl.LOCK_EX)
            if path.exists() and not replace:
                raise CalibrationError(
                    "calibration record already exists (use replace=True)"
                )
            if path.is_symlink():
                raise CalibrationError("unsafe calibration record symlink")
            payload = _bytes(record) + b"\n"
            fd, name = tempfile.mkstemp(prefix=".calibration-", dir=path.parent)
            try:
                with os.fdopen(fd, "wb") as out:
                    out.write(payload)
                    out.flush()
                    os.fsync(out.fileno())
                os.replace(name, path)
                parent_fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
                try:
                    os.fsync(parent_fd)
                finally:
                    os.close(parent_fd)
            finally:
                if os.path.exists(name):
                    os.unlink(name)
        finally:
            fcntl.flock(lock_fd, fcntl.LOCK_UN)
            os.close(lock_fd)
        return path
