"""Platform-owned, immutable v1 run records and provenance blobs (MRC01).

Only this library reads the store. Engines provide adapters; runtime binaries
emit inputs as files and never import this Python module. No filename/path hint
is authoritative: every input is retrieved by verified SHA-256 content.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import stat
import tempfile
from collections.abc import Iterator
from datetime import datetime
from pathlib import Path
from typing import Any

from .context import resolve_work_root

SCHEMA_PATH = Path(__file__).parent / "schemas" / "run-record-v1.schema.json"
STORE_DIR = Path("run-records") / "v1"
_HEX64 = re.compile(r"[0-9a-f]{64}\Z")
_METRIC = re.compile(r"[a-z][a-z0-9_]*\.[a-zA-Z_][a-zA-Z0-9_]*\Z")


class RecordError(ValueError):
    """A record, blob or immutable-store entry is missing or invalid."""


def load_schema() -> dict[str, Any]:
    """The published JSON Schema contract, shared by all engine adapters."""
    return json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))


def _json_bytes(value: object) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    except (ValueError, TypeError) as exc:
        raise RecordError(f"not canonical JSON: {exc}") from exc


def with_host_facts(record: dict[str, Any], facts: dict[str, Any] | None = None) -> dict[str, Any]:
    """Copy a measurement-owner's already captured v1 start snapshot.

    Adapters call this before `put_record`; existing stored records are immutable.
    A caller may inject a fixture snapshot, avoiding host access in unit tests.
    """
    from copy import deepcopy

    from .host_facts import validate_start_snapshot

    if facts is None:
        raise RecordError("start host facts must be captured by the measurement owner")
    updated = deepcopy(record)
    try:
        updated["host_facts"] = deepcopy(validate_start_snapshot(facts))
    except ValueError as exc:
        raise RecordError(f"invalid start host facts: {exc}") from exc
    validate_record(updated)
    return updated


def canonical_record_bytes(record: dict[str, Any]) -> bytes:
    validate_record(record)
    return _json_bytes(record)


def _check_schema(value: Any, spec: dict, root: dict, where: str) -> None:
    """Validate the intentionally small JSON Schema vocabulary used by v1.

    This is not a general JSON Schema implementation. The checked vocabulary
    matches the bundled schema; unknown schema keywords fail rather than
    silently weakening the local writer's contract.
    """
    if "$ref" in spec:
        ref = spec["$ref"]
        if not ref.startswith("#/$defs/") or ref[8:] not in root["$defs"]:
            raise RecordError(f"{where}: unsupported schema reference {ref}")
        _check_schema(value, root["$defs"][ref[8:]], root, where)
        return
    allowed = {
        "type",
        "const",
        "enum",
        "pattern",
        "minLength",
        "minimum",
        "required",
        "properties",
        "additionalProperties",
        "items",
        "format",
        "description",
        "propertyNames",
        "minItems",
        "$schema",
        "$id",
        "$defs",
        "title",
    }
    extra = set(spec) - allowed
    if extra:
        raise RecordError(f"{where}: unsupported schema keywords {sorted(extra)}")
    if "const" in spec and (
        type(value) is not type(spec["const"]) or value != spec["const"]
    ):
        raise RecordError(f"{where}: expected constant {spec['const']!r}")
    if "enum" in spec and value not in spec["enum"]:
        raise RecordError(f"{where}: not one of {spec['enum']!r}")
    kind = spec.get("type")
    if kind is not None:
        matches = {
            "object": lambda: isinstance(value, dict),
            "array": lambda: isinstance(value, list),
            "string": lambda: isinstance(value, str),
            "integer": lambda: type(value) is int,
            "number": lambda: type(value) in (int, float) and math.isfinite(value),
            "boolean": lambda: type(value) is bool,
        }
        if kind not in matches or not matches[kind]():
            raise RecordError(f"{where}: expected {kind}")
    if isinstance(value, str):
        if len(value) < spec.get("minLength", 0):
            raise RecordError(f"{where}: string is too short")
        if "pattern" in spec and re.search(spec["pattern"], value) is None:
            raise RecordError(f"{where}: invalid pattern")
        if spec.get("format") == "date-time":
            try:
                stamp = datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ")
            except ValueError as exc:
                raise RecordError(f"{where}: invalid UTC time") from exc
            if stamp.year < 1:
                raise RecordError(f"{where}: invalid year")
    if type(value) in (int, float) and "minimum" in spec and value < spec["minimum"]:
        raise RecordError(f"{where}: below minimum")
    if isinstance(value, dict):
        for name in spec.get("required", []):
            if name not in value:
                raise RecordError(f"{where}: missing {name}")
        defined = spec.get("properties", {})
        for name, item in value.items():
            if not isinstance(name, str):
                raise RecordError(f"{where}: object keys must be strings")
            if "propertyNames" in spec:
                _check_schema(name, spec["propertyNames"], root, f"{where} key")
            rule = defined.get(name, spec.get("additionalProperties", True))
            if rule is False:
                raise RecordError(f"{where}: unexpected key {name}")
            if isinstance(rule, dict):
                _check_schema(item, rule, root, f"{where}.{name}")
    if isinstance(value, list):
        if len(value) < spec.get("minItems", 0):
            raise RecordError(f"{where}: array too short")
        if "items" in spec:
            for index, item in enumerate(value):
                _check_schema(item, spec["items"], root, f"{where}[{index}]")


def validate_record(record: Any) -> dict[str, Any]:
    """Validate the schema and canonical-JSON safety, without touching disk."""
    schema = load_schema()
    _check_schema(record, schema, schema, "record")
    # Arbitrary workload/host details must still be JSON, with no NaN/Infinity.
    _json_bytes(record)
    # MRC02: a Radiance record must carry the SHA-bound measurement start
    # receipt on both read and write. Do not synthesize an old host snapshot.
    from . import engines
    if record["identity"]["engine"] == engines.RADIANCE:
        from .host_facts import validate_start_snapshot

        try:
            validate_start_snapshot(record["host_facts"])
            start = datetime.fromisoformat(record["host_facts"]["captured_at"])
            recorded = datetime.fromisoformat(record["recorded_at"])
            if start > recorded:
                raise ValueError("start receipt is after measured record")
        except (ValueError, TypeError, KeyError) as exc:
            raise RecordError(f"Radiance record requires sealed start host facts: {exc}") from exc

    # MRC05/#1226: exact per-phase metric direction and gate are mandatory.
    if "metric_specs" not in record:
        raise RecordError("metric_specs is required: every metric needs an explicit direction and gate")
    if set(record["metric_specs"]) != set(record["metrics"]):
        raise RecordError("metric_specs groups must match metrics groups")
    for group, values in record["metrics"].items():
        if set(record["metric_specs"][group]) != set(values):
            raise RecordError(f"metric_specs.{group}: direction missing or extra metric")
        for spec in record["metric_specs"][group].values():
            if type(spec["gated"]) is not bool:
                raise RecordError("metric gate must be a boolean")

    # VA25 validity must not contradict its reason codes.
    if "validity" in record:
        validity = record["validity"]
        if (validity["state"] == "valid") != (not validity["reason_codes"]):
            raise RecordError("validity: reason_codes disagree with execution verdict")
    if "calibration_record" in record:
        selection = record["calibration_record"]
        if selection["status"] == "none":
            if set(selection) != {"status", "reason_code", "reason_detail"}:
                raise RecordError("calibration none requires only a reason code and detail")
        elif selection["status"] == "selected":
            required = {"status", "kind", "key_sha256", "record_blob", "quantity_blobs"}
            if set(selection) != required:
                raise RecordError("calibration selected requires kind, key, record_blob and quantity_blobs")
            names = [entry["name"] for entry in selection["quantity_blobs"]]
            if len(names) != len(set(names)):
                raise RecordError("calibration has duplicate quantity blob names")
    return record


def blob_references(record: dict[str, Any]) -> Iterator[dict[str, Any]]:
    """Enumerate every mandatory and optional input in a validated bundle."""
    bundle = record["provenance"]
    for name in ("effective_config", "argv", "environment", "power_settings"):
        yield bundle[name]
    for item in bundle["inputs"]:
        yield item["blob"]
    if record.get("calibration_record", {}).get("status") == "selected":
        selected = record["calibration_record"]
        yield selected["record_blob"]
        for item in selected["quantity_blobs"]:
            yield item["blob"]


def _digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _check_digest(digest: str) -> str:
    if not isinstance(digest, str) or not _HEX64.fullmatch(digest):
        raise RecordError("invalid sha256 digest")
    return digest


def _pairs_no_duplicates(pairs: list[tuple[str, Any]]) -> dict:
    obj = {}
    for key, value in pairs:
        if key in obj:
            raise RecordError(f"duplicate JSON key {key!r}")
        obj[key] = value
    return obj


def _decode_record(raw: bytes) -> dict:
    try:
        record = json.loads(
            raw,
            object_pairs_hook=_pairs_no_duplicates,
            parse_constant=lambda token: (_ for _ in ()).throw(
                RecordError(f"invalid {token}")
            ),
        )
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise RecordError(f"unreadable record JSON: {exc}") from exc
    validate_record(record)
    if canonical_record_bytes(record) != raw:
        raise RecordError("record bytes are not canonical")
    return record


class RunRecordStore:
    """Content-addressed, write-once v1 store rooted at BC_WORK_ROOT."""

    def __init__(self, work_root: Path | str | None = None):
        self.root = resolve_work_root(work_root) / STORE_DIR

    def blob_path(self, digest: str) -> Path:
        digest = _check_digest(digest)
        return self.root / "blobs" / digest[:2] / digest

    def record_path(self, digest: str) -> Path:
        digest = _check_digest(digest)
        return self.root / "records" / digest[:2] / f"{digest}.json"

    def _safe_parents(self, directory: Path, *, create: bool) -> None:
        # Store subdirectories must never be redirected through symlinks.
        for item in (self.root.parent, self.root, directory.parent, directory):
            if create:
                item.mkdir(parents=True, exist_ok=True)
            if item.is_symlink() or not item.is_dir():
                raise RecordError(f"symlink or unsafe store directory: {item}")

    def _read(self, path: Path, expected: str) -> bytes:
        self._safe_parents(path.parent, create=False)
        flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
        try:
            fd = os.open(path, flags)
        except OSError as exc:
            raise RecordError(f"missing or unsafe object {path}: {exc}") from exc
        try:
            if not stat.S_ISREG(os.fstat(fd).st_mode):
                raise RecordError(f"object is not a regular file: {path}")
            with os.fdopen(fd, "rb") as stream:
                fd = -1
                data = stream.read()
        finally:
            if fd != -1:
                os.close(fd)
        if _digest(data) != expected:
            raise RecordError(f"sha256 digest mismatch for {path}")
        return data

    def _put(self, path: Path, data: bytes, digest: str) -> None:
        self._safe_parents(path.parent, create=True)
        fd, name = tempfile.mkstemp(prefix=".tmp-", dir=path.parent)
        try:
            with os.fdopen(fd, "wb") as stream:
                stream.write(data)
                stream.flush()
                os.fsync(stream.fileno())
            try:
                os.link(name, path)  # create-once; never replace existing content
            except FileExistsError:
                if self._read(path, digest) != data:
                    raise RecordError(f"conflicting immutable object at {path}")
            else:
                # A directory sync makes the hard-link publication durable.
                dfd = os.open(path.parent, os.O_RDONLY)
                try:
                    os.fsync(dfd)
                finally:
                    os.close(dfd)
        finally:
            os.unlink(name)

    def put_blob(self, data: bytes) -> dict[str, Any]:
        if not isinstance(data, bytes):
            raise RecordError("blob input must be bytes")
        digest = _digest(data)
        self._put(self.blob_path(digest), data, digest)
        return {"sha256": digest, "size_bytes": len(data)}

    def put_blob_file(self, path: Path) -> dict[str, Any]:
        """Copy inputs by value; the original path is not part of the identity."""
        return self.put_blob(Path(path).read_bytes())

    def get_blob(self, ref: dict[str, Any]) -> bytes:
        if (
            not isinstance(ref, dict)
            or set(ref) != {"sha256", "size_bytes"}
            or type(ref["size_bytes"]) is not int
            or ref["size_bytes"] < 0
        ):
            raise RecordError("malformed blob reference")
        digest = _check_digest(ref["sha256"])
        raw = self._read(self.blob_path(digest), digest)
        if len(raw) != ref["size_bytes"]:
            raise RecordError(f"blob size mismatch for {digest}")
        return raw

    def _check_calibration_record(self, record: dict[str, Any]) -> None:
        selection = record.get("calibration_record", {})
        if selection.get("status") != "selected":
            return
        from . import calibration_records
        try:
            raw = self.get_blob(selection["record_blob"])
            payload = json.loads(
                raw.decode("utf-8"), object_pairs_hook=_pairs_no_duplicates
            )
            calibration_records.validate_record(payload)
        except (ValueError, UnicodeError, KeyError, TypeError) as exc:
            raise RecordError(f"calibration record blob is invalid: {exc}") from exc
        if (selection["kind"] != payload["kind"] or
                selection["key_sha256"] != payload["key"]["sha256"]):
            raise RecordError("calibration record selected key/kind differs from blob")

    def put_record(self, record: dict[str, Any]) -> str:
        data = canonical_record_bytes(record)
        for ref in blob_references(record):
            self.get_blob(ref)
        self._check_calibration_record(record)
        digest = _digest(data)
        self._put(self.record_path(digest), data, digest)
        return digest

    def get_record(self, digest: str) -> dict[str, Any]:
        raw = self._read(self.record_path(digest), digest)
        record = _decode_record(raw)
        for ref in blob_references(record):
            self.get_blob(ref)
        self._check_calibration_record(record)
        return record

    def _objects(self, kind: str) -> Iterator[Path]:
        directory = self.root / kind
        if not directory.exists() and not directory.is_symlink():
            return
        self._safe_parents(directory, create=False)
        for shard in sorted(directory.iterdir()):
            if shard.is_symlink() or not shard.is_dir():
                raise RecordError(f"unsafe content shard: {shard}")
            for file in sorted(shard.iterdir()):
                if file.name.startswith(".tmp-"):
                    continue
                yield file

    def object_paths(self) -> list[Path]:
        return list(self._objects("records")) + list(self._objects("blobs"))

    def protected_paths(self) -> set[Path]:
        """Conservative GC roots: all published records and their blobs.

        Raises if *any* record is corrupt; retention must then protect every
        blob, never infer that it has no inbound reference.
        """
        protected = set()
        for path in self._objects("records"):
            doc = self.get_record(path.stem)
            protected.add(path)
            protected.update(
                self.blob_path(ref["sha256"]) for ref in blob_references(doc)
            )
        return protected

    def query(
        self,
        *,
        engine: str | None = None,
        run_id: str | None = None,
        layout: str | None = None,
        workload: str | None = None,
        metric: str | None = None,
        since: str | None = None,
        until: str | None = None,
    ) -> list[dict[str, Any]]:
        """Filter verified records by identity, workload, named metric group and UTC time.

        Each result has sha256 and record keys. This reader deliberately has
        no path-based record API or producer-specific interpretation.
        """
        if metric is not None and not _METRIC.fullmatch(metric):
            raise RecordError("metric must be group.name (for example decode.tokens_per_second or quality.kld_mean)")
        for label, stamp in (("since", since), ("until", until)):
            if stamp is not None:
                _check_schema(
                    stamp,
                    {
                        "type": "string",
                        "format": "date-time",
                        "pattern": r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$",
                    },
                    {},
                    label,
                )
        matches = []
        for path in self._objects("records"):
            digest = _check_digest(path.stem)
            doc = self.get_record(digest)
            if (
                run_id is not None
                and doc["run_id"] != run_id
                or engine is not None
                and doc["identity"]["engine"] != engine
                or layout is not None
                and doc["identity"]["layout"] != layout
                or workload is not None
                and doc["workload"]["name"] != workload
                or since is not None
                and doc["recorded_at"] < since
                or until is not None
                and doc["recorded_at"] > until
            ):
                continue
            if metric is not None:
                phase, name = metric.split(".", 1)
                if name not in doc["metrics"][phase]:
                    continue
            matches.append({"sha256": digest, "record": doc})
        return sorted(
            matches, key=lambda item: (item["record"]["recorded_at"], item["sha256"])
        )
