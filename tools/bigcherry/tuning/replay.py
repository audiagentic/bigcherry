"""Build a replay cache from a tuning run's measurements (HI15, review RV09).

This is the missing half of the dispatch pipeline. `hip-autotune-replay.cpp`
has always been able to *read* a cache; nothing has ever written one, so tuning
output could not reach production at all.

Written in Python rather than C++ for the same reason record mode is: nothing
in the production link path should be able to write this format, and a tuning
run killed at hour three should still yield a usable cache from whatever its
JSONL already flushed.

The layout below mirrors `hip-autotune-replay.cpp` byte for byte. Both sides
are little-endian and neither uses a struct overlay, so no compiler padding is
involved. If you change one, change the other and bump
``GGML_HIP_REPLAY_VERSION``; the loader rejects a version it does not know,
which is the intended failure.

Usage:

    python -m bigcherry.replay_cache tune.jsonl.measurements.jsonl \\
        --manifest artifacts/<rev>/hip-autotune-manifest.json \\
        --output dispatch.cache
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import re
import sqlite3
import struct
import tempfile
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from . import catalog as autotune_catalog
from . import dispatch_abi
from . import promotion_gate as correctness_gate
from . import verification_state
from ..identity_separation import IdentitySeparationError, validate_measurement_identity

MAGIC = 0x59484342
# HI31: v4 -> v5 adds a per-entry transform_id (uint16, 0 = no transform,
# ENT_TRANSFORM below); HI74 adds the trailing match_kind byte (ENT_MATCH_KIND
# below). Both mirror hip-autotune-replay.h/.cpp byte for byte. A v4 cache is
# rejected outright by the C++ loader's exact format-version check (rerun
# required) -- the production side has no dual-layout reader. The only v4
# reader in this module is read_cache_legacy_v4(), an OFFLINE ANALYSIS-ONLY
# reader for historical campaign artifacts and for merging v4 exports into a
# fresh v5 cache.
REPLAY_VERSION = 5
REPLAY_VERSION_V4 = 4
ARTIFACT_VERSION = 1
# HI119 review follow-up: derived from dispatch_abi (the single canonical
# Python-side source), not a separately hand-maintained literal -- see that
# module's own docstring for why this mattered.
SIGNATURE_SCHEMA_VERSION = dispatch_abi.SIGNATURE_SCHEMA_VERSION
HARDWARE_SCHEMA_VERSION = dispatch_abi.HARDWARE_SCHEMA_VERSION
DIGEST_BYTES = 16
PERSON_DISPATCH = b"llama-dispatch"
_REVISION_RE = re.compile(r"[0-9a-fA-F]{40}")


def validate_provenance_namespace(
    value: dict[str, Any], *, where: str = "provenance"
) -> dict[str, Any]:
    """Validate the identity namespace carried by a tuning result.

    Provenance is intentionally metadata, not part of the portable replay
    digest.  It must nevertheless be complete and canonical before results
    from different builds can be compared or selected.  Return a normalized
    copy so callers cannot accidentally compare mixed-case digests.
    """
    if not isinstance(value, dict):
        raise SystemExit(f"{where} must be an object")
    revision = value.get("source_revision")
    if not isinstance(revision, str) or not _REVISION_RE.fullmatch(revision):
        raise SystemExit(
            f"{where}.source_revision must be a 20-byte hexadecimal revision"
        )
    manifest = _digest_hex(value.get("manifest_hash"), f"{where}.manifest_hash")
    normalized: dict[str, Any] = {
        "source_revision": revision.lower(),
        "manifest_hash": manifest,
    }
    descriptor = value.get("build_descriptor_hash")
    if descriptor is not None:
        normalized["build_descriptor_hash"] = _digest_hex(
            descriptor, f"{where}.build_descriptor_hash"
        )
    variant_set = value.get("variant_set")
    if variant_set is not None:
        if not isinstance(variant_set, str) or not variant_set.strip():
            raise SystemExit(f"{where}.variant_set must be a non-empty string")
        normalized["variant_set"] = variant_set
    return normalized


def portable_tuning_key(
    hardware: Any, signature: Any, objective: Any = "latency"
) -> str:
    """Return the Python equivalent of the runtime portable dispatch digest.

    The canonical object and personalization match ``ggml_hip_dispatch_digest``
    exactly.  Build/source identity is deliberately excluded; callers should
    retain it separately through :func:`validate_provenance_namespace`.
    """
    hardware = _digest_hex(hardware, "hardware digest")
    signature = _digest_hex(signature, "signature digest")
    if not isinstance(objective, str) or not objective:
        raise SystemExit("objective must be a non-empty string")
    payload = json.dumps(
        {"hardware": hardware, "objective": objective, "signature": signature},
        sort_keys=True,
        separators=(",", ":"),
    )
    return blake2b_digest(payload.encode("utf-8")).hex()


def select_newest_winners(
    records: Iterable[dict[str, Any]], *, keep_generations: int = 1
) -> list[dict[str, Any]]:
    """Select deterministic newest winners without changing replay bytes.

    Records are grouped by their portable hardware/signature/objective key.
    ``generation`` is the only ordering field; source and manifest identities
    are tie-break identity, never an accidental ordering surrogate.  A
    conflicting winner for the same key and generation is rejected rather than
    making selection depend on input order.
    """
    if (
        isinstance(keep_generations, bool)
        or not isinstance(keep_generations, int)
        or keep_generations < 1
    ):
        raise SystemExit("keep_generations must be a positive integer")
    grouped: dict[str, dict[int, dict[tuple[str, str], dict[str, Any]]]] = {}
    for index, original in enumerate(records):
        if not isinstance(original, dict):
            raise SystemExit(f"winner record {index} must be an object")
        record = dict(original)
        provenance = record.get("provenance")
        if provenance is None and record.get("source_revision_digest") is not None:
            manifest = _digest_hex(
                record.get("manifest_hash"), f"winner record {index}.manifest_hash"
            )
            revision_digest = _digest_hex(
                record.get("source_revision_digest"),
                f"winner record {index}.source_revision_digest",
            )
            provenance = {
                "manifest_hash": manifest,
                "source_revision_digest": revision_digest,
            }
        elif provenance is None:
            provenance = {
                key: record.get(key)
                for key in (
                    "source_revision",
                    "manifest_hash",
                    "build_descriptor_hash",
                    "variant_set",
                )
                if key in record
            }
        if "source_revision_digest" in provenance:
            provenance = {
                "manifest_hash": _digest_hex(
                    provenance.get("manifest_hash"),
                    f"winner record {index}.provenance.manifest_hash",
                ),
                "source_revision_digest": _digest_hex(
                    provenance.get("source_revision_digest"),
                    f"winner record {index}.provenance.source_revision_digest",
                ),
            }
        else:
            provenance = validate_provenance_namespace(
                provenance, where=f"winner record {index}.provenance"
            )
        generation = record.get("generation", 0)
        if (
            isinstance(generation, bool)
            or not isinstance(generation, int)
            or generation < 0
        ):
            raise SystemExit(
                f"winner record {index}.generation must be a non-negative integer"
            )
        key = record.get("portable_key")
        if key is None:
            key = portable_tuning_key(
                record.get("hardware"),
                record.get("signature"),
                record.get("objective", "latency"),
            )
        key = _digest_hex(key, f"winner record {index}.portable_key")
        winner = record.get("winner")
        if not isinstance(winner, str) or not winner:
            raise SystemExit(f"winner record {index}.winner must be a non-empty string")
        record["portable_key"] = key
        record["provenance"] = provenance
        record["generation"] = generation
        bucket = grouped.setdefault(key, {})
        identity = (
            provenance.get("source_revision_digest")
            or source_revision_digest(provenance["source_revision"]),
            provenance["manifest_hash"],
        )
        existing = bucket.setdefault(generation, {})
        prior = existing.get(identity)
        if prior is not None and prior.get("winner") != winner:
            raise SystemExit(
                f"conflicting winners for portable key {key} generation {generation}"
            )
        existing[identity] = record

    selected: list[dict[str, Any]] = []
    for key in sorted(grouped):
        generations = sorted(grouped[key], reverse=True)[:keep_generations]
        for generation in generations:
            # A single generation may have multiple provenance identities;
            # retain all of them in stable order for a future exact match.
            selected.extend(
                grouped[key][generation][identity]
                for identity in sorted(grouped[key][generation])
            )
    return selected


# Wire-format boundary shared with hip-autotune-replay.cpp. Version 4 keeps the
# v3 ABI fields and adds per-entry provenance/generation. Older containers are
# intentionally not readable: HI51 made v3 the sole pre-HI23 wire contract.
REPLAY_HEADER_SIZE = 56

# Entry layout, from the ENT_* constants in hip-autotune-replay.cpp.
ENT_MANIFEST = 54
ENT_SOURCE_REVISION = ENT_MANIFEST + DIGEST_BYTES
ENT_GENERATION = ENT_SOURCE_REVISION + DIGEST_BYTES
ENT_TRANSFORM = ENT_GENERATION + 4
# HI74: match_kind discriminator -- see ggml_hip_replay_match_kind in
# hip-autotune-replay.h. Every producer here writes MATCH_KIND_EXACT; the
# field exists so a future generalised-entry feature (HI36b) has an
# extension point that does not force a v6 wire-format bump by itself.
ENT_MATCH_KIND = ENT_TRANSFORM + 2
ENT_SIZE = ENT_MATCH_KIND + 1
assert ENT_SIZE == 93

# v4 entry layout: the v5 fields minus the trailing transform_id +
# match_kind. Verified against the historical dispatch-27b.cache artifact
# (version=4, 59 entries, implied size 90).
ENT_SIZE_V4 = 90
assert ENT_SIZE_V4 + 3 == ENT_SIZE

MATCH_KIND_EXACT = 0


def source_revision_digest(source_revision: str) -> str:
    """Return the v4 fixed-width identity for a source revision string."""
    if not isinstance(source_revision, str) or not _REVISION_RE.fullmatch(
        source_revision
    ):
        raise SystemExit("source_revision must be a 20-byte hexadecimal revision")
    return blake2b_digest(source_revision.lower().encode("ascii")).hex()


def _digest_hex(value: Any, label: str) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-fA-F]{32}", value):
        raise SystemExit(f"{label} must be a 16-byte hexadecimal digest")
    return value.lower()


def blake2b_digest(data: bytes) -> bytes:
    """The project's content hash: blake2b, 16 bytes, personalised.

    Must match `ggml_hip_blake2b(..., GGML_HIP_PERSON_DISPATCH)`; the loader
    rejects the file on a mismatch, which is how a truncated or corrupted
    cache is caught before any offset in it is trusted.
    """
    return hashlib.blake2b(
        data, digest_size=DIGEST_BYTES, person=PERSON_DISPATCH
    ).digest()


def validate_blob(
    blob: bytes, *, manifest_hash: str | None = None, enforce_schema: bool = True,
) -> dict[str, int]:
    """Preflight a v5 cache using the same bounds as the production reader.

    This is deliberately a structural check only: candidate ABI validation
    remains the C++ registry's responsibility.  Running it before publishing
    an export catches accidental truncation, stale manifest binding, and
    duplicate/unterminated entries while the producer still has a useful
    error path.  It does not change or rewrite the wire bytes.

    A v4 cache fails here at the format-version check: production replay
    treats v4 as rerun-required, and a producer must never rewrite a v4
    file as if it understood it.  A v5 entry carrying a match_kind this
    tool does not recognise (anything but MATCH_KIND_EXACT) is rejected the
    same way -- reinterpreting a future generalised-entry key form as a
    plain exact match is exactly the misparse the discriminator exists to
    prevent (HI74 fail-closed contract, mirroring the C++ loader's
    per-entry skip).

    ``enforce_schema=False`` (HI119 review follow-up, 2026-08-25) skips ONLY
    the signature/hardware schema version checks below -- every other
    structural check (format/artifact version, checksum, bounds,
    duplicate/unterminated entries) still runs. This exists purely for
    offline inspection of real historical artifacts committed before a
    schema bump (e.g. docs/reference/.../dispatch-27b-v5.cache, captured
    under schema 1) -- every real production/merge call site keeps the
    default True and must never pass False, since that is precisely the
    fail-closed check a schema bump exists to enforce.
    """
    if not isinstance(blob, bytes) or len(blob) < REPLAY_HEADER_SIZE:
        raise SystemExit("replay cache is shorter than its header")
    magic, version, artifact = struct.unpack_from("<III", blob)
    if magic != MAGIC:
        raise SystemExit("replay cache has bad magic")
    if version != REPLAY_VERSION:
        raise SystemExit(
            "replay cache format version mismatch "
            f"(found {version}, this tool produces and consumes "
            f"{REPLAY_VERSION} only)"
        )
    if artifact != ARTIFACT_VERSION:
        raise SystemExit("replay cache artifact version mismatch")
    sig_schema, hw_schema, entry_count, string_bytes = struct.unpack_from(
        "<HHII", blob, 12
    )
    if enforce_schema and sig_schema != SIGNATURE_SCHEMA_VERSION:
        raise SystemExit("replay cache signature schema version mismatch")
    if enforce_schema and hw_schema != HARDWARE_SCHEMA_VERSION:
        raise SystemExit("replay cache hardware schema version mismatch")
    stored_manifest = blob[24:40]
    if manifest_hash is not None:
        expected_manifest = bytes.fromhex(_digest_hex(manifest_hash, "manifest_hash"))
        if stored_manifest != expected_manifest:
            raise SystemExit(
                "replay cache manifest hash differs from supplied manifest"
            )
    entries_bytes = entry_count * ENT_SIZE
    strings_at = REPLAY_HEADER_SIZE + entries_bytes
    expected = strings_at + string_bytes
    if strings_at < REPLAY_HEADER_SIZE or expected < strings_at:
        raise SystemExit("replay cache table size overflows")
    if len(blob) != expected:
        raise SystemExit("replay cache file is truncated or has trailing bytes")
    payload = blob[REPLAY_HEADER_SIZE:]
    if blob[40:56] != blake2b_digest(payload):
        raise SystemExit("replay cache content checksum mismatch")

    strings = blob[strings_at:expected]
    seen: set[tuple[bytes, bytes, bytes, int]] = set()
    for index in range(entry_count):
        entry = blob[
            REPLAY_HEADER_SIZE + index * ENT_SIZE : REPLAY_HEADER_SIZE
            + (index + 1) * ENT_SIZE
        ]
        dispatch = entry[:DIGEST_BYTES]
        manifest = entry[ENT_MANIFEST:ENT_SOURCE_REVISION]
        revision = entry[ENT_SOURCE_REVISION:ENT_GENERATION]
        generation = struct.unpack_from("<I", entry, ENT_GENERATION)[0]
        identity = (dispatch, manifest, revision, generation)
        if identity in seen:
            raise SystemExit("replay cache contains duplicate generation identity")
        seen.add(identity)
        match_kind = entry[ENT_MATCH_KIND]
        if match_kind != MATCH_KIND_EXACT:
            raise SystemExit(
                "replay cache entry "
                f"{dispatch.hex()[:16]}... carries match_kind {match_kind}, "
                "which this tool does not recognise; refusing to reinterpret "
                "a future key form as an exact match"
            )
        name_offset = struct.unpack_from("<I", entry, 32)[0]
        if name_offset >= string_bytes:
            raise SystemExit("replay cache entry names a string outside the table")
        if b"\0" not in strings[name_offset:]:
            raise SystemExit("replay cache string table is not NUL-terminated")
    return {"entry_count": entry_count, "string_bytes": string_bytes}


def read_cache(
    blob: bytes, *, enforce_schema: bool = True,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Read v5 entries for a deterministic merge without trusting bad bytes.

    ``enforce_schema=False``: see validate_blob's own docstring -- offline
    historical-artifact inspection only, never for a production/merge path."""
    validate_blob(blob, enforce_schema=enforce_schema)
    _, version, artifact = struct.unpack_from("<III", blob)
    sig_schema, hw_schema, entry_count, string_bytes = struct.unpack_from(
        "<HHII", blob, 12
    )
    entries_at = REPLAY_HEADER_SIZE
    strings_at = entries_at + entry_count * ENT_SIZE
    strings = blob[strings_at : strings_at + string_bytes]
    header = {
        "version": version,
        "artifact_version": artifact,
        "signature_schema": sig_schema,
        "hardware_schema": hw_schema,
        "manifest_hash": blob[24:40].hex(),
    }
    entries: list[dict[str, Any]] = []
    for index in range(entry_count):
        entry = blob[
            entries_at + index * ENT_SIZE : entries_at + (index + 1) * ENT_SIZE
        ]
        name_offset = struct.unpack_from("<I", entry, 32)[0]
        name_end = strings.index(b"\0", name_offset)
        entries.append(
            {
                "portable_key": entry[:DIGEST_BYTES].hex(),
                "dispatch": entry[:DIGEST_BYTES].hex(),
                "signature": entry[DIGEST_BYTES : 2 * DIGEST_BYTES].hex(),
                "winner": strings[name_offset:name_end].decode("utf-8"),
                "manifest_hash": entry[ENT_MANIFEST:ENT_SOURCE_REVISION].hex(),
                "source_revision_digest": entry[
                    ENT_SOURCE_REVISION:ENT_GENERATION
                ].hex(),
                "generation": struct.unpack_from("<I", entry, ENT_GENERATION)[0],
                "transform_id": struct.unpack_from("<H", entry, ENT_TRANSFORM)[0],
                "match_kind": entry[ENT_MATCH_KIND],
                "wire_entry": entry,
            }
        )
    return header, entries


def read_cache_legacy_v4(
    blob: bytes, *, enforce_schema: bool = True,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Offline analysis-only reader for historical v4 caches.

    NOT a production reader. The C++ loader rejects v4 outright as
    rerun-required and never reads it; this exists so the Python tools can
    still inspect historical campaign artifacts (e.g. dispatch-27b.cache) and
    so a v4 export can be merged into a fresh v5 cache without the operator
    having to hand-convert bytes. It shares every structural bound with
    :func:`validate_blob` (same header fields, same checksum, same bounds and
    duplicate/string checks) and only differs in the v4 entry size and the
    absence of the trailing transform_id/match_kind fields.

    Returned entries carry ``transform_id=0`` and ``match_kind=None`` because
    the v4 wire format had no such fields -- a v4 entry was always an exact
    match. ``wire_entry`` is the raw 90-byte v4 entry; it must not be
    repacked into a v5 cache as-is (see the retained-entry guard in
    :func:`build`), it is only exposed for inspection.

    ``enforce_schema=False`` (HI119 review follow-up, 2026-08-25): this
    reader has TWO real callers with different safety needs -- pure offline
    inspection of a historical artifact (schema irrelevant, the artifact is
    definitionally historical) and :func:`build`'s merge-into-fresh-v5-cache
    path (schema MUST gate here: merging a stale-schema v4 export's entries
    into a current-schema v5 cache without checking would silently launder
    old-schema data under the new schema's meaning -- the exact class of bug
    HI119's schema bump exists to prevent). :func:`build`'s own call site
    keeps the default True; pass False only for read-only inspection."""
    if not isinstance(blob, bytes) or len(blob) < REPLAY_HEADER_SIZE:
        raise SystemExit("replay cache is shorter than its header")
    magic, version, artifact = struct.unpack_from("<III", blob)
    if magic != MAGIC:
        raise SystemExit("replay cache has bad magic")
    if version != REPLAY_VERSION_V4:
        raise SystemExit(
            f"read_cache_legacy_v4 expects a v4 cache, found version {version}"
        )
    if artifact != ARTIFACT_VERSION:
        raise SystemExit("replay cache artifact version mismatch")
    sig_schema, hw_schema, entry_count, string_bytes = struct.unpack_from(
        "<HHII", blob, 12
    )
    if enforce_schema and sig_schema != SIGNATURE_SCHEMA_VERSION:
        raise SystemExit("replay cache signature schema version mismatch")
    if enforce_schema and hw_schema != HARDWARE_SCHEMA_VERSION:
        raise SystemExit("replay cache hardware schema version mismatch")
    entries_bytes = entry_count * ENT_SIZE_V4
    strings_at = REPLAY_HEADER_SIZE + entries_bytes
    expected = strings_at + string_bytes
    if strings_at < REPLAY_HEADER_SIZE or expected < strings_at:
        raise SystemExit("replay cache table size overflows")
    if len(blob) != expected:
        raise SystemExit("replay cache file is truncated or has trailing bytes")
    payload = blob[REPLAY_HEADER_SIZE:]
    if blob[40:56] != blake2b_digest(payload):
        raise SystemExit("replay cache content checksum mismatch")

    strings = blob[strings_at:expected]
    seen: set[tuple[bytes, bytes, bytes, int]] = set()
    entries: list[dict[str, Any]] = []
    for index in range(entry_count):
        entry = blob[
            REPLAY_HEADER_SIZE + index * ENT_SIZE_V4 : REPLAY_HEADER_SIZE
            + (index + 1) * ENT_SIZE_V4
        ]
        dispatch = entry[:DIGEST_BYTES]
        manifest = entry[ENT_MANIFEST:ENT_SOURCE_REVISION]
        revision = entry[ENT_SOURCE_REVISION:ENT_GENERATION]
        generation = struct.unpack_from("<I", entry, ENT_GENERATION)[0]
        identity = (dispatch, manifest, revision, generation)
        if identity in seen:
            raise SystemExit("replay cache contains duplicate generation identity")
        seen.add(identity)
        name_offset = struct.unpack_from("<I", entry, 32)[0]
        if name_offset >= string_bytes:
            raise SystemExit("replay cache entry names a string outside the table")
        name_end = strings.index(b"\0", name_offset)
        entries.append(
            {
                "portable_key": dispatch.hex(),
                "dispatch": dispatch.hex(),
                "signature": entry[DIGEST_BYTES : 2 * DIGEST_BYTES].hex(),
                "winner": strings[name_offset:name_end].decode("utf-8"),
                "manifest_hash": manifest.hex(),
                "source_revision_digest": revision.hex(),
                "generation": generation,
                # v4 has no transform/match fields; an exact match by construction.
                "transform_id": 0,
                "match_kind": None,
                "wire_entry": entry,
            }
        )
    header = {
        "version": version,
        "artifact_version": artifact,
        "signature_schema": sig_schema,
        "hardware_schema": hw_schema,
        "manifest_hash": blob[24:40].hex(),
    }
    return header, entries


def ggml_type_values(ggml_h: Path) -> dict[str, int]:
    """Numeric `ggml_type` values, parsed from upstream's own enum.

    Read rather than restated: the cache stores the type as a byte, and a
    hand-maintained table that drifted from upstream would repoint every
    entry at a different type without any error.
    """
    text = ggml_h.read_text(encoding="utf-8", errors="replace")
    values: dict[str, int] = {}
    for name, value in re.findall(r"(GGML_TYPE_\w+)\s*=\s*(\d+)", text):
        values[name] = int(value)
    if "GGML_TYPE_F32" not in values:
        raise SystemExit(f"could not parse ggml_type enum from {ggml_h}")
    return values


def read_results(
    path: Path, *, require_header: bool = True
) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
    """Winning results from a measurements JSONL.

    Tolerates a truncated final line by construction -- a tuning run killed
    mid-write still yields every result it had already flushed.
    """
    results = []
    header: dict[str, Any] | None = None
    seen: set[str] = set()
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise SystemExit(
                    f"malformed measurements JSONL: {path}: {exc}"
                ) from exc
            if record.get("kind") == "header":
                if header is not None:
                    raise SystemExit("duplicate measurements header")
                header = record
            elif record.get("kind") == "result" and record.get("winner"):
                try:
                    validate_measurement_identity(
                        record, header=header, where=f"result {len(results)}"
                    )
                except IdentitySeparationError as exc:
                    raise SystemExit(str(exc)) from exc
                dispatch = record.get("dispatch")
                dispatch = _digest_hex(dispatch, "result dispatch digest")
                if dispatch in seen:
                    raise SystemExit(f"duplicate dispatch digest: {dispatch}")
                seen.add(dispatch)
                record = dict(record)
                record["dispatch"] = dispatch
                results.append(record)
            elif record:
                raise SystemExit("unknown measurements record kind")
    if header is None and require_header:
        raise SystemExit("measurements header required")
    return header, results


def _native_claim_is_authoritative(
    native_name: str | None, by_name: dict[str, dict[str, Any]],
) -> bool:
    """Adversarial-review follow-up: a result row's own ``native`` field is
    self-reported by whatever produced the measurements artifact -- nothing
    upstream (inventory.py's ingest) independently proves the named
    candidate is actually the native implementation. Both
    _validate_promotion_gate and _validate_correctness_gate below used to
    exempt any row where ``winner == native`` unconditionally, which let a
    forged/crafted row claim ``winner=native=<any real challenger
    candidate>`` and skip BOTH gates entirely -- the same trust-pattern bug
    already fixed once for the "seeded" field (HI136 review round 1):
    authority must never come from the artifact's own claim about itself.

    Ground the claim in the supplied manifest's real ``source_class``
    instead -- the one field catalog.py itself uses as the authoritative
    native/generated distinction throughout the codebase. Absence from the
    manifest, or a manifest that doesn't classify it as native_wrapper,
    means the claim is NOT authoritative -- the row then falls through to
    the same promotion_status/correctness-evidence requirements a genuine
    non-native candidate would need, which is the correct fail-closed
    behavior (never an outright reject: a legitimate legacy/seed-only
    export may have no manifest candidate data at all for reasons unrelated
    to forgery, and still deserves the normal non-native evidence path)."""
    if not isinstance(native_name, str) or not native_name:
        return False
    candidate = by_name.get(native_name)
    if not isinstance(candidate, dict):
        return False
    return candidate.get("source_class") == "native_wrapper"


def _validate_promotion_gate(
    entries: dict[str, dict[str, Any]], by_name: dict[str, dict[str, Any]],
) -> None:
    """Fail closed (HI34/P0): a non-native winner ships only after fresh
    confirmation *and* experiment-wide BH correction -- see tune_promotion.py.
    A row whose winner differs from its recorded native and whose
    promotion_status is not exactly "promoted" (raw pending_bh, an explicit
    rejection, or a legacy/missing-provenance record with no promotion_status
    at all) has never been through that gate and has no business on
    production's hot path.

    A measurement_failure row is never exportable, including when its
    recorded winner is native: it is terminal evidence that the tuning
    experiment did not produce a trustworthy replay decision.

    Aborts the whole export rather than silently dropping the offending
    entries: a cache silently missing entries the operator expected to see
    is a worse failure than a build that stops and says exactly why. The
    caller excludes any dispatch covered by a manual `--seed` override
    before calling this -- an explicit operator override carries its own
    provenance, not the tuner's, and must not be blocked by (or need to
    pass) the promotion state of whatever raw measurement happened to exist
    for that same digest.
    """
    violations = []
    for digest_hex, record in entries.items():
        # Fail closed: any truthy encoding of the failure marker counts,
        # not only a strict boolean True.
        if bool(record.get("measurement_failure")):
            violations.append((digest_hex, record.get("winner"), "measurement_failure"))
            continue
        winner = record.get("winner")
        native = record.get("native")
        if winner == native and _native_claim_is_authoritative(native, by_name):
            continue  # native state, and the manifest confirms it really is
        if record.get("promotion_status") != "promoted":
            violations.append((digest_hex, winner, record.get("promotion_status")))
    if violations:
        shown = "\n".join(
            f"  {digest[:16]}...: winner={winner!r} promotion_status={status!r}"
            for digest, winner, status in violations[:20]
        )
        more = (
            f"\n  ... and {len(violations) - 20} more" if len(violations) > 20 else ""
        )
        raise SystemExit(
            f"refusing to export: {len(violations)} unsafe measurement result(s) "
            f"lack a valid replay promotion gate (failed measurements are never "
            f"exportable):\n"
            f"{shown}{more}"
        )


def _validate_correctness_gate(
    entries: dict[str, dict[str, Any]], dispatch_db: Path | None,
    by_name: dict[str, dict[str, Any]],
    source_build_id: int | None = None,
    source_provenance: dict[str, Any] | None = None,
) -> None:
    """Fail closed (HI67/RV49): every non-native winner ships only after it
    independently passes the real CPU-reference correctness gate for its
    EXACT (dispatch, signature, hardware, candidate) binding -- re-verified
    here against a real dispatch_db, not merely trusted from an earlier
    pipeline stage.

    This is deliberately separate from, and stricter than,
    _validate_promotion_gate above: that function proves a row went through
    tune_promotion.py's statistical BH correction; this proves it separately
    satisfied the RV49 correctness contract (promotion_correctness_gate.py),
    at export time, for the precise binding about to ship. A manual --seed
    override is explicitly NOT exempt -- an operator-authored override for a
    non-native candidate is exactly the kind of unverified escape hatch this
    check exists to close (see HI67's replay-cache export gate redesign).

    Only a candidate's own claim of correctness is ever accepted here: the
    dispatch_db itself supplies threshold_t/headroom_fraction from evidence
    that promotion_correctness_gate.py already re-derives from raw seed rows,
    never from a caller-supplied value.
    """
    violations: list[tuple[str, str | None, str]] = []
    conn: sqlite3.Connection | None = None
    source_build_id_verified = False
    try:
        for digest_hex, record in sorted(entries.items()):
            if bool(record.get("measurement_failure")):
                continue  # already rejected by _validate_promotion_gate
            winner = record.get("winner")
            native = record.get("native")
            if winner == native and _native_claim_is_authoritative(native, by_name):
                continue  # native state, and the manifest confirms it really is
            if source_build_id is not None and not source_build_id_verified:
                # Adversarial-review follow-up (HI126 composition gap): a
                # projected artifact's hi121_source_provenance.source_build_id
                # was previously trusted as-is (only range/type checked) and
                # fed directly into resolve_promotion_identity()'s
                # "AND build_id=?" constraint. Nothing proved that build_id
                # actually corresponds to the REST of the same provenance
                # tuple (source_revision/manifest_hash/build_descriptor_hash)
                # recorded right beside it in the same header -- an attacker
                # could retarget source_build_id to a DIFFERENT build sharing
                # the same dispatch/signature/hardware triple and inherit ITS
                # correctness evidence instead. Cross-check the claimed
                # build_id against the real DB row for it before trusting it
                # for anything. Deliberately lazy (only once, only when a
                # non-native winner actually needs correctness resolution) --
                # an all-native export must not be forced to require
                # --dispatch-db just because a header happens to carry
                # provenance nobody is going to use.
                if not isinstance(source_provenance, dict):
                    raise SystemExit(
                        "refusing to export: source_build_id present without its "
                        "hi121_source_provenance tuple to bind it to"
                    )
                if dispatch_db is None:
                    raise SystemExit(
                        "refusing to export: hi121_source_provenance.source_build_id is "
                        "present but no --dispatch-db was supplied to verify it against"
                    )
                if conn is None:
                    conn = sqlite3.connect(f"file:{dispatch_db}?mode=ro", uri=True)
                build_row = conn.execute(
                    "SELECT source_revision, manifest_hash, build_descriptor_hash "
                    "FROM build WHERE build_id = ?",
                    (source_build_id,),
                ).fetchone()
                if build_row is None:
                    raise SystemExit(
                        f"refusing to export: hi121_source_provenance.source_build_id="
                        f"{source_build_id} does not exist in {dispatch_db}"
                    )
                db_source_revision, db_manifest_hash, db_descriptor_hash = build_row
                claimed_revision = source_provenance.get("source_revision")
                claimed_manifest_hash = source_provenance.get("manifest_hash")
                claimed_descriptor_hash = source_provenance.get("build_descriptor_hash")
                if claimed_revision != db_source_revision or claimed_manifest_hash != db_manifest_hash:
                    raise SystemExit(
                        f"refusing to export: hi121_source_provenance claims source_build_id="
                        f"{source_build_id} but its source_revision/manifest_hash do not match "
                        f"that build's real DB row -- source_build_id is not bound to the rest "
                        f"of its own provenance tuple"
                    )
                if (
                    db_descriptor_hash is not None
                    and claimed_descriptor_hash is not None
                    and claimed_descriptor_hash != db_descriptor_hash
                ):
                    raise SystemExit(
                        f"refusing to export: hi121_source_provenance claims source_build_id="
                        f"{source_build_id} but its build_descriptor_hash does not match that "
                        f"build's real DB row"
                    )
                source_build_id_verified = True
            signature_hex = record.get("signature")
            hardware_hex = record.get("hardware")
            if not isinstance(signature_hex, str) or not isinstance(hardware_hex, str):
                violations.append((
                    digest_hex, winner,
                    "missing explicit signature/hardware identity needed to "
                    "resolve real correctness evidence",
                ))
                continue
            if not isinstance(native, str) or not native:
                violations.append((
                    digest_hex, winner, "missing native baseline identity",
                ))
                continue
            if dispatch_db is None:
                violations.append((
                    digest_hex, winner,
                    "no --dispatch-db supplied to verify RV49 correctness evidence",
                ))
                continue
            if conn is None:
                conn = sqlite3.connect(f"file:{dispatch_db}?mode=ro", uri=True)
            binding = correctness_gate.CorrectnessBinding(
                dispatch_hex=digest_hex, signature_hex=signature_hex,
                hardware_hex=hardware_hex, native_name=native,
                candidate_name=winner,
                source_build_id=source_build_id,
            )
            try:
                correctness_gate.require_correctness_binding(conn, binding)
            except correctness_gate.CorrectnessGateError as exc:
                violations.append((digest_hex, winner, str(exc)))
    finally:
        if conn is not None:
            conn.close()
    if violations:
        shown = "\n".join(
            f"  {digest[:16]}...: winner={winner!r}: {reason}"
            for digest, winner, reason in violations[:20]
        )
        more = (
            f"\n  ... and {len(violations) - 20} more" if len(violations) > 20 else ""
        )
        raise SystemExit(
            f"refusing to export: {len(violations)} non-native winner(s) lack "
            f"real RV49 correctness-gate evidence for their exact dispatch/"
            f"hardware binding:\n{shown}{more}"
        )


def build(
    measurements: Path,
    manifest_path: Path,
    ggml_h: Path,
    *,
    dispatch_db: Path | None = None,
    seed_file: Path | None = None,
    merge_into: Path | None = None,
    keep_generations: int = 3,
    generation: int = 0,
    require_winner_verification: bool = False,
) -> bytes:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    by_name = {c["stable_name"]: c for c in manifest["candidates"]}
    type_values = ggml_type_values(ggml_h)

    manifest_hash = _digest_hex(manifest.get("manifest_hash"), "manifest_hash")

    # A fully explicit seed file is an operator-authored provenance source and
    # may replace the measurements artifact entirely. Normal exports always
    # require the producer header.
    producer_header, results = read_results(
        measurements, require_header=seed_file is None
    )
    if not results:
        raise SystemExit(f"no winning results in {measurements}")
    source_build_id = None
    projected_provenance = None
    if seed_file is None:
        projected_provenance = producer_header.get("hi121_source_provenance") if producer_header else None
        if isinstance(projected_provenance, dict) and "source_build_id" in projected_provenance:
            source_build_id = projected_provenance["source_build_id"]
            if isinstance(source_build_id, bool) or not isinstance(source_build_id, int) or source_build_id < 1:
                raise SystemExit(
                    "refusing to export: measurements hi121_source_provenance.source_build_id "
                    "must be a positive integer"
                )
        expected_revision = manifest.get("source_revision")
        if not isinstance(expected_revision, str) or not expected_revision:
            raise SystemExit(
                "refusing to export: supplied manifest lacks producer provenance "
                "(source_revision/manifest_hash)"
            )
        manifest_provenance = validate_provenance_namespace(
            {"source_revision": expected_revision, "manifest_hash": manifest_hash},
            where="manifest provenance",
        )
        expected_revision = manifest_provenance["source_revision"]
        # manifest_hash is deliberately not compared here: it is scoped to
        # variant_set and candidate set (manifest_hash()), so a replay-slim
        # manifest can never share a hash with the workload/full-max manifest
        # that produced these measurements even when the export is correct.
        # source_revision is the actual producer/consumer identity that
        # matters -- the candidate-set compatibility check happens separately
        # when winners are resolved against this manifest's catalog below.
        if (
            producer_header is None
            or producer_header.get("source_revision", "").lower() != expected_revision
        ):
            raise SystemExit(
                "refusing to export: measurements producer source_revision does "
                "not match the supplied manifest's source_revision"
            )

    if (
        isinstance(generation, bool)
        or not isinstance(generation, int)
        or generation < 0
    ):
        raise SystemExit("generation must be a non-negative integer")
    if merge_into is not None and not merge_into.is_file():
        raise SystemExit(f"merge source does not exist: {merge_into}")

    # One entry per dispatch digest in this measurement run. Duplicate results
    # are rejected by read_results: silently taking the last row can combine
    # incompatible tuning epochs into one generation.
    entries: dict[str, dict[str, Any]] = {}
    for record in results:
        entries[record["dispatch"]] = record

    # Seed overrides: explicit operator choices take precedence over measured
    # winners, but only after their digest, candidate identity, and namespace
    # have been validated against this manifest (HI22).
    seed_overrides: dict[str, dict[str, str]] = {}
    if seed_file and seed_file.is_file():
        seed_overrides = _load_seed_overrides(
            seed_file, by_name=by_name, manifest=manifest, manifest_hash=manifest_hash
        )
        for digest_hex, override in seed_overrides.items():
            existing = entries.get(digest_hex)
            signature = (
                existing.get("signature")
                if existing is not None
                else override.get("signature")
            )
            if not isinstance(signature, str) or not re.fullmatch(
                r"[0-9a-fA-F]{32}", signature
            ):
                raise SystemExit(
                    f"seed override for unseen dispatch {digest_hex[:16]}... requires explicit signature"
                )
            # An explicit operator override remains authoritative for the
            # winner, while a measured row remains authoritative for its
            # dispatch signature.  This makes precedence deterministic.
            override["signature"] = signature.lower()

    if seed_overrides:
        # Apply overrides: replace or insert entries
        for digest_hex, override in seed_overrides.items():
            entry = dict(entries.get(digest_hex, {}))
            entry.update(
                {
                    "dispatch": digest_hex,
                    "winner": override["winner"],
                    "signature": override["signature"],
                    "seeded": True,
                }
            )
            # An explicit override may also name the hardware/native identity
            # a genuinely unseen dispatch has no measured row to inherit them
            # from; when present it takes precedence over any measured value,
            # matching signature's own override-wins precedence above.
            if "hardware" in override:
                entry["hardware"] = override["hardware"]
            if "native" in override:
                entry["native"] = override["native"]
            entries[digest_hex] = entry

    if require_winner_verification:
        if dispatch_db is None:
            raise SystemExit(
                "refusing to export: require_winner_verification=True requires --dispatch-db"
            )
        verified_entries: dict[str, dict[str, Any]] = {}
        conn = sqlite3.connect(f"file:{dispatch_db}?mode=ro", uri=True)
        try:
            for digest_hex, record in sorted(entries.items()):
                # Explicit operator seed overrides retain their existing
                # authority semantics; this gate protects measured winners.
                # adversarial-review follow-up: authority MUST come from
                # membership in the independently-loaded seed_overrides
                # dict, never from the record's own "seeded" field --
                # read_results() preserves arbitrary JSON fields from the
                # measurements artifact verbatim, so a forged/promoted row
                # could otherwise smuggle "seeded": true past this gate and
                # bypass winner_verification entirely (a native winner is
                # especially exposed, since existing gates already treat
                # native as safe without requiring correctness evidence).
                if digest_hex in seed_overrides:
                    verified_entries[digest_hex] = record
                    continue
                signature_hex = record.get("signature")
                try:
                    winner_id = verification_state.require_winner_row(
                        conn, row=record, signature_hex=signature_hex,
                    )
                except verification_state.WinnerRowIdentityError as exc:
                    raise SystemExit(
                        f"refusing to export: measured result {digest_hex!r} "
                        f"failed exact winner identity binding: {exc}"
                    ) from exc
                if verification_state.is_winner_verified(conn, winner_id=winner_id):
                    verified_entries[digest_hex] = record
        finally:
            conn.close()
        entries = verified_entries
        if not entries:
            raise SystemExit(
                "refusing to export: no measured winners carry current winner_verification"
            )

    # The gate runs on every retained dispatch NOT covered by an explicit
    # seed. A seed is a separate operator decision carrying its own
    # provenance (HI22), not the tuner's, and must not be blocked by (or need
    # to pass) the promotion state of a raw measurement for that digest.
    _validate_promotion_gate(
        {
            digest_hex: record
            for digest_hex, record in entries.items()
            if digest_hex not in seed_overrides
        },
        by_name,
    )

    # Runs on every entry, including seed overrides applied just above --
    # unlike _validate_promotion_gate, which a seed override is explicitly
    # allowed to bypass (HI22), this gate is never bypassable (HI67).
    _validate_correctness_gate(entries, dispatch_db, by_name, source_build_id, projected_provenance)

    current_entries: list[dict[str, Any]] = []
    producer_revision = manifest.get("source_revision")
    if not isinstance(producer_revision, str):
        if seed_file is None:
            raise SystemExit("manifest source_revision is required for v4 export")
        # Preserve the legacy flat seed-file API, but bind a source-less seed
        # to an intentionally non-matching identity. It can be inspected and
        # merged, yet cannot become a production replay winner until the
        # operator supplies explicit provenance in the versioned envelope.
        producer_revision = "0" * 40
    producer_revision_digest = source_revision_digest(producer_revision)
    for digest_hex, record in sorted(entries.items()):
        record = dict(record)
        record["portable_key"] = digest_hex
        record["generation"] = generation
        # HI31: the tuner's measurements.jsonl row (winner_transform_id) and
        # read_cache()'s retained-entry dict (transform_id) name this fact
        # differently -- normalize onto "transform_id" here so the packing
        # loop below has one field to read regardless of which of
        # current_entries/existing_entries a record came from. A seed
        # override entry (built by hand above, no tuner row behind it) has
        # neither key and defaults to 0 -- an operator-authored seed always
        # names a plain candidate, never a transform.
        record["transform_id"] = int(record.get("winner_transform_id", 0) or 0)
        record["provenance"] = {
            "source_revision": producer_revision,
            "manifest_hash": manifest_hash,
        }
        current_entries.append(record)

    existing_entries: list[dict[str, Any]] = []
    if merge_into is not None:
        merge_blob = merge_into.read_bytes()
        merge_version = (
            struct.unpack_from("<I", merge_blob, 4)[0] if len(merge_blob) >= 8 else None
        )
        if merge_version == REPLAY_VERSION_V4:
            # A historical v4 export can be folded into a fresh v5 cache. v4
            # entries have no transform/match fields and are exact by
            # construction; the legacy reader maps them to transform_id=0.
            # Their raw 90-byte wire entries are never repacked into v5 -- the
            # retained-entry guard below rejects them fail-closed.
            _, existing_entries = read_cache_legacy_v4(merge_blob)
        else:
            _, existing_entries = read_cache(merge_blob)
        # A re-tune of the same build/generation replaces that generation's
        # winners rather than creating an input-order-dependent conflict.
        existing_entries = [
            old
            for old in existing_entries
            if not (
                old.get("generation") == generation
                and old.get("manifest_hash") == manifest_hash
                and old.get("source_revision_digest") == producer_revision_digest
            )
        ]
    selected_entries = select_newest_winners(
        [*existing_entries, *current_entries], keep_generations=keep_generations
    )

    strings: dict[str, int] = {}
    string_blob = bytearray()

    def intern(name: str) -> int:
        if name not in strings:
            strings[name] = len(string_blob)
            string_blob.extend(name.encode("utf-8"))
            string_blob.append(0)
        return strings[name]

    packed = bytearray()
    for record in sorted(
        selected_entries,
        key=lambda item: (
            item["portable_key"],
            -item["generation"],
            item["provenance"].get("source_revision_digest")
            or source_revision_digest(item["provenance"]["source_revision"]),
            item["provenance"]["manifest_hash"],
        ),
    ):
        digest_hex = record["portable_key"]
        name = record["winner"]
        candidate = by_name.get(name)
        if candidate is None:
            # A retained older generation may name a candidate removed from
            # this catalog. Preserve its self-describing wire entry so the
            # matching older binary can still use it; the current binary will
            # ignore it during exact-provenance lookup.
            raw_entry = record.get("wire_entry")
            if not isinstance(raw_entry, bytes) or len(raw_entry) != ENT_SIZE:
                raise SystemExit(
                    f"winner '{name}' is not in {manifest_path.name} and has "
                    "no self-describing wire entry to retain"
                )
            retained = bytearray(raw_entry)
            struct.pack_into("<I", retained, 32, intern(name))
            packed += retained
            continue

        fields = autotune_catalog.variant_fields(candidate)
        type_name = fields["src0_type"]
        src0_type = 0 if type_name == "0" else type_values.get(type_name)
        if src0_type is None:
            raise SystemExit(f"unknown ggml type '{type_name}' for '{name}'")

        digest = bytes.fromhex(digest_hex)
        if len(digest) != DIGEST_BYTES:
            raise SystemExit(f"malformed dispatch digest: {digest_hex!r}")

        signature_value = record.get("signature")
        if signature_value is None:
            raise SystemExit(
                f"winner {digest_hex} is missing an explicit signature digest"
            )
        signature_hex = _digest_hex(
            signature_value, f"winner {digest_hex} signature digest"
        )

        # Current measurements also carry the hardware half of the portable
        # dispatch identity.  If it is present, verify the composite rather
        # than allowing a fixture or hand-edited journal to pair unrelated
        # dispatch/signature/hardware values.  Older rows without hardware
        # remain readable, but they still must carry an explicit signature.
        hardware_value = record.get("hardware")
        if hardware_value is not None:
            hardware_hex = _digest_hex(
                hardware_value, f"winner {digest_hex} hardware digest"
            )
            expected_dispatch = portable_tuning_key(
                hardware_hex, signature_hex, record.get("objective", "latency")
            )
            if expected_dispatch != digest_hex:
                raise SystemExit(
                    f"winner {digest_hex} does not match its hardware/signature "
                    "dispatch identity"
                )
        signature = bytes.fromhex(signature_hex)
        provenance = record["provenance"]
        entry_manifest = bytes.fromhex(provenance["manifest_hash"])
        entry_revision = bytes.fromhex(
            provenance.get("source_revision_digest")
            or source_revision_digest(provenance["source_revision"])
        )

        packed += digest  # ENT_DISPATCH
        packed += signature  # ENT_SIGNATURE
        packed += struct.pack("<I", intern(name))
        packed += struct.pack("<H", int(candidate.get("implementation_version", 1)))
        packed += struct.pack(
            "<iii", fields["primary"], fields["secondary"], fields["width"]
        )
        packed += struct.pack(
            "<BBBB", fields["acc_f16"], fields["fallback"], fields["small_k"], src0_type
        )
        packed += entry_manifest
        packed += entry_revision
        packed += struct.pack("<I", record["generation"])
        transform_id = int(record.get("transform_id", 0) or 0)
        if not 0 <= transform_id <= 0xFFFF:
            raise SystemExit(
                f"winner {digest_hex} transform_id out of range: {transform_id}"
            )
        packed += struct.pack("<H", transform_id)
        match_kind = int(record.get("match_kind", MATCH_KIND_EXACT) or MATCH_KIND_EXACT)
        if not 0 <= match_kind <= 0xFF:
            raise SystemExit(
                f"winner {digest_hex} match_kind out of range: {match_kind}"
            )
        packed += struct.pack("<B", match_kind)

    entry_count = len(packed) // ENT_SIZE
    payload = bytes(packed) + bytes(string_blob)

    header = bytearray()
    header += struct.pack("<III", MAGIC, REPLAY_VERSION, ARTIFACT_VERSION)
    header += struct.pack("<HH", SIGNATURE_SCHEMA_VERSION, HARDWARE_SCHEMA_VERSION)
    header += struct.pack("<II", entry_count, len(string_blob))
    header += bytes.fromhex(manifest_hash)
    header += blake2b_digest(payload)
    if len(header) != REPLAY_HEADER_SIZE:
        raise AssertionError("replay header layout drifted from the v5 wire format")

    blob = bytes(header) + payload
    validate_blob(blob, manifest_hash=manifest_hash)

    generations = sorted(
        {record["generation"] for record in selected_entries}, reverse=True
    )
    print(
        f"  {entry_count} winner(s) across generations {generations}, "
        f"{len(strings)} distinct candidate(s), {len(string_blob)} string byte(s)"
    )
    print(f"  manifest {manifest['manifest_hash']}")
    return blob


def _candidate_identity(candidate: dict[str, Any]) -> str:
    """Return the manifest-bound identity of a candidate descriptor."""
    payload = json.dumps(
        candidate, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return blake2b_digest(payload).hex()


def _load_seed_overrides(
    seed_file: Path,
    *,
    by_name: dict[str, dict[str, Any]],
    manifest: dict[str, Any],
    manifest_hash: str,
) -> dict[str, dict[str, str]]:
    """Load operator seeds with a manifest-bound, deterministic precedence.

    The original flat ``{dispatch: stable_name}`` format remains accepted for
    operators using an existing seed file.  The envelope form adds explicit
    provenance and candidate identities, and is required to match the
    manifest supplied to this export.  In both forms the manifest is the
    authority for candidate identity; a seed cannot smuggle in a descriptor.
    """
    try:
        document = json.loads(seed_file.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise SystemExit(f"seed file {seed_file} is not valid JSON: {exc}") from exc
    if not isinstance(document, dict):
        raise SystemExit("seed file must be a JSON object")

    if "overrides" in document:
        unknown = set(document) - {"version", "provenance", "overrides"}
        if unknown:
            raise SystemExit(
                f"seed file has unknown top-level fields: {sorted(unknown)}"
            )
        if document.get("version", 1) != 1:
            raise SystemExit("unsupported seed file version")
        seed_provenance = document.get("provenance")
        if not isinstance(seed_provenance, dict):
            raise SystemExit("seed file is missing its provenance object")
        provenance = validate_provenance_namespace(
            seed_provenance, where="seed provenance"
        )
        expected_revision = manifest.get("source_revision")
        if (
            not isinstance(expected_revision, str)
            or provenance["source_revision"] != expected_revision.lower()
            or provenance["manifest_hash"] != manifest_hash
        ):
            raise SystemExit("seed provenance does not match supplied manifest")
        raw_overrides = document["overrides"]
    else:
        # Backward-compatible operator format.  It is still bound to the
        # manifest hash and, when present, to its source revision.
        raw_overrides = document

    if not isinstance(raw_overrides, dict):
        raise SystemExit("seed overrides must be an object")
    normalized: dict[str, dict[str, str]] = {}
    for raw_digest, raw_value in sorted(
        raw_overrides.items(), key=lambda item: item[0].lower()
    ):
        digest_hex = _digest_hex(raw_digest, "seed override dispatch digest")
        if digest_hex in normalized:
            raise SystemExit(f"duplicate seed override dispatch digest: {digest_hex}")
        value = {"winner": raw_value} if isinstance(raw_value, str) else raw_value
        if not isinstance(value, dict) or not isinstance(value.get("winner"), str):
            raise SystemExit(
                "seed override must be a candidate name or object with winner"
            )
        allowed = {
            "winner", "signature", "candidate_digest", "provenance",
            "hardware", "native",
        }
        unknown = set(value) - allowed
        if unknown:
            raise SystemExit(f"seed override has unknown fields: {sorted(unknown)}")
        stable_name = value["winner"]
        candidate = by_name.get(stable_name)
        if candidate is None:
            raise SystemExit(
                f"seed override '{stable_name}' for dispatch {digest_hex[:16]}... "
                f"is not in the manifest"
            )
        candidate_digest = _candidate_identity(candidate)
        supplied_candidate_digest = value.get("candidate_digest", candidate_digest)
        if (
            _digest_hex(supplied_candidate_digest, "seed override candidate digest")
            != candidate_digest
        ):
            raise SystemExit(
                f"seed override candidate identity does not match manifest for '{stable_name}'"
            )
        if "provenance" in value:
            override_provenance = value["provenance"]
            if not isinstance(override_provenance, dict):
                raise SystemExit(
                    f"seed override '{stable_name}' has a non-object provenance"
                )
            provenance = validate_provenance_namespace(
                override_provenance, where="seed override provenance"
            )
            if provenance["manifest_hash"] != manifest_hash or (
                isinstance(manifest.get("source_revision"), str)
                and provenance["source_revision"] != manifest["source_revision"].lower()
            ):
                raise SystemExit(
                    "seed override provenance does not match supplied manifest"
                )
        # The caller supplies measurements separately; signature is checked
        # against that row below.  Unseen dispatches must carry one explicitly.
        signature = value.get("signature")
        if signature is not None:
            signature = _digest_hex(signature, "seed override signature digest")
        hardware = value.get("hardware")
        if hardware is not None:
            hardware = _digest_hex(hardware, "seed override hardware digest")
        native = value.get("native")
        if native is not None:
            if not isinstance(native, str):
                raise SystemExit(
                    f"seed override native for dispatch {digest_hex[:16]}... must be a string"
                )
            # HI89: validate against the manifest up front, the same way
            # `winner` already is -- an operator typo here previously only
            # surfaced later as an opaque CorrectnessGateError from the
            # binding-resolution DB lookup, fail-closed but not fail-fast.
            native_candidate = by_name.get(native)
            if native_candidate is None:
                raise SystemExit(
                    f"seed override native '{native}' for dispatch "
                    f"{digest_hex[:16]}... is not in the manifest"
                )
            if native_candidate.get("source_class") != "native_wrapper":
                raise SystemExit(
                    f"seed override native '{native}' for dispatch "
                    f"{digest_hex[:16]}... is not a native_wrapper candidate"
                )
        normalized[digest_hex] = {
            "winner": stable_name,
            "candidate_digest": candidate_digest,
            **({"signature": signature} if signature else {}),
            **({"hardware": hardware} if hardware else {}),
            **({"native": native} if native else {}),
        }
    return normalized


def main(argv: list[str] | None = None) -> None:
    root = Path(__file__).resolve().parent.parent.parent.parent
    parser = argparse.ArgumentParser(
        prog="bigcherry replay-cache",
        description="Build a replay cache from tuning measurements.",
    )
    parser.add_argument(
        "measurements",
        type=Path,
        help="JSONL written by GGML_HIP_DISPATCH_DB (the .measurements.jsonl file)",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        required=True,
        help="the manifest the tuning build was generated from",
    )
    parser.add_argument(
        "--output", type=Path, required=True, help="cache file to write"
    )
    parser.add_argument(
        "--ggml-header",
        type=Path,
        default=root / "vendor/llama.cpp/ggml/include/ggml.h",
        help="upstream ggml.h, for the ggml_type enum",
    )
    parser.add_argument(
        "--dispatch-db",
        type=Path,
        default=None,
        help="sqlite dispatch DB (schema 6) used to re-verify RV49 CPU-"
        "reference correctness evidence for every non-native winner "
        "(HI67); required whenever the export contains any non-native "
        "winner, including manual --seed overrides",
    )
    parser.add_argument(
        "--seed",
        type=Path,
        default=None,
        help="JSON file mapping dispatch_hex → stable_name "
        "for manual seed overrides (HI22)",
    )
    parser.add_argument(
        "--merge-into",
        type=Path,
        default=None,
        help="existing v4 cache whose generations should be retained",
    )
    parser.add_argument(
        "--keep-generations",
        type=int,
        default=3,
        help="maximum generations retained per portable key (default: 3)",
    )
    parser.add_argument(
        "--generation",
        type=int,
        default=0,
        help="monotonic generation number for this export (default: 0)",
    )
    args = parser.parse_args(argv)

    blob = build(
        args.measurements,
        args.manifest,
        args.ggml_header,
        dispatch_db=args.dispatch_db,
        seed_file=args.seed,
        merge_into=args.merge_into,
        keep_generations=args.keep_generations,
        generation=args.generation,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(
        prefix=args.output.name + ".tmp-", dir=args.output.parent
    )
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(blob)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_name, args.output)
    except BaseException:
        with contextlib.suppress(FileNotFoundError):
            os.unlink(temp_name)
        raise
    print(f"wrote {args.output} ({len(blob)} bytes)")
    print("Load it with GGML_HIP_DISPATCH_CACHE=<path> GGML_HIP_DISPATCH_MODE=replay")


if __name__ == "__main__":
    main()
