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
import hashlib
import json
import os
import re
import struct
import tempfile
from pathlib import Path
from typing import Any

from . import autotune_catalog

MAGIC = 0x59484342
REPLAY_VERSION = 3
ARTIFACT_VERSION = 1
SIGNATURE_SCHEMA_VERSION = 1
HARDWARE_SCHEMA_VERSION = 1
DIGEST_BYTES = 16
PERSON_DISPATCH = b"llama-dispatch"

# Wire-format boundary shared with hip-autotune-replay.cpp.  Version 3 is the
# first format that carries implementation_version, small_k, and src0_type;
# version 2 caches are not readable because those fields were absent.
REPLAY_HEADER_SIZE = 56

# Entry layout, from the ENT_* constants in hip-autotune-replay.cpp.
ENT_SIZE = 2 * DIGEST_BYTES + 4 + 2 + 4 + 4 + 4 + 1 + 1 + 1 + 1
assert ENT_SIZE == 54


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


def read_results(path: Path, *, require_header: bool = True) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
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
                raise SystemExit(f"malformed measurements JSONL: {path}: {exc}") from exc
            if record.get("kind") == "header":
                if header is not None:
                    raise SystemExit("duplicate measurements header")
                header = record
            elif record.get("kind") == "result" and record.get("winner"):
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


def _validate_promotion_gate(entries: dict[str, dict[str, Any]]) -> None:
    """Fail closed (HI34/P0): a non-native winner ships only after fresh
    confirmation *and* experiment-wide BH correction -- see tune_promotion.py.
    A row whose winner differs from its recorded native and whose
    promotion_status is not exactly "promoted" (raw pending_bh, an explicit
    rejection, or a legacy/missing-provenance record with no promotion_status
    at all) has never been through that gate and has no business on
    production's hot path.

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
        winner = record.get("winner")
        native = record.get("native")
        if winner == native:
            continue  # native state; always safe
        if record.get("promotion_status") != "promoted":
            violations.append((digest_hex, winner, record.get("promotion_status")))
    if violations:
        shown = "\n".join(
            f"  {digest[:16]}...: winner={winner!r} promotion_status={status!r}"
            for digest, winner, status in violations[:20]
        )
        more = f"\n  ... and {len(violations) - 20} more" if len(violations) > 20 else ""
        raise SystemExit(
            f"refusing to export: {len(violations)} non-native winner(s) lack "
            f"promotion_status=='promoted' (run `bigcherry tune-promote` first):\n"
            f"{shown}{more}"
        )


def build(
    measurements: Path,
    manifest_path: Path,
    ggml_h: Path,
    *,
    seed_file: Path | None = None,
) -> bytes:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    by_name = {c["stable_name"]: c for c in manifest["candidates"]}
    type_values = ggml_type_values(ggml_h)

    manifest_hash = _digest_hex(manifest.get("manifest_hash"), "manifest_hash")

    # A fully explicit seed file is an operator-authored provenance source and
    # may replace the measurements artifact entirely. Normal exports always
    # require the producer header.
    producer_header, results = read_results(measurements, require_header=seed_file is None)
    if not results:
        raise SystemExit(f"no winning results in {measurements}")
    if seed_file is None:
        expected_revision = manifest.get("source_revision")
        expected_manifest = manifest_hash
        if (not isinstance(expected_revision, str) or not expected_revision or
                not isinstance(expected_manifest, str) or not expected_manifest):
            raise SystemExit(
                "refusing to export: supplied manifest lacks producer provenance "
                "(source_revision/manifest_hash)"
            )
        if (producer_header is None or
                producer_header.get("source_revision") != expected_revision or
                producer_header.get("manifest_hash") != expected_manifest):
            raise SystemExit(
                "refusing to export: measurements producer provenance does not "
                "match the supplied manifest (source_revision/manifest_hash)"
            )

    # One entry per dispatch digest. Duplicate results are rejected by
    # read_results: silently taking the last row can combine incompatible
    # tuning epochs into one cache.
    entries: dict[str, dict[str, Any]] = {}
    for record in results:
        entries[record["dispatch"]] = record

    # Seed overrides: manual winner selection for specific dispatch digests.
    # Loaded from a JSON file mapping dispatch_hex → stable_name.
    # These override any measured winner (HI22).
    seed_overrides: dict[str, dict[str, str]] = {}
    if seed_file and seed_file.is_file():
        try:
            seed_overrides = json.loads(seed_file.read_text(encoding="utf-8"))
        except json.JSONDecodeError as e:
            raise SystemExit(f"seed file {seed_file} is not valid JSON: {e}") from e

        # Validate: every candidate must exist in the manifest
        normalized: dict[str, dict[str, str]] = {}
        for digest_hex, value in seed_overrides.items():
            digest_hex = _digest_hex(digest_hex, "seed override dispatch digest")
            if digest_hex in normalized:
                raise SystemExit(f"duplicate seed override dispatch digest: {digest_hex}")
            if isinstance(value, str):
                value = {"winner": value}
            if not isinstance(value, dict) or not isinstance(value.get("winner"), str):
                raise SystemExit("seed override must be a candidate name or object with winner")
            stable_name = value["winner"]
            if stable_name not in by_name:
                raise SystemExit(
                    f"seed override '{stable_name}' for dispatch {digest_hex[:16]}... "
                    f"is not in {manifest_path.name}"
                )
            existing = entries.get(digest_hex)
            signature = value.get("signature") if isinstance(value, dict) else None
            if existing is not None:
                signature = existing.get("signature")
            if not isinstance(signature, str) or not re.fullmatch(r"[0-9a-fA-F]{32}", signature):
                raise SystemExit(
                    f"seed override for unseen dispatch {digest_hex[:16]}... requires explicit signature"
                )
            normalized[digest_hex] = {"winner": stable_name, "signature": signature.lower()}
        seed_overrides = normalized

    # The gate runs on every dispatch NOT covered by an explicit seed --
    # a seed is a separate operator decision carrying its own provenance
    # (HI22), not the tuner's, and must not be blocked by (or need to pass)
    # the promotion state of whatever raw measurement happened to exist for
    # that same digest.
    _validate_promotion_gate({
        digest_hex: record for digest_hex, record in entries.items()
        if digest_hex not in seed_overrides
    })

    if seed_overrides:
        # Apply overrides: replace or insert entries
        for digest_hex, override in seed_overrides.items():
            entry = dict(entries.get(digest_hex, {}))
            entry.update({"dispatch": digest_hex, "winner": override["winner"],
                          "signature": override["signature"], "seeded": True})
            entries[digest_hex] = entry

    strings: dict[str, int] = {}
    string_blob = bytearray()

    def intern(name: str) -> int:
        if name not in strings:
            strings[name] = len(string_blob)
            string_blob.extend(name.encode("utf-8"))
            string_blob.append(0)
        return strings[name]

    packed = bytearray()
    skipped = 0
    for digest_hex, record in sorted(entries.items()):
        name = record["winner"]
        candidate = by_name.get(name)
        if candidate is None:
            # The manifest and the tuning run disagree, which means they came
            # from different generations. Skipping is wrong here -- it would
            # silently drop a winner -- so refuse the whole build.
            raise SystemExit(
                f"winner '{name}' is not in {manifest_path.name}; the "
                f"measurements were produced by a different catalog"
            )

        fields = autotune_catalog.variant_fields(candidate)
        type_name = fields["src0_type"]
        src0_type = 0 if type_name == "0" else type_values.get(type_name)
        if src0_type is None:
            raise SystemExit(f"unknown ggml type '{type_name}' for '{name}'")

        digest = bytes.fromhex(digest_hex)
        if len(digest) != DIGEST_BYTES:
            raise SystemExit(f"malformed dispatch digest: {digest_hex!r}")

        signature_hex = record.get("signature") or digest_hex
        if (not isinstance(signature_hex, str)
                or not re.fullmatch(r"[0-9a-fA-F]{32}", signature_hex)):
            raise SystemExit(f"winner {digest_hex} has malformed signature digest")
        signature = bytes.fromhex(signature_hex.lower())

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

    entry_count = len(packed) // ENT_SIZE
    payload = bytes(packed) + bytes(string_blob)

    header = bytearray()
    header += struct.pack("<III", MAGIC, REPLAY_VERSION, ARTIFACT_VERSION)
    header += struct.pack("<HH", SIGNATURE_SCHEMA_VERSION, HARDWARE_SCHEMA_VERSION)
    header += struct.pack("<II", entry_count, len(string_blob))
    header += bytes.fromhex(manifest_hash)
    header += blake2b_digest(payload)
    if len(header) != REPLAY_HEADER_SIZE:
        raise AssertionError("replay header layout drifted from the v3 wire format")

    print(
        f"  {entry_count} winner(s), {len(strings)} distinct candidate(s), "
        f"{len(string_blob)} string byte(s)"
    )
    print(f"  manifest {manifest['manifest_hash']}")
    return bytes(header) + payload


def main(argv: list[str] | None = None) -> None:
    root = Path(__file__).resolve().parent.parent.parent
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
        "--seed",
        type=Path,
        default=None,
        help="JSON file mapping dispatch_hex → stable_name "
        "for manual seed overrides (HI22)",
    )
    args = parser.parse_args(argv)

    blob = build(
        args.measurements,
        args.manifest,
        args.ggml_header,
        seed_file=args.seed,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=args.output.name + ".tmp-",
                                     dir=args.output.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(blob)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_name, args.output)
    except BaseException:
        try:
            os.unlink(temp_name)
        except FileNotFoundError:
            pass
        raise
    print(f"wrote {args.output} ({len(blob)} bytes)")
    print("Load it with GGML_HIP_DISPATCH_CACHE=<path> GGML_HIP_DISPATCH_MODE=replay")


if __name__ == "__main__":
    main()
