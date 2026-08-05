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
import re
import struct
from pathlib import Path
from typing import Any

from . import autotune_catalog

MAGIC = 0x59484342
REPLAY_VERSION = 1
ARTIFACT_VERSION = 1
SIGNATURE_SCHEMA_VERSION = 1
HARDWARE_SCHEMA_VERSION = 1
DIGEST_BYTES = 16
PERSON_DISPATCH = b"llama-dispatch"

# Entry layout, from the ENT_* constants in hip-autotune-replay.cpp.
ENT_SIZE = 2 * DIGEST_BYTES + 4 + 4 + 4 + 4 + 1 + 1 + 1 + 1


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


def read_results(path: Path) -> list[dict[str, Any]]:
    """Winning results from a measurements JSONL.

    Tolerates a truncated final line by construction -- a tuning run killed
    mid-write still yields every result it had already flushed.
    """
    results = []
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue  # truncated tail; everything before it stands
            if record.get("kind") == "result" and record.get("winner"):
                results.append(record)
    return results


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

    results = read_results(measurements)
    if not results:
        raise SystemExit(f"no winning results in {measurements}")

    # One entry per dispatch digest. A signature tuned more than once keeps the
    # last verdict, which is the one taken with the most warmed-up device.
    entries: dict[str, dict[str, Any]] = {}
    for record in results:
        entries[record["dispatch"]] = record

    # Seed overrides: manual winner selection for specific dispatch digests.
    # Loaded from a JSON file mapping dispatch_hex → stable_name.
    # These override any measured winner (HI22).
    seed_overrides: dict[str, str] = {}
    if seed_file and seed_file.is_file():
        try:
            seed_overrides = json.loads(seed_file.read_text(encoding="utf-8"))
        except json.JSONDecodeError as e:
            raise SystemExit(f"seed file {seed_file} is not valid JSON: {e}") from e

        # Validate: every candidate must exist in the manifest
        for digest_hex, stable_name in seed_overrides.items():
            if stable_name not in by_name:
                raise SystemExit(
                    f"seed override '{stable_name}' for dispatch {digest_hex[:16]}... "
                    f"is not in {manifest_path.name}"
                )

        # Apply overrides: replace or insert entries
        for digest_hex, stable_name in seed_overrides.items():
            entries[digest_hex] = {
                "dispatch": digest_hex,
                "winner": stable_name,
            }

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
            skipped += 1
            continue

        packed += digest  # ENT_DISPATCH
        packed += b"\x00" * DIGEST_BYTES  # ENT_SIGNATURE, unused by lookup
        packed += struct.pack("<I", intern(name))
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
    header += bytes.fromhex(manifest["manifest_hash"])
    header += blake2b_digest(payload)

    if skipped:
        print(f"  warning: {skipped} entry/entries had a malformed digest")
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
    args.output.write_bytes(blob)
    print(f"wrote {args.output} ({len(blob)} bytes)")
    print("Load it with GGML_HIP_DISPATCH_CACHE=<path> GGML_HIP_DISPATCH_MODE=replay")


if __name__ == "__main__":
    main()
