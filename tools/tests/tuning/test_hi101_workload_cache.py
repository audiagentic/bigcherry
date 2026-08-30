"""HI101: `bigcherry inventory workload-check --cache` -- read a real binary
v5 replay cache as the tuned-signature source instead of a
.measurements.jsonl file.

The tuned set is the UNION of entry signatures across all dispatch/generation
rows: v5 entries carry no separate hardware field (hardware is folded into
the portable dispatch digest), so this is the same hardware-agnostic
semantics the measurements path already uses, and workload_overlap() is
reused unchanged.
"""

from __future__ import annotations

import json
import shutil
import struct
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from bigcherry import replay_cache  # noqa: E402


def _python_executable() -> str:
    """Return a launchable interpreter even when sys.executable is an alias."""
    if Path(sys.executable).is_file():
        return sys.executable
    return shutil.which("python") or sys.executable

REPO_ROOT = Path(__file__).resolve().parents[3]
REFERENCE_V5_CACHE = (
    REPO_ROOT / "tools" / "tests" / "fixtures" / "replay" / "dispatch-v5-h36-27b.cache"
)

SIG_A = "a" * 32
SIG_B = "b" * 32
WINNER = "mmvq:native:v1"


def _v5_cache_with_signatures(tmp: Path, signatures: list[str]) -> Path:
    """Build a real binary v5 cache whose entries carry exactly `signatures`."""
    manifest_hash = "a" * 32
    source_revision = "b" * 40
    manifest_path = tmp / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "source_revision": source_revision,
                "manifest_hash": manifest_hash,
                "candidates": [
                    {
                        "stable_name": WINNER,
                        "family": "mmvq",
                        "source_class": "native_wrapper",
                        "implementation_version": 1,
                        "config": {},
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    ggml_h = tmp / "ggml.h"
    ggml_h.write_text("GGML_TYPE_F32 = 0,\n", encoding="utf-8")
    rows = [
        json.dumps(
            {
                "kind": "header",
                "source_revision": source_revision,
                "manifest_hash": manifest_hash,
            }
        )
    ]
    for index, signature in enumerate(signatures):
        rows.append(
            json.dumps(
                {
                    "kind": "result",
                    "dispatch": f"{index:032x}",
                    "signature": signature,
                    "winner": WINNER,
                    "native": WINNER,
                }
            )
        )
    measurements = tmp / "tune.measurements.jsonl"
    measurements.write_text("\n".join(rows) + "\n", encoding="utf-8")
    blob = replay_cache.build(measurements, manifest_path, ggml_h)
    cache_path = tmp / "cache.cache"
    cache_path.write_bytes(blob)
    return cache_path


def _empty_v5_cache(tmp: Path) -> Path:
    """A structurally valid v5 cache with zero entries (no winners exported)."""
    blob = bytearray(
        struct.pack(
            "<IIIHHII",
            replay_cache.MAGIC,
            replay_cache.REPLAY_VERSION,
            replay_cache.ARTIFACT_VERSION,
            replay_cache.SIGNATURE_SCHEMA_VERSION,
            replay_cache.HARDWARE_SCHEMA_VERSION,
            0,
            0,
        )
        + bytes(16)  # stored manifest hash (unbound: no entries)
    )
    assert len(blob) == 40
    blob += replay_cache.blake2b_digest(b"")[:16]
    assert len(blob) == replay_cache.REPLAY_HEADER_SIZE
    cache_path = tmp / "empty.cache"
    cache_path.write_bytes(bytes(blob))
    return cache_path


def _v5_to_v4(v5_blob: bytes) -> bytes:
    """Stamp a v5 blob as a structurally valid v4 cache (see test_replay_v5)."""
    version4 = bytearray(v5_blob)
    struct.pack_into("<I", version4, 4, replay_cache.REPLAY_VERSION_V4)
    entry_count, string_bytes = struct.unpack_from("<II", version4, 16)
    strings_at = replay_cache.REPLAY_HEADER_SIZE + entry_count * replay_cache.ENT_SIZE
    header = bytes(version4[: replay_cache.REPLAY_HEADER_SIZE])
    entries = bytearray()
    for i in range(entry_count):
        offset = replay_cache.REPLAY_HEADER_SIZE + i * replay_cache.ENT_SIZE
        entries += version4[offset : offset + replay_cache.ENT_SIZE_V4]
    strings = bytes(version4[strings_at : strings_at + string_bytes])
    payload = bytes(entries) + strings
    out = bytearray(header)
    struct.pack_into("<16s", out, 40, replay_cache.blake2b_digest(payload))
    out += payload
    return bytes(out)


def _record_path(tmp: Path) -> Path:
    record_path = tmp / "record.jsonl"
    record_path.write_text(
        "\n".join(
            [
                json.dumps({"kind": "header"}),
                json.dumps({"kind": "observation", "signature": SIG_A, "calls": 100}),
                json.dumps({"kind": "observation", "signature": SIG_B, "calls": 10}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return record_path


def _run_cli(*extra_args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [_python_executable(), "-m", "bigcherry", "inventory", "workload-check", *extra_args],
        cwd=str(Path(__file__).resolve().parents[2]),
        capture_output=True,
        text=True,
    )


class WorkloadCheckCacheCliTests(unittest.TestCase):
    def test_cache_path_reports_same_coverage_as_measurements_path(self):
        # One signature (SIG_A, the 99%-of-calls one) is tuned in both the
        # measurements file and the v5 cache; both paths must agree exactly.
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            record = _record_path(tmp_path)
            cache = _v5_cache_with_signatures(tmp_path, [SIG_A])
            measurements = tmp_path / "tune.measurements.jsonl"
            measurements.write_text(
                "\n".join(
                    [
                        json.dumps(
                            {"kind": "header", "source_revision": "b" * 40,
                             "manifest_hash": "a" * 32}
                        ),
                        json.dumps({"kind": "result", "signature": SIG_A}),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            via_cache = _run_cli(str(record), "--cache", str(cache))
            self.assertEqual(via_cache.returncode, 0, via_cache.stderr)
            self.assertIn("coverage", via_cache.stdout)
            self.assertIn("100 of 110 calls covered (90.9%)", via_cache.stdout)
            self.assertIn("1 of 2 observed signatures", via_cache.stdout)

            via_measurements = _run_cli(str(record), "--measurements", str(measurements))
            self.assertEqual(via_measurements.returncode, 0, via_measurements.stderr)
            # The coverage lines must be identical between the two paths;
            # only the tuned-source line differs.
            cache_lines = [
                line for line in via_cache.stdout.splitlines() if "tuned " not in line
            ]
            measurements_lines = [
                line
                for line in via_measurements.stdout.splitlines()
                if "tuned " not in line
            ]
            self.assertEqual(cache_lines, measurements_lines)

    def test_reference_cache_fixture_has_59_entries(self):
        # Committed acceptance artifact (HI74/HI15: dispatch-27b-v5.cache,
        # 0adcc3bb manifest, 59/59 winners usable). A missing committed
        # fixture is regression evidence, not a skippable environment gap.
        self.assertTrue(
            REFERENCE_V5_CACHE.is_file(),
            f"committed acceptance fixture missing: {REFERENCE_V5_CACHE}",
        )
        blob = REFERENCE_V5_CACHE.read_bytes()
        # This real historical campaign artifact was captured under
        # signature schema 1 (predates HI118/HI119's schema-2 bump) --
        # enforce_schema=False is required to inspect it at all now, and is
        # the documented offline-only escape hatch for exactly this case
        # (replay_cache.validate_blob's own docstring). Structural checks
        # (format/artifact version, checksum, bounds, duplicates) still run.
        header, entries = replay_cache.read_cache(blob, enforce_schema=False)
        self.assertEqual(header["version"], replay_cache.REPLAY_VERSION)
        self.assertEqual(header["signature_schema"], 1)
        self.assertEqual(len(entries), 59)
        # Do NOT assert len({e["signature"]}) == 59: the wire format permits
        # multiple entries collapsing to one signature through differing
        # dispatch/generation identities. Only the entry count is pinned.
        for entry in entries:
            self.assertRegex(entry["signature"], r"^[0-9a-f]{32}$")

    def test_reference_cache_fixture_is_rejected_by_default_schema_enforcement(self):
        # Positive proof of the schema bump's own fail-closed contract: the
        # SAME real historical artifact is correctly refused by the default
        # (production) read path now that it predates the current schema.
        blob = REFERENCE_V5_CACHE.read_bytes()
        with self.assertRaises(SystemExit):
            replay_cache.read_cache(blob)

    def test_zero_entry_cache_reports_zero_coverage_without_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            record = _record_path(tmp_path)
            cache = _empty_v5_cache(tmp_path)
            result = _run_cli(str(record), "--cache", str(cache))
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("0 of 2 observed signatures", result.stdout)
            self.assertIn("0.0%", result.stdout)

    def test_v4_cache_fails_loudly(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            record = _record_path(tmp_path)
            v5 = _v5_cache_with_signatures(tmp_path, [SIG_A])
            v4_path = tmp_path / "v4.cache"
            v4_path.write_bytes(_v5_to_v4(v5.read_bytes()))
            result = _run_cli(str(record), "--cache", str(v4_path))
            self.assertEqual(result.returncode, 1, result.stdout)
            self.assertIn("cannot read replay cache", result.stderr)

    def test_mutually_exclusive_tuned_sources(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            record = _record_path(tmp_path)
            cache = _v5_cache_with_signatures(tmp_path, [SIG_A])
            measurements = tmp_path / "tune.measurements.jsonl"
            measurements.write_text(
                json.dumps(
                    {"kind": "header", "source_revision": "b" * 40,
                     "manifest_hash": "a" * 32}
                )
                + "\n"
                + json.dumps({"kind": "result", "signature": SIG_A})
                + "\n",
                encoding="utf-8",
            )
            both = _run_cli(
                str(record),
                "--measurements",
                str(measurements),
                "--cache",
                str(cache),
            )
            self.assertEqual(both.returncode, 2)  # argparse: not allowed together
            neither = _run_cli(str(record))
            self.assertEqual(neither.returncode, 2)  # argparse: one is required


if __name__ == "__main__":
    unittest.main()
