"""HI74 replay v5 foundation tests: v4 legacy reader, v4->v5 merge,
unrecognised match_kind rejection, and v5 corruption-parity matrix.

The v4 layout is pinned by the historical dispatch-27b.cache campaign
artifact (version=4, 59 entries, 90-byte entries); the v5 layout adds the
trailing transform_id (u16) + match_kind (u8) for 93-byte entries. The
production C++ loader rejects v4 as rerun-required -- the only v4 reader in
this codebase is replay_cache.read_cache_legacy_v4, an offline
analysis/merge path.
"""

from __future__ import annotations

import json
import struct
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bigcherry import replay_cache  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
CAMPAIGN_V4 = REPO_ROOT / "docs" / "reference" / "h36-campaign-27b-r9700" / \
    "dispatch-27b.cache"


def _fixture(dispatch: str, root: Path, manifest_hash: str = "a" * 32):
    manifest = root / f"manifest-{dispatch[0]}.json"
    manifest.write_text(json.dumps({
        "source_revision": "b" * 40,
        "manifest_hash": manifest_hash,
        "candidates": [{
            "stable_name": "mmvq:native:v1", "family": "mmvq",
            "source_class": "native_wrapper", "implementation_version": 1,
            "config": {},
        }],
    }), encoding="utf-8")
    ggml_h = root / "ggml.h"
    ggml_h.write_text("GGML_TYPE_F32 = 0,\n", encoding="utf-8")
    measurements = root / f"measurements-{dispatch[0]}.jsonl"
    measurements.write_text("\n".join([
        json.dumps({"kind": "header", "source_revision": "b" * 40,
                    "manifest_hash": manifest_hash}),
        json.dumps({"kind": "result", "dispatch": dispatch,
                    "signature": "C" * 32, "winner": "mmvq:native:v1",
                    "native": "mmvq:native:v1"}),
    ]) + "\n", encoding="utf-8")
    return manifest, ggml_h, measurements


def _v5_to_v4(v5_blob: bytes) -> bytes:
    """Rebuild a v5 cache as a structurally valid v4 cache.

    Strips the trailing transform_id (u16) + match_kind (u8) from every
    entry, stamps the header version to 4, and recomputes the content
    checksum so the result passes every structural bound -- a genuine
    historical artifact, not a corrupted v5 file.
    """
    version4 = bytearray(v5_blob)
    struct.pack_into("<I", version4, 4, replay_cache.REPLAY_VERSION_V4)
    entry_count, string_bytes = struct.unpack_from("<II", version4, 16)
    strings_at = replay_cache.REPLAY_HEADER_SIZE + \
        entry_count * replay_cache.ENT_SIZE
    header = bytes(version4[:replay_cache.REPLAY_HEADER_SIZE])
    entries = bytearray()
    for i in range(entry_count):
        offset = replay_cache.REPLAY_HEADER_SIZE + i * replay_cache.ENT_SIZE
        entries += version4[offset:offset + replay_cache.ENT_SIZE_V4]
    strings = bytes(version4[strings_at:strings_at + string_bytes])
    payload = bytes(entries) + strings
    out = bytearray(header)
    struct.pack_into("<16s", out, 40,
                     replay_cache.blake2b_digest(payload))
    out += payload
    assert len(out) == replay_cache.REPLAY_HEADER_SIZE + \
        entry_count * replay_cache.ENT_SIZE_V4 + string_bytes
    return bytes(out)


class V4LegacyReaderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root = Path(tempfile.mkdtemp())

    def test_real_campaign_v4_artifact_parses_via_legacy_reader(self):
        if not CAMPAIGN_V4.is_file():
            self.skipTest("historical campaign artifact not present")
        blob = CAMPAIGN_V4.read_bytes()
        header, entries = replay_cache.read_cache_legacy_v4(blob)
        self.assertEqual(header["version"], 4)
        self.assertEqual(len(entries), 59)
        for entry in entries:
            self.assertEqual(len(entry["wire_entry"]), replay_cache.ENT_SIZE_V4)
            self.assertEqual(entry["transform_id"], 0)
            self.assertIsNone(entry["match_kind"])
            self.assertEqual(entry["portable_key"], entry["dispatch"])

    def test_synthetic_v4_blob_round_trips_legacy_reader(self):
        manifest, ggml_h, measurements = _fixture("A" * 32, self.root)
        v5 = replay_cache.build(measurements, manifest, ggml_h)
        v4 = _v5_to_v4(v5)
        header, entries = replay_cache.read_cache_legacy_v4(v4)
        self.assertEqual(header["version"], 4)
        self.assertEqual(len(entries), 1)
        self.assertEqual(entries[0]["winner"], "mmvq:native:v1")
        self.assertEqual(entries[0]["transform_id"], 0)
        self.assertIsNone(entries[0]["match_kind"])
        # Every non-trailing v4 field is byte-identical to the v5 original.
        v5_header, v5_entries = replay_cache.read_cache(v5)
        self.assertEqual(entries[0]["wire_entry"],
                         v5_entries[0]["wire_entry"][:replay_cache.ENT_SIZE_V4])

    def test_production_reader_rejects_v4_as_rerun_input(self):
        manifest, ggml_h, measurements = _fixture("A" * 32, self.root)
        v4 = _v5_to_v4(replay_cache.build(measurements, manifest, ggml_h))
        with self.assertRaises(SystemExit) as ctx:
            replay_cache.read_cache(v4)
        self.assertIn("format version mismatch", str(ctx.exception))

    def test_legacy_reader_rejects_v5(self):
        manifest, ggml_h, measurements = _fixture("A" * 32, self.root)
        v5 = replay_cache.build(measurements, manifest, ggml_h)
        with self.assertRaises(SystemExit) as ctx:
            replay_cache.read_cache_legacy_v4(v5)
        self.assertIn("expects a v4 cache", str(ctx.exception))

    def test_truncated_v4_is_rejected(self):
        manifest, ggml_h, measurements = _fixture("A" * 32, self.root)
        v4 = _v5_to_v4(replay_cache.build(measurements, manifest, ggml_h))
        for bad in (v4[:replay_cache.REPLAY_HEADER_SIZE - 4],
                    v4[:-7]):
            with self.assertRaises(SystemExit):
                replay_cache.read_cache_legacy_v4(bad)


class V4ToV5MergeTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp())

    def test_v4_cache_merges_into_fresh_v5_cache(self):
        # First "historical" export: dispatch B, converted to v4.
        manifest_b, ggml_h, measurements_b = _fixture("B" * 32, self.root)
        v4_path = self.root / "legacy.cache"
        v4_path.write_bytes(_v5_to_v4(
            replay_cache.build(measurements_b, manifest_b, ggml_h)))
        # Current export: dispatch A, merged into the v4 cache.
        manifest_a, _, measurements_a = _fixture("A" * 32, self.root)
        merged = replay_cache.build(measurements_a, manifest_a, ggml_h,
                                    merge_into=v4_path, generation=2)
        header, entries = replay_cache.read_cache(merged)
        self.assertEqual(header["version"], replay_cache.REPLAY_VERSION)
        self.assertEqual(len(entries), 2)
        # The v4-derived entry was repacked as a genuine 93-byte v5 entry.
        for entry in entries:
            self.assertEqual(len(entry["wire_entry"]), replay_cache.ENT_SIZE)
            self.assertEqual(entry["match_kind"],
                             replay_cache.MATCH_KIND_EXACT)
            self.assertEqual(entry["transform_id"], 0)
        generations = sorted(entry["generation"] for entry in entries)
        self.assertEqual(generations, [0, 2])


class UnrecognisedMatchKindTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp())

    def _v5_with_match_kind(self, kind: int) -> bytes:
        manifest, ggml_h, measurements = _fixture("A" * 32, self.root)
        blob = bytearray(replay_cache.build(measurements, manifest, ggml_h))
        offset = replay_cache.REPLAY_HEADER_SIZE + replay_cache.ENT_MATCH_KIND
        blob[offset] = kind
        # A future producer would emit a valid checksum; recompute it so the
        # match_kind check -- not the checksum -- is what rejects the file.
        struct.pack_into("<16s", blob, 40,
                         replay_cache.blake2b_digest(bytes(blob[56:])))
        return bytes(blob)

    def test_future_match_kind_is_rejected_not_reinterpreted(self):
        for kind in (1, 7, 255):
            with self.subTest(kind=kind):
                with self.assertRaises(SystemExit) as ctx:
                    replay_cache.read_cache(self._v5_with_match_kind(kind))
                self.assertIn("match_kind", str(ctx.exception))

    def test_exact_match_kind_still_round_trips(self):
        blob = self._v5_with_match_kind(replay_cache.MATCH_KIND_EXACT)
        _, entries = replay_cache.read_cache(blob)
        self.assertEqual(entries[0]["match_kind"],
                         replay_cache.MATCH_KIND_EXACT)


class V5CorruptionParityTests(unittest.TestCase):
    """v5 truncation/flip matrix mirroring the v4 corruption contract: every
    corrupted form is rejected deterministically before any offset in the
    file is trusted."""

    def setUp(self):
        self.root = Path(tempfile.mkdtemp())
        manifest, ggml_h, measurements = _fixture("A" * 32, self.root)
        self.blob = replay_cache.build(measurements, manifest, ggml_h)
        replay_cache.read_cache(self.blob)  # sanity: clean blob passes

    def test_truncation_matrix_is_rejected(self):
        for length in (0, 8, replay_cache.REPLAY_HEADER_SIZE - 1,
                       replay_cache.REPLAY_HEADER_SIZE + 30,
                       len(self.blob) - 1, len(self.blob) - 5):
            with self.subTest(length=length), self.assertRaises(SystemExit):
                replay_cache.read_cache(self.blob[:length])

    def test_trailing_bytes_are_rejected(self):
        for extra in (1, 4, 64):
            with self.subTest(extra=extra):
                with self.assertRaises(SystemExit) as ctx:
                    replay_cache.read_cache(self.blob + b"\x00" * extra)
                self.assertIn("truncated or has trailing bytes",
                              str(ctx.exception))

    def test_single_byte_flip_is_caught_by_checksum(self):
        for position in (57,  # first entry byte
                         replay_cache.REPLAY_HEADER_SIZE +
                         replay_cache.ENT_MANIFEST,  # mid-entry
                         len(self.blob) - 1):  # string table
            with self.subTest(position=position):
                flipped = bytearray(self.blob)
                flipped[position] ^= 0x01
                with self.assertRaises(SystemExit) as ctx:
                    replay_cache.read_cache(bytes(flipped))
                self.assertIn("checksum mismatch", str(ctx.exception))

    def test_entry_count_overflow_is_rejected(self):
        bad = bytearray(self.blob)
        struct.pack_into("<I", bad, 16, 1 << 20)  # entry_count
        with self.assertRaises(SystemExit):
            replay_cache.read_cache(bytes(bad))


class CppContractTests(unittest.TestCase):
    """Source-level contract: the C++ loader's rejection/warning strings must
    stay present so a future edit cannot silently change the fail-closed
    behaviour the Python side mirrors."""

    def test_cpp_loader_contract_strings(self):
        cpp = (REPO_ROOT / "src" / "ggml" / "src" / "ggml-cuda" /
               "hip-autotune-replay.cpp").read_text(encoding="utf-8")
        header = (REPO_ROOT / "src" / "ggml" / "src" / "ggml-cuda" /
                  "hip-autotune-replay.h").read_text(encoding="utf-8")
        for fragment in (
                "container format version mismatch",  # v4 -> rerun_required
                "match_kind this build does not recognise",
                "implementation_version that no longer matches",
                "artifact version mismatch",
                "signature schema version mismatch",
                "hardware key schema version mismatch"):
            with self.subTest(fragment=fragment):
                self.assertIn(fragment, cpp)
        for fragment in ("#define GGML_HIP_REPLAY_VERSION 5",
                         "GGML_HIP_REPLAY_MATCH_EXACT = 0"):
            with self.subTest(fragment=fragment):
                self.assertIn(fragment, header)


if __name__ == "__main__":
    unittest.main()
