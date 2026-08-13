"""HI51/HI56 replay-cache wire-format and provenance boundary tests."""

from __future__ import annotations

import hashlib
import json
import struct
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bigcherry import replay_cache  # noqa: E402


class ReplayCacheWireTests(unittest.TestCase):
    def _fixture(self, *, dispatch: str = "A" * 32, manifest_hash: str = "a" * 32):
        root = Path(tempfile.mkdtemp())
        manifest = root / "manifest.json"
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
        measurements = root / "measurements.jsonl"
        measurements.write_text("\n".join([
            json.dumps({"kind": "header", "source_revision": "b" * 40,
                        "manifest_hash": manifest_hash}),
            json.dumps({"kind": "result", "dispatch": dispatch,
                        "signature": "C" * 32, "winner": "mmvq:native:v1",
                        "native": "mmvq:native:v1"}),
        ]) + "\n", encoding="utf-8")
        return root, manifest, ggml_h, measurements

    def test_v3_output_round_trips_wire_fields_and_digest(self):
        _, manifest, ggml_h, measurements = self._fixture()
        blob = replay_cache.build(measurements, manifest, ggml_h)

        self.assertEqual(len(blob), replay_cache.REPLAY_HEADER_SIZE + replay_cache.ENT_SIZE + 15)
        magic, version, artifact = struct.unpack_from("<III", blob)
        self.assertEqual((magic, version, artifact),
                         (replay_cache.MAGIC, 3, replay_cache.ARTIFACT_VERSION))
        self.assertEqual(struct.unpack_from("<HHII", blob, 12), (1, 1, 1, 15))
        payload = blob[replay_cache.REPLAY_HEADER_SIZE:]
        expected = hashlib.blake2b(
            payload, digest_size=16, person=replay_cache.PERSON_DISPATCH).digest()
        self.assertEqual(blob[40:56], expected)
        entry = payload[:replay_cache.ENT_SIZE]
        self.assertEqual(entry[:16], bytes.fromhex("a" * 32))
        self.assertEqual(entry[16:32], bytes.fromhex("c" * 32))
        self.assertEqual(entry[32:36], b"\x00\x00\x00\x00")
        self.assertEqual(payload[replay_cache.ENT_SIZE:], b"mmvq:native:v1\x00")

    def test_duplicate_dispatch_is_rejected_case_insensitively(self):
        root, _, _, measurements = self._fixture()
        measurements.write_text("\n".join([
            json.dumps({"kind": "header"}),
            json.dumps({"kind": "result", "dispatch": "a" * 32, "winner": "x"}),
            json.dumps({"kind": "result", "dispatch": "A" * 32, "winner": "x"}),
        ]) + "\n", encoding="utf-8")
        with self.assertRaisesRegex(SystemExit, "duplicate dispatch digest"):
            replay_cache.read_results(measurements)

    def test_malformed_manifest_digest_is_rejected_before_writing(self):
        _, manifest, ggml_h, measurements = self._fixture(manifest_hash="short")
        with self.assertRaisesRegex(SystemExit, "manifest_hash"):
            replay_cache.build(measurements, manifest, ggml_h)

    def test_portable_key_is_stable_across_build_provenance(self):
        key = replay_cache.portable_tuning_key("a" * 32, "b" * 32)
        self.assertEqual(key, replay_cache.portable_tuning_key("A" * 32, "B" * 32))
        self.assertNotEqual(key, replay_cache.portable_tuning_key("a" * 32, "b" * 32, "throughput"))

    def test_provenance_namespace_is_strict_and_normalized(self):
        value = replay_cache.validate_provenance_namespace({
            "source_revision": "A" * 40,
            "manifest_hash": "B" * 32,
            "build_descriptor_hash": "C" * 32,
        })
        self.assertEqual(value["source_revision"], "a" * 40)
        self.assertEqual(value["manifest_hash"], "b" * 32)
        with self.assertRaisesRegex(SystemExit, "source_revision"):
            replay_cache.validate_provenance_namespace({
                "source_revision": "not-a-revision", "manifest_hash": "a" * 32})

    def test_newest_winner_selection_is_order_independent_and_bounded(self):
        def row(generation, winner, revision):
            return {"hardware": "a" * 32, "signature": "b" * 32,
                    "winner": winner, "generation": generation,
                    "source_revision": revision, "manifest_hash": "c" * 32}

        records = [row(1, "old", "1" * 40), row(3, "new", "3" * 40),
                   row(2, "middle", "2" * 40)]
        forward = replay_cache.select_newest_winners(records)
        reverse = replay_cache.select_newest_winners(reversed(records))
        self.assertEqual([(r["generation"], r["winner"]) for r in forward], [(3, "new")])
        self.assertEqual([(r["generation"], r["winner"]) for r in reverse], [(3, "new")])
        kept = replay_cache.select_newest_winners(records, keep_generations=2)
        self.assertEqual([r["generation"] for r in kept], [3, 2])

    def test_same_generation_conflict_fails_closed(self):
        common = {"hardware": "a" * 32, "signature": "b" * 32,
                  "generation": 4, "source_revision": "4" * 40,
                  "manifest_hash": "c" * 32}
        with self.assertRaisesRegex(SystemExit, "conflicting winners"):
            replay_cache.select_newest_winners([
                {**common, "winner": "a"}, {**common, "winner": "b"}])

    def test_cpp_reader_has_the_same_v3_boundary_and_fail_closed_checks(self):
        source = (Path(__file__).resolve().parents[2] /
                  "src/ggml/src/ggml-cuda/hip-autotune-replay.cpp").read_text(encoding="utf-8")
        header = (Path(__file__).resolve().parents[2] /
                  "src/ggml/src/ggml-cuda/hip-autotune-replay.h").read_text(encoding="utf-8")
        self.assertIn("#define GGML_HIP_REPLAY_VERSION 3", header)
        self.assertIn("constexpr size_t HDR_SIZE         = HDR_CONTENT + GGML_HIP_DIGEST_BYTES", source)
        self.assertIn("constexpr size_t ENT_SIZE      = ENT_SRC0_TYPE + 1", source)
        for check in ("file is truncated", "content checksum mismatch", "duplicate dispatch digest",
                      "entry implementation version differs from candidate registry"):
            self.assertIn(check, source)


if __name__ == "__main__":
    unittest.main()
