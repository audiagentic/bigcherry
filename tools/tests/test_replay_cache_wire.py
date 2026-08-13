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
