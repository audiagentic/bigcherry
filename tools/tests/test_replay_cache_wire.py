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

    def test_transform_id_round_trips_through_build_and_read_cache(self):
        # HI31: a winner reached through a routing transform (HI27/HI28)
        # must carry its transform_id through build()'s packing and back out
        # through read_cache() -- the whole point of the v5 wire bump.
        root = Path(tempfile.mkdtemp())
        manifest = root / "manifest.json"
        manifest.write_text(json.dumps({
            "source_revision": "b" * 40,
            "manifest_hash": "a" * 32,
            "candidates": [{
                "stable_name": "mmvf:f32:w1", "family": "mmvf",
                "source_class": "existing_alternative", "implementation_version": 1,
                "config": {"block_size": 1, "width": 1, "accumulator": "f32",
                          "type": "f32"},
            }],
        }), encoding="utf-8")
        ggml_h = root / "ggml.h"
        ggml_h.write_text("GGML_TYPE_F32 = 0,\n", encoding="utf-8")
        measurements = root / "measurements.jsonl"
        measurements.write_text("\n".join([
            json.dumps({"kind": "header", "source_revision": "b" * 40,
                        "manifest_hash": "a" * 32}),
            json.dumps({"kind": "result", "dispatch": "A" * 32,
                        "signature": "C" * 32, "winner": "mmvf:f32:w1",
                        "winner_transform": "transpose_weight_for_mmvf",
                        "winner_transform_id": 1,
                        "native": "blas:native:v1",
                        "promotion_status": "promoted"}),
        ]) + "\n", encoding="utf-8")

        blob = replay_cache.build(measurements, manifest, ggml_h)
        _, entries = replay_cache.read_cache(blob)
        self.assertEqual(len(entries), 1)
        self.assertEqual(entries[0]["transform_id"], 1)

    def test_plain_winner_has_zero_transform_id(self):
        _, manifest, ggml_h, measurements = self._fixture()
        blob = replay_cache.build(measurements, manifest, ggml_h)
        _, entries = replay_cache.read_cache(blob)
        self.assertEqual(entries[0]["transform_id"], 0)

    def test_v5_output_round_trips_wire_fields_and_digest(self):
        _, manifest, ggml_h, measurements = self._fixture()
        blob = replay_cache.build(measurements, manifest, ggml_h)

        self.assertEqual(len(blob), replay_cache.REPLAY_HEADER_SIZE + replay_cache.ENT_SIZE + 15)
        magic, version, artifact = struct.unpack_from("<III", blob)
        self.assertEqual((magic, version, artifact),
                         (replay_cache.MAGIC, replay_cache.REPLAY_VERSION,
                          replay_cache.ARTIFACT_VERSION))
        self.assertEqual(struct.unpack_from("<HHII", blob, 12), (1, 1, 1, 15))
        payload = blob[replay_cache.REPLAY_HEADER_SIZE:]
        expected = hashlib.blake2b(
            payload, digest_size=16, person=replay_cache.PERSON_DISPATCH).digest()
        self.assertEqual(blob[40:56], expected)
        entry = payload[:replay_cache.ENT_SIZE]
        self.assertEqual(entry[:16], bytes.fromhex("a" * 32))
        self.assertEqual(entry[16:32], bytes.fromhex("c" * 32))
        self.assertEqual(entry[32:36], b"\x00\x00\x00\x00")
        self.assertEqual(entry[54:70], bytes.fromhex("a" * 32))
        self.assertEqual(
            entry[70:86],
            bytes.fromhex(replay_cache.source_revision_digest("b" * 40)))
        self.assertEqual(entry[86:90], b"\x00\x00\x00\x00")
        self.assertEqual(payload[replay_cache.ENT_SIZE:], b"mmvq:native:v1\x00")

    def test_export_preflight_rejects_checksum_corruption_and_trailing_bytes(self):
        _, manifest, ggml_h, measurements = self._fixture()
        blob = replay_cache.build(measurements, manifest, ggml_h)
        self.assertEqual(
            replay_cache.validate_blob(blob, manifest_hash="a" * 32)["entry_count"], 1)

        corrupted = bytearray(blob)
        corrupted[-1] ^= 0x01
        with self.assertRaisesRegex(SystemExit, "checksum"):
            replay_cache.validate_blob(bytes(corrupted), manifest_hash="a" * 32)

        with self.assertRaisesRegex(SystemExit, "trailing"):
            replay_cache.validate_blob(blob + b"extra", manifest_hash="a" * 32)

    def test_export_preflight_rejects_manifest_mismatch_and_duplicate_entries(self):
        _, manifest, ggml_h, measurements = self._fixture()
        blob = replay_cache.build(measurements, manifest, ggml_h)
        with self.assertRaisesRegex(SystemExit, "manifest hash"):
            replay_cache.validate_blob(blob, manifest_hash="b" * 32)

        entry_start = replay_cache.REPLAY_HEADER_SIZE
        entry_end = entry_start + replay_cache.ENT_SIZE
        duplicate = bytearray(blob[:entry_start])
        duplicate.extend(blob[entry_start:entry_end])
        duplicate.extend(blob[entry_start:entry_end])
        duplicate.extend(blob[entry_end:])
        struct.pack_into("<I", duplicate, 16, 2)
        # Recompute the content digest so this exercises duplicate detection,
        # rather than stopping at the checksum boundary.
        duplicate[40:56] = replay_cache.blake2b_digest(
            bytes(duplicate[replay_cache.REPLAY_HEADER_SIZE:]))
        with self.assertRaisesRegex(SystemExit, "duplicate generation identity"):
            replay_cache.validate_blob(bytes(duplicate), manifest_hash="a" * 32)

    def test_v5_header_and_entry_sizes_are_explicit_wire_contract(self):
        self.assertEqual(replay_cache.REPLAY_HEADER_SIZE, 56)
        self.assertEqual(replay_cache.ENT_SIZE, 93)
        # Header: 3x u32, 2x u16, 2x u32, then two 16-byte digests.
        self.assertEqual(struct.calcsize("<IIIHHII16s16s"), 56)
        # Entry: two digests, name offset, implementation ABI, three i32
        # variants, four byte-sized variant fields, generation, (HI31, v5)
        # a u16 transform_id, and (HI74, v5) a trailing u8 match_kind.
        self.assertEqual(struct.calcsize("<16s16sIHiiiBBBB16s16sIHB"), 93)

    def test_deterministic_round_trip_is_independent_of_measurement_order(self):
        root, manifest, ggml_h, measurements = self._fixture()
        records = [
            {"kind": "header", "source_revision": "b" * 40,
             "manifest_hash": "a" * 32},
            {"kind": "result", "dispatch": "B" * 32,
             "signature": "D" * 32, "winner": "mmvq:native:v1",
             "native": "mmvq:native:v1"},
            {"kind": "result", "dispatch": "A" * 32,
             "signature": "C" * 32, "winner": "mmvq:native:v1",
             "native": "mmvq:native:v1"},
        ]
        measurements.write_text("\n".join(json.dumps(row) for row in records) + "\n",
                                encoding="utf-8")
        forward = replay_cache.build(measurements, manifest, ggml_h)
        measurements.write_text("\n".join(json.dumps(row) for row in [
            records[0], records[2], records[1]]) + "\n", encoding="utf-8")
        reverse = replay_cache.build(measurements, manifest, ggml_h)
        self.assertEqual(forward, reverse)

    def test_v4_merge_retains_bounded_generations_and_exact_provenance(self):
        root, manifest, ggml_h, measurements = self._fixture()
        first = root / "first.cache"
        first.write_bytes(replay_cache.build(
            measurements, manifest, ggml_h, generation=1))

        second_manifest = root / "manifest-second.json"
        second_manifest.write_text(json.dumps({
            "source_revision": "c" * 40,
            "manifest_hash": "d" * 32,
            "candidates": [{
                "stable_name": "mmvq:native:v1", "family": "mmvq",
                "source_class": "native_wrapper", "implementation_version": 1,
                "config": {},
            }],
        }), encoding="utf-8")
        second_measurements = root / "second.jsonl"
        second_measurements.write_text("\n".join([
            json.dumps({"kind": "header", "source_revision": "c" * 40,
                        "manifest_hash": "d" * 32}),
            json.dumps({"kind": "result", "dispatch": "A" * 32,
                        "signature": "C" * 32, "winner": "mmvq:native:v1",
                        "native": "mmvq:native:v1"}),
        ]) + "\n", encoding="utf-8")
        merged = replay_cache.build(
            second_measurements, second_manifest, ggml_h,
            merge_into=first, generation=2, keep_generations=2)
        _, entries = replay_cache.read_cache(merged)
        self.assertEqual([entry["generation"] for entry in entries], [2, 1])
        self.assertEqual([entry["manifest_hash"] for entry in entries],
                         ["d" * 32, "a" * 32])

    def test_v4_merge_replaces_same_generation_and_bounds_old_history(self):
        root, manifest, ggml_h, measurements = self._fixture()
        cache = root / "cache"
        cache.write_bytes(replay_cache.build(
            measurements, manifest, ggml_h, generation=1))
        for generation in (2, 3):
            cache.write_bytes(replay_cache.build(
                measurements, manifest, ggml_h, merge_into=cache,
                generation=generation, keep_generations=2))
        _, entries = replay_cache.read_cache(cache.read_bytes())
        self.assertEqual([entry["generation"] for entry in entries], [3, 2])

    def test_cpp_reader_rejects_partial_headers_and_records_before_offsets(self):
        source = (Path(__file__).resolve().parents[2] /
                  "src/ggml/src/ggml-cuda/hip-autotune-replay.cpp").read_text(encoding="utf-8")
        self.assertIn("bytes.size() < HDR_SIZE", source)
        self.assertIn("bytes.size() != expected", source)
        self.assertIn("content checksum mismatch", source)
        self.assertIn("Checksum before trusting any offset", source)

    def test_cpp_reader_enforces_abi_and_version_namespace_before_loading(self):
        source = (Path(__file__).resolve().parents[2] /
                  "src/ggml/src/ggml-cuda/hip-autotune-replay.cpp").read_text(encoding="utf-8")
        header = (Path(__file__).resolve().parents[2] /
                  "src/ggml/src/ggml-cuda/hip-autotune-replay.h").read_text(encoding="utf-8")
        for check in (
                "GGML_HIP_REPLAY_VERSION 5",
                "GGML_HIP_REPLAY_MAGIC   0x59484342u",
                "signature schema version mismatch",
                "hardware key schema version mismatch",
                "artifact version mismatch",
                "implementation_version that no longer matches"):
            self.assertTrue(check in source or check in header, check)

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

    def test_export_rejects_missing_signature_instead_of_substituting_dispatch(self):
        root, manifest, ggml_h, measurements = self._fixture()
        rows = measurements.read_text(encoding="utf-8").replace(
            ', "signature": "' + "C" * 32 + '"', "")
        measurements.write_text(rows, encoding="utf-8")
        with self.assertRaisesRegex(SystemExit, "missing an explicit signature"):
            replay_cache.build(measurements, manifest, ggml_h)

    def test_export_rejects_inconsistent_hardware_signature_dispatch(self):
        root, manifest, ggml_h, measurements = self._fixture()
        rows = [
            {"kind": "header", "source_revision": "b" * 40,
             "manifest_hash": "a" * 32},
            {"kind": "result", "dispatch": "A" * 32,
             "signature": "C" * 32, "hardware": "D" * 32,
             "winner": "mmvq:native:v1", "native": "mmvq:native:v1"},
        ]
        measurements.write_text("\n".join(json.dumps(row) for row in rows) + "\n",
                                encoding="utf-8")
        with self.assertRaisesRegex(SystemExit, "does not match its hardware/signature"):
            replay_cache.build(measurements, manifest, ggml_h)

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

    def test_cpp_reader_has_the_same_v5_boundary_and_fail_closed_checks(self):
        source = (Path(__file__).resolve().parents[2] /
                  "src/ggml/src/ggml-cuda/hip-autotune-replay.cpp").read_text(encoding="utf-8")
        header = (Path(__file__).resolve().parents[2] /
                  "src/ggml/src/ggml-cuda/hip-autotune-replay.h").read_text(encoding="utf-8")
        self.assertIn("#define GGML_HIP_REPLAY_VERSION 5", header)
        self.assertIn("constexpr size_t HDR_SIZE         = HDR_CONTENT + GGML_HIP_DIGEST_BYTES", source)
        self.assertIn("constexpr size_t ENT_SIZE      = ENT_MATCH_KIND + 1", source)
        for check in ("file is truncated", "content checksum mismatch", "duplicate generation identity",
                      "implementation_version that no longer matches"):
            self.assertIn(check, source)


if __name__ == "__main__":
    unittest.main()
