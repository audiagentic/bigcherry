"""Fail-closed promotion gate for the replay-cache exporter (HI34/P0).

Restores the pre-reset invariant that a non-native winner cannot reach a
replay cache without passing tune_promotion.py's experiment-wide BH
correction first. Tests `_validate_promotion_gate` directly rather than the
full `build()` round trip -- that needs a real manifest/ggml.h/variant-field
fixture unrelated to this gate, and is exercised end to end on real hardware
separately.
"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bigcherry import replay_cache  # noqa: E402


def entry(dispatch: str, winner: str, native: str = "native", **extra) -> dict:
    row = {"kind": "result", "dispatch": dispatch, "winner": winner, "native": native}
    row.update(extra)
    return row


class PromotionGateTests(unittest.TestCase):

    def test_native_winner_always_exports(self):
        entries = {"a" * 32: entry("a" * 32, "native", "native")}
        replay_cache._validate_promotion_gate(entries)  # must not raise

    def test_failed_native_measurement_cannot_export(self):
        entries = {"a" * 32: entry("a" * 32, "native", "native",
                                   measurement_failure=True)}
        with self.assertRaisesRegex(SystemExit, "measurement_failure"):
            replay_cache._validate_promotion_gate(entries)

    def test_promoted_non_native_exports(self):
        entries = {"a" * 32: entry("a" * 32, "candidate", "native",
                                   promotion_status="promoted")}
        replay_cache._validate_promotion_gate(entries)  # must not raise

    def test_raw_pending_bh_cannot_export(self):
        entries = {"a" * 32: entry("a" * 32, "candidate", "native",
                                   promotion_status="pending_bh")}
        with self.assertRaisesRegex(SystemExit, "promotion_status"):
            replay_cache._validate_promotion_gate(entries)

    def test_rejected_bh_cannot_export(self):
        entries = {"a" * 32: entry("a" * 32, "candidate", "native",
                                   promotion_status="rejected_bh")}
        with self.assertRaises(SystemExit):
            replay_cache._validate_promotion_gate(entries)

    def test_missing_promotion_metadata_cannot_export(self):
        # A non-native winner with no promotion_status at all -- e.g. a
        # measurements file from before this gate existed. Fail closed, not
        # a silent pass-through.
        entries = {"a" * 32: entry("a" * 32, "candidate", "native")}
        with self.assertRaises(SystemExit):
            replay_cache._validate_promotion_gate(entries)

    def test_missing_native_field_treated_as_non_native(self):
        entries = {"a" * 32: {"kind": "result", "dispatch": "a" * 32,
                              "winner": "candidate"}}
        with self.assertRaises(SystemExit):
            replay_cache._validate_promotion_gate(entries)

    def test_mixed_batch_reports_every_violation(self):
        entries = {
            "a" * 32: entry("a" * 32, "native", "native"),
            "b" * 32: entry("b" * 32, "candidate", "native",
                            promotion_status="promoted"),
            "c" * 32: entry("c" * 32, "candidate", "native",
                            promotion_status="pending_bh"),
            "d" * 32: entry("d" * 32, "candidate", "native",
                            promotion_status="rejected_bh"),
        }
        with self.assertRaisesRegex(SystemExit, "2 unsafe measurement result"):
            replay_cache._validate_promotion_gate(entries)


class SeedOverrideBypassesGateTests(unittest.TestCase):
    """HI22/P0: an explicit --seed override is a separate operator decision
    with its own provenance, not the tuner's -- build() must apply it AFTER
    _validate_promotion_gate rather than let it satisfy or dodge the gate,
    so a seeded entry can export even though its raw measurement never
    reached promotion_status=='promoted'.
    """

    def _fixture(self, root: Path) -> tuple[Path, Path, Path]:
        manifest = root / "manifest.json"
        manifest.write_text(json.dumps({
            "manifest_hash": "a" * 32,
            "candidates": [
                {"stable_name": "mmvq:native:v1", "family": "mmvq",
                 "source_class": "native_wrapper", "config": {}},
                {"stable_name": "mmvq:seeded:v1", "family": "mmvq",
                 "source_class": "native_wrapper", "config": {}},
            ],
        }), encoding="utf-8")
        ggml_h = root / "ggml.h"
        ggml_h.write_text("GGML_TYPE_F32 = 0,\n", encoding="utf-8")
        measurements = root / "tune.measurements.jsonl"
        # No promotion_status at all: this dispatch never went through
        # tune-promote. Without a seed override, build() must refuse it.
        measurements.write_text(json.dumps({
            "kind": "result", "dispatch": "b" * 32,
            "winner": "mmvq:native:v1", "native": "mmvq:native:v1",
        }) + "\n", encoding="utf-8")
        return manifest, ggml_h, measurements

    def test_unseeded_unpromoted_entry_is_refused(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest, ggml_h, measurements = self._fixture(root)
            # Overwrite with an unpromoted non-native winner.
            measurements.write_text(json.dumps({
                "kind": "result", "dispatch": "b" * 32,
                "winner": "mmvq:seeded:v1", "native": "mmvq:native:v1",
                "signature": "c" * 32,
            }) + "\n", encoding="utf-8")
            with self.assertRaises(SystemExit):
                replay_cache.build(measurements, manifest, ggml_h)

    def test_manifest_without_provenance_is_refused(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest, ggml_h, measurements = self._fixture(root)
            measurements.write_text("\n".join([
                json.dumps({"kind": "header", "source_revision": "a" * 40,
                            "manifest_hash": "a" * 32}),
                json.dumps({"kind": "result", "dispatch": "b" * 32,
                            "winner": "mmvq:native:v1", "native": "mmvq:native:v1"}),
            ]) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(SystemExit, "lacks producer provenance"):
                replay_cache.build(measurements, manifest, ggml_h)

    def test_explicit_seed_bypasses_the_gate(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest, ggml_h, measurements = self._fixture(root)
            # Same unpromoted non-native winner as above, but now overridden
            # by an explicit operator seed for this exact dispatch digest.
            measurements.write_text(json.dumps({
                "kind": "result", "dispatch": "b" * 32,
                "winner": "mmvq:seeded:v1", "native": "mmvq:native:v1",
                "signature": "c" * 32,
            }) + "\n", encoding="utf-8")
            seed_file = root / "seed.json"
            seed_file.write_text(json.dumps({"b" * 32: {"winner": "mmvq:seeded:v1",
                                                   "signature": "c" * 32}}), encoding="utf-8")
            blob = replay_cache.build(measurements, manifest, ggml_h, seed_file=seed_file)
            self.assertTrue(blob)  # exported without raising

    def test_envelope_requires_manifest_provenance(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest, ggml_h, measurements = self._fixture(root)
            document = {
                "version": 1,
                "provenance": {"source_revision": "a" * 40,
                                "manifest_hash": "f" * 32},
                "overrides": {"b" * 32: {"winner": "mmvq:seeded:v1",
                                          "signature": "c" * 32}},
            }
            seed_file = root / "seed.json"
            seed_file.write_text(json.dumps(document), encoding="utf-8")
            with self.assertRaisesRegex(SystemExit, "provenance"):
                replay_cache.build(measurements, manifest, ggml_h, seed_file=seed_file)

    def test_candidate_identity_digest_is_manifest_bound(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest, ggml_h, measurements = self._fixture(root)
            seed_file = root / "seed.json"
            seed_file.write_text(json.dumps({"b" * 32: {
                "winner": "mmvq:seeded:v1", "signature": "c" * 32,
                "candidate_digest": "d" * 32,
            }}), encoding="utf-8")
            with self.assertRaisesRegex(SystemExit, "candidate identity"):
                replay_cache.build(measurements, manifest, ggml_h, seed_file=seed_file)

    def test_override_winner_precedes_measurement_but_measurement_signature_wins(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest, ggml_h, measurements = self._fixture(root)
            measurements.write_text(json.dumps({
                "kind": "result", "dispatch": "b" * 32,
                "winner": "mmvq:native:v1", "native": "mmvq:native:v1",
                "signature": "e" * 32,
            }) + "\n", encoding="utf-8")
            seed_file = root / "seed.json"
            seed_file.write_text(json.dumps({"b" * 32: {
                "winner": "mmvq:seeded:v1", "signature": "c" * 32,
            }}), encoding="utf-8")
            blob = replay_cache.build(measurements, manifest, ggml_h, seed_file=seed_file)
            self.assertEqual(blob[replay_cache.REPLAY_HEADER_SIZE + 16:
                                  replay_cache.REPLAY_HEADER_SIZE + 32], bytes.fromhex("e" * 32))
            self.assertIn(b"mmvq:seeded:v1\0", blob)


class ProducerProvenanceTests(unittest.TestCase):

    def test_measurements_from_different_manifest_are_refused(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / "manifest.json"
            manifest.write_text(json.dumps({
                "source_revision": "b" * 40,
                "manifest_hash": "b" * 32,
                "candidates": [],
            }), encoding="utf-8")
            ggml_h = root / "ggml.h"
            ggml_h.write_text("GGML_TYPE_F32 = 0,\n", encoding="utf-8")
            measurements = root / "measurements.jsonl"
            measurements.write_text("\n".join([
                json.dumps({"kind": "header", "source_revision": "a" * 40,
                            "manifest_hash": "a" * 32}),
                json.dumps({"kind": "result", "dispatch": "a" * 32,
                            "winner": "native", "native": "native"}),
            ]) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(SystemExit, "producer source_revision"):
                replay_cache.build(measurements, manifest, ggml_h)

    def test_manifest_hash_mismatch_alone_is_accepted_when_source_revision_matches(self):
        # A replay-slim manifest always has a different manifest_hash than the
        # workload/full-max manifest that produced the tuning measurements --
        # manifest_hash() is scoped to variant_set and candidate set, and
        # replay-slim narrows both by design. Only source_revision identifies
        # a compatible producer/consumer pair; requiring manifest_hash to
        # match too made every replay-slim export refuse unconditionally.
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / "manifest.json"
            manifest.write_text(json.dumps({
                "source_revision": "a" * 40,
                "manifest_hash": "b" * 32,
                "candidates": [
                    {"stable_name": "mmvq:native:v1", "family": "mmvq",
                     "source_class": "native_wrapper", "config": {}},
                ],
            }), encoding="utf-8")
            ggml_h = root / "ggml.h"
            ggml_h.write_text("GGML_TYPE_F32 = 0,\n", encoding="utf-8")
            measurements = root / "measurements.jsonl"
            measurements.write_text("\n".join([
                json.dumps({"kind": "header", "source_revision": "a" * 40,
                            "manifest_hash": "c" * 32}),
                json.dumps({"kind": "result", "dispatch": "a" * 32,
                            "winner": "mmvq:native:v1", "native": "mmvq:native:v1",
                            "signature": "e" * 32}),
            ]) + "\n", encoding="utf-8")
            blob = replay_cache.build(measurements, manifest, ggml_h)
            self.assertGreater(len(blob), replay_cache.REPLAY_HEADER_SIZE)


if __name__ == "__main__":
    unittest.main()
