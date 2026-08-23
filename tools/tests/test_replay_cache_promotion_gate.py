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
import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bigcherry import correctness_evidence as ce  # noqa: E402
from bigcherry import paths  # noqa: E402
from bigcherry import replay_cache  # noqa: E402


def entry(dispatch: str, winner: str, native: str = "native", **extra) -> dict:
    row = {"kind": "result", "dispatch": dispatch, "winner": winner, "native": native}
    row.update(extra)
    return row


def _passing_dispatch_db(
    tmp_path: Path, *, dispatch_hex: str, signature_hex: str, hardware_hex: str,
    native_name: str, candidate_name: str,
) -> Path:
    """HI67: a real schema-6 dispatch DB carrying passing correctness_evidence
    for the exact (dispatch, signature, hardware, candidate) binding a
    replay_cache.build() correctness-gate test needs -- see
    test_promotion_correctness_gate.py's _Base fixture for the same shape."""
    db_path = tmp_path / "gate.sqlite"
    conn = sqlite3.connect(str(db_path))
    conn.executescript((paths.SQL / "dispatch-db.sql").read_text(encoding="utf-8"))
    conn.execute(
        "INSERT INTO build (source_revision, manifest_hash, signature_schema, "
        "hardware_schema, variant_set) VALUES (?, ?, 1, 1, 'inventory')",
        ("b" * 40, "a" * 32),
    )
    build_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
    conn.execute(
        "INSERT INTO hardware (hardware_digest, architecture, architecture_code, "
        "wave_size, compute_units, feature_flags, canonical_json) VALUES "
        "(?, 'gfx1100', 1, 32, 96, 0, '{}')",
        (bytes.fromhex(hardware_hex),),
    )
    hardware_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
    conn.execute(
        "INSERT INTO signature (signature_digest, base_digest, schema_version, op, "
        "src0_type, src1_type, dst_type, m, n, k, canonical_json) VALUES "
        "(?, x'02', 1, 'MUL_MAT', 'q8_0', 'f32', 'f32', 1, 1, 1, '{}')",
        (bytes.fromhex(signature_hex),),
    )
    signature_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
    conn.execute(
        "INSERT INTO candidate (build_id, stable_name, family, source_class, "
        "implementation_version, architectures, architecture_mask, graph_safe, "
        "deterministic, config_json) VALUES (?, ?, 'mmvq', 'native_wrapper', "
        "1, '[]', 0, 1, 1, '{}')",
        (build_id, native_name),
    )
    native_candidate_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
    conn.execute(
        "INSERT INTO candidate (build_id, stable_name, family, source_class, "
        "implementation_version, architectures, architecture_mask, graph_safe, "
        "deterministic, config_json) VALUES (?, ?, 'mmvq', "
        "'existing_alternative', 1, '[]', 0, 1, 1, '{}')",
        (build_id, candidate_name),
    )
    candidate_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
    conn.execute(
        "INSERT INTO measurement (build_id, hardware_id, signature_id, dispatch_digest, "
        "candidate_id, objective, stage, accepted) VALUES (?, ?, ?, ?, ?, 'latency', "
        "'final', 1)",
        (build_id, hardware_id, signature_id, bytes.fromhex(dispatch_hex), candidate_id),
    )
    seeds = [
        ce.SeedEvidence(seed=i, reference_digest=f"d{i}", e_n_nmse=1e-05, e_c_nmse=2e-05,
                         max_abs_native=0.001, max_abs_candidate=0.0009,
                         native_execution_status="ok", candidate_execution_status="ok",
                         threshold_t=5e-4)
        for i in (1, 2, 3)
    ]
    aggregate = ce.aggregate_seed_evidence(seeds)
    conn.execute(
        "INSERT INTO correctness_evidence (build_id, hardware_id, signature_id, "
        "candidate_id, native_candidate_id, contract_version, threshold_t, "
        "headroom_fraction, e_n_nmse, e_c_nmse, max_abs_native, max_abs_candidate, "
        "seed_count, tool_version) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'v1')",
        (build_id, hardware_id, signature_id, candidate_id, native_candidate_id,
         aggregate.contract_version, aggregate.threshold_t, aggregate.headroom_fraction,
         aggregate.e_n_nmse, aggregate.e_c_nmse, aggregate.max_abs_native,
         aggregate.max_abs_candidate, len(aggregate.seed_rows)),
    )
    evidence_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
    for row in aggregate.seed_rows:
        conn.execute(
            "INSERT INTO correctness_evidence_seed (correctness_evidence_id, seed, "
            "reference_digest, e_n_nmse, e_c_nmse, max_abs_native, max_abs_candidate, "
            "native_execution_status, candidate_execution_status, threshold_t) VALUES "
            "(?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (evidence_id, row.seed, row.reference_digest, row.e_n_nmse, row.e_c_nmse,
             row.max_abs_native, row.max_abs_candidate, row.native_execution_status,
             row.candidate_execution_status, row.threshold_t),
        )
    conn.commit()
    conn.close()
    return db_path


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

    def test_seed_override_native_not_in_manifest_is_refused_fast(self):
        # HI89: an operator typo in `native` used to only surface later as
        # an opaque CorrectnessGateError from the binding-resolution DB
        # lookup. Now it fails at seed-load time, matching `winner`'s own
        # existing manifest-bound validation, with a clear reason.
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest, ggml_h, measurements = self._fixture(root)
            seed_file = root / "seed.json"
            seed_file.write_text(json.dumps({"b" * 32: {
                "winner": "mmvq:seeded:v1", "signature": "c" * 32,
                "hardware": "d" * 32, "native": "mmvq:typo:v1",
            }}), encoding="utf-8")
            with self.assertRaisesRegex(SystemExit, "native 'mmvq:typo:v1'.*not in the manifest"):
                replay_cache.build(measurements, manifest, ggml_h, seed_file=seed_file)

    def test_seed_override_native_not_a_native_wrapper_is_refused(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / "manifest.json"
            manifest.write_text(json.dumps({
                "manifest_hash": "a" * 32,
                "candidates": [
                    {"stable_name": "mmvq:native:v1", "family": "mmvq",
                     "source_class": "native_wrapper", "config": {}},
                    {"stable_name": "mmvq:seeded:v1", "family": "mmvq",
                     "source_class": "native_wrapper", "config": {}},
                    {"stable_name": "mmvq:candidate:v1", "family": "mmvq",
                     "source_class": "existing_alternative", "config": {}},
                ],
            }), encoding="utf-8")
            ggml_h = root / "ggml.h"
            ggml_h.write_text("GGML_TYPE_F32 = 0,\n", encoding="utf-8")
            measurements = root / "tune.measurements.jsonl"
            measurements.write_text(json.dumps({
                "kind": "result", "dispatch": "b" * 32,
                "winner": "mmvq:native:v1", "native": "mmvq:native:v1",
            }) + "\n", encoding="utf-8")
            seed_file = root / "seed.json"
            seed_file.write_text(json.dumps({"b" * 32: {
                "winner": "mmvq:seeded:v1", "signature": "c" * 32,
                "hardware": "d" * 32, "native": "mmvq:candidate:v1",
            }}), encoding="utf-8")
            with self.assertRaisesRegex(SystemExit, "not a native_wrapper candidate"):
                replay_cache.build(measurements, manifest, ggml_h, seed_file=seed_file)

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

    def test_explicit_seed_bypasses_the_promotion_status_gate_not_the_correctness_gate(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest, ggml_h, measurements = self._fixture(root)
            # Same unpromoted non-native winner as above, but now overridden
            # by an explicit operator seed for this exact dispatch digest.
            # HI67: the seed still bypasses _validate_promotion_gate (HI22's
            # own contract), but NOT the RV49 correctness gate -- it must
            # carry a real hardware identity and a passing dispatch_db. The
            # dispatch digest itself must be the real portable key derived
            # from hardware+signature (replay_cache's own consistency check
            # rejects a hand-picked digest once "hardware" is present).
            hardware_hex = "d" * 32
            signature_hex = "c" * 32
            dispatch_hex = replay_cache.portable_tuning_key(hardware_hex, signature_hex)
            measurements.write_text(json.dumps({
                "kind": "result", "dispatch": dispatch_hex,
                "winner": "mmvq:seeded:v1", "native": "mmvq:native:v1",
                "signature": signature_hex,
            }) + "\n", encoding="utf-8")
            seed_file = root / "seed.json"
            seed_file.write_text(json.dumps({dispatch_hex: {
                "winner": "mmvq:seeded:v1", "signature": signature_hex,
                "hardware": hardware_hex,
            }}), encoding="utf-8")
            with self.assertRaisesRegex(SystemExit, "no --dispatch-db"):
                replay_cache.build(measurements, manifest, ggml_h, seed_file=seed_file)
            dispatch_db = _passing_dispatch_db(
                root, dispatch_hex=dispatch_hex, signature_hex=signature_hex,
                hardware_hex=hardware_hex, native_name="mmvq:native:v1",
                candidate_name="mmvq:seeded:v1",
            )
            blob = replay_cache.build(
                measurements, manifest, ggml_h, seed_file=seed_file, dispatch_db=dispatch_db,
            )
            self.assertTrue(blob)  # exported without raising, backed by real evidence

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
            # HI67: the correctness gate resolves against the FINAL (post-
            # precedence) signature -- the measurement's "e" * 32, not the
            # override's own "c" * 32 -- matching the assertion below that
            # the measurement's signature is what actually ships. The
            # dispatch digest must be the real portable key for hardware +
            # the WINNING signature, or the hardware/signature consistency
            # check below rejects it once "hardware" is present.
            hardware_hex = "d" * 32
            winning_signature_hex = "e" * 32
            dispatch_hex = replay_cache.portable_tuning_key(
                hardware_hex, winning_signature_hex
            )
            measurements.write_text(json.dumps({
                "kind": "result", "dispatch": dispatch_hex,
                "winner": "mmvq:native:v1", "native": "mmvq:native:v1",
                "signature": winning_signature_hex,
            }) + "\n", encoding="utf-8")
            seed_file = root / "seed.json"
            seed_file.write_text(json.dumps({dispatch_hex: {
                "winner": "mmvq:seeded:v1", "signature": "c" * 32,
                "hardware": hardware_hex,
            }}), encoding="utf-8")
            dispatch_db = _passing_dispatch_db(
                root, dispatch_hex=dispatch_hex, signature_hex=winning_signature_hex,
                hardware_hex=hardware_hex, native_name="mmvq:native:v1",
                candidate_name="mmvq:seeded:v1",
            )
            blob = replay_cache.build(
                measurements, manifest, ggml_h, seed_file=seed_file, dispatch_db=dispatch_db,
            )
            self.assertEqual(
                blob[replay_cache.REPLAY_HEADER_SIZE + 16:replay_cache.REPLAY_HEADER_SIZE + 32],
                bytes.fromhex(winning_signature_hex),
            )
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
