"""RE10-min (RV50-locked scope): tools/bigcherry/lifecycle.py stage coverage.

Each stage function is exercised end to end against a real ArtifactStore and
real filesystem, with only the actual runtime-binary launch replaced
(bigcherry.lifecycle.subprocess.run is patched via _fake_run -- the real
binary only runs on Brutus/Linux hardware; a published .py bundle member
cannot be exec'd directly on a Windows dev box either way). Everything
downstream of that launch runs unmodified: inventory.py's real SQLite
writer, tune_promotion.py's real BH/bootstrap statistics, replay_cache.py's
real wire-format builder, and ab_benchmark.py's real coverage validator.
Fixtures reuse the same record/tuning JSONL shapes as test_inventory.py and
the same manifest/ggml.h shape as test_replay_cache_wire.py.
"""

from __future__ import annotations

import dataclasses
import json
import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bigcherry import lifecycle  # noqa: E402
from bigcherry import provenance  # noqa: E402
from bigcherry.artifacts import ArtifactLocator, ArtifactStore  # noqa: E402
from bigcherry.context import ProjectContext  # noqa: E402
from bigcherry.runtime_smoke import RuntimeSmokeSpec  # noqa: E402

RECORD_HEADER = {
    "kind": "header",
    "source_revision": "abcdef1234567890",
    "manifest_hash": "deadbeef00112233",
    "signature_schema": 1,
    "hardware_schema": 1,
    "variant_set": "inventory",
}

RECORD_OBS = {
    "kind": "observation",
    "hardware": "a" * 32,
    "signature": "b" * 32,
    "native": "mmq:native:v1",
    "canonical": {
        "op": "MUL_MAT",
        "src0_type": 8,
        "src1_type": 0,
        "dst_type": 0,
        "ne0": [64, 512],
        "ned": [64, 128],
    },
    "hardware_key": {
        "architecture_code": "gfx1100",
        "wave_size": 64,
        "compute_units": 60,
        "feature_flags": 1,
    },
    "calls": 5,
    "est_bytes": 2048,
    "devices": [0],
}

TUNING_HEADER = {
    "kind": "header",
    "artifact_version": 1,
    "source_revision": "abcdef1234567890",
    "manifest_hash": "deadbeef00112233",
    "variant_set": "workload-max",
    "build_descriptor_hash": "build-descriptor-test",
}

TUNING_RESULT_NATIVE = {
    "kind": "result",
    "dispatch": "e" * 32,
    "winner": "mmq:native:v1",
    "improvement_pct": 0.0,
    "generated": 3,
    "eligible": 3,
    "measured": 2,
    "reason": "native retained",
    "candidates": [
        {
            "name": "mmq:native:v1",
            "status": "ok",
            "median_us": 1.5,
            "mad_us": 0.01,
            "p95_us": 1.6,
            "host_median_us": 0.4,
        }
    ],
}


def _jsonl(*rows: dict) -> str:
    return "".join(json.dumps(row) + "\n" for row in rows)


_RECORD_JSONL = _jsonl(RECORD_HEADER, RECORD_OBS)
_TUNE_JSONL = _jsonl(TUNING_HEADER, TUNING_RESULT_NATIVE)

# GPT audit fix (2026-08-18): the compiled binary never writes identity
# fields into these headers -- so a complete triple here is attacker-
# controlled (the only thing that ever produces them is a hostile file).
# Both lifecycle gates must hold for these to stay legacy-imported:
# _identity_from_provenance() rejects the imported-legacy bundle, AND
# build_database()/load_measurements() treat identity=None as
# authoritative (binary authority, no header fallback).
_HOSTILE_RECORD_HEADER = dict(
    RECORD_HEADER,
    source_slice_id="slice-hostile",
    build_plan_id="plan-hostile",
    effective_build_id="eb-hostile",
    campaign_run_id="run-hostile",
)
_HOSTILE_RECORD_JSONL = _jsonl(_HOSTILE_RECORD_HEADER, RECORD_OBS)
_HOSTILE_TUNE_HEADER = dict(
    TUNING_HEADER,
    source_slice_id="slice-hostile",
    build_plan_id="plan-hostile",
    effective_build_id="eb-hostile",
    campaign_run_id="run-hostile",
)
_HOSTILE_TUNE_JSONL = _jsonl(_HOSTILE_TUNE_HEADER, TUNING_RESULT_NATIVE)


class _FakeCompletedProcess:
    def __init__(self, returncode: int = 0, stderr: str = ""):
        self.returncode = returncode
        self.stderr = stderr


def _fake_run(
    *,
    record_jsonl: str | None = None,
    tune_jsonl: str | None = None,
    coverage: bool = False,
):
    """A patched bigcherry.lifecycle.subprocess.run -- the real runtime
    binary invocation is replaced (Windows dev boxes cannot exec a
    published .py bundle member directly; the real binary only runs on
    Brutus), but every downstream consumer of what it writes (inventory.py,
    tune_promotion.py, replay_cache.py, ab_benchmark.py) runs unmodified
    against whatever this writes to the SAME env-var paths lifecycle.py
    itself sets."""

    def run(argv, *, capture_output, text, env):
        if coverage:
            Path(env["GGML_HIP_DISPATCH_COVERAGE"]).write_text(
                json.dumps(
                    {
                        "total_dispatched": 1,
                        "total_executed": 1,
                        "replay": {
                            "schema_version": 2,
                            "exact": 1,
                            "candidate_unavailable": 0,
                            "rerun_required": 0,
                            "incompatible": 0,
                            "misses": 0,
                        },
                    }
                ),
                encoding="utf-8",
            )
            return _FakeCompletedProcess()
        db_path = env["GGML_HIP_DISPATCH_DB"]
        if record_jsonl is not None:
            # RE15 real-hardware finding: GGML_HIP_DISPATCH_MODE defaults to
            # "native" when unset -- the record build silently records
            # nothing without this.
            assert env.get("GGML_HIP_DISPATCH_MODE") == "record", env.get("GGML_HIP_DISPATCH_MODE")
            Path(db_path).write_text(record_jsonl, encoding="utf-8")
        if tune_jsonl is not None:
            assert env.get("GGML_HIP_DISPATCH_MODE") == "tune", env.get("GGML_HIP_DISPATCH_MODE")
            # RE15 real-hardware finding: without this, HIP graph replay
            # crashed (illegal memory access) at small ubatch sizes during
            # tune-mode candidate rotation.
            assert env.get("GGML_CUDA_DISABLE_GRAPHS") == "1", env.get("GGML_CUDA_DISABLE_GRAPHS")
            Path(db_path + ".measurements.jsonl").write_text(
                tune_jsonl, encoding="utf-8"
            )
        return _FakeCompletedProcess()

    return Mock(side_effect=run)


class _Fixture:
    """A real ArtifactStore plus a runtime-bundle publication, following
    the same shape test_re07_smoke_bundle_consumption.py's _Fixture uses.
    The actual binary launch is patched out per-call (see _fake_run) --
    everything else (store verification, provenance derivation,
    inventory.py/tune_promotion.py/replay_cache.py/ab_benchmark.py) is
    real."""

    def __init__(self, directory: Path):
        self.directory = directory
        self.context = ProjectContext(
            project_root=directory,
            config_path=directory / "recipes.toml",
            artifacts_root=directory / "artifacts",
            work_root=directory / "work",
            upstream_repo=directory / "upstream",
            overlay_root=directory / "src",
            patches_root=directory / "patches",
        )
        self.store = ArtifactStore(directory / "store")
        self.run_id = "run1"
        self.seed_doc = provenance.ProvenanceV2.from_document(
            provenance.make(
                project={"provenance_class": "development", "bigcherry_revision": "r1"},
                source={"source_slice_id": "s1"},
                build={"build_plan_id": "bp1", "effective_build_id": "eb1"},
                workload={},
                campaign={"run_id": "seed"},
            )
        )

        script_path = directory / "entrypoint.py"
        script_path.write_text("#!/usr/bin/env python3\n", encoding="utf-8")

        prefix = "builds/s1/bp1"
        script_relative = f"{prefix}/entrypoint.py"
        script_digest = self.store.publish_file(script_relative, script_path)
        manifest = {
            "entrypoint": "entrypoint.py",
            "members": {"entrypoint.py": script_digest},
        }
        bundle_relative = f"{prefix}/runtime-bundle.json"
        self.bundle_ref = self.store.publish_json_ref(
            bundle_relative, manifest, kind="runtime-bundle", provenance=self.seed_doc
        )

        model_path = directory / "model.gguf"
        model_path.write_bytes(b"fake-model-bytes")
        self.spec = RuntimeSmokeSpec(model_path=model_path)


class RecordStageTests(unittest.TestCase):
    def test_record_stage_publishes_a_valid_record_jsonl(self):
        with tempfile.TemporaryDirectory() as directory:
            fx = _Fixture(Path(directory))
            with patch(
                "bigcherry.lifecycle.subprocess.run",
                _fake_run(record_jsonl=_RECORD_JSONL),
            ):
                result = lifecycle.execute_record_stage(
                    context=fx.context,
                    store=fx.store,
                    run_id=fx.run_id,
                    runtime_bundle=ArtifactLocator(fx.bundle_ref.artifact_id),
                    spec=fx.spec,
                    local_provenance_class="development",
                )
            self.assertEqual(result.record_ref.kind, "record-jsonl")
            self.assertTrue(result.record_ref.artifact_id)
            rehydrated = fx.store.rehydrate(
                result.record_ref.artifact_id, expected_kind="record-jsonl"
            )
            self.assertEqual(rehydrated.content_hash, result.record_ref.content_hash)

    def test_record_stage_fails_closed_on_a_tampered_bundle_member(self):
        with tempfile.TemporaryDirectory() as directory:
            fx = _Fixture(Path(directory))
            (fx.store.root / "builds/s1/bp1/entrypoint.py").write_text(
                "TAMPERED", encoding="utf-8"
            )
            with patch(
                "bigcherry.lifecycle.subprocess.run",
                _fake_run(record_jsonl=_RECORD_JSONL),
            ) as fake_run:
                with self.assertRaises(lifecycle.LifecycleError):
                    lifecycle.execute_record_stage(
                        context=fx.context,
                        store=fx.store,
                        run_id=fx.run_id,
                        runtime_bundle=ArtifactLocator(fx.bundle_ref.artifact_id),
                        spec=fx.spec,
                        local_provenance_class="development",
                    )
                fake_run.assert_not_called()

    def test_record_stage_rejects_a_bare_ref_with_no_artifact_id(self):
        with tempfile.TemporaryDirectory() as directory:
            fx = _Fixture(Path(directory))
            bare = dataclasses.replace(fx.bundle_ref, artifact_id="")
            with self.assertRaises(lifecycle.LifecycleError):
                lifecycle.execute_record_stage(
                    context=fx.context,
                    store=fx.store,
                    run_id=fx.run_id,
                    runtime_bundle=bare,
                    spec=fx.spec,
                    local_provenance_class="development",
                )


def _record(fx: _Fixture):
    with patch(
        "bigcherry.lifecycle.subprocess.run", _fake_run(record_jsonl=_RECORD_JSONL)
    ):
        return lifecycle.execute_record_stage(
            context=fx.context,
            store=fx.store,
            run_id=fx.run_id,
            runtime_bundle=ArtifactLocator(fx.bundle_ref.artifact_id),
            spec=fx.spec,
            local_provenance_class="development",
        ).record_ref


class InventoryStageTests(unittest.TestCase):
    def test_inventory_stage_publishes_inventory_and_database(self):
        with tempfile.TemporaryDirectory() as directory:
            fx = _Fixture(Path(directory))
            record_ref = _record(fx)
            result = lifecycle.execute_inventory_stage(
                context=fx.context,
                store=fx.store,
                run_id=fx.run_id,
                record=ArtifactLocator(record_ref.artifact_id),
                local_provenance_class="development",
            )
            self.assertEqual(result.inventory_ref.kind, "inventory")
            self.assertEqual(result.database_ref.kind, "dispatch-db")
            self.assertTrue(result.workload_id)
            rehydrated_db = fx.store.rehydrate(
                result.database_ref.artifact_id, expected_kind="dispatch-db"
            )
            self.assertEqual(
                rehydrated_db.content_hash, result.database_ref.content_hash
            )

    def test_inventory_stage_derives_campaign_identity_from_a_complete_parent_triple(
        self,
    ):
        # The fixture's seed doc carries a complete source/build identity
        # triple -- _identity_from_provenance should derive a real
        # CampaignDatabaseIdentity from it, landing the DB row as
        # identity_scope='campaign'.
        with tempfile.TemporaryDirectory() as directory:
            fx = _Fixture(Path(directory))
            record_ref = _record(fx)
            result = lifecycle.execute_inventory_stage(
                context=fx.context,
                store=fx.store,
                run_id=fx.run_id,
                record=ArtifactLocator(record_ref.artifact_id),
                local_provenance_class="development",
            )
            conn = sqlite3.connect(str(result.database_ref.path))
            try:
                rows = conn.execute("SELECT identity_scope FROM build").fetchall()
            finally:
                conn.close()
            self.assertTrue(rows)
            for (scope,) in rows:
                self.assertEqual(scope, "campaign")

    def test_inventory_stage_falls_back_to_legacy_imported_without_a_complete_parent_triple(
        self,
    ):
        with tempfile.TemporaryDirectory() as directory:
            fx = _Fixture(Path(directory))
            fx.seed_doc = provenance.ProvenanceV2.from_document(
                provenance.make(
                    project={
                        "provenance_class": "development",
                        "bigcherry_revision": "r1",
                    },
                    source={"source_slice_id": "s1"},
                    build={},
                    workload={},
                    campaign={"run_id": "seed"},
                )
            )
            manifest = {
                "entrypoint": "entrypoint.py",
                "members": {
                    "entrypoint.py": fx.store.digest(
                        (fx.directory / "entrypoint.py").read_bytes()
                    )
                },
            }
            fx.bundle_ref = fx.store.publish_json_ref(
                "builds/s1/bp1/runtime-bundle-2.json",
                manifest,
                kind="runtime-bundle",
                provenance=fx.seed_doc,
            )
            record_ref = _record(fx)
            result = lifecycle.execute_inventory_stage(
                context=fx.context,
                store=fx.store,
                run_id=fx.run_id,
                record=ArtifactLocator(record_ref.artifact_id),
                local_provenance_class="development",
            )
            conn = sqlite3.connect(str(result.database_ref.path))
            try:
                rows = conn.execute("SELECT identity_scope FROM build").fetchall()
            finally:
                conn.close()
            self.assertTrue(rows)
            for (scope,) in rows:
                self.assertEqual(scope, "legacy-imported")

    def test_imported_legacy_parent_with_complete_triple_is_not_campaign_evidence(self):
        # RE09 audit fix: RE25.3's downgraded-Ref branch keeps an unverified
        # doc's source/build fields, so an imported-legacy runtime bundle CAN
        # carry a complete triple. Those fields must NOT establish a
        # 'campaign'-scoped DB row -- the class means "origin we cannot
        # verify", and RE09's own rule ("a partial identity is not campaign
        # evidence") extends to unverified ones.
        #
        # NOTE: the record stage runs at local_provenance_class="production"
        # deliberately -- that is what a real campaign lane does, and it is
        # what lets the imported-legacy taint SURVIVE re-derivation (a
        # non-production local class would mask it: derived_provenance_class
        # returns the local class when it is not production). The taint must
        # reach the inventory stage's identity derivation for the gate to
        # have anything to catch.
        with tempfile.TemporaryDirectory() as directory:
            fx = _Fixture(Path(directory))
            fx.seed_doc = provenance.ProvenanceV2.from_document(
                provenance.make(
                    project={
                        "provenance_class": "imported-legacy",
                        "bigcherry_revision": "r1",
                    },
                    source={"source_slice_id": "s1"},
                    build={"build_plan_id": "bp1", "effective_build_id": "eb1"},
                    workload={},
                    campaign={"run_id": "seed"},
                )
            )
            manifest = {
                "entrypoint": "entrypoint.py",
                "members": {
                    "entrypoint.py": fx.store.digest(
                        (fx.directory / "entrypoint.py").read_bytes()
                    )
                },
            }
            fx.bundle_ref = fx.store.publish_json_ref(
                "builds/s1/bp1/runtime-bundle-3.json",
                manifest,
                kind="runtime-bundle",
                provenance=fx.seed_doc,
            )
            with patch(
                "bigcherry.lifecycle.subprocess.run",
                _fake_run(record_jsonl=_RECORD_JSONL),
            ):
                record_ref = lifecycle.execute_record_stage(
                    context=fx.context,
                    store=fx.store,
                    run_id=fx.run_id,
                    runtime_bundle=ArtifactLocator(fx.bundle_ref.artifact_id),
                    spec=fx.spec,
                    local_provenance_class="production",
                ).record_ref
            # Sanity: the taint actually reached the record artifact.
            self.assertEqual(
                provenance.ProvenanceV2.from_document(
                    record_ref.provenance
                ).project.provenance_class,
                "imported-legacy",
            )
            result = lifecycle.execute_inventory_stage(
                context=fx.context,
                store=fx.store,
                run_id=fx.run_id,
                record=ArtifactLocator(record_ref.artifact_id),
                local_provenance_class="development",
            )
            conn = sqlite3.connect(str(result.database_ref.path))
            try:
                rows = conn.execute("SELECT identity_scope FROM build").fetchall()
            finally:
                conn.close()
            self.assertTrue(rows)
            for (scope,) in rows:
                self.assertEqual(scope, "legacy-imported")


    def test_hostile_record_header_stays_legacy_through_record_to_inventory(self):
        # GPT audit fix (2026-08-18): end-to-end regression. The record
        # JSONL header carries a complete (attacker-planted) campaign
        # triple, the bundle is imported-legacy, and the record stage runs
        # at the real lane's local_class='production' (so the imported
        # taint reaches inventory). The DB row must land legacy-imported
        # with visibly-NULL identity columns -- neither the bundle class
        # gate nor the writer's binary authority may fail.
        with tempfile.TemporaryDirectory() as directory:
            fx = _Fixture(Path(directory))
            fx.seed_doc = provenance.ProvenanceV2.from_document(provenance.make(
                project={"provenance_class": "imported-legacy", "bigcherry_revision": "r1"},
                source={"source_slice_id": "s1"},
                build={"build_plan_id": "bp1", "effective_build_id": "eb1"},
                workload={}, campaign={"run_id": "seed"},
            ))
            manifest = {"entrypoint": "entrypoint.py",
                        "members": {"entrypoint.py": fx.store.digest((fx.directory / "entrypoint.py").read_bytes())}}
            fx.bundle_ref = fx.store.publish_json_ref(
                "builds/s1/bp1/runtime-bundle-4.json", manifest,
                kind="runtime-bundle", provenance=fx.seed_doc)
            with patch("bigcherry.lifecycle.subprocess.run",
                       _fake_run(record_jsonl=_HOSTILE_RECORD_JSONL)):
                record_ref = lifecycle.execute_record_stage(
                    context=fx.context, store=fx.store, run_id=fx.run_id,
                    runtime_bundle=ArtifactLocator(fx.bundle_ref.artifact_id), spec=fx.spec,
                    local_provenance_class="production",
                ).record_ref
            result = lifecycle.execute_inventory_stage(
                context=fx.context, store=fx.store, run_id=fx.run_id,
                record=ArtifactLocator(record_ref.artifact_id),
                local_provenance_class="development",
            )
            conn = sqlite3.connect(str(result.database_ref.path))
            try:
                rows = conn.execute(
                    "SELECT source_slice_id, build_plan_id, effective_build_id, "
                    "campaign_run_id, identity_scope FROM build"
                ).fetchall()
            finally:
                conn.close()
            self.assertTrue(rows)
            for row in rows:
                self.assertEqual(row, (None, None, None, None, "legacy-imported"))


class TuneStageTests(unittest.TestCase):
    def _inventory(self, fx: _Fixture):
        record_ref = _record(fx)
        return lifecycle.execute_inventory_stage(
            context=fx.context,
            store=fx.store,
            run_id=fx.run_id,
            record=ArtifactLocator(record_ref.artifact_id),
            local_provenance_class="development",
        )

    def test_tune_stage_publishes_measurements_and_updated_database(self):
        with tempfile.TemporaryDirectory() as directory:
            fx = _Fixture(Path(directory))
            inv = self._inventory(fx)
            with patch(
                "bigcherry.lifecycle.subprocess.run", _fake_run(tune_jsonl=_TUNE_JSONL)
            ):
                result = lifecycle.execute_tune_stage(
                    context=fx.context,
                    store=fx.store,
                    run_id=fx.run_id,
                    runtime_bundle=ArtifactLocator(fx.bundle_ref.artifact_id),
                    dispatch_db=ArtifactLocator(inv.database_ref.artifact_id),
                    spec=fx.spec,
                    local_provenance_class="development",
                )
            self.assertEqual(result.measurements_ref.kind, "tuning-measurements")
            self.assertEqual(result.database_ref.kind, "dispatch-db")
            # RE09 4.5: the ORIGINAL published inventory DB bytes must be
            # untouched -- tune works on its own run-scoped copy.
            original_bytes = inv.database_ref.path.read_bytes()
            self.assertTrue(
                fx.store.verify(
                    inv.database_ref.path.resolve().relative_to(fx.store.root),
                    fx.store.digest(original_bytes),
                )
            )

    def test_hostile_tuning_header_stays_legacy_through_tune_to_db(self):
        # GPT audit fix (2026-08-18): the tune-side analogue. The tuning
        # JSONL header carries a complete attacker-planted triple and the
        # bundle is imported-legacy; the updated DB's build row must stay
        # legacy-imported with NULL identity columns even though the tune
        # stage runs at local_class='production' (the real lane's class).
        with tempfile.TemporaryDirectory() as directory:
            fx = _Fixture(Path(directory))
            fx.seed_doc = provenance.ProvenanceV2.from_document(provenance.make(
                project={"provenance_class": "imported-legacy", "bigcherry_revision": "r1"},
                source={"source_slice_id": "s1"},
                build={"build_plan_id": "bp1", "effective_build_id": "eb1"},
                workload={}, campaign={"run_id": "seed"},
            ))
            manifest = {"entrypoint": "entrypoint.py",
                        "members": {"entrypoint.py": fx.store.digest((fx.directory / "entrypoint.py").read_bytes())}}
            fx.bundle_ref = fx.store.publish_json_ref(
                "builds/s1/bp1/runtime-bundle-5.json", manifest,
                kind="runtime-bundle", provenance=fx.seed_doc)
            inv = self._inventory(fx)
            with patch("bigcherry.lifecycle.subprocess.run",
                       _fake_run(tune_jsonl=_HOSTILE_TUNE_JSONL)):
                result = lifecycle.execute_tune_stage(
                    context=fx.context, store=fx.store, run_id=fx.run_id,
                    runtime_bundle=ArtifactLocator(fx.bundle_ref.artifact_id),
                    dispatch_db=ArtifactLocator(inv.database_ref.artifact_id),
                    spec=fx.spec,
                    local_provenance_class="production",
                )
            conn = sqlite3.connect(str(result.database_ref.path))
            try:
                rows = conn.execute(
                    "SELECT source_slice_id, build_plan_id, effective_build_id, "
                    "campaign_run_id, identity_scope FROM build"
                ).fetchall()
            finally:
                conn.close()
            self.assertTrue(rows)
            hostile = ("slice-hostile", "plan-hostile", "eb-hostile")
            for row in rows:
                # The attacker-planted header triple must NOT appear on ANY
                # row -- that is the invariant under test.
                self.assertNotEqual(row[:3], hostile)
                # The pre-existing inventory row is campaign-scoped from the
                # dev-class bundle's LEGITIMATE complete triple (RE09's
                # completeness rule); no other triple may be campaign.
                if row[4] == "campaign":
                    self.assertEqual(row[:3], ("s1", "bp1", "eb1"))
            # The tune load itself landed as a NULL-identity legacy row.
            self.assertIn((None, None, None, None, "legacy-imported"), rows)

    def test_tune_stage_fails_before_running_when_the_db_artifact_was_tampered(self):
        with tempfile.TemporaryDirectory() as directory:
            fx = _Fixture(Path(directory))
            inv = self._inventory(fx)
            inv.database_ref.path.write_bytes(b"TAMPERED")
            with patch(
                "bigcherry.lifecycle.subprocess.run", _fake_run(tune_jsonl=_TUNE_JSONL)
            ) as fake_run:
                with self.assertRaises(lifecycle.LifecycleError):
                    lifecycle.execute_tune_stage(
                        context=fx.context,
                        store=fx.store,
                        run_id=fx.run_id,
                        runtime_bundle=ArtifactLocator(fx.bundle_ref.artifact_id),
                        dispatch_db=ArtifactLocator(inv.database_ref.artifact_id),
                        spec=fx.spec,
                        local_provenance_class="development",
                    )
                fake_run.assert_not_called()


class PromotionAndReplayExportTests(unittest.TestCase):
    def test_promotion_stage_publishes_promoted_winners(self):
        with tempfile.TemporaryDirectory() as directory:
            fx = _Fixture(Path(directory))
            measurements_bytes = _jsonl(
                {
                    "kind": "header",
                    "artifact_version": 1,
                    "source_revision": "b" * 40,
                    "manifest_hash": "a" * 32,
                },
                {
                    "kind": "result",
                    "dispatch": "a" * 32,
                    "native": "native",
                    "signature": "a" * 32,
                    "winner": "candidate",
                    "promotion_status": "pending_bh",
                    "provisional_winner": "candidate",
                    "schedule_seed": int.from_bytes(
                        bytes.fromhex("a" * 32)[:4], "little"
                    ),
                    "schedule": {
                        "schema_version": 1,
                        "selection_algorithm": "seeded-rotation-v1",
                        "confirmation_algorithm": "seeded-alternation-v1",
                        "candidates": ["candidate", "native", "native#twin"],
                    },
                    "improvement_pct": 5.0,
                    "confirmation": {
                        "p_value": 0.001,
                        "effect_pct": 5.0,
                        "wins": 12,
                        "rounds": 12,
                        "native_us": [100.0] * 12,
                        "winner_us": [95.0] * 12,
                    },
                },
            ).encode("utf-8")
            measurements_ref = fx.store.publish_bytes_ref(
                "runs/run1/tune/measurements.jsonl",
                measurements_bytes,
                kind="tuning-measurements",
                provenance=fx.seed_doc,
            )
            result = lifecycle.execute_promotion_stage(
                context=fx.context,
                store=fx.store,
                run_id=fx.run_id,
                measurements=ArtifactLocator(measurements_ref.artifact_id),
                resamples=1000,
                local_provenance_class="development",
            )
            self.assertEqual(result.promoted_winners_ref.kind, "promoted-winners")
            rows = [
                json.loads(line)
                for line in result.promoted_winners_ref.path.read_text().splitlines()
            ]
            self.assertEqual(rows[1]["promotion_status"], "promoted")

    def test_replay_export_stage_publishes_a_replay_cache(self):
        with tempfile.TemporaryDirectory() as directory:
            fx = _Fixture(Path(directory))
            winners_bytes = _jsonl(
                {
                    "kind": "header",
                    "source_revision": "b" * 40,
                    "manifest_hash": "a" * 32,
                },
                {
                    "kind": "result",
                    "dispatch": "A" * 32,
                    "signature": "C" * 32,
                    "winner": "mmvq:native:v1",
                    "native": "mmvq:native:v1",
                },
            ).encode("utf-8")
            winners_ref = fx.store.publish_bytes_ref(
                "runs/run1/promote/promoted-winners.jsonl",
                winners_bytes,
                kind="promoted-winners",
                provenance=fx.seed_doc,
            )
            manifest_doc = {
                "source_revision": "b" * 40,
                "manifest_hash": "a" * 32,
                "candidates": [
                    {
                        "stable_name": "mmvq:native:v1",
                        "family": "mmvq",
                        "source_class": "native_wrapper",
                        "implementation_version": 1,
                        "config": {},
                    }
                ],
            }
            manifest_ref = fx.store.publish_json_ref(
                "runs/run1/manifest.json",
                manifest_doc,
                kind="manifest",
                provenance=fx.seed_doc,
            )
            source_root = fx.directory / "source"
            (source_root / "ggml" / "include").mkdir(parents=True)
            (source_root / "ggml" / "include" / "ggml.h").write_text(
                "GGML_TYPE_F32 = 0,\n", encoding="utf-8"
            )

            result = lifecycle.execute_replay_export_stage(
                context=fx.context,
                store=fx.store,
                run_id=fx.run_id,
                promoted_winners=ArtifactLocator(winners_ref.artifact_id),
                manifest=ArtifactLocator(manifest_ref.artifact_id),
                source_root=source_root,
                local_provenance_class="development",
            )
            self.assertEqual(result.replay_cache_ref.kind, "replay-cache")
            self.assertTrue(result.replay_cache_ref.path.read_bytes())


class ReplayValidationStageTests(unittest.TestCase):
    def test_replay_validation_stage_publishes_full_coverage(self):
        with tempfile.TemporaryDirectory() as directory:
            fx = _Fixture(Path(directory))
            cache_ref = fx.store.publish_bytes_ref(
                "runs/run1/replay/replay.cache",
                b"fake-cache-bytes",
                kind="replay-cache",
                provenance=fx.seed_doc,
            )
            with patch("bigcherry.lifecycle.subprocess.run", _fake_run(coverage=True)):
                result = lifecycle.execute_replay_validation_stage(
                    context=fx.context,
                    store=fx.store,
                    run_id=fx.run_id,
                    runtime_bundle=ArtifactLocator(fx.bundle_ref.artifact_id),
                    replay_cache_artifact=ArtifactLocator(cache_ref.artifact_id),
                    spec=fx.spec,
                    local_provenance_class="development",
                )
            self.assertEqual(result.coverage_ref.kind, "replay-coverage")
            self.assertEqual(result.coverage["total_dispatched"], 1)


class RehydrationAcrossFreshStoreTests(unittest.TestCase):
    def test_every_stage_output_rehydrates_in_a_fresh_artifact_store_instance(self):
        with tempfile.TemporaryDirectory() as directory:
            fx = _Fixture(Path(directory))
            record_ref = _record(fx)
            fresh_store = ArtifactStore(fx.directory / "store")
            rehydrated = fresh_store.rehydrate(
                record_ref.artifact_id, expected_kind="record-jsonl"
            )
            self.assertEqual(rehydrated.content_hash, record_ref.content_hash)


if __name__ == "__main__":
    unittest.main()
