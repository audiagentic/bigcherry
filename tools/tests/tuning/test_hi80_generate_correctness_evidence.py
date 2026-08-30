"""HI80: tests for hi80_generate_correctness_evidence.py's CLI-shaped
orchestration -- finding promotable-but-non-native rows in a measurements
file, resolving each against a real schema-6 dispatch_db, mapping its
signature via signature_correctness_mapping, and driving correctness_
evidence's generation/write path with a fake test-backend-ops runner (no
real binary or HIP hardware needed to exercise the orchestration logic
itself)."""

from __future__ import annotations

import json
import sqlite3
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from bigcherry.tuning import correctness_evidence as ce # noqa: E402
from bigcherry import hi80_generate_correctness_evidence as cli  # noqa: E402
from bigcherry.tuning import promotion_gate as gate # noqa: E402
from bigcherry.tuning import signature_mapping as scm # noqa: E402

SCHEMA_SQL = Path(__file__).resolve().parents[3] / "sql" / "dispatch-db.sql"

DISPATCH_HEX = "11" * 16
SIGNATURE_HEX = "22" * 16
HARDWARE_HEX = "33" * 16

CANONICAL_SIGNATURE = {
    "op": 2,  # MUL_MAT in the fixture vendor tree below
    "src0_type": 0,  # f32
    "src1_type": 0,  # f32
    "dst_type": 0,  # f32
    "ne0": [256, 16, 1, 1],
    "ne1": [256, 1, 1, 1],
    "ned": [16, 1, 1, 1],
}


def _write_fixture_vendor(tmp_path: Path) -> Path:
    vendor = tmp_path / "vendor" / "llama.cpp"
    (vendor / "ggml" / "include").mkdir(parents=True, exist_ok=True)
    (vendor / "ggml" / "src").mkdir(parents=True, exist_ok=True)
    (vendor / "ggml" / "include" / "ggml.h").write_text(
        "enum ggml_type {\n    GGML_TYPE_F32  = 0,\n};\n"
        "enum ggml_op {\n    GGML_OP_NONE,\n    GGML_OP_ADD,\n    GGML_OP_MUL_MAT,\n"
        "    GGML_OP_COUNT,\n};\n",
        encoding="utf-8",
    )
    (vendor / "ggml" / "src" / "ggml.c").write_text(
        "static const struct ggml_type_traits type_traits[GGML_TYPE_COUNT] = {\n"
        '    [GGML_TYPE_F32] = {\n        .type_name = "f32",\n    },\n'
        "};\n"
        'static const char * GGML_OP_NAME[GGML_OP_COUNT] = {\n'
        '    "NONE",\n    "ADD",\n    "MUL_MAT",\n'
        "};\n",
        encoding="utf-8",
    )
    return vendor


def _fake_runner_factory(*, digest="cafebabe", threshold=5e-4, e_c=1e-5, max_abs_c=0.0004):
    def runner(argv, capture_output, text, env):
        mode = env.get("GGML_HIP_DISPATCH_MODE")
        tensor = "out"
        stderr = (
            # HI80 (2026-08-23): the --test-file/test_generic_op path's
            # destination tensor never gets pre-filled, so its digest comes
            # from a real leaf tensor ("leaf_0") instead of "out" -- see
            # correctness_evidence.collect_seed_evidence's digest_tensor.
            f"BIGCHERRY_REF_DIGEST name=leaf_0 call_index=0 digest={digest} nels=16\n"
            f"BIGCHERRY_CORRECTNESS_METRIC op=MUL_MAT tensor={tensor} "
            f"backend1=native backend2={'native' if mode == 'native' else 'candidate'} "
            f"err={e_c if mode != 'native' else 1e-6} max_abs={max_abs_c if mode != 'native' else 0.0005} "
            f"threshold={threshold} n=16\n"
        )
        return subprocess.CompletedProcess(argv, 0, stdout="", stderr=stderr)

    return runner


class _Base(unittest.TestCase):
    def setUp(self):
        self.conn = sqlite3.connect(":memory:")
        self.conn.execute("PRAGMA foreign_keys = ON")
        self.conn.executescript(SCHEMA_SQL.read_text(encoding="utf-8"))
        self.conn.execute(
            "INSERT INTO build (source_revision, manifest_hash, signature_schema, "
            "hardware_schema, variant_set) VALUES ('deadbeefdeadbeefdead', 'aa', 1, 1, 'inventory')"
        )
        self.build_id = self.conn.execute("SELECT last_insert_rowid()").fetchone()[0]
        self.conn.execute(
            "INSERT INTO hardware (hardware_digest, architecture, architecture_code, "
            "wave_size, compute_units, feature_flags, canonical_json) VALUES "
            "(?, 'gfx1100', 1, 32, 96, 0, '{}')",
            (bytes.fromhex(HARDWARE_HEX),),
        )
        self.hardware_id = self.conn.execute("SELECT last_insert_rowid()").fetchone()[0]
        self.conn.execute(
            "INSERT INTO signature (signature_digest, base_digest, schema_version, op, "
            "src0_type, src1_type, dst_type, m, n, k, canonical_json) VALUES "
            "(?, x'02', 1, 'MUL_MAT', 'f32', 'f32', 'f32', 16, 1, 256, ?)",
            (bytes.fromhex(SIGNATURE_HEX), json.dumps(CANONICAL_SIGNATURE)),
        )
        self.signature_id = self.conn.execute("SELECT last_insert_rowid()").fetchone()[0]
        self.conn.execute(
            "INSERT INTO candidate (build_id, stable_name, family, source_class, "
            "implementation_version, architectures, architecture_mask, graph_safe, "
            "deterministic, config_json) VALUES (?, 'native', 'mmq', 'native_wrapper', "
            "1, '[]', 0, 1, 1, '{}')",
            (self.build_id,),
        )
        self.native_candidate_id = self.conn.execute("SELECT last_insert_rowid()").fetchone()[0]
        self.conn.execute(
            "INSERT INTO candidate (build_id, stable_name, family, source_class, "
            "implementation_version, architectures, architecture_mask, graph_safe, "
            "deterministic, config_json) VALUES (?, 'mmq:fb1', 'mmq', 'existing_alternative', "
            "1, '[]', 0, 1, 1, '{}')",
            (self.build_id,),
        )
        self.candidate_id = self.conn.execute("SELECT last_insert_rowid()").fetchone()[0]
        self.conn.execute(
            "INSERT INTO measurement (build_id, hardware_id, signature_id, dispatch_digest, "
            "candidate_id, objective, stage, accepted) VALUES (?, ?, ?, ?, ?, 'latency', "
            "'final', 1)",
            (self.build_id, self.hardware_id, self.signature_id,
             bytes.fromhex(DISPATCH_HEX), self.candidate_id),
        )
        self.conn.commit()

        self._tmp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp_dir.cleanup)
        self.vendor = _write_fixture_vendor(Path(self._tmp_dir.name))

        self.row = {
            "dispatch": DISPATCH_HEX,
            "signature": SIGNATURE_HEX,
            "hardware": HARDWARE_HEX,
            "native": "native",
            "provisional_winner": "mmq:fb1",
        }


class FindCandidateRowsTests(unittest.TestCase):
    def test_finds_non_native_provisional_winner(self):
        rows = [
            {"provisional_winner": "mmq:fb1", "native": "native"},
            {"provisional_winner": "native", "native": "native"},
            {"native": "native"},
        ]
        found = cli.find_candidate_rows(rows)
        self.assertEqual(len(found), 1)
        self.assertEqual(found[0]["provisional_winner"], "mmq:fb1")


class GenerateForRowTests(_Base):
    def test_writes_evidence_for_a_fresh_row(self):
        outcome = cli.generate_for_row(
            self.conn, self.row, binary=Path("test-backend-ops"), vendor_root=self.vendor,
            seeds=(1, 2, 3), headroom_fraction=ce.DEFAULT_HEADROOM_FRACTION,
            contract_version=ce.CONTRACT_VERSION, tool_version="test",
            runner=_fake_runner_factory(),
        )
        self.assertIn("wrote evidence_id=", outcome)
        stored = self.conn.execute(
            "SELECT candidate_id, native_candidate_id FROM correctness_evidence"
        ).fetchone()
        self.assertEqual(stored, (self.candidate_id, self.native_candidate_id))

    def test_skips_a_row_with_existing_evidence(self):
        runner = _fake_runner_factory()
        cli.generate_for_row(
            self.conn, self.row, binary=Path("test-backend-ops"), vendor_root=self.vendor,
            seeds=(1, 2, 3), headroom_fraction=ce.DEFAULT_HEADROOM_FRACTION,
            contract_version=ce.CONTRACT_VERSION, tool_version="test", runner=runner,
        )
        outcome = cli.generate_for_row(
            self.conn, self.row, binary=Path("test-backend-ops"), vendor_root=self.vendor,
            seeds=(1, 2, 3), headroom_fraction=ce.DEFAULT_HEADROOM_FRACTION,
            contract_version=ce.CONTRACT_VERSION, tool_version="test", runner=runner,
        )
        self.assertIn("skip (already has evidence_id=", outcome)
        count = self.conn.execute("SELECT COUNT(*) FROM correctness_evidence").fetchone()[0]
        self.assertEqual(count, 1)

    def test_evidence_written_passes_the_real_promotion_gate(self):
        cli.generate_for_row(
            self.conn, self.row, binary=Path("test-backend-ops"), vendor_root=self.vendor,
            seeds=(1, 2, 3), headroom_fraction=ce.DEFAULT_HEADROOM_FRACTION,
            contract_version=ce.CONTRACT_VERSION, tool_version="test",
            runner=_fake_runner_factory(),
        )
        identity = gate.resolve_promotion_identity(
            self.conn, dispatch_hex=DISPATCH_HEX, signature_hex=SIGNATURE_HEX,
            hardware_hex=HARDWARE_HEX, native_name="native", candidate_name="mmq:fb1",
        )
        passed, status = gate.evaluate_correctness_gate(self.conn, identity)
        self.assertTrue(passed, status)

    def test_malformed_identity_fields_raise_cli_error(self):
        bad_row = dict(self.row)
        del bad_row["hardware"]
        with self.assertRaises(cli.CliError):
            cli.generate_for_row(
                self.conn, bad_row, binary=Path("test-backend-ops"), vendor_root=self.vendor,
                seeds=(1, 2, 3), headroom_fraction=ce.DEFAULT_HEADROOM_FRACTION,
                contract_version=ce.CONTRACT_VERSION, tool_version="test",
                runner=_fake_runner_factory(),
            )

    def test_unknown_dispatch_identity_raises_gate_error(self):
        bad_row = dict(self.row)
        bad_row["dispatch"] = "ff" * 16
        with self.assertRaises(gate.CorrectnessGateError):
            cli.generate_for_row(
                self.conn, bad_row, binary=Path("test-backend-ops"), vendor_root=self.vendor,
                seeds=(1, 2, 3), headroom_fraction=ce.DEFAULT_HEADROOM_FRACTION,
                contract_version=ce.CONTRACT_VERSION, tool_version="test",
                runner=_fake_runner_factory(),
            )

    def test_non_mul_mat_signature_raises_signature_mapping_error(self):
        # op=1 is GGML_OP_ADD in the fixture vendor tree -- signature_to_
        # op_filter is MUL_MAT-only this slice (HI80's own documented scope
        # limit), so this row must be reported/skipped, not silently mapped.
        self.conn.execute(
            "UPDATE signature SET canonical_json = ? WHERE signature_id = ?",
            (json.dumps({**CANONICAL_SIGNATURE, "op": 1}), self.signature_id),
        )
        self.conn.commit()
        with self.assertRaises(scm.SignatureMappingError):
            cli.generate_for_row(
                self.conn, self.row, binary=Path("test-backend-ops"), vendor_root=self.vendor,
                seeds=(1, 2, 3), headroom_fraction=ce.DEFAULT_HEADROOM_FRACTION,
                contract_version=ce.CONTRACT_VERSION, tool_version="test",
                runner=_fake_runner_factory(),
            )


class HI112BatchedSrc1PipelineTests(_Base):
    """HI112 (gpt-dev-agent review, 2026-08-25): the earlier real-hardware
    confirmation invoked test-backend-ops directly, bypassing the actual
    production pipeline this class exercises -- generate_for_row() ->
    dispatch_db -> promotion_correctness_gate. Proves a real HI109-shaped
    batched-src1 MUL_MAT signature (weights non-batched, src1 batched by 3
    MTP draft tokens, matching HI109's real observed batch counts of 2-4)
    drives real evidence through the actual orchestration and passes the
    real gate -- not just that signature_to_test_file_line() builds a
    syntactically valid line in isolation."""

    BATCHED_SIGNATURE_HEX = "77" * 16

    BATCHED_CANONICAL_SIGNATURE = {
        "op": 2,  # GGML_OP_MUL_MAT in the fixture vendor tree
        "src0_type": 0, "src1_type": 0, "dst_type": 0,  # f32
        "ne0": [2880, 32, 1, 1],
        "ne1": [2880, 1, 3, 1],
        "ned": [32, 1, 3, 1],
    }

    def setUp(self):
        super().setUp()
        self.conn.execute(
            "INSERT INTO signature (signature_digest, base_digest, schema_version, op, "
            "src0_type, src1_type, dst_type, m, n, k, canonical_json) VALUES "
            "(?, x'02', 1, 'MUL_MAT', 'f32', 'f32', 'f32', 32, 3, 2880, ?)",
            (bytes.fromhex(self.BATCHED_SIGNATURE_HEX), json.dumps(self.BATCHED_CANONICAL_SIGNATURE)),
        )
        self.batched_signature_id = self.conn.execute("SELECT last_insert_rowid()").fetchone()[0]
        self.conn.execute(
            "INSERT INTO measurement (build_id, hardware_id, signature_id, dispatch_digest, "
            "candidate_id, objective, stage, accepted) VALUES (?, ?, ?, ?, ?, 'latency', "
            "'final', 1)",
            (self.build_id, self.hardware_id, self.batched_signature_id,
             bytes.fromhex("88" * 16), self.candidate_id),
        )
        self.conn.commit()
        self.batched_row = {
            "dispatch": "88" * 16,
            "signature": self.BATCHED_SIGNATURE_HEX,
            "hardware": HARDWARE_HEX,
            "native": "native",
            "provisional_winner": "mmq:fb1",
        }

    def test_batched_src1_row_writes_evidence_and_passes_the_real_gate(self):
        outcome = cli.generate_for_row(
            self.conn, self.batched_row, binary=Path("test-backend-ops"), vendor_root=self.vendor,
            seeds=(1, 2, 3), headroom_fraction=ce.DEFAULT_HEADROOM_FRACTION,
            contract_version=ce.CONTRACT_VERSION, tool_version="test",
            runner=_fake_runner_factory(),
        )
        self.assertIn("wrote evidence_id=", outcome)
        identity = gate.resolve_promotion_identity(
            self.conn, dispatch_hex="88" * 16, signature_hex=self.BATCHED_SIGNATURE_HEX,
            hardware_hex=HARDWARE_HEX, native_name="native", candidate_name="mmq:fb1",
        )
        passed, status = gate.evaluate_correctness_gate(self.conn, identity)
        self.assertTrue(passed, status)


# HI105 (dev-gpt-agent review, 2026-08-24): proves the real bug found at
# HEAD 7f2e04c is fixed -- generate_for_row() previously called
# signature_to_test_file_line() (MUL_MAT-only) unconditionally, so a real
# MUL_MAT_ID row always raised SignatureMappingError and was silently
# SKIPPED by main()'s loop without incrementing `failed`, letting a run
# exit 0 having generated no evidence at all. This test drives
# generate_for_row() with a real MUL_MAT_ID signature (RD54 K=256's own
# shape) through a fake runner and asserts it actually reaches
# ce.generate_correctness_evidence() and writes real evidence -- not that
# it merely fails to raise.

MUL_MAT_ID_SIGNATURE_HEX = "44" * 16

MUL_MAT_ID_CANONICAL_SIGNATURE = {
    "op": 2,  # GGML_OP_MUL_MAT_ID in the extended fixture vendor tree below
    "flags": 15,  # SRC0|SRC1|DST contiguous (1|2|4) | HAS_IDS (8)
    "n_expert": 256, "n_expert_used": 8,
    "src0_type": 0, "src1_type": 0, "dst_type": 0,  # f32 (fixture only defines f32)
    "ne0": [256, 2048, 256, 1], "ne1": [256, 8, 1, 1], "ned": [2048, 8, 1, 1],
}


def _write_fixture_vendor_with_mul_mat_id(tmp_path: Path) -> Path:
    vendor = tmp_path / "vendor" / "llama.cpp"
    (vendor / "ggml" / "include").mkdir(parents=True, exist_ok=True)
    (vendor / "ggml" / "src").mkdir(parents=True, exist_ok=True)
    (vendor / "ggml" / "include" / "ggml.h").write_text(
        "enum ggml_type {\n    GGML_TYPE_F32  = 0,\n    GGML_TYPE_I32  = 26,\n};\n"
        "enum ggml_op {\n    GGML_OP_NONE,\n    GGML_OP_ADD,\n    GGML_OP_MUL_MAT_ID,\n"
        "    GGML_OP_COUNT,\n};\n",
        encoding="utf-8",
    )
    (vendor / "ggml" / "src" / "ggml.c").write_text(
        "static const struct ggml_type_traits type_traits[GGML_TYPE_COUNT] = {\n"
        '    [GGML_TYPE_F32] = {\n        .type_name = "f32",\n    },\n'
        '    [GGML_TYPE_I32] = {\n        .type_name = "i32",\n    },\n'
        "};\n"
        'static const char * GGML_OP_NAME[GGML_OP_COUNT] = {\n'
        '    "NONE",\n    "ADD",\n    "MUL_MAT_ID",\n'
        "};\n",
        encoding="utf-8",
    )
    return vendor


def _fake_mul_mat_id_runner_factory(*, digest="deadc0de", threshold=5e-4, e_c=1e-5, max_abs_c=0.0004):
    def runner(argv, capture_output, text, env):
        mode = env.get("GGML_HIP_DISPATCH_MODE")
        tensor = "out"
        stderr = (
            # HI105: digest_tensor is "leaf_2" (the ids/routing tensor) for
            # MUL_MAT_ID, not "leaf_0" -- see signature_to_mul_mat_id_test_
            # file_line's own docstring on why.
            f"BIGCHERRY_REF_DIGEST name=leaf_2 call_index=0 digest={digest} nels=8\n"
            f"BIGCHERRY_CORRECTNESS_METRIC op=MUL_MAT_ID tensor={tensor} "
            f"backend1=native backend2={'native' if mode == 'native' else 'candidate'} "
            f"err={e_c if mode != 'native' else 1e-6} max_abs={max_abs_c if mode != 'native' else 0.0005} "
            f"threshold={threshold} n=8\n"
        )
        return subprocess.CompletedProcess(argv, 0, stdout="", stderr=stderr)

    return runner


class MulMatIdGenerateForRowTests(unittest.TestCase):
    def setUp(self):
        self.conn = sqlite3.connect(":memory:")
        self.conn.execute("PRAGMA foreign_keys = ON")
        self.conn.executescript(SCHEMA_SQL.read_text(encoding="utf-8"))
        self.conn.execute(
            "INSERT INTO build (source_revision, manifest_hash, signature_schema, "
            "hardware_schema, variant_set) VALUES ('deadbeefdeadbeefdead', 'aa', 1, 1, 'inventory')"
        )
        self.build_id = self.conn.execute("SELECT last_insert_rowid()").fetchone()[0]
        self.conn.execute(
            "INSERT INTO hardware (hardware_digest, architecture, architecture_code, "
            "wave_size, compute_units, feature_flags, canonical_json) VALUES "
            "(?, 'gfx1100', 1, 32, 96, 0, '{}')",
            (bytes.fromhex(HARDWARE_HEX),),
        )
        self.hardware_id = self.conn.execute("SELECT last_insert_rowid()").fetchone()[0]
        self.conn.execute(
            "INSERT INTO signature (signature_digest, base_digest, schema_version, op, "
            "src0_type, src1_type, dst_type, m, n, k, canonical_json) VALUES "
            "(?, x'02', 1, 'MUL_MAT_ID', 'f32', 'f32', 'f32', 2048, 8, 256, ?)",
            (bytes.fromhex(MUL_MAT_ID_SIGNATURE_HEX), json.dumps(MUL_MAT_ID_CANONICAL_SIGNATURE)),
        )
        self.signature_id = self.conn.execute("SELECT last_insert_rowid()").fetchone()[0]
        self.conn.execute(
            "INSERT INTO candidate (build_id, stable_name, family, source_class, "
            "implementation_version, architectures, architecture_mask, graph_safe, "
            "deterministic, config_json) VALUES (?, 'native', 'mmvq', 'native_wrapper', "
            "1, '[]', 0, 1, 1, '{}')",
            (self.build_id,),
        )
        self.native_candidate_id = self.conn.execute("SELECT last_insert_rowid()").fetchone()[0]
        self.conn.execute(
            "INSERT INTO candidate (build_id, stable_name, family, source_class, "
            "implementation_version, architectures, architecture_mask, graph_safe, "
            "deterministic, config_json) VALUES (?, 'mmvq:q8_0:w1:nw1:rpb2:sk0:v1', 'mmvq', "
            "'new_generated_variant', 1, '[]', 0, 1, 1, '{}')",
            (self.build_id,),
        )
        self.candidate_id = self.conn.execute("SELECT last_insert_rowid()").fetchone()[0]
        self.conn.execute(
            "INSERT INTO measurement (build_id, hardware_id, signature_id, dispatch_digest, "
            "candidate_id, objective, stage, accepted) VALUES (?, ?, ?, ?, ?, 'latency', "
            "'final', 1)",
            (self.build_id, self.hardware_id, self.signature_id,
             bytes.fromhex(DISPATCH_HEX), self.candidate_id),
        )
        self.conn.commit()

        self._tmp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp_dir.cleanup)
        self.vendor = _write_fixture_vendor_with_mul_mat_id(Path(self._tmp_dir.name))

        self.row = {
            "dispatch": DISPATCH_HEX,
            "signature": MUL_MAT_ID_SIGNATURE_HEX,
            "hardware": HARDWARE_HEX,
            "native": "native",
            "provisional_winner": "mmvq:q8_0:w1:nw1:rpb2:sk0:v1",
        }

    def test_mul_mat_id_row_reaches_real_evidence_generation(self):
        outcome = cli.generate_for_row(
            self.conn, self.row, binary=Path("test-backend-ops"), vendor_root=self.vendor,
            seeds=(1, 2, 3), headroom_fraction=ce.DEFAULT_HEADROOM_FRACTION,
            contract_version=ce.CONTRACT_VERSION, tool_version="test",
            runner=_fake_mul_mat_id_runner_factory(),
        )
        self.assertIn("wrote evidence_id=", outcome)
        stored = self.conn.execute(
            "SELECT candidate_id, native_candidate_id FROM correctness_evidence"
        ).fetchone()
        self.assertEqual(stored, (self.candidate_id, self.native_candidate_id))

    def test_evidence_written_passes_the_real_promotion_gate(self):
        cli.generate_for_row(
            self.conn, self.row, binary=Path("test-backend-ops"), vendor_root=self.vendor,
            seeds=(1, 2, 3), headroom_fraction=ce.DEFAULT_HEADROOM_FRACTION,
            contract_version=ce.CONTRACT_VERSION, tool_version="test",
            runner=_fake_mul_mat_id_runner_factory(),
        )
        identity = gate.resolve_promotion_identity(
            self.conn, dispatch_hex=DISPATCH_HEX, signature_hex=MUL_MAT_ID_SIGNATURE_HEX,
            hardware_hex=HARDWARE_HEX, native_name="native",
            candidate_name="mmvq:q8_0:w1:nw1:rpb2:sk0:v1",
        )
        passed, status = gate.evaluate_correctness_gate(self.conn, identity)
        self.assertTrue(passed, status)


# HI119: the fused-GLU branch is structurally different from MUL_MAT/
# MUL_MAT_ID -- it routes through --moe-glu-file (patches 1239/1240) and
# additionally requires a real record-mode run to prove the synthetic
# graph reproduced the row's own dispatch signature before any correctness
# comparison is trusted (see _observed_signature_hex's own docstring).
# This class proves that whole path -- branch selection by op name,
# fused_glu/ids tensor naming, and the observed-digest gate's both
# outcomes -- without a real binary or hardware.

# The dispatch_db signature identity and the observed-record-mode hex must
# be the SAME real hex here, since generate_for_row's GLU branch compares
# them directly (_observed_signature_hex vs. the row's own signature_hex) --
# using the real hardware-observed value for both is what the "matches"
# path actually exercises.
GLU_SIGNATURE_HEX = "b9e208b3d066da12565a8912c1117c16"

GLU_CANONICAL_SIGNATURE = {
    "op": 4,  # GGML_OP_GLU in the extended fixture vendor tree below
    "src0_type": 8, "src1_type": 0, "dst_type": 0,  # q8_0 / f32 / f32
    "fusion": 2,  # GGML_HIP_FUSION_GATE
    "glu_op": 2,  # GGML_GLU_OP_SWIGLU
    "flags": 31,  # SRC0|SRC1|DST_CONTIGUOUS | HAS_IDS | BROADCAST_CH, no HI118 bias/scale bits
    "n_expert": 256, "n_expert_used": 8,
    "ne0": [2048, 256, 256, 1], "ne1": [2048, 1, 1, 1], "ned": [256, 8, 1, 1],
    "schema_version": scm.dispatch_abi.SIGNATURE_SCHEMA_VERSION,
}


def _write_fixture_vendor_with_glu(tmp_path: Path) -> Path:
    vendor = tmp_path / "vendor" / "llama.cpp"
    (vendor / "ggml" / "include").mkdir(parents=True, exist_ok=True)
    (vendor / "ggml" / "src").mkdir(parents=True, exist_ok=True)
    (vendor / "ggml" / "include" / "ggml.h").write_text(
        "enum ggml_type {\n    GGML_TYPE_F32  = 0,\n    GGML_TYPE_Q8_0 = 8,\n};\n"
        "enum ggml_op {\n    GGML_OP_NONE,\n    GGML_OP_ADD,\n    GGML_OP_MUL_MAT,\n"
        "    GGML_OP_MUL_MAT_ID,\n    GGML_OP_GLU,\n    GGML_OP_COUNT,\n};\n",
        encoding="utf-8",
    )
    (vendor / "ggml" / "src" / "ggml.c").write_text(
        "static const struct ggml_type_traits type_traits[GGML_TYPE_COUNT] = {\n"
        '    [GGML_TYPE_F32] = {\n        .type_name = "f32",\n    },\n'
        '    [GGML_TYPE_Q8_0] = {\n        .type_name = "q8_0",\n    },\n'
        "};\n"
        'static const char * GGML_OP_NAME[GGML_OP_COUNT] = {\n'
        '    "NONE",\n    "ADD",\n    "MUL_MAT",\n    "MUL_MAT_ID",\n    "GLU",\n'
        "};\n",
        encoding="utf-8",
    )
    return vendor


def _fake_glu_runner_factory(
    *, observed_hex=GLU_SIGNATURE_HEX, digest="f00dcafe",
    threshold=5e-4, e_c=1e-5, max_abs_c=0.0004,
):
    """Handles all three GGML_HIP_DISPATCH_MODE values generate_for_row's
    GLU branch drives: "record" (the observed-signature gate, writes a
    JSONL observation row to the path in env["GGML_HIP_DISPATCH_DB"]) and
    "native"/"replay" (the actual correctness comparison, emitting
    fused_glu/ids-named lines matching signature_to_moe_glu_file_line's
    real target_tensor="fused_glu"/digest_tensor="ids")."""

    def runner(argv, capture_output, text, env):
        mode = env.get("GGML_HIP_DISPATCH_MODE")
        if mode == "record":
            db_path = Path(env["GGML_HIP_DISPATCH_DB"])
            db_path.write_text(
                json.dumps({"kind": "observation", "signature": observed_hex, "canonical": {}}) + "\n",
                encoding="utf-8",
            )
            return subprocess.CompletedProcess(argv, 0, stdout="", stderr="")
        stderr = (
            f"BIGCHERRY_REF_DIGEST name=ids call_index=0 digest={digest} nels=8\n"
            f"BIGCHERRY_CORRECTNESS_METRIC op=GLU tensor=fused_glu "
            f"backend1=native backend2={'native' if mode == 'native' else 'candidate'} "
            f"err={e_c if mode != 'native' else 1e-6} max_abs={max_abs_c if mode != 'native' else 0.0005} "
            f"threshold={threshold} n=8\n"
        )
        return subprocess.CompletedProcess(argv, 0, stdout="", stderr=stderr)

    return runner


class GluGenerateForRowTests(unittest.TestCase):
    def setUp(self):
        self.conn = sqlite3.connect(":memory:")
        self.conn.execute("PRAGMA foreign_keys = ON")
        self.conn.executescript(SCHEMA_SQL.read_text(encoding="utf-8"))
        self.conn.execute(
            "INSERT INTO build (source_revision, manifest_hash, signature_schema, "
            "hardware_schema, variant_set) VALUES ('deadbeefdeadbeefdead', 'aa', 1, 1, 'inventory')"
        )
        self.build_id = self.conn.execute("SELECT last_insert_rowid()").fetchone()[0]
        self.conn.execute(
            "INSERT INTO hardware (hardware_digest, architecture, architecture_code, "
            "wave_size, compute_units, feature_flags, canonical_json) VALUES "
            "(?, 'gfx1100', 1, 32, 96, 0, '{}')",
            (bytes.fromhex(HARDWARE_HEX),),
        )
        self.hardware_id = self.conn.execute("SELECT last_insert_rowid()").fetchone()[0]
        self.conn.execute(
            "INSERT INTO signature (signature_digest, base_digest, schema_version, op, "
            "src0_type, src1_type, dst_type, m, n, k, canonical_json) VALUES "
            "(?, x'02', 1, 'GLU', 'q8_0', 'f32', 'f32', 256, 8, 2048, ?)",
            (bytes.fromhex(GLU_SIGNATURE_HEX), json.dumps(GLU_CANONICAL_SIGNATURE)),
        )
        self.signature_id = self.conn.execute("SELECT last_insert_rowid()").fetchone()[0]
        self.conn.execute(
            "INSERT INTO candidate (build_id, stable_name, family, source_class, "
            "implementation_version, architectures, architecture_mask, graph_safe, "
            "deterministic, config_json) VALUES (?, 'native', 'mmvq', 'native_wrapper', "
            "1, '[]', 0, 1, 1, '{}')",
            (self.build_id,),
        )
        self.native_candidate_id = self.conn.execute("SELECT last_insert_rowid()").fetchone()[0]
        self.conn.execute(
            "INSERT INTO candidate (build_id, stable_name, family, source_class, "
            "implementation_version, architectures, architecture_mask, graph_safe, "
            "deterministic, config_json) VALUES (?, 'mmvq:q8_0:w1:nw1:rpb1:sk0:v1', 'mmvq', "
            "'new_generated_variant', 1, '[]', 0, 1, 1, '{}')",
            (self.build_id,),
        )
        self.candidate_id = self.conn.execute("SELECT last_insert_rowid()").fetchone()[0]
        self.conn.execute(
            "INSERT INTO measurement (build_id, hardware_id, signature_id, dispatch_digest, "
            "candidate_id, objective, stage, accepted) VALUES (?, ?, ?, ?, ?, 'latency', "
            "'final', 1)",
            (self.build_id, self.hardware_id, self.signature_id,
             bytes.fromhex(DISPATCH_HEX), self.candidate_id),
        )
        self.conn.commit()

        self._tmp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp_dir.cleanup)
        self.vendor = _write_fixture_vendor_with_glu(Path(self._tmp_dir.name))

        self.row = {
            "dispatch": DISPATCH_HEX,
            "signature": GLU_SIGNATURE_HEX,
            "hardware": HARDWARE_HEX,
            "native": "native",
            "provisional_winner": "mmvq:q8_0:w1:nw1:rpb1:sk0:v1",
        }

    def test_glu_row_reaches_real_evidence_generation(self):
        outcome = cli.generate_for_row(
            self.conn, self.row, binary=Path("test-backend-ops"), vendor_root=self.vendor,
            seeds=(1, 2, 3), headroom_fraction=ce.DEFAULT_HEADROOM_FRACTION,
            contract_version=ce.CONTRACT_VERSION, tool_version="test",
            runner=_fake_glu_runner_factory(),
        )
        self.assertIn("wrote evidence_id=", outcome)
        stored = self.conn.execute(
            "SELECT candidate_id, native_candidate_id FROM correctness_evidence"
        ).fetchone()
        self.assertEqual(stored, (self.candidate_id, self.native_candidate_id))

    def test_evidence_written_passes_the_real_promotion_gate(self):
        cli.generate_for_row(
            self.conn, self.row, binary=Path("test-backend-ops"), vendor_root=self.vendor,
            seeds=(1, 2, 3), headroom_fraction=ce.DEFAULT_HEADROOM_FRACTION,
            contract_version=ce.CONTRACT_VERSION, tool_version="test",
            runner=_fake_glu_runner_factory(),
        )
        identity = gate.resolve_promotion_identity(
            self.conn, dispatch_hex=DISPATCH_HEX, signature_hex=GLU_SIGNATURE_HEX,
            hardware_hex=HARDWARE_HEX, native_name="native",
            candidate_name="mmvq:q8_0:w1:nw1:rpb1:sk0:v1",
        )
        passed, status = gate.evaluate_correctness_gate(self.conn, identity)
        self.assertTrue(passed, status)

    def test_observed_signature_mismatch_raises_cli_error_and_writes_no_evidence(self):
        # A wrong observed hex must be treated as "the synthetic graph may
        # not have reproduced the real fused dispatch" and refuse to
        # certify correctness -- this is the whole point of HI119's
        # observed-digest gate (dev-gpt-agent design review, 2026-08-25),
        # not an incidental side effect.
        with self.assertRaises(cli.CliError) as ctx:
            cli.generate_for_row(
                self.conn, self.row, binary=Path("test-backend-ops"), vendor_root=self.vendor,
                seeds=(1, 2, 3), headroom_fraction=ce.DEFAULT_HEADROOM_FRACTION,
                contract_version=ce.CONTRACT_VERSION, tool_version="test",
                runner=_fake_glu_runner_factory(observed_hex="ff" * 16),
            )
        self.assertIn("does not match", str(ctx.exception))
        count = self.conn.execute("SELECT COUNT(*) FROM correctness_evidence").fetchone()[0]
        self.assertEqual(count, 0)

    def test_no_observation_row_raises_cli_error(self):
        def broken_runner(argv, capture_output, text, env):
            if env.get("GGML_HIP_DISPATCH_MODE") == "record":
                Path(env["GGML_HIP_DISPATCH_DB"]).write_text("", encoding="utf-8")
                return subprocess.CompletedProcess(argv, 0, stdout="", stderr="")
            return _fake_glu_runner_factory()(argv, capture_output, text, env)

        # HI121/HI125: _observed_signature_hex() now delegates to
        # signature_digest_verification's generalized primitive, which
        # raises ce.EvidenceError (not CliError) -- still caught by the
        # CLI's own broader except (..., ce.EvidenceError, CliError).
        with self.assertRaises(ce.EvidenceError) as ctx:
            cli.generate_for_row(
                self.conn, self.row, binary=Path("test-backend-ops"), vendor_root=self.vendor,
                seeds=(1, 2, 3), headroom_fraction=ce.DEFAULT_HEADROOM_FRACTION,
                contract_version=ce.CONTRACT_VERSION, tool_version="test",
                runner=broken_runner,
            )
        self.assertIn("0 distinct signature(s)", str(ctx.exception))


def _fake_runner_with_digests_factory(*, digest="cafebabe", threshold=5e-4, e_c=1e-5, max_abs_c=0.0004):
    def runner(argv, capture_output, text, env):
        mode = env.get("GGML_HIP_DISPATCH_MODE")
        tensor = "out"
        stderr = (
            f"BIGCHERRY_REF_DIGEST name=leaf_0 call_index=0 digest={digest} nels=16\n"
            f"BIGCHERRY_CORRECTNESS_METRIC op=MUL_MAT tensor={tensor} "
            f"backend1=native backend2={'native' if mode == 'native' else 'candidate'} "
            f"err={e_c if mode != 'native' else 1e-6} max_abs={max_abs_c if mode != 'native' else 0.0005} "
            f"threshold={threshold} n=16 "
            f"backend1_digest={'aa' * 8 if mode == 'native' else 'bb' * 8} "
            f"backend2_digest={'cc' * 8}\n"
        )
        return subprocess.CompletedProcess(argv, 0, stdout="", stderr=stderr)
    return runner


class GenerateForCandidateTests(_Base):
    """HTR01 (2026-08-30): generate_for_candidate() -- the primitive
    recovery.py's lazy correctness-qualification path calls -- gets its
    own explicit-candidate-name, origin-recording, and native-seed-reuse
    behavior exercised directly, not just through generate_for_row's
    thin wrapper."""

    def setUp(self):
        super().setUp()
        self.conn.execute(
            "INSERT INTO candidate (build_id, stable_name, family, source_class, "
            "implementation_version, architectures, architecture_mask, graph_safe, "
            "deterministic, config_json) VALUES (?, 'mmq:fb2', 'mmq', 'existing_alternative', "
            "1, '[]', 0, 1, 1, '{}')",
            (self.build_id,),
        )
        self.conn.commit()

    def test_generates_evidence_for_an_explicit_candidate_not_provisional_winner(self):
        result = cli.generate_for_candidate(
            self.conn, self.row, candidate_name="mmq:fb2",
            binary=Path("test-backend-ops"), vendor_root=self.vendor,
            seeds=(1, 2, 3), headroom_fraction=ce.DEFAULT_HEADROOM_FRACTION,
            contract_version=ce.CONTRACT_VERSION, tool_version="test",
            origin=ce.EvidenceOrigin(reason="recovery_alternative", recovery_run_id="r1"),
            runner=_fake_runner_factory(),
        )
        self.assertEqual(result.status, "generated")
        self.assertIsInstance(result.evidence_id, int)
        origin_row = self.conn.execute(
            "SELECT reason, recovery_run_id FROM correctness_evidence_origin "
            "WHERE correctness_evidence_id = ?", (result.evidence_id,),
        ).fetchone()
        self.assertEqual(origin_row, ("recovery_alternative", "r1"))

    def test_existing_evidence_short_circuits_with_zero_subprocess_runs(self):
        first = cli.generate_for_candidate(
            self.conn, self.row, candidate_name="mmq:fb2",
            binary=Path("test-backend-ops"), vendor_root=self.vendor,
            seeds=(1, 2, 3), headroom_fraction=ce.DEFAULT_HEADROOM_FRACTION,
            contract_version=ce.CONTRACT_VERSION, tool_version="test",
            origin=ce.EvidenceOrigin(reason="recovery_alternative"),
            runner=_fake_runner_factory(),
        )
        second = cli.generate_for_candidate(
            self.conn, self.row, candidate_name="mmq:fb2",
            binary=Path("test-backend-ops"), vendor_root=self.vendor,
            seeds=(1, 2, 3), headroom_fraction=ce.DEFAULT_HEADROOM_FRACTION,
            contract_version=ce.CONTRACT_VERSION, tool_version="test",
            origin=ce.EvidenceOrigin(reason="recovery_alternative"),
            runner=_fake_runner_factory(),
        )
        self.assertEqual(second.status, "existing")
        self.assertEqual(second.evidence_id, first.evidence_id)
        self.assertEqual(second.subprocess_runs, 0)

    def test_native_seed_cache_reuse_halves_subprocess_runs_for_a_second_candidate(self):
        cache: dict[int, ce.NativeSeedEvidence] = {}
        first = cli.generate_for_candidate(
            self.conn, self.row, candidate_name="mmq:fb1",
            binary=Path("test-backend-ops"), vendor_root=self.vendor,
            seeds=(1, 2, 3), headroom_fraction=ce.DEFAULT_HEADROOM_FRACTION,
            contract_version=ce.CONTRACT_VERSION, tool_version="test",
            origin=ce.EvidenceOrigin(reason="promotion_winner"),
            native_seed_cache=cache, runner=_fake_runner_factory(),
        )
        # First candidate for this signature: 3 native + 3 candidate runs.
        self.assertEqual(first.subprocess_runs, 6)
        self.assertEqual(len(cache), 3)

        second = cli.generate_for_candidate(
            self.conn, self.row, candidate_name="mmq:fb2",
            binary=Path("test-backend-ops"), vendor_root=self.vendor,
            seeds=(1, 2, 3), headroom_fraction=ce.DEFAULT_HEADROOM_FRACTION,
            contract_version=ce.CONTRACT_VERSION, tool_version="test",
            origin=ce.EvidenceOrigin(reason="recovery_alternative"),
            native_seed_cache=cache, runner=_fake_runner_factory(),
        )
        # Second candidate, same signature, native already cached: 3
        # candidate runs only -- HTR01's whole cost-amortization point.
        self.assertEqual(second.subprocess_runs, 3)

    def test_output_digests_are_persisted_on_the_seed_rows(self):
        result = cli.generate_for_candidate(
            self.conn, self.row, candidate_name="mmq:fb2",
            binary=Path("test-backend-ops"), vendor_root=self.vendor,
            seeds=(1, 2, 3), headroom_fraction=ce.DEFAULT_HEADROOM_FRACTION,
            contract_version=ce.CONTRACT_VERSION, tool_version="test",
            origin=ce.EvidenceOrigin(reason="recovery_alternative"),
            runner=_fake_runner_with_digests_factory(),
        )
        seed_row = self.conn.execute(
            "SELECT native_output_digest, candidate_output_digest, reference_output_digest, "
            "output_nels FROM correctness_evidence_seed WHERE correctness_evidence_id = ? "
            "AND seed = 1", (result.evidence_id,),
        ).fetchone()
        self.assertIsNotNone(seed_row)
        native_digest, candidate_digest, reference_digest, nels = seed_row
        self.assertIsNotNone(native_digest)
        self.assertIsNotNone(candidate_digest)
        self.assertIsNotNone(reference_digest)
        self.assertEqual(nels, 16)


if __name__ == "__main__":
    unittest.main()
