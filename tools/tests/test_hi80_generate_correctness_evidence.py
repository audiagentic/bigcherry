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

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bigcherry import correctness_evidence as ce  # noqa: E402
from bigcherry import hi80_generate_correctness_evidence as cli  # noqa: E402
from bigcherry import promotion_correctness_gate as gate  # noqa: E402
from bigcherry import signature_correctness_mapping as scm  # noqa: E402

SCHEMA_SQL = Path(__file__).resolve().parents[2] / "sql" / "dispatch-db.sql"

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


if __name__ == "__main__":
    unittest.main()
