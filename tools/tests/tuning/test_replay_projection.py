"""HI121 M4: end-to-end tests for replay_projection.project_measurements() --
selective, capability-gated reuse feeding into the UNCHANGED replay.build()."""

from __future__ import annotations

import json
import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from bigcherry.tuning import catalog  # noqa: E402
from bigcherry.tuning import dispatch_abi  # noqa: E402
from bigcherry.tuning import replay as replay_module  # noqa: E402
from bigcherry.tuning import replay_projection as rp  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[3]
SCHEMA_SQL = REPO_ROOT / "sql" / "dispatch-db.sql"

EPOCH = dispatch_abi.SIGNATURE_IDENTITY_EPOCH

SOURCE_REVISION = "b" * 40
CORE_ONLY_HEX = "0" * 31 + "1"  # CORE_SIGNATURE_V1 only
ALL_FIVE_HEX = "0000000000000000000000000000001f"  # CORE + all 4 HI118 presence caps


def _write_fixture_vendor(tmp_path: Path, *, producer_capabilities_hex: str) -> Path:
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
    lo = int(producer_capabilities_hex[16:], 16)
    hi = int(producer_capabilities_hex[:16], 16)
    (vendor / "ggml" / "src" / "ggml-cuda").mkdir(parents=True, exist_ok=True)
    (vendor / "ggml" / "src" / "ggml-cuda" / "hip-autotune-types.h").write_text(
        f"#define GGML_HIP_PRODUCER_CAPABILITIES_LO UINT64_C({hex(lo)})\n"
        f"#define GGML_HIP_PRODUCER_CAPABILITIES_HI UINT64_C({hex(hi)})\n",
        encoding="utf-8",
    )
    return vendor


def _make_manifest(*, producer_capabilities_hex: str) -> dict:
    families = ("mmvq", "mmq", "mmvf", "mmf", "blas")
    manifest = {
        "artifact_version": 1,
        "variant_set": "inventory",
        "source_revision": SOURCE_REVISION,
        "architectures": ["gfx1100"],
        "signature_schema_version": EPOCH,
        "hardware_schema_version": 1,
        "producer_capabilities": producer_capabilities_hex,
        "candidates": [
            {
                "stable_name": f"{f}:native:v1", "family": f, "source_class": "native_wrapper",
                "implementation_version": 1, "architectures": ["gfx1100"], "architecture_mask": 1,
                "graph_safe": True, "deterministic": True, "config": {},
            }
            for f in families
        ],
        "summary": {
            "total": len(families),
            "by_family": dict.fromkeys(families, 1),
            "by_source_class": {"native_wrapper": len(families)},
        },
    }
    manifest["manifest_hash"] = catalog.manifest_hash(manifest)
    manifest["build_descriptor"] = catalog.build_descriptor(manifest)
    return manifest


class ProjectMeasurementsTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.tmp_path = Path(self._tmp.name)

        self.vendor = _write_fixture_vendor(self.tmp_path, producer_capabilities_hex=ALL_FIVE_HEX)

        self.dispatch_db = self.tmp_path / "dispatch.sqlite"
        self.conn = sqlite3.connect(str(self.dispatch_db))
        self.conn.execute("PRAGMA foreign_keys = ON")
        self.conn.executescript(SCHEMA_SQL.read_text(encoding="utf-8"))

        self.manifest = _make_manifest(producer_capabilities_hex=ALL_FIVE_HEX)
        self.manifest_path = self.tmp_path / "manifest.json"
        self.manifest_path.write_text(json.dumps(self.manifest), encoding="utf-8")

        self.conn.execute(
            "INSERT INTO build (source_revision, manifest_hash, signature_schema, "
            "hardware_schema, variant_set) VALUES (?, ?, ?, 1, 'inventory')",
            (SOURCE_REVISION, self.manifest["manifest_hash"], EPOCH),
        )
        self.build_id = self.conn.execute("SELECT last_insert_rowid()").fetchone()[0]

        # S1: plain MUL_MAT, requires CORE only.
        self.s1_hex = "11" * 16
        s1_canonical = {"schema_version": EPOCH, "op": 2, "flags": 0, "fusion": 0, "glu_op": 0}
        self.conn.execute(
            "INSERT INTO signature (signature_digest, base_digest, schema_version, op, "
            "src0_type, src1_type, dst_type, m, n, k, canonical_json) VALUES "
            "(?, x'02', ?, 'MUL_MAT', 'f32', 'f32', 'f32', 1, 1, 1, ?)",
            (bytes.fromhex(self.s1_hex), EPOCH, json.dumps(s1_canonical)),
        )
        # S2: GLU with all HI118 content flags zero -- requires all 4 presence caps.
        self.s2_hex = "22" * 16
        s2_canonical = {"schema_version": EPOCH, "op": 4, "flags": 1 << 3, "fusion": 2, "glu_op": 2}
        self.conn.execute(
            "INSERT INTO signature (signature_digest, base_digest, schema_version, op, "
            "src0_type, src1_type, dst_type, m, n, k, canonical_json) VALUES "
            "(?, x'02', ?, 'MUL_MAT_ID', 'q8_0', 'f32', 'f32', 1, 1, 1, ?)",
            (bytes.fromhex(self.s2_hex), EPOCH, json.dumps(s2_canonical)),
        )
        self.conn.commit()

        self.header = {
            "kind": "header",
            "artifact_version": 1,
            "source_revision": SOURCE_REVISION,
            "manifest_hash": self.manifest["manifest_hash"],
            "variant_set": "inventory",
        }
        hardware_hex = "bb" * 16
        self.s1_result = {
            "kind": "result",
            "dispatch": replay_module.portable_tuning_key(hardware_hex, self.s1_hex),
            "signature": self.s1_hex, "hardware": hardware_hex,
            "winner": "mmvq:native:v1", "native": "mmvq:native:v1",
            "source_revision": SOURCE_REVISION, "manifest_hash": self.manifest["manifest_hash"],
        }
        self.s2_result = {
            "kind": "result",
            "dispatch": replay_module.portable_tuning_key(hardware_hex, self.s2_hex),
            "signature": self.s2_hex, "hardware": hardware_hex,
            "winner": "mmvq:native:v1", "native": "mmvq:native:v1",
            "source_revision": SOURCE_REVISION, "manifest_hash": self.manifest["manifest_hash"],
        }
        self.measurements_path = self.tmp_path / "measurements.jsonl"
        with self.measurements_path.open("w", encoding="utf-8") as handle:
            handle.write(json.dumps(self.header) + "\n")
            handle.write(json.dumps(self.s1_result) + "\n")
            handle.write(json.dumps(self.s2_result) + "\n")

    def tearDown(self):
        self.conn.close()

    def _set_source_capabilities(self, mask_hex: str) -> None:
        self.conn.execute(
            "INSERT INTO build_capability (build_id, backend, producer_capabilities) "
            "VALUES (?, 'hip', ?)",
            (self.build_id, bytes.fromhex(mask_hex)),
        )
        self.conn.commit()

    def test_missing_source_capability_row_raises(self):
        with self.assertRaises(rp.ProjectionError):
            rp.project_measurements(
                self.measurements_path, self.tmp_path / "out.jsonl",
                dispatch_db=self.dispatch_db, source_build_id=self.build_id,
                target_manifest_path=self.manifest_path, vendor_root=self.vendor,
            )

    def test_full_capabilities_both_sides_retains_both_rows(self):
        self._set_source_capabilities(ALL_FIVE_HEX)
        output = self.tmp_path / "out.jsonl"
        summary = rp.project_measurements(
            self.measurements_path, output,
            dispatch_db=self.dispatch_db, source_build_id=self.build_id,
            target_manifest_path=self.manifest_path, vendor_root=self.vendor,
        )
        self.assertEqual(summary.examined, 2)
        self.assertEqual(summary.retained, 2)
        self.assertEqual(summary.omitted_missing_producer_capability, 0)
        self.assertEqual(summary.omitted_missing_target_capability, 0)
        self.assertEqual(summary.omitted_unsupported_domain, 0)

        lines = output.read_text(encoding="utf-8").splitlines()
        kinds = [json.loads(line)["kind"] for line in lines]
        self.assertEqual(kinds, ["header", "result", "result"])

    def test_core_only_source_capability_omits_glu_row_but_keeps_mul_mat(self):
        # S1 (plain MUL_MAT) needs only CORE; S2 (GLU) needs all 4 HI118
        # presence caps too -- a CORE-only source producer cannot certify S2.
        self._set_source_capabilities(CORE_ONLY_HEX)
        output = self.tmp_path / "out.jsonl"
        summary = rp.project_measurements(
            self.measurements_path, output,
            dispatch_db=self.dispatch_db, source_build_id=self.build_id,
            target_manifest_path=self.manifest_path, vendor_root=self.vendor,
        )
        self.assertEqual(summary.retained, 1)
        self.assertEqual(summary.omitted_missing_producer_capability, 1)
        results = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()
                   if json.loads(line).get("kind") == "result"]
        self.assertEqual(results[0]["signature"], self.s1_hex)

    def test_core_only_target_omits_glu_row_even_with_full_source_capability(self):
        # HI121's own added strengthening: the TARGET must also hold the
        # required capabilities, not just the source.
        self._set_source_capabilities(ALL_FIVE_HEX)
        core_only_manifest = _make_manifest(producer_capabilities_hex=CORE_ONLY_HEX)
        core_only_vendor = _write_fixture_vendor(
            self.tmp_path / "core-only-target", producer_capabilities_hex=CORE_ONLY_HEX
        )
        core_only_manifest_path = self.tmp_path / "core-only-manifest.json"
        core_only_manifest_path.write_text(json.dumps(core_only_manifest), encoding="utf-8")

        output = self.tmp_path / "out.jsonl"
        summary = rp.project_measurements(
            self.measurements_path, output,
            dispatch_db=self.dispatch_db, source_build_id=self.build_id,
            target_manifest_path=core_only_manifest_path, vendor_root=core_only_vendor,
        )
        self.assertEqual(summary.retained, 1)
        self.assertEqual(summary.omitted_missing_target_capability, 1)

    def test_projected_file_feeds_into_unchanged_replay_build(self):
        self._set_source_capabilities(ALL_FIVE_HEX)
        output = self.tmp_path / "out.jsonl"
        rp.project_measurements(
            self.measurements_path, output,
            dispatch_db=self.dispatch_db, source_build_id=self.build_id,
            target_manifest_path=self.manifest_path, vendor_root=self.vendor,
        )
        ggml_h = self.tmp_path / "ggml.h"
        ggml_h.write_text("enum ggml_type { GGML_TYPE_F32 = 0 };\n", encoding="utf-8")
        blob = replay_module.build(output, self.manifest_path, ggml_h)
        header, entries = replay_module.read_cache(blob)
        self.assertEqual(header["version"], replay_module.REPLAY_VERSION)
        self.assertEqual(len(entries), 2)

    def test_wrong_source_build_id_header_mismatch_raises(self):
        self.conn.execute(
            "INSERT INTO build (source_revision, manifest_hash, signature_schema, "
            "hardware_schema, variant_set) VALUES ('c'||?, 'different', ?, 1, 'inventory')",
            (SOURCE_REVISION[1:], EPOCH),
        )
        other_build_id = self.conn.execute("SELECT last_insert_rowid()").fetchone()[0]
        self.conn.execute(
            "INSERT INTO build_capability (build_id, backend, producer_capabilities) "
            "VALUES (?, 'hip', ?)",
            (other_build_id, bytes.fromhex(ALL_FIVE_HEX)),
        )
        self.conn.commit()
        with self.assertRaises(rp.ProjectionError):
            rp.project_measurements(
                self.measurements_path, self.tmp_path / "out.jsonl",
                dispatch_db=self.dispatch_db, source_build_id=other_build_id,
                target_manifest_path=self.manifest_path, vendor_root=self.vendor,
            )

    def test_cli_project_replay_command(self):
        from bigcherry.cli import main as cli_main

        self._set_source_capabilities(ALL_FIVE_HEX)
        output = self.tmp_path / "out.jsonl"
        exit_code = cli_main.main([
            "project-replay", str(self.measurements_path),
            "--dispatch-db", str(self.dispatch_db),
            "--source-build-id", str(self.build_id),
            "--target-manifest", str(self.manifest_path),
            "--vendor-root", str(self.vendor),
            "--output", str(output),
            "--json",
        ])
        self.assertEqual(exit_code, 0)
        self.assertTrue(output.is_file())
        results = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()
                   if json.loads(line).get("kind") == "result"]
        self.assertEqual(len(results), 2)

    def test_target_manifest_from_wrong_source_root_raises(self):
        self._set_source_capabilities(ALL_FIVE_HEX)
        wrong_vendor = _write_fixture_vendor(self.tmp_path / "wrong-root", producer_capabilities_hex=CORE_ONLY_HEX)
        with self.assertRaises(rp.ProjectionError):
            rp.project_measurements(
                self.measurements_path, self.tmp_path / "out.jsonl",
                dispatch_db=self.dispatch_db, source_build_id=self.build_id,
                # manifest claims ALL_FIVE_HEX but this vendor_root only declares CORE_ONLY_HEX
                target_manifest_path=self.manifest_path, vendor_root=wrong_vendor,
            )


if __name__ == "__main__":
    unittest.main()
