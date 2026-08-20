"""RE30 phase 2: Vulkan autotune identity-type tests.

Covers vk_autotune_types.py's validators/canonical-JSON/digest functions and
the new vk_* tables in sql/dispatch-db.sql. Real SQLite throughout (no
mocking), same convention as test_re09_schema_v4.py.
"""

from __future__ import annotations

import sqlite3
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bigcherry import vk_autotune_types as vk  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
SCHEMA_SQL = REPO_ROOT / "sql" / "dispatch-db.sql"


def _hex_digest(byte: int = 0xAB) -> str:
    return bytes([byte]) .hex() * vk.DIGEST_BYTES


def _sample_hardware() -> dict:
    return {
        "vendor_id": 0x1002,
        "device_id": 0x744C,
        "device_class": "rdna3.5-strix-halo",
        "driver_version": "24.20.1",
        "api_version": "1.3.280",
        "subgroup_size": 32,
        "subgroup_ops_mask": 0b1111,
        "extensions": ["VK_KHR_shader_float16_int8", "VK_KHR_cooperative_matrix"],
        "limits": {"maxComputeSharedMemorySize": 65536},
        "shader_toolchain_digest": _hex_digest(0xAB),
    }


def _sample_signature() -> dict:
    return {
        "op": "MUL_MAT",
        "src0_type": "Q8_0",
        "src1_type": "F32",
        "dst_type": "F32",
        "output_precision": "f32",
        "accumulation_precision": "f32",
        "m": 4096,
        "n": 1,
        "k": 4096,
        "layout": "row_major",
        "alignment_class": 4,
        "batching": "none",
        "conversion_route": "dequant_once",
        "split_k": 0,
        "fusion": "none",
    }


def _sample_candidate() -> dict:
    return {
        "stable_name": "vk:mul_mat:q8_0:coopmat:v1",
        "family": "mul_mat",
        "source_class": "native_wrapper",
        "pipeline_stage_count": 1,
        "shader_module_digests": [_hex_digest(0xCD)],
    }


class VkHardwareValidationTests(unittest.TestCase):
    def test_valid_hardware_round_trips(self):
        hw = _sample_hardware()
        vk.validate_vk_hardware(hw)  # must not raise

    def test_missing_field_rejected(self):
        hw = _sample_hardware()
        del hw["driver_version"]
        with self.assertRaises(vk.VkSchemaError):
            vk.validate_vk_hardware(hw)

    def test_bad_shader_digest_length_rejected(self):
        hw = _sample_hardware()
        hw["shader_toolchain_digest"] = "deadbeef"
        with self.assertRaises(vk.VkSchemaError):
            vk.validate_vk_hardware(hw)

    def test_canonical_json_is_deterministic(self):
        hw = _sample_hardware()
        first = vk.vk_hardware_canonical_json(hw)
        second = vk.vk_hardware_canonical_json(dict(hw))
        self.assertEqual(first, second)

    def test_extension_order_does_not_affect_digest(self):
        hw_a = _sample_hardware()
        hw_b = _sample_hardware()
        hw_b["extensions"] = list(reversed(hw_b["extensions"]))
        self.assertEqual(vk.vk_hardware_digest(hw_a), vk.vk_hardware_digest(hw_b))

    def test_digest_is_16_bytes(self):
        self.assertEqual(len(vk.vk_hardware_digest(_sample_hardware())), 16)


class VkSignatureValidationTests(unittest.TestCase):
    def test_valid_signature_round_trips(self):
        vk.validate_vk_signature(_sample_signature())  # must not raise

    def test_bad_layout_rejected(self):
        sig = _sample_signature()
        sig["layout"] = "diagonal"
        with self.assertRaises(vk.VkSchemaError):
            vk.validate_vk_signature(sig)

    def test_digest_changes_with_dimensions(self):
        sig_a = _sample_signature()
        sig_b = _sample_signature()
        sig_b["k"] = 8192
        self.assertNotEqual(vk.vk_signature_digest(sig_a), vk.vk_signature_digest(sig_b))

    def test_base_digest_ignores_refinements(self):
        sig_a = _sample_signature()
        sig_b = _sample_signature()
        sig_b["alignment_class"] = 0
        sig_b["batching"] = "batch4"
        # full digest differs (alignment_class/batching are refinements)...
        self.assertNotEqual(vk.vk_signature_digest(sig_a), vk.vk_signature_digest(sig_b))
        # ...but the base digest (refinements stripped) is identical.
        self.assertEqual(vk.vk_signature_base_digest(sig_a), vk.vk_signature_base_digest(sig_b))

    def test_signature_and_hardware_digest_namespaces_differ(self):
        # Same personalisation-prefix reasoning as HIP's
        # GGML_HIP_PERSON_SIGNATURE vs GGML_HIP_PERSON_HARDWARE: distinct
        # `person` bytes must be actually distinct, not accidentally equal.
        self.assertNotEqual(vk.PERSON_VK_SIGNATURE, vk.PERSON_VK_HARDWARE)


class VkCandidateValidationTests(unittest.TestCase):
    def test_valid_candidate_round_trips(self):
        vk.validate_vk_candidate(_sample_candidate())  # must not raise

    def test_bad_family_rejected(self):
        cand = _sample_candidate()
        cand["family"] = "mmq"  # HIP family name, not a Vulkan one
        with self.assertRaises(vk.VkSchemaError):
            vk.validate_vk_candidate(cand)

    def test_stage_count_digest_mismatch_rejected(self):
        cand = _sample_candidate()
        cand["pipeline_stage_count"] = 2
        # shader_module_digests still has only 1 entry -- must be rejected,
        # not silently truncated/padded (RE30: a candidate must be a
        # *complete* recipe, never a partial one).
        with self.assertRaises(vk.VkSchemaError):
            vk.validate_vk_candidate(cand)


class VkSchemaTablesTests(unittest.TestCase):
    """The new vk_* tables apply cleanly and existing HIP tables/behavior
    are provably unaffected."""

    def setUp(self):
        self.conn = sqlite3.connect(":memory:")
        self.conn.executescript(SCHEMA_SQL.read_text(encoding="utf-8"))

    def tearDown(self):
        self.conn.close()

    def test_vk_tables_exist(self):
        tables = {
            row[0] for row in self.conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
        }
        for name in ("vk_hardware", "vk_signature", "vk_candidate",
                      "vk_observation", "vk_measurement", "vk_winner"):
            self.assertIn(name, tables)

    def test_schema_version_unchanged_at_4(self):
        # Additive orthogonal namespace, not a reinterpretation of any
        # existing hashed field -- schema_version must NOT bump (see the
        # comment block above the vk_* tables in dispatch-db.sql).
        row = self.conn.execute(
            "SELECT value FROM schema_meta WHERE key = 'schema_version'"
        ).fetchone()
        self.assertEqual(row[0], "4")

    def test_hip_tables_and_constraints_unaffected(self):
        # A HIP candidate insert must still enforce the original HIP-only
        # family vocabulary -- proves the vk_candidate CHECK (mul_mat |
        # mul_mat_id) is a completely separate constraint, not a widened
        # shared one.
        self.conn.execute(
            "INSERT INTO build (source_revision, manifest_hash, signature_schema, "
            "hardware_schema, variant_set) VALUES ('deadbeefdeadbeefdead', 'aa', 1, 1, 'inventory')"
        )
        build_id = self.conn.execute("SELECT last_insert_rowid()").fetchone()[0]
        with self.assertRaises(sqlite3.IntegrityError):
            self.conn.execute(
                "INSERT INTO candidate (build_id, stable_name, family, source_class, "
                "implementation_version, architectures, architecture_mask, graph_safe, "
                "deterministic, config_json) VALUES (?, 'x', 'mul_mat', 'native_wrapper', "
                "1, '[]', 0, 0, 0, '{}')",
                (build_id,),
            )

    def test_vk_candidate_rejects_hip_family_name(self):
        self.conn.execute(
            "INSERT INTO build (source_revision, manifest_hash, signature_schema, "
            "hardware_schema, variant_set) VALUES ('deadbeefdeadbeefdead', 'aa', 1, 1, 'inventory')"
        )
        build_id = self.conn.execute("SELECT last_insert_rowid()").fetchone()[0]
        with self.assertRaises(sqlite3.IntegrityError):
            self.conn.execute(
                "INSERT INTO vk_candidate (build_id, stable_name, family, source_class, "
                "implementation_version, shader_module_digests_json, graph_safe, "
                "deterministic, config_json) VALUES (?, 'x', 'mmq', 'native_wrapper', "
                "1, '[]', 0, 0, '{}')",
                (build_id,),
            )

    def test_vk_winner_dispatch_digest_unique_index_present(self):
        indexes = {
            row[0] for row in self.conn.execute(
                "SELECT name FROM sqlite_master WHERE type='index'"
            )
        }
        self.assertIn("vk_winner_dispatch_idx", indexes)


if __name__ == "__main__":
    unittest.main()
