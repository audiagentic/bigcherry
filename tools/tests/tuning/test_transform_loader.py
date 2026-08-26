import json
import sqlite3
import tempfile
import unittest
from pathlib import Path

from bigcherry.transform_loader import load_transforms
from bigcherry.inventory import RecordError


ROOT = Path(__file__).resolve().parents[3]
SCHEMA = ROOT / "sql" / "dispatch-db.sql"


class TestTransformLoader(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="bigcherry_transform_")
        self.root = Path(self.tmp.name)
        self.db = self.root / "dispatch.sqlite"
        conn = sqlite3.connect(self.db)
        try:
            conn.executescript(SCHEMA.read_text(encoding="utf-8"))
            conn.execute(
                "INSERT INTO build(source_revision, manifest_hash, signature_schema, "
                "hardware_schema, variant_set, build_descriptor_hash) VALUES (?,?,?,?,?,?)",
                ("a" * 40, "b" * 64, 1, 1, "inventory", "c" * 64),
            )
            conn.execute(
                "INSERT INTO hardware(hardware_digest, architecture, architecture_code, "
                "wave_size, compute_units, feature_flags, canonical_json) VALUES (?,?,?,?,?,?,?)",
                (bytes.fromhex("d" * 32), "gfx1201", 1201, 64, 80, 0, "{}"),
            )
            conn.execute(
                "INSERT INTO signature(signature_digest, base_digest, schema_version, op, "
                "src0_type, src1_type, dst_type, m, n, k, canonical_json) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                (bytes.fromhex("e" * 32), bytes.fromhex("e" * 32), 1, "MUL_MAT",
                 "q4_k", "f32", "f32", 1, 8, 16, "{}"),
            )
            conn.execute(
                "INSERT INTO observation(build_id, hardware_id, signature_id, native_stable_name, "
                "calls, est_bytes) VALUES (1,1,1,?,?,?)", ("blas:native:v1", 17, 4096)
            )
            conn.commit()
        finally:
            conn.close()

    def tearDown(self):
        self.tmp.cleanup()

    def write_records(self, *records):
        path = self.root / "transforms.jsonl"
        header = {"kind": "header", "transform_schema_version": 1}
        with path.open("w", encoding="utf-8") as handle:
            handle.write(json.dumps(header) + "\n")
            for record in records:
                handle.write(json.dumps(record) + "\n")
        return path

    def common(self, kind):
        return {
            "kind": kind,
            "source_signature": "e" * 32,
            "hardware_provenance": {"digest": "d" * 32, "architecture": "gfx1201"},
            "build_provenance": {
                "source_revision": "a" * 40,
                "manifest_hash": "b" * 64,
                "build_descriptor_hash": "c" * 64,
            },
            "reason": "no faster route",
            "evidence_references": ["evidence://transform/1"],
        }

    def test_loads_attempt_and_gap_with_observation_join_values(self):
        attempt = self.common("transform-attempt")
        attempt.update({
            "transformation": {"id": 1, "name": "transpose", "source": "predefined"},
            "outcome": "rejected",
        })
        gap = self.common("transform-gap")
        gap.update({
            "pattern": "q4_k-to-f32",
            "native_family": "blas",
            "transformations_tried": [{"id": 1, "name": "transpose", "reason": "no"}],
        })
        result = load_transforms(self.write_records(attempt, gap), self.db, SCHEMA)
        self.assertEqual(result, {"attempts": 1, "gaps": 1})
        conn = sqlite3.connect(self.db)
        try:
            self.assertEqual(conn.execute("SELECT calls, est_bytes FROM transform_gap").fetchone(), (17, 4096))
            self.assertEqual(conn.execute(
                "SELECT COUNT(*) FROM transform_gap g JOIN signature s "
                "ON g.signature_digest=s.signature_digest JOIN observation o "
                "ON o.signature_id=s.signature_id"
            ).fetchone()[0], 1)
        finally:
            conn.close()

    def test_reload_is_idempotent(self):
        attempt = self.common("transform-attempt")
        attempt.update({
            "transformation": {"id": 1, "name": "transpose", "source": "predefined"},
            "outcome": "success",
        })
        path = self.write_records(attempt)
        self.assertEqual(load_transforms(path, self.db, SCHEMA), {"attempts": 1, "gaps": 0})
        self.assertEqual(load_transforms(path, self.db, SCHEMA), {"attempts": 0, "gaps": 0})

    def test_rejects_unbound_source_signature(self):
        attempt = self.common("transform-attempt")
        attempt.update({
            "source_signature": "f" * 32,
            "transformation": {"id": 1, "name": "transpose", "source": "predefined"},
            "outcome": "rejected",
        })
        with self.assertRaisesRegex(RecordError, "source_signature"):
            load_transforms(self.write_records(attempt), self.db, SCHEMA)


if __name__ == "__main__":
    unittest.main()
