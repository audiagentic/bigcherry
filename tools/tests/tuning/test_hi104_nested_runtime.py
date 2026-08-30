"""HI104: the runtime transform artifact uses complete nested provenance."""

from __future__ import annotations

import hashlib
import json
import re
import sqlite3
import tempfile
import unittest
from pathlib import Path

from bigcherry.analysis.gaps import analyze_gap_file
from bigcherry.transform_loader import load_transforms
from bigcherry.transform_records import load_artifact_records


ROOT = Path(__file__).resolve().parents[3]
SCHEMA = ROOT / "sql" / "dispatch-db.sql"
MANIFEST_PATH = ROOT / "vendor/llama.cpp/ggml/src/ggml-cuda/hip-autotune-manifest.json"
HASH_HEADER = ROOT / "vendor/llama.cpp/ggml/src/ggml-cuda/hip-autotune-build-hash.h"
TUNER_SOURCE = ROOT / "vendor/llama.cpp/ggml/src/ggml-cuda/hip-autotune-tuner.cu"


def _generated_identity() -> tuple[str, str, str]:
    header = HASH_HEADER.read_text(encoding="utf-8")
    revision = re.search(
        r"GGML_HIP_AUTOTUNE_SOURCE_REVISION_STR\s+\"([0-9a-f]+)\"", header
    ).group(1)
    manifest_hash = re.search(
        r"GGML_HIP_AUTOTUNE_MANIFEST_HASH_STR\s+\"([0-9a-f]+)\"", header
    ).group(1)
    descriptor_hash = re.search(
        r"GGML_HIP_AUTOTUNE_DESCRIPTOR_HASH_STR\s+\"([0-9a-f]+)\"", header
    ).group(1)
    return revision, manifest_hash, descriptor_hash


class TestHi104BuildDescriptorParity(unittest.TestCase):
    def test_cpp_flush_hash_matches_offline_manifest_descriptor_hash(self):
        """The generated C++ descriptor JSON and Python manifest are one hash."""
        header = HASH_HEADER.read_text(encoding="utf-8")
        json_literal = re.search(
            r"GGML_HIP_AUTOTUNE_DESCRIPTOR_JSON\s+(\"(?:\\.|[^\"])*\")",
            header,
        ).group(1)
        cpp_descriptor = json.loads(json.loads(json_literal))
        manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
        # The checked-in generated manifest is the shared fixture for the
        # C++ header.  Recompute its timestamp-free descriptor identity using
        # the same canonical JSON + BLAKE2b-128 operation as catalog.py.
        offline_descriptor = dict(manifest["build_descriptor"])
        expected_hash = offline_descriptor.pop("descriptor_hash")
        offline_hash = hashlib.blake2b(
            json.dumps(offline_descriptor, sort_keys=True,
                       separators=(",", ":")).encode(),
            digest_size=16,
        ).hexdigest()
        offline_descriptor["descriptor_hash"] = expected_hash

        self.assertEqual(cpp_descriptor, offline_descriptor)
        cpp_payload = dict(cpp_descriptor)
        cpp_payload.pop("descriptor_hash")
        cpp_hash = hashlib.blake2b(
            json.dumps(cpp_payload, sort_keys=True, separators=(",", ":")).encode(),
            digest_size=16,
        ).hexdigest()
        self.assertEqual(cpp_hash, _generated_identity()[2])
        self.assertEqual(expected_hash, offline_hash)
        self.assertEqual(offline_descriptor["descriptor_hash"], cpp_hash)

    def test_cpp_flush_writes_nested_runtime_records(self):
        source = TUNER_SOURCE.read_text(encoding="utf-8")
        flush = source[source.index("void ggml_hip_tuner_flush()"):]
        for field in (
            r'\"transform_schema_version\":1',
            r'\"hardware_provenance\":{\"digest\":\"%s\",\"architecture\":\"%s\"}',
            r'\"build_provenance\":{\"source_revision\":\"%s\",',
            r'\"manifest_hash\":\"%s\",\"build_descriptor_hash\":\"%s\"}',
            r'\"evidence_references\":[\"%s\"]',
        ):
            self.assertIn(field, flush)
        self.assertIn("runtime_build_descriptor_hash()", source)
        self.assertNotIn(r'\"original_sig\":\"%s\",\"hardware\":\"%s\"', flush)


class TestHi104NestedRuntimeArtifact(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="bigcherry_hi104_")
        self.root = Path(self.tmp.name)
        self.db = self.root / "dispatch.sqlite"
        self.revision, self.manifest_hash, self.descriptor_hash = _generated_identity()
        self.hardware = "d" * 32
        self.signature = "e" * 32
        connection = sqlite3.connect(self.db)
        try:
            connection.executescript(SCHEMA.read_text(encoding="utf-8"))
            connection.execute(
                "INSERT INTO build(source_revision, manifest_hash, signature_schema, "
                "hardware_schema, variant_set, build_descriptor_hash) VALUES (?,?,?,?,?,?)",
                (self.revision, self.manifest_hash, 1, 1, "full-max", self.descriptor_hash),
            )
            connection.execute(
                "INSERT INTO hardware(hardware_digest, architecture, architecture_code, "
                "wave_size, compute_units, feature_flags, canonical_json) VALUES (?,?,?,?,?,?,?)",
                (bytes.fromhex(self.hardware), "gfx1201", 1201, 64, 80, 0, "{}"),
            )
            connection.execute(
                "INSERT INTO signature(signature_digest, base_digest, schema_version, op, "
                "src0_type, src1_type, dst_type, m, n, k, canonical_json) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                (bytes.fromhex(self.signature), bytes.fromhex(self.signature), 1, "MUL_MAT",
                 "q4_k", "f32", "f32", 1, 8, 16, "{}"),
            )
            connection.execute(
                "INSERT INTO observation(build_id, hardware_id, signature_id, "
                "native_stable_name, calls, est_bytes) VALUES (1,1,1,?,?,?)",
                ("blas:native:v1", 17, 4096),
            )
            connection.commit()
        finally:
            connection.close()

    def tearDown(self):
        self.tmp.cleanup()

    def nested_runtime_artifact(self) -> Path:
        path = self.root / "runtime.transforms.jsonl"
        provenance = {
            "hardware_provenance": {
                "digest": self.hardware,
                "architecture": "gfx1201",
            },
            "build_provenance": {
                "source_revision": self.revision,
                "manifest_hash": self.manifest_hash,
                "build_descriptor_hash": self.descriptor_hash,
            },
            "source_schema": "runtime-nested-v1",
            "provenance_complete": True,
        }
        records = [
            {"kind": "header", "transform_schema_version": 1, "origin": "runtime"},
            {
                **provenance,
                "kind": "transform-attempt",
                "source_signature": self.signature,
                "transformation": {"id": 1, "name": "transpose", "source": "predefined"},
                "outcome": "success",
                "reason": "transform selected as winner",
                "evidence_references": ["artifact://runtime.transforms.jsonl"],
            },
            {
                **provenance,
                "kind": "transform-gap",
                "source_signature": self.signature,
                "pattern": "unclassified-runtime-gap",
                "native_family": "blas",
                "reason": "no transform served this signature",
                "evidence_references": ["artifact://runtime.transforms.jsonl#gap"],
                "transformations_tried": [
                    {"id": 1, "name": "transpose", "reason": "not selected"}
                ],
                "est_bytes": 4096,
            },
        ]
        path.write_text(
            "\n".join(json.dumps(record) for record in records) + "\n",
            encoding="utf-8",
        )
        return path

    def test_nested_runtime_records_are_complete_and_ingest_end_to_end(self):
        path = self.nested_runtime_artifact()
        records = load_artifact_records(path)
        self.assertEqual([record["source_schema"] for record in records],
                         ["runtime-nested-v1", "runtime-nested-v1"])
        self.assertTrue(all(record["provenance_complete"] for record in records))
        self.assertEqual(load_transforms(path, self.db, SCHEMA), {"attempts": 1, "gaps": 1})

        report = analyze_gap_file(path)
        self.assertEqual(report["record_count"], 1)
        self.assertTrue(report["provenance"]["hardware"]["architecture"] == "gfx1201")
        self.assertEqual(report["provenance"]["build"]["build_descriptor_hash"],
                         self.descriptor_hash)

        connection = sqlite3.connect(self.db)
        try:
            self.assertEqual(
                connection.execute("SELECT COUNT(*) FROM transform_attempt").fetchone()[0], 1
            )
            self.assertEqual(
                connection.execute("SELECT COUNT(*) FROM transform_gap").fetchone()[0], 1
            )
        finally:
            connection.close()


if __name__ == "__main__":
    unittest.main()
