"""Immutable artifact and provenance-v2 contracts."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from bigcherry.core.artifacts import ArtifactError, ArtifactStore  # noqa: E402
from bigcherry.core import provenance # noqa: E402


class ArtifactProvenanceTests(unittest.TestCase):
    def test_store_is_atomic_immutable_and_confines_paths(self):
        with tempfile.TemporaryDirectory() as directory:
            store = ArtifactStore(Path(directory))
            digest = store.publish_bytes("run/result.json", b"one")
            self.assertTrue(store.verify("run/result.json", digest))
            self.assertEqual(store.publish_bytes("run/result.json", b"one"), digest)
            with self.assertRaises(ArtifactError):
                store.publish_bytes("run/result.json", b"two")
            with self.assertRaises(ArtifactError):
                store.publish_bytes("../escape", b"bad")

    def test_provenance_requires_complete_v2_and_exact_namespaces(self):
        document = provenance.make(
            project={"bigcherry_revision": "b"},
            source={"source_slice_id": "s"},
            build={"build_id": "b"},
            workload={"workload_id": "w"},
            campaign={"campaign_run_id": "r"},
        )
        provenance.require_compatible(document, **{"source.source_slice_id": "s"})
        document.pop("workload")
        with self.assertRaises(provenance.ProvenanceError):
            provenance.validate(document)


if __name__ == "__main__":
    unittest.main()
