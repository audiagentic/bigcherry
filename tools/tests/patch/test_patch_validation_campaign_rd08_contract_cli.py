"""PA36 migration #6 (RD08/1204): patch-local validation producer tests.

Tests the RD08 producer's basic structure and manifest alignment.
"""

from __future__ import annotations

import unittest
from pathlib import Path


class TestRD08ProducerStructure(unittest.TestCase):
    """Test the RD08 producer's manifest and structure."""

    def test_producer_toml_exists(self):
        """The producer.toml must exist and be valid TOML."""
        import tomllib

        path = (
            Path(__file__).parent.parent.parent.parent
            / "patches"
            / "1204_rd08_q6k_mmvq_vdr2"
            / "validation"
            / "producer.toml"
        )
        self.assertTrue(path.is_file(), f"producer.toml not found at {path}")

        with open(path, "rb") as f:
            data = tomllib.load(f)

        self.assertIn("producer", data)
        self.assertIn("rd08", data["producer"])

    def test_producer_toml_has_required_fields(self):
        """The producer.toml must have all required fields."""
        import tomllib

        path = (
            Path(__file__).parent.parent.parent.parent
            / "patches"
            / "1204_rd08_q6k_mmvq_vdr2"
            / "validation"
            / "producer.toml"
        )
        with open(path, "rb") as f:
            data = tomllib.load(f)

        producer = data["producer"]["rd08"]
        self.assertEqual(producer["entrypoint"], "producer.py")
        self.assertEqual(producer["callable"], "run")
        self.assertEqual(producer["trace_probe"], "skip")
        self.assertEqual(producer["standard_campaign"], "run")
        self.assertEqual(producer["correctness_evidence_cli"], "forbid")
        self.assertEqual(producer["performance_benchmark_cli"], "forbid")

    def test_producer_toml_has_all_artifacts(self):
        """The producer.toml must list all 7 required artifacts."""
        import tomllib

        path = (
            Path(__file__).parent.parent.parent.parent
            / "patches"
            / "1204_rd08_q6k_mmvq_vdr2"
            / "validation"
            / "producer.toml"
        )
        with open(path, "rb") as f:
            data = tomllib.load(f)

        artifacts = data["producer"]["rd08"]["artifacts"]
        expected = [
            "rd08-decode-lane.json",
            "rd08-prefill-control.json",
            "rd08-correctness.json",
            "rd08-activation.json",
            "rd08-performance.json",
            "rd08-subject-trace.log",
            "rd08-control-trace.log",
        ]
        for artifact in expected:
            self.assertIn(artifact, artifacts)

    def test_producer_py_exists(self):
        """The producer.py must exist."""
        path = (
            Path(__file__).parent.parent.parent.parent
            / "patches"
            / "1204_rd08_q6k_mmvq_vdr2"
            / "validation"
            / "producer.py"
        )
        self.assertTrue(path.is_file(), f"producer.py not found at {path}")

    def test_producer_has_run_function(self):
        """The producer.py must have a run() function."""
        import ast

        path = (
            Path(__file__).parent.parent.parent.parent
            / "patches"
            / "1204_rd08_q6k_mmvq_vdr2"
            / "validation"
            / "producer.py"
        )
        with open(path) as f:
            tree = ast.parse(f.read())

        function_names = [
            node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)
        ]
        self.assertIn("run", function_names)


if __name__ == "__main__":
    unittest.main()
