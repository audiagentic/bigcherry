"""PA36 migration #6 (RD08/1204): patch-local validation producer tests.

Tests the RD08 producer's basic structure and manifest alignment.
"""

from __future__ import annotations

import unittest
from types import SimpleNamespace
from bigcherry.patch import validation_producer as vp
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





class RD08ProducerStructureTests(unittest.TestCase):
    """Structural test: verify the RD08 producer module loads and has
    the expected functions and structure."""

    def test_producer_module_loads(self) -> None:
        import importlib.util
        from pathlib import Path

        producer_path = (
            Path(__file__).resolve().parents[2] / ".." / "patches"
            / "1204_rd08_q6k_mmvq_vdr2" / "validation" / "producer.py"
        )
        spec = importlib.util.spec_from_file_location(
            "rd08_producer", producer_path
        )
        assert spec is not None
        mod = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        import sys
        sys.modules["rd08_producer"] = mod
        spec.loader.exec_module(mod)

        # Verify the key functions exist
        self.assertTrue(hasattr(mod, "run"))
        self.assertTrue(hasattr(mod, "_load_rd08_correctness"))
        self.assertTrue(hasattr(mod, "_run_correctness"))
        self.assertTrue(hasattr(mod, "_run_activation"))
        self.assertTrue(hasattr(mod, "_run_lanes"))

    def test_producer_has_correct_artifact_names(self) -> None:
        import importlib.util
        from pathlib import Path

        producer_path = (
            Path(__file__).resolve().parents[2] / ".." / "patches"
            / "1204_rd08_q6k_mmvq_vdr2" / "validation" / "producer.py"
        )
        spec = importlib.util.spec_from_file_location(
            "rd08_producer", producer_path
        )
        assert spec is not None
        mod = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        import sys
        sys.modules["rd08_producer"] = mod
        spec.loader.exec_module(mod)

        # Verify the producer references the expected artifact names
        src = Path(producer_path).read_text()
        self.assertIn("rd08-correctness.json", src)
        self.assertIn("rd08-performance.json", src)
        self.assertIn("rd08-subject-trace.log", src)
        self.assertIn("rd08-control-trace.log", src)
        self.assertIn("rd08-decode-lane.json", src)
        self.assertIn("rd08-prefill-control.json", src)
        self.assertIn("rd08-activation.json", src)


class _FakeRd08Runtime:
    """Minimal fake runtime for RD08 functional test."""

    def __init__(self, run_dir: Path) -> None:
        self.run_dir = run_dir
        self.fat_targets = SimpleNamespace(targets=("gfx1100",))

    def device_contexts(self, *, device_map):
        from bigcherry.experiment.attestation import ExecutionIdentity
        return (
            vp.ProducerDeviceContext(
                architecture="gfx1100",
                device_index=0,
                execution_identity=ExecutionIdentity(
                    backend="ROCm", architectures=("gfx1100",)
                ),
                env_overrides={},
                env_unset=(),
            ),
        )

    def build_materialized_pair(self, *, control_source, subject_source, targets, primary_target):
        from bigcherry.patch import validation_producer as vp
        return vp.ProducerBuildPair(
            base_revision="fake-rev",
            control_source=control_source,
            subject_source=subject_source,
            control_composition=(),
            subject_composition=(),
            control_bin=self.run_dir / "control-bin",
            subject_bin=self.run_dir / "subject-bin",
            validation_build_identities={
                "control": {"build_id": "ctrl-build"},
                "subject": {"build_id": "subj-build"},
            },
        )

    def run_trace_probe(self, *, binary, model, device, bench_prompt, bench_gen, log_context, disable_fusion=False):
        # Return a log that contains the marker for subject, not for control
        if "subject" in log_context:
            return "BIGCHERRY_PATCH_HIT patch=1204_rd08 path=q6k_mmvq_vdr2"
        return "no marker here"

    def run_paired_llama_benchmark(self, **kwargs):
        # Return a fake outcome with decode and prefill runs
        # Must match the shape expected by lane_effect_from_run()
        run = SimpleNamespace(
            runs=[1.0, 1.1, 1.2],
            stats={
                "geometric_effect_pct": 1.5,
                "ci95_low_pct": 0.5,
                "ci95_high_pct": 2.5,
                "pair_ratios": [1.0, 1.1, 0.9],
            },
        )
        return SimpleNamespace(
            runs={"decode": run, "prefill": run},
            commands=["fake-cmd"],
            raw_logs=["fake-log"],
        )

    def write_artifact(self, *, name, payload):
        import hashlib, json
        path = self.run_dir / "artifacts" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        data = json.dumps(payload, indent=2)
        path.write_text(data)
        return ArtifactRef(
            name=name,
            path=path.relative_to(self.run_dir).as_posix(),
            sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        )

    def write_text_artifact(self, *, name, text):
        import hashlib
        path = self.run_dir / "artifacts" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
        return ArtifactRef(
            name=name,
            path=path.relative_to(self.run_dir).as_posix(),
            sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        )


class _FakeRd08Context:
    """Minimal fake ProducerContext for RD08 functional test."""

    def __init__(self, runtime, workdir: Path) -> None:
        self.runtime = runtime
        self.workdir = workdir
        self.model = workdir / "model.gguf"
        self.device_map = {"gfx1100": (0,)}
        self.fat_targets = runtime.fat_targets
        self.base_revision = "fake-rev"
        self.repo_root = workdir.parent
        self.validation_binaries = {
            "control": {"llama-bench": workdir / "control-bench"},
            "subject": {"llama-bench": workdir / "subject-bench"},
        }
        self.validation_build_identities = {
            "control": {"build_id": "scaffold-ctrl"},
            "subject": {"build_id": "scaffold-subj"},
        }


if __name__ == "__main__":
    unittest.main()
