"""Tests for the RD73 (1233) patch-local validation producer."""

from __future__ import annotations

import unittest
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import sys

# Register the producer module in sys.modules before exec_module
PRODUCER_DIR = Path(
    "patches/1233_rd73_stable_graph_cache_key/validation"
)
PRODUCER_MODULE = "patches_1233_rd73_stable_graph_cache_key_validation_producer"


def _load_producer() -> Any:
    """Load the producer module with the required sys.modules registration."""
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        PRODUCER_MODULE,
        PRODUCER_DIR / "producer.py",
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[PRODUCER_MODULE] = module
    spec.loader.exec_module(module)
    return module


class _FakeVisibility:
    def __init__(self) -> None:
        self.device_ids = (0, 1)
        self.hip_visible_devices = ("0", "1")

    def document(self) -> dict[str, Any]:
        return {"observed": list(self.device_ids), "exact": 2, "satisfied": True}


class _FakeArtifactRef:
    def __init__(self, name: str) -> None:
        self.path = name
        self.sha256 = "fake-sha256"

class _FakeRuntime:
    def __init__(self) -> None:
        self.artifacts: dict[str, Any] = {}

    def write_artifact(self, *, name: str, payload: Any) -> _FakeArtifactRef:
        self.artifacts[name] = payload
        return _FakeArtifactRef(name)

    def write_text_artifact(self, *, name: str, text: str) -> _FakeArtifactRef:
        self.artifacts[name] = text
        return _FakeArtifactRef(name)


class _FakeProducerContext:
    def __init__(self) -> None:
        self.patch_id = "1233_rd73_stable_graph_cache_key"
        self.workdir = Path("/tmp/rd73-test")
        self.model = Path("/tmp/model.gguf")
        self.corpus = Path("/tmp/corpus.txt")
        self.build_env = {"PATH": "/usr/bin", "HIP_VISIBLE_DEVICES": "0,1"}
        self.validation_build_identities = {
            "control": {"digest": "ctrl-digest"},
            "subject": {"digest": "subj-digest"},
        }
        self.validation_binaries = {
            "control": {"llama-server": Path("/tmp/ctrl-server")},
            "subject": {"llama-server": Path("/tmp/subj-server")},
        }
        self.runtime = _FakeRuntime()


class Rd73ProducerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.module = _load_producer()
        self.ctx = _FakeProducerContext()

    def test_missing_model_fails_closed(self) -> None:
        # GPT round-2 MAJOR: the --validation-producer path does not
        # re-impose the legacy parser's model-required check, so the
        # producer must fail fast on a missing model before any hardware
        # use (otherwise ServerRunner would build `-m None`). The model
        # guard is the very first check in run(), so no server mocking is
        # needed.
        self.ctx.model = None
        with self.assertRaises(self.module.ValidationProducerError):
            self.module.run(self.ctx)

    @patch(f"{PRODUCER_MODULE}.require_device_visibility")
    @patch(f"{PRODUCER_MODULE}.AttestedServerSession")
    @patch(f"{PRODUCER_MODULE}.sc")
    @patch(f"{PRODUCER_MODULE}.run_paired_lane")
    def test_performance_artifact_complete_and_benchmark_passes(
        self, mock_paired, mock_sc, mock_session, mock_vis
    ) -> None:
        # GPT round-3 MAJOR: the producer's performance artifact must carry
        # "passed": True (evidence completeness -- the measurement was
        # performed and the artifact is bound) in addition to a non-empty
        # "metrics" dict, because the generic benchmark validator
        # (_builtin_benchmark -> _evidence_pass) requires BOTH. Without
        # "passed", both the required performance and controls checks
        # deterministically FAIL. The contract PASS/FAIL verdict stays
        # solely in the typed promotion gate, independent of this field.
        mock_vis.return_value = _FakeVisibility()
        mock_sc.load_corpus.return_value = (["prompt1", "prompt2"], "sha256")
        mock_sc.SamplingConfig.return_value = MagicMock()
        mock_sc.SessionConfig.return_value = MagicMock()
        mock_sc.HttpTransport.return_value = MagicMock()
        mock_sc.validate_server.return_value = None
        mock_sc.run_request.return_value = {
            "wall_tps": 50.0, "tg128": 100.0, "content": "test",
        }
        mock_session.return_value.__enter__ = MagicMock()
        mock_session.return_value.__exit__ = MagicMock()
        mock_paired_run = MagicMock()
        mock_paired_run.stats = {
            "geometric_effect_pct": 5.0, "ci95_low_pct": -1.0,
            "ci95_high_pct": 11.0, "paired_rounds": 10,
            "pair_ratios": [1.05], "metric": "mtp_wall_tps",
        }
        mock_paired.return_value = mock_paired_run

        def _mock_read_text(self, *args, **kwargs):
            path_str = str(self)
            if "resource-burst" in path_str:
                return "BIGCHERRY_RD73_RESOURCE graph_cache_entries=651\n"
            if "mtp-subject" in path_str:
                return "BIGCHERRY_PATCH_HIT patch=1233_rd73 path=stable_graph_cache_key\n"
            if "mtp-control" in path_str:
                return "no marker here\n"
            return "BIGCHERRY_PATCH_HIT patch=1233_rd73 path=stable_graph_cache_key\n"

        with patch.object(Path, "read_text", _mock_read_text):
            with patch.object(Path, "write_text"):
                self.module.run(self.ctx)

        # (1) The producer's performance artifact now carries passed + metrics.
        perf_payload = self.ctx.runtime.artifacts["rd73-performance.json"]
        self.assertIsInstance(perf_payload.get("metrics"), dict)
        self.assertTrue(perf_payload["metrics"])
        self.assertIs(perf_payload.get("passed"), True)

        # (2) Dispatcher-level proof: the generic benchmark validator
        # (_builtin_benchmark -> _evidence_pass) returns PASS for valid
        # bound evidence. Real Path I/O (the mocks above are out of scope).
        import hashlib
        import json
        import tempfile
        import types

        from bigcherry.patch.validation import CheckSpec, _builtin_benchmark

        with tempfile.TemporaryDirectory() as run_dir_str:
            run_dir = Path(run_dir_str)
            rel = "rd73-performance.json"
            target = run_dir / rel
            target.write_text(json.dumps(perf_payload), encoding="utf-8")
            digest = hashlib.sha256(target.read_bytes()).hexdigest()
            ctx = types.SimpleNamespace(
                performance_evidence={"artifact": {"path": rel, "sha256": digest}},
                run_dir=run_dir,
            )
            spec = CheckSpec(
                check_id="performance", capability="performance",
                validator="benchmark", required=True,
            )
            self.assertEqual(_builtin_benchmark(spec, ctx).status, "pass")

        # (3) Negative control: without "passed", the same artifact FAILs
        # (proving "passed" is load-bearing for the benchmark check, and
        # that the performance artifact's "passed" is evidence
        # completeness -- separate from the promotion gate's verdict).
        with tempfile.TemporaryDirectory() as run_dir_str:
            run_dir = Path(run_dir_str)
            rel = "rd73-performance.json"
            target = run_dir / rel
            no_passed = {
                k: v for k, v in perf_payload.items() if k != "passed"
            }
            target.write_text(json.dumps(no_passed), encoding="utf-8")
            digest = hashlib.sha256(target.read_bytes()).hexdigest()
            ctx = types.SimpleNamespace(
                performance_evidence={"artifact": {"path": rel, "sha256": digest}},
                run_dir=run_dir,
            )
            spec = CheckSpec(
                check_id="performance", capability="performance",
                validator="benchmark", required=True,
            )
            self.assertEqual(_builtin_benchmark(spec, ctx).status, "fail")

    @patch(f"{PRODUCER_MODULE}.require_device_visibility")
    @patch(f"{PRODUCER_MODULE}.AttestedServerSession")
    @patch(f"{PRODUCER_MODULE}.sc")
    @patch(f"{PRODUCER_MODULE}.run_paired_lane")
    def test_producer_success(
        self, mock_paired, mock_sc, mock_session, mock_vis
    ) -> None:
        mock_vis.return_value = _FakeVisibility()

        # Mock the corpus loading
        mock_sc.load_corpus.return_value = (["prompt1", "prompt2"], "sha256")
        mock_sc.SamplingConfig.return_value = MagicMock()
        mock_sc.SessionConfig.return_value = MagicMock()
        mock_sc.HttpTransport.return_value = MagicMock()
        mock_sc.validate_server.return_value = None
        mock_sc.run_request.return_value = {
            "wall_tps": 50.0,
            "tg128": 100.0,
            "content": "test content",
        }

        # Mock the server session context manager
        mock_session.return_value.__enter__ = MagicMock()
        mock_session.return_value.__exit__ = MagicMock()

        # Mock the paired lane
        mock_paired_run = MagicMock()
        mock_paired_run.stats = {
            "geometric_effect_pct": 5.0,
            "ci95_low_pct": -1.0,
            "ci95_high_pct": 11.0,
            "paired_rounds": 10,
            "pair_ratios": [1.05, 0.95, 1.10, 0.90, 1.00],
            "metric": "mtp_wall_tps",
        }
        mock_paired.return_value = mock_paired_run

        # Mock the log file writes
        def _mock_read_text(self, *args, **kwargs):
            path_str = str(self)
            if "resource-burst" in path_str:
                return "BIGCHERRY_RD73_RESOURCE graph_cache_entries=651\n"
            if "mtp-subject" in path_str:
                return "BIGCHERRY_PATCH_HIT patch=1233_rd73 path=stable_graph_cache_key\n"
            if "mtp-control" in path_str:
                return "no marker here\n"
            return "BIGCHERRY_PATCH_HIT patch=1233_rd73 path=stable_graph_cache_key\n"
        
        with patch.object(Path, "read_text", _mock_read_text):
            with patch.object(Path, "write_text"):
                result = self.module.run(self.ctx)

        self.assertEqual(result.correctness["disposition"], "passed")
        self.assertEqual(
            result.activation_evidence.status, "executed"
        )
        self.assertEqual(len(result.emitted_artifacts), 8)

    @patch(f"{PRODUCER_MODULE}.require_device_visibility")
    @patch(f"{PRODUCER_MODULE}.AttestedServerSession")
    @patch(f"{PRODUCER_MODULE}.sc")
    @patch(f"{PRODUCER_MODULE}.run_paired_lane")
    def test_device_visibility_exact_count(
        self, mock_paired, mock_sc, mock_session, mock_vis
    ) -> None:
        mock_vis.return_value = _FakeVisibility()
        mock_sc.load_corpus.return_value = (["prompt1"], "sha256")
        mock_sc.SamplingConfig.return_value = MagicMock()
        mock_sc.SessionConfig.return_value = MagicMock()
        mock_sc.HttpTransport.return_value = MagicMock()
        mock_sc.validate_server.return_value = None
        mock_sc.run_request.return_value = {
            "wall_tps": 50.0, "tg128": 100.0, "content": "test",
        }
        mock_session.return_value.__enter__ = MagicMock()
        mock_session.return_value.__exit__ = MagicMock()
        mock_paired_run = MagicMock()
        mock_paired_run.stats = {
            "geometric_effect_pct": 5.0, "ci95_low_pct": -1.0,
            "ci95_high_pct": 11.0, "paired_rounds": 10,
            "pair_ratios": [], "metric": "mtp_wall_tps",
        }
        mock_paired.return_value = mock_paired_run
        def _mock_read_text(self, *args, **kwargs):
            path_str = str(self)
            if "resource-burst" in path_str:
                return "BIGCHERRY_RD73_RESOURCE graph_cache_entries=651\n"
            if "mtp-subject" in path_str:
                return "BIGCHERRY_PATCH_HIT patch=1233_rd73 path=stable_graph_cache_key\n"
            if "mtp-control" in path_str:
                return "no marker here\n"
            return "BIGCHERRY_PATCH_HIT patch=1233_rd73 path=stable_graph_cache_key\n"
        with patch.object(Path, "read_text", _mock_read_text):
            with patch.object(Path, "write_text"):
                self.module.run(self.ctx)
        mock_vis.assert_called_once()
        call_kwargs = mock_vis.call_args[1]
        self.assertEqual(call_kwargs["exact_count"], 2)
        self.assertIn("env", call_kwargs)

    @patch(f"{PRODUCER_MODULE}.require_device_visibility")
    @patch(f"{PRODUCER_MODULE}.AttestedServerSession")
    @patch(f"{PRODUCER_MODULE}.sc")
    @patch(f"{PRODUCER_MODULE}.run_paired_lane")
    def test_promotion_resource_results(
        self, mock_paired, mock_sc, mock_session, mock_vis
    ) -> None:
        mock_vis.return_value = _FakeVisibility()

        mock_sc.load_corpus.return_value = (["prompt1"], "sha256")
        mock_sc.SamplingConfig.return_value = MagicMock()
        mock_sc.SessionConfig.return_value = MagicMock()
        mock_sc.HttpTransport.return_value = MagicMock()
        mock_sc.validate_server.return_value = None
        mock_sc.run_request.return_value = {
            "wall_tps": 50.0,
            "tg128": 100.0,
            "content": "test",
        }
        mock_session.return_value.__enter__ = MagicMock()
        mock_session.return_value.__exit__ = MagicMock()

        mock_paired_run = MagicMock()
        mock_paired_run.stats = {
            "geometric_effect_pct": 5.0,
            "ci95_low_pct": -1.0,
            "ci95_high_pct": 11.0,
            "paired_rounds": 10,
            "pair_ratios": [],
            "metric": "mtp_wall_tps",
        }
        mock_paired.return_value = mock_paired_run

        # Mock the resource burst log to contain telemetry
        # The read_text mock needs to return different values for different files
        def _mock_read_text(self, *args, **kwargs):
            path_str = str(self)
            if "resource-burst" in path_str:
                return "BIGCHERRY_RD73_RESOURCE graph_cache_entries=651\n"
            if "mtp-subject" in path_str:
                return "BIGCHERRY_PATCH_HIT patch=1233_rd73 path=stable_graph_cache_key\n"
            if "mtp-control" in path_str:
                return "no marker here\n"
            return "BIGCHERRY_PATCH_HIT patch=1233_rd73 path=stable_graph_cache_key\n"
        
        with patch.object(Path, "read_text", _mock_read_text):
            with patch.object(Path, "write_text"):
                result = self.module.run(self.ctx)

        self.assertIn(
            "RD73-STABLE-GRAPH-CACHE-KEY",
            result.promotion_resource_results,
        )
        resources = result.promotion_resource_results[
            "RD73-STABLE-GRAPH-CACHE-KEY"
        ]
        self.assertEqual(len(resources), 1)
        self.assertEqual(resources[0].metric, "graph_cache_entries")
        self.assertEqual(resources[0].subject_value, 651.0)

    @patch(f"{PRODUCER_MODULE}.require_device_visibility")
    @patch(f"{PRODUCER_MODULE}.AttestedServerSession")
    @patch(f"{PRODUCER_MODULE}.sc")
    @patch(f"{PRODUCER_MODULE}.run_paired_lane")
    def test_contract_correctness_results(
        self, mock_paired, mock_sc, mock_session, mock_vis
    ) -> None:
        mock_vis.return_value = _FakeVisibility()

        mock_sc.load_corpus.return_value = (["prompt1"], "sha256")
        mock_sc.SamplingConfig.return_value = MagicMock()
        mock_sc.SessionConfig.return_value = MagicMock()
        mock_sc.HttpTransport.return_value = MagicMock()
        mock_sc.validate_server.return_value = None
        # Different content for subject vs control (bit-identical fails)
        # Use a callable that returns different content based on call count
        call_count = {"n": 0}
        def _mock_run_request(*args, **kwargs):
            call_count["n"] += 1
            # Odd calls = subject, even calls = control (alternating)
            if call_count["n"] % 2 == 1:
                return {"wall_tps": 50.0, "tg128": 100.0, "content": "subject content"}
            return {"wall_tps": 50.0, "tg128": 100.0, "content": "control content"}
        mock_sc.run_request.side_effect = _mock_run_request
        mock_session.return_value.__enter__ = MagicMock()
        mock_session.return_value.__exit__ = MagicMock()

        mock_paired_run = MagicMock()
        mock_paired_run.stats = {
            "geometric_effect_pct": 5.0,
            "ci95_low_pct": -1.0,
            "ci95_high_pct": 11.0,
            "paired_rounds": 10,
            "pair_ratios": [],
            "metric": "mtp_wall_tps",
        }
        mock_paired.return_value = mock_paired_run

        def _mock_read_text(self, *args, **kwargs):
            path_str = str(self)
            if "resource-burst" in path_str:
                return "BIGCHERRY_RD73_RESOURCE graph_cache_entries=651\n"
            if "mtp-subject" in path_str:
                return "BIGCHERRY_PATCH_HIT patch=1233_rd73 path=stable_graph_cache_key\n"
            if "mtp-control" in path_str:
                return "no marker here\n"
            return "BIGCHERRY_PATCH_HIT patch=1233_rd73 path=stable_graph_cache_key\n"
        
        with patch.object(Path, "read_text", _mock_read_text):
            with patch.object(Path, "write_text"):
                result = self.module.run(self.ctx)

        self.assertEqual(len(result.contract_correctness_results), 1)
        self.assertEqual(
            result.contract_correctness_results[0].check, "bit_identical"
        )
        # Different content means bit-identical should fail
        self.assertFalse(result.contract_correctness_results[0].passed)


if __name__ == "__main__":
    unittest.main()
