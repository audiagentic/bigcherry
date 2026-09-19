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


class _FakeProducerContext:
    def __init__(self) -> None:
        self.patch_id = "1233_rd73_stable_graph_cache_key"
        self.workdir = Path("/tmp/rd73-test")
        self.model = Path("/tmp/model.gguf")
        self.build_env = {"PATH": "/usr/bin"}
        self.validation_build_identities = {
            "control": {"digest": "ctrl-digest"},
            "subject": {"digest": "subj-digest"},
        }
        self.validation_binaries = {
            "control": {"llama-server": Path("/tmp/ctrl-server")},
            "subject": {"llama-server": Path("/tmp/subj-server")},
        }
        self.corpus = None


class Rd73ProducerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.module = _load_producer()
        self.ctx = _FakeProducerContext()

    @patch("patches_1233_rd73_stable_graph_cache_key_validation_producer.require_device_visibility")
    @patch("patches_1233_rd73_stable_graph_cache_key_validation_producer.subprocess.run")
    def test_producer_success(self, mock_run, mock_vis) -> None:
        mock_vis.return_value = _FakeVisibility()
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout="tg128 100.0\nmtp_wall_tps 50.0\n",
            stderr="",
        )
        result = self.module.run(self.ctx)
        self.assertEqual(
            result.correctness["disposition"], "passed"
        )
        self.assertEqual(
            result.activation_evidence["disposition"], "activation-verified"
        )
        self.assertEqual(len(result.emitted_artifacts), 8)

    @patch("patches_1233_rd73_stable_graph_cache_key_validation_producer.require_device_visibility")
    @patch("patches_1233_rd73_stable_graph_cache_key_validation_producer.subprocess.run")
    def test_device_visibility_exact_count(self, mock_run, mock_vis) -> None:
        mock_vis.return_value = _FakeVisibility()
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout="tg128 100.0\nmtp_wall_tps 50.0\n",
            stderr="",
        )
        # Verify that exact_count=2 is passed
        self.module.run(self.ctx)
        mock_vis.assert_called_once()
        call_kwargs = mock_vis.call_args[1]
        self.assertEqual(call_kwargs["exact_count"], 2)

    @patch("patches_1233_rd73_stable_graph_cache_key_validation_producer.require_device_visibility")
    @patch("patches_1233_rd73_stable_graph_cache_key_validation_producer.subprocess.run")
    def test_promotion_resource_results(self, mock_run, mock_vis) -> None:
        mock_vis.return_value = _FakeVisibility()
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout="BIGCHERRY_RD73_RESOURCE graph_cache_entries=651\n",
            stderr="",
        )
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

    @patch("patches_1233_rd73_stable_graph_cache_key_validation_producer.require_device_visibility")
    @patch("patches_1233_rd73_stable_graph_cache_key_validation_producer.subprocess.run")
    def test_contract_correctness_results(self, mock_run, mock_vis) -> None:
        mock_vis.return_value = _FakeVisibility()
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout="tg128 100.0\n",
            stderr="",
        )
        result = self.module.run(self.ctx)
        self.assertEqual(len(result.contract_correctness_results), 1)
        self.assertEqual(
            result.contract_correctness_results[0].check, "bit_identical"
        )


if __name__ == "__main__":
    unittest.main()
