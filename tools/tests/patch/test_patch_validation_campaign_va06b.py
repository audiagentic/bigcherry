"""VA06 next slice: RD73's paired control/subject mtp_verify performance
lane over a real llama-server HTTP harness (run_rd73_mtp_server_lane()).
GPT scoping (session ses_89a3ef2b02b94469): reuse ServerRunner +
server_completion.py's real request/metrics machinery; target metric is
client-measured wall_tps, not the server's self-reported predicted_tps.
Hardware-free: ServerRunner and server_completion's transport/request
primitives are faked, exercising only this adapter's own control flow.
"""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from bigcherry.bench import server_completion as sc  # noqa: E402
from bigcherry.patch import validation_campaign as vc  # noqa: E402
from bigcherry.tuning import server_runner as sr  # noqa: E402


class _FakeServerRunner:
    """Stands in for tuning.server_runner.ServerRunner -- no real process,
    no real HTTP -- so this adapter's own orchestration logic (warmup vs.
    measured, arm routing, fail-closed on missing wall_tps) can be tested
    without a GPU."""

    instances: list["_FakeServerRunner"] = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.entered = False
        self.exited = False
        _FakeServerRunner.instances.append(self)

    def __enter__(self):
        self.entered = True
        return self

    def __exit__(self, exc_type, exc, tb):
        self.exited = True
        return False


class _FakeTransport:
    def __init__(self, base_url: str):
        self.base_url = base_url


def _fake_corpus():
    prompts = [
        sc.CorpusPrompt(id="p1", seed=1, category="prose", prompt="hi"),
        sc.CorpusPrompt(id="p2", seed=2, category="prose", prompt="ho"),
    ]
    return prompts, "corpus-sha"


class RunRd73MtpServerLaneTests(unittest.TestCase):
    def setUp(self) -> None:
        _FakeServerRunner.instances = []
        self._tmp = tempfile.TemporaryDirectory()
        self.run_dir = Path(self._tmp.name)
        self.control_binary = self.run_dir / "control-server"
        self.subject_binary = self.run_dir / "subject-server"
        self.model = self.run_dir / "model.gguf"
        self.corpus_path = self.run_dir / "corpus.jsonl"
        self.corpus_path.write_text("", encoding="utf-8")

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def _patches(self, wall_tps_by_arm):
        """wall_tps_by_arm: {"control": [values...], "subject": [values...]}
        -- one value consumed per real request to that arm, in order."""
        counters = {"control": 0, "subject": 0}

        def _fake_run_request(transport, prompt, config, *, pass_number, order_index):
            arm = "control" if "control" in config.session_id else "subject"
            values = wall_tps_by_arm[arm]
            index = counters[arm]
            counters[arm] += 1
            wall_tps = values[index]
            return {
                "tokens_predicted": 128, "wall_s": 1.0, "wall_tps": wall_tps,
                "content": f"{arm}-{index}",
            }

        return [
            mock.patch.object(sr, "ServerRunner", _FakeServerRunner),
            mock.patch.object(sc, "load_corpus", return_value=_fake_corpus()),
            mock.patch.object(sc, "HttpTransport", _FakeTransport),
            mock.patch.object(sc, "validate_server", return_value=None),
            mock.patch.object(sc, "run_request", side_effect=_fake_run_request),
        ]

    def _run(self, wall_tps_by_arm, **kwargs):
        patches = self._patches(wall_tps_by_arm)
        for p in patches:
            p.start()
        self.addCleanup(lambda: [p.stop() for p in patches])
        return vc.run_rd73_mtp_server_lane(
            control_binary=self.control_binary, subject_binary=self.subject_binary,
            model=self.model, corpus_path=self.corpus_path, run_dir=self.run_dir,
            warmup_pairs=1, measured_pairs=3, **kwargs,
        )

    def test_subject_faster_than_control_yields_positive_effect(self) -> None:
        result = self._run({
            "control": [10.0] * 4,  # 1 warmup + 3 measured
            "subject": [20.0] * 4,
        })
        self.assertGreater(result["effect"].geometric_effect_pct, 0.0)

    def test_warmup_requests_are_not_fed_into_statistics(self) -> None:
        # If the warmup pair's values (999.0) leaked into the paired stats,
        # the mean would be pulled far from the measured 10.0/20.0 values.
        result = self._run({
            "control": [999.0, 10.0, 10.0, 10.0],
            "subject": [999.0, 20.0, 20.0, 20.0],
        })
        self.assertEqual(len(result["control_requests"]), 4)
        self.assertEqual(len(result["subject_requests"]), 4)
        self.assertAlmostEqual(result["effect"].geometric_effect_pct, 100.0, delta=1.0)

    def test_both_servers_launched_and_torn_down(self) -> None:
        self._run({"control": [10.0] * 4, "subject": [20.0] * 4})
        self.assertEqual(len(_FakeServerRunner.instances), 2)
        for instance in _FakeServerRunner.instances:
            self.assertTrue(instance.entered)
            self.assertTrue(instance.exited)

    def test_missing_wall_tps_fails_closed(self) -> None:
        with self.assertRaises(vc.PatchCampaignError):
            self._run({"control": [None] * 4, "subject": [20.0] * 4})

    def test_content_retained_for_correctness_lane(self) -> None:
        result = self._run({"control": [10.0] * 4, "subject": [20.0] * 4})
        self.assertEqual(
            [r["content"] for r in result["control_requests"]],
            ["control-0", "control-1", "control-2", "control-3"],
        )

    def test_artifact_written(self) -> None:
        result = self._run({"control": [10.0] * 4, "subject": [20.0] * 4})
        self.assertTrue((self.run_dir / "artifacts" / "rd73-mtp-lane.json").is_file())
        self.assertIsNotNone(result["artifact"])

    def test_server_launched_with_real_llama_server_flags(self) -> None:
        # Regression coverage: an earlier draft invented "--spec-n-max"/
        # "--spec-draft-k"/"--spec-draft-v", none of which are real
        # llama-server flags (verified against vendor/llama.cpp's
        # common/arg.cpp before the real Brutus hardware run) -- the real
        # flag is --spec-draft-n-max, and -sm tensor is required for this
        # 27B model on 2x gfx1100.
        self._run({"control": [10.0] * 4, "subject": [20.0] * 4})
        for instance in _FakeServerRunner.instances:
            extra_args = instance.kwargs["extra_args"]
            self.assertIn("--spec-draft-n-max", extra_args)
            self.assertNotIn("--spec-n-max", extra_args)
            self.assertNotIn("--spec-draft-k", extra_args)
            self.assertNotIn("--spec-draft-v", extra_args)
            self.assertIn("-sm", extra_args)
            self.assertEqual(extra_args[extra_args.index("-sm") + 1], "tensor")
            # Real hardware finding: llama.cpp's automatic device-memory
            # fit feature (default on) raises "llama_params_fit is not
            # implemented for SPLIT_MODE_TENSOR" and aborts -- must be
            # explicitly disabled alongside -sm tensor.
            self.assertIn("--fit", extra_args)
            self.assertEqual(extra_args[extra_args.index("--fit") + 1], "off")


if __name__ == "__main__":
    unittest.main()
