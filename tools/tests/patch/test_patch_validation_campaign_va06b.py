"""VA06 next slice: RD73's paired control/subject mtp_verify performance
lane over a real llama-server HTTP harness.

PA36 RD73 legacy compatibility retirement: the lane moved from shared
validation_campaign.py (run_rd73_mtp_server_lane) into RD73's producer
(patches/1233_rd73_stable_graph_cache_key/validation/producer.py,
_run_mtp_server_lane). This test now exercises the producer-side lane
directly.

GPT scoping (session ses_89a3ef2b02b94469): reuse ServerRunner +
server_completion.py's real request/metrics machinery; target metric is
client-measured wall_tps, not the server's self-reported predicted_tps.
Hardware-free: ServerRunner and server_completion's transport/request
primitives are faked, exercising only the lane's own control flow.

Note: the producer's _run_mtp_server_lane returns a TUPLE
(lane_effect, subject_records, control_records, subject_log, control_log)
and does NOT write the rd73-mtp-lane.json artifact (that is written by
the producer's run(); covered by test_patch_validation_campaign_rd73_
contract_cli.py). The historical test_artifact_written assertion is
therefore retired with this migration.
"""

from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from bigcherry.bench import server_completion as sc  # noqa: E402
from bigcherry.experiment import attestation as att  # noqa: E402
from bigcherry.experiment import server_execution as se  # noqa: E402

PRODUCER_DIR = Path(
    "patches/1233_rd73_stable_graph_cache_key/validation"
)
PRODUCER_MODULE = "patches_1233_rd73_stable_graph_cache_key_validation_producer_va06b"


def _load_producer() -> Any:
    """Load the producer module with the required sys.modules registration."""
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


# The producer lane drives each arm through AttestedServerSession; this
# fixture's ExecutionIdentity is arbitrary but non-empty (see
# ExecutionIdentity.__post_init__) and is only compared against the
# also-fixed fake attestation below -- real attestation-content parsing
# is covered by test_attested_server_session.py and
# test_execution_attestation.py, not re-tested here.
_FAKE_EXPECTED_EXECUTION = att.ExecutionIdentity(backend="ROCm", architectures=("gfx1100", "gfx1100"))


class _FakeServerRunner:
    """Stands in for tuning.server_runner.ServerRunner -- no real process,
    no real HTTP -- so the lane's own orchestration logic (warmup vs.
    measured, arm routing, fail-closed on missing wall_tps) can be tested
    without a GPU. Exposes launch()/wait_healthy()/shutdown() matching what
    AttestedServerSession actually calls (it does not use ``with runner:``
    on the wrapped ServerRunner directly)."""

    instances: list["_FakeServerRunner"] = []

    concurrent_entries: list[int] = []
    _live_count = 0

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.entered = False
        self.exited = False
        self.host = kwargs.get("host", "127.0.0.1")
        self.port = kwargs.get("port", 0)
        # Real ServerRunner always creates its log file on launch (stdout
        # redirect); the lane reads per-request log files back (sequential
        # single-request-per-launch restart, a real hardware fix for
        # control+subject VRAM contention) so the fake must produce one too.
        log_path = kwargs.get("log_path")
        if log_path is not None:
            Path(log_path).parent.mkdir(parents=True, exist_ok=True)
            Path(log_path).write_text("", encoding="utf-8")
        _FakeServerRunner.instances.append(self)

    def launch(self) -> None:
        self.entered = True
        _FakeServerRunner._live_count += 1
        _FakeServerRunner.concurrent_entries.append(_FakeServerRunner._live_count)

    def wait_healthy(self, timeout_s: int = 180) -> None:
        pass

    def shutdown(self, timeout_s: int = 90):
        self.exited = True
        _FakeServerRunner._live_count -= 1
        return None


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
    @classmethod
    def setUpClass(cls) -> None:
        cls.producer = _load_producer()

    def setUp(self) -> None:
        _FakeServerRunner.instances = []
        _FakeServerRunner.concurrent_entries = []
        _FakeServerRunner._live_count = 0
        self._tmp = tempfile.TemporaryDirectory()
        self.run_dir = Path(self._tmp.name)
        self.control_binary = self.run_dir / "control-server"
        self.subject_binary = self.run_dir / "subject-server"
        self.model = self.run_dir / "model.gguf"
        self.corpus_path = self.run_dir / "corpus.jsonl"
        self.corpus_path.write_text("", encoding="utf-8")

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def _make_ctx(self) -> Any:
        class _Ctx:
            pass

        ctx = _Ctx()
        ctx.model = self.model
        ctx.corpus = self.corpus_path
        ctx.workdir = self.run_dir
        return ctx

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
            # The lane goes through AttestedServerSession, which imports
            # ServerRunner into its OWN module namespace -- patching
            # sr.ServerRunner (the source module attribute) would not reach
            # that already-bound name, so the fake must be installed on
            # server_execution's namespace.
            mock.patch.object(se, "ServerRunner", _FakeServerRunner),
            # Real attestation-content parsing is tested elsewhere
            # (test_attested_server_session.py, test_execution_attestation.py);
            # here it is fixed to always match, so this file's own tests stay
            # focused on the warmup/measured/arm-routing orchestration logic.
            mock.patch.object(
                se, "parse_llama_server_attestation",
                return_value=att.ExecutionAttestation(
                    backend="ROCm",
                    devices=(
                        att.ObservedDevice(architecture="gfx1100", locator=None),
                        att.ObservedDevice(architecture="gfx1100", locator=None),
                    ),
                ),
            ),
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
        return self.producer._run_mtp_server_lane(
            ctx=self._make_ctx(),
            control_binary=self.control_binary,
            subject_binary=self.subject_binary,
            expected_execution=_FAKE_EXPECTED_EXECUTION,
            selector_env={},
            warmup_pairs=1, measured_pairs=3, **kwargs,
        )

    def test_subject_faster_than_control_yields_positive_effect(self) -> None:
        lane_effect, _subject, _control, _subject_log, _control_log = self._run({
            "control": [10.0] * 4,  # 1 warmup + 3 measured
            "subject": [20.0] * 4,
        })
        self.assertGreater(lane_effect.geometric_effect_pct, 0.0)

    def test_warmup_requests_are_not_fed_into_statistics(self) -> None:
        # If the warmup pair's values (999.0) leaked into the paired stats,
        # the mean would be pulled far from the measured 10.0/20.0 values.
        lane_effect, subject_requests, control_requests, _sl, _cl = self._run({
            "control": [999.0, 10.0, 10.0, 10.0],
            "subject": [999.0, 20.0, 20.0, 20.0],
        })
        self.assertEqual(len(control_requests), 4)
        self.assertEqual(len(subject_requests), 4)
        self.assertAlmostEqual(lane_effect.geometric_effect_pct, 100.0, delta=1.0)

    def test_one_server_launched_per_single_request_both_arms(self) -> None:
        # Real hardware fix (2026-09-01): control and subject servers
        # must never run concurrently for this model (each needs
        # ~13GB/GPU, two copies exceed the 24.5GB/GPU cards) -- one fresh
        # server per single request, alternating arms. warmup_pairs=1 +
        # measured_pairs=3 -> 4 pairs x 2 arms = 8 server launches.
        self._run({"control": [10.0] * 4, "subject": [20.0] * 4})
        self.assertEqual(len(_FakeServerRunner.instances), 8)
        for instance in _FakeServerRunner.instances:
            self.assertTrue(instance.entered)
            self.assertTrue(instance.exited)

    def test_servers_never_run_concurrently(self) -> None:
        # Real hardware regression guard: control+subject running at the
        # same time OOMs (each needs ~13GB/GPU, cards are 24.5GB) --
        # at most one server may be "live" (entered but not yet exited)
        # at any point during this lane.
        self._run({"control": [10.0] * 4, "subject": [20.0] * 4})
        self.assertTrue(_FakeServerRunner.concurrent_entries)
        self.assertEqual(max(_FakeServerRunner.concurrent_entries), 1)

    def test_missing_wall_tps_fails_closed(self) -> None:
        with self.assertRaises(self.producer.ValidationProducerError):
            self._run({"control": [None] * 4, "subject": [20.0] * 4})

    def test_content_retained_for_correctness_lane(self) -> None:
        _le, _subject, control_requests, _sl, _cl = self._run({
            "control": [10.0] * 4, "subject": [20.0] * 4,
        })
        self.assertEqual(
            [r["content"] for r in control_requests],
            ["control-0", "control-1", "control-2", "control-3"],
        )

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

    def test_servers_always_carry_rd73_activation_and_resource_env(self) -> None:
        # User redirect (2026-09-01): activation/resource evidence is now
        # read from these SAME servers' own log files rather than a
        # separate llama-bench probe -- both markers must always be
        # enabled here.
        self._run({"control": [10.0] * 4, "subject": [20.0] * 4})
        for instance in _FakeServerRunner.instances:
            env_overrides = instance.kwargs["env_overrides"]
            self.assertEqual(env_overrides.get("BIGCHERRY_PATCH_TRACE"), "1")
            self.assertEqual(env_overrides.get("BIGCHERRY_RD73_RESOURCE_TRACE"), "1")

    def test_returns_server_log_paths_for_activation_and_resource_evidence(self) -> None:
        _le, _subject, _control, subject_log, control_log = self._run({
            "control": [10.0] * 4, "subject": [20.0] * 4,
        })
        self.assertIsInstance(subject_log, Path)
        self.assertIsInstance(control_log, Path)
        self.assertTrue(subject_log.is_file())
        self.assertTrue(control_log.is_file())


if __name__ == "__main__":
    unittest.main()
