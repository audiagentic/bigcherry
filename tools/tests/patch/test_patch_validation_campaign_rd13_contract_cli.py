"""PA36 migration #3 (dev-gpt-agent req_110d0729beb44d8b): the RD13
dedicated CLI path (--run-rd13-contract, run_rd13_backend_reference_check,
run_rd13_ppl_check, _load_rd13_correctness_module, the rd13_correctness.py
module) was DELETED from shared code and replaced by the generic
standard_campaign="run" producer path
(--validation-producer 1206_rd13_mul_mat_add_view_fusion/rd13).

This file covers the replaced path at two levels:
  * source-layout regression guards (the deleted surface must not come
    back, and the producer manifest pins the migration policy),
  * the end-to-end dispatcher (_run_validation_producer) driven against
    the REAL 1206 patch/descriptor/plan with the expensive producer
    boundary faked (a lightweight stand-in producer; the real
    full-vocabulary measurement semantics are covered by
    test_patch_validation_campaign_rd13_backend_reference.py) -- the
    NEW trace_probe="run" dispatcher behavior: the dispatcher itself
    runs the scaffold's generic two-probe activation probe, writes
    activation.json, and fails closed when a producer supplies its own
    activation/trace evidence.

Producer-level measurement semantics (full-vocabulary logprob
comparison, artifact shape, env sanitization) are covered by
test_patch_validation_campaign_rd13_backend_reference.py.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import cast
from unittest import mock

TOOLS_ROOT = Path(__file__).resolve().parents[2]
if str(TOOLS_ROOT) not in sys.path:
    sys.path.insert(0, str(TOOLS_ROOT))

from bigcherry.experiment import contract as experiment_contract  # noqa: E402
from bigcherry.experiment.attestation import ExecutionIdentity  # noqa: E402
from bigcherry.patch import activation as patch_activation  # noqa: E402
from bigcherry.patch import evidence as patch_evidence  # noqa: E402
from bigcherry.patch import source as psi  # noqa: E402
from bigcherry.patch import validation_campaign as vc  # noqa: E402
from bigcherry.patch import validation_producer as vp  # noqa: E402
from bigcherry.patch.validation import ArtifactRef  # noqa: E402

PATCH_ID = "1206_rd13_mul_mat_add_view_fusion"
CAMPAIGN_SRC = (
    TOOLS_ROOT / "bigcherry" / "patch" / "validation_campaign.py"
).read_text(encoding="utf-8")
PRODUCER_SRC = (
    TOOLS_ROOT.parent / "patches" / PATCH_ID / "validation" / "producer.py"
).read_text(encoding="utf-8")
EXPECTED_ARTIFACT_NAMES = frozenset(
    {
        "rd13-backend-reference.json",
        "rd13-performance.json",
        "rd13-subject-trace.log",
        "rd13-control-trace.log",
    }
)
_FAKE_ARTIFACT_NAMES = frozenset({"rd13-backend-reference.json"})


def cfg_pinned() -> str:
    from bigcherry.core import config as campaign_config
    from bigcherry.core import paths as bc_paths

    return campaign_config.load(bc_paths.RECIPES).pinned


# ---------------------------------------------------------- layout guards


class Rd13DedicatedPathDeletionTests(unittest.TestCase):
    def test_dedicated_cli_surface_is_gone(self) -> None:
        for needle in (
            "run_rd13_backend_reference_check",
            "run_rd13_ppl_check",
            "_load_rd13_correctness_module",
            "_rd13_stream_completion_rows",
            "_rd13_dense_logprobs",
            "_rd13_canonical_bytes",
            "run-rd13-contract",
            "rd13_correctness",
        ):
            self.assertNotIn(needle, CAMPAIGN_SRC, needle)

    def test_exclusion_tuples_do_not_carry_the_dead_flags(self) -> None:
        for line in CAMPAIGN_SRC.split("\n"):
            if '"run_rd58_state_restore", "run_rd73_contract"' in line:
                self.assertNotIn("run_rd13", line)
            if (
                "args.run_rd08_contract or " in line
                and "args.run_rd58_state_restore" in line
            ):
                self.assertNotIn("run_rd13", line)

    def test_producer_manifest_pins_the_migration_policy(self) -> None:
        selection = vp.resolve_producer(
            patch_dir=TOOLS_ROOT.parent / "patches" / PATCH_ID,
            producer_id="rd13",
        )
        self.assertEqual(selection.spec.patch_id, PATCH_ID)
        self.assertEqual(selection.spec.standard_campaign, "run")
        # The producer owns the real scaffold-pair trigger probes so the
        # promotion channel can bind truthful TriggerEvidence.
        self.assertEqual(selection.spec.trace_probe, "skip")
        self.assertEqual(selection.spec.correctness_evidence_cli, "forbid")
        self.assertEqual(selection.spec.performance_benchmark_cli, "forbid")
        self.assertEqual(selection.spec.artifact_names, EXPECTED_ARTIFACT_NAMES)

    def test_producer_owns_scaffold_trigger_and_performance_evidence(self) -> None:
        self.assertIn("run_trace_probe(", PRODUCER_SRC)
        self.assertIn('"rd13-performance.json"', PRODUCER_SRC)
        self.assertIn("promotion_lane_effects", PRODUCER_SRC)
        self.assertIn("promotion_trigger_evidence", PRODUCER_SRC)

    def test_dispatcher_trace_probe_block_runs_the_scaffold_probe(self) -> None:
        # The dispatcher (not the producer) runs the probe, against the
        # SCAFFOLD subject llama-bench, and fails closed when the
        # producer supplies its own evidence.
        core_src = inspect.getsource(vc._run_producer_trace_probes)
        self.assertIn("run_trace_activation_probes(", core_src)
        self.assertIn("result.activation_evidence is not None", core_src)
        self.assertIn("result.trace_evidence is not None", core_src)
        self.assertIn('validation_binaries.get("subject")', core_src)
        self.assertIn("llama-bench", core_src)
        self.assertIn("exactly one trace-marker activation check", core_src)


# ------------------------------------------------------ dispatcher (end-to-end)


class _FakeBuildEvidence:
    def __init__(self, role: str) -> None:
        self.role = role

    @property
    def effective_build_id(self) -> str:
        return f"{self.role}-build-id"

    @property
    def effective_configure(self) -> dict[str, str]:
        return {"CMAKE_BUILD_TYPE": "Release", "GGML_HIP": "ON"}

    @property
    def verification(self) -> SimpleNamespace:
        return SimpleNamespace(to_dict=lambda: {"verified": True, "role": self.role})

    @property
    def runtime_artifacts(self) -> dict[str, object]:
        digest = hashlib.sha256(self.role.encode("utf-8")).hexdigest()
        return {"main": digest}

    def campaign_identity(self) -> dict[str, object]:
        return {
            "effective_build_id": self.effective_build_id,
            "compile_verification_id": f"{self.role}-verify-id",
            "compile_commands_digest": f"{self.role}-cc-digest",
            "hip_compile_commands_digest": f"{self.role}-hip-cc-digest",
            "runtime_bundle_hash": f"{self.role}-bundle-hash",
            "runtime_artifacts": dict(sorted(self.runtime_artifacts.items())),
        }


class _FakeScaffold:
    def __init__(self, base_dir: Path) -> None:
        self.base_revision = "b" * 40
        self.control_composition = ()
        self.subject_composition = ((PATCH_ID, "c1"),)
        self.control_source = base_dir / "trees" / "control"
        self.subject_source = base_dir / "trees" / "subject"
        self.stock_source = base_dir / "trees" / "stock"
        for source in (self.control_source, self.subject_source, self.stock_source):
            source.mkdir(parents=True, exist_ok=True)
        self.control_idempotent = True
        self.subject_idempotent = True
        self.control_bin = base_dir / "bin" / "control"
        self.validation_subject_bin = base_dir / "bin" / "validation-subject"
        self.tune_build_evidence = _FakeBuildEvidence("tune")
        self.replay_build_evidence = _FakeBuildEvidence("replay")
        self.stock_build_evidence = _FakeBuildEvidence("stock")
        self.control_build_evidence = _FakeBuildEvidence("control")
        self.validation_subject_build_evidence = _FakeBuildEvidence(
            "validation-subject",
        )

    @property
    def campaign_build_identities(self) -> dict[str, dict[str, object]]:
        return {
            role: getattr(self, f"{role}_build_evidence").campaign_identity()
            for role in ("tune", "replay", "stock")
        }

    @property
    def scaffold_validation_build_identities(self) -> dict[str, dict[str, object]]:
        return {
            "control": self.control_build_evidence.campaign_identity(),
            "subject": self.validation_subject_build_evidence.campaign_identity(),
        }


class _FakeDispatcherRuntime:
    """Stand-in for vc.CampaignProducerRuntime: one fake llama-server
    pair (fat multi-arch, parity asserted), one device for the run
    architecture, real artifact files under run_dir."""

    def __init__(
        self,
        *,
        repo_root,
        patch_id,
        base_revision,
        workdir,
        hip_path,
        fat_targets,
        run_dir,
    ) -> None:
        self.run_dir = run_dir
        self.fat_targets = fat_targets
        self.pair = vp.ProducerBuildPair(
            base_revision=base_revision,
            control_source=workdir / "pair-trees" / "control",
            subject_source=workdir / "pair-trees" / "subject",
            control_composition=(),
            subject_composition=((patch_id, "p1"),),
            control_bin=workdir / "pair" / "CONTROL-BIN" / "llama-server",
            subject_bin=workdir / "pair" / "SUBJECT-BIN" / "llama-server",
            validation_build_identities={
                "control": {"build_id": "pair-control-build"},
                "subject": {"build_id": "pair-subject-build"},
            },
        )
        self.build_pair_calls: list[dict[str, object]] = []

    def build_pair(
        self,
        *,
        targets,
        primary_target,
        common_extra_patches=(),
        baseline_source="bigcherry",
        control_extra_cmake_args=(),
        subject_extra_cmake_args=(),
        require_parity: bool = False,
    ):
        self.build_pair_calls.append(
            {
                "targets": tuple(targets),
                "primary_target": primary_target,
                "baseline_source": baseline_source,
                "require_parity": require_parity,
            },
        )
        return self.pair

    def device_contexts(self, *, device_map):
        architecture = self.fat_targets.targets[0]
        return (
            vp.ProducerDeviceContext(
                architecture=architecture,
                device_index=0,
                execution_identity=ExecutionIdentity(
                    backend="ROCm",
                    architectures=(architecture,),
                ),
                env_overrides={"HIP_VISIBLE_DEVICES": "0"},
                env_unset=("ROCR_VISIBLE_DEVICES",),
            ),
        )

    def write_artifact(self, *, name, payload):
        return self._write(name, json.dumps(payload, indent=2))

    def write_text_artifact(self, *, name, text):
        return self._write(name, text)

    def run_paired_llama_benchmark(
        self,
        *,
        control_binary,
        subject_binary,
        model,
        workloads=("decode", "prefill"),
        patch_args=(),
        runtime_args=(),
        pairs=3,
        log_context,
        device=None,
    ):
        raise AssertionError("the RD13 producer must never benchmark")

    def _write(self, name: str, text: str) -> ArtifactRef:
        path = self.run_dir / "artifacts" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(text.encode("utf-8"))
        return ArtifactRef(
            name=name,
            path=path.relative_to(self.run_dir).as_posix(),
            sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        )


def _fake_tree(path: Path) -> str:
    return hashlib.sha256(str(path).encode("utf-8")).hexdigest()


def _fake_probe(
    *,
    workdir: Path,
    marker_regex: str,
    write_logs: bool = True,
    **kwargs: object,
) -> tuple[patch_activation.ActivationEvidence, dict[str, object]]:
    """Stand-in for vc.run_trace_activation_probes: the fake writes the
    two probe logs under the supplied workdir (the dispatcher binds
    their real bytes) and returns the (ActivationEvidence, detail)
    pair the dispatcher consumes."""
    if write_logs:
        logs = workdir / "logs"
        logs.mkdir(parents=True, exist_ok=True)
        (logs / "activation-positive.log").write_text(
            "BIGCHERRY_PATCH_HIT patch=1206_rd13 path=mul_mat_add_view_fusion_f\n",
            encoding="utf-8",
        )
        (logs / "activation-fusion-disabled.log").write_text(
            "no marker in negative control\n", encoding="utf-8"
        )
    evidence = patch_activation.ActivationEvidence(
        status="executed",
        mechanism="trace-marker",
        detail="fake probe: marker observed in positive, absent in negative",
    )
    detail = {
        "marker_regex": marker_regex,
        "positive": {
            "marker_observed": True,
            "log": "logs/activation-positive.log",
        },
        "negative_control": {
            "marker_observed": False,
            "log": "logs/activation-fusion-disabled.log",
        },
    }
    return evidence, detail


def _fake_producer(
    *,
    passed: bool = True,
    trace_evidence: dict[str, object] | None = None,
    activation_evidence: object = None,
) -> vp.ValidationProducer:
    def producer(ctx: vp.ProducerContext) -> vp.ProducerResult:
        return vp.ProducerResult(
            correctness={
                "disposition": "passed" if passed else "failed",
                "mechanism": "rd13-full-vocab-backend-reference",
                "detail": "fake producer measurement",
            },
            validation_build_identities={
                "control": {"build_id": "pair-control-build"},
                "subject": {"build_id": "pair-subject-build"},
            },
            activation_evidence=activation_evidence,
            performance_evidence=None,
            trace_evidence=trace_evidence,
            check_results=(),
            lane_effects=(),
            contract_correctness_results=(
                experiment_contract.CorrectnessResult(
                    check="backend_reference",
                    passed=passed,
                    detail="fake producer measurement",
                ),
            ),
            emitted_artifacts=frozenset({"rd13-backend-reference.json"}),
        )

    return producer


def _fake_selection(
    *,
    producer: vp.ValidationProducer,
    trace_probe: str = "run",
) -> vp.ProducerSelection:
    return vp.ProducerSelection(
        spec=vp.ProducerSpec(
            patch_id=PATCH_ID,
            producer_id="rd13",
            entrypoint=Path("producer.py"),
            callable_name="run",
            trace_probe=trace_probe,
            standard_campaign="run",
            correctness_evidence_cli="forbid",
            performance_benchmark_cli="forbid",
            artifact_names=_FAKE_ARTIFACT_NAMES,
        ),
        producer=producer,
    )


class _Recorder:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def __call__(self, **kwargs) -> dict[str, object]:
        self.calls.append(kwargs)
        return {"record": f"sentinel-{len(self.calls) - 1}"}


def _args(tmp: Path, **overrides) -> argparse.Namespace:
    base = {
        "patch": PATCH_ID,
        "model": tmp / "model" / "m.gguf",
        "hip_path": Path("/opt/rocm"),
        "amdgpu_targets": "gfx1100",
        "device_map": ["gfx1100=0"],
        "workdir": tmp / "workdir",
        "worktree_root": tmp / "worktrees",
        "build_root": tmp / "build",
        "correctness_evidence": None,
        "run_performance_benchmark": False,
        "producer_corpus": None,
        "baseline_source": "bigcherry",
        "bench_prompt": 512,
        "bench_gen": 128,
    }
    base.update(overrides)
    return argparse.Namespace(**base)


_UNSET_PROBE = object()


def _dispatch(
    tmp: Path,
    *,
    selection: vp.ProducerSelection | None = None,
    args_overrides: dict[str, object] | None = None,
    probe_side_effect: object = _UNSET_PROBE,
    recorders: dict[str, _Recorder] | None = None,
    model_bytes: bytes = b"fake-model-bytes-1206",
):
    """Run vc._run_validation_producer() against the REAL 1206
    patch/descriptor/plan with every expensive boundary faked. The
    producer is ALWAYS a fake (the real full-vocabulary measurement is
    covered by the producer-level test file); the trace probe is faked
    too (no real llama-bench run)."""
    recorders = recorders or {}
    scaffold_recorder = recorders.setdefault("scaffold", _Recorder())
    make_record_recorder = recorders.setdefault("make_record", _Recorder())
    write_record_recorder = recorders.setdefault("write_record", _Recorder())
    probe_recorder = recorders.setdefault("probe", _Recorder())

    scaffold = _FakeScaffold(tmp / "scaffold")
    env_backup = {key: os.environ.get(key) for key in ("ROCM_PATH", "HIP_PATH", "PATH")}

    def fake_scaffold(**kwargs) -> _FakeScaffold:
        scaffold_recorder.calls.append(kwargs)
        return scaffold

    def fake_write_record(record) -> Path:
        write_record_recorder.calls.append({"record": record})
        path = tmp / "records" / f"record-{len(write_record_recorder.calls) - 1}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"persisted": True}), encoding="utf-8")
        return path

    def fake_probe(**kwargs):
        probe_recorder.calls.append(kwargs)
        if probe_side_effect is not _UNSET_PROBE:
            if callable(probe_side_effect):
                return probe_side_effect(**kwargs)
            return probe_side_effect
        return _fake_probe(**kwargs)

    patches = [
        mock.patch.object(vc, "_build_standard_campaign_scaffold", fake_scaffold),
        mock.patch.object(vc, "CampaignProducerRuntime", _FakeDispatcherRuntime),
        mock.patch.object(psi, "git_worktree_tree", _fake_tree),
        mock.patch.object(psi, "patch_implementation_digest", lambda pid: "d" * 64),
        mock.patch.object(psi, "composition_digest", lambda c: "e" * 64),
        mock.patch.object(vc, "run_trace_activation_probes", fake_probe),
        mock.patch.object(patch_evidence, "make_record", make_record_recorder),
        mock.patch.object(patch_evidence, "write_record", fake_write_record),
    ]
    if selection is not None:
        patches.append(
            mock.patch.object(vc, "resolve_producer", lambda **kw: selection)
        )
    args = _args(tmp, **(args_overrides or {}))
    if args.model is not None:
        args.model.parent.mkdir(parents=True, exist_ok=True)
        args.model.write_bytes(model_bytes)
    for patcher in patches:
        patcher.start()
    try:
        exit_code = vc._run_validation_producer(
            args,
            producer_id="rd13",
            provided_inputs={},
        )
    finally:
        for key, value in env_backup.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        for patcher in patches:
            patcher.stop()

    run_dir = tmp / "workdir" / "campaign"
    outcome_path = run_dir / "producer-execution.json"
    outcome = (
        json.loads(outcome_path.read_text(encoding="utf-8"))
        if outcome_path.is_file()
        else None
    )
    return SimpleNamespace(
        exit_code=exit_code,
        scaffold=scaffold,
        scaffold_calls=scaffold_recorder.calls,
        make_record_calls=make_record_recorder.calls,
        write_record_calls=write_record_recorder.calls,
        probe_calls=probe_recorder.calls,
        run_dir=run_dir,
        outcome=outcome,
    )


class Rd13GenericDispatcherTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = Path(tempfile.mkdtemp())

    def tearDown(self) -> None:
        import shutil

        shutil.rmtree(self._tmp, ignore_errors=True)

    def test_run_path_probe_runs_activation_written_and_record_persisted(
        self,
    ) -> None:
        result = _dispatch(
            self._tmp,
            selection=_fake_selection(producer=_fake_producer()),
        )

        # Exit 0 iff execution + binding + persistence completed.
        self.assertEqual(result.exit_code, 0)
        self.assertEqual(len(result.make_record_calls), 1)
        self.assertEqual(len(result.write_record_calls), 1)
        self.assertEqual(len(result.scaffold_calls), 1)
        # The dispatcher ran the scaffold probe exactly once, against
        # the scaffold subject llama-bench (not a second build).
        self.assertEqual(len(result.probe_calls), 1)
        probe_kwargs = result.probe_calls[0]
        self.assertIn("llama-bench", str(probe_kwargs["binary"]))
        self.assertEqual(
            probe_kwargs["marker_regex"],
            "BIGCHERRY_PATCH_HIT patch=1206_rd13 path=mul_mat_add_view_fusion_(?:f|q)",
        )

        assert result.outcome is not None
        check_results = result.outcome["check_results"]
        # The declared trace-marker check PASSES from the real probe
        # output (no fabrication: the probe observed the marker).
        self.assertEqual(check_results["activation"]["status"], "pass")
        self.assertEqual(check_results["apply"]["status"], "pass")
        self.assertEqual(check_results["build"]["status"], "pass")
        self.assertEqual(check_results["correctness"]["status"], "pass")
        # RD13's performance/controls stay honestly unsatisfied (the
        # migration does not add benchmark semantics).
        self.assertNotEqual(check_results["performance"]["status"], "pass")
        self.assertNotEqual(check_results["controls"]["status"], "pass")

        # activation.json written by the dispatcher with the probe
        # disposition + trace_probe detail.
        activation_path = result.run_dir / "activation.json"
        self.assertTrue(activation_path.is_file())
        activation_doc = json.loads(activation_path.read_text(encoding="utf-8"))
        self.assertEqual(activation_doc["activation"]["status"], "executed")
        self.assertIn("trace_probe", activation_doc)
        self.assertIn("positive", activation_doc["trace_probe"])
        self.assertIn("negative_control", activation_doc["trace_probe"])

    def test_probe_failure_still_exit_zero_with_failed_record(self) -> None:
        # A failed probe (marker absent) is evidence, not process
        # failure: the run exits 0, the record persists, and the
        # activation check is blocked (no fabrication).
        def failing_probe(*, workdir: Path, marker_regex: str, **kwargs: object):
            logs = workdir / "logs"
            logs.mkdir(parents=True, exist_ok=True)
            (logs / "activation-positive.log").write_text(
                "no marker\n", encoding="utf-8"
            )
            (logs / "activation-fusion-disabled.log").write_text(
                "no marker\n", encoding="utf-8"
            )
            return (
                patch_activation.ActivationEvidence(
                    status="not_executed",
                    mechanism="trace-marker",
                    detail="marker absent from positive probe",
                ),
                {
                    "marker_regex": marker_regex,
                    "positive": {
                        "marker_observed": False,
                        "log": "logs/activation-positive.log",
                    },
                    "negative_control": {
                        "marker_observed": False,
                        "log": "logs/activation-fusion-disabled.log",
                    },
                },
            )

        result = _dispatch(
            self._tmp,
            selection=_fake_selection(producer=_fake_producer()),
            probe_side_effect=failing_probe,
        )
        self.assertEqual(result.exit_code, 0)
        self.assertEqual(len(result.make_record_calls), 1)
        assert result.outcome is not None
        self.assertNotEqual(
            result.outcome["check_results"]["activation"]["status"], "pass"
        )

    def test_probe_none_fails_closed(self) -> None:
        with self.assertRaisesRegex(
            vc.PatchCampaignError, "unexpectedly returned None"
        ):
            _dispatch(
                self._tmp,
                selection=_fake_selection(producer=_fake_producer()),
                probe_side_effect=None,
            )
        # A None probe is a hard error (the marker/description were both
        # non-empty, so None is a bug, not a legitimate skip): no
        # record is persisted.
        recorder = _Recorder()
        with self.assertRaises(vc.PatchCampaignError):
            _dispatch(
                self._tmp,
                selection=_fake_selection(producer=_fake_producer()),
                probe_side_effect=None,
                recorders={"make_record": recorder},
            )
        self.assertEqual(recorder.calls, [])

    def test_producer_supplied_trace_evidence_fails_closed(self) -> None:
        with self.assertRaisesRegex(vc.PatchCampaignError, "must not supply their own"):
            _dispatch(
                self._tmp,
                selection=_fake_selection(
                    producer=_fake_producer(
                        trace_evidence={
                            "positive": {"marker_regex": "x", "artifact": {}},
                            "negative": {"marker_regex": "x", "artifact": {}},
                        }
                    )
                ),
            )

    def test_producer_supplied_activation_evidence_fails_closed(self) -> None:
        with self.assertRaisesRegex(vc.PatchCampaignError, "must not supply their own"):
            _dispatch(
                self._tmp,
                selection=_fake_selection(
                    producer=_fake_producer(
                        activation_evidence=patch_activation.ActivationEvidence(
                            status="executed",
                            mechanism="fabricated",
                            detail="the producer must never supply this",
                        )
                    )
                ),
            )

    def test_forbid_cli_gates_fail_before_the_scaffold(self) -> None:
        for overrides, pattern in (
            (
                {"correctness_evidence": self._tmp / "ce.json"},
                "--correctness-evidence is ambiguous",
            ),
            (
                {"run_performance_benchmark": True},
                "--run-performance-benchmark is mutually exclusive",
            ),
        ):
            with (
                self.subTest(overrides=sorted(overrides)),
                self.assertRaisesRegex(vp.ValidationProducerError, pattern),
            ):
                _dispatch(
                    self._tmp,
                    selection=_fake_selection(producer=_fake_producer()),
                    args_overrides=cast("dict[str, object]", overrides),
                )


if __name__ == "__main__":
    unittest.main()
