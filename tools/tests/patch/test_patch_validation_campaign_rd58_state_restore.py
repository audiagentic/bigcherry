"""PA36 migration #4 (dev-gpt-agent req_82fbbafe52c0472d): RD58
state-restore producer tests -- hardware-free.

The legacy run_rd58_state_restore_evidence() orchestration tests are
REPLACED by direct tests of the 1234 patch-local producer module
(validation/producer.py): the legacy function, the
--run-rd58-state-restore CLI path, and the RD58-specific guards are
all DELETED from shared code in the same change. The three real,
independent claims (correctness, activation, controls) are preserved
1:1; the producer-specific guards (model required, one contract
architecture) and the preserved env sanitization (stale RROC /
BIGCHERRY_* strip) are new coverage for the migrated surface.
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
import unittest
from pathlib import Path
from typing import Any, cast
from unittest import mock

TOOLS_ROOT = Path(__file__).resolve().parents[2]
if str(TOOLS_ROOT) not in sys.path:
    sys.path.insert(0, str(TOOLS_ROOT))

from bigcherry.patch import validation_producer as vp  # noqa: E402
from bigcherry.patch.validation import ArtifactRef  # noqa: E402

SUBJECT_PATCH = "1234_rd58_pin_state_buffer_multigpu_restore"
PATCH_DIR = TOOLS_ROOT.parent / "patches" / SUBJECT_PATCH


def _load_producer() -> object:
    spec = importlib.util.spec_from_file_location(
        "_bc_rd58_producer", PATCH_DIR / "validation" / "producer.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    # The producer module must be registered in sys.modules before
    # exec_module (the _RunRecord dataclass needs it).
    sys.modules["_bc_rd58_producer"] = module
    spec.loader.exec_module(module)
    return module


class _FakeBuildPair:
    def __init__(self) -> None:
        self.control_bin = Path("/tmp/fake-control")
        self.subject_bin = Path("/tmp/fake-subject")
        self.validation_build_identities = {
            "control": {"build_id": "pair-control-build"},
            "subject": {"build_id": "pair-subject-build"},
        }


class _FakePairedLaneRun:
    def __init__(self, stats: dict[str, object]) -> None:
        self.stats = stats
        self.runs = ()


class _FakeOutcome:
    def __init__(self, stats: dict[str, object]) -> None:
        self.runs = {"decode": _FakePairedLaneRun(stats)}
        self.commands = {}
        self.raw_logs = ()


class _FakeDeviceVisibility:
    def document(self) -> dict[str, object]:
        return {"observed": ["0", "1"], "minimum": 2, "satisfied": True}


class _FakeRuntime:
    def __init__(self) -> None:
        self.pair = _FakeBuildPair()
        self.written_artifacts: dict[str, object] = {}
        self.written_text: dict[str, str] = {}
        self.benchmark_calls: list[dict[str, object]] = []

    def build_pair(
        self,
        *,
        targets: tuple[str, ...],
        primary_target: str,
        common_extra_patches: tuple[str, ...] = (),
        baseline_source: str = "bigcherry",
        control_extra_cmake_args: tuple[str, ...] = (),
        subject_extra_cmake_args: tuple[str, ...] = (),
        require_parity: bool = False,
    ) -> object:
        assert targets == ("gfx1100",)
        assert primary_target == "test-save-load-state"
        assert baseline_source == "bigcherry"
        assert require_parity is True
        return self.pair

    def write_artifact(self, *, name: str, payload: dict[str, object]) -> ArtifactRef:
        self.written_artifacts[name] = payload
        return ArtifactRef(
            name=name, path=str(f"/tmp/artifacts/{name}"), sha256="abc123"
        )

    def write_text_artifact(self, *, name: str, text: str) -> ArtifactRef:
        self.written_text[name] = text
        return ArtifactRef(
            name=name, path=str(f"/tmp/artifacts/{name}"), sha256="abc123"
        )

    def device_contexts(
        self, *, device_map: dict[str, tuple[int, ...]]
    ) -> tuple[object, ...]:
        return ()

    def run_paired_llama_benchmark(
        self,
        *,
        control_binary: Path,
        subject_binary: Path,
        model: Path,
        workloads: tuple[str, ...] = ("decode", "prefill"),
        patch_args: tuple[str, ...] = (),
        runtime_args: tuple[str, ...] = (),
        pairs: int = 3,
        log_context: str,
        device: object = None,
        env_overrides: dict[str, str] | None = None,
        env_unset: tuple[str, ...] = (),
    ) -> object:
        self.benchmark_calls.append(
            {
                "control_binary": control_binary,
                "subject_binary": subject_binary,
                "model": model,
                "workloads": workloads,
                "pairs": pairs,
                "log_context": log_context,
                "device": device,
                "env_overrides": env_overrides,
                "env_unset": env_unset,
                "runtime_args": runtime_args,
            }
        )
        return _FakeOutcome(
            {
                "geometric_effect_pct": 1.5,
                "ci95_low_pct": 0.5,
                "ci95_high_pct": 2.5,
                "paired_rounds": 3,
                "pair_ratios": [1.1, 0.9, 1.0],
            }
        )


def _make_context(runtime: _FakeRuntime) -> vp.ProducerContext:
    return vp.ProducerContext(
        repo_root=Path("/tmp/repo"),
        patch_dir=PATCH_DIR,
        workdir=Path("/tmp/workdir"),
        campaign_id="test-campaign",
        base_revision="abc123",
        hip_path=Path("/tmp/hip"),
        fat_targets=vp.FatTargetPlan(targets=("gfx1100",)),
        model=Path("/tmp/model.gguf"),
        corpus=None,
        build_env={"HIP_VISIBLE_DEVICES": "0,1", "ROCR_VISIBLE_DEVICES": "0,1"},
        inputs={},
        validation_build_identities=runtime.pair.validation_build_identities,
        patch_id=SUBJECT_PATCH,
        device_map={"gfx1100": (0, 1)},
        runtime=runtime,  # type: ignore[arg-type]
        validation_binaries={
            "control": {"llama-bench": Path("/tmp/fake-bench-control")},
            "subject": {"llama-bench": Path("/tmp/fake-bench-subject")},
        },
    )


class TestRD58Producer(unittest.TestCase):
    def _run_producer(
        self,
        *,
        subject_rc: int = 0,
        subject_marker: bool = True,
        control_marker: bool = False,
        control_rc: int = 0,
    ) -> tuple[vp.ProducerResult, _FakeRuntime]:
        module = _load_producer()
        runtime = _FakeRuntime()
        ctx = _make_context(runtime)

        def _fake_subprocess_run(
            command: list[str], **kwargs: object
        ) -> subprocess.CompletedProcess[str]:
            binary = str(command[0])
            is_subject = "fake-subject" in binary
            if is_subject:
                rc = subject_rc
                marker = subject_marker
            else:
                rc = control_rc
                marker = control_marker
            stdout = (
                "pinned state buffer (1024 bytes) for restore\n"
                if marker
                else "no marker\n"
            )
            return subprocess.CompletedProcess(
                args=command, returncode=rc, stdout=stdout, stderr=""
            )

        with mock.patch("subprocess.run", side_effect=_fake_subprocess_run), \
             mock.patch(
                 "bigcherry.experiment.execution.require_device_visibility",
                 return_value=_FakeDeviceVisibility(),
             ):
            result = cast(vp.ProducerResult, cast(Any, module).run(ctx))
        return result, runtime

    def test_correctness_pass(self) -> None:
        result, runtime = self._run_producer()
        self.assertEqual(cast(Any, result.correctness)["disposition"], "passed")
        self.assertEqual(
            cast(Any, result.correctness)["mechanism"], "rd58-state-restore-integrity"
        )
        # The state_restore_integrity correctness result.
        self.assertEqual(len(result.contract_correctness_results), 1)
        self.assertEqual(
            result.contract_correctness_results[0].check,
            "state_restore_integrity",
        )
        self.assertTrue(result.contract_correctness_results[0].passed)

    def test_correctness_fail(self) -> None:
        result, _ = self._run_producer(subject_rc=1)
        self.assertEqual(cast(Any, result.correctness)["disposition"], "failed")
        self.assertFalse(result.contract_correctness_results[0].passed)

    def test_activation_pass(self) -> None:
        result, _ = self._run_producer()
        self.assertEqual(cast(Any, result.activation_evidence).status, "executed")
        self.assertEqual(
            cast(Any, result.activation_evidence).mechanism, "rd58-trigger-marker"
        )

    def test_activation_fail_no_subject_hit(self) -> None:
        result, _ = self._run_producer(subject_marker=False)
        self.assertEqual(cast(Any, result.activation_evidence).status, "not_executed")

    def test_activation_fail_control_hit(self) -> None:
        result, _ = self._run_producer(control_marker=True)
        self.assertEqual(cast(Any, result.activation_evidence).status, "not_executed")

    def test_promotion_lane_effects(self) -> None:
        result, _ = self._run_producer()
        contract_id = "RD58-PIN-STATE-BUFFER-MULTIGPU-RESTORE"
        self.assertIn(contract_id, result.promotion_lane_effects)
        lane_effects = result.promotion_lane_effects[contract_id]
        self.assertEqual(len(lane_effects), 1)
        self.assertEqual(lane_effects[0].role, "control")
        self.assertEqual(lane_effects[0].metric, "tg128")
        self.assertEqual(lane_effects[0].geometric_effect_pct, 1.5)
        # The promotion target metric.
        self.assertEqual(result.promotion_target_metric[contract_id], "tg128")
        # The promotion trigger evidence.
        self.assertIn(contract_id, result.promotion_trigger_evidence)
        trigger = result.promotion_trigger_evidence[contract_id][0]
        self.assertEqual(trigger.role, "positive")
        self.assertEqual(trigger.lane_id, "rd58-subject")
        self.assertEqual(trigger.candidate_launches, 1)
        self.assertIsNone(trigger.expected_route_selected)

    def test_emitted_artifacts(self) -> None:
        result, _ = self._run_producer()
        self.assertEqual(
            result.emitted_artifacts,
            frozenset(
                {
                    "rd58-correctness.json",
                    "rd58-trigger-subject.log",
                    "rd58-trigger-control.log",
                    "rd58-trigger.json",
                    "performance.json",
                }
            ),
        )

    def test_benchmark_device_none(self) -> None:
        _, runtime = self._run_producer()
        self.assertEqual(len(runtime.benchmark_calls), 1)
        call = runtime.benchmark_calls[0]
        self.assertIsNone(call["device"])
        self.assertEqual(call["env_overrides"], {"GGML_CUDA_REGISTER_HOST": "1"})
        self.assertEqual(call["env_unset"], ("ROCR_VISIBLE_DEVICES",))
        self.assertEqual(call["workloads"], ("decode",))

    def test_env_sanitization(self) -> None:
        # GPT round 3: seed stale sanitizer keys to verify they are
        # stripped (the absence assertions are vacuous without them).
        module = _load_producer()
        runtime = _FakeRuntime()
        ctx = _make_context(runtime)
        # Seed the stale keys that sanitize_environment(mode="stock") strips.
        ctx.build_env["GGML_HIP_DISPATCH_MODE"] = "0"
        ctx.build_env["GGML_HIP_FORCE_DEVICE"] = "0"
        ctx.build_env["GGML_HIP_TUNE"] = "1"
        ctx.build_env["GGML_HIP_AUTOTUNE_MODE"] = "0"
        ctx.build_env["NCCL_DEBUG"] = "INFO"
        ctx.build_env["NCCL_DEBUG_SUBSYS"] = "INIT"
        ctx.build_env["NCCL_DEBUG_FILE"] = "/tmp/nccl.log"
        # A non-debug NCCL key that must SURVIVE sanitization.
        ctx.build_env["NCCL_SOCKET_IFNAME"] = "eth0"
        ctx.build_env["BIGCHERRY_STALE"] = "1"
        # Verify the env sanitization by checking the subprocess env.
        captured_envs: list[dict[str, str]] = []

        def _fake_subprocess_run(
            command: list[str], **kwargs: object
        ) -> subprocess.CompletedProcess[str]:
            env = kwargs.get("env")
            if env is not None:
                captured_envs.append(dict(cast(Any, env)))
            return subprocess.CompletedProcess(
                args=command,
                returncode=0,
                stdout="pinned state buffer (1024 bytes) for restore\n",
                stderr="",
            )

        with mock.patch("subprocess.run", side_effect=_fake_subprocess_run), \
             mock.patch(
                 "bigcherry.experiment.execution.require_device_visibility",
                 return_value=_FakeDeviceVisibility(),
             ):
            cast(Any, module).run(ctx)

        self.assertTrue(captured_envs)
        # All 6 runs (3 subject + 3 control) should have the same env.
        for env in captured_envs:
            self.assertNotIn("ROCR_VISIBLE_DEVICES", env)
            self.assertEqual(env["GGML_CUDA_REGISTER_HOST"], "1")
            self.assertEqual(env["HIP_VISIBLE_DEVICES"], "0,1")
            # No BIGCHERRY_* keys (GPT round 2 MAJOR: use canonical
            # sanitize_environment(mode="stock") which strips
            # GGML_HIP_DISPATCH_*, GGML_HIP_FORCE_*, GGML_HIP_TUNE_*,
            # GGML_HIP_AUTOTUNE_MODE, and NCCL_DEBUG*).
            self.assertFalse(any(k.startswith("BIGCHERRY_") for k in env))
            self.assertFalse(any(k.startswith("GGML_HIP_DISPATCH_") for k in env))
            self.assertFalse(any(k.startswith("GGML_HIP_FORCE_") for k in env))
            self.assertFalse(any(k.startswith("GGML_HIP_TUNE_") for k in env))
            self.assertNotIn("GGML_HIP_AUTOTUNE_MODE", env)
            self.assertNotIn("NCCL_DEBUG", env)
            self.assertNotIn("NCCL_DEBUG_SUBSYS", env)
            self.assertNotIn("NCCL_DEBUG_FILE", env)
            # Non-debug NCCL keys must survive (GPT round 3).
            self.assertEqual(env.get("NCCL_SOCKET_IFNAME"), "eth0")

    def test_model_required(self) -> None:
        module = _load_producer()
        runtime = _FakeRuntime()
        ctx = _make_context(runtime)
        # Rebuild the context with model=None.
        ctx_no_model = vp.ProducerContext(
            repo_root=ctx.repo_root,
            patch_dir=ctx.patch_dir,
            workdir=ctx.workdir,
            campaign_id=ctx.campaign_id,
            base_revision=ctx.base_revision,
            hip_path=ctx.hip_path,
            fat_targets=ctx.fat_targets,
            model=None,
            corpus=ctx.corpus,
            build_env=ctx.build_env,
            inputs=ctx.inputs,
            validation_build_identities=ctx.validation_build_identities,
            patch_id=ctx.patch_id,
            device_map=ctx.device_map,
            runtime=ctx.runtime,
            validation_binaries=ctx.validation_binaries,
        )
        with self.assertRaises(vp.ValidationProducerError):
            cast(Any, module).run(ctx_no_model)

    def test_wrong_architecture(self) -> None:
        module = _load_producer()
        runtime = _FakeRuntime()
        ctx = _make_context(runtime)
        # Rebuild the context with a wrong architecture.
        ctx_wrong_arch = vp.ProducerContext(
            repo_root=ctx.repo_root,
            patch_dir=ctx.patch_dir,
            workdir=ctx.workdir,
            campaign_id=ctx.campaign_id,
            base_revision=ctx.base_revision,
            hip_path=ctx.hip_path,
            fat_targets=vp.FatTargetPlan(targets=("gfx1030",)),
            model=ctx.model,
            corpus=ctx.corpus,
            build_env=ctx.build_env,
            inputs=ctx.inputs,
            validation_build_identities=ctx.validation_build_identities,
            patch_id=ctx.patch_id,
            device_map=ctx.device_map,
            runtime=ctx.runtime,
            validation_binaries=ctx.validation_binaries,
        )
        with self.assertRaises(vp.ValidationProducerError):
            cast(Any, module).run(ctx_wrong_arch)


if __name__ == "__main__":
    unittest.main()
