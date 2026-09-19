"""PA36 migration #4 (dev-gpt-agent req_110d0729beb44d8b): the RD26
dedicated CLI path (--run-rd26-contract,
run_rd26_decode_verify_bit_identity_check, run_rd26_ppl_check,
_load_rd26_correctness_module, the rd26_correctness.py module) was
DELETED from shared code and replaced by the generic
standard_campaign="run" producer path
(--validation-producer 1210_rd26_bitidentical_decode_verify_standalone/rd26).

This file covers the replaced path at two levels:
  * source-layout regression guards (the deleted surface must not come
    back, and the producer manifest pins the migration policy),
  * the end-to-end dispatcher (_run_validation_producer) driven against
    the REAL 1210 patch/descriptor/plan with the expensive producer
    boundary faked (a lightweight stand-in producer; the real
    raw-logit oracle mechanics are covered by
    test_patch_validation_campaign_rd26_bit_identity.py) -- the
    trace_probe="skip" path: no probe runs, no activation.json is
    written, and the honest correctness FAIL of the historical record
    is a valid exit-0 receipt (eligibility is evidence, not process
    success).

Producer-level measurement semantics (the decode-vs-verify raw-F32
byte-identity oracle, artifact shape, guards) are covered by
test_patch_validation_campaign_rd26_bit_identity.py.
"""

from __future__ import annotations

import argparse
import hashlib
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
from bigcherry.patch import evidence as patch_evidence  # noqa: E402
from bigcherry.patch import source as psi  # noqa: E402
from bigcherry.patch import validation_campaign as vc  # noqa: E402
from bigcherry.patch import validation_producer as vp  # noqa: E402
from bigcherry.patch.validation import ArtifactRef  # noqa: E402

PATCH_ID = "1210_rd26_bitidentical_decode_verify_standalone"
CAMPAIGN_SRC = (
    TOOLS_ROOT / "bigcherry" / "patch" / "validation_campaign.py"
).read_text(encoding="utf-8")
PRODUCER_SRC = (
    TOOLS_ROOT.parent / "patches" / PATCH_ID / "validation" / "producer.py"
).read_text(encoding="utf-8")
EXPECTED_ARTIFACT_NAMES = frozenset({"rd26-decode-verify-bit-identity.json"})


def cfg_pinned() -> str:
    from bigcherry.core import config as campaign_config
    from bigcherry.core import paths as bc_paths

    return campaign_config.load(bc_paths.RECIPES).pinned


# ---------------------------------------------------------- layout guards


class Rd26DedicatedPathDeletionTests(unittest.TestCase):
    def test_dedicated_cli_surface_is_gone(self) -> None:
        for needle in (
            "run_rd26_decode_verify_bit_identity_check",
            "run_rd26_ppl_check",
            "_load_rd26_correctness_module",
            "run-rd26-contract",
            "rd26_correctness",
        ):
            self.assertNotIn(needle, CAMPAIGN_SRC, needle)

    def test_exclusion_tuples_do_not_carry_the_dead_flags(self) -> None:
        for line in CAMPAIGN_SRC.split("\n"):
            if '"run_rd58_state_restore", "run_rd73_contract"' in line:
                self.assertNotIn("run_rd26", line)
            if (
                "args.run_rd08_contract or " in line
                and "args.run_rd58_state_restore" in line
            ):
                self.assertNotIn("run_rd26", line)

    def test_producer_manifest_pins_the_migration_policy(self) -> None:
        selection = vp.resolve_producer(
            patch_dir=TOOLS_ROOT.parent / "patches" / PATCH_ID,
            producer_id="rd26",
        )
        self.assertEqual(selection.spec.patch_id, PATCH_ID)
        self.assertEqual(selection.spec.standard_campaign, "run")
        # RD26 declares NO activation check (expected effect is
        # determinism/correctness only) -- there is no marker to probe.
        self.assertEqual(selection.spec.trace_probe, "skip")
        self.assertEqual(selection.spec.correctness_evidence_cli, "forbid")
        self.assertEqual(selection.spec.performance_benchmark_cli, "forbid")
        self.assertEqual(selection.spec.artifact_names, EXPECTED_ARTIFACT_NAMES)

    def test_producer_returns_no_activation_or_trace_evidence(self) -> None:
        # The honesty line: the producer owns ONLY the measurement;
        # activation/trace/performance stay untouched.
        self.assertIn("activation_evidence=None", PRODUCER_SRC)
        self.assertIn("trace_evidence=None", PRODUCER_SRC)
        self.assertIn("performance_evidence=None", PRODUCER_SRC)

    def test_producer_builds_the_llama_results_pair_once(self) -> None:
        # The isolated llama-results pair is the producer's own
        # build_pair() domain (fat multi-arch, bigcherry baseline,
        # parity asserted).
        self.assertIn("llama-results", PRODUCER_SRC)
        self.assertIn("require_parity=True", PRODUCER_SRC)
        self.assertIn('primary_target="llama-results"', PRODUCER_SRC)


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
    """Stand-in for vc.CampaignProducerRuntime: one fake llama-results
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
            control_bin=workdir / "pair" / "CONTROL-BIN" / "llama-results",
            subject_bin=workdir / "pair" / "SUBJECT-BIN" / "llama-results",
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
        raise AssertionError("the RD26 producer must never benchmark")

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


def _fake_producer(
    *,
    passed: bool = True,
) -> vp.ValidationProducer:
    def producer(ctx: vp.ProducerContext) -> vp.ProducerResult:
        return vp.ProducerResult(
            correctness={
                "disposition": "passed" if passed else "failed",
                "mechanism": "rd26-decode-verify-bit-identity",
                "detail": "fake producer measurement",
            },
            validation_build_identities={
                "control": {"build_id": "pair-control-build"},
                "subject": {"build_id": "pair-subject-build"},
            },
            activation_evidence=None,
            performance_evidence=None,
            trace_evidence=None,
            check_results=(),
            lane_effects=(),
            contract_correctness_results=(
                experiment_contract.CorrectnessResult(
                    check="bit_identical",
                    passed=passed,
                    detail="fake producer measurement",
                ),
            ),
            emitted_artifacts=frozenset({"rd26-decode-verify-bit-identity.json"}),
        )

    return producer


def _fake_selection(
    *,
    producer: vp.ValidationProducer,
    standard_campaign: str = "run",
) -> vp.ProducerSelection:
    return vp.ProducerSelection(
        spec=vp.ProducerSpec(
            patch_id=PATCH_ID,
            producer_id="rd26",
            entrypoint=Path("producer.py"),
            callable_name="run",
            trace_probe="skip",
            standard_campaign=standard_campaign,
            correctness_evidence_cli="forbid",
            performance_benchmark_cli="forbid",
            artifact_names=EXPECTED_ARTIFACT_NAMES,
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


def _dispatch(
    tmp: Path,
    *,
    selection: vp.ProducerSelection | None = None,
    args_overrides: dict[str, object] | None = None,
    recorders: dict[str, _Recorder] | None = None,
    model_bytes: bytes = b"fake-model-bytes-1210",
):
    """Run vc._run_validation_producer() against the REAL 1210
    patch/descriptor/plan with every expensive boundary faked. The
    producer is ALWAYS a fake (the real raw-logit oracle is covered by
    the producer-level test file). RD26 is trace_probe="skip": no
    probe machinery is involved."""
    recorders = recorders or {}
    scaffold_recorder = recorders.setdefault("scaffold", _Recorder())
    make_record_recorder = recorders.setdefault("make_record", _Recorder())
    write_record_recorder = recorders.setdefault("write_record", _Recorder())

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

    patches = [
        mock.patch.object(vc, "_build_standard_campaign_scaffold", fake_scaffold),
        mock.patch.object(vc, "CampaignProducerRuntime", _FakeDispatcherRuntime),
        mock.patch.object(psi, "git_worktree_tree", _fake_tree),
        mock.patch.object(psi, "patch_implementation_digest", lambda pid: "d" * 64),
        mock.patch.object(psi, "composition_digest", lambda c: "e" * 64),
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
            producer_id="rd26",
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
        run_dir=run_dir,
        outcome=outcome,
    )


class Rd26GenericDispatcherTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = Path(tempfile.mkdtemp())

    def tearDown(self) -> None:
        import shutil

        shutil.rmtree(self._tmp, ignore_errors=True)

    def test_run_path_exit_zero_no_probe_no_activation_json(self) -> None:
        result = _dispatch(
            self._tmp,
            selection=_fake_selection(producer=_fake_producer()),
        )

        # Exit 0 iff execution + binding + persistence completed.
        self.assertEqual(result.exit_code, 0)
        self.assertEqual(len(result.make_record_calls), 1)
        self.assertEqual(len(result.write_record_calls), 1)
        self.assertEqual(len(result.scaffold_calls), 1)

        assert result.outcome is not None
        check_results = result.outcome["check_results"]
        # 1210 declares NO activation check (trace_probe="skip"): no
        # activation entry at all, and no activation.json written.
        self.assertNotIn("activation", check_results)
        self.assertFalse(
            (result.run_dir / "activation.json").is_file(),
        )
        self.assertEqual(check_results["apply"]["status"], "pass")
        self.assertEqual(check_results["build"]["status"], "pass")
        self.assertEqual(check_results["correctness"]["status"], "pass")
        # The contract's controls check stays honestly unsatisfied (the
        # migration does not add benchmark semantics).
        self.assertNotEqual(check_results["controls"]["status"], "pass")

    def test_honest_failed_receipt_is_exit_zero_with_failed_record(self) -> None:
        # THIS MIGRATION PROVES PRODUCER EQUIVALENCE, NOT RD26
        # QUALIFICATION: a real run against the current 2-of-5
        # base-standalone subset may legitimately FAIL, and that honest
        # FAIL is the expected receipt (eligibility is evidence, not
        # process success).
        result = _dispatch(
            self._tmp,
            selection=_fake_selection(producer=_fake_producer(passed=False)),
        )
        self.assertEqual(result.exit_code, 0)
        self.assertEqual(len(result.make_record_calls), 1)
        assert result.outcome is not None
        self.assertEqual(
            result.outcome["check_results"]["correctness"]["status"], "fail"
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

    def test_skip_path_no_scaffold_no_record(self) -> None:
        selection = vp.ProducerSelection(
            spec=vp.ProducerSpec(
                patch_id=PATCH_ID,
                producer_id="rd26",
                entrypoint=Path("producer.py"),
                callable_name="run",
                trace_probe="skip",
                standard_campaign="skip",
                correctness_evidence_cli="forbid",
                performance_benchmark_cli="forbid",
                artifact_names=frozenset(),
            ),
            producer=lambda ctx: vp.ProducerResult(
                correctness=None,
                validation_build_identities={
                    "control": {"build_id": "c"},
                    "subject": {"build_id": "s"},
                },
                activation_evidence=None,
                performance_evidence=None,
                trace_evidence=None,
                check_results=(),
                lane_effects=(),
                emitted_artifacts=frozenset(),
            ),
        )
        result = _dispatch(self._tmp, selection=selection)

        self.assertEqual(result.exit_code, 0)
        self.assertEqual(result.scaffold_calls, [])
        self.assertEqual(result.make_record_calls, [])
        self.assertEqual(result.write_record_calls, [])
        # Self-contained producer semantics: its own run_dir, diagnostic
        # outcome only, no tracked record.
        skip_run_dir = self._tmp / "workdir" / "producer" / "rd26"
        self.assertTrue((skip_run_dir / "producer-execution.json").is_file())
        self.assertIsNone(result.outcome)  # campaign/ outcome never written


if __name__ == "__main__":
    unittest.main()
