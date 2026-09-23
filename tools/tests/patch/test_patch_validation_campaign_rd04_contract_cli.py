"""PA36 migration #2 (dev-gpt-agent req_69d011a2acf44acc): the RD04
dedicated CLI paths (--run-rd04-contract / --run-rd04-benchmark /
--rd04-corpus, run_rd04_contract_correctness,
run_rd04_benchmark_evidence) were DELETED from shared code and replaced
by the generic standard_campaign="run" producer path
(--validation-producer 1202_rd04_bf16_flash_attn_tile/rd04).

This file covers the replaced path at two levels:
  * source-layout regression guards (the deleted surface must not come
    back, and the producer manifest pins the migration policy),
  * the end-to-end dispatcher (_run_validation_producer) driven against
    the REAL 1202 patch/descriptor/plan with every expensive boundary
    (scaffold builds, pair build, PPL subprocess, benchmark) faked --
    exit semantics, record identity facts, the blocked-activation
    honesty, and the CLI forbid gates.

Producer-level measurement semantics (PPL comparison, artifact shapes,
scaffold-binary reuse) are covered by
test_patch_validation_campaign_rd04_correctness.py.
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
from collections.abc import Mapping
from pathlib import Path
from types import SimpleNamespace
from typing import cast
from unittest import mock

TOOLS_ROOT = Path(__file__).resolve().parents[2]
if str(TOOLS_ROOT) not in sys.path:
    sys.path.insert(0, str(TOOLS_ROOT))

from bigcherry.patch import evidence as patch_evidence  # noqa: E402
from bigcherry.patch import source as psi  # noqa: E402
from bigcherry.patch.campaign import producer as campaign_producer  # noqa: E402
from bigcherry.patch import validation_producer as vp  # noqa: E402
from bigcherry.patch.validation import ArtifactRef  # noqa: E402

PATCH_ID = "1202_rd04_bf16_flash_attn_tile"
CAMPAIGN_SRC = (
    TOOLS_ROOT / "bigcherry" / "patch" / "validation_campaign.py"
).read_text(encoding="utf-8")
EXPECTED_ARTIFACT_NAMES = frozenset(
    {
        "rd04-correctness-gfx1100.json",
        "rd04-correctness-gfx1201.json",
        "rd04-correctness-gfx1030.json",
        "rd04-performance-gfx1100.json",
        "rd04-performance-gfx1201.json",
        "rd04-performance-gfx1030.json",
    }
)


# ---------------------------------------------------------- layout guards


class Rd04DedicatedPathDeletionTests(unittest.TestCase):
    def test_dedicated_cli_surface_is_gone(self) -> None:
        for needle in (
            "run_rd04_contract_correctness",
            "run_rd04_benchmark_evidence",
            "run_rd04_contract",
            "run_rd04_benchmark",
            "run-rd04-contract",
            "run-rd04-benchmark",
            "rd04-corpus",
        ):
            self.assertNotIn(needle, CAMPAIGN_SRC, needle)

    def test_exclusion_tuples_do_not_carry_the_dead_flags(self) -> None:
        for line in CAMPAIGN_SRC.split("\n"):
            if '"run_rd58_state_restore", "run_rd73_contract"' in line:
                self.assertNotIn("run_rd04", line)
            if (
                "args.run_rd08_contract or " in line
                and "args.run_rd58_state_restore" in line
            ):
                self.assertNotIn("run_rd04", line)

    def test_producer_manifest_pins_the_migration_policy(self) -> None:
        selection = vp.resolve_producer(
            patch_dir=TOOLS_ROOT.parent / "patches" / PATCH_ID,
            producer_id="rd04",
        )
        self.assertEqual(selection.spec.patch_id, PATCH_ID)
        self.assertEqual(selection.spec.standard_campaign, "run")
        self.assertEqual(selection.spec.trace_probe, "skip")
        self.assertEqual(selection.spec.correctness_evidence_cli, "forbid")
        self.assertEqual(selection.spec.performance_benchmark_cli, "forbid")
        self.assertEqual(selection.spec.artifact_names, EXPECTED_ARTIFACT_NAMES)

    def test_producer_benchmark_consumes_the_scaffold_seam(self) -> None:
        # The design ruling's load-bearing line: the paired benchmark
        # REUSES the standard scaffold's llama-bench pair through
        # ProducerContext.validation_binaries -- it must never build a
        # second pair.
        producer_src = (
            TOOLS_ROOT.parent / "patches" / PATCH_ID / "validation" / "producer.py"
        ).read_text(encoding="utf-8")
        self.assertIn('ctx.validation_binaries.get("control"', producer_src)
        self.assertIn('ctx.validation_binaries.get("subject"', producer_src)
        self.assertIn('"llama-bench"', producer_src)
        # GPT re-review req_6e79607f075c479b verification item: the
        # benchmark MUST pass the resolved device through so the shared
        # runtime applies the explicit HIP selector + execution identity
        # (it only does when device is non-None).
        self.assertIn("device=device", producer_src)

    def test_dispatcher_fails_closed_on_plural_contracts(self) -> None:
        # GPT re-review req_6e79607f075c479b BLOCKER: the producer
        # dispatcher is plural-aware, but the named-result gate binds to a
        # single contract's authority. Pin the fail-closed guard so it
        # never silently picks bound_contracts[0] when several contracts
        # are bound, and never lets duplicate check names silently
        # overwrite each other in the {check: result} mapping.
        core_src = inspect.getsource(campaign_producer._producer_check_results)
        self.assertIn("len(bound_contracts) != 1", core_src)
        self.assertIn("requires exactly", core_src)
        self.assertIn("{result.check for result in named}", core_src)
        self.assertIn("duplicate check", core_src)


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
        # Values must be 64-hex digests (evidence._validate_build_identity).
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
        # The apply/build validators check the REAL directories exist.
        for source in (self.control_source, self.subject_source, self.stock_source):
            source.mkdir(parents=True, exist_ok=True)
        self.control_idempotent = True
        self.subject_idempotent = True
        # The real scaffold builds control/validation-subject with
        # llama-server+llama-bench targets under these bin dirs; the generic
        # path exposes them via ProducerContext.validation_binaries.
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
    """Stand-in for campaign_producer.CampaignProducerRuntime: one fake PPL pair, real
    artifact files under run_dir, one device for the run architecture, and
    a finite fake paired benchmark (the producer reuses the SCAFFOLD
    llama-bench binaries handed to it via ctx.validation_binaries)."""

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
            subject_composition=((PATCH_ID, "p1"),),
            control_bin=workdir / "pair" / "CONTROL-BIN" / "llama-perplexity",
            subject_bin=workdir / "pair" / "SUBJECT-BIN" / "llama-perplexity",
            validation_build_identities={
                "control": {"build_id": "ppl-pair-control-build"},
                "subject": {"build_id": "ppl-pair-subject-build"},
            },
        )
        self.build_pair_calls: list[dict[str, object]] = []
        self.benchmark_calls: list[dict[str, object]] = []

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
        from bigcherry.experiment.attestation import ExecutionIdentity

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
        self.benchmark_calls.append(
            {
                "control_binary": control_binary,
                "subject_binary": subject_binary,
                "patch_args": tuple(patch_args),
                "pairs": pairs,
                "log_context": log_context,
            },
        )

        def _lane(metric: str) -> SimpleNamespace:
            return SimpleNamespace(
                runs=[
                    {"metric": metric, "role": "control", "value": 100.0},
                    {"metric": metric, "role": "subject", "value": 103.4},
                ],
                stats={"geometric_effect_pct": 3.41, "p_value": 0.02},
            )

        return vp.ProducerPairedBenchmarkOutcome(
            runs={"decode": _lane("tg128"), "prefill": _lane("pp512")},
            commands={
                "decode": {
                    "control": ("c-bench", "-m", "m"),
                    "subject": ("s-bench", "-m", "m"),
                },
                "prefill": {
                    "control": ("c-bench", "-m", "m"),
                    "subject": ("s-bench", "-m", "m"),
                },
            },
            raw_logs=(
                {"path": "artifacts/rd04-benchmark-decode.log", "sha256": "0" * 64},
                {"path": "artifacts/rd04-benchmark-prefill.log", "sha256": "0" * 64},
            ),
        )

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
    """Real psi.git_worktree_tree() returns a git tree hash (hex); the
    campaign identity validator requires 40/64 hex for the source trees."""
    return hashlib.sha256(str(path).encode("utf-8")).hexdigest()


def _fake_ppl_run(*, subject_ppl: str = "10.50", control_ppl: str = "10.52"):
    def run(argv, **kwargs):
        executable = str(argv[0])
        if "SUBJECT-BIN" in executable:
            ppl = subject_ppl
        elif "CONTROL-BIN" in executable:
            ppl = control_ppl
        else:
            raise AssertionError(f"unexpected binary: {executable!r}")
        return SimpleNamespace(
            returncode=0,
            stdout=f"Final estimate: PPL = {ppl} +/- 0.05\n",
            stderr="",
        )

    return run


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
        "producer_corpus": tmp / "corpus" / "c.txt",
        "baseline_source": "bigcherry",
        "bench_prompt": 512,
        "bench_gen": 128,
    }
    base.update(overrides)
    return argparse.Namespace(**base)


def _dispatch(
    tmp: Path,
    *,
    selection=None,
    args_overrides: dict[str, object] | None = None,
    subject_ppl: str = "10.50",
    control_ppl: str = "10.52",
    recorders: dict[str, _Recorder] | None = None,
    model_bytes: bytes = b"fake-model-bytes-1202",
    corpus_bytes: bytes = b"fake-corpus-bytes-1202",
):
    """Run campaign_producer._run_validation_producer() against the REAL 1202 patch with
    every expensive boundary faked. Returns a namespace of everything the
    tests assert on."""
    fake_run = _fake_ppl_run(subject_ppl=subject_ppl, control_ppl=control_ppl)
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
        mock.patch.object(campaign_producer, "_build_standard_campaign_scaffold", fake_scaffold),
        mock.patch.object(campaign_producer, "CampaignProducerRuntime", _FakeDispatcherRuntime),
        mock.patch.object(psi, "git_worktree_tree", _fake_tree),
        mock.patch.object(psi, "patch_implementation_digest", lambda pid: "d" * 64),
        mock.patch.object(psi, "composition_digest", lambda c: "e" * 64),
        mock.patch("subprocess.run", fake_run),
        mock.patch.object(patch_evidence, "make_record", make_record_recorder),
        mock.patch.object(patch_evidence, "write_record", fake_write_record),
    ]
    if selection is not None:
        patches.append(
            mock.patch.object(campaign_producer, "resolve_producer", lambda **kw: selection)
        )
    args = _args(tmp, **(args_overrides or {}))
    # GPT review req_7a72896b609a48b5 BLOCKER #1: the dispatcher now binds
    # the real model/corpus FILE FACTS (path/size/sha256) into the campaign
    # identity, so the inputs must exist on disk -- create them with stable
    # bytes (a test that overrides one absent still works; the identity then
    # degrades accordingly).
    if args.model is not None:
        args.model.parent.mkdir(parents=True, exist_ok=True)
        args.model.write_bytes(model_bytes)
    if args.producer_corpus is not None:
        args.producer_corpus.parent.mkdir(parents=True, exist_ok=True)
        args.producer_corpus.write_bytes(corpus_bytes)
    for patcher in patches:
        patcher.start()
    try:
        exit_code = campaign_producer._run_validation_producer(
            args,
            producer_id="rd04",
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


class Rd04GenericDispatcherTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = Path(tempfile.mkdtemp())

    def tearDown(self) -> None:
        import shutil

        shutil.rmtree(self._tmp, ignore_errors=True)

    def test_run_path_exit_zero_and_record_persisted(self) -> None:
        result = _dispatch(self._tmp)

        # Exit 0 iff execution + binding + persistence completed --
        # eligibility is evidence, not process success.
        self.assertEqual(result.exit_code, 0)
        self.assertEqual(len(result.make_record_calls), 1)
        self.assertEqual(len(result.write_record_calls), 1)
        self.assertEqual(len(result.scaffold_calls), 1)
        scaffold_args = result.scaffold_calls[0]
        self.assertEqual(scaffold_args["patch_id"], PATCH_ID)
        self.assertEqual(scaffold_args["baseline_source"], "bigcherry")
        self.assertEqual(scaffold_args["amdgpu_targets"], "gfx1100")

        self.assertIsNotNone(result.outcome)
        assert result.outcome is not None
        # Activation stays honestly BLOCKED (1202 has no valid RD04
        # marker) -- ineligible, but the run succeeds.
        self.assertFalse(result.outcome["eligible"])
        self.assertIsInstance(result.outcome["evidence_record"], str)
        check_results = result.outcome["check_results"]
        # Scaffold-backed checks and the bound producer evidence PASS.
        self.assertEqual(check_results["apply"]["status"], "pass")
        self.assertEqual(check_results["build"]["status"], "pass")
        self.assertEqual(check_results["correctness"]["status"], "pass")
        self.assertEqual(check_results["performance"]["status"], "pass")
        self.assertEqual(check_results["controls"]["status"], "pass")
        # The declared trace-marker check stays blocked -- no fabrication.
        self.assertNotEqual(check_results["activation"]["status"], "pass")

        # Root canonical evidence written by the binder; NO activation.json
        # (activation_evidence=None -- nothing to bind).
        correctness = json.loads(
            (self._tmp / "workdir" / "campaign" / "correctness.json").read_text(
                encoding="utf-8",
            ),
        )
        self.assertEqual(correctness["disposition"], "passed")
        self.assertFalse(
            (self._tmp / "workdir" / "campaign" / "activation.json").is_file(),
        )

    def test_run_path_record_identity_facts(self) -> None:
        result = _dispatch(self._tmp)
        self.assertEqual(len(result.make_record_calls), 1)
        kwargs = result.make_record_calls[0]

        self.assertEqual(kwargs["patch_id"], PATCH_ID)
        self.assertEqual(kwargs["base_ref"], cfg_pinned())
        self.assertEqual(kwargs["base_revision"], "b" * 40)
        self.assertEqual(
            kwargs["patched_source_tree"], _fake_tree(result.scaffold.subject_source)
        )
        self.assertEqual(kwargs["gpu_architectures"], "gfx1100")
        self.assertEqual(kwargs["campaign_workdir"], result.run_dir)
        # Two DISTINCT provenance domains: campaign identities from the
        # scaffold, validation identities from the producer's own PPL pair
        # (which replaces the scaffold's at record level).
        self.assertEqual(
            kwargs["build_identities"],
            result.scaffold.campaign_build_identities,
        )
        self.assertEqual(
            kwargs["validation_build_identities"],
            {
                "control": {"build_id": "ppl-pair-control-build"},
                "subject": {"build_id": "ppl-pair-subject-build"},
            },
        )
        # GPT review req_7a72896b609a48b5 BLOCKER #1: RD04 consumes real
        # model/corpus files, so the campaign identity is the INPUT-BOUND
        # digest (not model-free) -- it hashes the real file facts
        # (path/size/sha256), so two runs whose model or corpus BYTES
        # differ never share a campaign identity. Computed from the
        # SCAFFOLD facts + the on-disk model/corpus (never the fat-three).
        model_file = self._tmp / "model" / "m.gguf"
        corpus_file = self._tmp / "corpus" / "c.txt"

        def _file_identity(path: Path) -> dict[str, object]:
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            return {"path": str(path), "size": path.stat().st_size, "sha256": digest}

        expected_digest = patch_evidence.producer_campaign_identity_digest(
            patch_name=PATCH_ID,
            patch_digest="d" * 64,
            patched_source_tree=_fake_tree(result.scaffold.subject_source),
            gpu_architecture="gfx1100",
            campaign_build_identities=result.scaffold.campaign_build_identities,
            base_revision=result.scaffold.base_revision,
            model=_file_identity(model_file),
            corpus=_file_identity(corpus_file),
        )
        self.assertEqual(kwargs["campaign_identity_digest"], expected_digest)

        # GPT review req_7a72896b609a48b5 BLOCKER #2: the real
        # contract-correctness gate over the producer's typed named results
        # is persisted as check_results._contract_correctness_gate (the
        # 1202 contract requires backend_reference + ppl_equality, both
        # passing here).
        gate = kwargs["check_results"]["_contract_correctness_gate"]
        self.assertIsInstance(gate, Mapping)
        self.assertEqual(gate["passed"], True)
        self.assertEqual(kwargs["producer_artifact_names"], EXPECTED_ARTIFACT_NAMES)
        bound_correctness = kwargs["correctness"]
        self.assertIsInstance(bound_correctness, Mapping)
        self.assertEqual(bound_correctness["disposition"], "passed")
        # No activation evidence -- disposition stays unknown, never a
        # fabricated "executed".
        self.assertIsNone(kwargs["activation_disposition"])
        self.assertEqual(kwargs["lane_effects"], ())
        # No promotions: every bound contract gets the explicit BLOCKED
        # verdict, so eligibility can never come from the producer path.
        self.assertFalse(kwargs["validation_eligible"])

    def test_different_model_bytes_yield_different_campaign_identity(
        self,
    ) -> None:
        # GPT review req_7a72896b609a48b5 BLOCKER #1: same scaffold, different
        # model BYTES -> a different input-bound campaign identity digest
        # (it hashes the file's sha256/size, so two materially different
        # RD04 runs never share one campaign identity).
        base = _dispatch(self._tmp, model_bytes=b"model-bytes-A")
        alt = _dispatch(self._tmp, model_bytes=b"model-bytes-B-different")
        self.assertEqual(len(base.make_record_calls), 1)
        self.assertEqual(len(alt.make_record_calls), 1)
        base_digest = base.make_record_calls[0]["campaign_identity_digest"]
        alt_digest = alt.make_record_calls[0]["campaign_identity_digest"]
        self.assertIsNotNone(base_digest)
        self.assertIsNotNone(alt_digest)
        self.assertNotEqual(base_digest, alt_digest)
        # Control: the SCAFFOLD campaign build identities are identical
        # across the two runs -- only the input-bound digest differs.
        self.assertEqual(
            base.make_record_calls[0]["build_identities"],
            alt.make_record_calls[0]["build_identities"],
        )

    def test_ppl_failure_still_exit_zero_with_failed_record(self) -> None:
        # control PPL far outside combined uncertainty: the correctness
        # check FAILS, but exit is 0 and the INELIGIBLE record is the
        # whole point of tracked evidence.
        result = _dispatch(self._tmp, subject_ppl="10.50", control_ppl="15.00")
        self.assertEqual(result.exit_code, 0)
        self.assertEqual(len(result.make_record_calls), 1)
        kwargs = result.make_record_calls[0]
        bound_correctness = kwargs["correctness"]
        self.assertIsInstance(bound_correctness, Mapping)
        self.assertEqual(bound_correctness["disposition"], "failed")
        self.assertFalse(result.outcome["eligible"])
        check_results = result.outcome["check_results"]
        self.assertEqual(check_results["correctness"]["status"], "fail")
        self.assertFalse(kwargs["validation_eligible"])

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
                    args_overrides=cast("dict[str, object]", overrides),
                )
        # Both gate failures happen before any scaffold build: the
        # scaffold recorder (fresh per _dispatch) must be empty.

    def test_producer_patch_id_mismatch_fails_closed(self) -> None:
        selection = vp.ProducerSelection(
            spec=vp.ProducerSpec(
                patch_id="9999_not_this_patch",
                producer_id="rd04",
                entrypoint=Path("producer.py"),
                callable_name="run",
                trace_probe="skip",
                standard_campaign="run",
                correctness_evidence_cli="forbid",
                performance_benchmark_cli="forbid",
                artifact_names=frozenset(),
            ),
            producer=lambda ctx: None,  # type: ignore[return-value]
        )
        with self.assertRaisesRegex(vp.ValidationProducerError, "does not match"):
            _dispatch(self._tmp, selection=selection)

    def test_skip_path_no_scaffold_no_record(self) -> None:
        def trivial_producer(ctx):
            return vp.ProducerResult(
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
            )

        selection = vp.ProducerSelection(
            spec=vp.ProducerSpec(
                patch_id=PATCH_ID,
                producer_id="rd04",
                entrypoint=Path("producer.py"),
                callable_name="run",
                trace_probe="skip",
                standard_campaign="skip",
                correctness_evidence_cli="forbid",
                performance_benchmark_cli="forbid",
                artifact_names=frozenset(),
            ),
            producer=trivial_producer,
        )
        result = _dispatch(self._tmp, selection=selection)

        self.assertEqual(result.exit_code, 0)
        self.assertEqual(result.scaffold_calls, [])
        self.assertEqual(result.make_record_calls, [])
        self.assertEqual(result.write_record_calls, [])
        # Self-contained producer semantics: its own run_dir, diagnostic
        # outcome only, no tracked record.
        skip_run_dir = self._tmp / "workdir" / "producer" / "rd04"
        self.assertTrue((skip_run_dir / "producer-execution.json").is_file())
        self.assertIsNone(result.outcome)  # campaign/ outcome never written


def cfg_pinned() -> str:
    from bigcherry.core import config as campaign_config
    from bigcherry.core import paths as bc_paths

    return campaign_config.load(bc_paths.RECIPES).pinned


if __name__ == "__main__":
    unittest.main()
