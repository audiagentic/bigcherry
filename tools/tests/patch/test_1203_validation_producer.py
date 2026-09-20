"""PA37 step 8: hardware-free end-to-end tests for patch 1203's
patch-local plural-contract validation producer (RD05/RD06/RD07).

Everything here is hardware-free -- fake/controlled ProducerRuntime
results throughout, no real compile and no real GPU. Real hardware
execution is PA39's job (PA37.md's own scope boundary).

Covers PA37.md step 8's required scenarios: all required architectures
present, missing architecture, failed architecture, scoped contract
routing, artifact binding, build identity, plural gate serialization,
and promotion fail-closed behavior.
"""

from __future__ import annotations

import dataclasses
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

TOOLS_DIR = Path(__file__).resolve().parents[2]
REPO_ROOT = TOOLS_DIR.parent
sys.path.insert(0, str(TOOLS_DIR))

from bigcherry.experiment import contract as experiment_contract  # noqa: E402
from bigcherry.experiment import execution as experiment_execution  # noqa: E402
from bigcherry.patch import evidence as patch_validation_evidence  # noqa: E402
from bigcherry.patch import validation as pv  # noqa: E402
from bigcherry.patch import validation_campaign as vc  # noqa: E402
from bigcherry.patch import validation_producer as vp  # noqa: E402

_PATCH_ID = "1203_rd050607_rdna4_wmma_fa_q6k_mmq"
_PATCH_DIR = REPO_ROOT / "patches" / _PATCH_ID
_CONTRACTS_TOML = REPO_ROOT / "config" / "experiment-contracts.toml"

_RD05 = "RD05-WMMA-FA-CORRECTNESS-BARRIERS"
_RD06 = "RD06-RDNA4-WMMA-FA-CONFIG"
_RD07 = "RD07-Q6K-MMQ-PREFILL-FOLD"


# PA39 defects #2/#3 fix: real stats for RD06's positive (tierA-qwen4b-q6k,
# decode/tg128) and control (tierM-gptoss20b-q6k, decode) lanes, shaped
# exactly like block_bootstrap_effect()'s real output
# (geometric_effect_pct/ci95_low_pct/ci95_high_pct/paired_rounds) --
# lane_effect_from_run() reads these fields directly. Defaults here clear
# RD06's real contract bound (target_kernel_gain_pct=0.5,
# max_control_regression_pct=1, min_paired_rounds=10): a real gain with
# ci95_low above 0.5, and a near-zero control regression with ci95_high
# below 1.
_RD06_PASSING_POSITIVE_STATS = {
    "effect_pct": 2.0,
    "ci95_low": 1.0,
    "ci95_high": 3.0,
    "paired_rounds": 10,
}
_RD06_PASSING_CONTROL_STATS = {
    "effect_pct": -0.1,
    "ci95_low": -0.4,
    "ci95_high": 0.2,
    "paired_rounds": 10,
}
# RD06's contract declares BOTH decode and prefill as positive workloads
# (config/experiment-contracts.toml [contract.RD06-RDNA4-WMMA-FA-CONFIG.
# positive]) -- PA39 P0 defect #2 fix. Same passing numbers as decode by
# default; tests that need to prove the missing-lane fail-closed path
# override this directly.
_RD06_PASSING_PREFILL_STATS = {
    "effect_pct": 2.0,
    "ci95_low": 1.0,
    "ci95_high": 3.0,
    "paired_rounds": 10,
}

# PA39 real-hardware-acceptance fix: RD05's new controls check runs its own
# positive(decode)/control(prefill) paired benchmark on tierA-qwen4b-q6k --
# aggregate_contract_effects() requires a real positive-role lane even
# though RD05's gate never checks target_kernel_gain_pct (RD05's acceptance
# declares it as None); a small real gain plus a near-zero control
# regression keeps the default fixture path a clean PASS.
_RD05_PASSING_POSITIVE_STATS = {
    "effect_pct": 1.0,
    "ci95_low": 0.5,
    "ci95_high": 1.5,
    "paired_rounds": 3,
}
_RD05_PASSING_CONTROL_STATS = {
    "effect_pct": -0.1,
    "ci95_low": -0.4,
    "ci95_high": 0.2,
    "paired_rounds": 3,
}

# PA39 real-hardware-acceptance fix (2026-09-20): RD07's quantitative
# evaluation now runs on all three contract-scoped architectures, using the
# same real evaluate_promotion_gate() contract promotion gate as RD05/RD06
# (point_estimate_v1 for RD07, which declares no effect_evidence_policy).
# RD07's acceptance declares only
# max_control_regression_pct=1 (no target_kernel_gain_pct), so the gate's
# gain half is a no-op; only the control (decode) regression bound is
# enforced. These defaults keep the control lane's regression upper bound
# under 1, so the default fixture is a clean PASS.
_RD07_PASSING_POSITIVE_STATS = {
    "effect_pct": 1.0,
    "ci95_low": 0.5,
    "ci95_high": 1.5,
    "paired_rounds": 3,
}
_RD07_PASSING_CONTROL_STATS = {
    "effect_pct": -0.1,
    "ci95_low": -0.4,
    "ci95_high": 0.2,
    "paired_rounds": 3,
}

# PA39 real-hardware-acceptance fix: the real BIGCHERRY_PATCH_TRACE markers
# from patch.py (PA37 part 3) -- the new activation checks' default fixture
# behavior returns both when BIGCHERRY_PATCH_TRACE is set in the probe's
# env and neither when it is not, so the default happy-path fixture is a
# clean PASS without every test needing to know about markers.
_RD06_ACTIVATION_MARKER = (
    "BIGCHERRY_PATCH_HIT patch=1203_rd050607 path=wmma_f16_dispatch contract=RD06"
)
_RD07_ACTIVATION_MARKER = (
    "BIGCHERRY_PATCH_HIT patch=1203_rd050607 path=q6k_mmq_dispatch contract=RD07"
)


def _make_subprocess_side_effect(
    *,
    ppl_subject: float = 10.0,
    ppl_control: float = 10.0,
    ppl_uncertainty: float = 0.01,
):
    """A single robust subprocess.run() fake covering BOTH real subprocess
    call sites in producer.py: the backend_reference PPL comparisons
    (_run_backend_reference, alternating subject/control per pair, an
    unbounded number of pairs -- one for rd0506 plus one per RD07
    architecture) and the new activation probes (_run_activation_probe,
    two calls each: BIGCHERRY_PATCH_TRACE=1 then unset). Replaces the old
    fixed-length side_effect list (which only ever anticipated the PPL
    calls) so adding the new activation probe's subprocess.run() calls
    does not exhaust it. Distinguishes the two call sites by binary name
    (llama-perplexity vs llama-bench), matching producer.py's own real
    binary-selection contract."""
    state = {"ppl_calls": 0}

    def _side_effect(command, *, env=None, **_kwargs):
        binary_name = Path(command[0]).name
        if "llama-perplexity" in binary_name:
            state["ppl_calls"] += 1
            value = ppl_subject if state["ppl_calls"] % 2 == 1 else ppl_control
            return _ppl_completed(value, ppl_uncertainty)
        # Activation probe: both the subject and control binaries are
        # probed with BIGCHERRY_PATCH_TRACE=1 (the real negative control is
        # the UNPATCHED control binary, not a trace-off run of the subject
        # -- see _run_activation_probe()'s docstring) -- only the SUBJECT
        # binary (built at .../subject-bin/... by _FakeProducerRuntime.
        # build_pair()) can ever emit either real marker; the control
        # binary structurally cannot, matching the real patched-vs-
        # unpatched compiled code difference.
        # PA39 real-hardware attempt #3d finding: "-p 0" (no prefill) gave
        # RD06/RD07's dispatch conditions no honest chance to fire on real
        # hardware, even on the correctly-patched subject binary -- fixed
        # to "-p 512" (matching this producer's own pp512 performance
        # workload). Assert the real argv shape here so a regression back
        # to a no-prefill probe fails every test using this fixture, not
        # just a real-hardware rerun.
        assert "-p" in command, f"activation probe argv missing -p: {command!r}"
        assert command[command.index("-p") + 1] != "0", (
            f"activation probe argv uses -p 0 (no prefill) -- PA39 attempt #3d "
            f"real-hardware regression: {command!r}"
        )
        binary_path = str(command[0])
        is_subject = "subject-bin" in binary_path.replace("\\", "/")
        stdout = (
            f"{_RD06_ACTIVATION_MARKER}\n{_RD07_ACTIVATION_MARKER}\n"
            if is_subject
            else ""
        )
        return mock.Mock(returncode=0, stdout=stdout, stderr="")

    return _side_effect


def _paired_lane_run(
    *,
    effect_pct: float,
    ci95_low: float,
    ci95_high: float,
    paired_rounds: int,
) -> experiment_execution.PairedLaneRun:
    return experiment_execution.PairedLaneRun(
        runs=(),
        stats={
            "geometric_effect_pct": effect_pct,
            "ci95_low_pct": ci95_low,
            "ci95_high_pct": ci95_high,
            "paired_rounds": paired_rounds,
        },
    )


def _fake_build_identity(role: str) -> dict[str, object]:
    return {
        "effective_build_id": f"{role}-build-id",
        "compile_verification_id": f"{role}-compile-verification",
        "compile_commands_digest": f"{role}-compile-commands-digest",
        "hip_compile_commands_digest": f"{role}-hip-compile-commands-digest",
        "runtime_bundle_hash": f"{role}-runtime-bundle-hash",
        "runtime_artifacts": {f"{role}.bin": "a" * 64},
    }


class _FakeProducerRuntime:
    """Controlled, fully offline ProducerRuntime. ``benchmark_should_fail``
    lets a test simulate a paired-benchmark execution failure without any
    real subprocess/hardware."""

    def __init__(
        self,
        run_dir: Path,
        *,
        device_map,
        benchmark_should_fail: bool = False,
        rd06_positive_stats: dict | None = None,
        rd06_control_stats: dict | None = None,
        rd06_prefill_stats: dict | None = None,
        rd06_missing_prefill: bool = False,
        rd05_positive_stats: dict | None = None,
        rd05_control_stats: dict | None = None,
        rd07_positive_stats: dict | None = None,
        rd07_control_stats: dict | None = None,
        rd07_missing_prefill: bool = False,
        rd07_missing_control_arch: str | None = None,
    ):
        self.run_dir = run_dir
        self._device_map = device_map
        self.benchmark_should_fail = benchmark_should_fail
        self.build_pair_calls = 0
        self.build_pair_targets: list[str] = []
        self.rd06_positive_stats = rd06_positive_stats or _RD06_PASSING_POSITIVE_STATS
        self.rd06_control_stats = rd06_control_stats or _RD06_PASSING_CONTROL_STATS
        self.rd06_prefill_stats = rd06_prefill_stats or _RD06_PASSING_PREFILL_STATS
        self.rd05_positive_stats = rd05_positive_stats or _RD05_PASSING_POSITIVE_STATS
        self.rd05_control_stats = rd05_control_stats or _RD05_PASSING_CONTROL_STATS
        self.rd07_positive_stats = rd07_positive_stats or _RD07_PASSING_POSITIVE_STATS
        self.rd07_control_stats = rd07_control_stats or _RD07_PASSING_CONTROL_STATS
        # PA39 real-hardware-acceptance fix (2026-09-20): when True, the
        # positive lane's paired-benchmark outcome omits "prefill" on every
        # RD07 architecture (as if a real llama-bench run never produced that
        # lane), so tests can prove the producer fails closed rather than
        # silently aggregating decode-only evidence.
        self.rd07_missing_prefill = rd07_missing_prefill
        # GPT NIT (2026-09-20): when set to a specific arch, the control
        # (decode) lane's paired-benchmark outcome omits "decode" on JUST that
        # arch (all others still return decode), so a test can pin the exact
        # rd07_controls per-architecture invariant (one arch's missing control
        # lane -> controls FAIL) that the all-arch missing-prefill test
        # cannot reach.
        self.rd07_missing_control_arch = rd07_missing_control_arch
        # PA39 P0 defect #2 proof knob: when True, the positive lane's
        # paired-benchmark outcome omits "prefill" entirely (as if a real
        # llama-bench run had somehow not produced that lane), so tests can
        # prove the producer fails closed rather than silently aggregating
        # decode-only evidence.
        self.rd06_missing_prefill = rd06_missing_prefill
        self.paired_benchmark_calls: list[dict] = []

    def build_pair(self, *, primary_target: str, **_kwargs) -> vp.ProducerBuildPair:
        # PA39 P0 defect #1 proof: real primary_target-specific binary
        # paths, so a test can assert exactly which binary a given
        # run_paired_llama_benchmark() call actually received.
        self.build_pair_calls += 1
        self.build_pair_targets.append(primary_target)
        exe = "" if sys.platform != "win32" else ".exe"
        return vp.ProducerBuildPair(
            base_revision="a" * 40,
            control_source=self.run_dir / "control-src",
            subject_source=self.run_dir / "subject-src",
            control_composition=(),
            subject_composition=(),
            control_bin=self.run_dir / "control-bin" / f"{primary_target}{exe}",
            subject_bin=self.run_dir / "subject-bin" / f"{primary_target}{exe}",
            validation_build_identities={
                "control": _fake_build_identity("control"),
                "subject": _fake_build_identity("subject"),
            },
        )

    def device_contexts(self, *, device_map) -> tuple[vp.ProducerDeviceContext, ...]:
        contexts = []
        for architecture, indices in device_map.items():
            if architecture not in self._device_map:
                continue
            for index in indices:
                contexts.append(
                    vp.ProducerDeviceContext(
                        architecture=architecture,
                        device_index=index,
                        execution_identity=object(),
                        env_overrides={"HIP_VISIBLE_DEVICES": str(index)},
                        env_unset=("ROCR_VISIBLE_DEVICES",),
                    )
                )
        return tuple(contexts)

    def write_artifact(self, *, name: str, payload) -> pv.ArtifactRef:
        import hashlib
        import json

        target = self.run_dir / "artifacts" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        encoded = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
        target.write_bytes(encoded)
        return pv.ArtifactRef(
            name=name,
            path=f"artifacts/{name}",
            sha256=hashlib.sha256(encoded).hexdigest(),
        )

    def run_paired_llama_benchmark(
        self,
        *,
        log_context: str,
        model,
        workloads,
        control_binary=None,
        subject_binary=None,
        **_kwargs,
    ) -> vp.ProducerPairedBenchmarkOutcome:
        self.paired_benchmark_calls.append(
            {
                "log_context": log_context,
                "model": model,
                "workloads": workloads,
                "control_binary": control_binary,
                "subject_binary": subject_binary,
            }
        )
        if self.benchmark_should_fail:
            raise vc.PatchCampaignError("simulated benchmark failure")
        if log_context == "rd06-performance-positive":
            runs = {"decode": _paired_lane_run(**self.rd06_positive_stats)}
            if not self.rd06_missing_prefill:
                runs["prefill"] = _paired_lane_run(**self.rd06_prefill_stats)
            return vp.ProducerPairedBenchmarkOutcome(
                runs=runs,
                commands={w: {"control": ("x",), "subject": ("x",)} for w in runs},
                raw_logs=(),
            )
        if log_context == "rd06-performance-control":
            return vp.ProducerPairedBenchmarkOutcome(
                runs={"decode": _paired_lane_run(**self.rd06_control_stats)},
                commands={"decode": {"control": ("x",), "subject": ("x",)}},
                raw_logs=(),
            )
        if log_context == "rd05-controls-positive":
            return vp.ProducerPairedBenchmarkOutcome(
                runs={"decode": _paired_lane_run(**self.rd05_positive_stats)},
                commands={"decode": {"control": ("x",), "subject": ("x",)}},
                raw_logs=(),
            )
        if log_context == "rd05-controls-control":
            return vp.ProducerPairedBenchmarkOutcome(
                runs={"prefill": _paired_lane_run(**self.rd05_control_stats)},
                commands={"prefill": {"control": ("x",), "subject": ("x",)}},
                raw_logs=(),
            )
        # rd07-{arch}-positive / rd07-{arch}-control: PA39 real-hardware-
        # acceptance fix (2026-09-20) -- RD07's quantitative evaluation now
        # runs per-architecture; the fake returns a real PairedLaneRun for
        # each declared workload (prefill positive, decode control) so the
        # real lane_effect_from_run()/aggregate_contract_effects()/
        # evaluate_promotion_gate() path is exercised, not the old object()
        # stub.
        if log_context.startswith("rd07-") and log_context.endswith("-positive"):
            runs: dict[str, object] = {}
            if not self.rd07_missing_prefill:
                runs["prefill"] = _paired_lane_run(**self.rd07_positive_stats)
            return vp.ProducerPairedBenchmarkOutcome(
                runs=runs,
                commands={w: {"control": ("x",), "subject": ("x",)} for w in runs},
                raw_logs=(),
            )
        if log_context.startswith("rd07-") and log_context.endswith("-control"):
            arch = log_context[len("rd07-") : -len("-control")]
            runs: dict[str, object] = {}
            if arch != self.rd07_missing_control_arch:
                runs["decode"] = _paired_lane_run(**self.rd07_control_stats)
            return vp.ProducerPairedBenchmarkOutcome(
                runs=runs,
                commands={w: {"control": ("x",), "subject": ("x",)} for w in runs},
                raw_logs=(),
            )
        # Generic fallback (should not be reached for the RD05/RD06/RD07
        # contexts handled above).
        return vp.ProducerPairedBenchmarkOutcome(
            runs={"decode": object(), "prefill": object()},
            commands={"decode": {"control": ("x",), "subject": ("x",)}},
            raw_logs=(),
        )


def _ppl_completed(ppl: float, uncertainty: float) -> mock.Mock:
    return mock.Mock(
        returncode=0,
        stdout=f"Final estimate: PPL = {ppl} +/- {uncertainty}\n",
        stderr="",
    )


_MISSING = object()  # sentinel: "use the default control_model", distinct from None


def _run_producer(
    *,
    run_dir: Path,
    device_map: dict[str, tuple[int, ...]],
    model: Path | None,
    corpus: Path | None,
    benchmark_should_fail: bool = False,
    control_model: Path | None = _MISSING,
    rd06_positive_stats: dict | None = None,
    rd06_control_stats: dict | None = None,
    rd06_prefill_stats: dict | None = None,
    rd06_missing_prefill: bool = False,
    rd05_positive_stats: dict | None = None,
    rd05_control_stats: dict | None = None,
    rd07_positive_stats: dict | None = None,
    rd07_control_stats: dict | None = None,
    rd07_missing_prefill: bool = False,
    rd07_missing_control_arch: str | None = None,
):
    checks = pv.parse_validation_toml(
        _PATCH_DIR / "validation.toml", patch_id=_PATCH_ID
    )
    plan = pv.ValidationPlan(
        patch_id=_PATCH_ID, checks=checks, universal_capabilities=()
    )

    registry = experiment_contract.load_contracts(_CONTRACTS_TOML)
    contracts = (registry[_RD05], registry[_RD06], registry[_RD07])
    context = pv.ValidationContext(
        descriptor=None,
        base_revision="a" * 40,
        control_source=None,
        subject_source=None,
        package_root=_PATCH_DIR,
        contracts=contracts,
        contract_hashes={c.id: c.contract_hash for c in contracts},
    )

    run_dir.mkdir(parents=True, exist_ok=True)
    runtime = _FakeProducerRuntime(
        run_dir,
        device_map=device_map,
        benchmark_should_fail=benchmark_should_fail,
        rd06_positive_stats=rd06_positive_stats,
        rd06_control_stats=rd06_control_stats,
        rd06_prefill_stats=rd06_prefill_stats,
        rd06_missing_prefill=rd06_missing_prefill,
        rd05_positive_stats=rd05_positive_stats,
        rd05_control_stats=rd05_control_stats,
        rd07_positive_stats=rd07_positive_stats,
        rd07_control_stats=rd07_control_stats,
        rd07_missing_prefill=rd07_missing_prefill,
        rd07_missing_control_arch=rd07_missing_control_arch,
    )
    producer_context = vp.ProducerContext(
        repo_root=REPO_ROOT,
        patch_dir=_PATCH_DIR,
        workdir=run_dir,
        campaign_id=f"{_PATCH_ID}/rd050607",
        base_revision="a" * 40,
        hip_path=Path("/hip"),
        fat_targets=vp.FatTargetPlan(targets=("gfx1100", "gfx1201", "gfx1030")),
        model=model,
        corpus=corpus,
        build_env={},
        inputs={},
        validation_build_identities={},
        patch_id=_PATCH_ID,
        device_map=device_map,
        runtime=runtime,
    )

    resolved_control_model = (
        run_dir / "control-model.gguf" if control_model is _MISSING else control_model
    )
    provided_inputs = (
        {}
        if resolved_control_model is None
        else {"control_model": str(resolved_control_model)}
    )

    execution = vc.execute_validation_producer(
        patch_dir=_PATCH_DIR,
        producer_id="rd050607",
        provided_inputs=provided_inputs,
        producer_context=producer_context,
        validation_plan=plan,
        validation_context=context,
        correctness_evidence_requested=False,
        performance_benchmark_requested=False,
    )

    record = patch_validation_evidence.make_record(
        patch_id=_PATCH_ID,
        patch_path=_PATCH_DIR / "patch.toml",
        patch_implementation_digest="deadbeef" * 8,
        base_ref="pinned-ref",
        base_revision="a" * 40,
        framework_baseline_digest="b" * 64,
        patched_source_tree="c" * 40,
        gpu_architectures="gfx1100;gfx1201;gfx1030",
        activation_evidence=None,
        activation_disposition=None,
        correctness=None,
        campaign_identity_digest="d" * 64,
        build_identities={
            role: _fake_build_identity(role) for role in ("tune", "replay", "stock")
        },
        validation_build_identities=execution.result.validation_build_identities,
        campaign_workdir=run_dir,
        producer_artifact_names=execution.selection.spec.artifact_names,
        check_results={
            check_id: dataclasses.asdict(result)
            for check_id, result in execution.evaluated.items()
        },
        validation_eligible=execution.verdict.eligible,
        lane_effects=execution.result.lane_effects,
        contracts=[{"id": c.id, "hash": c.contract_hash} for c in contracts],
        contract_verdicts=execution.contract_verdicts,
    )
    return execution, record, runtime


class Patch1203ValidationProducerTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory(prefix="pa37-1203-producer-")
        self.addCleanup(self._tmp.cleanup)
        self.run_dir = Path(self._tmp.name) / "run"
        self.model = Path(self._tmp.name) / "model.gguf"
        self.corpus = Path(self._tmp.name) / "corpus.txt"

    def test_all_required_architectures_present_passes_closed(self) -> None:
        device_map = {"gfx1100": (0,), "gfx1201": (1,), "gfx1030": (2,)}
        with mock.patch("subprocess.run") as run_mock:
            # Backend-reference PPL comparisons (rd0506 1x + rd07 x3 archs)
            # and the new RD06/RD07 activation probes all share this one
            # robust side_effect (see _make_subprocess_side_effect()).
            run_mock.side_effect = _make_subprocess_side_effect()
            execution, record, runtime = _run_producer(
                run_dir=self.run_dir,
                device_map=device_map,
                model=self.model,
                corpus=self.corpus,
            )

        # PA39 P0 defect #1 fix: two build pairs (one binary set each,
        # never rebuilt per architecture) -- llama-perplexity for
        # backend-reference correctness, llama-bench for paired
        # performance benchmarking.
        self.assertEqual(runtime.build_pair_calls, 2, "one build pair per binary set")
        self.assertEqual(
            sorted(runtime.build_pair_targets),
            ["llama-bench", "llama-perplexity"],
        )
        self.assertTrue(execution.verdict.eligible, execution.verdict.reasons)
        for check_id in (
            "rd05-backend-reference",
            "rd06-backend-reference",
            "rd06-performance",
            "rd07-backend-reference",
            "rd07-performance",
        ):
            self.assertEqual(execution.evaluated[check_id].status, pv.PASS, check_id)

        # plural gate serialization: all three contracts have a bool verdict.
        self.assertEqual(set(record["contract_verdicts"]), {_RD05, _RD06, _RD07})
        for verdict in record["contract_verdicts"].values():
            self.assertIs(verdict["passed"], True)

        # build identity: the ONE build pair's identities are what got recorded.
        self.assertEqual(
            record["validation_build_identities"]["control"]["effective_build_id"],
            "control-build-id",
        )

    def test_missing_architecture_fails_closed(self) -> None:
        # gfx1201 absent -> rd05/rd06/rd06-performance AND rd07 (needs all
        # three) must all fail closed; no subprocess call is ever made.
        device_map = {"gfx1100": (0,), "gfx1030": (2,)}
        with mock.patch("subprocess.run") as run_mock:
            execution, record, _runtime = _run_producer(
                run_dir=self.run_dir,
                device_map=device_map,
                model=self.model,
                corpus=self.corpus,
            )
            run_mock.assert_not_called()

        self.assertFalse(execution.verdict.eligible)
        for check_id in (
            "rd05-backend-reference",
            "rd06-backend-reference",
            "rd06-performance",
            "rd07-backend-reference",
            "rd07-performance",
        ):
            self.assertEqual(execution.evaluated[check_id].status, pv.FAIL, check_id)
        for verdict in record["contract_verdicts"].values():
            self.assertIs(verdict["passed"], False)

    def test_failed_architecture_comparison_fails_closed(self) -> None:
        # All architectures present, but the real PPL comparison itself
        # disagrees beyond tolerance -- sigma way past max_sigma=3.0.
        device_map = {"gfx1100": (0,), "gfx1201": (1,), "gfx1030": (2,)}
        with mock.patch("subprocess.run") as run_mock:
            run_mock.side_effect = _make_subprocess_side_effect(
                ppl_subject=10.0, ppl_control=50.0
            )
            execution, record, _runtime = _run_producer(
                run_dir=self.run_dir,
                device_map=device_map,
                model=self.model,
                corpus=self.corpus,
            )

        self.assertFalse(execution.verdict.eligible)
        self.assertEqual(execution.evaluated["rd05-backend-reference"].status, pv.FAIL)
        self.assertEqual(execution.evaluated["rd07-backend-reference"].status, pv.FAIL)
        # RD06/RD07 performance checks still execute (benchmark itself
        # "succeeds" -- ci95 bound evaluation is PA39's job) but promotion
        # is still fail-closed overall because correctness failed.
        self.assertIs(record["contract_verdicts"][_RD05]["passed"], False)

    def test_missing_model_and_corpus_fails_closed(self) -> None:
        # RD07's activation/performance/controls now key off control_model
        # (always supplied in this fixture), not ctx.model -- so they are
        # no longer blocked by ctx.model/corpus being None the way RD05/
        # RD06 and RD07's own backend_reference correctness check are.
        # Real subprocess.run() calls (RD07's activation probe) are
        # mocked so the test stays hardware-free; RD07's disposition still
        # requires rd07_ok (backend_reference), which fails closed on the
        # missing corpus, so overall eligibility stays False regardless.
        device_map = {"gfx1100": (0,), "gfx1201": (1,), "gfx1030": (2,)}
        with mock.patch("subprocess.run") as run_mock:
            run_mock.side_effect = _make_subprocess_side_effect()
            execution, record, _runtime = _run_producer(
                run_dir=self.run_dir,
                device_map=device_map,
                model=None,
                corpus=None,
            )
        self.assertFalse(execution.verdict.eligible)
        for verdict in record["contract_verdicts"].values():
            self.assertIs(verdict["passed"], False)

    def test_benchmark_execution_failure_propagates_as_producer_error(self) -> None:
        device_map = {"gfx1100": (0,), "gfx1201": (1,), "gfx1030": (2,)}
        with mock.patch("subprocess.run") as run_mock:
            run_mock.side_effect = _make_subprocess_side_effect()
            with self.assertRaises(vc.PatchCampaignError):
                _run_producer(
                    run_dir=self.run_dir,
                    device_map=device_map,
                    model=self.model,
                    corpus=self.corpus,
                    benchmark_should_fail=True,
                )

    def test_scoped_contract_routing(self) -> None:
        device_map = {"gfx1100": (0,), "gfx1201": (1,), "gfx1030": (2,)}
        with mock.patch("subprocess.run") as run_mock:
            run_mock.side_effect = _make_subprocess_side_effect()
            execution, _record, _runtime = _run_producer(
                run_dir=self.run_dir,
                device_map=device_map,
                model=self.model,
                corpus=self.corpus,
            )

        contract_ids_by_check = {
            record.check_id: record.contract_ids
            for record in execution.result.check_results
        }
        self.assertEqual(contract_ids_by_check["rd05-backend-reference"], (_RD05,))
        self.assertEqual(contract_ids_by_check["rd06-backend-reference"], (_RD06,))
        self.assertEqual(contract_ids_by_check["rd06-performance"], (_RD06,))
        self.assertEqual(contract_ids_by_check["rd07-backend-reference"], (_RD07,))
        self.assertEqual(contract_ids_by_check["rd07-performance"], (_RD07,))

    def test_artifact_binding_only_declared_artifacts_are_evidence(self) -> None:
        device_map = {"gfx1100": (0,), "gfx1201": (1,), "gfx1030": (2,)}
        with mock.patch("subprocess.run") as run_mock:
            run_mock.side_effect = _make_subprocess_side_effect()
            _execution, record, _runtime = _run_producer(
                run_dir=self.run_dir,
                device_map=device_map,
                model=self.model,
                corpus=self.corpus,
            )

        artifact_paths = {a["path"] for a in record["campaign_artifacts"]}
        for name in (
            "rd0506-backend-reference.json",
            "rd06-performance.json",
            "rd07-backend-reference.json",
            "rd07-performance.json",
        ):
            self.assertIn(f"artifacts/{name}", artifact_paths)

        # An adjacent, undeclared artifact dropped in the same directory
        # must never become evidence.
        sneaky = self.run_dir / "artifacts" / "sneaky.json"
        sneaky.write_text("{}", encoding="utf-8")
        with mock.patch("subprocess.run") as run_mock:
            run_mock.side_effect = _make_subprocess_side_effect()
            _execution2, record2, _runtime2 = _run_producer(
                run_dir=self.run_dir,
                device_map=device_map,
                model=self.model,
                corpus=self.corpus,
            )
        sneaky_paths = {a["path"] for a in record2["campaign_artifacts"]}
        self.assertNotIn("artifacts/sneaky.json", sneaky_paths)

    def test_promotion_cannot_follow_from_correctness_alone(self) -> None:
        # Correctness passes but performance checks are gated off (no
        # devices) -- overall verdict must still be ineligible, proving
        # a correctness-only result never sets promotion eligibility.
        device_map: dict[str, tuple[int, ...]] = {}
        execution, record, _runtime = _run_producer(
            run_dir=self.run_dir,
            device_map=device_map,
            model=None,
            corpus=None,
        )
        self.assertFalse(execution.verdict.eligible)
        self.assertEqual(execution.evaluated["rd06-performance"].status, pv.FAIL)
        self.assertIs(record["contract_verdicts"][_RD06]["passed"], False)

    # --- PA39 defects #2/#3: RD06 real ci95_threshold_bound_v1 gate -----

    def test_rd06_gate_actually_invoked_with_below_bound_gain_fails(self) -> None:
        # PA39 defect #2 proof: before this fix, producer.py set
        # `rd06_perf_ok = bool(outcome.runs)`, which is True for ANY
        # non-empty paired-benchmark outcome regardless of the measured
        # numbers -- a gain whose ci95_low sits well BELOW RD06's own
        # target_kernel_gain_pct=0.5 bound would still have reported PASS
        # under the old code. This test supplies exactly that: a positive
        # point estimate whose lower CI bound (0.1) never clears 0.5. Only
        # the real evaluate_promotion_gate() call added by this fix can
        # correctly turn that into a FAIL.
        device_map = {"gfx1100": (0,), "gfx1201": (1,), "gfx1030": (2,)}
        below_bound_positive = {
            "effect_pct": 0.6,
            "ci95_low": 0.1,
            "ci95_high": 1.1,
            "paired_rounds": 10,
        }
        with mock.patch("subprocess.run") as run_mock:
            run_mock.side_effect = _make_subprocess_side_effect()
            execution, record, runtime = _run_producer(
                run_dir=self.run_dir,
                device_map=device_map,
                model=self.model,
                corpus=self.corpus,
                rd06_positive_stats=below_bound_positive,
            )
        # The old `bool(outcome.runs)` stub would have passed here (the
        # benchmark ran and returned non-empty results); the real gate
        # must not.
        self.assertEqual(execution.evaluated["rd06-performance"].status, pv.FAIL)
        self.assertIs(record["contract_verdicts"][_RD06]["passed"], False)
        self.assertFalse(execution.verdict.eligible)
        # And RD05/RD07 (unaffected by RD06's gain) still pass -- proves
        # the failure is genuinely scoped to RD06's own gate, not a
        # blanket regression.
        self.assertEqual(execution.evaluated["rd05-backend-reference"].status, pv.PASS)

    def test_rd06_gate_actually_invoked_with_over_budget_regression_fails(self) -> None:
        # Mirror case: the GAIN clears its bound but the CONTROL lane's
        # regression upper bound exceeds max_control_regression_pct=1 --
        # again something `bool(outcome.runs)` could never detect.
        device_map = {"gfx1100": (0,), "gfx1201": (1,), "gfx1030": (2,)}
        over_budget_control = {
            "effect_pct": -1.5,
            "ci95_low": -2.0,
            "ci95_high": -1.2,
            "paired_rounds": 10,
        }
        with mock.patch("subprocess.run") as run_mock:
            run_mock.side_effect = _make_subprocess_side_effect()
            execution, record, _runtime = _run_producer(
                run_dir=self.run_dir,
                device_map=device_map,
                model=self.model,
                corpus=self.corpus,
                rd06_control_stats=over_budget_control,
            )
        self.assertEqual(execution.evaluated["rd06-performance"].status, pv.FAIL)
        self.assertIs(record["contract_verdicts"][_RD06]["passed"], False)

    def test_rd06_gate_passing_bound_actually_passes(self) -> None:
        # Positive control: a real gain clearing 0.5 and a real control
        # regression bound under 1 must produce a real PASS through the
        # actual evaluate_promotion_gate() path (not a default/stubbed
        # True) -- already implicitly covered by
        # test_all_required_architectures_present_passes_closed, asserted
        # directly here for clarity of what the gate itself decided.
        device_map = {"gfx1100": (0,), "gfx1201": (1,), "gfx1030": (2,)}
        with mock.patch("subprocess.run") as run_mock:
            run_mock.side_effect = _make_subprocess_side_effect()
            execution, record, _runtime = _run_producer(
                run_dir=self.run_dir,
                device_map=device_map,
                model=self.model,
                corpus=self.corpus,
            )
        self.assertEqual(execution.evaluated["rd06-performance"].status, pv.PASS)
        self.assertIs(record["contract_verdicts"][_RD06]["passed"], True)

    # --- PA39 real-hardware-acceptance fix (2026-09-20): RD07 quantitative
    # evaluation (was the execution-only bool(outcome.runs) stub) + the
    # three-architecture widening (was gfx1201-only).

    def test_rd07_gate_actually_invoked_with_over_budget_regression_fails(self) -> None:
        # PA39 real-hardware-acceptance fix (2026-09-20) proof: before this
        # fix, producer.py set `rd07_perf_ok = bool(outcome.runs)`, which is
        # True for ANY non-empty paired-benchmark outcome regardless of the
        # measured numbers -- a control (decode) lane whose regression upper
        # bound (ci95_high=-1.2) sits WORSE than RD07's own
        # max_control_regression_pct=1 would still have reported PASS under
        # the old stub. This test supplies exactly that on all three
        # contract-scoped architectures; only the real
        # evaluate_promotion_gate() call added by this fix can turn it into
        # a FAIL.
        device_map = {"gfx1100": (0,), "gfx1201": (1,), "gfx1030": (2,)}
        over_budget_control = {
            "effect_pct": -1.5,
            "ci95_low": -2.0,
            "ci95_high": -1.2,
            "paired_rounds": 3,
        }
        with mock.patch("subprocess.run") as run_mock:
            run_mock.side_effect = _make_subprocess_side_effect()
            execution, record, _runtime = _run_producer(
                run_dir=self.run_dir,
                device_map=device_map,
                model=self.model,
                corpus=self.corpus,
                rd07_control_stats=over_budget_control,
            )
        self.assertEqual(execution.evaluated["rd07-performance"].status, pv.FAIL)
        self.assertIs(record["contract_verdicts"][_RD07]["passed"], False)
        self.assertFalse(execution.verdict.eligible)
        # RD05/RD06 (unaffected by RD07's control regression) still pass --
        # proves the failure is genuinely scoped to RD07's own gate, not a
        # blanket regression.
        self.assertEqual(execution.evaluated["rd05-backend-reference"].status, pv.PASS)
        self.assertEqual(execution.evaluated["rd06-performance"].status, pv.PASS)

    def test_rd07_missing_prefill_evidence_fails_closed(self) -> None:
        # PA39 real-hardware-acceptance fix (2026-09-20) proof: if the
        # positive (prefill) lane's paired-benchmark outcome does not
        # actually contain "prefill" evidence on any architecture (as if a
        # real llama-bench run never produced that lane), the producer must
        # fail closed rather than silently calling
        # aggregate_contract_effects()/evaluate_promotion_gate() with a
        # partial lane set -- aggregate_contract_effects() is NOT
        # contract-aware and would happily compute a gain from the control
        # lane alone.
        device_map = {"gfx1100": (0,), "gfx1201": (1,), "gfx1030": (2,)}
        with mock.patch("subprocess.run") as run_mock:
            run_mock.side_effect = _make_subprocess_side_effect()
            execution, record, _runtime = _run_producer(
                run_dir=self.run_dir,
                device_map=device_map,
                model=self.model,
                corpus=self.corpus,
                rd07_missing_prefill=True,
            )
        self.assertEqual(execution.evaluated["rd07-performance"].status, pv.FAIL)
        self.assertIs(record["contract_verdicts"][_RD07]["passed"], False)
        self.assertFalse(execution.verdict.eligible)
        # RD05/RD06 (unaffected by RD07's missing prefill lane) still pass.
        self.assertEqual(execution.evaluated["rd05-backend-reference"].status, pv.PASS)

    def test_rd07_single_arch_missing_control_lane_fails_controls(self) -> None:
        # GPT NIT (2026-09-20) coverage: if the control (decode) lane's
        # paired-benchmark outcome is missing on exactly ONE of the three
        # contract-scoped architectures (the other two still produce decode),
        # rd07-controls must fail closed -- pinning the exact per-arch
        # invariant that the all-arch missing-prefill test cannot reach.
        # The producer's rd07_control_lane_ok requires all three archs' control
        # lanes present, so a single arch's missing decode lane must fail the
        # whole check (and the performance gate, which shares the missing
        # control-lane evidence), not just that arch.
        device_map = {"gfx1100": (0,), "gfx1201": (1,), "gfx1030": (2,)}
        with mock.patch("subprocess.run") as run_mock:
            run_mock.side_effect = _make_subprocess_side_effect()
            execution, record, _runtime = _run_producer(
                run_dir=self.run_dir,
                device_map=device_map,
                model=self.model,
                corpus=self.corpus,
                rd07_missing_control_arch="gfx1201",
            )
        self.assertEqual(execution.evaluated["rd07-controls"].status, pv.FAIL)
        self.assertEqual(execution.evaluated["rd07-performance"].status, pv.FAIL)
        self.assertIs(record["contract_verdicts"][_RD07]["passed"], False)
        self.assertFalse(execution.verdict.eligible)
        # RD05/RD06 (unaffected by RD07's missing control lane) still pass.
        self.assertEqual(execution.evaluated["rd05-backend-reference"].status, pv.PASS)
        self.assertEqual(execution.evaluated["rd06-performance"].status, pv.PASS)

    def test_rd07_gate_passing_bound_actually_passes(self) -> None:
        # Positive control: a real control (decode) regression bound under
        # 1 must produce a real PASS through the actual
        # evaluate_promotion_gate() path (not a default/stubbed True).
        device_map = {"gfx1100": (0,), "gfx1201": (1,), "gfx1030": (2,)}
        with mock.patch("subprocess.run") as run_mock:
            run_mock.side_effect = _make_subprocess_side_effect()
            execution, record, _runtime = _run_producer(
                run_dir=self.run_dir,
                device_map=device_map,
                model=self.model,
                corpus=self.corpus,
            )
        self.assertEqual(execution.evaluated["rd07-performance"].status, pv.PASS)
        self.assertIs(record["contract_verdicts"][_RD07]["passed"], True)

    def test_rd07_benchmarked_on_all_three_contract_architectures(self) -> None:
        # PA39 real-hardware-acceptance fix (2026-09-20) proof: RD07's
        # contract scope is all three architectures (gfx1100/gfx1201/
        # gfx1030), and its performance/controls now benchmark each one
        # individually -- a positive (prefill) and a control (decode) lane
        # per architecture. This proves all six per-arch lanes are
        # benchmarked (closing the prior "only gfx1201" gap) and that the
        # positive lane requests "prefill" and the control lane requests
        # "decode", both on RD07's declared model (control_model, never the
        # generic --model).
        device_map = {"gfx1100": (0,), "gfx1201": (1,), "gfx1030": (2,)}
        with mock.patch("subprocess.run") as run_mock:
            run_mock.side_effect = _make_subprocess_side_effect()
            _execution, _record, runtime = _run_producer(
                run_dir=self.run_dir,
                device_map=device_map,
                model=self.model,
                corpus=self.corpus,
            )
        calls_by_context = {c["log_context"]: c for c in runtime.paired_benchmark_calls}
        for arch in ("gfx1100", "gfx1201", "gfx1030"):
            positive_ctx = f"rd07-{arch}-positive"
            control_ctx = f"rd07-{arch}-control"
            self.assertIn(positive_ctx, calls_by_context)
            self.assertIn(control_ctx, calls_by_context)
            self.assertEqual(
                set(calls_by_context[positive_ctx]["workloads"]), {"prefill"}
            )
            self.assertEqual(
                set(calls_by_context[control_ctx]["workloads"]), {"decode"}
            )
            self.assertEqual(
                calls_by_context[positive_ctx]["model"],
                calls_by_context[control_ctx]["model"],
            )

    # --- PA39 P0 defect #1: real llama-bench binary, never llama-perplexity

    def test_rd06_performance_uses_real_llama_bench_binary_not_perplexity(self) -> None:
        # PA39 P0 defect #1 proof (GPT review req_e3d28b6b104a4a01): before
        # this fix, the ONE build_pair() call used
        # primary_target="llama-perplexity" for everything, so
        # run_paired_llama_benchmark() -- which constructs llama-bench-
        # style argv -- was handed a llama-perplexity binary path. This
        # test proves the real fix: the control/subject binaries actually
        # passed to run_paired_llama_benchmark() for RD06's positive AND
        # control lanes are llama-bench, never llama-perplexity.
        device_map = {"gfx1100": (0,), "gfx1201": (1,), "gfx1030": (2,)}
        with mock.patch("subprocess.run") as run_mock:
            run_mock.side_effect = _make_subprocess_side_effect()
            _execution, _record, runtime = _run_producer(
                run_dir=self.run_dir,
                device_map=device_map,
                model=self.model,
                corpus=self.corpus,
            )
        calls_by_context = {c["log_context"]: c for c in runtime.paired_benchmark_calls}
        # PA39 real-hardware-acceptance fix (2026-09-20): RD07's benchmark
        # now runs per-architecture (positive/control on all three archs),
        # so every one of those lanes must also receive the llama-bench
        # binary, never llama-perplexity.
        rd07_bench_contexts = tuple(
            f"rd07-{arch}-{role}"
            for arch in ("gfx1100", "gfx1201", "gfx1030")
            for role in ("positive", "control")
        )
        for context in (
            "rd06-performance-positive",
            "rd06-performance-control",
            *rd07_bench_contexts,
        ):
            control_binary = calls_by_context[context]["control_binary"]
            subject_binary = calls_by_context[context]["subject_binary"]
            self.assertIn("llama-bench", control_binary.name, context)
            self.assertIn("llama-bench", subject_binary.name, context)
            self.assertNotIn("llama-perplexity", control_binary.name, context)
            self.assertNotIn("llama-perplexity", subject_binary.name, context)
        # And the backend_reference (PPL) comparisons build_pair() is the
        # llama-perplexity one -- a genuinely different pair, not the same
        # object reused under a different name.
        self.assertEqual(
            sorted(runtime.build_pair_targets), ["llama-bench", "llama-perplexity"]
        )

    # --- PA39 P0 defect #2: prefill lane evidence actually gathered ------

    def test_rd06_prefill_positive_lane_actually_benchmarked(self) -> None:
        # PA39 P0 defect #2 proof: RD06's contract declares BOTH decode
        # and prefill as positive workloads. Before this fix, the
        # producer only ever requested "decode" -- this proves the real
        # fix requests both.
        device_map = {"gfx1100": (0,), "gfx1201": (1,), "gfx1030": (2,)}
        with mock.patch("subprocess.run") as run_mock:
            run_mock.side_effect = _make_subprocess_side_effect()
            _execution, _record, runtime = _run_producer(
                run_dir=self.run_dir,
                device_map=device_map,
                model=self.model,
                corpus=self.corpus,
            )
        calls_by_context = {c["log_context"]: c for c in runtime.paired_benchmark_calls}
        self.assertEqual(
            set(calls_by_context["rd06-performance-positive"]["workloads"]),
            {"decode", "prefill"},
        )

    def test_rd06_missing_prefill_evidence_fails_closed(self) -> None:
        # PA39 P0 defect #2 proof: if the positive lane's paired-benchmark
        # outcome does not actually contain "prefill" evidence (simulating
        # a real llama-bench run that somehow didn't produce that lane),
        # the producer must fail closed rather than silently calling
        # aggregate_contract_effects()/evaluate_promotion_gate() with a
        # partial lane set -- aggregate_contract_effects() itself is NOT
        # contract-aware and would happily compute a gain from decode
        # alone, which is exactly the false-positive-promotion risk this
        # fix closes.
        device_map = {"gfx1100": (0,), "gfx1201": (1,), "gfx1030": (2,)}
        with mock.patch("subprocess.run") as run_mock:
            run_mock.side_effect = _make_subprocess_side_effect()
            execution, record, _runtime = _run_producer(
                run_dir=self.run_dir,
                device_map=device_map,
                model=self.model,
                corpus=self.corpus,
                rd06_missing_prefill=True,
            )
        self.assertEqual(execution.evaluated["rd06-performance"].status, pv.FAIL)
        self.assertIs(record["contract_verdicts"][_RD06]["passed"], False)
        self.assertFalse(execution.verdict.eligible)
        # RD05/RD07 (unaffected by RD06's missing prefill lane) still pass
        # -- proves the failure is genuinely scoped to RD06.
        self.assertEqual(execution.evaluated["rd05-backend-reference"].status, pv.PASS)

    def test_rd06_control_model_actually_benchmarked_separately(self) -> None:
        # PA39 defect #3 proof: the control model (tierM-gptoss20b-q6k, via
        # --producer-input control_model=<path>) must be benchmarked in its
        # OWN paired run, distinct from the positive model (ctx.model) --
        # never silently reused/skipped.
        device_map = {"gfx1100": (0,), "gfx1201": (1,), "gfx1030": (2,)}
        control_model = self.run_dir.parent / "control-model.gguf"
        with mock.patch("subprocess.run") as run_mock:
            run_mock.side_effect = _make_subprocess_side_effect()
            _execution, _record, runtime = _run_producer(
                run_dir=self.run_dir,
                device_map=device_map,
                model=self.model,
                corpus=self.corpus,
                control_model=control_model,
            )
        calls_by_context = {c["log_context"]: c for c in runtime.paired_benchmark_calls}
        self.assertIn("rd06-performance-positive", calls_by_context)
        self.assertIn("rd06-performance-control", calls_by_context)
        self.assertEqual(
            calls_by_context["rd06-performance-positive"]["model"], self.model
        )
        self.assertEqual(
            calls_by_context["rd06-performance-control"]["model"], control_model
        )
        self.assertNotEqual(
            calls_by_context["rd06-performance-positive"]["model"],
            calls_by_context["rd06-performance-control"]["model"],
        )

    def test_missing_control_model_input_fails_closed_before_producer_runs(
        self,
    ) -> None:
        # control_model is a REQUIRED producer input (producer.toml) --
        # omitting it must fail closed at input-validation time, not
        # silently skip RD06's control lane.
        device_map = {"gfx1100": (0,), "gfx1201": (1,), "gfx1030": (2,)}
        with self.assertRaises(vp.ValidationProducerError):
            _run_producer(
                run_dir=self.run_dir,
                device_map=device_map,
                model=self.model,
                corpus=self.corpus,
                control_model=None,
            )

    # --- PA39 real-hardware-acceptance fix: apply/build/activation/controls

    def test_all_new_capability_checks_pass_in_happy_path(self) -> None:
        # PA39 real-hardware-acceptance fix proof: apply, build, RD05
        # controls, RD06 activation, RD06 controls, RD07 activation, RD07
        # controls all resolve to real PASS results (not just "present in
        # the plan") when every architecture/model/corpus/control_model is
        # supplied -- these are exactly the seven checks PA39's real
        # ConfigurationError (real-hardware acceptance attempt #1) found
        # missing entirely.
        device_map = {"gfx1100": (0,), "gfx1201": (1,), "gfx1030": (2,)}
        with mock.patch("subprocess.run") as run_mock:
            run_mock.side_effect = _make_subprocess_side_effect()
            execution, record, _runtime = _run_producer(
                run_dir=self.run_dir,
                device_map=device_map,
                model=self.model,
                corpus=self.corpus,
            )
        self.assertTrue(execution.verdict.eligible, execution.verdict.reasons)
        for check_id in (
            "apply",
            "build",
            "rd05-controls",
            "rd06-activation",
            "rd06-controls",
            "rd07-activation",
            "rd07-controls",
        ):
            self.assertEqual(execution.evaluated[check_id].status, pv.PASS, check_id)

    def test_rd06_activation_fails_closed_when_marker_never_observed(self) -> None:
        # If BIGCHERRY_PATCH_TRACE=1 never produces the real marker (e.g.
        # the patched binary does not actually contain/reach the
        # instrumented dispatch site), the activation check must FAIL, not
        # silently PASS on the basis that the benchmark itself executed
        # without error.
        device_map = {"gfx1100": (0,), "gfx1201": (1,), "gfx1030": (2,)}
        with mock.patch("subprocess.run") as run_mock:
            run_mock.side_effect = lambda *a, **k: mock.Mock(
                returncode=0, stdout="", stderr=""
            )
            execution, record, _runtime = _run_producer(
                run_dir=self.run_dir,
                device_map=device_map,
                model=self.model,
                corpus=self.corpus,
            )
        self.assertEqual(execution.evaluated["rd06-activation"].status, pv.FAIL)
        self.assertEqual(execution.evaluated["rd07-activation"].status, pv.FAIL)
        self.assertFalse(execution.verdict.eligible)

    def test_rd06_activation_fails_closed_when_marker_observed_on_control_binary(
        self,
    ) -> None:
        # A marker that fires on the UNPATCHED control binary too would
        # mean it does not uniquely identify the patched dispatch path
        # (e.g. a probe bug, or a marker string that isn't actually
        # gated on the patch's own code) -- the probe must catch that as
        # a FAIL (control_hit=True), not a PASS.
        device_map = {"gfx1100": (0,), "gfx1201": (1,), "gfx1030": (2,)}
        with mock.patch("subprocess.run") as run_mock:
            run_mock.side_effect = lambda *a, **k: mock.Mock(
                returncode=0,
                stdout=f"{_RD06_ACTIVATION_MARKER}\n",
                stderr="",
            )
            execution, record, _runtime = _run_producer(
                run_dir=self.run_dir,
                device_map=device_map,
                model=self.model,
                corpus=self.corpus,
            )
        self.assertEqual(execution.evaluated["rd06-activation"].status, pv.FAIL)
        self.assertFalse(execution.verdict.eligible)

    def test_rd05_controls_regression_over_budget_fails_closed(self) -> None:
        # RD05's own contract still declares acceptance.max_control_
        # regression_pct=1 even though it has no target_kernel_gain_pct --
        # a control (prefill) regression beyond that budget must fail the
        # new rd05-controls check, proving the gate is genuinely evaluated
        # rather than a fabricated PASS.
        device_map = {"gfx1100": (0,), "gfx1201": (1,), "gfx1030": (2,)}
        over_budget = {
            "effect_pct": -1.5,
            "ci95_low": -2.0,
            "ci95_high": -1.2,
            "paired_rounds": 3,
        }
        with mock.patch("subprocess.run") as run_mock:
            run_mock.side_effect = _make_subprocess_side_effect()
            execution, record, _runtime = _run_producer(
                run_dir=self.run_dir,
                device_map=device_map,
                model=self.model,
                corpus=self.corpus,
                rd05_control_stats=over_budget,
            )
        self.assertEqual(execution.evaluated["rd05-controls"].status, pv.FAIL)
        self.assertFalse(execution.verdict.eligible)
        # RD06/RD07 (unaffected by RD05's control regression) still pass.
        self.assertEqual(execution.evaluated["rd06-performance"].status, pv.PASS)

    def test_apply_build_fail_closed_if_build_pair_raises(self) -> None:
        # A real build/apply failure must surface as a real FAIL for both
        # universal capabilities, never a silently-skipped check -- and
        # must propagate as a producer error (matching how a genuine
        # infrastructure failure is already treated elsewhere in this
        # producer, e.g. test_benchmark_execution_failure_propagates_as_
        # producer_error), not a fabricated PASS reached by catching the
        # exception and moving on.
        device_map = {"gfx1100": (0,), "gfx1201": (1,), "gfx1030": (2,)}
        checks = pv.parse_validation_toml(
            _PATCH_DIR / "validation.toml", patch_id=_PATCH_ID
        )
        plan = pv.ValidationPlan(
            patch_id=_PATCH_ID, checks=checks, universal_capabilities=()
        )
        registry = experiment_contract.load_contracts(_CONTRACTS_TOML)
        contracts = (registry[_RD05], registry[_RD06], registry[_RD07])
        context = pv.ValidationContext(
            descriptor=None,
            base_revision="a" * 40,
            control_source=None,
            subject_source=None,
            package_root=_PATCH_DIR,
            contracts=contracts,
            contract_hashes={c.id: c.contract_hash for c in contracts},
        )
        self.run_dir.mkdir(parents=True, exist_ok=True)
        runtime = _FakeProducerRuntime(self.run_dir, device_map=device_map)

        def _raise_build_pair(**_kwargs):
            raise vc.PatchCampaignError("simulated apply/build failure")

        runtime.build_pair = _raise_build_pair  # type: ignore[assignment]
        producer_context = vp.ProducerContext(
            repo_root=REPO_ROOT,
            patch_dir=_PATCH_DIR,
            workdir=self.run_dir,
            campaign_id=f"{_PATCH_ID}/rd050607",
            base_revision="a" * 40,
            hip_path=Path("/hip"),
            fat_targets=vp.FatTargetPlan(targets=("gfx1100", "gfx1201", "gfx1030")),
            model=self.model,
            corpus=self.corpus,
            build_env={},
            inputs={},
            validation_build_identities={},
            patch_id=_PATCH_ID,
            device_map=device_map,
            runtime=runtime,
        )
        with self.assertRaises(vc.PatchCampaignError):
            vc.execute_validation_producer(
                patch_dir=_PATCH_DIR,
                producer_id="rd050607",
                provided_inputs={
                    "control_model": str(self.run_dir / "control-model.gguf")
                },
                producer_context=producer_context,
                validation_plan=plan,
                validation_context=context,
                correctness_evidence_requested=False,
                performance_benchmark_requested=False,
            )

    # --- PA39 real-hardware-acceptance fix: real plan-resolution proof ---
    # This is the exact class of gap PA39's real-hardware acceptance
    # attempt #1 exposed: every prior hardware-free test drove
    # execute_validation_producer() directly against a hand-built
    # ValidationPlan(universal_capabilities=()), so none of them ever
    # called require_execution_package()/build_validation_plan() against
    # the REAL validation.toml + producer.toml + bound Experiment
    # Contracts the real CLI path (_run_validation_producer()) actually
    # uses. That real path failed closed with ConfigurationError at plan
    # resolution, before any GPU work, and no existing test could have
    # caught it.

    def test_real_validation_plan_resolves_via_require_execution_package(self) -> None:
        from bigcherry.patch import registry as patch_registry
        from bigcherry.patch import validation_policy as patch_validation_policy

        registry = patch_registry.load_registry(REPO_ROOT / "patches")
        descriptor = registry.get(_PATCH_ID)
        # Must not raise ConfigurationError -- this is the real call
        # _run_validation_producer() makes before ever touching hardware.
        plan = patch_validation_policy.require_execution_package(
            descriptor,
            root=REPO_ROOT / "patches",
        )
        self.assertEqual(
            set(plan.required_capabilities),
            {
                "apply",
                "build",
                "correctness",
                "performance",
                "activation",
                "controls",
            },
        )
        # Every required (contract_id, capability) pair PA39's real error
        # listed as missing must now have a covering check.
        required_pairs = set(plan.contract_requirements)
        self.assertEqual(
            required_pairs,
            {
                (_RD05, "correctness"),
                (_RD05, "controls"),
                (_RD06, "correctness"),
                (_RD06, "performance"),
                (_RD06, "activation"),
                (_RD06, "controls"),
                (_RD07, "correctness"),
                (_RD07, "performance"),
                (_RD07, "activation"),
                (_RD07, "controls"),
            },
        )

    def test_pre_fix_validation_toml_shape_fails_plan_resolution(self) -> None:
        # Concretely reproduces PA39's real ConfigurationError: the
        # ORIGINAL five checks alone (correctness/performance only, no
        # apply/build/activation/controls) against the same real bound
        # contracts must fail build_validation_plan() with exactly the
        # capabilities PA39's real run reported missing. Proves this test
        # class actually catches the real regression, not just that the
        # current (fixed) validation.toml happens to resolve.
        from bigcherry.patch import validation as patch_validation

        checks = pv.parse_validation_toml(
            _PATCH_DIR / "validation.toml", patch_id=_PATCH_ID
        )
        pre_fix_checks = tuple(
            c
            for c in checks
            if c.check_id
            in (
                "rd05-backend-reference",
                "rd06-backend-reference",
                "rd06-performance",
                "rd07-backend-reference",
                "rd07-performance",
            )
        )
        registry = experiment_contract.load_contracts(_CONTRACTS_TOML)
        bindings = tuple(
            patch_validation.bind_contract(registry[cid])
            for cid in (_RD05, _RD06, _RD07)
        )
        with self.assertRaises(patch_validation.ConfigurationError) as ctx:
            patch_validation.build_validation_plan(
                _PATCH_ID,
                pre_fix_checks,
                bindings=bindings,
            )
        message = str(ctx.exception)
        self.assertIn("apply", message)
        self.assertIn("build", message)
        self.assertIn(f"{_RD05}:controls", message)
        self.assertIn(f"{_RD06}:activation", message)
        self.assertIn(f"{_RD06}:controls", message)
        self.assertIn(f"{_RD07}:activation", message)
        self.assertIn(f"{_RD07}:controls", message)

    def test_fallback_evaluate_check_is_fail_closed_without_producer(self) -> None:
        # Directly exercise validation.toml's callables (evaluate_check()
        # path, no --validation-producer) -- must never fabricate a PASS.
        checks = pv.parse_validation_toml(
            _PATCH_DIR / "validation.toml", patch_id=_PATCH_ID
        )
        plan = pv.ValidationPlan(
            patch_id=_PATCH_ID, checks=checks, universal_capabilities=()
        )
        registry = experiment_contract.load_contracts(_CONTRACTS_TOML)
        contracts = (registry[_RD05], registry[_RD06], registry[_RD07])
        context = pv.ValidationContext(
            descriptor=None,
            base_revision="a" * 40,
            control_source=None,
            subject_source=None,
            package_root=_PATCH_DIR,
            contracts=contracts,
            contract_hashes={c.id: c.contract_hash for c in contracts},
        )
        for spec in plan.checks:
            result = pv.evaluate_check(spec, context)
            self.assertEqual(result.status, pv.FAIL, spec.check_id)


if __name__ == "__main__":
    unittest.main()
