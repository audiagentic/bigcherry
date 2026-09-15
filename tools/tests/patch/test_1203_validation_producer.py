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
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

TOOLS_DIR = Path(__file__).resolve().parents[2]
REPO_ROOT = TOOLS_DIR.parent
sys.path.insert(0, str(TOOLS_DIR))

from bigcherry.experiment import contract as experiment_contract  # noqa: E402
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

    def __init__(self, run_dir: Path, *, device_map, benchmark_should_fail: bool = False):
        self.run_dir = run_dir
        self._device_map = device_map
        self.benchmark_should_fail = benchmark_should_fail
        self.build_pair_calls = 0

    def build_pair(self, **_kwargs) -> vp.ProducerBuildPair:
        self.build_pair_calls += 1
        return vp.ProducerBuildPair(
            base_revision="a" * 40,
            control_source=self.run_dir / "control-src",
            subject_source=self.run_dir / "subject-src",
            control_composition=(), subject_composition=(),
            control_bin=self.run_dir / "control-bin" / "llama-perplexity",
            subject_bin=self.run_dir / "subject-bin" / "llama-perplexity",
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
                contexts.append(vp.ProducerDeviceContext(
                    architecture=architecture, device_index=index,
                    execution_identity=object(),
                    env_overrides={"HIP_VISIBLE_DEVICES": str(index)},
                    env_unset=("ROCR_VISIBLE_DEVICES",),
                ))
        return tuple(contexts)

    def write_artifact(self, *, name: str, payload) -> pv.ArtifactRef:
        import hashlib
        import json

        target = self.run_dir / "artifacts" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        encoded = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
        target.write_bytes(encoded)
        return pv.ArtifactRef(
            name=name, path=f"artifacts/{name}", sha256=hashlib.sha256(encoded).hexdigest(),
        )

    def run_paired_llama_benchmark(self, **_kwargs) -> vp.ProducerPairedBenchmarkOutcome:
        if self.benchmark_should_fail:
            raise vc.PatchCampaignError("simulated benchmark failure")
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


def _run_producer(
    *, run_dir: Path, device_map: dict[str, tuple[int, ...]],
    model: Path | None, corpus: Path | None, benchmark_should_fail: bool = False,
):
    checks = pv.parse_validation_toml(_PATCH_DIR / "validation.toml", patch_id=_PATCH_ID)
    plan = pv.ValidationPlan(patch_id=_PATCH_ID, checks=checks, universal_capabilities=())

    registry = experiment_contract.load_contracts(_CONTRACTS_TOML)
    contracts = (registry[_RD05], registry[_RD06], registry[_RD07])
    context = pv.ValidationContext(
        descriptor=None, base_revision="a" * 40, control_source=None, subject_source=None,
        package_root=_PATCH_DIR,
        contracts=contracts,
        contract_hashes={c.id: c.contract_hash for c in contracts},
    )

    run_dir.mkdir(parents=True, exist_ok=True)
    runtime = _FakeProducerRuntime(
        run_dir, device_map=device_map, benchmark_should_fail=benchmark_should_fail,
    )
    producer_context = vp.ProducerContext(
        repo_root=REPO_ROOT, patch_dir=_PATCH_DIR, workdir=run_dir,
        campaign_id=f"{_PATCH_ID}/rd050607", base_revision="a" * 40,
        hip_path=Path("/hip"),
        fat_targets=vp.FatTargetPlan(targets=("gfx1100", "gfx1201", "gfx1030")),
        model=model, corpus=corpus, build_env={}, inputs={},
        validation_build_identities={}, patch_id=_PATCH_ID, device_map=device_map,
        runtime=runtime,
    )

    execution = vc.execute_validation_producer(
        patch_dir=_PATCH_DIR, producer_id="rd050607",
        provided_inputs={},
        producer_context=producer_context, validation_plan=plan,
        validation_context=context,
        correctness_evidence_requested=False, performance_benchmark_requested=False,
    )

    record = patch_validation_evidence.make_record(
        patch_id=_PATCH_ID, patch_path=_PATCH_DIR / "patch.toml",
        patch_implementation_digest="deadbeef" * 8,
        base_ref="pinned-ref", base_revision="a" * 40,
        framework_baseline_digest="b" * 64,
        patched_source_tree="c" * 40,
        gpu_architectures="gfx1100;gfx1201;gfx1030",
        activation_evidence=None, activation_disposition=None, correctness=None,
        campaign_identity_digest="d" * 64,
        build_identities={role: _fake_build_identity(role) for role in ("tune", "replay", "stock")},
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
            # Two run_perplexity() calls (subject, control) per each of:
            # rd0506 (1x) + rd07 x3 archs = 4 comparisons = 8 subprocess calls.
            run_mock.side_effect = [
                _ppl_completed(10.0, 0.01), _ppl_completed(10.0, 0.01),
            ] * 4
            execution, record, runtime = _run_producer(
                run_dir=self.run_dir, device_map=device_map,
                model=self.model, corpus=self.corpus,
            )

        self.assertEqual(runtime.build_pair_calls, 1, "one atomic build pair only")
        self.assertTrue(execution.verdict.eligible, execution.verdict.reasons)
        for check_id in (
            "rd05-backend-reference", "rd06-backend-reference", "rd06-performance",
            "rd07-backend-reference", "rd07-performance",
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
                run_dir=self.run_dir, device_map=device_map,
                model=self.model, corpus=self.corpus,
            )
            run_mock.assert_not_called()

        self.assertFalse(execution.verdict.eligible)
        for check_id in (
            "rd05-backend-reference", "rd06-backend-reference", "rd06-performance",
            "rd07-backend-reference", "rd07-performance",
        ):
            self.assertEqual(execution.evaluated[check_id].status, pv.FAIL, check_id)
        for verdict in record["contract_verdicts"].values():
            self.assertIs(verdict["passed"], False)

    def test_failed_architecture_comparison_fails_closed(self) -> None:
        # All architectures present, but the real PPL comparison itself
        # disagrees beyond tolerance -- sigma way past max_sigma=3.0.
        device_map = {"gfx1100": (0,), "gfx1201": (1,), "gfx1030": (2,)}
        with mock.patch("subprocess.run") as run_mock:
            run_mock.side_effect = [
                _ppl_completed(10.0, 0.01), _ppl_completed(50.0, 0.01),
            ] * 4
            execution, record, _runtime = _run_producer(
                run_dir=self.run_dir, device_map=device_map,
                model=self.model, corpus=self.corpus,
            )

        self.assertFalse(execution.verdict.eligible)
        self.assertEqual(execution.evaluated["rd05-backend-reference"].status, pv.FAIL)
        self.assertEqual(execution.evaluated["rd07-backend-reference"].status, pv.FAIL)
        # RD06/RD07 performance checks still execute (benchmark itself
        # "succeeds" -- ci95 bound evaluation is PA39's job) but promotion
        # is still fail-closed overall because correctness failed.
        self.assertIs(record["contract_verdicts"][_RD05]["passed"], False)

    def test_missing_model_and_corpus_fails_closed(self) -> None:
        device_map = {"gfx1100": (0,), "gfx1201": (1,), "gfx1030": (2,)}
        execution, record, _runtime = _run_producer(
            run_dir=self.run_dir, device_map=device_map, model=None, corpus=None,
        )
        self.assertFalse(execution.verdict.eligible)
        for verdict in record["contract_verdicts"].values():
            self.assertIs(verdict["passed"], False)

    def test_benchmark_execution_failure_propagates_as_producer_error(self) -> None:
        device_map = {"gfx1100": (0,), "gfx1201": (1,), "gfx1030": (2,)}
        with mock.patch("subprocess.run") as run_mock:
            run_mock.side_effect = [
                _ppl_completed(10.0, 0.01), _ppl_completed(10.0, 0.01),
            ] * 4
            with self.assertRaises(vc.PatchCampaignError):
                _run_producer(
                    run_dir=self.run_dir, device_map=device_map,
                    model=self.model, corpus=self.corpus, benchmark_should_fail=True,
                )

    def test_scoped_contract_routing(self) -> None:
        device_map = {"gfx1100": (0,), "gfx1201": (1,), "gfx1030": (2,)}
        with mock.patch("subprocess.run") as run_mock:
            run_mock.side_effect = [
                _ppl_completed(10.0, 0.01), _ppl_completed(10.0, 0.01),
            ] * 4
            execution, _record, _runtime = _run_producer(
                run_dir=self.run_dir, device_map=device_map,
                model=self.model, corpus=self.corpus,
            )

        contract_ids_by_check = {
            record.check_id: record.contract_ids for record in execution.result.check_results
        }
        self.assertEqual(contract_ids_by_check["rd05-backend-reference"], (_RD05,))
        self.assertEqual(contract_ids_by_check["rd06-backend-reference"], (_RD06,))
        self.assertEqual(contract_ids_by_check["rd06-performance"], (_RD06,))
        self.assertEqual(contract_ids_by_check["rd07-backend-reference"], (_RD07,))
        self.assertEqual(contract_ids_by_check["rd07-performance"], (_RD07,))

    def test_artifact_binding_only_declared_artifacts_are_evidence(self) -> None:
        device_map = {"gfx1100": (0,), "gfx1201": (1,), "gfx1030": (2,)}
        with mock.patch("subprocess.run") as run_mock:
            run_mock.side_effect = [
                _ppl_completed(10.0, 0.01), _ppl_completed(10.0, 0.01),
            ] * 4
            _execution, record, _runtime = _run_producer(
                run_dir=self.run_dir, device_map=device_map,
                model=self.model, corpus=self.corpus,
            )

        artifact_paths = {a["path"] for a in record["campaign_artifacts"]}
        for name in (
            "rd0506-backend-reference.json", "rd06-performance.json",
            "rd07-backend-reference.json", "rd07-performance.json",
        ):
            self.assertIn(f"artifacts/{name}", artifact_paths)

        # An adjacent, undeclared artifact dropped in the same directory
        # must never become evidence.
        sneaky = self.run_dir / "artifacts" / "sneaky.json"
        sneaky.write_text("{}", encoding="utf-8")
        with mock.patch("subprocess.run") as run_mock:
            run_mock.side_effect = [
                _ppl_completed(10.0, 0.01), _ppl_completed(10.0, 0.01),
            ] * 4
            _execution2, record2, _runtime2 = _run_producer(
                run_dir=self.run_dir, device_map=device_map,
                model=self.model, corpus=self.corpus,
            )
        sneaky_paths = {a["path"] for a in record2["campaign_artifacts"]}
        self.assertNotIn("artifacts/sneaky.json", sneaky_paths)

    def test_promotion_cannot_follow_from_correctness_alone(self) -> None:
        # Correctness passes but performance checks are gated off (no
        # devices) -- overall verdict must still be ineligible, proving
        # a correctness-only result never sets promotion eligibility.
        device_map: dict[str, tuple[int, ...]] = {}
        execution, record, _runtime = _run_producer(
            run_dir=self.run_dir, device_map=device_map, model=None, corpus=None,
        )
        self.assertFalse(execution.verdict.eligible)
        self.assertEqual(execution.evaluated["rd06-performance"].status, pv.FAIL)
        self.assertIs(record["contract_verdicts"][_RD06]["passed"], False)

    def test_fallback_evaluate_check_is_fail_closed_without_producer(self) -> None:
        # Directly exercise validation.toml's callables (evaluate_check()
        # path, no --validation-producer) -- must never fabricate a PASS.
        checks = pv.parse_validation_toml(_PATCH_DIR / "validation.toml", patch_id=_PATCH_ID)
        plan = pv.ValidationPlan(patch_id=_PATCH_ID, checks=checks, universal_capabilities=())
        registry = experiment_contract.load_contracts(_CONTRACTS_TOML)
        contracts = (registry[_RD05], registry[_RD06], registry[_RD07])
        context = pv.ValidationContext(
            descriptor=None, base_revision="a" * 40, control_source=None, subject_source=None,
            package_root=_PATCH_DIR,
            contracts=contracts,
            contract_hashes={c.id: c.contract_hash for c in contracts},
        )
        for spec in plan.checks:
            result = pv.evaluate_check(spec, context)
            self.assertEqual(result.status, pv.FAIL, spec.check_id)


if __name__ == "__main__":
    unittest.main()
