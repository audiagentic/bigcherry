"""PA36-F step 7: the synthetic zero-shared-change producer fixture (GPT
design req_8ec9b90c05f84a30, section 7).

Proves end to end, with ZERO shared implementation-code change, that a
brand-new patch-local validation producer can be declared, selected,
executed, validated, and evidence-bound through
``execute_validation_producer()``:

    resolve manifest -> input validation -> ProducerContext -> dynamic
    producer import -> producer execution -> typed result validation ->
    evaluate the remaining shared (unscoped) check -> compute_verdict ->
    make_record -> record has two canonical contracts -> each
    contract_verdict has bool passed -> the declared synthetic artifact is
    hashed -> an undeclared adjacent artifact is absent.

Then the same fixture patch directory is copied to a second, fresh patch
id and re-run through the identical, unmodified generic executor -- the
concrete proof that adding a normal producer requires no shared code
edit/registry/branch (PA36-F's actual load-bearing claim).
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

TOOLS_DIR = Path(__file__).resolve().parents[2]
REPO_ROOT = TOOLS_DIR.parent
sys.path.insert(0, str(TOOLS_DIR))

from bigcherry.experiment import contract as experiment_contract  # noqa: E402
from bigcherry.patch import evidence as patch_validation_evidence  # noqa: E402
from bigcherry.patch import validation as pv  # noqa: E402
from bigcherry.patch import validation_campaign as vc  # noqa: E402
from bigcherry.patch.campaign import producer as campaign_producer  # noqa: E402
from bigcherry.patch import validation_producer as vp  # noqa: E402

_FIXTURE_ROOT = TOOLS_DIR / "tests" / "fixtures" / "validation_producer"
_FIXTURE_PATCH_DIR = _FIXTURE_ROOT / "patches" / "0000_synthetic_validation_producer"
_FIXTURE_CONTRACTS = _FIXTURE_ROOT / "experiment-contracts.toml"


class _FakeProducerRuntime:
    """A minimal, fully offline ``ProducerRuntime`` -- the design's own
    words for step 7: producer.py "uses ctx.runtime.write_artifact() ...
    supplied by the fake runtime". Never touches
    ``CampaignProducerRuntime``/``validation_campaign``'s real build
    machinery -- this fixture proves the GENERIC DISPATCHER seam, not a
    real compile."""

    def __init__(self, run_dir: Path) -> None:
        self.run_dir = run_dir

    def write_artifact(self, *, name: str, payload) -> pv.ArtifactRef:
        target = self.run_dir / "artifacts" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        encoded = json.dumps(payload, sort_keys=True).encode("utf-8")
        target.write_bytes(encoded)
        return pv.ArtifactRef(
            name=name, path=f"artifacts/{name}", sha256=hashlib.sha256(encoded).hexdigest(),
        )

    def build_pair(self, **_kwargs):  # pragma: no cover - not exercised
        raise NotImplementedError("synthetic fixture never builds")

    def device_contexts(self, **_kwargs):  # pragma: no cover - not exercised
        raise NotImplementedError("synthetic fixture never selects devices")

    def run_paired_llama_benchmark(self, **_kwargs):  # pragma: no cover - not exercised
        raise NotImplementedError("synthetic fixture never benchmarks")


def _fake_build_identity(role: str) -> dict[str, object]:
    return {
        "effective_build_id": f"{role}-build-id",
        "compile_verification_id": f"{role}-compile-verification",
        "compile_commands_digest": f"{role}-compile-commands-digest",
        "hip_compile_commands_digest": f"{role}-hip-compile-commands-digest",
        "runtime_bundle_hash": f"{role}-runtime-bundle-hash",
        "runtime_artifacts": {f"{role}.bin": "a" * 64},
    }


def _run_synthetic_producer(patch_dir: Path, *, patch_id: str, run_dir: Path):
    """The generic, PATCH-AGNOSTIC drive sequence: everything here works
    identically for ANY patch_dir/patch_id -- nothing below names
    "0000_synthetic_validation_producer" -- which is exactly what the
    step-7 copy-to-a-second-patch-id proof (below) exercises."""
    checks = pv.parse_validation_toml(patch_dir / "validation.toml", patch_id=patch_id)
    plan = pv.ValidationPlan(patch_id=patch_id, checks=checks, universal_capabilities=())

    registry = experiment_contract.load_contracts(_FIXTURE_CONTRACTS)
    contracts = (registry["SYN-C1"], registry["SYN-C2"])
    context = pv.ValidationContext(
        descriptor=None, base_revision="a" * 40, control_source=None, subject_source=None,
        package_root=patch_dir,
        contracts=contracts,
        contract_hashes={c.id: c.contract_hash for c in contracts},
    )

    run_dir.mkdir(parents=True, exist_ok=True)
    producer_context = vp.ProducerContext(
        repo_root=REPO_ROOT, patch_dir=patch_dir, workdir=run_dir,
        campaign_id=f"{patch_id}/synthetic", base_revision="a" * 40,
        hip_path=Path("/hip"), fat_targets=vp.FatTargetPlan(targets=("gfx1100",)),
        model=None, corpus=None, build_env={}, inputs={},
        validation_build_identities={}, patch_id=patch_id, device_map={},
        runtime=_FakeProducerRuntime(run_dir),
    )

    execution = campaign_producer.execute_validation_producer(
        patch_dir=patch_dir, producer_id="synthetic",
        provided_inputs={"token": "synthetic-token-value"},
        producer_context=producer_context, validation_plan=plan,
        validation_context=context,
        correctness_evidence_requested=False, performance_benchmark_requested=False,
    )

    record = patch_validation_evidence.make_record(
        patch_id=patch_id, patch_path=patch_dir / "patch.toml",
        patch_implementation_digest="deadbeef" * 8,
        base_ref="pinned-ref", base_revision="a" * 40,
        framework_baseline_digest="b" * 64,
        patched_source_tree="c" * 40,
        gpu_architectures="gfx1100",
        activation_evidence=None, activation_disposition=None, correctness=None,
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
    return execution, record


class SyntheticValidationProducerExecutionTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory(prefix="pa36f-step7-")
        self.addCleanup(self._tmp.cleanup)
        self.run_dir = Path(self._tmp.name) / "run"

    def test_synthetic_producer_executes_and_binds_plural_contract_evidence(self) -> None:
        execution, record = _run_synthetic_producer(
            _FIXTURE_PATCH_DIR, patch_id="0000_synthetic_validation_producer",
            run_dir=self.run_dir,
        )

        # resolve manifest -> ... -> typed result validation: no exception
        # raised means resolve_producer()/validate_producer_inputs()/
        # validate_producer_cli_compatibility()/validate_producer_result()
        # all succeeded for a producer NONE of the shared modules know by
        # name.
        self.assertEqual(execution.selection.spec.producer_id, "synthetic")
        self.assertEqual(execution.selection.spec.patch_id, "0000_synthetic_validation_producer")

        # evaluate the remaining shared (unscoped) check via evaluate_check()
        # fallback -- proves execute_validation_producer()'s step 9 "else"
        # branch, not just the producer-supplied path.
        self.assertIn("syn-universal-smoke", execution.evaluated)
        self.assertEqual(execution.evaluated["syn-universal-smoke"].status, pv.PASS)

        # compute_verdict(): all three required checks passed.
        self.assertTrue(execution.verdict.eligible, execution.verdict.reasons)

        # record has two canonical contracts.
        self.assertEqual(
            {c["id"] for c in record["contracts"]}, {"SYN-C1", "SYN-C2"},
        )
        # each contract_verdict has bool passed.
        self.assertEqual(set(record["contract_verdicts"]), {"SYN-C1", "SYN-C2"})
        for verdict in record["contract_verdicts"].values():
            self.assertIs(verdict["passed"], True)

        # the declared synthetic artifact is hashed.
        artifact_paths = {a["path"] for a in record["campaign_artifacts"]}
        self.assertIn("artifacts/synthetic-correctness.json", artifact_paths)
        self.assertIn("artifacts/synthetic-correctness.json", record["artifact_hashes"])

        # an undeclared adjacent artifact -- dropped in the SAME artifacts/
        # directory by something other than the producer's own declared
        # write_artifact() call -- is absent from the evidence, proving
        # PA36-F step 4's declarative allowlist actually gates it.
        sneaky = self.run_dir / "artifacts" / "sneaky.json"
        sneaky.write_text("{}", encoding="utf-8")
        _, record_with_sneak = _run_synthetic_producer(
            _FIXTURE_PATCH_DIR, patch_id="0000_synthetic_validation_producer",
            run_dir=self.run_dir,
        )
        sneaky_paths = {a["path"] for a in record_with_sneak["campaign_artifacts"]}
        self.assertNotIn("artifacts/sneaky.json", sneaky_paths)
        self.assertNotIn("artifacts/sneaky.json", record_with_sneak["artifact_hashes"])
        self.assertIn("artifacts/synthetic-correctness.json", sneaky_paths)

    def test_second_fresh_patch_id_runs_through_the_same_unmodified_executor(self) -> None:
        """PA36-F step 7's concrete zero-shared-change proof: copy the
        fixture patch directory to a brand-new patch id, change only
        patch-local manifest/module/check content (the token input value
        and the synthetic artifact payload happen to differ too, but no
        shared code path changes), and run it through the exact same
        generic executor with no edit to validation_campaign.py/
        validation.py/validation_producer.py/evidence.py."""
        second_patch_id = "0001_synthetic_validation_producer_two"
        second_patch_dir = Path(self._tmp.name) / "patches" / second_patch_id
        shutil.copytree(_FIXTURE_PATCH_DIR, second_patch_dir)

        # Change only patch-local content: the patch's own declared id.
        toml_path = second_patch_dir / "patch.toml"
        toml_path.write_text(
            toml_path.read_text(encoding="utf-8").replace(
                "0000_synthetic_validation_producer", second_patch_id,
            ),
            encoding="utf-8",
        )

        execution, record = _run_synthetic_producer(
            second_patch_dir, patch_id=second_patch_id, run_dir=self.run_dir / "second",
        )

        self.assertTrue(execution.verdict.eligible, execution.verdict.reasons)
        self.assertEqual(record["patch_id"], second_patch_id)
        self.assertEqual(
            {c["id"] for c in record["contracts"]}, {"SYN-C1", "SYN-C2"},
        )
        for verdict in record["contract_verdicts"].values():
            self.assertIs(verdict["passed"], True)


if __name__ == "__main__":
    unittest.main()
