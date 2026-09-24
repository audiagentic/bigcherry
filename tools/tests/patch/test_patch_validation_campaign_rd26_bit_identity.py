"""PA36 migration #4 (dev-gpt-agent req_110d0729beb44d8b): RD26
decode-vs-verify bit-identity producer tests -- hardware-free.

The legacy run_rd26_decode_verify_bit_identity_check() orchestration
tests are REPLACED by direct tests of the 1210 patch-local producer
module (validation/producer.py): the legacy function, the
--run-rd26-contract CLI path, and the rd26_correctness.py module are
all DELETED from shared code in the same change. The seven measurement
scenarios are preserved 1:1 (the subprocess.run seam faked with
deterministic GGUF content per arm/mode/replicate); the
producer-specific guards (model required, one contract architecture,
one device, verify-width scope) and the result shape are new coverage
for the migrated surface.

The historical RD26 record's honest correctness FAIL is the expected
receipt for a real run against the current 2-of-5 base-standalone
subset -- these tests prove producer EQUIVALENCE (the oracle mechanics
are preserved), never RD26 qualification.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

TOOLS_ROOT = Path(__file__).resolve().parents[2]
if str(TOOLS_ROOT) not in sys.path:
    sys.path.insert(0, str(TOOLS_ROOT))

from bigcherry.experiment.attestation import ExecutionIdentity  # noqa: E402
from bigcherry.experiment import contract as experiment_contract  # noqa: E402
from bigcherry.patch import producer_support  # noqa: E402
from bigcherry.patch import validation_producer as vp  # noqa: E402
from bigcherry.patch.validation import ArtifactRef  # noqa: E402

SUBJECT_PATCH = "1210_rd26_bitidentical_decode_verify_standalone"
PATCH_DIR = TOOLS_ROOT.parent / "patches" / SUBJECT_PATCH
CONTRACT_ARCHITECTURES = ("gfx1100", "gfx1201", "gfx1030")
FAT_TARGETS = "gfx1100;gfx1201;gfx1030"

_PAIRED_IDENTITIES = {
    "control": {"build_id": "pair-control-build"},
    "subject": {"build_id": "pair-subject-build"},
}


def _load_producer() -> object:
    spec = importlib.util.spec_from_file_location(
        "_bc_rd26_producer", PATCH_DIR / "validation" / "producer.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    # The producer defines a dataclass (_RunRecord); the dataclass
    # machinery looks the module up by name in sys.modules, so it must
    # be registered before exec.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class _FakeRuntime:
    def __init__(
        self,
        *,
        run_dir: Path,
        pair: vp.ProducerBuildPair,
        device: vp.ProducerDeviceContext | None,
    ) -> None:
        self.run_dir = run_dir
        self.pair = pair
        self.device = device
        self.benchmark_calls: list[dict[str, object]] = []
        self.build_pair_calls: list[dict[str, object]] = []

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
    ) -> vp.ProducerBuildPair:
        self.build_pair_calls.append(
            {
                "targets": tuple(targets),
                "primary_target": primary_target,
                "baseline_source": baseline_source,
                "require_parity": require_parity,
            }
        )
        return self.pair

    def device_contexts(
        self,
        *,
        device_map: object,
    ) -> tuple[vp.ProducerDeviceContext, ...]:
        return (self.device,) if self.device is not None else ()

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
        device: vp.ProducerDeviceContext | None = None,
    ) -> SimpleNamespace:
        # PRBE20: the controls lane -- one 10-round tg128 decode run.
        (workload,) = workloads
        self.benchmark_calls.append({"workload": workload, "pairs": pairs, "model": model})
        return SimpleNamespace(runs={workload: SimpleNamespace(runs=[], stats={"paired_rounds": pairs})})

    def write_artifact(
        self,
        *,
        name: str,
        payload: vp.JsonObject,
    ) -> ArtifactRef:
        return self._write(name, json.dumps(payload, indent=2))

    def write_text_artifact(self, *, name: str, text: str) -> ArtifactRef:
        return self._write(name, text)

    def _write(self, name: str, text: str) -> ArtifactRef:
        path = self.run_dir / "artifacts" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(text.encode("utf-8"))
        return ArtifactRef(
            name=name,
            path=path.relative_to(self.run_dir).as_posix(),
            sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        )


def _make_pair(temp: Path) -> vp.ProducerBuildPair:
    return vp.ProducerBuildPair(
        base_revision="a" * 40,
        control_source=temp / "trees" / "control",
        subject_source=temp / "trees" / "subject",
        control_composition=(),
        subject_composition=((SUBJECT_PATCH, "c2"),),
        control_bin=temp / "pair" / "CONTROL-BIN" / "llama-results",
        subject_bin=temp / "pair" / "SUBJECT-BIN" / "llama-results",
        validation_build_identities=_PAIRED_IDENTITIES,
    )


def _make_device(architecture: str) -> vp.ProducerDeviceContext:
    return vp.ProducerDeviceContext(
        architecture=architecture,
        device_index=0,
        execution_identity=ExecutionIdentity(
            backend="ROCm",
            architectures=(architecture,),
        ),
        env_overrides={"HIP_VISIBLE_DEVICES": "0"},
        env_unset=("ROCR_VISIBLE_DEVICES",),
    )


def _fake_tree(path: Path) -> str:
    return hashlib.sha256(str(path).encode("utf-8")).hexdigest()


def _content(
    arm: str,
    mode: str,
    replicate: int,
    *,
    subject_identical: bool = True,
    control_diverges: bool = True,
    nondeterministic: bool = False,
) -> bytes:
    """Deterministic GGUF content per (arm, mode, replicate).

    - subject_identical: subject decode and verify bytes are equal
      (the RD26 invariant holds for the subject).
    - control_diverges: control decode and verify bytes differ (the
      non-vacuous effect gate).
    - nondeterministic: replicate 1's bytes differ from replicate 0's
      (same-configuration repeatability breaks).

    Every body is padded to a fixed length: the producer requires the
    decode and verify artifacts to be the SAME size (a size mismatch
    invalidates the byte comparison), so the divergent content must
    differ in bytes, not in length.
    """
    if arm == "subject":
        if subject_identical:
            body = b"SUBJECT-IDENTICAL-CONTENT"
        elif mode == "decode":
            body = b"SUBJECT-DECODE-CONTENT-00"
        else:
            body = b"SUBJECT-VERIFY-CONTENT-00"
    else:
        if control_diverges:
            body = (
                b"CONTROL-DECODE-CONTENT-00"
                if mode == "decode"
                else b"CONTROL-VERIFY-CONTENT-00"
            )
        else:
            body = b"CONTROL-IDENTICAL-CONTENT"
    if nondeterministic and replicate == 1:
        body = body[:-2] + b"01"
    return b"GGUF" + body


def _fake_run(
    *,
    subject_identical: bool = True,
    control_diverges: bool = True,
    nondeterministic: bool = False,
    returncode: int = 0,
    missing_output: bool = False,
):
    calls: list[dict[str, object]] = []

    def run(argv, **kwargs):
        args = list(argv)
        output_index = args.index("--output")
        output = Path(args[output_index + 1])
        ubatch_index = args.index("--ubatch-size")
        ubatch = int(args[ubatch_index + 1])
        executable = str(argv[0])
        arm = "control" if "CONTROL" in executable.upper() else "subject"
        mode = "decode" if ubatch == 1 else "verify"
        calls.append({"arm": arm, "mode": mode, "env": dict(kwargs.get("env") or {})})
        replicate = int(output.name.rsplit("rep", 1)[1].split(".")[0])
        if returncode == 0 and not missing_output:
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_bytes(
                _content(
                    arm,
                    mode,
                    replicate,
                    subject_identical=subject_identical,
                    control_diverges=control_diverges,
                    nondeterministic=nondeterministic,
                )
            )
        return SimpleNamespace(returncode=returncode, stdout="", stderr="fake stderr")

    return run, calls


def _scaffold_benches(temp: Path) -> dict[str, dict[str, Path]]:
    benches = {}
    for role in ("control", "subject"):
        binary = temp / "scaffold" / role / "llama-bench"
        binary.parent.mkdir(parents=True, exist_ok=True)
        binary.write_bytes(b"")
        benches[role] = {"llama-bench": binary}
    return benches


def _fake_lane_effect(outcome, *, workload, metric, role, rounds, label):
    effect = experiment_contract.LaneEffect(
        role=role, metric=metric, geometric_effect_pct=0.1, ci95_low_pct=-0.2, ci95_high_pct=0.3,
        paired_rounds=rounds, pair_ratios=(1.001,) * rounds,
    )
    return effect, outcome.runs[workload]


def _run_producer(
    module: object,
    *,
    subject_identical: bool = True,
    control_diverges: bool = True,
    nondeterministic: bool = False,
    returncode: int = 0,
    missing_output: bool = False,
    model_none: bool = False,
    with_device: bool = True,
    targets: tuple[str, ...] | None = None,
) -> tuple[vp.ProducerResult | None, _FakeRuntime, list[dict[str, object]]]:
    temp = Path(tempfile.mkdtemp())
    run_dir = temp / "run"
    run_dir.mkdir(parents=True)
    pair = _make_pair(temp)
    device = _make_device("gfx1100") if with_device else None
    runtime = _FakeRuntime(run_dir=run_dir, pair=pair, device=device)
    architecture = "gfx1100"
    if targets is None:
        targets = (architecture,)
    ctx = vp.ProducerContext(
        repo_root=TOOLS_ROOT.parent,
        patch_dir=PATCH_DIR,
        workdir=run_dir,
        campaign_id=f"{SUBJECT_PATCH}/rd26",
        base_revision="a" * 40,
        hip_path=Path("/opt/rocm"),
        fat_targets=vp.FatTargetPlan(targets=targets),
        model=None if model_none else Path("/models/m.gguf"),
        corpus=None,
        build_env={"HIP_PATH": "/opt/rocm"},
        inputs={},
        validation_build_identities={
            "control": {"build_id": "scaffold-control-build"},
            "subject": {"build_id": "scaffold-subject-build"},
        },
        patch_id=SUBJECT_PATCH,
        device_map={architecture: (0,)},
        runtime=runtime,  # type: ignore[arg-type]
        validation_binaries=_scaffold_benches(temp),
    )
    fake_run, calls = _fake_run(
        subject_identical=subject_identical,
        control_diverges=control_diverges,
        nondeterministic=nondeterministic,
        returncode=returncode,
        missing_output=missing_output,
    )
    with (
        mock.patch("subprocess.run", fake_run),
        mock.patch("bigcherry.patch.source.git_worktree_tree", _fake_tree),
        mock.patch.object(producer_support, "lane_effect", _fake_lane_effect),
    ):
        try:
            result = module.run(ctx)  # type: ignore[union-attr]
        except vp.ValidationProducerError:
            return None, runtime, calls
    assert isinstance(result, vp.ProducerResult)
    return result, runtime, calls


class Rd26BitIdentityProducerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.module = _load_producer()

    def test_subject_identity_and_control_divergence_pass_nonvacuously(self) -> None:
        result, runtime, calls = _run_producer(self.module)
        assert result is not None
        self.assertEqual(len(result.contract_correctness_results), 1)
        check = result.contract_correctness_results[0]
        self.assertEqual(check.check, "bit_identical")
        self.assertTrue(check.passed)
        assert result.correctness is not None
        self.assertEqual(result.correctness["disposition"], "passed")
        self.assertEqual(
            result.correctness["mechanism"], "rd26-decode-verify-bit-identity"
        )
        self.assertEqual(
            result.emitted_artifacts,
            frozenset({"rd26-decode-verify-bit-identity.json", "rd26-controls.json"}),
        )
        self.assertEqual(result.check_results, ())
        self.assertIsNone(result.activation_evidence)
        self.assertIsNone(result.trace_evidence)
        # PRBE20: the controls lane is bound as the performance evidence.
        assert result.performance_evidence is not None
        self.assertEqual(result.promotion_target_metric, {"RD26-DECODE-VERIFY-BIT-IDENTITY": "tg128"})
        self.assertEqual(runtime.benchmark_calls, [{"workload": "decode", "pairs": 10, "model": Path("/models/m.gguf")}])
        # The pair is built once, fat multi-arch, parity asserted, and
        # every arm/mode/replicate ran exactly as the oracle specifies.
        self.assertEqual(len(runtime.build_pair_calls), 1)
        self.assertEqual(runtime.build_pair_calls[0]["targets"], CONTRACT_ARCHITECTURES)
        self.assertTrue(runtime.build_pair_calls[0]["require_parity"])
        self.assertEqual(runtime.build_pair_calls[0]["primary_target"], "llama-results")
        # 2 arms x 2 modes x 2 replicates = 8 subprocess runs.
        self.assertEqual(len(calls), 8)
        self.assertEqual({c["arm"] for c in calls}, {"control", "subject"})
        self.assertEqual({c["mode"] for c in calls}, {"decode", "verify"})

    def test_subject_divergence_fails_even_when_control_diverges(self) -> None:
        result, _, _ = _run_producer(self.module, subject_identical=False)
        assert result is not None
        check = result.contract_correctness_results[0]
        self.assertFalse(check.passed)
        assert result.correctness is not None
        self.assertEqual(result.correctness["disposition"], "failed")
        self.assertIn("first_file_byte_mismatch", check.detail)

    def test_control_identity_fails_nonvacuous_effect_gate(self) -> None:
        result, _, _ = _run_producer(self.module, control_diverges=False)
        assert result is not None
        check = result.contract_correctness_results[0]
        self.assertFalse(check.passed)
        self.assertIn("non-authoritative", check.detail)

    def test_same_configuration_nondeterminism_is_hard_error(self) -> None:
        result, _, _ = _run_producer(self.module, nondeterministic=True)
        self.assertIsNone(result)

    def test_llama_results_process_failure_is_hard_error(self) -> None:
        result, _, _ = _run_producer(self.module, returncode=1)
        self.assertIsNone(result)

    def test_missing_output_is_hard_error(self) -> None:
        result, _, _ = _run_producer(self.module, missing_output=True)
        self.assertIsNone(result)

    def test_verify_width_over_rd26_scope_fails_before_build(self) -> None:
        self.module._SPEC_DRAFT_N_MAX = 9  # type: ignore[attr-defined]
        result, runtime, _ = _run_producer(self.module)
        self.assertIsNone(result)
        # The scope guard fires BEFORE any build.
        self.assertEqual(runtime.build_pair_calls, [])
        self.module._SPEC_DRAFT_N_MAX = 4  # type: ignore[attr-defined]

    def test_model_required_fails_before_build(self) -> None:
        result, runtime, _ = _run_producer(self.module, model_none=True)
        self.assertIsNone(result)
        self.assertEqual(runtime.build_pair_calls, [])

    def test_single_contract_architecture_guard(self) -> None:
        result, _, _ = _run_producer(self.module, targets=("gfx1100", "gfx1030"))
        self.assertIsNone(result)

    def test_device_selection_guard(self) -> None:
        result, _, _ = _run_producer(self.module, with_device=False)
        self.assertIsNone(result)


if __name__ == "__main__":
    unittest.main()
