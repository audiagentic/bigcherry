"""PA36 migration #3 (dev-gpt-agent req_110d0729beb44d8b): RD13
backend_reference producer tests -- hardware-free.

The legacy run_rd13_backend_reference_check() orchestration tests are
REPLACED by direct tests of the 1206 patch-local producer module
(validation/producer.py): the legacy function, its helper trio, the
_RD13_BACKEND_REFERENCE_* constants, and the --run-rd13-contract CLI
path are all DELETED from shared code in the same change. The four
measurement scenarios are preserved 1:1 (a small fake vocabulary via
the producer's module constants, mirroring the legacy test's
vocab_size=3/n_predict=2 parameters); the producer-specific guards
(model required, one contract architecture, one device) and the
preserved env sanitization (stale GGML_CUDA_DISABLE_FUSION) are new
coverage for the migrated surface.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

TOOLS_ROOT = Path(__file__).resolve().parents[2]
if str(TOOLS_ROOT) not in sys.path:
    sys.path.insert(0, str(TOOLS_ROOT))

from bigcherry.experiment.attestation import ExecutionIdentity  # noqa: E402
from bigcherry.patch import validation_producer as vp  # noqa: E402
from bigcherry.patch.validation import ArtifactRef  # noqa: E402

SUBJECT_PATCH = "1206_rd13_mul_mat_add_view_fusion"
PATCH_DIR = TOOLS_ROOT.parent / "patches" / SUBJECT_PATCH
CONTRACT_ARCHITECTURES = ("gfx1100", "gfx1201", "gfx1030")
FAT_TARGETS = "gfx1100;gfx1201;gfx1030"

_PAIRED_IDENTITIES = {
    "control": {"build_id": "pair-control-build"},
    "subject": {"build_id": "pair-subject-build"},
}


def _load_producer() -> object:
    spec = importlib.util.spec_from_file_location(
        "_bc_rd13_producer", PATCH_DIR / "validation" / "producer.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _FakeAttestation:
    def __init__(self, arm: str) -> None:
        self.arm = arm

    def document(self) -> dict[str, object]:
        return {"schema_version": 1, "backend": "ROCm", "arm": self.arm}


class _FakeSession:
    """Structural AttestedServerSession stand-in: arm from the binary,
    per-arm base URL (the fake urlopen routes on it), attestation on
    enter, and the exact env the producer passed (the test asserts the
    stale fusion-disable sanitization)."""

    def __init__(self, **kwargs: object) -> None:
        self.binary = Path(str(kwargs["binary"]))
        self.arm = "control" if "CONTROL" in str(self.binary).upper() else "subject"
        self.base_url = f"http://{self.arm}.invalid"
        self.env_overrides = dict(kwargs["env_overrides"])  # type: ignore[arg-type]
        self.env_unset = tuple(kwargs["env_unset"])  # type: ignore[arg-type]
        self.expected = kwargs.get("expected")
        self.architecture_by_locator = kwargs.get("architecture_by_locator")
        self.attestation: _FakeAttestation | None = None

    def __enter__(self) -> _FakeSession:
        self.attestation = _FakeAttestation(self.arm)
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        pass


class _FakeResponse:
    def __init__(self, lines: list[bytes]) -> None:
        self._lines = lines

    def __enter__(self) -> _FakeResponse:
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        pass

    def __iter__(self) -> object:
        return iter(self._lines)


def _sse_lines(rows: list[dict[str, object]]) -> list[bytes]:
    lines = [
        f"data: {json.dumps({'stop': False, 'completion_probabilities': [row]})}\n".encode()
        for row in rows
    ]
    lines.append(b"data: [DONE]\n")
    return lines


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
        self.build_pair_calls: list[dict[str, object]] = []
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
    ) -> vp.ProducerPairedBenchmarkOutcome:
        self.benchmark_calls.append(
            {
                "control_binary": control_binary,
                "subject_binary": subject_binary,
                "model": model,
                "workloads": workloads,
                "pairs": pairs,
                "log_context": log_context,
                "device": device,
            }
        )
        lane = types.SimpleNamespace(
            stats={
                "geometric_effect_pct": 0.8,
                "ci95_low_pct": 0.6,
                "ci95_high_pct": 1.0,
                "paired_rounds": 10,
                "pair_ratios": [1.01] * 10,
            },
            runs=[{"pair": i, "control": 10.0, "subject": 10.1} for i in range(10)],
        )
        return vp.ProducerPairedBenchmarkOutcome(
            runs={"decode": lane},
            commands={
                "decode": {
                    "control": (str(control_binary),),
                    "subject": (str(subject_binary),),
                }
            },
            raw_logs=(),
        )

    def run_trace_probe(
        self,
        *,
        binary: Path,
        model: Path,
        device: vp.ProducerDeviceContext,
        bench_prompt: int,
        bench_gen: int,
        log_context: str,
        disable_fusion: bool = False,
    ) -> str:
        return (
            "BIGCHERRY_PATCH_HIT patch=1206_rd13 path=mul_mat_add_view_fusion_f\n"
            if "SUBJECT" in str(binary)
            else ""
        )

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
        control_bin=temp / "pair" / "CONTROL-BIN" / "llama-server",
        subject_bin=temp / "pair" / "SUBJECT-BIN" / "llama-server",
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
        locator="0000:03:00.0",
    )


def _fake_tree(path: Path) -> str:
    return hashlib.sha256(str(path).encode("utf-8")).hexdigest()


def _row(generated_id: int, values: list[float]) -> dict[str, object]:
    return {
        "id": generated_id,
        "top_logprobs": [
            {"id": token_id, "logprob": value} for token_id, value in enumerate(values)
        ],
    }


def _run_producer(
    module: object,
    *,
    control_rows: list[dict[str, object]],
    subject_rows: list[dict[str, object]],
    model: Path | None = None,
    with_device: bool = True,
    build_env: dict[str, str] | None = None,
    targets: tuple[str, ...] | None = None,
    include_control_model: bool = True,
    control_model_name: str = "gpt-oss-20b-UD-Q6_K_XL.gguf",
) -> tuple[vp.ProducerResult, _FakeRuntime, list[_FakeSession]]:
    temp = Path(tempfile.mkdtemp())
    run_dir = temp / "run"
    run_dir.mkdir(parents=True)
    pair = _make_pair(temp)
    device = _make_device("gfx1100") if with_device else None
    runtime = _FakeRuntime(run_dir=run_dir, pair=pair, device=device)
    control_model = temp / control_model_name
    control_model.write_bytes(b"control-model")
    model_registry = temp / "models.toml"
    model_registry.write_text(
        """version = 1

[[models]]
 id = 'tierM-gptoss20b-q6k'
 path = 'gpt-oss-20b-UD-Q6_K_XL.gguf'
 size-bytes = {size}
""".replace("{size}", str(control_model.stat().st_size)),
        encoding="utf-8",
    )
    bench_control = temp / "scaffold" / "CONTROL" / "llama-bench"
    bench_subject = temp / "scaffold" / "SUBJECT" / "llama-bench"
    bench_control.parent.mkdir(parents=True)
    bench_subject.parent.mkdir(parents=True)
    bench_control.write_bytes(b"control-bench")
    bench_subject.write_bytes(b"subject-bench")
    architecture = "gfx1100"
    if targets is None:
        targets = (architecture,)
    ctx = vp.ProducerContext(
        repo_root=TOOLS_ROOT.parent,
        patch_dir=PATCH_DIR,
        workdir=run_dir,
        campaign_id=f"{SUBJECT_PATCH}/rd13",
        base_revision="a" * 40,
        hip_path=Path("/opt/rocm"),
        fat_targets=vp.FatTargetPlan(targets=targets),
        model=model if model is not None else Path("/models/m.gguf"),
        corpus=None,
        build_env=build_env if build_env is not None else {"HIP_PATH": "/opt/rocm"},
        inputs={"control_model": str(control_model)} if include_control_model else {},
        validation_build_identities={
            "control": {"build_id": "scaffold-control-build"},
            "subject": {"build_id": "scaffold-subject-build"},
        },
        patch_id=SUBJECT_PATCH,
        device_map={architecture: (0,)},
        runtime=runtime,  # type: ignore[arg-type]
        validation_binaries={
            "control": {"llama-bench": bench_control},
            "subject": {"llama-bench": bench_subject},
        },
    )
    # The small fake vocabulary (the legacy test's vocab_size=3 /
    # n_predict=2 parameters, now module constants).
    module._VOCAB_SIZE = 3  # type: ignore[attr-defined]
    module._N_PREDICT = 2  # type: ignore[attr-defined]

    sessions: list[_FakeSession] = []

    def urlopen(request: object, timeout: object = None) -> _FakeResponse:
        url = str(getattr(request, "full_url", ""))
        if "control" not in url and "subject" not in url:
            raise AssertionError(f"unexpected completion URL: {url!r}")
        rows = control_rows if "control" in url else subject_rows
        body = json.loads(getattr(request, "data", b"{}"))
        stream = body.get("stream")
        if not isinstance(stream, bool) or not stream:
            raise AssertionError("RD13 backend_reference must use streaming completion")
        post_sampling = body.get("post_sampling_probs")
        if not isinstance(post_sampling, bool) or post_sampling:
            raise AssertionError(
                "RD13 backend_reference must compare pre-sampling logprobs"
            )
        if body.get("n_probs") != 3:
            raise AssertionError("test expected the fake full vocabulary")
        return _FakeResponse(_sse_lines(rows))

    def session_factory(**kwargs: object) -> _FakeSession:
        session = _FakeSession(**kwargs)
        sessions.append(session)
        return session

    with (
        mock.patch("urllib.request.urlopen", side_effect=urlopen),
        mock.patch(
            "bigcherry.experiment.server_execution.AttestedServerSession",
            side_effect=session_factory,
        ),
        mock.patch("bigcherry.patch.source.git_worktree_tree", _fake_tree),
        mock.patch("bigcherry.core.paths.MODELS", model_registry),
    ):
        result = module.run(ctx)  # type: ignore[union-attr]
    assert isinstance(result, vp.ProducerResult)
    return result, runtime, sessions


class Rd13BackendReferenceProducerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.module = _load_producer()

    def test_within_tolerance_passes_and_binds_full_vocab_summary(self) -> None:
        result, runtime, sessions = _run_producer(
            self.module,
            control_rows=[
                _row(1, [-1.0, -2.0, -3.0]),
                _row(2, [-1.1, -2.1, -3.1]),
            ],
            subject_rows=[
                _row(1, [-1.0001, -2.0, -3.0]),
                _row(2, [-1.1, -2.1002, -3.1]),
            ],
        )
        self.assertEqual(len(result.contract_correctness_results), 1)
        check = result.contract_correctness_results[0]
        self.assertEqual(check.check, "backend_reference")
        self.assertTrue(check.passed)
        assert result.correctness is not None
        self.assertEqual(result.correctness["disposition"], "passed")
        self.assertEqual(
            result.correctness["mechanism"], "rd13-full-vocab-backend-reference"
        )
        self.assertEqual(
            result.emitted_artifacts,
            frozenset(
                {
                    "rd13-backend-reference.json",
                    "rd13-performance.json",
                    "rd13-subject-trace.log",
                    "rd13-control-trace.log",
                }
            ),
        )
        self.assertEqual(result.check_results, ())
        self.assertIsNotNone(result.activation_evidence)
        self.assertIsNotNone(result.performance_evidence)
        self.assertIsNotNone(result.trace_evidence)
        self.assertEqual(
            result.validation_build_identities,
            {
                "control": {"build_id": "scaffold-control-build"},
                "subject": {"build_id": "scaffold-subject-build"},
            },
        )
        self.assertEqual(len(runtime.benchmark_calls), 2)
        self.assertEqual(
            [Path(str(call["model"])).name for call in runtime.benchmark_calls],
            ["m.gguf", "gpt-oss-20b-UD-Q6_K_XL.gguf"],
        )
        self.assertEqual([call["pairs"] for call in runtime.benchmark_calls], [10, 10])
        self.assertEqual(
            result.promotion_target_metric,
            {"RD13-MUL-MAT-ADD-VIEW-FUSION": "tg128"},
        )
        # The pair is built once, fat multi-arch, parity asserted.
        self.assertEqual(len(runtime.build_pair_calls), 1)
        self.assertEqual(runtime.build_pair_calls[0]["targets"], CONTRACT_ARCHITECTURES)
        self.assertTrue(runtime.build_pair_calls[0]["require_parity"])
        # Both arms run against the one real device's sanctioned env.
        self.assertEqual(len(sessions), 2)
        self.assertEqual({session.arm for session in sessions}, {"control", "subject"})

    def test_numeric_delta_over_tolerance_is_a_correctness_failure(self) -> None:
        result, _, _ = _run_producer(
            self.module,
            control_rows=[
                _row(1, [-1.0, -2.0, -3.0]),
                _row(2, [-1.0, -2.0, -3.0]),
            ],
            subject_rows=[
                _row(1, [-1.0, -2.001, -3.0]),
                _row(2, [-1.0, -2.0, -3.0]),
            ],
        )
        check = result.contract_correctness_results[0]
        self.assertFalse(check.passed)
        assert result.correctness is not None
        self.assertEqual(result.correctness["disposition"], "failed")
        self.assertIn("tolerance", check.detail)

    def test_generated_token_divergence_is_a_correctness_failure(self) -> None:
        result, _, _ = _run_producer(
            self.module,
            control_rows=[
                _row(1, [-1.0, -2.0, -3.0]),
                _row(2, [-1.0, -2.0, -3.0]),
            ],
            subject_rows=[
                _row(2, [-1.0, -2.0, -3.0]),
                _row(2, [-1.0, -2.0, -3.0]),
            ],
        )
        check = result.contract_correctness_results[0]
        self.assertFalse(check.passed)
        self.assertIn("diverged at step 0", check.detail)

    def test_incomplete_full_vocab_response_fails_closed(self) -> None:
        with self.assertRaises(vp.ValidationProducerError):
            _run_producer(
                self.module,
                control_rows=[
                    _row(1, [-1.0, -2.0]),
                    _row(2, [-1.0, -2.0, -3.0]),
                ],
                subject_rows=[
                    _row(1, [-1.0, -2.0, -3.0]),
                    _row(2, [-1.0, -2.0, -3.0]),
                ],
            )

    def test_model_required_fails_before_build(self) -> None:
        # model=None must fail BEFORE any build: run through the guard
        # path directly.
        temp = Path(tempfile.mkdtemp())
        run_dir = temp / "run"
        run_dir.mkdir(parents=True)
        pair = _make_pair(temp)
        runtime = _FakeRuntime(
            run_dir=run_dir, pair=pair, device=_make_device("gfx1100")
        )
        ctx = vp.ProducerContext(
            repo_root=TOOLS_ROOT.parent,
            patch_dir=PATCH_DIR,
            workdir=run_dir,
            campaign_id=f"{SUBJECT_PATCH}/rd13",
            base_revision="a" * 40,
            hip_path=Path("/opt/rocm"),
            fat_targets=vp.FatTargetPlan(targets=("gfx1100",)),
            model=None,
            corpus=None,
            build_env={"HIP_PATH": "/opt/rocm"},
            inputs={},
            validation_build_identities={},
            patch_id=SUBJECT_PATCH,
            device_map={"gfx1100": (0,)},
            runtime=runtime,  # type: ignore[arg-type]
            validation_binaries={},
        )
        self.module._VOCAB_SIZE = 3  # type: ignore[attr-defined]
        self.module._N_PREDICT = 2  # type: ignore[attr-defined]
        with self.assertRaises(vp.ValidationProducerError):
            self.module.run(ctx)  # type: ignore[union-attr]
        self.assertEqual(runtime.build_pair_calls, [])

    def test_control_model_input_is_required_before_build(self) -> None:
        with self.assertRaises(vp.ValidationProducerError):
            _run_producer(
                self.module,
                control_rows=[
                    _row(1, [-1.0, -2.0, -3.0]),
                    _row(2, [-1.0, -2.0, -3.0]),
                ],
                subject_rows=[
                    _row(1, [-1.0, -2.0, -3.0]),
                    _row(2, [-1.0, -2.0, -3.0]),
                ],
                include_control_model=False,
            )

    def test_control_model_registry_basename_is_enforced(self) -> None:
        with self.assertRaises(vp.ValidationProducerError):
            _run_producer(
                self.module,
                control_rows=[
                    _row(1, [-1.0, -2.0, -3.0]),
                    _row(2, [-1.0, -2.0, -3.0]),
                ],
                subject_rows=[
                    _row(1, [-1.0, -2.0, -3.0]),
                    _row(2, [-1.0, -2.0, -3.0]),
                ],
                control_model_name="wrong-control-model.gguf",
            )

    def test_single_contract_architecture_guard(self) -> None:
        with self.assertRaises(vp.ValidationProducerError):
            _run_producer(
                self.module,
                control_rows=[_row(1, [-1.0, -2.0, -3.0]), _row(2, [-1.0, -2.0, -3.0])],
                subject_rows=[_row(1, [-1.0, -2.0, -3.0]), _row(2, [-1.0, -2.0, -3.0])],
                targets=("gfx1100", "gfx1030"),
            )

    def test_device_selection_guard(self) -> None:
        with self.assertRaises(vp.ValidationProducerError):
            _run_producer(
                self.module,
                control_rows=[_row(1, [-1.0, -2.0, -3.0]), _row(2, [-1.0, -2.0, -3.0])],
                subject_rows=[_row(1, [-1.0, -2.0, -3.0]), _row(2, [-1.0, -2.0, -3.0])],
                with_device=False,
            )

    def test_locator_backed_server_attestation_identity(self) -> None:
        # PA36 RD13/1206 migration (GPT req_760c0fe82d7b4609 BLOCKER): the
        # llama-server attestation channel derives the server architecture
        # ONLY via a verified locator->arch mapping, so the producer must
        # construct a locator-bearing ExecutionIdentity (not the
        # locator-less device.execution_identity) and the {locator: arch}
        # mapping from the verified physical locator.
        _result, _runtime, sessions = _run_producer(
            self.module,
            control_rows=[_row(1, [-1.0, -2.0, -3.0]), _row(2, [-1.0, -2.0, -3.0])],
            subject_rows=[_row(1, [-1.0, -2.0, -3.0]), _row(2, [-1.0, -2.0, -3.0])],
        )
        self.assertEqual(len(sessions), 2)
        for session in sessions:
            expected = session.expected
            self.assertIsInstance(expected, ExecutionIdentity)
            assert isinstance(expected, ExecutionIdentity)
            self.assertEqual(expected.locators, ("0000:03:00.0",))
            self.assertEqual(expected.architectures, ("gfx1100",))
            self.assertEqual(
                session.architecture_by_locator,
                {"0000:03:00.0": "gfx1100"},
            )

    def test_stale_ambient_fusion_disable_is_sanitized(self) -> None:
        _, _, sessions = _run_producer(
            self.module,
            control_rows=[_row(1, [-1.0, -2.0, -3.0]), _row(2, [-1.0, -2.0, -3.0])],
            subject_rows=[_row(1, [-1.0, -2.0, -3.0]), _row(2, [-1.0, -2.0, -3.0])],
            build_env={
                "HIP_PATH": "/opt/rocm",
                "GGML_CUDA_DISABLE_FUSION": "1",
            },
        )
        self.assertEqual(len(sessions), 2)
        for session in sessions:
            self.assertNotIn("GGML_CUDA_DISABLE_FUSION", session.env_overrides)
            self.assertIn("GGML_CUDA_DISABLE_FUSION", session.env_unset)
            # The sanctioned device selector env is still present.
            self.assertIn("HIP_VISIBLE_DEVICES", session.env_overrides)


if __name__ == "__main__":
    unittest.main()
