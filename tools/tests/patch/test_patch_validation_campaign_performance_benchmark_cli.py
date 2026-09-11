"""PVPS02 step 4: --run-performance-benchmark CLI wiring. The parser-level
argument contract is directly testable (no hardware needed, main() exits
via parser.error() before touching a build). Real end-to-end behavior
(actual cmake builds, actual llama-bench execution) is covered by
PVPS02's real-hardware merge gate on Brutus, consistent with the
project's established convention for this class of function (see
Rd04CliWiringTests in test_patch_validation_campaign_va04.py).

OrchestrationLogicTests below is a real, deliberate exception to that
"don't mock the whole pipeline" convention: a first real-hardware smoke
run of this exact CLI on Brutus (2026-09-11) hit a plain NameError
(require_device_visibility used without being imported) that every
test in this file at the time -- all either parser-only or source-
inspection-only -- was structurally incapable of catching, because none
of them actually called _run_performance_benchmark() far enough to
reach that line. Mocking build_tree()/run_paired_llama_benchmark()/
source materialization is exactly narrow enough to exercise the real
orchestration logic (device resolution, ExecutionIdentity construction,
the require_device_visibility() call itself, skip-cell handling)
hardware-free, without asserting on cmake/llama-bench output shape --
that stays the real-hardware gate's job.
"""

from __future__ import annotations

import contextlib
import json
import sys
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from bigcherry.patch import validation_campaign as vc  # noqa: E402


class PerformanceBenchmarkArgParsingTests(unittest.TestCase):
    def _parse_or_error(self, argv: list[str]) -> str:
        """main() calls parser.error() (SystemExit(2) + a usage message to
        stderr) on a contract violation -- capture that message instead of
        letting the real exit propagate."""
        import argparse
        import io
        import contextlib

        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr):
            with self.assertRaises(SystemExit):
                vc.main(argv)
        return stderr.getvalue()

    def test_missing_model_root_is_rejected(self) -> None:
        message = self._parse_or_error([
            "--patch", "1202_rd04_bf16_flash_attn_tile", "--run-performance-benchmark",
            "--hip-path", "H:/fake", "--workdir", "H:/fake-workdir",
            "--device-map", "gfx1100=0",
        ])
        self.assertIn("requires --model-root and --device-map", message)

    def test_missing_device_map_is_rejected(self) -> None:
        message = self._parse_or_error([
            "--patch", "1202_rd04_bf16_flash_attn_tile", "--run-performance-benchmark",
            "--hip-path", "H:/fake", "--workdir", "H:/fake-workdir",
            "--model-root", "H:/fake-models",
        ])
        self.assertIn("requires --model-root and --device-map", message)

    def test_mutually_exclusive_with_legacy_rd04_mode(self) -> None:
        message = self._parse_or_error([
            "--patch", "1202_rd04_bf16_flash_attn_tile", "--run-performance-benchmark",
            "--run-rd04-benchmark",
            "--hip-path", "H:/fake", "--workdir", "H:/fake-workdir",
            "--model-root", "H:/fake-models", "--device-map", "gfx1100=0",
        ])
        self.assertIn("mutually exclusive with the legacy RD modes", message)

    def test_does_not_require_model_manifest_or_amdgpu_targets(self) -> None:
        # Should get PAST arg validation (i.e. NOT hit parser.error()) and
        # into real dispatch, which then fails for an unrelated real-
        # hardware reason (no real ROCm toolchain here) -- proving the
        # legacy --model/--manifest/--amdgpu-targets gate was actually
        # bypassed for this mode, not just given defaults. Real source
        # materialization does run (this is a real-integration entry
        # point, not mocked) -- --worktree-root/--workdir are pointed at
        # an isolated tempdir so this does not pollute the real
        # C:\bc-worktrees content-addressed cache.
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            try:
                vc.main([
                    "--patch", "1202_rd04_bf16_flash_attn_tile", "--run-performance-benchmark",
                    "--hip-path", str(Path(__file__).resolve().parent),
                    "--workdir", str(tmp_path / "workdir"),
                    "--worktree-root", str(tmp_path / "worktrees"),
                    "--model-root", "H:/fake-models", "--device-map", "gfx1100=0",
                ])
            except SystemExit:
                self.fail("--run-performance-benchmark should not hit parser.error() here")
            except Exception:
                pass  # expected: fails later for a real, unrelated (non-arg-parsing) reason

    def test_legacy_mode_still_requires_amdgpu_targets(self) -> None:
        # Regression check: removing --amdgpu-targets's required=True (so
        # --run-performance-benchmark can omit it) must not silently make
        # it optional for the LEGACY single-architecture flow too.
        message = self._parse_or_error([
            "--patch", "1202_rd04_bf16_flash_attn_tile",
            "--hip-path", "H:/fake", "--workdir", "H:/fake-workdir",
            "--model", "H:/fake.gguf", "--manifest", "H:/fake-manifest.json",
        ])
        self.assertIn("--amdgpu-targets", message)


class PerformanceBenchmarkDispatchWiringTests(unittest.TestCase):
    """Source-inspection tests for the real-hardware integration path
    itself, matching the project's established convention for this class
    of function (see this file's own module docstring)."""

    def setUp(self) -> None:
        import inspect
        self.run_source = inspect.getsource(vc.run)
        self.impl_source = inspect.getsource(vc._run_performance_benchmark)

    def test_dispatches_before_the_legacy_single_architecture_flow(self) -> None:
        dispatch_index = self.run_source.index(
            'if getattr(args, "run_performance_benchmark", False):'
        )
        legacy_index = self.run_source.index("worktree_root: Path = args.worktree_root")
        self.assertLess(dispatch_index, legacy_index)

    def test_never_populates_contract_promotions(self) -> None:
        self.assertNotIn("contract_promotions", self.impl_source)

    def test_turns_on_execution_identity_and_device_visibility(self) -> None:
        self.assertIn("execution_identity=execution_identity", self.impl_source)
        self.assertIn("require_device_visibility(", self.impl_source)

    def test_inapplicable_cells_are_recorded_skipped_with_a_reason_not_omitted(self) -> None:
        self.assertIn('cell.update(status="skipped", reason=', self.impl_source)


class OrchestrationLogicTests(unittest.TestCase):
    """Hardware-free: mocks build_tree/generate_registry/source
    materialization/resolve_benchmark_wiring/resolve_benchmark_model so
    _run_performance_benchmark()'s real orchestration logic -- device
    resolution, require_device_visibility()/ExecutionIdentity
    construction, skip-cell handling -- actually runs. This is the test
    that would have caught the NameError found on the real-hardware
    smoke run (see module docstring)."""

    def setUp(self) -> None:
        import argparse
        import tempfile

        self._tmp = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self._tmp.name)

        self.args = argparse.Namespace(
            patch="1202_rd04_bf16_flash_attn_tile",
            hip_path=self.tmp_path / "hip",
            workdir=self.tmp_path / "workdir",
            worktree_root=self.tmp_path / "worktrees",
            build_root=None,
            model_root=self.tmp_path / "models",
            benchmark_architecture=["gfx1100"],
            benchmark_model=["test-model"],
            device_map=["gfx1100=0"],
            bench_repetitions=2,
            baseline_source="bigcherry",
        )
        self.args.workdir.mkdir(parents=True, exist_ok=True)

        self.descriptor = mock.Mock(
            patch_id="1202_rd04_bf16_flash_attn_tile",
            validation_architectures=("gfx1100",),
        )
        self.cfg = mock.Mock()
        self.cfg.pinned = "deadbeef"
        self.cfg.platforms = {"linux-multi": mock.Mock(targets=("gfx1100",))}

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def _run_with_patches(self, *, device_count: int = 1, runtime_args: tuple = ()):
        wiring = mock.Mock(executor="paired-llama-bench-v1", patch_args=())
        resolved_model = mock.Mock(
            path=self.tmp_path / "model.gguf", device_count=device_count,
            runtime_args=runtime_args,
        )
        composition = mock.Mock()
        materialized = mock.Mock()
        materialized.name = "src-deadbeef"
        outcome = mock.Mock(
            commands=[], raw_logs=[], runs={"decode": mock.Mock(stats={}, runs=[])},
        )

        executor_func = mock.Mock(return_value=outcome)

        patches = [
            mock.patch.object(vc, "resolve_benchmark_wiring", return_value=wiring),
            mock.patch.object(vc, "resolve_benchmark_model", return_value=resolved_model),
            mock.patch.object(vc, "generate_registry", return_value=None),
            mock.patch.object(vc, "build_tree", return_value=self.tmp_path / "bin"),
            mock.patch.object(vc, "run_paired_llama_benchmark", executor_func),
            mock.patch.dict(
                vc.BENCHMARK_EXECUTOR_FUNCS, {"paired-llama-bench-v1": executor_func},
            ),
            mock.patch.object(vc, "capture_completed_build_evidence", return_value=mock.Mock()),
            mock.patch.object(vc, "assert_validation_subject_parity", return_value=None),
            mock.patch(
                "bigcherry.patch.source.resolve_source_composition",
                return_value=("deadbeef" * 5, composition),
            ),
            mock.patch(
                "bigcherry.patch.source.materialize_composition",
                return_value=materialized,
            ),
        ]
        with contextlib.ExitStack() as stack:
            mocks = [stack.enter_context(p) for p in patches]
            require_visibility = stack.enter_context(
                mock.patch(
                    "bigcherry.experiment.execution.require_device_visibility",
                    return_value=mock.Mock(document=lambda: {}),
                )
            )
            result = vc._run_performance_benchmark(self.args, self.descriptor, self.cfg)
        return result, require_visibility, executor_func

    def test_reaches_and_calls_require_device_visibility(self) -> None:
        result, require_visibility, _ = self._run_with_patches()
        require_visibility.assert_called_once()
        _, kwargs = require_visibility.call_args
        self.assertEqual(kwargs["exact_count"], 1)
        self.assertIn("HIP_VISIBLE_DEVICES", kwargs["env"])

    def test_successful_cell_returns_zero(self) -> None:
        result, _, _ = self._run_with_patches()
        self.assertEqual(result, 0)

    def test_device_pool_shortfall_skips_the_cell_without_crashing(self) -> None:
        # device-map only offers 1 id but the model needs 2 -- must be
        # recorded as a skipped cell (not an uncaught exception).
        result, require_visibility, _ = self._run_with_patches(device_count=2)
        require_visibility.assert_not_called()
        self.assertEqual(result, 1)
        matrix = json.loads((self.args.workdir / "performance-matrix.json").read_text())
        self.assertEqual(matrix["cells"][0]["status"], "skipped")

    def test_topology_runtime_args_reach_the_executor(self) -> None:
        # GPT review (req_e608313764834497, 2026-09-11): tensor-2 selected
        # 2 devices but never passed llama-bench's -sm tensor flag through
        # to the executor -- this is the exact gap that finding closed.
        self.args.device_map = ["gfx1100=0,1"]
        _, _, executor_func = self._run_with_patches(
            device_count=2, runtime_args=("-sm", "tensor"),
        )
        executor_func.assert_called_once()
        _, kwargs = executor_func.call_args
        self.assertEqual(kwargs["runtime_args"], ("-sm", "tensor"))

    def test_executor_is_looked_up_via_the_dispatch_table_not_hardcoded(self) -> None:
        # GPT review: wiring.executor was validated but never actually
        # dispatched on -- _run_performance_benchmark() always called
        # run_paired_llama_benchmark() directly regardless of wiring.
        # Prove it really goes through BENCHMARK_EXECUTOR_FUNCS[executor]
        # by making the dict entry a DIFFERENT mock than the module-level
        # run_paired_llama_benchmark name -- only the dispatch-table path
        # would reach the dict's mock.
        import argparse
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            args = argparse.Namespace(
                patch="1202_rd04_bf16_flash_attn_tile",
                hip_path=tmp_path / "hip", workdir=tmp_path / "workdir",
                worktree_root=tmp_path / "worktrees", build_root=None,
                model_root=tmp_path / "models", benchmark_architecture=["gfx1100"],
                benchmark_model=["test-model"], device_map=["gfx1100=0"],
                bench_repetitions=2, baseline_source="bigcherry",
            )
            args.workdir.mkdir(parents=True, exist_ok=True)
            descriptor = mock.Mock(
                patch_id="1202_rd04_bf16_flash_attn_tile", validation_architectures=("gfx1100",),
            )
            cfg = mock.Mock()
            cfg.pinned = "deadbeef"
            cfg.platforms = {"linux-multi": mock.Mock(targets=("gfx1100",))}

            wiring = mock.Mock(executor="paired-llama-bench-v1", patch_args=())
            resolved_model = mock.Mock(
                path=tmp_path / "model.gguf", device_count=1, runtime_args=(),
            )
            composition = mock.Mock()
            materialized = mock.Mock()
            materialized.name = "src-deadbeef"
            dispatch_outcome = mock.Mock(
                commands=[], raw_logs=[], runs={"decode": mock.Mock(stats={}, runs=[])},
            )
            dispatch_func = mock.Mock(return_value=dispatch_outcome)
            wrong_func = mock.Mock(side_effect=AssertionError(
                "run_paired_llama_benchmark called directly, bypassing BENCHMARK_EXECUTOR_FUNCS"
            ))

            with contextlib.ExitStack() as stack:
                stack.enter_context(mock.patch.object(vc, "resolve_benchmark_wiring", return_value=wiring))
                stack.enter_context(mock.patch.object(vc, "resolve_benchmark_model", return_value=resolved_model))
                stack.enter_context(mock.patch.object(vc, "generate_registry", return_value=None))
                stack.enter_context(mock.patch.object(vc, "build_tree", return_value=tmp_path / "bin"))
                stack.enter_context(mock.patch.object(vc, "run_paired_llama_benchmark", wrong_func))
                stack.enter_context(mock.patch.dict(
                    vc.BENCHMARK_EXECUTOR_FUNCS, {"paired-llama-bench-v1": dispatch_func},
                ))
                stack.enter_context(mock.patch.object(vc, "capture_completed_build_evidence", return_value=mock.Mock()))
                stack.enter_context(mock.patch.object(vc, "assert_validation_subject_parity", return_value=None))
                stack.enter_context(mock.patch(
                    "bigcherry.patch.source.resolve_source_composition",
                    return_value=("deadbeef" * 5, composition),
                ))
                stack.enter_context(mock.patch(
                    "bigcherry.patch.source.materialize_composition", return_value=materialized,
                ))
                stack.enter_context(mock.patch(
                    "bigcherry.experiment.execution.require_device_visibility",
                    return_value=mock.Mock(document=lambda: {}),
                ))
                vc._run_performance_benchmark(args, descriptor, cfg)

            dispatch_func.assert_called_once()
            wrong_func.assert_not_called()


if __name__ == "__main__":
    unittest.main()
