"""PVPS02 step 4: --run-performance-benchmark CLI wiring. The parser-level
argument contract is directly testable (no hardware needed, main() exits
via parser.error() before touching a build); _run_performance_benchmark()
itself is a real-hardware integration entry point (source materialization,
per-architecture cmake builds) -- consistent with the project's established
convention for this class of function (see Rd04CliWiringTests in
test_patch_validation_campaign_va04.py), its dispatch wiring is proven via
source inspection rather than mocking the entire build pipeline; real
end-to-end behavior is covered by PVPS02's real-hardware merge gate on
Brutus.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

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


if __name__ == "__main__":
    unittest.main()
