from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from bigcherry.patch.campaign import benchmark as campaign_benchmark  # noqa: E402
from bigcherry.patch.campaign import build as campaign_build  # noqa: E402

import importlib.util  # noqa: E402

# PA43: the patch1000 helpers are lab code (tools/lab/patch1000/), which is
# intentionally not a package -- load the module by path.
_LAB_MODULE_PATH = (
    Path(__file__).resolve().parents[1].parent / "lab" / "patch1000" / "patch1000_verification.py"
)
_spec = importlib.util.spec_from_file_location("patch1000_verification", _LAB_MODULE_PATH)
p1000 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(p1000)


class _Result:
    def __init__(self, returncode: int, stdout: str, stderr: str = "") -> None:
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


class Patch1000CommandTests(unittest.TestCase):
    def test_perf_command_is_exact_upstream_n512_shape(self) -> None:
        command = p1000._patch1000_backend_ops_command(
            Path("test-backend-ops"), "Q2_K", mode="perf",
        )
        self.assertEqual(
            command,
            [
                "test-backend-ops", "perf", "-o", "MUL_MAT", "-p",
                "type_a=q2_K,type_b=f32,m=4096,n=512,k=14336",
            ],
        )

    def test_correctness_uses_quant_family_not_perf_only_shape(self) -> None:
        command = p1000._patch1000_backend_ops_command(
            Path("test-backend-ops"), "Q6_K", mode="test",
        )
        self.assertEqual(
            command,
            ["test-backend-ops", "test", "-o", "MUL_MAT", "-p", "type_a=q6_K,type_b=f32"],
        )

    def test_unknown_quant_fails_closed(self) -> None:
        with self.assertRaises(campaign_build.PatchCampaignError):
            p1000._patch1000_backend_ops_command(Path("test-backend-ops"), "Q4_K", mode="perf")


class Patch1000BackendPerfTests(unittest.TestCase):
    def setUp(self) -> None:
        self.real_run = p1000.subprocess.run

    def tearDown(self) -> None:
        p1000.subprocess.run = self.real_run

    def test_paired_perf_reports_control_over_subject_speedup(self) -> None:
        seen_envs: list[dict[str, str]] = []

        def fake_run(command, capture_output, text, check, env):  # noqa: ANN001
            seen_envs.append(dict(env))
            time_us = 20.0 if "control" in command[0] else 10.0
            stdout = (
                "ggml_cuda_init: found 1 ROCm devices\n"
                "  MUL_MAT(type_a=q6_K,type_b=f32,m=4096,"
                "n=512,k=14336,bs=[1,1],nr=[1,1],"
                "per=[0,1,2,3],k_v=0,o=1): "
                f"10 runs - {time_us:.2f} us/run - "
                "1.00 TFLOP/run - 50.00 TFLOPS\n"
            )
            return _Result(0, stdout)

        p1000.subprocess.run = fake_run

        old_rocr = os.environ.get("ROCR_VISIBLE_DEVICES")
        try:
            os.environ["ROCR_VISIBLE_DEVICES"] = "9"
            result = p1000.run_patch1000_backend_ops_perf(
                control_binary=Path("control/test-backend-ops"),
                subject_binary=Path("subject/test-backend-ops"),
                quant="Q6_K",
                hip_path=Path("/opt/rocm"),
                env_overrides={"HIP_VISIBLE_DEVICES": "2"},
                pairs=2,
            )
        finally:
            if old_rocr is None:
                os.environ.pop("ROCR_VISIBLE_DEVICES", None)
            else:
                os.environ["ROCR_VISIBLE_DEVICES"] = old_rocr

        self.assertAlmostEqual(result["stats"]["geometric_speedup_x"], 2.0)
        self.assertEqual(len(result["raw_logs"]), 4)
        for env in seen_envs:
            self.assertEqual(env["HIP_VISIBLE_DEVICES"], "2")
            self.assertNotIn("ROCR_VISIBLE_DEVICES", env)


class Patch1000CorrectnessTests(unittest.TestCase):
    def setUp(self) -> None:
        self.real_run = p1000.subprocess.run

    def tearDown(self) -> None:
        p1000.subprocess.run = self.real_run

    def test_successful_backend_reference_check_passes(self) -> None:
        def fake_run(command, capture_output, text, check, env):  # noqa: ANN001
            return _Result(0, "ggml_cuda_init: found 1 ROCm devices\n1/1 tests passed\n")

        p1000.subprocess.run = fake_run
        result = p1000.run_patch1000_backend_ops_correctness(
            binary=Path("test-backend-ops"),
            quant="Q2_K",
            hip_path=Path("/opt/rocm"),
            env_overrides={"HIP_VISIBLE_DEVICES": "2"},
            log_context="patch1000-test",
        )
        self.assertTrue(result["passed"])

    def test_failed_backend_reference_check_fails_closed(self) -> None:
        def fake_run(command, capture_output, text, check, env):  # noqa: ANN001
            return _Result(1, "FAIL\n", "bad result")

        p1000.subprocess.run = fake_run
        with self.assertRaises(campaign_build.PatchCampaignError):
            p1000.run_patch1000_backend_ops_correctness(
                binary=Path("test-backend-ops"),
                quant="Q6_K",
                hip_path=Path("/opt/rocm"),
                env_overrides={"HIP_VISIBLE_DEVICES": "2"},
                log_context="patch1000-test",
            )


class Patch1000FatBuildTests(unittest.TestCase):
    def test_builds_each_arm_once_with_full_target_set(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            recipes = root / "recipes.toml"
            recipes.write_text(
                '[patch-set.serving-core]\npatches = ["0100_fake_framework"]\n',
                encoding="utf-8",
            )
            (root / "src").mkdir()

            q2 = root / "q2.gguf"
            q6 = root / "q6.gguf"
            q2.write_bytes(b"q2")
            q6.write_bytes(b"q6")

            patch_id = p1000._PATCH1000_ID
            framework_comp = (("0100_fake_framework", "d0"),)
            subject_comp = (("0100_fake_framework", "d0"), (patch_id, "d1"))

            def resolve_source_composition(source_name, *, focal=None, extra_patches=(), **kwargs):  # noqa: ANN001
                self.assertIsNone(focal)
                if source_name == "bigcherry-serving-base":
                    return "base-sha", subject_comp
                if source_name != "llama-native":
                    raise AssertionError(source_name)
                if not extra_patches:
                    return "base-sha", ()
                if tuple(extra_patches) == ("0100_fake_framework",):
                    return "base-sha", framework_comp
                if tuple(extra_patches) == ("0100_fake_framework", patch_id):
                    return "base-sha", subject_comp
                raise AssertionError(extra_patches)

            def materialize_composition(*, composition, worktree_root, **kwargs):  # noqa: ANN001
                if not composition:
                    name = "stock"
                elif tuple(composition) == framework_comp:
                    name = "control"
                else:
                    name = "subject"
                return Path(worktree_root) / name

            fake_source = SimpleNamespace(
                REPO_ROOT=root,
                resolve_source_composition=resolve_source_composition,
                materialize_composition=materialize_composition,
            )

            build_calls: list[dict[str, object]] = []

            def fake_build(**kwargs):  # noqa: ANN003
                build_calls.append(dict(kwargs))
                return root / "build" / str(kwargs["name"]) / "bin"

            perf_calls: list[dict[str, object]] = []
            correctness_calls: list[dict[str, object]] = []
            bench_calls: list[dict[str, object]] = []

            def fake_perf(**kwargs):  # noqa: ANN003
                perf_calls.append(dict(kwargs))
                return {"passed": True}

            def fake_correctness(**kwargs):  # noqa: ANN003
                correctness_calls.append(dict(kwargs))
                return {"passed": True}

            def fake_bench(**kwargs):  # noqa: ANN003
                bench_calls.append(dict(kwargs))
                return campaign_benchmark.PairedBenchmarkOutcome(runs={}, commands={}, raw_logs=[])

            result = p1000.run_patch1000_verification(
                base_revision="base",
                hip_path=Path("/opt/rocm"),
                build_root=root / "work",
                q2k_model=q2,
                q6k_model=q6,
                devices={"gfx1100": "0", "gfx1201": "2", "gfx1030": "3"},
                _source_module=fake_source,
                _build_func=fake_build,
                _perf_func=fake_perf,
                _correctness_func=fake_correctness,
                _bench_func=fake_bench,
                _recipes_path=recipes,
            )

            self.assertEqual(len(build_calls), 3)
            for call in build_calls:
                self.assertEqual(call["amdgpu_targets"], "gfx1100;gfx1201;gfx1030")
                self.assertEqual(call["targets"], ["test-backend-ops", "llama-bench"])
                self.assertNotIn("gfx", str(call["name"]))

            self.assertEqual(len(perf_calls), 18)
            self.assertEqual(len(bench_calls), 18)
            self.assertEqual(len(correctness_calls), 12)

            self.assertEqual(result["compiled_targets"], ["gfx1100", "gfx1201", "gfx1030"])
            self.assertEqual(result["amdgpu_targets"], "gfx1100;gfx1201;gfx1030")


if __name__ == "__main__":
    unittest.main()
