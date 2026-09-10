"""Offline wiring checks for the HI168 framework-configuration campaign path."""

from __future__ import annotations

import tempfile
import unittest
from types import SimpleNamespace
from pathlib import Path
from unittest import mock

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from bigcherry.patch import validation_campaign as vc  # noqa: E402


class FrameworkConfigurationCampaignTests(unittest.TestCase):
    def test_single_composition_runs_all_real_adapters_and_persists_eligible_record(self):
        from bigcherry.patch import source, registry, evidence
        root = Path(__file__).resolve().parents[3]
        descriptor = registry.load_registry(root / "patches").get("0100_cmake_options")
        composition = ((descriptor.patch_id, descriptor.implementation_digest),)
        identity = {
            "effective_build_id": "build", "compile_verification_id": "verify",
            "compile_commands_digest": "a" * 64, "hip_compile_commands_digest": "b" * 64,
            "runtime_bundle_hash": "c" * 64, "runtime_artifacts": {"llama-server": "d" * 64},
        }
        completed = SimpleNamespace(campaign_identity=lambda: identity, to_dict=lambda: identity)
        source_identity = source._make_source_identity_v2(
            resolved_revision="b" * 40, composition=composition, overlay_root=root / "src")
        source_identity["materialization_plan_id"] = source_identity["source_key"]
        with tempfile.TemporaryDirectory() as temporary:
            work = Path(temporary)
            src = work / "source"
            src.mkdir()
            args = SimpleNamespace(workdir=work / "run", build_root=work / "builds",
                worktree_root=work / "trees", hip_path=work / "rocm", amdgpu_targets="gfx1100")

            def generate(**kwargs):
                directory = kwargs["generated_dir"]
                directory.mkdir(parents=True)
                for name in ("hip-autotune-registry.inc", "hip-autotune-build-hash.h",
                             "hip-autotune-arch.h", "hip-autotune-mmvq-instances.inc"):
                    (directory / name).write_text("// fixture\n", encoding="utf-8")

            def build(**kwargs):
                self.assertEqual(kwargs["targets"], ["llama-server"])
                directory = kwargs["workdir"] / kwargs["name"]
                self.assertIn(f"-DGGML_HIP_AUTOTUNE_GENERATED_DIR={directory / 'generated-inputs'}",
                              kwargs["extra_cmake_args"])
                for phase in ("preconfigure", "postconfigure-precompile", "postcompile"):
                    kwargs["generated_proof_callback"](phase, directory)
                return directory / "bin"

            records = []
            with mock.patch.object(source, "resolve_source_composition", return_value=("b" * 40, composition)), \
                 mock.patch.object(source, "materialize_composition", return_value=src), \
                 mock.patch.object(source, "verify_composition_idempotent", return_value=True), \
                 mock.patch.object(source, "git_worktree_tree", return_value="c" * 40), \
                 mock.patch.object(source, "_read_manifest", return_value={**source_identity, "source_tree_oid": "c" * 40, "source_slice_id": "e" * 32}), \
                 mock.patch.object(vc, "generate_registry", side_effect=generate), \
                 mock.patch.object(vc, "build_tree", side_effect=build), \
                 mock.patch.object(vc, "capture_completed_build_evidence", return_value=completed), \
                 mock.patch("bigcherry.build.builds.inspect_dispatch_build", side_effect=lambda directory: {
                     "issues": [], "hip_compile_command_count": 1,
                     "coverage_translation_unit": directory.name.endswith("diagnostic"),
                     "compiled_definition_counts": {
                         "GGML_HIP_DISPATCH_DIAGNOSTICS": int(directory.name.endswith("diagnostic")),
                         "GGML_HIP_AUTOTUNE": 0, "GGML_HIP_AUTOTUNE_RECORD": 0,
                         "GGML_HIP_REPLAY_DIAGNOSTICS": 0, "GGML_HIP_WORKSPACE_METRICS": 0}}), \
                 mock.patch.object(evidence, "write_record", side_effect=lambda record: records.append(record) or work / "record.json"):
                result = vc._run_framework_configuration(args, descriptor, SimpleNamespace(pinned="pin"))
            self.assertEqual(result, 0)
            self.assertEqual(set(records[0]["check_results"]), {"apply", "build", "coverage-source-selection"})
            self.assertTrue(records[0]["eligible_for_validated_state"])
            self.assertFalse(records[0]["hardware_execution_qualified"])

    def test_cli_exposes_explicit_framework_configuration_mode(self) -> None:
        with mock.patch.object(vc, "run", return_value=17) as run:
            result = vc.main(
                [
                    "--patch", "0100_cmake_options", "--framework-configuration",
                    "--model", "model.gguf", "--hip-path", "rocm",
                    "--amdgpu-targets", "gfx1201", "--manifest", "manifest.json",
                    "--workdir", "work",
                ]
            )
        self.assertEqual(result, 17)
        self.assertTrue(run.call_args.args[0].framework_configuration)

    def test_build_tree_calls_generated_proof_at_all_three_boundaries(self) -> None:
        phases: list[str] = []
        with tempfile.TemporaryDirectory() as d, mock.patch("subprocess.run") as run:
            run.return_value.returncode = 0
            root = Path(d)
            source = root / "source"
            source.mkdir()
            vc.build_tree(
                name="production", extra_cmake_args=[], hip_path=root / "rocm",
                amdgpu_targets="gfx1201", workdir=root / "build", targets=["llama-server"],
                source=source, generated_proof_callback=lambda phase, _: phases.append(phase),
            )
        self.assertEqual(phases, ["preconfigure", "postconfigure-precompile", "postcompile"])


if __name__ == "__main__":
    unittest.main()
