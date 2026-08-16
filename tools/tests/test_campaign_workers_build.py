"""RE14: make_build_worker's real cross-invocation build reuse via
builds.validate_reuse() -- unit-tested with a faked compiler (subprocess.run
patched) rather than a real one, so the reuse *decision logic* (skip
configure+build vs fail closed vs run for real and record metadata) is
covered without needing a compiler or a GPU. Round 9-11's real Brutus runs
already prove the compile/publish machinery this sits on top of works for
real; this file's job is the branch logic those runs never exercised twice
with the same build_plan_id in the same process.
"""

from __future__ import annotations

import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bigcherry import campaign_workers  # noqa: E402
from bigcherry import config as campaign_config  # noqa: E402
from bigcherry import generated_tree  # noqa: E402
from bigcherry.artifacts import ArtifactStore  # noqa: E402
from bigcherry.builds import BuildPlan, build_directory  # noqa: E402
from bigcherry.campaign_build import CampaignBuildError  # noqa: E402
from bigcherry.context import ProjectContext  # noqa: E402
from bigcherry.pipeline import ArtifactRef  # noqa: E402
from bigcherry import provenance  # noqa: E402

_CMAKE_CACHE = """\
CMAKE_C_COMPILER:FILEPATH=/opt/rocm/llvm/bin/clang
CMAKE_CXX_COMPILER:FILEPATH=/opt/rocm/llvm/bin/clang++
CMAKE_BUILD_TYPE:STRING=Release
AMDGPU_TARGETS:STRING=gfx1100
GGML_HIP:BOOL=ON
"""


class _Harness:
    """Everything one make_build_worker call needs, built once per test so
    each test only has to vary the part it's actually testing."""

    def __init__(self, directory: Path):
        self.directory = directory
        self.context = ProjectContext(
            project_root=directory, config_path=directory / "recipes.toml",
            artifacts_root=directory / "artifacts", work_root=directory / "work",
            upstream_repo=directory / "upstream", overlay_root=directory / "src",
            patches_root=directory / "patches",
        )
        self.store = ArtifactStore(directory / "store")
        self.run_id = "run1"
        self.source_slice_id = "s1"
        self.workload_id = "w1"
        self.build_plan = BuildPlan(
            source_slice_id=self.source_slice_id, phase="tune", platform="linux-multi",
            targets=("gfx1100",), variant_set="workload-max")
        self.platform = campaign_config.Platform(
            name="linux-multi", targets=("gfx1100",), options=(),
            c_compiler="/opt/rocm/llvm/bin/clang", cxx_compiler="/opt/rocm/llvm/bin/clang++")
        self.build_cfg = campaign_config.Build(
            name="tune", options=(), variant_set="workload-max", needs=frozenset({"inventory"}))
        self.calls: list[list[str]] = []

    def generate_inputs(
        self, *, registry_content: str = "registry", label: str = "generate-inputs",
    ) -> tuple[ArtifactRef, ...]:
        generated_root = self.context.work_root / "runs" / self.run_id / "generate" / "generated"
        registry = generated_root / "hip-autotune-registry.inc"
        registry.parent.mkdir(parents=True, exist_ok=True)
        # Overwrite, not append: a second call with different
        # registry_content (simulating a different --arch's generated
        # catalog) must leave the on-disk generated/ tree matching the
        # freshly-built tree_document below, or generated_tree.verify_tree()
        # would reject it as tampered before the reuse decision is even
        # reached.
        registry.write_text(registry_content, encoding="utf-8")
        tree_document = generated_tree.build_manifest(generated_root, compile_inputs=(registry,))

        doc = provenance.make(
            project={}, source={"source_slice_id": self.source_slice_id},
            build={"build_plan_id": self.build_plan.build_plan_id},
            workload={"workload_id": self.workload_id}, campaign={"run_id": self.run_id})
        # label-scoped store paths: two calls with different content must
        # not collide on ArtifactStore's immutable-publish check the way two
        # calls with the SAME content correctly do (see FreshBuildTests).
        manifest_digest = self.store.publish_json(f"{label}/manifest.json", {"candidates": []})
        tree_digest = self.store.publish_json(f"{label}/generated-tree.json", tree_document)
        return (
            ArtifactRef(kind="manifest", path=self.store.resolve(f"{label}/manifest.json"),
                        content_hash=manifest_digest, provenance=doc),
            ArtifactRef(kind="generated-tree", path=self.store.resolve(f"{label}/generated-tree.json"),
                        content_hash=tree_digest, provenance=doc),
        )

    def worker(self):
        return campaign_workers.make_build_worker(
            context=self.context, source_root=self.directory / "source", run_id=self.run_id,
            build_plan=self.build_plan, platform=self.platform, build=self.build_cfg,
            store=self.store, binary_relative_path="bin/llama-bench",
            source_slice_id=self.source_slice_id, workload_id=self.workload_id,
            cmake_targets=("llama-bench",),
        )

    def fake_compiler(self, cmake_cache_text: str = _CMAKE_CACHE, hip_so_content: bytes = b"hip-dispatch-v1"):
        build_dir = build_directory(self.context, self.source_slice_id, self.build_plan)

        def run(cmd, cwd=None, check=None):
            self.calls.append(list(cmd))
            if "--build" in cmd:
                binary = build_dir / "bin" / "llama-bench"
                binary.parent.mkdir(parents=True, exist_ok=True)
                binary.write_bytes(b"compiled-binary-bytes")
                # A real build always produces libggml-hip.so alongside the
                # launcher (see resolve_runtime_artifacts's docstring) --
                # include one so runtime-bundle tests can tamper it
                # independently of the launcher itself.
                (build_dir / "bin" / "libggml-hip.so.0.19.0").write_bytes(hip_so_content)
            else:
                (build_dir / "CMakeCache.txt").write_text(cmake_cache_text, encoding="utf-8")
            return subprocess.CompletedProcess(cmd, 0)

        return run


class FreshBuildTests(unittest.TestCase):
    def test_compiles_for_real_and_records_reuse_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            harness = _Harness(Path(directory))
            inputs = harness.generate_inputs()
            with patch("bigcherry.campaign_workers.subprocess.run", harness.fake_compiler()):
                refs = harness.worker()(inputs)

            self.assertEqual(len(harness.calls), 2)  # configure, then build
            self.assertEqual(len(refs), 1)
            self.assertEqual(refs[0].kind, "binary")

            build_dir = build_directory(harness.context, harness.source_slice_id, harness.build_plan)
            metadata_path = build_dir / "bigcherry-build-metadata-llama-bench.json"
            self.assertTrue(metadata_path.is_file())
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            self.assertEqual(metadata["source_slice_id"], "s1")
            self.assertEqual(metadata["build_plan_id"], harness.build_plan.build_plan_id)
            self.assertIn("CMAKE_C_COMPILER", metadata["effective_configure"])


class ReuseTests(unittest.TestCase):
    def test_second_call_with_same_identity_reuses_without_compiling(self):
        with tempfile.TemporaryDirectory() as directory:
            harness = _Harness(Path(directory))
            with patch("bigcherry.campaign_workers.subprocess.run", harness.fake_compiler()):
                harness.worker()(harness.generate_inputs())
            self.assertEqual(len(harness.calls), 2)

            # A second, independent worker call -- as if a different
            # process/run_id encountered the same content-addressed
            # build_directory(). subprocess.run is patched to always raise,
            # so any attempt to recompile fails the test outright rather
            # than merely going undetected.
            with patch("bigcherry.campaign_workers.subprocess.run",
                       side_effect=AssertionError("must not recompile on reuse")):
                refs = harness.worker()(harness.generate_inputs())

            self.assertEqual(len(refs), 1)
            self.assertEqual(refs[0].kind, "binary")

    def test_metadata_present_but_binary_tampered_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            harness = _Harness(Path(directory))
            with patch("bigcherry.campaign_workers.subprocess.run", harness.fake_compiler()):
                harness.worker()(harness.generate_inputs())

            build_dir = build_directory(harness.context, harness.source_slice_id, harness.build_plan)
            (build_dir / "bin" / "llama-bench").write_bytes(b"tampered-after-the-fact")

            with patch("bigcherry.campaign_workers.subprocess.run",
                       side_effect=AssertionError("must not silently recompile over a failed check")):
                with self.assertRaisesRegex(CampaignBuildError, "failed reuse validation"):
                    harness.worker()(harness.generate_inputs())

    def test_different_generated_catalog_under_the_same_build_plan_recompiles_not_reuses(self):
        # gpt-auto-agent review finding, verified real before fixing (see
        # campaign_workers.py's module docstring): BuildPlan does not depend
        # on the catalog's --arch, so two generate runs asked for different
        # candidate architectures can share a build_plan_id/build_dir while
        # producing genuinely different generated/ catalogs. Without a
        # compile_inputs_hash check, the second run would silently reuse a
        # binary built from an entirely different candidate catalog -- a
        # real false-reuse bug, not a hypothetical one.
        with tempfile.TemporaryDirectory() as directory:
            harness = _Harness(Path(directory))
            with patch("bigcherry.campaign_workers.subprocess.run", harness.fake_compiler()):
                harness.worker()(harness.generate_inputs(
                    registry_content="registry-for-gfx1100", label="arch-a"))
            first_calls = len(harness.calls)
            self.assertEqual(first_calls, 2)

            # A DIFFERENT generated catalog (simulating a different --arch),
            # same build_plan_id/build_dir. This must NOT silently reuse the
            # first build's binary -- it must recompile for real.
            arch_b_inputs = harness.generate_inputs(
                registry_content="registry-for-gfx1201-DIFFERENT", label="arch-b")
            with patch("bigcherry.campaign_workers.subprocess.run", harness.fake_compiler()):
                refs = harness.worker()(arch_b_inputs)

            self.assertEqual(len(harness.calls), first_calls + 2)  # recompiled, not reused
            self.assertEqual(len(refs), 1)
            self.assertEqual(refs[0].kind, "binary")

            arch_b_tree_document = json.loads(
                arch_b_inputs[1].path.read_text(encoding="utf-8"))
            build_dir = build_directory(harness.context, harness.source_slice_id, harness.build_plan)
            metadata = json.loads(
                (build_dir / "bigcherry-build-metadata-llama-bench.json").read_text(encoding="utf-8"))
            # The recorded hash now reflects the SECOND (arch-b) catalog,
            # not the first -- a third call with arch-b's exact content
            # would correctly reuse; a third call with arch-a's would not.
            self.assertEqual(
                metadata["generated_compile_inputs_hash"],
                arch_b_tree_document["compile_inputs_hash"])

    def test_dependent_library_tampered_fails_closed_even_though_launcher_is_untouched(self):
        # gpt-auto-agent review finding: "you're hashing one executable, not
        # necessarily the build-output closure" -- RE09 established the
        # real HIP dispatch logic lives in libggml-hip.so, not the launcher.
        # A reuse check based on binary_hash alone would miss this.
        with tempfile.TemporaryDirectory() as directory:
            harness = _Harness(Path(directory))
            with patch("bigcherry.campaign_workers.subprocess.run", harness.fake_compiler()):
                harness.worker()(harness.generate_inputs())

            build_dir = build_directory(harness.context, harness.source_slice_id, harness.build_plan)
            # The launcher itself is byte-identical; only its dependent
            # library changed.
            (build_dir / "bin" / "libggml-hip.so.0.19.0").write_bytes(b"TAMPERED-hip-dispatch")

            with patch("bigcherry.campaign_workers.subprocess.run",
                       side_effect=AssertionError("must not silently recompile over a failed check")):
                with self.assertRaisesRegex(CampaignBuildError, "failed reuse validation"):
                    harness.worker()(harness.generate_inputs())

    def test_metadata_present_but_build_plan_differs_fails_closed(self):
        # Same build_dir (forced by publishing under the first plan's
        # identity) but a caller now asking for a DIFFERENT build_plan_id's
        # worth of identity -- e.g. a corrupted/mismatched metadata file,
        # not a real collision (build_directory() is itself keyed by
        # build_plan_id, so this specific scenario cannot happen through the
        # real code path; it proves validate_reuse's own field check is
        # actually wired in, not just present in builds.py).
        with tempfile.TemporaryDirectory() as directory:
            harness = _Harness(Path(directory))
            with patch("bigcherry.campaign_workers.subprocess.run", harness.fake_compiler()):
                harness.worker()(harness.generate_inputs())

            build_dir = build_directory(harness.context, harness.source_slice_id, harness.build_plan)
            metadata_path = build_dir / "bigcherry-build-metadata-llama-bench.json"
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            metadata["build_plan_id"] = "a-different-build-plan-id"
            metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

            with patch("bigcherry.campaign_workers.subprocess.run",
                       side_effect=AssertionError("must not silently recompile over a failed check")):
                with self.assertRaisesRegex(CampaignBuildError, "failed reuse validation"):
                    harness.worker()(harness.generate_inputs())


if __name__ == "__main__":
    unittest.main()
