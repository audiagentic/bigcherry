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

    def generate_inputs(self) -> tuple[ArtifactRef, ...]:
        generated_root = self.context.work_root / "runs" / self.run_id / "generate" / "generated"
        registry = generated_root / "hip-autotune-registry.inc"
        registry.parent.mkdir(parents=True, exist_ok=True)
        registry.write_text("registry", encoding="utf-8")
        tree_document = generated_tree.build_manifest(generated_root, compile_inputs=(registry,))

        doc = provenance.make(
            project={}, source={"source_slice_id": self.source_slice_id},
            build={"build_plan_id": self.build_plan.build_plan_id},
            workload={"workload_id": self.workload_id}, campaign={"run_id": self.run_id})
        manifest_digest = self.store.publish_json("manifest.json", {"candidates": []})
        tree_digest = self.store.publish_json("generated-tree.json", tree_document)
        return (
            ArtifactRef(kind="manifest", path=self.store.resolve("manifest.json"),
                        content_hash=manifest_digest, provenance=doc),
            ArtifactRef(kind="generated-tree", path=self.store.resolve("generated-tree.json"),
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

    def fake_compiler(self, cmake_cache_text: str = _CMAKE_CACHE):
        build_dir = build_directory(self.context, self.source_slice_id, self.build_plan)

        def run(cmd, cwd=None, check=None):
            self.calls.append(list(cmd))
            if "--build" in cmd:
                binary = build_dir / "bin" / "llama-bench"
                binary.parent.mkdir(parents=True, exist_ok=True)
                binary.write_bytes(b"compiled-binary-bytes")
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
