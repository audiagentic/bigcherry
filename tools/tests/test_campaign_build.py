"""RE14 campaign build executor: cmake args, isolated materialisation, publish."""

from __future__ import annotations

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bigcherry import config as campaign_config  # noqa: E402
from bigcherry.artifacts import ArtifactStore  # noqa: E402
from bigcherry.builds import BuildPlan, build_directory  # noqa: E402
from bigcherry.campaign_build import (CampaignBuildError, cmake_build_args,  # noqa: E402
                                      cmake_configure_args, execute_build_stage,
                                      materialize_source, publish_build_outputs,
                                      toolchain_request_for_platform)
from bigcherry.context import ProjectContext  # noqa: E402
from bigcherry.workspace import SourcePlan  # noqa: E402


def _git(root: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(root), *args], check=True, capture_output=True, text=True
    ).stdout.strip()


def _init_upstream(directory: Path) -> tuple[Path, str]:
    repo = directory / "upstream"
    _git(directory, "init", str(repo))
    _git(repo, "config", "user.email", "test@example.invalid")
    _git(repo, "config", "user.name", "Test")
    (repo / "source.txt").write_text("one\n", encoding="utf-8")
    _git(repo, "add", "source.txt")
    _git(repo, "commit", "-m", "initial")
    revision = _git(repo, "rev-parse", "HEAD")
    return repo, revision


def _context(directory: Path, upstream_repo: Path) -> ProjectContext:
    return ProjectContext(
        project_root=directory, config_path=directory / "recipes.toml",
        artifacts_root=directory / "artifacts", work_root=directory / "work",
        upstream_repo=upstream_repo, overlay_root=directory / "src",
        patches_root=directory / "patches",
    )


class CmakeArgsTests(unittest.TestCase):
    def _build_platform(self):
        build = campaign_config.Build(
            name="tune", options=(("GGML_HIP_AUTOTUNE", "ON"),),
            variant_set="workload-max", needs=frozenset({"inventory"}))
        platform = campaign_config.Platform(
            name="linux-multi", targets=("gfx1100", "gfx1030"),
            options=(("GGML_HIP", "ON"),))
        return build, platform

    def test_configure_args_include_platform_and_build_options(self):
        build, platform = self._build_platform()
        args = cmake_configure_args(
            build, platform, Path("/src"), Path("/build"))
        joined = " ".join(args)
        self.assertIn("-DAMDGPU_TARGETS=gfx1100;gfx1030", joined)
        self.assertIn("-DGGML_HIP=ON", joined)
        self.assertIn("-DGGML_HIP_AUTOTUNE=ON", joined)
        self.assertIn("-DGGML_HIP_AUTOTUNE_VARIANT_SET=workload-max", joined)
        self.assertIn("-DCMAKE_BUILD_TYPE=Release", joined)

    def test_generated_root_and_inventory_only_apply_with_variant_set(self):
        build = campaign_config.Build(
            name="stock", options=(), variant_set=None, needs=frozenset())
        platform = campaign_config.Platform(
            name="p", targets=("gfx1100",), options=())
        args = cmake_configure_args(
            build, platform, Path("/src"), Path("/build"),
            generated_root=Path("/gen"), inventory=Path("/inv.json"))
        joined = " ".join(args)
        self.assertNotIn("GGML_HIP_AUTOTUNE_GENERATED_DIR", joined)
        self.assertNotIn("GGML_HIP_AUTOTUNE_SIGNATURE_FILE", joined)

    def test_build_args_include_targets(self):
        args = cmake_build_args(Path("/build"), targets=("llama-server",))
        self.assertEqual(args, ["cmake", "--build", str(Path("/build")), "-j", "--target", "llama-server"])


class ToolchainRequestTests(unittest.TestCase):
    def test_includes_compilers_when_declared(self):
        platform = campaign_config.Platform(
            name="linux-multi", targets=("gfx1100",), options=(),
            c_compiler="/opt/rocm/llvm/bin/clang", cxx_compiler="/opt/rocm/llvm/bin/clang++")
        self.assertEqual(
            toolchain_request_for_platform(platform),
            (("CMAKE_CXX_COMPILER", "/opt/rocm/llvm/bin/clang++"),
             ("CMAKE_C_COMPILER", "/opt/rocm/llvm/bin/clang"),
             ("CMAKE_GENERATOR", "Ninja")))

    def test_omits_compilers_when_not_declared(self):
        platform = campaign_config.Platform(
            name="windows-gfx1100", targets=("gfx1100",), options=())
        self.assertEqual(
            toolchain_request_for_platform(platform),
            (("CMAKE_GENERATOR", "Ninja"),))


class MaterializeSourceTests(unittest.TestCase):
    def test_materialize_is_idempotent_and_reused(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            upstream, revision = _init_upstream(root)
            context = _context(root, upstream)
            plan = SourcePlan(revision, False, (), None)

            first = materialize_source(context, plan, allow_dirty_bigcherry=True)
            second = materialize_source(context, plan, allow_dirty_bigcherry=True)
            self.assertEqual(first["source_slice_id"], second["source_slice_id"])

    def test_materialize_detects_metadata_mismatch(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            upstream, revision = _init_upstream(root)
            context = _context(root, upstream)
            plan = SourcePlan(revision, False, (), None)
            materialize_source(context, plan, allow_dirty_bigcherry=True)

            # Corrupt the stored metadata to simulate a stale/mismatched cache.
            from bigcherry import campaign_source
            plan_id = campaign_source.source_plan_id(plan)
            destination = context.work_root / "sources" / plan_id
            metadata_path = destination.parent / f"{destination.name}.metadata.json"
            import json
            record = json.loads(metadata_path.read_text(encoding="utf-8"))
            record["plan"]["upstream_revision"] = "0" * 40
            metadata_path.write_text(json.dumps(record), encoding="utf-8")

            with self.assertRaises(CampaignBuildError):
                materialize_source(context, plan, allow_dirty_bigcherry=True)


class PublishBuildOutputsTests(unittest.TestCase):
    def test_publish_verifies_every_artifact(self):
        with tempfile.TemporaryDirectory() as directory:
            store = ArtifactStore(Path(directory))
            binary = Path(directory) / "binary.bin"
            binary.write_bytes(b"fake-binary-bytes")
            published = publish_build_outputs(
                store, source_slice_id="s1", build_plan_id="b1",
                manifest={"m": 1}, descriptor={"d": 2}, binary=binary)
            self.assertEqual(len(published), 3)
            for relative, digest in published.items():
                self.assertTrue(store.verify(relative, digest))


class ExecuteBuildStageTests(unittest.TestCase):
    def test_end_to_end_with_fake_compiler(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            upstream, revision = _init_upstream(root)
            context = _context(root, upstream)
            source_plan = SourcePlan(revision, False, (), None)

            # Real materialisation determines the real source_slice_id; the
            # build plan below is intentionally built with a placeholder
            # first, then corrected, to prove execute_build_stage rejects a
            # mismatched source_slice_id rather than silently trusting it.
            build_plan_wrong = BuildPlan(
                source_slice_id="wrong", phase="tune", platform="linux-multi",
                targets=("gfx1100",))
            build = campaign_config.Build(
                name="stock", options=(), variant_set=None, needs=frozenset())
            platform = campaign_config.Platform(
                name="linux-multi", targets=("gfx1100",), options=())
            store = ArtifactStore(context.work_root / "artifacts")

            def fake_runner(args, cwd):
                if args[0] == "cmake" and "--build" in args:
                    build_dir = Path(args[2])
                    build_dir.mkdir(parents=True, exist_ok=True)
                    (build_dir / "llama-server").write_bytes(b"fake-binary")

            with self.assertRaises(CampaignBuildError):
                execute_build_stage(
                    context, source_plan=source_plan, build_plan=build_plan_wrong,
                    build=build, platform=platform, artifact_store=store,
                    binary_relative_path="llama-server", runner=fake_runner,
                    allow_dirty_bigcherry=True)

            # Now with the correct source_slice_id derived the same way the
            # real caller would: materialise once to learn it.
            from bigcherry.campaign_build import materialize_source
            metadata = materialize_source(context, source_plan, allow_dirty_bigcherry=True)
            build_plan = BuildPlan(
                source_slice_id=metadata["source_slice_id"], phase="tune",
                platform="linux-multi", targets=("gfx1100",))

            published = execute_build_stage(
                context, source_plan=source_plan, build_plan=build_plan,
                build=build, platform=platform, artifact_store=store,
                binary_relative_path="llama-server", runner=fake_runner,
                manifest={"m": 1}, allow_dirty_bigcherry=True)
            self.assertTrue(any(name.endswith("llama-server") for name in published))
            self.assertTrue(any(name.endswith("manifest.json") for name in published))

            expected_dir = build_directory(context, metadata["source_slice_id"], build_plan)
            self.assertTrue((expected_dir / "llama-server").is_file())


if __name__ == "__main__":
    unittest.main()
