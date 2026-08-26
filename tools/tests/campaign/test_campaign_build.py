"""RE14 campaign build executor: cmake args, isolated materialisation, publish."""

from __future__ import annotations

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from unittest.mock import patch  # noqa: E402

from bigcherry.core import config as campaign_config # noqa: E402
from bigcherry.core.artifacts import ArtifactStore  # noqa: E402
from bigcherry.build.builds import BuildPlan, build_directory  # noqa: E402
from bigcherry.campaign.build import (CampaignBuildError, cmake_build_args,  # noqa: E402
                                      cmake_configure_args, execute_build_stage,
                                      materialize_source, publish_build_outputs,
                                      resolve_build_environment, resolve_toolchain_versions,
                                      toolchain_request_for_platform)
from bigcherry.core.context import ProjectContext  # noqa: E402
from bigcherry.source.workspace import SourcePlan  # noqa: E402


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

    def test_default_backend_is_hip_unchanged(self):
        """RE30 phase 1 regression proof: adding the ``backend`` kwarg must
        not change a single byte of output for any existing (HIP) caller."""
        build, platform = self._build_platform()
        explicit = cmake_configure_args(build, platform, Path("/src"), Path("/build"), backend="hip")
        default = cmake_configure_args(build, platform, Path("/src"), Path("/build"))
        self.assertEqual(explicit, default)
        self.assertIn("-DAMDGPU_TARGETS=gfx1100;gfx1030", " ".join(default))

    def test_vulkan_backend_omits_amdgpu_targets_and_sets_ggml_vulkan(self):
        build = campaign_config.Build(name="vulkan-stock", options=(), variant_set=None, needs=frozenset())
        platform = campaign_config.Platform(name="p", targets=(), options=())
        args = cmake_configure_args(
            build, platform, Path("/src"), Path("/build"), backend="vulkan")
        joined = " ".join(args)
        self.assertNotIn("AMDGPU_TARGETS", joined)
        self.assertIn("-DGGML_VULKAN=ON", joined)

    def test_unknown_backend_is_rejected(self):
        build, platform = self._build_platform()
        with self.assertRaises(ValueError):
            cmake_configure_args(build, platform, Path("/src"), Path("/build"), backend="cuda")


class ToolchainRequestTests(unittest.TestCase):
    # resolve_toolchain_versions() is mocked away in these two -- they test
    # the requested-path bookkeeping specifically (already covered before
    # version resolution existed); real version-probing behavior has its
    # own dedicated tests below, deterministic regardless of what cmake/
    # ninja/compilers happen to be installed on whatever machine runs this.
    def test_includes_compilers_when_declared(self):
        platform = campaign_config.Platform(
            name="linux-multi", targets=("gfx1100",), options=(),
            c_compiler="/opt/rocm/llvm/bin/clang", cxx_compiler="/opt/rocm/llvm/bin/clang++")
        with patch("bigcherry.campaign.build.resolve_toolchain_versions", return_value={}):
            self.assertEqual(
                toolchain_request_for_platform(platform),
                (("CMAKE_CXX_COMPILER", "/opt/rocm/llvm/bin/clang++"),
                 ("CMAKE_C_COMPILER", "/opt/rocm/llvm/bin/clang"),
                 ("CMAKE_GENERATOR", "Ninja")))

    def test_omits_compilers_when_not_declared(self):
        platform = campaign_config.Platform(
            name="windows-gfx1100", targets=("gfx1100",), options=())
        with patch("bigcherry.campaign.build.resolve_toolchain_versions", return_value={}):
            self.assertEqual(
                toolchain_request_for_platform(platform),
                (("CMAKE_GENERATOR", "Ninja"),))

    def test_real_version_resolution_feeds_into_the_result(self):
        platform = campaign_config.Platform(
            name="linux-multi", targets=("gfx1100",), options=(),
            c_compiler="/opt/rocm/llvm/bin/clang", cxx_compiler=None)
        with patch("bigcherry.campaign.build.resolve_toolchain_versions",
                   return_value={"rocm_version": "7.2.4"}):
            self.assertIn(("rocm_version", "7.2.4"), toolchain_request_for_platform(platform))


class ResolveToolchainVersionsTests(unittest.TestCase):
    """RE14: real, content-level toolchain identity, not just a requested
    path -- gpt-auto-agent review item 3. The core scenario this exists
    for: an in-place ROCm upgrade at the same symlinked path, or explicitly
    pointing two BuildPlans at two different ROCm installs to run them
    side by side as a genuine comparison, must produce two different
    identities rather than colliding on one build_directory().
    """

    def _rocm_install(self, root: Path, version: str) -> Path:
        clang = root / "llvm" / "bin" / "clang"
        clang.parent.mkdir(parents=True, exist_ok=True)
        clang.write_text("", encoding="utf-8")  # only needs to exist, not run
        info_dir = root / ".info"
        info_dir.mkdir(exist_ok=True)
        (info_dir / "version").write_text(f"{version}\n", encoding="utf-8")
        return clang

    def test_two_different_rocm_installs_produce_different_identities(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            clang_a = self._rocm_install(root / "rocm-6.9.9", "6.9.9")
            clang_b = self._rocm_install(root / "rocm-7.2.4", "7.2.4")

            with patch("bigcherry.campaign.build._version_probe",
                       return_value="AMD clang version 22.0.0"):
                values_a = resolve_toolchain_versions(str(clang_a), None)
                values_b = resolve_toolchain_versions(str(clang_b), None)

            self.assertEqual(values_a["rocm_version"], "6.9.9")
            self.assertEqual(values_b["rocm_version"], "7.2.4")
            self.assertNotEqual(values_a["c_compiler_realpath"], values_b["c_compiler_realpath"])

    def test_in_place_upgrade_via_symlink_swap_changes_identity_at_the_same_path(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            clang_old = self._rocm_install(root / "rocm-7.1.0", "7.1.0")
            clang_new = self._rocm_install(root / "rocm-7.2.4", "7.2.4")
            current = root / "rocm-current" / "llvm" / "bin" / "clang"
            current.parent.mkdir(parents=True, exist_ok=True)
            try:
                current.symlink_to(clang_old)
            except OSError as exc:
                self.skipTest(
                    "toolchain symlink-swap coverage requires symlink creation "
                    f"privilege, unavailable in this environment: {exc}"
                )

            with patch("bigcherry.campaign.build._version_probe",
                       return_value="AMD clang version 22.0.0"):
                before = resolve_toolchain_versions(str(current), None)
                current.unlink()
                current.symlink_to(clang_new)
                after = resolve_toolchain_versions(str(current), None)

            # Same requested path both times -- only the real toolchain
            # underneath changed, which is exactly the case a
            # requested-path-only identity would miss.
            self.assertEqual(before["rocm_version"], "7.1.0")
            self.assertEqual(after["rocm_version"], "7.2.4")
            self.assertNotEqual(before["c_compiler_realpath"], after["c_compiler_realpath"])

    def test_missing_tools_are_omitted_not_an_error(self):
        with tempfile.TemporaryDirectory() as directory:
            values = resolve_toolchain_versions(
                str(Path(directory) / "does-not-exist" / "clang"), None)
            # cmake/ninja may genuinely be installed on whatever machine
            # runs this test -- only the compiler-specific keys are
            # guaranteed absent for a path that doesn't exist.
            self.assertNotIn("c_compiler_realpath", values)
            self.assertNotIn("c_compiler_version", values)
            self.assertNotIn("rocm_version", values)

    def test_no_explicit_compiler_falls_back_to_path(self):
        # This repo's linux-multi platform declares no c_compiler in
        # recipes.toml at all -- if resolution only engaged for an
        # explicit override, the PATH-default build (the common case all
        # session) would never get real toolchain fingerprinting.
        with patch("bigcherry.campaign.build.shutil.which", return_value=None):
            values = resolve_toolchain_versions(None, None)
        self.assertNotIn("c_compiler_realpath", values)  # nothing found on PATH either -- fine, not an error


class CmakePrefixPathTests(unittest.TestCase):
    def test_derives_prefix_path_from_resolved_compiler_when_hip_config_present(self):
        with tempfile.TemporaryDirectory() as directory:
            rocm_root = Path(directory) / "rocm-7.2.4"
            clang = rocm_root / "llvm" / "bin" / "clang"
            clang.parent.mkdir(parents=True)
            clang.write_text("", encoding="utf-8")
            hip_config = rocm_root / "lib" / "cmake" / "hip" / "hip-config.cmake"
            hip_config.parent.mkdir(parents=True)
            hip_config.write_text("", encoding="utf-8")

            build = campaign_config.Build(name="tune", options=(), variant_set=None, needs=frozenset())
            platform = campaign_config.Platform(name="p", targets=("gfx1100",), options=())
            args = cmake_configure_args(
                build, platform, Path("/src"), Path("/build"), c_compiler=str(clang))

            joined = " ".join(args)
            self.assertIn(f"-DCMAKE_PREFIX_PATH={rocm_root.as_posix()}", joined)

    def test_no_prefix_path_when_hip_config_absent(self):
        with tempfile.TemporaryDirectory() as directory:
            clang = Path(directory) / "llvm" / "bin" / "clang"
            clang.parent.mkdir(parents=True)
            clang.write_text("", encoding="utf-8")

            build = campaign_config.Build(name="tune", options=(), variant_set=None, needs=frozenset())
            platform = campaign_config.Platform(name="p", targets=("gfx1100",), options=())
            args = cmake_configure_args(
                build, platform, Path("/src"), Path("/build"), c_compiler=str(clang))

            self.assertNotIn("CMAKE_PREFIX_PATH", " ".join(args))


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
            from bigcherry.campaign import source as campaign_source
            plan_id = campaign_source.materialization_plan_id(
                campaign_source.resolve_materialization_identity(context, plan))
            destination = context.work_root / "sources" / plan_id
            metadata_path = destination.parent / f"{destination.name}.metadata.json"
            import json
            record = json.loads(metadata_path.read_text(encoding="utf-8"))
            record["plan"]["upstream_revision"] = "0" * 40
            metadata_path.write_text(json.dumps(record), encoding="utf-8")

            with self.assertRaises(CampaignBuildError):
                materialize_source(context, plan, allow_dirty_bigcherry=True)

    def test_cache_reuse_rejects_head_change_even_when_tree_is_identical(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            upstream, revision_a = _init_upstream(root)
            # Different commit identity, identical tree.
            _git(upstream, "commit", "--allow-empty", "-m", "different commit")
            revision_b = _git(upstream, "rev-parse", "HEAD")
            context = _context(root, upstream)
            plan = SourcePlan(revision_a, False, (), None)
            materialize_source(context, plan, allow_dirty_bigcherry=True)
            from bigcherry.campaign import source as campaign_source
            plan_id = campaign_source.materialization_plan_id(
                campaign_source.resolve_materialization_identity(context, plan))
            destination = context.work_root / "sources" / plan_id
            _git(destination, "reset", "--hard", revision_b)
            with self.assertRaises(CampaignBuildError):
                materialize_source(context, plan, allow_dirty_bigcherry=True)

    def test_cache_reuse_rejects_forged_object_format(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            upstream, revision = _init_upstream(root)
            context = _context(root, upstream)
            plan = SourcePlan(revision, False, (), None)
            materialize_source(context, plan, allow_dirty_bigcherry=True)
            from bigcherry.campaign import source as campaign_source
            plan_id = campaign_source.materialization_plan_id(
                campaign_source.resolve_materialization_identity(context, plan))
            metadata_path = context.work_root / "sources" / f"{plan_id}.metadata.json"
            import json
            record = json.loads(metadata_path.read_text(encoding="utf-8"))
            record["git_object_format"] = "sha256"
            record["source_slice_id"] = "forged"
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
            from bigcherry.campaign.build import materialize_source
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


class ResolveBuildEnvironmentTests(unittest.TestCase):
    def test_only_the_allowlisted_vars_are_captured(self):
        with patch.dict("os.environ", {
            "PATH": "/usr/bin", "ROCM_PATH": "/opt/rocm",
            "UNRELATED_SHELL_VAR": "should-not-appear",
        }, clear=True):
            env = dict(resolve_build_environment())
        self.assertEqual(env, {"PATH": "/usr/bin", "ROCM_PATH": "/opt/rocm"})

    def test_changing_a_relevant_var_changes_build_plan_id(self):
        with patch.dict("os.environ", {"PATH": "/usr/bin"}, clear=True):
            plan_a = BuildPlan(source_slice_id="s1", phase="tune", platform="p",
                               targets=("gfx1100",), environment=resolve_build_environment())
        with patch.dict("os.environ", {"PATH": "/opt/rocm/bin"}, clear=True):
            plan_b = BuildPlan(source_slice_id="s1", phase="tune", platform="p",
                               targets=("gfx1100",), environment=resolve_build_environment())
        self.assertNotEqual(plan_a.build_plan_id, plan_b.build_plan_id)


if __name__ == "__main__":
    unittest.main()
