"""Content-addressed build identity contracts."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from bigcherry.build import builds # noqa: E402
from bigcherry.build.builds import (BuildIdentityError, BuildPlan, CommandRequirement,
                              binary_hash, effective_build_id, parse_effective_configure,
                              post_build_verify, resolve_runtime_artifacts,
                              runtime_bundle_hash, validate_reuse)  # noqa: E402


class BuildIdentityTests(unittest.TestCase):
    def _plan(self, **changes):
        values = dict(
            source_slice_id="s1", phase="record", platform="brutus",
            targets=("gfx1100",), cmake_options=(("GGML_HIP", "ON"),),
            variant_set="inventory", environment=(("CC", "clang"),),
        )
        values.update(changes)
        return BuildPlan(**values)

    def test_build_plan_changes_for_every_material_input(self):
        base = self._plan().build_plan_id
        for field, value in {
            "source_slice_id": "s2", "phase": "tune", "platform": "other",
            "targets": ("gfx1201",), "cmake_options": (("GGML_HIP", "OFF"),),
            "variant_set": "full-max", "catalog_architectures": ("gfx1201",),
            "input_hashes": (("inventory", "i"), ("promoted-winners", "w")),
            "resource_report_hashes": ("r",),
            "environment": (("CXXFLAGS", "-O0"),),
        }.items():
            self.assertNotEqual(base, self._plan(**{field: value}).build_plan_id, field)

    def test_reuse_validates_configure_and_binary_hash(self):
        with tempfile.TemporaryDirectory() as directory:
            binary = Path(directory) / "bench"
            binary.write_bytes(b"binary")
            plan = self._plan()
            configure = {"source": "s1", "generator": "Ninja", "options": {"A": "B"}}
            metadata = {
                "source_slice_id": plan.source_slice_id,
                "build_plan_id": plan.build_plan_id,
                "effective_configure": configure,
                "build_id": effective_build_id(configure),
                "binary_hash": binary_hash(binary),
            }
            validate_reuse(metadata, plan, binary=binary)
            binary.write_bytes(b"changed")
            with self.assertRaisesRegex(BuildIdentityError, "binary hash"):
                validate_reuse(metadata, plan, binary=binary)


class RuntimeBundleTests(unittest.TestCase):
    def test_resolve_runtime_artifacts_includes_real_shared_libs_not_symlinks(self):
        with tempfile.TemporaryDirectory() as directory:
            bin_dir = Path(directory) / "bin"
            bin_dir.mkdir()
            binary = bin_dir / "llama-bench"
            binary.write_bytes(b"launcher")
            hip_so = bin_dir / "libggml-hip.so.0.19.0"
            hip_so.write_bytes(b"hip-dispatch-logic")
            try:
                (bin_dir / "libggml-hip.so").symlink_to(hip_so.name)
                (bin_dir / "libggml-hip.so.0").symlink_to(hip_so.name)
            except OSError as exc:
                self.skipTest(
                    "runtime-bundle symlink coverage requires symlink creation "
                    f"privilege, unavailable in this environment: {exc}"
                )
            # A file that happens to match *.so* but isn't actually a
            # library some other tool dropped there -- still real, still
            # part of the closure by this function's directory-membership
            # rule (deliberately simple; see its docstring).
            other_so = bin_dir / "libggml.so.0.19.0"
            other_so.write_bytes(b"core-ops")

            artifacts = resolve_runtime_artifacts(binary)

            self.assertEqual(set(artifacts), {binary, hip_so, other_so})
            for path in artifacts:
                self.assertFalse(path.is_symlink())

    def test_resolve_runtime_artifacts_includes_extra_binaries(self):
        # RE26: a lane can request additional executables (e.g.
        # test-backend-ops alongside the tune lane's llama-bench) that share
        # the same runtime bundle -- they must appear in the resolved
        # artifact set alongside the main binary and its shared libs.
        with tempfile.TemporaryDirectory() as directory:
            bin_dir = Path(directory) / "bin"
            bin_dir.mkdir()
            binary = bin_dir / "llama-bench"
            binary.write_bytes(b"launcher")
            hip_so = bin_dir / "libggml-hip.so.0.19.0"
            hip_so.write_bytes(b"hip-dispatch-logic")
            extra = bin_dir / "test-backend-ops"
            extra.write_bytes(b"direct-op-corpus")

            artifacts = resolve_runtime_artifacts(binary, extra_binaries=(extra,))

            self.assertEqual(set(artifacts), {binary, hip_so, extra})

    def test_runtime_bundle_hash_changes_when_any_dependent_library_changes(self):
        # The exact gpt-auto-agent finding: a reuse check based only on the
        # requested launcher's hash could accept a cache hit even though
        # libggml-hip.so (where the real HIP dispatch logic lives, per
        # RE09) changed underneath it.
        base = {"llama-bench": "aaa", "libggml-hip.so.0.19.0": "bbb"}
        changed_hip_lib = {"llama-bench": "aaa", "libggml-hip.so.0.19.0": "ccc"}
        self.assertNotEqual(runtime_bundle_hash(base), runtime_bundle_hash(changed_hip_lib))

    def test_validate_reuse_accepts_matching_runtime_bundle_and_rejects_mismatch(self):
        with tempfile.TemporaryDirectory() as directory:
            binary = Path(directory) / "bench"
            binary.write_bytes(b"binary")
            plan = BuildPlan(source_slice_id="s1", phase="tune", platform="p",
                             targets=("gfx1100",))
            configure = {"generator": "Ninja"}
            artifacts = {"bench": binary_hash(binary), "libggml-hip.so.0.19.0": "bbb"}
            metadata = {
                "source_slice_id": plan.source_slice_id,
                "build_plan_id": plan.build_plan_id,
                "effective_configure": configure,
                "build_id": effective_build_id(configure),
                "binary_hash": binary_hash(binary),
                "runtime_bundle_hash": runtime_bundle_hash(artifacts),
            }

            validate_reuse(metadata, plan, binary=binary,
                           runtime_bundle_hash=runtime_bundle_hash(artifacts))

            # The launcher itself is untouched, but a dependent library
            # changed -- runtime_bundle_hash must catch this even though
            # binary_hash alone would not.
            tampered_artifacts = {**artifacts, "libggml-hip.so.0.19.0": "TAMPERED"}
            with self.assertRaisesRegex(BuildIdentityError, "runtime bundle hash"):
                validate_reuse(metadata, plan, binary=binary,
                               runtime_bundle_hash=runtime_bundle_hash(tampered_artifacts))


_CMAKE_CACHE = """\
# This is the CMakeCache file.
# For build in directory: /tmp/build
//No help, variable specified on the command line.
CMAKE_C_COMPILER:FILEPATH=/opt/rocm/llvm/bin/clang
CMAKE_CXX_COMPILER:FILEPATH=/opt/rocm/llvm/bin/clang++
CMAKE_BUILD_TYPE:STRING=Release
AMDGPU_TARGETS:STRING=gfx1100;gfx1201
GGML_HIP:BOOL=ON
GGML_HIP_AUTOTUNE_VARIANT_SET:STRING=workload-max
CMAKE_GENERATOR:INTERNAL=Ninja
FETCHCONTENT_BASE_DIR:PATH=/tmp/build/_deps
//Some unrelated cached find_package probe result
Boost_INCLUDE_DIR:PATH=/usr/include
"""


class ParseEffectiveConfigureTests(unittest.TestCase):
    def test_extracts_only_the_build_identity_relevant_keys(self):
        with tempfile.TemporaryDirectory() as directory:
            cache = Path(directory) / "CMakeCache.txt"
            cache.write_text(_CMAKE_CACHE, encoding="utf-8")
            record = parse_effective_configure(cache)
            self.assertEqual(record, {
                "CMAKE_C_COMPILER": "/opt/rocm/llvm/bin/clang",
                "CMAKE_CXX_COMPILER": "/opt/rocm/llvm/bin/clang++",
                "CMAKE_BUILD_TYPE": "Release",
                "AMDGPU_TARGETS": "gfx1100;gfx1201",
                "GGML_HIP": "ON",
                "GGML_HIP_AUTOTUNE_VARIANT_SET": "workload-max",
            })
            # Unrelated cache noise (generator internals, unrelated
            # find_package probe results) must not leak in -- it would make
            # effective_build_id() sensitive to changes that say nothing
            # about what was actually built.
            self.assertNotIn("CMAKE_GENERATOR", record)
            self.assertNotIn("FETCHCONTENT_BASE_DIR", record)
            self.assertNotIn("Boost_INCLUDE_DIR", record)

    def test_missing_cache_file_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(BuildIdentityError, "no CMakeCache.txt"):
                parse_effective_configure(Path(directory) / "CMakeCache.txt")

    def test_cache_with_no_relevant_keys_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            cache = Path(directory) / "CMakeCache.txt"
            cache.write_text("CMAKE_GENERATOR:INTERNAL=Ninja\n", encoding="utf-8")
            with self.assertRaisesRegex(BuildIdentityError, "no relevant configure keys"):
                parse_effective_configure(cache)

    def test_real_configure_then_reuse_round_trip(self):
        # Ties parse_effective_configure directly to validate_reuse/
        # effective_build_id, the way campaign_workers.make_build_worker
        # actually uses it: parse a cache, build a metadata record from it,
        # confirm a real BuildPlan reuses cleanly against its own recorded
        # identity and rejects a binary that no longer matches.
        with tempfile.TemporaryDirectory() as directory:
            cache = Path(directory) / "CMakeCache.txt"
            cache.write_text(_CMAKE_CACHE, encoding="utf-8")
            binary = Path(directory) / "llama-bench"
            binary.write_bytes(b"real-binary-bytes")

            plan = BuildPlan(
                source_slice_id="s1", phase="tune", platform="linux-multi",
                targets=("gfx1100", "gfx1201"), variant_set="workload-max")
            effective_configure = parse_effective_configure(cache)
            metadata = {
                "source_slice_id": plan.source_slice_id,
                "build_plan_id": plan.build_plan_id,
                "effective_configure": effective_configure,
                "build_id": effective_build_id(effective_configure),
                "binary_hash": binary_hash(binary),
            }

            validate_reuse(metadata, plan, binary=binary)  # must not raise

            binary.write_bytes(b"a different compile produced this")
            with self.assertRaisesRegex(BuildIdentityError, "binary hash"):
                validate_reuse(metadata, plan, binary=binary)


class CompileVerificationTests(unittest.TestCase):
    """HI82: post_build_verify() proves configured intent (arch,
    CMAKE_HIP_FLAGS) reached the real compiled command line, not just
    CMakeCache.txt's claimed configuration -- generalizes the HI81 Windows
    CMAKE_HIP_FLAGS-propagation gap into an automatic check."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.source = self.root / "source"
        self.build = self.root / "build"
        self.source.mkdir()
        self.build.mkdir()

    def tearDown(self):
        self._tmp.cleanup()

    def _cache(self, *, hip_flags: str = "") -> None:
        (self.build / "CMakeCache.txt").write_text(
            "\n".join((
                "CMAKE_GENERATOR:INTERNAL=Ninja",
                "CMAKE_HIP_ARCHITECTURES:STRING=gfx1100",
                f"CMAKE_HIP_FLAGS:STRING={hip_flags}",
                "",
            )),
            encoding="utf-8",
        )

    def _commands(self, hip_command: str) -> None:
        payload = [
            {
                "directory": str(self.build),
                "file": str(self.source / "kernel.cu"),
                "command": hip_command,
            },
            {
                "directory": str(self.build),
                "file": str(self.source / "host.cpp"),
                "command": f"clang++ -c {self.source / 'host.cpp'} -o {self.build / 'host.o'}",
            },
        ]
        (self.build / "compile_commands.json").write_text(json.dumps(payload), encoding="utf-8")

    def test_accepts_hip_flags_that_reached_compiler(self):
        flag = "-funsafe-math-optimizations"
        self._cache(hip_flags=flag)
        self._commands(
            f"clang++ -c {self.source / 'kernel.cu'} --offload-arch=gfx1100 "
            f"{flag} -o {self.build / 'kernel.o'}"
        )

        evidence = post_build_verify(self.build, source_root=self.source, architecture="gfx1100")

        self.assertEqual(evidence.command_source, "compile_commands.json")
        self.assertEqual(evidence.hip_compile_command_count, 1)

        labels = {check.label for check in evidence.checks}
        self.assertIn("hip-architecture", labels)
        self.assertIn("cmake-hip-flags-propagation", labels)

    def test_rejects_cmake_hip_flags_that_did_not_propagate(self):
        self._cache(hip_flags="-funsafe-math-optimizations")
        self._commands(
            f"clang++ -c {self.source / 'kernel.cu'} --offload-arch=gfx1100 "
            f"-o {self.build / 'kernel.o'}"
        )

        with self.assertRaisesRegex(BuildIdentityError, "did not propagate"):
            post_build_verify(self.build, source_root=self.source, architecture="gfx1100")

    def test_rejects_wrong_real_architecture(self):
        self._cache()
        self._commands(
            f"clang++ -c {self.source / 'kernel.cu'} --offload-arch=gfx1201 "
            f"-o {self.build / 'kernel.o'}"
        )

        with self.assertRaisesRegex(BuildIdentityError, "gfx1100"):
            post_build_verify(self.build, source_root=self.source, architecture="gfx1100")

    def test_custom_required_command_token(self):
        self._cache()
        self._commands(
            f"clang++ -c {self.source / 'kernel.cu'} --offload-arch=gfx1100 "
            f"-DBIGCHERRY_EXAMPLE=1 -o {self.build / 'kernel.o'}"
        )

        evidence = post_build_verify(
            self.build, source_root=self.source, architecture="gfx1100",
            command_requirements=(
                CommandRequirement(
                    label="example patch flag", selector_regex=r"(?i)\.cu\b",
                    required_tokens=("-DBIGCHERRY_EXAMPLE=1",),
                ),
            ),
        )

        self.assertEqual(evidence.checks[-1].label, "example patch flag")
        self.assertEqual(evidence.checks[-1].status, "pass")

    def test_custom_forbidden_command_token(self):
        self._cache()
        self._commands(
            f"clang++ -c {self.source / 'kernel.cu'} --offload-arch=gfx1100 "
            f"-DBIGCHERRY_BAD_FLAG=1 -o {self.build / 'kernel.o'}"
        )

        with self.assertRaisesRegex(BuildIdentityError, "forbidden_present"):
            post_build_verify(
                self.build, source_root=self.source, architecture="gfx1100",
                command_requirements=(
                    CommandRequirement(
                        label="negative control", selector_regex=r"(?i)\.cu\b",
                        forbidden_tokens=("-DBIGCHERRY_BAD_FLAG=1",),
                    ),
                ),
            )

    def test_runtime_artifacts_include_windows_dlls(self):
        with tempfile.TemporaryDirectory() as directory:
            bin_dir = Path(directory)
            binary = bin_dir / "llama-bench.exe"
            binary.write_bytes(b"launcher")
            hip = bin_dir / "ggml-hip.dll"
            hip.write_bytes(b"hip")
            core = bin_dir / "ggml.dll"
            core.write_bytes(b"core")

            self.assertEqual(set(resolve_runtime_artifacts(binary)), {binary, hip, core})

    def test_requested_config_must_match_cache(self):
        self._cache(hip_flags="")
        (self.build / "CMakeCache.txt").write_text(
            "\n".join((
                "GGML_HIP:BOOL=ON",
                "GGML_HIP_AUTOTUNE:BOOL=OFF",
                "AMDGPU_TARGETS:STRING=gfx1100",
                "",
            )),
            encoding="utf-8",
        )
        self._commands(
            f"clang++ -c {self.source / 'kernel.cu'} --offload-arch=gfx1100 "
            f"-o {self.build / 'kernel.o'}"
        )

        with self.assertRaisesRegex(BuildIdentityError, "requested CMake configuration"):
            post_build_verify(
                self.build, source_root=self.source, architecture="gfx1100",
                requested_cmake_args={
                    "GGML_HIP": "ON", "GGML_HIP_AUTOTUNE": "ON", "AMDGPU_TARGETS": "gfx1100",
                },
            )

    def test_ninja_link_command_is_not_hip_compile(self):
        command = builds._Command(
            source="", directory="/tmp/build",
            text="clang++ objects/foo.cu.o objects/bar.cpp.o -o bin/llama-bench",
        )
        self.assertFalse(builds._is_hip_compile(command))


if __name__ == "__main__":
    unittest.main()
