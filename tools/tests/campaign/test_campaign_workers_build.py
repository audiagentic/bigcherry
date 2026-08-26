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
from typing import cast
from unittest.mock import patch
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from bigcherry.campaign import workers as campaign_workers # noqa: E402
from bigcherry.core import config as campaign_config # noqa: E402
from bigcherry.build import generated_tree # noqa: E402
from bigcherry.core.artifacts import ArtifactStore  # noqa: E402
from bigcherry.build.builds import BuildPlan, build_directory  # noqa: E402
from bigcherry.campaign.build import CampaignBuildError  # noqa: E402
from bigcherry.core.context import ProjectContext  # noqa: E402
from bigcherry.core.pipeline import ArtifactRef  # noqa: E402
from bigcherry.core import provenance # noqa: E402
from bigcherry.core.provenance import ProvenanceClass  # noqa: E402

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
            project_root=directory,
            config_path=directory / "recipes.toml",
            artifacts_root=directory / "artifacts",
            work_root=directory / "work",
            upstream_repo=directory / "upstream",
            overlay_root=directory / "src",
            patches_root=directory / "patches",
        )
        self.store = ArtifactStore(directory / "store")
        self.run_id = "run1"
        self.source_slice_id = "s1"
        self.workload_id = "w1"
        self.build_plan = BuildPlan(
            source_slice_id=self.source_slice_id,
            phase="tune",
            platform="linux-multi",
            targets=("gfx1100",),
            variant_set="workload-max",
        )
        self.platform = campaign_config.Platform(
            name="linux-multi",
            targets=("gfx1100",),
            options=(),
            c_compiler="/opt/rocm/llvm/bin/clang",
            cxx_compiler="/opt/rocm/llvm/bin/clang++",
        )
        self.build_cfg = campaign_config.Build(
            name="tune",
            options=(),
            variant_set="workload-max",
            needs=frozenset({"inventory"}),
        )
        self.calls: list[list[str]] = []

    def generate_inputs(
        self,
        *,
        registry_content: str = "registry",
        label: str = "generate-inputs",
    ) -> tuple[ArtifactRef, ...]:
        generated_root = (
            self.context.work_root / "runs" / self.run_id / "generate" / "generated"
        )
        registry = generated_root / "hip-autotune-registry.inc"
        registry.parent.mkdir(parents=True, exist_ok=True)
        # Overwrite, not append: a second call with different
        # registry_content (simulating a different --arch's generated
        # catalog) must leave the on-disk generated/ tree matching the
        # freshly-built tree_document below, or generated_tree.verify_tree()
        # would reject it as tampered before the reuse decision is even
        # reached.
        registry.write_text(registry_content, encoding="utf-8")
        tree_document = generated_tree.build_manifest(
            generated_root, compile_inputs=(registry,)
        )

        doc = provenance.make(
            project={},
            source={"source_slice_id": self.source_slice_id},
            build={"build_plan_id": self.build_plan.build_plan_id},
            workload={"workload_id": self.workload_id},
            campaign={"run_id": self.run_id},
        )
        # label-scoped store paths: two calls with different content must
        # not collide on ArtifactStore's immutable-publish check the way two
        # calls with the SAME content correctly do (see FreshBuildTests).
        manifest_digest = self.store.publish_json(
            f"{label}/manifest.json", {"candidates": []}
        )
        tree_digest = self.store.publish_json(
            f"{label}/generated-tree.json", tree_document
        )
        return (
            ArtifactRef(
                kind="manifest",
                path=self.store.resolve(f"{label}/manifest.json"),
                content_hash=manifest_digest,
                provenance=doc,
            ),
            ArtifactRef(
                kind="generated-tree",
                path=self.store.resolve(f"{label}/generated-tree.json"),
                content_hash=tree_digest,
                provenance=doc,
            ),
        )

    def worker(
        self,
        local_provenance_class: ProvenanceClass = "development",
        backend: str = "hip",
        extra_binary_names: tuple[str, ...] = (),
    ):
        return campaign_workers.make_build_worker(
            context=self.context,
            source_root=self.directory / "source",
            run_id=self.run_id,
            build_plan=self.build_plan,
            platform=self.platform,
            build=self.build_cfg,
            store=self.store,
            binary_relative_path="bin/llama-bench",
            source_slice_id=self.source_slice_id,
            workload_id=self.workload_id,
            cmake_targets=("llama-bench",) + extra_binary_names,
            extra_binary_names=extra_binary_names,
            # RE25.3: these are mechanism/reuse tests with fixture (not
            # production) provenance docs; development class keeps the
            # production-only publish-time kind contract out of scope.
            local_provenance_class=local_provenance_class,
            backend=backend,
        )

    def fake_compiler(
        self,
        cmake_cache_text: str = _CMAKE_CACHE,
        hip_so_content: bytes = b"hip-dispatch-v1",
        extra_binary_names: tuple[str, ...] = (),
    ):
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
                (build_dir / "bin" / "libggml-hip.so.0.19.0").write_bytes(
                    hip_so_content
                )
                for name in extra_binary_names:
                    (build_dir / "bin" / name).write_bytes(f"extra-{name}".encode())
            else:
                (build_dir / "CMakeCache.txt").write_text(
                    cmake_cache_text, encoding="utf-8"
                )
            return subprocess.CompletedProcess(cmd, 0)

        return run


class FreshBuildTests(unittest.TestCase):
    def test_compiles_for_real_and_records_reuse_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            harness = _Harness(Path(directory))
            inputs = harness.generate_inputs()
            with patch(
                "bigcherry.campaign.workers.subprocess.run", harness.fake_compiler()
            ):
                refs = harness.worker()(inputs)

            self.assertEqual(len(harness.calls), 2)  # configure, then build
            self.assertEqual(len(refs), 2)
            self.assertEqual(refs[0].kind, "binary")
            self.assertEqual(refs[1].kind, "runtime-bundle")

            build_dir = build_directory(
                harness.context, harness.source_slice_id, harness.build_plan
            )
            metadata_path = build_dir / "bigcherry-build-metadata-llama-bench.json"
            self.assertTrue(metadata_path.is_file())
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            self.assertEqual(metadata["source_slice_id"], "s1")
            self.assertEqual(
                metadata["build_plan_id"], harness.build_plan.build_plan_id
            )
            self.assertIn("CMAKE_C_COMPILER", metadata["effective_configure"])


class BackendThreadingTests(unittest.TestCase):
    """RE30 phase 3: make_build_worker's backend= parameter must actually
    reach the real configure argv, not just default silently to HIP."""

    def test_default_backend_configures_hip_unchanged(self):
        with tempfile.TemporaryDirectory() as directory:
            harness = _Harness(Path(directory))
            inputs = harness.generate_inputs()
            with patch(
                "bigcherry.campaign.workers.subprocess.run", harness.fake_compiler()
            ):
                harness.worker()(inputs)  # backend defaults to "hip"
            configure_call = harness.calls[0]
            self.assertIn("-DAMDGPU_TARGETS=gfx1100", configure_call)
            self.assertFalse(any("GGML_VULKAN" in arg for arg in configure_call))

    def test_vulkan_backend_reaches_the_real_configure_call(self):
        with tempfile.TemporaryDirectory() as directory:
            harness = _Harness(Path(directory))
            inputs = harness.generate_inputs()
            with patch(
                "bigcherry.campaign.workers.subprocess.run", harness.fake_compiler()
            ):
                harness.worker(backend="vulkan")(inputs)
            configure_call = harness.calls[0]
            self.assertIn("-DGGML_VULKAN=ON", configure_call)
            self.assertFalse(any("AMDGPU_TARGETS" in arg for arg in configure_call))


class Re25ParentLineageTests(unittest.TestCase):
    def test_build_outputs_name_their_real_parent_artifacts(self):
        # RE25.2 review fix: build must record the artifacts it actually
        # consumed -- generate's manifest/tree plus the lane inputs -- into
        # its own provenance, or the lineage chain terminates at build:
        # generate records its parents, but a build republished with an
        # empty parent set severs exactly the chain release provenance
        # needs. Pre-descriptor refs (no artifact_id yet) carry their
        # content_hash as the identity; RE25.3 replaces that fallback with
        # real descriptors.
        with tempfile.TemporaryDirectory() as directory:
            harness = _Harness(Path(directory))
            inputs = harness.generate_inputs()
            inventory_digest = harness.store.publish_bytes(
                "inputs/inventory.json", b'{"mmq_types": ["q8_0"]}'
            )
            # RE25.3: in production every lane input is normalized by
            # _resolve_lane_inputs before a worker ever sees it (a raw
            # Path/imported ref comes out with a valid schema-v2 doc), so
            # the build worker's fail-closed parent-doc check needs a
            # realistic document here -- not an empty dict.
            inventory_ref = ArtifactRef(
                kind="inventory",
                path=harness.store.resolve("inputs/inventory.json"),
                content_hash=inventory_digest,
                provenance=provenance.make(
                    project={"provenance_class": "imported-legacy"},
                    source={},
                    build={},
                    workload={},
                    campaign={},
                ),
            )
            worker = campaign_workers.make_build_worker(
                context=harness.context,
                source_root=harness.directory / "source",
                run_id=harness.run_id,
                build_plan=harness.build_plan,
                platform=harness.platform,
                build=harness.build_cfg,
                store=harness.store,
                binary_relative_path="bin/llama-bench",
                source_slice_id=harness.source_slice_id,
                workload_id=harness.workload_id,
                cmake_targets=("llama-bench",),
                lane_inputs={"inventory": inventory_ref},
                # RE25.3: mechanism test -- development class (see harness).
                local_provenance_class="development",
            )
            with patch(
                "bigcherry.campaign.workers.subprocess.run", harness.fake_compiler()
            ):
                refs = worker(inputs)

            expected_ids = sorted(
                [ref.content_hash for ref in inputs] + [inventory_digest]
            )
            by_kind = {ref.kind: ref for ref in refs}
            self.assertEqual(set(by_kind), {"binary", "runtime-bundle"})
            for kind in ("binary", "runtime-bundle"):
                doc = by_kind[kind].provenance
                # provenance is a dict[str, object] (JSON-shaped), so the
                # nested namespaces are `object` to pyright -- cast each level.
                campaign_doc = cast("dict[str, object]", doc["campaign"])
                # campaign.producer_artifact_ids names the actual parents.
                self.assertEqual(
                    campaign_doc.get("producer_artifact_ids"), expected_ids
                )
                build_doc = cast("dict[str, object]", doc["build"])
                raw_inputs = cast("list[object]", build_doc.get("inputs") or [])
                # build.inputs records them by name, with the content-hash
                # identity fallback for refs that have no descriptor yet.
                input_entries = {
                    cast("dict[str, object]", entry).get("name"): cast(
                        "dict[str, object]", entry
                    ).get("artifact_id")
                    for entry in raw_inputs
                }
                self.assertEqual(
                    set(input_entries), {"manifest", "generated-tree", "inventory"}
                )
                for parent in inputs:
                    self.assertEqual(input_entries[parent.kind], parent.content_hash)
                self.assertEqual(input_entries["inventory"], inventory_digest)


class ReuseTests(unittest.TestCase):
    def test_second_call_with_same_identity_reuses_without_compiling(self):
        with tempfile.TemporaryDirectory() as directory:
            harness = _Harness(Path(directory))
            with patch(
                "bigcherry.campaign.workers.subprocess.run", harness.fake_compiler()
            ):
                harness.worker()(harness.generate_inputs())
            self.assertEqual(len(harness.calls), 2)

            # A second, independent worker call -- as if a different
            # process/run_id encountered the same content-addressed
            # build_directory(). subprocess.run is patched to always raise,
            # so any attempt to recompile fails the test outright rather
            # than merely going undetected.
            with patch(
                "bigcherry.campaign.workers.subprocess.run",
                side_effect=AssertionError("must not recompile on reuse"),
            ):
                refs = harness.worker()(harness.generate_inputs())

            self.assertEqual(len(refs), 2)
            self.assertEqual(refs[0].kind, "binary")
            self.assertEqual(refs[1].kind, "runtime-bundle")

    def test_metadata_present_but_binary_tampered_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            harness = _Harness(Path(directory))
            with patch(
                "bigcherry.campaign.workers.subprocess.run", harness.fake_compiler()
            ):
                harness.worker()(harness.generate_inputs())

            build_dir = build_directory(
                harness.context, harness.source_slice_id, harness.build_plan
            )
            (build_dir / "bin" / "llama-bench").write_bytes(b"tampered-after-the-fact")

            with patch(
                "bigcherry.campaign.workers.subprocess.run",
                side_effect=AssertionError(
                    "must not silently recompile over a failed check"
                ),
            ):
                with self.assertRaisesRegex(
                    CampaignBuildError, "failed reuse validation"
                ):
                    harness.worker()(harness.generate_inputs())

    def test_different_generated_catalog_under_the_same_build_plan_recompiles_not_reuses(
        self,
    ):
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
            with patch(
                "bigcherry.campaign.workers.subprocess.run", harness.fake_compiler()
            ):
                harness.worker()(
                    harness.generate_inputs(
                        registry_content="registry-for-gfx1100", label="arch-a"
                    )
                )
            first_calls = len(harness.calls)
            self.assertEqual(first_calls, 2)

            # A DIFFERENT generated catalog (simulating a different --arch),
            # same build_plan_id/build_dir. This must NOT silently reuse the
            # first build's binary -- it must recompile for real.
            arch_b_inputs = harness.generate_inputs(
                registry_content="registry-for-gfx1201-DIFFERENT", label="arch-b"
            )
            with patch(
                "bigcherry.campaign.workers.subprocess.run", harness.fake_compiler()
            ):
                refs = harness.worker()(arch_b_inputs)

            self.assertEqual(
                len(harness.calls), first_calls + 2
            )  # recompiled, not reused
            self.assertEqual(len(refs), 2)
            self.assertEqual(refs[0].kind, "binary")
            self.assertEqual(refs[1].kind, "runtime-bundle")

            arch_b_tree_document = json.loads(
                arch_b_inputs[1].path.read_text(encoding="utf-8")
            )
            build_dir = build_directory(
                harness.context, harness.source_slice_id, harness.build_plan
            )
            metadata = json.loads(
                (build_dir / "bigcherry-build-metadata-llama-bench.json").read_text(
                    encoding="utf-8"
                )
            )
            # The recorded hash now reflects the SECOND (arch-b) catalog,
            # not the first -- a third call with arch-b's exact content
            # would correctly reuse; a third call with arch-a's would not.
            self.assertEqual(
                metadata["generated_compile_inputs_hash"],
                arch_b_tree_document["compile_inputs_hash"],
            )

    def test_cmake_generated_dir_is_stable_across_run_ids(self):
        # RD100: GGML_HIP_AUTOTUNE_GENERATED_DIR used to be the raw,
        # run_id-scoped staging path (work_root/runs/<run_id>/generate/
        # generated) -- an absolute path that differs on every invocation
        # and leaks into the compiled output (include search paths /
        # embedded diagnostics), making two builds of IDENTICAL generated
        # content byte-different purely because run_id differed. That then
        # collided with ArtifactStore's immutability check on re-publish.
        # A build worker must always pass the SAME (build_dir-scoped)
        # absolute path to CMake for the same (source_slice_id,
        # build_plan_id), regardless of which run_id's generate stage
        # produced the content.
        with tempfile.TemporaryDirectory() as directory:
            harness = _Harness(Path(directory))
            build_dir = build_directory(
                harness.context, harness.source_slice_id, harness.build_plan
            )

            def generated_dir_arg(cmd: list[str]) -> str:
                prefix = "-DGGML_HIP_AUTOTUNE_GENERATED_DIR="
                (arg,) = (a for a in cmd if a.startswith(prefix))
                return arg[len(prefix):]

            harness.run_id = "run1"
            with patch(
                "bigcherry.campaign.workers.subprocess.run", harness.fake_compiler()
            ):
                harness.worker()(
                    harness.generate_inputs(registry_content="content-a", label="a")
                )
            first_configure = harness.calls[0]
            first_dir = generated_dir_arg(first_configure)

            # Force a second REAL compile (not a reuse) by changing the
            # generated content, same as the arch-a/arch-b test above --
            # but this time also change run_id, which previously would have
            # changed the generated-dir path passed to CMake too.
            harness.run_id = "run2"
            with patch(
                "bigcherry.campaign.workers.subprocess.run", harness.fake_compiler()
            ):
                harness.worker()(
                    harness.generate_inputs(registry_content="content-b", label="b")
                )
            second_configure = harness.calls[2]
            second_dir = generated_dir_arg(second_configure)

            self.assertEqual(first_dir, second_dir)
            self.assertEqual(Path(first_dir), (build_dir / "generated-inputs").resolve())
            self.assertNotIn("run1", first_dir)
            self.assertNotIn("run2", second_dir)

    def test_missing_extra_binary_forces_recompile_not_silent_reuse(self):
        # RD100 (gpt-auto-agent review follow-up): current_runtime_hash was
        # left None whenever an expected extra binary was absent, so
        # validate_reuse() skipped the runtime-bundle comparison entirely --
        # but `reused` was still computed from compile_inputs_hash alone,
        # so a build_dir missing a requested extra binary (e.g. pruned, or
        # from before this lane asked for it) could be treated as a valid
        # reuse and returned without ever recompiling the missing artifact.
        with tempfile.TemporaryDirectory() as directory:
            harness = _Harness(Path(directory))
            with patch(
                "bigcherry.campaign.workers.subprocess.run",
                harness.fake_compiler(extra_binary_names=("extra-tool",)),
            ):
                harness.worker(extra_binary_names=("extra-tool",))(
                    harness.generate_inputs()
                )
            first_calls = len(harness.calls)
            self.assertEqual(first_calls, 2)

            build_dir = build_directory(
                harness.context, harness.source_slice_id, harness.build_plan
            )
            (build_dir / "bin" / "extra-tool").unlink()

            # Same identity, same generated content (a genuine reuse
            # candidate) -- but the extra binary is now missing. This must
            # recompile, not silently report success while missing the
            # requested extra.
            with patch(
                "bigcherry.campaign.workers.subprocess.run",
                harness.fake_compiler(extra_binary_names=("extra-tool",)),
            ):
                refs = harness.worker(extra_binary_names=("extra-tool",))(
                    harness.generate_inputs()
                )

            self.assertEqual(len(harness.calls), first_calls + 2)  # recompiled
            self.assertTrue((build_dir / "bin" / "extra-tool").is_file())
            self.assertEqual(len(refs), 2)

    def test_dependent_library_tampered_fails_closed_even_though_launcher_is_untouched(
        self,
    ):
        # gpt-auto-agent review finding: "you're hashing one executable, not
        # necessarily the build-output closure" -- RE09 established the
        # real HIP dispatch logic lives in libggml-hip.so, not the launcher.
        # A reuse check based on binary_hash alone would miss this.
        with tempfile.TemporaryDirectory() as directory:
            harness = _Harness(Path(directory))
            with patch(
                "bigcherry.campaign.workers.subprocess.run", harness.fake_compiler()
            ):
                harness.worker()(harness.generate_inputs())

            build_dir = build_directory(
                harness.context, harness.source_slice_id, harness.build_plan
            )
            # The launcher itself is byte-identical; only its dependent
            # library changed.
            (build_dir / "bin" / "libggml-hip.so.0.19.0").write_bytes(
                b"TAMPERED-hip-dispatch"
            )

            with patch(
                "bigcherry.campaign.workers.subprocess.run",
                side_effect=AssertionError(
                    "must not silently recompile over a failed check"
                ),
            ):
                with self.assertRaisesRegex(
                    CampaignBuildError, "failed reuse validation"
                ):
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
            with patch(
                "bigcherry.campaign.workers.subprocess.run", harness.fake_compiler()
            ):
                harness.worker()(harness.generate_inputs())

            build_dir = build_directory(
                harness.context, harness.source_slice_id, harness.build_plan
            )
            metadata_path = build_dir / "bigcherry-build-metadata-llama-bench.json"
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            metadata["build_plan_id"] = "a-different-build-plan-id"
            metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

            with patch(
                "bigcherry.campaign.workers.subprocess.run",
                side_effect=AssertionError(
                    "must not silently recompile over a failed check"
                ),
            ):
                with self.assertRaisesRegex(
                    CampaignBuildError, "failed reuse validation"
                ):
                    harness.worker()(harness.generate_inputs())


if __name__ == "__main__":
    unittest.main()
