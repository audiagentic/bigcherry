"""RE07 (RV48 audit fix): build-plan identity completeness and the
published runtime-bundle artifact, negative/falsification tests.

test_build_identity.py already proves every OTHER BuildPlan field
participates in build_plan_id; this file's job is the two fields RE07
added (catalog_architectures, input_hashes) plus the runtime-bundle
publication B5 required.
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bigcherry.artifacts import ArtifactError, ArtifactStore  # noqa: E402
from bigcherry.builds import BuildPlan, build_directory  # noqa: E402
from bigcherry.context import ProjectContext  # noqa: E402


def _plan(**overrides) -> BuildPlan:
    values = dict(
        source_slice_id="s1", phase="tune", platform="linux-multi",
        targets=("gfx1100", "gfx1201"), variant_set="workload-max",
        catalog_architectures=("gfx1100",),
        input_hashes=(("inventory", "hash-a"),),
    )
    values.update(overrides)
    return BuildPlan(**values)


class CatalogArchitecturesParticipateInIdentityTests(unittest.TestCase):
    def test_changing_catalog_architectures_changes_build_plan_id(self):
        a = _plan(catalog_architectures=("gfx1100",))
        b = _plan(catalog_architectures=("gfx1100", "gfx1201"))
        self.assertNotEqual(a.build_plan_id, b.build_plan_id)

    def test_changing_catalog_architectures_changes_build_directory(self):
        context = ProjectContext(
            project_root=Path("."), config_path=Path("recipes.toml"),
            artifacts_root=Path("artifacts"), work_root=Path("/tmp/bigcherry-work"),
            upstream_repo=Path("upstream"), overlay_root=Path("src"),
            patches_root=Path("patches"),
        )
        a = _plan(catalog_architectures=("gfx1100",))
        b = _plan(catalog_architectures=("gfx1201",))
        # Same everything else -- ONLY the requested generation architecture
        # set differs. RV48's exact finding: these two used to share a
        # build_dir despite representing two different generated catalogs.
        self.assertNotEqual(
            build_directory(context, "s1", a), build_directory(context, "s1", b))

    def test_two_lanes_same_plan_different_architectures_never_share_a_build_dir(self):
        context = ProjectContext(
            project_root=Path("."), config_path=Path("recipes.toml"),
            artifacts_root=Path("artifacts"), work_root=Path("/tmp/bigcherry-work"),
            upstream_repo=Path("upstream"), overlay_root=Path("src"),
            patches_root=Path("patches"),
        )
        lane1 = _plan(catalog_architectures=("gfx1100",))
        lane2 = _plan(catalog_architectures=("gfx1030",))
        dirs = {build_directory(context, "s1", lane1), build_directory(context, "s1", lane2)}
        self.assertEqual(len(dirs), 2)


class InputHashesParticipateInIdentityTests(unittest.TestCase):
    def test_changing_a_declared_needs_input_bytes_changes_build_plan_id(self):
        a = _plan(input_hashes=(("inventory", "hash-a"),))
        b = _plan(input_hashes=(("inventory", "hash-b"),))
        self.assertNotEqual(a.build_plan_id, b.build_plan_id)

    def test_a_third_need_kind_beyond_inventory_and_winners_participates_too(self):
        # RE07's whole point: input_hashes is generic, not hard-coded to
        # {inventory, promoted-winners} -- HI66's correctness-evidence need
        # (or any future kind) must change identity the same way.
        a = _plan(input_hashes=(("inventory", "hash-a"),))
        b = _plan(input_hashes=(
            ("inventory", "hash-a"), ("correctness-evidence", "hash-c")))
        self.assertNotEqual(a.build_plan_id, b.build_plan_id)

    def test_inventory_hash_and_winners_hash_compat_properties_still_work(self):
        plan = _plan(input_hashes=(
            ("inventory", "hash-a"), ("promoted-winners", "hash-w")))
        self.assertEqual(plan.inventory_hash, "hash-a")
        self.assertEqual(plan.winners_hash, "hash-w")

    def test_compat_properties_are_none_when_absent(self):
        plan = _plan(input_hashes=())
        self.assertIsNone(plan.inventory_hash)
        self.assertIsNone(plan.winners_hash)


class RuntimeBundlePublicationTests(unittest.TestCase):
    """B5: the full runtime .so closure is published to ArtifactStore, not
    just the launcher -- verified end-to-end through the real
    make_build_worker path (fake compiler, real publish/verify).
    """

    def test_runtime_bundle_includes_every_so_member_not_just_the_launcher(self):
        from bigcherry import campaign_workers
        from bigcherry import config as campaign_config

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            context = ProjectContext(
                project_root=root, config_path=root / "recipes.toml",
                artifacts_root=root / "artifacts", work_root=root / "work",
                upstream_repo=root / "upstream", overlay_root=root / "src",
                patches_root=root / "patches",
            )
            store = ArtifactStore(root / "store")
            build_plan = BuildPlan(
                source_slice_id="s1", phase="tune", platform="linux-multi",
                targets=("gfx1100",), variant_set=None,
            )
            platform = campaign_config.Platform(
                name="linux-multi", targets=("gfx1100",), options=())
            build_cfg = campaign_config.Build(
                name="stock", options=(), variant_set=None, needs=frozenset())

            build_dir = build_directory(context, "s1", build_plan)

            def fake_compiler(calls):
                def run(args, cwd=None, check=None, **kwargs):
                    calls.append(args)
                    if "--build" in args:
                        build_dir.mkdir(parents=True, exist_ok=True)
                        (build_dir / "llama-bench").write_bytes(b"launcher-bytes")
                        (build_dir / "libggml-hip.so.0").write_bytes(b"hip-dispatch-bytes")
                        (build_dir / "libggml.so.0").write_bytes(b"ggml-base-bytes")
                    else:
                        build_dir.mkdir(parents=True, exist_ok=True)
                        (build_dir / "CMakeCache.txt").write_text(
                            "CMAKE_C_COMPILER:FILEPATH=/opt/rocm/llvm/bin/clang\n"
                            "AMDGPU_TARGETS:STRING=gfx1100\n"
                            "GGML_HIP:BOOL=ON\n", encoding="utf-8")
                    return subprocess.CompletedProcess(args, 0)
                return run

            worker = campaign_workers.make_build_worker(
                context=context, source_root=root / "src-tree", run_id="run1",
                build_plan=build_plan, platform=platform, build=build_cfg,
                store=store, binary_relative_path="llama-bench",
                source_slice_id="s1", workload_id=None,
                has_generate_stage=False, cmake_targets=("llama-bench",),
            )

            calls: list = []
            with patch("bigcherry.campaign_workers.subprocess.run", fake_compiler(calls)):
                refs = worker(())

            bundle_ref = next(ref for ref in refs if ref.kind == "runtime-bundle")
            manifest = json.loads(bundle_ref.path.read_text(encoding="utf-8"))
            self.assertEqual(manifest["entrypoint"], "llama-bench")
            self.assertEqual(
                set(manifest["members"]),
                {"llama-bench", "libggml-hip.so.0", "libggml.so.0"},
                "the runtime bundle must include every .so member alongside "
                "the launcher, not just the launcher itself")
            # Every published member is real, verified, immutable store content.
            for name in manifest["members"]:
                relative = bundle_ref.path.relative_to(store.root).parent / name
                self.assertTrue(store.verify(relative, manifest["members"][name]))

    def test_tampered_bundle_member_bytes_fail_store_verification(self):
        with tempfile.TemporaryDirectory() as directory:
            store = ArtifactStore(Path(directory))
            digest = store.publish_bytes("builds/s1/b1/libggml-hip.so.0", b"real-bytes")
            self.assertTrue(store.verify("builds/s1/b1/libggml-hip.so.0", digest))

            # Tamper with the published bytes directly on disk.
            (Path(directory) / "builds/s1/b1/libggml-hip.so.0").write_bytes(b"tampered!!")
            self.assertFalse(store.verify("builds/s1/b1/libggml-hip.so.0", digest))

            # And a second publish attempt at the same path with the ORIGINAL
            # (correct) bytes must fail closed too -- the immutability
            # guarantee an ArtifactDescriptor consumer relies on.
            with self.assertRaises(ArtifactError):
                store.publish_bytes("builds/s1/b1/libggml-hip.so.0", b"real-bytes")


if __name__ == "__main__":
    unittest.main()
