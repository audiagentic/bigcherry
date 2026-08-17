"""RE24: extended adversarial cutover matrix for the RV48-audit-found
integrity failures, proven END-TO-END through execute_campaign_lane()
itself -- the real public production API -- not just the focused
per-function unit tests RE04/RE07/RE08 already carry (test_re04_
materialization_safety.py, test_re07_build_identity.py, test_re07_
smoke_bundle_consumption.py, test_re08_provenance_import_boundary.py).

Real git materialize, real patch application, real BuildPlan identity,
real ArtifactStore publish/verify throughout. Only campaign_workers.
subprocess.run (the actual cmake/compiler invocation) is faked, so these
tests run without a real C++ toolchain -- exactly the same seam
test_re07_build_identity.py's real make_build_worker test already uses.
The build (``stock``: needs=[], variant_set=None) has no generate stage,
so nothing here depends on llama.cpp's real source tree shape either.
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bigcherry import config as campaign_config  # noqa: E402
from bigcherry.artifacts import ArtifactStore  # noqa: E402
from bigcherry.builds import build_directory  # noqa: E402
from bigcherry.campaign_lane import (CampaignLaneError, CampaignLaneExecutionSpec,  # noqa: E402
                                     execute_campaign_lane)
from bigcherry.context import ProjectContext  # noqa: E402
from bigcherry.workspace import WorkspaceError  # noqa: E402


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


def _init_project(directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    _git(directory, "init", str(directory))
    _git(directory, "config", "user.email", "test@example.invalid")
    (directory / ".gitkeep").write_text("", encoding="utf-8")
    _git(directory, "add", ".gitkeep")
    _git(directory, "config", "user.name", "Test")
    _git(directory, "commit", "-m", "initial")


def _write_marker_patch(patches_root: Path, *, marker_text: str) -> None:
    patches_root.mkdir(parents=True, exist_ok=True)
    (patches_root / "0001_marker.py").write_text(
        "from bigcherry.patcher import Edit, FilePatch\n"
        "GROUP = 'core'\n"
        "STATE = 'validated'\n"
        "PATCH = FilePatch(\n"
        "    path='source.txt',\n"
        "    edits=(Edit(\n"
        "        id='marker',\n"
        "        anchor=r'one',\n"
        f"        text='{marker_text}',\n"
        "        mode='insert_after',\n"
        "    ),),\n"
        ")\n",
        encoding="utf-8",
    )


_REAL_SUBPROCESS_RUN = subprocess.run


def _fake_compiler(calls: list):
    """Stands in for cmake/the compiler only -- ``import subprocess``
    (whole-module import) means patching ``bigcherry.campaign_workers.
    subprocess.run`` patches the SAME global ``subprocess.run`` every
    other module (workspace.py's own git calls, this file's own ``_git``
    helper) sees too, so anything that is not actually a cmake/compiler
    invocation must be passed straight through to the real subprocess.run
    -- otherwise real git commands silently get a fake CompletedProcess
    with no stdout/stderr and crash their own callers.

    Writes a real CMakeCache.txt on configure and a real (fake-content)
    binary + one .so member on build, so every REAL downstream identity/
    publish/verify code path still runs.
    """
    def run(args, cwd=None, check=None, **kwargs):
        argv = args if isinstance(args, (list, tuple)) else [args]
        if not argv or Path(str(argv[0])).name.split(".")[0] not in ("cmake",):
            return _REAL_SUBPROCESS_RUN(args, cwd=cwd, check=check, **kwargs)
        calls.append(args)
        if "--build" in argv:
            build_dir = Path(argv[2])
            (build_dir / "bin").mkdir(parents=True, exist_ok=True)
            (build_dir / "bin" / "llama-bench").write_bytes(b"launcher-bytes")
            (build_dir / "bin" / "libggml-hip.so.0").write_bytes(b"hip-dispatch-bytes")
        else:
            build_dir = Path(argv[argv.index("-B") + 1]) if "-B" in argv else None
            if build_dir is not None:
                build_dir.mkdir(parents=True, exist_ok=True)
                (build_dir / "CMakeCache.txt").write_text(
                    "CMAKE_C_COMPILER:FILEPATH=/opt/rocm/llvm/bin/clang\n"
                    "AMDGPU_TARGETS:STRING=gfx1100\n"
                    "GGML_HIP:BOOL=ON\n", encoding="utf-8")
        return subprocess.CompletedProcess(args, 0, stdout="", stderr="")
    return run


class _Harness:
    def __init__(self, directory: Path, *, with_patch: bool, marker_text: str = "marker-v1"):
        self.directory = directory
        upstream, revision = _init_upstream(directory)
        self.revision = revision
        project = directory / "project"
        _init_project(project)
        self.context = ProjectContext(
            project_root=project, config_path=directory / "recipes.toml",
            artifacts_root=directory / "artifacts", work_root=directory / "work",
            upstream_repo=upstream, overlay_root=directory / "src",
            patches_root=directory / "patches",
        )
        self.store = ArtifactStore(directory / "store")
        if with_patch:
            _write_marker_patch(self.context.patches_root, marker_text=marker_text)
        patch_sets = {"core": campaign_config.PatchSet(
            name="core", patches=("0001_marker",), required_state="validated")} if with_patch else {}
        self.cfg = campaign_config.Config(
            pinned=revision, patch_sets=patch_sets,
            sources={"test-source": campaign_config.Source(
                name="test-source", ref=revision, overlay=False,
                patch_sets=("core",) if with_patch else ())},
            builds={"stock": campaign_config.Build(
                name="stock", options=(), variant_set=None, needs=frozenset())},
            platforms={"linux-multi": campaign_config.Platform(
                name="linux-multi", targets=("gfx1100",), options=())},
            experiments={}, campaigns={}, path=directory / "recipes.toml",
        )

    def spec(self) -> CampaignLaneExecutionSpec:
        return CampaignLaneExecutionSpec(
            source_name="test-source", build_name="stock", platform_name="linux-multi",
            architectures=("gfx1100",), inputs=(), validation=None,
        )

    def run(self, *, run_id: str, allow_dirty_bigcherry: bool = True, calls: list | None = None):
        calls = calls if calls is not None else []
        # campaign_source.source_plan_for() resolves patch SELECTION via
        # patchset.catalog() with no directory override -- it always reads
        # paths.PATCHES (the real project's patches/), never context.
        # patches_root. Only workspace.materialize()/resolve_materialization_
        # identity() read context.patches_root explicitly. Patch the shared
        # default so this harness's isolated patch fixtures are what gets
        # resolved during materialize, matching what a real isolated
        # checkout would see.
        with patch("bigcherry.paths.PATCHES", self.context.patches_root), \
             patch("bigcherry.campaign_workers.subprocess.run", _fake_compiler(calls)):
            return execute_campaign_lane(
                self.spec(), cfg=self.cfg, context=self.context, store=self.store,
                run_id=run_id, allow_dirty_bigcherry=allow_dirty_bigcherry)


class PatchByteSpoofingTests(unittest.TestCase):
    """RE24 item 1: edit a patch module's bytes under its unchanged
    canonical ID -> the canonical build must not reuse the old source
    materialisation (RE04 fix)."""

    def test_editing_patch_content_under_the_same_id_produces_a_different_source_slice(self):
        with tempfile.TemporaryDirectory() as directory:
            harness = _Harness(Path(directory), with_patch=True, marker_text="marker-v1")
            first = harness.run(run_id="run1")

            _write_marker_patch(harness.context.patches_root, marker_text="marker-v2")
            second = harness.run(run_id="run2")

            self.assertNotEqual(first.source_slice_id, second.source_slice_id)


class CachedTreeTamperTests(unittest.TestCase):
    """RE24 item 2: modify a file inside an already-cached source worktree
    -> reuse fails closed (RE04 fix)."""

    def test_tampering_the_cached_worktree_fails_closed_on_reuse(self):
        with tempfile.TemporaryDirectory() as directory:
            harness = _Harness(Path(directory), with_patch=False)
            first = harness.run(run_id="run1")

            source_root = None
            for candidate in (harness.context.work_root / "sources").iterdir():
                if candidate.is_dir():
                    source_root = candidate
            self.assertIsNotNone(source_root)
            (source_root / "source.txt").write_text("tampered\n", encoding="utf-8")

            with self.assertRaises(CampaignLaneError):
                harness.run(run_id="run2")
            self.assertTrue(first.source_slice_id)


class DirtyProjectLeakTests(unittest.TestCase):
    """RE24 item 3: dirty BigCherry source -> canonical lane path rejects
    on BOTH fresh materialisation and cache reuse unless the explicit
    development override is set; execute_campaign_lane default is clean
    (RE04 fix)."""

    def test_default_rejects_dirty_project_on_fresh_materialisation(self):
        with tempfile.TemporaryDirectory() as directory:
            harness = _Harness(Path(directory), with_patch=False)
            (harness.context.project_root / "dirty.txt").write_text("x", encoding="utf-8")
            with self.assertRaises((CampaignLaneError, WorkspaceError)):
                harness.run(run_id="run1", allow_dirty_bigcherry=False)

    def test_default_rejects_dirty_project_on_cache_reuse_too(self):
        with tempfile.TemporaryDirectory() as directory:
            harness = _Harness(Path(directory), with_patch=False)
            harness.run(run_id="run1", allow_dirty_bigcherry=False)
            (harness.context.project_root / "dirty.txt").write_text("x", encoding="utf-8")
            with self.assertRaises((CampaignLaneError, WorkspaceError)):
                harness.run(run_id="run2", allow_dirty_bigcherry=False)

    def test_explicit_override_allows_dirty_project_on_both_paths(self):
        with tempfile.TemporaryDirectory() as directory:
            harness = _Harness(Path(directory), with_patch=False)
            harness.run(run_id="run1", allow_dirty_bigcherry=True)
            (harness.context.project_root / "dirty.txt").write_text("x", encoding="utf-8")
            result = harness.run(run_id="run2", allow_dirty_bigcherry=True)
            self.assertTrue(result.source_slice_id)


class ProvenanceLaunderingTests(unittest.TestCase):
    """RE24 item 4: pass a wrong-source inventory as a raw path while
    building a different source -> visibly classified imported-legacy,
    never silently relabelled as this lane's own production evidence
    (RE08 fix)."""

    def test_raw_inventory_import_is_visibly_classified_imported_legacy(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            harness = _Harness(root, with_patch=False)
            # "tune" needs inventory -- add it to this harness's config.
            from dataclasses import replace
            harness.cfg = replace(harness.cfg, builds={
                **harness.cfg.builds,
                "tune": campaign_config.Build(
                    name="tune", options=(), variant_set=None, needs=frozenset({"inventory"})),
            })
            inventory = root / "inventory.json"
            inventory.write_text('{"mmq_types": ["q8_0"]}', encoding="utf-8")
            spec = CampaignLaneExecutionSpec(
                source_name="test-source", build_name="tune", platform_name="linux-multi",
                architectures=("gfx1100",), inputs=(("inventory", inventory),), validation=None,
            )
            calls: list = []
            with patch("bigcherry.campaign_workers.subprocess.run", _fake_compiler(calls)):
                result = execute_campaign_lane(
                    spec, cfg=harness.cfg, context=harness.context, store=harness.store,
                    run_id="run1", allow_dirty_bigcherry=True)
            inventory_ref = dict(result.input_refs)["inventory"]
            self.assertEqual(
                inventory_ref.provenance["project"]["provenance_class"], "imported-legacy")


class CatalogArchitectureIdentityTests(unittest.TestCase):
    """RE24 item 5: change the requested generation architectures with
    everything else fixed -> different build_plan_id and build dir
    (RE07 fix)."""

    def test_changing_architectures_changes_build_plan_id_and_build_dir(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            harness = _Harness(root, with_patch=False)
            from dataclasses import replace as dc_replace
            harness.cfg = dc_replace(harness.cfg, builds={
                **harness.cfg.builds,
                "gen-build": campaign_config.Build(
                    name="gen-build", options=(), variant_set="inventory", needs=frozenset()),
            })

            def run_with_arch(architectures, run_id):
                spec = CampaignLaneExecutionSpec(
                    source_name="test-source", build_name="gen-build", platform_name="linux-multi",
                    architectures=architectures, inputs=(), validation=None,
                )
                calls: list = []

                # A real autotune_catalog.emit() needs a real llama.cpp
                # source tree (TYPES_MMQ etc) this harness's fixture repo
                # does not have. Faking make_generate_worker wholesale --
                # rather than autotune_catalog's internals -- keeps every
                # OTHER real code path (BuildPlan.catalog_architectures
                # identity, real generated_tree.verify_tree, real build
                # worker, real ArtifactStore) exercised for real; only the
                # catalog-content derivation itself is stubbed.
                def fake_make_generate_worker(*, context, source_root, run_id,
                                              variant_set, architectures, upstream_revision,
                                              required_needs=frozenset()):
                    from bigcherry import generated_tree as gt

                    def generate(inputs):
                        stage_root = context.work_root / "runs" / run_id / "generate"
                        generated_root = stage_root / "generated"
                        generated_root.mkdir(parents=True, exist_ok=True)
                        header = generated_root / "hip-autotune-arch.h"
                        header.write_text(
                            f"// architectures: {','.join(sorted(architectures))}\n",
                            encoding="utf-8")
                        tree_manifest = gt.build_manifest(
                            generated_root, compile_inputs=(header,))
                        return {"manifest": {"architectures": sorted(architectures)},
                                "generated_tree": tree_manifest}
                    return generate

                with patch("bigcherry.campaign_workers.subprocess.run", _fake_compiler(calls)), \
                     patch("bigcherry.campaign_workers.make_generate_worker",
                           side_effect=fake_make_generate_worker):
                    return execute_campaign_lane(
                        spec, cfg=harness.cfg, context=harness.context, store=harness.store,
                        run_id=run_id, allow_dirty_bigcherry=True)

            result_a = run_with_arch(("gfx1100",), "run-a")
            result_b = run_with_arch(("gfx1100", "gfx1201"), "run-b")

            self.assertNotEqual(result_a.build_plan_id, result_b.build_plan_id)
            self.assertNotEqual(
                build_directory(harness.context, result_a.source_slice_id, result_a.build_plan),
                build_directory(harness.context, result_b.source_slice_id, result_b.build_plan))


class DeclaredNeedIdentityTests(unittest.TestCase):
    """RE24 item 6: change any declared build need's bytes -> different
    build identity (RE07 fix)."""

    def test_changing_inventory_bytes_changes_build_plan_id(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            harness = _Harness(root, with_patch=False)
            from dataclasses import replace as dc_replace
            harness.cfg = dc_replace(harness.cfg, builds={
                **harness.cfg.builds,
                "tune": campaign_config.Build(
                    name="tune", options=(), variant_set=None, needs=frozenset({"inventory"})),
            })

            def run_with_inventory(content, run_id):
                inventory = root / f"inventory-{run_id}.json"
                inventory.write_text(content, encoding="utf-8")
                spec = CampaignLaneExecutionSpec(
                    source_name="test-source", build_name="tune", platform_name="linux-multi",
                    architectures=("gfx1100",), inputs=(("inventory", inventory),), validation=None,
                )
                calls: list = []
                with patch("bigcherry.campaign_workers.subprocess.run", _fake_compiler(calls)):
                    return execute_campaign_lane(
                        spec, cfg=harness.cfg, context=harness.context, store=harness.store,
                        run_id=run_id, allow_dirty_bigcherry=True)

            result_a = run_with_inventory('{"mmq_types": ["q8_0"]}', "run-a")
            result_b = run_with_inventory('{"mmq_types": ["f16"]}', "run-b")
            self.assertNotEqual(result_a.build_plan_id, result_b.build_plan_id)


class RuntimeBundleIntegrityTests(unittest.TestCase):
    """RE24 item 7: delete or byte-flip one runtime .so member of a
    published bundle -> downstream consumption fails closed (RE07 fix).

    Real end-to-end here means: real materialize, real build worker
    (fake compiler), real runtime-smoke stage genuinely re-verifying the
    published bundle before running -- with the smoke binary faked
    (returns a valid llama-bench JSON shape) so this test needs no GPU.
    """

    def test_tampering_a_published_so_member_fails_smoke_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            harness = _Harness(root, with_patch=False)
            from bigcherry.runtime_smoke import RuntimeSmokeSpec
            model = root / "model.gguf"
            model.write_bytes(b"fake-model-bytes")
            spec = CampaignLaneExecutionSpec(
                source_name="test-source", build_name="stock", platform_name="linux-multi",
                architectures=("gfx1100",), inputs=(),
                validation=RuntimeSmokeSpec(model_path=model),
            )

            import json as _json
            smoke_stdout = _json.dumps([
                {"n_prompt": 512, "n_gen": 0, "avg_ts": 100.0, "samples_ts": [100.0]},
                {"n_prompt": 0, "n_gen": 128, "avg_ts": 50.0, "samples_ts": [50.0]},
            ])

            calls: list = []

            def fake_subprocess(args, cwd=None, check=None, **kwargs):
                if isinstance(args, list) and args and "llama-bench" in str(args[0]):
                    return subprocess.CompletedProcess(args, 0, stdout=smoke_stdout, stderr="")
                return _fake_compiler(calls)(args, cwd=cwd, check=check, **kwargs)

            with patch("bigcherry.campaign_workers.subprocess.run", side_effect=fake_subprocess):
                result = execute_campaign_lane(
                    spec, cfg=harness.cfg, context=harness.context, store=harness.store,
                    run_id="run-tamper-target", allow_dirty_bigcherry=True)
            self.assertTrue(result.smoke_ref)

            # Now tamper with a PUBLISHED runtime-bundle member's bytes
            # directly on disk, then rerun smoke fresh (new run_id, build
            # reused since identity is unchanged) -- must fail closed.
            so_path = None
            for candidate in result.runtime_bundle_ref.path.parent.iterdir():
                if candidate.name.startswith("lib") and candidate.is_file():
                    so_path = candidate
            self.assertIsNotNone(so_path)
            so_path.write_bytes(b"TAMPERED-BYTES")

            # The tampered bytes live in the STORE (the published copy), not
            # the local build_dir cache -- so even the build stage's own
            # reuse path (which re-publishes from build_dir, still holding
            # the real bytes, on every run) will hit ArtifactStore's
            # immutability check on republish before smoke ever gets to
            # consume anything. Either failure surface proves the tamper is
            # caught, not silently served -- this test's job is closure, not
            # pinning which exact stage's check fires first. CampaignRun.
            # execute() deliberately does not propagate the underlying
            # exception's own text into CampaignLaneError, so this only
            # asserts closure, not which check fired.
            with patch("bigcherry.campaign_workers.subprocess.run", side_effect=fake_subprocess):
                with self.assertRaises(CampaignLaneError):
                    execute_campaign_lane(
                        spec, cfg=harness.cfg, context=harness.context, store=harness.store,
                        run_id="run-tamper-check", allow_dirty_bigcherry=True)


class ReleaseProbeDecouplingTests(unittest.TestCase):
    """RE24 item 8: release_validate invokes no command scheduled for
    RE23 deletion, static check (RE13 migration status)."""

    def test_release_validate_does_not_invoke_a_command_scheduled_for_re23_deletion(self):
        import bigcherry.release_validate as release_validate
        source = Path(release_validate.__file__).read_text(encoding="utf-8")
        # RE13 (2026-08-17): probe() now runs execute_campaign_lane()
        # directly -- the same canonical production API `build` uses --
        # instead of shelling out to `bigcherry ... legacy-build`. RE23 can
        # delete legacy-build's mutable-checkout mechanics without breaking
        # this compatibility probe. Checks for actual invocation syntax
        # (quoted CLI arg / subprocess command token), not the word
        # appearing in prose comments describing the historical design.
        self.assertNotIn('"legacy-build"', source)
        self.assertNotIn("'legacy-build'", source)
        self.assertIn("execute_campaign_lane", source)


if __name__ == "__main__":
    unittest.main()
