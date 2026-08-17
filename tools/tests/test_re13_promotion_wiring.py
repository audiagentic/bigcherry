"""RE13: promotion pointer wired to a real execute_campaign_lane() result
and a real ReleaseRecord promotion, closing the two pieces RE13's probe
migration (see test_release_validate.py) deliberately left open --
PromotionPointer.pointer_from_campaign_result() and releases.promote().

Real git materialize, real BuildPlan identity, real ArtifactStore
publish/verify throughout (matching test_campaign_cutover_audit.py's own
end-to-end pattern); only campaign_workers.subprocess.run (the actual
cmake/compiler invocation) is faked.
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
from bigcherry import releases  # noqa: E402
from bigcherry.artifacts import ArtifactStore  # noqa: E402
from bigcherry.campaign_lane import CampaignLaneExecutionSpec, execute_campaign_lane  # noqa: E402
from bigcherry.context import ProjectContext  # noqa: E402
from bigcherry.promotion import PromotionError, pointer_from_campaign_result  # noqa: E402


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
    return repo, _git(repo, "rev-parse", "HEAD")


def _init_project(directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    _git(directory, "init", str(directory))
    _git(directory, "config", "user.email", "test@example.invalid")
    _git(directory, "config", "user.name", "Test")
    (directory / ".gitkeep").write_text("", encoding="utf-8")
    _git(directory, "add", ".gitkeep")
    _git(directory, "commit", "-m", "initial")


_REAL_SUBPROCESS_RUN = subprocess.run


def _fake_compiler(calls: list):
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
                    "AMDGPU_TARGETS:STRING=gfx1100\nGGML_HIP:BOOL=ON\n", encoding="utf-8")
        return subprocess.CompletedProcess(args, 0, stdout="", stderr="")
    return run


def _run_real_lane(root: Path):
    upstream, revision = _init_upstream(root)
    project = root / "project"
    _init_project(project)
    context = ProjectContext.resolve(
        project_root=project, config_path=root / "recipes.toml",
        artifacts_root=root / "artifacts", work_root=root / "work",
        upstream_repo=upstream)
    store = ArtifactStore(root / "store")
    cfg = campaign_config.Config(
        pinned=revision, patch_sets={},
        sources={"test-source": campaign_config.Source(
            name="test-source", ref=revision, overlay=False, patch_sets=())},
        builds={"stock": campaign_config.Build(
            name="stock", options=(), variant_set=None, needs=frozenset())},
        platforms={"linux-multi": campaign_config.Platform(
            name="linux-multi", targets=("gfx1100",), options=())},
        experiments={}, campaigns={}, path=root / "recipes.toml",
    )
    spec = CampaignLaneExecutionSpec(
        source_name="test-source", build_name="stock", platform_name="linux-multi",
        architectures=("gfx1100",), inputs=(), validation=None,
    )
    calls: list = []
    with patch("bigcherry.campaign_workers.subprocess.run", _fake_compiler(calls)):
        return execute_campaign_lane(
            spec, cfg=cfg, context=context, store=store,
            run_id="promotion-wiring-run", allow_dirty_bigcherry=True)


def _run_real_lane_with_generate(root: Path, *, architectures: tuple[str, ...]):
    """Same shape as _run_real_lane, but a real generate stage (variant_set
    set) so the result's build_plan.catalog_architectures is non-empty --
    needed to exercise pointer_from_campaign_result's architecture
    cross-check for real. autotune_catalog.emit() itself is faked (needs a
    real llama.cpp source tree this fixture doesn't have), matching
    test_campaign_cutover_audit.py's CatalogArchitectureIdentityTests
    pattern; everything else (BuildPlan identity, generated_tree.verify_tree,
    real build worker, real ArtifactStore) is real.
    """
    upstream, revision = _init_upstream(root)
    project = root / "project"
    _init_project(project)
    context = ProjectContext.resolve(
        project_root=project, config_path=root / "recipes.toml",
        artifacts_root=root / "artifacts", work_root=root / "work",
        upstream_repo=upstream)
    store = ArtifactStore(root / "store")
    cfg = campaign_config.Config(
        pinned=revision, patch_sets={},
        sources={"test-source": campaign_config.Source(
            name="test-source", ref=revision, overlay=False, patch_sets=())},
        builds={"gen-build": campaign_config.Build(
            name="gen-build", options=(), variant_set="inventory", needs=frozenset())},
        platforms={"linux-multi": campaign_config.Platform(
            name="linux-multi", targets=architectures, options=())},
        experiments={}, campaigns={}, path=root / "recipes.toml",
    )
    spec = CampaignLaneExecutionSpec(
        source_name="test-source", build_name="gen-build", platform_name="linux-multi",
        architectures=architectures, inputs=(), validation=None,
    )

    def fake_make_generate_worker(*, context, source_root, run_id, variant_set,
                                  architectures, upstream_revision, required_needs=frozenset()):
        from bigcherry import generated_tree as gt

        def generate(inputs):
            generated_root = context.work_root / "runs" / run_id / "generate" / "generated"
            generated_root.mkdir(parents=True, exist_ok=True)
            header = generated_root / "hip-autotune-arch.h"
            header.write_text(f"// architectures: {','.join(sorted(architectures))}\n",
                              encoding="utf-8")
            tree_manifest = gt.build_manifest(generated_root, compile_inputs=(header,))
            return {"manifest": {"architectures": sorted(architectures)},
                    "generated_tree": tree_manifest}
        return generate

    calls: list = []
    with patch("bigcherry.campaign_workers.subprocess.run", _fake_compiler(calls)), \
         patch("bigcherry.campaign_workers.make_generate_worker",
               side_effect=fake_make_generate_worker):
        return execute_campaign_lane(
            spec, cfg=cfg, context=context, store=store,
            run_id="promotion-wiring-generate-run", allow_dirty_bigcherry=True)


class PointerFromCampaignResultTests(unittest.TestCase):
    def test_pointer_reads_identities_off_the_real_lane_result(self):
        with tempfile.TemporaryDirectory() as directory:
            result = _run_real_lane(Path(directory))
            pointer = pointer_from_campaign_result(
                result=result, release_tag="b10362", campaign_plan_id="plan1",
                architectures=("gfx1100",), report=b"a real report body", valid=True)
            self.assertEqual(pointer.revision, result.resolved_revision)
            self.assertEqual(pointer.campaign_run_id, result.run_id)
            self.assertEqual(pointer.source_slice_id, result.source_slice_id)
            self.assertEqual(pointer.build_id, result.build_plan.build_plan_id)
            self.assertEqual(pointer.binary_hash, result.binary_ref.content_hash)
            self.assertEqual(pointer.replay_artifact_hash,
                             result.runtime_bundle_ref.content_hash)
            self.assertNotEqual(pointer.binary_hash, pointer.replay_artifact_hash)

    def test_invalid_report_refuses_to_produce_a_pointer(self):
        with tempfile.TemporaryDirectory() as directory:
            result = _run_real_lane(Path(directory))
            with self.assertRaises(PromotionError):
                pointer_from_campaign_result(
                    result=result, release_tag="b10362", campaign_plan_id="plan1",
                    architectures=("gfx1100",), report=b"bad report", valid=False)

    def test_pointer_promotes_a_real_release_record(self):
        with tempfile.TemporaryDirectory() as directory:
            result = _run_real_lane(Path(directory))
            pointer = pointer_from_campaign_result(
                result=result, release_tag="b10362", campaign_plan_id="plan1",
                architectures=("gfx1100",), report=b"a real report body", valid=True)
            record = releases.ReleaseRecord(
                revision=result.resolved_revision, release_tag="b10362", stage="tested")
            releases.promote(record, pointer)
            self.assertEqual(record.stage, "validated")
            self.assertEqual(record.promotion["promoted_source"]["binary_hash"],
                             result.binary_ref.content_hash)

    def test_claimed_architectures_beyond_the_real_catalog_are_rejected(self):
        # GPT-auto-agent review (RE13 follow-up, 2026-08-17): a valid
        # result for gfx1100 only must not be wrappable in a pointer
        # claiming broader coverage.
        with tempfile.TemporaryDirectory() as directory:
            result = _run_real_lane_with_generate(Path(directory), architectures=("gfx1100",))
            self.assertEqual(result.build_plan.catalog_architectures, ("gfx1100",))
            with self.assertRaisesRegex(PromotionError, "exceed"):
                pointer_from_campaign_result(
                    result=result, release_tag="b10362", campaign_plan_id="plan1",
                    architectures=("gfx1100", "gfx1201"), report=b"report", valid=True)

    def test_claimed_architectures_within_the_real_catalog_are_accepted(self):
        with tempfile.TemporaryDirectory() as directory:
            result = _run_real_lane_with_generate(
                Path(directory), architectures=("gfx1100", "gfx1201"))
            pointer = pointer_from_campaign_result(
                result=result, release_tag="b10362", campaign_plan_id="plan1",
                architectures=("gfx1100",), report=b"report", valid=True)
            self.assertEqual(pointer.required_architectures, ("gfx1100",))

    def test_no_generate_result_cannot_claim_an_uncompiled_architecture(self):
        # GPT-auto-agent review (RE13 follow-up, 2026-08-17): a residual
        # gap in the first architecture-check fix -- a no-generate build's
        # catalog_architectures is empty by construction (RE07: "no catalog
        # to disambiguate", not "no data"), and the first fix skipped the
        # check entirely in that case. A no-generate result compiled only
        # for gfx1100 must still not be wrappable in a pointer claiming a
        # DIFFERENT architecture -- fall back to build_plan.targets.
        with tempfile.TemporaryDirectory() as directory:
            result = _run_real_lane(Path(directory))
            self.assertEqual(result.build_plan.catalog_architectures, ())
            self.assertEqual(result.build_plan.targets, ("gfx1100",))
            with self.assertRaisesRegex(PromotionError, "exceed"):
                pointer_from_campaign_result(
                    result=result, release_tag="b10362", campaign_plan_id="plan1",
                    architectures=("gfx1201",), report=b"report", valid=True)
            # The architecture it WAS actually compiled for is still fine.
            pointer = pointer_from_campaign_result(
                result=result, release_tag="b10362", campaign_plan_id="plan1",
                architectures=("gfx1100",), report=b"report", valid=True)
            self.assertEqual(pointer.required_architectures, ("gfx1100",))


if __name__ == "__main__":
    unittest.main()
