"""VA26 execution phase, RD08 adapter: bridges a QualificationCell's
composition identity to a real measured CellResult, by delegating to
RD08's existing evidence producers (patch/validation_campaign.py) rather
than reimplementing them.

Design consulted with dev-gpt-agent (req_a5e5f0987117413b, 2026-09-08),
grounded in direct reads of the real build/evidence code:

- QualificationCell.control/.subject already carry `module_hashes` in the
  EXACT shape patch/source.py's materialize_composition() takes directly
  (an ordered [(patch_id, content_hash)] list) -- no named-source/focal
  resolution step is needed since qualification_matrix.py already
  resolved it.
- build_tree() (validation_campaign.py) is fully patch-agnostic and
  content-addressed-cache-reusing; this module never re-derives it.
- run_rd08_contract_qualification() itself is NOT called here -- it
  immediately computes its own correctness gate / aggregation / trigger
  proof / promotion, which would duplicate exactly what
  campaign.qualification_execution.execute_qualification_plan() does
  ONCE at the plan level, across every cell. This module calls the three
  lower-level producers it is built from instead:
  run_rd08_validation_lanes() / run_rd08_contract_correctness() /
  run_rd08_contract_trigger().
- run_rd08_contract_correctness() does not vary by cell composition at
  all -- it materializes its OWN internal VDR2-subject/VDR1-control A/B
  pair (RD08_PATCH_STACK vs apply_vdr1_control), independent of which
  cell.control/cell.subject compositions are being measured. It varies
  only by architecture, so it is cached per-architecture, run once
  regardless of how many contrasts (isolated/release_delta) share that
  architecture.
- A correctness FAILURE (Rd08CorrectnessError, e.g. RD08's own real,
  already-confirmed bit-identical divergence) is evidence, not an error:
  run_rd08_contract_correctness() already converts it to a
  CorrectnessResult(passed=False) internally, and that is what this
  adapter returns as part of a normal CellResult (error=None) -- only a
  genuine build/materialization/infrastructure failure becomes
  CellResult(error=...), which forces execute_qualification_plan()'s
  fail-closed "invalid" path. A correctness failure must reach the
  plan-level correctness gate as real (failing) evidence instead.

The design review flagged a latent bug this adapter would have exposed on
a multi-architecture plan: qualification_execution.py originally merged
correctness_results with a plain dict.update(), so a later architecture's
PASSING "bit_identical" result could silently overwrite an earlier
architecture's FAILING one. Fixed directly (same session) in
qualification_execution.execute_qualification_plan(): a failing result for
a given check name now always wins the merge over a passing one for that
same check, regardless of call order.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

from . import qualification_execution as qe
from .qualification_matrix import QualificationCell, QualificationComposition
from ..build.builds import capture_completed_build_evidence
from ..patch import source as patch_source
from ..patch.validation_campaign import (
    LLAMA_CPP_SRC,
    REPO_ROOT,
    _full_requested_cmake_args,
    assert_validation_subject_parity,
    build_tree,
    generate_registry,
    run_rd08_contract_correctness,
    run_rd08_contract_trigger,
    run_rd08_validation_lanes,
)

_EXE = ".exe" if sys.platform == "win32" else ""


@dataclass(frozen=True)
class _BuiltComposition:
    source: Path
    bin_dir: Path
    build_identity: dict[str, object]
    build_evidence: object


def make_rd08_run_cell(
    *,
    contract: object,
    base_revision: str,
    hip_path: Path,
    model: Path,
    model_ref: str,
    marker_regex: str,
    worktree_root: Path,
    build_root: Path,
    run_root: Path,
    build_env: dict[str, str],
    base_repo: Path = LLAMA_CPP_SRC,
    pairs: int = 3,
) -> qe.LaneRunner:
    """Build a ``run_cell`` callback bound to one qualification run's
    execution context. Everything here is per-RUN configuration (model,
    toolchain, work directories), never per-cell identity -- a
    QualificationCell already carries its own architecture and exact
    control/subject compositions, so only ``cell`` varies call to call.

    Materializes/builds each DISTINCT (architecture, composition) pair at
    most once (cached across cells that share one, e.g. the isolated and
    release_delta contrasts' shared native/release control side); RD08's
    own correctness evidence is cached per-architecture only, matching
    its own real independence from cell composition.
    """
    build_cache: dict[tuple[str, str], _BuiltComposition] = {}
    correctness_cache: dict[str, dict[str, object]] = {}

    def _get_build(architecture: str, composition: QualificationComposition) -> _BuiltComposition:
        key = (architecture, composition.patch_set_id)
        cached = build_cache.get(key)
        if cached is not None:
            return cached
        name = f"{architecture}-{composition.patch_set_id}"
        source = patch_source.materialize_composition(
            base_repo=base_repo, worktree_root=worktree_root / name,
            resolved_revision=base_revision, composition=composition.module_hashes,
            overlay_root=REPO_ROOT / "src", requested_revision=base_revision,
        )
        cell_build_root = build_root / name
        generated_dir = cell_build_root / "generated"
        generate_registry(source=source, amdgpu_targets=architecture, generated_dir=generated_dir)
        bin_dir = build_tree(
            name=name, hip_path=hip_path, amdgpu_targets=architecture,
            workdir=cell_build_root, targets=["llama-bench"], source=source,
            extra_cmake_args=[],
        )
        cmake_args = _full_requested_cmake_args(
            hip_path=hip_path, amdgpu_targets=architecture, extra_cmake_args=[],
        )
        evidence = capture_completed_build_evidence(
            cell_build_root / name, source_root=source, architecture=architecture,
            binary=bin_dir / f"llama-bench{_EXE}", requested_cmake_args=cmake_args,
            build_env=build_env,
        )
        built = _BuiltComposition(
            source=source, bin_dir=bin_dir, build_identity=evidence.campaign_identity(),
            build_evidence=evidence,
        )
        build_cache[key] = built
        return built

    def _get_correctness(architecture: str) -> dict[str, object]:
        cached = correctness_cache.get(architecture)
        if cached is not None:
            return cached
        correctness_doc = run_rd08_contract_correctness(
            base_revision=base_revision, hip_path=hip_path, amdgpu_targets=architecture,
            worktree_root=worktree_root / "rd08-correctness" / architecture,
            build_root=build_root / architecture, build_env=build_env,
            run_dir=run_root / architecture / "correctness",
        )
        results = correctness_doc["results"]
        correctness_cache[architecture] = results
        return results

    def run_cell(cell: QualificationCell) -> qe.CellResult:
        control = _get_build(cell.architecture, cell.control)
        subject = _get_build(cell.architecture, cell.subject)
        assert_validation_subject_parity(
            control.build_evidence, subject.build_evidence, patch_id=cell.patch_id,
        )

        cell_dir = run_root / cell.architecture / cell.contrast
        cell_dir.mkdir(parents=True, exist_ok=True)

        lanes = run_rd08_validation_lanes(
            contract=contract, control_binary=control.bin_dir / f"llama-bench{_EXE}",
            subject_binary=subject.bin_dir / f"llama-bench{_EXE}", model=model,
            model_ref=model_ref, hip_path=hip_path, run_dir=cell_dir,
            control_build_identity=control.build_identity,
            subject_build_identity=subject.build_identity, pairs=pairs,
        )
        trigger = run_rd08_contract_trigger(
            marker_regex=marker_regex, control_binary=control.bin_dir / f"llama-bench{_EXE}",
            subject_binary=subject.bin_dir / f"llama-bench{_EXE}", model=model,
            hip_path=hip_path, workdir=cell_dir, run_dir=cell_dir,
        )
        correctness = _get_correctness(cell.architecture)

        return qe.CellResult(
            cell=cell, lane_effects=tuple(lanes["effects"]),
            correctness_results=dict(correctness), trigger_evidence=tuple(trigger["evidence"]),
            smoke_passed=None, error=None,
        )

    return run_cell
