"""Ad-hoc real-hardware driver for the 1205 RD12 patch-local producer
(PA36 sub-slice 2, T10: replaces the deleted run_rd12_correctness_check()
driver, which called the deleted CLI path directly).

Runs `--validation-producer 1205_rd12_paired_mmvq_dual_output/rd12` once
per contract architecture through the GENERIC dispatcher, which owns the
five-build standard-campaign scaffold, the producer's own build pair,
evidence binding, and the tracked record. Each invocation is one contract
architecture (the historical RD12 rule); the worktree/build roots are
shared across iterations so the fat multi-arch binaries are built once
and reused (build_tree's cmake-cache reuse) while each run_dir stays
per-architecture (the producer namespaces its own artifacts by arch).

Usage:
  PYTHONPATH=tools python tools/lab/rd12-correctness/run_real.py
"""

from __future__ import annotations

import argparse
import json
import sys
import tomllib
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "tools"))

from bigcherry.patch.campaign import producer as campaign_producer  # noqa: E402

HIP_PATH = Path("/home/audumla/rocm-shim")
PATCH = "1205_rd12_paired_mmvq_dual_output"
ARCHITECTURES = ("gfx1100", "gfx1201", "gfx1030")
# config/environment.toml's real Brutus device inventory: index is the
# HIP/ROCR visible-devices ordinal. Each architecture runs against its
# own matching device via the producer's sanctioned HIP-only selector
# env (never ambient HIP_VISIBLE_DEVICES -- the producer's runtime owns
# that, and a fixed ambient selector would silently run the wrong device).
DEVICE_INDEX_BY_ARCH = {"gfx1100": "0", "gfx1201": "2", "gfx1030": "3"}


def main() -> int:
    recipes = tomllib.loads((REPO_ROOT / "config" / "recipes.toml").read_text())
    pinned = recipes["pinned"]  # noqa: F841 (informational; the dispatcher pins it itself)

    run_root = REPO_ROOT / "artifacts" / "lab" / "rd12-correctness"
    # Shared across every architecture: the producer compiles one fat
    # multi-arch binary once and reuses it for every device run.
    worktree_root = run_root / "worktrees"
    build_root = run_root / "build"
    for p in (worktree_root, build_root):
        p.mkdir(parents=True, exist_ok=True)

    overall_pass = True
    for arch in ARCHITECTURES:
        print(f"=== RD12 validation producer: {arch} ===")
        args = argparse.Namespace(
            patch=PATCH,
            hip_path=HIP_PATH,
            amdgpu_targets=arch,
            device_map=[f"{arch}={DEVICE_INDEX_BY_ARCH[arch]}"],
            workdir=run_root / f"run-{arch}",
            worktree_root=worktree_root,
            build_root=build_root,
            model=None,
            correctness_evidence=None,
            run_performance_benchmark=False,
            producer_corpus=None,
            baseline_source="bigcherry",
        )
        exit_code = campaign_producer._run_validation_producer(
            args,
            producer_id="rd12",
            provided_inputs={},
        )
        if exit_code != 0:
            print(f"[{arch}] producer execution FAILED (exit {exit_code})")
            overall_pass = False
            continue
        outcome_path = run_root / f"run-{arch}" / "campaign" / "producer-execution.json"
        outcome = json.loads(outcome_path.read_text(encoding="utf-8"))
        correctness = outcome["check_results"].get("correctness", {})
        activation = outcome["check_results"].get("activation", {})
        print(f"[{arch}] correctness: {correctness.get('status')}")
        print(f"[{arch}] activation: {activation.get('status')}")
        print(
            f"[{arch}] eligible: {outcome['eligible']} "
            f"(record: {outcome['evidence_record']})"
        )
        overall_pass = overall_pass and (
            correctness.get("status") == "pass" and activation.get("status") == "pass"
        )

    return 0 if overall_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
