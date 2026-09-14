"""Ad-hoc real-hardware driver for run_rd12_correctness_check().

Mirrors tools/lab/rd30-correctness/run_real.py's pattern. RD12's contract
(RD12-PAIRED-MMVQ-DUAL) scopes gfx1100/gfx1201/gfx1030 -- runs all three
in sequence, one real build+run per architecture. The CLI equivalent is
`--run-rd12-contract` (one contract architecture per invocation); this
driver exists to exercise all three against one shared run_dir, which is
why the producer namespaces its per-architecture artifacts and logs.

Usage:
  PYTHONPATH=tools python tools/lab/rd12-correctness/run_real.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "tools"))

from bigcherry.patch import validation_campaign as vc  # noqa: E402

HIP_PATH = Path("/home/audumla/rocm-shim")
ARCHITECTURES = ("gfx1100", "gfx1201", "gfx1030")
# config/environment.toml's real Brutus device inventory: index is the
# HIP/ROCR visible-devices ordinal. Each architecture MUST run against its
# own matching device -- the loop below sets this per iteration since a
# single fixed ambient HIP_VISIBLE_DEVICES would silently run gfx1201/
# gfx1030's builds against whatever device index 0 happens to be.
DEVICE_INDEX_BY_ARCH = {"gfx1100": "0", "gfx1201": "2", "gfx1030": "3"}


def main() -> int:
    import tomllib

    recipes = tomllib.loads((REPO_ROOT / "config" / "recipes.toml").read_text())
    base_revision = recipes["pinned"]

    # worktree_root/build_root are SHARED across every architecture in this
    # loop -- the producer compiles one fat multi-arch binary once and
    # reuses it for every device run (build_tree()'s cmake-cache-reuse).
    # Per-arch subdirectories would defeat that. run_dir is shared too:
    # the producer namespaces its own artifact filenames by architecture.
    run_root = REPO_ROOT / "artifacts" / "lab" / "rd12-correctness"
    run_dir = run_root / "run"
    worktree_root = run_root / "worktrees"
    build_root = run_root / "build"
    for p in (run_dir, worktree_root, build_root):
        p.mkdir(parents=True, exist_ok=True)

    overall_pass = True
    for arch in ARCHITECTURES:
        print(f"=== RD12 correctness: {arch} ===")
        device_index = DEVICE_INDEX_BY_ARCH[arch]
        os.environ["HIP_VISIBLE_DEVICES"] = device_index
        # Setting BOTH HIP_VISIBLE_DEVICES and ROCR_VISIBLE_DEVICES to the
        # same index double-filters: ROCR selects device N from the real
        # device list first, then HIP re-applies its own index-N filter
        # against that already-filtered (now 1-device) list, landing on
        # nothing -- "no ROCm-capable device is detected" (a known trap in
        # this project, e.g. RD58's PVPS02 finding). HIP_VISIBLE_DEVICES
        # alone is sufficient.
        os.environ.pop("ROCR_VISIBLE_DEVICES", None)
        result = vc.run_rd12_correctness_check(
            base_revision=base_revision,
            hip_path=HIP_PATH,
            amdgpu_targets=arch,
            worktree_root=worktree_root,
            build_root=build_root,
            build_env=vc._hip_env(HIP_PATH),
            run_dir=run_dir,
        )
        bit_identical = result["results"]["bit_identical"]
        backend_reference = result["results"]["backend_reference"]
        activation = result["results"]["activation"]
        print(f"[{arch}] bit_identical: {'PASS' if bit_identical.passed else 'FAIL'}")
        print(f"[{arch}] {bit_identical.detail}")
        print(f"[{arch}] backend_reference: {'PASS' if backend_reference.passed else 'FAIL'}")
        print(f"[{arch}] activation: {'PASS' if activation.passed else 'FAIL'}")
        print(f"[{arch}] artifact: {result['artifact']['path']}")
        overall_pass = overall_pass and bit_identical.passed

    return 0 if overall_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
