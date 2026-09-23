"""Ad-hoc real-hardware driver for run_rd30_correctness_check().

No --run-rd30-correctness CLI flag exists yet (RD58-style CLI wiring into
run()'s mutually-exclusive execution modes is separate follow-up scope).
Mirrors the same pattern used for RD13's real-hardware driver
(tools/lab/rd13-backend-reference/run_real.py) -- a direct call to the real
producer function. Run on Brutus (single gfx1100 XTX; the contract's own
scope.architectures is gfx1100-only), never on a shared/ambiguous tree
without checking for live leases first.

Usage:
  PYTHONPATH=tools python tools/lab/rd30-correctness/run_real.py
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "tools"))

from bigcherry.patch import validation_campaign as vc  # noqa: E402
from bigcherry.patch.campaign import build as campaign_build  # noqa: E402

HIP_PATH = Path("/home/audumla/rocm-shim")
AMDGPU_TARGETS = "gfx1100"


def main() -> int:
    import tomllib

    recipes = tomllib.loads((REPO_ROOT / "config" / "recipes.toml").read_text())
    base_revision = recipes["pinned"]

    run_root = REPO_ROOT / "artifacts" / "lab" / "rd30-correctness"
    run_dir = run_root / "run"
    worktree_root = run_root / "worktrees"
    build_root = run_root / "build"
    for p in (run_dir, worktree_root, build_root):
        p.mkdir(parents=True, exist_ok=True)

    result = vc.run_rd30_correctness_check(
        base_revision=base_revision,
        hip_path=HIP_PATH,
        amdgpu_targets=AMDGPU_TARGETS,
        worktree_root=worktree_root,
        build_root=build_root,
        build_env=campaign_build._hip_env(HIP_PATH),
        run_dir=run_dir,
    )
    bit_identical = result["results"]["bit_identical"]
    backend_reference = result["results"]["backend_reference"]
    print(f"RD30 bit_identical: {'PASS' if bit_identical.passed else 'FAIL'}")
    print(bit_identical.detail)
    print(f"RD30 backend_reference: {'PASS' if backend_reference.passed else 'FAIL'}")
    print(backend_reference.detail)
    print(f"artifact: {result['artifact']['path']}")
    return 0 if bit_identical.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
