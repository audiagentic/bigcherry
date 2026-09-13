"""Ad-hoc real-hardware driver for run_rd13_backend_reference_check().

No --run-rd13-backend-reference CLI flag exists yet in validation_campaign.py
(wiring it into run()'s mutually-exclusive execution-mode branches, RD58-
style, is separate follow-up scope). This script invokes the real producer
function directly, exactly the way RD58's real evidence was first gathered
before its own CLI flag existed. Run on Brutus (2x gfx1100 XTX or single
gfx1100), never on a shared/ambiguous tree without checking for live leases
first.

Usage:
  PYTHONPATH=tools python tools/lab/rd13-backend-reference/run_real.py
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "tools"))

from bigcherry.experiment.attestation import ExecutionIdentity  # noqa: E402
from bigcherry.patch import validation_campaign as vc  # noqa: E402

HIP_PATH = Path("/home/audumla/rocm-shim")
AMDGPU_TARGETS = "gfx1100"
MODEL = Path("/mnt/vault/llm-models/qwen3.5-4B/gguf/mtp/Qwen3.5-4B-UD-Q6_K_XL.gguf")
BASE_REVISION = None  # resolved from the active pin below
# rocm-smi --showbus: GPU[0] = 0000:03:00.0, a real gfx1100 XTX. HIP_VISIBLE_
# DEVICES=0 below pins execution to exactly this device.
EXPECTED_EXECUTION = ExecutionIdentity(
    backend="ROCm", architectures=("gfx1100",), locators=("0000:03:00.0",),
)


def main() -> int:
    import tomllib

    recipes = tomllib.loads((REPO_ROOT / "config" / "recipes.toml").read_text())
    base_revision = recipes["pinned"]

    run_root = REPO_ROOT / "artifacts" / "lab" / "rd13-backend-reference"
    run_dir = run_root / "run"
    worktree_root = run_root / "worktrees"
    build_root = run_root / "build"
    for p in (run_dir, worktree_root, build_root):
        p.mkdir(parents=True, exist_ok=True)

    result = vc.run_rd13_backend_reference_check(
        base_revision=base_revision,
        hip_path=HIP_PATH,
        amdgpu_targets=AMDGPU_TARGETS,
        worktree_root=worktree_root,
        build_root=build_root,
        model=MODEL,
        run_dir=run_dir,
        expected_execution=EXPECTED_EXECUTION,
    )
    correctness = result["results"]["backend_reference"]
    print(f"RD13 backend_reference: {'PASS' if correctness.passed else 'FAIL'}")
    print(correctness.detail)
    print(f"artifact: {result['artifact']['path']}")
    return 0 if correctness.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
