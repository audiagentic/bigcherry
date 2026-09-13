"""Ad-hoc real-hardware driver for run_rd06_contract_correctness().

No --run-rd06-correctness CLI flag exists yet. RD06's contract
(RD06-RDNA4-WMMA-FA-CONFIG) is gfx1201-only.

Usage:
  HIP_VISIBLE_DEVICES=2 PYTHONPATH=tools python tools/lab/rd06-correctness/run_real.py
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "tools"))

from bigcherry.patch import validation_campaign as vc  # noqa: E402

HIP_PATH = Path("/home/audumla/rocm-shim")
AMDGPU_TARGETS = "gfx1201"
MODEL = Path("/mnt/vault/llm-models/qwen3.5-4B/gguf/mtp/Qwen3.5-4B-UD-Q6_K_XL.gguf")
CORPUS = Path("/tmp/wikitext2-small-slice.txt")


def main() -> int:
    import tomllib

    recipes = tomllib.loads((REPO_ROOT / "config" / "recipes.toml").read_text())
    base_revision = recipes["pinned"]

    run_root = REPO_ROOT / "artifacts" / "lab" / "rd06-correctness"
    run_dir = run_root / "run"
    worktree_root = run_root / "worktrees"
    build_root = run_root / "build"
    for p in (run_dir, worktree_root, build_root):
        p.mkdir(parents=True, exist_ok=True)

    result = vc.run_rd06_contract_correctness(
        base_revision=base_revision,
        hip_path=HIP_PATH,
        amdgpu_targets=AMDGPU_TARGETS,
        worktree_root=worktree_root,
        build_root=build_root,
        model=MODEL,
        corpus=CORPUS,
        run_dir=run_dir,
    )
    correctness = result["results"]["backend_reference"]
    print(f"RD06 backend_reference: {'PASS' if correctness.passed else 'FAIL'}")
    print(correctness.detail)
    print(f"artifact: {result['artifact']['path']}")
    return 0 if correctness.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
