"""Ad-hoc real-hardware driver for run_rd26_decode_verify_bit_identity_check().

No --run-rd26-bit-identity CLI flag exists yet (separate follow-up scope).
RD26's contract (RD26-DECODE-VERIFY-BIT-IDENTITY) scopes gfx1100/gfx1201/
gfx1030 -- runs gfx1100 first (simplest single-GPU topology); gfx1201/
gfx1030 are real follow-up runs, not yet executed here.

A real run against the CURRENT 1210 patch may legitimately FAIL: patch
1210 contains only 2 of a 5-commit determinism cluster (the "base-
standalone" hunks), and the complete decode/verify determinism property
belongs to the full cluster. A FAIL here is real, useful evidence of that
documented remaining gap, not a bug -- do not treat it as one.

Usage:
  HIP_VISIBLE_DEVICES=0 ROCR_VISIBLE_DEVICES=0 \
  PYTHONPATH=tools python tools/lab/rd26-bit-identity/run_real.py
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "tools"))

from bigcherry.patch import validation_campaign as vc  # noqa: E402

HIP_PATH = Path("/home/audumla/rocm-shim")
AMDGPU_TARGETS = "gfx1100"
MODEL = Path("/mnt/vault/llm-models/qwen3.5-4B/gguf/mtp/Qwen3.5-4B-UD-Q6_K_XL.gguf")


def main() -> int:
    import tomllib

    recipes = tomllib.loads((REPO_ROOT / "config" / "recipes.toml").read_text())
    base_revision = recipes["pinned"]

    run_root = REPO_ROOT / "artifacts" / "lab" / "rd26-bit-identity"
    run_dir = run_root / "run"
    worktree_root = run_root / "worktrees"
    build_root = run_root / "build"
    for p in (run_dir, worktree_root, build_root):
        p.mkdir(parents=True, exist_ok=True)

    result = vc.run_rd26_decode_verify_bit_identity_check(
        base_revision=base_revision,
        hip_path=HIP_PATH,
        amdgpu_targets=AMDGPU_TARGETS,
        worktree_root=worktree_root,
        build_root=build_root,
        model=MODEL,
        run_dir=run_dir,
    )
    correctness = result["results"]["bit_identical"]
    print(f"RD26 bit_identical: {'PASS' if correctness.passed else 'FAIL'}")
    print(correctness.detail)
    print(f"artifact: {result['artifact']['path']}")
    return 0 if correctness.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
