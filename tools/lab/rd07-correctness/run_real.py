"""Ad-hoc real-hardware driver for run_rd07_contract_correctness().

No --run-rd07-correctness CLI flag exists yet. RD07's contract
(RD07-Q6K-MMQ-PREFILL-FOLD) covers gfx1100/gfx1201/gfx1030 -- loops all
three, setting the matching device index per iteration (never both
HIP_VISIBLE_DEVICES and ROCR_VISIBLE_DEVICES to the same index -- that
double-filters to zero devices, a known trap in this project).

Usage:
  PYTHONPATH=tools python tools/lab/rd07-correctness/run_real.py
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
DEVICE_INDEX_BY_ARCH = {"gfx1100": "0", "gfx1201": "2", "gfx1030": "3"}
MODEL = Path("/mnt/vault/llm-models/qwen3.5-4B/gguf/mtp/Qwen3.5-4B-UD-Q6_K_XL.gguf")
CORPUS = Path("/tmp/wikitext2-small-slice.txt")


def main() -> int:
    import tomllib

    recipes = tomllib.loads((REPO_ROOT / "config" / "recipes.toml").read_text())
    base_revision = recipes["pinned"]

    overall_pass = True
    for arch in ARCHITECTURES:
        run_root = REPO_ROOT / "artifacts" / "lab" / "rd07-correctness" / arch
        run_dir = run_root / "run"
        worktree_root = run_root / "worktrees"
        build_root = run_root / "build"
        for p in (run_dir, worktree_root, build_root):
            p.mkdir(parents=True, exist_ok=True)

        print(f"=== RD07 correctness: {arch} ===")
        device_index = DEVICE_INDEX_BY_ARCH[arch]
        os.environ["HIP_VISIBLE_DEVICES"] = device_index
        os.environ.pop("ROCR_VISIBLE_DEVICES", None)

        result = vc.run_rd07_contract_correctness(
            base_revision=base_revision,
            hip_path=HIP_PATH,
            amdgpu_targets=arch,
            worktree_root=worktree_root,
            build_root=build_root,
            model=MODEL,
            corpus=CORPUS,
            run_dir=run_dir,
        )
        correctness = result["results"]["backend_reference"]
        print(f"[{arch}] backend_reference: {'PASS' if correctness.passed else 'FAIL'}")
        print(f"[{arch}] {correctness.detail}")
        print(f"[{arch}] artifact: {result['artifact']['path']}")
        overall_pass = overall_pass and correctness.passed

    return 0 if overall_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
