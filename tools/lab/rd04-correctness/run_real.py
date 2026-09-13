"""Ad-hoc real-hardware driver for run_rd04_contract_correctness().

No --run-rd04-correctness CLI flag exists yet. RD04's contract
(RD04-BF16-FLASH-ATTN-TILE) covers gfx1100/gfx1201/gfx1030 -- loops all
three, setting the matching device index per iteration (never both
HIP_VISIBLE_DEVICES and ROCR_VISIBLE_DEVICES to the same index -- that
double-filters to zero devices, a known trap in this project).

Usage:
  PYTHONPATH=tools python tools/lab/rd04-correctness/run_real.py
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

    # worktree_root/build_root are SHARED across every architecture in this
    # loop -- the producer now compiles one fat multi-arch binary once and
    # reuses it for every device run (build_tree()'s cmake-cache-reuse).
    # Per-arch subdirectories here would defeat that by giving each run a
    # genuinely different build location with nothing to reuse. Only
    # run_dir stays shared too: the producer already namespaces its own
    # artifact filenames by run architecture, so no collision risk.
    run_root = REPO_ROOT / "artifacts" / "lab" / "rd04-correctness"
    run_dir = run_root / "run"
    worktree_root = run_root / "worktrees"
    build_root = run_root / "build"
    for p in (run_dir, worktree_root, build_root):
        p.mkdir(parents=True, exist_ok=True)

    overall_pass = True
    for arch in ARCHITECTURES:
        print(f"=== RD04 correctness: {arch} ===")
        device_index = DEVICE_INDEX_BY_ARCH[arch]
        os.environ["HIP_VISIBLE_DEVICES"] = device_index
        os.environ.pop("ROCR_VISIBLE_DEVICES", None)

        result = vc.run_rd04_contract_correctness(
            base_revision=base_revision,
            hip_path=HIP_PATH,
            amdgpu_targets=arch,
            worktree_root=worktree_root,
            build_root=build_root,
            model=MODEL,
            corpus=CORPUS,
            run_dir=run_dir,
        )
        backend_reference = result["results"]["backend_reference"]
        ppl_equality = result["results"]["ppl_equality"]
        print(f"[{arch}] backend_reference: {'PASS' if backend_reference.passed else 'FAIL'}")
        print(f"[{arch}] ppl_equality: {'PASS' if ppl_equality.passed else 'FAIL'}")
        print(f"[{arch}] {backend_reference.detail}")
        print(f"[{arch}] artifact: {result['artifact']['path']}")
        overall_pass = overall_pass and backend_reference.passed

    return 0 if overall_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
