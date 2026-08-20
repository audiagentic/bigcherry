"""One-shot: build each RD-prefixed experimental patch in isolation (audit
build, gfx1201 only) and run the project's own documented correctness
recipe (docs/reference/TEST.md) against it -- native baseline, then a
screen=1/final=1 tune-mode pass (correctness-only, not a timing run) so
every registered candidate (including whatever the patch adds) actually
gets exercised and its GPU-vs-CPU-reference comparison checked.

Not a permanent tool -- a one-shot validation harness, same status as
re15_tamper_evidence.py.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from bigcherry import config
from bigcherry.artifacts import ArtifactStore
from bigcherry.campaign_lane import CampaignLaneError, CampaignLaneExecutionSpec, execute_campaign_lane
from bigcherry.context import ProjectContext

EXPERIMENTS_AND_OPS = {
    "rd04-only": ("FLASH_ATTN_EXT",),
    "rd05-07-only": ("FLASH_ATTN_EXT", "MUL_MAT"),
    "rd08-only": ("MUL_MAT",),
    "rd12-only": ("MUL_MAT",),
    "rd13-only": ("MUL_MAT",),
    "rd17-only": ("MUL_MAT",),
    "rd19-only": ("MUL_MAT",),
    "rd20-only": ("MUL_MAT",),
    "rd21-only": ("MUL_MAT",),
    "rd22-only": ("MUL_MAT",),
    "rd26a-only": ("MUL_MAT",),
}


def run_correctness(binary: Path, op: str, *, mode: str, work_dir: Path) -> tuple[int, str]:
    env = dict(os.environ)
    env["GGML_HIP_DISPATCH_MODE"] = mode
    argv = [str(binary), "test", "-o", op]
    if mode == "tune":
        env["GGML_HIP_TUNE_SCREEN_SAMPLES"] = "1"
        env["GGML_HIP_TUNE_FINAL_SAMPLES"] = "1"
        env["GGML_CUDA_DISABLE_GRAPHS"] = "1"
        db_path = work_dir / f"{op}-{mode}"
        env["GGML_HIP_DISPATCH_DB"] = str(db_path)
    completed = subprocess.run(argv, capture_output=True, text=True, env=env, timeout=900)
    tail = (completed.stdout[-1500:] + completed.stderr[-1500:])
    return completed.returncode, tail


def scan_rejects(measurements_path: Path) -> list[str]:
    """result rows carry a "candidates" list, each with its own status --
    see inventory.py's load_measurements() for the authoritative shape.
    Most non-"ok" statuses ("ineligible", "architecture", ...) just mean
    "this candidate does not apply to this shape" -- normal, expected, and
    not what this check cares about. Flag only statuses that name an
    actual correctness/execution problem (nmse/tolerance rejection, launch
    or measurement failure, crash) -- exactly the class of thing a patch
    could plausibly introduce."""
    if not measurements_path.is_file():
        return []
    concerning = ("reject", "crash", "fail", "error", "abort")
    rejects = []
    for line in measurements_path.read_text(encoding="utf-8").splitlines():
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        if rec.get("kind") != "result":
            continue
        for cand in rec.get("candidates", []):
            name = cand.get("name", "")
            if name.endswith("#twin"):
                continue
            status = str(cand.get("status", "ok"))
            if status != "ok" and any(word in status.lower() for word in concerning):
                rejects.append(f"{name}: {status}")
    return rejects


def main() -> int:
    cfg = config.load("recipes.toml")
    context = ProjectContext.resolve()
    store = ArtifactStore(context.artifacts_root)

    overall_ok = True
    for name, ops in EXPERIMENTS_AND_OPS.items():
        print(f"=== {name} ===")
        try:
            spec = CampaignLaneExecutionSpec(
                source_name="bigcherry", build_name="audit", platform_name="linux-multi",
                architectures=("gfx1201",), extra_cmake_targets=("test-backend-ops",),
                experiment=name,
            )
            result = execute_campaign_lane(spec, cfg=cfg, context=context, store=store)
        except CampaignLaneError as exc:
            print(f"  [BUILD FAIL] {exc}")
            overall_ok = False
            continue
        except Exception as exc:  # noqa: BLE001 -- one patch's crash must not
            # abort the batch; every other experiment still needs its own
            # independent verdict.
            print(f"  [BUILD FAIL] {type(exc).__name__}: {exc}")
            overall_ok = False
            continue

        binary_dir = result.runtime_bundle_ref.path.parent
        test_binary = binary_dir / "test-backend-ops"
        if not test_binary.is_file():
            print(f"  [BUILD FAIL] test-backend-ops missing at {test_binary}")
            overall_ok = False
            continue

        work_dir = context.work_root / "rd-validate" / name
        work_dir.mkdir(parents=True, exist_ok=True)

        patch_ok = True
        for op in ops:
            try:
                rc, tail = run_correctness(test_binary, op, mode="native", work_dir=work_dir)
                if rc != 0:
                    print(f"  [FAIL] {op} native baseline exited {rc}\n{tail}")
                    patch_ok = False
                    continue
                print(f"  [ok] {op} native baseline")

                rc, tail = run_correctness(test_binary, op, mode="tune", work_dir=work_dir)
                if rc != 0:
                    print(f"  [FAIL] {op} tune-mode (screen=1/final=1) exited {rc}\n{tail}")
                    patch_ok = False
                    continue
                rejects = scan_rejects(work_dir / f"{op}-tune.measurements.jsonl")
                if rejects:
                    print(f"  [FAIL] {op} tune-mode: {len(rejects)} candidate(s) rejected on correctness:")
                    for r in rejects:
                        print(f"           {r}")
                    patch_ok = False
                else:
                    print(f"  [ok] {op} tune-mode, no correctness rejects")
            except Exception as exc:  # noqa: BLE001 -- same batch-resilience
                # reasoning as the build try/except above.
                print(f"  [FAIL] {op}: {type(exc).__name__}: {exc}")
                patch_ok = False

        print(f"  => {name}: {'PASS' if patch_ok else 'FAIL'}")
        overall_ok = overall_ok and patch_ok

    print("=== SUMMARY ===", "ALL PASS" if overall_ok else "SOME FAILED")
    return 0 if overall_ok else 1


if __name__ == "__main__":
    sys.exit(main())
