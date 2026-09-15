"""PA35 step 1 narrow lab driver for patch 1000_rdna4_mmq_q2k_q6k_fix.

Not shared production code -- a one-off hardware-evidence driver, per GPT
design-review guidance (dev-gpt-agent, req_71c1aaa166f446a1, 2026-09-16):
"Do not invoke run_patch1000_verification() unchanged ... Use the existing
backend-ops helpers/narrow lab driver without adding new shared production
code." It does NOT touch validation_campaign.py and does NOT attempt the
PA36 patch-local producer-ownership migration (PA36-F is not complete and
patch 1000 is migration #8 in PA36's atomic sequence; GPT: "Option C").

Scope, per GPT's confirmed correction of PA35 step 1:
  - gfx1201 (RDNA4) only -- this is the patch's target architecture.
  - control  = serving-core composition WITHOUT patch 1000
    subject  = the same composition WITH patch 1000
    (stock and the third-arch sweep are explicitly NOT part of this
    minimum pass.)
  - Q2_K and Q6_K exact-shape backend-ops correctness (full registered
    MUL_MAT type_a=q2_K/q6_K,type_b=f32 corpus) and performance
    (exact m=4096,n=512,k=14336, 5 paired runs) via the already-authored
    run_patch1000_backend_ops_correctness / run_patch1000_backend_ops_perf.
  - Q2_K model-level llama-bench lane: SKIPPED, no fixture registered in
    config/models.toml.
  - Q6_K model-level llama-bench lane: SKIPPED for this minimum pass per
    GPT guidance -- recorded as not executed, not as evidence.

Usage (on Brutus, from repo root):
    PYTHONPATH=tools python tools/lab/patch1000/run_pa35_step1.py \
        --hip-path /opt/rocm --device 2 --out artifacts/patch1000-pa35-step1.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "tools"))

from bigcherry.patch import source as psi  # noqa: E402
from bigcherry.patch import validation_campaign as vc  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hip-path", type=Path, default=Path("/opt/rocm"))
    parser.add_argument("--device", default="0", help="HIP_VISIBLE_DEVICES selector for gfx1201")
    parser.add_argument("--pairs", type=int, default=5)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--build-root", type=Path, default=REPO_ROOT / "artifacts" / "patch1000-pa35-step1-build",
    )
    args = parser.parse_args()

    architecture = "gfx1201"
    patch_id = vc._PATCH1000_ID

    import tomllib

    recipes_path = REPO_ROOT / "config" / "recipes.toml"
    recipes = tomllib.loads(recipes_path.read_text(encoding="utf-8"))
    serving_core_ids = tuple(recipes["patch-set"]["serving-core"]["patches"])
    if patch_id in serving_core_ids:
        raise SystemExit("patch1000 unexpectedly already in serving-core")

    control_revision, control_composition = psi.resolve_source_composition(
        "llama-native", extra_patches=serving_core_ids, base_repo=vc.LLAMA_CPP_SRC,
    )
    subject_revision, subject_composition = psi.resolve_source_composition(
        "llama-native", extra_patches=(*serving_core_ids, patch_id), base_repo=vc.LLAMA_CPP_SRC,
    )
    if control_revision != subject_revision:
        raise SystemExit("control/subject resolved different base revisions")

    source_root = args.build_root / "sources"
    control_src = psi.materialize_composition(
        base_repo=vc.LLAMA_CPP_SRC, worktree_root=source_root / "control",
        resolved_revision=control_revision, composition=control_composition,
        overlay_root=REPO_ROOT / "src", requested_revision=control_revision,
    )
    subject_src = psi.materialize_composition(
        base_repo=vc.LLAMA_CPP_SRC, worktree_root=source_root / "subject",
        resolved_revision=subject_revision, composition=subject_composition,
        overlay_root=REPO_ROOT / "src", requested_revision=subject_revision,
    )

    build_workdir = args.build_root / "builds"
    build_args = {
        "hip_path": args.hip_path,
        "amdgpu_targets": architecture,
        "workdir": build_workdir,
        "targets": ["test-backend-ops"],
        "extra_cmake_args": [],
    }
    control_bin = vc.build_tree(name="patch1000-pa35-control", source=control_src, **build_args)
    subject_bin = vc.build_tree(name="patch1000-pa35-subject", source=subject_src, **build_args)

    exe = ".exe" if sys.platform == "win32" else ""
    control_ops = control_bin / f"test-backend-ops{exe}"
    subject_ops = subject_bin / f"test-backend-ops{exe}"

    selector_env = {"HIP_VISIBLE_DEVICES": args.device}
    from bigcherry.experiment import execution as experiment_execution

    visibility = experiment_execution.require_device_visibility(
        context="patch1000 pa35-step1", env=selector_env, exact_count=1,
    )
    expected_execution = vc.ExecutionIdentity(backend="ROCm", architectures=(architecture,))

    result: dict[str, object] = {
        "producer": "patch1000-pa35-step1-lab-driver",
        "patch": patch_id,
        "architecture": architecture,
        "device_visibility": visibility.document(),
        "control_composition": list(control_composition),
        "subject_composition": list(subject_composition),
        "base_revision": control_revision,
        "correctness": {},
        "microbenchmark": {},
        "skipped": {
            "q2k_model_llama_bench": "no Q2_K model fixture registered in config/models.toml",
            "q6k_model_llama_bench": (
                "not executed this pass; GPT design-review guidance "
                "(req_71c1aaa166f446a1) scoped PA35 step 1's minimum bar to the "
                "model-free exact-shape backend-ops lane only"
            ),
            "stock_arm": "not part of this minimum pass (control-vs-subject only)",
            "other_architectures": "gfx1100/gfx1030 not part of this minimum pass",
        },
    }

    for quant in ("Q2_K", "Q6_K"):
        correctness_doc = {}
        for arm, binary in (("control", control_ops), ("subject", subject_ops)):
            correctness_doc[arm] = vc.run_patch1000_backend_ops_correctness(
                binary=binary, quant=quant, hip_path=args.hip_path,
                env_overrides=selector_env,
                log_context=f"patch1000 pa35-step1 {quant} {arm} correctness",
                execution_identity=expected_execution,
            )
        result["correctness"][quant] = correctness_doc

        result["microbenchmark"][quant] = vc.run_patch1000_backend_ops_perf(
            control_binary=control_ops, subject_binary=subject_ops, quant=quant,
            hip_path=args.hip_path, env_overrides=selector_env, pairs=args.pairs,
            execution_identity=expected_execution,
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    print(f"wrote {args.out}")
    for quant in ("Q2_K", "Q6_K"):
        speedup = result["microbenchmark"][quant]["stats"]["geometric_speedup_x"]
        print(f"{quant}: control->subject speedup = {speedup:.3f}x")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
