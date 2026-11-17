#!/usr/bin/env python3
"""Run 5: official Radiance adapter strict-v2 measured-manifest attempt.

No guessed host pool, no unmeasured/fallback manifest, no ABI-swapped helpers.
Any adapter measurement failure is terminal: inspect logs and stop the A/B.
"""
import argparse
import json
import os
from pathlib import Path
import sys
import traceback

ROOTS = {
    "a": Path(os.environ["RCS13_ARM_A_SOURCE_ROOT"]),
    "b": Path(os.environ["RCS13_ARM_B_SOURCE_ROOT"]),
}
TMP = Path(os.environ.get("RCS13_RUN_ROOT","."))
MODEL = Path(os.environ["BC_MODEL_ROOT"])/"radiance/bigcherry/qwen3.8-next-flash-hcbf16/qwen3.8-next-flash-hcbf16.rad"
HEAT = Path(os.environ["BC_WORK_ROOT"])/"runs/perf51/heat.rank0.hot-first"

def main():
    p = argparse.ArgumentParser()
    p.add_argument("arm", choices=ROOTS.keys())
    p.add_argument("--build", required=True, type=Path)
    a = p.parse_args()
    root = ROOTS[a.arm]
    sys.path.insert(0, str(root / "tools"))
    os.chdir(root)
    os.environ["BIGCHERRY_ENVIRONMENT"] = str(Path(os.environ["BC_WORK_ROOT"])/"environment.local.toml")
    os.environ["BIGCHERRY_LAYOUTS"] = str(root / "engines/radiance/layouts")


    from bigcherry.engine_adapters.radiance import RadianceAdapter
    from bigcherry.engine_adapters.radiance_launch import _collect
    out = TMP / "run5-serving" / ("plan-" + a.arm)
    out.mkdir(parents=True, exist_ok=True)
    build = a.build.resolve()
    try:
        opts = argparse.Namespace(layout="flashnext-tp2xtx-r9700",
            model=MODEL, bc_serve=build/"radiance_home/bin/bc-serve",
            radiance_home=build/"radiance_home", heat=HEAT,
            histogram="/dev/null", sessions=1, context=8192, ngram="disk")
        plan = _collect(opts)
        (out / "budget.json").write_text(json.dumps(plan.explain(), sort_keys=True, indent=2) + "\n")
        print("RCS13_RUN5_DERIVED_BUDGET", plan.host_pool_mib, plan.host_kv_mib, flush=True)
        manifest = RadianceAdapter().write_serving_manifest(
            "rcs13-run5-" + a.arm,
            {"layout": "flashnext-tp2xtx-r9700", "heat": str(HEAT),
             "histogram": "/dev/null",
             "set": {
               "engine.ngram_placement": "disk",
               "tuning.weights_disk_tier": "on",
               "memory.host_pool_mib": plan.host_pool_mib,
               "memory.host_kv_mib": plan.host_kv_mib,
               "service.max_context": 8192,
               "service.max_sessions": 1,
               "engine.max_num_batched_tokens": 256,
               "handoff.max_rows": 64,
               "handoff.small_rows": 8,
               "handoff.histogram_period_s": 1}},
            binary=build/"bin/radiance", model=MODEL,
            env={**os.environ, "ROCR_VISIBLE_DEVICES": "0,1",
                 "HIP_VISIBLE_DEVICES": "0,1",
                 "RADIANCE_HOME": str(build/"radiance_home")},
            run_dir=out)
        print("RCS13_RUN5_MEASURED_MANIFEST", manifest, flush=True)
    except Exception as exc:
        print("RCS13_RUN5_MEASURE_FAIL", type(exc).__name__, repr(exc), flush=True)
        traceback.print_exc()
        raise SystemExit(1)

if __name__ == "__main__":
    main()
