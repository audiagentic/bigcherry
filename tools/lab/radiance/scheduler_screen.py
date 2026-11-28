#!/usr/bin/env python3
"""Deterministic orchestration of the native burst test, not a scheduler.

Cost inputs are frozen hypotheses, not measured step records. No production
selection or inferred hardware speedup. Screening ranks Pareto tradeoffs.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess


def native_run(binary, env_args):
    env = {k: v for k, v in os.environ.items()
           if not k.startswith(("SIM_", "BC_SIM_", "RAD_TEST_"))}
    env.update(ROCR_VISIBLE_DEVICES="99", HIP_VISIBLE_DEVICES="",
               OMP_WAIT_POLICY="PASSIVE", OMP_NUM_THREADS="4",
               RAD_TEST_STRICT="1", RAD_SCHED_SIM="1", BC_SIM_SCREEN="1")
    env.update({k: str(v) for k, v in env_args.items()})
    r = subprocess.run([str(binary), "sim_burst_of_long_prompts"], env=env,
                       capture_output=True, text=True, timeout=30)
    lines = r.stdout.splitlines()
    result = [x.split()[1:] for x in lines if x.startswith("BC_RESULT ")]
    trace = [x for x in lines if x.startswith(("BC_STEP ", "BC_EMIT "))]
    if r.returncode or len(result) != 1 or not trace or "FAIL " in r.stderr or "SKIP " in r.stderr:
        raise ValueError(f"native simulation failed: {r.returncode} {r.stderr[-2000:]}")
    vals = list(map(float, result[0]))
    if len(vals) != 8 or not all(math.isfinite(v) and v >= 0 for v in vals) or vals[0] <= 0:
        raise ValueError("invalid native result")
    wall, ttft, p95, maxgap, steps, decsteps, decrows, mixed = vals
    step_lines = [x.split() for x in trace if x.startswith("BC_STEP ")]
    # These are native plan rows and the hypothesis clock, not GPU measurements.
    mixed_rows = []
    target_ns = int(env_args.get("SIM_MIXED_MS", 0)) * 1000000
    for step in step_lines:
        try:
            pf, dec, ns = map(int, step[2:5])
            entries = [tuple(map(int, x.split(":"))) for x in step[5:]]
            if (pf < 0 or dec < 0 or ns <= 0 or not entries
                    or any(len(e) != 5 or e[1] not in (0, 1) or e[2] <= 0 for e in entries)
                    or sum(e[2] for e in entries if e[1]) != pf
                    or sum(e[2] for e in entries if not e[1]) != dec):
                raise ValueError("row counts differ from entries")
        except (ValueError, IndexError) as exc:
            raise ValueError("malformed native step geometry") from exc
        if pf and dec:
            mixed_rows.append((pf, sum(e[1] for e in entries), ns))
    emitted = {}
    for x in trace:
        if x.startswith("BC_EMIT "):
            _, seq, count, stamp = x.split()
            count, stamp = int(count), int(stamp)
            prev_count, prev_stamp = emitted.get(seq, (0, -1))
            if count <= prev_count or stamp <= prev_stamp:
                raise ValueError("nonmonotonic emissions")
            emitted[seq] = count, stamp
    if (steps != len(step_lines) or any(int(x[1]) != i for i,x in enumerate(step_lines))
            or len(emitted) != env_args["SIM_N"]
            or any(v[0] != env_args["SIM_OUT"] for v in emitted.values())
            or not 0 < ttft <= wall or not 0 <= p95 <= maxgap <= wall
            or len(mixed_rows) != mixed
            or not 0 <= mixed <= decsteps <= steps):
        raise ValueError("incomplete native trace")
    return {"schedule_sha256": hashlib.sha256("\n".join(trace).encode()).hexdigest(),
            "wall_s": wall, "ttft_max_s": ttft, "gap_p95_s": p95, "gap_max_s": maxgap,
            "steps": int(steps), "decode_steps": int(decsteps),
            "decode_rows": int(decrows), "mixed_steps": int(mixed),
            "output_tps_simulated": env_args["SIM_N"] * env_args["SIM_OUT"] / wall,
            "mixed_geometry": {
                "max_prefill_rows": max((r[0] for r in mixed_rows), default=0),
                "max_prefill_participants": max((r[1] for r in mixed_rows), default=0),
                "target_misses_simulated": sum(r[2] > target_ns for r in mixed_rows) if target_ns else None,
                "target_ms": target_ns / 1000000 if target_ns else None,
            }}


def policies():
    for il, share in [(0,1),(1,1),(2,1),(-1,.1),(-1,.25),(-1,1)]:
        for ms in (0,100,150,200,250):
            yield {"SIM_IL": il, "SIM_SHARE": share, "SIM_MIXED_MS": ms, "SIM_MAXTOK": 2048}
    for ms in (100,150,200,250):
        yield {"SIM_IL":0, "SIM_SHARE":1,"SIM_MIXED_MS":ms,"SIM_MAXTOK":4096}


def dominates(a,b):
    return all(x <= y for x,y in zip(a,b)) and any(x < y for x,y in zip(a,b))


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--binary",type=Path,required=True)
    ap.add_argument("--costs",type=Path,required=True)
    ap.add_argument("--output",type=Path,required=True)
    args=ap.parse_args()
    costs=json.loads(args.costs.read_text())
    if costs.get("kind") != "hypothesis-not-calibrated":
        raise ValueError("this screen accepts explicitly labelled hypotheses only")
    if not costs.get("profiles"):
        raise ValueError("no cost profiles")
    for cost in costs["profiles"].values():
        if set(cost) != {"SIM_DEC0","SIM_DECR","SIM_PF0","SIM_PFB"}:
            raise ValueError("unexpected cost fields")
        if any(type(v) not in (int,float) or not math.isfinite(v) or v < 0 for v in cost.values()):
            raise ValueError("invalid cost values")
        if cost["SIM_DEC0"] <= 0 or cost["SIM_PFB"] <= 0:
            raise ValueError("nonpositive required cost")
    scenarios=[(8,512,128),(8,2048,128),(8,4096,128),(8,4096,512)]
    rows=[]
    for name, cost in sorted(costs["profiles"].items()):
        group=[]
        for policy in policies():
            records=[]
            for n,prompt,out in scenarios:
                env={**cost,**policy,"SIM_N":n,"SIM_LEN":prompt,"SIM_OUT":out,
                     "SIM_SEQS":n,"SIM_INTERVAL":4096,"SIM_ACCEPT":2,"SIM_STAGE_ROWS":0}
                a=native_run(args.binary.resolve(),env)
                b=native_run(args.binary.resolve(),env)
                if a!=b:
                    raise ValueError(f"nondeterministic {name} {env}")
                records.append({"scenario": [n,prompt,out], "result":a})
            group.append({"profile":name,"policy":policy,"cases":records})
        base=next(x for x in group if x["policy"]["SIM_IL"]==-1 and x["policy"]["SIM_SHARE"]==1 and x["policy"]["SIM_MIXED_MS"]==0)
        for x in group:
            r=[a["result"] for a in x["cases"]]
            b=[a["result"] for a in base["cases"]]
            x["objectives"]=[max(a["wall_s"]/c["wall_s"] for a,c in zip(r,b)),
                             max(a["ttft_max_s"]/c["ttft_max_s"] for a,c in zip(r,b)),
                             max(a["gap_p95_s"] for a in r),
                             max(a["gap_max_s"] for a in r)]
        for x in group:
            x["pareto"]=not any(dominates(y["objectives"],x["objectives"]) for y in group)
        rows.extend(group)
        print(name, "Pareto", [(x["policy"],x["objectives"]) for x in group if x["pareto"]], flush=True)
    args.output.write_text(json.dumps({"kind":"diagnostic-only","binary_sha256":hashlib.sha256(args.binary.read_bytes()).hexdigest(),
                                      "costs_sha256":hashlib.sha256(args.costs.read_bytes()).hexdigest(),"costs":costs,"rows":rows},indent=2)+"\n")

if __name__=="__main__":
    main()
