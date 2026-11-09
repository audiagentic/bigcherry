#!/usr/bin/env python3
"""Strict CPU-only RAD50 S1 combine A/B evidence validator.

There is no GPU/VRAM measurement in this format. Full generation data are
verified in the bench; compact logs emit samples but report quantiles over all
verified iterations. Incomplete logs are provisional, not speedup evidence.
"""
from __future__ import annotations
import argparse
import math
from pathlib import Path

ROWS={1,16,64,256,768}
HELPERS={1,2,4}
LAYOUTS={"hot","unique"}
TIMES=("direct_us","stage_combine_us","stage_copy_us","stage_total_us")

def fields(line:str,prefix:str)->dict[str,str]:
    out={}
    for bit in line[len(prefix):].split():
        if "=" not in bit:
            raise ValueError(f"malformed field in {prefix}: {bit}")
        key,value=bit.split("=",1)
        if key in out:
            raise ValueError("duplicate field "+key)
        out[key]=value
    return out

def integer(d:dict[str,str],key:str)->int:
    try: return int(d[key])
    except (ValueError,KeyError) as e: raise ValueError("invalid "+key) from e

def duration(d:dict[str,str],key:str)->float:
    try: value=float(d[key])
    except (ValueError,KeyError) as e: raise ValueError("invalid "+key) from e
    if not math.isfinite(value) or value<0:
        raise ValueError("invalid nonnegative "+key)
    return value

def scheduled(n:int)->list[int]:
    return [i for i in range(1,n+1)
            if n<=15 or i<=3 or i%10==0 or i==n]

def quant(values:list[float],q:float)->float:
    v=sorted(values)
    return v[int((len(v)-1)*q)]

def parse(text:str)->dict:
    exit_ok=timeout_ok=truncation_ok=False
    context=None
    samples=[]
    summary=None
    complete=False
    result=False
    for line in text.splitlines():
        if line.startswith("exitCode: "):
            if exit_ok or line!="exitCode: 0":
                raise ValueError("nonzero or duplicate process exit")
            exit_ok=True
        elif line.startswith("timedOut: "):
            if timeout_ok or line!="timedOut: false":
                raise ValueError("host timeout or duplicate flag")
            timeout_ok=True
        elif line.startswith("truncated: "):
            if truncation_ok or line!="truncated: false":
                raise ValueError("truncated or duplicate capture flag")
            truncation_ok=True
        elif line.startswith("RAD50_CPU_CONTEXT "):
            if context is not None or samples or summary:
                raise ValueError("duplicate or late context")
            context=fields(line,"RAD50_CPU_CONTEXT ")
            rows=integer(context,"rows")
            helpers=integer(context,"helpers")
            iterations=integer(context,"iterations")
            if (rows not in ROWS or helpers not in HELPERS or
                not 10<=iterations<=200 or integer(context,"width")!=2560 or
                context.get("layout") not in LAYOUTS or
                context.get("paths")!="host-direct,host-stage-copy" or
                context.get("source")!="deterministic_bf16" or
                context.get("bitwise_oracle")!="1" or
                context.get("gpu")!="0"):
                raise ValueError("invalid CPU-only context contract")
        elif line.startswith("RAD50_CPU_SAMPLE "):
            if context is None or summary is not None or complete:
                raise ValueError("sample out of sequence")
            d=fields(line,"RAD50_CPU_SAMPLE ")
            offsets=scheduled(integer(context,"iterations"))
            if (len(samples)>=len(offsets) or
                integer(d,"iteration")!=offsets[len(samples)] or
                integer(d,"rows")!=integer(context,"rows") or
                integer(d,"helpers")!=integer(context,"helpers") or
                d.get("layout")!=context.get("layout") or
                integer(d,"mismatches")!=0 or integer(d,"status")!=0):
                raise ValueError("missing, wrong or unverified CPU iteration")
            for k in TIMES:duration(d,k)
            if abs(duration(d,"stage_total_us")-
                   duration(d,"stage_combine_us")-
                   duration(d,"stage_copy_us"))>.008:
                raise ValueError("stage timings not additive")
            samples.append(d)
        elif line.startswith("RAD50_CPU_SUMMARY "):
            if context is None or summary is not None or complete:
                raise ValueError("invalid summary sequence")
            d=fields(line,"RAD50_CPU_SUMMARY ")
            if (integer(d,"rows")!=integer(context,"rows") or
                integer(d,"helpers")!=integer(context,"helpers") or
                d.get("layout")!=context.get("layout") or
                integer(d,"iterations")!=integer(context,"iterations") or
                integer(d,"verified")!=integer(context,"iterations") or
                integer(d,"emitted")!=len(scheduled(integer(context,"iterations"))) or
                len(samples)!=integer(d,"emitted") or
                integer(d,"status")!=0):
                raise ValueError("incomplete summary")
            for k in ("direct_p50_us","direct_p95_us","direct_max_us",
                      "stage_combine_p50_us","stage_copy_p50_us",
                      "stage_total_p50_us","stage_total_p95_us",
                      "stage_total_max_us"):
                duration(d,k)
            for label in ("direct","stage_total"):
                p50,p95,max_v=(duration(d,label+"_"+key+"_us")
                               for key in ("p50","p95","max"))
                if not p50<=p95<=max_v:
                    raise ValueError("invalid quantile order")
                if any(duration(s,label+"_us")>max_v+.006 for s in samples):
                    raise ValueError("sample exceeds reported maximum")
                if integer(context,"iterations")<=15:
                    for key,q in (("p50",.5),("p95",.95),("max",1)):
                        actual=quant([duration(s,label+"_us") for s in samples],q)
                        if abs(actual-duration(d,label+"_"+key+"_us"))>.006:
                            raise ValueError("sample quantile mismatch")
            summary=d
        elif line.startswith("RAD50_CPU_COMPLETE "):
            if context is None or summary is None or complete:
                raise ValueError("completion before summary")
            d=fields(line,"RAD50_CPU_COMPLETE ")
            if (integer(d,"rows")!=integer(context,"rows") or
                integer(d,"helpers")!=integer(context,"helpers") or
                d.get("layout")!=context.get("layout") or
                integer(d,"verified")!=integer(context,"iterations") or
                integer(d,"status")!=0):
                raise ValueError("completion mismatch")
            complete=True
        elif line.startswith("RAD50 RESULT test=combine-cpu "):
            if result or line!="RAD50 RESULT test=combine-cpu exit=0":
                raise ValueError("tester process failure")
            result=True
        elif line.startswith("RAD50_CPU_"):
            raise ValueError("unrecognized CPU evidence or failure: "+line)
    if not (context and summary and complete and result and
            exit_ok and timeout_ok and truncation_ok):
        raise ValueError("incomplete CPU-only capture; retain as provisional")
    return {"context":context,"summary":summary,"samples":samples}

def report(logs:list[Path])->str:
    lines=["RAD50 S1 CPU host-only validated results: not BAR/VRAM or inference.",
           "rows helpers layout iterations direct_p50_us direct_p95_us stage_copy_p50_us stage_total_p50_us"]
    for f in logs:
        run=parse(f.read_text())
        c,s=run["context"],run["summary"]
        lines.append(" ".join([c["rows"],c["helpers"],c["layout"],c["iterations"],
                               s["direct_p50_us"],s["direct_p95_us"],
                               s["stage_copy_p50_us"],s["stage_total_p50_us"]]))
    lines.append("All reported samples and success envelopes were checked; compact runs include all generations in in-process quantiles.")
    return "\n".join(lines)

def provisional(text:str)->str:
    lines=["RAD50 S1 PROVISIONAL: incomplete/failed capture, NO speedup proof."]
    for line in text.splitlines():
        if line.startswith(("RAD50_CPU_","RAD50 RESULT ","exitCode: ",
                            "timedOut: ","truncated: ")):
            lines.append(line)
    lines.append("NOT_VERIFIED=1")
    return "\n".join(lines)

def main()->int:
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--provisional",action="store_true")
    p.add_argument("log",type=Path,nargs="+")
    a=p.parse_args()
    try:
        print(provisional("\n".join(f.read_text() for f in a.log))
              if a.provisional else report(a.log))
    except (ValueError,OSError) as e:
        p.exit(2,"RAD50 S1 CPU analysis rejected input: "+str(e)+"\n")
    return 0

if __name__=="__main__":
    raise SystemExit(main())
