#!/usr/bin/env python3
"""Gate 4 slice analysis: strict complete-run proofs and provisional diagnostics.

Consumes raw command capture(s) from dmabuf-live-parallel / dmabuf-live-relay.
Never interprets a partial stdout capture, a process exit alone, or an
unpaired measurement as proof of verified live GPU readiness.
"""
from __future__ import annotations
import argparse
import math
from pathlib import Path

KINDS={"PARALLEL":"hip-register-mapped-direct-parallel",
       "RELAY":"hip-host-coherent-relay"}
ROWS={1,16,64,256,768}
TIMING=("arm_us","cpu_combine_us","dma_sync_us","helper_ack_us",
        "post_ack_complete_us","serial_us")

def emitted_indices(n:int)->list[int]:
    if n<=15:return list(range(n))
    return [i for i in range(n) if i<3 or (i+1)%10==0 or i+1==n]

def expected_offsets(count:int)->list[int]:
    """The tester reports the first three, every tenth and the final sample.
    Counts <=15 emit every generation. Suppressed samples still undergo full
    GPU verification and contribute to the in-process summary quantiles.
    """
    return [i for i in range(1,count+1)
            if count<=15 or i<=3 or i%10==0 or i==count]


def fields(line:str,prefix:str)->dict[str,str]:
    result={}
    for bit in line[len(prefix):].split():
        if "=" not in bit:raise ValueError(f"malformed field in {prefix}: {bit}")
        k,v=bit.split("=",1)
        if k in result:raise ValueError(f"duplicate {k}")
        result[k]=v
    return result


def number(row:dict[str,str],name:str)->int:
    try:return int(row[name])
    except (KeyError,ValueError) as e:raise ValueError(f"invalid integer {name}") from e


def decimal(row:dict[str,str],name:str)->float:
    try:v=float(row[name])
    except (KeyError,ValueError) as e:raise ValueError(f"invalid float {name}") from e
    if not math.isfinite(v) or v<0:raise ValueError(f"invalid nonnegative {name}")
    return v


def parse(text:str)->list[dict]:
    output=[]
    active=None
    for line in text.splitlines():
        if line.startswith("exitCode: ") and line!="exitCode: 0":
            raise ValueError("nonzero command exit")
        if line.startswith("timedOut: ") and line!="timedOut: false":
            raise ValueError("host timeout")
        if line.startswith("truncated: ") and line!="truncated: false":
            raise ValueError("tool marked captured output truncated")
        if line.startswith(("RAD50_RELAY_FAIL ","RAD50_PARALLEL_FAIL ")):
            raise ValueError("hardware failure: "+line)
        if line.startswith("RAD50 RESULT test=dmabuf-live-") and not line.endswith("exit=0"):
            raise ValueError("tester reported failure")
        if line.startswith("RAD50_LIVE_CONTEXT "):
            if active is not None:raise ValueError("previous run incomplete")
            meta=fields(line,"RAD50_LIVE_CONTEXT ")
            kind=next((k for k,v in KINDS.items() if v==meta.get("control")),None)
            if kind is None:raise ValueError("unknown slice control mode")
            device,rows,count=(number(meta,k) for k in ("device","rows","iterations"))
            if device<0 or rows not in ROWS or not 10<=count<=200 or \
               number(meta,"bytes")!=rows*5120:
                raise ValueError("invalid context shape")
            if (meta.get("consumer_armed_before_write")!="1" or
                meta.get("dma_buf_sync")!="1" or
                meta.get("ready_scope")!="system" or
                meta.get("reuse_after_done")!="1"):
                raise ValueError("missing GPU-live ordering contract")
            blocks=int(meta.get("gpu_parallel_blocks","256"))
            if blocks not in {1,32,64,128,256}:
                raise ValueError("invalid GPU verifier block count")
            active={"kind":kind,"device":device,"rows":rows,"count":count,
                    "blocks":blocks,"samples":[],"summarized":False,"summary":None}
        else:
            found=False
            for kind in KINDS:
                prefix=f"RAD50_{kind}_SAMPLE "
                if line.startswith(prefix):
                    found=True
                    if active is None or active["kind"]!=kind or active["summarized"]:
                        raise ValueError("sample outside matching run")
                    row=fields(line,prefix)
                    dev,rows,size,gen=(number(row,k) for k in
                                       ("device","rows","bytes","gen"))
                    index=len(active["samples"])
                    offsets=expected_offsets(active["count"])
                    if index>=len(offsets):raise ValueError("extra emitted sample")
                    if (dev!=active["device"] or rows!=active["rows"] or
                        size!=rows*5120 or gen!=100000+rows*1000+offsets[index] or
                        index>=active["count"] or number(row,"mismatches") or
                        number(row,"status") or
                        number(row,"gpu_seen_ready")!=gen or
                        number(row,"gpu_parallel_blocks")!=active["blocks"]):
                        raise ValueError("wrong/stale/unverified GPU-ready generation")
                    if ((kind=="PARALLEL" and
                         row.get("delivery")!="child-after-dma-sync-end") or
                        (kind=="RELAY" and
                         row.get("delivery")!="parent-after-socket-ack")):
                        raise ValueError("publication source changed")
                    for k in TIMING:decimal(row,k)
                    active["samples"].append(row)
                    break
                prefix=f"RAD50_{kind}_SUMMARY "
                if line.startswith(prefix):
                    found=True
                    if active is None or active["kind"]!=kind or active["summarized"]:
                        raise ValueError("summary outside matching run")
                    row=fields(line,prefix)
                    if (number(row,"device")!=active["device"] or
                        number(row,"rows")!=active["rows"] or
                        number(row,"bytes")!=active["rows"]*5120 or
                        number(row,"iterations")!=active["count"] or
                        number(row,"verified")!=active["count"] or
                        number(row,"emitted")!=len(expected_offsets(active["count"])) or
                        int(row.get("gpu_parallel_blocks","256"))!=active["blocks"] or
                        number(row,"status")!=0 or
                        len(active["samples"])!=len(expected_offsets(active["count"]))):
                        raise ValueError("incomplete or mismatched summary")
                    times=sorted(decimal(s,"serial_us") for s in active["samples"])
                    qs=[decimal(row,key) for key in
                        ("p50_us","p95_us","p99_us","max_us")]
                    if not qs[0]<=qs[1]<=qs[2]<=qs[3]:
                        raise ValueError("summary quantiles out of order")
                    if active["count"]<=15:
                        for field,index in (("p50_us",(len(times)-1)//2),
                                            ("p95_us",int(.95*(len(times)-1))),
                                            ("p99_us",int(.99*(len(times)-1)))):
                            if abs(decimal(row,field)-times[index])>.04:
                                raise ValueError(f"summary {field} disagrees with samples")
                    if any(t>qs[3]+.04 for t in times):
                        raise ValueError("sample greater than reported maximum")
                    active["summary"]=row
                    active["summarized"]=True
                    break
                prefix=f"RAD50_{kind}_COMPLETE "
                if line.startswith(prefix):
                    found=True
                    if active is None or active["kind"]!=kind or not active["summarized"]:
                        raise ValueError("completion without verified summary")
                    row=fields(line,prefix)
                    if (number(row,"device")!=active["device"] or
                        number(row,"rows")!=active["rows"] or
                        number(row,"iterations")!=active["count"] or
                        number(row,"verified")!=active["count"] or
                        number(row,"status")!=0):
                        raise ValueError("incorrect post-cleanup completion")
                    output.append(active)
                    active=None
                    break
            if not found and line.startswith("RAD50_") and \
               any(k in line for k in ("_SAMPLE","_SUMMARY","_COMPLETE")):
                # Avoid silently mixing unrelated benchmark sample families.
                if active is not None:raise ValueError("unexpected benchmark row")
    if active is not None:raise ValueError("incomplete run, partial log retained")
    if not output:raise ValueError("no completed slice runs")
    return output


def report(text:str)->str:
    runs=parse(text)
    output=["RAD50 GPU-live slice results: complete, bit-checked, post-cleanup",
            "NOT an inference speedup or a general ROCm coherence proof.",
            "mode gpu rows iterations median_us p95_us median_cpu_combine_us "
            "median_arm_us median_ack_us median_post_ack_us"]
    for run in runs:
        summary=run["summary"]
        output.append(f"{run['kind']} {run['device']} {run['rows']} {run['count']} "
                      f"{float(summary['p50_us']):.2f} {float(summary['p95_us']):.2f} "
                      f"{float(summary['combine_p50_us']):.2f} "
                      f"{float(summary['arm_p50_us']):.2f} "
                      f"{float(summary['ack_p50_us']):.2f} "
                      f"{float(summary['post_ack_complete_p50_us']):.2f}")
    output.append("Compact rows are sampled; all iterations are verified by the tester, "
                  "and summary quantiles are computed over all generations.")
    output.append("100 iterations still insufficient for a production p99 claim.")
    return "\n".join(output)


def provisional(text:str)->str:
    output=["PROVISIONAL: all available partial GPU-live observations, NOT verified."]
    context=0;count=0;complete=0
    for line in text.splitlines():
        if line.startswith("RAD50_LIVE_CONTEXT "):
            context+=1;output.append("CONTEXT "+line[len("RAD50_LIVE_CONTEXT "):])
        elif line.startswith(("RAD50_PARALLEL_SAMPLE ","RAD50_RELAY_SAMPLE ")):
            count+=1;output.append("OBSERVED_SAMPLE "+line)
        elif line.startswith(("RAD50_PARALLEL_COMPLETE ","RAD50_RELAY_COMPLETE ")):
            complete+=1;output.append("OBSERVED_COMPLETION "+line)
        elif line.startswith(("RAD50_RELAY_FAIL ","RAD50_PARALLEL_FAIL ",
                              "exitCode: ","timedOut: ","truncated: ","RAD50 RESULT ")):
            output.append("DIAGNOSTIC "+line)
    output.append(f"CONTEXTS={context} OBSERVED_SAMPLES={count} "
                  f"COMPLETIONS={complete} INCOMPLETE={max(0,context-complete)} "
                  "NOT_VERIFIED=1")
    return "\n".join(output)


def main()->int:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provisional",action="store_true")
    parser.add_argument("log",type=Path,nargs="+")
    args=parser.parse_args()
    try:
        data="\n".join(p.read_text() for p in args.log)
        print(provisional(data) if args.provisional else report(data))
    except (ValueError,OSError) as e:
        parser.exit(2,f"RAD50 slice analysis rejected input: {e}\n")
    return 0


if __name__=="__main__":
    raise SystemExit(main())
