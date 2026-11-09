#!/usr/bin/env python3
"""RAD50 Gate 4 cross-process LIVE GPU-ready evidence parser.

Default is fail-closed: require every expected generation, GPU verification,
summary, post-cleanup completion and non-failed command status. Provisional
mode retains any observed samples and the exact failure/timeout reason.
Hardware pass does NOT prove a portable ROCm memory-model guarantee or
that serving MoE throughput improves.
"""
from __future__ import annotations
import argparse
import math
from pathlib import Path

ROWS = {1, 16, 64, 256, 768}
REPORT_FIELDS = ("cpu_combine_us", "dma_sync_us",
                 "helper_ack_after_arm_us", "serial_us")


def fields(line: str, prefix: str) -> dict[str, str]:
    values: dict[str, str] = {}
    for item in line[len(prefix):].split():
        if "=" not in item:
            raise ValueError(f"malformed {prefix.strip()} entry: {item}")
        k,v=item.split("=",1)
        if k in values:
            raise ValueError(f"duplicate field {k}")
        values[k]=v
    return values


def integers(row: dict[str,str], *names: str) -> tuple[int, ...]:
    try:
        return tuple(int(row[name]) for name in names)
    except (ValueError, KeyError) as error:
        raise ValueError(f"invalid integer field: {error}") from error


def validate_context(row: dict[str,str]) -> tuple[int,int,int]:
    dev, rows, size, count=integers(row,"device","rows","bytes","iterations")
    if dev<0 or rows not in ROWS or size!=rows*5120 or count<10 or count>200:
        raise ValueError("invalid live context shape or sample count")
    if (row.get("consumer_armed_before_write")!="1" or
        row.get("dma_buf_sync")!="1" or row.get("ready_scope")!="system" or
        row.get("reuse_after_done")!="1"):
        raise ValueError("live publication contract omitted")
    return dev,rows,count


def parse(text: str) -> list[dict]:
    completed=[]
    active=None
    for line in text.splitlines():
        if line.startswith("exitCode: ") and line.strip()!="exitCode: 0":
            raise ValueError("nonzero shell exit")
        if line.startswith("timedOut: ") and line.strip()!="timedOut: false":
            raise ValueError("shell timeout")
        if line.startswith("truncated: ") and line.strip()!="truncated: false":
            raise ValueError("truncated output")
        if line.startswith("RAD50_LIVE_FAIL "):
            raise ValueError(f"hardware live failure: {line}")
        if line.startswith("RAD50 RESULT test=dmabuf-live ") and not line.endswith("exit=0"):
            raise ValueError("tester reported failure")
        if line.startswith("RAD50_LIVE_CONTEXT "):
            if active:
                raise ValueError("previous live run has not completed")
            row=fields(line,"RAD50_LIVE_CONTEXT ")
            dev,rows,count=validate_context(row)
            active={"dev":dev,"rows":rows,"count":count,"samples":[],
                    "summarized":False}
        elif line.startswith("RAD50_LIVE_SAMPLE "):
            if active is None or active["summarized"]:
                raise ValueError("sample outside active live run")
            row=fields(line,"RAD50_LIVE_SAMPLE ")
            dev,rows,size,gen,errors,status=integers(
                row,"device","rows","bytes","gen","mismatches","status")
            n=len(active["samples"])
            expected=100000+rows*1000+n+1
            if (dev!=active["dev"] or rows!=active["rows"] or size!=rows*5120
                or n>=active["count"] or gen!=expected or errors or status
                or row.get("armed_before_helper")!="1"
                or row.get("stale_ready_retained")!="1"):
                raise ValueError(f"wrong/stale/unverified live sample at gen {gen}")
            if "gpu_seen_ready" in row and integers(row,"gpu_seen_ready")[0]!=gen:
                raise ValueError("GPU did not observe expected ready generation")
            for name in REPORT_FIELDS:
                try:
                    val=float(row[name])
                except (ValueError, KeyError) as error:
                    raise ValueError(f"bad live timing {name}") from error
                if not math.isfinite(val) or val<0:
                    raise ValueError(f"invalid live timing {name}")
            active["samples"].append(row)
        elif line.startswith("RAD50_LIVE_SUMMARY "):
            if active is None or active["summarized"]:
                raise ValueError("unexpected live summary")
            row=fields(line,"RAD50_LIVE_SUMMARY ")
            dev,rows,size,count,verified,status=integers(
                row,"device","rows","bytes","iterations","verified","status")
            if (dev!=active["dev"] or rows!=active["rows"] or size!=rows*5120
                or count!=active["count"] or verified!=count or
                len(active["samples"])!=count or status!=0):
                raise ValueError("incomplete/incorrect live summary")
            active["summarized"]=True
        elif line.startswith("RAD50_LIVE_COMPLETE "):
            if active is None or not active["summarized"]:
                raise ValueError("unverified or misplaced completion")
            row=fields(line,"RAD50_LIVE_COMPLETE ")
            dev,rows,count,verified,status=integers(
                row,"device","rows","iterations","verified","status")
            if (dev!=active["dev"] or rows!=active["rows"] or
                count!=active["count"] or verified!=count or status):
                raise ValueError("invalid post-cleanup completion")
            completed.append(active)
            active=None
    if active:
        raise ValueError("truncated/incomplete live run")
    if not completed:
        raise ValueError("no completed live runs")
    return completed


def report(text: str) -> str:
    runs=parse(text)
    out=["RAD50 live concurrent producer/consumer serial-generation evidence.",
         "No portability or inference-speed guarantee."]
    for run in runs:
        durations=sorted(float(s["serial_us"]) for s in run["samples"])
        n=len(durations)
        out.append(f"PASS device={run['dev']} rows={run['rows']} "
                   f"verified={n} p50_us={durations[(n-1)//2]:.2f} "
                   f"p95_us={durations[int((n-1)*.95)]:.2f} "
                   "stale_ready_retained=1")
    return "\n".join(out)


def provisional(text: str) -> str:
    out=["PROVISIONAL RAD50 LIVE diagnostics; no verified PASS inference or portable-ordering claim."]
    contexts=samples=completions=0
    for line in text.splitlines():
        for prefix,tag in (("RAD50_LIVE_CONTEXT ","CONTEXT"),
                           ("RAD50_LIVE_SAMPLE ","OBSERVED_SAMPLE"),
                           ("RAD50_LIVE_SUMMARY ","OBSERVED_SUMMARY"),
                           ("RAD50_LIVE_COMPLETE ","OBSERVED_COMPLETION"),
                           ("RAD50_LIVE_FAIL ","FAILURE")):
            if line.startswith(prefix):
                out.append(f"{tag} {line[len(prefix):]}")
                contexts+=tag=="CONTEXT"
                samples+=tag=="OBSERVED_SAMPLE"
                completions+=tag=="OBSERVED_COMPLETION"
                break
        if (line.startswith("exitCode: ") or line.startswith("timedOut: ") or
            line.startswith("truncated: ") or
            line.startswith("RAD50 RESULT test=dmabuf-live ")):
            out.append("OBSERVED_STATUS "+line)
    out.append(f"OBSERVED_TOTAL contexts={contexts} samples={samples} "
               f"completions={completions} incomplete={max(contexts-completions,0)} "
               "NOT_VERIFIED=1")
    return "\n".join(out)


def main()->int:
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--provisional",action="store_true")
    ap.add_argument("log",type=Path,nargs="+")
    opts=ap.parse_args()
    try:
        data="\n".join(p.read_text() for p in opts.log)
        print(provisional(data) if opts.provisional else report(data))
    except (OSError,ValueError) as error:
        ap.exit(2,f"RAD50 live log analysis rejected: {error}\n")
    return 0

if __name__=="__main__":
    raise SystemExit(main())
