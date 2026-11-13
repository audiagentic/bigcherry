#!/usr/bin/env python3
"""Pre-armed RAD50 GPU-live verification while the primary GPU reads host memory.

Requires an exclusive GPU lease and no foreign KFD processes. Child-owned
background reader only. Tests one destination at a time, not three-way TP3.
"""
import argparse
import json
from pathlib import Path
import re
import subprocess
import sys
import time

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--reader-binary",type=Path,required=True)
    ap.add_argument("--rad50-binary",type=Path,required=True)
    ap.add_argument("--output-dir",type=Path,required=True)
    ap.add_argument("--primary",type=int,required=True)
    ap.add_argument("--victims",nargs="+",type=int,required=True)
    ap.add_argument("--iterations",type=int,default=100)
    args=ap.parse_args()
    if args.primary in args.victims or len(set(args.victims))!=len(args.victims):
        ap.error("primary and victims must be distinct")
    args.output_dir.mkdir(parents=True,exist_ok=True)
    cmd=[str(args.reader_binary),str(args.primary),"coherent","read","20","64"]
    bg=subprocess.Popen(cmd,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
    try:
        time.sleep(1)
        if bg.poll() is not None:
            raise RuntimeError("primary reader ended before test began")
        test=[sys.executable,str(Path(__file__).with_name("run_live_repeats.py")),
              "--binary",str(args.rad50_binary),"--output-dir",str(args.output_dir),
              "--devices",*[str(v) for v in args.victims],
              "--iterations",str(args.iterations),"--rows","768"]
        cp=subprocess.run(test,stdout=subprocess.PIPE,stderr=subprocess.PIPE,
                          text=True,timeout=80)
        (args.output_dir/"runner.stdout.log").write_text(cp.stdout)
        (args.output_dir/"runner.stderr.log").write_text(cp.stderr)
        if cp.returncode or bg.poll() is not None:
            raise RuntimeError("GPU-live test failed or primary reader ended early")
    finally:
        if bg.poll() is None:
            bg.terminate()
        try:
            out,err=bg.communicate(timeout=15)
        except subprocess.TimeoutExpired:
            bg.kill()
            out,err=bg.communicate(timeout=5)
        (args.output_dir/"background.stdout.log").write_text(out)
        (args.output_dir/"background.stderr.log").write_text(err)
    match=re.search(r"HL02_RESULT .*device_gbps=([\d.]+).*gpu_word_errors=(\d+) status=(\w+)",out)
    if not match or int(match.group(2))!=0 or match.group(3)!="PASS" or float(match.group(1))<10:
        raise RuntimeError("primary background reader not validated")
    payload=json.loads((args.output_dir/"live-summary.json").read_text())
    if any(not r.get("passed") for r in payload["results"]):
        raise RuntimeError("one or more consumers failed full-word verification")
    payload["primary_read_gbps"]=float(match.group(1))
    payload["primary_source"]="coherent GPU-read mapped host 64 MiB"
    (args.output_dir/"contended-summary.json").write_text(json.dumps(payload,indent=2)+"\n")
    print("HL02_CONTENDED_LIVE_PASS",json.dumps(payload,sort_keys=True))

if __name__=="__main__":
    main()
