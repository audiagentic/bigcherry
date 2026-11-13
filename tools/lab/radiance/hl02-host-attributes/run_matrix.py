#!/usr/bin/env python3
"""HL02 lab-only paired, verified HIP mapped-host attribute matrix.

Caller MUST acquire exclusive GPU lease and verify no foreign KFD processes.
No process outside this script's own children is signaled.
"""
import argparse
import datetime as dt
import json
from pathlib import Path
import subprocess
import time

MODES = ("coherent", "noncoherent", "writecombined")


def execute(binary, device, mode, direction, seconds, mib, timeout):
    cmd = [str(binary), str(device), mode, direction, str(seconds), str(mib)]
    start = time.time()
    cp = subprocess.run(cmd, text=True, capture_output=True, timeout=timeout)
    return {"cmd": cmd, "returncode": cp.returncode, "elapsed_s": time.time()-start,
            "stdout": cp.stdout, "stderr": cp.stderr}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--binary", required=True, type=Path)
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--primary", type=int, required=True)
    ap.add_argument("--victims", nargs="+", type=int, required=True)
    ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("--directions", nargs="+", choices=("read","write"), default=("read","write"))
    ap.add_argument("--reverse", action="store_true", help="reverse mode order for paired order-control rounds")
    ap.add_argument("--seconds", type=float, default=1.0)
    ap.add_argument("--mib", type=int, default=64)
    args = ap.parse_args()
    if args.rounds not in range(1, 5) or not 0.5 <= args.seconds <= 4:
        ap.error("rounds 1..4 and seconds 0.5..4 required")
    if args.primary in args.victims or len(set(args.victims)) != len(args.victims):
        ap.error("distinct device ordinals are required")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as log:
        def record(x):
            log.write(json.dumps(x, sort_keys=True) + "\n")
            log.flush()
        record({"kind": "header", "started_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
                "primary":args.primary,"victims":args.victims,"rounds":args.rounds,
                "seconds":args.seconds,"mib":args.mib,"binary":str(args.binary)})
        for round_no in range(args.rounds):
            modes = MODES if (round_no + int(args.reverse)) % 2 == 0 else tuple(reversed(MODES))
            for victim in args.victims:
                for direction in args.directions:
                    for mode in modes:
                        r=execute(args.binary,victim,mode,direction,args.seconds,args.mib,30)
                        record({"kind":"measurement","round":round_no,"primary_action":"idle",
                                "victim":victim,"mode":mode,"direction":direction,**r})
                        if r["returncode"]:
                            raise RuntimeError("idle run failed; see JSONL")
                    for background_direction in ("read", "write"):
                        bg_cmd=[str(args.binary),str(args.primary),"coherent",
                                background_direction,"12",str(args.mib)]
                        bg=subprocess.Popen(bg_cmd,stdout=subprocess.PIPE,
                                             stderr=subprocess.PIPE,text=True)
                        try:
                            time.sleep(1.0)
                            if bg.poll() is not None:
                                out,err=bg.communicate()
                                raise RuntimeError(f"background exited early {bg.returncode}: {out} {err}")
                            for mode in modes:
                                r=execute(args.binary,victim,mode,direction,args.seconds,args.mib,30)
                                record({"kind":"measurement","round":round_no,
                                        "primary_action":background_direction,
                                        "victim":victim,"mode":mode,"direction":direction,**r})
                                if r["returncode"] or bg.poll() is not None:
                                    raise RuntimeError("victim failed or background finished early; see JSONL")
                        finally:
                            # We only signal a direct child created by this script.
                            if bg.poll() is None:
                                bg.terminate()
                            try:out,err=bg.communicate(timeout=15)
                            except subprocess.TimeoutExpired:
                                bg.kill()
                                out,err=bg.communicate(timeout=5)
                            record({"kind":"background","round":round_no,
                                    "primary_action":background_direction,
                                    "victim":victim,"direction":direction,"cmd":bg_cmd,
                                    "returncode":bg.returncode,
                                    "stdout":out,"stderr":err})
        record({"kind":"complete","ended_utc":dt.datetime.now(dt.timezone.utc).isoformat()})
    print("HL02 matrix complete",args.output)


if __name__=="__main__":
    main()
