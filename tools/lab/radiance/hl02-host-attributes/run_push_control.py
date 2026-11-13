#!/usr/bin/env python3
"""RAD50's existing 4-MiB serial pull/push A/B under controlled primary-GPU load.

Caller must hold the GPU lease. Lab experiment, not an engine transport or
concurrent three-GPU push test. Only terminates its own background child.
"""
import argparse
import datetime as dt
import json
from pathlib import Path
import subprocess
import time


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--reader-binary",required=True,type=Path)
    ap.add_argument("--rad50-binary",required=True,type=Path)
    ap.add_argument("--output",required=True,type=Path)
    ap.add_argument("--primary",required=True,type=int)
    ap.add_argument("--victim",required=True,type=int)
    ap.add_argument("--iterations",type=int,default=10)
    ap.add_argument("--reverse",action="store_true")
    args=ap.parse_args()
    if args.victim==args.primary or args.iterations<10 or args.iterations>30:
        ap.error("devices must differ; iterations 10..30")
    args.output.parent.mkdir(parents=True,exist_ok=True)
    order=["idle","read","write","idle"] if not args.reverse else ["idle","write","read","idle"]
    with args.output.open("w") as output:
        def emit(row):
            output.write(json.dumps(row,sort_keys=True)+"\n")
            output.flush()
        emit({"kind":"header","utc":dt.datetime.now(dt.timezone.utc).isoformat(),"primary":args.primary,
              "victim":args.victim,"order":order,"iterations":args.iterations})
        for arm in order:
            background=None
            try:
                if arm!="idle":
                    bcmd=[str(args.reader_binary),str(args.primary),"coherent",arm,"20","64"]
                    background=subprocess.Popen(bcmd,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
                    time.sleep(1.0)
                    if background.poll() is not None:
                        raise RuntimeError("background exited before rad50 start")
                cmd=[str(args.rad50_binary),"--test","dmabuf-ab","--device",str(args.victim),
                     "--iterations",str(args.iterations),"--mode","direct",
                     "--bytes","4194304","--lease-confirmed"]
                started=time.time()
                result=subprocess.run(cmd,capture_output=True,text=True,timeout=25)
                emit({"kind":"comparison","arm":arm,"cmd":cmd,"elapsed_s":time.time()-started,
                      "returncode":result.returncode,"stdout":result.stdout,"stderr":result.stderr})
                if result.returncode or "RAD50_AB_COMPLETE" not in result.stdout:
                    raise RuntimeError("RAD50 comparison incomplete")
                if background is not None and background.poll() is not None:
                    raise RuntimeError("background did not overlap entire comparison")
            finally:
                if background is not None:
                    if background.poll() is None:
                        background.terminate()
                    try:
                        bout,berr=background.communicate(timeout=15)
                    except subprocess.TimeoutExpired:
                        background.kill()
                        bout,berr=background.communicate(timeout=5)
                    emit({"kind":"background","arm":arm,"stdout":bout,
                          "stderr":berr,"returncode":background.returncode})
        emit({"kind":"complete","utc":dt.datetime.now(dt.timezone.utc).isoformat()})
    print("HL02 RAD50 push control complete",args.output)


if __name__=="__main__":
    main()
