#!/usr/bin/env python3
"""RCS13 run 5: minimal CPU-only repetition of the failed owner-map preflight.

DO NOT substitute another heat file or retry with edited budgets. This only
duplicates the exact --write-manifest --unmeasured call made by the maintained
radiance_launch._collect() at its owner-map gate. No engine/helper is launched.
"""
import argparse, json, os, subprocess, tempfile
from pathlib import Path

MODEL=Path(os.environ["BC_MODEL_ROOT"])/"radiance/bigcherry/qwen3.8-next-flash-hcbf16/qwen3.8-next-flash-hcbf16.rad"
HEAT=Path(os.environ["BC_WORK_ROOT"])/"runs/perf51/heat.rank0.hot-first"
LAYOUT="flashnext-tp2xtx-r9700"

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--build",required=True,type=Path)
    ap.add_argument("--source-root",required=True,type=Path)
    ns=ap.parse_args()
    build=ns.build.resolve()
    assert (build/"radiance_home/bin/bc-serve").is_file()
    assert MODEL.is_file()
    assert not HEAT.is_file(),f"reproducer precondition changed: {HEAT} now exists"
    with tempfile.TemporaryDirectory(prefix="rcs13-run5-minimal-") as temp:
        t=Path(temp)
        site=t/"site.toml"
        site.write_text("[engine]\nmodel = "+json.dumps(str(MODEL))+
            "\nradiance_home = "+json.dumps(str(build/"radiance_home"))+
            "\n[handoff]\nheat = "+json.dumps(str(HEAT))+
            "\nhistogram = "+json.dumps("/dev/null")+"\n")
        cmd=[str(build/"radiance_home/bin/bc-serve"),"--write-manifest",
             str(t/"owner-map.json"),"--unmeasured","--site",str(site),
             "--power-profile","stock","--set",f"layout.name={LAYOUT}",
             "--set","service.max_sessions=1","--set","service.max_context=8192"]
        env=os.environ.copy()
        env.update({"BIGCHERRY_LAYOUTS":str(ns.source_root.resolve()/"engines/radiance/layouts"),
                    "BIGCHERRY_ENVIRONMENT":str(Path(os.environ["BC_WORK_ROOT"])/"environment.local.toml"),
                    "BC_WORK_ROOT":os.environ["BC_WORK_ROOT"],"BC_MODEL_ROOT":os.environ["BC_MODEL_ROOT"]})
        res=subprocess.run(cmd,text=True,capture_output=True,timeout=90,env=env)
        print("exit",res.returncode)
        print("stderr",res.stderr.strip())
        assert res.returncode==2, f"unexpected exit {res.returncode}"
        assert f"R3_MOE_WORKER_HEAT {HEAT}: want 65536 counts, read 0" in res.stderr
        assert "owned-expert selection refused" in res.stderr
        assert not (t/"owner-map.json").exists()
        print("confirmed: identical deterministic preflight error; no measured manifest")
if __name__=="__main__":main()
