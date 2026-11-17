#!/usr/bin/env python3
"""Card-free Run 9 source-isolation gate. Never opens a GPU or starts a server.

Run from any checkout of audiagentic/bigcherry containing these commits:
    python3 cpu_source_gate.py --repo /path/to/bigcherry
This reproduces why retained e28734f09 is not a valid sole-RCS13 control
for candidate 17a16f5d7, despite the clean branch diff against main.
"""
import argparse, json, subprocess
from pathlib import Path
BASE="e28734f097fe7c0ff59415ecfe0494271a1e22a0"
CANDIDATE="17a16f5d7132efaa1c07dc858e111dfb7645ad10"
SEVEN={
 "engines/radiance/components/bc-gfx12/bc_gfx12_plugin.cpp",
 "engines/radiance/components/bc-gfx12/bc_moe_gemv.hip",
 "engines/radiance/components/moe-remote/gemv_geometry.h",
 "engines/radiance/components/moe-remote/gemv_geometry_test.cpp",
 "engines/radiance/components/moe-remote/moe_lane_gemm.h",
 "engines/radiance/components/moe-remote/moe_remote_gpu.h",
 "engines/radiance/model_constant_guard.toml",
}
parser=argparse.ArgumentParser()
parser.add_argument("--repo",type=Path,default=Path.cwd())
args=parser.parse_args()
def git(*cmd):
    return subprocess.check_output(["git","-C",str(args.repo),*cmd],text=True).strip()
branch=set(git("diff","--name-only","7c0273976627c339fa8f50c54543da2c1419ac2f..."+CANDIDATE).splitlines())
binary_sources=set(git("diff","--name-only",BASE,CANDIDATE).splitlines())
assert branch==SEVEN,("branch itself drifted",sorted(branch.symmetric_difference(SEVEN)))
extra=sorted(binary_sources-SEVEN)
assert extra, "A/B source isolation surprisingly restored"
for required in ("engines/radiance/components/serving/session.cpp",
                 "engines/radiance/components/serving/runtime_guards.cpp",
                 "engines/radiance/components/serving/calibration_lab.cpp"):
    assert required in extra, ("expected additional serving change absent",required)
hunks=git("diff","--unified=0",BASE,CANDIDATE,"--",
          "engines/radiance/components/serving/session.cpp")
lines=[line for line in hunks.splitlines() if line.startswith("@@")]
assert lines and any("+741," in line or "+741 " in line for line in lines)
obj={"cpu_only":True,"branch_isolated_seven":True,
     "actual_build_sources_isolated":False,
     "baseline":BASE,"candidate":CANDIDATE,
     "branch_delta_count":len(branch),
     "actual_source_delta_count":len(binary_sources),
     "extra_non_rcs13_files":extra,
     "serving_session_hunks":lines,
     "gate":"STOP_OLD_BASELINE_NOT_AN_RCS13_ONLY_COMPARISON",
     "requires":"fresh baseline current-main build OR candidate rebuilt against old baseline"}
print(json.dumps(obj,indent=2,sort_keys=True))
