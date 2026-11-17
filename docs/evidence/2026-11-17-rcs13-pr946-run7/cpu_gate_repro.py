#!/usr/bin/env python3
"""Run 7 CPU-only owner-map mismatch replay. Requires existing build and planner draft."""
import hashlib,json,os,subprocess
from pathlib import Path
source=Path(os.environ["RCS13_ARM_A_SOURCE_ROOT"])
build=Path(os.environ["RCS13_ARM_A_BUILD_DIR"])
plan=Path(os.environ["RCS13_PLAN_DIR"])
heat=source/"tools/lab/radiance/heat/perf51.rank0.hot-first"
assert hashlib.sha256(heat.read_bytes()).hexdigest()=="cbf8e17ad4e0e57470da2c000904d863621d5affcf8876c9b87960b9472ba06e"
assert (plan/"budget.json").is_file() and (plan/"manifest-measuring.json").is_file()
out=plan/"repro-preflight-owner.json"
cmd=[str(build/"radiance_home/bin/bc-serve"),"--write-manifest",str(out),
 "--unmeasured","--site",str(plan/"site.toml"),"--power-profile","stock",
 "--set","layout.name=flashnext-tp2xtx-r9700",
 "--set","service.max_sessions=1","--set","service.max_context=8192"]
env={**os.environ,"BIGCHERRY_LAYOUTS":str(source/"engines/radiance/layouts")}
p=subprocess.run(cmd,env=env,text=True,capture_output=True,timeout=90)
assert p.returncode==0,(p.returncode,p.stderr)
pre=json.loads(out.read_text())["spec"]["plan"]["owner_map"]
mea=json.loads((plan/"manifest-measuring.json").read_text())["spec"]["plan"]["owner_map"]
budget=json.loads((plan/"budget.json").read_text())
assert pre["owner_map_sha256"]==budget["owner_map_digest"],"preflight reproduction diverged"
counts=lambda d:[h["count"] for h in d["helpers"]]
a=[set(map(tuple,h["experts"])) for h in pre["helpers"]]
b=[set(map(tuple,h["experts"])) for h in mea["helpers"]]
print(json.dumps({"cpu_only":True,"preflight":counts(pre),"measuring":counts(mea),
 "preflight_digest":pre["owner_map_sha256"],"measuring_digest":mea["owner_map_sha256"],
 "only_preflight":[len(x-y) for x,y in zip(a,b)],
 "only_measuring":[len(y-x) for x,y in zip(a,b)]},indent=2))
assert pre["owner_map_sha256"]!=mea["owner_map_sha256"],"expected identity mismatch absent"
