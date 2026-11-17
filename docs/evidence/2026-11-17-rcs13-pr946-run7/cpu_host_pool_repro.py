#!/usr/bin/env python3
"""CPU-only reproducibility of Run7 measured-plan exit 11, no engine/helper start.

Prerequisites: the source-matched A build, original plan-a directory with the
site.toml, plan-measured.json, plan-ranks.json, manifest-measuring.json and
the original plan-cache record, exposed as RCS13_PLAN_CACHE_RECORD.
Configure BIGCHERRY_ENVIRONMENT and BC_WORK_ROOT for the original test host.
This replayer writes an isolated plan cache inside RCS13_PLAN_DIR.
"""
import json,os,re,subprocess
from pathlib import Path
root=Path(os.environ["RCS13_ARM_A_SOURCE_ROOT"])
build=Path(os.environ["RCS13_ARM_A_BUILD_DIR"])
d=Path(os.environ["RCS13_PLAN_DIR"])
stored=json.loads(Path(os.environ["RCS13_PLAN_CACHE_RECORD"]).read_text())
manifest=json.loads((d/"manifest-measuring.json").read_text())
assert stored["plan"]["host_pool_mib"]==32005
assert stored["resident"]=={"engine_mib":183,"helpers_mib":509}
sets={"engine.max_num_batched_tokens":256,"engine.ngram_placement":"disk",
 "engine.port":manifest["spec"]["processes"]["engine"]["server"]["port"],
 "handoff.histogram_period_s":1,"handoff.max_rows":64,"handoff.small_rows":8,
 "layout.name":"flashnext-tp2xtx-r9700",
 "memory.host_pool_mib":32005,"memory.host_kv_mib":0,
 "service.max_context":8192,"service.max_sessions":1,
 "tuning.weights_disk_tier":"on"}
serialize=lambda v:json.dumps(v) if isinstance(v,str) else str(v)
argsets=[a for k in sorted(sets) for a in ("--set",k+"="+serialize(sets[k]))]
cmd=[str(build/"radiance_home/bin/bc-serve"),"--write-manifest",
 str(d/"cpu-host-pool-repro.json"),"--plan-cache",
 str(d/"cpu-repro-isolated-cache"),"--engine-binary",str(build/"bin/radiance"),
 "--plan-extra",stored["key_inputs"]["engine_args"],"--site",str(d/"site.toml"),
 "--power-profile","stock",*argsets,"--measured-plan",str(d/"plan-measured.json"),
 "--rank-demand",str(d/"plan-ranks.json"),
 "--helpers-resident-mib",str(stored["resident"]["helpers_mib"]),
 "--engine-resident-mib",str(stored["resident"]["engine_mib"])]
env={**os.environ,"BIGCHERRY_LAYOUTS":str(root/"engines/radiance/layouts")}
p=subprocess.run(cmd,env=env,text=True,capture_output=True,timeout=120)
m=re.search(r"the plan was measured with a ([0-9]+) MiB host pool; the measured inputs need ([0-9]+) MiB",p.stderr)
print(json.dumps({"cpu_only":True,"exit":p.returncode,
 "pool_mib":int(m.group(1)) if m else None,
 "derived_required_mib":int(m.group(2)) if m else None},indent=2))
assert p.returncode==11 and m and tuple(map(int,m.groups()))==(32005,32301),"Run7 exit-11 failure did not reproduce"
