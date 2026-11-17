#!/usr/bin/env python3
"""CPU-only recheck of RCS13 Run8 B measured owner-budget refusal.

Provide paths to archived B measuring draft and the run8 measured-plan
stderr (for example, via RCS13_MEASURING_DRAFT and RCS13_PLAN_LOG).
No GPU access, child starts, or live capacity sampling is performed.
"""
import json, os, re
from pathlib import Path
draft=json.loads(Path(os.environ["RCS13_MEASURING_DRAFT"]).read_text())
log=Path(os.environ["RCS13_PLAN_LOG"]).read_text(errors="replace")
h=draft["spec"]["processes"]["helpers"]
assert len(h)==1
assert draft["spec"]["plan"]["state"]=="unmeasured"
need=int(h[0]["vram_budget_mib"])
match=re.search(
  r"the measured owner map needs ([0-9]+) MiB of experts, but live claimable VRAM after the worker footprint allows only ([0-9]+) MiB",
  log)
assert match and "BC_CFG_ADMISSION_REFUSED" in log
reported, ceiling=map(int,match.groups())
assert reported==need and need>ceiling>0
assert "the measured manifest was not written (2)" in log
print(json.dumps({"cpu_only":True,"measured_owner_mib":need,
 "captured_live_ceiling_mib":ceiling,"shortfall_mib":need-ceiling,
 "admission_refused":True,"historic_live_capacity_independently_measured":False},
 indent=2,sort_keys=True))
