#!/usr/bin/env python3
"""One-time source edits; run only on the original rad0029 with --apply."""
from pathlib import Path
import argparse
p=argparse.ArgumentParser()
p.add_argument("--apply", action="store_true", required=True)
p.parse_args()
f=Path(__file__).resolve().parents[3]/"engines/radiance/patches/rad0029_rank_own_split/patch.py"
s=f.read_text()
a='''                         ? fmt(", and %lld MiB stay unclaimed", (long long)rs.unusable_mib).c_str()
                         : "");
        }'''
b='''                RAD_INFO("card %d uses %lld MiB of room above its own headroom for resident experts",
                         ranks_[i]->device, (long long)add);
            }
        }'''
assert s.count(a)==2
s=s.replace(a,b)
s=s.replace("cfg_.gpu_memory_mib <= 0;", "!broker_ && cfg_.vram_claim <= 0 && cfg_.gpu_memory_mib <= 0;")
s=s.replace("int64_t free_min = -1, statics_max = 0;","int64_t statics_max = 0;")
s=s.replace("                if (free_min < 0 || f.free < free_min) free_min = f.free;\n","")
s=s.replace("std::max<int64_t>(facts[i].free - free_min, 0) / kMiB", "vram_budget_rank_spare_mib(cfg_, facts, i, false)")
s=s.replace("free memory it measured above the smallest card's", "room above the least rank after its own reserves")
s=s.replace("free memory it\n         * measured above the smallest card's", "room above the least rank after its own reserves")
a="PATCHES = (BRINGUP,)"
assert s.count(a)==1
s=s.replace(a,r'''# Reuse the native room/reserve arithmetic. Keep the native named-only default for
# the expert path; the dense path deliberately also uses unnamed card spare room.
_SPARE_SIG = """int64_t vram_budget_rank_spare_mib(const Config& cfg, const std::vector<VramFacts>& per_rank,
                                   size_t rank)"""
def _edit(id, old, new, guard):
    return Edit(id=id, anchor=re.escape(old), mode="replace", text=new,
                guard=re.escape(guard), expect_matches=1, max_span_lines=old.count("\n")+1,
                rationale="VTP33 reuse native per-card reserve accounting")

BUDGET_HEADER = FilePatch(path="core/mem/vram_budget.h", language="none", description="Dense rank spare", edits=(
    _edit("own-spare-decl", _SPARE_SIG + ";",
          _SPARE_SIG.replace("size_t rank)", "size_t rank, bool named_only = true)") + ";",
          "bool named_only = true)"),
))
BUDGET = FilePatch(path="core/mem/vram_budget.cpp", language="none", description="Dense rank spare", edits=(
    _edit("own-spare-def", _SPARE_SIG + " {",
          _SPARE_SIG.replace("size_t rank)", "size_t rank, bool named_only)") + " {",
          "size_t rank, bool named_only)"),
    _edit("own-spare-filter", "    if (!named) return 0;",
          "    if (named_only && !named) return 0; /* BC_VTP33_SPARE_FILTER */", "BC_VTP33_SPARE_FILTER"),
))
PATCHES = (BUDGET_HEADER, BUDGET, BRINGUP)''')
f.write_text(s)
