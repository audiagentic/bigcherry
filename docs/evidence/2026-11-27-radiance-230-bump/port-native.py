#!/usr/bin/env python3
"""One-time port from the 58ad10011 active rewrite inventory, not a runtime tool."""
import argparse
from pathlib import Path
import re
a=argparse.ArgumentParser()
a.add_argument("--apply",action="store_true",required=True)
a.parse_args()
p=Path(__file__).resolve().parents[3]
n=p/"engines/radiance/components/libr3/native"
def remove_blocks(s, ids):
    idx=0
    def sub(m):
        nonlocal idx
        idx+=1
        return "" if idx in ids else m[0]
    t=re.sub(r"(?ms)^@@ find[^\n]*\n.*?^@@ end\n",sub,s)
    assert ids <= set(range(1,idx+1)),(ids,idx)
    return t
assert (n/"r4d_gemm_fp8a8.hip.rw").is_file()
f=n/"r4d_gemm_8a8_prefill.h.rw"
s=f.read_text()
s=s[:s.index("# Single-band IQR:")]
s=s.replace("# Native (the int8 instantiations only; the fp8 ones keep the compatibility layer):","# Native (upstream 2.3 int8-only body):")
s=s.replace("mma8<I8>(af, bf[j],\n                                         ","mma_i8(af, bf[j],\n                                        ")
a=s.index("            if constexpr (I8) {")
b=s.index("            } else {",a)
c=s.index("\n@@ end",b)
s=s[:a]+"\n".join(x[4:] if x.startswith("    ") else x for x in s[a+len("            if constexpr (I8) {\n"):b].rstrip().splitlines())+s[c:]
a=s.index("        if constexpr (I8) {")
b=s.index("\n@@ end",a)
inner=s[a+len("        if constexpr (I8) {\n"):b].rstrip()
assert inner.endswith("        }")
s=s[:a]+"\n".join(x[4:] if x.startswith("    ") else x for x in inner[:-9].rstrip().splitlines())+s[b:]
f.write_text(s.rstrip()+"\n")
f=n/"r4d_gemm_fp8a8.hip.rw"
s=remove_blocks(f.read_text(),{4,5}).replace("mma8<true>","mma_i8")
(n/"r4d_gemm_i8a8.hip.rw").write_text(s)
f.unlink()
f=n/"r4d_gemm_fp8a16_nt_m1.hip.rw"
s=f.read_text().replace(" && I8","").replace("e4m3x8_to_bf16<I8>","codes8_to_bf16")
(n/"r4d_logits_gemm_i8.hip.rw").write_text(s)
f.unlink()
f=n/"r4d_gemm_mxfp4a8_gemv.hip.rw"
s=remove_blocks(f.read_text(),{9,*range(11,23)}).replace("@@ find 2\n          acc =", "@@ find 1\n          acc =")
a=s.index("# ---- the gated forms:")
b=s.index("# ---- the WMMA launcher:",a)
s=s[:a]+s[b:]
s=s.replace("the plain WMMA form and the two gated forms","the retained plain WMMA form").replace("shared by the plain and the gated-wide forms","plain form")
f.write_text(s)
f=n/"r4d_selftest.cpp.rw"
s=f.read_text().replace("the 8-bit trunk GEMMs -- block E4M3 W8A8 and the int8","the int8 trunk GEMMs (W8A8 and the i8a8 / IQ8R rows) at")
f.write_text(s)
for rel in ("engines/radiance/components/libr3/r3_groups_test.py","tools/lab/radiance/unit_families.toml"):
    f=p/rel
    f.write_text(f.read_text().replace("r4d_gemm_fp8a8","r4d_gemm_i8a8"))
