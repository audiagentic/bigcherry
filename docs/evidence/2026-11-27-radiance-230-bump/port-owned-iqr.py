#!/usr/bin/env python3
"""One-time owned IQR source port. Preserve the separately qualified fused and native paths."""
import argparse
from pathlib import Path
a=argparse.ArgumentParser()
a.add_argument("--apply", action="store_true", required=True)
a.parse_args()
p=Path(__file__).resolve().parents[3]/"engines/radiance/components/bc-iqr"
for name in ("bc_iqr_gemm.hip","bc_iqr_vocab.hip"):
    f=p/name
    s=f.read_text().replace("mma8<true>","mma_i8").replace("acc_val<true>","acc_i32")
    s=s.replace("gemm_8a8_prefill_body<true, ","gemm_8a8_prefill_body<")
    s=s.replace("r4d_gemm_fp8a8.hip","r4d_gemm_i8a8.hip").replace("gemm_fp8a8_body at I8","gemm_i8a8_narrow_body")
    f.write_text(s)
f=p/"bc_iqr_embed.hip"
s=f.read_text().replace("the values the table's bf16\n// widening at load produced, bit for bit.","the table's exact decode to\n// bf16, bit for bit.")
f.write_text(s)
f=p/"bc_gemm_8a8_prefill.h"
s=f.read_text()
assert "template <bool I8, int RB" in s
s=s.replace("template <bool I8, int RB","template <int RB").replace("(I8 && RB == 0 && SB == 256u)","(RB == 0 && SB == 256u)")
s=s.replace('    static_assert(!RB || I8, "RB is the int8 tile\'s");\n',"")
# Both architecture branches already have the identical loop and fold. Remove only the
# discarded false-I8 branch and leave arch selection, operands and accumulation intact.
s=s.replace("            if constexpr (I8) {\n","",1)
i=s.index("            } else {\n                i2v bf[kPTA];")
j=s.index("#else  // bc-gfx11",i)
s=s[:i]+s[j:]
s=s.replace("                            false));\n                }\n#else  // bc-gfx11", "                            false));\n#else  // bc-gfx11",1)
s=s.replace("mma8<I8>","mma_i8").replace("acc_val<I8>","acc_i32")
s=s.replace("static_assert(I8 && SB == kBlock","static_assert(SB == kBlock")
s=s.replace("uint4 w8[I8 && !RB ?","uint4 w8[!RB ?")
s=s.replace("        } else if constexpr (I8) {\n","        } else {\n",1)
s=s.replace("        } else {\n            ws = bscale(Sw[(size_t)(n0 / kBlock) * sw_ld + kb]);\n","",1)
s=s.replace("#if defined(__gfx1100__)  // bc-gfx11\n        }\n        if constexpr (I8) {\n","#if defined(__gfx1100__)  // bc-gfx11\n",1)
s=s.replace("#endif  // bc-gfx11\n        }\n        if constexpr (RB != 0) {"," #endif  // bc-gfx11\n        if constexpr (RB != 0) {",1)
# Keep the tile's kTPB scope closed on both architectures.
s=s.replace("            tile(kb, 3u, w1, w0);\n#if defined(__gfx1100__)","            tile(kb, 3u, w1, w0);\n        }\n#if defined(__gfx1100__)",1)
s=s.replace(" #endif  // bc-gfx11","#endif  // bc-gfx11")
i=s.index("                if constexpr (I8) {\n")
j=s.index("                } else {",i)
k=s.index("                }\n#pragma unroll",j)+len("                }\n")
inner=s[i+len("                if constexpr (I8) {\n"):j]
s=s[:i]+"\n".join(x[4:] if x.startswith("    ") else x for x in inner.rstrip().splitlines())+"\n"+s[k:]
s=s.replace("r4d_gemm_fp8a8.hip","r4d_gemm_i8a8.hip").replace("the int8 and fp8 prefill tile's body","the int8 prefill tile's body")
assert "if constexpr (I8)" not in s and "mma8<I8>" not in s and "acc_val<I8>" not in s
f.write_text(s)
