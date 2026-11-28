#!/usr/bin/env python3
"""Historical source transformation of rad0061 for target 2.3.0; not a runtime tool."""
from pathlib import Path
import argparse
import textwrap
a=argparse.ArgumentParser()
a.add_argument("--apply",action="store_true",required=True)
a.parse_args()
p=Path(__file__).resolve().parents[3]
f=p/"engines/radiance/patches/rad0061_iq8r_f32_scale/patch.py"
s=f.read_text()
begin=s.index("    FilePatch(path='libr4d/r4d_gemm_fp8a8.hip'")
end=s.index("    FilePatch(path='libr4d/r4d_selftest.cpp'",begin)
def block(path, edits):
    lines=[f"    FilePatch(path={path!r}, language=\"none\",", '              description="RUM15 IQ8R F32 scale reads", edits=(']
    for ident,old,new in edits:
        lines.append(f"        rep({ident!r}, {old!r}, {new!r}),")
    return "\n".join(lines)+"\n    )),\n"
def norm(x):
    return textwrap.dedent(x).strip("\n")+"\n"
old="""    constexpr uint32_t kWS = MW * kFolds * 8u;
"""
new="""    // One dword per wide IQ8R scale, two BF16 values per dword otherwise.
    constexpr uint32_t kScaleWords_rad0061 = SB == 256u ? 16u : 8u;
    constexpr uint32_t kWS = MW * kFolds * kScaleWords_rad0061;
"""
ed=[("rum15-decode-kpar-narrow-1",old,new)]
old="""        const uint32_t tr = (m0 + (i / (kFolds * 8u)) * 16u) / 16u;
        const uint32_t kb = c / SB + (i / 8u) % kFolds;
        return ((Full || tr * 16u < rows) && kb < nkb8)
            ? ((const uint32_t*)Sa)[((size_t)tr * nkb8 + kb) * 8u + i % 8u] : 0u;
"""
ed.append(("rum15-decode-kpar-narrow-2",old,old.replace("8u","kScaleWords_rad0061")))
# Keep scale branch and integer sums exactly as native, change only the source representation.
old="""                float wsr[8];
                {
                    const uint4 w8 =
                        *(const uint4*)&s_ws[(wave * kFolds + f) * 8u + (lane >> 4) * 4u];
                    const uint32_t d[4] = { w8.x, w8.y, w8.z, w8.w };
"""
new="""                float wsr[8];
                if constexpr (SB == 256u) {
                    const uint32_t* scale = &s_ws[(wave * kFolds + f) * 16u + (lane >> 4) * 8u];
#pragma unroll
                    for (uint32_t t = 0; t < 8; ++t) wsr[t] = __uint_as_float(scale[t]);
                } else {
                    const uint4 w8 =
                        *(const uint4*)&s_ws[(wave * kFolds + f) * 8u + (lane >> 4) * 4u];
                    const uint32_t d[4] = { w8.x, w8.y, w8.z, w8.w };
"""
ed.append(("rum15-decode-kpar-narrow-3",old,new))
# Keep native narrow ABI pointer type: only the compile-time SB=256 instantiation interprets float.
# This avoids edits to the multi (SB128) and row-major call interfaces.
old="""        const uint4 q4 = *(const uint4*)(Sa + ((size_t)rt * ngrp + g) * 16u + acc_row(lane, 0));
        const uint32_t qd[4] = { q4.x, q4.y, q4.z, q4.w };
#pragma unroll
        for (uint32_t j = 0; j < NT; ++j) {
            const float as = xsr[j][g];
#pragma unroll
            for (uint32_t t = 0; t < 4; ++t) {
                out[j][2 * t] = __builtin_fmaf(acc_i32(raw[j][2 * t]),
                                               __uint_as_float(qd[t] << 16) * as, out[j][2 * t]);
                out[j][2 * t + 1] = __builtin_fmaf(acc_i32(raw[j][2 * t + 1]),
                                                   __uint_as_float(qd[t] & 0xFFFF0000u) * as,
                                                   out[j][2 * t + 1]);
            }
        }
"""
new="""        if constexpr (SB == 256u) {
            const float* scale = (const float*)Sa + ((size_t)rt * ngrp + g) * 16u + acc_row(lane, 0);
#pragma unroll
            for (uint32_t j = 0; j < NT; ++j) {
                const float as = xsr[j][g];
#pragma unroll
                for (uint32_t t = 0; t < 8; ++t)
                    out[j][t] = __builtin_fmaf(acc_i32(raw[j][t]), scale[t] * as, out[j][t]);
            }
        } else {
"""+old+"        }\n"
ed.append(("rum15-kpar-fold",old,new))
new=block("libr4d/r4d_gemm_i8a8.hip",ed)
ed=[]
old="""        uint4 w8[!RB ? kPTW : 1u];
"""
ed.append(("rum15-prefill-1",old,"""        uint4 w8[!RB && SB != 256u ? kPTW : 1u];
        float w32[!RB && SB == 256u ? kPTW * 8u : 1u];
"""))
old="""                if constexpr (ROWM) {
                    /* Eight bf16 scales, one a row, packed as the tile-major uint4 the fold reads. */
"""
ed.append(("rum15-prefill-2",old,"""                if constexpr (SB == 256u) {
                    const float* scale = (const float*)Sw + ((size_t)(wt0 + i) * nkb + kb) * 16u +
                                         (lane >> 4) * 8u;
#pragma unroll
                    for (uint32_t t = 0; t < 8; ++t) w32[i * 8u + t] = scale[t];
                } else if constexpr (ROWM) {
                    /* Eight bf16 scales, one a row, packed as the tile-major uint4 the fold reads. */
"""))
old="""                float s[8];
                const uint32_t d[4] = { w8[i].x, w8[i].y, w8[i].z, w8[i].w };
#pragma unroll
                for (uint32_t t = 0; t < 8; ++t)
                    s[t] = as * bscale((uint16_t)(d[t / 2] >> (16 * (t % 2))));
"""
ed.append(("rum15-prefill-3",old,"""                float s[8];
                if constexpr (SB == 256u) {
#pragma unroll
                    for (uint32_t t = 0; t < 8; ++t) s[t] = as * w32[i * 8u + t];
                } else {
                    const uint32_t d[4] = { w8[i].x, w8[i].y, w8[i].z, w8[i].w };
#pragma unroll
                    for (uint32_t t = 0; t < 8; ++t)
                        s[t] = as * bscale((uint16_t)(d[t / 2] >> (16 * (t % 2))));
                }
"""))
new+=block("libr4d/r4d_gemm_8a8_prefill.h",ed)
f.write_text(s[:begin]+new+s[end:])
# Put the gfx11 accumulator conversion before the shared native scale comment;
# both the production and diagnostic folds consume gfx12-format raw.
f=p/"engines/radiance/components/libr3/native/r4d_gemm_i8a8.hip.rw"
s=f.read_text()
old="        const uint4 q4 = *(const uint4*)(Sa + ((size_t)rt * ngrp + g) * 16u + acc_row(lane, 0));"
assert s.count(old)==2
s=s.replace(old,"        /* The lane's eight rows' bf16 scales of this group are consecutive in the tile-major\n         * plane: one 16-byte load. */")
f.write_text(s)
for rel in ("engines/radiance/patches/rad0061_iq8r_f32_scale/patch.toml","engines/radiance/patches/unit-sharing.toml","tools/tests/patch/test_unit_ownership.py"):
    f=p/rel
    f.write_text(f.read_text().replace("r4d_gemm_fp8a8","r4d_gemm_i8a8"))
