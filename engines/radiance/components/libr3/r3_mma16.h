// The 16x16x16 tile for r3_compat.h's WMMA wrappers, on gfx11: the WMMA instruction (A: lane l holds row l & 15, B: lane l holds column
// l & 15, both 16 values of K; C/D: lane l holds column l & 15 of rows 2t + (l >> 4), t = 0..7). Not self-contained: r3_compat.h includes
// it after the vector types and r3_lane() / r3_from_lane() it shares with the other port of this layout, libr2, which has a file of the
// same name beside its own sources (no WMMA on gfx10.3), found first because r3_compat.h includes it with <> and libr2 lists its own directory first.
#pragma once
__device__ __forceinline__ r3_v8f r3_mma16_f16(r3_v16h a, r3_v16h b, r3_v8f acc) {
    return __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, b, acc);
}
__device__ __forceinline__ r3_v8f r3_mma16_bf16(r3_v16bf a, r3_v16bf b, r3_v8f acc) {
    return __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(a, b, acc);
}
