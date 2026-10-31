// The 16x16x16 tile for r3_compat.h's WMMA wrappers, on gfx10.3 (RDNA2), which has no WMMA. D = A x B + C in the gfx11 fragment layout
// (A: lane l holds row l & 15, B: lane l holds column l & 15, both 16 values of K; C/D: lane l holds column l & 15 of rows 2t + (l >> 4),
// t = 0..7): each lane takes the 8 dwords of its 8 rows from the lanes that hold them (ds_bpermute) and runs them against its own
// column with f32 FMAs. f16 is the native dot2 form below; bf16 (no dot2 on gfx10.3) stays the exchange-and-FMA form: correct and slow, and not on the MoE path.
// (Taking the halves out of the v16h fragment with a vector bit_cast gave wrong tiles on gfx1030; the dword form is checked against a
// CPU reference on a 6900 XT.) Not self-contained: r3_compat.h includes it after the vector types and r3_lane() / r3_from_lane().
#pragma once
__device__ __forceinline__ r3_v8f r3_mma16_f16(r3_v16h a, r3_v16h b, r3_v8f acc) {
    // Native form (RAD23): no cross-lane exchange through LDS. A row m sits in lane m of each 16-lane row (both rows hold all 16); this
    // lane (row `half`, column n) needs A rows 2t + half. The upper row first swaps neighbouring lanes (quad_perm 1,0,3,2), so lane 2t of
    // either row holds A row 2t + half; DPP row_share then broadcasts that lane across the row, and v_dot2_f32_f16 takes two products a dword.
    // 8 swaps + 64 shares + 64 dot2 per tile, all VALU. (The __builtin_amdgcn_fdot2 form gave wrong sums on gfx1030, like the vector
    // bit_cast the older form notes; the instruction by asm is checked against a CPU reference on a 6900 XT.)
    const r3_u8 av0 = __builtin_bit_cast(r3_u8, a), bv = __builtin_bit_cast(r3_u8, b);
    const bool upper = (r3_lane() >> 4) != 0;
    r3_u8 av;
#pragma unroll
    for (int j = 0; j < 8; ++j) {
        const unsigned swapped = (unsigned)__builtin_amdgcn_update_dpp(0, (int)av0[j], 0xB1, 0xF, 0xF, false);
        av[j] = upper ? swapped : av0[j];
    }
    r3_v8f out = acc;
#pragma unroll
    for (int t = 0; t < 8; ++t) {
        float s = acc[t];
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            // row_share:n is DPP control 0x150 + n; the source lane is 2t, a compile-time constant after unrolling
            const unsigned x = (unsigned)__builtin_amdgcn_update_dpp(0, (int)av[j], 0x150 + 2 * t, 0xF, 0xF, false);
            asm("v_dot2_f32_f16 %0, %1, %2, %0" : "+v"(s) : "v"(x), "v"(bv[j]));
        }
        out[t] = s;
    }
    return out;
}
__device__ __forceinline__ r3_v8f r3_mma16_bf16(r3_v16bf a, r3_v16bf b, r3_v8f acc) {
    const r3_u8 av = __builtin_bit_cast(r3_u8, a), bv = __builtin_bit_cast(r3_u8, b);
    const unsigned half = r3_lane() >> 4;
    r3_v8f out = acc;
#pragma unroll
    for (int t = 0; t < 8; ++t) {
        const unsigned src = 2u * t + half;
        float s = acc[t];
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            const unsigned x = r3_from_lane(src, av[j]), y = bv[j];
            s = __builtin_fmaf(__builtin_bit_cast(float, x << 16), __builtin_bit_cast(float, y << 16), s);
            s = __builtin_fmaf(__builtin_bit_cast(float, x & 0xffff0000u), __builtin_bit_cast(float, y & 0xffff0000u), s);
        }
        out[t] = s;
    }
    return out;
}
