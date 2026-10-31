// The 16x16x16 tile for r3_compat.h's WMMA wrappers, on gfx10.3 (RDNA2), which has no WMMA. D = A x B + C in the gfx11 fragment layout
// (A: lane l holds row l & 15, B: lane l holds column l & 15, both 16 values of K; C/D: lane l holds column l & 15 of rows 2t + (l >> 4),
// t = 0..7): each lane takes the 8 dwords of its 8 rows from the lanes that hold them (ds_bpermute) and runs them against its own
// column with f32 FMAs. Correct and slow (64 exchanges, 128 FMAs a tile); a native tile belongs here when one is worth the work.
// (Taking the halves out of the v16h fragment with a vector bit_cast gave wrong tiles on gfx1030; the dword form is checked against a
// CPU reference on a 6900 XT.) Not self-contained: r3_compat.h includes it after the vector types and r3_lane() / r3_from_lane().
#pragma once
__device__ __forceinline__ r3_v8f r3_mma16_f16(r3_v16h a, r3_v16h b, r3_v8f acc) {
    const r3_u8 av = __builtin_bit_cast(r3_u8, a), bv = __builtin_bit_cast(r3_u8, b);
    const unsigned half = r3_lane() >> 4;
    r3_v8f out = acc;
#pragma unroll
    for (int t = 0; t < 8; ++t) {
        const unsigned src = 2u * t + half;
        float s = acc[t];
#pragma unroll
        for (int j = 0; j < 8; ++j)
        {
            const unsigned x = r3_from_lane(src, av[j]), y = bv[j];
            s = __builtin_fmaf((float)__builtin_bit_cast(_Float16, (unsigned short)x), (float)__builtin_bit_cast(_Float16, (unsigned short)y), s);
            s = __builtin_fmaf((float)__builtin_bit_cast(_Float16, (unsigned short)(x >> 16)), (float)__builtin_bit_cast(_Float16, (unsigned short)(y >> 16)), s);
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
