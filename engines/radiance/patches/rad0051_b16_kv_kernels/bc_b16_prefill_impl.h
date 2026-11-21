/* rad0051-owned typed prefill kernel, extracted from the frozen 2.2 ABI44 source. */
#pragma once
#include "r4d_b16_kv.h"

template<int NWARPS, int TILE, int HEAD_DIM, int GQA, int BS, int KVP, int OPT, int WIN = 0, RadKVLayout Layout = RAD_KVQ_NONE>
__global__ __launch_bounds__(NWARPS * 32)
void r4d_attn_prefill_kernel(const R4DArgs a)
{
    constexpr int F16   = (OPT & O_F16)   ? 1 : 0;
    constexpr int MSKIP = (OPT & O_MSKIP) ? 1 : 0;
    constexpr int FOLDQ = (OPT & O_FOLDQ) ? 1 : 0;
    constexpr int CTG   = (OPT & O_CONTIG) ? 1 : 0;
    constexpr int PF    = 1 << ((OPT >> 10) & 3);
    constexpr int SGB   = (OPT & O_SGB) ? 1 : 0;
    constexpr int DOT2  = ((OPT & O_DOT2) && (OPT & O_F16) && !(OPT & O_PV8)) ? 1 : 0;
    constexpr int BTS   = (OPT & O_BTS)   ? 1 : 0;
    constexpr int V4    = ((OPT & O_V4) && !KVP) ? 1 : 0;
    constexpr int K6    = ((OPT & O_K6) && V4) ? 1 : 0;
    constexpr int I8    = ((OPT & O_I8) && K6) ? 1 : 0;
    constexpr bool KB   = r4d_b16_layout(Layout);
    constexpr bool KI   = Layout == RAD_KVQ_B16I8;
    constexpr bool KBF  = KVP || KB;
    constexpr bool VI8  = I8 || KI;
    constexpr bool VSTAGE = V4 || KI;
    constexpr int VG    = KI ? R4dKvBytes<Layout>::group : 32;
    static_assert(!KB || (HEAD_DIM == 256 && !KVP && !V4 && F16), "hybrid rows stage BF16 K");
    static_assert(!I8 || !(OPT & (O_QK8 | O_PV8)), "the i8 cache has no e4m3 legs");
    // A token's row in a page, and where in it V's codes, V's scales and K's scales start.
    constexpr int ROWB  = KB ? HEAD_DIM / 16 * R4dKvBytes<Layout>::row16
                        : I8 ? 2 * HEAD_DIM + 4
                        : K6 ? HEAD_DIM * 22 / 16 : V4 ? HEAD_DIM * 25 / 16 : 2 * HEAD_DIM;
    constexpr int VBOFF = KB ? HEAD_DIM / 16 * R4dKvBytes<Layout>::value16 : HEAD_DIM;
    constexpr int VISOFF = HEAD_DIM / 16 * R4dKvBytes<Layout>::value_scale16;
    constexpr int VOFF  = I8 ? HEAD_DIM : K6 ? HEAD_DIM * 3 / 4 : HEAD_DIM;
    constexpr int VSOFF = I8 ? 2 * HEAD_DIM + 2 : K6 ? HEAD_DIM * 21 / 16 : HEAD_DIM * 3 / 2;
    constexpr int KSOFF = I8 ? 2 * HEAD_DIM : HEAD_DIM * 5 / 4;
    constexpr int GPREV = ((OPT & O_GPREV) && !VSTAGE) ? 1 : 0;
    constexpr int LDSB  = (OPT & O_LDSB)  ? 1 : 0;
    constexpr int KWIDE = (OPT & O_KWIDE) ? 1 : 0;
    // fp8_v4 takes none of the three (KLATE, the assume, the scale pack): they gain at 100k but lose
    // 3-7 % at 25k and 50k.
    constexpr int OLDV4 = V4 && !K6;
    constexpr int KLATE = ((OPT & O_KLATE) && KWIDE && !KBF && !OLDV4) ? 1 : 0;
    constexpr int QK8   = ((OPT & O_QK8) && !KVP) ? 1 : 0;
    constexpr int PV8   = ((OPT & O_PV8) && !KVP) ? 1 : 0;
    static_assert(!(OPT & (O_QK8 | O_PV8)) || (!KVP && !KB),
                  "lossy prefill products require the supported native or V4/K6 cache rows");
    constexpr int SPLIT = (OPT & O_PSPLIT) ? 1 : 0;
    static_assert(!SPLIT || !WIN, "the split partition has no window floor (the merge's WIN has)");
    static_assert(!QK8 || KWIDE, "QK8 raw-byte staging is written for the wide K fetch");
    typedef DT16<F16> D;
    typedef typename D::frag frag16;

    constexpr int ROWS    = NWARPS * 16;
    constexpr int BLOCK_Q = ROWS / GQA;
    constexpr int NKS     = HEAD_DIM / 16;
    constexpr int MT      = TILE / 16;
    constexpr int KSTR    = HEAD_DIM + (CTG ? 8 : 4);
    constexpr int VSTR    = TILE + (CTG ? 8 : 4);
    constexpr int NTHR    = NWARPS * 32;
    constexpr int NB      = TILE / BS;                       // block-table entries per tile
    constexpr int KEL     = KWIDE ? 16 : 8;                  // K elements fetched per thread-pass
    constexpr int KITEMS  = TILE * (HEAD_DIM / KEL);          // K passes needed for one tile
    constexpr int KPASS   = (KITEMS + NTHR - 1) / NTHR;        // passes per thread
    constexpr int KREGS   = KPASS * (KBF ? 2 : 1);             // uint4 of raw K per thread
    constexpr int NTR     = (TILE / 8) * (HEAD_DIM / (KVP ? 32 : 64));
    constexpr int VREGS   = (NTR + NWARPS - 1) / NWARPS;        // uint4 of raw V per thread
    constexpr float PSHIFT = D::SHIFT;
    constexpr float PGROW  = PV8 ? 8.0f : D::GROW;   // e4m3 max is 448 = 2^8.8
    static_assert(!KWIDE || KPASS == 1, "the wide K fetch is one straight-line pass");
    static_assert(!K6 || I8 || KWIDE, "k6_v4 K is staged sixteen dims a thread");
    static_assert((!QK8 && !PV8) || CTG, "the 8-bit fragment maps are written for CONTIG");

    constexpr int KSTR8 = HEAD_DIM + 8, VSTR8 = TILE + 8;
    __shared__ uint16_t sK[QK8 ? (TILE * KSTR8 + 1) / 2 : TILE * KSTR];
    __shared__ uint16_t sV16[PV8 ? (HEAD_DIM * VSTR8 + 1) / 2 : HEAD_DIM * VSTR];
    uint8_t* const sK8 = (uint8_t*)sK;
    uint8_t* const sV8 = (uint8_t*)sV16;

    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    // The launch is NTHR threads. Stated, so the wide K fetch's `key >= TILE` test (never true:
    // KITEMS == NTHR) folds: as a branch its join made the compiler wait on every load in flight right
    // after the K fetch, the V stores' cover included.
    if constexpr (!OLDV4) __builtin_assume(tid < NTHR);
    const int c = lane & 15, h = lane >> 4;

    // SPLIT: blockIdx.x = qtile * splits + segment; wave-uniform, so the divide is scalar.
    const int qb = SPLIT ? (int)blockIdx.x / a.splits : (int)blockIdx.x;
    const int sp = SPLIT ? (int)blockIdx.x - qb * a.splits : 0;
    const int kvh = blockIdx.y, seq = blockIdx.z;
    const int ctx = a.seqused_k[seq];
    if (ctx <= 0) return;

    // Ragged batches: this sequence owns rows [qoff, qoff+qlen) of q and out. grid.x is sized from
    // the batch maximum q_len, so blocks past a shorter sequence's end exit here.
    const int qoff = r4d_q_off(a.cu_q, seq, a.q_len);
    const int qlen = r4d_q_len(a.cu_q, seq, a.q_len);
    if (qb * BLOCK_Q >= qlen) return;

    const int row = warp * 16 + c;
    const int qi = row / GQA, hi = row - qi * GQA;
    const int qpos = qb * BLOCK_Q + qi;
    const bool live = (qpos < qlen);
    const int qrow = live ? qpos : (qlen - 1);
    const int qhead = kvh * GQA + hi;

    const int* __restrict__ bt = a.block_table + (size_t)seq * a.max_blocks;

    // SPLIT: tps tiles a segment, as r4d_attn_tps cuts tiles(ctx). A segment at or past the merge's
    // segment count u = ceil(tiles / tps) is never read: exit. One wholly past this workgroup's
    // causal end must still be written, since the merge reads all u segments of every row: its walk
    // is empty, so the epilogue stores the empty partial (O = 0, m_ref = -3e38, l = 0), and the
    // prologue's prefetch of tile tstart stays inside the sequence. (A separate early store spills
    // k6_v4's next-tile fetch addresses inside the tile loop.)
    const int sp_tps = SPLIT ? ((ctx + TILE - 1) / TILE + a.splits - 1) / a.splits : 0;
    if constexpr (SPLIT) {
        if (sp * sp_tps >= (ctx + TILE - 1) / TILE) return;
    }

    const float kdesc = a.k_descale ? a.k_descale[seq * a.kv_heads + kvh] : 1.0f;
    const float vdesc = a.v_descale ? a.v_descale[seq * a.kv_heads + kvh] : 1.0f;
    const float mul   = a.scale * kdesc * 1.44269504089f;

    frag16 qf[NKS];
    v2i32_r4d qf8[QK8 ? NKS : 1];
    {
        const uint16_t* qp = (const uint16_t*)a.q
                           + ((size_t)(qoff + qrow) * a.q_heads + qhead) * HEAD_DIM;
        #pragma unroll
        for (int t = 0; t < NKS; ++t) {
            // CONTIG reads the 8 dims as one run; the split map takes them 4+4.
            const uint2 l0 = *(const uint2*)(qp + 16 * t + (CTG ? 8 * h : 4 * h));
            const uint2 h0 = *(const uint2*)(qp + 16 * t + (CTG ? 8 * h + 4 : 8 + 4 * h));
            if constexpr (QK8) {
                const float m_ = FOLDQ ? mul : 1.0f;
                #define R4D_QB(w) __builtin_bit_cast(float, (w) << 16) * m_, \
                                  __builtin_bit_cast(float, (w) & 0xffff0000u) * m_
                qf8[t] = (v2i32_r4d){(int)pk_fp8x4(R4D_QB(l0.x), R4D_QB(l0.y)),
                                     (int)pk_fp8x4(R4D_QB(h0.x), R4D_QB(h0.y))};
                #undef R4D_QB
            } else {
                qf[t] = D::mk(make_uint2(r4d_qcvt<D, FOLDQ, F16>(l0.x, mul),
                                         r4d_qcvt<D, FOLDQ, F16>(l0.y, mul)),
                              make_uint2(r4d_qcvt<D, FOLDQ, F16>(h0.x, mul),
                                         r4d_qcvt<D, FOLDQ, F16>(h0.y, mul)));
            }
        }
    }

    const int klimit = ctx - qlen + qpos;
    // First key in this row's window, floored at zero. Same formula as r4d_attn_wlo, which is
    // defined in the decode unit, included after this one.
    int wlo = (WIN && a.window > 0) ? (ctx - qlen + qpos - a.window + 1) : 0;
    if (WIN && a.window > 0 && wlo < 0) wlo = 0;

    float m_ref = -3.0e38f;
    float off   = PSHIFT + 3.0e38f;
    float l_i   = 0.0f;
    v8f acc[NKS];
    #pragma unroll
    for (int t = 0; t < NKS; ++t) acc[t] = (v8f){0,0,0,0,0,0,0,0};

    const int qmax_cta   = min(qb * BLOCK_Q + BLOCK_Q - 1, qlen - 1);
    const int klimit_cta = ctx - qlen + qmax_cta;
    const int ctiles = (min(klimit_cta + 1, ctx) + TILE - 1) / TILE;
    // SPLIT: the segment's tiles, clipped at the causal end (empty when it starts past it).
    const int ntiles = SPLIT ? min((sp + 1) * sp_tps, ctiles) : ctiles;
    // First window tile, taken at this workgroup's first query row (the one reaching furthest
    // back), not the chunk's row 0, so fully masked tiles are never walked. Starts at 0 without a
    // window.
    const int wlo0 = (WIN && a.window > 0) ? (ctx - qlen + qb * BLOCK_Q - a.window + 1) : 0;
    const int tstart = SPLIT ? sp * sp_tps
                     : (WIN && a.window > 0 && wlo0 > 0) ? (wlo0 / TILE) : 0;

    // ---- staging, split into a global fetch (registers) and an LDS store ------------------------
    // The split lets GPREV move the fetch out from between the barriers.
    const int vg = lane >> 3, vj = lane & 7;
    uint4 kreg[KREGS], vreg[VREGS];
    // i8: the K scale of the key a pass stages.
    uint32_t ksc8[I8 ? KPASS : 1];

    // ---- fp8_v4 / k6_v4: V is int4 in groups of 32 with an f16 scale (absmax / 7): 16 bytes of
    // codes a group at VOFF (dim d at bits 4*(d%8) of word d/8), the group scales at VSOFF. A thread stages one key pair's 8 dims: two code words and the pair's scale,
    // fetched a tile ahead like GPREV's V, so the loads are in flight across the m-tile loop. They
    // widen into sV16 ([dim][key]) as a dword per dim holding both keys: with u = code ^ 8 (offset
    // binary), 0x6400 | u is the f16 1024 + u, so (x - 1032) * s is the value times s, rounded
    // once. Key pairs run fastest across lanes,
    // which keeps the dword stores of one dim row in distinct banks.
    static_assert(!V4 || F16, "fp8_v4 widens V to f16");
    constexpr int NP4 = (TILE / 2) * (HEAD_DIM / 8);
    constexpr int U4  = VSTAGE ? (NP4 + NTHR - 1) / NTHR : 1;
    // The pair's two scales stay apart until storeV4 packs them: packed here, the prefetch a tile
    // ahead would be waited on right after it is issued (fp8_v4 packs here, see KLATE).
    // i8: V's codes are bytes, two words a key's 8 dims, and a key has one scale; the rest is the
    // same with u = code ^ 0x80 and (x - 1152) * s.
    uint32_t v4c[U4][VI8 ? 4 : 2], v4s[U4][2];
    auto fetchV4 = [&](int k0, const int* blks) {
        #pragma unroll
        for (int r = 0; r < U4; ++r) {
            const int u = tid + r * NTHR;
            if (NP4 % NTHR == 0 || u < NP4) {
                const int kp = u % (TILE / 2), o = u / (TILE / 2);
                uint32_t sc[2];
                #pragma unroll
                for (int j = 0; j < 2; ++j) {
                    const int kk = min(k0 + 2 * kp + j, ctx - 1);
                    const int bk = BTS ? blks[(kk - k0) / BS] : bt[kk / BS];
                    const uint8_t* base = (const uint8_t*)a.kv + (size_t)bk * a.kv_block_stride
                                        + (size_t)kvh * a.kv_head_stride + (size_t)(kk % BS) * ROWB;
                    if constexpr (VI8) {
                        const uint2 cw = *(const uint2*)(base + (KI ? VBOFF : VOFF) + o * 8);
                        v4c[r][2 * j]     = cw.x;
                        v4c[r][2 * j + 1] = cw.y;
                        sc[j] = *(const uint16_t*)(base + (KI ? VISOFF + 2 * (o * 8 / VG) : VSOFF));
                    } else {
                        v4c[r][j] = *(const uint32_t*)(base + VOFF + o * 4);
                        sc[j]     = *(const uint16_t*)(base + VSOFF + (o >> 2) * 2);
                    }
                }
                if constexpr (OLDV4) { v4s[r][0] = sc[0] | (sc[1] << 16); v4s[r][1] = 0; }
                else                 { v4s[r][0] = sc[0]; v4s[r][1] = sc[1]; }
            }
        }
    };
    auto storeV4 = [&]() {
        typedef _Float16 h2 __attribute__((ext_vector_type(2)));
        #pragma unroll
        for (int r = 0; r < U4; ++r) {
            const int u = tid + r * NTHR;
            if (NP4 % NTHR == 0 || u < NP4) {
                const int kp = u % (TILE / 2), o = u / (TILE / 2);
                // Codes are two's complement nibbles; ^ 8 makes them offset binary (value + 8). i8:
                // bytes, ^ 0x80; `n` is a key's dims 0-3 of the octet and `m` its dims 4-7.
                const uint32_t n0 = VI8 ? v4c[r][0] ^ 0x80808080u
                                       : (v4c[r][0] & 0x0F0F0F0Fu) ^ 0x08080808u;
                const uint32_t m0 = VI8 ? v4c[r][1] ^ 0x80808080u
                                       : ((v4c[r][0] >> 4) & 0x0F0F0F0Fu) ^ 0x08080808u;
                const uint32_t n1 = VI8 ? v4c[r][VI8 ? 2 : 1] ^ 0x80808080u
                                       : (v4c[r][1] & 0x0F0F0F0Fu) ^ 0x08080808u;
                const uint32_t m1 = VI8 ? v4c[r][VI8 ? 3 : 1] ^ 0x80808080u
                                       : ((v4c[r][1] >> 4) & 0x0F0F0F0Fu) ^ 0x08080808u;
                const h2 s2 = __builtin_bit_cast(h2, OLDV4 ? v4s[r][0] : (v4s[r][0] | (v4s[r][1] << 16)));
                const h2 bias = { (_Float16)(VI8 ? 1152.0f : 1032.0f), (_Float16)(VI8 ? 1152.0f : 1032.0f) };
                uint16_t* dst = &sV16[(o * 8) * VSTR + 2 * kp];
                uint8_t* dst8 = &sV8[(o * 8) * VSTR8 + 2 * kp];
                #pragma unroll
                for (int b = 0; b < 4; ++b) {
                    // Byte b of the first key's word low, of the second key's word in the high half:
                    // perm selects source bytes 0-3 from its second operand, 4-7 from its first,
                    // and 0x0c is a zero byte.
                    const uint32_t sel = (uint32_t)b | 0x0c00u | ((uint32_t)(4 + b) << 16) | 0x0c000000u;
                    const uint32_t lo = __builtin_amdgcn_perm(n1, n0, sel) | 0x64006400u;   // dim 2b
                    const uint32_t hi = __builtin_amdgcn_perm(m1, m0, sel) | 0x64006400u;   // dim 2b+1
                    const h2 vl = (__builtin_bit_cast(h2, lo) - bias) * s2;
                    const h2 vh = (__builtin_bit_cast(h2, hi) - bias) * s2;
                    if constexpr (PV8) {
                        // [dim][key] bytes: the pair's two keys as one 16-bit store a dim
                        *(uint16_t*)(dst8 + (2 * b) * VSTR8)     = (uint16_t)pk_fp8x2_h(__builtin_bit_cast(uint32_t, vl));
                        *(uint16_t*)(dst8 + (2 * b + 1) * VSTR8) = (uint16_t)pk_fp8x2_h(__builtin_bit_cast(uint32_t, vh));
                        continue;
                    }
                    // The octet's rows: nibbles interleave the dims (2b, 2b+1), bytes run b, 4+b.
                    *(uint32_t*)(dst + (VI8 ? b : 2 * b) * VSTR)         = __builtin_bit_cast(uint32_t, vl);
                    *(uint32_t*)(dst + (VI8 ? 4 + b : 2 * b + 1) * VSTR) = __builtin_bit_cast(uint32_t, vh);
                }
            }
        }
    };

    auto fetchK = [&](int k0, const int* blks) {
        #pragma unroll
        for (int p = 0; p < KPASS; ++p) {
            const int i = tid + p * NTHR;
            if (KITEMS % NTHR == 0 || i < KITEMS) {
                const int key = KWIDE ? (2 * (i >> 5) + (i & 1)) : (i / (HEAD_DIM / KEL));
                const int ch  = KWIDE ? ((i & 31) >> 1) : (i - key * (HEAD_DIM / KEL));
                if (KWIDE && key >= TILE) continue;
                const int kk  = min(k0 + key, ctx - 1);
                const int blk = BTS ? blks[(kk - k0) / BS] : bt[kk / BS];
                const size_t o = (size_t)blk * a.kv_block_stride
                               + (size_t)kvh * a.kv_head_stride + (size_t)(kk % BS) * ROWB;
                if constexpr (KBF) {
                    const auto* row = (const uint8_t*)a.kv + o * (KVP ? 2 : 1);
                    const uint16_t* g = (const uint16_t*)row + ch * KEL;
                    #pragma unroll
                    for (int q = 0; q < KEL / 8; ++q)
                        kreg[p * (KEL / 8) + q] = *(const uint4*)(g + 8 * q);
                } else if (I8) {
                    // i8: the chunk's codes (sixteen, or eight without the wide fetch) and the
                    // key's scale.
                    const uint8_t* rp = (const uint8_t*)a.kv + o;
                    if constexpr (KWIDE) {
                        kreg[p] = *(const uint4*)(rp + ch * 16);
                    } else {
                        const uint2 r = *(const uint2*)(rp + ch * 8);
                        kreg[p] = make_uint4(r.x, r.y, 0, 0);
                    }
                    ksc8[p] = *(const uint16_t*)(rp + KSOFF);
                } else if (K6) {
                    // k6_v4: the chunk's sixteen low nibbles, its high-bit word, its group's scale.
                    const uint8_t* rp = (const uint8_t*)a.kv + o;
                    const uint2 lo = *(const uint2*)(rp + ch * 8);
                    kreg[p] = make_uint4(lo.x, lo.y, *(const uint32_t*)(rp + HEAD_DIM / 2 + ch * 4),
                                         *(const uint16_t*)(rp + KSOFF + (ch >> 1) * 2));
                } else if (KWIDE) {
                    kreg[p] = *(const uint4*)((const uint8_t*)a.kv + o + ch * 16);
                } else {
                    const uint2 r = *(const uint2*)((const uint8_t*)a.kv + o + ch * 8);
                    kreg[p] = make_uint4(r.x, r.y, 0, 0);
                }
            }
        }
    };

    auto fetchV = [&](int k0, const int* blks) {
        #pragma unroll
        for (int r = 0; r < VREGS; ++r) {
            const int idx = warp + r * NWARPS;
            if (NTR % NWARPS == 0 || idx < NTR) {
                constexpr int DG = HEAD_DIM / (KVP ? 32 : 64);
                const int kg = idx / DG, dg = idx - kg * DG;
                const int kk = min(k0 + kg * 8 + vj, ctx - 1);
                const int d0 = dg * (KVP ? 32 : 64) + vg * (KVP ? 8 : 16);
                const int blk = BTS ? blks[(kk - k0) / BS] : bt[kk / BS];
                const size_t o = (size_t)blk * a.kv_block_stride
                               + (size_t)kvh * a.kv_head_stride
                               + (size_t)(kk % BS) * ROWB + VBOFF + d0;
                const void* p = KVP ? (const void*)((const uint16_t*)a.kv + o)
                                    : (const void*)((const uint8_t*)a.kv + o);
                vreg[r] = __builtin_bit_cast(uint4, load_tr_b128(p));
            }
        }
    };

    auto storeK = [&]() {
        #pragma unroll
        for (int p = 0; p < KPASS; ++p) {
            const int i = tid + p * NTHR;
            if (KITEMS % NTHR == 0 || i < KITEMS) {
                const int key = KWIDE ? (2 * (i >> 5) + (i & 1)) : (i / (HEAD_DIM / KEL));
                const int ch  = KWIDE ? ((i & 31) >> 1) : (i - key * (HEAD_DIM / KEL));
                if (KWIDE && key >= TILE) continue;
                if constexpr (QK8 && K6) {
                    // k6_v4: widen to f16 as below, then round each value once to e4m3
                    const uint4 r = kreg[p];
                    #pragma unroll
                    for (int q = 0; q < 2; ++q) {
                        uint32_t w4[4];
                        k6x8_to_f16x4w(q ? r.y : r.x, r.z, q, r.w | (r.w << 16), w4);
                        *(uint2*)(&sK8[key * KSTR8 + ch * KEL + 8 * q]) =
                            make_uint2(pk_fp8x4_h(w4[0], w4[1]), pk_fp8x4_h(w4[2], w4[3]));
                    }
                    continue;
                } else if constexpr (QK8) {
                    *(uint4*)(&sK8[key * KSTR8 + ch * KEL]) = kreg[p];
                    continue;
                }
                uint16_t* kd = &sK[key * KSTR + ch * KEL];
                #pragma unroll
                for (int q = 0; q < KEL / 8; ++q) {
                    uint32_t w4[4];
                    if constexpr (KBF) {
                        const uint4 r = kreg[p * (KEL / 8) + q];
                        w4[0] = D::from_bf16w(r.x); w4[1] = D::from_bf16w(r.y);
                        w4[2] = D::from_bf16w(r.z); w4[3] = D::from_bf16w(r.w);
                    } else {
                        const uint4 r = kreg[p];
                        if constexpr (I8)
                            i8x8_to_f16x4w(q ? make_uint2(r.z, r.w) : make_uint2(r.x, r.y),
                                           ksc8[p] | (ksc8[p] << 16), w4);
                        else if constexpr (K6) k6x8_to_f16x4w(q ? r.y : r.x, r.z, q, r.w | (r.w << 16), w4);
                        else fp8x8_to_16x4w<D>(q ? make_uint2(r.z, r.w) : make_uint2(r.x, r.y), w4);
                    }
                    *(uint2*)(kd + 8 * q)     = make_uint2(w4[0], w4[1]);
                    *(uint2*)(kd + 8 * q + 4) = make_uint2(w4[2], w4[3]);
                }
            }
        }
    };

    auto storeV = [&]() {
        #pragma unroll
        for (int r = 0; r < VREGS; ++r) {
            const int idx = warp + r * NWARPS;
            if (NTR % NWARPS == 0 || idx < NTR) {
                constexpr int DG = HEAD_DIM / (KVP ? 32 : 64);
                const int kg = idx / DG, dg = idx - kg * DG;
                const int d0 = dg * (KVP ? 32 : 64) + vg * (KVP ? 8 : 16);
                const uint4 v = vreg[r];
                if (KVP) {
                    uint16_t* dst = &sV16[(d0 + vj) * VSTR + kg * 8];
                    *(uint2*)(dst)     = make_uint2(D::from_bf16w(v.x), D::from_bf16w(v.y));
                    *(uint2*)(dst + 4) = make_uint2(D::from_bf16w(v.z), D::from_bf16w(v.w));
                } else {
                    // tr_b128 transposes 16-bit lanes, so each short holds two fp8 from adjacent
                    // dims; v_perm_b32 splits them into two dim rows.
                    const uint32_t lo0 = byte_gather(v.x, v.y, 0), lo1 = byte_gather(v.z, v.w, 0);
                    const uint32_t hi0 = byte_gather(v.x, v.y, 1), hi1 = byte_gather(v.z, v.w, 1);
                    if constexpr (PV8) {
                        *(uint2*)(&sV8[(d0 + 2 * vj)     * VSTR8 + kg * 8]) = make_uint2(lo0, lo1);
                        *(uint2*)(&sV8[(d0 + 2 * vj + 1) * VSTR8 + kg * 8]) = make_uint2(hi0, hi1);
                        continue;
                    }
                    uint32_t wl[4], wh[4];
                    fp8x8_to_16x4w<D>(make_uint2(lo0, lo1), wl);
                    fp8x8_to_16x4w<D>(make_uint2(hi0, hi1), wh);
                    uint16_t* a0 = &sV16[(d0 + 2 * vj)     * VSTR + kg * 8];
                    uint16_t* a1 = &sV16[(d0 + 2 * vj + 1) * VSTR + kg * 8];
                    *(uint2*)(a0)     = make_uint2(wl[0], wl[1]);
                    *(uint2*)(a0 + 4) = make_uint2(wl[2], wl[3]);
                    *(uint2*)(a1)     = make_uint2(wh[0], wh[1]);
                    *(uint2*)(a1 + 4) = make_uint2(wh[2], wh[3]);
                }
            }
        }
    };

    // Tile ti's block-table entries are loaded during tile ti-1, off the KV address chain.
    // A multi-block tile can run past the sequence's last block (on the last row, past the end of
    // the table). Keys are clamped to ctx - 1, so entries past lastb are never used; only tiles
    // reaching past lastb take the clamped load. With NB == 1 the test folds away.
    const int lastb = (ctx - 1) / BS;
    auto load_bt = [&](int t0, int* out) {
        if (NB > 1 && t0 + NB - 1 > lastb) {
            #pragma unroll
            for (int i = 0; i < NB; ++i) out[i] = bt[min(t0 + i, lastb)];
        } else {
            #pragma unroll
            for (int i = 0; i < NB; ++i) out[i] = bt[t0 + i];
        }
    };
    int blk[NB];
    if (BTS) load_bt(tstart * NB, blk);
    if (GPREV) fetchV(tstart * TILE, blk);
    if (VSTAGE) fetchV4(tstart * TILE, blk);
    if (KLATE) fetchK(tstart * TILE, blk);

    for (int ti = tstart; ti < ntiles; ++ti) {
        const int k0 = ti * TILE;
        int blkn[NB];
        if (BTS && ti + 1 < ntiles) {
            load_bt((ti + 1) * NB, blkn);
        }
        if (LDSB) lds_barrier(); else __syncthreads();
        if (!KLATE) fetchK(k0, blk);
        // V first: already in registers.
        if constexpr (VSTAGE) { storeV4(); } else { if (!GPREV) fetchV(k0, blk); storeV(); }
        storeK();
        if (LDSB) lds_barrier(); else __syncthreads();
        // Issue the next tile's V loads before the WMMAs below; they are consumed after the next
        // barrier.
        if (GPREV && ti + 1 < ntiles) fetchV(k0 + TILE, blkn);
        if (VSTAGE && ti + 1 < ntiles) fetchV4(k0 + TILE, blkn);

        #pragma unroll 1
        for (int mt = 0; mt < MT; ++mt) {
            v8f s = (v8f){0,0,0,0,0,0,0,0};
            if constexpr (QK8) {
                const uint8_t* krow8 = &sK8[(mt * 16 + c) * KSTR8] + 8 * h;
                #pragma unroll
                for (int g = 0; g < NKS; g += PF) {
                    v2i32_r4d kb8[PF];
                    #pragma unroll
                    for (int j = 0; j < PF; ++j)
                        kb8[j] = *(const v2i32_r4d*)(krow8 + 16 * (g + j));
                    if (SGB) sched_barrier();
                    #pragma unroll
                    for (int j = 0; j < PF; ++j)
                        s = __builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(kb8[j], qf8[g + j], s);
                }
            } else {
                const uint16_t* krow = &sK[(mt * 16 + c) * KSTR] + (CTG ? 8 * h : 0);
                // Load PF fragments before issuing their WMMAs, so each WMMA waits on
                // s_wait_dscnt PF-1 instead of a full drain.
                #pragma unroll
                for (int g = 0; g < NKS; g += PF) {
                    frag16 kb[PF];
                    #pragma unroll
                    for (int j = 0; j < PF; ++j) {
                        const uint16_t* kp = krow + 16 * (g + j);
                        if (CTG) {
                            const uint4 u = *(const uint4*)kp;
                            kb[j] = D::mk(make_uint2(u.x, u.y), make_uint2(u.z, u.w));
                        } else {
                            kb[j] = D::mk(*(const uint2*)(kp + 4 * h),
                                          *(const uint2*)(kp + 8 + 4 * h));
                        }
                    }
                    if (SGB) sched_barrier();
                    #pragma unroll
                    for (int j = 0; j < PF; ++j) s = D::wmma(kb[j], qf[g + j], s);
                }
            }
            if (KLATE && mt == MT - 1 && ti + 1 < ntiles) fetchK(k0 + TILE, blkn);
            float sc[8];
            #pragma unroll
            for (int e = 0; e < 8; ++e) sc[e] = FOLDQ ? s[e] : s[e] * mul;

            const int kbase = k0 + mt * 16 + 8 * h;
            // Mask keys past klimit and, with WIN, before wlo. MSKIP skips tiles straddling neither.
            if (!MSKIP || wave_any(kbase + 7 > klimit) || (WIN && wave_any(kbase < wlo))) {
                #pragma unroll
                for (int e = 0; e < 8; ++e) sc[e] = ((kbase + e <= klimit) && (!WIN || kbase + e >= wlo))
                                                    ? sc[e] : -INFINITY;
            }

            float smax = sc[0];
            #pragma unroll
            for (int e = 1; e < 8; ++e) smax = fmaxf(smax, sc[e]);
            // Both half-waves hold the same query and different keys. The branch must be
            // wave-uniform, but each query may change scale only on its own growth threshold;
            // otherwise other live query rows change its rounding when the chunk length changes.
            const float rmax = fmaxf(smax, swap16(smax));
            const bool row_grow = live && rmax > m_ref + PGROW;
            if (wave_any(row_grow)) {
                const float mnew = row_grow ? fmaxf(fmaxf(m_ref, rmax), -3.0e38f) : m_ref;
                const float alpha = ex2(m_ref - mnew);
                #pragma unroll
                for (int dt = 0; dt < NKS; ++dt)
                    #pragma unroll
                    for (int e = 0; e < 8; ++e) acc[dt][e] *= alpha;
                l_i *= alpha;
                m_ref = mnew;
                off   = PSHIFT - mnew;
            }

            float pf[8], lsum = 0.0f;
            #pragma unroll
            for (int e = 0; e < 8; ++e) {
                const float p = ex2(sc[e] + off);
                pf[e] = p;
                if (!DOT2) lsum += p;
            }
            if (!DOT2) l_i += lsum;

            if constexpr (PV8) {
                const v2i32_r4d p8 = (v2i32_r4d){(int)pk_fp8x4(pf[0], pf[1], pf[2], pf[3]),
                                                 (int)pk_fp8x4(pf[4], pf[5], pf[6], pf[7])};
                const uint8_t* vbase8 = &sV8[c * VSTR8 + mt * 16 + 8 * h];
                #pragma unroll
                for (int g = 0; g < NKS; g += PF) {
                    v2i32_r4d vb8[PF];
                    #pragma unroll
                    for (int j = 0; j < PF; ++j)
                        vb8[j] = *(const v2i32_r4d*)(vbase8 + (size_t)(g + j) * 16 * VSTR8);
                    if (SGB) sched_barrier();
                    #pragma unroll
                    for (int j = 0; j < PF; ++j)
                        acc[g + j] = __builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(vb8[j], p8, acc[g + j]);
                }
            } else {
                uint32_t pw[4];
                #pragma unroll
                for (int e = 0; e < 8; e += 2) pw[e >> 1] = D::pk(pf[e], pf[e + 1]);
                if (DOT2) {
                    const v2hf one = {(__fp16)1.0f, (__fp16)1.0f};
                    #pragma unroll
                    for (int e = 0; e < 4; ++e)
                        l_i = __builtin_amdgcn_fdot2(__builtin_bit_cast(v2hf, pw[e]), one, l_i, false);
                }
                const uint2 lo  = make_uint2(pw[0], pw[1]);
                const uint2 hi2 = make_uint2(pw[2], pw[3]);
                // QK leaves lane (c,h) holding keys 8h+e. Under CONTIG that is already the B
                // fragment; the split map needs a cross-half exchange.
                frag16 p16;
                if (CTG) {
                    p16 = D::mk(lo, hi2);
                } else {
                    const uint2 y = swap16_u2p((h == 0) ? hi2 : lo);
                    p16 = (h == 0) ? D::mk(lo, y) : D::mk(y, hi2);
                }
                const uint16_t* vbase = &sV16[c * VSTR + mt * 16] + (CTG ? 8 * h : 0);
                #pragma unroll
                for (int g = 0; g < NKS; g += PF) {
                    frag16 vb[PF];
                    #pragma unroll
                    for (int j = 0; j < PF; ++j) {
                        const uint16_t* vp = vbase + (g + j) * 16 * VSTR;
                        if (CTG) {
                            const uint4 u = *(const uint4*)vp;
                            vb[j] = D::mk(make_uint2(u.x, u.y), make_uint2(u.z, u.w));
                        } else {
                            vb[j] = D::mk(*(const uint2*)(vp + 4 * h),
                                          *(const uint2*)(vp + 8 + 4 * h));
                        }
                    }
                    if (SGB) sched_barrier();
                    #pragma unroll
                    for (int j = 0; j < PF; ++j) acc[g + j] = D::wmma(vb[j], p16, acc[g + j]);
                }
            }
        }
        if (BTS) {
            #pragma unroll
            for (int i = 0; i < NB; ++i) blk[i] = blkn[i];
        }
    }

    if (!live) return;
    // fetch-inactive: the split store reads l_i under `if (h == 0)` only, so the exchange may be sunk
    // into that half; swap16_fi still reads the partner's half of the row sum.
    l_i += swap16_fi(l_i);
    if constexpr (SPLIT) {
        // Unnormalised: the merge weighs segments by 2^(m - max m) and applies v_descale itself.
        r4d_prefill_store_partial<NKS, HEAD_DIM>(a, qoff + qpos, qhead, sp, acc, m_ref, l_i, h);
        return;
    }
    const float inv = (l_i > 0.0f) ? (vdesc / l_i) : 0.0f;
    uint16_t* op = (uint16_t*)a.out
                 + ((size_t)(qoff + qpos) * a.q_heads + qhead) * HEAD_DIM;
    #pragma unroll
    for (int dt = 0; dt < NKS; ++dt) {
        uint32_t tw[4];
        #pragma unroll
        for (int e = 0; e < 8; e += 2)
            tw[e >> 1] = (uint32_t)f32_to_bf16(acc[dt][e] * inv)
                       | ((uint32_t)f32_to_bf16(acc[dt][e + 1] * inv) << 16);
        *(uint4*)(op + dt * 16 + 8 * h) = make_uint4(tw[0], tw[1], tw[2], tw[3]);
    }
}
