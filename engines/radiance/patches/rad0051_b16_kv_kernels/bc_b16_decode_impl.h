/* rad0051-owned typed decode kernels, extracted from the frozen 2.2 ABI44 source. */
#pragma once
#include "r4d_b16_kv.h"

template<int NWARPS, int TILE, int HEAD_DIM, int GQA, int BS, int KVP, int OPT,
         int WIN = 0, RadKVLayout Layout = RAD_KVQ_NONE>
__device__ __forceinline__
void r4d_attn_decode_body(const R4DArgs& a, int splits)
{
    constexpr int F16   = (OPT & O_F16)   ? 1 : 0;
    constexpr int MSKIP = (OPT & O_MSKIP) ? 1 : 0;
    constexpr int FOLDQ = (OPT & O_FOLDQ) ? 1 : 0;
    constexpr int KLDS  = (OPT & O_KLDS0) ? 0 : 1;
    constexpr int PF16  = (OPT & O_PF16)  ? 1 : 0;
    constexpr int CTG   = (OPT & O_CONTIG) ? 1 : 0;
    constexpr int PF    = 1 << ((OPT >> 10) & 3);
    constexpr int SGB   = (OPT & O_SGB) ? 1 : 0;
    constexpr int DOT2  = ((OPT & O_DOT2) && (OPT & O_F16)) ? 1 : 0;
    constexpr int BTS   = (OPT & O_BTS)  ? 1 : 0;
    constexpr int LDSB  = (OPT & O_LDSB) ? 1 : 0;
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
    // A token's row in a page, and where in it V's codes, V's scales and K's scales start.
    constexpr int ROWB  = KB ? HEAD_DIM / 16 * R4dKvBytes<Layout>::row16
                        : I8 ? 2 * HEAD_DIM + 4
                        : K6 ? HEAD_DIM * 22 / 16 : V4 ? HEAD_DIM * 25 / 16 : 2 * HEAD_DIM;
    constexpr int VBOFF = KB ? HEAD_DIM / 16 * R4dKvBytes<Layout>::value16 : HEAD_DIM;
    constexpr int VISOFF = HEAD_DIM / 16 * R4dKvBytes<Layout>::value_scale16;
    constexpr int VOFF  = I8 ? HEAD_DIM : K6 ? HEAD_DIM * 3 / 4 : HEAD_DIM;
    constexpr int VSOFF = I8 ? 2 * HEAD_DIM + 2 : K6 ? HEAD_DIM * 21 / 16 : HEAD_DIM * 3 / 2;
    constexpr int KSOFF = I8 ? 2 * HEAD_DIM : HEAD_DIM * 5 / 4;
    constexpr int V8    = ((OPT & O_V8) && !KVP && !VSTAGE) ? 1 : 0;
    static_assert(!V8 || (OPT & O_CONTIG), "the code tile is read in the contiguous layout");
    constexpr int KRAW  = ((OPT & O_KRAW) && !KLDS) ? 1 : 0;
    static_assert(!KRAW || (OPT & O_CONTIG), "the raw K round reads the contiguous layout");
    static_assert(!K6 || (!KLDS && (OPT & O_CONTIG) && !KRAW),
                  "k6_v4 K is widened from the paged cache, a lane's eight dims contiguous");
    static_assert(!V4 || (OPT & O_F16), "fp8_v4 widens to f16");
    constexpr int QLD   = (OPT & O_QLD)   ? 1 : 0;
    constexpr int KVREP = (OPT & O_KVREP) ? 1 : 0;
    constexpr int TREE  = (OPT & O_TREE)  ? 1 : 0;
    typedef DT16<F16> D;
    typedef typename D::frag frag16;

    constexpr int NKS   = HEAD_DIM / 16;
    constexpr int MT    = TILE / 16;
    constexpr int KSTR  = HEAD_DIM + (CTG ? 8 : 4);
    constexpr int VSTR  = TILE + (CTG ? 8 : 4);
    constexpr int NTHR  = NWARPS * 32;
    constexpr int NB    = TILE / BS;
    constexpr int NTR   = (TILE / 8) * (HEAD_DIM / (KVP ? 32 : 64));
    constexpr int VREGS = (NTR + NWARPS - 1) / NWARPS;
    // A V prefetch costs VREGS x 4 VGPRs on top of a 128-VGPR accumulator and a 64-VGPR query.
    // Up to three uint4 it fits in 255 VGPRs without spilling (two: +6-8% at depth; three, the
    // 48-row shape q_len 8 x gqa 6 at NWARPS 3: a further 2-15%). At four or more (NWARPS 1-2, or a
    // bf16 cache) it spills and loses ~10%, so the geometry gates it, not just the flag.
    constexpr int GPREV = ((OPT & O_GPREV) && VREGS <= 3 && !VSTAGE) ? 1 : 0;
    constexpr float PSHIFT = D::SHIFT;
    constexpr float PGROW  = D::GROW;

    __shared__ uint16_t sK[KLDS ? TILE * KSTR : 1];
    // V4 at TILE 16: dim octet o's 8 rows start S(o) = 16 (o & 3) + 8 ((o >> 2) & 1) + 128 (o >> 2)
    // halfwords later, so the 4 (8) octets of a storeV4 store land on distinct banks (unskewed they
    // share them: 8 rows = 96 dwords = 0 mod 32). The 128 (o >> 2) term (64 dwords, unseen by the
    // banks) keeps octet o's last row clear of octet o + 1's first; multiples of 8 halfwords keep the
    // b128 PV read aligned. NWARPS 1 keeps the plain tile: its 10 workgroups a WGP would not fit.
    constexpr int VSK  = (VSTAGE && TILE == 16 && CTG && NWARPS >= 2) ? 1 : 0;
    constexpr int VSKN = HEAD_DIM / 8 - 1;
    constexpr int VSKX = VSK ? 16 * (VSKN & 3) + 8 * ((VSKN >> 2) & 1) + 128 * (VSKN >> 2) : 0;
    __shared__ uint16_t sV16[V8 ? 1 : HEAD_DIM * VSTR + VSKX];
    /* A code-tile row is 16 keys + 8 padding = 24 bytes (six dwords): the sixteen lanes of a
     * fragment read hit distinct banks, and a lane's eight codes are one aligned 8-byte load. */
    constexpr int VSTR8 = TILE + 8;
    __shared__ uint8_t sV8[V8 ? HEAD_DIM * VSTR8 : 1];

    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    // The launch is NTHR threads. Stated (k6_v4), so the V4 rounds wholly below NP4 lose their range
    // test: as exec-masked blocks, each round's loads issued only after the previous round's returned.
    // fp8_v4 keeps the tested rounds: batched there, its 4-row steps ran 6 % slower.
    if constexpr (K6 || KI) __builtin_assume(tid < NTHR);
    const int c = lane & 15, h = lane >> 4;

    const int sp = blockIdx.x, kvh = blockIdx.y, seq = blockIdx.z;
    // kvh is the query group; under KVREP the cache head it reads is kvh / kv_rep
    const int kvc = KVREP ? kvh / a.kv_rep : kvh;
    const int ctx = a.seqused_k[seq];
    if (ctx <= 0) return;

    // Ragged batches (r4d.h, R4DArgs::cu_q): this sequence's q/out rows and query length. With
    // cu_q, a.q_len is the batch maximum that sized the wave count; rows past qlen are not live.
    const int qoff = r4d_q_off(a.cu_q, seq, a.q_len);
    const int qlen = r4d_q_len(a.cu_q, seq, a.q_len);
    if (qlen <= 0) return;                 // a slot carrying no query rows this step

    const int ntl  = r4d_attn_tiles(ctx, TILE);
    // A sliding-window layer must not read below the window: those blocks may already be freed
    // and reassigned to another sequence. The walk starts at the tile holding row 0's first
    // window token (row 0 reaches furthest back), and the splits divide that range.
    const int wlo0 = WIN ? r4d_attn_wlo(ctx, qlen, 0, a.window) : 0;
    const int tb   = WIN ? wlo0 / TILE : 0;
    const int neff = ntl - tb;
    const int tps  = (neff + splits - 1) / splits;
    const int t_lo = tb + sp * tps;
    if (t_lo >= ntl) return;
    const int t_hi = min(t_lo + tps, ntl);

    // TREE: the block's parents staged once, so each row's walk below reads LDS rather than a
    // chain of dependent global loads. Every early return above is uniform across the block.
    __shared__ int s_tpar[TREE ? 64 : 1];
    if constexpr (TREE) {
        if (a.spec_parent) {
            for (int r = tid; r < qlen && r < 64; r += NTHR) s_tpar[r] = a.spec_parent[qoff + r];
            __syncthreads();
        }
    }

    const int row = warp * 16 + c;
    const int qi = row / GQA, hi = row - qi * GQA;
    const bool live = (qi < qlen);
    const int qrow = live ? qi : (qlen - 1);
    const int qhead = kvh * GQA + hi;

    const int* __restrict__ bt = a.block_table + (size_t)seq * a.max_blocks;
    const float kdesc = a.k_descale
                      ? a.k_descale[seq * (KVREP ? a.kv_heads / a.kv_rep : a.kv_heads) + kvc]
                      : 1.0f;
    const float mul   = a.scale * kdesc * 1.44269504089f;

    frag16 qf[NKS];
    {
        const uint16_t* qp = QLD ? (const uint16_t*)a.q + (size_t)(qoff + qrow) * a.q_ld
                                                        + (size_t)qhead * HEAD_DIM
                                 : (const uint16_t*)a.q
                                   + ((size_t)(qoff + qrow) * a.q_heads + qhead) * HEAD_DIM;
        #pragma unroll
        for (int t = 0; t < NKS; ++t) {
            const uint2 l0 = *(const uint2*)(qp + 16 * t + (CTG ? 8 * h : 4 * h));
            const uint2 h0 = *(const uint2*)(qp + 16 * t + (CTG ? 8 * h + 4 : 8 + 4 * h));
            qf[t] = D::mk(make_uint2(r4d_qcvt<D, FOLDQ, F16>(l0.x, mul),
                                     r4d_qcvt<D, FOLDQ, F16>(l0.y, mul)),
                          make_uint2(r4d_qcvt<D, FOLDQ, F16>(h0.x, mul),
                                     r4d_qcvt<D, FOLDQ, F16>(h0.y, mul)));
        }
    }

    // The row's absolute position. Causal masking bounds the key by it; non-causal masking
    // bounds it by the sequence, with the window applied on both sides (a block-diffusion
    // drafter's own block).
    const int pos = ctx - qlen + qrow;
    const int cz  = a.causal_seq ? (int)((const unsigned char*)a.causal_seq)[seq] : a.causal;
    const int klimit = cz ? pos
                          : ((WIN && a.window > 0) ? min(ctx - 1, pos + a.window - 1)
                                                   : ctx - 1);
    // A tree verify's row (TREE): a bit per row of the block for itself and each ancestor, and
    // its depth, from which its window is measured. Parents precede children, so the walk ends.
    const int tbase = ctx - qlen;
    uint64_t tmask = 0;
    int tdep = 0;
    bool trow = false;
    if (TREE && a.spec_parent && qrow < 64) {
        int pj = s_tpar[qrow];
        if (pj != -2) {
            trow  = true;
            tmask = 1ull << qrow;
            for (int at = qrow; pj >= 0 && pj < at; ) {
                tmask |= 1ull << pj;
                ++tdep;
                at = pj;
                pj = s_tpar[pj];
            }
        }
    }
    const int wpos   = (TREE && trow) ? tbase + tdep : pos;
    const int wlo    = (WIN && a.window > 0) ? (wpos - a.window + 1 > 0
                                               ? wpos - a.window + 1 : 0) : 0;
    float m_ref = -3.0e38f;
    float off   = PSHIFT + 3.0e38f;
    float l_i   = 0.0f;
    v8f acc[NKS];
    #pragma unroll
    for (int t = 0; t < NKS; ++t) acc[t] = (v8f){0,0,0,0,0,0,0,0};

    const int vg = lane >> 3, vj = lane & 7;
    uint4 vreg[VREGS];

    // ---- fp8_v4 / k6_v4: V's int4 codes into sV16 ([dim][key]) as the prefill kernel stages them
    // (r4d_attn_prefill_h256_gqa6.hip: a key pair's 8 dims a thread, two code words and the pair's
    // scale, widened with packed f16 into one dword a dim), fetched in the tile: no prefetch.
    constexpr int NP4 = (TILE / 2) * (HEAD_DIM / 8);
    constexpr int U4  = VSTAGE ? (NP4 + NTHR - 1) / NTHR : 1;
    // The pair's two scales stay apart until storeV4 packs them: packed here, the fetch is waited on
    // right after it is issued. fp8_v4 packs here, as the prefill kernel does.
    constexpr int OLDV4 = V4 && !K6;
    // i8: a key's 8 dims are two code words, so a pair's are four.
    uint32_t v4c[U4][VI8 ? 4 : 2], v4s[U4][2];
    auto fetchV4 = [&](int k0, const int* blks) {
        #pragma unroll
        for (int r = 0; r < U4; ++r) {
            const int u = tid + r * NTHR;
            // k6_v4, a partial round (NWARPS 3: the third, warps 0-1): u < NP4 <=> warp < (NP4 - r NTHR)
            // / 32 when NP4 is a multiple of 32, tested on the wave-uniform warp: a scalar branch.
            if (NP4 % NTHR == 0 || ((K6 || KI) ? ((r + 1) * NTHR <= NP4
                || (NP4 % 32 == 0 ? __builtin_amdgcn_readfirstlane(warp) * 32 < NP4 - r * NTHR
                                  : u < NP4)) : u < NP4)) {
                const int kp = u % (TILE / 2), o = u / (TILE / 2);
                uint32_t sc[2];
                #pragma unroll
                for (int j = 0; j < 2; ++j) {
                    int kk = min(k0 + 2 * kp + j, ctx - 1);
                    if (WIN && kk < wlo0) kk = wlo0;
                    const int bk = BTS ? blks[(kk - k0) / BS] : bt[kk / BS];
                    const uint8_t* base = (const uint8_t*)a.kv + (size_t)bk * a.kv_block_stride
                                        + (size_t)kvc * a.kv_head_stride + (size_t)(kk % BS) * ROWB;
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
            // k6_v4, a partial round (NWARPS 3: the third, warps 0-1): u < NP4 <=> warp < (NP4 - r NTHR)
            // / 32 when NP4 is a multiple of 32, tested on the wave-uniform warp: a scalar branch.
            if (NP4 % NTHR == 0 || ((K6 || KI) ? ((r + 1) * NTHR <= NP4
                || (NP4 % 32 == 0 ? __builtin_amdgcn_readfirstlane(warp) * 32 < NP4 - r * NTHR
                                  : u < NP4)) : u < NP4)) {
                const int kp = u % (TILE / 2), o = u / (TILE / 2);
                // Two's complement nibbles; ^ 8 makes them offset binary (value + 8). i8: bytes,
                // ^ 0x80 (value + 128); `n` is a key's dims 0-3 of the octet and `m` its dims 4-7.
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
                uint16_t* dst = &sV16[(o * 8) * VSTR + 2 * kp
                                      + (VSK ? 16 * (o & 3) + 8 * ((o >> 2) & 1) + 128 * (o >> 2) : 0)];
                #pragma unroll
                for (int b = 0; b < 4; ++b) {
                    const uint32_t sel = (uint32_t)b | 0x0c00u | ((uint32_t)(4 + b) << 16) | 0x0c000000u;
                    const uint32_t lo = __builtin_amdgcn_perm(n1, n0, sel) | 0x64006400u;
                    const uint32_t hi = __builtin_amdgcn_perm(m1, m0, sel) | 0x64006400u;
                    const h2 vl = (__builtin_bit_cast(h2, lo) - bias) * s2;
                    const h2 vh = (__builtin_bit_cast(h2, hi) - bias) * s2;
                    // The octet's rows: nibbles interleave the dims (2b, 2b+1), bytes run b, 4+b.
                    *(uint32_t*)(dst + (VI8 ? b : 2 * b) * VSTR)         = __builtin_bit_cast(uint32_t, vl);
                    *(uint32_t*)(dst + (VI8 ? 4 + b : 2 * b + 1) * VSTR) = __builtin_bit_cast(uint32_t, vh);
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
                int kk = min(k0 + kg * 8 + vj, ctx - 1);
                if (WIN && kk < wlo0) kk = wlo0;
                const int d0 = dg * (KVP ? 32 : 64) + vg * (KVP ? 8 : 16);
                const int blk = BTS ? blks[(kk - k0) / BS] : bt[kk / BS];
                const size_t o = (size_t)blk * a.kv_block_stride
                               + (size_t)kvc * a.kv_head_stride
                               + (size_t)(kk % BS) * ROWB + VBOFF + d0;
                const void* p = KVP ? (const void*)((const uint16_t*)a.kv + o)
                                    : (const void*)((const uint8_t*)a.kv + o);
                vreg[r] = __builtin_bit_cast(uint4, load_tr_b128(p));
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
                } else if (V8) {
                    const uint32_t lo0 = byte_gather(v.x, v.y, 0), lo1 = byte_gather(v.z, v.w, 0);
                    const uint32_t hi0 = byte_gather(v.x, v.y, 1), hi1 = byte_gather(v.z, v.w, 1);
                    *(uint2*)&sV8[(d0 + 2 * vj)     * VSTR8 + kg * 8] = make_uint2(lo0, lo1);
                    *(uint2*)&sV8[(d0 + 2 * vj + 1) * VSTR8 + kg * 8] = make_uint2(hi0, hi1);
                } else {
                    const uint32_t lo0 = byte_gather(v.x, v.y, 0), lo1 = byte_gather(v.z, v.w, 0);
                    const uint32_t hi0 = byte_gather(v.x, v.y, 1), hi1 = byte_gather(v.z, v.w, 1);
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

    // A tile's NB entries are read whole, and a multi-block tile can run past the sequence's last
    // block (on the last row, past the end of the table). Keys are clamped to ctx - 1, so entries
    // past lastb are never used; only tiles reaching past lastb take the clamped load, and the
    // rest keep one wide scalar load. With NB == 1 the test folds away.
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
    if (BTS) load_bt(t_lo * NB, blk);
    if (GPREV) fetchV(t_lo * TILE, blk);

    for (int ti = t_lo; ti < t_hi; ++ti) {
        const int k0 = ti * TILE;
        int blkn[NB];
        if (BTS && ti + 1 < t_hi) {
            load_bt((ti + 1) * NB, blkn);
        }
        if (LDSB) lds_barrier(); else __syncthreads();

        if (KLDS) {
            #pragma unroll 1
            for (int i = tid; i < TILE * (HEAD_DIM / 8); i += NTHR) {
                const int key = i / (HEAD_DIM / 8), ch = i - key * (HEAD_DIM / 8);
                int kk = min(k0 + key, ctx - 1);
                if (WIN && kk < wlo0) kk = wlo0;
                const int kb = BTS ? blk[(kk - k0) / BS] : bt[kk / BS];
                const size_t o = (size_t)kb * a.kv_block_stride
                               + (size_t)kvc * a.kv_head_stride + (size_t)(kk % BS) * ROWB;
                uint16_t* kd = &sK[key * KSTR];
                uint32_t w4[4];
                if constexpr (KBF) {
                    const auto* row = (const uint8_t*)a.kv + o * (KVP ? 2 : 1);
                    const uint4 r = *(const uint4*)((const uint16_t*)row + ch * 8);
                    w4[0] = D::from_bf16w(r.x); w4[1] = D::from_bf16w(r.y);
                    w4[2] = D::from_bf16w(r.z); w4[3] = D::from_bf16w(r.w);
                } else {
                    fp8x8_to_16x4w<D>(*(const uint2*)((const uint8_t*)a.kv + o + ch * 8), w4);
                }
                *(uint2*)(kd + ch * 8)     = make_uint2(w4[0], w4[1]);
                *(uint2*)(kd + ch * 8 + 4) = make_uint2(w4[2], w4[3]);
            }
        }
        if constexpr (VSTAGE) { fetchV4(k0, blk); storeV4(); }
        else { if (!GPREV) fetchV(k0, blk); storeV(); }
        if (LDSB) lds_barrier(); else __syncthreads();
        if (GPREV && ti + 1 < t_hi) fetchV(k0 + TILE, blkn);

        #pragma unroll 1
        for (int mt = 0; mt < MT; ++mt) {
            const uint8_t* grow = nullptr;
            if (!KLDS) {
                int kk = min(k0 + mt * 16 + c, ctx - 1);
                if (WIN && kk < wlo0) kk = wlo0;
                const int kb = BTS ? blk[(kk - k0) / BS] : bt[kk / BS];
                grow = (const uint8_t*)a.kv + ((size_t)kb * a.kv_block_stride
                     + (size_t)kvc * a.kv_head_stride + (size_t)(kk % BS) * ROWB)
                     * (KVP ? 2 : 1);
            }
            // i8: the key's one K scale, read once an m-tile.
            const uint32_t ksb = I8 ? (uint32_t)*(const uint16_t*)(grow + KSOFF) : 0u;
            // A fragment's raw K: 8 elements of 1 byte (fp8) or 2 (bf16).
            typedef typename std::conditional<KBF, uint4, uint2>::type kraw_t;
            kraw_t kraw[KRAW ? NKS : 1];
            if constexpr (KRAW) {
                #pragma unroll
                for (int t = 0; t < NKS; ++t)
                    kraw[t] = *(const kraw_t*)(grow + (16 * t + 8 * h) * (KBF ? 2 : 1));
            }
            v8f s = (v8f){0,0,0,0,0,0,0,0};
            #pragma unroll
            for (int g = 0; g < NKS; g += PF) {
                frag16 kb[PF];
                #pragma unroll
                for (int j = 0; j < PF; ++j) {
                    const int t = g + j;
                    if (KLDS) {
                        const uint16_t* kp = &sK[(mt * 16 + c) * KSTR + 16 * t];
                        if (CTG) {
                            const uint4 u = *(const uint4*)(kp + 8 * h);
                            kb[j] = D::mk(make_uint2(u.x, u.y), make_uint2(u.z, u.w));
                        } else {
                            kb[j] = D::mk(*(const uint2*)(kp + 4 * h), *(const uint2*)(kp + 8 + 4 * h));
                        }
                    } else if constexpr (KBF) {
                        const uint16_t* kp = (const uint16_t*)grow + 16 * t;
                        uint2 l0, h0;
                        if constexpr (KRAW && KBF) {
                            const uint4 u = kraw[t];
                            l0 = make_uint2(u.x, u.y);
                            h0 = make_uint2(u.z, u.w);
                        } else {
                            l0 = *(const uint2*)(kp + (CTG ? 8 * h : 4 * h));
                            h0 = *(const uint2*)(kp + (CTG ? 8 * h + 4 : 8 + 4 * h));
                        }
                        kb[j] = D::mk(make_uint2(D::from_bf16w(l0.x), D::from_bf16w(l0.y)),
                                      make_uint2(D::from_bf16w(h0.x), D::from_bf16w(h0.y)));
                    } else if constexpr (I8) {
                        // i8: this lane's eight dims (16t + 8h ..) as bytes, times the key's scale.
                        uint32_t w4[4];
                        i8x8_to_f16x4w(*(const uint2*)(grow + 16 * t + 8 * h), ksb | (ksb << 16), w4);
                        kb[j] = D::mk(make_uint2(w4[0], w4[1]), make_uint2(w4[2], w4[3]));
                    } else if constexpr (K6) {
                        // k6_v4: this lane's eight dims (16t + 8h ..) from their low nibbles, the
                        // chunk's high-bit word and the group's scale.
                        uint32_t w4[4];
                        const uint32_t sb = *(const uint16_t*)(grow + KSOFF + (t >> 1) * 2);
                        k6x8_to_f16x4w(*(const uint32_t*)(grow + 8 * t + 4 * h),
                                       *(const uint32_t*)(grow + HEAD_DIM / 2 + 4 * t), h,
                                       sb | (sb << 16), w4);
                        kb[j] = D::mk(make_uint2(w4[0], w4[1]), make_uint2(w4[2], w4[3]));
                    } else {
                        const uint8_t* kp = grow + 16 * t;
                        uint32_t w4[4];
                        // CONTIG: one 8-byte load; the split map needs two 4-byte loads.
                        uint2 raw;
                        if constexpr (KRAW && !KBF) raw = kraw[t];
                        else raw = CTG ? *(const uint2*)(kp + 8 * h)
                                       : make_uint2(*(const uint32_t*)(kp + 4 * h),
                                                    *(const uint32_t*)(kp + 8 + 4 * h));
                        fp8x8_to_16x4w<D>(raw, w4);
                        kb[j] = D::mk(make_uint2(w4[0], w4[1]), make_uint2(w4[2], w4[3]));
                    }
                }
                if (SGB) sched_barrier();
                #pragma unroll
                for (int j = 0; j < PF; ++j) s = D::wmma(kb[j], qf[g + j], s);
            }
            float sc[8];
            #pragma unroll
            for (int e = 0; e < 8; ++e) sc[e] = FOLDQ ? s[e] : s[e] * mul;

            const int kbase = k0 + mt * 16 + 8 * h;
            if (!MSKIP || wave_any(kbase + 7 > klimit) || (WIN && wave_any(kbase < wlo)) ||
                (TREE && wave_any(trow && kbase + 7 >= tbase))) {
                #pragma unroll
                for (int e = 0; e < 8; ++e)
                    sc[e] = ((kbase + e <= klimit) && (!WIN || kbase + e >= wlo) &&
                             (!TREE || !trow || kbase + e < tbase ||
                              ((tmask >> (kbase + e - tbase)) & 1ull)))
                          ? sc[e] : -INFINITY;
            }
            float smax = sc[0];
            #pragma unroll
            for (int e = 1; e < 8; ++e) smax = fmaxf(smax, sc[e]);

            if (wave_any(smax > m_ref + PGROW)) {
                const float rmax  = fmaxf(smax, swap16(smax));
                const float mnew  = fmaxf(fmaxf(m_ref, rmax), -3.0e38f);
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
                const float p = ex2(sc[e] + off); pf[e] = p;
                if (!DOT2) lsum += p;
            }
            if (!DOT2) l_i += lsum;

            {
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
                frag16 p16;
                if (CTG) {
                    p16 = D::mk(lo, hi2);          // QK already left the keys lane-local
                } else {
                    const uint2 y = swap16_u2p((h == 0) ? hi2 : lo);
                    p16 = (h == 0) ? D::mk(lo, y) : D::mk(y, hi2);
                }
                // Skewed tile: row 16 G + c is in octet 2 G + (c >> 3), so its S is 16 (c >> 3) here
                // plus 32 (G & 1) + 8 ((G >> 1) & 1) + 128 (G >> 1), a compile-time offset below.
                const uint16_t* vbase = &sV16[c * VSTR + mt * 16] + (CTG ? 8 * h : 0)
                                      + (VSK ? 16 * (c >> 3) : 0);
                #pragma unroll
                for (int g = 0; g < NKS; g += PF) {
                    frag16 vb[PF];
                    #pragma unroll
                    for (int j = 0; j < PF; ++j) {
                        const uint16_t* vp = vbase + (g + j) * 16 * VSTR
                            + (VSK ? 32 * ((g + j) & 1) + 8 * (((g + j) >> 1) & 1) + 128 * ((g + j) >> 1)
                                   : 0);
                        if (V8) {
                            uint32_t w4[4];
                            fp8x8_to_16x4w<D>(*(const uint2*)&sV8[(c + (g + j) * 16) * VSTR8 +
                                                                 mt * 16 + 8 * h], w4);
                            vb[j] = D::mk(make_uint2(w4[0], w4[1]), make_uint2(w4[2], w4[3]));
                        } else if (CTG) {
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
    // fetch-inactive: the f32 partials read lrow under `if (h == 0)` only (see the prefill kernel's
    // epilogue).
    const float lrow = l_i + swap16_fi(l_i);
    const int tok = qoff + qrow;
    const size_t idx = (size_t)(tok * a.q_heads + qhead) * splits + sp;
    const size_t nrow = (size_t)r4d_q_total(a) * a.q_heads * splits;
    if (PF16) {
        // O/l in f16: bounded by max|v|, so no overflow, and the merge weight becomes w*l.
        uint16_t* so = (uint16_t*)a.scratch;
        float* sm = (float*)(so + nrow * HEAD_DIM);
        float* sl = sm + nrow;
        uint16_t* op = so + idx * HEAD_DIM;
        const float inv = (lrow > 0.0f) ? (1.0f / lrow) : 0.0f;
        #pragma unroll
        for (int dt = 0; dt < NKS; ++dt) {
            uint32_t tw[4];
            #pragma unroll
            for (int e = 0; e < 8; e += 2) tw[e >> 1] = DT16<1>::pk(acc[dt][e] * inv,
                                                                    acc[dt][e + 1] * inv);
            *(uint4*)(op + dt * 16 + 8 * h) = make_uint4(tw[0], tw[1], tw[2], tw[3]);
        }
        if (h == 0) { sm[idx] = m_ref; sl[idx] = lrow; }
    } else {
        float* so = (float*)a.scratch;
        float* sm = so + nrow * HEAD_DIM;
        float* sl = sm + nrow;
        float* op = so + idx * HEAD_DIM;
        #pragma unroll
        for (int dt = 0; dt < NKS; ++dt) *(v8f*)(op + dt * 16 + 8 * h) = acc[dt];
        if (h == 0) { sm[idx] = m_ref; sl[idx] = lrow; }
    }
}

// ---- Output gate and fp8 quantiser as the merge's epilogue (attn_paged_gate_quant) ----
//
// A gated attention's merged row feeds gate_quant_fp8: bf16(sigmoid(gate)) times the row, rounded
// to bf16, then E4M3 codes with one scale per 128 columns. A merge row is one (token, head) pair
// and a scale group is 128 columns, so the `_gq` forms finish the row in registers instead of in a
// second dispatch.
//
// Byte-identical to gate_quant_fp8 (r4d_fused_quant_fp8.hip): gate rounded to bf16 before the
// multiply, product rounded, scale = group absmax / 448 (1 at zero), codes clamped to +-448, and
// the bf16 product stored to `out`. A padded slot merges to zeros and takes the same epilogue.
// GQ == 2 is the int8 form (scale amax / 127, round to nearest-even, r4d_q8.h).
// GQ == 3 (attn_paged_gate_had_quant) is the int8 form with the width-128 Hadamard between the
// bf16 product and the absmax, byte-identical to gate_had_quant_i8g: `out` keeps the unrotated
// product, and the stages run by ascending stride as r4d_fwht_wave's do, whose butterflies give
// the same bits under any thread map.
struct R4DGateQuant {
    const uint16_t* g;          // gate, row `job` at g + job * g_ld
    uint8_t*        q;          // E4M3 (or int8, `i8`) codes, row `job` at q + job * q_ld
    float*          s;          // one scale per 128 columns, row `job` at s + job * s_ld
    unsigned        g_ld, q_ld, s_ld;
    int             i8;         // read by the launcher only: which combine to instantiate
    int             rot;        // likewise: the rotated int8 form (GQ 3)
};
constexpr int kGqGroup = 128;

__device__ __forceinline__ uint16_t r4d_gq_gate(uint16_t gb, uint16_t xb) {
    const float    gv = bf16_to_f32(gb);
    const uint16_t ab = f32_to_bf16(1.0f / (1.0f + expf(-gv)));
    return f32_to_bf16(bf16_to_f32(ab) * bf16_to_f32(xb));
}
__device__ __forceinline__ float r4d_gq_clamp(float v) { return fminf(fmaxf(v, -448.0f), 448.0f); }

// The plain decode kernel, and the chained one (r4dchain*, r4d_chain.h) over the same body.
template<int NWARPS, int TILE, int HEAD_DIM, int GQA, int BS, int KVP, int OPT,
         int WIN = 0, RadKVLayout Layout = RAD_KVQ_NONE>
__global__ __launch_bounds__(NWARPS * 32)
void r4d_attn_decode_kernel(const R4DArgs a, int splits)
{
    r4d_attn_decode_body<NWARPS, TILE, HEAD_DIM, GQA, BS, KVP, OPT, WIN, Layout>(a, splits);
}
template<int NWARPS, int TILE, int HEAD_DIM, int GQA, int BS, int KVP, int OPT,
         int WIN = 0, RadKVLayout Layout = RAD_KVQ_NONE>
__global__ __launch_bounds__(NWARPS * 32)
void r4dchain_attn_decode(const R4DArgs a, int splits)
{
    r4d_chain_wait_vec();
    r4d_attn_decode_body<NWARPS, TILE, HEAD_DIM, GQA, BS, KVP, OPT, WIN, Layout>(a, splits);
    r4d_chain_done();
}

// End of rad0051-owned decode templates.
