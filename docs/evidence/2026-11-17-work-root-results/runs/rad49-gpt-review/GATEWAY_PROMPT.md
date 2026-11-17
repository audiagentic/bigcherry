# Brief: review + online deep search — Flash-Next on a 4-GPU AMD consumer host

You are reviewing a performance project and searching the web for techniques from other inference engines that could
help. You cannot read our files; the essential kernel source is pasted at the end. Answer with the report itself.

## The system

- Model: Qwen3.8-Next-Flash style MoE ("qwen4exp"): 49 MoE layers, 512 routed experts a layer, top-k 10, hidden 2560, expert
  intermediate 384 (gate/up N=768, down K=384), hybrid attention (GDN linear-attention + full attention), MTP speculation
  depth 3. Experts are "w4nl64": 4-bit NON-LINEAR codebook nibbles (a 16-entry value table, like llama.cpp IQ4_NL) with a
  per-64 scale; activations are quantised to E4M3 (fp8) with a per-128 scale before the expert GEMMs.
- Engine: Radiance (C++/HIP inference engine) plus BigCherry components. Host: i7-12700KF, 94 GiB RAM, no GPU peer-to-peer
  (all cross-card traffic goes through host memory). Cards:
  - R9700 (gfx1201, RDNA4, 32 GB, PCIe Gen4 x4): runs the model (attention, router, shared parts, some experts).
  - 2x RX 7900 XTX (gfx1100, RDNA3, 24 GB, Gen4 x8): "helpers" holding the hottest experts (22 GiB each).
  - RX 6900 XT (gfx1030, RDNA2, 16 GB, PCIe Gen3 x4): "tail" helper holding the coldest experts (14 GiB).
  - Remaining experts sit in a 9 GiB pinned host pool the engine reads over its link. A 47.7 GiB n-gram embedding table is
    pinned in host RAM; a 12 GiB host KV tier keeps finished sessions.
- Remote-MoE protocol: per layer the engine publishes routes + E4M3 activations in shared host memory; each helper computes
  its owned experts full-width, reduces per token (weighted by the engine's bf16 router weights) and writes bf16/fp16 rows
  back; helper 0 can combine all helpers' rows on the CPU (AVX2/F16C) into one plane the engine reads.

## Measured today (all A/B, same session)

- Best layout `flashnext-4gpu` (above): decode 1 stream ~24.9 ms/step, 6 streams ~47 ms/step, 44k doc prefill ~3,650 tok/s,
  8 sessions x 262k context. Tensor-parallel on the two XTX (host-memory all-reduce) is worse on every metric (prefill ~1,950).
- Cold / unusual prompts (synthetic 27k-token text) take 11.4 s TTFT (~2,370 tok/s): the 6900 XT tail is busy ~73% of the
  wall time because spread-out routing sends it ~2,900 routes a 2048-row request. Its per-request stage times vs an XTX
  helper (us): gate_up 5,427 vs 1,270; down 2,334 vs 733; reduce (result write over Gen3 x4) 1,835 vs 634; activation copy
  1,075 vs 405.
- Tail kernel trace (rocprofv3): `lane_moe_rows_w4_quaddpp<gate_up,16 rows>` 50%, `<down,16>` 21%, result write 17%,
  activation copy 9%. Achieved ~2.3 TFLOPS of the card's ~46 (fp16 dot2). The kernel decodes each W4 tile once per 16 rows
  (DPP quad shuffles), f16 dot2 (`__builtin_amdgcn_fdot2`) with f32 accumulate. A replay of 40 captured real requests timed six
  existing forms: production quad16 333 ms, quad 334, quad_v2 384, output_lane 600, rowlane16 760, rowlane8 875 (all
  bit-identical). The forms were tuned before (RAD46) for accuracy; an older faster kernel failed a KL quality gate.
- A smaller tail (7 GiB, experts moved to the host pool, funded by a smaller KV tier) gives cold TTFT 8.4 s but costs ~21% of
  sustained 8-session prefill: experts in the host pool are read over the R9700's x4 link.
- CPU combine pushed straight into R9700 VRAM via a DMA-BUF mapping (instead of the GPU pulling it from host memory):
  serial 1.25x at 768 rows, ~1.0x at 1-16 rows (decode). Only prefill-sized steps could gain.

## Code to read (paths relative to <repo>)

- `engines/radiance/components/moe-remote/moe_lane_gemm.h` — the helper's expert kernels (`lane_moe_rows_w4_quaddpp` ~line 834,
  `lane_moe_gemv_w4h2`, `lane_moe_rows_w4_outputlane`, rowlane).
- `engines/radiance/components/moe-remote/moe_remote_gpu.h` — dispatch and form selection (~lines 640-1000).
- `engines/radiance/components/moe-remote/rad46_rows_policy.h` — measured crossovers between forms.
- `engines/radiance/components/moe-remote/moe_lane_reduce.h`, `moe_lane_front.h`, `combine_f16.h` — reduce / result write.
- Findings with run references: `<work-root>/runs/rad49-findings-20261109.md`.
- BAR-push lab (RAD50): `<work-root>/wt-rad42-cpu-bar-push/tools/lab/rad42-bar-push/` (PROGRESS.md, SLICE_OPTIMIZATION.md).

## What I want

1. **Deep web search** for techniques in other engines and papers that apply here, with links and the specific file or
   section, e.g.:
   - llama.cpp HIP/ROCm MMQ/MMVQ for IQ4_NL / Q4 on RDNA2 (gfx1030) and RDNA3: do they decode the non-linear codebook to
     int8 and use `v_dot4_i32_i8` / dp4a with int8 activations (q8_1)? What throughput do they reach on a 6900 XT for MoE
     (`mul_mat_id`) prefill? Could our w4nl + E4M3 activation path use an int8 dot4 route on RDNA2 (bit-exactness vs our f16
     path is a concern; quality is gated by KL)?
   - Grouped/MoE GEMM kernels for many small experts on GPUs without matrix cores (vLLM fused_moe, SGLang, Triton MoE,
     ExLlama/ExLlamaV2/V3, Marlin-style W4 GEMM, AMD Composable Kernel / hipBLASLt grouped GEMM, rocWMMA alternatives).
   - Heterogeneous / offloaded MoE serving: KTransformers (CPU+GPU expert split), Fiddler, MoE-Lightning, PowerInfer,
     llama.cpp `--override-tensor` expert offload, DeepSpeed-MoE / expert parallelism, HybriMoE, MoE-Infinity, Pre-gated MoE,
     expert placement by load and by device speed (heterogeneous devices), expert replication of hot experts.
   - Reducing PCIe traffic for remote experts: sending fp8/int8 activations and results, computing on the CPU for cold
     experts, overlapping transfer with compute (chunked/pipelined MoE), prefetching experts by predicted routing.
2. **Review** our kernels and the protocol against what you find: concrete, ranked opportunities (expected gain, effort,
   risk to the KL quality gate), especially (a) the RDNA2 tail GEMM (gate_up/down), (b) the tail's result write and
   activation copy over its Gen3 x4 link, (c) prefill throughput of the whole layout, (d) placement across devices of
   different speed.
3. Flag anything in our measurements or conclusions that looks wrong or that other engines contradict.

Be specific: cite file paths/lines in our code and URLs/commits in the references. Prefer a short ranked list of actionable
ideas over a survey. Mark clearly what is verified from a source versus your inference.


## Appendix: the RDNA2 tail kernel today (moe_lane_gemm.h lines 820-960)

```cpp
}

/* RAD46 quad-DPP experiment: four adjacent lanes own one output column. Each
 * computes one dot16 in a K64 group. Only quad lane zero's ordered DPP fold is
 * used; other lanes must remain active as DPP sources, including padded rows.
 * Dot8 resets, dot16 additions, K64 ordering and the K128 scale fold match the
 * output-lane experiment. This is not a claim of universal libr2 equivalence.
 * Host dispatch admits only complete K128 groups and complete N32 tiles. */
constexpr uint32_t kQuadDppN = 32, kQuadDppRB = 16;
// RAD46 v2: halve the 16-row live accumulation/staging set for gate/up.
// Each row has an independent, bit-identical K-fold; the down path keeps the
// established 16-row batch until its separate regression gate is measured.
constexpr uint32_t kQuadV2GuRB = kQuadDppRB / 2;
template <bool TOKEN, uint32_t RB>
__global__ __launch_bounds__(128) void lane_moe_rows_w4_quaddpp(
    const uint8_t* aq, const float* as, const int32_t* sorted, const int32_t* off, uint32_t top_k,
    void* const* code0, void* const* scale0, void* const* code1, void* const* scale1,
    uint32_t n0, uint32_t n1, uint32_t k0, uint32_t k1, uint32_t a_stride, uint32_t out_stride,
    uint16_t* out, uint32_t part_pitch) {
#pragma clang fp contract(off)
#pragma clang fp reassociate(off)
    static_assert(RB > 0 && RB <= 16, "quad-DPP row batch exceeds staging geometry");
    const uint32_t e = blockIdx.x, tid = threadIdx.x, qlane = tid & 3u;
    const uint32_t rs = (uint32_t)off[e], re = (uint32_t)off[e + 1];
    if (rs >= re) return;
    const uint32_t cls = e & 1u, ix = e >> 1;
    const uint32_t N = cls ? n1 : n0, K = cls ? k1 : k0;
    const uint32_t n_base = blockIdx.y * kQuadDppN, n = n_base + tid / 4u;
    if (n_base >= N) return;
    const uint8_t* codes = static_cast<const uint8_t*>(cls ? code1[ix] : code0[ix]);
    const uint8_t* scales = static_cast<const uint8_t*>(cls ? scale1[ix] : scale0[ix]);
    if (!codes || !scales) return;
    __shared__ uint4 activation[2][RB][16];
    __shared__ float activation_scale[2][RB];
    __shared__ uint32_t arow[RB];
    __shared__ uint32_t lut[16];
    // Store exact half bits: weight decode needs no float-to-half conversions.
    if (tid < 16) lut[tid] = __builtin_bit_cast(uint16_t, (_Float16)kGemvW4nl[tid]);
    auto pack = [](_Float16 x, _Float16 y) {
        return (uint32_t)__builtin_bit_cast(uint16_t, x) |
               ((uint32_t)__builtin_bit_cast(uint16_t, y) << 16);
    };
    auto decode = [&](uint32_t w) {
        return make_uint4(
            lut[w & 15u] | (lut[(w >> 8) & 15u] << 16),
            lut[(w >> 16) & 15u] | (lut[(w >> 24) & 15u] << 16),
            lut[(w >> 4) & 15u] | (lut[(w >> 12) & 15u] << 16),
            lut[(w >> 20) & 15u] | (lut[(w >> 28) & 15u] << 16));
    };
    auto dot8 = [](uint4 x, uint4 w) {
        float s = 0.0f;
        asm("v_dot2_f32_f16 %0, %1, %2, %0" : "+v"(s) : "v"(x.x), "v"(w.x));
        asm("v_dot2_f32_f16 %0, %1, %2, %0" : "+v"(s) : "v"(x.y), "v"(w.y));
        asm("v_dot2_f32_f16 %0, %1, %2, %0" : "+v"(s) : "v"(x.z), "v"(w.z));
        asm("v_dot2_f32_f16 %0, %1, %2, %0" : "+v"(s) : "v"(x.w), "v"(w.w));
        return s;
    };
    auto quad_sum = [](float partial) {
        // Keep each broadcast next to its consuming add. Separate intrinsics
        // let LLVM hoist 96 broadcast results across next-tile staging. Both
        // early-clobbers prevent either temporary overwriting the source that
        // the later broadcasts still need. All four source lanes stay active.
        float sum, exchanged;
        asm("v_mov_b32_dpp %1, %2 quad_perm:[1,1,1,1] row_mask:0xf bank_mask:0xf\n\t"
            "v_add_f32 %0, %2, %1\n\t"
            "v_mov_b32_dpp %1, %2 quad_perm:[2,2,2,2] row_mask:0xf bank_mask:0xf\n\t"
            "v_add_f32 %0, %0, %1\n\t"
            "v_mov_b32_dpp %1, %2 quad_perm:[3,3,3,3] row_mask:0xf bank_mask:0xf\n\t"
            "v_add_f32 %0, %0, %1"
            : "=&v"(sum), "=&v"(exchanged) : "v"(partial));
        return sum;
    };
    for (uint32_t r0 = rs; r0 < re; r0 += RB) {
        const uint32_t nr = re - r0 < RB ? re - r0 : RB;
        if (tid < RB)
            arow[tid] = tid < nr ? (TOKEN ? (uint32_t)sorted[r0 + tid] / top_k : r0 + tid) : 0u;
        float acc[RB];
#pragma unroll
        for (uint32_t r = 0; r < RB; ++r) acc[r] = 0.0f;
        uint2 prefetched[2];
        uint16_t prefetched_scales;
        auto fetch = [&](uint32_t kc) {
#pragma unroll
            for (uint32_t group = 0; group < 2; ++group)
                prefetched[group] = *reinterpret_cast<const uint2*>(
                    codes + (uint64_t)n * (K / 2u) + kc / 2u + group * 32u + qlane * 8u);
            prefetched_scales = reinterpret_cast<const uint16_t*>(
                scales + (uint64_t)n * (K / 64u))[kc / 128u];
        };
        auto stage = [&](uint32_t kc, uint32_t buf) {
            if (tid < RB)
                activation_scale[buf][tid] = tid < nr ?
                    as[(uint64_t)arow[tid] * (a_stride / 128u) + kc / 128u] : 0.0f;
            for (uint32_t i = tid; i < RB * 8u; i += 128u) {
                const uint32_t r = i / 8u, p16 = i % 8u;
                // Zero every invalid row, so the hot arithmetic has no row predicates.
                const uint4 q = r < nr ? *reinterpret_cast<const uint4*>(
                    aq + (uint64_t)arow[r] * a_stride + kc + p16 * 16u) : make_uint4(0, 0, 0, 0);
                const uint32_t qw[4] = {q.x, q.y, q.z, q.w};
                uint32_t hw[8];
#pragma unroll
                for (uint32_t j = 0; j < 4; ++j) {
                    hw[2 * j] = pack((_Float16)gemv_e4m3(qw[j] & 255u),
                                    (_Float16)gemv_e4m3((qw[j] >> 8) & 255u));
                    hw[2 * j + 1] = pack((_Float16)gemv_e4m3((qw[j] >> 16) & 255u),
                                        (_Float16)gemv_e4m3(qw[j] >> 24));
                }
                activation[buf][r][2 * p16] = make_uint4(hw[0], hw[1], hw[2], hw[3]);
                activation[buf][r][2 * p16 + 1] = make_uint4(hw[4], hw[5], hw[6], hw[7]);
            }
        };
        fetch(0);
        __syncthreads();
        stage(0, 0);
        __syncthreads();
        uint32_t cur = 0;
        for (uint32_t kc = 0; kc < K; kc += 128u, cur ^= 1u) {
            const uint2 current[2] = {prefetched[0], prefetched[1]};
            const uint16_t ws = prefetched_scales;
            const float ws0 = gemv_e4m3(ws & 255u) * 0x1p-13f;
            const float ws1 = gemv_e4m3(ws >> 8) * 0x1p-13f;
            if (kc + 128u < K) fetch(kc + 128u);
            float first_weighted[RB];
#pragma unroll
            for (uint32_t group = 0; group < 2; ++group) {
                const uint4 w0 = decode(current[group].x), w1 = decode(current[group].y);
                const uint32_t kw = group * 8u + qlane * 2u;
#pragma unroll
                for (uint32_t r = 0; r < RB; ++r) {
                    const float lo = dot8(activation[cur][r][kw], w0);
                    const float hi = dot8(activation[cur][r][kw + 1u], w1);
                    const float raw = quad_sum(lo + hi);
                    if (group == 0) first_weighted[r] = raw * ws0;
                    else {
                        const float second_weighted = raw * ws1;
                        const float folded = first_weighted[r] + second_weighted;
                        acc[r] += folded * activation_scale[cur][r];
                    }
                }
            }
            if (kc + 128u < K) stage(kc + 128u, cur ^ 1u);
```
