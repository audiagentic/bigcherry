"""RD05/06/07: RDNA4 WMMA flash-attn and Q6_K mmq prefill performance work.

Provenance (group 'rdna-boosts' patches are external backports; the
machine-readable PROVENANCE dict below is cross-checked against
external-sources.toml by tools/tests/test_external_sources.py):

  source:            stew675-rdna-boosts
  repo:              https://github.com/stew675/llama.cpp
  locator:           rdna-boosts (branch name is a locator only, NOT identity)
  fork commit:       1d525bd45f9e8f844856ecbc5dd8ae33c8d34eff
                     (snapshot v2; v1 ledger items 5/6/7, 5e5667a85,
                     0226a6b23, b3a95b554, are the pre-rebase identities of
                     the SAME logical change, content-identical per git
                     patch-id)
                     "cuda : RDNA4 WMMA flash-attn and Q6_K mmq prefill
                     performance work"
  reviewed snapshot: v2 -- head 9e46e1fdc7a880f9ae9a2f9a693ae3e14c142a22
                     on base 4df29be4f4c3673f428170fda944a5b19f743bb8
  plan items:        RD05 (WMMA FA head-256 combine race fix + RDNA4
                     config), RD06 (WMMA FA head 320/512/576 enablement +
                     config), RD07 (Q6_K mmq sub-scale fold)
  mainline status:   NOT merged into ggml-org/llama.cpp master as of tip
                     6d0549831 (git cherry patch-id check, 2026-08-18)

What it does (performance, kernel-level; one squashed commit, 8 upstream
sub-changes):
  - fix the head-256 WMMA flash-attn combine race and tune its RDNA4
    config (nthreads 256, nbatch_fa 32, nbatch_combine 32)
  - fix the tile_Q reuse race for np == 1 configs: sync between flash-attn
    tile iterations in the kernel's kbc loop
  - tune the head 320x256/512x512/576x512 WMMA FA configs and enable the
    WMMA path up to head 576 by default on RDNA4
    (GGML_CUDA_FA_WMMA_256=0 opts out; RDNA3/3.5 stay at head <= 128)
  - hoist and fold the Q6_K mmq sub-scales into the row base-scale to
    remove the int-mul from the wmma-result scale chain
    (fork: Q6_K mmq 40 -> 58 TFLOPS)
  - per-op timing instrumentation (GGML_CUDA_OP_TIMING env-gated) +
    GGML_CUDA_MMQ_J_MAX override + mmq/FA perf test cases

Porting notes:
  - Six files: fattn-mma-f16.cuh (config table + 2 sync fixes), fattn.cu
    (softcap read + RDNA4 WMMA head gating), mmq-vec-dot.cuh (Q6_K
    sub-scale fold), mmq.cuh (J_MAX env), ggml-cuda.cu (op timing),
    tests/test-backend-ops.cpp (perf cases + softcap matrix).
  - ADAPTATION (1000 cast): the mmq-vec-dot.cuh sum line in our base
    carries BigCherry patch 1000's "((float) C.x[l])" cast (upstream PR
    #25940, validated in the framework set). The anchor matches the
    post-1000 text and the replacement keeps an equivalent single "(float)"
    cast -- semantically identical to the fork's final line.
  - ADAPTATION (test positions): the fork perf-hunk context was added by
    this same commit's earlier lines in a different layout; our base's
    make_test_cases_perf anchors on the HI70 direct-op corpus instead
    (test-only position deviation). The fork's "eval" test hunk sits
    inside a "#if 0" dead-code block in BOTH the fork base and ours and
    is intentionally OMITTED.
  - RD08 (1204) relationship: this commit's op-timing instrumentation
    records events inside the graph path; RD08's later commit makes
    GGML_CUDA_OP_TIMING disable graph capture so the two work together.
    Standalone, op timing is OFF unless the env var is set, and with it
    set plus graph capture active the pre-RD08 abort behaviour remains
    (diagnostic path only, never the default).
  - Base co-tenancy: fattn.cu / fattn-mma-f16.cuh have ZERO framework
    patches; mmq-vec-dot.cuh is touched by 1000 (see above); ggml-cuda.cu
    co-tenants (0200/0700/0830/0900/1004) are in other functions.

Isolation and promotion (first-sweep policy, RD review 2026-08-18):
  - GROUP 'rdna-boosts' + STATE 'untested' keeps this OUT of the
    production 'framework' and 'validated-enhancements' patch-sets.
  - Bench: FA workloads at head 128/256/320/512/576 vs GGML_CUDA_FA_WMMA_256=0
    (tile kernel), plus Q6_K prefill (mmq path) at the qwen35-27B shapes.
    Correctness: FLASH_ATTN_EXT + MUL_MAT backend-ops must stay green.

Maintenance (future pin bumps / fork movement):
  - fattn config tables and the mmq Q6_K path are upstream-touched;
    re-derive from the tracked fork commit in external-sources.toml and
    run `python -m bigcherry sources check` before every pin bump.
"""

import re

from bigcherry.patcher import Edit, FilePatch

GROUP = "rdna-boosts"
STATE = "untested"

PROVENANCE = {
    "source-id": "stew675-rdna-boosts",
    "plan-item": "RD05/RD06/RD07",
    "fork-commit": "1d525bd45f9e8f844856ecbc5dd8ae33c8d34eff",
    "fork-commit-title": "cuda : RDNA4 WMMA flash-attn and Q6_K mmq prefill performance work",
    "original-commit": "5e5667a85c201a5f43f071d25b3050d2783274b5",
    "snapshot-head": "9e46e1fdc7a880f9ae9a2f9a693ae3e14c142a22",
    "snapshot-base": "4df29be4f4c3673f428170fda944a5b19f743bb8",
    "adaptations": [
        "mmq-vec-dot.cuh: sum line anchored on post-1000 text "
        "(\"((float) C.x[l])\" cast from framework patch 1000, upstream "
        "PR #25940); replacement keeps an equivalent single (float) cast.",
        "tests: perf cases inserted after the HI70 direct-op corpus; the "
        "fork eval-test hunk sits in a #if 0 dead block and is omitted.",
    ],
}


# ------------------------------------------------------------ fattn-mma-f16.cuh

_CONFIG_OLD = """    GGML_CUDA_FATTN_MMA_CONFIG_CASE(256, 256, 64, 128, 2,  64, 128, 128,  64, 1, true);

    GGML_CUDA_FATTN_MMA_CONFIG_CASE(320, 256, 32, 128, 2,  32, 160, 128, 128, 1, true);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE(320, 256, 64, 128, 2,  32, 160, 128, 128, 1, true);

    GGML_CUDA_FATTN_MMA_CONFIG_CASE(512, 512,  8, 128, 3,  64,  96,  64, 128, 1, true);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE(512, 512, 16, 128, 3,  64,  96,  64, 128, 1, true);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE(512, 512, 32, 128, 2,  32, 128, 128, 128, 1, true);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE(512, 512, 64, 128, 2,  32, 128, 128, 128, 1, true);

    GGML_CUDA_FATTN_MMA_CONFIG_CASE(576, 512,  8, 128, 3,  64,  96,  64, 128, 1, true);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE(576, 512, 16, 128, 3,  64,  96,  64, 128, 1, true);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE(576, 512, 32, 128, 2,  32, 160, 128, 128, 1, true);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE(576, 512, 64, 128, 2,  32, 160, 128, 128, 1, true);"""

_CONFIG_NEW = """    GGML_CUDA_FATTN_MMA_CONFIG_CASE(256, 256, 64, 256, 2,  32, 128, 128,  32, 1, true);

    GGML_CUDA_FATTN_MMA_CONFIG_CASE(320, 256, 32, 256, 2,  64,  96,  16,  16, 1, true);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE(320, 256, 64, 256, 2,  64,  96,  16,  16, 1, true);

    GGML_CUDA_FATTN_MMA_CONFIG_CASE(512, 512,  8, 128, 3,  64,  96,  64, 128, 1, true);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE(512, 512, 16, 128, 2,  64,  96,  16,  16, 1, true);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE(512, 512, 32, 256, 2, 128,  96,  16,  16, 1, true);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE(512, 512, 64, 128, 2,  32, 128, 128, 128, 1, true);

    GGML_CUDA_FATTN_MMA_CONFIG_CASE(576, 512,  8, 128, 3,  64,  96,  64, 128, 1, true);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE(576, 512, 16, 128, 2,  64,  96,  16,  16, 1, true);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE(576, 512, 32, 256, 2, 128,  96,  64,  16, 1, true);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE(576, 512, 64, 128, 2,  32, 160, 128, 128, 1, true);"""

_K00_SYNC_OLD = """        if (np > 1) {
            __syncthreads();
        }"""

_K00_SYNC_NEW = """        // The tile_Q buffer is reused for the next k00 iteration, so all warps must sync here
        // before its data is overwritten. With np > 1 only some warps read back, but they all write.
        if (np > 1 || k00 + nbatch_combine < DV/2) {
            __syncthreads();
        }"""

_DKQ_GATE_OLD = """    if (ncols1*ncols2 < 16 || ncols2 == 1 || DKQ > 128) {"""

_DKQ_GATE_NEW = """    if (ncols1*ncols2 < 16 || ncols2 == 1 || DKQ > 576) {"""

_KBC_ANCHOR_OLD = """        }

        kbc += iter_k;
        kbc -= kbc % iter_k;"""

_KBC_ANCHOR_NEW = """        }

        // The next process_tile call reuses the tile_Q buffer for its Q/K tiles, so all warps must
        // have finished reading the combined results before any of them starts the next call.
        // (With np == 1 the end-of-k00 barrier does not fire, so this is required for correctness.)
        __syncthreads();

        kbc += iter_k;
        kbc -= kbc % iter_k;"""

# --------------------------------------------------------------------- fattn.cu

# The trailing comment is blank space in the anchor view (space run).
_SOFTCAP_OLD = (r"    float max_bias = 0.0f;\n"
                r"    memcpy\(&max_bias, \(const float \*\) KQV->op_params \+ 1, sizeof\(float\)\);\n"
                r"\n    [ ]{70,80}\n")

_SOFTCAP_NEW = """    float max_bias = 0.0f;
    memcpy(&max_bias, (const float *) KQV->op_params + 1, sizeof(float));

    float logit_softcap = 0.0f;
    memcpy(&logit_softcap, (const float *) KQV->op_params + 2, sizeof(float));

    // The effective batch size for the kernel can be increased by gqa_ratio."""

# Anchor on the code lines only (the comment above is unchanged and stays).
_WMMA_OLD = """    if ((amd_wmma_available(cc) && gqa_opt_applies && Q->ne[0] <= 128) && Q->ne[0] != 40 && Q->ne[0] != 72 && Q->ne[1] * gqa_ratio_eff > 8) {
        return BEST_FATTN_KERNEL_MMA_F16;
    }"""

_WMMA_NEW = """    // RDNA4 WMMA has higher throughput than RDNA3; heads up to 576 (incl. the DKQ != DV shapes)
    // are enabled by default there. Set GGML_CUDA_FA_WMMA_256=0 to force the WMMA path off for
    // heads > 128 (e.g. to compare against the tile kernel).
    const char * wmma_256_env = getenv("GGML_CUDA_FA_WMMA_256");
    const bool wmma_256 = wmma_256_env == nullptr || std::atoi(wmma_256_env) != 0;
    const int wmma_max_head = (wmma_256 && GGML_CUDA_CC_IS_RDNA4(cc)) ? 576 : 128;
    if ((amd_wmma_available(cc) && gqa_opt_applies && Q->ne[0] <= wmma_max_head) && Q->ne[0] != 40 && Q->ne[0] != 72 && Q->ne[1] * gqa_ratio_eff > 8) {
        // The kernel instantiates logit_softcap only for heads 128/256/512.
        if (logit_softcap == 0.0f || Q->ne[0] == 128 || Q->ne[0] == 256 || Q->ne[0] == 512) {
            return BEST_FATTN_KERNEL_MMA_F16;
        }
    }"""

# -------------------------------------------------------------- mmq-vec-dot.cuh

_DF_HOIST_OLD = """    const int   * x_sc = (const int   *) x_df + MMQ_TILE_NE_K/QI6_K;
    const int   * y_qs = (const int   *) y + 4;
    const float * y_df = (const float *) y;

    const int i0 = (threadIdx.y / ntx) * rows_per_warp;"""

_DF_HOIST_NEW = """    const int   * x_sc = (const int   *) x_df + MMQ_TILE_NE_K/QI6_K;
    const int   * y_qs = (const int   *) y + 4;
    const float * y_df = (const float *) y;

    const int i0 = (threadIdx.y / ntx) * rows_per_warp;

    // Row base scales are invariant over the k01 and j0 loops; load them once.
    // Each thread owns fixed elements of the C tile, so one value per element suffices.
    float x_df_reg[ntx][tile_C::ne];
#pragma unroll
    for (int n = 0; n < ntx; ++n) {
#pragma unroll
        for (int l = 0; l < tile_C::ne; ++l) {
            const int i = i0 + n*tile_C::I + tile_C::get_i(l);
            x_df_reg[n][l] = x_df[i*sram_stride];
        }
    }"""

# Order-dependent: anchors on the text inserted by rd07-hoist-base-scale
# (declared first in this file patch), which makes the window unique -- the
# surrounding k01/j0 loop text is identical in the other arch branches.
_SC_FOLD_OLD = """            x_df_reg[n][l] = x_df[i*sram_stride];
        }
    }

    for (int k01 = 0; k01 < MMQ_TILE_NE_K; k01 += 4) {
        const int k0 = k00 + k01;

        tile_A A[ntx];
#pragma unroll
        for (int n = 0; n < ntx; ++n) {
            load_ldmatrix(A[n], x_qs + (i0 + n*tile_A::I)*sram_stride + k0, sram_stride);
        }

#pragma unroll
        for (int j0 = 0; j0 < J; j0 += ntx*tile_C::J) {"""

_SC_FOLD_NEW = """            x_df_reg[n][l] = x_df[i*sram_stride];
        }
    }

    for (int k01 = 0; k01 < MMQ_TILE_NE_K; k01 += 4) {
        const int k0 = k00 + k01;

        tile_A A[ntx];
#pragma unroll
        for (int n = 0; n < ntx; ++n) {
            load_ldmatrix(A[n], x_qs + (i0 + n*tile_A::I)*sram_stride + k0, sram_stride);
        }

        // Sub-scales for this k01 chunk; invariant over the j0 loop.
        int8_t x_sc_reg[ntx][tile_C::ne];
#pragma unroll
        for (int n = 0; n < ntx; ++n) {
#pragma unroll
            for (int l = 0; l < tile_C::ne; ++l) {
                const int i = i0 + n*tile_C::I + tile_C::get_i(l);
                x_sc_reg[n][l] = ((const int8_t *) (x_sc + i*sram_stride + k00/16))[k01/4];
            }
        }

        // Fold the sub-scale and the row base scale into one f32 per element;
        // saves one int-multiply and one convert per element in the j0 loop.
        float x_s2_reg[ntx][tile_C::ne];
#pragma unroll
        for (int n = 0; n < ntx; ++n) {
#pragma unroll
            for (int l = 0; l < tile_C::ne; ++l) {
                x_s2_reg[n][l] = (float) x_sc_reg[n][l] * x_df_reg[n][l];
            }
        }

#pragma unroll
        for (int j0 = 0; j0 < J; j0 += ntx*tile_C::J) {"""

# Base line carries framework patch 1000's ((float) C.x[l]) cast; the
# replacement keeps an equivalent single (float) cast.
_SUM_OLD = """                for (int l = 0; l < tile_C::ne; ++l) {
                    const int i = i0 + n*tile_C::I + tile_C::get_i(l);
                    const int8_t * sc = (const int8_t *) (x_sc + i*sram_stride + k00/16);
                    sum[(j0/tile_C::J + n)*tile_C::ne + l] += ((float) C.x[l]) * sc[k01/4] * x_df[i*sram_stride] * dB;
                }"""

_SUM_NEW = """                for (int l = 0; l < tile_C::ne; ++l) {
                    sum[(j0/tile_C::J + n)*tile_C::ne + l] += (float) C.x[l] * x_s2_reg[n][l] * dB;
                }"""

# ---------------------------------------------------------------------- mmq.cuh

_JMAX_OLD = """    int ret = std::min(ne11, int64_t(512));
    ret -= ret % 8;"""

_JMAX_NEW = """    int ret = std::min(ne11, int64_t(512));
    ret -= ret % 8;
    const char * env = getenv("GGML_CUDA_MMQ_J_MAX");
    if (env != nullptr) {
        ret = std::min(ret, std::atoi(env));
    }"""

# ------------------------------------------------------------- ggml-cuda.cu

_INCLUDE_OLD = """#include <algorithm>
#include <array>"""

_INCLUDE_NEW = """#include <algorithm>
#include <numeric>
#include <array>"""

_TIMING_HEAD_OLD = """    bool graph_evaluated_or_captured = false;
"""

_TIMING_HEAD_NEW = """    bool graph_evaluated_or_captured = false;

    // per-op timing instrumentation (env-gated, diagnostic only)
    const bool op_timing = getenv("GGML_CUDA_OP_TIMING") != nullptr;
    std::vector<cudaEvent_t> op_ev0;
    std::vector<cudaEvent_t> op_ev1;
    std::vector<std::pair<const ggml_tensor *, int>> op_nodes;
    if (op_timing) {
        op_ev0.resize(cgraph->n_nodes);
        op_ev1.resize(cgraph->n_nodes);
        op_nodes.reserve(cgraph->n_nodes);
        for (int i = 0; i < cgraph->n_nodes; i++) {
#ifdef GGML_USE_HIP
            CUDA_CHECK(cudaEventCreateWithFlags(&op_ev0[i], hipEventDefault));
            CUDA_CHECK(cudaEventCreateWithFlags(&op_ev1[i], hipEventDefault));
#else
            CUDA_CHECK(cudaEventCreateWithFlags(&op_ev0[i], cudaEventDefault));
            CUDA_CHECK(cudaEventCreateWithFlags(&op_ev1[i], cudaEventDefault));
#endif
        }
    }
"""

# The LOG_ERROR line is string-literal noise in the anchor view, so the two
# event blocks are two separate insertions: before 'bool ok = ...' (count 1)
# and after 'GGML_ASSERT(ok);'.
_TIMING_MID0_OLD = """                bool ok = ggml_cuda_compute_forward(*cuda_ctx, node);
"""

_TIMING_MID0_NEW = """                if (op_timing) {
                    CUDA_CHECK(cudaEventRecord(op_ev0[i], cuda_ctx->stream()));
                }

                bool ok = ggml_cuda_compute_forward(*cuda_ctx, node);
"""

_TIMING_MID1_OLD = """                GGML_ASSERT(ok);
"""

_TIMING_MID1_NEW = """                GGML_ASSERT(ok);

                if (op_timing) {
                    CUDA_CHECK(cudaEventRecord(op_ev1[i], cuda_ctx->stream()));
                    op_nodes.emplace_back(node, i);
                }
"""

# The // comment after #endif is blank space in the anchor view.
_TIMING_TAIL_OLD = """        graph_evaluated_or_captured = true;
#endif[ ]{10,30}
    }
}"""

_TIMING_TAIL_NEW = """        graph_evaluated_or_captured = true;
#endif  // USE_CUDA_GRAPH
    }

    if (op_timing) {
        CUDA_CHECK(cudaStreamSynchronize(cuda_ctx->stream()));
        static std::map<std::string, double> op_ms_total;
        static std::map<std::string, int>    op_cnt_total;
        std::map<std::string, double> op_ms;
        std::map<std::string, int>    op_cnt;
        for (const auto & [node, idx] : op_nodes) {
            float ms = 0.0f;
#ifdef GGML_USE_HIP
            CUDA_CHECK(hipEventElapsedTime(&ms, (hipEvent_t) op_ev0[idx], (hipEvent_t) op_ev1[idx]));
#else
            CUDA_CHECK(cudaEventElapsedTime(&ms, op_ev0[idx], op_ev1[idx]));
#endif
            std::string key = ggml_op_name(node->op);
            key += " ";
            key += node->name;
            if (node->op == GGML_OP_MUL_MAT && node->src[0] != nullptr && node->src[1] != nullptr) {
                char buf[64];
                snprintf(buf, sizeof(buf), " [%lldx%lldx%lld]",
                         (long long) node->src[0]->ne[0], (long long) node->src[0]->ne[1],
                         (long long) node->src[1]->ne[1]);
                key += buf;
            }
            op_ms[key] += ms;
            op_cnt[key]++;
            op_ms_total[key] += ms;
            op_cnt_total[key]++;
        }
        std::vector<std::pair<std::string, double>> sorted(op_ms.begin(), op_ms.end());
        std::sort(sorted.begin(), sorted.end(),
                  [](const auto & a, const auto & b) { return a.second > b.second; });
        double total = 0.0;
        for (const auto & [k, v] : sorted) {
            total += v;
        }
        GGML_LOG_INFO("%s: op timing: total %.2f ms over %zu nodes:\\n", __func__, total, op_nodes.size());
        for (const auto & [k, v] : sorted) {
            GGML_LOG_INFO("  %8.3f ms %5.1f%%  x%-4d %s\\n", v, 100.0 * v / total, op_cnt[k], k.c_str());
        }
        GGML_LOG_INFO("%s: op timing cumulative: %.2f ms over %d nodes\\n", __func__,
                      std::accumulate(op_ms_total.begin(), op_ms_total.end(), 0.0,
                                      [](double acc, const auto & p) { return acc + p.second; }),
                      std::accumulate(op_cnt_total.begin(), op_cnt_total.end(), 0,
                                      [](int acc, const auto & p) { return acc + p.second; }));
        for (cudaEvent_t e : op_ev0) {
            CUDA_CHECK(cudaEventDestroy(e));
        }
        for (cudaEvent_t e : op_ev1) {
            CUDA_CHECK(cudaEventDestroy(e));
        }
    }
}"""

# --------------------------------------------------- tests/test-backend-ops.cpp

_SOFTLOOP_OLD = """                            if (hsk != 128 && logit_softcap != 0.0f) continue;"""

_SOFTLOOP_NEW = """                            // The mma kernel instantiates logit_softcap for heads 128/256/512 only.
                            if (hsk != 128 && hsk != 256 && hsk != 512 && logit_softcap != 0.0f) continue;"""

# Position adaptation: anchor on the HI70 direct-op corpus (see header).
_PERF_ANCHOR_OLD = """    test_cases.emplace_back(new test_mul_mat(GGML_TYPE_Q4_K, GGML_TYPE_F32, 127, 128, 256, {1, 1}, {1, 1}));"""

_PERF_NEW = """    test_cases.emplace_back(new test_mul_mat(GGML_TYPE_Q4_K, GGML_TYPE_F32, 127, 128, 256, {1, 1}, {1, 1}));

    // rdna-boosts (RD05/06/07): Qwen3.6-27B Q6_K prefill shapes + FA perf:
    test_cases.emplace_back(new test_mul_mat(GGML_TYPE_Q6_K, GGML_TYPE_F32, 17408, 512, 5120, {1, 1}, {1, 1})); // ffn_up/ffn_gate
    test_cases.emplace_back(new test_mul_mat(GGML_TYPE_Q6_K, GGML_TYPE_F32, 5120,  512, 17408, {1, 1}, {1, 1})); // ffn_out
    test_cases.emplace_back(new test_mul_mat(GGML_TYPE_Q6_K, GGML_TYPE_F32, 10240, 512, 5120, {1, 1}, {1, 1})); // attn qkv
    test_cases.emplace_back(new test_mul_mat(GGML_TYPE_Q8_0, GGML_TYPE_F32, 17408, 512, 5120, {1, 1}, {1, 1})); // q8_0 at ffn shape
    // f16/f32 references at the same shapes:
    test_cases.emplace_back(new test_mul_mat(GGML_TYPE_F16, GGML_TYPE_F32, 17408, 512, 5120, {1, 1}, {1, 1}));
    test_cases.emplace_back(new test_mul_mat(GGML_TYPE_F16, GGML_TYPE_F32, 5120,  512, 17408, {1, 1}, {1, 1}));
    test_cases.emplace_back(new test_mul_mat(GGML_TYPE_F32, GGML_TYPE_F32, 17408, 512, 5120, {1, 1}, {1, 1}));

    // Flash attention perf: head 256 (default-on WMMA) vs 512/320/576 (DKQ != DV).
    // A/B with GGML_CUDA_FA_WMMA_256=0 (forces the tile kernel for head > 128 on RDNA4).
    for (const auto & fa : {std::tuple<int,int,int,int,int>{256, 256, 8, 4, 256},
                            {512, 512, 4, 8, 128},
                            {320, 256, 4, 32, 128},
                            {576, 512, 4, 4, 128},
                            {192, 128, 4, 8, 128},
                            {512, 512, 4, 8, 1},
                            {576, 512, 4, 4, 1}}) {
        const auto [hsk, hsv, nh, nr2, nb] = fa;
        test_cases.emplace_back(new test_flash_attn_ext(hsk, hsv, nh, {nr2, 1}, 16384, nb, true, false, 0, 0,
                                                        GGML_PREC_F32, GGML_TYPE_F16, GGML_TYPE_F16));
    }
"""


PATCHES = [
    FilePatch(
        path="ggml/src/ggml-cuda/fattn-mma-f16.cuh",
        description="WMMA FA config table + tile_Q reuse race fixes "
                    "(rdna-boosts 1d525bd45 / RD05+RD06)",
        edits=(
            Edit(
                id="rd0506-config-table",
                anchor=re.escape(_CONFIG_OLD),
                rationale="ggml_cuda_fattn_mma_get_config: RDNA4-tuned "
                          "configs for heads 256/320/512/576 (fork values)",
                mode="replace",
                text=_CONFIG_NEW,
                guard=r"GGML_CUDA_FATTN_MMA_CONFIG_CASE\(256, 256, 64, 256, 2,  32, 128, 128,  32, 1, true\);",
            ),
            Edit(
                id="rd05-k00-sync",
                anchor=re.escape(_K00_SYNC_OLD),
                rationale="flash_attn_ext_f16_process_tile: tile_Q reuse "
                          "race fix for np == 1 configs (fork logic)",
                mode="replace",
                text=_K00_SYNC_NEW,
                guard=r"if \(np > 1 \|\| k00 \+ nbatch_combine < DV/2\) \{",
            ),
            Edit(
                id="rd06-dkq-gate",
                anchor=re.escape(_DKQ_GATE_OLD),
                rationale="flash_attn_ext_f16: allow the WMMA path up to "
                          "head 576 (fork logic)",
                mode="replace",
                text=_DKQ_GATE_NEW,
                guard=r"DKQ > 576",
            ),
            Edit(
                id="rd05-kbc-sync",
                anchor=re.escape(_KBC_ANCHOR_OLD),
                rationale="flash_attn_ext_f16 kbc loop: sync between tile "
                          "iterations so the reused tile_Q buffer is not "
                          "overwritten (fork logic)",
                mode="replace",
                text=_KBC_ANCHOR_NEW,
                guard=r"reuses the tile_Q buffer for its Q/K tiles",
            ),
        ),
    ),
    FilePatch(
        path="ggml/src/ggml-cuda/fattn.cu",
        description="RDNA4 WMMA head gating + softcap read "
                    "(rdna-boosts 1d525bd45 / RD05+RD06)",
        edits=(
            Edit(
                id="rd0506-softcap-read",
                anchor=_SOFTCAP_OLD,
                rationale="ggml_cuda_get_best_fattn_kernel: read "
                          "logit_softcap from the op params (fork logic)",
                mode="replace",
                text=_SOFTCAP_NEW,
                guard=r"memcpy\(&logit_softcap, \(const float \*\) KQV->op_params \+ 2, sizeof\(float\)\);",
            ),
            Edit(
                id="rd06-wmma-gating",
                anchor=re.escape(_WMMA_OLD),
                rationale="ggml_cuda_get_best_fattn_kernel: enable the "
                          "WMMA path up to head 576 on RDNA4, opt-out via "
                          "GGML_CUDA_FA_WMMA_256=0, softcap guard (fork "
                          "logic)",
                mode="replace",
                text=_WMMA_NEW,
                guard=r"const int wmma_max_head = \(wmma_256 && GGML_CUDA_CC_IS_RDNA4\(cc\)\) \? 576 : 128;",
            ),
        ),
    ),
    FilePatch(
        path="ggml/src/ggml-cuda/mmq-vec-dot.cuh",
        description="Q6_K mmq sub-scale fold into row base-scale "
                    "(rdna-boosts 1d525bd45 / RD07)",
        edits=(
            Edit(
                id="rd07-hoist-base-scale",
                anchor=re.escape(_DF_HOIST_OLD),
                rationale="Q6_K mmq warp kernel: hoist the row base scales "
                          "out of the k01/j0 loops (fork logic)",
                mode="replace",
                text=_DF_HOIST_NEW,
                guard=r"float x_df_reg\[ntx\]\[tile_C::ne\];",
            ),
            Edit(
                id="rd07-fold-subscale",
                anchor=re.escape(_SC_FOLD_OLD),
                rationale="Q6_K mmq warp kernel: precompute the folded "
                          "sub-scale x base-scale per element (fork logic; "
                          "anchor rides on the hoist edit inserted above)",
                mode="replace",
                text=_SC_FOLD_NEW,
                guard=r"float x_s2_reg\[ntx\]\[tile_C::ne\];",
            ),
            Edit(
                id="rd07-sum-line",
                anchor=re.escape(_SUM_OLD),
                rationale="Q6_K mmq warp kernel: the j0 accumulation uses "
                          "the pre-folded scale (fork logic; keeps the "
                          "1000 float cast, equivalent form)",
                mode="replace",
                text=_SUM_NEW,
                guard=r"sum\[\(j0/tile_C::J \+ n\)\*tile_C::ne \+ l\] \+= \(float\) C\.x\[l\] \* x_s2_reg\[n\]\[l\] \* dB;",
            ),
        ),
    ),
    FilePatch(
        path="ggml/src/ggml-cuda/mmq.cuh",
        description="GGML_CUDA_MMQ_J_MAX override (rdna-boosts 1d525bd45 / "
                    "RD07 tooling)",
        edits=(
            Edit(
                id="rd07-jmax-env",
                anchor=re.escape(_JMAX_OLD),
                rationale="ggml_cuda_mmq_get_J_max: env override for the "
                          "J sweep (fork logic)",
                mode="replace",
                text=_JMAX_NEW,
                guard=r"const char \* env = getenv\(\"GGML_CUDA_MMQ_J_MAX\"\);",
            ),
        ),
    ),
    FilePatch(
        path="ggml/src/ggml-cuda/ggml-cuda.cu",
        description="Per-op timing instrumentation in the graph path "
                    "(rdna-boosts 1d525bd45 / RD07 tooling)",
        edits=(
            Edit(
                id="rd07-include-numeric",
                anchor=re.escape(_INCLUDE_OLD),
                rationale="include <numeric> for std::accumulate in the "
                          "timing summary (fork hunk)",
                mode="replace",
                text=_INCLUDE_NEW,
                guard=r"#include <numeric>",
            ),
            Edit(
                id="rd07-timing-head",
                anchor=_TIMING_HEAD_OLD,
                rationale="ggml_cuda_graph_evaluate_and_capture: create the "
                          "per-node events when GGML_CUDA_OP_TIMING is set "
                          "(fork logic)",
                mode="replace",
                text=_TIMING_HEAD_NEW,
                guard=r"std::vector<std::pair<const ggml_tensor \*, int>> op_nodes;",
            ),
            Edit(
                id="rd07-timing-mid0",
                anchor=re.escape(_TIMING_MID0_OLD),
                rationale="ggml_cuda_graph_evaluate_and_capture: record "
                          "the start event before the node runs (fork "
                          "logic)",
                mode="replace",
                text=_TIMING_MID0_NEW,
                guard=r"CUDA_CHECK\(cudaEventRecord\(op_ev0\[i\], cuda_ctx->stream\(\)\)\);",
            ),
            Edit(
                id="rd07-timing-mid1",
                anchor=re.escape(_TIMING_MID1_OLD),
                rationale="ggml_cuda_graph_evaluate_and_capture: record "
                          "the end event after the node ran (fork logic)",
                mode="replace",
                text=_TIMING_MID1_NEW,
                guard=r"CUDA_CHECK\(cudaEventRecord\(op_ev1\[i\], cuda_ctx->stream\(\)\)\);",
            ),
            Edit(
                id="rd07-timing-tail",
                anchor=_TIMING_TAIL_OLD,
                rationale="ggml_cuda_graph_evaluate_and_capture: per-op "
                          "timing summary + event teardown (fork logic)",
                mode="replace",
                text=_TIMING_TAIL_NEW,
                guard=r"op timing cumulative: %.2f ms over %d nodes",
            ),
        ),
    ),
    FilePatch(
        path="tests/test-backend-ops.cpp",
        description="Q6_K prefill + FA perf cases and the softcap matrix "
                    "(rdna-boosts 1d525bd45 / RD05+RD06+RD07 tooling)",
        edits=(
            Edit(
                id="rd0506-softcap-matrix",
                anchor=re.escape(_SOFTLOOP_OLD),
                rationale="FLASH_ATTN_EXT matrix: allow softcap for heads "
                          "256/512 now that the mma kernel instantiates "
                          "them (fork logic)",
                mode="replace",
                text=_SOFTLOOP_NEW,
                guard=r"if \(hsk != 128 && hsk != 256 && hsk != 512 && logit_softcap != 0\.0f\) continue;",
            ),
            Edit(
                id="rd07-perf-cases",
                anchor=re.escape(_PERF_ANCHOR_OLD),
                rationale="make_test_cases_perf: qwen35-27B Q6_K prefill "
                          "shapes + FA perf loop (position adaptation -- "
                          "after the HI70 direct-op corpus)",
                mode="replace",
                text=_PERF_NEW,
                guard=r"rdna-boosts \(RD05/06/07\): Qwen3.6-27B Q6_K prefill shapes",
            ),
        ),
    ),
]
