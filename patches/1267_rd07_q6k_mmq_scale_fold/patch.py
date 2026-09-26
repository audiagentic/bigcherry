"""PRBE110/RD07: extract the Q6_K MMQ scale-fold optimization from rejected 1203.

Only RD07-owned kernel changes, the J_MAX test knob, a Q6_K-only perf corpus,
and local activation instrumentation are retained. RD05/RD06 and op-timing
instrumentation are intentionally excluded.
"""

import re

from bigcherry.patcher import Edit, FilePatch

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

        // Fold the sub-scale and row base scale once per element.
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

_SUM_OLD_PLAIN = "sum[(j0/tile_C::J + n)*tile_C::ne + l] += C.x[l] * sc[k01/4] * x_df[i*sram_stride] * dB;"
_SUM_OLD_CAST = "sum[(j0/tile_C::J + n)*tile_C::ne + l] += ((float) C.x[l]) * sc[k01/4] * x_df[i*sram_stride] * dB;"
_SUM_OLD = re.escape("""                for (int l = 0; l < tile_C::ne; ++l) {
                    const int i = i0 + n*tile_C::I + tile_C::get_i(l);
                    const int8_t * sc = (const int8_t *) (x_sc + i*sram_stride + k00/16);
                    sum[(j0/tile_C::J + n)*tile_C::ne + l] += """) + (
    r"(?:" + re.escape(_SUM_OLD_CAST.split("+= ", 1)[1]) + r"|" + re.escape(_SUM_OLD_PLAIN.split("+= ", 1)[1]) + r")"
) + re.escape("""
                }""")
_SUM_NEW = """                for (int l = 0; l < tile_C::ne; ++l) {
                    sum[(j0/tile_C::J + n)*tile_C::ne + l] += (float) C.x[l] * x_s2_reg[n][l] * dB;
                }"""

_MMQ_INCLUDES_OLD = """#include <cstdint>

static void ggml_cuda_mul_mat_q_switch_type(ggml_backend_cuda_context & ctx, const mmq_args & args,"""
_MMQ_INCLUDES_NEW = """#include <atomic>
#include <cstdint>

static void ggml_cuda_mul_mat_q_switch_type(ggml_backend_cuda_context & ctx, const mmq_args & args,"""
_MMQ_SWITCH_OLD = """        case GGML_TYPE_Q6_K:
            mul_mat_q_case<GGML_TYPE_Q6_K>(ctx, args, stream, forced_J);
            break;"""
_MMQ_SWITCH_NEW = """        case GGML_TYPE_Q6_K: {
            // bigcherry: PRBE110/RD07 activation evidence, not source-port logic.
            if (getenv("BIGCHERRY_PATCH_TRACE") != nullptr) {
                static std::atomic_flag bigcherry_rd07_logged = ATOMIC_FLAG_INIT;
                if (!bigcherry_rd07_logged.test_and_set(std::memory_order_relaxed)) {
                    GGML_LOG_WARN("BIGCHERRY_PATCH_HIT patch=1267_rd07_q6k_mmq_scale_fold path=q6k_mmq_dispatch contract=PRBE110-RD07-Q6K-MMQ-SCALE-FOLD\\n");
                }
            }
            mul_mat_q_case<GGML_TYPE_Q6_K>(ctx, args, stream, forced_J);
            break;
        }"""

_JMAX_OLD = """    int ret = std::min(ne11, int64_t(512));
    ret -= ret % 8;"""
_JMAX_NEW = """    int ret = std::min(ne11, int64_t(512));
    ret -= ret % 8;
    const char * env = getenv("GGML_CUDA_MMQ_J_MAX");
    if (env != nullptr) {
        ret = std::min(ret, std::atoi(env));
    }"""

_PERF_OLD = """        test_cases.emplace_back(new test_l2_norm_batch(GGML_TYPE_F32, { n, 16, 16, 1 }, 4, 1e-12f, true));
    }


    return test_cases;
}"""
_PERF_NEW = """        test_cases.emplace_back(new test_l2_norm_batch(GGML_TYPE_F32, { n, 16, 16, 1 }, 4, 1e-12f, true));
    }

    // bigcherry PRBE110/RD07: Q6_K MMQ target shapes plus non-target controls.
    test_cases.emplace_back(new test_mul_mat(GGML_TYPE_Q6_K, GGML_TYPE_F32, 17408, 512, 5120, {1, 1}, {1, 1}));
    test_cases.emplace_back(new test_mul_mat(GGML_TYPE_Q6_K, GGML_TYPE_F32, 5120,  512, 17408, {1, 1}, {1, 1}));
    test_cases.emplace_back(new test_mul_mat(GGML_TYPE_Q6_K, GGML_TYPE_F32, 10240, 512, 5120, {1, 1}, {1, 1}));
    test_cases.emplace_back(new test_mul_mat(GGML_TYPE_Q8_0, GGML_TYPE_F32, 17408, 512, 5120, {1, 1}, {1, 1}));
    test_cases.emplace_back(new test_mul_mat(GGML_TYPE_F16, GGML_TYPE_F32, 17408, 512, 5120, {1, 1}, {1, 1}));
    test_cases.emplace_back(new test_mul_mat(GGML_TYPE_F32, GGML_TYPE_F32, 17408, 512, 5120, {1, 1}, {1, 1}));

    return test_cases;
}"""

PATCHES = [
    FilePatch(
        path="ggml/src/ggml-cuda/mmq-vec-dot.cuh",
        description="RD07 Q6_K MMQ scale hoist/fold",
        edits=(
            Edit(id="rd07-hoist-base-scale", anchor=re.escape(_DF_HOIST_OLD), mode="replace", text=_DF_HOIST_NEW,
                 guard=r"float x_df_reg\[ntx\]\[tile_C::ne\];", rationale="Hoist invariant Q6_K row base scales.", expect_matches=1, max_span_lines=6),
            Edit(id="rd07-fold-subscale", anchor=re.escape(_SC_FOLD_OLD), mode="replace", text=_SC_FOLD_NEW,
                 guard=r"float x_s2_reg\[ntx\]\[tile_C::ne\];", rationale="Fold per-k01 Q6_K sub-scales with hoisted row scales.", expect_matches=1, max_span_lines=16),
            Edit(id="rd07-sum-line", anchor=_SUM_OLD, mode="replace", text=_SUM_NEW,
                 guard=r"x_s2_reg\[n\]\[l\] \* dB;", rationale="Use the pre-folded scale in the Q6_K accumulation.", expect_matches=1, max_span_lines=5),
        ),
    ),
    FilePatch(
        path="ggml/src/ggml-cuda/mmq.cu",
        description="RD07 activation marker at composed forced-J Q6_K MMQ dispatch",
        edits=(
            Edit(id="rd07-atomic-include", anchor=re.escape(_MMQ_INCLUDES_OLD), mode="replace", text=_MMQ_INCLUDES_NEW,
                 guard=r"#include <atomic>", rationale="Support once-per-process activation instrumentation.", expect_matches=1, max_span_lines=3),
            Edit(id="rd07-activation-marker", anchor=re.escape(_MMQ_SWITCH_OLD), mode="replace", text=_MMQ_SWITCH_NEW,
                 guard=re.escape("BIGCHERRY_PATCH_HIT patch=1267_rd07_q6k_mmq_scale_fold"), rationale="Anchor after 0300_mmq_forced_j and prove the optimized Q6_K specialization was dispatched.", expect_matches=1, max_span_lines=3),
        ),
    ),
    FilePatch(
        path="ggml/src/ggml-cuda/mmq.cuh",
        description="RD07 J_MAX test/tuning override",
        edits=(Edit(id="rd07-jmax-env", anchor=re.escape(_JMAX_OLD), mode="replace", text=_JMAX_NEW,
                    guard=re.escape("GGML_CUDA_MMQ_J_MAX"), rationale="Allow bounded J sweeps for RD07 qualification.", expect_matches=1, max_span_lines=2),),
    ),
    FilePatch(
        path="tests/test-backend-ops.cpp",
        description="RD07-only Q6_K MMQ performance cases",
        edits=(Edit(id="rd07-perf-cases", anchor=re.escape(_PERF_OLD), mode="replace", text=_PERF_NEW,
                    guard=re.escape("bigcherry PRBE110/RD07: Q6_K MMQ target shapes"), rationale="Add only RD07 MMQ target/control cases; exclude RD05/RD06 FA cases.", expect_matches=1, max_span_lines=7),),
    ),
]
