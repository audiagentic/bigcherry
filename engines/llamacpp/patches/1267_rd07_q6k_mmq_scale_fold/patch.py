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
_MMQ_DISPATCH_CALL = "            mul_mat_q_case<GGML_TYPE_Q6_K>(ctx, args, stream, forced_J);"
_MMQ_MARKER = """            // bigcherry: PRBE110/RD07 activation evidence, not source-port logic.
            if (getenv("BIGCHERRY_PATCH_TRACE") != nullptr) {
                static std::atomic_flag bigcherry_rd07_logged = ATOMIC_FLAG_INIT;
                if (!bigcherry_rd07_logged.test_and_set(std::memory_order_relaxed)) {
                    GGML_LOG_WARN("BIGCHERRY_PATCH_HIT patch=1267_rd07_q6k_mmq_scale_fold path=q6k_mmq_dispatch contract=PRBE110-RD07-Q6K-MMQ-SCALE-FOLD\\n");
                }
            }
"""

_JMAX_OLD = """    int ret = std::min(ne11, int64_t(512));
    ret -= ret % 8;"""
_JMAX_NEW = """    int ret = std::min(ne11, int64_t(512));
    ret -= ret % 8;
    const char * env = getenv("GGML_CUDA_MMQ_J_MAX");
    if (env != nullptr) {
        // GPT code review 2026-09-27 (req_6c90e1ebba83464a): plain atoi() on a
        // qualification-sweep override silently turned garbage/negative/non-
        // multiple-of-8 input into an out-of-range J (atoi("garbage")==0,
        // atoi("-1")==-1), which is exactly the tile-width invariant the
        // preceding `ret -= ret % 8` line exists to enforce. Parse strictly,
        // reject anything that fails to parse or isn't a positive integer,
        // and round the accepted value down to a multiple of 8 the same way
        // the unconditional default above is; an override that rounds to 0
        // is rejected outright rather than launching a degenerate J=0 tile.
        char * end = nullptr;
        const long parsed = std::strtol(env, &end, 10);
        if (end != env && *end == '\\0' && parsed > 0) {
            int j = (int) std::min<long>(parsed, ret);
            j -= j % 8;
            if (j > 0) {
                ret = j;
            }
        }
    }"""

_PERF_ANCHOR = """        test_cases.emplace_back(new test_l2_norm_batch(GGML_TYPE_F32, { n, 16, 16, 1 }, 4, 1e-12f, true));
    }"""
_PERF_CASES = """

    // bigcherry PRBE110/RD07: Q6_K MMQ target shapes plus non-target controls.
    test_cases.emplace_back(new test_mul_mat(GGML_TYPE_Q6_K, GGML_TYPE_F32, 17408, 512, 5120, {1, 1}, {1, 1}));
    test_cases.emplace_back(new test_mul_mat(GGML_TYPE_Q6_K, GGML_TYPE_F32, 5120,  512, 17408, {1, 1}, {1, 1}));
    test_cases.emplace_back(new test_mul_mat(GGML_TYPE_Q6_K, GGML_TYPE_F32, 10240, 512, 5120, {1, 1}, {1, 1}));
    test_cases.emplace_back(new test_mul_mat(GGML_TYPE_Q8_0, GGML_TYPE_F32, 17408, 512, 5120, {1, 1}, {1, 1}));
    test_cases.emplace_back(new test_mul_mat(GGML_TYPE_F16, GGML_TYPE_F32, 17408, 512, 5120, {1, 1}, {1, 1}));
    test_cases.emplace_back(new test_mul_mat(GGML_TYPE_F32, GGML_TYPE_F32, 17408, 512, 5120, {1, 1}, {1, 1}));"""

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
            Edit(id="rd07-activation-marker", anchor=re.escape(_MMQ_DISPATCH_CALL), mode="insert_before", text=_MMQ_MARKER,
                 guard=re.escape("BIGCHERRY_PATCH_HIT patch=1267_rd07_q6k_mmq_scale_fold"), rationale="Insert before the unique Q6_K dispatch call without consuming it, so 1006 can compose in either order.", expect_matches=1, max_span_lines=1),
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
        edits=(Edit(id="rd07-perf-cases", anchor=re.escape(_PERF_ANCHOR), mode="insert_after", text=_PERF_CASES,
                    guard=re.escape("bigcherry PRBE110/RD07: Q6_K MMQ target shapes"), rationale="Insert after make_test_cases_perf's unique final l2_norm_batch loop line+closing brace; the anchor remains intact for 1203/1269 in any order.", expect_matches=1, max_span_lines=2),),
    ),
]
