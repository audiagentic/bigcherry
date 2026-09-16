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
  - Seven files: fattn-mma-f16.cuh (config table + 2 sync fixes), fattn.cu
    (softcap read + RDNA4 WMMA head gating + RD05/RD06 activation
    markers), mmq-vec-dot.cuh (Q6_K sub-scale fold), mmq.cuh (J_MAX env),
    mmq.cu (RD07 activation marker only -- the fold itself is in
    mmq-vec-dot.cuh), ggml-cuda.cu (op timing),
    tests/test-backend-ops.cpp (perf cases + softcap matrix).

Activation evidence (PA37, added after GPT-confirmed host/device boundary
review, 2026-09-16, gpt-auto req_c2c26de482d845ad): RD05's barrier fix and
RD07's sub-scale fold both live entirely in device code with no safe
host-adjacent insertion point, so their BIGCHERRY_PATCH_TRACE markers sit
at the nearest real host-side dispatch site instead, mirroring
1204_rd08_q6k_mmvq_vdr2's precedent (once-per-process std::atomic_flag +
GGML_LOG_WARN, not GGML_LOG_INFO -- VA21 found llama-bench filters INFO):
  - RD05: fattn.cu's ggml_cuda_flash_attn_ext(), case
    BEST_FATTN_KERNEL_MMA_F16, immediately before the call into
    ggml_cuda_flash_attn_ext_mma_f16(). Unconditional -- fires whenever the
    WMMA-F16 kernel (which contains RD05's barrier fixes) is dispatched.
    Proves the fixed kernel executed, not that a race was avoided (that is
    backend-reference correctness evidence, per the RD05 contract).
    NOTE: GPT explicitly rejected an earlier proposal to place this marker
    inside ggml_cuda_get_best_fattn_kernel() (the selector) -- that
    function has non-launch callers too, so a marker there does not prove
    the kernel actually ran.
  - RD06: same call site as RD05, second independent marker, additionally
    gated on GGML_CUDA_CC_IS_RDNA4(cc) && Q->ne[0] > 128 -- proves the
    RDNA4-extended head ceiling (RD06's chooser change) was the reason
    this dispatch happened, not merely that some WMMA kernel ran.
  - RD07: mmq.cu's ggml_cuda_mul_mat_q_switch_type(), case
    GGML_TYPE_Q6_K, immediately before mul_mat_q_case<GGML_TYPE_Q6_K>().
    mmq.cu was not previously in 1203's edit set.
  - ADAPTATION (1000/1006 cast, corrected 2026-09-16): framework patch
    1000 (upstream PR #25940, combined Q2_K+Q6_K) would add a
    "((float) C.x[l])" cast to the mmq-vec-dot.cuh sum line, but its state
    is "rejected" -- not part of the applied patchset. Its Q6_K half was
    later split out as 1006_rdna4_mmq_q6k_codegen_fix (state "untested",
    NOT rejected), which independently inserts that same cast at the same
    site. Depending on which patch selection is composed ahead of 1203
    (1006 present or absent), the real current-pin base this edit's anchor
    sees carries the cast or does not -- both are real, both confirmed
    2026-09-16 against isolated worktrees at pin b10901 /
    28ff0958291ce3465fabd7bd679d4b0edd742bd9. The 'rd07-sum-line' anchor
    now matches either shape via a regex alternation (not re.escape() of a
    single literal); the replacement always applies its own explicit
    "(float)" cast to C.x[l], so the emitted promotion is semantically
    identical to the fork's final line regardless of which shape matched.
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
        "mmq-vec-dot.cuh: sum line anchor accepts either real base shape "
        "-- plain pin text (no cast; 1000/PR #25940 is rejected) or the "
        "cast already inserted by active co-tenant patch 1006 (split from "
        "1000's Q6_K half) -- via a regex alternation; replacement always "
        "applies its own equivalent single (float) cast.",
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

# RD05/RD06 activation markers need std::atomic_flag; fattn.cu does not
# already include <atomic> (confirmed against the real vendored file,
# 2026-09-16), same reasoning as 1204_rd08's mmvq.cu include addition.
_FATTN_INCLUDES_OLD = """#include "common.cuh"
#include "fattn-common.cuh"
#include "fattn-mma-f16.cuh"
#include "fattn-tile.cuh"
#include "fattn-vec.cuh"
#include "fattn.cuh"
"""

_FATTN_INCLUDES_NEW = """#include "common.cuh"
#include "fattn-common.cuh"
#include "fattn-mma-f16.cuh"
#include "fattn-tile.cuh"
#include "fattn-vec.cuh"
#include "fattn.cuh"

#include <atomic>
"""

# RD05/RD06 activation-evidence markers (PA37, GPT-confirmed placement,
# req_c2c26de482d845ad): ggml_cuda_flash_attn_ext() is the real host-side
# dispatcher -- its BEST_FATTN_KERNEL_MMA_F16 case is the actual launch
# site, unlike ggml_cuda_get_best_fattn_kernel() (the selector), which has
# non-launch callers and so cannot prove the kernel executed. Q and cc are
# read the same way ggml_cuda_flash_attn_ext_mma_f16() itself reads them
# a few lines later, so this costs one extra tensor/device-info read when
# BIGCHERRY_PATCH_TRACE is unset and is still gated by the atomic_flag once
# it is set. Same once-per-process BIGCHERRY_PATCH_TRACE / atomic_flag /
# GGML_LOG_WARN pattern as 1204_rd08's real precedent (GGML_LOG_INFO is
# filtered by llama-bench's default verbosity, per VA21).
_DISPATCH_OLD = """        case BEST_FATTN_KERNEL_MMA_F16:
            ggml_cuda_flash_attn_ext_mma_f16(ctx, dst);
            break;"""

_DISPATCH_NEW = """        case BEST_FATTN_KERNEL_MMA_F16: {
            // bigcherry: RD05/RD06 activation-evidence instrumentation,
            // not part of the ported fork change.
            if (getenv("BIGCHERRY_PATCH_TRACE") != nullptr) {
                static std::atomic_flag bigcherry_rd05_logged = ATOMIC_FLAG_INIT;
                if (!bigcherry_rd05_logged.test_and_set(std::memory_order_relaxed)) {
                    GGML_LOG_WARN("BIGCHERRY_PATCH_HIT patch=1203_rd050607 path=wmma_f16_dispatch contract=RD05\\n");
                }

                const int cc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;
                const ggml_tensor * Q = dst->src[0];
                if (GGML_CUDA_CC_IS_RDNA4(cc) && Q->ne[0] > 128) {
                    static std::atomic_flag bigcherry_rd06_logged = ATOMIC_FLAG_INIT;
                    if (!bigcherry_rd06_logged.test_and_set(std::memory_order_relaxed)) {
                        GGML_LOG_WARN("BIGCHERRY_PATCH_HIT patch=1203_rd050607 path=wmma_f16_dispatch contract=RD06\\n");
                    }
                }
            }

            ggml_cuda_flash_attn_ext_mma_f16(ctx, dst);
            break;
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

# Base line, two real upstream shapes (checked 2026-09-16 against the real
# pin b10901 / 28ff0958291ce3465fabd7bd679d4b0edd742bd9, isolated worktree):
#
#   1. Plain pin source (no other co-tenant patch applied first): the sum
#      line has NO explicit float cast --
#      "sum[...] += C.x[l] * sc[k01/4] * x_df[i*sram_stride] * dB;". This is
#      what a standalone/focal application of 1203 (its own REQUIRES closure
#      only, no framework companions) sees -- confirmed against a real
#      isolated subject worktree and matching the PA39 real-hardware
#      failure (attempt #2, commit 63b68e88) this anchor was originally
#      fixed for.
#   2. Pin source with active co-tenant patch 1006_rdna4_mmq_q6k_codegen_fix
#      (order 1006 < 1203's order, state "untested" -- NOT rejected; the
#      rejected patch at this same site is 1000_rdna4_mmq_q2k_q6k_fix, a
#      different id) applied first: 1006 independently inserts its own
#      "((float) C.x[l])" cast at this exact line (see
#      patches/1006_rdna4_mmq_q6k_codegen_fix/patch.py). Any full-registry
#      composition that includes 1006 ahead of 1203 (e.g.
#      `patch-rebase-check --all`) presents this cast shape to 1203.
#
# Both are real, both are reachable depending on which patch selection is
# composed, and both are semantically identical to what the replacement
# below computes -- so the anchor accepts either explicitly rather than
# assuming one caller's selection is the only one that matters. This is a
# regex alternation on the anchor text, not re.escape(), since re.escape()
# cannot express "one of two literal shapes".
_SUM_OLD_PLAIN = ("sum[(j0/tile_C::J + n)*tile_C::ne + l] += "
                  "C.x[l] * sc[k01/4] * x_df[i*sram_stride] * dB;")
_SUM_OLD_CAST = ("sum[(j0/tile_C::J + n)*tile_C::ne + l] += "
                 "((float) C.x[l]) * sc[k01/4] * x_df[i*sram_stride] * dB;")

_SUM_OLD = re.escape("""                for (int l = 0; l < tile_C::ne; ++l) {
                    const int i = i0 + n*tile_C::I + tile_C::get_i(l);
                    const int8_t * sc = (const int8_t *) (x_sc + i*sram_stride + k00/16);
                    sum[(j0/tile_C::J + n)*tile_C::ne + l] += """) + (
    r"(?:" + re.escape(_SUM_OLD_CAST.split("+= ", 1)[1]) + r"|"
    + re.escape(_SUM_OLD_PLAIN.split("+= ", 1)[1]) + r")"
) + re.escape("""
                }""")

_SUM_NEW = """                for (int l = 0; l < tile_C::ne; ++l) {
                    sum[(j0/tile_C::J + n)*tile_C::ne + l] += (float) C.x[l] * x_s2_reg[n][l] * dB;
                }"""

# ---------------------------------------------------------------------- mmq.cu

# RD07 activation marker: the sub-scale fold itself (mmq-vec-dot.cuh) is
# device code with no safe host-adjacent insertion point (confirmed same
# as the prior session's finding), so the marker sits at the nearest real
# host-side Q6_K MMQ dispatch site instead -- structurally identical to
# 1204_rd08's real mul_mat_vec_q_switch_ncols_dst precedent (GPT-confirmed
# placement, req_c2c26de482d845ad). mmq.cu is not otherwise touched by
# 1203 and needs <atomic> added, same reasoning as mmvq.cu in 1204_rd08.
_MMQ_INCLUDES_OLD = """#include <cstdint>

static void ggml_cuda_mul_mat_q_switch_type(ggml_backend_cuda_context & ctx, const mmq_args & args,"""

_MMQ_INCLUDES_NEW = """#include <atomic>
#include <cstdint>

static void ggml_cuda_mul_mat_q_switch_type(ggml_backend_cuda_context & ctx, const mmq_args & args,"""

_MMQ_SWITCH_OLD = """        case GGML_TYPE_Q6_K:
            mul_mat_q_case<GGML_TYPE_Q6_K>(ctx, args, stream, forced_J);
            break;"""

_MMQ_SWITCH_NEW = """        case GGML_TYPE_Q6_K: {
            // bigcherry: RD07 activation-evidence instrumentation, not
            // part of the ported fork change. Proves the Q6_K MMQ
            // specialization (which contains RD07's folded sub-scale
            // kernel) was dispatched, not that the fold instruction itself
            // executed.
            if (getenv("BIGCHERRY_PATCH_TRACE") != nullptr) {
                static std::atomic_flag bigcherry_rd07_logged = ATOMIC_FLAG_INIT;
                if (!bigcherry_rd07_logged.test_and_set(std::memory_order_relaxed)) {
                    GGML_LOG_WARN("BIGCHERRY_PATCH_HIT patch=1203_rd050607 path=q6k_mmq_dispatch contract=RD07\\n");
                }
            }

            mul_mat_q_case<GGML_TYPE_Q6_K>(ctx, args, stream, forced_J);
            break;
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
        description="RDNA4 WMMA head gating + softcap read + RD05/RD06 "
                    "activation markers (rdna-boosts 1d525bd45 / RD05+RD06)",
        edits=(
            Edit(
                id="rd0506-atomic-include",
                anchor=re.escape(_FATTN_INCLUDES_OLD),
                rationale="fattn.cu's RD05/RD06 activation markers need "
                          "std::atomic_flag; <atomic> is not already "
                          "included here or transitively",
                mode="replace",
                text=_FATTN_INCLUDES_NEW,
                guard=r"#include <atomic>",
            ),
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
            Edit(
                id="rd0506-activation-markers",
                anchor=re.escape(_DISPATCH_OLD),
                rationale="ggml_cuda_flash_attn_ext: RD05 (unconditional) "
                          "and RD06 (RDNA4 + head>128) activation evidence "
                          "at the real WMMA-F16 launch site -- PA37, "
                          "GPT-confirmed (req_c2c26de482d845ad)",
                mode="replace",
                text=_DISPATCH_NEW,
                guard=r"BIGCHERRY_PATCH_HIT patch=1203_rd050607 path=wmma_f16_dispatch contract=RD06",
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
                anchor=_SUM_OLD,
                rationale="Q6_K mmq warp kernel: the j0 accumulation uses "
                          "the pre-folded scale (fork logic; matches the "
                          "sum line whether or not co-tenant patch 1006 "
                          "already inserted its own float cast here)",
                mode="replace",
                text=_SUM_NEW,
                guard=r"sum\[\(j0/tile_C::J \+ n\)\*tile_C::ne \+ l\] \+= \(float\) C\.x\[l\] \* x_s2_reg\[n\]\[l\] \* dB;",
            ),
        ),
    ),
    FilePatch(
        path="ggml/src/ggml-cuda/mmq.cu",
        description="RD07 activation marker at the Q6_K MMQ host-side "
                    "dispatch site (PA37, not part of the ported fork "
                    "change)",
        edits=(
            Edit(
                id="rd07-atomic-include",
                anchor=re.escape(_MMQ_INCLUDES_OLD),
                rationale="mmq.cu's RD07 activation marker needs "
                          "std::atomic_flag; <atomic> is not already "
                          "included here",
                mode="replace",
                text=_MMQ_INCLUDES_NEW,
                guard=r"#include <atomic>\n#include <cstdint>",
            ),
            Edit(
                id="rd07-activation-marker",
                anchor=re.escape(_MMQ_SWITCH_OLD),
                rationale="ggml_cuda_mul_mat_q_switch_type: RD07 activation "
                          "evidence at the real Q6_K MMQ dispatch site -- "
                          "PA37, GPT-confirmed (req_c2c26de482d845ad), "
                          "mirrors 1204_rd08's real precedent",
                mode="replace",
                text=_MMQ_SWITCH_NEW,
                guard=r"BIGCHERRY_PATCH_HIT patch=1203_rd050607 path=q6k_mmq_dispatch contract=RD07",
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
