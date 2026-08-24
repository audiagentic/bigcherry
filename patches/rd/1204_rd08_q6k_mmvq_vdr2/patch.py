"""RD08: Q6_K mmvq VDR=2 decode kernel.

Provenance (group 'rdna-boosts' patches are external backports; the
machine-readable PROVENANCE dict below is cross-checked against
external-sources.toml by tools/tests/test_external_sources.py):

  source:            stew675-rdna-boosts
  repo:              https://github.com/stew675/llama.cpp
  locator:           rdna-boosts (branch name is a locator only, NOT identity)
  fork commit:       4591cc9806d24c5151a35007d24b69550df93e32
                     (snapshot v2; v1 ledger item 8, 25009e5f3, is the
                     pre-rebase identity of the SAME logical change,
                     content-identical per git patch-id)
                     "cuda : Q6_K mmvq VDR=2 decode kernel"
  reviewed snapshot: v2 -- head 9e46e1fdc7a880f9ae9a2f9a693ae3e14c142a22
                     on base 4df29be4f4c3673f428170fda944a5b19f743bb8
  plan item:         RD08 (docs/planning/active/rdna-boost-experiments/RD08.md)
  mainline status:   NOT merged into ggml-org/llama.cpp master as of tip
                     6d0549831 (git cherry patch-id check, 2026-08-18)

What it does (performance, kernel-level):
  Processes both 8-element chunks of a Q6_K vec_dot in one call so the
  ql/qh/scales/d8 loads are amortized over 4 dp4a ops instead of 2, and
  the thread block covers the same row with half the loop iterations.
  Switches get_vec_dot_q_cuda's Q6_K case to the new vdr2 entry point.
  Fork-reported: bit-identical to the VDR=1 kernel (Python simulation +
  1194/1194 MUL_MAT test-backend-ops), modest tg64 gains on gfx1201
  (23.35 -> 23.48 t/s d8192) since decode is DRAM-bound.
  Also: GGML_CUDA_OP_TIMING now disables CUDA graph capture (event
  records are illegal during capture; previously it aborted), plus
  decode-shaped (n=1) Q6_K/Q8_0 MUL_MAT perf test cases.

Porting notes:
  - Four files: vecdotq.cuh (the vdr2 impl + entry point, VDR define),
    mmvq.cu (the type switch line), ggml-cuda.cu (op_timing graph-capture
    guard), tests/test-backend-ops.cpp (decode perf cases).
  - Base co-tenancy: 0600/0650/0700 edit mmvq.cu launch/geometry, not the
    get_vec_dot_q_cuda switch; all anchors verified byte-identical on
    the framework-patched base 2026-08-19.

Isolation and promotion (first-sweep policy, RD review 2026-08-18):
  - GROUP 'rdna-boosts' + STATE 'untested' keeps this OUT of the
    production 'framework' and 'validated-enhancements' patch-sets.
  - Bench: Q6_K model decode; output equality vs VDR=1 is the primary
    gate (the fork claims bit-identity), plus the new n=1 perf cases.

Maintenance (future pin bumps / fork movement):
  - vecdotq.cuh is upstream-touched; re-derive from the tracked fork
    commit in external-sources.toml and run
    `python -m bigcherry sources check` before every pin bump.
"""

import re

from bigcherry.patcher import Edit, FilePatch

PROVENANCE = {
    "source-id": "stew675-rdna-boosts",
    "plan-item": "RD08",
    "fork-commit": "4591cc9806d24c5151a35007d24b69550df93e32",
    "fork-commit-title": "cuda : Q6_K mmvq VDR=2 decode kernel",
    "original-commit": "25009e5f3e5633d005b5c25b1b67b7f4e75a6355",
    "snapshot-head": "9e46e1fdc7a880f9ae9a2f9a693ae3e14c142a22",
    "snapshot-base": "4df29be4f4c3673f428170fda944a5b19f743bb8",
    "adaptations": [
        "tests/test-backend-ops.cpp: the decode perf cases are inserted "
        "after the HI70 direct-op corpus instead of the fork position -- "
        "the fork hunk's context lines were added by RD05/06/07 "
        "(1d525bd45), an experimental patch our base does not have. Same "
        "cases, same function, different position (test-only).",
    ],
}


# ----------------------------------------------------------------- vecdotq.cuh

_VDR_DEFINE_OLD = """#define VDR_Q6_K_Q8_1_MMVQ 1
#define VDR_Q6_K_Q8_1_MMQ  8"""

_VDR_DEFINE_NEW = """#define VDR_Q6_K_Q8_1_MMVQ 2
#define VDR_Q6_K_Q8_1_MMQ  8"""

# The "// contiguous v/x + u/y values" comment is blank space in the
# noise-stripped anchor view; the space-run class keeps the insertion
# BEFORE the comment, matching the fork layout.
_IMPL_ANCHOR = r"""    [ ]{20,40}
static __device__ __forceinline__ float vec_dot_q6_K_q8_1_impl_mmq\("""

_VDR2_IMPL = """// VDR=2 variant: processes two adjacent 8-element chunks (16 elements).
// Both chunks share the q8_1 block pair, the two sub-scales and the d8 values,
// so the loads are amortized over twice the dp4a work of the VDR=1 kernel.
static __device__ __forceinline__ float vec_dot_q6_K_q8_1_impl_mmvq_vdr2(
    const int & vl0, const int & vl1, const int & vh0, const int & vh1,
    const int * __restrict__ u, const int8_t * __restrict__ scales,
    const float & d, const float * __restrict__ d8) {

    float sumf = 0.0f;

#pragma unroll
    for (int i = 0; i < 2*QR6_K; ++i) {
        const int sc = scales[4*(i&1)];

        const int vl = (i < QR6_K) ? vl0 : vl1;
        const int vh = (i < QR6_K) ? vh0 : vh1;

        const int vil = (vl >> (4*(i&1))) & 0x0F0F0F0F;

        const int vih = ((vh >> (4*(i&1))) << 4) & 0x30303030;

        const int vi = __vsubss4((vil | vih), 0x20202020); // vi = (vil | vih) - 32

        sumf += d8[i&1] * (ggml_cuda_dp4a(vi, u[i], 0) * sc); // SIMD dot product
    }

    return d*sumf;
}

"""

_VDR2_ENTRY = """// VDR=2 entry point: iqs must be even (the mmvq kernel strides kqs by VDR).
// Processes 16 elements per call, splitting the ql/qh/u loads over two chunks.
static __device__ __forceinline__ float vec_dot_q6_K_q8_1_vdr2(
    const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1, const int & kbx, const int & iqs) {

    const block_q6_K * bq6_K = (const block_q6_K *) vbq + kbx;

    // both chunks share these offsets: iqs+1 lands in the same (iqs/16, (iqs%16)/8) cell
    const int bq8_offset = 2 * QR6_K * (iqs / (QI6_K/2)) + (iqs % (QI6_K/2)) / (QI6_K/4);
    const int scale_offset = (QI6_K/4) * (iqs / (QI6_K/2)) + (iqs % (QI6_K/2)) / (QI6_K/8);
    const int vh_shift = 2 * ((iqs % (QI6_K/2)) / (QI6_K/4));
    const int vh_idx = (QI6_K/4) * (iqs / (QI6_K/2)) + iqs % (QI6_K/4);

    const int vl0 = get_int_b4(bq6_K->ql, iqs);
    const int vl1 = get_int_b4(bq6_K->ql, iqs + 1);
    const int vh0 = get_int_b2(bq6_K->qh, vh_idx) >> vh_shift;
    const int vh1 = get_int_b2(bq6_K->qh, vh_idx + 1) >> vh_shift;

    const int8_t * scales = bq6_K->scales + scale_offset;

    int    u[2*QR6_K];
    float d8[QR6_K];

    // iqs is even, so iqs%QI8_1 and (iqs%QI8_1)+1 are adjacent int32 groups in the block
    const int i8 = iqs % QI8_1;
    u[0] = get_int_b4(bq8_1[bq8_offset + 0].qs, i8);
    u[1] = get_int_b4(bq8_1[bq8_offset + 2].qs, i8);
    u[2] = get_int_b4(bq8_1[bq8_offset + 0].qs, i8 + 1);
    u[3] = get_int_b4(bq8_1[bq8_offset + 2].qs, i8 + 1);
    d8[0] = __low2float(bq8_1[bq8_offset + 0].ds);
    d8[1] = __low2float(bq8_1[bq8_offset + 2].ds);

    return vec_dot_q6_K_q8_1_impl_mmvq_vdr2(vl0, vl1, vh0, vh1, u, scales, bq6_K->d, d8);
}

"""

_IQ2_ANCHOR = """#define VDR_IQ2_XXS_Q8_1_MMVQ 2
#define VDR_IQ2_XXS_Q8_1_MMQ  2"""

_IQ2_NEW = _VDR2_ENTRY + _IQ2_ANCHOR

# -------------------------------------------------------------------- mmvq.cu

_SWITCH_OLD = """        case GGML_TYPE_Q6_K:    return vec_dot_q6_K_q8_1;"""

_SWITCH_NEW = """        case GGML_TYPE_Q6_K:    return vec_dot_q6_K_q8_1_vdr2;"""

# The activation marker below needs std::atomic_flag. ggml-cuda.cu (where
# RD12's own marker lives) already #includes <atomic>; mmvq.cu does not, and
# common.cuh/mmvq.cuh (confirmed by direct inspection of the real vendored
# tree, 2026-08-21) pull in <cstdint>/<cstdlib>/<mutex>/<memory> but never
# <atomic> -- relying on it arriving transitively through <mutex> would be an
# unwarranted assumption about the standard library implementation, not a
# portable guarantee, so this patch adds the include explicitly.
_INCLUDES_OLD = """#include <cstdint>
#include <type_traits>"""

_INCLUDES_NEW = """#include <atomic>
#include <cstdint>
#include <type_traits>"""

# HI83 activation marker: mul_mat_vec_q_switch_ncols_dst's `case 1:` block is
# the real host-side launch point for the ncols_dst=1 mmvq path this patch's
# vdr2 kernel actually serves (confirmed against the real vendored file,
# 2026-08-21 -- this anchor is byte-identical there). `type` is this
# function's own template parameter (ggml_type type), so the if constexpr
# resolves at compile time and costs nothing for any other quant type. Same
# once-per-process GGML_LOG_INFO + BIGCHERRY_PATCH_TRACE pattern as
# patches/1205_rd12_paired_mmvq_dual_output.py's activation marker.
#
# The anchor is a REGEX (not a literal re.escape'd string): edits match
# against strip_noise()'d text, which blanks comment bodies to same-length
# whitespace while keeping the surrounding indentation and newlines intact
# (see csource.strip_noise / _IMPL_ANCHOR's own [ ]{...} class above for the
# same technique) -- so the literal "// static, else ..." comment text on the
# line after "case 1: {" is invisible to anchor matching and must be matched
# as a whitespace-only line instead. The replacement text below restores the
# real comment (it splices into the ORIGINAL, non-stripped text).
_ACTIVATION_ANCHOR = r"""        case 1: \{
[ ]{60,90}
            static constexpr int c_ncols_dst = 1;
"""

_ACTIVATION_MARKER = """        case 1: {
            // static, else MSVC lambda capture breaks the constexpr uses below
            static constexpr int c_ncols_dst = 1;

            // bigcherry: HI83 activation-evidence instrumentation, not part
            // of the ported fork change. Q6_K-gated so this only fires for
            // RD08's own kernel, not every ncols_dst=1 mmvq dispatch.
            if constexpr (type == GGML_TYPE_Q6_K) {
                if (getenv("BIGCHERRY_PATCH_TRACE") != nullptr) {
                    static std::atomic_flag bigcherry_rd08_logged = ATOMIC_FLAG_INIT;
                    if (!bigcherry_rd08_logged.test_and_set(std::memory_order_relaxed)) {
                        GGML_LOG_INFO("BIGCHERRY_PATCH_HIT patch=1204_rd08 path=q6k_mmvq_vdr2 ncols_dst=1\\n");
                    }
                }
            }
"""

# ------------------------------------------------------------- ggml-cuda.cu

_CAPTURE_HEAD_OLD = """    bool cuda_graph_update_required = false;
    const void * graph_key = nullptr;
"""

_CAPTURE_HEAD_NEW = """    bool cuda_graph_update_required = false;
    const void * graph_key = nullptr;

    // op timing instruments each node with stream events, which is not possible during capture
    const bool op_timing = getenv("GGML_CUDA_OP_TIMING") != nullptr;
"""

_CAPTURE_GATE_OLD = """    if (graph->is_enabled()) {"""

_CAPTURE_GATE_NEW = """    if (!op_timing && graph->is_enabled()) {"""

# --------------------------------------------------- tests/test-backend-ops.cpp

# POSITION ADAPTATION (deviation, recorded): the fork hunk anchors on the
# qwen35 27B ffn perf lines that RD05/06/07 (1d525bd45) added earlier in
# the branch -- our base does not have them (pin drift + the fork hunk's
# context depends on that experimental commit). The decode cases are
# inserted after BigCherry's own HI70 direct-op corpus instead; same test
# function, same cases, different position.
_PERF_ANCHOR = re.escape(
    "    test_cases.emplace_back(new test_mul_mat(GGML_TYPE_Q8_0, GGML_TYPE_F32, 127, 128, 256, {1, 1}, {1, 1}));\n")

_PERF_NEW = """    test_cases.emplace_back(new test_mul_mat(GGML_TYPE_Q8_0, GGML_TYPE_F32, 127, 128, 256, {1, 1}, {1, 1}));

    // rdna-boosts (RD08): Qwen3.6-27B decode shapes (n=1, mmvq path),
    // Q6_K vs Q8_0 -- the VDR=2 kernel's bench shapes.
    test_cases.emplace_back(new test_mul_mat(GGML_TYPE_Q6_K, GGML_TYPE_F32, 10240, 1, 5120, {1, 1}, {1, 1})); // ffn
    test_cases.emplace_back(new test_mul_mat(GGML_TYPE_Q6_K, GGML_TYPE_F32, 248320, 1, 5120, {1, 1}, {1, 1})); // lm_head
    test_cases.emplace_back(new test_mul_mat(GGML_TYPE_Q6_K, GGML_TYPE_F32, 12288, 1, 5120, {1, 1}, {1, 1})); // qkv
    test_cases.emplace_back(new test_mul_mat(GGML_TYPE_Q6_K, GGML_TYPE_F32, 6144, 1, 5120, {1, 1}, {1, 1})); // ssm z
    test_cases.emplace_back(new test_mul_mat(GGML_TYPE_Q6_K, GGML_TYPE_F32, 1024, 1, 5120, {1, 1}, {1, 1})); // kv proj
    test_cases.emplace_back(new test_mul_mat(GGML_TYPE_Q8_0, GGML_TYPE_F32, 10240, 1, 5120, {1, 1}, {1, 1})); // ffn
    test_cases.emplace_back(new test_mul_mat(GGML_TYPE_Q8_0, GGML_TYPE_F32, 248320, 1, 5120, {1, 1}, {1, 1})); // lm_head
    test_cases.emplace_back(new test_mul_mat(GGML_TYPE_Q8_0, GGML_TYPE_F32, 12288, 1, 5120, {1, 1}, {1, 1})); // qkv
    test_cases.emplace_back(new test_mul_mat(GGML_TYPE_Q8_0, GGML_TYPE_F32, 6144, 1, 5120, {1, 1}, {1, 1})); // ssm z
    test_cases.emplace_back(new test_mul_mat(GGML_TYPE_Q8_0, GGML_TYPE_F32, 1024, 1, 5120, {1, 1}, {1, 1})); // kv proj
"""


PATCHES = [
    FilePatch(
        path="ggml/src/ggml-cuda/vecdotq.cuh",
        description="Q6_K VDR=2 mmvq vec_dot impl + entry point "
                    "(rdna-boosts 4591cc980 / RD08)",
        edits=(
            Edit(
                id="rd08-vdr-define",
                anchor=re.escape(_VDR_DEFINE_OLD),
                rationale="VDR_Q6_K_Q8_1_MMVQ 1 -> 2: the mmvq kernel now "
                          "strides kqs by 2",
                mode="replace",
                text=_VDR_DEFINE_NEW,
                guard=r"#define VDR_Q6_K_Q8_1_MMVQ 2",
            ),
            Edit(
                id="rd08-vdr2-impl",
                anchor=_IMPL_ANCHOR,
                rationale="vecdotq.cuh: the vdr2 impl goes between the "
                          "VDR=1 impl and the mmq impl (fork layout)",
                # NOT mode="replace": _IMPL_ANCHOR is a REGEX (whitespace
                # class + escaped paren), not literal source text -- the
                # anchor matches real file content, but _IMPL_ANCHOR itself
                # is not that content. A prior version of this edit built
                # its replacement text as `_VDR2_IMPL + _IMPL_ANCHOR`, which
                # spliced the regex SOURCE (literally "[ ]{20,40}" and
                # "vec_dot_q6_K_q8_1_impl_mmq\(") into the file instead of
                # the real matched whitespace + mmq-impl declaration,
                # producing invalid C++ (confirmed via a real build attempt,
                # 2026-08-21: "expected unqualified-id" at the injected
                # regex text). insert_before preserves the actual matched
                # text untouched and only adds the new impl ahead of it.
                mode="insert_before",
                text=_VDR2_IMPL,
                guard=r"vec_dot_q6_K_q8_1_impl_mmvq_vdr2\(",
            ),
            Edit(
                id="rd08-vdr2-entry",
                anchor=re.escape(_IQ2_ANCHOR),
                rationale="vecdotq.cuh: the vdr2 entry point goes after "
                          "vec_dot_q6_K_q8_1, before the IQ2 section "
                          "(fork layout)",
                mode="replace",
                text=_IQ2_NEW,
                guard=r"vec_dot_q6_K_q8_1_vdr2\(",
            ),
        ),
    ),
    FilePatch(
        path="ggml/src/ggml-cuda/mmvq.cu",
        description="Route Q6_K in get_vec_dot_q_cuda to the vdr2 entry "
                    "(rdna-boosts 4591cc980 / RD08)",
        edits=(
            Edit(
                id="rd08-atomic-include",
                anchor=re.escape(_INCLUDES_OLD),
                rationale="mmvq.cu's activation marker below needs "
                          "std::atomic_flag; <atomic> is not already "
                          "included here or transitively via common.cuh",
                mode="replace",
                text=_INCLUDES_NEW,
                guard=r"#include <atomic>\n#include <cstdint>",
            ),
            Edit(
                id="rd08-switch-q6k",
                anchor=re.escape(_SWITCH_OLD),
                rationale="get_vec_dot_q_cuda: Q6_K now resolves to the "
                          "VDR=2 kernel",
                mode="replace",
                text=_SWITCH_NEW,
                guard=r"case GGML_TYPE_Q6_K:    return vec_dot_q6_K_q8_1_vdr2;",
            ),
            Edit(
                id="rd08-activation-marker",
                anchor=_ACTIVATION_ANCHOR,
                rationale="mul_mat_vec_q_switch_ncols_dst's case 1: block is "
                          "the ncols_dst=1 host-side launch point the vdr2 "
                          "kernel serves -- HI83 activation evidence",
                mode="replace",
                text=_ACTIVATION_MARKER,
                guard=r"BIGCHERRY_PATCH_HIT patch=1204_rd08",
            ),
        ),
    ),
    FilePatch(
        path="ggml/src/ggml-cuda/ggml-cuda.cu",
        description="GGML_CUDA_OP_TIMING disables CUDA graph capture "
                    "(rdna-boosts 4591cc980 / RD08)",
        edits=(
            Edit(
                id="rd08-op-timing-flag",
                anchor=re.escape(_CAPTURE_HEAD_OLD),
                rationale="ggml_backend_cuda_graph_compute: read the op "
                          "timing env var up front",
                mode="replace",
                text=_CAPTURE_HEAD_NEW,
                guard=r"const bool op_timing = getenv\(\"GGML_CUDA_OP_TIMING\"\) != nullptr;",
            ),
            Edit(
                id="rd08-op-timing-gate",
                anchor=re.escape(_CAPTURE_GATE_OLD),
                rationale="ggml_backend_cuda_graph_compute: skip graph "
                          "capture while op timing is active (events are "
                          "illegal during capture)",
                mode="replace",
                text=_CAPTURE_GATE_NEW,
                guard=r"if \(!op_timing && graph->is_enabled\(\)\) \{",
            ),
        ),
    ),
    FilePatch(
        path="tests/test-backend-ops.cpp",
        description="Decode-shaped (n=1) Q6_K/Q8_0 MUL_MAT perf cases "
                    "(rdna-boosts 4591cc980 / RD08)",
        edits=(
            Edit(
                id="rd08-perf-cases",
                anchor=_PERF_ANCHOR,
                rationale="make_test_cases_perf: qwen35-27B decode shapes "
                          "after the HI70 direct-op corpus (position "
                          "adaptation -- the fork's anchor context depends "
                          "on RD05/06/07, which our base lacks)",
                mode="replace",
                text=_PERF_NEW,
                guard=r"rdna-boosts \(RD08\): Qwen3\.6-27B decode shapes ",
            ),
        ),
    ),
]
