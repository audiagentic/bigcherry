"""Upstream backport: RDNA4 MMQ Q6_K codegen fix (split from 1000).

Extracted 2026-09-16 (PA35) from the rejected combined patch
``1000_rdna4_mmq_q2k_q6k_fix`` after real hardware evidence found the Q6_K
half genuinely improves (1.365x, CI95 1.362x-1.367x, statistically
significant) while the Q2_K half regresses (0.959x, CI95 0.954x-0.963x) --
see ``patches/1000_rdna4_mmq_q2k_q6k_fix/SUMMARY.md``'s DEMOTION section for
the full disposition record and GPT design-review reference
(dev-gpt-agent, req_24ceb64100cb41dd). The two edits were already
structurally independent in 1000's ``patch.py``, so this extraction is a
plain split, not a reimplementation.

Cherry-picked from an unmerged upstream PR
(https://github.com/ggml-org/llama.cpp/pull/25940, still ``OPEN`` as of this
writing). ``ggml_cuda_mmq_vec_dot_q6_K_q8_1_mma`` -- the MFMA/WMMA tile-based
dot product used by RDNA3+/CDNA hardware -- multiplies an ``int`` tile value
against an ``int8_t`` scale without an explicit float promotion first; an
explicit ``(float)`` cast on ``C.x[l]`` is enough to change the code ROCm
generates.

The PR's own ``test-backend-ops perf -o MUL_MAT`` numbers (RDNA4, ROCm
7.15/TheRock 20260717): Q6_K n=512 36.62 -> 69.66 tok/s (1.90x) -- retained
here as the historical upstream claim. This project's own combined-package
real hardware measurement (2026-09-16, gfx1201, exact-shape backend-ops
paired A/B) found 1.365x [CI95 1.362x-1.367x], a real but smaller gain than
claimed. That measurement was made against the combined 1000 composition
(Q2_K change also present) and is NOT reused as sufficient promotion
evidence here -- this patch's own composition identity differs (Q6_K only),
so ``state`` starts at ``"untested"`` pending a fresh Q6-only hardware A/B
before any promotion to ``"validated"``.

Deliberately **not** backported: the PR's Q2_K unroll-guard change (real,
statistically significant regression on this project's hardware -- see
1000's rejection) and its second change (a hand-written RDNA4 native-select
heuristic narrowing ``ggml_cuda_should_use_mmq``'s heuristic), for the same
reason 1000 originally excluded it: this project's tuner already measures
candidates head-to-head per exact shape and hardware.
"""

import re

from bigcherry.patcher import Edit, FilePatch

GROUP = "rdna-boosts"
STATE = "untested"

PATCH = FilePatch(
    path="ggml/src/ggml-cuda/mmq-vec-dot.cuh",
    description="RDNA4: float promotion for Q6_K MMA dot product "
                "(upstream PR #25940, split from 1000)",
    edits=(
        Edit(
            id="q6k-mma-float-promotion",
            # Unique in the file: the only occurrence of this exact
            # multiplicand chain (other sum[...] += C.x[l] * ... lines in this
            # file multiply against different operands entirely).
            anchor=r"sum\[\(j0/tile_C::J \+ n\)\*tile_C::ne \+ l\] \+= "
                   r"C\.x\[l\] \* sc\[k01/4\] \* x_df\[i\*sram_stride\] \* dB;",
            rationale="the Q6_K MMA accumulation inside "
                      "ggml_cuda_mmq_vec_dot_q6_K_q8_1_mma",
            mode="replace",
            text="sum[(j0/tile_C::J + n)*tile_C::ne + l] += "
                 "((float) C.x[l]) * sc[k01/4] * x_df[i*sram_stride] * dB;",
            guard=r"\(\(float\) C\.x\[l\]\) \* sc\[k01/4\] \* x_df\[i\*sram_stride\] \* dB;",
        ),
    ),
)

# RDNA4-MMQ-Q6K-CODEGEN (validation contract, added when authoring the
# contract itself) requires activation evidence, but the Q6_K float
# promotion above lives inside a __device__ kernel body (no host I/O, so a
# getenv()-gated marker cannot live there). Following 1267/RD07's own
# precedent exactly (same dispatch site, same file): instrument the HOST
# dispatch switch in ggml_cuda_mul_mat_q_switch_type (mmq.cu) at its
# case GGML_TYPE_Q6_K: branch -- proves the Q6_K MMQ path was actually
# dispatched, once per process, only under BIGCHERRY_PATCH_TRACE=1.
_MMQ_INCLUDES_OLD = """#include <cstdint>

static void ggml_cuda_mul_mat_q_switch_type(ggml_backend_cuda_context & ctx, const mmq_args & args,"""
_MMQ_INCLUDES_NEW = """#include <atomic>
#include <cstdint>

static void ggml_cuda_mul_mat_q_switch_type(ggml_backend_cuda_context & ctx, const mmq_args & args,"""

_MMQ_SWITCH_OLD = """        case GGML_TYPE_Q6_K:
            mul_mat_q_case<GGML_TYPE_Q6_K>(ctx, args, stream, forced_J);
            break;"""
_MMQ_SWITCH_NEW = """        case GGML_TYPE_Q6_K: {
            // bigcherry: RDNA4-MMQ-Q6K-CODEGEN activation evidence, not source-port logic.
            if (getenv("BIGCHERRY_PATCH_TRACE") != nullptr) {
                static std::atomic_flag bigcherry_1006_logged = ATOMIC_FLAG_INIT;
                if (!bigcherry_1006_logged.test_and_set(std::memory_order_relaxed)) {
                    GGML_LOG_WARN("BIGCHERRY_PATCH_HIT patch=1006_rdna4_mmq_q6k_codegen_fix path=q6k_mmq_dispatch contract=RDNA4-MMQ-Q6K-CODEGEN\\n");
                }
            }
            mul_mat_q_case<GGML_TYPE_Q6_K>(ctx, args, stream, forced_J);
            break;
        }"""

ACTIVATION_PATCH = FilePatch(
    path="ggml/src/ggml-cuda/mmq.cu",
    description="RDNA4-MMQ-Q6K-CODEGEN: activation evidence at the Q6_K MMQ dispatch site",
    edits=(
        Edit(
            id="mmq-includes-atomic",
            anchor=re.escape(_MMQ_INCLUDES_OLD),
            rationale="add <atomic> for the once-per-process activation flag",
            mode="replace",
            text=_MMQ_INCLUDES_NEW,
            guard=r"#include <atomic>",
            max_span_lines=3,
        ),
        Edit(
            id="mmq-q6k-dispatch-marker",
            # Unique in the file: ggml_cuda_mul_mat_q_switch_type's own
            # Q6_K case, not any of the other switch statements in mmq.cu
            # that also branch on GGML_TYPE_Q6_K (verified anchor text
            # includes the specific mul_mat_q_case<GGML_TYPE_Q6_K> call).
            anchor=re.escape(_MMQ_SWITCH_OLD),
            rationale="prove the Q6_K MMQ path was actually dispatched",
            mode="replace",
            text=_MMQ_SWITCH_NEW,
            guard=re.escape("BIGCHERRY_PATCH_HIT patch=1006_rdna4_mmq_q6k_codegen_fix"),
            expect_matches=1,
            max_span_lines=3,
        ),
    ),
)

PATCHES = [PATCH, ACTIVATION_PATCH]
