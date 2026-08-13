"""Upstream backport: RDNA4 MMQ codegen fixes for Q2_K and Q6_K.

Cherry-picked from an unmerged upstream PR
(https://github.com/ggml-org/llama.cpp/pull/25940, still ``OPEN`` as of this
writing) rather than waiting for it to land. Two independent, narrowly-scoped
fixes in ``ggml_cuda_mmq_vec_dot_q2_K_q8_1_mma`` and
``ggml_cuda_mmq_vec_dot_q6_K_q8_1_mma`` -- the MFMA/WMMA tile-based dot
product used by RDNA3+/CDNA hardware -- that the PR author attributes to
ROCm's compiler, not to the kernel's own logic:

* Q2_K: the inner ``k01`` loop over-unrolls on ROCm's LLVM and spills into
  scratch memory. ``#pragma unroll 1`` forces it back to a plain loop.
* Q6_K: ``sum[...] += C.x[l] * sc[k01/4] * ...`` multiplies an ``int`` tile
  value against an ``int8_t`` scale without an explicit float promotion
  first; an explicit ``(float)`` cast on ``C.x[l]`` is enough to change the
  code ROCm generates.

The PR's own ``test-backend-ops perf -o MUL_MAT`` numbers (RDNA4, ROCm
7.15/TheRock 20260717): Q6_K n=512 36.62 -> 69.66 tok/s (1.90x), Q2_K n=512
2.51 -> 70.78 tok/s (28.2x). Both quant types are in this project's own test
corpus (``tierA-qwen4b-q6k``, ``tierM-gptoss20b-q6k``), and RDNA4 (gfx1201)
is hardware this project tunes on directly, so the fix is worth taking now
rather than waiting on upstream review.

Deliberately **not** backported: the PR's second change, narrowing
``ggml_cuda_should_use_mmq``'s RDNA4 heuristic from "always force MMQ" to a
table of per-type ``ne11`` thresholds. That heuristic only feeds this
project's ``*:native:v1`` reference candidates (see
``ggml_hip_native_select`` in ``hip-autotune-dispatch.cu``); real ``mmq`` and
``blas`` candidate eligibility never consulted it, and the tuner already
measures both families head-to-head per exact shape and hardware -- strictly
better than any hand-written cutoff table, upstream's or ours.

Grouped separately from the dispatch-engine patches (``GROUP =
"upstream-fixes"``, vs. their implicit ``"core"``) so a build can include or
exclude it independently: `--groups core` isolates the tuning engine's own
effect from this fix, `--groups upstream-fixes` isolates the fix's effect
against raw stock, and the default (no `--groups`) takes both for a final
binary.
"""

from bigcherry.patcher import Edit, FilePatch

GROUP = "upstream-fixes"
STATE = "validated"

PATCH = FilePatch(
    path="ggml/src/ggml-cuda/mmq-vec-dot.cuh",
    description="RDNA4: unroll guard for Q2_K MMA dot product, float "
                "promotion for Q6_K MMA dot product (upstream PR #25940)",
    edits=(
        Edit(
            id="q2k-mma-unroll-guard",
            # Three functions in this file open a k01 loop with this exact,
            # four-space-indented line (a fourth, more deeply nested
            # occurrence exists but does not match this anchor). Only the
            # second (ggml_cuda_mmq_vec_dot_q2_K_q8_1_mma, the MFMA/WMMA
            # variant) is the one the upstream PR's own description names
            # ("Q2_K loop unrolls and spill into scratch"). expect_matches
            # pins the total so upstream adding, removing, or re-indenting one
            # of the three is caught rather than silently mis-targeted.
            anchor=r"^    for \(int k01 = 0; k01 < MMQ_TILE_NE_K; k01 \+= 4\) \{$",
            expect_matches=3,
            occurrence=1,
            rationale="the k01 loop inside ggml_cuda_mmq_vec_dot_q2_K_q8_1_mma "
                      "(second of three matching occurrences)",
            mode="insert_before",
            text="    #pragma unroll 1\n",
            guard=r"#pragma unroll 1\n    for \(int k01 = 0; k01 < MMQ_TILE_NE_K; k01 \+= 4\) \{",
        ),
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

PATCHES = [PATCH]
