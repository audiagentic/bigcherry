"""Widen the MMVQ MUL_MAT+GLU fusion gate from ncols_dst==1 to the whole
MMVQ batch range, so fusion actually fires on MTP speculative-decode shapes.

WHY (real profiling evidence, 2026-09-04, dual gfx1100 / Qwen3.8-27B-Q8_0 /
-sm tensor / MTP spec_draft_n_max=5):

A rocprofv3 decode capture (5.9s window, 632,620 dispatches, union-of-spans)
broke the 123,090 MMVQ dispatches down by template parameters. The MMVQ
template is ``mul_mat_vec_q<type, ncols_dst, has_fusion, small_k,
halve_iters>``, and the observed distribution is:

    112,110 (91%)  ncols=6  has_fusion=false
      7,848        ncols=1  has_fusion=false
      2,000        ncols=2  has_fusion=false
      1,120        ncols=1  has_fusion=TRUE
      1,010        ncols=3  has_fusion=false
      1,008        ncols=4  has_fusion=false

i.e. MUL_MAT+GLU fusion fires on 1,120 of 123,090 dispatches -- 0.9% -- and
only ever at ncols_dst==1.

This workload runs at ncols_dst==6 because MTP speculative decoding verifies
spec_draft_n_max=5 drafted tokens + 1. So upstream's fusion is structurally
disabled on 91% of decode work. The same capture shows the consequences
directly: all 20,561 same-shape consecutive-MMVQ pairs per GPU carry a
``quantize_q8_1`` between them (pattern ``quantize -> MMVQ -> quantize ->
MMVQ``) instead of one fused op, 7,345 of those pairs are immediately
followed by ``unary_gated_op_kernel`` (FFN gate/up SwiGLU -- the textbook
fusable shape), and there are 30,500 separate ``unary_gated_op_kernel``
dispatches (38.2ms) that fusion would have absorbed.

The same ncols_dst==1 assumption is also why patch 1241 (RD33) measured a
null result: its own eligibility gate is ncols_dst==1, so it never fired on
~93% of MMVQ work either. This is one root cause behind several stalled
items, not a coincidence.

WHY WIDENING IS PLAUSIBLE (source-verified), AND THE REAL RISK:

The fused kernel body is written generically over ncols_dst. In
``mul_mat_vec_q`` (mmvq.cu): ``x_biases[ncols_dst]`` / ``gate_biases[ncols_dst]``
are sized by the template parameter, and the bias/gate-bias prefetch, the
per-column accumulation, and the GLU combine (``switch (active_glu)``) all sit
inside ``for (int j = 0; j < ncols_dst; ++j)`` loops. No dense/no-ids indexing
or stride correctness dependency on ncols==1 was found.

CORRECTION (gpt-dev-agent review, req_6d9ac5777c094cea): an earlier revision of
this docstring also cited the kernel comment "Block: (warp_size, ncols_dst) --
each warp handles one token independently" as evidence of safety. **That comment
belongs to the separate MoE kernel, not to dense MMVQ, and the citation was
wrong.** It materially understated the real risk, which is REGISTER PRESSURE:
dense RDNA3 uses nwarps=1 and rows_per_cuda_block=1 for ncols>1, so at ncols=6
every thread carries ``tmp[6]``, ``tmp_gate[6]``, ``x_biases[6]`` and
``gate_biases[6]`` -- roughly 24 live float slots before vec-dot temporaries --
and the bias/gate-bias flags are RUNTIME flags, so a bias-free model does not
necessarily let the compiler fold those arrays away. Scratch/spill is the
threat to watch, not shared-memory sizing. Checking VGPR and especially
private/scratch usage of the new specialization is therefore a go/no-go gate
BEFORE any timing work: scratch appearing at all is an abandon signal for a
~1% opportunity.

The restriction lives entirely in the eligibility/instantiation layer:
  * ``mul_mat_vec_q_switch_fusion`` only instantiates the fused variant inside
    ``if constexpr (c_ncols_dst == 1)``, so for ncols>1 the fused template is
    never emitted at all;
  * ``ggml_cuda_mul_mat_vec_q`` asserts ``ids || dst->ne[1] == 1`` in its
    fusion prologue;
  * ``ggml_cuda_should_fuse_mul_mat_vec_q`` returns false for
    ``dst->ne[1] != 1``.

Upstream states the limitation flatly ("we only support fusion for
ncols_dst = 1") with no rationale given, and the generic kernel body
indicates conservatism rather than a correctness constraint.

SCOPE -- deliberately a Q8_0 + ncols=6 PROTOTYPE, not a general widening:

  * Fused instantiation is added for exactly ``(c_ncols_dst == 6 && type ==
    GGML_TYPE_Q8_0)`` alongside the existing ncols==1 case. A general
    ``c_ncols_dst >= 1`` widening (the first revision of this patch) would have
    created roughly 23 quantized types x 7 extra widths = ~161 additional fused
    kernel specializations, which is grossly disproportionate to a ~1% target.
    This form adds about one specialization per compiled GPU architecture.
    If it wins, Q8_0 widths 2..6 costs only five more and covers tunable MTP
    widths (spec_draft_n_max is user-configurable, so the needed width moves
    with config -- 6 is today's production value, n_max=5 + 1).
  * Host-side selection is gated to the same shape AND to gfx1100
    (``GGML_CUDA_CC_IS_RDNA3_0``), matching patch 1241/RD30's precedent of
    excluding RDNA3.5 pending its own hardware evidence. Host gating does not
    itself save instantiations -- the compile-time condition above does -- but
    it keeps the behavioural blast radius on the architecture the evidence
    covers.
  * Dense ``GGML_OP_MUL_MAT`` only. The ``GGML_OP_MUL_MAT_ID`` (MoE routing)
    gate is left at ``!= 1`` untouched: this project's target model is dense,
    the profiling evidence above is dense-only, and MoE expert routing has its
    own indexing that this change has no evidence for.
  * MMVQ only. The identical ncols_dst==1 gate on the MMVF path
    (ggml_cuda_should_fuse_mul_mat_vec_f) is left alone -- this model is Q8_0,
    so MMVQ is the path that matters, and leaving MMVF untouched keeps the
    blast radius to one kernel family.

Note ``ggml_cuda_should_use_mmvq()`` carries its own architecture/type-specific
thresholds; RDNA3 dense Q8_0 falls through to ``<= MMVQ_MAX_BATCH_SIZE``, so
ncols=6 is a valid MMVQ shape for this target (corroborated by the trace above,
which observes 112,110 real ncols=6 Q8_0 MMVQ dispatches). Restricting this
patch to Q8_0/gfx1100 keeps it inside the regime the evidence actually covers
rather than enabling multi-column fused MMVQ for type/architecture combinations
where normal dispatch may stop using MMVQ earlier.

CORRECTNESS GATE IS TOLERANCE, NOT BIT-IDENTITY: fusion changes the
accumulation/rounding structure relative to running the two matmuls plus a
separate GLU, so a widened path must be compared against a CPU reference with
tolerance, exactly as patch 1241 (RD33) had to be.

STATE: untested. Not built, not run, not measured on real hardware.
"""

GROUP = "gpu-collectives"
STATE = "untested"

from bigcherry.patcher import Edit, FilePatch

MMVQ_CU = FilePatch(
    path="ggml/src/ggml-cuda/mmvq.cu",
    description="instantiate the fused MMVQ kernel for the whole ncols_dst "
                "batch range, and widen the dense fusion-prologue assert",
    edits=(
        Edit(
            id="instantiate-fusion-q8_0-ncols6",
            anchor=(
                r"    if constexpr \(c_ncols_dst == 1\) \{\n"
                r"        if \(has_fusion\) \{"
            ),
            rationale="mul_mat_vec_q_switch_fusion only emits the fused "
                      "template inside this if-constexpr, so for ncols_dst>1 "
                      "no fused instantiation exists at all",
            mode="replace",
            text=(
                "    if constexpr (c_ncols_dst == 1 ||\n"
                "                  (c_ncols_dst == 6 && type == GGML_TYPE_Q8_0)) {\n"
                "        if (has_fusion) {"
            ),
            guard=r"c_ncols_dst == 6 && type == GGML_TYPE_Q8_0",
        ),
        # NOTE: the GGML_ASSERT(!has_fusion && "fusion only supported for
        # ncols_dst=1") immediately after that block is deliberately left in
        # place. With the guard above always true, the fused branch returns
        # before reaching it whenever has_fusion is set, so the assert is only
        # ever evaluated with !has_fusion and passes trivially. Leaving it
        # alone also avoids anchoring through a string literal, which this
        # project's patcher blanks before matching.
        Edit(
            id="widen-dense-fusion-prologue-assert",
            anchor=r"        GGML_ASSERT\(  ids \|\| dst->ne\[1\] == 1\);",
            rationale="ggml_cuda_mul_mat_vec_q's fusion prologue asserts the "
                      "dense case is single-column; widen to the MMVQ batch "
                      "bound while leaving the MUL_MAT_ID assert untouched",
            mode="replace",
            text="        GGML_ASSERT(  ids || dst->ne[1] <= MMVQ_MAX_BATCH_SIZE);",
            guard=r"GGML_ASSERT\(  ids \|\| dst->ne\[1\] <= MMVQ_MAX_BATCH_SIZE\);",
        ),
    ),
)

GGML_CUDA_CU = FilePatch(
    path="ggml/src/ggml-cuda/ggml-cuda.cu",
    description="widen the dense MMVQ fusion eligibility predicate from "
                "ncols_dst==1 to the MMVQ batch bound",
    edits=(
        Edit(
            id="widen-mmvq-fusion-eligibility",
            # Anchored on the trailing `return use_mul_mat_vec_q;` so this
            # matches the MMVQ predicate and never the byte-identical block in
            # ggml_cuda_should_fuse_mul_mat_vec_f (which returns ..._f).
            # Deliberately starts AFTER the "//we only support fusion for
            # ncols_dst = 1" comment: this project's patcher blanks comments
            # before matching, so an anchor spanning one silently fails.
            anchor=(
                r"    if \(tensor->op == GGML_OP_MUL_MAT && dst->ne\[1\] != 1\) \{\n"
                r"        return false;\n"
                r"    \}\n"
                r"\n"
                r"    if \(tensor->op == GGML_OP_MUL_MAT_ID && dst->ne\[2\] != 1\) \{\n"
                r"        return false;\n"
                r"    \}\n"
                r"\n"
                r"    return use_mul_mat_vec_q;"
            ),
            rationale="the host-side gate that refuses MMVQ fusion for any "
                      "dst->ne[1] != 1, i.e. for every MTP speculative-verify "
                      "shape this workload actually runs",
            mode="replace",
            text=(
                "    if (tensor->op == GGML_OP_MUL_MAT &&\n"
                "        !(dst->ne[1] == 1 ||\n"
                "          (dst->ne[1] == 6 && src0->type == GGML_TYPE_Q8_0 && GGML_CUDA_CC_IS_RDNA3_0(cc)))) {\n"
                "        return false;\n"
                "    }\n"
                "\n"
                "    if (tensor->op == GGML_OP_MUL_MAT_ID && dst->ne[2] != 1) {\n"
                "        return false;\n"
                "    }\n"
                "\n"
                "    return use_mul_mat_vec_q;"
            ),
            guard=r"dst->ne\[1\] == 6 && src0->type == GGML_TYPE_Q8_0 && GGML_CUDA_CC_IS_RDNA3_0\(cc\)",
        ),
    ),
)

PATCHES = [MMVQ_CU, GGML_CUDA_CU]
