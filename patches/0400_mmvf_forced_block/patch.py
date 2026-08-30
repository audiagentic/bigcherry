"""HI07 - MMVF forced block-size and accumulator-mode dispatch.

The forced value is an explicit parameter, threaded to the launcher that
consumes it.

An earlier version of this patch used a thread-local override, because
threading looked like the larger diff. That was the wrong call, for a reason
only visible in production: a replay build would read the override on *every*
MMVF launch, and in production it is always zero. The native path would be
paying, on the hottest path in the system, for a feature only the tuner uses.

With an explicit parameter the native path is untouched -- no branch, no
thread-local read, byte-identical to upstream -- and only a forced launch
carries the value.

The diff stays small because every added parameter is **appended and
defaulted**, so all existing call sites compile unchanged. Only the calls on
the path a forced value travels are updated:

    ggml_cuda_mul_mat_vec_f          + 2 defaulted params, 3 forwarding sites
      -> mul_mat_vec_f_cuda          + 2 defaulted params, 2 forwarding sites
        -> ..._switch_ncols_dst      + 1 defaulted param, 10 forwarding sites
          -> launch_..._cuda         + 1 defaulted param, applies it

The ten launcher calls are one `replace_all` edit rather than ten near-identical
Edit objects. The match count is asserted, so upstream adding or removing an
`ncols_dst` case fails loudly instead of leaving the patch half-applied.

What this costs: the forced path is now genuinely separate code, so "forced with
the value native would have chosen produces identical output" stops being
structurally guaranteed and becomes a property a test must check. That is HI16,
and it is worth having regardless -- it is exactly the test that would catch a
mistake in this patch.

Accumulator mode (standards 3.2) travels the same way and never *upgrades*
precision: F16 accumulation is only selected where the native policy would also
have selected it. Forcing F16 against an F32 request is a different operation,
not a variant of this one (3.1), and `ggml_hip_mmvf_can_execute` rejects it
before it reaches here.
"""

GROUP = "core"
STATE = "validated"

from bigcherry.patcher import Edit, FilePatch

_ELIGIBILITY_AND_ENTRY = """
// bigcherry (HI07): hard eligibility (standards 12.4). Answers "could this run
// at all", never "would it be fast".
bool ggml_cuda_mmvf_variant_is_eligible(
        ggml_type type, int block_size, bool acc_f16, int warp_size,
        size_t shared_mem_limit, int64_t ncols, int64_t width,
        bool has_fusion) {
    GGML_UNUSED(ncols);

    // The compiled cases (32..256 step 32) assume wave32. On wave64 hardware
    // the reachable set differs, so test the property, not the membership.
    if (block_size < warp_size || block_size % warp_size != 0) {
        return false;
    }
    if (block_size > 256) {
        return false;
    }
    // Standards 3.2: two accumulator modes exist for F16 sources only.
    if (acc_f16 && type != GGML_TYPE_F16) {
        return false;
    }
    // Upstream supports fusion on the ncols_dst == 1 path only.
    if (has_fusion && width != 1) {
        return false;
    }
    const size_t nbytes_shared =
        (size_t) warp_size * sizeof(float) * (has_fusion ? 2 : 1);
    return nbytes_shared <= shared_mem_limit;
}

// bigcherry (HI07): forced-variant entry point for measured dispatch.
// forced_block_size == 0 and forced_acc_f16 < 0 mean "native policy", so the
// registry's native wrapper and every forced variant share one path -- and the
// tensor-argument marshalling is written exactly once.
void ggml_cuda_mul_mat_vec_f_variant(
        ggml_backend_cuda_context & ctx, const ggml_tensor * src0,
        const ggml_tensor * src1, const ggml_tensor * ids, ggml_tensor * dst,
        const ggml_cuda_mm_fusion_args_host * fusion,
        int forced_block_size, int forced_acc_f16) {
    ggml_cuda_mul_mat_vec_f(ctx, src0, src1, ids, dst, fusion,
                            forced_block_size, forced_acc_f16);
}

"""

_BLOCK_APPLY = """

    // bigcherry (HI07): a forced block size replaces the scan's answer. Zero
    // means "native policy", so with no override the code below is exactly
    // upstream's. The switch that follows is unchanged and remains the single
    // launcher for both paths.
    if (forced_block_size != 0) {
        block_size_best = forced_block_size;
    }
"""

_ACC_APPLY = """    if constexpr(std::is_same_v<T, half>) {
        // bigcherry (HI07): accumulator mode is a performance variant for F16
        // sources only (standards 3.2). A forced mode can decline F16
        // accumulation but never select it where the native policy would not:
        // upgrading a request's precision is a different operation (3.1), not
        // a faster form of this one.
        const bool native_wants_f16 = prec == GGML_PREC_DEFAULT;
        const bool use_f16_acc = forced_acc_f16 < 0
            ? native_wants_f16
            : (forced_acc_f16 != 0 && native_wants_f16);
        if (use_f16_acc) {"""

SOURCE_PATCH = FilePatch(
    path="ggml/src/ggml-cuda/mmvf.cu",
    description="MMVF forced block size and accumulator mode, threaded explicitly",
    edits=(
        # -- innermost launcher: take the value and apply it ------------------
        Edit(
            id="mmvf-launcher-param",
            anchor=r"        const int64_t nsamples_or_ntokens, const int64_t ids_stride, "
                   r"cudaStream_t stream\) \{",
            rationale="launch_mul_mat_vec_f_cuda's parameter list",
            mode="replace",
            text="        const int64_t nsamples_or_ntokens, const int64_t ids_stride,\n"
                 "        cudaStream_t stream, int forced_block_size = 0) {",
            guard=r"cudaStream_t stream, int forced_block_size = 0\) \{",
        ),
        Edit(
            id="mmvf-block-apply",
            anchor=r"    for \(int64_t block_size = 2\*warp_size; "
                   r"block_size <= max_block_size; block_size \+= warp_size\) \{\n"
                   r"        const int64_t niter = \(ncols \+ 2\*block_size - 1\) / \(2\*block_size\);\n"
                   r"        if \(niter < niter_best\) \{\n"
                   r"            niter_best      = niter;\n"
                   r"            block_size_best = block_size;\n"
                   r"        \}\n"
                   r"    \}",
            rationale="upstream's block-size scan, whose answer a forced value replaces",
            text=_BLOCK_APPLY,
            guard=r"if \(forced_block_size != 0\)",
            max_span_lines=10,
        ),

        # -- ncols_dst dispatcher: take the value and forward it --------------
        Edit(
            id="mmvf-ncols-param",
            anchor=r"        const int64_t ids_stride, cudaStream_t stream\) \{\n\n"
                   r"    const bool has_ids = ids != nullptr;",
            rationale="mul_mat_vec_f_cuda_switch_ncols_dst's parameter list",
            mode="replace",
            text="        const int64_t ids_stride, cudaStream_t stream,\n"
                 "        int forced_block_size = 0) {\n\n"
                 "    const bool has_ids = ids != nullptr;",
            guard=r"int forced_block_size = 0\) \{\n\n    const bool has_ids",
        ),
        Edit(
            id="mmvf-forward-ncols-dst",
            anchor=r"ncols_dst, ids_stride, stream\);",
            rationale="the two multi-token-id launcher calls",
            mode="replace_all",
            expect_matches=2,
            text="ncols_dst, ids_stride, stream, forced_block_size);",
            guard=r"ncols_dst, ids_stride, stream, forced_block_size\);",
        ),
        Edit(
            id="mmvf-forward-nsamples",
            anchor=r"nsamples_dst, ids_stride, stream\);",
            rationale="the eight ncols_dst-case launcher calls",
            mode="replace_all",
            expect_matches=8,
            text="nsamples_dst, ids_stride, stream, forced_block_size);",
            guard=r"nsamples_dst, ids_stride, stream, forced_block_size\);",
        ),

        # -- accumulator selection: take both values --------------------------
        Edit(
            id="mmvf-cuda-param",
            anchor=r"        const int64_t ids_stride, enum ggml_prec prec, cudaStream_t stream\) \{",
            rationale="mul_mat_vec_f_cuda's parameter list",
            mode="replace",
            text="        const int64_t ids_stride, enum ggml_prec prec, cudaStream_t stream,\n"
                 "        int forced_block_size = 0, int forced_acc_f16 = -1) {",
            guard=r"int forced_block_size = 0, int forced_acc_f16 = -1\) \{",
        ),
        Edit(
            id="mmvf-acc-apply",
            anchor=r"    if constexpr\(std::is_same_v<T, half>\) \{\n"
                   r"        if \(prec == GGML_PREC_DEFAULT\) \{",
            rationale="the F16 accumulator branch in mul_mat_vec_f_cuda",
            mode="replace",
            text=_ACC_APPLY,
            guard=r"const bool use_f16_acc = forced_acc_f16 < 0",
            max_span_lines=4,
        ),
        Edit(
            id="mmvf-forward-switch-ncols",
            anchor=r"stride_sample_dst, ids_stride, stream\);",
            rationale="the two mul_mat_vec_f_cuda_switch_ncols_dst calls, one "
                      "per accumulator mode",
            mode="replace_all",
            expect_matches=2,
            text="stride_sample_dst, ids_stride, stream, forced_block_size);",
            guard=r"stride_sample_dst, ids_stride, stream, forced_block_size\);",
        ),

        # -- public entry: accept and forward ---------------------------------
        Edit(
            id="mmvf-public-param",
            anchor=r"^void ggml_cuda_mul_mat_vec_f\(ggml_backend_cuda_context & ctx, "
                   r"const ggml_tensor \* src0, const ggml_tensor \* src1, "
                   r"const ggml_tensor \* ids, ggml_tensor \* dst,\n"
                   r"    const ggml_cuda_mm_fusion_args_host \* fusion\) \{",
            rationale="the public MMVF entry point",
            mode="replace",
            text="void ggml_cuda_mul_mat_vec_f(ggml_backend_cuda_context & ctx, "
                 "const ggml_tensor * src0, const ggml_tensor * src1, "
                 "const ggml_tensor * ids, ggml_tensor * dst,\n"
                 "    const ggml_cuda_mm_fusion_args_host * fusion,\n"
                 "    int forced_block_size, int forced_acc_f16) {",
            guard=r"int forced_block_size, int forced_acc_f16\) \{",
        ),
        Edit(
            id="mmvf-forward-public",
            anchor=r"ids_stride, prec, ctx\.stream\(\)\);",
            rationale="the three per-type calls in the public entry",
            mode="replace_all",
            expect_matches=3,
            text="ids_stride, prec, ctx.stream(), forced_block_size, forced_acc_f16);",
            guard=r"ctx\.stream\(\), forced_block_size, forced_acc_f16\);",
        ),
        Edit(
            id="mmvf-eligibility-and-entry",
            anchor=r"^void ggml_cuda_op_mul_mat_vec_f\($",
            rationale="the function following ggml_cuda_mul_mat_vec_f",
            mode="insert_before",
            text=_ELIGIBILITY_AND_ENTRY,
            guard=r"bool ggml_cuda_mmvf_variant_is_eligible\(",
        ),
    ),
)

HEADER_PATCH = FilePatch(
    path="ggml/src/ggml-cuda/mmvf.cuh",
    description="MMVF public entry gains defaulted forced-variant parameters",
    edits=(
        Edit(
            id="mmvf-public-decl",
            anchor=r"    const ggml_cuda_mm_fusion_args_host \* fusion = nullptr\);",
            rationale="the declaration of ggml_cuda_mul_mat_vec_f",
            mode="replace",
            # Defaulted, so every existing caller elsewhere in the tree is
            # unaffected and keeps getting the native policy.
            text="    const ggml_cuda_mm_fusion_args_host * fusion = nullptr,\n"
                 "    int forced_block_size = 0, int forced_acc_f16 = -1);",
            guard=r"int forced_block_size = 0, int forced_acc_f16 = -1\);",
        ),
    ),
)

PATCHES = [HEADER_PATCH, SOURCE_PATCH]
