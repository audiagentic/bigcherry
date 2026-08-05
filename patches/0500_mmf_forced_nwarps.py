"""HI08 - MMF forced-nwarps dispatch.

Same shape as HI07: the forced value is an explicit, appended, defaulted
parameter, so the native path is byte-identical to upstream and only a forced
launch carries anything extra. See that patch for why an override variable was
the wrong answer.

MMF happens to be the tidiest of the three families, because its three
dispatchers share an identical signature tail and an identical call tail:

    mul_mat_f_switch_rows_per_block   -.
      mul_mat_f_switch_cols_per_block  |-- all end `cudaStream_t stream,
        mul_mat_f_cuda                -'      const mmf_ids_data * ids_data)`

So three signatures and eighteen forwarding calls are two `replace_all` edits
rather than twenty-one Edit objects. Both assert their match counts, so upstream
adding a `cols_per_block` case or a rows variant fails loudly here instead of
leaving the patch silently half-applied.

**Shared memory comes out right because of where the value is applied.** HI08
requires dynamic shared memory to be recalculated from the forced nwarps rather
than the native choice. Upstream computes `nbytes_shared_iter` and
`nbytes_shared_combine` from `nwarps_best` *after* the scan, so applying the
forced value immediately after the scan means both are already derived from it.
Applying it at the switch instead would need them recomputed by hand, and a
forced nwarps larger than the native choice would under-allocate.
"""

from bigcherry.patcher import Edit, FilePatch

_NWARPS_APPLY = """

    // bigcherry (HI08): a forced nwarps replaces the scan's answer. Applied
    // here, above the shared-memory computation, so nbytes_shared_iter and
    // nbytes_shared_combine are both derived from the forced value -- a forced
    // nwarps larger than the native choice would otherwise under-allocate.
    // Zero means "native policy", so with no override this is exactly upstream.
    if (forced_nwarps != 0) {
        nwarps_best = forced_nwarps;
    }
"""

_ELIGIBILITY_AND_ENTRY = """
// bigcherry (HI08): hard eligibility (standards 12.4). Answers "could this run
// at all", never "would it be fast".
bool ggml_cuda_mmf_variant_is_eligible(
        ggml_type type, int nwarps, int cc, int warp_size,
        size_t shared_mem_limit, int64_t width, int64_t rows_per_block) {
    GGML_UNUSED(type);

    if (nwarps < 1 || nwarps > 8) {
        return false; // the compiled switch covers 1..8
    }
    // The reachable maximum is architecture dependent, not a flat 8: upstream
    // scans up to mmf_get_max_block_size(cc)/warp_size. Using the same
    // expression keeps eligibility honest on wave64 parts, where it halves.
    if (cc != 0 && warp_size != 0
            && nwarps > mmf_get_max_block_size(cc) / warp_size) {
        return false;
    }

    // Mirrors the shared-memory computation in mul_mat_f_cuda for the forced
    // value, so a geometry that would not fit is rejected before launch.
    const int tile_i = 16;
    const int nbytes_shared_iter =
        nwarps * tile_i * (warp_size + mmf_get_padding(cc)) * 4;
    const int nbytes_shared_combine =
        GGML_PAD((int) width, tile_i) *
        (nwarps * (int) rows_per_block + mmf_get_padding(cc)) * 4;
    const size_t nbytes_shared =
        (size_t) std::max(nbytes_shared_iter, nbytes_shared_combine);

    return nbytes_shared <= shared_mem_limit;
}

// bigcherry (HI08): forced-variant entry point for measured dispatch.
// forced_nwarps == 0 means "native policy", so the registry's native wrapper
// and every forced variant share one path.
void ggml_cuda_mul_mat_f_variant(
        ggml_backend_cuda_context & ctx, const ggml_tensor * src0,
        const ggml_tensor * src1, const ggml_tensor * ids, ggml_tensor * dst,
        int forced_nwarps) {
    ggml_cuda_mul_mat_f(ctx, src0, src1, ids, dst, forced_nwarps);
}

"""

HEADER_PATCH = FilePatch(
    path="ggml/src/ggml-cuda/mmf.cuh",
    description="MMF dispatchers take a forced nwarps and forward it",
    edits=(
        Edit(
            id="mmf-dispatcher-params",
            # All three dispatchers share this tail. Appended and defaulted, so
            # any call site not on the forced path compiles unchanged.
            anchor=r"        cudaStream_t stream, const mmf_ids_data \* ids_data\) \{",
            rationale="the shared parameter-list tail of mul_mat_f_cuda, "
                      "mul_mat_f_switch_cols_per_block and "
                      "mul_mat_f_switch_rows_per_block",
            mode="replace_all",
            expect_matches=3,
            text="        cudaStream_t stream, const mmf_ids_data * ids_data,\n"
                 "        int forced_nwarps = 0) {",
            guard=r"const mmf_ids_data \* ids_data,\n        int forced_nwarps = 0\) \{",
        ),
        Edit(
            id="mmf-instantiation-macro",
            # DECL_MMF_CASE_HELPER restates mul_mat_f_cuda's whole parameter
            # list to explicitly instantiate it, so it has to gain the new
            # parameter too. Deliberately without `= 0`: repeating a default
            # argument on an explicit instantiation is ill-formed.
            anchor=r"        cudaStream_t stream, const mmf_ids_data \* ids_data\);",
            rationale="the explicit-instantiation macro DECL_MMF_CASE_HELPER",
            mode="replace_all",
            expect_matches=1,
            text="        cudaStream_t stream, const mmf_ids_data * ids_data, "
                 "int forced_nwarps);",
            guard=r"const mmf_ids_data \* ids_data, int forced_nwarps\);",
        ),
        Edit(
            id="mmf-dispatcher-forward",
            anchor=r"stream, ids_data\);",
            rationale="every call between the three MMF dispatchers",
            mode="replace_all",
            expect_matches=18,
            text="stream, ids_data, forced_nwarps);",
            guard=r"stream, ids_data, forced_nwarps\);",
        ),
        Edit(
            id="mmf-nwarps-apply",
            # Anchored on the whole scan, so a change to how the native choice
            # is made breaks loudly here rather than leaving the forced value
            # applied at the wrong point.
            anchor=r"    for \(int64_t nwarps = 2; nwarps <= max_block_size/warp_size; nwarps\+\+\) \{\n"
                   r"        const int64_t niter = \(ncols_x \+ nwarps\*warp_size\*2 - 1\) / \(nwarps\*warp_size\*2\);\n"
                   r"        if \(niter < niter_best\) \{\n"
                   r"            niter_best  = niter;\n"
                   r"            nwarps_best = nwarps;\n"
                   r"        \}\n"
                   r"    \}",
            rationale="upstream's nwarps scan, which must precede the "
                      "shared-memory computation",
            text=_NWARPS_APPLY,
            guard=r"if \(forced_nwarps != 0\)",
            max_span_lines=10,
        ),
        Edit(
            id="mmf-public-decl",
            anchor=r"^void ggml_cuda_mul_mat_f\(ggml_backend_cuda_context & ctx, "
                   r"const ggml_tensor \* src0, const ggml_tensor \* src1, "
                   r"const ggml_tensor \* ids, ggml_tensor \* dst\);$",
            rationale="the declaration of the public MMF entry point",
            mode="replace",
            text="void ggml_cuda_mul_mat_f(ggml_backend_cuda_context & ctx, "
                 "const ggml_tensor * src0, const ggml_tensor * src1, "
                 "const ggml_tensor * ids, ggml_tensor * dst,\n"
                 "    int forced_nwarps = 0);",
            guard=r"int forced_nwarps = 0\);",
        ),
    ),
)

SOURCE_PATCH = FilePatch(
    path="ggml/src/ggml-cuda/mmf.cu",
    description="MMF public entry accepts and forwards a forced nwarps",
    edits=(
        Edit(
            id="mmf-public-param",
            anchor=r"^void ggml_cuda_mul_mat_f\(ggml_backend_cuda_context & ctx, "
                   r"const ggml_tensor \* src0, const ggml_tensor \* src1, "
                   r"const ggml_tensor \* ids, ggml_tensor \* dst\) \{$",
            rationale="the public MMF entry point",
            mode="replace",
            text="void ggml_cuda_mul_mat_f(ggml_backend_cuda_context & ctx, "
                 "const ggml_tensor * src0, const ggml_tensor * src1, "
                 "const ggml_tensor * ids, ggml_tensor * dst,\n"
                 "    int forced_nwarps) {",
            guard=r"ggml_tensor \* dst,\n    int forced_nwarps\) \{",
        ),
        Edit(
            id="mmf-public-forward",
            anchor=r"ctx\.stream\(\), ids_info_ptr\);",
            rationale="the three per-type calls into the rows_per_block "
                      "dispatcher",
            mode="replace_all",
            expect_matches=3,
            text="ctx.stream(), ids_info_ptr, forced_nwarps);",
            guard=r"ctx\.stream\(\), ids_info_ptr, forced_nwarps\);",
        ),
        Edit(
            id="mmf-eligibility-and-entry",
            anchor=r"^bool ggml_cuda_should_use_mmf\(enum ggml_type type, int cc, "
                   r"int warp_size, const int64_t \* src0_ne,$",
            rationale="the function following ggml_cuda_mul_mat_f",
            mode="insert_before",
            text=_ELIGIBILITY_AND_ENTRY,
            guard=r"bool ggml_cuda_mmf_variant_is_eligible\(",
        ),
    ),
)

PATCHES = [HEADER_PATCH, SOURCE_PATCH]
