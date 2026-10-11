"""HI06 - MMQ forced-J variant dispatch.

b11552 computes J_best in ggml_cuda_mul_mat_q before activation padding,
workspace allocation, and mmq_args construction. Forced J therefore belongs at
that same selection point: overriding it later in mul_mat_q_case/switch_J would
let the launch use a wider tile than the buffer was padded for.

The public MMQ entry point carries one required forced_J value. Native callers
pass 0 explicitly; the autotune variant passes its candidate J. On HIP, a
non-zero explicit value wins over GGML_HIP_MMQ_FORCE_J, while zero permits the
environment override. After selection, b11552's native args.J_best dispatch is
left unchanged, so no forced parameter is threaded through case/switch
signatures and no compatibility default is retained.
"""

GROUP = "core"
STATE = "validated"

from bigcherry.patcher import Edit, FilePatch


_HELPERS = r"""
// bigcherry (HI06): hard MMQ config eligibility. b11552 made activation
// precision explicit throughout the config API, so callers must name it.
static inline bool ggml_cuda_mmq_config_is_eligible(
        const ggml_type type, const int J, const bool fallback,
        const int cc, const size_t shared_mem_limit, const ggml_prec prec_src1) {
    if (J < 8 || J > 128 || J % 8 != 0) {
        return false;
    }
    const ggml_cuda_mmq_config config =
        ggml_cuda_mmq_get_config(type, J, fallback, cc, prec_src1);
    if (config.type == GGML_TYPE_COUNT) {
        return false;
    }
    return mmq_get_nbytes_shared(config, cc) <= shared_mem_limit;
}

// bigcherry (HI24): native's b11552 J search as a query for the tuner/noise
// canary. ncols_opt is the same optimization width computed by upstream.
int ggml_cuda_mmq_native_j_best(
        ggml_type type, bool fallback, int64_t ncols_opt, ggml_prec prec_src1);
"""

_FORCE_SELECTION = r"""
    // bigcherry (HI06): b11552 sizes src1 padding and MoE ids_dst from J_best,
    // so force the value here, before either allocation. An explicit autotune
    // candidate wins; zero means native policy unless the HIP diagnostic env
    // override requests a tile width.
    int bc_forced_J = forced_J;
#ifdef GGML_USE_HIP
    if (bc_forced_J == 0) {
        const char * bc_force_J_env = std::getenv("GGML_HIP_MMQ_FORCE_J");
        if (bc_force_J_env != nullptr && bc_force_J_env[0] != '\0') {
            bc_forced_J = std::atoi(bc_force_J_env);
        }
    }
#endif
    if (bc_forced_J != 0) {
        GGML_ASSERT(ggml_cuda_mmq_config_is_eligible(
            src0->type, bc_forced_J, fallback, cc, smpbo, prec_src1));
        const ggml_cuda_mmq_config bc_forced_config =
            ggml_cuda_mmq_get_config(src0->type, bc_forced_J, fallback, cc, prec_src1);
        J_best = bc_forced_J;
        nthreads_best = bc_forced_config.nthreads;
    }
"""

_MMQ_SOURCE = r"""
// bigcherry (HI06): hard eligibility used by measured dispatch. b11552 pads
// the activation/workspace from the selected J itself, so the former HI71
// ggml_cuda_mmq_get_J_max tail-envelope check is obsolete (that helper was
// removed upstream). A real width is still required because this predicate is
// only valid for a real dispatch signature.
bool ggml_cuda_mmq_variant_is_eligible(
        ggml_type type, int j, bool fallback, int cc, size_t shared_mem_limit,
        int64_t ncols_max) {
    const int id = ggml_cuda_get_device();
    if (cc == 0) {
        cc = ggml_cuda_info().devices[id].cc;
    }
    if (shared_mem_limit == 0) {
        shared_mem_limit = ggml_cuda_info().devices[id].smpbo;
    }
    GGML_ASSERT(ncols_max > 0 &&
        "ggml_cuda_mmq_variant_is_eligible requires a real batch width");
    return ggml_cuda_mmq_config_is_eligible(
        type, j, fallback, cc, shared_mem_limit, GGML_PREC_Q8);
}

// bigcherry (HI24): query the same J-search policy b11552 moved into
// ggml_cuda_mul_mat_q. The tuner uses this only at HIP's reachable Q8
// activation precision and passes ncols_opt, not the launch width.
int ggml_cuda_mmq_native_j_best(
        ggml_type type, bool fallback, int64_t ncols_opt, ggml_prec prec_src1) {
    const int    id    = ggml_cuda_get_device();
    const int    cc    = ggml_cuda_info().devices[id].cc;
    const size_t smpbo = ggml_cuda_info().devices[id].smpbo;

    int J_best        = 0;
    int ntiles_J_best = INT_MAX;
    for (int J = 8; J <= 128 && ntiles_J_best > 1; J += 8) {
        const ggml_cuda_mmq_config config =
            ggml_cuda_mmq_get_config(type, J, fallback, cc, prec_src1);
        if (config.type == GGML_TYPE_COUNT) {
            continue;
        }
        if (mmq_get_nbytes_shared(config, cc) > smpbo) {
            continue;
        }
        const int ntiles_x = (int) ((ncols_opt + config.J - 1) / config.J);
        if (ntiles_x < ntiles_J_best) {
            J_best = J;
            ntiles_J_best = ntiles_x;
        }
    }
    return J_best;
}

bool ggml_cuda_mmq_type_is_supported(
        ggml_type type, int cc, size_t shared_mem_limit) {
    const int id = ggml_cuda_get_device();
    if (cc == 0) {
        cc = ggml_cuda_info().devices[id].cc;
    }
    if (shared_mem_limit == 0) {
        shared_mem_limit = ggml_cuda_info().devices[id].smpbo;
    }
    for (int j = 8; j <= 128; j += 8) {
        if (ggml_cuda_mmq_config_is_eligible(
                type, j, false, cc, shared_mem_limit, GGML_PREC_Q8)
         || ggml_cuda_mmq_config_is_eligible(
                type, j, true, cc, shared_mem_limit, GGML_PREC_Q8)) {
            return true;
        }
    }
    return false;
}

// bigcherry (HI06): measured-dispatch entry. forced_j == 0 retains native
// selection (or permits the explicit HIP env override handled above).
void ggml_cuda_mul_mat_q_variant(
        ggml_backend_cuda_context & ctx, const ggml_tensor * src0,
        const ggml_tensor * src1, const ggml_tensor * ids, ggml_tensor * dst,
        int forced_j, int forced_fallback) {
    GGML_UNUSED(forced_fallback);
    ggml_cuda_mul_mat_q(ctx, src0, src1, ids, dst, forced_j);
}
"""


HEADER_PATCH = FilePatch(
    path="ggml/src/ggml-cuda/mmq.cuh",
    description="MMQ forced-J eligibility/query helpers and required public parameter",
    edits=(
        Edit(
            id="mmq-helpers",
            anchor=r"^template <ggml_type type, bool fallback, ggml_prec prec_src1>\n"
                   r"void mul_mat_q_switch_J\(ggml_backend_cuda_context & ctx, "
                   r"const mmq_args & args, cudaStream_t stream\) \{$",
            rationale="b11552 J dispatcher, after mmq_get_nbytes_shared is defined",
            mode="insert_before",
            text=_HELPERS,
            guard=r"static inline bool ggml_cuda_mmq_config_is_eligible\(",
            max_span_lines=3,
        ),
        Edit(
            id="mmq-public-decl",
            anchor=r"^void ggml_cuda_mul_mat_q\(\n"
                   r"        ggml_backend_cuda_context & ctx, const ggml_tensor \* src0, "
                   r"const ggml_tensor \* src1, const ggml_tensor \* ids, ggml_tensor \* dst\);$",
            rationale="b11552 public MMQ declaration",
            mode="replace",
            text="void ggml_cuda_mul_mat_q(\n"
                 "        ggml_backend_cuda_context & ctx, const ggml_tensor * src0, "
                 "const ggml_tensor * src1, const ggml_tensor * ids, ggml_tensor * dst,\n"
                 "        int forced_J);",
            guard=r"ggml_tensor \* dst,\n        int forced_J\);",
            max_span_lines=3,
        ),
    ),
)


SOURCE_PATCH = FilePatch(
    path="ggml/src/ggml-cuda/mmq.cu",
    description="resolve forced J before b11552 padding/allocation and expose tuner helpers",
    edits=(
        Edit(
            id="mmq-cstdlib",
            anchor=r"^#include <cstdint>$",
            rationale="stdlib declarations for the HIP force-J environment override",
            mode="insert_before",
            text="#include <cstdlib>\n",
            guard=r"^#include <cstdlib>$",
            max_span_lines=2,
        ),
        Edit(
            id="mmq-public-param",
            anchor=r"^void ggml_cuda_mul_mat_q\(\n"
                   r"        ggml_backend_cuda_context & ctx, const ggml_tensor \* src0, "
                   r"const ggml_tensor \* src1, const ggml_tensor \* ids, ggml_tensor \* dst\) \{$",
            rationale="b11552 public MMQ definition",
            mode="replace",
            text="void ggml_cuda_mul_mat_q(\n"
                 "        ggml_backend_cuda_context & ctx, const ggml_tensor * src0, "
                 "const ggml_tensor * src1, const ggml_tensor * ids, ggml_tensor * dst,\n"
                 "        int forced_J) {",
            guard=r"ggml_tensor \* dst,\n        int forced_J\) \{",
            max_span_lines=3,
        ),
        Edit(
            id="mmq-force-before-padding",
            anchor=r"^    GGML_ASSERT\(J_best > 0\);$",
            rationale="b11552 point after native J search and before J-sized padding/allocation",
            mode="insert_before",
            text=_FORCE_SELECTION,
            guard=r"int bc_forced_J = forced_J;",
            max_span_lines=2,
        ),
        Edit(
            id="mmq-eligibility-and-entry",
            anchor=r"^bool ggml_cuda_should_use_mmq\(enum ggml_type type, int cc, "
                   r"int64_t ne11, int64_t n_experts\) \{$",
            rationale="function following ggml_cuda_mul_mat_q",
            mode="insert_before",
            text=_MMQ_SOURCE,
            guard=r"void ggml_cuda_mul_mat_q_variant\(",
            max_span_lines=2,
        ),
    ),
)


CALLER_PATCH = FilePatch(
    path="ggml/src/ggml-cuda/ggml-cuda.cu",
    description="migrate native MMQ callers to the required forced-J argument",
    edits=(
        Edit(
            id="mmq-dense-native-call",
            anchor=r"^        ggml_cuda_mul_mat_q\(ctx, src0, src1, nullptr, dst\);$",
            rationale="native dense MMQ call",
            mode="replace",
            text="        ggml_cuda_mul_mat_q(ctx, src0, src1, nullptr, dst, 0);",
            guard=r"ggml_cuda_mul_mat_q\(ctx, src0, src1, nullptr, dst, 0\);",
            max_span_lines=2,
        ),
        Edit(
            id="mmq-moe-native-call",
            anchor=r"^            ggml_cuda_mul_mat_q\(ctx, src0, src1, ids, dst\);$",  # b11552: inside `if (ggml_cuda_should_use_mmq(...))`
            rationale="native MUL_MAT_ID MMQ call",
            mode="replace",
            text="            ggml_cuda_mul_mat_q(ctx, src0, src1, ids, dst, 0);",
            guard=r"ggml_cuda_mul_mat_q\(ctx, src0, src1, ids, dst, 0\);",
            max_span_lines=2,
        ),
    ),
)


PATCHES = [HEADER_PATCH, SOURCE_PATCH, CALLER_PATCH]
