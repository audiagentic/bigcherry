// bigcherry: measured dispatch — selection, resolution, launch (HI04).
//
// Upstream's `ggml_cuda_mul_mat` decides and launches in one motion: each
// `if (should_use_X)` ends in a call. That is fine when the decision is fixed
// policy, but it leaves nothing to measure, nothing to store, and nothing to
// replay -- the choice exists only as control flow.
//
// This header splits the motion in three:
//
//     auto native = ggml_hip_native_select(ctx, op);      // pure decision
//     auto bound  = ggml_hip_dispatch_resolve(sig, native);
//     ggml_hip_dispatch_launch(bound, launch_context);    // the launch
//
// `ggml_hip_native_select` must reproduce upstream's branch order exactly. In
// native mode the resolver returns that candidate untouched, so the three-step
// form runs the same kernel with the same arguments as the one-step form did
// (standards 4.2). That equivalence is the thing the parity tests check, and
// nothing else in this project is trustworthy without it.

#pragma once

#include "hip-autotune-types.h"

// The public API travels with the dispatch header rather than being included
// separately by every consumer: ggml-cuda.cu needs ggml_hip_autotune_flush()
// for its teardown hook, and this is the only bigcherry header it includes.
#include "../../include/ggml-hip-autotune.h"

// Standards 12.2: gated on GGML_USE_HIP *and* the feature flag. These files
// are picked up by the ggml-cuda/*.cu glob, so without the second condition
// they would compile into every HIP build, including ones that carry none of
// the machinery they depend on.
#if defined(GGML_USE_HIP) && defined(GGML_HIP_DISPATCH)

// ------------------------------------------------------- family entry points
//
// The registry (hip-autotune-registry.inc) is a flat table of descriptors that
// point at these functions -- one triple per family, not per candidate. A
// candidate differs from its siblings only in `variant`, which the launcher
// reads from the descriptor it is handed. Thousands of table rows compile in
// moments; thousands of generated thunks would not.

#define GGML_HIP_DECLARE_FAMILY(name)                                          \
    bool name##_can_execute(const ggml_hip_candidate_descriptor * self,        \
                            const ggml_hip_dispatch_signature_v1 & sig,        \
                            const ggml_hip_hardware_key_v1 & hw);              \
    void name##_launch(const ggml_hip_candidate_descriptor * self,             \
                       const ggml_hip_launch_context & lc);                    \
    size_t name##_workspace(const ggml_hip_candidate_descriptor * self,        \
                            const ggml_hip_dispatch_signature_v1 & sig);

GGML_HIP_DECLARE_FAMILY(ggml_hip_mmq)
GGML_HIP_DECLARE_FAMILY(ggml_hip_mmvq)
GGML_HIP_DECLARE_FAMILY(ggml_hip_mmvf)
GGML_HIP_DECLARE_FAMILY(ggml_hip_mmf)
GGML_HIP_DECLARE_FAMILY(ggml_hip_blas)

#undef GGML_HIP_DECLARE_FAMILY

// --------------------------------------------------------- forced variants
//
// Implemented by the anchored patches to mmq.cu / mmvf.cu / mmf.cu (HI06-HI08).
// Each extracts the existing "compute the best value, then switch on it" into
// a launcher that takes the value as an argument. The native path then computes
// its best value and calls the same launcher, which is what makes forced-native
// bit-identical to native by construction rather than by testing.
//
// A variant value of 0 means "let the native policy choose", so the same entry
// point serves both modes.

void ggml_cuda_mul_mat_q_variant(
    ggml_backend_cuda_context & ctx, const ggml_tensor * src0,
    const ggml_tensor * src1, const ggml_tensor * ids, ggml_tensor * dst,
    int forced_j, int forced_fallback);

void ggml_cuda_mul_mat_vec_f_variant(
    ggml_backend_cuda_context & ctx, const ggml_tensor * src0,
    const ggml_tensor * src1, const ggml_tensor * ids, ggml_tensor * dst,
    const ggml_cuda_mm_fusion_args_host * fusion,
    int forced_block_size, int forced_acc_f16);

void ggml_cuda_mul_mat_f_variant(
    ggml_backend_cuda_context & ctx, const ggml_tensor * src0,
    const ggml_tensor * src1, const ggml_tensor * ids, ggml_tensor * dst,
    int forced_nwarps);

// forced_nwarps == 0 && forced_rows_per_block == 0 means "native policy"; in
// that case forced_small_k is ignored, because native decides small_k from the
// shape via should_use_small_k.
void ggml_cuda_mul_mat_vec_q_variant(
    ggml_backend_cuda_context & ctx, const ggml_tensor * src0,
    const ggml_tensor * src1, const ggml_tensor * ids, ggml_tensor * dst,
    const ggml_cuda_mm_fusion_args_host * fusion,
    int forced_nwarps, int forced_rows_per_block, bool forced_small_k);

// BLAS-1 carries a structured native-only plan through a generated side table.
// The resolve/apply/execute seam records the complete native call facts while
// the final vendor launcher remains the authority for exact cuBLAS arguments.
// The HI04 patch still exposes that file-local entry point through a forwarder.
void ggml_cuda_mul_mat_cublas_dispatch(
    ggml_backend_cuda_context & ctx, const ggml_tensor * src0,
    const ggml_tensor * src1, ggml_tensor * dst,
    const struct ggml_hip_blas_execution_options_v1 * options);

enum ggml_hip_blas_api_kind {
    GGML_HIP_BLAS_API_SGEMM = 0,
    GGML_HIP_BLAS_API_GEMMEX,
    GGML_HIP_BLAS_API_STRIDED_BATCHED,
    GGML_HIP_BLAS_API_POINTER_BATCHED,
};

// Runtime-only experiment controls. These are deliberately separate from
// ggml_hip_blas_plan_v1: they are never serialized, ranked, or replayed.
// The first supported route is the measured gfx1100 F16 temporary-output path.
enum ggml_hip_blas_experiment_compute_v1 {
    GGML_HIP_BLAS_EXPERIMENT_COMPUTE_F16 = 0,
};

enum ggml_hip_blas_experiment_output_v1 {
    GGML_HIP_BLAS_EXPERIMENT_OUTPUT_F16 = 0,
};

enum ggml_hip_blas_experiment_conversion_v1 {
    GGML_HIP_BLAS_EXPERIMENT_OUTPUT_TEMPORARY_TO_F32 = 0,
};

struct ggml_hip_blas_execution_options_v1 {
    uint8_t compute_type;
    uint8_t output_type;
    uint8_t output_conversion;
    size_t output_temp_bytes;
    size_t workspace_bytes;
};

struct ggml_hip_launch_context;

// Resolved native BLAS facts are intentionally observation/runtime state, not
// candidate or replay identity. They let the first BLAS-1 seam validate the
// plan and keep setup, output conversion, and final vendor launch together.
struct ggml_hip_blas_call_v1 {
    const ggml_hip_blas_plan_v1 * plan;
    uint8_t operand_type;
    uint8_t accumulation_type;
    uint8_t output_type;
    uint8_t api;
    uint8_t source_a_conversion;
    uint8_t source_b_conversion;
    uint8_t output_conversion;
    uint8_t numerical_class;
    uint8_t strict_precision;
    uint8_t has_ids;
    int64_t m;
    int64_t n;
    int64_t k;
    int64_t batch_count;
    int64_t lda;
    int64_t ldb;
    int64_t ldc;
    int64_t stride_a;
    int64_t stride_b;
    int64_t stride_c;
    size_t source_a_temp_bytes;
    size_t source_b_temp_bytes;
    size_t output_temp_bytes;
    size_t workspace_bytes;
};

bool ggml_hip_resolve_native_blas_call(
    ggml_backend_cuda_context & ctx,
    const ggml_tensor * src0,
    const ggml_tensor * src1,
    const ggml_tensor * ids,
    ggml_tensor * dst,
    ggml_hip_blas_call_v1 * call);

bool ggml_hip_apply_blas_plan(
    const ggml_hip_blas_plan_v1 * plan,
    ggml_hip_blas_call_v1 * call);

// Validate the resolved call and, when present, the native-only plan before
// the shared vendor forwarder is entered.  A null plan is the ordinary native
// fallback; it still requires a well-formed resolved call.
bool ggml_hip_blas_plan_matches_call(
    const ggml_hip_blas_plan_v1 * plan,
    const ggml_hip_blas_call_v1 * call);

void ggml_hip_execute_blas_call(
    ggml_backend_cuda_context & ctx,
    const ggml_hip_blas_call_v1 & call,
    const ggml_hip_launch_context & lc);

// Hard eligibility, evaluated before any launch (standards 12.4). Each answers
// "could this variant run at all", never "would it be fast".
bool ggml_cuda_mmq_variant_is_eligible(
    ggml_type type, int j, bool fallback, int cc, size_t shared_mem_limit,
    int64_t ncols_max);

// Whether MMQ implements this type at all on this device, derived from
// upstream's per-architecture config tables. Pass 0 for cc and
// shared_mem_limit to have the current device's values filled in.
bool ggml_cuda_mmq_type_is_supported(
    ggml_type type, int cc, size_t shared_mem_limit);

// The J native's own scan would choose for this shape, or 0 if nothing is
// eligible. A forced candidate at this J runs identical code to native, so the
// pair calibrates measurement noise for free (HI24).
int ggml_cuda_mmq_native_j_best(ggml_type type, bool fallback, int64_t ncols_max);

bool ggml_cuda_mmvf_variant_is_eligible(
    ggml_type type, int block_size, bool acc_f16, int warp_size,
    size_t shared_mem_limit, int64_t ncols, int64_t width, bool has_fusion);

bool ggml_cuda_mmf_variant_is_eligible(
    ggml_type type, int nwarps, int cc, int warp_size,
    size_t shared_mem_limit, int64_t width, int64_t rows_per_block);

// ------------------------------------------------------------ the three steps

// Upstream's policy, expressed as a decision rather than a launch. Reproduces
// the branch order of `ggml_cuda_mul_mat` / `ggml_cuda_mul_mat_id` exactly.
ggml_hip_native_selection ggml_hip_native_select(
    ggml_backend_cuda_context & ctx, const ggml_tensor * src0,
    const ggml_tensor * src1, const ggml_tensor * ids, const ggml_tensor * dst);

// HI158: the native selection, computed ON DEMAND rather than up front.
//
// ggml_hip_native_select() costs ~193.6ns and the L1 warm-cache lookup it
// feeds costs ~52.6ns (real gfx1100, 110,020-dispatch sweep, 99.8% L1 hit
// rate). On a hit the resolver overwrites the native seed immediately, so
// that ~194ns was spent producing a value nothing read -- on 99.8% of
// dispatches.
//
// This holds the inputs instead of the answer and computes at most once, when
// something actually needs it. Deliberately a plain struct with no
// std::function and no allocation: it sits on the hottest path in the system.
//
// Semantics are IDENTICAL to eager evaluation -- same function, same inputs,
// same result -- which is why this was chosen over caching native selections
// keyed by signature. That alternative would need proof that the signature
// captures every input native_select reads (cc/warp_size, shapes incl.
// ne11/ne12/dst->ne[2], contiguity, a padding condition); if it missed one,
// two different selections would collide on a key and the cache would return
// the WRONG kernel. Deferral needs no such proof.
struct ggml_hip_native_provider {
    ggml_backend_cuda_context * ctx;
    const ggml_tensor *         src0;
    const ggml_tensor *         src1;
    const ggml_tensor *         ids;
    const ggml_tensor *         dst;
    mutable ggml_hip_native_selection cached;
    mutable bool                      computed;
};

ggml_hip_native_provider ggml_hip_make_native_provider(
    ggml_backend_cuda_context & ctx, const ggml_tensor * src0,
    const ggml_tensor * src1, const ggml_tensor * ids, const ggml_tensor * dst);

// A provider that already has its answer and will never call native_select.
// The family collection points need this: standards 11.1 says the family is
// already decided there, so native selection is deliberately NOT consulted --
// a fused MMVQ must never be re-selected into MMQ. Expressing that as a
// pre-resolved provider keeps "the resolver takes a provider" true everywhere
// instead of carving out a second resolver entry point.
ggml_hip_native_provider ggml_hip_make_native_provider_resolved(
    const ggml_hip_native_selection & selection);

// Force the selection, computing it at most once. Every caller that genuinely
// needs the native answer goes through this; the L1-hit path must not.
const ggml_hip_native_selection & ggml_hip_native_force(
    const ggml_hip_native_provider & provider);

// Resolve a signature to the candidate that should run.
//
// Cold path on first sight of a dispatch key: hash, look up, cache. Warm path
// is a dictionary hit on the digest with no hashing at all (standards 15.2).
// In native mode, and on any miss, this returns the native selection
// unchanged -- forcing the provider at that point, not before.
// `lc` is needed because tune mode measures candidates from inside the
// resolver, and a measurement has to launch the real operation with real
// arguments. In every other mode it is unused.
ggml_hip_resolved_dispatch ggml_hip_dispatch_resolve(
    ggml_backend_cuda_context & ctx,
    const ggml_hip_dispatch_signature_v1 & sig,
    const ggml_hip_native_provider & native,
    const ggml_hip_launch_context & lc);

// Execute the resolved candidate's complete path.
void ggml_hip_dispatch_launch(const ggml_hip_resolved_dispatch & bound,
                              const ggml_hip_launch_context & lc);

// The entry point the patched `ggml_cuda_mul_mat` calls. Returns false when
// the dispatch layer declines to handle this operation, in which case the
// caller runs its original code unchanged -- which is how a build with the
// layer compiled in but inactive stays exactly upstream.
bool ggml_hip_dispatch_mul_mat(
    ggml_backend_cuda_context & ctx, const ggml_tensor * src0,
    const ggml_tensor * src1, const ggml_tensor * ids, ggml_tensor * dst,
    const ggml_cuda_mm_fusion_args_host * fusion);

// Entry point for the *family* collection sites (standards 9.1 sites 3-5):
// the fused graph paths and the public family entries, which the graph
// optimiser calls directly without passing through the dense selector.
//
// Differs from ggml_hip_dispatch_mul_mat in one essential way: it never
// re-selects the family. Standards 11.1 makes a fused graph pattern a distinct
// semantic operation, tuned within its already-chosen family -- comparing a
// fused MMVQ against an unfused MMQ decomposition is a graph-level question,
// not a matmul-level one. So this resolves geometry only, inside `family`.
//
// Returns false when the caller should run its own code unchanged: native mode,
// a re-entrant call already handled by dispatch, or a declined operation.
bool ggml_hip_dispatch_family(
    ggml_backend_cuda_context & ctx, const ggml_tensor * src0,
    const ggml_tensor * src1, const ggml_tensor * ids, ggml_tensor * dst,
    const ggml_cuda_mm_fusion_args_host * fusion,
    ggml_hip_kernel_family family);

// True while a dispatched candidate is executing.
//
// A dispatched launch re-enters its own family entry point, so anything
// counting or hooking there sees the operation twice. Callers use this to act
// only on the outermost entry.
bool ggml_hip_dispatch_is_reentrant();

// Mark a region as "already inside dispatch".
//
// The family entry points are collection sites (HI13), so *any* launch made
// from inside the dispatch layer re-enters them. For a normal dispatched
// launch that is handled by the scope held in ggml_hip_dispatch_launch. The
// tuner needs the same protection around its measurement launches: without it
// a measured launch re-enters the family hook, which resolves, which calls the
// tuner, which launches -- unbounded recursion until the stack dies.
//
// Exposed as enter/leave rather than an RAII type because the guard lives in
// dispatch.cu's anonymous namespace and callers are in other translation
// units. Prefer ggml_hip_dispatch_scope below to calling these directly.
void ggml_hip_dispatch_scope_enter();
void ggml_hip_dispatch_scope_leave();

struct ggml_hip_dispatch_scope {
    ggml_hip_dispatch_scope()  { ggml_hip_dispatch_scope_enter(); }
    ~ggml_hip_dispatch_scope() { ggml_hip_dispatch_scope_leave(); }
};

// ------------------------------------------------------------------ registry

size_t ggml_hip_registry_size();
const ggml_hip_candidate_descriptor * ggml_hip_registry_at(size_t index);
const ggml_hip_candidate_descriptor * ggml_hip_registry_find(const char * stable_name);

// The registry's native wrapper for a family. Every family has exactly one;
// the manifest schema refuses to validate otherwise.
const ggml_hip_candidate_descriptor * ggml_hip_registry_native(
    ggml_hip_kernel_family family);

// Current mode, after applying build capability to GGML_HIP_DISPATCH_MODE.
// A replay-only build asked to tune reports NATIVE rather than pretending.
int ggml_hip_dispatch_mode();

#endif // GGML_USE_HIP && GGML_HIP_DISPATCH
