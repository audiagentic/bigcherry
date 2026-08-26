// bigcherry: stable ABI types for HIP measured dispatch.
//
// Everything here is versioned, because these structs outlive the process that
// wrote them: signatures and hardware keys are hashed into database and cache
// keys, and a replay cache written by one build is read by another. Any change
// to a hashed field is an ABI break, so each hashed struct carries an explicit
// schema version and the loader refuses caches that disagree.
//
// Runtime identity and persistent identity are deliberately different things:
//   * `runtime_id` is a per-build uint32 index into the registry. Cheap, and
//     meaningless outside this binary.
//   * `stable_name` ("mmq:q8_0:j64:fb0:...:v1") is the durable identity used in
//     every database row and cache entry (standards 2.1).
// Never persist a runtime_id.

#pragma once

#include "common.cuh"

#include <stdint.h>

// Standards 12.2: gated on GGML_USE_HIP *and* the feature flag. These files
// are picked up by the ggml-cuda/*.cu glob, so without the second condition
// they would compile into every HIP build, including ones that carry none of
// the machinery they depend on.
#if defined(GGML_USE_HIP) && defined(GGML_HIP_DISPATCH)

// HI121 (round 9): this is now a FROZEN identity epoch, not a counter bumped
// for every new semantic distinction. Additive/scoped semantic content (a
// new independent flag bit, a field scoped to the op classes that use it)
// and new producer capabilities (GGML_HIP_PRODUCER_CAPABILITIES_LO/HI below)
// never bump this -- HI121's whole point is that those are expressed as new
// capability bits (a compatibility-gate concern, checked offline, never
// hashed) instead of a global epoch bump that invalidates every existing
// signature's digest regardless of whether it's even affected.
//
// Bump ONLY when the canonical signature REPRESENTATION/encoding itself
// changes incompatibly -- the canonicalization algorithm changes, or an
// EXISTING hashed field's on-wire meaning is reinterpreted such that old and
// new semantics cannot be told apart through content + capability
// applicability alone. Older replay caches and database rows are then
// rejected rather than misread.
//
// v1 -> v2 (HI119 review follow-up): the `flags` bitfield's bits 7-10
// (GGML_HIP_SIG_FUSION_X_BIAS/GATE_BIAS/X_SCALE/GATE_SCALE, added by HI118)
// reinterpreted previously-unused bits with new load-bearing meaning, and
// schema 1 had no way to represent "producer evaluated bias/scale presence
// at all" -- a v1 signature's flags==0 there cannot be trusted to mean "no
// bias/scale present". That was a real representational-completeness gap,
// not an ordinary additive change, which is why it was the one case that
// did warrant a bump. See tools/bigcherry/tuning/dispatch_abi.py for the
// mirrored Python-side constant (SIGNATURE_IDENTITY_EPOCH) and its own
// agreement test against this #define.
#define GGML_HIP_SIGNATURE_SCHEMA_VERSION 2
#define GGML_HIP_HARDWARE_SCHEMA_VERSION  1

// HI121: source-owned declaration of what THIS producer's code knows how to
// correctly evaluate -- a distinct axis from the signature's own content
// (GGML_HIP_SIG_* flags above, which describe what a dispatch actually HAS).
// APPEND-ONLY: capability IDs are permanent once allocated, never reused or
// renamed. A discovered evaluator bug is fixed by allocating a NEW `*_V2`
// bit and updating the relevant applicability rule to require it --
// `*_V1` is never redefined, so old measurements that only ever claimed
// `*_V1` correctly stop qualifying once `*_V2` is required, rather than
// silently being treated as still-correct.
//
// This is a plain compile-time declaration, not inferred from any other
// code's behavior (deliberately -- see tools/bigcherry/tuning/hip_capabilities.py's
// own docstring for why behavioral/source-pattern inference was rejected as
// an unsafe capability authority). The compiled record/tuner producer emits
// these same two values in its own JSON header so a consumer can verify
// what was actually compiled, not just what a manifest claims was compiled.
//
// bit 0  CORE_SIGNATURE_V1              -- op/types/prec, coarse fusion+glu_op,
//                                          extents/strides, expert counts, flags 0-6
// bit 1  FUSION_X_BIAS_PRESENCE_V1      -- correctly evaluates fusion->x_bias
// bit 2  FUSION_GATE_BIAS_PRESENCE_V1   -- correctly evaluates fusion->gate_bias
// bit 3  FUSION_X_SCALE_PRESENCE_V1     -- correctly evaluates fusion->x_scale
// bit 4  FUSION_GATE_SCALE_PRESENCE_V1  -- correctly evaluates fusion->gate_scale
#define GGML_HIP_PRODUCER_CAPABILITIES_LO UINT64_C(0x000000000000001f)
#define GGML_HIP_PRODUCER_CAPABILITIES_HI UINT64_C(0x0000000000000000)

// 128-bit blake2b digest (standards 5.4). Persisted verbatim.
#define GGML_HIP_DIGEST_BYTES 16

struct ggml_hip_digest {
    uint8_t bytes[GGML_HIP_DIGEST_BYTES];
};

// ------------------------------------------------------------------ families

// A kernel family is a major algorithmic path, not a configuration of one
// (standards 1). Candidates never cross family boundaries.
enum ggml_hip_kernel_family {
    GGML_HIP_FAMILY_MMVQ = 0,
    GGML_HIP_FAMILY_MMQ  = 1,
    GGML_HIP_FAMILY_MMVF = 2,
    GGML_HIP_FAMILY_MMF  = 3,
    GGML_HIP_FAMILY_BLAS = 4,
    GGML_HIP_FAMILY_COUNT
};

// Standards 2.3. Drives build decisions: which candidates need new translation
// units and which reuse code already in the tree.
enum ggml_hip_source_class {
    GGML_HIP_SOURCE_NATIVE_WRAPPER        = 0,
    GGML_HIP_SOURCE_EXISTING_RUNTIME      = 1,
    GGML_HIP_SOURCE_EXISTING_ALTERNATIVE  = 2,
    GGML_HIP_SOURCE_NEW_GENERATED_VARIANT = 3,
    GGML_HIP_SOURCE_VENDOR_AUTO           = 4,
    GGML_HIP_SOURCE_VENDOR_EXPLICIT       = 5,
    GGML_HIP_SOURCE_COUNT
};

// -------------------------------------------------------------- hardware key

// The architecture enumeration is generated from the single source of truth in
// tools/bigcherry/autotune_schema.py, because the enumerator value *is* the bit
// position in architecture_mask. A hand-written copy that drifted would make
// candidates claim support for hardware they were never measured on.
#include "hip-autotune-arch.h"

// Capability bits that change which candidates are legal or fast. These belong
// in the hardware key because two GPUs with the same gfx name but different
// available shared memory are not interchangeable for tuning purposes.
enum ggml_hip_feature_flag {
    GGML_HIP_FEATURE_WMMA      = 1u << 0,
    GGML_HIP_FEATURE_MFMA      = 1u << 1,
    GGML_HIP_FEATURE_DP4A      = 1u << 2,
    GGML_HIP_FEATURE_FP16_MMA  = 1u << 3,
    GGML_HIP_FEATURE_GRAPHS    = 1u << 4,
};

// Executing GPU class. Contains no device ordinal and no PCIe identity: two
// identical GPUs in one box must produce the same key so they can share a
// winner (standards 10.2).
struct ggml_hip_hardware_key_v1 {
    uint16_t schema_version;  // GGML_HIP_HARDWARE_SCHEMA_VERSION
    uint16_t architecture_code;
    uint16_t wave_size;
    uint16_t compute_units;
    uint32_t feature_flags;
    uint32_t shared_memory_per_block; // smpbo; bounds MMQ/MMF eligibility
};

// ------------------------------------------------------------------ signature

enum ggml_hip_signature_flag {
    GGML_HIP_SIG_SRC0_CONTIGUOUS = 1u << 0,
    GGML_HIP_SIG_SRC1_CONTIGUOUS = 1u << 1,
    GGML_HIP_SIG_DST_CONTIGUOUS  = 1u << 2,
    GGML_HIP_SIG_HAS_IDS         = 1u << 3, // MUL_MAT_ID
    GGML_HIP_SIG_BROADCAST_CH    = 1u << 4, // nchannels_dst != nchannels_src0
    GGML_HIP_SIG_BROADCAST_SMP   = 1u << 5,
    GGML_HIP_SIG_BAD_PADDING     = 1u << 6, // forces the BLAS path upstream

    // HI118: ggml_hip_fusion_kind() collapses fusion->x_bias and
    // fusion->gate_bias into one coarse GGML_HIP_FUSION_BIAS/GATE_BIAS value
    // -- real, since mmvq.cu's kernel (~line 1435-1450) handles them as two
    // independently-nullable fields, and a dispatch could in principle carry
    // only one. These flags record which one(s) are actually present, so a
    // consumer reconstructing the fused computation (HI119) knows exactly
    // which synthetic bias tensor(s) to build rather than guessing from the
    // coarse fusion byte alone. Enough of each fusion tensor's real geometry
    // to construct a semantically equivalent synthetic replacement (not
    // necessarily its exact original ne/nb) is derivable from fields this
    // signature already records -- verified against real source, corrected
    // 2026-08-25 (dev-gpt-agent review) after an initial pass cited only
    // mmvq.cu's own GGML_ASSERT calls, which is NOT sufficient proof on its
    // own: ggml_are_same_stride() compares nb[] only, not ne[], so it alone
    // does not prove gate's shape matches src0's. The real proof is one layer
    // up, in ggml-cuda.cu's ggml_cuda_should_fuse_mul_mat() -- the gate for
    // fusion in the first place requires `ffn_up->src[0]->type ==
    // ffn_gate->src[0]->type && ggml_are_same_shape(...) &&
    // ggml_are_same_stride(...)` (~ggml-cuda.cu line 1998), so a fusion this
    // tuner ever sees already had gate's shape/type/stride matched against
    // src0's before it could fuse at all -- mmvq.cu's assert is a redundant
    // downstream safety check, not the source of the guarantee.
    // fusion->x_bias/gate_bias are F32 with `ne[0] == dst->ne[0]` and (if
    // ids) `ne[1] == src0->ne[2]` (== this signature's own n_expert field) --
    // sufficient to build a correctly-shaped synthetic bias, though not a
    // full round-trip of the original tensor's own ne[2]/ne[3]/nb[].
    // x_scale/gate_scale are F32, contiguous, sized `nelements == (ids ?
    // src0->ne[2] : 1)` under the upstream/pristine kernel path (NVFP4-only;
    // never observed on the real q8_0 dispatches HI108 found) -- enough to
    // build a synthetic per-expert/per-tensor scale array for that path.
    // KNOWN REPO-WIDE GAP, not fixed here: patch 1207_rd17_moe_topk_down_
    // fold.py's x_scale_channel_dst mode changes x_scale's real semantics
    // (length dst->ne[1], non-NVFP4-gated, indexed by destination channel
    // instead of expert) -- this X_SCALE presence flag does not distinguish
    // that mode from the upstream one; out of scope for HI118/HI119's
    // upstream-GLU-fusion case, filed separately for whoever hardens 1207's
    // own signature coverage. dst_gate is deliberately NOT tracked here: it
    // exists on ggml_cuda_mm_fusion_args_host only under patch
    // patches/rd/1205_rd12_paired_mmvq_dual_output/patch.py (RD12; an
    // experimental, non-default patch -- 1207 only comments on sharing RD12's
    // struct slot, it does not add the field itself, corrected 2026-08-25
    // after initially misattributing this to 1207), not in the
    // upstream/pristine struct every other build compiles against --
    // confirmed the hard way (a real Brutus build without 1205 failed with
    // "no member named 'dst_gate'"). RD12's own fused dispatch (gate set,
    // dst_gate set, glu_op == NONE) DOES reach this tuner's family collection
    // point and gets signed today -- it is not silently aliased with a real
    // GLU signature (GLU's own glu_op is never NONE), but it relies on the
    // implicit invariant "fusion == GATE with glu_op == NONE means RD12,"
    // not an explicit flag. Filed separately for whoever hardens RD12's own
    // signature coverage; out of scope for HI118/HI119's upstream-GLU case.
    GGML_HIP_SIG_FUSION_X_BIAS     = 1u << 7,
    GGML_HIP_SIG_FUSION_GATE_BIAS  = 1u << 8,
    GGML_HIP_SIG_FUSION_X_SCALE    = 1u << 9,
    GGML_HIP_SIG_FUSION_GATE_SCALE = 1u << 10,
};

// How a graph pattern fused this operation. A fused operation is a different
// semantic operation, not a variant of the unfused one (standards 11.1), so
// this is hard identity.
enum ggml_hip_fusion_kind {
    GGML_HIP_FUSION_NONE       = 0,
    GGML_HIP_FUSION_BIAS       = 1,
    GGML_HIP_FUSION_GATE       = 2,
    GGML_HIP_FUSION_GATE_BIAS  = 3,
    GGML_HIP_FUSION_GLU        = 4,
};

// Canonical device-local description of one operation (standards 5).
//
// Every field here is hashed. Diagnostic identity -- model name, layer index,
// pointer values, device ordinal, wall clock -- is deliberately absent: it
// travels beside the signature as observation metadata and must never reach the
// digest (standards 5.1, 15.1).
//
// The dimensions are the *device-local* slice, filled in after tensor-split
// slicing, so two GPUs given unequal shares of a node get different signatures
// (standards 5.2).
struct ggml_hip_dispatch_signature_v1 {
    uint16_t schema_version;   // GGML_HIP_SIGNATURE_SCHEMA_VERSION
    uint16_t op;               // ggml_op
    uint8_t  src0_type;        // ggml_type
    uint8_t  src1_type;
    uint8_t  dst_type;
    uint8_t  prec;             // ggml_prec of the request
    uint8_t  fusion;           // ggml_hip_fusion_kind
    uint8_t  glu_op;           // ggml_glu_op, when fusion == GLU
    uint16_t flags;            // ggml_hip_signature_flag

    int64_t  ne0[4];           // device-local src0 extents
    int64_t  ne1[4];           // device-local src1 extents
    int64_t  ned[4];           // device-local dst extents
    int64_t  nb0[4];           // strides, in elements, not bytes
    int64_t  nb1[4];
    int64_t  nbd[4];

    int64_t  n_expert;         // MUL_MAT_ID only, else 0
    int64_t  n_expert_used;

    // ---- optional refinements (standards 5.5) --------------------------
    // Present only when a measurement proved they change the winner. A
    // signature with refinements hashes to a refined key whose lookup falls
    // back to the base key; `has_refinements` says which digest to build.
    uint8_t  has_refinements;
    uint8_t  alignment_class;  // log2 of the common alignment of src0/src1/dst
    uint8_t  occupancy_bucket; // MoE routing density bucket
    uint8_t  offset_modulo;
};

// ------------------------------------------------------------------ candidate

// The performance-variant knob a candidate turns. Each family reads only the
// fields it owns; the rest are zero. Kept as one flat struct rather than a
// union so it can be memcpy'd into a launch record during graph capture.
struct ggml_hip_variant_params {
    int32_t primary;    // MMQ: J. MMVF: block_size. MMF: nwarps. MMVQ: nwarps.
    int32_t secondary;  // MMVQ: rows_per_block. Others: 0.
    int32_t width;      // ncols_dst the variant is compiled for, 0 = any.
    uint8_t acc_f16;    // MMVF only: 1 = F16 accumulator (standards 3.2).
    uint8_t fallback;   // MMQ only: the `fallback` template argument.
    // MMVQ only: the `small_k` template argument. It is a compiled-in dimension
    // of the instance, not a runtime branch, so a candidate that cannot request
    // it cannot name what actually ran -- and the catalog has always enumerated
    // it (`:sk0:`/`:sk1:` in the stable name). Taken from `reserved` so the
    // struct size is unchanged.
    uint8_t small_k;
    // The src0 type this candidate was built for, as a ggml_type. Every family
    // has always carried it in the stable name (`mmq:q8_0:`, `mmvf:f32:`) and
    // none of them carried it here, which made it unavailable to eligibility.
    //
    // That is not a naming nicety. Every family's launch path dispatches on the
    // *runtime* tensor type and applies the candidate's variant to whatever it
    // finds, so a candidate is only meaningful for its own type:
    //
    //   - MMVQ instances are generated per (type, geometry); a q8_0 candidate
    //     on a q4_0 matmul finds no compiled instance and hits the HI09 abort.
    //   - MMQ's config table is per type, so `j16` names a different kernel
    //     configuration for iq1_s than for q8_0. Forcing one type's J onto
    //     another reaches the device-side NO_DEVICE_CODE guard (mmq.cuh:98).
    //   - MMF/MMVF survive it only because upstream instantiates every type,
    //     but the winner is then recorded under a name that misstates what ran.
    //
    // The per-family `_is_eligible` helpers do not catch this: they are asked
    // whether the *signature's* type is servable, never whether it is the type
    // the candidate claims. Both crashes above were found this way.
    //
    // Only meaningful for generated variants; a native wrapper leaves it 0 and
    // is accepted before the check. Takes the last `reserved` byte, so the
    // struct size is unchanged.
    uint8_t src0_type;
};

// HI17 BLAS-1 plan identity. These fields live beside the registry rather than
// in ggml_hip_variant_params: replay v3 serializes the latter's exact payload,
// while a structured BLAS plan is runtime metadata until a later slice adds a
// genuinely different launcher. The generated registry carries a side-table
// pointer indexed by runtime_id.
enum ggml_hip_blas_operand_type {
    GGML_HIP_BLAS_OPERAND_NATIVE = 0,
    GGML_HIP_BLAS_OPERAND_F32    = 1,
    GGML_HIP_BLAS_OPERAND_F16    = 2,
    GGML_HIP_BLAS_OPERAND_BF16   = 3,
};

enum ggml_hip_blas_accumulation_type {
    GGML_HIP_BLAS_ACCUMULATION_NATIVE = 0,
    GGML_HIP_BLAS_ACCUMULATION_F16    = 1,
    GGML_HIP_BLAS_ACCUMULATION_F32    = 2,
};

enum ggml_hip_blas_output_type {
    GGML_HIP_BLAS_OUTPUT_NATIVE = 0,
    GGML_HIP_BLAS_OUTPUT_F16    = 1,
    GGML_HIP_BLAS_OUTPUT_BF16   = 2,
    GGML_HIP_BLAS_OUTPUT_F32    = 3,
};

enum ggml_hip_blas_conversion_route {
    GGML_HIP_BLAS_CONVERSION_NONE           = 0,
    GGML_HIP_BLAS_CONVERSION_CONTIGUOUS     = 1,
    GGML_HIP_BLAS_CONVERSION_NON_CONTIGUOUS = 2,
};

enum ggml_hip_blas_output_conversion {
    GGML_HIP_BLAS_OUTPUT_CONVERSION_NONE            = 0,
    GGML_HIP_BLAS_OUTPUT_CONVERSION_TEMPORARY_TO_F32 = 1,
};

enum ggml_hip_blas_numerical_class {
    GGML_HIP_BLAS_NUMERICAL_EXACT_BASELINE = 0,
    GGML_HIP_BLAS_NUMERICAL_EQUIVALENT     = 1,
    GGML_HIP_BLAS_NUMERICAL_REDUCED        = 2,
};

struct ggml_hip_blas_plan_v1 {
    uint8_t operand_type;
    uint8_t accumulation_type;
    uint8_t output_type;
    uint8_t source_a_conversion;
    uint8_t source_b_conversion;
    uint8_t output_conversion;
    uint8_t numerical_class;
};

// Everything a launch needs that is not in the signature. Mirrors the arguments
// upstream's family entry points already take, so a native_wrapper candidate is
// a direct forward with no marshalling.
struct ggml_hip_launch_context {
    ggml_backend_cuda_context * ctx;
    const ggml_tensor *         src0;
    const ggml_tensor *         src1;
    const ggml_tensor *         ids;   // MUL_MAT_ID only, else nullptr
    ggml_tensor *               dst;
    ggml_cuda_mm_fusion_args_host fusion;
    bool                        has_fusion;
    cudaStream_t                stream;
};

struct ggml_hip_candidate_descriptor;

// Hard eligibility, evaluated before any launch is attempted (standards 12.4).
// Returning false must be cheap and must never have side effects: the tuner
// calls it across the whole catalog for every signature.
typedef bool (*ggml_hip_can_execute_fn)(
    const ggml_hip_candidate_descriptor * self,
    const ggml_hip_dispatch_signature_v1 & sig,
    const ggml_hip_hardware_key_v1 & hw);

// Execute the complete candidate path -- setup, quantization, main kernel,
// reductions, output conversion. Not just the terminal kernel: the tuner times
// exactly what this function does, and a candidate that hid its quantization
// cost outside it would win on a lie (standards 7.1).
typedef void (*ggml_hip_launch_fn)(
    const ggml_hip_candidate_descriptor * self,
    const ggml_hip_launch_context & lc);

// Upper bound on scratch bytes this candidate will request for `sig`. Used to
// filter candidates against max_workspace_bytes before measuring, and as a
// tiebreaker afterwards (standards 7.3).
typedef size_t (*ggml_hip_workspace_fn)(
    const ggml_hip_candidate_descriptor * self,
    const ggml_hip_dispatch_signature_v1 & sig);

struct ggml_hip_candidate_descriptor {
    uint32_t     runtime_id;             // per-build index; never persisted
    const char * stable_name;            // durable identity (standards 2.1)
    uint8_t      family;                 // ggml_hip_kernel_family
    uint8_t      source_class;           // ggml_hip_source_class
    uint16_t     implementation_version; // the vN suffix of stable_name

    // Bit i set means the candidate supports architecture code i. One fat
    // binary can carry candidates for several architectures without any
    // runtime string comparison (standards 2.6).
    uint64_t     architecture_mask;

    uint8_t      graph_safe;    // launchable inside HIP graph capture/update
    uint8_t      deterministic; // bitwise-identical output for identical input
    uint8_t      is_native;     // this is the upstream policy's own choice
    uint8_t      reserved;

    ggml_hip_variant_params variant;

    ggml_hip_can_execute_fn can_execute;
    ggml_hip_launch_fn      launch;
    ggml_hip_workspace_fn   workspace;
};

// What upstream's own policy chose, expressed as a candidate rather than as a
// launch that already happened (standards 4.1). Native selection is a pure
// function; nothing is dispatched until `launch`.
struct ggml_hip_native_selection {
    const ggml_hip_candidate_descriptor * candidate;
    ggml_hip_variant_params               variant;
    bool                                  valid;
};

#ifdef GGML_HIP_ROUTING_TRANSFORM
// Forward-declared: the full definition (further down this file) needs
// GGML_HIP_ROUTING_TRANSFORM too, but a pointer member does not need the
// complete type.
struct ggml_hip_routing_transformation;
#endif

// The result of resolving a dispatch key: the candidate to launch, plus room
// for per-signature state the resolver computed once and the launch can reuse.
struct ggml_hip_resolved_dispatch {
    const ggml_hip_candidate_descriptor * candidate;
    ggml_hip_variant_params               variant;
    void *                                prepared_state; // owned by the resolver
    bool                                  from_cache;     // false = native fallback
#ifdef GGML_HIP_ROUTING_TRANSFORM
    // HI31: nullptr (the default -- see every construction site) means
    // `candidate` is reached directly; a real pointer means it was only
    // reachable through this routing transformation (HI27/HI28), and the
    // dispatch-time launch must route through ggml_hip_transform_launch()
    // rather than calling candidate->launch() directly. See hip-autotune-
    // dispatch.cu's ggml_hip_dispatch_launch() for the fast-path contract:
    // a nullptr check here is the ONLY cost a non-transformed binding pays.
    const ggml_hip_routing_transformation * transform = nullptr;
#endif
};

// -------------------------------------------------------- routing transforms
//
// A routing transformation rewrites one launch context into another that a
// *different* family can serve. It exists because a signature's family is not
// a property of the arithmetic -- it is a property of how the operands happen
// to be arranged. Upstream already proves the point: ggml-cuda.cu:1879 takes
// F32[1xK] x F32[KxN] with N above MMVF's batch bound, swaps the operands, and
// runs MMVF against a reshaped dst rather than falling to BLAS.
//
// What that path does by hand for one shape, this models generally: a
// transformation is a pure function from (signature, launch context) to a new
// pair, plus the mathematical argument for why the two compute the same thing.
// The tuner then measures both and keeps whichever is faster -- so a
// transformation is a *hypothesis*, never an assumption.
//
// Everything here is compiled out unless GGML_HIP_ROUTING_TRANSFORM is on.
#ifdef GGML_HIP_ROUTING_TRANSFORM

enum ggml_hip_transform_id {
    GGML_HIP_TRANSFORM_NONE = 0,
    GGML_HIP_TRANSFORM_TRANSPOSE_WEIGHT_FOR_MMVF = 1,
    GGML_HIP_TRANSFORM_BATCH_FOR_MMVF            = 2,
    // 3-15 reserved for further predefined transforms. Agent-discovered ones
    // start at 16 so an id alone says which half of the registry it came from
    // even in a record written by a build that no longer has it.
    GGML_HIP_TRANSFORM_DISCOVERED_BASE = 16,
};

// Deliberately *not* GGML_HIP_SOURCE_*: that prefix already belongs to
// ggml_hip_source_class above, and two unrelated enums sharing it would make
// `GGML_HIP_SOURCE_PREDEFINED` read as a fifth source class.
enum ggml_hip_transform_source {
    GGML_HIP_TRANSFORM_SOURCE_PREDEFINED = 0,  // hand-written, equivalence proved
    GGML_HIP_TRANSFORM_SOURCE_DISCOVERED = 1,  // agent-proposed, must pass validation
};

// Scratch the *caller* owns for the life of one transformed launch.
//
// Applying a transform allocates nothing. A rewritten launch context still has
// to point at `ggml_tensor` headers describing the new shape, so the caller
// puts this on its stack and `apply` fills it in. That matters because HI31
// applies a transform on every dispatched launch of a transformed binding: a
// heap allocation there would be a per-call cost paid to avoid a per-call cost.
struct ggml_hip_transform_ctx {
    ggml_tensor src0;
    ggml_tensor src1;
    ggml_tensor dst;

    // Batch decomposition. A transform that produces exactly one launch leaves
    // n_batches at 1, and `select_batch` is null.
    int64_t n_batches   = 1;
    int64_t batch_width = 0;   // dst columns per batch
    int64_t total_width = 0;   // dst columns in the original operation

    // Unmoving base pointers for a batching transform's `select_batch`, so
    // re-pointing at batch i is arithmetic on the original data pointer
    // rather than repeated mutation of an already-offset one.
    char * src1_base = nullptr;
    char * dst_base  = nullptr;
};

// Hard applicability, from the signature alone. Must be cheap and side-effect
// free: the tuner asks every transform about every single-family signature.
typedef bool (*ggml_hip_transform_can_apply_fn)(
    const ggml_hip_dispatch_signature_v1 & sig);

// Rewrite orig_lc into (*out_sig, *out_lc), using `ctx` as storage for the
// rewritten tensor headers. No separate orig_sig parameter: everything an
// apply function needs is already reachable from orig_lc's tensor headers,
// and out_sig is rebuilt from the rewritten headers (never copied from
// orig_sig and hand-edited -- see the note on signature_from_ctx). Returns
// false if the rewrite could not be built -- which `can_apply` should
// already have prevented, so a false here is a bug rather than a routine
// outcome.
typedef bool (*ggml_hip_transform_apply_fn)(
    const ggml_hip_launch_context &        orig_lc,
    ggml_hip_transform_ctx *               ctx,
    ggml_hip_dispatch_signature_v1 *       out_sig,
    ggml_hip_launch_context *              out_lc);

// Re-point `ctx` at batch `index` of its decomposition. Null for a transform
// that produces a single launch. The last batch may be narrower than
// batch_width; the implementation adjusts the shape, not just the pointers.
// No `lc` parameter: `lc` already points at ctx's headers (see
// launch_context_from_ctx), so mutating ctx in place is what re-points the
// launch -- there is nothing in lc itself for a batch selector to touch.
typedef bool (*ggml_hip_transform_select_batch_fn)(
    ggml_hip_transform_ctx * ctx,
    int64_t                  index);

// Scratch bytes beyond what the target candidate itself requests, so a
// transform that trades memory for a family change is filtered against
// max_workspace_bytes on the same terms as a candidate (standards 7.3).
typedef size_t (*ggml_hip_transform_overhead_fn)(
    const ggml_hip_dispatch_signature_v1 & sig);

struct ggml_hip_routing_transformation {
    ggml_hip_transform_id     id;
    const char *              name;    // durable identity, as for candidates
    ggml_hip_transform_source source;
    bool                      equivalence_verified;

    // The family this rewrite exists to reach. The tuner checks it can serve
    // the transformed signature before measuring anything: a transform whose
    // target still cannot serve is a rewrite that bought nothing.
    ggml_hip_kernel_family    target_family;

    ggml_hip_transform_overhead_fn     overhead_bytes;
    ggml_hip_transform_can_apply_fn    can_apply;
    ggml_hip_transform_apply_fn        apply;
    ggml_hip_transform_select_batch_fn select_batch;  // null = single launch
};

#endif // GGML_HIP_ROUTING_TRANSFORM

static inline uint64_t ggml_hip_arch_bit(int architecture_code) {
    return architecture_code >= 0 && architecture_code < 64
        ? (uint64_t(1) << architecture_code)
        : uint64_t(0);
}

static inline bool ggml_hip_candidate_supports_arch(
        const ggml_hip_candidate_descriptor & candidate,
        const ggml_hip_hardware_key_v1 & hw) {
    return (candidate.architecture_mask & ggml_hip_arch_bit(hw.architecture_code)) != 0;
}

#endif // GGML_USE_HIP && GGML_HIP_DISPATCH
