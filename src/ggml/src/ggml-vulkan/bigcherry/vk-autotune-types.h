// bigcherry: Vulkan measured-dispatch identity types (RE30 phase 2).
//
// Mirrors src/ggml/src/ggml-cuda/hip-autotune-types.h's identity model
// rather than inventing a divergent shape, per RE30's own design notes.
// Naming follows this project's OWN ggml_hip_*/ggml_vk_* subsystem
// convention (not upstream ggml-vulkan.cpp's internal lowercase vk_*
// types, which are a different, unrelated naming scheme for unrelated
// implementation-detail structs).
//
// ABI STABILITY (GPT review finding, 2026-08-20 -- corrected from an
// earlier draft that called this "stable ABI types"): only the PERSISTENT
// identity types (ggml_vk_dispatch_signature_v1, ggml_vk_hardware_key_v1,
// the digest/replay-entry shapes) get a real stability promise, the same
// one HIP's schema_version fields make -- bump the version when a hashed
// field changes, never reinterpret silently. `ggml_vk_candidate_descriptor`
// and `ggml_vk_can_execute_fn`'s signature are explicitly PROVISIONAL: this
// header's hardware key deliberately omits extensions/limits/driver
// version (they travel as canonical JSON, not a fixed struct -- see
// ggml_vk_hardware_key_v1 below), so `can_execute` cannot yet evaluate the
// real Vulkan eligibility question ("does this device support the
// extension/limits this candidate needs"). Phase 3 will very likely need
// to widen what `can_execute`/`ggml_vk_candidate_descriptor` receive, which
// may not be an additive change. Do not treat this struct as frozen ABI
// before that decision is made with real Vulkan capability data in hand.
//
// UNINTEGRATED SCAFFOLDING (2026-08-20): this header is not included by
// any .cpp/.cu file yet, defines no runtime hook, and is not wired into
// any CMake target. It compiles standalone -- no <vulkan/vulkan.h>, no
// dependency on ggml-vulkan.cpp's internal types -- so it can be reviewed
// and unit-adjacent-tested (structurally, via the Python mirror in
// tools/bigcherry/vk_autotune_types.py) before any dispatch-hook code
// exists to call it. The real measurement/dispatch code that runs ON a
// Vulkan device (record, tune, replay resolver bodies) is RE30 phase 3,
// real-hardware-only, and is explicitly NOT in this file.
//
// Runtime identity and persistent identity are deliberately different
// things, same rule as HIP's ggml_hip_dispatch_signature_v1:
//   * `runtime_id` is a per-build uint32 index into the registry. Cheap,
//     and meaningless outside this binary.
//   * `stable_name` is the durable identity used in every database row
//     and cache entry.
// Never persist a runtime_id.

#pragma once

#include <stddef.h> // size_t (ggml_vk_workspace_fn's return type)
#include <stdint.h>

// Gated the same way HIP's autotune headers are gated on GGML_USE_HIP +
// GGML_HIP_DISPATCH: these files would otherwise compile into every
// Vulkan build, including ones that carry none of the machinery they
// depend on. GGML_VULKAN_AUTOTUNE does not exist as a build option yet
// (RE30 phase 1 owns introducing it to campaign_build.py/CMake) -- this
// guard names the flag the way it will be introduced, so the header is
// inert (compiles to nothing) until that flag is real.
#if defined(GGML_USE_VULKAN) && defined(GGML_VULKAN_AUTOTUNE)

// Bump when a hashed field is added, removed, or reinterpreted. Older
// replay caches and database rows are then rejected rather than misread.
// Independent of GGML_HIP_SIGNATURE_SCHEMA_VERSION/HARDWARE_SCHEMA_VERSION
// -- Vulkan and HIP identity evolve on separate schedules.
#define GGML_VK_SIGNATURE_SCHEMA_VERSION 1
#define GGML_VK_HARDWARE_SCHEMA_VERSION  1

// 128-bit blake2b digest, same construction as ggml_hip_digest. Persisted
// verbatim into the vk_* database tables (sql/dispatch-db.sql).
#define GGML_VK_DIGEST_BYTES 16

struct ggml_vk_digest {
    uint8_t bytes[GGML_VK_DIGEST_BYTES];
};

// ------------------------------------------------------------------ families
//
// A Vulkan "family" is a major algorithmic path, same rule as HIP's
// ggml_hip_kernel_family (standards 1): candidates never cross family
// boundaries. Starts narrow -- RE30's staged rollout begins with exactly
// one operation family (MUL_MAT) and grows this enum only as later phases
// add real, measured families, never speculatively ahead of them.
enum ggml_vk_kernel_family {
    GGML_VK_FAMILY_MUL_MAT    = 0,
    GGML_VK_FAMILY_MUL_MAT_ID = 1,
    GGML_VK_FAMILY_COUNT
};

// Mirrors ggml_hip_source_class's vocabulary exactly (same meaning, same
// build-decision role: which candidates need new shader compilation vs.
// reuse an existing SPIR-V module already in the tree).
enum ggml_vk_source_class {
    GGML_VK_SOURCE_NATIVE_WRAPPER        = 0,
    GGML_VK_SOURCE_EXISTING_RUNTIME      = 1,
    GGML_VK_SOURCE_EXISTING_ALTERNATIVE  = 2,
    GGML_VK_SOURCE_NEW_GENERATED_VARIANT = 3,
    GGML_VK_SOURCE_VENDOR_AUTO           = 4,
    GGML_VK_SOURCE_VENDOR_EXPLICIT       = 5,
    GGML_VK_SOURCE_COUNT
};

// -------------------------------------------------------------- hardware key
//
// Executing GPU+driver *class*, never a device ordinal (same standards-10
// sharing rule as HIP's hardware key: two identical GPUs in one box must
// produce the same key so they can share a winner). Vulkan needs more axes
// than HIP's single architecture_code because ICD/driver/extension surface
// varies more across vendors than ROCm's gfx target strings do (RE30
// detailed_solution's "Identity and persistence" section).
enum ggml_vk_subgroup_op_flag {
    GGML_VK_SUBGROUP_BASIC      = 1u << 0,
    GGML_VK_SUBGROUP_VOTE       = 1u << 1,
    GGML_VK_SUBGROUP_ARITHMETIC = 1u << 2,
    GGML_VK_SUBGROUP_BALLOT     = 1u << 3,
    GGML_VK_SUBGROUP_SHUFFLE    = 1u << 4,
    GGML_VK_SUBGROUP_CLUSTERED  = 1u << 5,
    GGML_VK_SUBGROUP_QUAD       = 1u << 6,
};

struct ggml_vk_hardware_key_v1 {
    uint16_t schema_version;      // GGML_VK_HARDWARE_SCHEMA_VERSION
    uint32_t vendor_id;
    uint32_t device_id;
    uint16_t subgroup_size;
    uint32_t subgroup_ops_mask;   // ggml_vk_subgroup_op_flag
    // extensions/limits/driver+API version/shader-toolchain fingerprint are
    // variable-length and travel as canonical JSON (see
    // ggml_vk_hardware_json below), same split HIP's hardware key does not
    // need but Vulkan's ICD surface does -- a fixed-width struct cannot
    // hold an open-ended extension list.
};

// ------------------------------------------------------------------ signature
//
// Canonical device-local description of one Vulkan operation (same rule as
// HIP's ggml_hip_dispatch_signature_v1, standards 5). Every field here is
// hashed; diagnostic identity (model name, layer index, pointers, device
// ordinal, wall clock) is deliberately absent -- it travels beside the
// signature as observation metadata (standards 5.1, 15.1).
enum ggml_vk_signature_flag {
    GGML_VK_SIG_SRC0_CONTIGUOUS = 1u << 0,
    GGML_VK_SIG_SRC1_CONTIGUOUS = 1u << 1,
    GGML_VK_SIG_DST_CONTIGUOUS  = 1u << 2,
    GGML_VK_SIG_HAS_IDS         = 1u << 3, // MUL_MAT_ID
    GGML_VK_SIG_BROADCAST_CH    = 1u << 4,
    GGML_VK_SIG_BROADCAST_SMP   = 1u << 5,
};

enum ggml_vk_layout {
    GGML_VK_LAYOUT_ROW_MAJOR = 0,
    GGML_VK_LAYOUT_COL_MAJOR = 1,
    GGML_VK_LAYOUT_COOPMAT   = 2,
};

enum ggml_vk_conversion_route {
    GGML_VK_CONVERSION_NONE             = 0,
    GGML_VK_CONVERSION_DEQUANT_ONCE     = 1, // e.g. RD68a: dequant q8_0 KV once
    GGML_VK_CONVERSION_TILED_TRANSPOSE  = 2, // e.g. RD68b: tiled 0<->2 transpose
};

// Mirrors ggml_hip_fusion_kind's role: whether/how this dispatch fuses a
// following op into the same launch (standards-5 hard signature identity --
// GPT review, 2026-08-20, caught this struct having no field for the
// "fusion" the vk_signature SQL table's own comment already documents;
// added here so the persistent schema and the runtime struct describe the
// same identity, matching the HIP side's fusion/glu_op split).
enum ggml_vk_fusion_kind {
    GGML_VK_FUSION_NONE      = 0,
    GGML_VK_FUSION_BIAS      = 1,
    GGML_VK_FUSION_GATE      = 2,
    GGML_VK_FUSION_GATE_BIAS = 3,
    GGML_VK_FUSION_GLU       = 4,
};

struct ggml_vk_dispatch_signature_v1 {
    uint16_t schema_version;  // GGML_VK_SIGNATURE_SCHEMA_VERSION
    uint16_t op;              // ggml_op
    uint8_t  src0_type;       // ggml_type
    uint8_t  src1_type;
    uint8_t  dst_type;
    uint8_t  output_precision;
    uint8_t  accumulation_precision;
    uint8_t  layout;          // ggml_vk_layout
    uint8_t  fusion;          // ggml_vk_fusion_kind
    uint16_t flags;           // ggml_vk_signature_flag

    int64_t  ne0[4];          // device-local src0 extents
    int64_t  ne1[4];          // device-local src1 extents
    int64_t  ned[4];          // device-local dst extents

    uint8_t  conversion_route; // ggml_vk_conversion_route
    uint8_t  split_k;          // 0 = disabled

    // ---- optional refinements (mirrors HIP's has_refinements split) ----
    uint8_t  has_refinements;
    uint8_t  alignment_class;
    uint8_t  batching_class;
};

// ------------------------------------------------------------------ candidate
//
// A Vulkan candidate is a complete executable *pipeline recipe*
// (preparation + main pipeline + optional reduction), never a bare
// dispatch -- RE30 detailed_solution's "pinned upstream seam" note: timing
// only the terminal kernel would let a candidate hide its quantisation or
// pipeline-creation cost outside the timed sample and win on a lie (same
// standards-7.1 rule HIP's launch_fn documents). `shader_stage_count` says
// how many command-buffer stages the recipe comprises.
struct ggml_vk_variant_params {
    int32_t workgroup_size_x;
    int32_t workgroup_size_y;
    int32_t tile_m;
    int32_t tile_n;
    int32_t tile_k;
    uint8_t coopmat;       // 1 = uses VK_KHR_cooperative_matrix
    uint8_t split_k;
    uint8_t shader_stage_count;
};

struct ggml_vk_candidate_descriptor;

// Hard eligibility, evaluated before any pipeline is created (mirrors
// ggml_hip_can_execute_fn: cheap, side-effect free, called across the
// whole catalog for every signature). Must reject unsupported
// extension/device/limits combinations before a candidate result is ever
// recorded (RE30 detailed_solution step 4).
typedef bool (*ggml_vk_can_execute_fn)(
    const ggml_vk_candidate_descriptor * self,
    const ggml_vk_dispatch_signature_v1 & sig,
    const ggml_vk_hardware_key_v1 & hw);

// Upper bound on scratch bytes this candidate will request for `sig`.
// Mirrors ggml_hip_workspace_fn's role exactly.
typedef size_t (*ggml_vk_workspace_fn)(
    const ggml_vk_candidate_descriptor * self,
    const ggml_vk_dispatch_signature_v1 & sig);

// NOTE: deliberately no `launch_fn` typedef here yet. HIP's
// ggml_hip_launch_fn takes a ggml_hip_launch_context built from CUDA/HIP
// stream + tensor pointers; the Vulkan equivalent needs a command-buffer +
// descriptor-set + push-constant launch context type that does not exist
// until RE30 phase 3 defines the real pipeline-recipe execution path (see
// RE30 detailed_solution's launch_recipe code sample). Adding a
// launch_fn typedef now, ahead of that context type, would either dangle
// or force a premature Vulkan-context shape decision -- phase 3's job, not
// this scaffolding's.
struct ggml_vk_candidate_descriptor {
    uint32_t     runtime_id;             // per-build index; never persisted
    const char * stable_name;            // durable identity
    uint8_t      family;                 // ggml_vk_kernel_family
    uint8_t      source_class;           // ggml_vk_source_class
    uint16_t     implementation_version;

    uint8_t      graph_safe;
    uint8_t      deterministic;
    uint8_t      is_native;
    uint8_t      reserved;

    ggml_vk_variant_params variant;

    ggml_vk_can_execute_fn can_execute;
    ggml_vk_workspace_fn   workspace;
    // launch intentionally omitted -- see note above.
};

#endif // GGML_USE_VULKAN && GGML_VULKAN_AUTOTUNE
