// bigcherry: compact immutable replay cache for Vulkan (RE30 phase 2).
//
// Mirrors src/ggml/src/ggml-cuda/hip-autotune-replay.h's shape and its
// fail-safe philosophy exactly: production resolves a dispatch key to a
// stored winner and launches it, nothing else. A cache from another build,
// a truncated file, a corrupted entry, or a winner whose provenance no
// longer matches all resolve to "use native selection", never to a wrong
// launch -- same reasoning HIP's header states: a production binary that
// silently ran the wrong pipeline would be far worse than one that quietly
// ran the upstream default.
//
// UNINTEGRATED SCAFFOLDING (2026-08-20): declares the resolver's shape and
// resolution vocabulary only. No .cpp implementation exists (there is
// nothing to implement against yet -- no Vulkan dispatch hook calls this,
// no writer produces a cache file). See vk-autotune-types.h's header
// comment for the same caveat, which applies here identically. RE30 phase
// 3 (real Vulkan hardware) is required before any of this can be
// implemented or tested end-to-end.

#pragma once

#include "vk-autotune-types.h"

#if defined(GGML_USE_VULKAN) && defined(GGML_VULKAN_AUTOTUNE)

#include <stddef.h>
#include <stdint.h>

// "BCVK", little-endian byte order (same convention as HIP's
// GGML_HIP_REPLAY_MAGIC == 0x59484342u == bytes "BCHY"): the constant's
// low byte is the string's first character. Distinct magic from HIP's so a
// Vulkan replay cache can never be loaded by a HIP reader or vice versa
// (RE30 detailed_solution: "give Vulkan replay a distinct magic/version").
// GPT review (2026-08-20) caught the original constant (0x4B435642)
// actually serializing to bytes "BVCK", not "BCVK" -- fixed here.
#define GGML_VK_REPLAY_MAGIC   0x4B564342u
#define GGML_VK_REPLAY_VERSION 1

// Header of the on-disk Vulkan replay cache. Fixed size, little-endian, no
// padding assumed -- same portability rule as ggml_hip_replay_header.
struct ggml_vk_replay_header {
    uint32_t magic;
    uint32_t format_version;
    uint32_t artifact_version;
    uint16_t signature_schema;
    uint16_t hardware_schema;
    uint32_t entry_count;
    uint32_t string_bytes;
    uint8_t  manifest_hash[GGML_VK_DIGEST_BYTES];
    uint8_t  content_digest[GGML_VK_DIGEST_BYTES]; // over entries + strings
};

// One stored winner.
struct ggml_vk_replay_entry {
    uint8_t  dispatch_digest[GGML_VK_DIGEST_BYTES];
    uint8_t  signature_digest[GGML_VK_DIGEST_BYTES];
    uint32_t stable_name_offset;   // into the string table
    uint16_t implementation_version;
    int32_t  variant_tile_m;
    int32_t  variant_tile_n;
    int32_t  variant_tile_k;
    uint8_t  variant_coopmat;
    uint8_t  variant_split_k;
    uint8_t  manifest_hash[GGML_VK_DIGEST_BYTES];
    uint8_t  source_revision_digest[GGML_VK_DIGEST_BYTES];
    uint32_t generation;
};

// Same five-way classification as ggml_hip_resolution_v2
// (hip-autotune-replay.h), same semantics per outcome -- kept identical on
// purpose rather than inventing a Vulkan-specific taxonomy, since the
// *reasons* a lookup can fail to produce a usable winner (missing entry,
// obsolete producer, corrupt cache, resolved-but-unusable candidate) are
// backend-independent:
//   exact                  -- a winner resolved AND is actually usable now.
//   miss                   -- no stored entry for this dispatch key.
//   rerun_required         -- cache/entries are from an obsolete-but-
//                              recognisable producer; a tune rerun fixes it.
//   incompatible           -- malformed/corrupt/structurally wrong cache.
//   candidate_unavailable  -- matched a stored winner that cannot be used
//                              (unregistered candidate, extension/limits
//                              rejection, variant-identity mismatch).
enum ggml_vk_resolution_v2 : uint8_t {
    GGML_VK_RESOLVE_EXACT = 0,
    GGML_VK_RESOLVE_CANDIDATE_UNAVAILABLE,
    GGML_VK_RESOLVE_RERUN_REQUIRED,
    GGML_VK_RESOLVE_INCOMPATIBLE,
    GGML_VK_RESOLVE_MISS,
    GGML_VK_RESOLVE_COUNT,
};

// Load the cache named by GGML_VK_DISPATCH_CACHE, if any. Safe to call more
// than once; the first call wins. Returns false when no usable cache was
// loaded, which is not an error -- it means native selection. NOT
// implemented (declaration only) -- see file header.
bool ggml_vk_replay_init();

// Resolve a dispatch key to a stored winner. sig/hw are needed to run the
// same extension/limits/can_execute/variant checks that decide EXACT vs
// CANDIDATE_UNAVAILABLE -- classification happens here, not after the
// caller has already committed to a binding (mirrors
// ggml_hip_replay_lookup's contract exactly). NOT implemented.
//
// Only GGML_VK_RESOLVE_EXACT sets *out_candidate/*out_variant; every other
// outcome leaves them untouched and the caller uses native selection.
ggml_vk_resolution_v2 ggml_vk_replay_lookup(
                            const ggml_vk_digest & dispatch_digest,
                            const ggml_vk_digest & signature_digest,
                            const ggml_vk_dispatch_signature_v1 & sig,
                            const ggml_vk_hardware_key_v1 & hw,
                            const ggml_vk_candidate_descriptor ** out_candidate,
                            ggml_vk_variant_params * out_variant);

// Per-outcome resolution counters, same always-tracked/relaxed-atomics
// design as ggml_hip_replay_resolution_count. NOT implemented.
size_t ggml_vk_replay_resolution_count(ggml_vk_resolution_v2 outcome);
const char * ggml_vk_replay_resolution_name(ggml_vk_resolution_v2 outcome);

#endif // GGML_USE_VULKAN && GGML_VULKAN_AUTOTUNE
