// bigcherry: the routing-transformation registry (HI27).
//
// A flat table of transformations, exactly as the candidate registry is a flat
// table of descriptors. Both halves of it -- the hand-written transforms and
// any the agent pipeline adds -- live in one array so the tuner iterates one
// thing and a record says which half an entry came from without consulting a
// second table.
//
// The whole file is inert unless GGML_HIP_ROUTING_TRANSFORM is defined. That is
// not caution about the feature's correctness: it is the same rule the rest of
// the layer follows (standards 12.2), so a production replay build carries no
// symbol it does not dispatch through.

#pragma once

#include "hip-autotune-types.h"

#if defined(GGML_USE_HIP) && defined(GGML_HIP_DISPATCH) && defined(GGML_HIP_ROUTING_TRANSFORM)

// Registry access. Indices are per-build and never persisted; `id` and `name`
// are the durable identities, exactly as for candidates (standards 2.1).
int ggml_hip_transform_count();
const ggml_hip_routing_transformation * ggml_hip_transform_at(int index);
const ggml_hip_routing_transformation * ggml_hip_transform_find(
    ggml_hip_transform_id id);
const ggml_hip_routing_transformation * ggml_hip_transform_find_by_name(
    const char * name);

// Whether `sig` is one this layer should even consider rewriting.
//
// Two exclusions, both hard. A fused pattern is a distinct semantic operation
// (standards 11.1) and rewriting its operands would change what is being
// computed, not just how. MUL_MAT_ID carries a routing tensor whose indices
// address the original operand layout, so any rewrite of that layout
// invalidates the ids without the ids ever saying so.
bool ggml_hip_transform_signature_is_eligible(
    const ggml_hip_dispatch_signature_v1 & sig);

// Run `candidate` over the *whole* decomposition `ctx` describes.
//
// This is the complete path of a transformed dispatch, and it is one function
// rather than two because the tuner and the dispatcher must not be able to
// disagree about what a transformed launch consists of. A batching transform
// costs what its full loop costs (standards 7.1); if the tuner timed one batch
// and the dispatcher ran forty, the recorded winner would describe a fortieth
// of the work it was chosen to do.
//
// `variant` overrides the descriptor's own, exactly as ggml_hip_dispatch_launch
// does, so a probe can drive a registry candidate at a variant it does not
// itself carry.
void ggml_hip_transform_launch(
    const ggml_hip_routing_transformation * transform,
    const ggml_hip_candidate_descriptor *   candidate,
    const ggml_hip_variant_params &         variant,
    ggml_hip_transform_ctx *                ctx,
    const ggml_hip_launch_context &         lc);

#endif // GGML_USE_HIP && GGML_HIP_DISPATCH && GGML_HIP_ROUTING_TRANSFORM
