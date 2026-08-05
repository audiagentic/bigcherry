// bigcherry: canonical signature and hardware key construction (HI05).
//
// A signature is the canonical, device-local description of one operation. Two
// executions that a tuned winner may legitimately be shared between must
// produce the same signature; two that must not, must not. Everything else
// about this file follows from that one requirement:
//
//   * Dimensions come from the *device-local* slice, after tensor-split
//     slicing, so two GPUs given unequal shares of a node get different
//     signatures (standards 5.2).
//   * Diagnostic identity -- model name, layer index, pointers, device
//     ordinal, wall clock -- is excluded from the hash. It is observation
//     metadata and travels separately (standards 5.1, 15.1).
//   * Serialisation is explicit canonical JSON with sorted keys, never a raw
//     struct. A padded struct's contents depend on the compiler; a persistent
//     key must not (standards 5.4).

#pragma once

#include "hip-autotune-types.h"

// Standards 12.2: gated on GGML_USE_HIP *and* the feature flag. These files
// are picked up by the ggml-cuda/*.cu glob, so without the second condition
// they would compile into every HIP build, including ones that carry none of
// the machinery they depend on.
#if defined(GGML_USE_HIP) && defined(GGML_HIP_DISPATCH)

#include <string>

// Personalisation prefixes (standards 5.4). Distinct prefixes are what keep a
// signature digest and a hardware digest over coincidentally identical bytes
// from colliding into one key.
#define GGML_HIP_PERSON_SIGNATURE "llama-hip-tune"
#define GGML_HIP_PERSON_HARDWARE  "llama-hardware"
#define GGML_HIP_PERSON_DISPATCH  "llama-dispatch"

// Observation metadata. Recorded alongside a signature, never hashed into it.
// Keeping it in a separate struct is the mechanism that enforces standards
// 5.1: there is no field here that *could* reach the digest.
struct ggml_hip_diagnostics {
    const char * model_name;  // may be nullptr
    const char * node_name;
    int          layer_index; // -1 when unknown
    int          device_ordinal;
};

// Build a signature from the device-local tensors of one operation.
//
// `src0`/`src1`/`dst` must already be the local slices this device will
// actually execute -- not the global node shape. Callers inside the
// tensor-split loop pass their slice views; callers outside it pass the tensors
// unchanged, which for a single device is the same thing.
ggml_hip_dispatch_signature_v1 ggml_hip_make_signature(
    const ggml_tensor * src0,
    const ggml_tensor * src1,
    const ggml_tensor * ids,
    const ggml_tensor * dst,
    const ggml_cuda_mm_fusion_args_host * fusion);

// Read the executing device's hardware key. `device` is a CUDA/HIP device
// index; the resulting key contains no ordinal, so two identical GPUs in one
// box produce identical keys and can share a winner (standards 10.2).
ggml_hip_hardware_key_v1 ggml_hip_make_hardware_key(int device);

// Map a HIP `cc` value to the closed architecture-code enumeration used by the
// architecture mask. Returns GGML_HIP_ARCH_UNKNOWN for anything unrecognised,
// which excludes every candidate and therefore falls back to native.
int ggml_hip_architecture_code_from_cc(int cc);

// Canonical JSON. Deterministic, sorted keys, no whitespace. Also stored in the
// database so a stored key can be explained without re-deriving it.
std::string ggml_hip_signature_json(const ggml_hip_dispatch_signature_v1 & sig,
                                    bool include_refinements);
std::string ggml_hip_hardware_json(const ggml_hip_hardware_key_v1 & hw);

// Digests over the canonical JSON, with the prefixes above.
ggml_hip_digest ggml_hip_signature_digest(
    const ggml_hip_dispatch_signature_v1 & sig);

// The base digest is the same signature with refinements stripped. Lookup goes
// refined -> base -> native, so a refinement that has not been measured costs
// nothing (standards 5.5).
ggml_hip_digest ggml_hip_signature_base_digest(
    const ggml_hip_dispatch_signature_v1 & sig);

ggml_hip_digest ggml_hip_hardware_digest(const ggml_hip_hardware_key_v1 & hw);

// The full dispatch key: software namespace + hardware + signature + objective
// (standards 1). The software namespace is the source revision and manifest
// hash, so a changed upstream selector or candidate set cannot silently reuse
// old winners (standards 13.1).
ggml_hip_digest ggml_hip_dispatch_digest(const ggml_hip_digest & hardware,
                                         const ggml_hip_digest & signature,
                                         const char * objective);

std::string ggml_hip_digest_hex(const ggml_hip_digest & digest);

bool ggml_hip_digest_equal(const ggml_hip_digest & a, const ggml_hip_digest & b);

#endif // GGML_USE_HIP && GGML_HIP_DISPATCH
