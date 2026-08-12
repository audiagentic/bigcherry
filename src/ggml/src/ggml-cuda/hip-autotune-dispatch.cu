// bigcherry: measured dispatch — selection, resolution, launch (HI04).

#include "hip-autotune-dispatch.cuh"

// Standards 12.2: gated on GGML_USE_HIP *and* the feature flag. These files
// are picked up by the ggml-cuda/*.cu glob, so without the second condition
// they would compile into every HIP build, including ones that carry none of
// the machinery they depend on.
#if defined(GGML_USE_HIP) && defined(GGML_HIP_DISPATCH)

#include "hip-autotune-build-hash.h"
#include "hip-autotune-coverage.h"
#include "hip-autotune-signature.h"
#include "ggml-hip-autotune.h"

#include "mmq.cuh"
#include "mmvq.cuh"
#include "mmvf.cuh"
#include "mmf.cuh"

#include <mutex>
#include <stdlib.h>
#include <string.h>
#include <string>
#include <unordered_map>

#ifdef GGML_HIP_AUTOTUNE_RECORD
#include "hip-autotune-record.h"
#endif
#ifdef GGML_HIP_AUTOTUNE
#include "hip-autotune-tuner.cuh"
#endif
#ifdef GGML_HIP_DISPATCH_REPLAY
#include "hip-autotune-replay.h"
#endif

// The generated table. Included exactly once, here, after the family entry
// points it references have been declared.
#include "hip-autotune-registry.inc"

// ----------------------------------------------------------------- registry

size_t ggml_hip_registry_size() {
    return GGML_HIP_AUTOTUNE_CANDIDATE_COUNT;
}

const ggml_hip_candidate_descriptor * ggml_hip_registry_at(size_t index) {
    return index < ggml_hip_registry_size()
        ? &ggml_hip_candidate_registry[index] : nullptr;
}

const ggml_hip_candidate_descriptor * ggml_hip_registry_find(
        const char * stable_name) {
    // Linear, but only ever walked by tools and tests. The hot path resolves
    // through the dispatch-key cache and never looks a candidate up by name.
    for (size_t i = 0; i < ggml_hip_registry_size(); ++i) {
        if (strcmp(ggml_hip_candidate_registry[i].stable_name, stable_name) == 0) {
            return &ggml_hip_candidate_registry[i];
        }
    }
    return nullptr;
}

const ggml_hip_candidate_descriptor * ggml_hip_registry_native(
        ggml_hip_kernel_family family) {
    static const ggml_hip_candidate_descriptor * cache[GGML_HIP_FAMILY_COUNT] = {};
    static std::once_flag once;
    std::call_once(once, []() {
        for (size_t i = 0; i < ggml_hip_registry_size(); ++i) {
            const auto & candidate = ggml_hip_candidate_registry[i];
            if (candidate.source_class == GGML_HIP_SOURCE_NATIVE_WRAPPER) {
                cache[candidate.family] = &candidate;
            }
        }
    });
    return family < GGML_HIP_FAMILY_COUNT ? cache[family] : nullptr;
}

// --------------------------------------------------------------------- mode

static int ggml_hip_parse_mode() {
    const char * requested = getenv("GGML_HIP_DISPATCH_MODE");
    if (requested == nullptr) {
        return GGML_HIP_DISPATCH_MODE_NATIVE;
    }

    int mode = GGML_HIP_DISPATCH_MODE_NATIVE;
    // "native" is the documented name of the default and was not accepted, so
    // asking for it explicitly produced an "unknown mode" warning. Harmless in
    // effect -- the fallback is native either way -- but it teaches anyone
    // reading the log to distrust the mode they asked for, and it makes a
    // deliberate native baseline run look like a typo. Baselines matter here:
    // comparing against native is how RV08 and the mixed-topology segfault
    // were each attributed correctly.
    if (strcmp(requested, "native") == 0) {
        mode = GGML_HIP_DISPATCH_MODE_NATIVE;
    } else if (strcmp(requested, "record") == 0) {
        mode = GGML_HIP_DISPATCH_MODE_RECORD;
    } else if (strcmp(requested, "tune") == 0) {
        mode = GGML_HIP_DISPATCH_MODE_TUNE;
    } else if (strcmp(requested, "replay") == 0) {
        mode = GGML_HIP_DISPATCH_MODE_REPLAY;
    } else {
        GGML_LOG_WARN("%s: unknown GGML_HIP_DISPATCH_MODE=%s; using native\n",
                      __func__, requested);
        return GGML_HIP_DISPATCH_MODE_NATIVE;
    }

    // Record and tune are separate capabilities and must downgrade separately.
    //
    // Folding them together is the defect PACK_REVIEW A1 describes: the
    // documented inventory profile is DISPATCH_REPLAY=ON, AUTOTUNE=OFF,
    // MODE=record, so a single `#ifndef GGML_HIP_AUTOTUNE` guard silently turns
    // the one build whose entire job is recording into a build that records
    // nothing. Every downstream stage then starts from an empty inventory with
    // no error anywhere.
    //
    // Standards 9.1 still holds: a production replay build defines neither
    // symbol and downgrades both.
#ifndef GGML_HIP_AUTOTUNE_RECORD
    if (mode == GGML_HIP_DISPATCH_MODE_RECORD) {
        GGML_LOG_WARN("%s: this build cannot record (configure with "
                      "GGML_HIP_AUTOTUNE_RECORD=ON); using native\n", __func__);
        return GGML_HIP_DISPATCH_MODE_NATIVE;
    }
#endif
#ifndef GGML_HIP_AUTOTUNE
    if (mode == GGML_HIP_DISPATCH_MODE_TUNE) {
        GGML_LOG_WARN("%s: this build has no tuning engine "
                      "(built without GGML_HIP_AUTOTUNE); using native\n", __func__);
        return GGML_HIP_DISPATCH_MODE_NATIVE;
    }
#endif
    return mode;
}

int ggml_hip_dispatch_mode() {
    static const int mode = ggml_hip_parse_mode();
    return mode;
}

// -------------------------------------------------------------- native select

// Reproduces the branch order of upstream `ggml_cuda_mul_mat`. Any divergence
// here is a native-parity failure, which is why the order is written out
// explicitly rather than being factored into something cleverer.
ggml_hip_native_selection ggml_hip_native_select(
        ggml_backend_cuda_context & ctx, const ggml_tensor * src0,
        const ggml_tensor * src1, const ggml_tensor * ids,
        const ggml_tensor * dst) {
    ggml_hip_native_selection selection = {};
    selection.valid = false;

    const int cc        = ggml_cuda_info().devices[ctx.device].cc;
    const int warp_size = ggml_cuda_info().devices[ctx.device].warp_size;
    const int64_t ne11  = src1->ne[1];
    const int64_t ne12  = src1->ne[2];

    const bool bad_padding_clear =
        src0->buffer != nullptr
        && ggml_backend_buffer_get_usage(src0->buffer) == GGML_BACKEND_BUFFER_USAGE_COMPUTE
        && ggml_nbytes(src0) != ggml_backend_buffer_get_alloc_size(src0->buffer, src0)
        && src0->view_src != nullptr;

    ggml_hip_kernel_family family = GGML_HIP_FAMILY_BLAS;

    if (bad_padding_clear || src1->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        family = GGML_HIP_FAMILY_BLAS;
    } else if (ids != nullptr) {
        // MUL_MAT_ID has its own, shorter ladder upstream: MMVF, then MMQ, then
        // MMF, then MMVQ. Mirroring it separately is less elegant than one
        // shared ladder and considerably more likely to stay correct.
        if (ggml_cuda_should_use_mmvf(src0->type, cc, src0->ne, src0->nb, ne12)) {
            family = GGML_HIP_FAMILY_MMVF;
        } else if (ggml_cuda_should_use_mmq(src0->type, cc, ne12, src0->ne[2])) {
            family = GGML_HIP_FAMILY_MMQ;
        } else if (ggml_cuda_should_use_mmf(src0->type, cc, warp_size, src0->ne,
                                            src0->nb, ne12, /*mul_mat_id =*/ true)) {
            family = GGML_HIP_FAMILY_MMF;
        } else if (ggml_cuda_should_use_mmvq(src0->type, cc, ne12)) {
            family = GGML_HIP_FAMILY_MMVQ;
        } else {
            family = GGML_HIP_FAMILY_BLAS;
        }
    } else if (ggml_cuda_should_use_mmvf(src0->type, cc, src0->ne, src0->nb, ne11)) {
        family = GGML_HIP_FAMILY_MMVF;
    } else if (ggml_cuda_should_use_mmf(src0->type, cc, warp_size, src0->ne,
                                        src0->nb, ne11, /*mul_mat_id =*/ false)) {
        family = GGML_HIP_FAMILY_MMF;
    } else if (ggml_cuda_should_use_mmvq(src0->type, cc, ne11)) {
        family = GGML_HIP_FAMILY_MMVQ;
    } else if (ggml_cuda_should_use_mmq(src0->type, cc, ne11, /*n_experts =*/ 0)) {
        family = GGML_HIP_FAMILY_MMQ;
    } else {
        family = GGML_HIP_FAMILY_BLAS;
    }

    selection.candidate = ggml_hip_registry_native(family);
    // Zero variant params mean "ask the native policy", which is exactly what
    // a native wrapper does. The concrete J or block size it picks is decided
    // inside the launcher, from the actual shape.
    memset(&selection.variant, 0, sizeof(selection.variant));
    selection.valid = selection.candidate != nullptr;
    return selection;
}

// ------------------------------------------------------------ process cache

namespace {

struct DigestHash {
    size_t operator()(const ggml_hip_digest & digest) const {
        // The digest is already a uniform hash; folding its first bytes into
        // size_t is all the bucket index needs.
        size_t value = 0;
        for (size_t i = 0; i < sizeof(size_t) && i < GGML_HIP_DIGEST_BYTES; ++i) {
            value |= (size_t) digest.bytes[i] << (8 * i);
        }
        return value;
    }
};

struct DigestEqual {
    bool operator()(const ggml_hip_digest & a, const ggml_hip_digest & b) const {
        return ggml_hip_digest_equal(a, b);
    }
};

// Standards 15.2: the first encounter of a dispatch key performs the lookup;
// every later encounter is a dictionary hit with no hashing cost. `Binding` is
// what gets cached -- the resolved candidate, not the query.
struct Binding {
    const ggml_hip_candidate_descriptor * candidate;
    ggml_hip_variant_params               variant;
    bool                                  from_cache;
};

std::unordered_map<ggml_hip_digest, Binding, DigestHash, DigestEqual> g_bindings;
std::mutex g_bindings_mutex;

// Set while a dispatched candidate is executing.
//
// The family entry points are themselves collection sites, so an operation that
// already came through ggml_hip_dispatch_mul_mat would otherwise be caught a
// second time on its way into the family it was dispatched to -- resolving,
// launching, and recursing. The guard makes a family hook fire only for calls
// that reached the family by some *other* route, which is exactly the traffic
// the dense-selector hook cannot see.
//
// thread_local because the backend runs one host thread per device context;
// two devices dispatching concurrently must not observe each other's guard.
thread_local int g_in_dispatch = 0;

struct DispatchScope {
    DispatchScope()  { ++g_in_dispatch; }
    ~DispatchScope() { --g_in_dispatch; }
};

// HI22: GGML_HIP_FORCE_CANDIDATE env var — force a specific candidate for every
// dispatch. Used for manual testing of a single geometry without tuning.
struct ForcedCandidate {
    const char * stable_name = nullptr;
    const ggml_hip_candidate_descriptor * candidate = nullptr;

    static const ForcedCandidate & instance() {
        static ForcedCandidate inst = [] {
            ForcedCandidate fc;
            if (const char * name = getenv("GGML_HIP_FORCE_CANDIDATE")) {
                fc.stable_name = name;
                fc.candidate = ggml_hip_registry_find(name);
                if (!fc.candidate) {
                    GGML_LOG_WARN("bigcherry: GGML_HIP_FORCE_CANDIDATE=%s not "
                                  "found in registry — force-candidate disabled\n",
                                  name);
                } else {
                    GGML_LOG_INFO("bigcherry: GGML_HIP_FORCE_CANDIDATE=%s "
                                  "(family %d, source %d)\n",
                                  name, fc.candidate->family,
                                  fc.candidate->source_class);
                }
            }
            return fc;
        }();
        return inst;
    }
};

#ifdef GGML_HIP_AUTOTUNE
// Is this stream mid graph capture?
//
// Measurement is illegal there. The tuner records events, synchronises and
// reads elapsed time; under capture those are not executed but *recorded*, and
// the first one poisons the capture. The failure surfaces later and elsewhere
// as "operation failed due to a previous error during capture", pointing at
// whoever ends the capture rather than at the tuner.
//
// Failing closed on a query error is deliberate: not tuning costs one
// signature's measurement, while tuning inside a capture costs the graph.
bool ggml_hip_stream_is_capturing(cudaStream_t stream) {
    hipStreamCaptureStatus status = hipStreamCaptureStatusNone;
    if (hipStreamIsCapturing(stream, &status) != hipSuccess) {
        return true;
    }
    return status != hipStreamCaptureStatusNone;
}

// Warn once rather than per launch: a capture-heavy run would otherwise bury
// the log, and the fact that matters is "some signatures went unmeasured",
// which needs saying exactly once.
void ggml_hip_warn_tuning_skipped_under_capture() {
    static std::once_flag once;
    std::call_once(once, []() {
        GGML_LOG_WARN("%s: HIP graph capture is active; those launches keep the "
                      "native binding and are left unmeasured. Coverage will "
                      "report them as undispatched. Disable graphs "
                      "(GGML_CUDA_DISABLE_GRAPHS=1) for a complete tuning run.\n",
                      __func__);
    });
}
#endif // GGML_HIP_AUTOTUNE

} // namespace

// ---------------------------------------------------------------- resolution

ggml_hip_resolved_dispatch ggml_hip_dispatch_resolve(
        ggml_backend_cuda_context & ctx,
        const ggml_hip_dispatch_signature_v1 & sig,
        const ggml_hip_native_selection & native,
        const ggml_hip_launch_context & lc) {
    ggml_hip_resolved_dispatch resolved = {};
    resolved.candidate      = native.candidate;
    resolved.variant        = native.variant;
    resolved.prepared_state = nullptr;
    resolved.from_cache     = false;

    const int mode = ggml_hip_dispatch_mode();
    if (mode == GGML_HIP_DISPATCH_MODE_NATIVE || !native.valid) {
        return resolved;
    }

    // HI22: force-candidate bypass — use a specific candidate for manual testing.
    if (const auto & forced = ForcedCandidate::instance(); forced.candidate != nullptr) {
        const ggml_hip_hardware_key_v1 hw = ggml_hip_make_hardware_key(ctx.device);
        if (forced.candidate->can_execute(forced.candidate, sig, hw)) {
            resolved.candidate  = forced.candidate;
            resolved.variant    = forced.candidate->variant;
            resolved.from_cache = true;
            return resolved;
        }
        // Not eligible — fall through to normal resolution with a one-shot warning
        static std::once_flag once;
        std::call_once(once, [&forced, &sig]() {
            GGML_LOG_WARN("bigcherry: GGML_HIP_FORCE_CANDIDATE=%s is not eligible "
                          "for this signature (%s %s %s m=%lld n=%lld k=%lld) — "
                          "using normal resolution instead\n",
                          forced.stable_name,
                          sig.src0_type ? "t" : "?",
                          sig.src1_type ? "t" : "?",
                          sig.dst_type ? "t" : "?",
                          (long long)sig.ne1[1],
                          (long long)sig.ned[1],
                          (long long)sig.ne0[0]);
        });
    }

    const ggml_hip_hardware_key_v1 hw = ggml_hip_make_hardware_key(ctx.device);
    const ggml_hip_digest hardware_digest  = ggml_hip_hardware_digest(hw);
    const ggml_hip_digest signature_digest = ggml_hip_signature_digest(sig);
    const ggml_hip_digest dispatch_digest =
        ggml_hip_dispatch_digest(hardware_digest, signature_digest, "latency");

    {
        std::lock_guard<std::mutex> lock(g_bindings_mutex);
        const auto found = g_bindings.find(dispatch_digest);
        if (found != g_bindings.end()) {
            resolved.candidate  = found->second.candidate;
            resolved.variant    = found->second.variant;
            resolved.from_cache = found->second.from_cache;
#ifdef GGML_HIP_AUTOTUNE_RECORD
            // The warm path is where nearly every execution lands, so record
            // mode has to count here or `calls` never exceeds 1 and the
            // hot-signature ranking (standards 7.4) orders by nothing. It is
            // also the only place a second GPU sharing a dispatch key
            // (standards 10.2) is ever seen.
            if (mode == GGML_HIP_DISPATCH_MODE_RECORD) {
                ggml_hip_record_touch(signature_digest, hardware_digest,
                                      ctx.device);
            }
#endif
            return resolved;
        }
    }

    Binding binding = { native.candidate, native.variant, false };

#ifdef GGML_HIP_AUTOTUNE_RECORD
    if (mode == GGML_HIP_DISPATCH_MODE_RECORD) {
        // Record mode observes and changes nothing. The binding it installs is
        // the native one, so the run behaves identically to native mode while
        // the signature is captured.
        ggml_hip_record_observation(ctx, sig, hw, signature_digest,
                                    hardware_digest, native);
    }
#endif
#ifdef GGML_HIP_AUTOTUNE
    if (mode == GGML_HIP_DISPATCH_MODE_TUNE) {
        if (ggml_hip_stream_is_capturing(lc.stream)) {
            // Keep the native binding. Measuring here would poison the capture,
            // and a winner chosen from poisoned timings would be worse than no
            // winner at all.
            ggml_hip_warn_tuning_skipped_under_capture();
        } else {
            const ggml_hip_candidate_descriptor * winner =
                ggml_hip_tuner_resolve(ctx, sig, hw, dispatch_digest, native, lc);
            if (winner != nullptr) {
                binding.candidate  = winner;
                binding.variant    = winner->variant;
                binding.from_cache = true;
            }
        }
    }
#endif

#ifdef GGML_HIP_DISPATCH_REPLAY
    if (mode == GGML_HIP_DISPATCH_MODE_REPLAY) {
        const ggml_hip_candidate_descriptor * winner = nullptr;
        ggml_hip_variant_params variant = {};
        if (ggml_hip_replay_lookup(dispatch_digest, signature_digest,
                                   &winner, &variant)) {
            binding.candidate  = winner;
            binding.variant    = variant;
            binding.from_cache = true;
#ifdef GGML_HIP_REPLAY_DIAGNOSTICS
            ggml_hip_replay_record_hit(dispatch_digest, signature_digest, winner);
#endif
        } else {
            // Standards 9.2: a miss falls back to native and records the miss.
            // Production never attempts online measurement.
            ggml_hip_replay_record_miss(dispatch_digest, signature_digest, sig,
                                        hw, native.candidate);
        }
    }
#endif

    // A stored winner that cannot run on this hardware, or that has since been
    // blacklisted, must not be launched merely because it was once the winner.
    if (binding.candidate != nullptr
            && (!ggml_hip_candidate_supports_arch(*binding.candidate, hw)
                || !binding.candidate->can_execute(binding.candidate, sig, hw))) {
        binding.candidate  = native.candidate;
        binding.variant    = native.variant;
        binding.from_cache = false;
    }

    {
        std::lock_guard<std::mutex> lock(g_bindings_mutex);
        g_bindings.emplace(dispatch_digest, binding);
    }

    resolved.candidate  = binding.candidate;
    resolved.variant    = binding.variant;
    resolved.from_cache = binding.from_cache;
    return resolved;
}

void ggml_hip_dispatch_launch(const ggml_hip_resolved_dispatch & bound,
                              const ggml_hip_launch_context & lc) {
    GGML_ASSERT(bound.candidate != nullptr);
    GGML_ASSERT(bound.candidate->launch != nullptr);

    // The descriptor carries the variant for a fixed candidate, but a resolved
    // dispatch may override it (a tuner probe, a seeded winner). Launching
    // through a copy keeps the registry table immutable and shareable.
    ggml_hip_candidate_descriptor effective = *bound.candidate;
    effective.variant = bound.variant;
    effective.launch(&effective, lc);
}

// ------------------------------------------------------------- entry point

bool ggml_hip_dispatch_mul_mat(
        ggml_backend_cuda_context & ctx, const ggml_tensor * src0,
        const ggml_tensor * src1, const ggml_tensor * ids, ggml_tensor * dst,
        const ggml_cuda_mm_fusion_args_host * fusion) {
    if (ggml_hip_dispatch_mode() == GGML_HIP_DISPATCH_MODE_NATIVE) {
        return false;
    }

    // Upstream has a narrow path that transposes a vector operand and calls
    // MMVF with src0/src1 swapped against a synthesised dst. Its launch
    // context is not the one the signature describes, so rather than model the
    // swap, the dispatch layer declines and upstream's own code runs. It is a
    // rare shape and describing it wrongly would be worse than not tuning it.
    const int64_t ne01 = src0->ne[1];
    const int64_t ne11 = src1->ne[1];
    if (ids == nullptr && ne01 == 1 && ne11 > MMVF_MAX_BATCH_SIZE
            && dst->ne[2] == 1 && dst->ne[3] == 1
            && src0->type == GGML_TYPE_F32
            && ggml_is_contiguous(src0) && ggml_is_contiguous(src1)
            && ggml_is_contiguous(dst)) {
        return false;
    }

    const ggml_hip_native_selection native =
        ggml_hip_native_select(ctx, src0, src1, ids, dst);
    if (!native.valid) {
        return false;
    }

    // Standards 5.2: built from the tensors this device will actually execute.
    // Callers inside the tensor-split loop have already sliced them.
    const ggml_hip_dispatch_signature_v1 sig =
        ggml_hip_make_signature(src0, src1, ids, dst, fusion);

    // Built before resolving, not after: tune mode measures candidates from
    // inside the resolver and needs the real arguments to launch them with.
    ggml_hip_launch_context lc = {};
    lc.ctx        = &ctx;
    lc.src0       = src0;
    lc.src1       = src1;
    lc.ids        = ids;
    lc.dst        = dst;
    lc.has_fusion = fusion != nullptr;
    if (fusion != nullptr) {
        lc.fusion = *fusion;
    }
    lc.stream = ctx.stream();

    const ggml_hip_resolved_dispatch bound =
        ggml_hip_dispatch_resolve(ctx, sig, native, lc);

    // Both counters fire here, because this is the outermost point at which
    // this operation is visible. The family entry will see the same launch
    // again as a re-entrant call and deliberately skips it.
    //
    // Between the two sites every launch is counted exactly once:
    //
    //   dense, dispatched -> here (family entry sees it re-entrant, skips)
    //   dense, declined   -> family entry (this returned false, counted nothing)
    //   fused             -> family entry (outermost; never reaches this site)
    //   legacy split path -> family entry (outermost)
    //
    // Getting this wrong is not a small error: counting unconditionally at the
    // family entry inflated `executed` by exactly the dispatched count and made
    // full coverage read as 75%; counting only there, gated, dropped the dense
    // route entirely and made dispatched exceed executed.
    const ggml_hip_kernel_family chosen =
        (ggml_hip_kernel_family) bound.candidate->family;
    ggml_hip_coverage_count_executed(chosen);
    ggml_hip_coverage_count_dispatched(chosen);

    const DispatchScope scope;
    ggml_hip_dispatch_launch(bound, lc);
    return true;
}

bool ggml_hip_dispatch_is_reentrant() {
    return g_in_dispatch != 0;
}

void ggml_hip_dispatch_scope_enter() {
    ++g_in_dispatch;
}

void ggml_hip_dispatch_scope_leave() {
    --g_in_dispatch;
}

// ------------------------------------------------- family collection points

bool ggml_hip_dispatch_family(
        ggml_backend_cuda_context & ctx, const ggml_tensor * src0,
        const ggml_tensor * src1, const ggml_tensor * ids, ggml_tensor * dst,
        const ggml_cuda_mm_fusion_args_host * fusion,
        ggml_hip_kernel_family family) {
    if (ggml_hip_dispatch_mode() == GGML_HIP_DISPATCH_MODE_NATIVE) {
        return false;
    }
    // Already handled on the way in; this is the dispatched launch arriving at
    // its own family. Letting it through would recurse.
    if (g_in_dispatch != 0) {
        return false;
    }

    const ggml_hip_candidate_descriptor * native_candidate =
        ggml_hip_registry_native(family);
    if (native_candidate == nullptr) {
        return false;
    }

    // Standards 11.1: the family is already decided. This is the whole
    // difference from the dense entry -- native selection is *not* consulted,
    // so a fused MMVQ is never re-selected into MMQ. Only geometry is tuned.
    ggml_hip_native_selection native = {};
    native.candidate = native_candidate;
    memset(&native.variant, 0, sizeof(native.variant));
    native.valid = true;

    const ggml_hip_dispatch_signature_v1 sig =
        ggml_hip_make_signature(src0, src1, ids, dst, fusion);

    ggml_hip_launch_context lc = {};
    lc.ctx        = &ctx;
    lc.src0       = src0;
    lc.src1       = src1;
    lc.ids        = ids;
    lc.dst        = dst;
    lc.has_fusion = fusion != nullptr;
    if (fusion != nullptr) {
        lc.fusion = *fusion;
    }
    lc.stream = ctx.stream();

    const ggml_hip_resolved_dispatch bound =
        ggml_hip_dispatch_resolve(ctx, sig, native, lc);

    // A stored winner from another family would be a graph-level decision
    // arriving through a matmul-level door. Refuse it and run native.
    if (bound.candidate == nullptr || bound.candidate->family != family) {
        return false;
    }

    ggml_hip_coverage_count_dispatched(family);

    const DispatchScope scope;
    ggml_hip_dispatch_launch(bound, lc);
    return true;
}

// ------------------------------------------------------- family entry points
//
// Each launcher forwards to the corresponding forced-variant entry point added
// by HI06-HI09. A zero variant value means "native policy chooses", so the
// native wrapper and a forced variant share one code path -- which is what
// makes forced-native identical to native by construction.


// Whether a family is *defined* for this operation at all.
//
// Before cross-family tuning these predicates were never exercised: native
// selection only ever routed an operation to a family that accepted it, so
// `can_execute` was only ever asked about combinations that already worked.
// Comparing families asks the question properly for the first time, and each
// missing clause below showed up as an abort deep inside a family entry point
// -- "unsupported type: q8_0", then `GGML_ASSERT(ncols_dst <= 8)`.
//
// Standards 11.3 is explicit that this must check *correctness and resource
// limits only*, never upstream's performance guesses. So this deliberately
// does not call `ggml_cuda_should_use_*` -- those encode heuristics about what
// is fast, and using them would make the tuner unable to contradict the very
// choices it exists to test.
//
// Everything here mirrors an assertion or a switch default that the family
// itself would hit.
static inline bool ggml_hip_family_can_serve(
        ggml_hip_kernel_family family,
        const ggml_hip_dispatch_signature_v1 & sig) {
    const ggml_type src0 = (ggml_type) sig.src0_type;
    const int64_t   k    = sig.ne0[0];      // shared dimension
    const int64_t   width = sig.ned[1];     // ncols_dst

    const bool is_float = src0 == GGML_TYPE_F32
                       || src0 == GGML_TYPE_F16
                       || src0 == GGML_TYPE_BF16;

    // All four matmul families require an F32 activation and say so with a
    // hard GGML_ASSERT on entry (mmvf.cu:673, mmf.cu:35, mmvq.cu:1320,
    // mmq.cu:109). Upstream never trips them because it routes a non-F32 src1
    // to the dense path before reaching any of them; the dispatch layer picks
    // candidates itself, so it has to encode the same precondition. BLAS is
    // exempt -- it is that dense path.
    if (family != GGML_HIP_FAMILY_BLAS && sig.src1_type != GGML_TYPE_F32) {
        return false;
    }

    switch (family) {
        case GGML_HIP_FAMILY_MMQ:
            // Quantised only; the vec-dot machinery has no float path. But
            // quantised is not sufficient: MMQ implements a strict subset, and
            // `ggml_is_quantized` alone admitted iq1_m, tq1_0 and tq2_0, whose
            // launch reaches the abort in mmq_get_q8_1_ds_layout's default
            // branch (mmq.cuh:98). The native wrapper made that reachable
            // because it is accepted before every other check in
            // ggml_hip_mmq_can_execute -- upstream avoids it by routing those
            // types elsewhere, which is a decision our family predicate has to
            // reproduce rather than assume.
            //
            // Asked of upstream's config tables, not of a restated list, so a
            // type upstream adds or drops needs no change here.
            return ggml_is_quantized(src0)
                && ggml_cuda_mmq_type_is_supported(src0, /*cc =*/ 0,
                                                   /*shared_mem_limit =*/ 0);

        case GGML_HIP_FAMILY_MMVQ:
            // mmvq.cu asserts ncols_dst <= MMVQ_MAX_BATCH_SIZE.
            return ggml_is_quantized(src0) && width <= MMVQ_MAX_BATCH_SIZE;

        case GGML_HIP_FAMILY_MMVF:
            // mmvf.cu asserts ncols % 2 == 0 and bounds the batch.
            return is_float && (k % 2) == 0 && width <= MMVF_MAX_BATCH_SIZE;

        case GGML_HIP_FAMILY_MMF:
            // mmf.cuh switches cols_per_block over 1..16 and asserts
            // ncols_x % 2 == 0.
            return is_float && (k % 2) == 0 && width >= 1 && width <= 16;

        case GGML_HIP_FAMILY_BLAS:
            // hipBLAS takes anything -- the dense path dequantises first --
            // but upstream has no BLAS route for MUL_MAT_ID.
            return (sig.flags & GGML_HIP_SIG_HAS_IDS) == 0;

        default:
            return false;
    }
}

bool ggml_hip_mmq_can_execute(const ggml_hip_candidate_descriptor * self,
                              const ggml_hip_dispatch_signature_v1 & sig,
                              const ggml_hip_hardware_key_v1 & hw) {
    if (!ggml_hip_family_can_serve(GGML_HIP_FAMILY_MMQ, sig)) {
        return false;
    }
    if (self->source_class == GGML_HIP_SOURCE_NATIVE_WRAPPER) {
        return true;
    }

    // MMQ's config table is per type, so every other field of this candidate's
    // identity -- J, the tile sizes, the SRAM type -- describes a row that only
    // exists for the type it was named for. Forcing that J onto a different
    // type asks for an instantiation the table never defined and reaches the
    // device-side NO_DEVICE_CODE guard in mmq.cuh. Observed: an
    // `mmq:q8_0:j16:...` candidate winning an iq1_s signature.
    if (self->variant.src0_type != sig.src0_type) {
        return false;
    }

    // `fallback` is not a free choice -- mul_mat_q_case derives it from row
    // divisibility -- so a candidate whose fallback disagrees with this shape
    // will be launched with the *other* value. That matters because the config
    // table is sparse in both dimensions: (type, J, fb1) existing says nothing
    // about (type, J, fb0). Launching the mismatch reaches the device-side
    // `NO_DEVICE_CODE` guard in mmq.cuh and aborts.
    //
    // So reject the disagreement here rather than discovering it at launch
    // (standards 12.4). ne0[1] is ne01, the same quantity mul_mat_q_case uses.
    const bool shape_fallback = (sig.ne0[1] % 128) != 0;
    if ((self->variant.fallback != 0) != shape_fallback) {
        return false;
    }

    // EX02 quarantine, 2026-08-11 (docs/planning/active/external-fixes/EX02.md,
    // docs/reference/FINDINGS.md): mmq:q6_k:j112:fb0:t256:o2:i128:sram-q6_k:
    // k256:sk0:v1 crashes with an illegal memory access on gfx1100,
    // reproduced deterministically three times across two independent
    // instrumentation approaches. Only reachable via bigcherry's own
    // exhaustive sweep -- native's own J-selection always picks the
    // smallest J that covers the batch in one tile, so it never forces
    // J=112 onto MTP's narrow draft-decode batches the way a full sweep
    // does. Not a shared-memory-limit or workspace-sizing gap (both
    // checked against real numbers/source and ruled out; see EX02) --
    // root cause is still open, needs rocgdb, not more source reading.
    //
    // Scoped to exactly this stable identity, on gfx1100 only: this
    // candidate's config row is shared with other architectures whose own
    // tables define the same (type, J, I, sram_layout) -- excluding it
    // everywhere would quarantine hardware this was never proven to crash
    // on. Remove this block once EX02 lands a real fix; do not widen it.
    if (self->variant.src0_type == GGML_TYPE_Q6_K &&
            self->variant.primary == 112 &&
            self->variant.fallback == 0 &&
            hw.architecture_code == GGML_HIP_ARCH_GFX1100) {
        return false;
    }

    return ggml_cuda_mmq_variant_is_eligible(
        (ggml_type) sig.src0_type, self->variant.primary,
        shape_fallback,
        /*cc =*/ 0, hw.shared_memory_per_block, sig.ned[1]);
}

void ggml_hip_mmq_launch(const ggml_hip_candidate_descriptor * self,
                         const ggml_hip_launch_context & lc) {
    ggml_cuda_mul_mat_q_variant(*lc.ctx, lc.src0, lc.src1, lc.ids, lc.dst,
                                self->variant.primary, self->variant.fallback);
}

size_t ggml_hip_mmq_workspace(const ggml_hip_candidate_descriptor * self,
                              const ggml_hip_dispatch_signature_v1 & sig) {
    GGML_UNUSED(self);
    // HI54: this declaration was measured against the pool's own real
    // high-water mark (HI52 part 1) and found short in 269 of 310 cases,
    // median 1.11x, worst 554x. It was missing two things `mmq.cu`'s actual
    // ggml_cuda_mul_mat_q allocates (lines 158-159 for dense, 209-228 for
    // MUL_MAT_ID): the J_max activation term, and -- on the MoE path only --
    // three id-routing buffers. Rederived directly from that source rather
    // than patched by ratio, so the two stay in sync structurally, not just
    // numerically for today's shapes.
    //
    // A third term (the stream-k fixup buffer) was investigated and NOT added
    // -- see the comment above this function for why it is unreachable dead
    // code on AMD hardware, and HI54's notes for what still explains the
    // residual gap this fix does not close.
    const int64_t k_padded = GGML_PAD(sig.ne1[0], MATRIX_ROW_PADDING);
    const bool fallback = (sig.ne0[1] % 128) != 0;
    const ggml_type type = (ggml_type) sig.src0_type;

    // `ggml_cuda_mmq_get_J_max` needs the compute-capability code the config
    // table is keyed on. No `hw` reaches this function (see
    // ggml_hip_workspace_fn's signature), so it is read the same way
    // ggml_hip_mmf_can_execute already does a few lines below: from the
    // current device, not passed in.
    const int cc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;
    // Same `ne11` (src1->ne[1]) feeds ggml_cuda_mmq_get_J_max on both the
    // dense and MoE call sites in mmq.cu -- only the activation row-count
    // term below differs between the two paths.
    const size_t j_max_bytes =
        (size_t) ggml_cuda_mmq_get_J_max(type, fallback, cc, sig.ne1[1])
        * sizeof(block_q8_1_mmq);

    if ((sig.flags & GGML_HIP_SIG_HAS_IDS) != 0) {
        // MUL_MAT_ID (mmq.cu:227-228): row count is ne12*n_expert_used, not
        // ne11 -- the dense-path row count is unrelated on this path.
        const size_t activation_bytes =
            (size_t) (sig.ne1[2] * sig.n_expert_used) * (size_t) k_padded
            * sizeof(block_q8_1_mmq) / QK8_1_MMQ
            + j_max_bytes;

        // mmq.cu:209-211: ids_src1, ids_dst (int32, ne12*n_expert_used each)
        // and expert_bounds (int32, n_expert+1). Absent from the declaration
        // entirely before this fix -- the MoE path had zero workspace
        // accounting of its own, on top of missing the J_max term every path
        // was missing.
        const size_t ne_get_rows = (size_t) (sig.ne1[2] * sig.n_expert_used);
        const size_t id_bytes =
            2 * ne_get_rows * sizeof(int32_t)
            + (size_t) (sig.n_expert + 1) * sizeof(int32_t);
        return activation_bytes + id_bytes;
    }

    // Dense (mmq.cu:158-159): row count is ne13*ne12*ne11.
    const size_t activation_bytes =
        (size_t) (sig.ne1[3] * sig.ne1[2] * sig.ne1[1]) * (size_t) k_padded
        * sizeof(block_q8_1_mmq) / QK8_1_MMQ
        + j_max_bytes;
    return activation_bytes;
}

bool ggml_hip_mmvf_can_execute(const ggml_hip_candidate_descriptor * self,
                               const ggml_hip_dispatch_signature_v1 & sig,
                               const ggml_hip_hardware_key_v1 & hw) {
    if (!ggml_hip_family_can_serve(GGML_HIP_FAMILY_MMVF, sig)) {
        return false;
    }
    if (self->source_class == GGML_HIP_SOURCE_NATIVE_WRAPPER) {
        return true;
    }
    // MMF/MMVF do not crash on a type mismatch the way MMQ and MMVQ do --
    // upstream instantiates every type, so something always runs. That is
    // precisely why it needs stating: the winner would be recorded under a
    // stable name naming a type it never ran on, and nothing would signal it.
    if (self->variant.src0_type != sig.src0_type) {
        return false;
    }

    // Standards 3.1: an F16 accumulator against a request that did not ask for
    // reduced precision is a different operation, not a faster variant of this
    // one. Rejecting it here keeps it out of the measurement set entirely.
    if (self->variant.acc_f16 && sig.src0_type != GGML_TYPE_F16) {
        return false;
    }
    if (self->variant.width != 0 && self->variant.width != sig.ned[1]) {
        return false;
    }
    return ggml_cuda_mmvf_variant_is_eligible(
        (ggml_type) sig.src0_type, self->variant.primary,
        self->variant.acc_f16 != 0, hw.wave_size, hw.shared_memory_per_block,
        sig.ne0[0], sig.ned[1], sig.fusion != GGML_HIP_FUSION_NONE);
}

void ggml_hip_mmvf_launch(const ggml_hip_candidate_descriptor * self,
                          const ggml_hip_launch_context & lc) {
    ggml_cuda_mul_mat_vec_f_variant(
        *lc.ctx, lc.src0, lc.src1, lc.ids, lc.dst,
        lc.has_fusion ? &lc.fusion : nullptr,
        self->variant.primary, self->variant.acc_f16);
}

size_t ggml_hip_mmvf_workspace(const ggml_hip_candidate_descriptor * self,
                               const ggml_hip_dispatch_signature_v1 & sig) {
    GGML_UNUSED(self);
    GGML_UNUSED(sig);
    return 0; // MMVF reads the activations in place.
}

bool ggml_hip_mmf_can_execute(const ggml_hip_candidate_descriptor * self,
                              const ggml_hip_dispatch_signature_v1 & sig,
                              const ggml_hip_hardware_key_v1 & hw) {
    if (!ggml_hip_family_can_serve(GGML_HIP_FAMILY_MMF, sig)) {
        return false;
    }

    // Upstream's own capability gate, and it has to run before the native
    // short-circuit below because the native wrapper is as capable of naming an
    // uncompiled kernel as any forced variant is.
    //
    // ggml_cuda_mmf_variant_is_eligible does not cover this: it discards the
    // type outright (GGML_UNUSED) and checks only nwarps bounds and shared
    // memory. The gap is not subtle -- MMF's F32 path requires Ampere MMA or
    // AMD MFMA, and RDNA3 has neither (it has WMMA), so *every* mmf:f32
    // candidate is unrunnable on gfx1100 while looking perfectly eligible.
    // Launching one does not abort cleanly: the kernel body compiled to a
    // NO_DEVICE_CODE stub, so it faults as HSA_STATUS_ERROR_EXCEPTION, takes
    // the queue down, and leaves the device unusable. See review RV03.
    //
    // Called whole rather than split into a capability half. The alignment and
    // contiguity checks inside it are crash-prevention, not policy, and the
    // ncols bounds track which instances upstream actually compiles. The one
    // thing it costs is exploring ncols 9..16 on RDNA3.0, where upstream caps
    // at 8; those candidates now report as ineligible, which is a category
    // HI12's coverage report already distinguishes rather than hides.
    {
        const ggml_type type  = (ggml_type) sig.src0_type;
        const size_t    ts    = ggml_type_size(type);
        const int64_t   blck  = ggml_blck_size(type);

        // sig.nb* are normalised to elements (see ggml_hip_fill_strides);
        // should_use_mmf wants the original byte strides back.
        int64_t ne[GGML_MAX_DIMS];
        size_t  nb[GGML_MAX_DIMS];
        for (int i = 0; i < GGML_MAX_DIMS; ++i) {
            ne[i] = sig.ne0[i];
            nb[i] = (size_t) (sig.nb0[i] * (int64_t) ts / (blck == 0 ? 1 : blck));
        }

        const int id = ggml_cuda_get_device();
        const int cc = ggml_cuda_info().devices[id].cc;

        if (!ggml_cuda_should_use_mmf(type, cc, (int) hw.wave_size, ne, nb,
                                      (int) sig.ned[1],
                                      (sig.flags & GGML_HIP_SIG_HAS_IDS) != 0)) {
            return false;
        }
    }

    if (self->source_class == GGML_HIP_SOURCE_NATIVE_WRAPPER) {
        return true;
    }
    // As for MMVF: no crash, but a winner named for the wrong type is a false
    // record, and a silent one.
    if (self->variant.src0_type != sig.src0_type) {
        return false;
    }
    if (self->variant.width != 0 && self->variant.width != sig.ned[1]) {
        return false;
    }
    return ggml_cuda_mmf_variant_is_eligible(
        (ggml_type) sig.src0_type, self->variant.primary, /*cc =*/ 0,
        hw.wave_size, hw.shared_memory_per_block, sig.ned[1], sig.ne0[1]);
}

void ggml_hip_mmf_launch(const ggml_hip_candidate_descriptor * self,
                         const ggml_hip_launch_context & lc) {
    ggml_cuda_mul_mat_f_variant(*lc.ctx, lc.src0, lc.src1, lc.ids, lc.dst,
                                self->variant.primary);
}

size_t ggml_hip_mmf_workspace(const ggml_hip_candidate_descriptor * self,
                              const ggml_hip_dispatch_signature_v1 & sig) {
    GGML_UNUSED(self);
    GGML_UNUSED(sig);
    return 0;
}

bool ggml_hip_mmvq_can_execute(const ggml_hip_candidate_descriptor * self,
                               const ggml_hip_dispatch_signature_v1 & sig,
                               const ggml_hip_hardware_key_v1 & hw) {
    if (!ggml_hip_family_can_serve(GGML_HIP_FAMILY_MMVQ, sig)) {
        return false;
    }
    GGML_UNUSED(hw);
    if (self->source_class == GGML_HIP_SOURCE_NATIVE_WRAPPER) {
        return true;
    }

    // Multi-token MUL_MAT_ID goes to upstream's dedicated MoE kernel, which has
    // no geometry dimension at all. There is nothing to force, and letting a
    // forced candidate take that path would attribute the MoE kernel's timing
    // to a geometry it never used. mul_mat_vec_q_switch_ncols_dst aborts if one
    // reaches it, so this predicate is what keeps that abort unreachable.
    if ((sig.flags & GGML_HIP_SIG_HAS_IDS) != 0 && sig.ned[1] > 1) {
        return false;
    }

    // Instances are generated per (type, geometry) and the launch chain
    // dispatches on the runtime tensor type, so a candidate named for one type
    // cannot serve another: `mul_mat_vec_q_switch_type` would look up the
    // signature's type with this candidate's geometry, find nothing compiled,
    // and abort. The variant set decides which types exist at all (an inventory
    // built from a Q8_0 workload generates only q8_0 instances), so this is the
    // ordinary case, not an exotic one.
    //
    // It is also what keeps the recorded winner honest: without it a q8_0
    // candidate measured on a q4_0 signature would be written down under the
    // q8_0 name, which is the same misattribution the MMQ `fallback` check
    // above exists to prevent.
    if (self->variant.src0_type != sig.src0_type) {
        return false;
    }

    // Generated MMVQ instances are compiled for one exact geometry, so the
    // width has to match rather than merely fit -- an instance built for
    // ncols_dst=4 cannot serve a 2-column launch.
    return self->variant.width == sig.ned[1];
}

void ggml_hip_mmvq_launch(const ggml_hip_candidate_descriptor * self,
                          const ggml_hip_launch_context & lc) {
    ggml_cuda_mul_mat_vec_q_variant(
        *lc.ctx, lc.src0, lc.src1, lc.ids, lc.dst,
        lc.has_fusion ? &lc.fusion : nullptr,
        self->variant.primary, self->variant.secondary,
        self->variant.small_k != 0);
}

size_t ggml_hip_mmvq_workspace(const ggml_hip_candidate_descriptor * self,
                               const ggml_hip_dispatch_signature_v1 & sig) {
    GGML_UNUSED(self);
    const int64_t k_padded = GGML_PAD(sig.ne1[0], MATRIX_ROW_PADDING);
    return (size_t) (sig.ne1[3] * sig.ne1[2] * sig.ne1[1] * k_padded)
         * sizeof(block_q8_1) / QK8_1;
}

bool ggml_hip_blas_can_execute(const ggml_hip_candidate_descriptor * self,
                               const ggml_hip_dispatch_signature_v1 & sig,
                               const ggml_hip_hardware_key_v1 & hw) {
    if (!ggml_hip_family_can_serve(GGML_HIP_FAMILY_BLAS, sig)) {
        return false;
    }
    GGML_UNUSED(self);
    GGML_UNUSED(hw);
    return true;   // the capability predicate above is the whole check
}

void ggml_hip_blas_launch(const ggml_hip_candidate_descriptor * self,
                          const ggml_hip_launch_context & lc) {
    GGML_UNUSED(self);
    ggml_cuda_mul_mat_cublas_dispatch(*lc.ctx, lc.src0, lc.src1, lc.dst);
}

size_t ggml_hip_blas_workspace(const ggml_hip_candidate_descriptor * self,
                               const ggml_hip_dispatch_signature_v1 & sig) {
    GGML_UNUSED(self);
    // Dequantised src0 plus converted src1, as the cuBLAS path allocates.
    return (size_t) (sig.ne0[0] * sig.ne0[1] + sig.ne1[0] * sig.ne1[1])
         * sizeof(half);
}

// --------------------------------------------------------------- public API

bool ggml_hip_autotune_available(void) {
    return true;
}

bool ggml_hip_autotune_can_tune(void) {
#ifdef GGML_HIP_AUTOTUNE
    return true;
#else
    return false;
#endif
}

enum ggml_hip_dispatch_mode ggml_hip_dispatch_get_mode(void) {
    return (enum ggml_hip_dispatch_mode) ggml_hip_dispatch_mode();
}

const char * ggml_hip_autotune_manifest_hash(void) {
    return GGML_HIP_AUTOTUNE_MANIFEST_HASH_STR;
}

const char * ggml_hip_autotune_source_revision(void) {
    return GGML_HIP_AUTOTUNE_SOURCE_REVISION_STR;
}

size_t ggml_hip_autotune_candidate_count(void) {
    return ggml_hip_registry_size();
}

const char * ggml_hip_autotune_candidate_name(size_t index) {
    const ggml_hip_candidate_descriptor * candidate = ggml_hip_registry_at(index);
    return candidate != nullptr ? candidate->stable_name : nullptr;
}

void ggml_hip_autotune_flush(void) {
    // Each subsystem flushes only what it owns, and only if it is compiled in.
    // A replay build reaches nothing here except its bounded miss log.
#ifdef GGML_HIP_AUTOTUNE_RECORD
    ggml_hip_record_flush();
#endif
#ifdef GGML_HIP_AUTOTUNE
    ggml_hip_tuner_flush();
#endif
#ifdef GGML_HIP_DISPATCH_REPLAY
    ggml_hip_replay_flush_misses();
#ifdef GGML_HIP_REPLAY_DIAGNOSTICS
    ggml_hip_replay_flush_hits();
#endif
#endif
    ggml_hip_coverage_report();
}

void ggml_hip_autotune_write_report(const char * path) {
#ifdef GGML_HIP_AUTOTUNE_RECORD
    ggml_hip_record_write_report(path);
#else
    GGML_UNUSED(path);
#endif
}

#endif // GGML_USE_HIP && GGML_HIP_DISPATCH
