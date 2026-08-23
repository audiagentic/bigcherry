// bigcherry: the tuning engine (HI12).

#include "hip-autotune-tuner.cuh"

#if defined(GGML_USE_HIP) && defined(GGML_HIP_AUTOTUNE)

#include "hip-autotune-build-hash.h"
#include "hip-autotune-canary.h"
#include "hip-autotune-dispatch.cuh"
#include "hip-autotune-io.h"
#include "hip-autotune-journal.h"
#include "hip-autotune-signature.h"
#include "hip-autotune-smi.h"
#include "hip-autotune-transform.cuh"

#include <algorithm>
#include <atomic>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <ctime>
#include <cerrno>
#include <limits>
#include <mutex>
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

// E6: compiler identity, so a measurements file that outlives a ROCm upgrade
// can say which compiler produced it (RV01-RV05 were invalidated by exactly
// this kind of unrecorded build-side difference).
#if defined(__clang_version__)
#  define GGML_HIP_COMPILER_STR "clang " __clang_version__
#elif defined(__VERSION__)
#  define GGML_HIP_COMPILER_STR __VERSION__
#else
#  define GGML_HIP_COMPILER_STR "unknown"
#endif

#define GGML_HIP_STRINGIFY_(x) #x
#define GGML_HIP_STRINGIFY(x) GGML_HIP_STRINGIFY_(x)

#if defined(HIP_VERSION_MAJOR) && defined(HIP_VERSION_MINOR)
#  define GGML_HIP_VERSION_STR \
       GGML_HIP_STRINGIFY(HIP_VERSION_MAJOR) "." GGML_HIP_STRINGIFY(HIP_VERSION_MINOR)
#else
#  define GGML_HIP_VERSION_STR "unknown"
#endif

namespace {

struct Measurement {
    const ggml_hip_candidate_descriptor * candidate = nullptr;
    ggml_hip_reject_reason reason      = GGML_HIP_REJECT_NONE;

#ifdef GGML_HIP_ROUTING_TRANSFORM
    // HI30: route identity is (candidate, variant, transform_id), not
    // candidate alone -- a transformed measurement runs a different amount
    // of work than the same candidate's direct route, and must never be
    // confused with it in the schedule, the JSONL, or winner selection.
    // nullptr means "direct route, no transform" everywhere below.
    const ggml_hip_routing_transformation * transform = nullptr;
#endif
    bool     measured                  = false;
    double   median_us                 = 0.0;
    double   mad_us                    = 0.0;   // standards: GPU times are
    double   p95_us                    = 0.0;   // right-skewed; MAD survives an
    double   host_median_us            = 0.0;   // outlier where stddev does not
    double   nmse                      = 0.0;
    double   max_abs_error             = 0.0;
    size_t   workspace_bytes           = 0;
    size_t   pool_peak_bytes           = 0;
    int      samples                   = 0;

    // E3: max(gpu_median, host_median - host_sync_overhead) -- whichever
    // resource is actually binding. Used for ranking; median_us stays the
    // recorded and reported figure.
    double   effective_us              = 0.0;

    // E1/E2: round-aligned final-stage samples. NaN marks a round where this
    // candidate failed to launch, so index r always means "round r" even
    // after a failure -- required for the paired sign test to compare like
    // rounds, and persisted verbatim as measurement.samples_json.
    std::vector<double> final_gpu_us;
    std::vector<double> final_host_us;

    // E1: one-sided paired sign-test result against native, over final_gpu_us.
    double   sign_p                    = 1.0;
    int      sign_wins                 = 0;
    int      sign_rounds               = 0;

    // HI24 step 4 / HI50: marks the same-kernel double-native replicate,
    // measured so every family (not only the MMQ J-best minority) gets a
    // repeatability canary. A synthetic measurement role, never a candidate:
    // excluded from winner selection, not counted in result.measured, and
    // emitted as "<stable_name>#twin" via measurement_name().
    bool     is_native_twin            = false;
};

// HI24 step 4: the identity of a measurement *instance*, distinct from its
// candidate. The double-native twin carries native's own descriptor, so its
// rows are emitted as "<stable_name>#twin" -- '#' is not legal in a stable
// name (standards 2.1), so no real candidate can collide with it. One helper
// everywhere the JSONL or the schedule names a measurement instance.
static std::string measurement_name(const Measurement & m) {
    std::string name = std::string(m.candidate ? m.candidate->stable_name : "")
           + (m.is_native_twin ? "#twin" : "");
#ifdef GGML_HIP_ROUTING_TRANSFORM
    // HI30: a transformed route is a different route identity from the same
    // candidate's direct one -- '#' is not legal in a stable name, so this
    // cannot collide with a real candidate or with the #twin suffix above.
    if (m.transform != nullptr) {
        name += "#xform:" + std::string(m.transform->name);
    }
#endif
    return name;
}

// HI50: whether the noise canary (same-kernel pair, see below) confirmed
// this signature's timings are trustworthy. NOT_AVAILABLE means no canary
// pair existed to check (e.g. no MMQ J-best twin); UNRESOLVED means a pair
// existed but never converged within noise_canary_retries -- a policy must
// not promote a challenger over an unresolved canary, since the margin it
// would be promoted on cannot be told apart from measurement noise.
enum ggml_hip_canary_state_v1 {
    GGML_HIP_CANARY_NOT_AVAILABLE = 0,
    GGML_HIP_CANARY_PASS,
    GGML_HIP_CANARY_RETRIED_PASS,
    GGML_HIP_CANARY_UNRESOLVED,
};

const char * canary_state_name(ggml_hip_canary_state_v1 state) {
    switch (state) {
        case GGML_HIP_CANARY_NOT_AVAILABLE: return "not_available";
        case GGML_HIP_CANARY_PASS:          return "pass";
        case GGML_HIP_CANARY_RETRIED_PASS:  return "retried_pass";
        case GGML_HIP_CANARY_UNRESOLVED:    return "unresolved";
        default:                            return "?";
    }
}

struct Result {
    const ggml_hip_candidate_descriptor * winner = nullptr;
#ifdef GGML_HIP_ROUTING_TRANSFORM
    // HI30: route identity is (candidate, variant, transform_id), not
    // candidate alone. nullptr means winner reached the destination directly
    // (or is native); a real value here means `winner` was only reachable
    // through this routing transformation.
    const ggml_hip_routing_transformation * winner_transform = nullptr;
    // HI30: transforms tried during the registry scan that produced no
    // measurable Measurement at all, with why -- consumed by record_result()
    // to emit HI29's TransformGapRecord for this signature.
    std::vector<std::pair<ggml_hip_transform_id, std::string>> transform_scan_gaps;
#endif
    std::string native_name;   // recorded once, up front -- promotion needs
                                // "did this row change from native" without
                                // relying on winner->stable_name still being
                                // comparable after promotion_status demotes
                                // winner back to native.candidate.
    std::vector<Measurement> measurements;
    int generated  = 0;   // candidates in the registry for this family/shape
    int applicable = 0;   // E5: right arch + right src0_type (varies; generated does not)
    int eligible   = 0;
    int measured   = 0;
    double improvement_pct = 0.0;
    double confidence      = 0.0;   // E1: 1 - sign_p of the chosen winner
    std::string reason;

    // HI34: shared batch size for this signature, derived once from a native
    // pilot and reused for every subsequent launch (native and candidate
    // alike) so the comparison is never biased by per-candidate calibration.
    int launches_per_sample = 1;

    // HI34: the seed that ordered the final-measurement rounds and (below)
    // the confirmation holdout, plus the finalist set it was drawn over --
    // together enough for an offline reader (tune_promotion.py) to verify
    // the schedule this run actually used rather than trust a claim.
    uint32_t schedule_seed = 0;
    std::vector<std::string> schedule_candidates;

    // HI34: fresh, disjoint confirmation holdout for the provisional winner.
    // Populated only when selection nominates a non-native challenger.
    std::vector<double> confirmation_native_us;
    std::vector<double> confirmation_winner_us;
    double p_value               = 1.0;
    int    confirmation_wins     = 0;
    int    confirmation_rounds   = 0;
    double confirmation_effect_pct = 0.0;

    // HI52 part 2: device clock/power/thermal snapshot, before and after this
    // signature's measurement work. Falsification only -- never a ranking
    // axis -- so a drift check can tell "the winner changed because the
    // hardware got slower" from "the winner changed because the code did".
    // "{}" (not a zeroed struct) when the capture is off, unavailable, or
    // could not resolve the device.
    std::string device_state_pre_json  = "{}";
    std::string device_state_post_json = "{}";
    ggml_hip_device_state device_state_pre;
    ggml_hip_device_state device_state_post;
    std::string device_clock_drift_json = "{\"status\":\"unavailable\"}";

    // HI60: observation-only evidence for per-round clock counterbalancing.
    // These fields deliberately do not participate in any identity, ranking,
    // or replay key.
    int clock_drift_rounds = 0;
    int reverse_retime_attempts = 0;
    int reverse_retime_passed = 0;
    std::string retime_status = "not_needed";
    bool measurement_failure = false;

    // HI50: every compiled-in policy's full verdict, for offline comparison
    // (bigcherry rank-replay). Which one actually governed this signature's
    // promotion.
    ggml_hip_canary_state_v1 canary_state = GGML_HIP_CANARY_NOT_AVAILABLE;
    std::string ranking_decisions_json    = "[]";
    std::string production_policy_name    = "latency-v1";
    int         production_policy_version = 1;

    // HI50: the ranking-stage pick, fixed before the confirmation holdout
    // (and any later promotion demotion) can change `winner` again -- the
    // stable target for offline replay to validate a policy's ranking
    // output against.
    std::string provisional_winner;

    // HI34/promotion: "native" (native retained, nothing to confirm),
    // "confirmation_rejected" (a provisional winner failed fresh holdout,
    // native retained), or "pending_bh" (fresh holdout passed; this result
    // is a *candidate* for promotion, not yet safe to export to a replay
    // cache -- experiment-wide Benjamini-Hochberg correction, run offline
    // over every pending_bh result together via `bigcherry tune-promote`,
    // decides that). A replay-cache exporter must treat anything other than
    // "native" or "promoted" (a status this file never sets; tune_promotion.py
    // sets it after BH) as unsafe to ship.
    std::string promotion_status = "native";

    // The two halves the dispatch digest was built from.
    //
    // Recorded because the dispatch digest also mixes in this build's manifest
    // hash and source revision (standards 13.1), so it is only meaningful to
    // the build that produced it. Without the components, no tool can compute
    // any other key from a measurements file -- not a re-keyed cache for a
    // slimmed catalog, not a migration after an upstream bump, not even an
    // offline check of which signatures two runs share. See HI23.
    ggml_hip_digest signature_digest = {};
    ggml_hip_digest hardware_digest  = {};
    std::string canonical_json;

    // HI24 noise canary. `canary_pct` is the divergence between two
    // measurements of the *same kernel*; anything above zero is pure
    // measurement error. -1 means the check could not be run for this
    // signature (no same-kernel pair reached final measurement).
    double canary_pct    = -1.0;
    int    canary_retries = 0;   // number of stability probes performed (0 or 1, HI68)
    std::string canary_pair;   // which two entries were compared
    // HI68: true once the fresh complete finalist block has been produced
    // mechanically (complete, non-poisoned, retime-resolved) and its window
    // has replaced every finalist's statistics -- set BEFORE the E4
    // post-block scrutiny so a row rejected on that scrutiny still says the
    // evidence is the fresh block's. Invariants: canary_state == retried_pass
    // implies canary_fresh_block; canary_fresh_block + canary_unresolved means
    // fresh evidence was collected but remained non-decision-grade (terminal,
    // native retained); a false flag always means original-block evidence.
    // That last invariant holds even when a fresh attempt STARTED but failed
    // mechanically: the attempt is transactional -- every field it could have
    // mutated is snapshotted beforehand and restored on both failure paths
    // (measurement_failure/poison, retime-unresolved), so partial fresh
    // statistics can never be serialized under a false flag.
    bool canary_fresh_block = false;
};

// HI30/HI31: which transform (if any) the FINAL winner reached its
// candidate through -- distinct from measurement_name()'s per-measurement
// "#xform:" suffix, this is the one durable fact ggml_hip_tuner_flush()'s
// measurements.jsonl row must carry so tune_promotion.py's promoted-winners
// output and replay_cache.py's exporter (HI31) know which transform_id to
// pack into the v5 replay entry. Previously only the diagnostic journal
// (journal_result_summary()) carried this -- the actual production artifact
// silently dropped it, which would have made every transformed winner
// replay as its bare candidate against the wrong (untransformed) signature.
static const char * winner_transform_name(const Result & r) {
#ifdef GGML_HIP_ROUTING_TRANSFORM
    return r.winner_transform != nullptr ? r.winner_transform->name : "";
#else
    (void) r;
    return "";
#endif
}

// The numeric id alongside the name (above): replay_cache.py packs this
// directly into the v5 replay entry's transform_id field without needing
// its own name -> id mapping of the transform registry.
static int winner_transform_id(const Result & r) {
#ifdef GGML_HIP_ROUTING_TRANSFORM
    return r.winner_transform != nullptr ? (int) r.winner_transform->id : 0;
#else
    (void) r;
    return 0;
#endif
}

struct DigestHash {
    size_t operator()(const ggml_hip_digest & d) const {
        size_t v = 0;
        for (size_t i = 0; i < sizeof(size_t) && i < GGML_HIP_DIGEST_BYTES; ++i) {
            v |= (size_t) d.bytes[i] << (8 * i);
        }
        return v;
    }
};

struct DigestEqual {
    bool operator()(const ggml_hip_digest & a, const ggml_hip_digest & b) const {
        return ggml_hip_digest_equal(a, b);
    }
};

std::unordered_map<ggml_hip_digest, Result, DigestHash, DigestEqual> g_results;
std::mutex g_mutex;

#ifdef GGML_HIP_ROUTING_TRANSFORM
// bigcherry (HI29): transform-attempt / transform-gap recording, written
// alongside the ordinary measurements.jsonl (see ggml_hip_tuner_flush()'s
// .transforms.jsonl block). A transform-attempt row is written whenever
// HI30's tuner integration tries a routing transformation (HI27/HI28)
// against a signature; a transform-gap row is written when none of the
// registered transforms worked. Kept structurally separate from Result --
// these are diagnostic records for offline (agent-driven) pattern analysis
// across many signatures, not the primary dispatch-identity/promotion
// evidence Result already carries for the signature that was actually
// tuned.
struct TransformAttemptRecord {
    ggml_hip_digest        original_sig = {};
    ggml_hip_digest        hardware     = {};
    ggml_hip_transform_id  transform_id = GGML_HIP_TRANSFORM_NONE;
    std::string            transform_name;
    std::string            source;                 // "predefined" | "discovered"
    std::string            original_native_family;
    std::string            result;                 // "success" | "rejected"
    std::string            rejection_reason;        // empty on success
    std::string            transformed_winner;      // candidate stable_name, empty if rejected
    double                 original_us      = 0.0;
    double                 transformed_us   = 0.0;
    double                 improvement_pct  = 0.0;
    double                 nmse             = 0.0;
    double                 max_abs_error    = 0.0;
};

// One (transform, rejection reason) pair -- part of a TransformGapRecord's
// "everything HI30 tried and why none of it worked" trail.
struct TransformTriedEntry {
    ggml_hip_transform_id transform_id;
    std::string           rejection_reason;
};

struct TransformGapRecord {
    ggml_hip_digest                    sig          = {};
    ggml_hip_digest                    hardware     = {};
    std::string                        native_family;
    int64_t                            est_bytes = 0;
    std::vector<TransformTriedEntry>   tried;
};

std::vector<TransformAttemptRecord> g_transform_attempts;
std::vector<TransformGapRecord>     g_transform_gaps;
std::mutex                          g_transform_mutex;

// bigcherry family enum -> the lowercase name candidate/catalog code already
// uses elsewhere (config/recipes.toml patch-set naming, catalog.toml
// backend fields) -- kept local since no shared helper does this today.
const char * transform_family_name(int family) {
    switch (family) {
        case GGML_HIP_FAMILY_MMVQ: return "mmvq";
        case GGML_HIP_FAMILY_MMQ:  return "mmq";
        case GGML_HIP_FAMILY_MMVF: return "mmvf";
        case GGML_HIP_FAMILY_MMF:  return "mmf";
        case GGML_HIP_FAMILY_BLAS: return "blas";
        default:                   return "?";
    }
}

void ggml_hip_record_transform_attempt(const TransformAttemptRecord & record) {
    std::lock_guard<std::mutex> lock(g_transform_mutex);
    g_transform_attempts.push_back(record);
}

void ggml_hip_record_transform_gap(const TransformGapRecord & record) {
    std::lock_guard<std::mutex> lock(g_transform_mutex);
    g_transform_gaps.push_back(record);
}
#endif // GGML_HIP_ROUTING_TRANSFORM

// GPU measurements are process-global device state.  Serialize the complete
// cold-path experiment so concurrent first encounters cannot perturb one
// another or publish different winners for the same dispatch key.
// This is deliberately process-wide rather than per-key. HIP measurements
// mutate process-global device/workspace state, so a per-key gate would still
// allow two different first encounters to perturb each other. Holding this
// gate from the cache lookup through record_result() -- and, since HI64
// (2026-08-22), through the public wrapper's own post-call measurement_
// failure read too -- gives every dispatch key single-flight semantics.
// That last clause matters now in a way it didn't before HI64: one
// measurement/publish no longer means every later waiter observes the
// SAME committed winner forever -- a device-local failure is retained only
// as fallback evidence, and a subsequent healthy device's success replaces
// it (record_result()'s failure->success rule). Held by
// ggml_hip_tuner_resolve() (the public wrapper), not the internal impl, so
// no other thread can slip a replacement into g_results between the impl
// returning and the wrapper reading back what actually got recorded.
std::mutex g_single_flight_mutex;

// bigcherry (HI64, RV49 F6 scope extension, 2026-08-17 GPT adjudication):
// poison/cache state keyed by the CURRENT hipGetDevice() value, not a
// single process-global flag -- a transient fault on one GPU (e.g. a WDDM
// TDR-class hiccup) must not disable tuning on every OTHER device sharing
// this process. Every existing .load()/.store() call site keeps working
// unchanged; only the storage moves from one flag to one-per-device.
//
// Deliberately NO auto-clear of any kind (no time-based expiry, no
// clear-after-one-successful-operation): a device that poisoned stays
// poisoned for the rest of the process. Per GPT's own requirement, recovery
// may only happen once the surrounding backend can PROVE the HIP context/
// device execution generation was recreated -- ggml-cuda.cu exposes no such
// generation/reset signal today, so that recovery path is explicitly NOT
// implemented here; a poisoned device requires a fresh process, exactly as
// before this change. This file still never calls hipDeviceReset() itself.
template <typename T>
class PerDeviceState {
public:
    explicit PerDeviceState(T initial) : initial_(initial) {}

    T load(std::memory_order = std::memory_order_relaxed) const {
        std::lock_guard<std::mutex> lock(mutex_);
        const int device = current_device();
        const auto it = values_.find(device);
        return it != values_.end() ? it->second : initial_;
    }

    void store(T value, std::memory_order = std::memory_order_relaxed) {
        std::lock_guard<std::mutex> lock(mutex_);
        values_[current_device()] = value;
    }

private:
    static int current_device() {
        int device = -1;
        // No resolvable device -- bucket under a shared sentinel key rather
        // than silently reading/writing some other device's state on this
        // caller's behalf.
        return hipGetDevice(&device) == hipSuccess && device >= 0 ? device : -1;
    }

    T initial_;
    mutable std::mutex mutex_;
    std::unordered_map<int, T> values_;
};

// A failed HIP measurement can leave the context poisoned. Once that happens,
// no later candidate, correctness copy, final round, determinism check, or
// device query may be treated as valid for THAT DEVICE in this process --
// other devices' tuning is unaffected (see PerDeviceState's comment above).
PerDeviceState<bool> g_tuner_poisoned(false);

// UTC, second resolution, ISO-ish -- only used to name a journal experiment
// uniquely per process, not parsed back by anything, so exact format is not
// load-bearing.
std::string utc_timestamp() {
    std::time_t now = std::time(nullptr);
    std::tm tm_utc;
#if defined(_WIN32)
    gmtime_s(&tm_utc, &now);
#else
    gmtime_r(&now, &tm_utc);
#endif
    char buf[32];
    std::strftime(buf, sizeof(buf), "%Y%m%dT%H%M%SZ", &tm_utc);
    return std::string(buf);
}

// A compact per-candidate summary for the journal's "result" event -- not
// the full ~80-field measurements-file record (that is built separately by
// ggml_hip_tuner_flush()), just enough to identify which candidate won,
// why, and how much evidence backed the answer, so a killed run's partial
// journal is still useful without a human re-deriving it from raw samples.
std::string journal_result_summary(
        const ggml_hip_digest & dispatch_digest, const Result & result) {
    return std::string("{\"kind\":\"result\",\"dispatch\":\"") +
        ggml_hip_digest_hex(dispatch_digest) +
        "\",\"signature\":\"" +
        ggml_hip_digest_hex(result.signature_digest) +
        "\",\"hardware\":\"" +
        ggml_hip_digest_hex(result.hardware_digest) +
        "\",\"winner\":\"" +
        (result.winner != nullptr ? result.winner->stable_name : "(none)") +
#ifdef GGML_HIP_ROUTING_TRANSFORM
        "\",\"winner_transform\":\"" +
        (result.winner_transform != nullptr ? result.winner_transform->name : "") +
#endif
        "\",\"native\":\"" + result.native_name +
        "\",\"promotion_status\":\"" + result.promotion_status +
        "\",\"reason\":\"" + result.reason +
        "\",\"generation\":1" +
        ",\"generated\":" + std::to_string(result.generated) +
        ",\"applicable\":" + std::to_string(result.applicable) +
        ",\"eligible\":" + std::to_string(result.eligible) +
        ",\"measured\":" + std::to_string(result.measured) +
        ",\"improvement_pct\":" + std::to_string(result.improvement_pct) +
        ",\"confidence\":" + std::to_string(result.confidence) +
        ",\"retime_status\":\"" + result.retime_status +
        "\",\"clock_drift_rounds\":" + std::to_string(result.clock_drift_rounds) +
        ",\"reverse_retime_attempts\":" +
        std::to_string(result.reverse_retime_attempts) +
        ",\"reverse_retime_passed\":" +
        std::to_string(result.reverse_retime_passed) +
        ",\"measurement_failure\":" +
        (result.measurement_failure ? "true" : "false") + "}";
}

// Called from every g_results.emplace() site (there are several -- most are
// early "native not eligible/rejected" exits). Emplacing into g_results and
// appending to the journal are two independent, differently-mutexed
// operations on purpose: g_mutex protects the in-memory map that
// ggml_hip_tuner_flush() reads at shutdown, while the journal's own mutex
// (inside hip-autotune-journal.cpp) guards the file handle -- holding
// g_mutex across a filesystem fsync would serialise every concurrent
// dispatch behind disk I/O for no reason.
void open_tuning_journal_once(const ggml_hip_digest & hardware_digest) {
    static std::atomic<bool> journal_open_attempted{false};
    if (journal_open_attempted.exchange(true)) {
        return;
    }
    const char * db_path = getenv("GGML_HIP_DISPATCH_DB");
    if (db_path != nullptr && db_path[0] != '\0') {
        const std::string journal_path = std::string(db_path) + ".journal.jsonl";
        const std::string experiment_id = "tune-" + utc_timestamp();
        if (!ggml_hip_journal_open(
                journal_path.c_str(), experiment_id,
                GGML_HIP_AUTOTUNE_SOURCE_REVISION_STR,
                GGML_HIP_AUTOTUNE_MANIFEST_HASH_STR, hardware_digest)) {
            GGML_LOG_WARN("bigcherry: cannot open tuning journal '%s'; "
                          "a killed run will not be incrementally recoverable\n",
                          journal_path.c_str());
        }
    }
}

#ifdef GGML_HIP_ROUTING_TRANSFORM
// HI30: derives HI29's transform-attempt/transform-gap diagnostics from the
// finished Result, at the one choke point every resolve() exit already
// funnels through (record_result). Keeping this out of resolve() itself
// means every early-return path gets correct diagnostics for free instead
// of needing its own recorder call, and the records always describe what
// was actually measured -- never a claim made before the outcome is known.
void record_transform_diagnostics(const Result & result) {
    bool any_transform_measurement = false;
    for (const Measurement & m : result.measurements) {
        if (m.transform != nullptr) { any_transform_measurement = true; break; }
    }
    if (!any_transform_measurement && result.transform_scan_gaps.empty()) {
        return;
    }

    const Measurement * native_measurement = nullptr;
    for (const Measurement & m : result.measurements) {
        if (m.transform == nullptr && !m.is_native_twin &&
                m.candidate != nullptr && m.candidate->stable_name == result.native_name) {
            native_measurement = &m;
            break;
        }
    }
    const std::string native_family = native_measurement != nullptr
        ? transform_family_name((int) native_measurement->candidate->family) : "?";

    // Best measured candidate per transform id actually attempted.
    std::unordered_map<int, const Measurement *> best_by_transform;
    for (const Measurement & m : result.measurements) {
        if (m.transform == nullptr) continue;
        if (!m.measured) continue;
        const int id = (int) m.transform->id;
        const auto it = best_by_transform.find(id);
        if (it == best_by_transform.end() || m.median_us < it->second->median_us) {
            best_by_transform[id] = &m;
        }
    }

    std::unordered_set<int> attempted_ids;
    for (const Measurement & m : result.measurements) {
        if (m.transform == nullptr) continue;
        const int id = (int) m.transform->id;
        if (!attempted_ids.insert(id).second) continue;

        TransformAttemptRecord rec;
        rec.original_sig             = result.signature_digest;
        rec.hardware                 = result.hardware_digest;
        rec.transform_id             = m.transform->id;
        rec.transform_name           = m.transform->name;
        rec.source                   = m.transform->source == GGML_HIP_TRANSFORM_SOURCE_PREDEFINED
                                          ? "predefined" : "discovered";
        rec.original_native_family   = native_family;

        const bool is_success = result.winner_transform != nullptr &&
            result.winner_transform->id == m.transform->id;
        const auto found = best_by_transform.find(id);
        const Measurement * best = found != best_by_transform.end() ? found->second : nullptr;

        rec.result             = is_success ? "success" : "rejected";
        rec.rejection_reason   = is_success ? "" : (best != nullptr
            ? "not selected as winner" : "no target candidate measured successfully");
        rec.transformed_winner = best != nullptr ? best->candidate->stable_name : "";
        rec.original_us        = native_measurement != nullptr ? native_measurement->median_us : 0.0;
        rec.transformed_us     = best != nullptr ? best->median_us : 0.0;
        rec.improvement_pct    = (rec.original_us > 0.0 && rec.transformed_us > 0.0)
            ? 100.0 * (rec.original_us - rec.transformed_us) / rec.original_us : 0.0;
        rec.nmse             = best != nullptr ? best->nmse : 0.0;
        rec.max_abs_error    = best != nullptr ? best->max_abs_error : 0.0;
        ggml_hip_record_transform_attempt(rec);
    }

    if (!result.transform_scan_gaps.empty()) {
        TransformGapRecord gap;
        gap.sig           = result.signature_digest;
        gap.hardware      = result.hardware_digest;
        gap.native_family = native_family;
        gap.est_bytes     = 0;
        for (const auto & entry : result.transform_scan_gaps) {
            gap.tried.push_back({entry.first, entry.second});
        }
        ggml_hip_record_transform_gap(gap);
    }
}
#endif // GGML_HIP_ROUTING_TRANSFORM

void record_result(const ggml_hip_digest & dispatch_digest, const Result & result) {
#ifdef GGML_HIP_ROUTING_TRANSFORM
    record_transform_diagnostics(result);
#endif
    {
        std::lock_guard<std::mutex> lock(g_mutex);
        // HI64 (2026-08-22, real dual-XTX hardware finding): g_results is
        // keyed only by the portable dispatch_digest (hardware identity
        // deliberately excludes device ordinal so identical GPUs can share
        // a winner, standards 10.2) -- a blind emplace() here let a single
        // device's fatal measurement failure permanently own that portable
        // key, silently blocking every OTHER device sharing it (including
        // an identical twin GPU) from ever attempting its own measurement,
        // confirmed on real hardware: a fault injected on the first-
        // processed device also poisoned the second device's tuning even
        // though the second device never itself failed.
        //
        // Fix: a failed result is retained ONLY as fallback evidence (so an
        // all-failed run still appears in measurements.jsonl instead of
        // looking empty) and is replaced the first time a non-failed result
        // arrives for the same key. A non-failed result is never replaced
        // by a later failure -- the successful result stays authoritative.
        // See the matching g_results.find() guard in
        // ggml_hip_tuner_resolve_impl(), which must treat a failed entry as
        // NOT a cache hit for this to actually let another device retry.
        const auto found = g_results.find(dispatch_digest);
        if (found == g_results.end()) {
            g_results.emplace(dispatch_digest, result);
        } else if (found->second.measurement_failure && !result.measurement_failure) {
            found->second = result;
        }
    }

    // The journal is opened at cold-path entry when the signature and hardware
    // digests are first known, so this call is only a fallback for early result
    // paths and remains attempt-once.
    open_tuning_journal_once(result.hardware_digest);
    if (ggml_hip_journal_is_open()) {
        ggml_hip_journal_append_result(journal_result_summary(dispatch_digest, result));
    }
}

const char * reason_name(ggml_hip_reject_reason r) {
    switch (r) {
        case GGML_HIP_REJECT_NONE:          return "ok";
        case GGML_HIP_REJECT_ARCHITECTURE:  return "architecture";
        case GGML_HIP_REJECT_INELIGIBLE:    return "ineligible";
        case GGML_HIP_REJECT_WORKSPACE:     return "workspace";
        case GGML_HIP_REJECT_LAUNCH_FAILED: return "launch_failed";
        case GGML_HIP_REJECT_NAN_INF:       return "nan_inf";
        case GGML_HIP_REJECT_TOLERANCE:     return "tolerance";
        case GGML_HIP_REJECT_UNSTABLE:      return "unstable";
        case GGML_HIP_REJECT_NOISY:         return "noisy";
        default:                            return "?";
    }
}

// E1: NaN marks a round where a candidate failed to launch (position must
// still equal round number, so the pairing survives). std::sort over a range
// containing NaN is undefined behaviour, not merely an odd answer, so every
// statistic below filters first.
std::vector<double> finite_only(const std::vector<double> & v) {
    std::vector<double> out;
    out.reserve(v.size());
    for (double x : v) {
        if (!std::isnan(x)) out.push_back(x);
    }
    return out;
}

double median_of(const std::vector<double> & v_in) {
    std::vector<double> v = finite_only(v_in);
    if (v.empty()) return 0.0;
    std::sort(v.begin(), v.end());
    const size_t n = v.size();
    return n % 2 ? v[n / 2] : 0.5 * (v[n / 2 - 1] + v[n / 2]);
}

double percentile_of(const std::vector<double> & v_in, double p) {
    std::vector<double> v = finite_only(v_in);
    if (v.empty()) return 0.0;
    std::sort(v.begin(), v.end());
    const size_t idx = (size_t) (p * (double) (v.size() - 1));
    return v[idx];
}

// Median absolute deviation. Preferred over stddev here because a single
// preemption or clock excursion makes stddev meaningless while MAD barely
// moves -- and GPU timings are full of both.
double mad_of(const std::vector<double> & v_in, double median) {
    std::vector<double> v = finite_only(v_in);
    if (v.empty()) return 0.0;
    std::vector<double> deviations;
    deviations.reserve(v.size());
    for (double x : v) {
        deviations.push_back(std::fabs(x - median));
    }
    return median_of(deviations);
}

// E1: P(X >= k) for X ~ Binomial(n, 0.5), exact. Summed downward from i = n
// with a running ratio, so no binomial coefficient is ever materialised and
// nothing overflows for any n the tuner can produce.
//   C(n, i-1) = C(n, i) * i / (n - i + 1)
double binomial_tail_ge(int k, int n) {
    if (k <= 0) return 1.0;
    if (k >  n) return 0.0;
    double term = std::exp(-(double) n * std::log(2.0));   // C(n,n) / 2^n
    double sum  = term;
    for (int i = n; i > k; --i) {
        term *= (double) i / (double) (n - i + 1);
        sum  += term;
    }
    return std::min(sum, 1.0);
}

// E1: paired sign test over the interleaved final rounds. Only the direction
// of each paired difference is used, so one preempted round costs a single
// vote instead of dominating a mean -- the same reasoning that put MAD in
// this file rather than stddev. Returns the one-sided p-value for "candidate
// is faster than baseline", and 1.0 when too few usable rounds survive to
// say anything.
double paired_sign_test(const std::vector<double> & baseline,
                        const std::vector<double> & candidate,
                        int min_rounds, int & wins_out, int & rounds_out) {
    wins_out = rounds_out = 0;
    const size_t rounds = std::min(baseline.size(), candidate.size());
    int wins = 0;
    int n    = 0;
    for (size_t r = 0; r < rounds; ++r) {
        if (std::isnan(baseline[r]) || std::isnan(candidate[r])) continue;
        if (candidate[r] == baseline[r]) continue;   // tie: excluded, standard
        if (candidate[r] < baseline[r]) ++wins;
        ++n;
    }
    wins_out = wins; rounds_out = n;
    return n < min_rounds ? 1.0 : binomial_tail_ge(wins, n);
}

// E5: aggregate the per-candidate reject reasons so a 1151-signature sweep is
// readable during the run rather than only after post-processing.
void reject_counts(const Result & r, int counts[GGML_HIP_REJECT_COUNT]) {
    std::fill(counts, counts + GGML_HIP_REJECT_COUNT, 0);
    for (const Measurement & m : r.measurements) {
        ++counts[m.reason];
    }
}

// Compare a candidate's output against native's, on the host.
//
// Both NMSE and max absolute error, because they fail differently: NMSE catches
// a systematically wrong result, max-abs catches a single catastrophic element
// that averaging would hide.
bool compare_outputs(const std::vector<float> & reference,
                     const std::vector<float> & actual,
                     double & nmse, double & max_abs) {
    nmse = 0.0;
    max_abs = 0.0;
    if (reference.size() != actual.size() || reference.empty()) {
        return false;
    }

    double sq_error = 0.0;
    double sq_ref   = 0.0;
    for (size_t i = 0; i < reference.size(); ++i) {
        const double a = actual[i];
        if (std::isnan(a) || std::isinf(a)) {
            return false;   // standards 7.2: introduced NaN/Inf is fatal
        }
        const double d = a - reference[i];
        sq_error += d * d;
        sq_ref   += (double) reference[i] * reference[i];
        max_abs = std::max(max_abs, std::fabs(d));
    }
    nmse = sq_ref > 0.0 ? sq_error / sq_ref : sq_error;
    return true;
}

// Diagnostic only -- attributes a device-side crash (illegal memory access,
// XNACK fault) to the candidate that caused it. Not a replacement for the
// crash-safe journal, a companion to it: a candidate that faults never
// reaches record_result(), so the journal's last "result" entry alone is one
// candidate short of the truth. This writes an "attempt" event into the SAME
// journal, *before* the risky GPU work, so the crashing candidate is the
// last line even when the process never executes another line of C++
// afterward.
//
// The journal is opened at cold-path entry after the resolver computes the
// signature and hardware digests, so the first candidate is traceable too.
// Off by default (one getenv + atomic check when unset) -- do not enable for a
// real tuning run, fsync-per-launch
// is the wrong trade when nothing is crashing. Correlate against
// `HIP_LAUNCH_BLOCKING=1`: without it, several launches can be in-flight on
// the device ahead of the one that actually faults, and the last logged
// attempt would name an innocent candidate that merely happened to be
// dispatched most recently, not the one whose kernel corrupted memory.
thread_local ggml_hip_digest g_trace_signature_digest = {};
// Keep the complete device-local identity beside the compact digest while the
// cold-path signature is being measured. Re-serializing this POD value at the
// trace seam avoids depending on Result's later journal/measurements-file
// lifetime. This is only serialized when GGML_HIP_TUNE_TRACE_ATTEMPTS is
// enabled; normal tuning and production dispatch do not write this payload.
thread_local ggml_hip_dispatch_signature_v1 g_trace_signature = {};

void trace_launch_attempt(const char * stable_name, const char * protocol_stage) {
    static std::atomic<bool> enabled{false};
    static std::atomic<bool> checked{false};
    if (!checked.exchange(true)) {
        const char * flag = getenv("GGML_HIP_TUNE_TRACE_ATTEMPTS");
        enabled = (flag != nullptr && flag[0] != '\0' && flag[0] != '0');
    }
    if (!enabled || !ggml_hip_journal_is_open()) return;
    const std::string signature_json = ggml_hip_signature_json(g_trace_signature, true);
    const std::string payload =
        "{\"candidate\":\"" + std::string(stable_name ? stable_name : "(null)") +
        "\",\"signature\":\"" + ggml_hip_digest_hex(g_trace_signature_digest) +
        "\",\"signature_json\":" + signature_json +
        ",\"stage\":\"" + std::string(protocol_stage ? protocol_stage : "unknown") +
        "\",\"t_us\":" + std::to_string(ggml_time_us()) + "}";
    ggml_hip_journal_append_attempt(payload);
}

// HI40/HI54: emit the actual workspace lifecycle when explicitly requested.
// This is diagnostic-only and completely dormant in normal tuning and
// production dispatch. Opening per event keeps the trace durable enough for a
// killed tuning process without adding a persistent file descriptor to the
// hot path when tracing is disabled.
void trace_workspace_event(const char * stage, const char * event,
                           const char * stable_name) {
    const char * path = getenv("GGML_HIP_WORKSPACE_TRACE");
    if (path == nullptr || path[0] == '\0') return;
    static std::mutex trace_mutex;
    std::lock_guard<std::mutex> lock(trace_mutex);
    FILE * file = fopen(path, "ab");
    if (file == nullptr) return;
    fprintf(file,
            "{\"stage\":\"%s\",\"event\":\"%s\","
            "\"candidate\":\"%s\",\"t_us\":%lld}\n",
            stage ? stage : "unknown", event ? event : "unknown",
            stable_name ? stable_name : "(null)",
            (long long) ggml_time_us());
    fflush(file);
    fclose(file);
}

// Launch one candidate into scratch and time the complete path.
//
// `lc` already points at scratch, so the caller's real destination is never
// touched -- a candidate that produces garbage must not corrupt the run it is
// being measured inside.
static bool hip_ok(hipError_t status, const char * what) {
    if (status == hipSuccess) return true;
    GGML_LOG_WARN("bigcherry: %s failed: %s\n", what, hipGetErrorString(status));
    return false;
}

// HIP event destruction is part of the measurement transaction.  Keep it in
// one checked seam so every event is attempted exactly once, even when an
// earlier cleanup fails.  Clearing the handle makes accidental double cleanup
// harmless for callers that need to report a primary measurement failure.
static bool hip_event_destroy_checked(hipEvent_t & event, const char * what) {
    if (event == nullptr) return true;
    const hipError_t status = hipEventDestroy(event);
    event = nullptr;
    return hip_ok(status, what);
}

// A failed HIP timing transaction can poison the context even after its
// pending error is consumed. SMI capture calls ggml_cuda_get_device(), which
// is deliberately fail-fast on that condition, so disable further SMI reads
// for THIS DEVICE once any timing failure is observed on it -- other
// devices keep capturing SMI normally (see PerDeviceState's comment above).
// Measurements remain valid as explicit unavailable evidence; timing
// failures themselves still reject the affected round/signature.
PerDeviceState<bool> g_smi_runtime_disabled(false);

static void disable_smi_after_measurement_failure() {
    g_tuner_poisoned.store(true, std::memory_order_relaxed);
    g_smi_runtime_disabled.store(true, std::memory_order_relaxed);
    (void) hipGetLastError();
}

// Test-only fault injection. Both selectors are required, and the one-shot
// guard makes the failure deterministic without turning a normal production
// process into a poisoned tuner. The attempt is traced before this hook so a
// failed candidate remains attributable in the durable journal.
static bool inject_test_measurement_failure(const char * candidate,
                                             const char * stage) {
    const char * requested_candidate = getenv("GGML_HIP_TUNE_TEST_FAIL_CANDIDATE");
    const char * requested_stage = getenv("GGML_HIP_TUNE_TEST_FAIL_STAGE");
    if (requested_candidate == nullptr || requested_candidate[0] == '\0'
            || requested_stage == nullptr || requested_stage[0] == '\0'
            || candidate == nullptr || stage == nullptr
            || std::strcmp(requested_candidate, candidate) != 0
            || std::strcmp(requested_stage, stage) != 0) {
        return false;
    }
    static std::atomic<bool> consumed{false};
    if (consumed.exchange(true, std::memory_order_relaxed)) {
        return false;
    }
    GGML_LOG_WARN("bigcherry: injected test measurement failure for %s at %s\n",
                  candidate, stage);
    disable_smi_after_measurement_failure();
    return true;
}

static bool smi_capture_enabled() {
    return ggml_hip_smi_enabled() &&
        !g_smi_runtime_disabled.load(std::memory_order_relaxed);
}

// HI34 step 3 (Slice B0): evict the last level of the cache hierarchy.
//
// Sized to the last level, not the first: on Navi 31 that is 64-96 MB of
// Infinity Cache behind a 6 MB L2, and hipDeviceAttributeL2CacheSize reports
// the L2 only. A 12 MB eviction buffer leaves a fully-resident src0 resident
// in Infinity Cache, so the flushed arm would read from the same place as the
// unflushed one and report "no effect" for a reason unrelated to the
// hypothesis. The size is therefore an explicit megabyte count
// (GGML_HIP_TUNE_FLUSH_MB), 256 MB by default: comfortably past the largest
// MALL on this generation, a rounding error against GPU memory a tuning run
// already holds.
//
// `volatile` on the store: a kernel whose only effect is dead stores is a
// kernel that flushes nothing, and it would fail silently.
__global__ void ggml_hip_cache_evict(volatile char * __restrict__ buffer,
                                      size_t n) {
    const size_t stride = (size_t) gridDim.x * blockDim.x;
    for (size_t i = blockIdx.x * (size_t) blockDim.x + threadIdx.x;
         i < n; i += stride) {
        buffer[i] = (char) (i & 0xff);
    }
}

// One buffer PER DEVICE for the process lifetime, not per signature: a
// per-signature allocation through ctx.pool() would churn the allocator harder
// than the thing being measured. Per device because hipMalloc allocations are
// device-specific -- a pointer allocated while tuning GPU A is not valid to
// launch on GPU B's stream without peer access, and an eviction experiment
// must not silently depend on peer capability.
struct ggml_hip_flush_evict_slot {
    int device = -1;
    char * buffer = nullptr;
};
static std::vector<ggml_hip_flush_evict_slot> & flush_evict_slots() {
    // The tuner is single-flight (one signature at a time under the
    // measurement mutex), so plain vector bookkeeping is safe.
    static std::vector<ggml_hip_flush_evict_slot> slots;
    return slots;
}
static char * flush_evict_buffer(int device) {
    for (const auto & slot : flush_evict_slots()) {
        if (slot.device == device) {
            return slot.buffer;
        }
    }
    const ggml_hip_tuner_config & config = ggml_hip_tuner_get_config();
    if (config.pre_sample_mode == GGML_HIP_PRE_SAMPLE_NONE) {
        return nullptr;
    }
    const size_t bytes = (size_t) std::max(config.flush_evict_mb, 1) << 20;
    char * p = nullptr;
    // Allocated on the current device, which the tuner has already selected
    // for this measurement; record the pairing so a later device never
    // reuses a foreign pointer.
    if (hipMalloc((void **) &p, bytes) != hipSuccess) {
        GGML_LOG_WARN("bigcherry: cannot allocate %zu MB on device %d for "
                      "the cache eviction; a run with GGML_HIP_TUNE_FLUSH_L2 "
                      "set cannot proceed without it\n", bytes >> 20, device);
    }
    flush_evict_slots().push_back({device, p});
    return p;
}

// The eviction must COMPLETE before the sample's measurement begins -- not
// merely be ordered before hipEventRecord(start) by stream sequencing.
// Stream ordering keeps it out of the GPU event interval, but the tuner also
// ranks on host wall time (effective_us = max(gpu, host - sync overhead)),
// and an unsynchronized 256 MB eviction tail would ride inside that host
// window: the final hipEventSynchronize(stop) waits for all earlier stream
// work, including the eviction, so host_us would absorb it. The explicit
// stream synchronization below closes that gap; it is outside both clocks.
// It must also not sit between the launches within a batch, which is why
// flush mode forces launches_per_sample to 1 in ggml_hip_tuner_get_config.
static bool launch_cache_evict(const ggml_hip_launch_context & lc) {
    int device = -1;
    if (hipGetDevice(&device) != hipSuccess || device < 0) {
        GGML_LOG_WARN("bigcherry: cannot determine current HIP device for "
                      "cache eviction\n");
        return false;
    }
    char * buffer = flush_evict_buffer(device);
    if (buffer == nullptr) {
        // Fail closed rather than silently measure unflushed: a run that
        // requested eviction is an experiment whose variable is the eviction,
        // and data collected without it must not masquerade as having it.
        return false;
    }
    const size_t bytes = (size_t) std::max(
        ggml_hip_tuner_get_config().flush_evict_mb, 1) << 20;
    ggml_hip_cache_evict<<<1024, 256, 0, lc.stream>>>(buffer, bytes);
    if (!hip_ok(hipGetLastError(), "cache evict launch")) {
        return false;
    }
    // Complete the eviction before any measurement clock starts (see the
    // comment above). Checked: a failure here must reject the sample.
    if (hipStreamSynchronize(lc.stream) != hipSuccess) {
        GGML_LOG_WARN("bigcherry: cache eviction synchronization failed on "
                      "device %d\n", device);
        return false;
    }
    return true;
}

// HI65: resolved-mode provenance string for the artifact header. The mode is
// the variable of the attribution experiment; flush_l2 in the same header is
// its 0/1 wire mirror kept for older readers.
static const char * pre_sample_mode_name(ggml_hip_pre_sample_mode mode) {
    switch (mode) {
        case GGML_HIP_PRE_SAMPLE_EVICT: return "evict";
        case GGML_HIP_PRE_SAMPLE_EVICT_REWARM: return "evict_rewarm";
        default: return "none";
    }
}

bool time_candidate(const ggml_hip_candidate_descriptor * candidate,
                    const ggml_hip_launch_context & lc,
                    int warmup, int samples, int launches_per_sample,
                    ggml_hip_pre_sample_mode pre_sample,
                    std::vector<double> & gpu_us,
                    std::vector<double> & host_us,
                    ggml_backend_cuda_context * workspace_ctx = nullptr,
                     bool isolate_workspace = false,
                     size_t * pool_peak_bytes = nullptr,
                     const char * protocol_stage = nullptr
#ifdef GGML_HIP_ROUTING_TRANSFORM
                     // HI30: every launch this function issues -- warmup,
                     // rewarm, and timed samples alike -- must run the exact
                     // same transformed route the caller is measuring, so
                     // the tuner can never rank a candidate against work it
                     // did not actually dispatch.
                     , const ggml_hip_routing_transformation * transform = nullptr
#endif
                     ) {
    const char * stage = protocol_stage != nullptr
        ? protocol_stage : (isolate_workspace ? "isolated_workspace" : "final");
    if (g_tuner_poisoned.load(std::memory_order_relaxed)) {
        return false;
    }
    trace_launch_attempt(candidate ? candidate->stable_name : nullptr, stage);
    if (inject_test_measurement_failure(candidate ? candidate->stable_name : nullptr,
                                        stage)) {
        return false;
    }

    hipEvent_t start = nullptr;
    hipEvent_t stop = nullptr;
    auto destroy_events = [&]() {
        // Do not short-circuit: both event handles must have a checked,
        // exactly-once destruction attempt.
        const bool start_ok = hip_event_destroy_checked(start, "hipEventDestroy(start)");
        const bool stop_ok = hip_event_destroy_checked(stop, "hipEventDestroy(stop)");
        return start_ok && stop_ok;
    };
    if (!hip_ok(hipEventCreate(&start), "hipEventCreate(start)")) {
        disable_smi_after_measurement_failure();
        return false;
    }
    if (!hip_ok(hipEventCreate(&stop), "hipEventCreate(stop)")) {
        destroy_events();
        disable_smi_after_measurement_failure();
        return false;
    }

    ggml_hip_candidate_descriptor effective = *candidate;

#ifdef GGML_HIP_ROUTING_TRANSFORM
    // Stack-allocated for the life of this call (standards: no heap
    // allocation on a measurement hot path). Reused across every launch
    // below -- apply() rewrites it in place each time, exactly as the
    // dispatcher (HI31) will on the runtime path.
    ggml_hip_transform_ctx xform_ctx;
    auto do_launch = [&]() -> bool {
        if (transform == nullptr) {
            effective.launch(&effective, lc);
            return true;
        }
        ggml_hip_launch_context out_lc;
        if (!transform->apply(lc, &xform_ctx, nullptr, &out_lc)) {
            return false;
        }
        ggml_hip_transform_launch(transform, &effective, effective.variant,
                                  &xform_ctx, out_lc);
        return true;
    };
#else
    auto do_launch = [&]() -> bool {
        effective.launch(&effective, lc);
        return true;
    };
#endif

#ifdef GGML_HIP_WORKSPACE_METRICS
    size_t workspace_baseline = 0;
    if (workspace_ctx != nullptr && isolate_workspace) {
        trace_workspace_event(stage, "clear_cache", candidate->stable_name);
        workspace_ctx->pool().bc_workspace_clear_cache();
        workspace_baseline = workspace_ctx->pool().bc_workspace_bytes();
    }
#else
    (void) workspace_ctx;
    (void) isolate_workspace;
    (void) pool_peak_bytes;
#endif

    trace_workspace_event(stage, "warmup_begin", candidate->stable_name);
    for (int i = 0; i < warmup; ++i) {
        if (!do_launch()) {
            destroy_events();
            disable_smi_after_measurement_failure();
            return false;
        }
    }
    trace_workspace_event(stage, "warmup_complete", candidate->stable_name);
    if (hipStreamSynchronize(lc.stream) != hipSuccess) {
        destroy_events();
        disable_smi_after_measurement_failure();
        return false;
    }
    if (hipGetLastError() != hipSuccess) {
        destroy_events();
        disable_smi_after_measurement_failure();
        return false;
    }
    trace_workspace_event(stage, "synchronize", candidate->stable_name);

#ifdef GGML_HIP_WORKSPACE_METRICS
    if (workspace_ctx != nullptr && isolate_workspace) {
        // Warmup establishes the reusable allocation baseline. Timed launches
        // are measured above that baseline, which is explicitly checked again
        // after the final synchronization below.
        workspace_baseline = workspace_ctx->pool().bc_workspace_bytes();
        workspace_ctx->pool().bc_workspace_reset_peak();
        trace_workspace_event(stage, "rebase_peak", candidate->stable_name);
    }
#endif

    gpu_us.reserve(samples);
    host_us.reserve(samples);

    for (int s = 0; s < samples; ++s) {
        if (pre_sample != GGML_HIP_PRE_SAMPLE_NONE
                && !launch_cache_evict(lc)) {
            destroy_events();
            disable_smi_after_measurement_failure();
            return false;
        }
        // HI65 EVICT_REWARM: after the eviction completes, run ONE untimed
        // rewarm launch (checked, synchronized) so every timed window starts
        // from a defined post-eviction state instead of a raw cold cache.
        // It sits outside both clocks for the same reason the eviction does:
        // an unsynchronized tail would ride inside host_us via the final
        // hipEventSynchronize(stop).
        if (pre_sample == GGML_HIP_PRE_SAMPLE_EVICT_REWARM) {
            trace_workspace_event(stage, "rewarm_begin", candidate->stable_name);
            if (!do_launch()) {
                destroy_events();
                disable_smi_after_measurement_failure();
                return false;
            }
            if (!hip_ok(hipGetLastError(), "rewarm launch")
                    || hipStreamSynchronize(lc.stream) != hipSuccess) {
                destroy_events();
                disable_smi_after_measurement_failure();
                return false;
            }
            trace_workspace_event(stage, "rewarm_complete", candidate->stable_name);
        }
        trace_workspace_event(stage, "timed_sample_begin", candidate->stable_name);
        const int64_t host_start = ggml_time_us();
        if (!hip_ok(hipEventRecord(start, lc.stream), "hipEventRecord(start)")) {
            destroy_events();
            disable_smi_after_measurement_failure();
            return false;
        }
        // Several launches per sample when one kernel is below event
        // resolution; the mean of the batch is the sample.
        for (int i = 0; i < launches_per_sample; ++i) {
            if (!do_launch()) {
                destroy_events();
                disable_smi_after_measurement_failure();
                return false;
            }
        }
        if (!hip_ok(hipEventRecord(stop, lc.stream), "hipEventRecord(stop)")) {
            destroy_events();
            disable_smi_after_measurement_failure();
            return false;
        }
        if (!hip_ok(hipEventSynchronize(stop), "hipEventSynchronize(stop)")) {
            destroy_events();
            disable_smi_after_measurement_failure();
            return false;
        }
        const int64_t host_end = ggml_time_us();

        float ms = 0.0f;
        if (!hip_ok(hipEventElapsedTime(&ms, start, stop), "hipEventElapsedTime")
                || !std::isfinite(ms) || ms <= 0.0f) {
            destroy_events();
            disable_smi_after_measurement_failure();
            return false;
        }
        const double us = (double) ms * 1000.0 / (double) launches_per_sample;
        if (!std::isfinite(us) || us <= 0.0) {
            destroy_events();
            disable_smi_after_measurement_failure();
            return false;
        }
        gpu_us.push_back(us);
        host_us.push_back((double) (host_end - host_start)
                          / (double) launches_per_sample);
        trace_workspace_event(stage, "timed_sample_end", candidate->stable_name);
    }

    const bool destroy_ok = destroy_events();
#ifdef GGML_HIP_WORKSPACE_METRICS
    if (pool_peak_bytes != nullptr && workspace_ctx != nullptr && isolate_workspace) {
        const size_t peak = workspace_ctx->pool().bc_workspace_peak_bytes();
        const size_t current = workspace_ctx->pool().bc_workspace_bytes();
        if (current != workspace_baseline) {
            GGML_LOG_WARN("bigcherry: workspace did not return to baseline "
                          "(%zu != %zu)\n", current, workspace_baseline);
            disable_smi_after_measurement_failure();
            return false;
        }
        *pool_peak_bytes = peak >= workspace_baseline ? peak - workspace_baseline : 0;
    }
#endif
    const hipError_t last_status = hipGetLastError();
    if (!destroy_ok || last_status != hipSuccess) {
        disable_smi_after_measurement_failure();
        return false;
    }
    return true;
}

// HI34: derive the shared launch batch from a one-launch-per-sample native
// pilot. Below min_sample_us a kernel is not resolvable against HIP event
// timer noise; batching several launches into one timed sample raises the
// floor without changing what is being measured.
int calibrated_launches_per_sample(double native_pilot_us,
                                   const ggml_hip_tuner_config & config) {
    if (!(native_pilot_us > 0.0) || !(config.min_sample_us > 0.0)) return 1;
    const int needed = (int) std::ceil(config.min_sample_us / native_pilot_us);
    return std::max(1, std::min(needed, config.max_launches_per_sample));
}

// HI52 part 2: serialise a device-state snapshot. Emits "{}" when the capture
// is off, unavailable, or could not resolve the device -- never a row of zeros,
// which would be indistinguishable from a genuinely idle, cold GPU.
std::string device_state_json(const ggml_hip_device_state & s) {
    if (!s.valid) return "{}";
    char buf[512];
    snprintf(buf, sizeof(buf),
             "{\"identity_valid\":%s,\"hip_device\":%d,"
             "\"pci_bdf\":\"%04x:%02x:%02x\","
             "\"sclk_mhz\":%llu,\"mclk_mhz\":%llu,\"edge_temp_mc\":%llu,"
             "\"junction_temp_mc\":%llu,\"socket_power_uw\":%llu,"
             "\"busy_percent\":%u}",
             s.identity_valid ? "true" : "false", s.hip_device,
             s.pci_domain, s.pci_bus, s.pci_device,
             (unsigned long long) s.sclk_mhz, (unsigned long long) s.mclk_mhz,
             (unsigned long long) s.edge_temp_mc,
             (unsigned long long) s.junction_temp_mc,
             (unsigned long long) s.socket_power_uw, s.busy_percent);
    return std::string(buf);
}

constexpr double k_clock_drift_threshold_pct = 5.0;

std::string device_clock_drift_json(const ggml_hip_device_state & pre,
                                    const ggml_hip_device_state & post) {
    if (!pre.valid || !post.valid || !pre.identity_valid || !post.identity_valid) {
        return "{\"status\":\"unavailable\"}";
    }
    if (pre.hip_device != post.hip_device || pre.pci_domain != post.pci_domain ||
            pre.pci_bus != post.pci_bus || pre.pci_device != post.pci_device) {
        return "{\"status\":\"identity_mismatch\"}";
    }
    if (pre.sclk_mhz == 0 || post.sclk_mhz == 0 || pre.mclk_mhz == 0 ||
            post.mclk_mhz == 0) {
        return "{\"status\":\"clock_unavailable\"}";
    }
    const double sclk_pct = 100.0 * std::abs((double) post.sclk_mhz -
                                             (double) pre.sclk_mhz) /
                            (double) pre.sclk_mhz;
    const double mclk_pct = 100.0 * std::abs((double) post.mclk_mhz -
                                             (double) pre.mclk_mhz) /
                            (double) pre.mclk_mhz;
    char buf[256];
    snprintf(buf, sizeof(buf),
             "{\"status\":\"captured\",\"sclk_delta_mhz\":%lld,"
             "\"mclk_delta_mhz\":%lld,\"max_abs_pct\":%.3f,"
             "\"drift\":%s,\"threshold_pct\":%.3f}",
             (long long) post.sclk_mhz - (long long) pre.sclk_mhz,
             (long long) post.mclk_mhz - (long long) pre.mclk_mhz,
             std::max(sclk_pct, mclk_pct),
             std::max(sclk_pct, mclk_pct) > k_clock_drift_threshold_pct ? "true" : "false",
             k_clock_drift_threshold_pct);
    return std::string(buf);
}

enum class RetimeStatus {
    not_needed,
    corrected,
    unresolved,
    unavailable,
};

const char * retime_status_name(RetimeStatus status) {
    switch (status) {
        case RetimeStatus::not_needed:  return "not_needed";
        case RetimeStatus::corrected:   return "corrected";
        case RetimeStatus::unresolved:  return "unresolved";
        case RetimeStatus::unavailable: return "unavailable";
    }
    return "unresolved";
}

struct ClockDriftObservation {
    bool comparable = false;
    bool identity_mismatch = false;
    bool drift = false;
    double max_abs_pct = 0.0;
};

ClockDriftObservation observe_clock_drift(const ggml_hip_device_state & pre,
                                          const ggml_hip_device_state & post) {
    ClockDriftObservation out;
    if (!pre.valid || !post.valid || !pre.identity_valid || !post.identity_valid) {
        return out;
    }
    if (pre.hip_device != post.hip_device || pre.pci_domain != post.pci_domain ||
            pre.pci_bus != post.pci_bus || pre.pci_device != post.pci_device) {
        out.identity_mismatch = true;
        return out;
    }
    if (pre.sclk_mhz == 0 || post.sclk_mhz == 0 || pre.mclk_mhz == 0 ||
            post.mclk_mhz == 0) {
        return out;
    }
    const double sclk_pct = 100.0 * std::abs((double) post.sclk_mhz -
                                             (double) pre.sclk_mhz) /
                            (double) pre.sclk_mhz;
    const double mclk_pct = 100.0 * std::abs((double) post.mclk_mhz -
                                             (double) pre.mclk_mhz) /
                            (double) pre.mclk_mhz;
    out.comparable = true;
    out.max_abs_pct = std::max(sclk_pct, mclk_pct);
    out.drift = out.max_abs_pct > k_clock_drift_threshold_pct;
    return out;
}

struct CounterbalancedRound {
    std::vector<double> gpu_us;
    std::vector<double> host_us;
    RetimeStatus status = RetimeStatus::not_needed;
    bool complete = false;
    bool clock_drift = false;
    bool reverse_attempted = false;
    bool reverse_passed = false;
};

// Measure one round in a deterministic order. When a comparable clock drift
// is observed, throw away the first order and retry the exact same candidates
// once in reverse order. Historical timings are never rescaled by a clock
// ratio: only a stable reverse retry can replace a drifted round.
// `flush_l2` is threaded explicitly rather than read from the config inside
// the measurement paths: every caller must state the eviction mode it wants,
// so a path that silently measures differently from its siblings becomes a
// compile error instead of a silent inconsistency (Slice B0 invariant).
CounterbalancedRound run_counterbalanced_round(
        const std::vector<Measurement *> & candidates,
        size_t offset, bool reverse,
        const ggml_hip_launch_context & lc,
        int launches_per_sample,
        ggml_hip_pre_sample_mode pre_sample,
        const char * protocol_stage) {
    CounterbalancedRound out;
    out.gpu_us.assign(candidates.size(), std::numeric_limits<double>::quiet_NaN());
    out.host_us.assign(candidates.size(), std::numeric_limits<double>::quiet_NaN());
    if (candidates.empty()) {
        out.complete = true;
        return out;
    }

    const bool smi_enabled = smi_capture_enabled();
    const int hip_device = smi_enabled ? ggml_cuda_get_device() : -1;
    auto run_order = [&](bool reverse_order, std::vector<double> & gpu,
                         std::vector<double> & host,
                         ggml_hip_device_state & pre,
                         ggml_hip_device_state & post) {
        gpu.assign(candidates.size(), std::numeric_limits<double>::quiet_NaN());
        host.assign(candidates.size(), std::numeric_limits<double>::quiet_NaN());
        pre = smi_enabled ? ggml_hip_query_device_state(hip_device)
                          : ggml_hip_device_state{};
        bool complete = true;
        for (size_t position = 0; position < candidates.size(); ++position) {
            const size_t index = reverse_order
                ? (offset + candidates.size() - 1 - position) % candidates.size()
                : (offset + position) % candidates.size();
            std::vector<double> one_gpu;
            std::vector<double> one_host;
            if (!time_candidate(candidates[index]->candidate, lc, 0, 1,
                                launches_per_sample, pre_sample,
                                one_gpu, one_host,
                                nullptr, false, nullptr, protocol_stage
#ifdef GGML_HIP_ROUTING_TRANSFORM
                                , candidates[index]->transform
#endif
                                ) ||
                    one_gpu.empty() || one_host.empty()) {
                complete = false;
                // A failed HIP launch may leave an asynchronous error pending
                // even after the timing transaction has rejected the sample.
                // Do not query SMI through ggml_cuda_get_device() while that
                // error is live, and do not launch another candidate in this
                // round. The round is discarded below.
                (void) hipGetLastError();
                break;
            }
            gpu[index] = one_gpu[0];
            host[index] = one_host[0];
        }
        post = (smi_enabled && complete) ? ggml_hip_query_device_state(hip_device)
                                         : ggml_hip_device_state{};
        return complete;
    };

    ggml_hip_device_state pre;
    ggml_hip_device_state post;
    const bool first_complete = run_order(reverse, out.gpu_us, out.host_us, pre, post);
    const ClockDriftObservation first_observation = observe_clock_drift(pre, post);
    if (!smi_enabled) {
        out.status = RetimeStatus::unavailable;
        out.complete = first_complete;
        return out;
    }
    if (!first_observation.comparable && !first_observation.identity_mismatch) {
        out.status = first_complete ? RetimeStatus::unavailable
                                    : RetimeStatus::not_needed;
        out.complete = first_complete;
        return out;
    }
    if (!first_observation.drift && !first_observation.identity_mismatch) {
        out.complete = first_complete;
        return out;
    }

    out.clock_drift = true;
    out.reverse_attempted = true;
    std::vector<double> reverse_gpu;
    std::vector<double> reverse_host;
    ggml_hip_device_state reverse_pre;
    ggml_hip_device_state reverse_post;
    const bool reverse_complete = run_order(!reverse, reverse_gpu, reverse_host,
                                             reverse_pre, reverse_post);
    const ClockDriftObservation reverse_observation =
        observe_clock_drift(reverse_pre, reverse_post);
    if (reverse_complete && reverse_observation.comparable &&
            !reverse_observation.drift && !reverse_observation.identity_mismatch) {
        out.gpu_us = std::move(reverse_gpu);
        out.host_us = std::move(reverse_host);
        out.status = RetimeStatus::corrected;
        out.complete = true;
        out.reverse_passed = true;
        return out;
    }
    out.gpu_us.assign(candidates.size(), std::numeric_limits<double>::quiet_NaN());
    out.host_us.assign(candidates.size(), std::numeric_limits<double>::quiet_NaN());
    out.status = RetimeStatus::unresolved;
    out.complete = false;
    return out;
}

void merge_retime_status(Result & result, RetimeStatus status) {
    if (status == RetimeStatus::unresolved) {
        result.retime_status = "unresolved";
    } else if (status == RetimeStatus::corrected && result.retime_status != "unresolved") {
        result.retime_status = "corrected";
    } else if (status == RetimeStatus::unavailable &&
            result.retime_status == "not_needed") {
        result.retime_status = "unavailable";
    }
}

void record_retime_observation(Result & result, const CounterbalancedRound & round) {
    if (!round.complete) result.measurement_failure = true;
    if (round.clock_drift) ++result.clock_drift_rounds;
    if (round.reverse_attempted) ++result.reverse_retime_attempts;
    if (round.reverse_passed) ++result.reverse_retime_passed;
    merge_retime_status(result, round.status);
}

// E3: the floor cost of one timed sample with no work in it -- two event
// records, a synchronize, and the two host clock reads. Subtracting it is
// what makes a host-side number comparable with a GPU-side one. Measured
// once per DEVICE for the process lifetime, not once per process (bigcherry
// HI64: a value measured on one GPU is not this file's business to hand out
// as another GPU's overhead, and keeping it process-global left exactly the
// kind of cross-device collateral GPT's poison-scoping review flagged --
// requirement 5, "make SMI-disable state and the cached host-sync-overhead
// device-local too"). Cached at namespace scope (not function-local) so
// `ggml_hip_tuner_flush` can report it in the measurements header for audit.
struct HostSyncOverheadCache {
    double us = -1.0;
    bool valid = false;
};
PerDeviceState<HostSyncOverheadCache> g_host_sync_overhead(HostSyncOverheadCache{});

double host_sync_overhead_us(cudaStream_t stream) {
    if (g_tuner_poisoned.load(std::memory_order_relaxed)) {
        return 0.0;
    }
    HostSyncOverheadCache cached = g_host_sync_overhead.load();
    if (cached.valid && cached.us >= 0.0) {
        return cached.us;
    }
    g_host_sync_overhead.store(HostSyncOverheadCache{});

    hipEvent_t a = nullptr;
    hipEvent_t b = nullptr;
    auto destroy_events = [&]() {
        const bool a_ok = hip_event_destroy_checked(a, "hipEventDestroy(overhead start)");
        const bool b_ok = hip_event_destroy_checked(b, "hipEventDestroy(overhead stop)");
        return a_ok && b_ok;
    };
    auto fail_measurement = [&]() {
        disable_smi_after_measurement_failure();
        g_host_sync_overhead.store(HostSyncOverheadCache{});
        return 0.0;
    };
    if (!hip_ok(hipEventCreate(&a), "hipEventCreate(overhead start)")) {
        return fail_measurement();
    }
    if (!hip_ok(hipEventCreate(&b), "hipEventCreate(overhead stop)")) {
        (void) destroy_events();
        return fail_measurement();
    }
    std::vector<double> samples;
    for (int i = 0; i < 20; ++i) {
        const int64_t t0 = ggml_time_us();
        if (!hip_ok(hipEventRecord(a, stream), "hipEventRecord(overhead start)")
                || !hip_ok(hipEventRecord(b, stream), "hipEventRecord(overhead stop)")
                || !hip_ok(hipEventSynchronize(b), "hipEventSynchronize(overhead)")) {
            (void) destroy_events();
            return fail_measurement();
        }
        samples.push_back((double) (ggml_time_us() - t0));
    }
    const bool destroy_ok = destroy_events();
    if (!destroy_ok || samples.empty()) {
        return fail_measurement();
    }
    const double overhead = median_of(samples);
    if (!std::isfinite(overhead) || overhead < 0.0) {
        return fail_measurement();
    }
    g_host_sync_overhead.store(HostSyncOverheadCache{overhead, true});
    return overhead;
}

// E3: max(), not a mode switch -- whichever resource is binding is the one
// that bounds the call, and the other term is smaller by construction. When
// the host runs ahead of the GPU (large kernels, deep queue) the second term
// is smaller and the GPU number wins; when submission dominates (tiny
// kernels) the host term wins.
double effective_us_of(double median_us, double host_median_us, double sync_overhead_us) {
    return std::max(median_us, host_median_us - sync_overhead_us);
}

// E4: one launch, then fetch the result to host. Used only for the winner's
// determinism recheck -- two launches and two copies, not the full timing
// harness above.
bool launch_and_fetch(const ggml_hip_candidate_descriptor * candidate,
                      const ggml_hip_launch_context & lc,
                      void * dst_device, size_t dst_bytes,
                      std::vector<float> & out_host
#ifdef GGML_HIP_ROUTING_TRANSFORM
                      , const ggml_hip_routing_transformation * transform = nullptr
#endif
                      ) {
    if (g_tuner_poisoned.load(std::memory_order_relaxed)) {
        return false;
    }
    ggml_hip_candidate_descriptor effective = *candidate;
#ifdef GGML_HIP_ROUTING_TRANSFORM
    if (transform != nullptr) {
        ggml_hip_transform_ctx xform_ctx;
        ggml_hip_launch_context out_lc;
        if (!transform->apply(lc, &xform_ctx, nullptr, &out_lc)) {
            disable_smi_after_measurement_failure();
            return false;
        }
        ggml_hip_transform_launch(transform, &effective, effective.variant,
                                  &xform_ctx, out_lc);
    } else {
        effective.launch(&effective, lc);
    }
#else
    effective.launch(&effective, lc);
#endif
    if (hipStreamSynchronize(lc.stream) != hipSuccess) {
        disable_smi_after_measurement_failure();
        return false;
    }
    if (hipGetLastError() != hipSuccess) {
        disable_smi_after_measurement_failure();
        return false;
    }
    if (hipMemcpy(out_host.data(), dst_device, dst_bytes,
                  hipMemcpyDeviceToHost) != hipSuccess) {
        disable_smi_after_measurement_failure();
        return false;
    }
    return true;
}

// E2: raw finalist samples as a JSON array, so a winner is recomputable
// offline without a GPU. NaN (a round where the candidate failed to launch)
// serialises as `null` -- JSON has no NaN literal, and dropping the entry
// entirely would silently reintroduce the round-misalignment bug E1 fixes.
//
// RE28: six decimal places, not three. The paired sign test that produces
// confirmation_wins/confirmation_rounds runs at full double precision; an
// offline reader (tune_promotion.py's _paired_rounds()) deliberately
// recomputes the same tie/win test from these persisted values rather than
// trusting the declared counts blindly. At %.3f, two rounds whose true
// timings differ by under half a nanosecond-scale microsecond fraction can
// round to the identical displayed value, turning a real (non-tied) round
// into an apparent tie on replay and failing that cross-check even though
// the producer's own accounting was correct. Six decimals keeps collisions
// below any timing precision this measurement stack can actually resolve.
std::string samples_json(const std::vector<double> & v) {
    std::string out = "[";
    for (size_t i = 0; i < v.size(); ++i) {
        if (i) out += ",";
        if (std::isnan(v[i])) {
            out += "null";
        } else {
            char buf[32];
            snprintf(buf, sizeof(buf), "%.6f", v[i]);
            out += buf;
        }
    }
    out += "]";
    return out;
}

// HI34: the finalist-name schedule an offline reader (tune_promotion.py)
// re-derives from schedule_seed to verify this run's confirmation rounds
// actually used the seed it claims, rather than trusting the claim.
std::string string_array_json(const std::vector<std::string> & v) {
    std::string out = "[";
    for (size_t i = 0; i < v.size(); ++i) {
        if (i) out += ",";
        out += "\"" + v[i] + "\"";
    }
    out += "]";
    return out;
}

} // namespace

const ggml_hip_tuner_config & ggml_hip_tuner_get_config() {
    static ggml_hip_tuner_config config = [] {
        ggml_hip_tuner_config c;
        auto int_env = [&](const char * name, int min_v, int max_v, int & out) {
            const char * v = getenv(name); if (!v) return;
            if (*v == '\0' || std::isspace((unsigned char) *v)) {
                GGML_LOG_WARN("bigcherry: invalid %s=%s (whitespace/empty value)\n", name, v);
                c.valid = false; return;
            }
            errno = 0; char * end = nullptr; const long parsed = strtol(v, &end, 10);
            if (errno || end == v || *end != '\0' || parsed < min_v || parsed > max_v) {
                GGML_LOG_WARN("bigcherry: invalid %s=%s (expected %d..%d)\n", name, v, min_v, max_v);
                c.valid = false; return;
            }
            out = (int) parsed;
        };
        auto size_env = [&](const char * name, size_t & out) {
            const char * v = getenv(name); if (!v) return;
            if (*v == '\0' || std::isspace((unsigned char) *v)) {
                GGML_LOG_WARN("bigcherry: invalid %s=%s (whitespace/empty value)\n", name, v);
                c.valid = false; return;
            }
            errno = 0; char * end = nullptr; const unsigned long long parsed = strtoull(v, &end, 10);
            if (errno || end == v || *end != '\0' || v[0] == '-'
                    || parsed > (unsigned long long) std::numeric_limits<size_t>::max()) {
                GGML_LOG_WARN("bigcherry: invalid %s=%s\n", name, v); c.valid = false; return;
            }
            out = (size_t) parsed;
        };
        auto double_env = [&](const char * name, double min_v, double max_v, double & out) {
            const char * v = getenv(name); if (!v) return;
            if (*v == '\0' || std::isspace((unsigned char) *v)) {
                GGML_LOG_WARN("bigcherry: invalid %s=%s (whitespace/empty value)\n", name, v);
                c.valid = false; return;
            }
            errno = 0; char * end = nullptr; const double parsed = strtod(v, &end);
            if (errno || end == v || *end != '\0' || !std::isfinite(parsed)
                    || parsed <= min_v || parsed >= max_v) {
                GGML_LOG_WARN("bigcherry: invalid %s=%s\n", name, v); c.valid = false; return;
            }
            out = parsed;
        };
        // Environment overrides exist so a long production tune can be traded
        // against precision without a rebuild.
        if (const char * v = getenv("GGML_HIP_TUNE_FINAL_SAMPLES")) {
            int_env("GGML_HIP_TUNE_FINAL_SAMPLES", 2, 100000, c.final_samples);
        }
        if (const char * v = getenv("GGML_HIP_TUNE_SCREEN_SAMPLES")) {
            int_env("GGML_HIP_TUNE_SCREEN_SAMPLES", 1, 100000, c.screen_samples);
        }
        if (const char * v = getenv("GGML_HIP_TUNE_MAX_WORKSPACE")) {
            size_env("GGML_HIP_TUNE_MAX_WORKSPACE", c.max_workspace_bytes);
        }
        if (const char * v = getenv("GGML_HIP_TUNE_NOISE_PCT")) {
            double_env("GGML_HIP_TUNE_NOISE_PCT", 0.0, std::numeric_limits<double>::max(), c.noise_canary_pct);
        }
        if (const char * v = getenv("GGML_HIP_TUNE_DOUBLE_NATIVE")) {
            int_env("GGML_HIP_TUNE_DOUBLE_NATIVE", 0, 1, c.double_native);
        }
        if (const char * v = getenv("GGML_HIP_TUNE_ALPHA")) {
            double_env("GGML_HIP_TUNE_ALPHA", 0.0, 1.0, c.confidence_alpha);
        }
        if (const char * v = getenv("GGML_HIP_TUNE_NOISY_MAD")) {
            double_env("GGML_HIP_TUNE_NOISY_MAD", 0.0, std::numeric_limits<double>::max(), c.noisy_mad_ratio);
        }
        if (const char * v = getenv("GGML_HIP_TUNE_VERIFY_DETERMINISM")) {
            int_env("GGML_HIP_TUNE_VERIFY_DETERMINISM", 0, 1, c.verify_determinism);
        }
        if (const char * v = getenv("GGML_HIP_TUNE_EMIT_SAMPLES")) {
            int_env("GGML_HIP_TUNE_EMIT_SAMPLES", 0, 1, c.emit_samples);
        }
        if (const char * v = getenv("GGML_HIP_TUNE_PILOT_SAMPLES")) {
            int_env("GGML_HIP_TUNE_PILOT_SAMPLES", 1, 100000, c.pilot_samples);
        }
        if (const char * v = getenv("GGML_HIP_TUNE_MIN_SAMPLE_US")) {
            double_env("GGML_HIP_TUNE_MIN_SAMPLE_US", 0.0, std::numeric_limits<double>::max(), c.min_sample_us);
        }
        if (const char * v = getenv("GGML_HIP_TUNE_MAX_LPS")) {
            int_env("GGML_HIP_TUNE_MAX_LPS", 1, 100000, c.max_launches_per_sample);
        }
        if (const char * v = getenv("GGML_HIP_TUNE_CONFIRM_SAMPLES")) {
            int_env("GGML_HIP_TUNE_CONFIRM_SAMPLES", 2, 100000, c.confirmation_samples);
        }
        int flush_l2_req = 0;
        int flush_rewarm_req = 0;
        if (const char * v = getenv("GGML_HIP_TUNE_FLUSH_L2")) {
            int_env("GGML_HIP_TUNE_FLUSH_L2", 0, 1, flush_l2_req);
        }
        if (const char * v = getenv("GGML_HIP_TUNE_FLUSH_REWARM")) {
            int_env("GGML_HIP_TUNE_FLUSH_REWARM", 0, 1, flush_rewarm_req);
        }
        if (const char * v = getenv("GGML_HIP_TUNE_FLUSH_MB")) {
            int_env("GGML_HIP_TUNE_FLUSH_MB", 1, 65536, c.flush_evict_mb);
        }
        // HI65: resolve the two request flags into the single mode. Both set
        // is a contradictory configuration (two different post-eviction
        // states cannot both be \"the\" state) and fails closed at parse time.
        if (flush_l2_req != 0 && flush_rewarm_req != 0) {
            c.valid = false;
            GGML_LOG_WARN("bigcherry: GGML_HIP_TUNE_FLUSH_L2 and "
                          "GGML_HIP_TUNE_FLUSH_REWARM are mutually exclusive; "
                          "tuning disabled\n");
        } else if (flush_rewarm_req != 0) {
            c.pre_sample_mode = GGML_HIP_PRE_SAMPLE_EVICT_REWARM;
        } else if (flush_l2_req != 0) {
            c.pre_sample_mode = GGML_HIP_PRE_SAMPLE_EVICT;
        }
        // Wire-format mirror for artifact emission and backward compatibility;
        // measurement code branches on pre_sample_mode, never on this field.
        c.flush_l2 = (c.pre_sample_mode != GGML_HIP_PRE_SAMPLE_NONE) ? 1 : 0;
        if (c.pre_sample_mode != GGML_HIP_PRE_SAMPLE_NONE
                && c.max_launches_per_sample > 1) {
            // A flush cannot reach inside a batch: a batched sample would
            // measure one cold launch plus lps-1 hot launches and report the
            // mean as if it were one number. Enforced here, where every
            // downstream calibration reads its ceiling, rather than at each
            // measurement site where a missed thread-through would silently
            // coexist flush and batching (Slice B0 invariant).
            GGML_LOG_WARN("bigcherry: cache-eviction pre-sample mode forces "
                          "max_launches_per_sample=1 (was %d); a flush cannot "
                          "reach inside a batch\n", c.max_launches_per_sample);
            c.max_launches_per_sample = 1;
        }
        if (const char * v = getenv("GGML_HIP_TUNE_PRODUCTION_POLICY")) {
            c.production_policy = v;
        }
        if (const char * v = getenv("GGML_HIP_TUNE_ACTIVE_POLICIES")) {
            c.active_policies = v;
        }
        if (!c.valid) GGML_LOG_WARN("bigcherry: invalid tuning configuration; tuning disabled\n");
        return c;
    }();
    return config;
}

// --- HI50: ranking-policy table -----------------------------------------
//
// Pure CPU-side ranking over already-measured finalists, no extra GPU
// launches, no mutation beyond the per-candidate sign-test bookkeeping
// every policy in this file has always recorded. That purity is what makes
// shadow-evaluating several policies on every tuning pass safe -- it
// changes nothing about what gets dispatched, confirmed, or promoted.
// Exactly one policy (config.production_policy) is ever plugged into the
// determinism recheck and confirmation holdout; every other active
// policy's output is an unconfirmed prediction, recorded for offline
// comparison and nothing else.

// Per-candidate outcome under one policy's ranking, for the JSON report --
// deliberately covers every finalist a policy considered, not only the one
// it picked, so a rejected or near-tie candidate stays visible afterward.
struct PolicyCandidateVerdict {
    Measurement * m = nullptr;
    std::string   verdict;            // winner|qualified|near_tie_below_threshold|outside_tie_band|not_attempted|rejected
    std::string   rejection_reason;   // set for "rejected" (reason_name) and "not_attempted" (why ranking never ran)
};

struct PolicySelection {
    // Empty means this policy retained native (no challenger qualified, or
    // ranking never ran -- see any_short_rounds/ranked for why). front() is
    // the pick, if any.
    std::vector<Measurement *> qualified;
    bool any_short_rounds = false;
    std::vector<PolicyCandidateVerdict> ranked;
};

// Mirrors the winner-selection algorithm this file has always run
// (standards 7.3, HI12 E1): rank finalists by effective_us, keep everything
// within tie_pct of the best as a near-tie set (native included only if it
// lands in that band), require replacement_threshold_pct improvement to
// qualify, and break ties by (p95_us, workspace_bytes, native-preference,
// stable_name). Selection only nominates by effect size; statistical
// significance is a separate, disjoint question answered by the fresh
// confirmation holdout below (HI34) -- not folded into selection, so the
// same evidence never both picks a candidate and promotes it.
static PolicySelection select_latency_v1(
        const std::vector<Measurement *> & finalists,
        Measurement *                      native_m,
        ggml_hip_canary_state_v1           canary_state,
        const ggml_hip_tuner_config &      config) {
    PolicySelection out;

    Measurement * best_m = native_m;
    for (Measurement * m : finalists) {
        if (!m->is_native_twin && m->measured && m->reason == GGML_HIP_REJECT_NONE &&
                m->effective_us < best_m->effective_us) {
            best_m = m;
        }
    }

    if (best_m == native_m || canary_state == GGML_HIP_CANARY_UNRESOLVED) {
        // Nothing beat native at all, or a beat is unconfirmed noise -- no
        // tied/qualified set is built in this case, while still reporting a
        // verdict per finalist.
        const char * why = canary_state == GGML_HIP_CANARY_UNRESOLVED
            ? "canary_unresolved" : "native_not_beaten";
        for (Measurement * m : finalists) {
            if (m->is_native_twin) continue;
            if (m == native_m) {
                out.ranked.push_back({m, "winner", ""});
                continue;
            }
            if (!m->measured || m->reason != GGML_HIP_REJECT_NONE) {
                out.ranked.push_back({m, "rejected", reason_name(m->reason)});
            } else {
                out.ranked.push_back({m, "not_attempted", why});
            }
        }
        return out;
    }

    std::vector<Measurement *> tied;
    for (Measurement * m : finalists) {
        if (m->is_native_twin) continue;
        if (!m->measured || m->reason != GGML_HIP_REJECT_NONE) {
            out.ranked.push_back({m, "rejected", reason_name(m->reason)});
            continue;
        }
        if (m->effective_us <= best_m->effective_us * (1.0 + config.tie_pct / 100.0)) {
            tied.push_back(m);
        } else {
            out.ranked.push_back({m, "outside_tie_band", ""});
        }
    }

    auto tie_order = [&](const Measurement * a, const Measurement * b) {
        if (a->p95_us != b->p95_us) return a->p95_us < b->p95_us;
        if (a->workspace_bytes != b->workspace_bytes) {
            return a->workspace_bytes < b->workspace_bytes;
        }
        const bool an = a->candidate == native_m->candidate;
        const bool bn = b->candidate == native_m->candidate;
        if (an != bn) return an;   // native wins a genuine tie
        return strcmp(a->candidate->stable_name,
                      b->candidate->stable_name) < 0;
    };

    for (Measurement * m : tied) {
        if (m == native_m) {
            out.qualified.push_back(m);
            out.ranked.push_back({m, "qualified", ""});
            continue;
        }
        int wins = 0, rounds = 0;
        m->sign_p = paired_sign_test(native_m->final_gpu_us, m->final_gpu_us,
                                     config.min_paired_rounds, wins, rounds);
        m->sign_wins   = wins;
        m->sign_rounds = rounds;
        if (rounds < config.min_paired_rounds) out.any_short_rounds = true;
        const double impr = native_m->effective_us > 0.0
            ? 100.0 * (native_m->effective_us - m->effective_us) / native_m->effective_us
            : 0.0;
        if (impr >= config.replacement_threshold_pct) {
            out.qualified.push_back(m);
            out.ranked.push_back({m, "qualified", ""});
        } else {
            out.ranked.push_back({m, "near_tie_below_threshold", ""});
        }
    }
    std::sort(out.qualified.begin(), out.qualified.end(), tie_order);
    if (!out.qualified.empty()) {
        for (auto & rv : out.ranked) {
            if (rv.m == out.qualified.front()) { rv.verdict = "winner"; break; }
        }
    }
    return out;
}

struct PolicyTableEntry {
    const char * name;
    int          version;
    PolicySelection (*fn)(const std::vector<Measurement *> &, Measurement *,
                          ggml_hip_canary_state_v1, const ggml_hip_tuner_config &);
};

// Compiled-in policies. Adding a policy is adding one entry here; nothing
// else in resolve()/flush() changes.
const PolicyTableEntry g_policy_table[] = {
    { "latency-v1", 1, select_latency_v1 },
};
constexpr size_t g_policy_table_size =
    sizeof(g_policy_table) / sizeof(g_policy_table[0]);

// Which table entry governs real dispatch. An unrecognized production_policy
// is invalid and must fail closed before any measurement is performed.
static size_t resolve_production_policy_index(const ggml_hip_tuner_config & config) {
    for (size_t i = 0; i < g_policy_table_size; ++i) {
        if (config.production_policy == g_policy_table[i].name) return i;
    }
    return g_policy_table_size;
}

static bool policy_name_is_active(const std::string & name, const std::string & active_policies) {
    if (active_policies == "all") return true;
    size_t start = 0;
    while (start <= active_policies.size()) {
        const size_t comma = active_policies.find(',', start);
        const std::string tok = active_policies.substr(
            start, comma == std::string::npos ? std::string::npos : comma - start);
        if (tok == name) return true;
        if (comma == std::string::npos) break;
        start = comma + 1;
    }
    return false;
}

static bool policy_list_is_valid(const std::string & active_policies) {
    if (active_policies == "all") return true;
    if (active_policies.empty()) return false;
    std::unordered_set<std::string> seen;
    size_t start = 0;
    while (start <= active_policies.size()) {
        const size_t comma = active_policies.find(',', start);
        const std::string tok = active_policies.substr(
            start, comma == std::string::npos ? std::string::npos : comma - start);
        if (tok.empty()) return false;
        bool known = false;
        for (size_t i = 0; i < g_policy_table_size; ++i) {
            if (tok == g_policy_table[i].name) { known = true; break; }
        }
        if (!known) return false;
        if (!seen.insert(tok).second) return false;
        if (comma == std::string::npos) break;
        start = comma + 1;
    }
    return true;
}

static std::string policy_candidates_json(const std::vector<PolicyCandidateVerdict> & ranked) {
    std::string out = "[";
    bool first = true;
    for (const auto & rv : ranked) {
        if (rv.m == nullptr) continue;
        if (!first) out += ",";
        first = false;
        const std::string emitted_name = measurement_name(*rv.m);
        char eff_buf[32];
        snprintf(eff_buf, sizeof(eff_buf), "%.3f", rv.m->effective_us);
        out += "{\"name\":\"";
        out += emitted_name;
        out += "\",\"effective_us\":";
        out += eff_buf;
        out += ",\"verdict\":\"";
        out += rv.verdict;
        out += "\",\"rejection_reason\":\"";
        out += rv.rejection_reason;
        out += "\"}";
    }
    out += "]";
    return out;
}

static std::string ranking_decision_json(const PolicyTableEntry & entry,
                                  const PolicySelection & sel,
                                  Measurement * native_m, bool is_production) {
    Measurement * picked = sel.qualified.empty() ? native_m : sel.qualified.front();
    const std::string predicted_name = measurement_name(*picked);
    std::string out = "{\"policy_name\":\"";
    out += entry.name;
    out += "\",\"policy_version\":";
    out += std::to_string(entry.version);
    out += ",\"is_production\":";
    out += is_production ? "true" : "false";
    out += ",\"predicted_winner\":\"";
    out += predicted_name;
    out += "\",\"candidates\":";
    out += policy_candidates_json(sel.ranked);
    out += "}";
    return out;
}

// HI64 (2026-08-22): renamed from the old public ggml_hip_tuner_resolve --
// this is now a file-local implementation detail. The public entry point
// below wraps this to additionally report whether the resolution ended in
// a device-local measurement failure, without touching any of this
// function's ~25 existing `return native.candidate;`/`return
// result.winner;` exit sites: every one of them already goes through
// record_result() first (or, for the cache-hit path just below, reuses an
// entry record_result() already wrote on a prior call), so a second,
// cheap g_results lookup after this function returns is guaranteed to see
// the correct, just-recorded measurement_failure value for this exact
// dispatch_digest.
static const ggml_hip_candidate_descriptor * ggml_hip_tuner_resolve_impl(
        ggml_backend_cuda_context & ctx,
        const ggml_hip_dispatch_signature_v1 & sig,
        const ggml_hip_hardware_key_v1 & hw,
        const ggml_hip_digest & dispatch_digest,
        const ggml_hip_native_selection & native,
        const ggml_hip_launch_context & lc_in) {
    // HI64 (2026-08-22): g_single_flight_mutex is held by the CALLER
    // (the public ggml_hip_tuner_resolve() wrapper below), across this
    // entire call AND its own post-call g_results re-check -- not
    // acquired in here. Locking only around this function would let a
    // second thread's record_result() replace this dispatch_digest's
    // entry (failure -> success) in the window between this function
    // returning and the wrapper reading measurement_failure back out,
    // making the wrapper report a stale/wrong value for what actually
    // happened on THIS invocation. The lock must still cover both the
    // cache lookup and the complete cold path -- that part of the
    // original rationale is unchanged, it just now starts one frame up.
    {
        std::lock_guard<std::mutex> lock(g_mutex);
        const auto found = g_results.find(dispatch_digest);
        // HI64: a failed/poisoned result is device-local evidence, not a
        // portable outcome -- only a non-failed result may short-circuit
        // resolution here. Otherwise one device's fatal measurement
        // failure permanently blocks every other device (including an
        // identical twin GPU) sharing this portable dispatch key from ever
        // getting its own attempt. See record_result()'s matching
        // failure->success replacement for the other half of this fix.
        if (found != g_results.end() && !found->second.measurement_failure) {
            return found->second.winner;
        }
    }

    if (g_tuner_poisoned.load(std::memory_order_relaxed)) {
        Result poisoned;
        poisoned.winner = native.candidate;
        poisoned.native_name = native.candidate ? native.candidate->stable_name : "";
        poisoned.signature_digest = ggml_hip_signature_digest(sig);
        poisoned.hardware_digest = ggml_hip_hardware_digest(hw);
        // The stub skips the cold-path setup below; canonical_json has no
        // valid default, and an empty value here used to serialize as
        // "canonical":, -- invalid JSON in every row written after a fatal
        // failure. Set it from the signature that is already in scope.
        poisoned.canonical_json = ggml_hip_signature_json(sig, true);
        poisoned.reason = "tuning disabled after fatal measurement failure";
        poisoned.measurement_failure = true;
        open_tuning_journal_once(poisoned.hardware_digest);
        record_result(dispatch_digest, poisoned);
        return native.candidate;
    }

    const ggml_hip_tuner_config & config = ggml_hip_tuner_get_config();
    const size_t resolved_production_index = resolve_production_policy_index(config);
    const bool active_policies_valid =
        resolved_production_index < g_policy_table_size
        && policy_list_is_valid(config.active_policies)
        && policy_name_is_active(g_policy_table[resolved_production_index].name,
                                 config.active_policies);
    if (!config.valid || resolved_production_index == g_policy_table_size || !active_policies_valid) {
        GGML_LOG_WARN("bigcherry: invalid tuning policy/configuration; tuning disabled\n");
        Result invalid;
        invalid.winner = native.candidate;
        invalid.native_name = native.candidate ? native.candidate->stable_name : "";
        // Same serialization guard as the poisoned stub above: every field
        // the flush prints must be valid JSON, and this path also bypasses
        // the cold-path setup that would otherwise fill these in.
        invalid.signature_digest = ggml_hip_signature_digest(sig);
        invalid.hardware_digest = ggml_hip_hardware_digest(hw);
        invalid.canonical_json = ggml_hip_signature_json(sig, true);
        invalid.reason = "invalid tuning configuration";
        record_result(dispatch_digest, invalid);
        return native.candidate;
    }
    Result result;
    result.winner = native.candidate;
    result.native_name = native.candidate ? native.candidate->stable_name : "";
    // Default provisional_winner to native up front, once, rather than at
    // each of the (many, growing) early-return sites below: every exit that
    // returns before the ranking-stage pick (further down) inherits the
    // correct value for free instead of needing its own assignment. Exits
    // AFTER a real ranking pick has been made (confirmation/retime failures)
    // must NOT let this default stand uncorrected -- see the comments at
    // those sites for why overwriting a nominated challenger's identity back
    // to native is itself a bug, not a fix.
    result.provisional_winner = result.native_name;
    // Recomputed rather than threaded down from the resolver: the two are the
    // same values, and recomputing here keeps this function's signature stable
    // for a field the caller has no other use for. Cold path, once per
    // signature.
    result.signature_digest = ggml_hip_signature_digest(sig);
    result.hardware_digest  = ggml_hip_hardware_digest(hw);
    result.canonical_json   = ggml_hip_signature_json(sig, true);
    g_trace_signature_digest = result.signature_digest;
    g_trace_signature = sig;
    open_tuning_journal_once(result.hardware_digest);

    // Held for the whole run, not just around each launch.
    //
    // Every measurement launch goes through a family entry point, and HI13
    // made those entry points collection sites. Without this the first
    // measured launch re-enters the hook, which resolves, which calls the
    // tuner, which launches again -- unbounded recursion, and the stack dies
    // inside the HSA runtime where no bigcherry frame is visible. That is
    // exactly how this was found.
    const ggml_hip_dispatch_scope no_reentry;

    // Runs once per dispatch key, inline, on the first encounter. It stalls
    // that one execution; the resolver's process cache makes every later one
    // free. Inline rather than on a background thread so the measured launches
    // sit on the same stream and device as the real work -- which is the only
    // way the numbers describe production rather than a laboratory.

    // --- gather the candidate set ---------------------------------------
    //
    // Whether other families compete depends on whether this operation is
    // fused, and the two rules pull in opposite directions:
    //
    //   plan 11.3   -- for an ordinary signature, include every legal family
    //                  default and BLAS auto. Upstream picks the family from a
    //                  heuristic ladder (`should_use_mmq` and friends), and
    //                  measuring whether that heuristic is right for a given
    //                  shape is the single biggest lever this project has.
    //                  Restricting to native's family throws it away.
    //
    //   standards 11.1 -- for a *fused* pattern, tune only within the family
    //                  already selected. A fused MMVQ is a different semantic
    //                  operation, and comparing it against an unfused MMQ
    //                  decomposition is a graph-level question that the matmul
    //                  tuner is not entitled to answer.
    //
    // So: cross-family for unfused operations, single-family for fused ones.
    const bool fused = sig.fusion != GGML_HIP_FUSION_NONE;

    for (size_t i = 0; i < ggml_hip_registry_size(); ++i) {
        const ggml_hip_candidate_descriptor * candidate = ggml_hip_registry_at(i);

        if (fused && candidate->family != native.candidate->family) {
            continue;
        }
        ++result.generated;

        Measurement m;
        m.candidate = candidate;

        const bool arch_ok = ggml_hip_candidate_supports_arch(*candidate, hw);

        // E5: a candidate is *applicable* when it could plausibly serve this
        // signature at all: right architecture, and built for this src0 type.
        // Without this counter `generated` is the registry size and never
        // varies, which is why every signature in TUNING-DETAIL.md reports
        // 270 and why COVERAGE_AUDIT's "generated high, eligible low"
        // diagnostic can never fire. variant.src0_type is 0 for native
        // wrappers, which are admitted unconditionally (hip-autotune-types.h:197).
        const bool type_ok = candidate->variant.src0_type == 0
                          || candidate->variant.src0_type == sig.src0_type;
        if (arch_ok && type_ok) {
            ++result.applicable;
        }

        if (!arch_ok) {
            m.reason = GGML_HIP_REJECT_ARCHITECTURE;
        } else if (!candidate->can_execute(candidate, sig, hw)) {
            m.reason = GGML_HIP_REJECT_INELIGIBLE;
        } else {
            m.workspace_bytes = candidate->workspace(candidate, sig);
            if (config.max_workspace_bytes != 0
                    && m.workspace_bytes > config.max_workspace_bytes) {
                m.reason = GGML_HIP_REJECT_WORKSPACE;
            } else {
                ++result.eligible;
            }
        }
        result.measurements.push_back(m);
    }

    // HI24 step 4. The J_best canary only applies where native is MMQ -- a
    // minority of signatures -- so every other family records canary_pct =
    // -1, which is exactly the set deciding the finest margins. A second
    // Measurement over native's own descriptor gives every family a
    // same-kernel repeatability check by construction: any divergence between
    // the two rows is measurement error.
    //
    // Location is load-bearing: `screening` below holds raw pointers into
    // result.measurements, so the twin must be pushed BEFORE that vector is
    // built or a later reallocation would dangle every one of them. At this
    // point native has not been timed yet, so the copy's timing fields are
    // pristine; nothing in the registry loop writes transient state.
    if (config.double_native) {
        for (const Measurement & m : result.measurements) {
            if (m.candidate == native.candidate &&
                    m.reason == GGML_HIP_REJECT_NONE &&
                    !m.is_native_twin) {
                Measurement twin = m;
                twin.is_native_twin = true;
                result.measurements.push_back(std::move(twin));
                break;
            }
        }
    }

#ifdef GGML_HIP_ROUTING_TRANSFORM
    // HI30: routing transformations compete only when nothing already
    // reached this signature by the direct route (2026-08-21 adjudication).
    // "Reached" means a real, unmodified candidate screened in above --
    // BLAS is excluded from that set deliberately: BLAS auto-selection is
    // upstream's fallback of last resort, not evidence a fast family already
    // fits, and is exactly the case these transforms exist to challenge.
    // Fused patterns are excluded entirely (standards 11.1, same rule the
    // registry loop above already applies to direct candidates).
    if (!fused) {
        std::unordered_set<int> direct_eligible_families;
        for (const Measurement & m : result.measurements) {
            if (m.reason == GGML_HIP_REJECT_NONE) {
                direct_eligible_families.insert((int) m.candidate->family);
            }
        }
        const bool blas_only = direct_eligible_families.size() == 1 &&
            direct_eligible_families.count((int) GGML_HIP_FAMILY_BLAS) != 0;
        const bool transform_trigger = direct_eligible_families.empty() || blas_only;

        if (transform_trigger) {
            for (int t = 0; t < ggml_hip_transform_count(); ++t) {
                const ggml_hip_routing_transformation * transform = ggml_hip_transform_at(t);
                if (transform == nullptr || !transform->equivalence_verified) {
                    continue;
                }
                if (!ggml_hip_transform_signature_is_eligible(sig) ||
                        !transform->can_apply(sig)) {
                    result.transform_scan_gaps.push_back(
                        {transform->id, "signature not eligible for this transform"});
                    continue;
                }

                // Apply once against the real launch context to get the exact
                // transformed signature every target candidate below is
                // screened against -- the central HI30 invariant: what is
                // scanned here is the same transformed shape that will later
                // be measured through ggml_hip_transform_launch().
                ggml_hip_transform_ctx   probe_ctx;
                ggml_hip_dispatch_signature_v1 out_sig{};
                ggml_hip_launch_context  probe_lc{};
                if (!transform->apply(lc_in, &probe_ctx, &out_sig, &probe_lc)) {
                    result.transform_scan_gaps.push_back(
                        {transform->id, "transform apply() failed"});
                    continue;
                }

                bool any_target_eligible = false;
                for (size_t i = 0; i < ggml_hip_registry_size(); ++i) {
                    const ggml_hip_candidate_descriptor * candidate = ggml_hip_registry_at(i);
                    if (candidate->family != transform->target_family) {
                        continue;
                    }
                    if (!ggml_hip_candidate_supports_arch(*candidate, hw)) {
                        continue;
                    }
                    const bool type_ok = candidate->variant.src0_type == 0
                                      || candidate->variant.src0_type == out_sig.src0_type;
                    if (!type_ok || !candidate->can_execute(candidate, out_sig, hw)) {
                        continue;
                    }

                    const size_t candidate_bytes = candidate->workspace(candidate, out_sig);
                    const size_t overhead_bytes  = transform->overhead_bytes(sig);
                    if (candidate_bytes > std::numeric_limits<size_t>::max() - overhead_bytes) {
                        continue;   // overflow; not eligible rather than wrap
                    }
                    const size_t workspace_bytes = candidate_bytes + overhead_bytes;
                    if (config.max_workspace_bytes != 0
                            && workspace_bytes > config.max_workspace_bytes) {
                        continue;
                    }

                    any_target_eligible = true;
                    ++result.generated;
                    ++result.applicable;
                    ++result.eligible;

                    Measurement m;
                    m.candidate       = candidate;
                    m.transform       = transform;
                    m.workspace_bytes = workspace_bytes;
                    m.reason          = GGML_HIP_REJECT_NONE;
                    result.measurements.push_back(m);
                }
                if (!any_target_eligible) {
                    result.transform_scan_gaps.push_back(
                        {transform->id, "no eligible candidate in target family"});
                }
            }
        }
    }
#endif

    std::vector<Measurement *> screening;
    for (Measurement & m : result.measurements) {
        if (m.reason == GGML_HIP_REJECT_NONE) {
            screening.push_back(&m);
        }
    }

    Measurement * native_m = nullptr;
    for (Measurement * m : screening) {
        // Role identity, not descriptor identity: the twin shares native's
        // descriptor and must never be mistaken for the baseline. HI30: a
        // transformed Measurement can legitimately share native's candidate
        // pointer (the transform's target family may include it) and must
        // not be mistaken for the untransformed baseline either.
        if (!m->is_native_twin && m->candidate == native.candidate
#ifdef GGML_HIP_ROUTING_TRANSFORM
                && m->transform == nullptr
#endif
                ) {
            native_m = m; break;
        }
    }
    if (native_m == nullptr) {
        // Standards 7.3: without a measured native there is no correctness
        // reference and no baseline, so this signature is rejected rather than
        // producing a winner chosen against nothing.
        result.reason = "native not eligible; run rejected";
        // provisional_winner already defaults to native (set once at
        // Result construction, above) -- this exit is before any ranking
        // pick exists, so the default is already correct.
        record_result(dispatch_digest, result);
        return native.candidate;
    }

    result.device_state_pre = smi_capture_enabled()
        ? ggml_hip_query_device_state(ggml_cuda_get_device())
        : ggml_hip_device_state{};
    result.device_state_pre_json = device_state_json(result.device_state_pre);
    auto capture_device_state_post = [&]() {
        if (result.measurement_failure || !smi_capture_enabled()) {
            // A failed HIP launch can poison the context even though the
            // timing transaction has rejected the sample. Do not call
            // ggml_cuda_get_device()/RSMI after that point; the post snapshot
            // is explicitly unavailable for this result.
            result.device_state_post = ggml_hip_device_state{};
            result.device_state_post_json = "{}";
            result.device_clock_drift_json = "{\"status\":\"unavailable\"}";
            return;
        }
        result.device_state_post = smi_capture_enabled()
            ? ggml_hip_query_device_state(ggml_cuda_get_device())
            : ggml_hip_device_state{};
        result.device_state_post_json = device_state_json(result.device_state_post);
        result.device_clock_drift_json = device_clock_drift_json(
            result.device_state_pre, result.device_state_post);
    };
    auto record_measurement_failure = [&](Measurement * measurement,
                                          const char * reason) {
        if (measurement != nullptr) {
            measurement->reason = GGML_HIP_REJECT_LAUNCH_FAILED;
        }
        result.measurement_failure = true;
        result.winner = native.candidate;
        result.improvement_pct = 0.0;
        result.promotion_status = "native";
        result.reason = reason;
        capture_device_state_post();
        record_result(dispatch_digest, result);
    };

    // --- scratch destinations -------------------------------------------
    //
    // Two buffers: one holds native's output as the correctness reference, the
    // other receives each candidate under test. The caller's real destination
    // is never written by a measurement, so a candidate producing garbage
    // cannot corrupt the run it is being measured inside.
    const size_t dst_bytes = ggml_nbytes(lc_in.dst);
    const size_t dst_floats = dst_bytes / sizeof(float);

    ggml_cuda_pool_alloc<char> reference_buf(ctx.pool(), dst_bytes);
    ggml_cuda_pool_alloc<char> candidate_buf(ctx.pool(), dst_bytes);

    std::vector<float> reference_host(dst_floats);
    std::vector<float> candidate_host(dst_floats);

    ggml_tensor scratch_dst = *lc_in.dst;
    ggml_hip_launch_context lc = lc_in;
    lc.dst = &scratch_dst;

    // --- HI34: pilot native once, calibrate the shared batch size --------
    scratch_dst.data = reference_buf.get();
    {
        std::vector<double> pilot_gpu;
        std::vector<double> pilot_host;
        if (!time_candidate(native.candidate, lc, config.warmup_launches,
                            config.pilot_samples, 1, config.pre_sample_mode,
                            pilot_gpu, pilot_host,
                            &ctx, true, nullptr)) {
            native_m->reason = GGML_HIP_REJECT_LAUNCH_FAILED;
            result.measurement_failure = true;
            result.reason = "native pilot failed; run rejected";
            record_result(dispatch_digest, result);
            return native.candidate;
        }
        result.launches_per_sample = calibrated_launches_per_sample(
            median_of(pilot_gpu), config);
    }

    // --- native reference, and native's own timing ----------------------
    {
        std::vector<double> gpu;
        std::vector<double> host;
        if (!time_candidate(native.candidate, lc, config.warmup_launches,
                            config.screen_samples, result.launches_per_sample,
                            config.pre_sample_mode,
                            gpu, host, &ctx, true, &native_m->pool_peak_bytes)) {
            native_m->reason = GGML_HIP_REJECT_LAUNCH_FAILED;
            result.measurement_failure = true;
            result.reason = "native failed to launch; run rejected";
            record_result(dispatch_digest, result);
            return native.candidate;
        }
        native_m->median_us      = median_of(gpu);
        native_m->mad_us         = mad_of(gpu, native_m->median_us);
        native_m->p95_us         = percentile_of(gpu, 0.95);
        native_m->host_median_us = median_of(host);
        native_m->samples        = (int) gpu.size();
        native_m->measured       = true;
        ++result.measured;

        if (!hip_ok(hipMemcpyAsync(reference_host.data(), reference_buf.get(),
                                   dst_bytes, hipMemcpyDeviceToHost, lc.stream),
                    "hipMemcpyAsync(reference)")
                || !hip_ok(hipStreamSynchronize(lc.stream),
                           "hipStreamSynchronize(reference)")) {
            disable_smi_after_measurement_failure();
            record_measurement_failure(native_m,
                "native correctness copy failed; tuning experiment poisoned");
            return native.candidate;
        }
    }

    // --- screening (standards 11.4) -------------------------------------
    scratch_dst.data = candidate_buf.get();
    for (Measurement * m : screening) {
        // Primary native was measured in its own block above; the twin is
        // deliberately NOT skipped -- it traverses the same measurement path.
        if (m == native_m) {
            continue;
        }
        std::vector<double> gpu;
        std::vector<double> host;
        if (!time_candidate(m->candidate, lc, config.warmup_launches,
                            config.screen_samples, result.launches_per_sample,
                            config.pre_sample_mode,
                            gpu, host, &ctx, true, &m->pool_peak_bytes, nullptr
#ifdef GGML_HIP_ROUTING_TRANSFORM
                            , m->transform
#endif
                            )) {
            m->reason = GGML_HIP_REJECT_LAUNCH_FAILED;
            result.measurement_failure = true;
            result.reason = "screening measurement failed; tuning experiment poisoned";
            record_result(dispatch_digest, result);
            return native.candidate;
        }

        if (!hip_ok(hipMemcpyAsync(candidate_host.data(), candidate_buf.get(),
                                   dst_bytes, hipMemcpyDeviceToHost, lc.stream),
                    "hipMemcpyAsync(candidate)")
                || !hip_ok(hipStreamSynchronize(lc.stream),
                           "hipStreamSynchronize(candidate)")) {
            disable_smi_after_measurement_failure();
            record_measurement_failure(m,
                "candidate correctness copy failed; tuning experiment poisoned");
            return native.candidate;
        }

        double nmse = 0.0;
        double max_abs = 0.0;
        if (!compare_outputs(reference_host, candidate_host, nmse, max_abs)) {
            // For an ordinary challenger this is a normal rejection. For the
            // double-native twin it is stronger evidence: the same descriptor,
            // same signature, same inputs produced a compatible output as
            // primary native and NaN/Inf moments later. The measurement
            // context is not trustworthy for this signature -- reject it
            // rather than drop the canary and carry on. (Launch/copy
            // failures remain the global poison case; a numerical
            // disagreement only invalidates this signature.)
            if (m->is_native_twin) {
                m->reason = GGML_HIP_REJECT_NAN_INF;
                result.reason = "double-native twin failed correctness; signature rejected";
                record_result(dispatch_digest, result);
                return native.candidate;
            }
            m->reason = GGML_HIP_REJECT_NAN_INF;
            continue;
        }
        m->nmse          = nmse;
        m->max_abs_error = max_abs;
        if (nmse > config.max_nmse || max_abs > config.max_abs_error) {
            // Same reasoning as the NaN/Inf path above: an identical native
            // descriptor disagreeing with itself numerically is a
            // signature-level integrity failure, not an ordinary challenger.
            m->reason = GGML_HIP_REJECT_TOLERANCE;
            if (m->is_native_twin) {
                result.reason = "double-native twin failed correctness; signature rejected";
                record_result(dispatch_digest, result);
                return native.candidate;
            }
            continue;
        }

        m->median_us      = median_of(gpu);
        m->mad_us         = mad_of(gpu, m->median_us);
        m->p95_us         = percentile_of(gpu, 0.95);
        m->host_median_us = median_of(host);
        m->samples        = (int) gpu.size();
        m->measured       = true;
        // The twin is a synthetic measurement role, not another registry
        // candidate: generated >= applicable >= eligible >= measured must stay
        // a funnel over the registry or offline consumers see invalid adaptive
        // evidence.
        if (!m->is_native_twin) {
            ++result.measured;
        }
    }

    // --- retain finalists -----------------------------------------------
    //
    // native always, plus the top 3 by median, plus everything within 10% of
    // the best. Finalist count dominates total tune time, and without this the
    // final stage would re-measure every candidate at 100 samples.
    std::vector<Measurement *> survivors;
    for (Measurement * m : screening) {
        if (m->measured) survivors.push_back(m);
    }
    std::sort(survivors.begin(), survivors.end(),
              [](const Measurement * a, const Measurement * b) {
                  return a->median_us < b->median_us;
              });

    std::vector<Measurement *> finalists;
    const double best = survivors.empty() ? 0.0 : survivors.front()->median_us;
    for (size_t i = 0; i < survivors.size(); ++i) {
        Measurement * m = survivors[i];
        // Role identity again: with the twin present, descriptor equality
        // would classify both rows as native.
        const bool is_native      = (m == native_m);
        const bool is_twin        = m->is_native_twin;
        const bool in_top         = (int) i < config.screen_keep_top;
        const bool near_best      = best > 0.0 &&
            m->median_us <= best * (1.0 + config.screen_keep_within_pct / 100.0);
        // The twin is retained unconditionally (once it has survived
        // screening and correctness): it is the calibration instrument, and
        // letting screening noise drop it would defeat its purpose.
        if (is_native || is_twin || in_top || near_best) {
            finalists.push_back(m);
        }
    }

    // --- final measurement (standards 11.5, 8.2) ------------------------
    //
    // Interleaved rather than one candidate at a time. Measuring A to
    // completion and then B lets thermal drift or a clock change across the
    // run masquerade as a difference between them.
    //
    // HI68: one complete finalist measurement block, extracted as a unit so
    // the canary's fresh-block path measures EXACTLY what the normal final
    // stage measures (same interleaving, same counterbalance schedule, same
    // retime bookkeeping) and nothing else. Every call fills a self-contained
    // statistics set; whatever block the run ultimately ranks is one of these,
    // whole -- never a patchwork of rounds from different blocks.
    auto measure_finalist_block = [&]() {
        for (Measurement * m : finalists) {
            m->final_gpu_us.assign(config.final_samples,
                                   std::numeric_limits<double>::quiet_NaN());
            m->final_host_us.assign(config.final_samples,
                                    std::numeric_limits<double>::quiet_NaN());
            // A fresh block is a fresh timing verdict. Clear the reasons a
            // discarded block left (a candidate rejected noisy or failed on
            // that block must not carry the verdict into the block being
            // ranked). Correctness rejects exist only on unmeasured rows, so
            // nothing authoritative can be cleared here.
            if (m->reason == GGML_HIP_REJECT_NOISY ||
                    m->reason == GGML_HIP_REJECT_LAUNCH_FAILED) {
                m->reason = GGML_HIP_REJECT_NONE;
            }
        }
        for (int round = 0; round < config.final_samples; ++round) {
            const size_t offset = (result.schedule_seed + (uint32_t) round) % finalists.size();
            const bool reverse = ((result.schedule_seed ^ (uint32_t) round) & 1u) != 0;
            const CounterbalancedRound measured = run_counterbalanced_round(
                finalists, offset, reverse, lc, result.launches_per_sample,
                config.pre_sample_mode, "final");
            record_retime_observation(result, measured);
            if (!measured.complete) {
                break;
            }
            for (size_t index = 0; index < finalists.size(); ++index) {
                finalists[index]->final_gpu_us[round]  = measured.gpu_us[index];
                finalists[index]->final_host_us[round] = measured.host_us[index];
            }
        }
        for (Measurement * m : finalists) {
            const std::vector<double> finite_gpu = finite_only(m->final_gpu_us);
            if (finite_gpu.empty()) {
                // Never let a screening median survive a completely failed
                // final stage.  That stale value could still be ranked and
                // promoted as if the candidate had completed the paired
                // measurement protocol.
                m->reason = GGML_HIP_REJECT_LAUNCH_FAILED;
                m->measured = false;
                continue;
            }
            m->median_us      = median_of(finite_gpu);
            m->mad_us         = mad_of(finite_gpu, m->median_us);
            m->p95_us         = percentile_of(finite_gpu, 0.95);
            m->host_median_us = median_of(m->final_host_us);
            m->samples        = (int) finite_gpu.size();
        }
    };
    if (finalists.size() > 1) {
        // HI34: deterministic order, then a per-signature seed (derived from
        // the signature digest, so it is reproducible offline without
        // storing anything extra) rotates which candidate starts each round.
        // A fixed order would let whichever candidate is always measured
        // first in a round absorb a systematic share of thermal drift; the
        // rotation spreads that instead of cancelling it only in aggregate.
        // The twin shares native's stable name, so plain strcmp leaves an
        // equivalent-key pair whose order std::sort does not guarantee. Break
        // ties explicitly (primary before twin) so the schedule is a
        // deterministic, name-unique description of what was actually run.
        std::sort(finalists.begin(), finalists.end(),
                  [](const Measurement * a, const Measurement * b) {
                      const int cmp = strcmp(a->candidate->stable_name,
                                             b->candidate->stable_name);
                      if (cmp != 0) {
                          return cmp < 0;
                      }
                      return a->is_native_twin < b->is_native_twin;
                  });
        result.schedule_seed = (uint32_t) result.signature_digest.bytes[0]
                             | ((uint32_t) result.signature_digest.bytes[1] << 8)
                             | ((uint32_t) result.signature_digest.bytes[2] << 16)
                             | ((uint32_t) result.signature_digest.bytes[3] << 24);
        for (const Measurement * m : finalists) {
            result.schedule_candidates.push_back(measurement_name(*m));
        }
        measure_finalist_block();
    }

    if (result.measurement_failure || g_tuner_poisoned.load(std::memory_order_relaxed)) {
        result.winner = native.candidate;
        result.improvement_pct = 0.0;
        result.promotion_status = "native";
        result.reason = "tuning experiment poisoned; later measurements suppressed";
        capture_device_state_post();
        record_result(dispatch_digest, result);
        return native.candidate;
    }

    if (result.retime_status == "unresolved") {
        result.winner = native.candidate;
        result.improvement_pct = 0.0;
        result.promotion_status = "native";
        result.reason = "clock drift retime unresolved; run rejected";
        capture_device_state_post();
        record_result(dispatch_digest, result);
        return native.candidate;
    }

    // Post-block verdicts for whatever statistics the finalists currently
    // hold: the E4 dispersion rejection plus the "native must be a usable
    // baseline" checks (standards 7.3). Returns a non-empty reason when the
    // whole run must be rejected, empty otherwise. Runs after EVERY block
    // that becomes the ranking dataset -- including HI68's fresh canary
    // block, which gets exactly the same scrutiny as the original one.
    auto post_block_reject_reason = [&]() -> std::string {
        // E4: dispersion rejection, after the final stage. A candidate whose
        // own spread is wider than the margins being decided cannot be
        // ranked, however good its median looks.
        for (Measurement * m : finalists) {
            if (!m->measured || m->median_us <= 0.0) continue;
            if (m->mad_us / m->median_us > config.noisy_mad_ratio) {
                m->reason = GGML_HIP_REJECT_NOISY;
            }
        }
        // A native baseline is only valid when it survived every measurement
        // stage.  In particular, a finalist can have a valid screening
        // median and then fail every final interleaved launch; retaining that
        // stale median would make the ranking stage non-conservative.
        if (native_m->reason == GGML_HIP_REJECT_LAUNCH_FAILED || !native_m->measured) {
            return "native final measurement failed; run rejected";
        }
        if (native_m->reason == GGML_HIP_REJECT_NOISY) {
            // Same treatment as an unmeasurable native (standards 7.3): a
            // baseline that cannot be pinned down is not a baseline, and
            // every improvement_pct in this signature would be computed
            // against a number that moved.
            return "native timing unstable; run rejected";
        }
        return "";
    };

    {
        const std::string initial_reject = post_block_reject_reason();
        if (!initial_reject.empty()) {
            result.reason = initial_reject;
            record_result(dispatch_digest, result);
            return native.candidate;
        }
    }

    // --- noise canary (HI24) --------------------------------------------
    //
    // Native and a forced MMQ candidate at J == J_best are the *same kernel*:
    // `mul_mat_q_switch_J` overwrites J_best with forced_J and calls one
    // launcher, so forcing J_best is native rather than merely equivalent to
    // it (RV21). Any divergence between their medians is therefore measurement
    // error, and needs no external reference to calibrate.
    //
    // This is worth more than a repeatability check on native alone, because
    // it also holds the forced path to producing native's timing -- the
    // invariant the whole dispatch design rests on. If an upstream change ever
    // breaks it, this fires constantly, which is the correct alarm.
    //
    // Costs nothing when the pair is already present: both were going to be
    // measured anyway.
    {
        // HI24 step 4: the explicit double-native replicate. It is the
        // fallback for every family the J_best form cannot cover (non-MMQ
        // natives), and it measures repeatability only -- so the J_best pair,
        // which additionally verifies the forced-dispatch/native equivalence
        // invariant, stays preferred where it exists.
        Measurement * native_twin = nullptr;
        for (Measurement * m : finalists) {
            if (m->is_native_twin && m->measured) {
                native_twin = m;
                break;
            }
        }

        Measurement * twin = nullptr;
        if (native.candidate != nullptr &&
                native.candidate->family == GGML_HIP_FAMILY_MMQ) {
            const bool fb = (sig.ne0[1] % 128) != 0;
            const int j_best = ggml_cuda_mmq_native_j_best(
                (ggml_type) sig.src0_type, fb,
                (sig.flags & GGML_HIP_SIG_HAS_IDS) ? sig.ned[2] : sig.ned[1]);
            if (j_best != 0) {
                for (Measurement * m : finalists) {
                    // The synthetic replicate is not a forced-J candidate; it
                    // is the fallback, considered only when no J-best pair
                    // exists. State the exclusion rather than rely on the
                    // native wrapper's zero-variant descriptor never matching.
                    if (m == native_m || m->is_native_twin ||
                            !m->measured || m->candidate == nullptr) {
                        continue;
                    }
                    if (m->candidate->family == GGML_HIP_FAMILY_MMQ &&
                            m->candidate->variant.primary == j_best &&
                            (m->candidate->variant.fallback != 0) == fb) {
                        twin = m;
                        break;
                    }
                }
            }
        }
        if (twin == nullptr) {
            twin = native_twin;
        }

        // HI68: the transition between blocks is a pure CPU state machine
        // (hip-autotune-canary.h, host-unit-tested). Whatever block the run
        // ranks is always ONE complete finalist measurement; no partial
        // round's statistics are ever substituted into another block's.
        if (twin != nullptr && native_m->median_us > 0.0) {
            // Measurement-instance identity: a J-best pair emits the
            // challenger's name, the fallback emits e.g. "mmvq:native:v1#twin".
            result.canary_pair = measurement_name(*twin);

            const ggml_hip_canary_verdict initial = ggml_hip_canary_transition(
                GGML_HIP_CANARY_STAGE_INITIAL, native_m->median_us,
                twin->median_us, config.noise_canary_pct,
                config.noise_canary_retries);
            result.canary_pct = initial.pct;

            if (initial.next == GGML_HIP_CANARY_RANK) {
                // The original block's pair is quiet: rank it.
                result.canary_state = GGML_HIP_CANARY_PASS;
            } else if (initial.next == GGML_HIP_CANARY_STOP_UNRESOLVED) {
                // Divergent and no probe allowed by config. Report rather
                // than discard: the winner may still be right; what is
                // established is that this signature's margins are not
                // resolvable at these sample counts.
                result.canary_state = GGML_HIP_CANARY_UNRESOLVED;
                GGML_LOG_WARN("bigcherry: noise canary %.1f%% on this "
                              "signature (native vs %s, identical kernels); "
                              "timings are unreliable at these sample counts\n",
                              result.canary_pct, result.canary_pair.c_str());
            } else {  // GGML_HIP_CANARY_RUN_PROBE
                // Pessimistic terminal state from here on: every path below
                // (probe failure, fresh-block rejection, fresh canary failure)
                // ends UNRESOLVED; only a passed fresh canary upgrades it.
                result.canary_state = GGML_HIP_CANARY_UNRESOLVED;
                // One pair-only stability probe. Its samples answer "did the
                // environment settle?" -- they are NOT a measurement of the
                // pair, and their statistics are discarded (RV49/F2: the old
                // code overwrote the ranked medians with exactly this
                // self-selected fresh draw).
                ++result.canary_retries;
                Measurement * pair[2] = { native_m, twin };
                std::vector<double> probe_native, probe_twin;
                for (int round = 0; round < config.final_samples; ++round) {
                    const bool reverse = ((result.schedule_seed ^ (uint32_t) round) & 1u) != 0;
                    const CounterbalancedRound measured = run_counterbalanced_round(
                        {pair[0], pair[1]}, 0, reverse, lc,
                        result.launches_per_sample, config.pre_sample_mode, "final");
                    record_retime_observation(result, measured);
                    if (!measured.complete) {
                        break;
                    }
                    probe_native.push_back(measured.gpu_us[0]);
                    probe_twin.push_back(measured.gpu_us[1]);
                }
                const std::vector<double> pn = finite_only(probe_native);
                const std::vector<double> pt = finite_only(probe_twin);
                const ggml_hip_canary_verdict probe = ggml_hip_canary_transition(
                    GGML_HIP_CANARY_STAGE_PROBE,
                    pn.empty() ? -1.0 : median_of(pn),
                    pt.empty() ? -1.0 : median_of(pt),
                    config.noise_canary_pct, 0);

                if (probe.next == GGML_HIP_CANARY_RUN_FRESH) {
                    // The environment settled: measure ONE fresh complete
                    // finalist block. It becomes the SOLE ranking dataset --
                    // the original block is exactly the one whose QC failed,
                    // and it is discarded for ranking. Its canary is evaluated
                    // once, on this block: never retried until quiet (that
                    // would re-select a favorable baseline).
                    //
                    // HI68 fault-path fix (GPT adjudication): measure_
                    // finalist_block() is NOT transactional -- it clears and
                    // refills the final-sample arrays and recomputes every
                    // finalist's statistics as soon as any fresh round
                    // completes. If the attempt then fails mechanically
                    // (incomplete round, poison, unresolved retime), this row
                    // must still carry ORIGINAL-block evidence: canary_
                    // fresh_block=false means "original-block evidence" by
                    // contract, and a validator assigns unresolved/1/false the
                    // meaning "probe failed; original block retained". So
                    // snapshot every authoritative field the block can mutate,
                    // and restore them on both mechanical-failure paths below.
                    // Only a mechanically valid fresh block is ever committed.
                    struct FreshEvidenceSnapshot {
                        std::vector<double> final_gpu_us;
                        std::vector<double> final_host_us;
                        ggml_hip_reject_reason reason;
                        double median_us      = 0.0;
                        double mad_us         = 0.0;
                        double p95_us         = 0.0;
                        double host_median_us = 0.0;
                        int    samples        = 0;
                        bool   measured       = false;
                    };
                    std::vector<FreshEvidenceSnapshot> fresh_snapshot(finalists.size());
                    for (size_t i = 0; i < finalists.size(); ++i) {
                        const Measurement * m = finalists[i];
                        FreshEvidenceSnapshot & s = fresh_snapshot[i];
                        s.final_gpu_us   = m->final_gpu_us;
                        s.final_host_us  = m->final_host_us;
                        s.reason         = m->reason;
                        s.median_us      = m->median_us;
                        s.mad_us         = m->mad_us;
                        s.p95_us         = m->p95_us;
                        s.host_median_us = m->host_median_us;
                        s.samples        = m->samples;
                        s.measured       = m->measured;
                    }
                    const auto restore_fresh_evidence = [&]() {
                        for (size_t i = 0; i < finalists.size(); ++i) {
                            Measurement * m = finalists[i];
                            const FreshEvidenceSnapshot & s = fresh_snapshot[i];
                            m->final_gpu_us   = s.final_gpu_us;
                            m->final_host_us  = s.final_host_us;
                            m->reason         = s.reason;
                            m->median_us      = s.median_us;
                            m->mad_us         = s.mad_us;
                            m->p95_us         = s.p95_us;
                            m->host_median_us = s.host_median_us;
                            m->samples        = s.samples;
                            m->measured       = s.measured;
                        }
                    };
                    measure_finalist_block();
                    if (result.measurement_failure ||
                            g_tuner_poisoned.load(std::memory_order_relaxed)) {
                        // The fresh attempt's partial rounds have already
                        // mutated the finalists' evidence. Roll it back so the
                        // rejected row still serializes original-block
                        // statistics; the rejection itself stands.
                        restore_fresh_evidence();
                        result.reason = "tuning experiment poisoned; later measurements suppressed";
                        capture_device_state_post();
                        record_result(dispatch_digest, result);
                        return native.candidate;
                    }
                    if (result.retime_status == "unresolved") {
                        // Same transactional guarantee for the retime gate:
                        // a fresh attempt that ended with unresolved clock
                        // drift leaves original-block evidence in place.
                        restore_fresh_evidence();
                        result.reason = "clock drift retime unresolved; run rejected";
                        record_result(dispatch_digest, result);
                        return native.candidate;
                    }
                    // Pessimistic provenance from here on: the fresh window
                    // has already replaced every finalist's statistics, so
                    // EVERY row serialized after this point -- including the
                    // E4 rejections below -- must say the evidence is the
                    // fresh block's. Setting it first is what makes
                    // canary_fresh_block an authoritative measurement-window
                    // marker rather than a "everything passed" flag.
                    result.canary_fresh_block = true;
                    const std::string fresh_reject = post_block_reject_reason();
                    if (!fresh_reject.empty()) {
                        result.reason = fresh_reject;
                        record_result(dispatch_digest, result);
                        return native.candidate;
                    }
                    // Measured guards: a finalist that failed to launch on
                    // the fresh block still holds its stale original-block
                    // median; judging the canary against that would mix two
                    // measurement windows. (Native is guaranteed measured by
                    // post_block_reject_reason above.)
                    const ggml_hip_canary_verdict fresh = ggml_hip_canary_transition(
                        GGML_HIP_CANARY_STAGE_FRESH,
                        native_m->measured ? native_m->median_us : -1.0,
                        twin->measured ? twin->median_us : -1.0,
                        config.noise_canary_pct, 0);
                    result.canary_pct = fresh.pct;
                    // The state was set UNRESOLVED pessimistically before the
                    // probe; upgrade it only on a passed fresh canary.
                    if (fresh.passed) {
                        result.canary_state = GGML_HIP_CANARY_RETRIED_PASS;
                    }
                    if (!fresh.passed) {
                        // Report rather than discard: native is retained on
                        // the fresh block's evidence, and that is what this
                        // row records.
                        GGML_LOG_WARN("bigcherry: noise canary %.1f%% on "
                                      "this signature (native vs %s, identical "
                                      "kernels, fresh block); native retained\n",
                                      result.canary_pct, result.canary_pair.c_str());
                    }
                } else {
                    // The probe still failed: the environment did not settle.
                    // Native is retained on the ORIGINAL block's evidence --
                    // no statistics were swapped, so nothing was self-
                    // selected. The state is already UNRESOLVED (set
                    // pessimistically above). Report rather than discard.
                    GGML_LOG_WARN("bigcherry: noise canary %.1f%% on this "
                                  "signature (native vs %s, identical kernels); "
                                  "stability probe still divergent; native retained\n",
                                  initial.pct, result.canary_pair.c_str());
                }
            }
        }
    }

    if (result.measurement_failure || g_tuner_poisoned.load(std::memory_order_relaxed)) {
        result.winner = native.candidate;
        result.improvement_pct = 0.0;
        result.promotion_status = "native";
        result.reason = "tuning experiment poisoned; later measurements suppressed";
        capture_device_state_post();
        record_result(dispatch_digest, result);
        return native.candidate;
    }

    // E3: rank on whichever resource is actually binding, computed after the
    // canary section above (which under HI68 may replace the entire finalist
    // statistics set with a fresh complete block) so this is never stale
    // relative to what median_us/host_median_us hold.
    // For most candidates host_median ~= gpu_median + sync_overhead and the
    // adjustment does nothing; it matters when host-side submission cost is
    // a material fraction of a tiny call (TUNING-DETAIL.md's 12.36us
    // signatures).
    {
        const double sync_overhead = host_sync_overhead_us(lc.stream);
        if (g_tuner_poisoned.load(std::memory_order_relaxed)) {
            record_measurement_failure(nullptr,
                "host synchronization overhead failed; tuning experiment poisoned");
            return native.candidate;
        }
        for (Measurement * m : finalists) {
            if (!m->measured) continue;
            m->effective_us = effective_us_of(m->median_us, m->host_median_us,
                                              sync_overhead);
        }
    }

    // --- winner selection (standards 7.3, HI12 E1, HI50) ------------------
    //
    // HI50: every policy named in config.active_policies (default "all") is
    // evaluated here against the same already-measured finalists -- pure
    // CPU-side ranking, zero extra GPU launches -- and each one's full
    // candidate verdict list is recorded so a rejected or near-tie
    // candidate stays visible afterward, not just the winner. The
    // production policy always evaluates regardless of active_policies, so
    // a misconfigured allow-list can only shrink shadow reporting, never
    // stop promotion. Only its pick continues below into the determinism
    // recheck and confirmation holdout; every other policy's pick is an
    // unconfirmed shadow prediction that never touches a GPU launch or a
    // rejection reason on its own.
    const size_t production_index = resolved_production_index;
    PolicySelection production_sel;
    {
        std::string decisions_json = "[";
        bool first = true;
        for (size_t i = 0; i < g_policy_table_size; ++i) {
            const PolicyTableEntry & entry = g_policy_table[i];
            const bool is_production = (i == production_index);
            if (!is_production && !policy_name_is_active(entry.name, config.active_policies)) {
                continue;
            }
            PolicySelection sel = entry.fn(finalists, native_m, result.canary_state, config);
            if (!first) decisions_json += ",";
            first = false;
            decisions_json += ranking_decision_json(entry, sel, native_m, is_production);
            if (is_production) {
                production_sel = std::move(sel);
            }
        }
        decisions_json += "]";
        result.ranking_decisions_json     = decisions_json;
        result.production_policy_name     = g_policy_table[production_index].name;
        result.production_policy_version  = g_policy_table[production_index].version;
    }

    std::vector<Measurement *> qualified = production_sel.qualified;
    bool any_short_rounds = production_sel.any_short_rounds;
    Measurement * winner_m = nullptr;

    // E4: winner-only determinism recheck. Bounded at two extra launches
    // regardless of catalog size -- the winner is the only candidate whose
    // determinism this run ends up asserting in a shipped cache. Operates
    // only on the production policy's qualified set -- a shadow policy's
    // pick is never GPU-retimed on its own.
    for (;;) {
        winner_m = qualified.empty() ? nullptr : qualified.front();
        if (winner_m == nullptr || winner_m == native_m ||
                !config.verify_determinism || !winner_m->candidate->deterministic) {
            break;
        }
        std::vector<float> first(dst_floats);
        std::vector<float> second(dst_floats);
        const bool launched =
            launch_and_fetch(winner_m->candidate, lc, candidate_buf.get(),
                             dst_bytes, first
#ifdef GGML_HIP_ROUTING_TRANSFORM
                             , winner_m->transform
#endif
                             ) &&
            launch_and_fetch(winner_m->candidate, lc, candidate_buf.get(),
                             dst_bytes, second
#ifdef GGML_HIP_ROUTING_TRANSFORM
                             , winner_m->transform
#endif
                             );
        if (!launched) {
            winner_m->reason = GGML_HIP_REJECT_LAUNCH_FAILED;
            result.measurement_failure = true;
            result.winner = native.candidate;
            result.improvement_pct = 0.0;
            result.promotion_status = "native";
            // GPT review (2026-08-21): this exit lands after the ranking
            // policy has already produced ranking_decisions but before
            // provisional_winner is ever assigned below -- native is
            // operationally safe here (matches the inherited default) even
            // though it doesn't capture which candidate determinism was
            // being checked when it failed. A dedicated evaluation-audit
            // field would be needed to record that; out of scope for now.
            result.reason = "determinism measurement failed; tuning experiment poisoned";
            capture_device_state_post();
            record_result(dispatch_digest, result);
            return native.candidate;
        }
        if (std::memcmp(first.data(), second.data(),
                        dst_floats * sizeof(float)) != 0) {
            winner_m->reason = GGML_HIP_REJECT_UNSTABLE;
            qualified.erase(qualified.begin());
            continue;
        }
        break;
    }

    // HI50: the ranking-stage pick, fixed before the confirmation holdout
    // (and any later promotion demotion) can change `result.winner` again --
    // the stable target for offline replay to validate a policy's ranking
    // output against.
    result.provisional_winner = (winner_m == nullptr || winner_m == native_m)
        ? native.candidate->stable_name
        : measurement_name(*winner_m);

    if (winner_m == nullptr || winner_m == native_m) {
        result.winner            = native.candidate;
        result.improvement_pct   = 0.0;
        result.promotion_status  = "native";
        result.reason            = any_short_rounds
            ? "native retained (too few paired rounds to confirm any challenger)"
            : "native retained";
    } else {
        result.confidence = 1.0 - winner_m->sign_p;

        // HI34: fresh, disjoint confirmation holdout. Only the provisional
        // winner versus native, with new rounds in schedule-seeded
        // alternating order -- the selection rounds above (which chose this
        // candidate out of every finalist) cannot also be the evidence that
        // promotes it. Reusing them would be winner's-curse: whichever
        // candidate got lucky on the selection rounds is exactly the one
        // most likely to look good again by chance.
        const int rounds = std::max(config.confirmation_samples, config.min_paired_rounds);
        result.confirmation_native_us.assign(
            rounds, std::numeric_limits<double>::quiet_NaN());
        result.confirmation_winner_us.assign(
            rounds, std::numeric_limits<double>::quiet_NaN());
        Measurement * pair[2] = { native_m, winner_m };
        for (int round = 0; round < rounds; ++round) {
            const bool reverse = ((result.schedule_seed ^ (uint32_t) round) & 1u) != 0;
            const CounterbalancedRound measured = run_counterbalanced_round(
                {pair[0], pair[1]}, 0, reverse, lc,
                result.launches_per_sample, config.pre_sample_mode, "confirmation");
            record_retime_observation(result, measured);
            if (!measured.complete) {
                // GPT review (2026-08-21): the dispatch winner falls back to
                // native (never ship an unconfirmed candidate), but
                // provisional_winner is deliberately LEFT ALONE -- it was set
                // to the challenger's name by the "HI50 ranking-stage pick"
                // above, and that is a true, useful fact: ranking nominated
                // this candidate; the attempt to confirm it produced invalid
                // evidence. Overwriting it to native would both discard that
                // fact and desync it from ranking_decisions.predicted_winner
                // (which still names the challenger), tripping a DIFFERENT
                // invariant in tune_promotion.py's _validate_policy_identity.
                // promotion_status uses a distinct non-"native" terminal
                // state so the challenger-provisional/native-status pairing
                // reads as "evaluation aborted", not "ranking chose native".
                result.winner = native.candidate;
                result.improvement_pct = 0.0;
                result.promotion_status = "evaluation_failed";
                result.reason = "confirmation measurement failed; tuning experiment poisoned";
                capture_device_state_post();
                record_result(dispatch_digest, result);
                return native.candidate;
            }
            result.confirmation_native_us[round] = measured.gpu_us[0];
            result.confirmation_winner_us[round] = measured.gpu_us[1];
        }
        result.p_value = paired_sign_test(
            result.confirmation_native_us, result.confirmation_winner_us,
            config.min_paired_rounds, result.confirmation_wins,
            result.confirmation_rounds);
        const double confirmation_native = median_of(result.confirmation_native_us);
        const double confirmation_winner = median_of(result.confirmation_winner_us);
        result.confirmation_effect_pct = confirmation_native > 0.0
            ? 100.0 * (confirmation_native - confirmation_winner) / confirmation_native
            : 0.0;
        const bool confirmed =
            result.confirmation_rounds >= config.min_paired_rounds &&
            result.confirmation_effect_pct >= config.replacement_threshold_pct &&
            result.p_value <= config.confidence_alpha;

        if (!confirmed) {
            result.winner            = native.candidate;
            result.improvement_pct   = 0.0;
            result.promotion_status  = "confirmation_rejected";
            result.reason            = "native retained (fresh confirmation rejected provisional winner)";
        } else {
            result.winner            = winner_m->candidate;
#ifdef GGML_HIP_ROUTING_TRANSFORM
            result.winner_transform  = winner_m->transform;
#endif
            result.improvement_pct   = result.confirmation_effect_pct;
            result.promotion_status  = "pending_bh";
            result.reason            = "fresh confirmation passed; experiment-wide BH pending, " +
                std::to_string(result.confirmation_wins) + "/" +
                std::to_string(result.confirmation_rounds) + " confirmation rounds" +
                (winner_m->candidate->family == native.candidate->family
                    ? "" : " (different family from native)");
        }
    }

    // Confirmation is also a promotion gate. A drifted confirmation round
    // that cannot be stabilized must never leave the provisional winner in a
    // promotable state merely because earlier ranking rounds were clean.
    if (result.retime_status == "unresolved") {
        result.winner = native.candidate;
        result.improvement_pct = 0.0;
        // GPT review (2026-08-21): provisional_winner is deliberately left
        // untouched here (see the confirmation-failure branch above for the
        // full rationale) -- but unlike that branch, this one can also fire
        // when winner_m == native_m, i.e. no real challenger was ever
        // nominated (provisional_winner already reads native, promotion_status
        // is already "native" from the branch above). Only relabel as an
        // aborted evaluation when a challenger's identity is actually at
        // stake; otherwise "native" already correctly describes this exit.
        if (result.provisional_winner != result.native_name) {
            result.promotion_status = "evaluation_failed";
        } else {
            result.promotion_status = "native";
        }
        result.reason = "clock drift retime unresolved; run rejected";
    }

    capture_device_state_post();

    {
        int counts[GGML_HIP_REJECT_COUNT];
        reject_counts(result, counts);
        GGML_LOG_INFO("bigcherry: tuned %s -- gen/appl/elig/meas %d/%d/%d/%d, "
                      "rejects arch=%d inelig=%d ws=%d launch=%d nan=%d tol=%d "
                      "noisy=%d unstable=%d, winner %s (%.2f%% vs native, "
                      "confidence=%.4f, promotion_status=%s)\n",
                      ggml_hip_digest_hex(dispatch_digest).c_str(),
                      result.generated, result.applicable, result.eligible,
                      result.measured,
                      counts[GGML_HIP_REJECT_ARCHITECTURE],
                      counts[GGML_HIP_REJECT_INELIGIBLE],
                      counts[GGML_HIP_REJECT_WORKSPACE],
                      counts[GGML_HIP_REJECT_LAUNCH_FAILED],
                      counts[GGML_HIP_REJECT_NAN_INF],
                      counts[GGML_HIP_REJECT_TOLERANCE],
                      counts[GGML_HIP_REJECT_NOISY],
                      counts[GGML_HIP_REJECT_UNSTABLE],
                      result.winner->stable_name, result.improvement_pct,
                      result.confidence, result.promotion_status.c_str());
    }

    record_result(dispatch_digest, result);
    return result.winner;
}

// HI64 (2026-08-22): public entry point. Every exit of
// ggml_hip_tuner_resolve_impl() -- the cache hit above and every deeper
// `record_result(...); return ...;` site -- leaves a fresh or reused
// g_results[dispatch_digest] entry in place by the time control reaches
// here, so this second lookup reliably reports the resolution's real
// measurement_failure state without requiring the impl's ~25 exit sites to
// each be touched individually.
//
// g_single_flight_mutex is acquired HERE, not inside the impl, and held
// across both the impl call and this function's own g_results re-check.
// Without that, a second thread could acquire the (impl-local) lock and
// replace this exact dispatch_digest's entry -- failure -> success, via
// record_result() -- in the window between the impl returning and this
// read, and this invocation would then wrongly report
// measurement_failure=false for a call that actually failed. Today's only
// caller (dispatch.cu's process_binding_cacheable gate) happens to stay
// safe either way once a real success exists, but the public contract
// promises "true only when THIS invocation hit a fatal failure" and must
// actually mean that.
ggml_hip_tuner_resolution ggml_hip_tuner_resolve(
        ggml_backend_cuda_context & ctx,
        const ggml_hip_dispatch_signature_v1 & sig,
        const ggml_hip_hardware_key_v1 & hw,
        const ggml_hip_digest & dispatch_digest,
        const ggml_hip_native_selection & native,
        const ggml_hip_launch_context & lc) {
    std::unique_lock<std::mutex> single_flight_lock(g_single_flight_mutex);
    ggml_hip_tuner_resolution out;
    out.winner = ggml_hip_tuner_resolve_impl(ctx, sig, hw, dispatch_digest, native, lc);
    std::lock_guard<std::mutex> lock(g_mutex);
    const auto found = g_results.find(dispatch_digest);
    out.measurement_failure = found != g_results.end() && found->second.measurement_failure;
    return out;
}

void ggml_hip_tuner_flush() {
    std::lock_guard<std::mutex> lock(g_mutex);
    if (g_results.empty()) {
        return;
    }

    const char * path = getenv("GGML_HIP_DISPATCH_DB");
    if (path == nullptr || path[0] == '\0') {
        GGML_LOG_WARN("bigcherry: tuned %zu signature(s) but "
                      "GGML_HIP_DISPATCH_DB is unset; nothing written\n",
                      g_results.size());
        return;
    }

    // HI48: same-directory temp file, fsync'd and atomically renamed over the
    // target -- a crash mid-write leaves the previous good measurements file
    // in place rather than a truncated one masquerading as complete.
    std::string measurements_path = std::string(path) + ".measurements.jsonl";
    ggml_hip_atomic_file measurements_atomic;
    if (!ggml_hip_atomic_begin(measurements_path.c_str(), measurements_atomic)) {
        GGML_LOG_WARN("bigcherry: cannot write '%s'\n",
                      measurements_path.c_str());
        return;
    }
    FILE * file = measurements_atomic.file;

    const ggml_hip_tuner_config & config = ggml_hip_tuner_get_config();
    const HostSyncOverheadCache sync_overhead = g_host_sync_overhead.load();

    fprintf(file,
            "{\"kind\":\"header\",\"artifact_version\":%d,"
            "\"source_revision\":\"%s\",\"manifest_hash\":\"%s\","
            "\"compiler\":\"%s\",\"hip_version\":\"%s\","
            "\"variant_set\":\"%s\",\"build_descriptor_hash\":\"%s\","
            "\"host_sync_overhead_us\":%.3f,\"host_sync_overhead_valid\":%s,\"final_samples\":%d,"
            "\"warmup_launches\":%d,\"screen_samples\":%d,"
            "\"confirmation_samples\":%d,\"replacement_threshold_pct\":%.4f,"
            "\"production_policy\":\"%s\",\"active_policies\":\"%s\","
                "\"alpha\":%.4f,\"double_native\":%d,"
                "\"flush_l2\":%d,\"flush_evict_mb\":%d,"
                "\"pre_sample_mode\":\"%s\"}\n",
            GGML_HIP_AUTOTUNE_ARTIFACT_VERSION,
            GGML_HIP_AUTOTUNE_SOURCE_REVISION_STR,
            GGML_HIP_AUTOTUNE_MANIFEST_HASH_STR,
            GGML_HIP_COMPILER_STR, GGML_HIP_VERSION_STR,
            GGML_HIP_AUTOTUNE_VARIANT_SET_STR, GGML_HIP_AUTOTUNE_DESCRIPTOR_HASH_STR,
            sync_overhead.us, sync_overhead.valid ? "true" : "false", config.final_samples,
            config.warmup_launches, config.screen_samples,
            config.confirmation_samples, config.replacement_threshold_pct,
            config.production_policy.c_str(), config.active_policies.c_str(),
            config.confidence_alpha, config.double_native,
            // Measurement-affecting knobs belong in the evidence: a flush=0
            // artifact and a flush=1 artifact are not measurement-equivalent
            // even with identical build/input digests (HI34 step 3). The
            // resolved pre_sample_mode string is the provenance of record;
            // flush_l2 is its 0/1 wire mirror (HI65).
            config.flush_l2, config.flush_evict_mb,
            pre_sample_mode_name(config.pre_sample_mode));

    for (const auto & entry : g_results) {
        const Result & r = entry.second;
        int counts[GGML_HIP_REJECT_COUNT];
        reject_counts(r, counts);

        // Coverage first, because it is what says whether the winner means
        // anything. generated/applicable/eligible/measured being far apart
        // is the signal that an eligibility predicate is too strict or a
        // candidate set never matched the observed shapes.
        fprintf(file,
                "{\"kind\":\"result\",\"dispatch\":\"%s\","
                "\"signature\":\"%s\",\"hardware\":\"%s\","
                "\"canonical\":%s,\"winner\":\"%s\","
                "\"winner_transform\":\"%s\",\"winner_transform_id\":%d,"
                "\"native\":\"%s\","
                "\"improvement_pct\":%.3f,\"confidence\":%.4f,"
                "\"generated\":%d,\"applicable\":%d,\"eligible\":%d,"
                "\"measured\":%d,\"reason\":\"%s\","
                "\"rejects\":{\"architecture\":%d,\"ineligible\":%d,"
                "\"workspace\":%d,\"launch_failed\":%d,\"nan_inf\":%d,"
                "\"tolerance\":%d,\"noisy\":%d,\"unstable\":%d},"
                "\"canary_pct\":%.3f,\"canary_retries\":%d,"
                "\"canary_fresh_block\":%s,"
                "\"canary_pair\":\"%s\","
                "\"promotion_status\":\"%s\","
                "\"launches_per_sample\":%d,\"schedule_seed\":%u,"
                "\"device_state_pre\":%s,\"device_state_post\":%s,"
                "\"device_clock_drift\":%s,"
                "\"clock_drift_rounds\":%d,\"reverse_retime_attempts\":%d,"
                "\"reverse_retime_passed\":%d,\"retime_status\":\"%s\","
                "\"measurement_failure\":%s,"
                "\"canary_state\":\"%s\",\"provisional_winner\":\"%s\","
                "\"production_policy\":{\"name\":\"%s\",\"version\":%d},"
                "\"ranking_decisions\":%s",
                ggml_hip_digest_hex(entry.first).c_str(),
                ggml_hip_digest_hex(r.signature_digest).c_str(),
                ggml_hip_digest_hex(r.hardware_digest).c_str(),
                // Fail closed on the only flush field without a valid JSON
                // default: an empty canonical serializes as null rather than
                // corrupting the row ("canonical":,). The stubs above set it;
                // this keeps future stub paths from breaking every consumer.
                r.canonical_json.empty() ? "null" : r.canonical_json.c_str(),
                r.winner ? r.winner->stable_name : "",
                winner_transform_name(r),
                winner_transform_id(r),
                r.native_name.c_str(),
                r.improvement_pct, r.confidence,
                r.generated, r.applicable, r.eligible, r.measured,
                r.reason.c_str(),
                counts[GGML_HIP_REJECT_ARCHITECTURE],
                counts[GGML_HIP_REJECT_INELIGIBLE],
                counts[GGML_HIP_REJECT_WORKSPACE],
                counts[GGML_HIP_REJECT_LAUNCH_FAILED],
                counts[GGML_HIP_REJECT_NAN_INF],
                counts[GGML_HIP_REJECT_TOLERANCE],
                counts[GGML_HIP_REJECT_NOISY],
                counts[GGML_HIP_REJECT_UNSTABLE],
                r.canary_pct, r.canary_retries,
                r.canary_fresh_block ? "true" : "false", r.canary_pair.c_str(),
                r.promotion_status.c_str(),
                r.launches_per_sample, r.schedule_seed,
                r.device_state_pre_json.c_str(), r.device_state_post_json.c_str(),
                r.device_clock_drift_json.c_str(),
                r.clock_drift_rounds, r.reverse_retime_attempts,
                r.reverse_retime_passed, r.retime_status.c_str(),
                r.measurement_failure ? "true" : "false",
                canary_state_name(r.canary_state), r.provisional_winner.c_str(),
                r.production_policy_name.c_str(), r.production_policy_version,
                r.ranking_decisions_json.c_str());

        // HI34: schedule + fresh confirmation evidence, only meaningful (and
        // only present) once a non-native winner was nominated and put
        // through the holdout above -- a "native" row has nothing to verify.
        if (!r.schedule_candidates.empty()) {
            fprintf(file,
                    ",\"schedule\":{\"schema_version\":1,"
                    "\"selection_algorithm\":\"seeded-rotation-v1\","
                    "\"confirmation_algorithm\":\"seeded-alternation-v1\","
                    "\"counterbalance_algorithm\":\"seeded-reverse-v1\","
                    "\"candidates\":%s}",
                    string_array_json(r.schedule_candidates).c_str());
        }
        if (r.promotion_status == "pending_bh" || r.promotion_status == "confirmation_rejected") {
            fprintf(file,
                    ",\"confirmation\":{\"p_value\":%.4f,"
                    "\"effect_pct\":%.3f,\"wins\":%d,\"rounds\":%d,"
                    "\"native_us\":%s,\"winner_us\":%s}",
                    r.p_value, r.confirmation_effect_pct,
                    r.confirmation_wins, r.confirmation_rounds,
                    samples_json(r.confirmation_native_us).c_str(),
                    samples_json(r.confirmation_winner_us).c_str());
        }
        fprintf(file, ",\"candidates\":[");

        bool first = true;
        for (const Measurement & m : r.measurements) {
            fprintf(file,
                    "%s{\"name\":\"%s\",\"status\":\"%s\",\"median_us\":%.3f,"
                    "\"effective_us\":%.3f,"
                    "\"mad_us\":%.3f,\"p95_us\":%.3f,\"host_median_us\":%.3f,"
                    "\"nmse\":%.6g,\"max_abs\":%.6g,\"workspace\":%zu,"
                    "\"pool_peak_bytes\":%zu,"
                    "\"samples\":%d,\"sign_p\":%.4f,\"sign_wins\":%d,"
                    "\"sign_rounds\":%d",
                    first ? "" : ",",
                    measurement_name(m).c_str(),
                    reason_name(m.reason), m.median_us, m.effective_us,
                    m.mad_us, m.p95_us,
                    m.host_median_us, m.nmse, m.max_abs_error,
                    m.workspace_bytes, m.pool_peak_bytes, m.samples, m.sign_p, m.sign_wins,
                    m.sign_rounds);
            if (config.emit_samples && !m.final_gpu_us.empty()) {
                fprintf(file, ",\"samples_us\":%s",
                        samples_json(m.final_gpu_us).c_str());
            }
            fprintf(file, "}");
            first = false;
        }
        fprintf(file, "]}\n");
    }
    if (!ggml_hip_atomic_commit(measurements_atomic)) {
        GGML_LOG_WARN("bigcherry: atomic measurements replacement failed for '%s'\n",
                      measurements_path.c_str());
        return;
    }

    GGML_LOG_INFO("bigcherry: wrote %zu tuning result(s) to '%s'\n",
                  g_results.size(), measurements_path.c_str());

    // HI24 step 8: canary_pct is recorded per signature and nothing
    // aggregated it, so nobody read it. This is the line that says whether
    // this run's numbers can be believed at all. canary_pct == -1.0 means no
    // same-kernel pair was available for that signature (e.g. a non-MMQ
    // native with double_native disabled) and is excluded from these counts,
    // not treated as zero divergence.
    {
        size_t checked = 0, flagged = 0, retried = 0;
        double worst = 0.0;
        std::string worst_dispatch;
        for (const auto & entry : g_results) {
            const Result & r = entry.second;
            if (r.canary_pct < 0.0) {
                continue;
            }
            ++checked;
            if (r.canary_pct > config.noise_canary_pct) {
                ++flagged;
            }
            if (r.canary_retries > 0) {
                ++retried;
            }
            if (r.canary_pct > worst) {
                worst = r.canary_pct;
                worst_dispatch = ggml_hip_digest_hex(entry.first);
            }
        }
        if (checked > 0) {
            GGML_LOG_INFO(
                "bigcherry: canary -- %zu/%zu signature(s) checked, %zu flagged "
                "above %.2f%%, worst %.2f%% (%s), %zu re-measured\n",
                checked, g_results.size(), flagged, config.noise_canary_pct,
                worst, worst_dispatch.c_str(), retried);
        } else {
            GGML_LOG_INFO(
                "bigcherry: canary -- 0/%zu signature(s) had a same-kernel pair "
                "to check\n",
                g_results.size());
        }
    }

#ifdef GGML_HIP_ROUTING_TRANSFORM
    // bigcherry (HI29): a separate file, separate schema, separate consumer
    // (offline agent-driven pattern analysis across many gaps, not a human
    // reading one tuner run) -- same GGML_HIP_DISPATCH_DB prefix and the
    // same crash-safety mechanism (HI48 atomic same-directory temp file +
    // rename) as measurements.jsonl above, so a killed run leaves either the
    // previous good file or nothing, never a truncated one masquerading as
    // complete.
    {
        std::lock_guard<std::mutex> transform_lock(g_transform_mutex);
        if (!g_transform_attempts.empty() || !g_transform_gaps.empty()) {
            std::string transforms_path = std::string(path) + ".transforms.jsonl";
            ggml_hip_atomic_file transforms_atomic;
            if (!ggml_hip_atomic_begin(transforms_path.c_str(), transforms_atomic)) {
                GGML_LOG_WARN("bigcherry: cannot write '%s'\n", transforms_path.c_str());
                return;
            }
            FILE * tfile = transforms_atomic.file;

            fprintf(tfile,
                    "{\"kind\":\"header\",\"artifact_version\":%d,"
                    "\"source_revision\":\"%s\",\"manifest_hash\":\"%s\"}\n",
                    GGML_HIP_AUTOTUNE_ARTIFACT_VERSION,
                    GGML_HIP_AUTOTUNE_SOURCE_REVISION_STR,
                    GGML_HIP_AUTOTUNE_MANIFEST_HASH_STR);

            for (const TransformAttemptRecord & r : g_transform_attempts) {
                fprintf(tfile,
                        "{\"kind\":\"transform-attempt\","
                        "\"original_sig\":\"%s\",\"hardware\":\"%s\","
                        "\"transformation_id\":%d,\"transformation_name\":\"%s\","
                        "\"source\":\"%s\",\"original_native_family\":\"%s\","
                        "\"result\":\"%s\",\"rejection_reason\":\"%s\","
                        "\"transformed_winner\":\"%s\","
                        "\"original_us\":%.3f,\"transformed_us\":%.3f,"
                        "\"improvement_pct\":%.3f,\"nmse\":%.6g,\"max_abs_error\":%.6g}\n",
                        ggml_hip_digest_hex(r.original_sig).c_str(),
                        ggml_hip_digest_hex(r.hardware).c_str(),
                        (int) r.transform_id, r.transform_name.c_str(),
                        r.source.c_str(), r.original_native_family.c_str(),
                        r.result.c_str(), r.rejection_reason.c_str(),
                        r.transformed_winner.c_str(),
                        r.original_us, r.transformed_us,
                        r.improvement_pct, r.nmse, r.max_abs_error);
            }

            for (const TransformGapRecord & r : g_transform_gaps) {
                fprintf(tfile,
                        "{\"kind\":\"transform-gap\","
                        "\"sig\":\"%s\",\"hardware\":\"%s\","
                        "\"native_family\":\"%s\",\"est_bytes\":%lld,"
                        "\"transformations_tried\":[",
                        ggml_hip_digest_hex(r.sig).c_str(),
                        ggml_hip_digest_hex(r.hardware).c_str(),
                        r.native_family.c_str(), (long long) r.est_bytes);
                for (size_t i = 0; i < r.tried.size(); ++i) {
                    fprintf(tfile, "%s{\"id\":%d,\"reason\":\"%s\"}",
                            i == 0 ? "" : ",",
                            (int) r.tried[i].transform_id,
                            r.tried[i].rejection_reason.c_str());
                }
                fprintf(tfile, "]}\n");
            }

            if (!ggml_hip_atomic_commit(transforms_atomic)) {
                GGML_LOG_WARN("bigcherry: atomic transforms replacement failed for '%s'\n",
                              transforms_path.c_str());
                return;
            }
            GGML_LOG_INFO("bigcherry: wrote %zu transform-attempt(s), %zu transform-gap(s) to '%s'\n",
                          g_transform_attempts.size(), g_transform_gaps.size(),
                          transforms_path.c_str());
        }
    }
#endif // GGML_HIP_ROUTING_TRANSFORM
}

#endif // GGML_USE_HIP && GGML_HIP_AUTOTUNE
