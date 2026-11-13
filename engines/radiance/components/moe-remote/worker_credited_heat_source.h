#pragma once
// Inactive RHP02 policy input. The reader is a CPU-side queue snapshot, NOT a
// GPU histogram collector. Each credited row owns its data after transport;
// the borrowed engine callback pointer must have been copied before enqueue.
// The trusted start boundary is established out of band for one rank and one
// owner-map epoch. No implicit baseline across gaps, restarts or owner changes.
#include "worker_heat_source.h"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <string>
#include <utility>
#include <vector>

namespace r3::policy {

enum class CreditedPhase : uint8_t { PREFILL = 1, DECODE = 2, MIXED = 3 };

// Raw landed histogram counts are multiplied by reports exactly once,
// matching HeatEngine::observe(hist, reports). weighted_sum and the cumulative
// observation count are independently supplied and must agree with that math.
struct CreditedHeatRow {
    uint64_t run_generation = 0;
    uint64_t owner_epoch = 0;
    uint64_t source_step = 0;
    uint64_t bank_generation = 0;
    uint64_t credit_sequence = 0;     // consecutive per-rank credited row ID
    uint64_t issued_watermark = 0;
    uint64_t completed_watermark = 0;
    uint64_t credited_watermark = 0;
    uint64_t dropped_watermark = 0;  // any increase invalidates the window
    uint64_t observed_monotonic_ns = 0;
    uint64_t weighted_sum = 0;
    uint64_t observation_count = 0; // cumulative weighted expert selections
    uint32_t rank = 0, layer = 0, reports = 0;
    CreditedPhase phase = CreditedPhase::MIXED;
    bool source_complete = false;    // required, never independently sufficient
    std::vector<int32_t> counts;     // unweighted, model-global expert IDs
};

// A single bounded queue snapshot, with explicit loss/invalid indication.
// The reader must not retain a borrowed engine histogram pointer. Draining
// into this batch happens off the inference thread, with no GPU access.
enum class CreditedRead : uint8_t { EMPTY, ROWS, GAP, INVALID };
class CreditedRowReader {
public:
    virtual ~CreditedRowReader() = default;
    virtual CreditedRead read_batch(std::vector<CreditedHeatRow>* out) = 0;
};

struct CreditedBoundary {
    uint32_t rank = 0, layers = 0, experts = 0;
    uint64_t run_generation = 0, owner_epoch = 0;
    uint64_t next_credit_sequence = 0;
    uint64_t observation_count = 0;
    uint64_t issued_watermark = 0, completed_watermark = 0;
    uint64_t dropped_watermark = 0;
    uint64_t max_age_ns = 0;        // policy's configured staleness limit
};

struct AcceptedCreditedRow {
    uint64_t source_step = 0, credit_sequence = 0, bank_generation = 0;
    uint32_t rank = 0, layer = 0, reports = 0;
    CreditedPhase phase = CreditedPhase::MIXED;
    std::vector<int32_t> weighted_counts;
};

enum class CreditedReject : uint8_t {
    NONE, CONFIGURATION, READER, COVERAGE_GAP, COVERAGE_REGRESSION,
    GENERATION, DOMAIN, PHASE, INCOMPLETE, REPLAY, STALE,
    COUNT_SHAPE, COUNT_NEGATIVE, WEIGHT_OVERFLOW, WEIGHT_SUM,
    OBSERVATION_REGRESSION, OBSERVATION_SUM
};
inline const char* credited_reason(CreditedReject r) {
    switch (r) {
        case CreditedReject::NONE: return "none";
        case CreditedReject::CONFIGURATION: return "configuration";
        case CreditedReject::READER: return "reader";
        case CreditedReject::COVERAGE_GAP: return "coverage_gap";
        case CreditedReject::COVERAGE_REGRESSION: return "coverage_regression";
        case CreditedReject::GENERATION: return "generation";
        case CreditedReject::DOMAIN: return "domain";
        case CreditedReject::PHASE: return "phase";
        case CreditedReject::INCOMPLETE: return "incomplete";
        case CreditedReject::REPLAY: return "replay";
        case CreditedReject::STALE: return "stale";
        case CreditedReject::COUNT_SHAPE: return "count_shape";
        case CreditedReject::COUNT_NEGATIVE: return "count_negative";
        case CreditedReject::WEIGHT_OVERFLOW: return "weight_overflow";
        case CreditedReject::WEIGHT_SUM: return "weight_sum";
        case CreditedReject::OBSERVATION_REGRESSION: return "observation_regression";
        case CreditedReject::OBSERVATION_SUM: return "observation_sum";
    }
    return "unknown";
}

class CreditedEngineSource final : public HeatSource {
    CreditedRowReader& reader_;
    CreditedBoundary bound_;
    std::vector<uint64_t> last_step_;       // one monotonic step per layer
    std::vector<uint8_t> step_seen_;
    std::vector<AcceptedCreditedRow> accepted_;
    CreditedReject fault_ = CreditedReject::NONE;
    bool configured_ = false;

    static bool valid(const CreditedBoundary& b) {
        return b.layers && b.layers <= moe_shm::kMaxLayers &&
               b.experts && uint64_t(b.experts) * sizeof(int32_t) <= moe_shm::kMaxHeatLdsBytes &&
               b.rank < moe_shm::kMaxEngineRanks &&
               b.run_generation && b.owner_epoch && b.next_credit_sequence &&
               b.max_age_ns && b.issued_watermark >= b.completed_watermark &&
               b.completed_watermark >= b.next_credit_sequence - 1;
    }
    HeatPoll fail(CreditedReject why, HeatDelta* out, std::string* error) {
        fault_ = why;
        accepted_.clear();
        if (out) *out = {};
        if (error) *error = credited_reason(why);
        return HeatPoll::INVALID;
    }

public:
    CreditedEngineSource(CreditedRowReader& reader, CreditedBoundary boundary)
        : reader_(reader), bound_(boundary), last_step_(valid(boundary) ? boundary.layers : 0, 0),
          step_seen_(valid(boundary) ? boundary.layers : 0, 0), configured_(valid(boundary)) {}

    CreditedReject fault() const { return fault_; }
    const std::vector<AcceptedCreditedRow>& last_rows() const { return accepted_; }

    HeatPoll poll(uint64_t monotonic_now_ns, uint64_t /*realtime_now_ns*/,
                  HeatDelta* out, std::string* error) override {
        if (!out || !monotonic_now_ns || !configured_)
            return fail(CreditedReject::CONFIGURATION, out, error);
        *out = {};
        accepted_.clear();
        if (fault_ != CreditedReject::NONE) {
            if (error) *error = credited_reason(fault_);
            return HeatPoll::INVALID; // no silent rebaseline following a loss
        }

        std::vector<CreditedHeatRow> rows;
        const CreditedRead read = reader_.read_batch(&rows);
        if (read != CreditedRead::ROWS) {
            if (read == CreditedRead::EMPTY && rows.empty()) return HeatPoll::UNCHANGED;
            return fail(read == CreditedRead::GAP ? CreditedReject::COVERAGE_GAP :
                        CreditedReject::READER, out, error);
        }
        if (rows.empty()) return fail(CreditedReject::INCOMPLETE, out, error);

        // Stage the entire batch: even a bad trailing row must not publish
        // half a window, alter the next expected sequence, or decay policy heat.
        CreditedBoundary next = bound_;
        auto steps = last_step_;
        auto seen = step_seen_;
        const size_t cells = size_t(bound_.layers) * bound_.experts;
        std::vector<int32_t> weighted(cells, 0);
        std::vector<AcceptedCreditedRow> staged;
        staged.reserve(rows.size());
        uint64_t interval = 0, age = 0;
        for (const CreditedHeatRow& row : rows) {
            if (row.run_generation != next.run_generation ||
                row.owner_epoch != next.owner_epoch || !row.bank_generation)
                return fail(CreditedReject::GENERATION, out, error);
            if (row.rank != next.rank || row.layer >= next.layers)
                return fail(CreditedReject::DOMAIN, out, error);
            if (row.phase != CreditedPhase::PREFILL && row.phase != CreditedPhase::DECODE &&
                row.phase != CreditedPhase::MIXED)
                return fail(CreditedReject::PHASE, out, error);
            if (!row.source_complete || !row.reports)
                return fail(CreditedReject::INCOMPLETE, out, error);
            if (row.credit_sequence < next.next_credit_sequence ||
                row.credited_watermark < next.next_credit_sequence)
                return fail(CreditedReject::REPLAY, out, error);
            if (row.credit_sequence != next.next_credit_sequence ||
                row.credited_watermark != row.credit_sequence ||
                row.dropped_watermark != next.dropped_watermark)
                return fail(CreditedReject::COVERAGE_GAP, out, error);
            if (row.completed_watermark < row.credited_watermark ||
                row.issued_watermark < row.completed_watermark)
                return fail(CreditedReject::INCOMPLETE, out, error);
            if (row.issued_watermark < next.issued_watermark ||
                row.completed_watermark < next.completed_watermark)
                return fail(CreditedReject::COVERAGE_REGRESSION, out, error);
            if (seen[row.layer] && row.source_step <= steps[row.layer])
                return fail(CreditedReject::REPLAY, out, error);
            if (!row.observed_monotonic_ns || row.observed_monotonic_ns > monotonic_now_ns ||
                monotonic_now_ns - row.observed_monotonic_ns > next.max_age_ns)
                return fail(CreditedReject::STALE, out, error);
            if (row.counts.size() != next.experts)
                return fail(CreditedReject::COUNT_SHAPE, out, error);
            uint64_t sum = 0;
            AcceptedCreditedRow credited;
            credited.source_step = row.source_step;
            credited.credit_sequence = row.credit_sequence;
            credited.bank_generation = row.bank_generation;
            credited.rank = row.rank; credited.layer = row.layer;
            credited.reports = row.reports; credited.phase = row.phase;
            credited.weighted_counts.reserve(next.experts);
            for (uint32_t e = 0; e < next.experts; ++e) {
                const int32_t count = row.counts[e];
                if (count < 0) return fail(CreditedReject::COUNT_NEGATIVE, out, error);
                const uint64_t w = uint64_t(count) * row.reports;
                const size_t index = size_t(row.layer) * next.experts + e;
                if (w > INT32_MAX || uint64_t(weighted[index]) + w > INT32_MAX ||
                    sum > UINT64_MAX - w || interval > UINT64_MAX - w)
                    return fail(CreditedReject::WEIGHT_OVERFLOW, out, error);
                sum += w;
                interval += w;
                weighted[index] += int32_t(w);
                credited.weighted_counts.push_back(int32_t(w));
            }
            if (sum != row.weighted_sum)
                return fail(CreditedReject::WEIGHT_SUM, out, error);
            if (row.observation_count < next.observation_count)
                return fail(CreditedReject::OBSERVATION_REGRESSION, out, error);
            if (sum > UINT64_MAX - next.observation_count ||
                row.observation_count != next.observation_count + sum)
                return fail(CreditedReject::OBSERVATION_SUM, out, error);
            age = std::max(age, monotonic_now_ns - row.observed_monotonic_ns);
            next.observation_count = row.observation_count;
            if (next.next_credit_sequence == UINT64_MAX)
                return fail(CreditedReject::COVERAGE_GAP, out, error);
            next.next_credit_sequence++;
            next.issued_watermark = row.issued_watermark;
            next.completed_watermark = row.completed_watermark;
            steps[row.layer] = row.source_step;
            seen[row.layer] = 1;
            staged.push_back(std::move(credited));
        }
        bound_ = next;
        last_step_ = std::move(steps);
        step_seen_ = std::move(seen);
        accepted_ = std::move(staged);
        out->provenance = HeatProvenance::CREDITED_ENGINE;
        out->layers = bound_.layers; out->experts = bound_.experts;
        out->counts = std::move(weighted);
        out->observation_count = bound_.observation_count;
        out->interval_observations = interval;
        out->age_ns = age;
        out->complete_cells = true;
        out->complete_observations = true; // derived only after full validation
        return interval ? HeatPoll::DELTA : HeatPoll::UNCHANGED;
    }
};

} // namespace r3::policy
