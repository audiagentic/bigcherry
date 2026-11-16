#include "worker_credited_heat_source.h"
#include "worker_policy_input.h"

#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <string>
#include <vector>
#include <ctime>
#include <unistd.h>

using namespace r3::policy;

static void check(bool ok, const char* why) {
    if (!ok) { std::fprintf(stderr, "r3_credited_heat_source_test: %s\n", why); std::exit(1); }
}

struct Queue final : CreditedRowReader {
    std::deque<std::vector<CreditedHeatRow>> batches;
    CreditedRead status = CreditedRead::EMPTY;
    CreditedRead read_batch(std::vector<CreditedHeatRow>* out) override {
        if (status != CreditedRead::EMPTY) return status;
        if (batches.empty()) return CreditedRead::EMPTY;
        *out = std::move(batches.front());
        batches.pop_front();
        return CreditedRead::ROWS;
    }
    void push(std::vector<CreditedHeatRow> batch) { batches.push_back(std::move(batch)); }
};

static CreditedBoundary boundary(uint64_t previous = 0) {
    CreditedBoundary b;
    b.rank = 0; b.layers = 2; b.experts = 4;
    b.run_generation = 11; b.owner_epoch = 29;
    b.next_credit_sequence = 1; b.observation_count = previous;
    b.max_age_ns = 2000;
    return b;
}

static CreditedHeatRow row(uint64_t seq, uint64_t step, uint32_t layer,
                           CreditedPhase phase, uint32_t reports,
                           std::vector<int32_t> raw, uint64_t previous) {
    CreditedHeatRow r;
    r.run_generation = 11; r.owner_epoch = 29;
    r.bank_generation = seq; r.source_step = step; r.credit_sequence = seq;
    r.issued_watermark = seq; r.completed_watermark = seq;
    r.credited_watermark = seq; r.dropped_watermark = 0;
    r.observed_monotonic_ns = 999900;
    r.source_complete = true;
    r.rank = 0; r.layer = layer; r.phase = phase; r.reports = reports;
    r.counts = std::move(raw);
    for (int32_t x : r.counts) r.weighted_sum += uint64_t(x) * reports;
    r.observation_count = previous + r.weighted_sum;
    return r;
}

static uint64_t realtime() {
    timespec t{};
    check(clock_gettime(CLOCK_REALTIME, &t) == 0, "clock_gettime");
    return uint64_t(t.tv_sec) * 1000000000ULL + uint64_t(t.tv_nsec);
}

struct FileFixture {
    std::string dir, path;
    std::vector<uint32_t> cumulative;
    FileFixture() : cumulative(size_t(r3::moe_shm::kMaxLayers) * 4, 0) {
        const char* tmpdir = std::getenv("TMPDIR");
        const std::string tmp = std::string(tmpdir && *tmpdir ? tmpdir : "/tmp") + "/r3credit-XXXXXX";
        std::vector<char> name(tmp.begin(), tmp.end()); name.push_back('\0');
        char* made = ::mkdtemp(name.data());
        check(made != nullptr, "mkdtemp");
        dir = made; path = dir + "/heat.rank0";
    }
    ~FileFixture() {
        ::unlink(path.c_str()); ::unlink((path + ".tmp").c_str()); ::rmdir(dir.c_str());
    }
    void write() {
        const std::string tmp = path + ".tmp";
        FILE* f = std::fopen(tmp.c_str(), "wb");
        check(f != nullptr, "open file");
        check(std::fwrite(cumulative.data(), sizeof(uint32_t), cumulative.size(), f) ==
              cumulative.size(), "write file");
        check(std::fclose(f) == 0 && std::rename(tmp.c_str(), path.c_str()) == 0,
              "atomic histogram replacement");
    }
    void add(const CreditedHeatRow& r) {
        for (uint32_t e = 0; e < 4; ++e)
            cumulative[size_t(r.layer) * 4 + e] += uint32_t(r.counts[e]) * r.reports;
    }
};

static void compare(const HeatDelta& file, const HeatDelta& credited,
                    r3::heat::Heat* old_heat, r3::heat::Heat* new_heat,
                    uint64_t* old_used, uint64_t* new_used) {
    check(file.counts == credited.counts, "exact per-(layer,expert) deltas");
    check(file.interval_observations == credited.interval_observations &&
          file.observation_count == credited.observation_count, "weighted observation parity");
    check(file.provenance == HeatProvenance::ENGINE_HISTOGRAM_FILE &&
          !file.complete_observations &&
          credited.provenance == HeatProvenance::CREDITED_ENGINE &&
          credited.complete_observations && credited.complete_cells,
          "honest, distinct source provenance");
    check(consume_heat(file, old_heat, old_used, true), "consume old interval");
    check(consume_heat(credited, new_heat, new_used), "consume credited interval");
    check(old_heat->counts == new_heat->counts && old_heat->samples == new_heat->samples,
          "identical policy decay, weighted scores and samples on matched windows");
}

static void parity_oracle() {
    FileFixture file;
    HistogramFileSource legacy(file.path, 2, 4);
    Queue q;
    CreditedEngineSource credited(q, boundary());
    HeatDelta old_delta, new_delta;
    std::string err;
    file.write();
    check(legacy.poll(1000000, realtime(), &old_delta, &err) == HeatPoll::BASELINE,
          "old file baseline");
    check(credited.poll(1000000, 0, &new_delta, &err) == HeatPoll::UNCHANGED,
          "empty credited queue unchanged");

    // Layer 1 is a protected/MTP fixture layer. Protection affects placement,
    // not visibility of model-global routing evidence. A mixed phase is not
    // split or guessed; reports=2 and reports=4 multiply the last bank once.
    auto a = row(1, 10, 0, CreditedPhase::PREFILL, 1, {2, 0, 1, 0}, 0);
    auto b = row(2, 10, 1, CreditedPhase::MIXED, 2, {0, 3, 1, 0}, a.observation_count);
    q.push({a, b});
    file.add(a); file.add(b); file.write();
    check(credited.poll(1000000, 0, &new_delta, &err) == HeatPoll::DELTA,
          "first credited interval");
    check(legacy.poll(1000000, realtime(), &old_delta, &err) == HeatPoll::DELTA,
          "first legacy interval");
    r3::heat::Heat old_heat(2, 4, 1000), new_heat(2, 4, 1000);
    uint64_t old_used = 0, new_used = 0;
    compare(old_delta, new_delta, &old_heat, &new_heat, &old_used, &new_used);
    check(new_delta.counts[4 + 1] == 6 && new_delta.counts[4 + 2] == 2,
          "protected MTP layer preserved, multiplicity=2");
    check(credited.last_rows().size() == 2 &&
          credited.last_rows()[1].reports == 2 &&
          credited.last_rows()[1].phase == CreditedPhase::MIXED &&
          credited.last_rows()[1].source_step == 10 &&
          credited.last_rows()[1].weighted_counts == std::vector<int32_t>({0, 6, 2, 0}),
          "matched rank/step/phase/multiplicity and expert weights");

    auto c = row(3, 11, 0, CreditedPhase::DECODE, 4, {1, 0, 2, 1}, b.observation_count);
    auto d = row(4, 12, 1, CreditedPhase::PREFILL, 1, {1, 2, 0, 2}, c.observation_count);
    q.push({c, d});
    file.add(c); file.add(d); file.write();
    check(credited.poll(1000000, 0, &new_delta, &err) == HeatPoll::DELTA,
          "second credited interval");
    check(legacy.poll(1000000, realtime(), &old_delta, &err) == HeatPoll::DELTA,
          "second legacy interval");
    compare(old_delta, new_delta, &old_heat, &new_heat, &old_used, &new_used);
    check(credited.last_rows()[0].weighted_counts ==
          std::vector<int32_t>({4, 0, 8, 4}) &&
          credited.last_rows()[0].reports == 4 &&
          new_delta.observation_count == 32, "MTP four-report exact weighting");
    check(credited.poll(1000000, 0, &new_delta, &err) == HeatPoll::UNCHANGED &&
          credited.last_rows().empty(), "old weights never replayed on empty poll");
}

static void expect(CreditedHeatRow test, CreditedReject reason,
                   CreditedBoundary b = boundary()) {
    Queue q;
    q.push({std::move(test)});
    CreditedEngineSource source(q, b);
    HeatDelta delta;
    std::string err;
    check(source.poll(1000000, 0, &delta, &err) == HeatPoll::INVALID &&
          source.fault() == reason && err == credited_reason(reason),
          "typed rejection from invalid credited row");
    check(!delta.complete_observations && delta.counts.empty() &&
          source.last_rows().empty(), "rejected row cannot advertise coverage or output");
    check(source.poll(1000000, 0, &delta, &err) == HeatPoll::INVALID,
          "invalid source is permanently faulted");
}

static void rejection_oracle() {
    const auto good = row(1, 10, 0, CreditedPhase::PREFILL, 2, {1, 1, 0, 0}, 0);
    auto bad = good; bad.run_generation++; expect(bad, CreditedReject::GENERATION);
    bad = good; bad.owner_epoch++; expect(bad, CreditedReject::GENERATION);
    bad = good; bad.bank_generation = 0; expect(bad, CreditedReject::GENERATION);
    bad = good; bad.rank++; expect(bad, CreditedReject::DOMAIN);
    bad = good; bad.layer = 2; expect(bad, CreditedReject::DOMAIN);
    bad = good; bad.phase = static_cast<CreditedPhase>(255); expect(bad, CreditedReject::PHASE);
    bad = good; bad.source_complete = false; expect(bad, CreditedReject::INCOMPLETE);
    bad = good; bad.reports = 0; expect(bad, CreditedReject::INCOMPLETE);
    bad = good; bad.credit_sequence = 2; bad.credited_watermark = 2;
    expect(bad, CreditedReject::COVERAGE_GAP);
    bad = good; bad.credited_watermark = 0; expect(bad, CreditedReject::REPLAY);
    bad = good; bad.dropped_watermark = 1; expect(bad, CreditedReject::COVERAGE_GAP);
    bad = good; bad.completed_watermark = 0; expect(bad, CreditedReject::INCOMPLETE);
    bad = good; bad.issued_watermark = 0; expect(bad, CreditedReject::INCOMPLETE);
    bad = good; bad.observed_monotonic_ns = 900000; expect(bad, CreditedReject::STALE);
    bad = good; bad.observed_monotonic_ns = 1000001; expect(bad, CreditedReject::STALE);
    bad = good; bad.observed_monotonic_ns = 0; expect(bad, CreditedReject::STALE);
    bad = good; bad.counts.pop_back(); expect(bad, CreditedReject::COUNT_SHAPE);
    bad = good; bad.counts[0] = -1; expect(bad, CreditedReject::COUNT_NEGATIVE);
    bad = good; bad.counts[0] = INT32_MAX; expect(bad, CreditedReject::WEIGHT_OVERFLOW);
    bad = good; bad.weighted_sum++; expect(bad, CreditedReject::WEIGHT_SUM);
    bad = good; bad.observation_count++; expect(bad, CreditedReject::OBSERVATION_SUM);
    bad = good; bad.observation_count = 9; expect(bad, CreditedReject::OBSERVATION_REGRESSION,
                                                   boundary(10));
    // A forged complete flag cannot repair a missing row, broken credit
    // watermark or a source-reported queue loss.
    for (int scenario = 0; scenario < 3; ++scenario) {
        auto r = good;
        if (scenario == 0) { r.credit_sequence = 3; r.credited_watermark = 3; }
        if (scenario == 1) r.completed_watermark = 0;
        if (scenario == 2) r.dropped_watermark = 2;
        r.source_complete = true;
        expect(r, scenario == 0 ? CreditedReject::COVERAGE_GAP :
               scenario == 1 ? CreditedReject::INCOMPLETE : CreditedReject::COVERAGE_GAP);
    }

    Queue gone;
    gone.status = CreditedRead::GAP;
    CreditedEngineSource gap(gone, boundary());
    HeatDelta delta; std::string err;
    check(gap.poll(1000000, 0, &delta, &err) == HeatPoll::INVALID &&
          gap.fault() == CreditedReject::COVERAGE_GAP, "reader gap is not a baseline");
    Queue corrupt;
    corrupt.status = CreditedRead::INVALID;
    CreditedEngineSource bad_reader(corrupt, boundary());
    check(bad_reader.poll(1000000, 0, &delta, &err) == HeatPoll::INVALID &&
          bad_reader.fault() == CreditedReject::READER, "invalid transport refused");
    auto config = boundary(); config.experts = 0;
    Queue q;
    CreditedEngineSource invalid(q, config);
    check(invalid.poll(1000000, 0, &delta, &err) == HeatPoll::INVALID &&
          invalid.fault() == CreditedReject::CONFIGURATION, "invalid domain refused");
    auto oversized = boundary(); oversized.layers = UINT32_MAX;
    CreditedEngineSource invalid_shape(q, oversized);
    check(invalid_shape.poll(1000000, 0, &delta, &err) == HeatPoll::INVALID &&
          invalid_shape.fault() == CreditedReject::CONFIGURATION,
          "invalid layer dimension rejected without allocation");
    CreditedEngineSource invalid_time(q, boundary());
    check(invalid_time.poll(0, 0, &delta, &err) == HeatPoll::INVALID, "missing clock refused");
}

static void batch_atomicity_and_replay() {
    const auto a = row(1, 5, 0, CreditedPhase::PREFILL, 1, {1, 0, 0, 0}, 0);
    auto wrong = row(2, 6, 0, CreditedPhase::DECODE, 1, {1, 0, 0, 0}, 1);
    wrong.weighted_sum++;
    Queue q;
    q.push({a, wrong});
    CreditedEngineSource src(q, boundary());
    HeatDelta delta; std::string err;
    check(src.poll(1000000, 0, &delta, &err) == HeatPoll::INVALID &&
          src.last_rows().empty() && delta.counts.empty(),
          "invalid trailing row rejects whole poll atomically");

    Queue replay;
    replay.push({a}); replay.push({a});
    CreditedEngineSource duplicate(replay, boundary());
    check(duplicate.poll(1000000, 0, &delta, &err) == HeatPoll::DELTA,
          "first row valid");
    check(duplicate.poll(1000000, 0, &delta, &err) == HeatPoll::INVALID &&
          duplicate.fault() == CreditedReject::REPLAY, "cross-poll replay rejected");
    Queue same_batch;
    same_batch.push({a, a});
    CreditedEngineSource dup_batch(same_batch, boundary());
    check(dup_batch.poll(1000000, 0, &delta, &err) == HeatPoll::INVALID &&
          dup_batch.fault() == CreditedReject::REPLAY, "same-batch duplicate rejected");

    Queue repeated_step;
    repeated_step.push({a, row(2, 5, 0, CreditedPhase::DECODE, 1, {1, 0, 0, 0}, 1)});
    CreditedEngineSource repeated(repeated_step, boundary());
    check(repeated.poll(1000000, 0, &delta, &err) == HeatPoll::INVALID &&
          repeated.fault() == CreditedReject::REPLAY,
          "same layer and step cannot be credited twice even with different bank generations");

    Queue watermark;
    auto first = a;
    first.issued_watermark = 5; first.completed_watermark = 5;
    auto second = row(2, 6, 0, CreditedPhase::DECODE, 1, {1, 0, 0, 0}, 1);
    second.issued_watermark = 4; second.completed_watermark = 4;
    watermark.push({first, second});
    CreditedEngineSource regression(watermark, boundary());
    check(regression.poll(1000000, 0, &delta, &err) == HeatPoll::INVALID &&
          regression.fault() == CreditedReject::COVERAGE_REGRESSION,
          "source watermarks cannot regress");

    Queue zeros;
    const auto empty = row(1, 1, 0, CreditedPhase::PREFILL, 1, {0, 0, 0, 0}, 0);
    zeros.push({empty});
    zeros.push({row(2, 2, 0, CreditedPhase::DECODE, 1, {0, 0, 1, 0}, 0)});
    CreditedEngineSource sparse(zeros, boundary());
    check(sparse.poll(1000000, 0, &delta, &err) == HeatPoll::UNCHANGED &&
          delta.complete_observations && delta.observation_count == 0,
          "zero-route credited row advances coverage but not heat");
    check(sparse.poll(1000000, 0, &delta, &err) == HeatPoll::DELTA &&
          delta.interval_observations == 1 && delta.counts[2] == 1,
          "next credited row after zero route remains contiguous");
}

int main() {
    parity_oracle();
    rejection_oracle();
    batch_atomicity_and_replay();
    std::puts("r3_credited_heat_source_test: ok");
}
