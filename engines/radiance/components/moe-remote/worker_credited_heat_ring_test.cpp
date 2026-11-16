#include "worker_credited_heat_ring.h"
#include "worker_policy_input.h"
#include <atomic>
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <thread>
#include <vector>
#include <sys/mman.h>
#include <sys/syscall.h>
#include <unistd.h>

using namespace r3::policy;
namespace ring = r3::policy::heat_ring;
static void check(bool ok, const char* message) {
    if (!ok) { std::fprintf(stderr, "r3_credited_heat_ring_test: %s\n", message); std::exit(1); }
}
struct Mappings {
    size_t bytes;
    void *producer, *consumer;
    explicit Mappings(ring::Shape shape) {
        size_t row = 0;
        check(ring::layout(shape, &row, &bytes), "valid shape");
#ifdef SYS_memfd_create
        int fd = static_cast<int>(syscall(SYS_memfd_create, "credited-ring", 0));
#else
        int fd = -1;
#endif
        check(fd >= 0, "memfd_create");
        check(ftruncate(fd, static_cast<off_t>(bytes)) == 0, "ftruncate");
        producer = mmap(nullptr, bytes, PROT_READ|PROT_WRITE, MAP_SHARED, fd, 0);
        consumer = mmap(nullptr, bytes, PROT_READ|PROT_WRITE, MAP_SHARED, fd, 0);
        close(fd);
        check(producer != MAP_FAILED && consumer != MAP_FAILED &&
              producer != consumer, "independent mmap virtual addresses");
        check(ring::format(producer, bytes, shape), "format");
    }
    ~Mappings() { munmap(producer, bytes); munmap(consumer, bytes); }
};
static CreditedHeatRow row(uint64_t seq, uint32_t experts = 4,
                           uint64_t run = 11, uint64_t epoch = 29) {
    CreditedHeatRow r;
    r.rank = 0; r.layer = uint32_t(seq % 2);
    r.run_generation = run; r.owner_epoch = epoch;
    r.source_step = seq; r.bank_generation = seq;
    r.credit_sequence = seq; r.issued_watermark = seq;
    r.completed_watermark = seq; r.credited_watermark = seq;
    r.observed_monotonic_ns = 999900;
    r.source_complete = true; r.reports = 2; r.phase = CreditedPhase::MIXED;
    r.counts.assign(experts, 1);
    r.weighted_sum = experts * 2;
    r.observation_count = seq * experts * 2;
    return r;
}
static CreditedBoundary boundary() {
    CreditedBoundary b;
    b.rank = 0; b.layers = 2; b.experts = 4;
    b.run_generation = 11; b.owner_epoch = 29;
    b.next_credit_sequence = 1; b.max_age_ns = 2000;
    return b;
}
static void malformed() {
    const ring::Shape shape{0, 2, 4, 3};
    Mappings m(shape);
    ring::Producer p;
    ring::Consumer c;
    ring::Error why{};
    check(!c.attach(static_cast<char*>(m.consumer)+1, m.bytes-1, shape, &why) &&
          why == ring::Error::ALIGN, "reject unaligned mapping");
    check(!c.attach(m.consumer, ring::kPage-1, shape, &why) &&
          why == ring::Error::SIZE, "reject short descriptor");
    check(!c.attach(m.consumer, m.bytes-1, shape, &why) &&
          why == ring::Error::SIZE, "reject truncated slot mapping");
    check(!c.attach(m.consumer, m.bytes, {1,2,4,3}, &why) &&
          why == ring::Error::MISMATCH, "reject wrong rank");
    check(!c.attach(m.consumer, m.bytes, {0,2,5,3}, &why) &&
          why == ring::Error::MISMATCH, "reject expert width");
    check(!c.attach(m.consumer, m.bytes, {0,3,4,3}, &why) &&
          why == ring::Error::MISMATCH, "reject layer count");
    auto* h = static_cast<ring::Header*>(m.producer);
    h->version++;
    check(!c.attach(m.consumer, m.bytes, shape, &why) &&
          why == ring::Error::VERSION, "reject version");
    h->version--;
    h->row_bytes++;
    check(!p.attach(m.producer, m.bytes, shape, &why) &&
          why == ring::Error::MAGIC, "reject row stride");
    h->row_bytes--;
    h->produced.value = UINT64_MAX;
    check(!p.attach(m.producer, m.bytes, shape, &why) &&
          why == ring::Error::CURSOR, "reject cursor overflow");
    h->produced.value = 0;
    h->consumed.value = 1;
    check(!c.attach(m.consumer, m.bytes, shape, &why) &&
          why == ring::Error::CURSOR, "reject inverted cursor");
    h->consumed.value = 0;
    size_t stride, bytes;
    check(!ring::layout({0,2,4,UINT64_MAX}, &stride, &bytes),
          "reject total size overflow");
    check(!ring::layout({0,2,UINT32_MAX,UINT64_MAX}, &stride, &bytes),
          "reject multiplication overflow");
}
static void late_and_policy() {
    const ring::Shape shape{0,2,4,4};
    Mappings m(shape);
    ring::Producer p; ring::Consumer c;
    check(p.attach(m.producer, m.bytes, shape) &&
          c.attach(m.consumer, m.bytes, shape), "attach both mappings");
    std::vector<CreditedHeatRow> got;
    check(c.read_batch(&got) == CreditedRead::EMPTY, "late engine empty");
    check(p.start(1), "new producer generation");
    CreditedEngineSource source(c, boundary());
    HeatDelta delta;
    std::string err;
    check(source.poll(1000000, 0, &delta, &err) == HeatPoll::UNCHANGED,
          "empty reader through policy");
    auto a = row(1), b = row(2);
    check(p.push(a) == ring::Write::STORED &&
          p.push(b) == ring::Write::STORED, "put two credited rows");
    check(source.poll(1000000, 0, &delta, &err) == HeatPoll::DELTA,
          "end-to-end credited source");
    check(delta.complete_observations && delta.complete_cells &&
          delta.interval_observations == 16 && delta.observation_count == 16 &&
          delta.counts == std::vector<int32_t>({2,2,2,2,2,2,2,2}),
          "same counts and exactly-once reports weighting as B1 oracle");
    check(source.last_rows().size() == 2 &&
          source.last_rows()[0].weighted_counts == std::vector<int32_t>({2,2,2,2}) &&
          source.last_rows()[1].phase == CreditedPhase::MIXED,
          "multiplicity and phase preserved");
    check(source.poll(1000000, 0, &delta, &err) == HeatPoll::UNCHANGED,
          "no second delivery");
}
static void overflow_and_restart() {
    const ring::Shape shape{0,2,4,1};
    Mappings m(shape);
    ring::Producer p; ring::Consumer c;
    check(p.attach(m.producer, m.bytes, shape) &&
          c.attach(m.consumer, m.bytes, shape) && p.start(1), "attach small ring");
    check(p.push(row(1)) == ring::Write::STORED, "first slot");
    check(p.push(row(2)) == ring::Write::FULL &&
          p.drops() == 1 && p.gap_number() == 1 && p.last_gap_credit() == 2,
          "full queue drops new row with numbered gap");
    std::vector<CreditedHeatRow> rows;
    CreditedEngineSource policy(c, boundary());
    HeatDelta rejected;
    std::string reason;
    check(policy.poll(1000000, 0, &rejected, &reason) == HeatPoll::INVALID &&
          policy.fault() == CreditedReject::COVERAGE_GAP &&
          rejected.counts.empty(), "B1 fails closed on transport loss");
    check(c.read_batch(&rows) == CreditedRead::ROWS && rows.size() == 1 &&
          rows[0].credit_sequence == 1, "previous immutable slot kept");
    check(p.push(row(3)) == ring::Write::STORED, "producer recovers without allocation");
    check(c.read_batch(&rows) == CreditedRead::ROWS && rows.size() == 1 &&
          rows[0].credit_sequence == 3 && rows[0].dropped_watermark == 1,
          "next row carries dropped watermark");
    check(p.push(row(4)) == ring::Write::STORED, "old generation queued");
    check(p.start(2), "restart generation");
    check(c.read_batch(&rows) == CreditedRead::GAP && rows.empty(),
          "generation switch is explicit gap");
    check(c.read_batch(&rows) == CreditedRead::GAP && rows.empty(),
          "old generation slot cannot be current");
    check(p.push(row(1,4,12,30)) == ring::Write::STORED,
          "new generation reuses drained slot");
    check(c.read_batch(&rows) == CreditedRead::ROWS && rows.size() == 1 &&
          rows[0].run_generation == 12 && rows[0].owner_epoch == 30 &&
          rows[0].dropped_watermark == 0,
          "new generation only");
}
static void threaded_wrap() {
    const ring::Shape shape{0,2,4,8};
    Mappings m(shape);
    ring::Producer p; ring::Consumer c;
    check(p.attach(m.producer, m.bytes, shape) &&
          c.attach(m.consumer, m.bytes, shape) && p.start(7), "threaded attach");
    constexpr uint64_t n = 10000;
    std::atomic<uint64_t> processed{0};
    std::atomic<bool> ok{true};
    std::thread consumer([&] {
        std::vector<CreditedHeatRow> rows;
        while (processed.load() < n && ok.load()) {
            auto status = c.read_batch(&rows);
            if (status == CreditedRead::EMPTY) { std::this_thread::yield(); continue; }
            if (status != CreditedRead::ROWS) { ok = false; break; }
            for (const auto& r : rows) {
                uint64_t expected = processed.fetch_add(1)+1;
                if (r.credit_sequence != expected ||
                    r.counts != std::vector<int32_t>(4,1) ||
                    r.weighted_sum != 8 || r.observation_count != expected*8)
                    ok = false;
            }
        }
    });
    std::thread producer([&] {
        for (uint64_t seq=1; seq<=n; ++seq) {
            auto item = row(seq);
            auto* h = static_cast<ring::Header*>(m.producer);
            while (ring::get(&h->produced.value) - ring::get(&h->consumed.value)
                   == shape.capacity && ok.load()) std::this_thread::yield();
            if (!ok || p.push(item) != ring::Write::STORED) { ok = false; return; }
        }
    });
    producer.join();
    if (!ok.load()) {
        // Even a failing producer must release the consumer promptly.
        ok = false;
    }
    consumer.join();
    check(ok && processed == n && p.drops() == 0, "concurrent wrap copies immutable rows");
}
int main() {
    malformed();
    late_and_policy();
    overflow_and_restart();
    threaded_wrap();
    return 0;
}
