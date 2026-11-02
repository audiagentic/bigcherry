/* GPU-only integration proxy for the stream-ordered ticket binding.
 *
 * The component test cannot directly instantiate the engine's AQL PassTape:
 * its implementation is private to core/device/aql.cpp and the executable,
 * not radiance::abi. A HIP graph captures a fixed kernel argument pointer and
 * replays the identical graph 32 times with intervening live ticket dispatches.
 * This proves the queue/lane protocol on HIP streams, NOT AQL's PlayRun barrier
 * and foreign-packet detection. That last proof needs an engine AQL test.
 *
 * No intermediate device synchronisation, even when two rank-local cells
 * submit concurrently. The negative control drops lane1->lane0 join, with a
 * long delayed lane1 read to expose a ticket overwritten by the next play.
 */
#include "step_ticket_kernel.h"
#include <hip/hip_runtime.h>
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

namespace {
constexpr int kPlays = 32;
constexpr int kRanks = 2;
constexpr int kStages = 4;
constexpr uint64_t kDelayCycles = 20000000ull;

struct Seen {
    uint64_t ticket_id;
    uint32_t epoch;
    uint32_t slot;
};
struct RequestCopy {
    uint64_t ticket_id;
    uint32_t epoch;
    uint32_t slot;
};
static_assert(sizeof(Seen) == 16 && sizeof(RequestCopy) == 16);

__device__ void save(Seen* out, int i, int stage, uint64_t id, uint32_t epoch, uint32_t slot) {
    out[i * kStages + stage] = Seen{id, epoch, slot};
}

__global__ void publish_like(const StepTicket* ticket, RequestCopy* requests, Seen* out) {
    if (blockIdx.x || threadIdx.x) return;
    const StepTicket t = *ticket;
    if (t.ticket_id < 1 || t.ticket_id > kPlays) return;
    const int i = (int)t.ticket_id - 1;
    requests[i] = RequestCopy{t.ticket_id, t.placement_epoch, t.generation_slot};
    save(out, i, 0, t.ticket_id, t.placement_epoch, t.generation_slot);
}

/* The add/finish-like consumers use the captured Request, not the ticket
 * fields. The pass's immutable ticket_id selects the output slot. */
__global__ void request_reader(const StepTicket* ticket, const RequestCopy* requests,
                               Seen* out, int stage) {
    if (blockIdx.x || threadIdx.x) return;
    const uint64_t id = ticket->ticket_id;
    if (id < 1 || id > kPlays) return;
    const int i = (int)id - 1;
    const RequestCopy r = requests[i];
    save(out, i, stage, r.ticket_id, r.epoch, r.slot);
}

/* This deliberately reads the cell late, rather than the request, to test
 * the strongest lane-1 lifetime invariant. */
__global__ void late_lane_reader(const StepTicket* ticket, Seen* out, uint64_t delay) {
    if (blockIdx.x || threadIdx.x) return;
    const uint64_t start = clock64();
    while (clock64() - start < delay) asm volatile("" ::: "memory");
    const StepTicket t = *ticket;
    if (t.ticket_id < 1 || t.ticket_id > kPlays) return;
    save(out, (int)t.ticket_id - 1, 3, t.ticket_id, t.placement_epoch, t.generation_slot);
}

bool hip_ok(hipError_t e, const char* where) {
    if (e == hipSuccess) return true;
    std::fprintf(stderr, "%s: %s\n", where, hipGetErrorString(e));
    return false;
}

struct Rank {
    StepTicket* cell = nullptr;
    RequestCopy* requests = nullptr;
    Seen* observed = nullptr;
    hipStream_t main = nullptr, lane1 = nullptr;
    hipGraph_t graph = nullptr;
    hipGraphExec_t replay = nullptr;
    std::array<hipEvent_t, kPlays> ready{}, joined{};
    void clear() {
        if (main) (void)hipStreamSynchronize(main);
        if (lane1) (void)hipStreamSynchronize(lane1);
        for (auto e : ready) if (e) (void)hipEventDestroy(e);
        for (auto e : joined) if (e) (void)hipEventDestroy(e);
        if (replay) (void)hipGraphExecDestroy(replay);
        if (graph) (void)hipGraphDestroy(graph);
        if (main) (void)hipStreamDestroy(main);
        if (lane1) (void)hipStreamDestroy(lane1);
        if (observed) (void)hipFree(observed);
        if (requests) (void)hipFree(requests);
        if (cell) (void)hipFree(cell);
    }
};

bool init(Rank& r) {
    if (!hip_ok(hipMalloc((void**)&r.cell, sizeof(StepTicket)), "cell allocation") ||
        !hip_ok(hipMalloc((void**)&r.requests, kPlays * sizeof(RequestCopy)), "request allocation") ||
        !hip_ok(hipMalloc((void**)&r.observed, kPlays * kStages * sizeof(Seen)), "observation allocation") ||
        !hip_ok(hipStreamCreateWithFlags(&r.main, hipStreamNonBlocking), "main queue") ||
        !hip_ok(hipStreamCreateWithFlags(&r.lane1, hipStreamNonBlocking), "lane1 queue"))
        return false;
    for (int i=0; i<kPlays; ++i) {
        if (!hip_ok(hipEventCreateWithFlags(&r.ready[i], hipEventDisableTiming), "ready event") ||
            !hip_ok(hipEventCreateWithFlags(&r.joined[i], hipEventDisableTiming), "join event"))
            return false;
    }
    if (!hip_ok(hipStreamBeginCapture(r.main, hipStreamCaptureModeGlobal), "begin capture"))
        return false;
    hipLaunchKernelGGL(publish_like, dim3(1), dim3(1), 0, r.main,
                       r.cell, r.requests, r.observed);
    hipLaunchKernelGGL(request_reader, dim3(1), dim3(1), 0, r.main,
                       r.cell, r.requests, r.observed, 1);
    hipLaunchKernelGGL(request_reader, dim3(1), dim3(1), 0, r.main,
                       r.cell, r.requests, r.observed, 2);
    if (!hip_ok(hipStreamEndCapture(r.main, &r.graph), "end capture") ||
        !hip_ok(hipGraphInstantiate(&r.replay, r.graph, nullptr, nullptr, 0), "graph instantiate"))
        return false;
    return true;
}

StepTicket make_ticket(int i, int rank) {
    StepTicket t{};
    t.ticket_id = (uint64_t)i+1;
    t.placement_epoch = (uint32_t)(10 + ((i+rank)&1));
    t.generation_slot = (uint8_t)((i+rank)&1);
    return t;
}

/* A/B per rank: tickets change but the captured graph and pointers do not. */
bool execute(std::array<Rank,kRanks>& ranks, bool join) {
    for (auto& r : ranks) {
        if (!hip_ok(hipMemset(r.requests, 0, kPlays*sizeof(RequestCopy)), "clear requests") ||
            !hip_ok(hipMemset(r.observed, 0, kPlays*kStages*sizeof(Seen)), "clear observations"))
            return false;
    }
    /* The clears run on the null stream, which the non-blocking rank queues do not order against: finish them before
     * the first play, or a late clear wipes early observations (it did: rank 1, plays 0-1). This is setup, not the
     * replay under test, which still has no synchronisation between plays. */
    if (!hip_ok(hipDeviceSynchronize(), "clears complete"))
        return false;
    for (int i=0; i<kPlays; ++i) for (int k=0; k<kRanks; ++k) {
        Rank& r = ranks[k];
        StepTicket t = make_ticket(i,k);
        /* The setter and graph launch are asynchronous. No event or sync
         * separates this loop's iterations or the two rank-local cells. */
        hipLaunchKernelGGL(mailbox_set_ticket_kernel, dim3(1), dim3(1), 0, r.main,
                           r.cell, t);
        if (!hip_ok(hipGraphLaunch(r.replay, r.main), "replay") ||
            !hip_ok(hipEventRecord(r.ready[i], r.main), "ready record") ||
            !hip_ok(hipStreamWaitEvent(r.lane1, r.ready[i], 0), "lane1 wait"))
            return false;
        hipLaunchKernelGGL(late_lane_reader, dim3(1), dim3(1), 0, r.lane1,
                           r.cell, r.observed, join ? 10000ull : kDelayCycles);
        if (!hip_ok(hipEventRecord(r.joined[i], r.lane1), "join record"))
            return false;
        if (join && !hip_ok(hipStreamWaitEvent(r.main, r.joined[i], 0), "join wait"))
            return false;
    }
    for (auto& r : ranks) {
        if (!hip_ok(hipStreamSynchronize(r.main), "final main sync") ||
            !hip_ok(hipStreamSynchronize(r.lane1), "final lane1 sync"))
            return false;
    }
    return true;
}

bool inspect(std::array<Rank,kRanks>& ranks, bool join, int* mismatches) {
    *mismatches = 0;
    for (int k=0; k<kRanks; ++k) {
        std::array<Seen,kPlays*kStages> observed{};
        if (!hip_ok(hipMemcpy(observed.data(), ranks[k].observed, sizeof(observed),
                              hipMemcpyDeviceToHost), "download observations"))
            return false;
        for (int i=0; i<kPlays; ++i) {
            const StepTicket t = make_ticket(i,k);
            for (int stage=0; stage<kStages; ++stage) {
                const Seen& got = observed[i*kStages+stage];
                if (got.ticket_id != t.ticket_id || got.epoch != t.placement_epoch ||
                    got.slot != t.generation_slot) {
                    ++*mismatches;
                    if (*mismatches <= 8)
                        std::fprintf(stderr,"rank=%d play=%d stage=%d expected=(%llu,%u,%u) got=(%llu,%u,%u)\n",
                            k,i,stage,(unsigned long long)t.ticket_id,t.placement_epoch,
                            (unsigned)t.generation_slot,(unsigned long long)got.ticket_id,
                            got.epoch,got.slot);
                }
            }
        }
    }
    std::printf("%s: %d plays x %d ranks, mismatches=%d\n",
                join ? "joined" : "negative-unjoined",kPlays,kRanks,*mismatches);
    return true;
}
} // namespace

int main() {
    int device_count = 0;
    if (!hip_ok(hipGetDeviceCount(&device_count), "HIP device discovery") || device_count < 1) {
        std::fprintf(stderr, "this is a GPU-only test; no device available\n");
        return 2;
    }
    std::array<Rank,kRanks> ranks{};
    bool ok = true;
    for (auto& r : ranks) if (!(ok=init(r))) break;
    int positive=0, negative=0;
    if (ok) ok=execute(ranks,true) && inspect(ranks,true,&positive) && positive==0;
    if (ok) {
        ok=execute(ranks,false) && inspect(ranks,false,&negative);
        if (ok && negative==0)
            std::fprintf(stderr, "negative control did not reproduce on this GPU; no claim of detection\n");
    }
    for (auto& r : ranks) r.clear();
    return ok ? 0 : 1;
}
