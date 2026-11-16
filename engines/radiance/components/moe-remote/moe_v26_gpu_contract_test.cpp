#include "moe_v26_gpu_contract.h"
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

namespace v = r3::moe_shm::v26;
namespace gpu = r3::moe_shm::v26::gpu;
using r3::moe_shm::LayoutShape;

static void check(bool ok, const char* label) {
    if (!ok) {
        std::fprintf(stderr, "r3_v26_gpu_contract_test: FAILED %s\n", label);
        std::exit(1);
    }
}
static void exercise(const LayoutShape& shape, uint32_t layers, v::Topology t) {
    v::Layout l{};
    check(v::layout_for(shape, layers, t, &l) == v::Refusal::None,
          "v26 layout accepted");
    gpu::Scratch scratch{};
    check(gpu::scratch_for(t, shape.max_rows, &scratch) == gpu::Refusal::None,
          "scratch geometry accepted");
    check(scratch.sequence_count == l.counts.lanes &&
          scratch.slot_flags == l.counts.slots &&
          scratch.combine_word == l.counts.slots &&
          scratch.chunk_mirror_begin == l.counts.slots + 1u &&
          scratch.ticket_begin == scratch.chunk_mirror_begin + l.counts.chunks &&
          scratch.published_words == scratch.ticket_begin +
                                     sizeof(gpu::TicketWords) / sizeof(uint32_t),
          "separate slot flags, combine, chunk mirrors and ticket words");
    check(scratch.consumed_records == l.counts.lanes + 1 &&
          scratch.helper_words == t.helpers &&
          scratch.sequence_bytes == uint64_t(l.counts.lanes) * sizeof(uint32_t) &&
          scratch.published_bytes == uint64_t(scratch.published_words) * sizeof(uint32_t) &&
          scratch.consumed_bytes == uint64_t(l.counts.lanes + 1) *
                                    sizeof(r3::moe_shm::ConsumedRecord) &&
          scratch.helper_bytes == uint64_t(t.helpers) * sizeof(uint32_t),
          "counted rank-local storage bytes include all consumers");

    for (uint32_t cap : {1u, 2u, 3u, 32u, 64u, 256u}) {
        gpu::FinishLaunch launch{};
        const auto decision = gpu::finish_launch(t, cap, &launch);
        if (uint64_t(l.counts.slots) + 1u > cap) {
            check(decision == gpu::Refusal::InvalidBlockLimit,
                  "unsafe multi-block finish must refuse before seal");
            continue;
        }
        check(decision == gpu::Refusal::None &&
              launch.work_items == l.counts.slots + 1 &&
              launch.threads == launch.work_items && launch.blocks == 1,
              "single-block finish follows device limit");
        std::vector<uint8_t> visited(launch.work_items);
        for (uint32_t block = 0; block < launch.blocks; ++block)
            for (uint32_t thread = 0; thread < launch.threads; ++thread) {
                const uint64_t index = uint64_t(block) * launch.threads + thread;
                if (index >= launch.work_items) continue;
                check(!visited[index], "finish never processes a slot twice");
                visited[index] = 1;
            }
        for (uint8_t n : visited) check(n == 1, "finish covers every slot and combine");
    }
    for (uint32_t helper = 0; helper < t.helpers; ++helper) {
        for (uint32_t rank = 0; rank < t.ranks; ++rank) {
            const uint32_t layer = rank;
            const uint32_t slot = helper * t.ranks + rank;
            const uint32_t cls = r3::moe_shm::kSmallClasses;
            const uint32_t lane = cls * l.counts.slots + slot;
            v::Request r{};
            check(gpu::populate_request(&r, l, shape, t, layer, layers, helper,
                                        cls, 0, 1, shape.top_k, 1, 0, 1),
                  "producer encodes a complete single-result v26 request");
            check(r.slot == slot && r.consumer == rank && r.helper == helper &&
                  r.world == t.ranks &&
                  r.activation_offset == v::element_offset(l, v::Region::Activations, rank) &&
                  r.result_offset == v::element_offset(l, v::Region::ResultPlanes, slot) &&
                  r.result_stride == shape.hidden * sizeof(uint16_t) &&
                  gpu::request_identity_valid(r, l, shape, t, layers, lane),
                  "producer/consumer share the exact identity and result plane");
            uint32_t combined = UINT32_MAX;
            check(v::combine_for_rank(t, rank, &combined) &&
                  r.result_offset != v::element_offset(l, v::Region::ResultPlanes, combined),
                  "v26 consumer result never aliases dedicated combine plane");
            auto bad = r;
            bad.consumer = (rank + 1) % t.ranks;
            if (t.ranks == 1) bad.consumer = 1;
            check(!gpu::request_identity_valid(bad, l, shape, t, layers, lane),
                  "consumer mismatch refused");
            bad = r; bad.helper = t.helpers;
            check(!gpu::request_identity_valid(bad, l, shape, t, layers, lane),
                  "wrong helper refused");
            bad = r; bad.world = t.ranks + 1;
            check(!gpu::request_identity_valid(bad, l, shape, t, layers, lane),
                  "wrong engine width refused");
            bad = r; bad.result_offset = v::element_offset(l, v::Region::ResultPlanes, combined);
            check(!gpu::request_identity_valid(bad, l, shape, t, layers, lane),
                  "combined output cannot masquerade as a slot result");
            bad = r; bad.result_stride = shape.hidden;
            check(!gpu::request_identity_valid(bad, l, shape, t, layers, lane),
                  "result stride mismatch refused");
            bad = r; bad.route_offset += sizeof(r3::moe_shm::Route);
            check(!gpu::request_identity_valid(bad, l, shape, t, layers, lane),
                  "route offset mismatch refused");
            bad = r; bad.activation_q_offset += sizeof(uint8_t);
            check(!gpu::request_identity_valid(bad, l, shape, t, layers, lane),
                  "activation staging mismatch refused");
            bad = r; bad.activation_s_offset += sizeof(float);
            check(!gpu::request_identity_valid(bad, l, shape, t, layers, lane),
                  "activation scales mismatch refused");
            bad = r; bad.generation_slot = v::kGenerationSlots;
            check(!gpu::request_identity_valid(bad, l, shape, t, layers, lane),
                  "generation mismatch refused");
            bad = r; bad.lane = (lane + 1) % l.counts.lanes;
            check(!gpu::request_identity_valid(bad, l, shape, t, layers, lane),
                  "lane index mismatch refused");
            bad = r; bad.lane_slot = 1;
            check(!gpu::request_identity_valid(bad, l, shape, t, layers, lane),
                  "lane ring index mismatch refused");
            bad = r; bad.placement_epoch = 0;
            check(!gpu::request_identity_valid(bad, l, shape, t, layers, lane),
                  "epoch zero refused");
            bad = r; bad.route_count = shape.top_k + 1;
            check(!gpu::request_identity_valid(bad, l, shape, t, layers, lane),
                  "route overflow refused");
        }
    }
}
static std::string source(const char* file) {
    std::ifstream in(std::filesystem::path(__FILE__).parent_path() / file);
    check(bool(in), "live HIP/host consumer source exists");
    return {std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>()};
}
static bool has(const std::string& text, const char* token) {
    return text.find(token) != std::string::npos;
}
static bool live_contract(const std::string& hip, const std::string& host) {
    for (const auto* bad : {"kMaxSlots", "kMaxHelpers", "kMaxEngineRanks",
                             "kLanes", "kResultPlanes", "kRingSlots", "ConsumedRecords"})
        if (has(hip, bad) || has(host, bad)) return false;
    // Deliberately match the *live* GPU path, not only the offline geometry.
    for (const auto* required : {
            "q->result_offset = h->result_offset[slot]",
            "combine_index(rank, h->slot_count)",
            "rec[lanes].sequence = cg", "rec[lane]",
            "h->lane[lane].consumed_sequence",
            "h->combine[rank].result_consumed_generation",
            "published_ticket_offset(h->slot_count, h->max_rows)",
            "published[h->slot_count]",
            "mailbox_add_finish_kernel, dim3(1), dim3(d->finish_threads)"})
        if (!has(hip, required)) return false;
    for (const auto* required : {
            "scratch.sequence_bytes", "scratch.published_bytes",
            "scratch.consumed_bytes", "gpu::scratch_for",
            "gpu::finish_launch", "s.dev.finish_threads = finish.threads"})
        if (!has(host, required)) return false;
    return true;
}
static void live_contract_mutation_guards() {
    const std::string hip = source("moe_mailbox_ops.hip");
    const std::string host = source("moe_remote_hook.cpp");
    check(live_contract(hip, host), "real live v26 HIP/host paths are counted");
    auto change = [&](const std::string& src, const char* from, const char* to) {
        auto mutated = src;
        const auto at = mutated.find(from);
        check(at != std::string::npos, "expected live contract anchor");
        mutated.replace(at, std::strlen(from), to);
        return mutated;
    };
    check(!live_contract(change(hip, "published[h->slot_count]", "published[kMaxSlots]"), host),
          "mutation: old compiled slot ceiling is caught");
    check(!live_contract(change(hip, "rec[lanes].sequence = cg", "rec->combine_sequence = cg"), host),
          "mutation: old ACK record base is caught");
    check(!live_contract(change(hip, "q->result_offset = h->result_offset[slot]",
                                "q->result_offset = h->result_offset[result_index(slot, rank)]"), host),
          "mutation: old result plane index is caught");
    check(!live_contract(hip, change(host, "scratch.published_bytes",
                                     "(kMaxSlots + 1u) * sizeof(uint32_t)")),
          "mutation: old fixed GPU scratch allocation is caught");
}

int main() {
    live_contract_mutation_guards();
    const v::Topology topologies[] = {
        {1,1}, {2,1}, {1,2}, {2,2}, {1,3}, {2,3}, {3,2}, {4,1}
    };
    struct Case { LayoutShape shape; uint32_t layers; };
    const Case cases[] = {
        {{2560,10,512,64,128,640},49},
        {{2048,4,256,64,128,768},32},
        {{2560,16,1024,64,128,3072},64}
    };
    for (const auto& c : cases)
        for (const auto& t : topologies)
            exercise(c.shape, c.layers, t);
    gpu::Scratch scratch{};
    gpu::FinishLaunch launch{};
    check(gpu::scratch_for({0,1}, 256, &scratch) == gpu::Refusal::InvalidTopology &&
          gpu::scratch_for({1,1}, 0, &scratch) == gpu::Refusal::InvalidTopology &&
          gpu::scratch_for({1,1}, 256, nullptr) == gpu::Refusal::InvalidTopology,
          "bad scratch shapes refused");
    check(gpu::finish_launch({1,3}, 0, &launch) == gpu::Refusal::InvalidBlockLimit &&
          gpu::finish_launch({1,0}, 64, &launch) == gpu::Refusal::InvalidTopology &&
          gpu::finish_launch({1,3}, 64, nullptr) == gpu::Refusal::InvalidTopology,
          "invalid device and rank limits refused");
    std::puts("r3_v26_gpu_contract_test: PASS");
}
