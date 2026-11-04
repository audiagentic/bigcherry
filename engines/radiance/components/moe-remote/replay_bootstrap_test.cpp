#include "worker_bootstrap.h"
#include "worker_slot_pool.h"
#include "worker_generation_tables.h"

#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <new>
#include <vector>
#include <set>
#include <sys/mman.h>

using namespace r3::moe_shm;

static void check(bool ok, const char* what) {
    if (!ok) { std::fprintf(stderr, "replay_bootstrap_test: FAIL %s\n", what); std::exit(1); }
}
struct Selected {
    std::vector<std::pair<uint32_t, uint32_t>> expert;
};
struct FakeAllocator {
    bool allocate(uint64_t bytes, void** out) {
        if (!out || !bytes || bytes > SIZE_MAX) return false;
        *out = new (std::nothrow) uint8_t[size_t(bytes)];
        return *out != nullptr;
    }
    void release(void* p) { delete[] static_cast<uint8_t*>(p); }
};
struct FakeRunner {
    static constexpr uint64_t stride = sizeof(uint64_t) * 8;
    FakeAllocator allocator;
    r3::worker_slots::Pool<FakeAllocator> pool;
    r3::worker_tables::Model<uint32_t> banks;
    const Selected& selected;
    uint32_t layers_, experts_;
    std::vector<int32_t> loaded_local;
    uint32_t loads = 0;
    bool fail_load = false;

    FakeRunner(const Selected& s, uint32_t layers, uint32_t experts)
        : pool(stride * s.expert.size(), stride, uint32_t(s.expert.size()), allocator),
          banks(layers, experts), selected(s),
          layers_(layers), experts_(experts),
          loaded_local(size_t(layers) * experts, -1) {}

    bool prepare_pool() { return pool.prepare(); }
    bool load_layer(uint32_t layer, const int32_t* slots) {
        if (fail_load || !pool.valid() || !slots || loads++) return false;
        std::vector<uint32_t> bindings(size_t(layers_) * experts_);
        for (auto [l, e] : selected.expert) {
            if (l != layer || slots[e] < 0) continue;
            uint32_t id = UINT32_MAX;
            if (pool.bootstrap(l * experts_ + e, stride, 1, &id) != r3::residency::Result::OK)
                return false;
            if (!pool.address(id)) return false;
            bindings[size_t(l) * experts_ + e] = id + 1;
            loaded_local[size_t(l) * experts_ + e] = slots[e];
        }
        for (uint32_t gen = 0; gen < r3::worker_tables::kGenerations; ++gen)
            if (!banks.build(gen, bindings)) return false;
        return true;
    }
    bool resident(uint32_t layer, uint32_t expert, uint32_t rank, uint32_t local) const {
        if (rank >= 2 || layer >= layers_ || expert >= experts_ ||
            loaded_local[size_t(layer) * experts_ + expert] != int32_t(local)) return false;
        uint32_t first = 0, second = 0;
        if (!banks.select(0, layer, expert, &first) ||
            !banks.select(1, layer, expert, &second) ||
            !first || first != second) return false;
        for (const auto& slot : pool.ledger().slots())
            if (slot.expert == layer * experts_ + expert &&
                slot.state == r3::residency::State::ACTIVE &&
                slot.completed_planes == 15 && first == slot.id + 1)
                return true;
        return false;
    }
    bool before_launch(const void* map, uint32_t layer, uint32_t expert) const {
        if (!bootstrap_generation_sealed(map) || loads != 1) return false;
        uint32_t first = 0, second = 0;
        return banks.select(0, layer, expert, &first) &&
               banks.select(1, layer, expert, &second) &&
               first == second && first != 0;
    }
};
static void run(bool bad_load) {
    const LayoutShape shape{128, 2, 4, 8, 128, 64};
    Layout layout{};
    check(layout_for(shape, 2, slots_for(1), &layout), "layout");
    void* map = mmap(nullptr, layout.bytes, PROT_READ | PROT_WRITE,
                     MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    check(map != MAP_FAILED, "mmap");
    // One owner survives on the replayed layer, one is protected and one
    // belongs to a layer which this isolated replay does not declare.
    const Selected selected{{{0, 0}, {0, 1}, {1, 2}}};
    const int32_t runtime_slots[]{0, -1, 1, 2};
    const uint32_t small_rows[]{8};
    FakeRunner device(selected, 2, shape.experts);
    check(device.prepare_pool(), "production slot pool preparation");
    device.fail_load = bad_load;
    PlacementControl control{};
    const char* why = nullptr;
    const bool ok = r3::epoch::prepare_compare_bootstrap(
        map, 2, shape, selected, 0, runtime_slots, 1, small_rows,
        [&](uint32_t layer, const int32_t* slots) {
            return device.load_layer(layer, slots);
        }, &control, &why);
    check(ok == !bad_load, bad_load ? "failed upload refuses seal" : "full replay bootstrap");
    if (!bad_load) {
        const Header* h = static_cast<const Header*>(map);
        check(load(&h->helper_count) == 1 && load(&h->layer_map_consumed[0]) == 1,
              "same publication and consumed ACK as serving");
        check(load(&h->bootstrap_filtered_helpers) == 1 &&
              generation_header(map, 0)->ready_helpers == 1 &&
              control.seq == 2 && bootstrap_generation_sealed(map),
              "bootstrap sealed before first launch");
        const auto* owners = generation_map(map, 0);
        check(owners[0] == 0 && owners[1] == kOwnerLocal &&
              owners[shape.experts + 2] == kOwnerLocal,
              "protected and undeclared experts filtered");
        check(device.before_launch(map, 0, 0), "real pool and two table banks prepared");
        check(r3::epoch::compare_request_ranks(1, h->slot_count) == 1 &&
              r3::epoch::compare_request_ranks(2, h->slot_count) == 2 &&
              r3::epoch::compare_request_ranks(3, h->slot_count) == 0,
              "one-rank full-width is not two duplicate TP requests");
        const std::vector<Route> captured{{0, 0, 0x3f80, 0}};
        auto resident = [&](uint32_t l, uint32_t e, uint32_t rank, uint32_t local) {
            return device.resident(l, e, rank, local);
        };
        check(r3::epoch::validate_compare_routes(map, 0, 2, captured, 2, resident),
              "captured route has both TP slices and both generation banks");
        const std::vector<Route> protected_route{{0, 1, 0x3f80, 0}};
        check(!r3::epoch::validate_compare_routes(map, 0, 2, protected_route, 2, resident),
              "captured protected route fails closed");
        const std::vector<Route> duplicate{{0, 0, 0x3f80, 0}, {0, 0, 0x3f80, 0}};
        check(!r3::epoch::validate_compare_routes(map, 0, 2, duplicate, 2, resident),
              "duplicate top-k route fails closed");
        check(!r3::epoch::validate_compare_routes(
                  map, 0, 2, captured, 2,
                  [&](uint32_t l, uint32_t e, uint32_t rank, uint32_t local) {
                      return rank == 0 && device.resident(l, e, rank, local);
                  }),
              "missing rank-1 weight slice fails closed");
        check(!r3::epoch::validate_compare_routes(
                  map, 0, 2, captured, 2,
                  [&](uint32_t l, uint32_t e, uint32_t rank, uint32_t local) {
                      return device.resident(l, e, rank, local + 1);
                  }),
              "mismatched runtime local slot fails closed");
        Request* q = lane_request(map, lane_of(0, 0), 1);
        check(r3::epoch::bind_compare_request(map, q, control) &&
              q->placement_epoch == control.active_epoch &&
              q->generation_slot == control.active_slot,
              "request bound to sealed generation before kernel launch");
        check(!r3::epoch::publish_owned(map, selected, 0) &&
              !r3::epoch::mark_bootstrap_filtered(map, 0),
              "no post-seal owner mutation");
    } else {
        check(!bootstrap_generation_sealed(map) &&
              !load(&static_cast<Header*>(map)->layer_map_consumed[0]) &&
              !r3::epoch::bind_compare_request(map, lane_request(map, lane_of(0,0), 1), control),
              "failed device load never admits a request");
    }
    munmap(map, layout.bytes);
}
// Optional real-capture check: invoked by the non-GPU developer gate with
// the recorded .bin argument; CI's no-argument test remains self-contained.
static void run_capture(const char* path) {
    FILE* f = std::fopen(path, "rb");
    check(f != nullptr, "capture opens");
    uint32_t hd[9]{};
    check(std::fread(hd, sizeof(uint32_t), 9, f) == 9 &&
          hd[0] == 0x52335243 && hd[1] == 3 && hd[2] < kMaxLayers &&
          hd[3] && hd[3] <= kMaxRows && hd[4] && hd[4] <= kMaxRoutes &&
          hd[6] && hd[6] <= kExperts && hd[7] && hd[7] <= kMaxTopK,
          "capture header valid");
    std::vector<int32_t> runtime_slots(hd[6]);
    std::vector<Route> routes(hd[4]);
    check(std::fread(runtime_slots.data(), sizeof(int32_t), runtime_slots.size(), f)
              == runtime_slots.size() &&
          std::fread(routes.data(), sizeof(Route), routes.size(), f) == routes.size(),
          "capture slot map and routes readable");
    std::fclose(f);
    Selected selected{};
    std::set<uint32_t> distinct;
    for (const Route& r : routes) {
        check(r.expert >= 0 && uint32_t(r.expert) < hd[6], "capture expert in range");
        if (distinct.insert(uint32_t(r.expert)).second)
            selected.expert.emplace_back(hd[2], uint32_t(r.expert));
    }
    // The real capture is a 2048-row request with 122 routes to 23 experts.
    check(hd[2] == 6 && hd[3] == 2048 && routes.size() == 122 &&
          selected.expert.size() == 23, "expected req-18-7 fixture");
    LayoutShape shape{128, hd[7], hd[6], hd[3], 128, 640};
    Layout layout{};
    check(layout_for(shape, hd[2] + 1, slots_for(1), &layout), "capture layout");
    void* map = mmap(nullptr, layout.bytes, PROT_READ | PROT_WRITE,
                     MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    check(map != MAP_FAILED, "capture mmap");
    FakeRunner device(selected, hd[2] + 1, hd[6]);
    check(device.prepare_pool(), "capture physical expert pool");
    PlacementControl control{};
    const uint32_t small_rows[]{8};
    const char* why = nullptr;
    check(r3::epoch::prepare_compare_bootstrap(
              map, hd[2] + 1, shape, selected, hd[2], runtime_slots.data(),
              1, small_rows,
              [&](uint32_t layer, const int32_t* slots) {
                  return device.load_layer(layer, slots);
              }, &control, &why),
          why ? why : "real capture bootstraps");
    check(r3::epoch::validate_compare_routes(
              map, hd[2], hd[3], routes, 2,
              [&](uint32_t l, uint32_t e, uint32_t rank, uint32_t local) {
                  return device.resident(l, e, rank, local);
              }, &why),
          why ? why : "all captured routes bound to loaded physical residents");
    check(device.before_launch(map, hd[2], uint32_t(routes[0].expert)),
          "both pointer table generations selected");
    Request* q = lane_request(map, lane_of(0, 0), 1);
    check(r3::epoch::bind_compare_request(map, q, control),
          "actual capture binds sealed request generation");
    check(!r3::epoch::validate_compare_routes(
              map, hd[2], hd[3], routes, 2,
              [&](uint32_t l, uint32_t e, uint32_t rank, uint32_t local) {
                  return rank == 0 && device.resident(l, e, rank, local);
              }),
          "all captured routes require second rank residency");
    std::printf("replay_bootstrap_test: real capture %zu routes / %zu residents / 2 slices PASS\n",
                routes.size(), distinct.size());
    munmap(map, layout.bytes);
}
int main(int argc, char** argv) {
    run(false);
    run(true);
    if (argc > 1) run_capture(argv[1]);
    std::puts("replay_bootstrap_test: PASS");
}
