#include "moe_remote_shm.h"
#include <cstdio>
#include <fcntl.h>
#include <string>
#include <sys/mman.h>
#include <thread>
#include <vector>
#include <unistd.h>
#include <cstdlib>

// Release builds define NDEBUG, which removes assert() and any call inside it; these checks always run.
static void check(bool ok, const char* what) {
    if (!ok) { std::fprintf(stderr, "r3_moe_shm_test: FAILED %s\n", what); std::exit(1); }
}

using namespace r3::moe_shm;
int main() {
    const std::string name = "/r3-moe-shm-test-" + std::to_string((long long)getpid());
    const Layout l = layout();
    int fd = shm_open(name.c_str(), O_CREAT | O_EXCL | O_RDWR, 0600);
    check(fd >= 0, "shm_open");
    check(ftruncate(fd, (off_t)l.bytes) == 0, "ftruncate");
    void* m = mmap(nullptr, l.bytes, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    check(m != MAP_FAILED, "mmap"); check(init(m, 48, legacy_shape()), "init"); check(valid((Header*)m, l.bytes, legacy_shape()), "header valid");
    {   // a segment of another protocol version (an engine and a helper from different builds) is rejected
        Header* hv = (Header*)m;
        const uint32_t ver = hv->version;
        hv->version = ver - 1;
        check(!valid(hv, l.bytes, legacy_shape()), "an older protocol version is rejected");
        hv->version = ver + 1;
        check(!valid(hv, l.bytes, legacy_shape()), "a newer protocol version is rejected");
        hv->version = ver;
        check(valid(hv, l.bytes, legacy_shape()), "the current protocol version is accepted");
    }
    auto* h = (Header*)m;
    {   // RAD12: ten lanes (four small sizes and the large lane for each rank), a class table the helper sets and the engine validates
        check(kLanes == kMaxRanks * (kSmallClasses + 1) && kRingSlots == kLanes * kLaneSlots, "lane count and ring size");
        check(lane_of(0, 0) == 0 && lane_of(1, 0) == 1 && lane_of(0, 1) == 2 && lane_of(1, kSmallClasses) == 9,
              "lane index is class * ranks + rank");
        check(lane_class(lane_of(1, 3)) == 3 && lane_is_small(lane_of(1, 3)) && !lane_is_small(lane_of(0, kSmallClasses)),
              "lane class and size");
        check(lane_request(m, kLanes, 1) == nullptr && lane_request(m, 0, 0) == nullptr, "lane bounds");
        check(lane_request(m, 1, 1) == lane_request(m, 0, 1) + kLaneSlots &&
              lane_request(m, 1, kLaneSlots + 1) == lane_request(m, 1, 1), "lane rings are private and wrap");
        h->small_class_count = 4;
        h->small_class_rows[0] = 4; h->small_class_rows[1] = 8; h->small_class_rows[2] = 16; h->small_class_rows[3] = 24;
        check(valid(h, l.bytes, legacy_shape()), "a four-class table is accepted");
        check(step_class(h, 1) == 0 && step_class(h, 4) == 0 && step_class(h, 5) == 1 && step_class(h, 8) == 1 &&
              step_class(h, 9) == 2 && step_class(h, 16) == 2 && step_class(h, 17) == 3 && step_class(h, 24) == 3 &&
              step_class(h, 25) == kSmallClasses && step_class(h, kMaxRows) == kSmallClasses,
              "a step takes the first class that fits, else the large lane");
        h->small_class_count = 2;
        check(step_class(h, 9) == kSmallClasses && step_class(h, 8) == 1, "classes past the count are not used");
        h->small_class_rows[1] = 4;
        check(!valid(h, l.bytes, legacy_shape()), "a non-ascending class table is rejected");
        h->small_class_rows[1] = 8;
        h->small_class_count = 0;
        check(!valid(h, l.bytes, legacy_shape()), "an empty class table is rejected");
        h->small_class_count = kSmallClasses + 1;
        check(!valid(h, l.bytes, legacy_shape()), "too many classes are rejected");
        h->small_class_count = 1;
        h->small_class_rows[0] = 8;
        check(valid(h, l.bytes, legacy_shape()), "the one-class table of the production configuration is accepted");
    }
    check(layer_slot(m, 7, 23) == -1, "layer map starts unpublished");
    int32_t* slots = layer_slots(m, 7);
    check(slots != nullptr, "layer map pointer");
    for (uint32_t e = 0; e < kExperts; ++e) slots[e] = (int32_t)e;
    slots[3] = -1;
    slots[23] = 22;
    store(&h->layer_map_ready[7], 1);
    check(layer_slot(m, 7, 3) == -1 && layer_slot(m, 7, 23) == 22,
          "runtime map preserves protection and compact slots");
    check(!owns(m, 7, 23, 0), "ownership starts clear");
    check(set_owned(m, 7, 23, 0), "publish ownership");
    check(owns(m, 7, 23, 0) && !owns(m, 7, 23, 1) && h->owned_count == 1, "ownership is visible, per helper");
    check(!set_owned(m, 7, 23, 0) && h->owned_count == 1, "duplicate ownership is idempotent");
    check(clear_owned(m, 7, 23, 0) && !owns(m, 7, 23, 0) && h->owned_count == 0,
          "runtime protection clears candidate ownership");
    check(set_owned(m, 7, 23, 0) && h->owned_count == 1, "ownership can be restored");
    check(h->result_offset[0] != h->result_offset[kMaxRanks] &&
          h->result_offset[1] != h->result_offset[kMaxRanks + 1],
          "producer ranks have isolated result slots");

    std::thread helper([&] {
        Request* q = request(m, 1);
        while (load(&q->sequence) != 1) {
            uint32_t seen = load(&h->publish_sequence);
            if (seen < 1) futex_wait(&h->publish_sequence, seen);
        }
        check(q->layer == 7 && q->rank == 0 && q->rows == 1 && q->route_count == 1, "q->layer == 7 && q->rank == 0 && q->rows");
        const Route* r = ptr<Route>(m, q->route_offset);
        check(r[0].token == 0 && r[0].expert == 23 && r[0].weight_bf16 == 0x3f80 && r[0].slot == 7,
              "route payload preserves token/expert/weight/top-k slot");
        uint16_t* out = ptr<uint16_t>(m, q->result_offset[0]); out[0] = 0x4000;
        q->status = 0; store(&q->completion_sequence, 1); futex_wake(&q->completion_sequence);
    });

    uint32_t seq = fetch_add(&h->next_sequence, 1); check(seq == 1, "first sequence");
    Request* q = request(m, seq);
    Route* r = ptr<Route>(m, h->route_offset[0]); r[0] = {0, 23, 0x3f80, 7};
    q->layer = 7; q->rank = 0; q->rank_count = 2; q->rows = 1; q->hidden = kHidden;
    q->route_count = 1; q->route_offset = h->route_offset[0]; q->activation_offset = h->activation_offset[0];
    q->activation_stride = kHidden * 2; q->result_offset[0] = h->result_offset[0]; q->result_stride[0] = kHidden * 2;
    store(&q->sequence, seq); store(&h->publish_sequence, seq); futex_wake(&h->publish_sequence);
    wait_until(&q->completion_sequence, seq);
    check(q->status == 0 && ptr<uint16_t>(m, q->result_offset[0])[0] == 0x4000, "q->status == 0 && ptr<uint16_t>(m, q->re");

    helper.join(); munmap(m, l.bytes); close(fd); shm_unlink(name.c_str());

    {   // RCS09-02: the runtime layout reproduces the compile-time one byte for byte for the legacy shape
        Layout rt{};
        check(layout_for(legacy_shape(), &rt), "layout_for(legacy_shape()) succeeds");
        check(rt.ring == l.ring && rt.owner == l.owner && rt.layer_map == l.layer_map && rt.bytes == l.bytes,
              "runtime layout matches the compile-time ring, owner, layer map and total");
        for (uint32_t r = 0; r < kMaxRanks; ++r) {
            check(rt.route[r] == l.route[r] && rt.activation[r] == l.activation[r], "route and activation offsets match");
            for (uint32_t o = 0; o < kMaxRanks; ++o) check(rt.result[r * kMaxRanks + o] == l.result[r * kMaxRanks + o], "result offsets match");
        }
        // another model's shape (Qwen3.6-35B-A3B: hidden 2048, 256 experts, top-k 8): valid, page aligned, in order
        Layout q{};
        check(layout_for(LayoutShape{2048, 8, 256, 4096, 128, 512}, &q), "a second shape lays out");
        check(q.bytes % kPage == 0 && q.ring == kPage && q.owner > q.ring && q.layer_map > q.owner && q.route[0] > q.layer_map,
              "regions are page aligned and in order");
        check(q.route[1] > q.result[kMaxRanks - 1] && q.bytes > q.result[kMaxRanks * kMaxRanks - 1], "per-rank regions follow in order");
        check(q.bytes < l.bytes, "a smaller hidden width and expert count take less memory");
        // refusals: nothing is written on failure
        Layout keep{}; keep.bytes = 77;
        check(!layout_for(LayoutShape{0, 8, 256, 4096, 128, 512}, &keep) && keep.bytes == 77, "zero hidden is refused");
        check(!layout_for(LayoutShape{2048, 0, 256, 4096, 128, 512}, &keep), "zero top-k is refused");
        check(!layout_for(LayoutShape{2048, 8, 256, 0, 128, 512}, &keep), "zero row capacity is refused");
        check(!layout_for(LayoutShape{2048, 8, 256, 4096, 0, 512}, &keep), "a zero activation group is refused");
        check(!layout_for(LayoutShape{2048, 300, 256, 4096, 128, 512}, &keep), "top-k above the expert count is refused");
        check(!layout_for(LayoutShape{2000, 8, 256, 4096, 128, 512}, &keep), "a hidden width that is not whole groups is refused");
        check(!layout_for(LayoutShape{UINT32_MAX - 127, 8, 256, UINT32_MAX, 128, 512}, &keep) && keep.bytes == 77,
              "an overflowing shape is refused");
        check(!layout_for(LayoutShape{65536, 8, 256, 65536, 128, 512}, &keep), "a slot that does not fit 32 bits is refused");
        check(!layout_for(legacy_shape(), nullptr), "a null output is refused");
    }
    {   // RCS09-03: a v14 segment records its shape and is accepted only by a side that expects the same shape
        const LayoutShape qs{2048, 8, 256, 4096, 128, 512};
        Layout ql{};
        check(layout_for(qs, &ql), "second shape lays out");
        std::vector<uint8_t> seg((size_t)ql.bytes);
        check(init(seg.data(), 41, qs), "init for the second shape");
        auto* h = reinterpret_cast<Header*>(seg.data());
        check(h->version == kVersion && h->hidden == 2048 && h->expert_count == 256 && h->top_k == 8 && h->intermediate == 512 &&
              h->activation_group == 128 && h->shape_hash == shape_hash(qs), "the header records the shape");
        check(valid(h, seg.size(), qs), "the same shape is accepted");
        check(!valid(h, seg.size(), legacy_shape()), "a side expecting another shape refuses the segment");
        LayoutShape other = qs; other.intermediate = 640;
        check(!valid(h, seg.size(), other), "a different expert width is refused");
        other = qs; other.act_group = 64;
        check(!valid(h, seg.size(), other), "a different activation group is refused");
        check(!valid(h, seg.size() - 1, qs), "a mapping shorter than the segment is refused");
        {   // a corrupt header cannot move a region: each field is checked against the layout rebuilt from the shape
            Header keep = *h;
            h->route_offset[0] += kPage; check(!valid(h, seg.size(), qs), "a moved route region is refused"); *h = keep;
            h->result_offset[1] += kPage; check(!valid(h, seg.size(), qs), "a moved result region is refused"); *h = keep;
            h->owner_offset += kPage; check(!valid(h, seg.size(), qs), "a moved owner region is refused"); *h = keep;
            h->segment_bytes += kPage; check(!valid(h, seg.size(), qs), "a wrong segment length is refused"); *h = keep;
            h->shape_hash ^= 1; check(!valid(h, seg.size(), qs), "a wrong shape hash is refused"); *h = keep;
            h->hidden = 2560; check(!valid(h, seg.size(), qs), "edited shape fields are refused"); *h = keep;
            h->version = 14; check(!valid(h, seg.size(), qs), "a stale v14 segment is refused"); *h = keep;
            h->small_class_rows[0] = h->max_rows + 1; check(!valid(h, seg.size(), qs), "a class above the row capacity is refused"); *h = keep;
            check(h->combine.enabled == 0 && h->combine.request == 0 && h->combine.done == 0,
                  "a new segment starts with R3_MOE_WORKER_COMBINE off and no generation");
            check(valid(h, seg.size(), qs), "and the restored header is accepted again");
        }
        check(!init(seg.data(), 41, LayoutShape{2048, 8, 256, 4096, 100, 512}), "init refuses a shape that does not lay out");
        check(!init(seg.data(), 0, qs) && !init(seg.data(), kMaxLayers + 1, qs), "init refuses a layer count outside the protocol bound");
    }
    std::printf("r3_moe_shm_test: PASS\n");
    std::puts("r3_moe_shm_test: ok");
    return 0;
}
