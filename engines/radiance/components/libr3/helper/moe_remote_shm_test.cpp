#include "moe_remote_shm.h"
#include <cstdio>
#include <fcntl.h>
#include <string>
#include <sys/mman.h>
#include <thread>
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
    check(m != MAP_FAILED, "mmap"); init(m, 48); check(valid((Header*)m, l.bytes), "header valid");
    {   // a segment of another protocol version (an engine and a helper from different builds) is rejected
        Header* hv = (Header*)m;
        const uint32_t ver = hv->version;
        hv->version = ver - 1;
        check(!valid(hv, l.bytes), "an older protocol version is rejected");
        hv->version = ver + 1;
        check(!valid(hv, l.bytes), "a newer protocol version is rejected");
        hv->version = ver;
        check(valid(hv, l.bytes), "the current protocol version is accepted");
    }
    auto* h = (Header*)m;
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
    std::printf("r3_moe_shm_test: PASS\n");
    std::puts("r3_moe_shm_test: ok");
    return 0;
}
