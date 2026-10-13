#include "moe_remote_shm.h"
#include <cassert>
#include <cstdio>
#include <fcntl.h>
#include <string>
#include <sys/mman.h>
#include <thread>
#include <unistd.h>

using namespace r3::moe_shm;
int main() {
    const std::string name = "/r3-moe-shm-test-" + std::to_string((long long)getpid());
    const Layout l = layout();
    int fd = shm_open(name.c_str(), O_CREAT | O_EXCL | O_RDWR, 0600);
    assert(fd >= 0 && ftruncate(fd, (off_t)l.bytes) == 0);
    void* m = mmap(nullptr, l.bytes, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    assert(m != MAP_FAILED); init(m, 48); assert(valid((Header*)m, l.bytes));
    auto* h = (Header*)m;

    std::thread helper([&] {
        Request* q = request(m, 1);
        while (load(&q->sequence) != 1) {
            uint32_t seen = load(&h->publish_sequence);
            if (seen < 1) futex_wait(&h->publish_sequence, seen);
        }
        assert(q->layer == 7 && q->rank == 0 && q->rows == 1 && q->route_count == 1);
        const Route* r = ptr<Route>(m, q->route_offset);
        assert(r[0].token == 0 && r[0].expert == 23 && r[0].weight_bf16 == 0x3f80);
        uint16_t* out = ptr<uint16_t>(m, q->result_offset[0]); out[0] = 0x4000;
        q->status = 0; store(&q->completion_sequence, 1); futex_wake(&q->completion_sequence);
    });

    uint32_t seq = fetch_add(&h->next_sequence, 1); assert(seq == 1);
    Request* q = request(m, seq);
    Route* r = ptr<Route>(m, h->route_offset[0]); r[0] = {0, 23, 0x3f80, 0};
    q->layer = 7; q->rank = 0; q->rank_count = 2; q->rows = 1; q->hidden = kHidden;
    q->route_count = 1; q->route_offset = h->route_offset[0]; q->activation_offset = h->activation_offset[0];
    q->activation_stride = kHidden * 2; q->result_offset[0] = h->result_offset[0]; q->result_stride[0] = kHidden * 2;
    store(&q->sequence, seq); store(&h->publish_sequence, seq); futex_wake(&h->publish_sequence);
    wait_until(&q->completion_sequence, seq);
    assert(q->status == 0 && ptr<uint16_t>(m, q->result_offset[0])[0] == 0x4000);

    helper.join(); munmap(m, l.bytes); close(fd); shm_unlink(name.c_str());
    std::puts("r3_moe_shm_test: ok");
    return 0;
}
