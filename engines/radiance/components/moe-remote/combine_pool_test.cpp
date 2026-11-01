// RAD23: CombinePool (worker_combine.h) never loses its first job. A worker thread that started after combine() had already
// posted the first job used to take that job as seen and sleep, and combine() waited for it forever. Here a fresh pool runs
// one combine at once, many times, under an alarm that fails the test instead of hanging it; the output is checked too.
#include <atomic>
#include <climits>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <thread>
#include <unistd.h>
#include <vector>
#include "moe_remote_shm.h"
namespace { inline bool stopping() { return false; } }   // worker_container.h supplies this in the worker
#include "worker_combine.h"

int main() {
    if (!combine_cpu_supported()) { std::printf("combine pool: skipped (no AVX2/F16C)\n"); return 0; }
    std::signal(SIGALRM, [](int) { std::fprintf(stderr, "FAIL: a combine never finished (lost first job)\n"); std::_Exit(1); });
    alarm(60);
    const uint32_t rows = 64, hidden = 256;
    std::vector<uint16_t> a((size_t)rows * hidden), b((size_t)rows * hidden), out((size_t)rows * hidden);
    for (size_t i = 0; i < a.size(); ++i) { a[i] = 0x3f80; b[i] = 0x4000; }   // bf16 1.0 and 2.0
    std::vector<uint8_t> mask(rows, 3);
    const uint16_t* src[2] = {a.data(), b.data()};
    for (int round = 0; round < 2000; ++round) {
        CombinePool pool(8);   // combine straight away, before the workers are known to be up
        std::memset(out.data(), 0, out.size() * 2);
        pool.combine(src, 2, out.data(), mask.data(), rows, hidden);
        for (uint16_t v : out)
            if (v != 0x4200) { std::fprintf(stderr, "FAIL: 1 + 2 is not fp16 3.0 (0x%04x)\n", v); return 1; }
    }
    std::printf("combine pool: ok\n");
    return 0;
}
