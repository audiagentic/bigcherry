// Feasibility probe for RAD03: can two processes on different GPUs signal each other through a
// pinned POSIX shm page using only stream operations (hipStreamWriteValue32 / hipStreamWaitValue32),
// with no CPU in the loop? Run "consumer" on the R9700 first, then "producer" on an XTX.
// Measures round-trip latency of N ping-pongs: producer writes ping=i, consumer stream waits for it,
// writes pong=i, producer stream waits for pong.
#include <hip/hip_runtime.h>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>

#define CK(x) do { hipError_t e_ = (x); if (e_ != hipSuccess) { std::fprintf(stderr, "%s: %s\n", #x, hipGetErrorString(e_)); std::exit(1); } } while (0)

int main(int argc, char** argv) {
    if (argc < 3) { std::fprintf(stderr, "usage: mailbox_probe consumer|producer N\n"); return 2; }
    const bool producer = std::strcmp(argv[1], "producer") == 0;
    const int n = std::atoi(argv[2]);
    const char* name = "/r3-mailbox-probe";
    int fd = shm_open(name, O_CREAT | O_RDWR, 0600);
    if (fd < 0 || ftruncate(fd, 4096) != 0) { std::perror("shm"); return 1; }
    auto* page = static_cast<uint32_t*>(mmap(nullptr, 4096, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0));
    if (page == MAP_FAILED) { std::perror("mmap"); return 1; }
    CK(hipHostRegister(page, 4096, hipHostRegisterPortable | hipHostRegisterMapped));
    void* dev = nullptr;
    CK(hipHostGetDevicePointer(&dev, page, 0));
    auto* ping = static_cast<uint32_t*>(dev);
    auto* pong = ping + 64;  // separate cache lines
    hipStream_t s;
    CK(hipStreamCreateWithFlags(&s, hipStreamNonBlocking));
    if (!producer) {
        page[0] = 0; page[64] = 0;
        std::printf("consumer ready\n"); std::fflush(stdout);
        for (int i = 1; i <= n; ++i) {
            CK(hipStreamWaitValue32(s, ping, (uint32_t)i, hipStreamWaitValueGte, 0xffffffffu));
            CK(hipStreamWriteValue32(s, pong, (uint32_t)i, 0));
        }
        CK(hipStreamSynchronize(s));
        std::printf("consumer done\n");
    } else {
        // warm-up then timed
        const auto t0 = std::chrono::steady_clock::now();
        for (int i = 1; i <= n; ++i) {
            CK(hipStreamWriteValue32(s, ping, (uint32_t)i, 0));
            CK(hipStreamWaitValue32(s, pong, (uint32_t)i, hipStreamWaitValueGte, 0xffffffffu));
        }
        CK(hipStreamSynchronize(s));
        const double us = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count();
        std::printf("producer: %d round trips, %.1f us each (GPU-signalled only)\n", n, us / n);
        shm_unlink(name);
    }
    return 0;
}
