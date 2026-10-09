// Read-only HIP topology probe. Never enables peer access or modifies the machine.
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstring>
#include <cstdlib>

int main() {
    int n = 0;
    hipError_t e = hipGetDeviceCount(&n);
    if (e != hipSuccess) {
        std::fprintf(stderr, "hipGetDeviceCount: %s\n", hipGetErrorString(e));
        return 1;
    }
    std::printf("visible_gpus=%d\n", n);
    for (int i = 0; i < n; ++i) {
        hipDeviceProp_t p{};
        e = hipGetDeviceProperties(&p, i);
        if (e != hipSuccess) {
            std::fprintf(stderr, "hipGetDeviceProperties(%d): %s\n", i, hipGetErrorString(e));
            return 1;
        }
        std::printf("gpu=%d name=%s arch=%s vram_mib=%zu\n",
                    i, p.name, p.gcnArchName,
                    static_cast<size_t>(p.totalGlobalMem / (1024 * 1024)));
        if (std::strncmp(p.gcnArchName, "gfx1100", 7) != 0) {
            std::fprintf(stderr, "GPU %d is %s, not gfx1100; refusing mixed-arch P2P conclusions\\n",
                         i, p.gcnArchName);
            return 2;
        }
    }
    if (n != 2) {
        std::fprintf(stderr, "Expected exactly two visible XTX GPUs; set ROCR_VISIBLE_DEVICES to their indices.\n");
        return 2;
    }
    for (int i = 0; i < n; ++i) {
        int peer = 0;
        e = hipDeviceCanAccessPeer(&peer, i, 1-i);
        if (e != hipSuccess) {
            std::fprintf(stderr, "hipDeviceCanAccessPeer(%d,%d): %s\n",i,1-i,hipGetErrorString(e));
            return 1;
        }
        std::printf("peer_access %d->%d=%d\n", i, 1-i, peer);
    }
    return 0;
}
