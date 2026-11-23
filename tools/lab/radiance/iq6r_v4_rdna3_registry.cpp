// Card-free ABI inspection of the actual managed libr3 row registry.
// Does not call rad_plugin_open, HIP runtime, launch, init, scratch or layout.
#include "rad_abi.h"
#include <dlfcn.h>
#include <cstdio>
#include <cstring>
#include <cstdlib>

int main(int argc, char** argv) {
    if (argc != 2) {
        std::fprintf(stderr, "usage: iq6r_v4_rdna3_registry LIBR3_SO\n");
        return 2;
    }
    void* so = dlopen(argv[1], RTLD_NOW | RTLD_LOCAL);
    if (!so) {
        std::fprintf(stderr, "dlopen: %s\n", dlerror());
        return 2;
    }
    using Count = int (*)(void);
    using At = const RadKernelInfo* (*)(int);
    auto count = reinterpret_cast<Count>(dlsym(so, "rad_kernel_count"));
    auto at = reinterpret_cast<At>(dlsym(so, "rad_kernel_at"));
    if (!count || !at) {
        std::fprintf(stderr, "missing ABI registry functions\n");
        return 2;
    }
    const char* required[] = {
        "embed_lookup_iqr", "logits_gemm_vocab_iqr",
        "logits_gemm_iqr", "logit_rerank_iqr",
        "gemm_iqr_nt_m64", "gemm_iqr_gated", "gemm_iqr_tiled",
    };
    bool present[7] = {};
    int n = count();
    if (n <= 0 || n > 4096) {
        std::fprintf(stderr, "invalid registry count: %d\n", n);
        return 2;
    }
    for (int i = 0; i < n; ++i) {
        const RadKernelInfo* row = at(i);
        if (!row || !row->name || !row->op) {
            std::fprintf(stderr, "malformed live row %d\n", i);
            return 2;
        }
        for (int j = 0; j < 7; ++j) {
            if (std::strcmp(row->name, required[j]) == 0) {
                if (present[j]) {
                    std::fprintf(stderr, "duplicate row %s\n", required[j]);
                    return 2;
                }
                present[j] = true;
                if (!row->launch || !row->layout || !row->relayout) {
                    std::fprintf(stderr, "incomplete live row %s\n", required[j]);
                    return 2;
                }
                std::printf("ROW\t%s\t%s\t%d\t%d\n",
                            row->name, row->op, row->domain, row->priority);
            }
        }
    }
    int verified = 0;
    for (int i = 0; i < 7; ++i) {
        if (present[i]) ++verified;
        else std::fprintf(stderr, "missing live row: %s\n", required[i]);
    }
    std::printf("RESULT\t%d\t%d\n", n, verified);
    dlclose(so);
    return verified == 7 ? 0 : 1;
}
