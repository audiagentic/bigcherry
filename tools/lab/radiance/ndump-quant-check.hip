// RAD15: real E4M3 round-trip error on ground-truth N (dumped from a live hc_read call via the throwaway
// BC_HC_NDUMP_ROW instrumentation), per-128-of-K amax scale -- replaces a crude Python approximation of E4M3
// rounding (which isn't uniform-step, so an approximation there isn't trustworthy) with the real cast.
// Build: /opt/rocm-7.2.4/bin/hipcc --offload-arch=gfx1201 -O2 -I/opt/rocm-7.2.4/include ndump-quant-check.hip \
//          -o ndump-quant-check
#include <hip/hip_runtime.h>
#include <rocwmma/rocwmma.hpp>
#include <cstdio>
#include <cmath>
#include <vector>

using rocwmma::float8_t;

int main(int argc, char** argv) {
    const char* path = argc > 1 ? argv[1] : "/mnt/data/bigcherry-work/ndump.bin";
    FILE* f = fopen(path, "rb");
    if (!f) { printf("cannot open %s\n", path); return 1; }
    std::vector<float> a(10240);
    size_t got = fread(a.data(), sizeof(float), a.size(), f);
    fclose(f);
    if (got != a.size()) { printf("short read: %zu\n", got); return 1; }

    const int KB = 128, NKB = (int)a.size() / KB;
    double sumsq_err = 0.0, sumsq_ref = 0.0;
    double maxabs = 0.0;
    int saturated = 0;
    for (int g = 0; g < NKB; ++g) {
        float amax = 0.0f;
        for (int k = 0; k < KB; ++k) amax = std::max(amax, std::fabs(a[g * KB + k]));
        const float scale = amax > 0.0f ? amax / 448.0f : 1.0f;
        for (int k = 0; k < KB; ++k) {
            const float x = a[g * KB + k];
            const float q = x / scale;
            const float8_t c = float8_t(q);        // the real E4M3 cast
            const float back = float(c) * scale;
            if (std::fabs(float(c)) >= 448.0f - 1e-3f) saturated++;
            const double d = back - x;
            sumsq_err += d * d;
            sumsq_ref += (double)x * x;
            maxabs = std::max(maxabs, std::fabs(d));
        }
    }
    printf("real E4M3 round-trip on ground-truth N (%d groups of %d, amax/448 scale):\n", NKB, KB);
    printf("  RMS relative error: %.4f%%\n", 100.0 * std::sqrt(sumsq_err / (NKB * KB)) / std::sqrt(sumsq_ref / (NKB * KB)));
    printf("  max abs error: %.6g   saturated codes: %d of %d\n", maxabs, saturated, NKB * KB);
    return 0;
}
