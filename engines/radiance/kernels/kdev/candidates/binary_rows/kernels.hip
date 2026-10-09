// Candidate binary_rows: bf16 add and mul, each thread takes a run of 8 consecutive elements of one row.
// Fewer threads and no divide per element; compared with binary_flat to see which the card prefers per shape.
#include "kdev_binary.h"

namespace {

constexpr uint32_t RUN = 8;

template <bool MUL>
__global__ void rows_kernel(const uint16_t* a, const uint16_t* b, uint16_t* y, uint32_t rows, uint32_t n,
                            uint32_t a_ld, uint32_t b_ld, uint32_t y_ld) {
    const uint32_t runs_per_row = (n + RUN - 1) / RUN;
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= rows * runs_per_row) return;
    const uint32_t r = i / runs_per_row, c0 = (i - r * runs_per_row) * RUN;
    const uint16_t* ar = a + r * a_ld;
    const uint16_t* br = b + r * b_ld;  // b_ld == 0: the one broadcast row
    uint16_t* yr = y + r * y_ld;
    const uint32_t c1 = c0 + RUN < n ? c0 + RUN : n;
    for (uint32_t c = c0; c < c1; ++c) {
        const float av = kdev_bf16_to_f32(ar[c]);
        const float bv = kdev_bf16_to_f32(br[c]);
        yr[c] = kdev_f32_to_bf16(MUL ? av * bv : av + bv);
    }
}

template <bool MUL>
int launch(const RadArgs* args, RadStream stream) {
    KdevBinary g;
    const int status = kdev_binary_args(args, &g);
    if (status != RAD_OK) return status;
    const uint32_t threads = 256, total = g.rows * ((g.n + RUN - 1) / RUN);
    hipLaunchKernelGGL((rows_kernel<MUL>), dim3((total + threads - 1) / threads), dim3(threads), 0,
                       kdev_stream(stream), g.a, g.b, g.y, g.rows, g.n, g.a_ld, g.b_ld, g.y_ld);
    return kdev_launch_status();
}

}  // namespace

extern "C" int k_rows_add(const RadArgs* a, RadStream s) { return launch<false>(a, s); }
extern "C" int k_rows_mul(const RadArgs* a, RadStream s) { return launch<true>(a, s); }
