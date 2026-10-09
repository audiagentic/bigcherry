// Candidate binary_flat: bf16 add and mul, one thread per element over the flattened [rows, n] output.
// The plainest form: the baseline every other add/mul candidate is compared with.
#include "kdev_binary.h"

namespace {

template <bool MUL>
__global__ void flat_kernel(const uint16_t* a, const uint16_t* b, uint16_t* y, uint32_t rows, uint32_t n,
                            uint32_t a_ld, uint32_t b_ld, uint32_t y_ld) {
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= rows * n) return;
    const uint32_t r = i / n, c = i - r * n;
    const float av = kdev_bf16_to_f32(a[r * a_ld + c]);
    const float bv = kdev_bf16_to_f32(b[r * b_ld + c]);  // b_ld == 0: the one broadcast row
    y[r * y_ld + c] = kdev_f32_to_bf16(MUL ? av * bv : av + bv);
}

template <bool MUL>
int launch(const RadArgs* args, RadStream stream) {
    KdevBinary g;
    const int status = kdev_binary_args(args, &g);
    if (status != RAD_OK) return status;
    const uint32_t threads = 256, total = g.rows * g.n;
    hipLaunchKernelGGL((flat_kernel<MUL>), dim3((total + threads - 1) / threads), dim3(threads), 0,
                       kdev_stream(stream), g.a, g.b, g.y, g.rows, g.n, g.a_ld, g.b_ld, g.y_ld);
    return kdev_launch_status();
}

}  // namespace

extern "C" int k_flat_add(const RadArgs* a, RadStream s) { return launch<false>(a, s); }
extern "C" int k_flat_mul(const RadArgs* a, RadStream s) { return launch<true>(a, s); }
