// Minimal gfx1100 BF16 add/mul: no gfx12 WMMA, FP8 or peer-memory assumptions.
#include "rad_abi.h"
#include <hip/hip_runtime.h>
#include <cstdint>
#include <climits>
#include <cstring>

namespace {

__device__ __forceinline__ float bf16_to_f32(uint16_t x) {
    return __uint_as_float(static_cast<uint32_t>(x) << 16);
}

// Round-to-nearest-even, matching libref; retain NaN rather than overflowing to infinity.
__device__ __forceinline__ uint16_t f32_to_bf16(float x) {
    const uint32_t v = __float_as_uint(x);
    if ((v & 0x7f800000u) == 0x7f800000u) {
        return static_cast<uint16_t>((v >> 16) | ((v & 0x007fffffu) ? 0x0040u : 0u));
    }
    return static_cast<uint16_t>((v + 0x7fffu + ((v >> 16) & 1u)) >> 16);
}

template <bool MULTIPLY>
__global__ void binary_bf16(const uint16_t* a, const uint16_t* b, uint16_t* y,
                            uint32_t total, uint32_t n, bool broadcast_b) {
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total) return;
    const float av = bf16_to_f32(a[i]);
    const float bv = bf16_to_f32(b[broadcast_b ? (i % n) : i]);
    y[i] = f32_to_bf16(MULTIPLY ? (av * bv) : (av + bv));
}

template <bool MULTIPLY>
int launch_binary(const RadArgs* a, RadStream stream) {
    if (!a || a->n_t < 3 || !a->t) return RAD_E_INVAL;
    const RadTensor& A = a->t[0];
    const RadTensor& B = a->t[1];
    const RadTensor& Y = a->t[2];
    if (!A.data || !B.data || !Y.data) return RAD_E_INVAL;
    const char* dtype = rad_args_gets(a, "dtype");
    if (!dtype || std::strcmp(dtype, "bf16") != 0 ||
        A.dtype != RAD_BF16 || B.dtype != RAD_BF16 || Y.dtype != RAD_BF16)
        return RAD_E_DTYPE;
    long long m = 0, n = 0;
    if (!rad_args_geti(a, "M", &m) || !rad_args_geti(a, "n", &n) ||
        m < 1 || n < 1 || m > static_cast<long long>(UINT32_MAX) / n)
        return RAD_E_SHAPE;
    const int64_t total = m * n;
    const int64_t bn = rad_tensor_numel(&B);
    if (rad_tensor_numel(&Y) != total || rad_tensor_numel(&A) != total ||
        (bn != total && !(m > 1 && bn == n)))
        return RAD_E_SHAPE;
    if (!rad_tensor_is_contiguous(&A) || !rad_tensor_is_contiguous(&B) ||
        !rad_tensor_is_contiguous(&Y))
        return RAD_E_STRIDE;
    if (bn == n && m > 1 && B.data == Y.data)
        return RAD_E_INVAL; // Broadcast source must not alias the output.
    const auto threads = 256u;
    const auto count = static_cast<uint32_t>(total);
    hipLaunchKernelGGL((binary_bf16<MULTIPLY>),
        dim3((count + threads - 1) / threads), dim3(threads), 0,
        reinterpret_cast<hipStream_t>(stream),
        static_cast<const uint16_t*>(A.data), static_cast<const uint16_t*>(B.data),
        static_cast<uint16_t*>(Y.data), count, static_cast<uint32_t>(n),
        bn == n && m > 1);
    return hipGetLastError() == hipSuccess ? RAD_OK : RAD_E_DEVICE;
}

} // namespace

extern "C" int r11_add_bf16(const RadArgs* a, RadStream s) {
    return launch_binary<false>(a, s);
}
extern "C" int r11_mul_bf16(const RadArgs* a, RadStream s) {
    return launch_binary<true>(a, s);
}
