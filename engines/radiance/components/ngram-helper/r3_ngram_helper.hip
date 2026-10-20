#include "r3_ngram_api.h"
#include "rad_plugin.h"

#include <hip/hip_runtime.h>

#include <cstdint>
#include <cstring>
#include <memory>
#include <mutex>
#include <unordered_map>

namespace {

struct State {
    int device = -1;
    int64_t first = 0;
    int64_t rows = 0;
    int n = 0;
    int max_m = 0;
    uint8_t* codes = nullptr;
    int32_t* ids = nullptr;
    uint16_t* out = nullptr;
    uint16_t* lut = nullptr;
    uint32_t lut_scale = 0;
    bool lut_ready = false;

    ~State() {
        int old = -1;
        hipGetDevice(&old);
        if (device >= 0) hipSetDevice(device);
        if (codes) hipFree(codes);
        if (ids) hipFree(ids);
        if (out) hipFree(out);
        if (lut) hipFree(lut);
        if (old >= 0) hipSetDevice(old);
    }
};

std::mutex g_mu;
std::unordered_map<int, std::unique_ptr<State>> g_state;

struct DeviceGuard {
    int old = -1;
    bool ok = false;
    explicit DeviceGuard(int d) {
        if (hipGetDevice(&old) == hipSuccess && hipSetDevice(d) == hipSuccess) ok = true;
    }
    ~DeviceGuard() { if (ok && old >= 0) hipSetDevice(old); }
};

__global__ void gather_kernel(const int32_t* ids, int m, int vocab_offset,
                              int64_t first, int64_t rows, const uint8_t* table,
                              int n, const uint16_t* lut, uint16_t* out) {
    const uint64_t g = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    const uint64_t total = (uint64_t)m * (uint64_t)n;
    if (g >= total) return;
    const int row = (int)(g / (uint64_t)n);
    const int col = (int)(g % (uint64_t)n);
    const int64_t id = (int64_t)ids[row] - vocab_offset;
    uint16_t v = 0;
    if (id >= first && id < first + rows) {
        const uint8_t q = table[(size_t)(id - first) * (size_t)n + (size_t)col];
        v = lut[q];
    }
    out[g] = v;
}

int hip_status(hipError_t e) { return e == hipSuccess ? RAD_OK : RAD_E_DEVICE; }

}  // namespace

extern "C" uint32_t r3_ngram_helper_abi(void) { return R3_NGRAM_HELPER_ABI; }

extern "C" int r3_ngram_helper_capacity(int device, uint64_t reserve_bytes, uint64_t* usable_bytes) {
    if (!usable_bytes) return RAD_E_INVAL;
    DeviceGuard g(device);
    if (!g.ok) return RAD_E_DEVICE;
    size_t free_b = 0, total_b = 0;
    if (hipMemGetInfo(&free_b, &total_b) != hipSuccess) return RAD_E_DEVICE;
    *usable_bytes = free_b > reserve_bytes ? (uint64_t)free_b - reserve_bytes : 0;
    return RAD_OK;
}

extern "C" int r3_ngram_helper_prepare(int device, int64_t first_row, int64_t rows,
                                        int n_embd, int max_m) {
    if (first_row < 0 || rows <= 0 || n_embd <= 0 || max_m <= 0) return RAD_E_INVAL;
    DeviceGuard guard(device);
    if (!guard.ok) return RAD_E_DEVICE;

    std::unique_ptr<State> s(new State());
    s->device = device;
    s->first = first_row;
    s->rows = rows;
    s->n = n_embd;
    s->max_m = max_m;

    const size_t code_bytes = (size_t)rows * (size_t)n_embd;
    const size_t id_bytes = (size_t)max_m * sizeof(int32_t);
    const size_t out_bytes = (size_t)max_m * (size_t)n_embd * sizeof(uint16_t);
    if (hipMalloc((void**)&s->codes, code_bytes) != hipSuccess ||
        hipMalloc((void**)&s->ids, id_bytes) != hipSuccess ||
        hipMalloc((void**)&s->out, out_bytes) != hipSuccess ||
        hipMalloc((void**)&s->lut, 256 * sizeof(uint16_t)) != hipSuccess) {
        return RAD_E_NOMEM;
    }

    std::lock_guard<std::mutex> lk(g_mu);
    g_state[device] = std::move(s);
    return RAD_OK;
}

extern "C" int r3_ngram_helper_load(int device, int64_t offset, const uint8_t* host_codes, int64_t bytes) {
    if (!host_codes || bytes <= 0 || offset < 0) return RAD_E_INVAL;
    std::lock_guard<std::mutex> lk(g_mu);
    auto it = g_state.find(device);
    if (it == g_state.end()) return RAD_E_STATE;
    State& s = *it->second;
    const int64_t want = s.rows * (int64_t)s.n;
    if (offset + bytes > want) return RAD_E_SHAPE;
    DeviceGuard guard(device);
    if (!guard.ok) return RAD_E_DEVICE;
    return hip_status(hipMemcpy((uint8_t*)s.codes + offset, host_codes, (size_t)bytes, hipMemcpyHostToDevice));
}

extern "C" int r3_ngram_helper_gather(int device, const int32_t* ids, int m, int vocab_offset,
                                       float scale, uint16_t* out_bf16) {
    if (!ids || !out_bf16 || m <= 0) return RAD_E_INVAL;
    std::lock_guard<std::mutex> lk(g_mu);
    auto it = g_state.find(device);
    if (it == g_state.end()) return RAD_E_STATE;
    State& s = *it->second;
    if (m > s.max_m) return RAD_E_SHAPE;

    DeviceGuard guard(device);
    if (!guard.ok) return RAD_E_DEVICE;
    const size_t id_bytes = (size_t)m * sizeof(int32_t);
    const size_t out_bytes = (size_t)m * (size_t)s.n * sizeof(uint16_t);
    uint32_t scale_bits = 0;
    std::memcpy(&scale_bits, &scale, sizeof scale_bits);
    if (!s.lut_ready || s.lut_scale != scale_bits) {
        uint16_t lut[256];
        for (int i = 0; i < 256; ++i)
            lut[i] = rad_f32_to_bf16(rad_fp8e4m3_to_f32((uint8_t)i) * scale);
        if (hipMemcpy(s.lut, lut, sizeof(lut), hipMemcpyHostToDevice) != hipSuccess) return RAD_E_DEVICE;
        s.lut_scale = scale_bits;
        s.lut_ready = true;
    }
    if (hipMemcpy(s.ids, ids, id_bytes, hipMemcpyHostToDevice) != hipSuccess) return RAD_E_DEVICE;
    const uint64_t total = (uint64_t)m * (uint64_t)s.n;
    const dim3 block(256), grid((uint32_t)((total + block.x - 1) / block.x));
    hipLaunchKernelGGL(gather_kernel, grid, block, 0, 0, s.ids, m, vocab_offset,
                       s.first, s.rows, s.codes, s.n, s.lut, s.out);
    if (hipGetLastError() != hipSuccess) return RAD_E_DEVICE;
    return hip_status(hipMemcpy(out_bf16, s.out, out_bytes, hipMemcpyDeviceToHost));
}

extern "C" void r3_ngram_helper_release(int device) {
    std::lock_guard<std::mutex> lk(g_mu);
    g_state.erase(device);
}
