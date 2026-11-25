// PWR12: platform-side numerical screen, not a power writer or engine component.
// Inspiration: Sweet-Spot Tuner v3.5.15 rdna4_aiprobe (MIT), independently
// implemented with the installed SDK and an exact CPU oracle.
// https://gitlab.com/spawnitems-group/sweetspot-tuner/-/tree/v3.5.15
#include <hip/hip_runtime.h>
#include <hipblaslt/hipblaslt.h>
#include <rocblas/rocblas.h>
#include <time.h>
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace {
struct Failure : std::runtime_error {
    int code;
    Failure(int c, const std::string& s) : std::runtime_error(s), code(c) {}
};
void hip(hipError_t e, const char* what) {
    if (e != hipSuccess) throw Failure(4, std::string(what) + ": " + hipGetErrorString(e));
}
void lt(hipblasStatus_t e, const char* what) {
    if (e == HIPBLAS_STATUS_NOT_SUPPORTED) throw Failure(3, std::string(what) + ": unsupported");
    if (e != HIPBLAS_STATUS_SUCCESS)
        throw Failure(4, std::string(what) + ": " + std::to_string(e));
}
#define HIP(expr) hip((expr), #expr)
#define LT(expr) lt((expr), #expr)
struct Buffer {
    void* p = nullptr;
    explicit Buffer(size_t n) { HIP(hipMalloc(&p, n)); }
    ~Buffer() { if (p) (void)hipFree(p); }
    Buffer(const Buffer&) = delete;
};
struct Handles {
    hipblasLtHandle_t h = nullptr;
    hipblasLtMatmulDesc_t desc = nullptr;
    hipblasLtMatrixLayout_t in = nullptr, out = nullptr;
    hipblasLtMatmulPreference_t pref = nullptr;
    ~Handles() {
        if (pref) hipblasLtMatmulPreferenceDestroy(pref);
        if (out) hipblasLtMatrixLayoutDestroy(out);
        if (in) hipblasLtMatrixLayoutDestroy(in);
        if (desc) hipblasLtMatmulDescDestroy(desc);
        if (h) hipblasLtDestroy(h);
    }
};
void rb(rocblas_status e, const char* what) {
    if (e == rocblas_status_not_implemented) throw Failure(3, std::string(what) + ": unsupported");
    if (e != rocblas_status_success) throw Failure(4, std::string(what) + ": " + std::to_string(e));
}
#define RB(expr) rb((expr), #expr)
struct RocHandle {
    rocblas_handle h = nullptr;
    ~RocHandle() { if (h) (void)rocblas_destroy_handle(h); }
};
rocblas_datatype roc_type(hipDataType t) {
    switch (t) {
    case HIP_R_16F: return rocblas_datatype_f16_r;
    case HIP_R_16BF: return rocblas_datatype_bf16_r;
    case HIP_R_32F: return rocblas_datatype_f32_r;
    case HIP_R_8I: return rocblas_datatype_i8_r;
    case HIP_R_32I: return rocblas_datatype_i32_r;
    default: throw Failure(3, "dtype not supported by rocBLAS screen");
    }
}
int64_t mono_ns() {
    timespec ts{};
    if (clock_gettime(CLOCK_MONOTONIC, &ts)) throw Failure(4, "monotonic clock unavailable");
    return int64_t(ts.tv_sec)*1000000000 + ts.tv_nsec;
}
struct Format {
    hipDataType in, out, scale;
    hipblasComputeType_t compute;
    size_t in_bytes, out_bytes;
};
Format format(const std::string& s) {
    if (s == "fp16") return {HIP_R_16F, HIP_R_16F, HIP_R_32F, HIPBLAS_COMPUTE_32F, 2, 2};
    if (s == "bf16") return {HIP_R_16BF, HIP_R_32F, HIP_R_32F, HIPBLAS_COMPUTE_32F, 2, 4};
    if (s == "fp32") return {HIP_R_32F, HIP_R_32F, HIP_R_32F, HIPBLAS_COMPUTE_32F, 4, 4};
    if (s == "fp8") return {HIP_R_8F_E4M3, HIP_R_32F, HIP_R_32F, HIPBLAS_COMPUTE_32F, 1, 4};
    if (s == "int8") return {HIP_R_8I, HIP_R_32I, HIP_R_32I, HIPBLAS_COMPUTE_32I, 1, 4};
    throw Failure(3, "unknown dtype");
}
// Exact numerical equivalence: only +0/-0 are interchangeable for floats.
bool equal_element(const uint8_t* actual, const uint8_t* expected, size_t bytes, bool integer) {
    if (std::memcmp(actual, expected, bytes) == 0) return true;
    if (integer) return false;
    uint32_t a = 0, e = 0;
    std::memcpy(&a, actual, bytes); std::memcpy(&e, expected, bytes);
    const uint32_t magnitude = bytes == 2 ? 0x7fff : 0x7fffffff;
    return (a & magnitude) == 0 && (e & magnitude) == 0;
}
// Compare every result before the next launch may overwrite it. The CPU
// oracle remains independent; host verifies the last output and reference
// buffer in every batch. Count compared elements as well as mismatches.
template<class T> __global__ void verify_matrix(const T* actual, const T* expected,
                                                size_t elements, bool integer, unsigned* counters) {
    unsigned seen = 0, bad = 0;
    constexpr T magnitude = sizeof(T) == 2 ? T(0x7fff) : T(0x7fffffff);
    for (size_t i = blockIdx.x * size_t(blockDim.x) + threadIdx.x;
         i < elements; i += size_t(blockDim.x) * gridDim.x) {
        T a = actual[i], e = expected[i];
        ++seen;
        bad += a != e && (integer || (a & magnitude) || (e & magnitude));
    }
    __shared__ unsigned seen_by_thread[256], bad_by_thread[256];
    seen_by_thread[threadIdx.x] = seen;
    bad_by_thread[threadIdx.x] = bad;
    __syncthreads();
    for (unsigned stride = blockDim.x / 2; stride; stride /= 2) {
        if (threadIdx.x < stride) {
            seen_by_thread[threadIdx.x] += seen_by_thread[threadIdx.x + stride];
            bad_by_thread[threadIdx.x] += bad_by_thread[threadIdx.x + stride];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        atomicAdd(counters, seen_by_thread[0]);
        if (bad_by_thread[0]) atomicAdd(counters + 1, bad_by_thread[0]);
    }
}
void check_matrix_on_device(void* actual, void* expected, size_t elements,
                            size_t bytes, bool integer, void* counters) {
    const unsigned blocks = unsigned(std::min<size_t>((elements + 255) / 256, 4096));
    if (bytes == 2)
        verify_matrix<<<blocks, 256>>>(static_cast<const uint16_t*>(actual),
            static_cast<const uint16_t*>(expected), elements, integer, static_cast<unsigned*>(counters));
    else
        verify_matrix<<<blocks, 256>>>(static_cast<const uint32_t*>(actual),
            static_cast<const uint32_t*>(expected), elements, integer, static_cast<unsigned*>(counters));
    HIP(hipGetLastError());
}
template<class T> void store(std::vector<uint8_t>& b, size_t i, T value) {
    std::memcpy(b.data() + i * sizeof(T), &value, sizeof(T));
}
// Integer sum / 256 is exactly representable in FP16 for |sum| <= 2048.
// Construct IEEE binary16 directly; no host/GPU rounding dependence.
uint16_t exact_half(int sum) {
    if (!sum) return 0;
    unsigned a = unsigned(sum < 0 ? -sum : sum), p = 0;
    for (unsigned t = a; t > 1; t >>= 1) ++p;
    unsigned sig = (a << (10 - std::min(p, 10u))) >> (p > 10 ? p - 10 : 0);
    return uint16_t((sum < 0 ? 0x8000 : 0) | ((p + 7) << 10) | (sig & 1023));
}
void encode(std::vector<uint8_t>& v, size_t i, int sign, const std::string& dtype) {
    if (dtype == "fp16") store(v, i, uint16_t(sign < 0 ? 0xac00 : 0x2c00));
    else if (dtype == "bf16") store(v, i, uint16_t(sign < 0 ? 0xbd80 : 0x3d80));
    else if (dtype == "fp8") store(v, i, uint8_t(sign < 0 ? 0x98 : 0x18)); // E4M3, not FNUZ
    else if (dtype == "fp32") store(v, i, float(sign) / 16);
    else store(v, i, int8_t(sign));
}
int sign(uint64_t& seed) {
    seed = seed * 6364136223846793005ULL + 1442695040888963407ULL;
    return seed >> 63 ? 1 : -1;
}
struct Corpus {
    std::vector<uint8_t> a, b, expected;
    Corpus(int n, const std::string& dtype, Format f)
        : a(size_t(n)*n*f.in_bytes), b(a.size()), expected(size_t(n)*n*f.out_bytes) {
        std::vector<int8_t> x(size_t(n)*n), y(x.size());
        uint64_t seed = 7;
        for (size_t i = 0; i < x.size(); ++i) {
            x[i] = sign(seed); y[i] = sign(seed);
            encode(a, i, x[i], dtype); encode(b, i, y[i], dtype);
        }
        // Column-major C = A B. Cache-friendly independent integer oracle.
        std::vector<int> sums(x.size(), 0);
        for (int j = 0; j < n; ++j)
            for (int k = 0; k < n; ++k)
                for (int i = 0; i < n; ++i)
                    sums[i+j*n] += x[i+k*n] * y[k+j*n];
        for (size_t i = 0; i < sums.size(); ++i) {
            if (dtype == "fp16") store(expected, i, exact_half(sums[i]));
            else if (dtype == "int8") store(expected, i, int32_t(sums[i]));
            else store(expected, i, float(sums[i]) / 256);
        }
    }
};
// Address-dependent data covers all 32 bits, changes every pass, and makes
// stale/cross-address data observable. Two buffers exercise independent VRAM.
__host__ __device__ uint32_t memory_word(uint64_t i, uint32_t seed) {
    uint32_t x = uint32_t(i) ^ uint32_t(i >> 32) ^ seed;
    x ^= x >> 16; x *= 0x7feb352dU; x ^= x >> 15; x *= 0x846ca68bU; x ^= x >> 16;
    return x;
}
__global__ void memory_transform(uint32_t* dst, const uint32_t* src, size_t words) {
    for (size_t i = blockIdx.x * size_t(blockDim.x) + threadIdx.x;
         i < words; i += size_t(blockDim.x) * gridDim.x)
        dst[i] = ~src[i];
}
void memory_screen(const std::string& bdf, const std::string& arch, int hip_version,
                   int driver_version, int mib, int headroom_mib, int seconds, int pause) {
    size_t free = 0, total = 0;
    HIP(hipMemGetInfo(&free, &total));
    const size_t budget = size_t(mib) << 20, headroom = size_t(headroom_mib) << 20;
    if (budget > free || free - budget < headroom)
        throw Failure(3, "requested memory budget exceeds free VRAM minus reserved headroom");
    const size_t bytes = budget / 2, words = bytes / sizeof(uint32_t);
    Buffer a(bytes), b(bytes);
    size_t free_after = 0, total_after = 0;
    HIP(hipMemGetInfo(&free_after, &total_after));
    if (free_after < headroom || total_after != total)
        throw Failure(3, "memory headroom not preserved after allocation");
    const size_t chunk = std::min(words, size_t(1) << 20);
    std::vector<uint32_t> host(chunk);
    std::cout << "{\"bdf\":\"" << bdf << "\",\"arch\":\"" << arch
              << "\",\"dtype\":\"memory\",\"n\":" << mib
              << ",\"hip\":" << hip_version << ",\"driver\":" << driver_version
              << ",\"provider\":\"hip-memory\",\"library_version\":\"" << hip_version
              << "\",\"workspace\":0,\"algorithm\":\"address-pattern-v1\",\"batch\":1,\"allocated_bytes\":" << budget
              << ",\"free_before_bytes\":" << free << ",\"free_after_bytes\":" << free_after
              << ",\"total_bytes\":" << total;
    const int64_t start = mono_ns();
    int64_t iterations = 0;
    do {
        uint32_t seed = uint32_t(iterations) * 0x9e3779b9U + 0x712db419U;
        for (size_t off = 0; off < words; off += chunk) {
            size_t len = std::min(chunk, words-off);
            for (size_t i=0;i<len;++i) host[i] = memory_word(off+i, seed);
            HIP(hipMemcpy(static_cast<uint32_t*>(a.p)+off, host.data(), len*sizeof(uint32_t), hipMemcpyHostToDevice));
        }
        HIP(hipMemcpy(b.p, a.p, bytes, hipMemcpyDeviceToDevice));
        memory_transform<<<std::min<size_t>((words+255)/256, 4096), 256>>>(
            static_cast<uint32_t*>(a.p), static_cast<const uint32_t*>(b.p), words);
        HIP(hipGetLastError());
        HIP(hipDeviceSynchronize());
        for (size_t off=0; off<words; off+=chunk) {
            size_t len = std::min(chunk, words-off);
            for (int copy=0;copy<2;++copy) {
                HIP(hipMemcpy(host.data(), static_cast<uint32_t*>(copy?a.p:b.p)+off,
                              len*sizeof(uint32_t), hipMemcpyDeviceToHost));
                for (size_t i=0;i<len;++i) {
                    uint32_t want = memory_word(off+i, seed);
                    if (copy) want = ~want;
                    if (host[i] != want) throw Failure(2, "memory/transfer mismatch at word " + std::to_string(off+i));
                }
            }
        }
        ++iterations;
        if (pause) std::this_thread::sleep_for(std::chrono::milliseconds(pause));
    } while (iterations < 2 || mono_ns()-start < int64_t(seconds)*1000000000);
    int64_t end = mono_ns();
    std::cout << ",\"iterations\":" << iterations << ",\"elapsed_s\":" << std::setprecision(9)
              << double(end-start)/1e9 << ",\"start_ns\":" << start << ",\"end_ns\":" << end
              << ",\"status\":\"pass\"}\n";
}

int parse_int(const char* s, int lo, int hi) {
    size_t used = 0;
    int v = std::stoi(s, &used);
    if (used != std::strlen(s) || v < lo || v > hi) throw Failure(3, "integer argument out of range");
    return v;
}
} // namespace

// Called by the lease/thermal/profile-aware platform command, one type per process.
// Exit 0 = screen passed; 2 = numerical mismatch; 3 = unavailable/invalid; 4 = device/runtime failure.
int main(int argc, char** argv) {
    try {
        if (argc == 2 && std::string(argv[1]) == "--self-test") {
            // Exact small CPU oracle and conversion tests, with no HIP calls.
            if (exact_half(1) != 0x1c00 || exact_half(-256) != 0xbc00 ||
                exact_half(2048) != 0x4800 || exact_half(0) != 0) return 2;
            auto f = format("int8");
            Corpus c(16, "int8", f);
            int32_t first = 0; std::memcpy(&first, c.expected.data(), sizeof(first));
            int32_t manual = 0;
            for (int k=0;k<16;++k) manual += int8_t(c.a[k*16])*int8_t(c.b[k]);
            if (first != manual) return 2;
            uint16_t pos = 0, neg = 0x8000, nonzero = 1;
            if (!equal_element(reinterpret_cast<uint8_t*>(&pos), reinterpret_cast<uint8_t*>(&neg), 2, false) ||
                equal_element(reinterpret_cast<uint8_t*>(&pos), reinterpret_cast<uint8_t*>(&nonzero), 2, false)) return 2;
            uint32_t pos32 = 0, neg32 = 0x80000000;
            if (!equal_element(reinterpret_cast<uint8_t*>(&pos32), reinterpret_cast<uint8_t*>(&neg32), 4, false) ||
                equal_element(reinterpret_cast<uint8_t*>(&pos32), reinterpret_cast<uint8_t*>(&neg32), 4, true)) return 2;
            std::cout << "CPU oracle self-test passed\n";
            return 0;
        }
        if (argc != 10) throw Failure(3, "usage: ai-probe BDF arch dtype N seconds pause-ms provider headroom-MiB batch");
        std::string bdf = argv[1], expected_arch = argv[2], dtype = argv[3];
        std::string provider = argv[7];
        int n = parse_int(argv[4], dtype == "memory" ? 2 : 16, dtype == "memory" ? 65536 : 2048);
        if (dtype == "memory" && n % 2) throw Failure(3, "memory budget must be even MiB");
        int headroom = parse_int(argv[8], 256, 65536);
        int batch = parse_int(argv[9], 1, 256);
        int seconds = parse_int(argv[5], 1, 3600);
        int pause = parse_int(argv[6], 0, 1000);
        int count = 0, dev = -1;
        HIP(hipGetDeviceCount(&count));
        for (int i = 0; i < count; ++i) {
            char bus[32] = {};
            HIP(hipDeviceGetPCIBusId(bus, sizeof(bus), i));
            if (bdf == bus) {
                if (dev != -1) throw Failure(3, "ambiguous BDF");
                dev = i;
            }
        }
        if (dev < 0) throw Failure(3, "requested BDF not HIP-visible");
        HIP(hipSetDevice(dev));
        int current = -1; HIP(hipGetDevice(&current));
        if (current != dev) throw Failure(3, "device selection mismatch");
        hipDeviceProp_t prop{}; HIP(hipGetDeviceProperties(&prop, current));
        std::string arch(prop.gcnArchName);
        arch = arch.substr(0, arch.find(':'));
        if (arch != expected_arch) throw Failure(3, "architecture mismatch");
        int hip_version = 0, driver_version = 0;
        HIP(hipRuntimeGetVersion(&hip_version)); HIP(hipDriverGetVersion(&driver_version));
        if (dtype == "memory") {
            if (provider != "hip-memory" || batch != 1) throw Failure(3, "memory requires hip-memory and batch 1");
            memory_screen(bdf, arch, hip_version, driver_version, n, headroom, seconds, pause);
            return 0;
        }
        if (provider != "hipblaslt" && provider != "rocblas") throw Failure(3, "unknown provider");
        Format f = format(dtype);
        Corpus c(n, dtype, f);
        Handles h;
        RocHandle r;
        std::string version, algorithm;
        hipblasLtMatmulHeuristicResult_t algo{};
        size_t workspace = 0;
        rocblas_datatype rin{}, rout{}, rcompute{};
        if (provider == "hipblaslt") {
            LT(hipblasLtCreate(&h.h));
            int lt_version = 0; LT(hipblasLtGetVersion(h.h, &lt_version));
            version = std::to_string(lt_version);
            auto setup = hipblasLtMatmulDescCreate(&h.desc, f.compute, f.scale);
            if (setup == HIPBLAS_STATUS_NOT_SUPPORTED) throw Failure(3, "dtype not supported");
            lt(setup, "create matmul descriptor");
            LT(hipblasLtMatrixLayoutCreate(&h.in, f.in, n, n, n));
            LT(hipblasLtMatrixLayoutCreate(&h.out, f.out, n, n, n));
            LT(hipblasLtMatmulPreferenceCreate(&h.pref));
            const size_t limit = 32ull << 20;
            LT(hipblasLtMatmulPreferenceSetAttribute(h.pref, HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &limit, sizeof(limit)));
            int found = 0;
            auto st = hipblasLtMatmulAlgoGetHeuristic(h.h, h.desc, h.in, h.in, h.out, h.out, h.pref, 1, &algo, &found);
            if (st == HIPBLAS_STATUS_NOT_SUPPORTED || (st == HIPBLAS_STATUS_SUCCESS && !found))
                throw Failure(3, "no supported algorithm");
            lt(st, "get heuristic");
            if (algo.state != HIPBLAS_STATUS_SUCCESS || algo.workspaceSize > limit)
                throw Failure(3, "invalid heuristic");
            workspace = algo.workspaceSize;
            const char* hex = "0123456789abcdef";
            for (uint8_t v : algo.algo.data) { algorithm += hex[v >> 4]; algorithm += hex[v & 15]; }
        } else {
            rin = roc_type(f.in); rout = roc_type(f.out); rcompute = roc_type(f.scale);
            RB(rocblas_create_handle(&r.h));
            RB(rocblas_set_atomics_mode(r.h, rocblas_atomics_not_allowed));
            char v[128]{};
            RB(rocblas_get_version_string(v, sizeof(v)));
            version = v;
            // Stable public API: no beta solution enumeration; honestly record
            // default dispatch, not an invented exact kernel identity.
            algorithm = "rocblas-default-no-atomics";
        }
        Buffer a(c.a.size()), b(c.b.size()), out(c.expected.size()), ws(std::max(size_t(1), workspace));
        HIP(hipMemcpy(a.p, c.a.data(), c.a.size(), hipMemcpyHostToDevice));
        HIP(hipMemcpy(b.p, c.b.data(), c.b.size(), hipMemcpyHostToDevice));
        std::vector<uint8_t> actual(c.expected.size());
        float alpha = 1, beta = 0;
        int32_t ia = 1, ib = 0;
        const void* pa = dtype == "int8" ? static_cast<void*>(&ia) : static_cast<void*>(&alpha);
        const void* pb = dtype == "int8" ? static_cast<void*>(&ib) : static_cast<void*>(&beta);
        const size_t elements = size_t(n) * n;
        const bool integer = dtype == "int8";
        std::unique_ptr<Buffer> expected, counters;
        // Only the explicit batched mode uses the device-side verifier.
        // With batch <= 256 and n <= 2048, counters cannot overflow.
        if (batch > 1) {
            expected = std::make_unique<Buffer>(c.expected.size());
            counters = std::make_unique<Buffer>(2 * sizeof(unsigned));
            HIP(hipMemcpy(expected->p, c.expected.data(), c.expected.size(), hipMemcpyHostToDevice));
        }
        auto verify_counts = [&](unsigned want_seen, unsigned want_bad) {
            unsigned counts[2] = {};
            HIP(hipDeviceSynchronize());
            HIP(hipMemcpy(counts, counters->p, sizeof(counts), hipMemcpyDeviceToHost));
            if (counts[0] != want_seen || counts[1] != want_bad)
                throw Failure(2, "device verifier count/mismatch: " + std::to_string(counts[0]) +
                                  "/" + std::to_string(want_seen) + " compared, " +
                                  std::to_string(counts[1]) + "/" + std::to_string(want_bad) + " wrong");
        };
        auto verifier_selfcheck = [&]() {
            if (batch == 1) return;
            // Exercise the same comparator with a known one-element error.
            HIP(hipMemcpy(out.p, c.expected.data(), c.expected.size(), hipMemcpyHostToDevice));
            uint8_t flipped = c.expected[0] ^ 1;
            HIP(hipMemcpy(out.p, &flipped, 1, hipMemcpyHostToDevice));
            HIP(hipMemsetAsync(counters->p, 0, 2 * sizeof(unsigned)));
            check_matrix_on_device(out.p, expected->p, elements, f.out_bytes, integer, counters->p);
            verify_counts(unsigned(elements), 1);
        };
        auto compare_host = [&](int64_t iterations) {
            HIP(hipMemcpy(actual.data(), out.p, actual.size(), hipMemcpyDeviceToHost));
            if (actual == c.expected) return;
            size_t mismatches = 0;
            for (size_t i = 0; i < actual.size(); i += f.out_bytes) {
                if (!equal_element(actual.data()+i, c.expected.data()+i, f.out_bytes, integer)) {
                    if (mismatches < 4) {
                        uint32_t a_bits = 0, e_bits = 0;
                        std::memcpy(&a_bits, actual.data()+i, f.out_bytes);
                        std::memcpy(&e_bits, c.expected.data()+i, f.out_bytes);
                        std::cerr << "index " << i/f.out_bytes << ": observed 0x" << std::hex << a_bits
                                  << " expected 0x" << e_bits << std::dec << '\n';
                    }
                    ++mismatches;
                }
            }
            if (mismatches)
                throw Failure(2, "exact CPU oracle mismatch: " + std::to_string(mismatches) +
                                 " elements after " + std::to_string(iterations) + " clean iterations");
        };
        verifier_selfcheck();
        std::cout << "{\"bdf\":\"" << bdf << "\",\"arch\":\"" << arch
                  << "\",\"dtype\":\"" << dtype << "\",\"n\":" << n
                  << ",\"hip\":" << hip_version << ",\"driver\":" << driver_version
                  << ",\"provider\":\"" << provider << "\",\"library_version\":\"" << version
                  << "\",\"workspace\":" << workspace << ",\"algorithm\":\"" << algorithm
                  << "\",\"batch\":" << batch;
        auto start = mono_ns();
        int64_t iterations = 0, host_checks = 0;
        do {
            if (batch > 1) HIP(hipMemsetAsync(counters->p, 0, 2 * sizeof(unsigned)));
            for (int j=0; j<batch; ++j) {
                // All operations use the same default stream, in issue order.
                HIP(hipMemsetAsync(out.p, 0xff, actual.size()));
                if (provider == "hipblaslt") {
                    LT(hipblasLtMatmul(h.h, h.desc, pa, a.p, h.in, b.p, h.in, pb,
                                      out.p, h.out, out.p, h.out, &algo.algo, ws.p, workspace, nullptr));
                } else {
                    RB(rocblas_gemm_ex(r.h, rocblas_operation_none, rocblas_operation_none,
                                      n,n,n,pa,a.p,rin,n,b.p,rin,n,pb,out.p,rout,n,out.p,rout,n,
                                      rcompute,rocblas_gemm_algo_standard,0,0));
                }
                if (batch > 1)
                    check_matrix_on_device(out.p, expected->p, elements, f.out_bytes, integer, counters->p);
            }
            if (batch > 1) verify_counts(unsigned(elements * batch), 0);
            else HIP(hipDeviceSynchronize());
            compare_host(iterations);
            if (batch > 1) {
                HIP(hipMemcpy(actual.data(), expected->p, actual.size(), hipMemcpyDeviceToHost));
                if (actual != c.expected) throw Failure(2, "GPU reference buffer changed");
            }
            iterations += batch;
            ++host_checks;
            if (pause) std::this_thread::sleep_for(std::chrono::milliseconds(pause));
        } while (iterations < 2 || mono_ns()-start < int64_t(seconds)*1000000000);
        int64_t end = mono_ns();
        verifier_selfcheck();
        std::cout << ",\"iterations\":" << iterations << ",\"host_checks\":" << host_checks
                  << ",\"elapsed_s\":" << std::setprecision(9) << double(end-start)/1e9
                  << ",\"start_ns\":" << start << ",\"end_ns\":" << end << ",\"status\":\"pass\"}\n";
        return 0;
    } catch (const Failure& e) {
        std::cerr << e.what() << '\n';
        return e.code;
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 3;
    }
}
