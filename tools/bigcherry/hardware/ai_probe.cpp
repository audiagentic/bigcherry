// PWR12: platform-side numerical screen, not a power writer or engine component.
// Inspiration: Sweet-Spot Tuner v3.5.15 rdna4_aiprobe (MIT), independently
// implemented with the installed SDK and an exact CPU oracle.
// https://gitlab.com/spawnitems-group/sweetspot-tuner/-/tree/v3.5.15
#include <hip/hip_runtime.h>
#include <hipblaslt/hipblaslt.h>
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace {
using Clock = std::chrono::steady_clock;
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
        if (argc != 7) throw Failure(3, "usage: ai-probe BDF arch dtype N seconds pause-ms");
        std::string bdf = argv[1], expected_arch = argv[2], dtype = argv[3];
        Format f = format(dtype);
        int n = parse_int(argv[4], 16, 2048);
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
        Corpus c(n, dtype, f);
        Handles h;
        LT(hipblasLtCreate(&h.h));
        int lt_version = 0; LT(hipblasLtGetVersion(h.h, &lt_version));
        auto setup = hipblasLtMatmulDescCreate(&h.desc, f.compute, f.scale);
        if (setup == HIPBLAS_STATUS_NOT_SUPPORTED) throw Failure(3, "dtype not supported");
        lt(setup, "create matmul descriptor");
        LT(hipblasLtMatrixLayoutCreate(&h.in, f.in, n, n, n));
        LT(hipblasLtMatrixLayoutCreate(&h.out, f.out, n, n, n));
        LT(hipblasLtMatmulPreferenceCreate(&h.pref));
        const size_t limit = 32ull << 20;
        LT(hipblasLtMatmulPreferenceSetAttribute(h.pref, HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &limit, sizeof(limit)));
        hipblasLtMatmulHeuristicResult_t algo{};
        int found = 0;
        auto st = hipblasLtMatmulAlgoGetHeuristic(h.h, h.desc, h.in, h.in, h.out, h.out, h.pref, 1, &algo, &found);
        if (st == HIPBLAS_STATUS_NOT_SUPPORTED || (st == HIPBLAS_STATUS_SUCCESS && !found))
            throw Failure(3, "no supported algorithm");
        lt(st, "get heuristic");
        if (algo.state != HIPBLAS_STATUS_SUCCESS || algo.workspaceSize > limit)
            throw Failure(3, "invalid heuristic");
        Buffer a(c.a.size()), b(c.b.size()), out(c.expected.size()), ws(std::max(size_t(1), algo.workspaceSize));
        HIP(hipMemcpy(a.p, c.a.data(), c.a.size(), hipMemcpyHostToDevice));
        HIP(hipMemcpy(b.p, c.b.data(), c.b.size(), hipMemcpyHostToDevice));
        std::vector<uint8_t> actual(c.expected.size());
        float alpha = 1, beta = 0;
        int32_t ia = 1, ib = 0;
        const void* pa = dtype == "int8" ? static_cast<void*>(&ia) : static_cast<void*>(&alpha);
        const void* pb = dtype == "int8" ? static_cast<void*>(&ib) : static_cast<void*>(&beta);
        // Print only stable, validated strings; algorithm bytes are opaque and
        // compared only within the same library version and executable.
        std::cout << "{\"bdf\":\"" << bdf << "\",\"arch\":\"" << arch
                  << "\",\"dtype\":\"" << dtype << "\",\"n\":" << n
                  << ",\"hip\":" << hip_version << ",\"driver\":" << driver_version
                  << ",\"hipblaslt\":" << lt_version << ",\"workspace\":" << algo.workspaceSize
                  << ",\"algorithm\":\"";
        for (uint8_t v : algo.algo.data) std::cout << std::hex << std::setw(2) << std::setfill('0') << unsigned(v);
        std::cout << std::dec << "\"";
        auto start = Clock::now();
        int64_t iterations = 0;
        do {
            // Poison D before every call: stale/no-write output cannot pass.
            HIP(hipMemset(out.p, 0xff, actual.size()));
            LT(hipblasLtMatmul(h.h, h.desc, pa, a.p, h.in, b.p, h.in, pb,
                              out.p, h.out, out.p, h.out, &algo.algo, ws.p, algo.workspaceSize, nullptr));
            HIP(hipDeviceSynchronize());
            HIP(hipMemcpy(actual.data(), out.p, actual.size(), hipMemcpyDeviceToHost));
            if (actual != c.expected) {
                size_t mismatches = 0;
                for (size_t i = 0; i < actual.size(); i += f.out_bytes) {
                    if (!equal_element(actual.data()+i, c.expected.data()+i, f.out_bytes, dtype == "int8")) {
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
                if (mismatches) {
                    std::cerr << "exact CPU oracle mismatch: " << mismatches << " elements after "
                              << iterations << " clean iterations\n";
                    return 2;
                }
            }
            ++iterations;
            if (pause) std::this_thread::sleep_for(std::chrono::milliseconds(pause));
        } while (iterations < 2 || Clock::now() - start < std::chrono::seconds(seconds));
        std::cout << ",\"iterations\":" << iterations << ",\"elapsed_s\":"
                  << std::chrono::duration<double>(Clock::now() - start).count() << ",\"status\":\"pass\"}\n";
        return 0;
    } catch (const Failure& e) {
        std::cerr << e.what() << '\n';
        return e.code;
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 3;
    }
}
