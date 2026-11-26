// PWR12: platform-side numerical screen, not a power writer or engine component.
// Inspiration: Sweet-Spot Tuner v3.5.15 rdna4_aiprobe (MIT), independently
// implemented with the installed SDK and an exact CPU oracle.
// https://gitlab.com/spawnitems-group/sweetspot-tuner/-/tree/v3.5.15
#include <hip/hip_runtime.h>
#include <hipblaslt/hipblaslt.h>
#include <rocblas/rocblas.h>
#include <time.h>
#include <algorithm>
#include <array>
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
constexpr int protocol_version = 3;
constexpr size_t corpus_count = 3;
constexpr size_t guard_bytes = 256;
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
    if (sign == 0) return; // value-initialized corpus buffers
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
    Corpus(int n, const std::string& dtype, Format f, size_t pattern)
        : a(size_t(n)*n*f.in_bytes), b(a.size()), expected(size_t(n)*n*f.out_bytes) {
        std::vector<int8_t> x(size_t(n)*n), y(x.size());
        if (pattern >= corpus_count) throw Failure(3, "invalid corpus pattern");
        uint64_t seed = 7 + pattern * 0x9e3779b97f4a7c15ULL;
        for (size_t i = 0; i < x.size(); ++i) {
            x[i] = sign(seed); y[i] = sign(seed);
            if (pattern == 1) {
                if (i % 5 < 2) x[i] = 0;
                if (i % 7 < 3) y[i] = 0;
            } else if (pattern == 2) {
                x[i] = i % n == i / n ? ((i / n) % 2 ? -1 : 1) : 0;
            }
            encode(a, i, x[i], dtype); encode(b, i, y[i], dtype);
        }
        // Column-major C = A B. Cache-friendly independent integer oracle.
        std::vector<int> sums(x.size(), 0);
        for (int j = 0; j < n; ++j)
            for (int k = 0; k < n; ++k) {
                const int v = y[k+j*n];
                if (v == 0) continue;
                if (pattern == 2) sums[k+j*n] = x[k+k*n] * v;
                else for (int i = 0; i < n; ++i) sums[i+j*n] += x[i+k*n] * v;
            }
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
    std::cout << "{\"protocol_version\":" << protocol_version << ",\"bdf\":\"" << bdf << "\",\"arch\":\"" << arch
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

// Independent instruction cases. Each lane returns four exact uint32 words;
// all outputs are read by the host, not a second GPU-side oracle.
constexpr unsigned instruction_block = 256;
struct Words {
    uint32_t a, b, c, d;
};
static_assert(sizeof(Words) == 4*sizeof(uint32_t));
enum class Instruction { alu, wave, lds };
const char* instruction_algorithm(Instruction kind) {
    switch (kind) {
    case Instruction::alu: return "integer-wrap-v1";
    case Instruction::wave: return "wave-exchange-v1";
    case Instruction::lds: return "lds-exchange-v1";
    }
    throw Failure(3, "unknown instruction case");
}
__global__ void instruction_alu(const uint32_t* input, Words* out, uint32_t seed) {
    const unsigned i = blockIdx.x*blockDim.x + threadIdx.x;
    const uint32_t a = input[i] ^ seed;
    const uint32_t b = input[i^1u] ^ (seed+0x9e3779b9u);
    const unsigned shift = (seed & 15u)+1;
    out[i] = {a*b+(b^seed), __umulhi(a,b),
              ((a << shift)|(a >> (32-shift)))^b,
              (a >> ((seed & 3u)*8)) & 255u};
}
__global__ void instruction_wave(const uint32_t* input, Words* out, uint32_t seed) {
    const unsigned i = blockIdx.x*blockDim.x + threadIdx.x;
    const unsigned width = warpSize, lane = threadIdx.x % width;
    const uint32_t v = input[i] ^ seed;
    const unsigned offset = (seed & (width-1))|1u;
    const unsigned mask = ((seed >> 8) & (width-1))|1u;
    const auto votes = __ballot((v >> (seed&31u)) & 1u);
    out[i] = {__shfl(v, (lane+offset)%width, width), __shfl_xor(v, mask, width),
              uint32_t(votes), uint32_t(uint64_t(votes)>>32)};
}
__global__ void instruction_lds(const uint32_t* input, Words* out, uint32_t seed) {
    __shared__ uint32_t scratch[instruction_block];
    const unsigned lane = threadIdx.x, i = blockIdx.x*blockDim.x + lane;
    const unsigned offset = ((seed >> 8)&(instruction_block-1))|1u;
    const uint32_t v = input[i]^seed;
    scratch[lane] = v;
    __syncthreads();
    const uint32_t a = scratch[(lane+offset)&(instruction_block-1)];
    __syncthreads(); // all old readers complete before reusing LDS
    scratch[lane] = a ^ (0x9e3779b9u*(lane+1));
    __syncthreads();
    out[i] = {a, scratch[lane^(instruction_block/2)], v, lane};
}
bool same_words(const Words& a, const Words& b) {
    return a.a==b.a && a.b==b.b && a.c==b.c && a.d==b.d;
}
void instruction_reference(Instruction kind, const std::vector<uint32_t>& input,
                           std::vector<Words>& expected, unsigned width, uint32_t seed) {
    if (kind == Instruction::wave) {
        for (size_t base=0; base<input.size(); base+=width) {
            uint64_t votes = 0;
            for (unsigned lane=0; lane<width; ++lane)
                if (((input[base+lane]^seed) >> (seed%32))&1u) votes |= uint64_t(1)<<lane;
            const unsigned offset = (seed%width)|1u, mask = ((seed/256)%width)|1u;
            for (unsigned lane=0; lane<width; ++lane)
                expected[base+lane] = {input[base+(lane+offset)%width]^seed,
                                       input[base+(lane^mask)]^seed,
                                       uint32_t(votes), uint32_t(votes>>32)};
        }
        return;
    }
    for (size_t i=0; i<input.size(); ++i) {
        const uint32_t a = input[i]^seed;
        if (kind == Instruction::alu) {
            const uint32_t b = input[i^size_t(1)]^(seed+0x9e3779b9u);
            const uint64_t product = uint64_t(a)*b;
            const unsigned shift = seed%16+1;
            const uint64_t expanded = uint64_t(a)*(uint64_t(1)<<shift);
            const uint32_t rotate = uint32_t(expanded)+uint32_t(expanded>>32);
            expected[i] = {uint32_t(product+(b^seed)), uint32_t(product>>32),
                           rotate^b, uint32_t(a/(uint64_t(1)<<(8*(seed%4))))%256};
        } else {
            const size_t base = i/instruction_block*instruction_block;
            const unsigned lane = i%instruction_block, other = lane^(instruction_block/2);
            const unsigned offset = ((seed/256)%instruction_block)|1u;
            const uint32_t first = input[base+(lane+offset)%instruction_block]^seed;
            const uint32_t second = input[base+(other+offset)%instruction_block]^seed;
            expected[i] = {first, second ^ uint32_t(uint64_t(0x9e3779b9u)*(other+1)), a, lane};
        }
    }
}
void instruction_screen(Instruction kind, const std::string& dtype, const std::string& bdf,
                        const std::string& arch, int hip_version, int driver_version,
                        unsigned width, int blocks, int headroom_mib, int seconds, int pause) {
    if ((width!=32 && width!=64) || instruction_block%width)
        throw Failure(3, "unsupported HIP wave width");
    const size_t elements = size_t(blocks)*instruction_block;
    const size_t input_bytes = elements*sizeof(uint32_t), output_bytes = elements*sizeof(Words);
    const size_t budget = input_bytes+output_bytes+2*guard_bytes;
    const size_t reserve = size_t(headroom_mib)<<20;
    size_t free=0, total=0;
    HIP(hipMemGetInfo(&free,&total));
    if (budget>free || free-budget<reserve)
        throw Failure(3, "instruction allocation exceeds VRAM headroom");
    Buffer device_input(input_bytes), device_output(output_bytes+2*guard_bytes);
    size_t after=0, total_after=0;
    HIP(hipMemGetInfo(&after,&total_after));
    if (after<reserve || total_after!=total)
        throw Failure(3, "instruction allocation did not preserve headroom");
    auto* out = reinterpret_cast<Words*>(static_cast<uint8_t*>(device_output.p)+guard_bytes);
    std::vector<uint32_t> input(elements), input_check(elements);
    std::vector<Words> actual(elements), expected(elements);
    for (size_t i=0;i<elements;++i) input[i]=memory_word(i,0x712db419u);
    HIP(hipMemcpy(device_input.p,input.data(),input_bytes,hipMemcpyHostToDevice));
    auto check_input = [&] {
        HIP(hipMemcpy(input_check.data(),device_input.p,input_bytes,hipMemcpyDeviceToHost));
        if (input!=input_check) throw Failure(2,"instruction input changed");
    };
    HIP(hipMemset(device_output.p,0xa5,output_bytes+2*guard_bytes));
    check_input();
    const int64_t start=mono_ns();
    int64_t iterations=0;
    do {
        const uint32_t seed=uint32_t(iterations)*0x9e3779b9u+0x712db419u;
        instruction_reference(kind,input,expected,width,seed);
        HIP(hipMemsetAsync(out,0xff,output_bytes));
        auto* in = static_cast<const uint32_t*>(device_input.p);
        switch (kind) {
        case Instruction::alu: instruction_alu<<<blocks,instruction_block>>>(in,out,seed); break;
        case Instruction::wave: instruction_wave<<<blocks,instruction_block>>>(in,out,seed); break;
        case Instruction::lds: instruction_lds<<<blocks,instruction_block>>>(in,out,seed); break;
        }
        HIP(hipGetLastError());
        HIP(hipDeviceSynchronize());
        HIP(hipMemcpy(actual.data(),out,output_bytes,hipMemcpyDeviceToHost));
        for (size_t i=0;i<elements;++i)
            if (!same_words(actual[i],expected[i]))
                throw Failure(2,"instruction oracle mismatch at lane "+std::to_string(i)+
                              " after "+std::to_string(iterations)+" clean launches");
        std::array<uint8_t,guard_bytes> guard{};
        for (size_t off : {size_t(0),guard_bytes+output_bytes}) {
            HIP(hipMemcpy(guard.data(),static_cast<uint8_t*>(device_output.p)+off,
                          guard.size(),hipMemcpyDeviceToHost));
            if (!std::all_of(guard.begin(),guard.end(),[](uint8_t v){return v==0xa5;}))
                throw Failure(2,"instruction output guard changed");
        }
        ++iterations;
        if (pause) std::this_thread::sleep_for(std::chrono::milliseconds(pause));
    } while (iterations<2 || mono_ns()-start<int64_t(seconds)*1000000000);
    const int64_t end=mono_ns();
    check_input();
    std::cout << "{\"protocol_version\":" << protocol_version << ",\"bdf\":\"" << bdf << "\",\"arch\":\"" << arch
              << "\",\"dtype\":\"" << dtype << "\",\"n\":" << blocks
              << ",\"hip\":" << hip_version << ",\"driver\":" << driver_version
              << ",\"provider\":\"hip-native\",\"library_version\":\"" << hip_version
              << "\",\"workspace\":0,\"algorithm\":\"" << instruction_algorithm(kind)
              << "\",\"batch\":1,\"block_size\":" << instruction_block << ",\"wave_size\":" << width
              << ",\"iterations\":" << iterations << ",\"host_checks\":" << iterations
              << ",\"checked_lanes\":" << uint64_t(iterations)*elements
              << ",\"host_elements_checked\":" << uint64_t(iterations)*elements*4
              << ",\"guard_checks\":" << uint64_t(iterations)*2 << ",\"input_checks\":2"
              << ",\"allocated_bytes\":" << budget << ",\"free_before_bytes\":" << free
              << ",\"free_after_bytes\":" << after
              << ",\"elapsed_s\":" << std::setprecision(9) << double(end-start)/1e9
              << ",\"start_ns\":" << start << ",\"end_ns\":" << end << ",\"status\":\"pass\"}\n";
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
        if (argc == 2 && std::string(argv[1]) == "--protocol-version") {
            std::cout << protocol_version << '\n';
            return 0;
        }
        if (argc == 2 && std::string(argv[1]) == "--self-test") {
            // Exact small CPU oracle and conversion tests, with no HIP calls.
            if (exact_half(1) != 0x1c00 || exact_half(-256) != 0xbc00 ||
                exact_half(2048) != 0x4800 || exact_half(0) != 0) return 2;
            auto f = format("int8");
            std::vector<uint8_t> previous;
            for (size_t pattern = 0; pattern < corpus_count; ++pattern) {
                Corpus c(17, "int8", f, pattern);
                if (c.expected == previous) return 2;
                previous = c.expected;
                for (int j=0;j<17;++j) for (int i=0;i<17;++i) {
                    int32_t want = 0, got = 0;
                    for (int k=0;k<17;++k) want += int8_t(c.a[i+k*17])*int8_t(c.b[k+j*17]);
                    std::memcpy(&got, c.expected.data()+(i+j*17)*sizeof(got), sizeof(got));
                    if (want != got) return 2;
                }
            }
            uint16_t pos = 0, neg = 0x8000, nonzero = 1;
            if (!equal_element(reinterpret_cast<uint8_t*>(&pos), reinterpret_cast<uint8_t*>(&neg), 2, false) ||
                equal_element(reinterpret_cast<uint8_t*>(&pos), reinterpret_cast<uint8_t*>(&nonzero), 2, false)) return 2;
            uint32_t pos32 = 0, neg32 = 0x80000000;
            if (!equal_element(reinterpret_cast<uint8_t*>(&pos32), reinterpret_cast<uint8_t*>(&neg32), 4, false) ||
                equal_element(reinterpret_cast<uint8_t*>(&pos32), reinterpret_cast<uint8_t*>(&neg32), 4, true)) return 2;
            std::vector<uint32_t> lanes(instruction_block);
            for (unsigned i=0;i<lanes.size();++i) lanes[i]=i;
            std::vector<Words> words(lanes.size());
            instruction_reference(Instruction::alu,lanes,words,32,0);
            if (!same_words(words[0],{0x9e3779b8u,0,0x9e3779b8u,0})) return 2;
            instruction_reference(Instruction::wave,lanes,words,32,0);
            if (!same_words(words[0],{1,1,0xaaaaaaaau,0}) ||
                !same_words(words[31],{0,30,0xaaaaaaaau,0})) return 2;
            instruction_reference(Instruction::wave,lanes,words,64,0);
            if (!same_words(words[63],{0,62,0xaaaaaaaau,0xaaaaaaaau})) return 2;
            instruction_reference(Instruction::lds,lanes,words,32,0);
            if (!same_words(words[0],{1,129u^uint32_t(0x9e3779b9ull*129),0,0}) ||
                same_words(words[0],{0,words[0].b,0,0})) return 2;
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
        if (dtype=="alu" || dtype=="wave" || dtype=="lds") {
            if (provider!="hip-native" || batch!=1)
                throw Failure(3,"instruction cases require hip-native and batch 1");
            const auto kind = dtype=="alu" ? Instruction::alu : dtype=="wave" ? Instruction::wave : Instruction::lds;
            instruction_screen(kind,dtype,bdf,arch,hip_version,driver_version,prop.warpSize,
                               n,headroom,seconds,pause);
            return 0;
        }
        if (provider != "hipblaslt" && provider != "rocblas") throw Failure(3, "unknown provider");
        Format f = format(dtype);
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
        struct Slot {
            Corpus c;
            Buffer a, b, out, expected;
            int64_t launches = 0, checks = 0;
            Slot(int n, const std::string& dtype, Format f, size_t pattern)
                : c(n, dtype, f, pattern), a(c.a.size()), b(c.b.size()),
                  out(c.expected.size()+2*guard_bytes), expected(c.expected.size()) {
                HIP(hipMemcpy(a.p, c.a.data(), c.a.size(), hipMemcpyHostToDevice));
                HIP(hipMemcpy(b.p, c.b.data(), c.b.size(), hipMemcpyHostToDevice));
                HIP(hipMemcpy(expected.p, c.expected.data(), c.expected.size(), hipMemcpyHostToDevice));
                HIP(hipMemset(out.p, 0xa5, c.expected.size()+2*guard_bytes));
            }
            void* data() { return static_cast<uint8_t*>(out.p)+guard_bytes; }
        };
        std::vector<std::unique_ptr<Slot>> slots;
        for (size_t i=0;i<corpus_count;++i)
            slots.push_back(std::make_unique<Slot>(n,dtype,f,i));
        Buffer ws(std::max(size_t(1), workspace));
        std::vector<uint8_t> actual(slots[0]->c.expected.size());
        float alpha = 1, beta = 0;
        int32_t ia = 1, ib = 0;
        const void* pa = dtype == "int8" ? static_cast<void*>(&ia) : static_cast<void*>(&alpha);
        const void* pb = dtype == "int8" ? static_cast<void*>(&ib) : static_cast<void*>(&beta);
        const size_t elements = size_t(n) * n;
        const bool integer = dtype == "int8";
        std::unique_ptr<Buffer> counters;
        // With batch <= 256 and n <= 2048, per-batch counters cannot overflow.
        if (batch > 1) counters = std::make_unique<Buffer>(2 * sizeof(unsigned));
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
            for (auto& ptr : slots) {
                auto& s = *ptr;
                for (size_t i : {size_t(0), elements/2, elements-1}) {
                    HIP(hipMemcpy(s.data(), s.c.expected.data(), actual.size(), hipMemcpyHostToDevice));
                    HIP(hipMemsetAsync(counters->p, 0, 2 * sizeof(unsigned)));
                    check_matrix_on_device(s.data(), s.expected.p, elements, f.out_bytes, integer, counters->p);
                    verify_counts(unsigned(elements), 0);
                    size_t off = i*f.out_bytes;
                    uint8_t flipped = s.c.expected[off] ^ 1;
                    HIP(hipMemcpy(static_cast<uint8_t*>(s.data())+off, &flipped, 1, hipMemcpyHostToDevice));
                    HIP(hipMemsetAsync(counters->p, 0, 2 * sizeof(unsigned)));
                    check_matrix_on_device(s.data(), s.expected.p, elements, f.out_bytes, integer, counters->p);
                    verify_counts(unsigned(elements), 1);
                }
            }
        };
        auto check_bytes = [&](void* device, const std::vector<uint8_t>& want, const char* what) {
            std::vector<uint8_t> got(want.size());
            HIP(hipMemcpy(got.data(), device, got.size(), hipMemcpyDeviceToHost));
            if (got != want) throw Failure(2, what);
        };
        auto check_inputs = [&](Slot& s) {
            check_bytes(s.a.p, s.c.a, "input A changed");
            check_bytes(s.b.p, s.c.b, "input B changed");
            check_bytes(s.expected.p, s.c.expected, "GPU reference buffer changed");
        };
        auto compare_host = [&](Slot& s, int64_t iterations) {
            std::array<uint8_t, guard_bytes> guard{};
            for (size_t off : {size_t(0), guard_bytes+actual.size()}) {
                HIP(hipMemcpy(guard.data(), static_cast<uint8_t*>(s.out.p)+off,
                              guard.size(), hipMemcpyDeviceToHost));
                if (!std::all_of(guard.begin(), guard.end(), [](uint8_t b){return b==0xa5;}))
                    throw Failure(2, "output guard changed");
            }
            HIP(hipMemcpy(actual.data(), s.data(), actual.size(), hipMemcpyDeviceToHost));
            if (actual != s.c.expected) {
                size_t mismatches = 0;
                for (size_t i = 0; i < actual.size(); i += f.out_bytes)
                    if (!equal_element(actual.data()+i, s.c.expected.data()+i, f.out_bytes, integer))
                        ++mismatches;
                if (mismatches) throw Failure(2, "exact CPU oracle mismatch: " + std::to_string(mismatches) +
                                                  " elements after " + std::to_string(iterations) + " clean iterations");
            }
            check_bytes(s.expected.p, s.c.expected, "GPU reference buffer changed");
            ++s.checks;
        };
        for (auto& s : slots) check_inputs(*s);
        verifier_selfcheck();
        std::cout << "{\"protocol_version\":" << protocol_version << ",\"bdf\":\"" << bdf << "\",\"arch\":\"" << arch
                  << "\",\"dtype\":\"" << dtype << "\",\"n\":" << n
                  << ",\"hip\":" << hip_version << ",\"driver\":" << driver_version
                  << ",\"provider\":\"" << provider << "\",\"library_version\":\"" << version
                  << "\",\"workspace\":" << workspace << ",\"algorithm\":\"" << algorithm
                  << "\",\"batch\":" << batch;
        auto start = mono_ns();
        int64_t iterations = 0, host_checks = 0;
        do {
            std::array<bool, corpus_count> used{};
            if (batch > 1) HIP(hipMemsetAsync(counters->p, 0, 2 * sizeof(unsigned)));
            for (int j=0; j<batch; ++j) {
                const size_t index = size_t(iterations+j) % corpus_count;
                auto& s = *slots[index];
                used[index] = true;
                HIP(hipMemsetAsync(s.data(), integer ? 0x80 : 0xff, actual.size()));
                if (provider == "hipblaslt") {
                    LT(hipblasLtMatmul(h.h, h.desc, pa, s.a.p, h.in, s.b.p, h.in, pb,
                                      s.data(), h.out, s.data(), h.out, &algo.algo, ws.p, workspace, nullptr));
                } else {
                    RB(rocblas_gemm_ex(r.h, rocblas_operation_none, rocblas_operation_none,
                                      n,n,n,pa,s.a.p,rin,n,s.b.p,rin,n,pb,s.data(),rout,n,s.data(),rout,n,
                                      rcompute,rocblas_gemm_algo_standard,0,0));
                }
                if (batch > 1)
                    check_matrix_on_device(s.data(), s.expected.p, elements, f.out_bytes, integer, counters->p);
                ++s.launches;
            }
            if (batch > 1) verify_counts(unsigned(elements * batch), 0);
            else HIP(hipDeviceSynchronize());
            for (size_t i=0;i<corpus_count;++i) if (used[i]) {
                compare_host(*slots[i], iterations);
                ++host_checks;
            }
            iterations += batch;
            if (pause) std::this_thread::sleep_for(std::chrono::milliseconds(pause));
        } while (iterations < int64_t(corpus_count) || mono_ns()-start < int64_t(seconds)*1000000000);
        int64_t end = mono_ns();
        for (auto& s : slots) check_inputs(*s);
        verifier_selfcheck();
        std::cout << ",\"iterations\":" << iterations << ",\"host_checks\":" << host_checks
                  << ",\"corpus_iterations\":[";
        for (size_t i=0;i<corpus_count;++i) std::cout << (i ? "," : "") << slots[i]->launches;
        std::cout << "],\"corpus_host_checks\":[";
        for (size_t i=0;i<corpus_count;++i) std::cout << (i ? "," : "") << slots[i]->checks;
        std::cout << "],\"device_elements_checked\":" << (batch>1 ? uint64_t(iterations)*elements : 0)
                  << ",\"host_elements_checked\":" << uint64_t(host_checks)*elements
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
