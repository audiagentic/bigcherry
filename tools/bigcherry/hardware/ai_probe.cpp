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
#include <type_traits>
#include <cmath>
#include <openssl/sha.h>

namespace {
constexpr int protocol_version = 7;
constexpr unsigned precision_epochs = 8;
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
    hipblasLtMatrixLayout_t a = nullptr, b = nullptr, out = nullptr;
    hipblasLtMatmulPreference_t pref = nullptr;
    ~Handles() {
        if (pref) hipblasLtMatmulPreferenceDestroy(pref);
        if (out) hipblasLtMatrixLayoutDestroy(out);
        if (b) hipblasLtMatrixLayoutDestroy(b);
        if (a) hipblasLtMatrixLayoutDestroy(a);
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
struct Geometry {
    int m, n, k, lda, ldb, ldc;
    size_t a_elements() const { return size_t(lda)*k; }
    size_t b_elements() const { return size_t(ldb)*n; }
    size_t c_elements() const { return size_t(ldc)*n; }
    size_t logical() const { return size_t(m)*n; }
};
uint8_t output_poison(const std::string& dtype) {
    return dtype=="int8" ? 0x80 : 0xff;
}
struct Corpus {
    std::vector<uint8_t> a, b, expected;
    Corpus(Geometry g, const std::string& dtype, Format f, size_t pattern)
        : a(g.a_elements()*f.in_bytes), b(g.b_elements()*f.in_bytes),
          expected(g.c_elements()*f.out_bytes, output_poison(dtype)) {
        std::vector<int8_t> x(size_t(g.m)*g.k), y(size_t(g.k)*g.n);
        if (pattern >= corpus_count) throw Failure(3, "invalid corpus pattern");
        uint64_t seed = 7 + pattern * 0x9e3779b97f4a7c15ULL;
        for (size_t i=0; i<std::max(x.size(),y.size()); ++i) {
            int a_sign=sign(seed), b_sign=sign(seed);
            if (i<x.size()) {
                x[i]=a_sign;
                if (pattern==1 && i%5<2) x[i]=0;
                if (pattern==2)
                    x[i]=i%g.m==i/g.m ? ((i/g.m)%2 ? -1 : 1) : 0;
                encode(a, i%g.m+(i/g.m)*g.lda, x[i], dtype);
            }
            if (i<y.size()) {
                y[i]=b_sign;
                if (pattern==1 && i%7<3) y[i]=0;
                encode(b, i%g.k+(i/g.k)*g.ldb, y[i], dtype);
            }
        }
        // Padding is distinct from all logical input encodings.
        for (int j=0;j<g.k;++j)
            std::fill(a.begin()+(size_t(j)*g.lda+g.m)*f.in_bytes,
                      a.begin()+size_t(j+1)*g.lda*f.in_bytes,0x5a);
        for (int j=0;j<g.n;++j)
            std::fill(b.begin()+(size_t(j)*g.ldb+g.k)*f.in_bytes,
                      b.begin()+size_t(j+1)*g.ldb*f.in_bytes,0x5a);
        // Column-major C = A B. Each sum is bounded by k <= 2048.
        std::vector<int> sums(g.logical(),0);
        for (int j=0;j<g.n;++j)
            for (int k=0;k<g.k;++k) {
                const int v=y[k+j*g.k];
                if (!v) continue;
                if (pattern==2) {
                    if (k<g.m) sums[k+j*g.m]=x[k+k*g.m]*v;
                } else for (int i=0;i<g.m;++i) sums[i+j*g.m]+=x[i+k*g.m]*v;
            }
        for (int j=0;j<g.n;++j) for (int i=0;i<g.m;++i) {
            size_t pos=i+size_t(j)*g.ldc;
            int sum=sums[i+j*g.m];
            if (dtype=="fp16") store(expected,pos,exact_half(sum));
            else if (dtype=="int8") store(expected,pos,int32_t(sum));
            else store(expected,pos,float(sum)/256);
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
// Separate, nonblocking memory queue; never claim measured overlap from
// enqueue order alone. Each epoch changes and verifies one resident chunk.
constexpr size_t resident_chunk_words = size_t(1)<<20;
__global__ void resident_xor(uint32_t* data, size_t words, uint32_t mask) {
    for (size_t i=blockIdx.x*size_t(blockDim.x)+threadIdx.x;
         i<words; i+=size_t(blockDim.x)*gridDim.x) data[i]^=mask;
}
struct Stream {
    hipStream_t value = nullptr;
    Stream() { HIP(hipStreamCreateWithFlags(&value,hipStreamNonBlocking)); }
    ~Stream() { if (value) (void)hipStreamDestroy(value); }
    Stream(const Stream&) = delete;
};
struct ResidentTraffic {
    size_t words, chunks;
    Buffer data;
    Stream stream;
    std::vector<uint32_t> host, masks;
    uint64_t epochs = 0, checked = 0;
    bool pending = false;
    explicit ResidentTraffic(int mib)
        : words((size_t(mib)<<20)/sizeof(uint32_t)),
          chunks((words+resident_chunk_words-1)/resident_chunk_words),
          data(words*sizeof(uint32_t)), host(std::min(words,resident_chunk_words)), masks(chunks,0) {
        for (size_t c=0;c<chunks;++c) {
            size_t off=c*resident_chunk_words, len=length(c);
            for (size_t i=0;i<len;++i) host[i]=memory_word(off+i,0xa17e58c3u);
            HIP(hipMemcpy(ptr(c),host.data(),len*sizeof(uint32_t),hipMemcpyHostToDevice));
        }
        check_all();
    }
    size_t length(size_t c) const { return std::min(resident_chunk_words,words-c*resident_chunk_words); }
    uint32_t* ptr(size_t c) { return static_cast<uint32_t*>(data.p)+c*resident_chunk_words; }
    uint32_t epoch_mask() const { return (uint32_t(epochs)*0x9e3779b9u+0x6a09e667u)|1u; }
    void check(size_t c) {
        size_t off=c*resident_chunk_words, len=length(c);
        HIP(hipMemcpy(host.data(),ptr(c),len*sizeof(uint32_t),hipMemcpyDeviceToHost));
        for (size_t i=0;i<len;++i)
            if (host[i]!=(memory_word(off+i,0xa17e58c3u)^masks[c]))
                throw Failure(2,"resident memory mismatch at word "+std::to_string(off+i));
        checked+=len;
    }
    void check_all() {
        if (pending) throw Failure(4,"resident queue not completed");
        for (size_t c=0;c<chunks;++c) check(c);
    }
    void enqueue() {
        if (pending) throw Failure(4,"resident epoch already pending");
        size_t c=epochs%chunks, len=length(c);
        resident_xor<<<std::min<size_t>((len+255)/256,4096),256,0,stream.value>>>(ptr(c),len,epoch_mask());
        HIP(hipGetLastError());
        pending=true;
    }
    void finish() {
        if (!pending) throw Failure(4,"resident epoch missing");
        HIP(hipStreamSynchronize(stream.value));
        size_t c=epochs%chunks;
        masks[c]^=epoch_mask();
        check(c);
        ++epochs;
        pending=false;
    }
};
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
enum class Instruction { alu, wave, lds, fp, dot4, wmma_fp16, wmma_bf16, wmma_fp16_pos, wmma_bf16_pos, dot2, dot2_pos, wmma_iu8, dot2_exact, wmma_fp16_bias, wmma_bf16_bias };
bool is_wmma(Instruction k) { return k==Instruction::wmma_fp16 || k==Instruction::wmma_bf16 ||
           k==Instruction::wmma_fp16_pos || k==Instruction::wmma_bf16_pos || k==Instruction::wmma_iu8 ||
           k==Instruction::wmma_fp16_bias || k==Instruction::wmma_bf16_bias; }
bool is_instruction(const std::string& name) {
    return name=="alu" || name=="wave" || name=="lds" || name=="vector-fp" ||
           name=="dot4" || name=="dot2" || name=="dot2-pos" || name=="wmma-iu8" || name=="wmma-fp16" || name=="wmma-bf16" ||
           name=="wmma-fp16-pos" || name=="wmma-bf16-pos" || name=="dot2-exact" ||
           name=="wmma-fp16-bias" || name=="wmma-bf16-bias" ||
           name=="dot2-rdna3-reference" || name=="wmma-fp16-rdna3-reference" || name=="wmma-bf16-rdna3-reference";
}
Instruction instruction_kind(const std::string& name) {
    if (name=="alu") return Instruction::alu;
    if (name=="wave") return Instruction::wave;
    if (name=="lds") return Instruction::lds;
    if (name=="vector-fp") return Instruction::fp;
    if (name=="dot4") return Instruction::dot4;
    if (name=="dot2" || name=="dot2-rdna3-reference") return Instruction::dot2;
    if (name=="dot2-exact") return Instruction::dot2_exact;
    if (name=="dot2-pos") return Instruction::dot2_pos;
    if (name=="wmma-iu8") return Instruction::wmma_iu8;
    if (name=="wmma-fp16" || name=="wmma-fp16-rdna3-reference") return Instruction::wmma_fp16;
    if (name=="wmma-fp16-bias") return Instruction::wmma_fp16_bias;
    if (name=="wmma-bf16" || name=="wmma-bf16-rdna3-reference") return Instruction::wmma_bf16;
    if (name=="wmma-bf16-bias") return Instruction::wmma_bf16_bias;
    if (name=="wmma-fp16-pos") return Instruction::wmma_fp16_pos;
    if (name=="wmma-bf16-pos") return Instruction::wmma_bf16_pos;
    throw Failure(3,"unknown instruction case");
}
const char* instruction_algorithm(Instruction kind) {
    switch (kind) {
    case Instruction::alu: return "integer-wrap-v1";
    case Instruction::wave: return "wave-exchange-v1";
    case Instruction::lds: return "lds-exchange-v1";
    case Instruction::fp: return "vector-dyadic-v1";
    case Instruction::dot4: return "signed-dot4-v1";
    case Instruction::dot2: return "half-dot2-v1";
    case Instruction::dot2_pos: return "half-dot2-nonnegative-v1";
    case Instruction::dot2_exact: return "half-dot2-vector-fma-v1";
    case Instruction::wmma_fp16_bias: return "wmma-fp16-biased-corpus-v1";
    case Instruction::wmma_bf16_bias: return "wmma-bf16-biased-corpus-v1";
    case Instruction::wmma_iu8: return "wmma-iu8-mixed-tile-v1";
    case Instruction::wmma_fp16: return "wmma-fp16-tile-v1";
    case Instruction::wmma_bf16: return "wmma-bf16-tile-v1";
    case Instruction::wmma_fp16_pos: return "wmma-fp16-nonnegative-v1";
    case Instruction::wmma_bf16_pos: return "wmma-bf16-nonnegative-v1";
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
// Exact dyadic values, no denormals/NaNs or tolerance-based comparison.
__device__ uint32_t canonical_float(float x) { return x==0.0f ? 0u : __float_as_uint(x); }
__global__ void instruction_fp(const uint32_t* input, Words* out, uint32_t seed) {
    const unsigned i=blockIdx.x*blockDim.x+threadIdx.x;
    const uint32_t a=input[i]^seed, b=input[i^1u]+seed;
    const float x=float(int(a&31u)-15)*0.0625f, y=float(int(b&31u)-15)*0.0625f;
    const float z=float(int((a>>8)&31u)-15)*0.0625f;
    out[i]={canonical_float(__builtin_fmaf(x,y,z)), canonical_float(x*y),
            canonical_float(x+y), uint32_t(int(x*16.0f))};
}
__device__ int native_sdot4(uint32_t a,uint32_t b,int c) {
#if defined(__gfx1030__)
    return __builtin_amdgcn_sdot4(int(a),int(b),c,false);
#elif defined(__gfx1100__) || defined(__gfx1201__)
    return __builtin_amdgcn_sudot4(true,int(a),true,int(b),c,false);
#else
    (void)a; (void)b; (void)c;
    return 0; // runtime refuses targets without an implemented native case
#endif
}
__global__ void instruction_dot4(const uint32_t* input, Words* out, uint32_t seed) {
    const unsigned i=blockIdx.x*blockDim.x+threadIdx.x;
    const uint32_t a=input[i]^seed, b=input[i^1u]+seed;
    const int c=int((a>>24)&255u)-128;
    const int d=native_sdot4(a,b,c);
    const int e=native_sdot4(a^0x80808080u,b,-c);
    const int f=native_sdot4(a,b^0x80808080u,0);
    const int g=native_sdot4(a,b,d);
    out[i]={uint32_t(d),uint32_t(e),uint32_t(f),uint32_t(g)};
}
// Explicit packed FP16 dot instruction. On gfx1030 use the native asm
// form because the historical builtin path produced incorrect sums.
__device__ float native_dot2(uint32_t a,uint32_t b,float acc) {
#if defined(__gfx1030__)
    asm("v_dot2_f32_f16 %0, %1, %2, %0" : "+v"(acc) : "v"(a), "v"(b));
    return acc;
#elif defined(__gfx1100__) || defined(__gfx1201__)
    using H2 = __attribute__((ext_vector_type(2))) _Float16;
    return __builtin_amdgcn_fdot2(__builtin_bit_cast(H2,a),__builtin_bit_cast(H2,b),acc,false);
#else
    (void)a; (void)b; return acc;
#endif
}
__device__ uint32_t pack_halves(int x,int y) {
    const _Float16 a=float(x)/16.0f,b=float(y)/16.0f;
    return uint32_t(__builtin_bit_cast(uint16_t,a)) | (uint32_t(__builtin_bit_cast(uint16_t,b))<<16);
}
template<bool Positive> __global__ void instruction_dot2(const uint32_t* input,Words* out,uint32_t seed) {
    const unsigned i=blockIdx.x*blockDim.x+threadIdx.x;
    const uint32_t a=input[i]^seed,b=input[i^1u]+seed;
    const int x=int(a&31u)-(Positive?0:15),y=int((a>>8)&31u)-(Positive?0:15);
    const int z=int(b&31u)-(Positive?0:15),w=int((b>>8)&31u)-(Positive?0:15);
    const uint32_t ab=pack_halves(x,y),cd=pack_halves(z,w);
    const float d=native_dot2(ab,cd,0.0f);
    out[i]={canonical_float(d),canonical_float(native_dot2(ab,cd,d)),
            canonical_float(native_dot2(pack_halves(y,x),cd,0.0f)),
            canonical_float(native_dot2(ab,pack_halves(Positive?z:-z,Positive?w:-w),1.0f))};
}
// Workaround experiment: FP32 multiply-add on the same signed dyadic inputs.
// This does not claim native packed-dot coverage.
__global__ void instruction_dot2_exact(const uint32_t* input,Words* out,uint32_t seed) {
    const unsigned i=blockIdx.x*blockDim.x+threadIdx.x;
    const uint32_t a=input[i]^seed,b=input[i^1u]+seed;
    const float x=(int(a&31u)-15)/16.0f,y=(int((a>>8)&31u)-15)/16.0f;
    const float z=(int(b&31u)-15)/16.0f,w=(int((b>>8)&31u)-15)/16.0f;
    const float d=__builtin_fmaf(x,z,y*w);
    out[i]={canonical_float(d),canonical_float(__builtin_fmaf(x,z,__builtin_fmaf(y,w,d))),
            canonical_float(__builtin_fmaf(y,z,x*w)),canonical_float(__builtin_fmaf(-x,z,__builtin_fmaf(-y,w,1.0f)))};
}
// One independent 16x16 tile per wave. Input words encode 16 signed
// two-bit coefficients per row/column. Native fragment layouts differ.
template<bool BF, bool Positive, bool Bias=false> __global__ void instruction_wmma(const uint32_t* input, Words* out, uint32_t seed) {
#if defined(__gfx1100__) || defined(__gfx1201__)
    using F = __attribute__((ext_vector_type(8))) float;
    const unsigned i=blockIdx.x*blockDim.x+threadIdx.x, lane=threadIdx.x%32;
    const unsigned base=i-lane, col=lane%16;
    const uint32_t a=input[base+col]^seed, b=input[base+16+col]+seed;
#if defined(__gfx1100__)
    using H = __attribute__((ext_vector_type(16))) _Float16;
    using B = __attribute__((ext_vector_type(16))) __bf16;
    constexpr unsigned count=16;
#else
    using H = __attribute__((ext_vector_type(8))) _Float16;
    using B = __attribute__((ext_vector_type(8))) __bf16;
    constexpr unsigned count=8;
#endif
    using V = std::conditional_t<BF,B,H>;
    V av,bv;
    for(unsigned j=0;j<count;++j) {
#if defined(__gfx1100__)
        const unsigned k=j;
#else
        const unsigned k=(j/4)*8+(lane/16)*4+j%4;
#endif
        av[j]=int((a>>(2*k))&3u)-((Positive || Bias)?0:1);
        bv[j]=int((b>>(2*k))&3u)-((Positive || Bias)?0:1);
    }
    F acc;
    for(unsigned j=0;j<8;++j) {
#if defined(__gfx1100__)
        const unsigned row=2*j+lane/16;
#else
        const unsigned row=8*(lane/16)+j;
#endif
        acc[j]=Bias?0.0f:float(Positive?int(row+col):int(row)-int(col));
    }
#if defined(__gfx1100__)
    if constexpr(BF) acc=__builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(av,bv,acc);
    else acc=__builtin_amdgcn_wmma_f32_16x16x16_f16_w32(av,bv,acc);
#else
    if constexpr(BF) acc=__builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(av,bv,acc);
    else acc=__builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(av,bv,acc);
#endif
    // (A+1)(B+1) - sum(A+1) - sum(B+1) + 16 + C.
    // Exact on this bounded integer-valued FP16/BF16 corpus, not a general FP rewrite.
    if constexpr(Bias) {
        for(unsigned j=0;j<8;++j) {
#if defined(__gfx1100__)
            const unsigned row=2*j+lane/16;
#else
            const unsigned row=8*(lane/16)+j;
#endif
            const uint32_t ar=input[base+row]^seed;
            int sa=0,sb=0;
            for(unsigned k=0;k<16;++k) { sa+=(ar>>(2*k))&3u; sb+=(b>>(2*k))&3u; }
            acc[j] += float(16-sa-sb+int(row)-int(col));
        }
    }
    out[2*i]={canonical_float(acc[0]),canonical_float(acc[1]),canonical_float(acc[2]),canonical_float(acc[3])};
    out[2*i+1]={canonical_float(acc[4]),canonical_float(acc[5]),canonical_float(acc[6]),canonical_float(acc[7])};
#else
    (void)input; (void)out; (void)seed;
#endif
}
__global__ void instruction_wmma_iu8(const uint32_t* input,Words* out,uint32_t seed) {
#if defined(__gfx1100__) || defined(__gfx1201__)
    using I8 = __attribute__((ext_vector_type(8))) int;
#if defined(__gfx1100__)
    using I = __attribute__((ext_vector_type(4))) int;
    constexpr unsigned count=4;
#else
    using I = __attribute__((ext_vector_type(2))) int;
    constexpr unsigned count=2;
#endif
    const unsigned i=blockIdx.x*blockDim.x+threadIdx.x,lane=threadIdx.x%32;
    const unsigned base=i-lane,col=lane%16;
    const uint32_t a=input[base+col]^seed,b=input[base+16+col]+seed;
    I av,bv;
    for(unsigned j=0;j<count;++j) {
        uint32_t ap=0,bp=0;
        for(unsigned q=0;q<4;++q) {
#if defined(__gfx1100__)
            const unsigned k=j*4+q;
#else
            const unsigned k=j*8+(lane/16)*4+q;
#endif
            const unsigned x=((a>>(2*k))&3u)*85;
            const int y=int((b>>(2*k))&3u)*85-128;
            ap|=x<<(8*q);
            bp|=(uint32_t(y)&255u)<<(8*q);
        }
        av[j]=__builtin_bit_cast(int,ap);
        bv[j]=__builtin_bit_cast(int,bp);
    }
    I8 acc;
    for(unsigned j=0;j<8;++j) {
#if defined(__gfx1100__)
        const unsigned row=2*j+lane/16;
#else
        const unsigned row=8*(lane/16)+j;
#endif
        acc[j]=int(row)-int(col);
    }
#if defined(__gfx1100__)
    acc=__builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(false,av,true,bv,acc,false);
#else
    acc=__builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(false,av,true,bv,acc,false);
#endif
    out[2*i]={uint32_t(acc[0]),uint32_t(acc[1]),uint32_t(acc[2]),uint32_t(acc[3])};
    out[2*i+1]={uint32_t(acc[4]),uint32_t(acc[5]),uint32_t(acc[6]),uint32_t(acc[7])};
#else
    (void)input; (void)out; (void)seed;
#endif
}
uint32_t host_float_bits(float x) {
    uint32_t bits=0;
    if(x!=0.0f) std::memcpy(&bits,&x,sizeof(bits));
    return bits;
}
int signed_dot4(uint32_t a,uint32_t b) {
    int result=0;
    for(unsigned k=0;k<4;++k) {
        const int x=(a>>(8*k))&255u, y=(b>>(8*k))&255u;
        result+=(x<128?x:x-256)*(y<128?y:y-256);
    }
    return result;
}
bool same_words(const Words& a, const Words& b) {
    return a.a==b.a && a.b==b.b && a.c==b.c && a.d==b.d;
}
void instruction_reference(Instruction kind, const std::vector<uint32_t>& input,
                           std::vector<Words>& expected, unsigned width, uint32_t seed,
                           bool gfx12=false) {
    if (is_wmma(kind)) {
        for(size_t base=0;base<input.size();base+=32) {
            int tile[16][16];
            const bool positive=kind==Instruction::wmma_fp16_pos || kind==Instruction::wmma_bf16_pos;
            const bool integer=kind==Instruction::wmma_iu8;
            for(unsigned row=0;row<16;++row) for(unsigned col=0;col<16;++col) {
                const uint32_t a=input[base+row]^seed,b=input[base+16+col]+seed;
                int sum=positive?int(row+col):int(row)-int(col);
                for(unsigned k=0;k<16;++k)
                    if (integer) sum+=int((a>>(2*k))&3u)*85*(int((b>>(2*k))&3u)*85-128);
                    else sum+=(int((a>>(2*k))&3u)-(positive?0:1))*(int((b>>(2*k))&3u)-(positive?0:1));
                tile[row][col]=sum;
            }
            for(unsigned lane=0;lane<32;++lane) for(unsigned j=0;j<8;j+=4) {
                uint32_t v[4];
                for(unsigned q=0;q<4;++q) {
                    unsigned row=gfx12?8*(lane/16)+j+q:2*(j+q)+lane/16;
                    v[q]=integer?uint32_t(tile[row][lane%16]):host_float_bits(float(tile[row][lane%16]));
                }
                expected[2*(base+lane)+j/4]={v[0],v[1],v[2],v[3]};
            }
        }
        return;
    }
    if (kind==Instruction::fp || kind==Instruction::dot4 || kind==Instruction::dot2 || kind==Instruction::dot2_pos || kind==Instruction::dot2_exact) {
        for(size_t i=0;i<input.size();++i) {
            const uint32_t a=input[i]^seed,b=input[i^size_t(1)]+seed;
            if(kind==Instruction::fp) {
                const int x=int(a%32)-15,y=int(b%32)-15,z=int((a/256)%32)-15;
                expected[i]={host_float_bits(float(x*y+16*z)/256),
                             host_float_bits(float(x*y)/256),host_float_bits(float(x+y)/16),uint32_t(x)};
            } else if (kind==Instruction::dot2 || kind==Instruction::dot2_pos || kind==Instruction::dot2_exact) {
                const bool positive=kind==Instruction::dot2_pos;
                const int offset=positive?0:15;
                const int x=int(a%32)-offset,y=int((a/256)%32)-offset;
                const int z=int(b%32)-offset,w=int((b/256)%32)-offset,d=x*z+y*w;
                expected[i]={host_float_bits(float(d)/256),host_float_bits(float(2*d)/256),
                             host_float_bits(float(y*z+x*w)/256),host_float_bits(float(256+(positive?d:-d))/256)};
            } else {
                const int c=int(a/16777216u)-128;
                const int d=signed_dot4(a,b)+c;
                expected[i]={uint32_t(d),uint32_t(signed_dot4(a^0x80808080u,b)-c),
                             uint32_t(signed_dot4(a,b^0x80808080u)),uint32_t(signed_dot4(a,b)+d)};
            }
        }
        return;
    }
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
bool precision_reference_case(const std::string& dtype) {
    return dtype=="dot2-rdna3-reference" || dtype=="wmma-fp16-rdna3-reference" ||
           dtype=="wmma-bf16-rdna3-reference";
}
float as_float(uint32_t bits) { float x; std::memcpy(&x,&bits,sizeof(x)); return x; }
std::string precision_sha(const std::vector<Words>& words) {
    unsigned char hash[SHA256_DIGEST_LENGTH];
    if (!SHA256(reinterpret_cast<const unsigned char*>(words.data()),words.size()*sizeof(Words),hash))
        throw Failure(4,"reference fingerprint failed");
    std::string out;
    constexpr char digits[]="0123456789abcdef";
    for(unsigned char c:hash) { out+=digits[c>>4]; out+=digits[c&15]; }
    return out;
}
void instruction_screen(Instruction kind, const std::string& dtype, const std::string& bdf,
                        const std::string& arch, int hip_version, int driver_version,
                        unsigned width, int blocks, int headroom_mib, int seconds, int pause) {
    if ((width!=32 && width!=64) || instruction_block%width)
        throw Failure(3, "unsupported HIP wave width");
    if ((kind==Instruction::dot4 || kind==Instruction::dot2 || kind==Instruction::dot2_pos) && arch!="gfx1030" && arch!="gfx1100" && arch!="gfx1201")
        throw Failure(3,"native dot4 unavailable for this target");
    if (is_wmma(kind) && (width!=32 || (arch!="gfx1100" && arch!="gfx1201")))
        throw Failure(3,"native WMMA unavailable for this target/wave width");
    const bool precision=precision_reference_case(dtype);
    if (precision && (arch!="gfx1100" || blocks>64))
        throw Failure(3,"RDNA3 reference scope requires gfx1100 and at most 64 groups");
    const size_t elements = size_t(blocks)*instruction_block;
    const size_t result_elements=elements*(is_wmma(kind)?2:1);
    const size_t input_bytes = elements*sizeof(uint32_t), output_bytes = result_elements*sizeof(Words);
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
    std::vector<Words> actual(result_elements), expected(result_elements);
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
    std::array<std::vector<Words>,precision_epochs> references;
    uint64_t different=0;
    double max_abs=0;
    do {
        const unsigned epoch=unsigned(precision?iterations%precision_epochs:iterations);
        const uint32_t seed=uint32_t(epoch)*0x9e3779b9u+0x712db419u;
        instruction_reference(kind,input,expected,width,seed,arch=="gfx1201");
        HIP(hipMemsetAsync(out,0xff,output_bytes));
        auto* in = static_cast<const uint32_t*>(device_input.p);
        switch (kind) {
        case Instruction::alu: instruction_alu<<<blocks,instruction_block>>>(in,out,seed); break;
        case Instruction::wave: instruction_wave<<<blocks,instruction_block>>>(in,out,seed); break;
        case Instruction::lds: instruction_lds<<<blocks,instruction_block>>>(in,out,seed); break;
        case Instruction::fp: instruction_fp<<<blocks,instruction_block>>>(in,out,seed); break;
        case Instruction::dot4: instruction_dot4<<<blocks,instruction_block>>>(in,out,seed); break;
        case Instruction::dot2: instruction_dot2<false><<<blocks,instruction_block>>>(in,out,seed); break;
        case Instruction::dot2_pos: instruction_dot2<true><<<blocks,instruction_block>>>(in,out,seed); break;
        case Instruction::dot2_exact: instruction_dot2_exact<<<blocks,instruction_block>>>(in,out,seed); break;
        case Instruction::wmma_fp16_bias: instruction_wmma<false,false,true><<<blocks,instruction_block>>>(in,out,seed); break;
        case Instruction::wmma_bf16_bias: instruction_wmma<true,false,true><<<blocks,instruction_block>>>(in,out,seed); break;
        case Instruction::wmma_iu8: instruction_wmma_iu8<<<blocks,instruction_block>>>(in,out,seed); break;
        case Instruction::wmma_fp16: instruction_wmma<false,false><<<blocks,instruction_block>>>(in,out,seed); break;
        case Instruction::wmma_bf16: instruction_wmma<true,false><<<blocks,instruction_block>>>(in,out,seed); break;
        case Instruction::wmma_fp16_pos: instruction_wmma<false,true><<<blocks,instruction_block>>>(in,out,seed); break;
        case Instruction::wmma_bf16_pos: instruction_wmma<true,true><<<blocks,instruction_block>>>(in,out,seed); break;
        }
        HIP(hipGetLastError());
        HIP(hipDeviceSynchronize());
        HIP(hipMemcpy(actual.data(),out,output_bytes,hipMemcpyDeviceToHost));
        if (precision) {
            // Discovery only: verify the independently derived dyadic output
            // lattice, then fingerprint and replay exact bytes. No numerical
            // pass is issued by discovery, even when all repeats agree.
            const double grid=kind==Instruction::dot2 ? 1.0/256 : 1.0;
            for(size_t i=0;i<result_elements;++i) {
                const Words& a=actual[i]; const Words& e=expected[i];
                const uint32_t got[]={a.a,a.b,a.c,a.d}, want[]={e.a,e.b,e.c,e.d};
                for(unsigned q=0;q<4;++q) {
                    const double v=as_float(got[q]), w=as_float(want[q]);
                    const double err=std::abs(v-w);
                    if (!std::isfinite(v) || !(err<grid/2))
                        throw Failure(2,"RDNA3 reference outside exact output lattice");
                    if (got[q]!=want[q]) { ++different; max_abs=std::max(max_abs,err); }
                }
            }
            if (iterations<precision_epochs) references[epoch]=actual;
            else if (std::memcmp(references[epoch].data(),actual.data(),output_bytes))
                throw Failure(2,"RDNA3 reference changed on repeated epoch");
        } else for (size_t i=0;i<result_elements;++i)
            if (!same_words(actual[i],expected[i])) {
                auto describe = [](const Words& w) {
                    return std::to_string(w.a)+","+std::to_string(w.b)+","+
                           std::to_string(w.c)+","+std::to_string(w.d);
                };
                throw Failure(2,"instruction oracle mismatch at word group "+std::to_string(i)+
                              " after "+std::to_string(iterations)+" clean launches: got "+
                              describe(actual[i])+" expected "+describe(expected[i]));
            }
        std::array<uint8_t,guard_bytes> guard{};
        for (size_t off : {size_t(0),guard_bytes+output_bytes}) {
            HIP(hipMemcpy(guard.data(),static_cast<uint8_t*>(device_output.p)+off,
                          guard.size(),hipMemcpyDeviceToHost));
            if (!std::all_of(guard.begin(),guard.end(),[](uint8_t v){return v==0xa5;}))
                throw Failure(2,"instruction output guard changed");
        }
        ++iterations;
        if (pause) std::this_thread::sleep_for(std::chrono::milliseconds(pause));
    } while (iterations<(precision?2*precision_epochs:2) || mono_ns()-start<int64_t(seconds)*1000000000);
    const int64_t end=mono_ns();
    check_input();
    std::cout << "{\"protocol_version\":" << protocol_version << ",\"bdf\":\"" << bdf << "\",\"arch\":\"" << arch
              << "\",\"dtype\":\"" << dtype << "\",\"n\":" << blocks
              << ",\"hip\":" << hip_version << ",\"driver\":" << driver_version
              << ",\"provider\":\"hip-native\",\"library_version\":\"" << hip_version
              << "\",\"workspace\":0,\"algorithm\":\"" << (precision ? dtype : instruction_algorithm(kind))
              << "\",\"batch\":1,\"block_size\":" << instruction_block << ",\"wave_size\":" << width
              << ",\"iterations\":" << iterations << ",\"host_checks\":" << iterations
              << ",\"checked_lanes\":" << uint64_t(iterations)*elements
              << ",\"host_elements_checked\":" << uint64_t(iterations)*result_elements*4
              << ",\"guard_checks\":" << uint64_t(iterations)*2 << ",\"input_checks\":2"
              << ",\"allocated_bytes\":" << budget << ",\"free_before_bytes\":" << free
              << ",\"free_after_bytes\":" << after
              << ",\"elapsed_s\":" << std::setprecision(9) << double(end-start)/1e9
              << ",\"start_ns\":" << start << ",\"end_ns\":" << end;
    if(precision) {
        std::cout << ",\"precision_epochs\":" << precision_epochs
                  << ",\"precision_different_words\":" << different
                  << ",\"precision_max_abs_error\":" << std::setprecision(17) << max_abs
                  << ",\"precision_fingerprints\":[";
        for(unsigned i=0;i<precision_epochs;++i)
            std::cout << (i?",":"") << "\"" << precision_sha(references[i]) << "\"";
        std::cout << "],\"status\":\"precision_observed\"}\n";
    } else std::cout << ",\"status\":\"pass\"}\n";
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
            for (Geometry g : {Geometry{17,17,17,17,17,17},
                               Geometry{15,19,7,18,11,21},
                               Geometry{5,3,17,8,20,9}}) {
                std::vector<uint8_t> previous;
                for (size_t pattern=0;pattern<corpus_count;++pattern) {
                    Corpus c(g,"int8",f,pattern);
                    if (c.expected==previous) return 2;
                    previous=c.expected;
                    for (int j=0;j<g.n;++j) for (int i=0;i<g.m;++i) {
                        int32_t want=0,got=0;
                        for (int k=0;k<g.k;++k) want+=int8_t(c.a[i+k*g.lda])*int8_t(c.b[k+j*g.ldb]);
                        std::memcpy(&got,c.expected.data()+(i+j*g.ldc)*sizeof(got),sizeof(got));
                        if (want!=got) return 2;
                    }
                    for (int j=0;j<g.n;++j)
                        for (int i=g.m;i<g.ldc;++i)
                            for (size_t b=0;b<f.out_bytes;++b)
                                if (c.expected[(i+j*g.ldc)*f.out_bytes+b]!=output_poison("int8")) return 2;
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
            if (signed_dot4(0x807f01ffu,0x01010101u)!=-1) return 2;
            instruction_reference(Instruction::fp,lanes,words,32,0);
            if (!same_words(words[0],{host_float_bits(-30.0f/256),host_float_bits(210.0f/256),
                                      host_float_bits(-29.0f/16),uint32_t(-15)})) return 2;
            std::vector<uint32_t> uniform(256,0x55555555u); // A/B coefficients zero
            words.resize(512);
            for (bool gfx12 : {false,true}) {
                instruction_reference(Instruction::wmma_fp16,uniform,words,32,0,gfx12);
                if (words[0].a!=0 || words[0].b!=host_float_bits(gfx12?1.0f:2.0f)) return 2;
            }
            std::cout << "CPU oracle self-test passed\n";
            return 0;
        }
        if (argc != 16) throw Failure(3, "usage: ai-probe BDF arch dtype N seconds pause-ms provider headroom-MiB batch M K LDA LDB LDC resident-MiB");
        std::string bdf = argv[1], expected_arch = argv[2], dtype = argv[3];
        std::string provider = argv[7];
        const bool matrix = dtype!="memory" && !is_instruction(dtype);
        int n = parse_int(argv[4], dtype=="memory" ? 2 : matrix ? 1 : 16,
                          dtype=="memory" ? 65536 : 2048);
        Geometry g{parse_int(argv[10],0,2048),n,parse_int(argv[11],0,2048),
                   parse_int(argv[12],0,2304),parse_int(argv[13],0,2304),
                   parse_int(argv[14],0,2304)};
        int resident_mib=parse_int(argv[15],0,65536);
        if (resident_mib && !matrix) throw Failure(3,"resident traffic requires a matrix case");
        if (matrix) {
            if (!g.m || !g.k || g.lda<g.m || g.lda>g.m+256 ||
                g.ldb<g.k || g.ldb>g.k+256 || g.ldc<g.m || g.ldc>g.m+256)
                throw Failure(3,"invalid matrix geometry");
        } else if (g.m || g.k || g.lda || g.ldb || g.ldc)
            throw Failure(3,"matrix geometry is not valid for this case");
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
        if (is_instruction(dtype)) {
            if (provider!="hip-native" || batch!=1)
                throw Failure(3,"instruction cases require hip-native and batch 1");
            const auto kind = instruction_kind(dtype);
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
            LT(hipblasLtMatrixLayoutCreate(&h.a,f.in,g.m,g.k,g.lda));
            LT(hipblasLtMatrixLayoutCreate(&h.b,f.in,g.k,g.n,g.ldb));
            LT(hipblasLtMatrixLayoutCreate(&h.out,f.out,g.m,g.n,g.ldc));
            LT(hipblasLtMatmulPreferenceCreate(&h.pref));
            const size_t limit = 32ull << 20;
            LT(hipblasLtMatmulPreferenceSetAttribute(h.pref, HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &limit, sizeof(limit)));
            int found = 0;
            auto st = hipblasLtMatmulAlgoGetHeuristic(h.h, h.desc, h.a, h.b, h.out, h.out, h.pref, 1, &algo, &found);
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
            Slot(Geometry g, const std::string& dtype, Format f, size_t pattern)
                : c(g, dtype, f, pattern), a(c.a.size()), b(c.b.size()),
                  out(c.expected.size()+2*guard_bytes), expected(c.expected.size()) {
                HIP(hipMemcpy(a.p, c.a.data(), c.a.size(), hipMemcpyHostToDevice));
                HIP(hipMemcpy(b.p, c.b.data(), c.b.size(), hipMemcpyHostToDevice));
                HIP(hipMemcpy(expected.p, c.expected.data(), c.expected.size(), hipMemcpyHostToDevice));
                HIP(hipMemset(out.p, 0xa5, c.expected.size()+2*guard_bytes));
            }
            void* data() { return static_cast<uint8_t*>(out.p)+guard_bytes; }
        };
        const size_t resident_bytes=size_t(resident_mib)<<20;
        const size_t allocation=corpus_count*(g.a_elements()*f.in_bytes+g.b_elements()*f.in_bytes+
                                2*g.c_elements()*f.out_bytes+2*guard_bytes)+
                                std::max(size_t(1),workspace)+(batch>1 ? 2*sizeof(unsigned) : 0)+resident_bytes;
        const size_t reserve=size_t(headroom)<<20;
        size_t free_before=0,total=0;
        HIP(hipMemGetInfo(&free_before,&total));
        if (allocation>free_before || free_before-allocation<reserve)
            throw Failure(3,"matrix/resident budget exceeds VRAM headroom");
        std::vector<std::unique_ptr<Slot>> slots;
        for (size_t i=0;i<corpus_count;++i)
            slots.push_back(std::make_unique<Slot>(g,dtype,f,i));
        Buffer ws(std::max(size_t(1), workspace));
        std::vector<uint8_t> actual(slots[0]->c.expected.size());
        float alpha = 1, beta = 0;
        int32_t ia = 1, ib = 0;
        const void* pa = dtype == "int8" ? static_cast<void*>(&ia) : static_cast<void*>(&alpha);
        const void* pb = dtype == "int8" ? static_cast<void*>(&ib) : static_cast<void*>(&beta);
        const size_t elements = g.c_elements(), logical=g.logical();
        const bool integer = dtype == "int8";
        std::unique_ptr<Buffer> counters;
        // Bounded ldc <= 2304, n <= 2048, batch <= 256: counter fits uint32.
        if (batch > 1) counters = std::make_unique<Buffer>(2 * sizeof(unsigned));
        std::unique_ptr<ResidentTraffic> resident;
        if (resident_mib) resident=std::make_unique<ResidentTraffic>(resident_mib);
        size_t free_after=0,total_after=0;
        HIP(hipMemGetInfo(&free_after,&total_after));
        if (free_after<reserve || total_after!=total)
            throw Failure(3,"matrix/resident allocation did not preserve headroom");
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
                  << "\",\"batch\":" << batch
                  << ",\"matrix_geometry\":[" << g.m << "," << g.n << "," << g.k << ","
                  << g.lda << "," << g.ldb << "," << g.ldc << "]"
                  << ",\"allocated_bytes\":" << allocation
                  << ",\"free_before_bytes\":" << free_before << ",\"free_after_bytes\":" << free_after
                  << ",\"resident_bytes\":" << resident_bytes;
        auto start = mono_ns();
        int64_t iterations = 0, host_checks = 0;
        do {
            std::array<bool, corpus_count> used{};
            if (resident) resident->enqueue();
            if (batch > 1) HIP(hipMemsetAsync(counters->p, 0, 2 * sizeof(unsigned)));
            for (int j=0; j<batch; ++j) {
                const size_t index = size_t(iterations+j) % corpus_count;
                auto& s = *slots[index];
                used[index] = true;
                HIP(hipMemsetAsync(s.data(), output_poison(dtype), actual.size()));
                if (provider == "hipblaslt") {
                    LT(hipblasLtMatmul(h.h, h.desc, pa, s.a.p, h.a, s.b.p, h.b, pb,
                                      s.data(), h.out, s.data(), h.out, &algo.algo, ws.p, workspace, nullptr));
                } else {
                    RB(rocblas_gemm_ex(r.h, rocblas_operation_none, rocblas_operation_none,
                                      g.m,g.n,g.k,pa,s.a.p,rin,g.lda,s.b.p,rin,g.ldb,pb,s.data(),rout,g.ldc,s.data(),rout,g.ldc,
                                      rcompute,rocblas_gemm_algo_standard,0,0));
                }
                if (batch > 1)
                    check_matrix_on_device(s.data(), s.expected.p, elements, f.out_bytes, integer, counters->p);
                ++s.launches;
            }
            if (batch > 1) verify_counts(unsigned(elements * batch), 0);
            else HIP(hipDeviceSynchronize());
            if (resident) resident->finish();
            for (size_t i=0;i<corpus_count;++i) if (used[i]) {
                compare_host(*slots[i], iterations);
                ++host_checks;
            }
            iterations += batch;
            if (pause) std::this_thread::sleep_for(std::chrono::milliseconds(pause));
        } while (iterations < int64_t(corpus_count) ||
                 (resident && resident->epochs<resident->chunks) ||
                 mono_ns()-start < int64_t(seconds)*1000000000);
        int64_t end = mono_ns();
        if (resident) resident->check_all();
        for (auto& s : slots) check_inputs(*s);
        verifier_selfcheck();
        std::cout << ",\"iterations\":" << iterations << ",\"host_checks\":" << host_checks
                  << ",\"corpus_iterations\":[";
        for (size_t i=0;i<corpus_count;++i) std::cout << (i ? "," : "") << slots[i]->launches;
        std::cout << "],\"corpus_host_checks\":[";
        for (size_t i=0;i<corpus_count;++i) std::cout << (i ? "," : "") << slots[i]->checks;
        std::cout << "]";
        if (resident)
            std::cout << ",\"resident_chunk_words\":" << resident_chunk_words
                      << ",\"resident_chunks\":" << resident->chunks
                      << ",\"resident_epochs\":" << resident->epochs
                      << ",\"resident_words_checked\":" << resident->checked
                      << ",\"resident_full_sweeps\":" << resident->epochs/resident->chunks;
        std::cout << ",\"device_elements_checked\":" << (batch>1 ? uint64_t(iterations)*logical : 0)
                  << ",\"host_elements_checked\":" << uint64_t(host_checks)*logical
                  << ",\"device_padding_checked\":" << (batch>1 ? uint64_t(iterations)*(elements-logical) : 0)
                  << ",\"host_padding_checked\":" << uint64_t(host_checks)*(elements-logical)
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
