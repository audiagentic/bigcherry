// HL02 Gate B lab harness: concurrent CPU BAR push to all GPUs versus concurrent GPU pull from host memory, at
// all-reduce message sizes, with optional GPU producers writing host memory and a CPU three-partial BF16 sum.
// Every received word is verified on the GPU after every timed transfer. Lab only; requires the exclusive gpu lease.
//
// Build: hipcc -O2 -mavx2 --offload-arch=gfx1100 --offload-arch=gfx1201 threeway_push.hip -o threeway_push -lpthread -lhsa-runtime64
// Usage: threeway_push --sizes 327680,1048576,4194304,10485760 --iters 30 --cpus 0,2,4 --push-one 0 [--reverse]
//   prints one JSON line per (arm, size, iteration) and a summary per (arm, size).
#include <hip/hip_runtime.h>
#include <hsa/hsa.h>
#include <hsa/hsa_ext_amd.h>
#include <immintrin.h>
#include <linux/dma-buf.h>
#include <pthread.h>
#include <sched.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <unistd.h>
#include <algorithm>
#include <atomic>
#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>
#include <time.h>
#include <vector>

#define CK(x) do { hipError_t e_ = (x); if (e_ != hipSuccess) { fprintf(stderr, "FAIL %s: %s\n", #x, hipGetErrorString(e_)); exit(2); } } while (0)
#define HK(x) do { hsa_status_t s_ = (x); if (s_ != HSA_STATUS_SUCCESS) { fprintf(stderr, "FAIL %s: hsa %d\n", #x, (int)s_); exit(2); } } while (0)

static constexpr int MAXD = 4;

__host__ __device__ inline uint32_t value_at(uint32_t idx, uint32_t gen) {
  uint32_t x = idx ^ (gen * 0x9e3779b9u);
  x ^= x >> 16; x *= 0x7feb352du; x ^= x >> 15; x *= 0x846ca68bu; x ^= x >> 16;
  return x;
}
__global__ void verify_words(const uint32_t* p, uint32_t n, uint32_t gen, unsigned* err) {
  unsigned bad = 0;
  for (uint32_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) bad += p[i] != value_at(i, gen);
  if (bad) atomicAdd(err, bad);
}
__global__ void clear_words(uint32_t* p, uint32_t n) {
  for (uint32_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) p[i] = 0xdeadbeefu;
}
// GPU pull: the receiving card copies the reduced message from mapped host memory into its own memory.
__global__ void pull_copy(const uint4* __restrict__ src, uint4* __restrict__ dst, size_t n16) {
  for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n16; i += (size_t)gridDim.x * blockDim.x) dst[i] = src[i];
}
// GPU producer: writes its partial to mapped host memory until told to stop (host-visible stop flag) or until its time
// limit, so a transfer kernel the hardware queues behind it cannot deadlock the run (such a transfer is reported).
__global__ void producer(uint4* dst, size_t n16, volatile int* stop, unsigned long long* bytes, unsigned long long limit) {
  unsigned long long rounds = 0; const unsigned long long t0 = wall_clock64();
  while (!*stop && wall_clock64() - t0 < limit) {
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n16; i += (size_t)gridDim.x * blockDim.x)
      dst[i] = make_uint4((uint32_t)i, (uint32_t)rounds, 1u, 2u);
    ++rounds;
  }
  if (threadIdx.x == 0) atomicAdd(bytes, rounds * (n16 * 16ull / gridDim.x));
}

struct Inventory { std::vector<hsa_agent_t> cpus, gpus; };
static hsa_status_t collect(hsa_agent_t a, void* c) {
  hsa_device_type_t k{}; hsa_agent_get_info(a, HSA_AGENT_INFO_DEVICE, &k);
  auto& v = *(Inventory*)c;
  if (k == HSA_DEVICE_TYPE_CPU) v.cpus.push_back(a);
  if (k == HSA_DEVICE_TYPE_GPU) v.gpus.push_back(a);
  return HSA_STATUS_SUCCESS;
}
struct PoolChoice { Inventory* inv; bool found = false; hsa_amd_memory_pool_t pool{}; size_t granule = 0; };
static hsa_status_t pick_pool(hsa_amd_memory_pool_t p, void* raw) {
  auto& c = *(PoolChoice*)raw;
  if (c.found) return HSA_STATUS_SUCCESS;
  hsa_amd_segment_t seg{}; hsa_amd_memory_pool_location_t loc{}; bool alloc = false;
  hsa_amd_memory_pool_get_info(p, HSA_AMD_MEMORY_POOL_INFO_SEGMENT, &seg);
  if (seg != HSA_AMD_SEGMENT_GLOBAL) return HSA_STATUS_SUCCESS;
  hsa_amd_memory_pool_get_info(p, HSA_AMD_MEMORY_POOL_INFO_LOCATION, &loc);
  hsa_amd_memory_pool_get_info(p, HSA_AMD_MEMORY_POOL_INFO_RUNTIME_ALLOC_ALLOWED, &alloc);
  if (loc != HSA_AMD_MEMORY_POOL_LOCATION_GPU || !alloc) return HSA_STATUS_SUCCESS;
  for (auto cpu : c.inv->cpus) {
    hsa_amd_memory_pool_access_t acc{};
    hsa_amd_agent_memory_pool_get_info(cpu, p, HSA_AMD_AGENT_MEMORY_POOL_INFO_ACCESS, &acc);
    if (acc == HSA_AMD_MEMORY_POOL_ACCESS_NEVER_ALLOWED) continue;
    hsa_amd_memory_pool_get_info(p, HSA_AMD_MEMORY_POOL_INFO_RUNTIME_ALLOC_GRANULE, &c.granule);
    c.pool = p; c.found = true; break;
  }
  return HSA_STATUS_SUCCESS;
}

struct Card {
  int dev = 0; char pci[32]{};
  int fd = -1; size_t alloc = 0;
  void* gpu_va = nullptr;     // card-local VMM buffer (receive buffer), GPU view
  uint8_t* cpu_map = nullptr; // the same memory through the exported DMA-BUF, CPU view (BAR)
  uint8_t* host_src = nullptr;  // mapped host memory the card pulls from (GPU pull arm)
  uint8_t* host_part = nullptr; // mapped host memory the producer writes (partials)
  unsigned* err = nullptr; unsigned long long* prod_bytes = nullptr;
  hipStream_t s{}, ps{};
};

static double now_us() { timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec * 1e6 + t.tv_nsec / 1e3; }
static void pin(int cpu) { cpu_set_t m; CPU_ZERO(&m); CPU_SET(cpu, &m); pthread_setaffinity_np(pthread_self(), sizeof(m), &m); }
static void dmabuf_sync(int fd, uint64_t flags) {
  dma_buf_sync s{}; s.flags = flags; int rc;
  do { rc = ioctl(fd, DMA_BUF_IOCTL_SYNC, &s); } while (rc < 0 && (errno == EAGAIN || errno == EINTR));
  if (rc < 0) { fprintf(stderr, "FAIL DMA_BUF_IOCTL_SYNC errno %d\n", errno); exit(2); }
}
// CPU push: non-temporal 32-byte stores from a cached host buffer into the card's BAR mapping, then a store fence.
static void push_nt(uint8_t* dst, const uint8_t* src, size_t n) {
  size_t i = 0;
  for (; i + 128 <= n; i += 128) {
    __m256i a = _mm256_load_si256((const __m256i*)(src + i)), b = _mm256_load_si256((const __m256i*)(src + i + 32));
    __m256i c = _mm256_load_si256((const __m256i*)(src + i + 64)), d = _mm256_load_si256((const __m256i*)(src + i + 96));
    _mm256_stream_si256((__m256i*)(dst + i), a); _mm256_stream_si256((__m256i*)(dst + i + 32), b);
    _mm256_stream_si256((__m256i*)(dst + i + 64), c); _mm256_stream_si256((__m256i*)(dst + i + 96), d);
  }
  for (; i < n; i += 4) *(volatile uint32_t*)(dst + i) = *(const uint32_t*)(src + i);
  _mm_sfence();
}
// CPU three-partial BF16 sum (fp32 accumulate, round to nearest even), the host reduce of the all-reduce.
static inline __m256 bf16x8_to_f32(__m128i v) { return _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_cvtepu16_epi32(v), 16)); }
static inline __m128i f32x8_to_bf16(__m256 f) {
  __m256i x = _mm256_castps_si256(f);
  __m256i r = _mm256_add_epi32(x, _mm256_add_epi32(_mm256_set1_epi32(0x7fff), _mm256_and_si256(_mm256_srli_epi32(x, 16), _mm256_set1_epi32(1))));
  r = _mm256_srli_epi32(r, 16);
  return _mm_packus_epi32(_mm256_castsi256_si128(r), _mm256_extracti128_si256(r, 1));
}
static void sum3_bf16(uint16_t* out, const uint16_t* a, const uint16_t* b, const uint16_t* c, size_t n) {
  for (size_t i = 0; i < n; i += 8) {
    __m256 s = _mm256_add_ps(_mm256_add_ps(bf16x8_to_f32(_mm_loadu_si128((const __m128i*)(a + i))), bf16x8_to_f32(_mm_loadu_si128((const __m128i*)(b + i)))),
                             bf16x8_to_f32(_mm_loadu_si128((const __m128i*)(c + i))));
    _mm_storeu_si128((__m128i*)(out + i), f32x8_to_bf16(s));
  }
}

struct Spin { std::atomic<int> go{0}; };

int main(int argc, char** argv) {
  std::vector<size_t> sizes{327680, 1048576, 4194304, 10485760};
  std::vector<int> cpus{0, 2, 4};
  int iters = 30, push_one = 0, push_pair = -1; bool reverse = false;
  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    auto next = [&]() { if (i + 1 >= argc) { fprintf(stderr, "missing value for %s\n", a.c_str()); exit(1); } return std::string(argv[++i]); };
    auto list = [](const std::string& s) { std::vector<long long> v; size_t p = 0; while (p < s.size()) { size_t q = s.find(',', p); if (q == std::string::npos) q = s.size(); v.push_back(atoll(s.substr(p, q - p).c_str())); p = q + 1; } return v; };
    if (a == "--sizes") { sizes.clear(); for (auto x : list(next())) sizes.push_back((size_t)x); }
    else if (a == "--cpus") { cpus.clear(); for (auto x : list(next())) cpus.push_back((int)x); }
    else if (a == "--iters") iters = atoi(next().c_str());
    else if (a == "--push-one") push_one = atoi(next().c_str());
    else if (a == "--push-pair") push_pair = atoi(next().c_str());
    else if (a == "--reverse") reverse = true;
    else { fprintf(stderr, "unknown argument %s\n", a.c_str()); return 1; }
  }
  for (auto s : sizes) if (s % 128 || s < 4096 || s > (64u << 20)) { fprintf(stderr, "size must be a multiple of 128 in 4 KiB..64 MiB\n"); return 1; }
  const size_t maxs = *std::max_element(sizes.begin(), sizes.end());

  int nd = 0; CK(hipGetDeviceCount(&nd));
  if (push_pair < 0) push_pair = nd - 1;  // default: the last device pushes with --push-one in the pair arm
  if (nd < 2 || nd > MAXD || (int)cpus.size() < nd) { fprintf(stderr, "need 2..%d devices and one cpu per device\n", MAXD); return 1; }
  HK(hsa_init());
  Inventory inv; HK(hsa_iterate_agents(collect, &inv));
  std::vector<Card> cards(nd);
  int wall_khz = 0; CK(hipDeviceGetAttribute(&wall_khz, hipDeviceAttributeWallClockRate, 0));
  const double prod_limit_ms = 60.0; const unsigned long long prod_ticks = (unsigned long long)(prod_limit_ms * wall_khz);
  int* stop_flag = nullptr; CK(hipHostMalloc(&stop_flag, sizeof(int), hipHostMallocMapped | hipHostMallocCoherent | hipHostMallocPortable));
  for (int d = 0; d < nd; ++d) {
    Card& c = cards[d]; c.dev = d;
    CK(hipSetDevice(d)); CK(hipDeviceGetPCIBusId(c.pci, sizeof(c.pci), d));
    unsigned dom, bus, dv, fn; sscanf(c.pci, "%x:%x:%x.%x", &dom, &bus, &dv, &fn);
    uint32_t want = (bus << 8) | (dv << 3) | fn; hsa_agent_t gpu{}; int match = 0;
    for (auto g : inv.gpus) { uint32_t b = 0, dm = 0;
      hsa_agent_get_info(g, (hsa_agent_info_t)HSA_AMD_AGENT_INFO_BDFID, &b); hsa_agent_get_info(g, (hsa_agent_info_t)HSA_AMD_AGENT_INFO_DOMAIN, &dm);
      if (b == want && dm == dom) { gpu = g; ++match; } }
    if (match != 1) { fprintf(stderr, "FAIL HSA agent for %s\n", c.pci); return 2; }
    PoolChoice pc{&inv}; HK(hsa_amd_agent_iterate_memory_pools(gpu, pick_pool, &pc));
    if (!pc.found) { fprintf(stderr, "FAIL no CPU-accessible GPU pool on %s\n", c.pci); return 2; }
    c.alloc = (maxs + pc.granule - 1) / pc.granule * pc.granule;
    hsa_amd_vmem_alloc_handle_t h{};
    HK(hsa_amd_vmem_handle_create(pc.pool, c.alloc, MEMORY_TYPE_PINNED, 0, &h));
    HK(hsa_amd_vmem_export_shareable_handle(&c.fd, h, 0));
    HK(hsa_amd_vmem_address_reserve(&c.gpu_va, c.alloc, 0, 0));
    HK(hsa_amd_vmem_map(c.gpu_va, c.alloc, 0, h, 0));
    hsa_amd_memory_access_desc_t g{HSA_ACCESS_PERMISSION_RW, gpu}; HK(hsa_amd_vmem_set_access(c.gpu_va, c.alloc, &g, 1));
    void* m = mmap(nullptr, c.alloc, PROT_READ | PROT_WRITE, MAP_SHARED, c.fd, 0);
    if (m == MAP_FAILED) { fprintf(stderr, "FAIL mmap DMA-BUF %s errno %d\n", c.pci, errno); return 2; }
    c.cpu_map = (uint8_t*)m;
    CK(hipHostMalloc((void**)&c.host_src, maxs, hipHostMallocMapped | hipHostMallocCoherent | hipHostMallocPortable));
    CK(hipHostMalloc((void**)&c.host_part, maxs, hipHostMallocMapped | hipHostMallocCoherent | hipHostMallocPortable));
    CK(hipMalloc(&c.err, sizeof(unsigned))); CK(hipMalloc(&c.prod_bytes, sizeof(unsigned long long)));
    CK(hipStreamCreateWithFlags(&c.s, hipStreamNonBlocking)); CK(hipStreamCreateWithFlags(&c.ps, hipStreamNonBlocking));
    printf("{\"device\":%d,\"pci\":\"%s\",\"alloc\":%zu}\n", d, c.pci, c.alloc);
  }
  // The reduced message in cached host memory (CPU push source); copies in each card's mapped host source (GPU pull).
  uint8_t* reduced = (uint8_t*)aligned_alloc(64, maxs);
  std::vector<uint16_t*> parts(3); for (auto& p : parts) { p = (uint16_t*)aligned_alloc(64, maxs); for (size_t i = 0; i < maxs / 2; ++i) p[i] = (uint16_t)(0x3f80 + (i & 63)); }
  uint16_t* sum_out = (uint16_t*)aligned_alloc(64, maxs);

  const char* arms_fwd[] = {"pull_all", "push_all", "push_one_pull_rest", "push_pair_pull_rest", "push_all_producers", "pull_all_producers", "cpu_sum3"};
  std::vector<std::string> arms(arms_fwd, arms_fwd + 7);
  if (reverse) std::reverse(arms.begin(), arms.end());
  uint32_t gen = 1; unsigned long long total_bad = 0;

  for (size_t S : sizes) {
    const uint32_t words = S / 4;
    for (int it = 0; it < iters; ++it) {
      std::vector<std::string> order = arms; std::rotate(order.begin(), order.begin() + (it % order.size()), order.end());
      for (const std::string& arm : order) {
        ++gen;
        for (uint32_t i = 0; i < words; ++i) ((uint32_t*)reduced)[i] = value_at(i, gen);
        for (auto& c : cards) memcpy(c.host_src, reduced, S);
        for (auto& c : cards) { CK(hipSetDevice(c.dev)); clear_words<<<64, 256, 0, c.s>>>((uint32_t*)c.gpu_va, words); CK(hipStreamSynchronize(c.s)); }
        if (arm == "cpu_sum3") {
          Spin sp; std::vector<double> t0(nd), t1(nd); std::vector<std::thread> th;
          size_t per = (S / 2 / nd + 7) / 8 * 8;
          for (int d = 0; d < nd; ++d) th.emplace_back([&, d] { pin(cpus[d]); size_t b = std::min(S / 2, per * d), e = std::min(S / 2, per * (d + 1));
            while (!sp.go.load(std::memory_order_acquire)) {} t0[d] = now_us(); sum3_bf16(sum_out + b, parts[0] + b, parts[1] + b, parts[2] + b, e - b); t1[d] = now_us(); });
          sp.go.store(1, std::memory_order_release); for (auto& t : th) t.join();
          double mk = *std::max_element(t1.begin(), t1.end()) - *std::min_element(t0.begin(), t0.end());
          printf("{\"arm\":\"%s\",\"bytes\":%zu,\"iter\":%d,\"makespan_us\":%.2f,\"read_gbs\":%.3f}\n", arm.c_str(), S, it, mk, 3.0 * S / mk / 1e3);
          continue;
        }
        const bool prod = arm.find("producers") != std::string::npos;
        if (prod) {
          *stop_flag = 0;
          for (auto& c : cards) { CK(hipSetDevice(c.dev)); CK(hipMemsetAsync(c.prod_bytes, 0, 8, c.ps));
            producer<<<64, 256, 0, c.ps>>>((uint4*)c.host_part, S / 16, (volatile int*)stop_flag, c.prod_bytes, prod_ticks); }
          double w = now_us(); while (now_us() - w < 2000) {}  // producers running before the transfer starts
        }
        Spin sp; std::vector<double> t0(nd), t1(nd); std::vector<int> mode(nd);
        for (int d = 0; d < nd; ++d)
          mode[d] = (arm == "pull_all" || arm == "pull_all_producers") ? 0 : (arm == "push_one_pull_rest" ? (d == push_one ? 1 : 0)
                  : arm == "push_pair_pull_rest" ? (d == push_one || d == push_pair ? 1 : 0) : 1);
        std::vector<std::thread> th;
        for (int d = 0; d < nd; ++d) th.emplace_back([&, d] {
          pin(cpus[d]); Card& c = cards[d]; CK(hipSetDevice(c.dev));
          while (!sp.go.load(std::memory_order_acquire)) {}
          t0[d] = now_us();
          if (mode[d] == 1) {
            dmabuf_sync(c.fd, DMA_BUF_SYNC_START | DMA_BUF_SYNC_WRITE);
            push_nt(c.cpu_map, reduced, S);
            dmabuf_sync(c.fd, DMA_BUF_SYNC_END | DMA_BUF_SYNC_WRITE);
          } else {
            pull_copy<<<64, 256, 0, c.s>>>((const uint4*)c.host_src, (uint4*)c.gpu_va, S / 16);
            CK(hipStreamSynchronize(c.s));
          }
          t1[d] = now_us();
        });
        sp.go.store(1, std::memory_order_release); for (auto& t : th) t.join();
        double prod_gbs = 0;
        if (prod) {
          double w0 = now_us(); *stop_flag = 1; __sync_synchronize();
          for (auto& c : cards) { CK(hipSetDevice(c.dev)); CK(hipStreamSynchronize(c.ps)); }
          (void)w0;
          unsigned long long pb = 0, x = 0;
          for (auto& c : cards) { CK(hipSetDevice(c.dev)); CK(hipMemcpy(&x, c.prod_bytes, 8, hipMemcpyDeviceToHost)); pb += x; }
          prod_gbs = pb;  // bytes written by producers during the whole window (reported, not rated)
        }
        // a transfer that took most of the producers' time limit ran behind them, not beside them: report it
        int serialized = 0; if (prod) for (int d = 0; d < nd; ++d) serialized += (t1[d] - t0[d]) > 0.8e3 * prod_limit_ms;
        unsigned bad_total = 0; std::string per = "";
        for (int d = 0; d < nd; ++d) {
          Card& c = cards[d]; CK(hipSetDevice(c.dev)); CK(hipMemsetAsync(c.err, 0, 4, c.s));
          verify_words<<<256, 256, 0, c.s>>>((const uint32_t*)c.gpu_va, words, gen, c.err);
          unsigned bad = 0; CK(hipMemcpyAsync(&bad, c.err, 4, hipMemcpyDeviceToHost, c.s)); CK(hipStreamSynchronize(c.s));
          bad_total += bad;
          char buf[200]; snprintf(buf, sizeof buf, "%s{\"dev\":%d,\"mode\":\"%s\",\"us\":%.2f,\"gbs\":%.3f,\"bad\":%u}", d ? "," : "", d,
                                  mode[d] ? "cpu_push" : "gpu_pull", t1[d] - t0[d], S / (t1[d] - t0[d]) / 1e3, bad);
          per += buf;
        }
        total_bad += bad_total;
        double mk = *std::max_element(t1.begin(), t1.end()) - *std::min_element(t0.begin(), t0.end());
        printf("{\"arm\":\"%s\",\"bytes\":%zu,\"iter\":%d,\"makespan_us\":%.2f,\"producer_bytes\":%.0f,\"serialized\":%d,\"bad\":%u,\"cards\":[%s]}\n",
               arm.c_str(), S, it, mk, prod_gbs, serialized, bad_total, per.c_str());
        fflush(stdout);
      }
    }
  }
  printf("{\"complete\":true,\"bad_words\":%llu}\n", total_bad);
  return total_bad ? 3 : 0;
}
