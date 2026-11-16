// VTP36 item: the R9700's two-shot step in isolation, fused kernel versus kernel-out plus copy-engine-in.
// With the duplex weights the rank behind the narrow link owns no shard: per call it sends its whole partial (S) to host
// memory and assembles the whole reduced result (S) from host memory, in P pieces.
//   A fused   : one kernel; step 0 sends piece 0, step k sends piece k and assembles piece k-1, the last step assembles.
//   B dmalane : one kernel sends the pieces and raises a per-piece flag; a second stream waits on each flag
//               (hipStreamWaitValue32) and copies that piece in with hipMemcpyAsync (the copy engine).
//   C free    : kernel out of S beside one memcpy in of S, no dependency (the link's duplex ceiling).
// In isolation a piece is ready to assemble as soon as it has been sent; in the engine it also waits on the peers.
// No kernel spins on a flag, so nothing here can hang on the device. Every assembled word is checked on the host.
#include <hip/hip_runtime.h>
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#define OK(x) do { hipError_t e_ = (x); if (e_ != hipSuccess) { std::fprintf(stderr, "%s: %s\n", #x, hipGetErrorString(e_)); std::exit(1); } } while (0)

// block b owns a contiguous slice of every piece; the grid walks the pieces in order
__global__ void fused(const int4* __restrict__ part, int4* __restrict__ hout, const int4* __restrict__ hin,
                      int4* __restrict__ res, int n16, int P) {
  const int pn = n16 / P;
  const int w = (pn + gridDim.x - 1) / gridDim.x, s0 = blockIdx.x * w;
  const int e0 = min(s0 + w, pn);
  for (int step = 0; step <= P; ++step) {
    const int send = step < P ? step : -1, asm_ = step > 0 ? step - 1 : -1;
    for (int k = s0 + threadIdx.x; k < e0; k += blockDim.x) {
      int4 v{};
      if (asm_ >= 0) v = hin[asm_ * pn + k];
      if (send >= 0) hout[send * pn + k] = part[send * pn + k];
      if (asm_ >= 0) res[asm_ * pn + k] = v;
    }
    __syncthreads();
  }
  __threadfence_system();
}
// sends piece by piece; after each piece the last block to finish it raises flag[piece] (system scope)
__global__ void send_flag(const int4* __restrict__ part, int4* __restrict__ hout, int n16, int P,
                          unsigned* count, volatile unsigned* flag, unsigned gen) {
  const int pn = n16 / P;
  const int w = (pn + gridDim.x - 1) / gridDim.x, s0 = blockIdx.x * w;
  const int e0 = min(s0 + w, pn);
  for (int p = 0; p < P; ++p) {
    for (int k = s0 + threadIdx.x; k < e0; k += blockDim.x) hout[p * pn + k] = part[p * pn + k];
    __threadfence_system();
    __syncthreads();
    if (threadIdx.x == 0) {
      if (atomicAdd(&count[p], 1u) == gridDim.x - 1) { count[p] = 0; __threadfence_system(); flag[p] = gen; }
    }
  }
}
__global__ void send_all(const int4* __restrict__ part, int4* __restrict__ hout, int n16) {
  const int w = (n16 + gridDim.x - 1) / gridDim.x, s0 = blockIdx.x * w, e0 = min(s0 + w, n16);
  for (int k = s0 + threadIdx.x; k < e0; k += blockDim.x) hout[k] = part[k];
  __threadfence_system();
}

static double now() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }

int main(int argc, char** argv) {
  const int dev = argc > 1 ? std::atoi(argv[1]) : 2;
  const int P = argc > 2 ? std::atoi(argv[2]) : 4;
  const int nb = 4, nt = 256, iters = 40;
  OK(hipSetDevice(dev));
  char bus[64] = {0}; OK(hipDeviceGetPCIBusId(bus, sizeof bus, dev));
  const size_t sizes[] = {1u << 20, 4u << 20, 10485760};
  const size_t NMAX = 10485760;
  const unsigned fl = hipHostMallocCoherent | hipHostMallocMapped | hipHostMallocPortable;
  int4 *hout, *hin, *part, *res; unsigned *count, *flag;
  OK(hipHostMalloc((void**)&hout, NMAX, fl)); OK(hipHostMalloc((void**)&hin, NMAX, fl));
  OK(hipHostMalloc((void**)&flag, 64 * sizeof(unsigned), fl)); std::memset(flag, 0, 64 * sizeof(unsigned));
  OK(hipMalloc((void**)&part, NMAX)); OK(hipMalloc((void**)&res, NMAX)); OK(hipMalloc((void**)&count, 64 * sizeof(unsigned)));
  OK(hipMemset(count, 0, 64 * sizeof(unsigned))); OK(hipMemset(part, 3, NMAX));
  hipStream_t s1, s2; OK(hipStreamCreateWithFlags(&s1, hipStreamNonBlocking)); OK(hipStreamCreateWithFlags(&s2, hipStreamNonBlocking));
  std::vector<unsigned> chk(NMAX / 4);
  std::printf("device %d %s, %d pieces, grid %d x %d, %d iterations (median of the last %d)\n", dev, bus, P, nb, nt, iters, iters - 4);
  unsigned gen = 0; long bad_total = 0;
  for (size_t S : sizes) {
    const int n16 = (int)(S / 16); const size_t pb = S / P;
    if (S % (16 * (size_t)P)) { std::fprintf(stderr, "size %zu not divisible\n", S); return 1; }
    for (const char* arm : {"A fused", "B dmalane", "C free duplex"}) {
      std::vector<double> t; long bad = 0;
      for (int it = 0; it < iters; ++it) {
        ++gen;
        unsigned* h = (unsigned*)hin; for (size_t i = 0; i < S / 4; ++i) h[i] = (unsigned)i * 2654435761u + gen;
        OK(hipMemset(res, 0, S)); OK(hipDeviceSynchronize());
        const double t0 = now();
        if (arm[0] == 'A') {
          fused<<<nb, nt, 0, s1>>>(part, hout, hin, res, n16, P);
        } else if (arm[0] == 'B') {
          send_flag<<<nb, nt, 0, s1>>>(part, hout, n16, P, count, flag, gen);
          for (int p = 0; p < P; ++p) {
            OK(hipStreamWaitValue32(s2, flag + p, gen, hipStreamWaitValueEq, 0xffffffffu));
            OK(hipMemcpyAsync((char*)res + p * pb, (char*)hin + p * pb, pb, hipMemcpyHostToDevice, s2));
          }
        } else {
          send_all<<<nb, nt, 0, s1>>>(part, hout, n16);
          OK(hipMemcpyAsync(res, hin, S, hipMemcpyHostToDevice, s2));
        }
        // watchdog: never leave a copy-engine wait pending; after 2 s the host raises the flags itself and fails the run
        bool timed_out = false;
        while (hipStreamQuery(s1) == hipErrorNotReady || hipStreamQuery(s2) == hipErrorNotReady) {
          if (now() - t0 > 2.0) { timed_out = true; for (int p = 0; p < P; ++p) ((volatile unsigned*)flag)[p] = gen; break; }
        }
        OK(hipStreamSynchronize(s1)); OK(hipStreamSynchronize(s2));
        if (timed_out) { std::fprintf(stderr, "TIMEOUT arm %s size %zu: flags released by the host\n", arm, S); return 4; }
        const double dt = now() - t0;
        OK(hipMemcpy(chk.data(), res, S, hipMemcpyDeviceToHost));
        for (size_t i = 0; i < S / 4; ++i) bad += chk[i] != h[i];
        if (it >= 4) t.push_back(dt);
      }
      std::sort(t.begin(), t.end());
      const double med = t[t.size() / 2], p90 = t[t.size() * 9 / 10];
      std::printf("  %9zu B  %-14s p50 %7.0f us  p90 %7.0f us  %6.2f GB/s both ways  bad %ld\n", S, arm, med * 1e6, p90 * 1e6, 2.0 * S / med / 1e9, bad);
      bad_total += bad;
    }
  }
  std::printf("bad_words %ld\n", bad_total);
  return bad_total ? 3 : 0;
}
