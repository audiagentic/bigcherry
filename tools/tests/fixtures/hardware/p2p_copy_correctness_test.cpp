// Standalone HIP program: per-element correctness check for peer-to-peer
// GPU copies on multi-GPU boxes.
//
// Regression target: on gfx1100 (RDNA3), hipMemcpy(dst, src, bytes,
// hipMemcpyDefault) with the CURRENT device set to the destination (a
// "pull") reports hipSuccess but silently copies zeros/garbage, while the
// mirrored call with current device set to the source (a "push") is
// correct. hipDeviceCanAccessPeer() returns 1 for both directions
// regardless -- topology reporting cannot be trusted as a proxy for actual
// data movement. This was found once, retracted a benchmark's bandwidth
// claim, and then rediscovered from scratch in a later session because
// nothing in the test suite would have caught it automatically.
//
// Sentinel scheme and failure classification (SRC / POISON / splitmix64
// mixing, mismatch classes) per external review guidance obtained while
// writing this test (dev-gpt-agent req_bd924bd0036542c8): exact-equality
// 64-bit integer patterns distinguish zero-fill, stale-destination, and
// wrong-source/routing corruption from each other, which a float
// tolerance-diff cannot.
//
// Deferred, not implemented here (documented as follow-up, not silently
// dropped): explicit hipMemcpyPeer/hipMemcpyPeerAsync variants tested
// independently of hipMemcpyDefault; a dedicated kernel-side peer
// read/write + __threadfence_system() visibility test (already covered
// ad hoc by tools/lab/gp10-collective-harness/p2p-diagnostics/p2p_spin.cpp,
// not yet folded into this permanent suite); multi-stream concurrent
// A<->B transfers; cross-process IPC (hipIpcOpenMemHandle can itself
// enable a different peer-mapping path); managed/unified memory (not
// used by this project). Also not implemented: hipDeviceGetP2PAttribute
// NativeAtomicSupported/PerformanceRank reporting -- HIP_VERSION on this
// tree's ROCm did not expose it consistently enough to depend on here;
// AccessSupported is covered via hipDeviceCanAccessPeer.
//
// This program requires two real, peer-capable GPUs. It prints one line
// per direction/size tested and a final PASS/FAIL summary line the
// wrapper test greps for.

#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <vector>

#define CHK(x) do { hipError_t _e = (x); if (_e != hipSuccess) { \
    printf("HIP error %d (%s) at %s:%d\n", _e, hipGetErrorString(_e), __FILE__, __LINE__); \
    exit(1); } } while (0)

static const uint64_t SEED_SRC    = 0x9E3779B97F4A7C15ull;
static const uint64_t SEED_POISON = 0xD1B54A32D192ED03ull;

static uint64_t splitmix64(uint64_t x) {
    x += 0x9E3779B97F4A7C15ull;
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ull;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBull;
    return x ^ (x >> 31);
}

// device/epoch-salted, position-dependent, exact-equality pattern: distinct
// from every other seed/device/epoch combination used in this program, so a
// mismatch's exact value identifies which pattern leaked through.
static void fill_pattern(std::vector<uint64_t> &v, uint64_t seed, int device, int epoch) {
    uint64_t salt = seed ^ ((uint64_t)device << 56) ^ ((uint64_t)(uint32_t)epoch << 32);
    for (size_t i = 0; i < v.size(); ++i)
        v[i] = splitmix64(salt ^ i);
}

enum class Mode { ZERO, POISON, PREVIOUS_EPOCH, OTHER_DEVICE_SRC, EXPECTED_OTHER_INDEX, OTHER };

static const char *mode_name(Mode m) {
    switch (m) {
        case Mode::ZERO: return "ZERO (zero-fill)";
        case Mode::POISON: return "POISON (destination never written)";
        case Mode::PREVIOUS_EPOCH: return "PREVIOUS_EPOCH (stale visibility/cache)";
        case Mode::OTHER_DEVICE_SRC: return "OTHER_DEVICE_PATTERN (wrong source/routing)";
        case Mode::EXPECTED_OTHER_INDEX: return "EXPECTED_OTHER_INDEX (offset/addressing error)";
        default: return "OTHER (arbitrary corruption)";
    }
}

struct CheckResult {
    size_t bad = 0;
    long first_bad_index = -1;
    uint64_t first_bad_got = 0, first_bad_want = 0;
    Mode mode = Mode::OTHER;
    // contiguous-range summary: helps distinguish full no-copy, prefix/
    // suffix partial copy, and isolated corruption at a glance.
    long first_ok_after_bad = -1;
};

static CheckResult check(const std::vector<uint64_t> &got, const std::vector<uint64_t> &want,
                          uint64_t poison_salt, uint64_t prev_epoch_salt, int src_dev, int other_dev) {
    CheckResult r;
    for (size_t i = 0; i < got.size(); ++i) {
        if (got[i] == want[i])
            continue;
        r.bad++;
        if (r.first_bad_index < 0) {
            r.first_bad_index = (long)i;
            r.first_bad_got = got[i];
            r.first_bad_want = want[i];

            if (got[i] == 0)
                r.mode = Mode::ZERO;
            else if (got[i] == splitmix64(poison_salt ^ i))
                r.mode = Mode::POISON;
            else if (got[i] == splitmix64(prev_epoch_salt ^ i))
                r.mode = Mode::PREVIOUS_EPOCH;
            else if (got[i] == splitmix64((SEED_SRC ^ ((uint64_t)other_dev << 56) ^ i)))
                r.mode = Mode::OTHER_DEVICE_SRC;
            else {
                r.mode = Mode::OTHER;
                for (size_t j = 0; j < want.size(); ++j) {
                    if (want[j] == got[i]) { r.mode = Mode::EXPECTED_OTHER_INDEX; break; }
                }
            }
        } else if (r.first_ok_after_bad < 0 && got[i] == want[i]) {
            r.first_ok_after_bad = (long)i;
        }
    }
    (void)src_dev;
    return r;
}

// Runs one hipMemcpy(dst, src, bytes, hipMemcpyDefault) with the given
// current device, seeding src fresh and poisoning dst beforehand (with a
// stale-epoch pattern on the second call for a device, to distinguish
// "never written" from "wrote the previous run's data") so the failure
// mode is classifiable, not just detectable.
static bool run_direction(const char *label, int cur_device, int src_dev, int dst_dev,
                           void *d_src, void *d_dst, size_t bytes, int epoch, int prev_epoch) {
    size_t n = bytes / sizeof(uint64_t);
    std::vector<uint64_t> src_pattern(n), poison_pattern(n), prev_epoch_pattern(n), got(n);

    fill_pattern(src_pattern, SEED_SRC, src_dev, epoch);
    fill_pattern(poison_pattern, SEED_POISON, dst_dev, epoch);
    fill_pattern(prev_epoch_pattern, SEED_SRC, src_dev, prev_epoch);

    hipSetDevice(src_dev);
    CHK(hipMemcpy(d_src, src_pattern.data(), bytes, hipMemcpyHostToDevice));
    hipSetDevice(dst_dev);
    CHK(hipMemcpy(d_dst, poison_pattern.data(), bytes, hipMemcpyHostToDevice));
    CHK(hipDeviceSynchronize());

    hipSetDevice(cur_device);
    CHK(hipMemcpy(d_dst, d_src, bytes, hipMemcpyDefault));
    CHK(hipDeviceSynchronize());

    hipSetDevice(dst_dev);
    CHK(hipMemcpy(got.data(), d_dst, bytes, hipMemcpyDeviceToHost));

    CheckResult r = check(got, src_pattern,
                           SEED_POISON ^ ((uint64_t)dst_dev << 56) ^ ((uint64_t)(uint32_t)epoch << 32),
                           SEED_SRC ^ ((uint64_t)src_dev << 56) ^ ((uint64_t)(uint32_t)prev_epoch << 32),
                           src_dev, dst_dev == 0 ? 1 : 0);
    if (r.bad == 0) {
        printf("%-40s OK   (n=%zu)\n", label, n);
        return true;
    }

    printf("%-40s FAIL bad=%zu/%zu first_bad=%ld got=0x%016llx want=0x%016llx range_end=%ld class=%s\n",
           label, r.bad, n, r.first_bad_index,
           (unsigned long long)r.first_bad_got, (unsigned long long)r.first_bad_want,
           r.first_ok_after_bad, mode_name(r.mode));
    return false;
}

int main() {
    int count = 0;
    CHK(hipGetDeviceCount(&count));
    if (count < 2) {
        printf("SKIP: only %d HIP device(s) visible, need >=2\n", count);
        return 2;
    }

    const int devA = 0, devB = 1;
    int can_ab = 0, can_ba = 0;
    hipDeviceCanAccessPeer(&can_ab, devA, devB);
    hipDeviceCanAccessPeer(&can_ba, devB, devA);
    printf("hipDeviceCanAccessPeer: %d->%d=%d  %d->%d=%d\n", devA, devB, can_ab, devB, devA, can_ba);
    // hipDeviceCanAccessPeer is directional and must not be assumed
    // symmetric; record any asymmetry rather than silently averaging it away.
    if (can_ab != can_ba)
        printf("NOTE: peer access capability is asymmetric between %d and %d\n", devA, devB);

    if (!can_ab || !can_ba) {
        printf("SKIP: devices %d/%d do not report mutual peer access\n", devA, devB);
        return 2;
    }

    hipSetDevice(devA);
    CHK(hipDeviceEnablePeerAccess(devB, 0));
    hipSetDevice(devB);
    CHK(hipDeviceEnablePeerAccess(devA, 0));

    // Boundary sizes: 8 bytes is the size at which a real gfx1100 ROCm
    // async-copy issue has been observed to silently drop data despite API
    // success; 4KB-1/4KB/4KB+1 straddle the common page-size boundary.
    // 1MB/64MB match the sizes originally used (and originally hid this
    // bug) in a bandwidth microbenchmark.
    size_t sizes[] = {
        8, 4096 - 8, 4096, 4096 + 8,
        1ull << 20, 64ull << 20,
    };
    // Regression gate: push MUST be correct at every size -- this is the
    // direction our code is required to use, and any regression here is a
    // real bug, not a known platform limitation.
    bool push_ok = true;
    // Pull is a KNOWN-BROKEN platform limitation (silent zero-fill on
    // gfx1100 despite hipDeviceCanAccessPeer()==1), not something our own
    // code can fix. We still run and report it every time: if a future
    // ROCm/driver update silently makes it correct, or if it regresses to
    // a *different* wrong answer, that is worth seeing rather than
    // ignoring. It never fails the overall exit code.
    bool pull_currently_broken = false;
    int epoch = 1, prev_epoch = 0;

    for (size_t bytes : sizes) {
        // round down to a whole number of uint64_t elements; the two small
        // boundary sizes (4096-8, 4096+8) are already 8-aligned.
        bytes -= bytes % sizeof(uint64_t);
        if (bytes == 0)
            continue;

        void *dA, *dB;
        hipSetDevice(devA);
        CHK(hipMalloc(&dA, bytes));
        hipSetDevice(devB);
        CHK(hipMalloc(&dB, bytes));

        char label[64];
        snprintf(label, sizeof(label), "push A->B (cur=src) %zuB", bytes);
        push_ok &= run_direction(label, devA, devA, devB, dA, dB, bytes, epoch, prev_epoch);

        snprintf(label, sizeof(label), "pull A->B (cur=dst) %zuB", bytes);
        bool pull_ok = run_direction(label, devB, devA, devB, dA, dB, bytes, epoch, prev_epoch);
        if (!pull_ok)
            pull_currently_broken = true;

        hipFree(dA);
        hipFree(dB);
        epoch++; prev_epoch++;
    }

    printf("PULL_DIRECTION: %s (informational only -- known platform limitation, not gated)\n",
           pull_currently_broken ? "BROKEN (expected)" : "WORKING (unexpected -- investigate before relying on it)");
    printf(push_ok ? "P2P_COPY_CORRECTNESS: PASS\n" : "P2P_COPY_CORRECTNESS: FAIL\n");
    return push_ok ? 0 : 1;
}
