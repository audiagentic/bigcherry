#pragma once
#include "experts.h"

// The .rad container reader and the process signal flag. Included only by r3_moe_worker.cpp, at the position the code had in the single file (RCS03, pure move).

namespace {
using namespace r3::moe_shm;
/* The stop word: set by the signal handler and by any failing thread, and futex-woken so the main thread sleeps on it
 * instead of polling. A lock-free atomic store plus the raw futex syscall are async-signal-safe. */
std::atomic<uint32_t> g_stop_word{0};
static_assert(std::atomic<uint32_t>::is_always_lock_free && sizeof(std::atomic<uint32_t>) == sizeof(uint32_t));
inline bool stopping() { return g_stop_word.load(std::memory_order_acquire) != 0; }
inline void request_stop() {
    g_stop_word.store(1, std::memory_order_release);
    (void)::syscall(SYS_futex, reinterpret_cast<uint32_t*>(&g_stop_word), FUTEX_WAKE_PRIVATE, INT_MAX, nullptr, nullptr, 0);
}
void on_signal(int) { request_stop(); }

/* Test only (RAD24 kill test): R3_MOE_WORKER_TEST_STOP_AT=<point>:<helper index> stops that helper with SIGSTOP when it
 * reaches <point>, so the test can SIGKILL it at exactly that moment. Points: park_drain (a small lane has set
 * kParkDraining and not yet re-checked the burst). Unset in every real run. */
inline void test_stop_at(const char* point, uint32_t helper_index) {
    const char* const want = r3_serving::settings().test_stop_at.c_str();
    if (!want || !*want) return;
    const size_t n = std::strlen(point);
    if (std::strncmp(want, point, n) != 0 || want[n] != ':' || (uint32_t)std::atoi(want + n + 1) != helper_index) return;
    std::fprintf(stderr, "r3_moe_worker: test stop at %s\n", point);
    (void)::raise(SIGSTOP);
}

float bf16(uint16_t v) {
    uint32_t u = (uint32_t)v << 16; float f; std::memcpy(&f, &u, 4); return f;
}
uint16_t to_bf16(float f) {
    uint32_t u; std::memcpy(&u, &f, 4);
    const uint32_t lsb = (u >> 16) & 1u;
    u += 0x7fffu + lsb;
    return (uint16_t)(u >> 16);
}
float f16(uint16_t h) {
    const uint32_t s = (h >> 15) & 1, e = (h >> 10) & 31, m = h & 1023;
    if (!e) return std::ldexp((s ? -1.f : 1.f) * (float)m, -24);
    if (e == 31) return m ? NAN : (s ? -INFINITY : INFINITY);
    return std::ldexp((s ? -1.f : 1.f) * (1024.f + (float)m), (int)e - 25);
}
float f8e4m3(uint8_t b) {
    const float sign = (b & 0x80) ? -1.f : 1.f;
    const int e = (b >> 3) & 15, m = b & 7;
    if (e == 0) return sign * std::ldexp((float)m, -9);
    if (e == 15 && m == 7) return NAN; /* OCP E4M3FN's sole NaN payload. */
    return sign * std::ldexp(1.f + (float)m / 8.f, e - 7);
}

uint64_t ubits(const uint8_t* row, int bits, int64_t col) {
    const uint64_t bit = (uint64_t)col * (uint64_t)bits;
    const uint64_t byte = bit >> 3, shift = bit & 7;
    uint64_t x = row[byte];
    if (shift + (uint64_t)bits > 8) x |= (uint64_t)row[byte + 1] << 8;
    return (x >> shift) & ((1ull << bits) - 1);
}
int64_t sbits(uint64_t x, int bits) {
    const uint64_t sign = 1ull << (bits - 1);
    return (int64_t)((x ^ sign) - sign);
}

/* The .rad container reader is bc::experts::Container (components/serving/experts.h): the one reader bc-serve plans
 * ownership with, over Radiance's installed container ABI only. */
using bc::experts::Container;
} // namespace
