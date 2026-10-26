#pragma once
// Included only by r3_moe_worker.cpp. The helper's process-wide settings, read from the environment once and
// validated: a malformed or out-of-range value is an error at start-up, never a silent default.
#include <charconv>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <string>

struct WorkerConfig {
    uint32_t helper_count = 1;      /* R3_MOE_WORKER_COUNT: helpers sharing the segment, 1..kMaxRanks */
    uint32_t helper_index = 0;      /* R3_MOE_WORKER_INDEX: this helper, < helper_count; 0 creates the segment */
    uint32_t engine_ranks = 0;      /* R3_MOE_WORKER_ENGINE_RANKS: 0 = the runner's default */
    bool cpu = false;               /* R3_MOE_WORKER_CPU: host fp32 reference instead of the GPU */
    bool power_save = true;         /* R3_MOE_WORKER_POWER_SAVE (RAD20): waits armed only inside bursts */
    bool queue_warm = true;         /* R3_MOE_WORKER_QUEUE_WARM (RAD20): oversubscribe only inside bursts */
    bool combine = false;           /* R3_MOE_WORKER_COMBINE (RAD18): helper 0 adds both helpers' rows on the CPU */
    uint32_t combine_min_rows = 256;   /* R3_MOE_WORKER_COMBINE_MIN_ROWS */
    uint32_t combine_threads = 8;      /* R3_MOE_WORKER_COMBINE_THREADS, capped at the CPU count */

    /* False with *err naming the variable on a malformed or out-of-range value. */
    bool from_environment(uint32_t max_helpers, std::string* err) {
        return u32("R3_MOE_WORKER_COUNT", 1, max_helpers, &helper_count, err) &&
               u32("R3_MOE_WORKER_INDEX", 0, max_helpers - 1, &helper_index, err) &&
               u32("R3_MOE_WORKER_ENGINE_RANKS", 1, max_helpers, &engine_ranks, err) &&
               flag("R3_MOE_WORKER_CPU", &cpu, err) &&
               flag("R3_MOE_WORKER_POWER_SAVE", &power_save, err) &&
               flag("R3_MOE_WORKER_QUEUE_WARM", &queue_warm, err) &&
               flag("R3_MOE_WORKER_COMBINE", &combine, err) &&
               u32("R3_MOE_WORKER_COMBINE_MIN_ROWS", 1, UINT32_MAX, &combine_min_rows, err) &&
               u32("R3_MOE_WORKER_COMBINE_THREADS", 1, 1024, &combine_threads, err) &&
               check(helper_index < helper_count, "R3_MOE_WORKER_INDEX must be below R3_MOE_WORKER_COUNT", err);
    }

    /* Before any configuration is read (GPU_MAX_HW_QUEUES must be set before HIP initialises): the same parse, but a
     * bad value just falls back here and is reported by from_environment() later. */
    static uint32_t peek_u32(const char* name, uint32_t fallback) {
        uint32_t v = fallback;
        std::string ignored;
        return u32(name, 0, UINT32_MAX, &v, &ignored) ? v : fallback;
    }
    static bool peek_flag(const char* name, bool fallback) {
        bool v = fallback;
        std::string ignored;
        return flag(name, &v, &ignored) ? v : fallback;
    }

private:
    /* Unset or empty keeps *out. */
    static bool u32(const char* name, uint32_t lo, uint32_t hi, uint32_t* out, std::string* err) {
        const char* s = std::getenv(name);
        if (!s || !*s) return true;
        uint32_t v = 0;
        const char* end = s + std::strlen(s);
        const auto [p, ec] = std::from_chars(s, end, v);
        if (ec != std::errc{} || p != end || v < lo || v > hi) {
            *err = std::string(name) + "=" + s + ": want an integer in " + std::to_string(lo) + ".." + std::to_string(hi);
            return false;
        }
        *out = v;
        return true;
    }
    /* "0" off, "1" on; unset or empty keeps *out. */
    static bool flag(const char* name, bool* out, std::string* err) {
        const char* s = std::getenv(name);
        if (!s || !*s) return true;
        if (!std::strcmp(s, "0") || !std::strcmp(s, "1")) {
            *out = s[0] == '1';
            return true;
        }
        *err = std::string(name) + "=" + s + ": want 0 or 1";
        return false;
    }
    static bool check(bool ok, const char* what, std::string* err) {
        if (!ok) *err = what;
        return ok;
    }
};
