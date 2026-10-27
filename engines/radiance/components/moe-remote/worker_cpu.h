#pragma once
// Included only by r3_moe_worker.cpp. PWR03: the helper's CPU placement. Its lane threads spend bursts waiting on the GPU
// (the runtime spins before it blocks, RAD25), so on a CPU with efficiency cores they cost far less power there: helpers
// on the four E-cores of the bench host cut CPU package power 77 -> 42 W in decode, 83 -> 52 W in prefill, with decode
// and prefill speed unchanged (pwr03-ecore-ab.sh). Efficiency cores are found generically: CPUs whose cpu_capacity is
// below the largest (any asymmetric CPU that reports it), else the hybrid "atom" PMU's CPU list. None found: unchanged.
#include <sched.h>

#include <algorithm>

#include <cstdio>
#include <fstream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

namespace worker_cpu {

/* "0-3,8,10-11" -> {0,1,2,3,8,10,11}; empty on a malformed list. */
inline std::vector<int> parse_list(const std::string& s) {
    std::vector<int> out;
    std::stringstream ss(s);
    std::string part;
    while (std::getline(ss, part, ',')) {
        if (part.empty()) continue;
        const size_t dash = part.find('-');
        try {
            const int a = std::stoi(part.substr(0, dash));
            const int b = dash == std::string::npos ? a : std::stoi(part.substr(dash + 1));
            if (a < 0 || b < a) return {};
            for (int c = a; c <= b; ++c) out.push_back(c);
        } catch (...) {
            return {};
        }
    }
    return out;
}

inline std::vector<int> efficiency_cpus() {
    std::map<int, long> cap;
    for (int c = 0; c < 4096; ++c) {
        std::ifstream f("/sys/devices/system/cpu/cpu" + std::to_string(c) + "/cpu_capacity");
        if (!f) {
            if (c > 0 && !std::ifstream("/sys/devices/system/cpu/cpu" + std::to_string(c))) break;
            continue;
        }
        long v = 0;
        if (f >> v) cap[c] = v;
    }
    long lo = -1, hi = -1;
    for (const auto& [c, v] : cap) {
        lo = lo < 0 ? v : std::min(lo, v);
        hi = std::max(hi, v);
    }
    std::vector<int> out;
    if (!cap.empty() && lo < hi) {
        for (const auto& [c, v] : cap)
            if (v < hi) out.push_back(c);
        return out;
    }
    std::ifstream atom("/sys/devices/cpu_atom/cpus");
    std::string list;
    if (atom && std::getline(atom, list)) return parse_list(list);
    return {};
}

/* R3_MOE_WORKER_CPU_SET=auto: pin this process (and every thread it starts later) to the efficiency cores when the CPU
 * has them. Called before any thread is created. */
inline void place(bool auto_set) {
    if (!auto_set) return;
    const std::vector<int> cpus = efficiency_cpus();
    if (cpus.empty()) return;
    cpu_set_t set;
    CPU_ZERO(&set);
    for (int c : cpus) CPU_SET(c, &set);
    if (sched_setaffinity(0, sizeof(set), &set) != 0) {
        std::perror("r3_moe_worker: sched_setaffinity (efficiency cores)");
        return;
    }
    std::fprintf(stderr, "r3_moe_worker: R3_MOE_WORKER_CPU_SET auto - on %zu efficiency core(s)\n", cpus.size());
}

}  // namespace worker_cpu
