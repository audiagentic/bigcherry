#pragma once
// RSC02: a card-free, BDF-scoped read/wait boundary for post-measurement VRAM accounting.
// C++17 only: shared headers must also remain consumable by the remote-MoE worker.
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>
namespace bc::serving::session::vram {
struct Refused : std::runtime_error {
    using std::runtime_error::runtime_error;
};
struct Hardware {
    // Production reads the sysfs file; tests can replace this OS boundary.
    std::function<uint64_t(const std::filesystem::path&)> read_used;
    // The session owner services its stop socket while waiting; false cancels admission.
    std::function<bool(std::chrono::milliseconds)> wait;
};
struct Observation {
    uint64_t baseline_bytes = 0; // highest observed pre-start usage
    uint64_t tolerance_bytes = 0; // observed baseline spread, never a fixed MiB fudge
};
using Baseline = std::map<std::string, Observation>;
Baseline snapshot(const std::vector<std::string>& bdfs, const std::filesystem::path& root,
                  const Hardware& hardware);
void await_release(const Baseline& baseline, const std::filesystem::path& root,
                   std::chrono::milliseconds stop_timeout, const Hardware& hardware);
} // namespace bc::serving::session::vram
