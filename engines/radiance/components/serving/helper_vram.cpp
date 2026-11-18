#include "helper_vram.h"
#include <algorithm>
#include <charconv>
#include <chrono>
#include <fstream>
#include <limits>
#include <sstream>
#include <thread>
#include <utility>
namespace bc::serving::session::vram {
namespace {
namespace fs = std::filesystem;
using Clock = std::chrono::steady_clock;
using namespace std::chrono_literals;
bool valid_bdf(const std::string& bdf) {
    if (bdf.size() != 12 || bdf[4] != ':' || bdf[7] != ':' || bdf[10] != '.' || bdf[11] < '0' || bdf[11] > '7')
        return false;
    for (size_t i : {0U, 1U, 2U, 3U, 5U, 6U, 8U, 9U}) {
        const char c = bdf[i];
        if (!((c >= '0' && c <= '9') || (c >= 'a' && c <= 'f') || (c >= 'A' && c <= 'F')))
            return false;
    }
    return true;
}
fs::path used_path(const fs::path& root, const std::string& bdf) {
    if (!valid_bdf(bdf))
        throw Refused("BC_CFG_ADMISSION_REFUSED (RAD_E_NOMEM): invalid measuring helper PCI BDF " + bdf);
    return root / "sys/bus/pci/devices" / bdf / "mem_info_vram_used";
}
uint64_t read_used(const fs::path& path, const Hardware& hardware) {
    if (hardware.read_used)
        return hardware.read_used.require("helper_vram.hardware.read", path);
    std::ifstream in(path);
    std::string token, extra;
    if (!(in >> token) || (in >> extra))
        throw std::runtime_error("missing or malformed unsigned byte counter");
    uint64_t value = 0;
    const auto parsed = std::from_chars(token.data(), token.data() + token.size(), value);
    if (parsed.ec != std::errc() || parsed.ptr != token.data() + token.size())
        throw std::runtime_error("invalid unsigned byte counter");
    return value;
}
uint64_t sample(const fs::path& root, const std::string& bdf, const Hardware& hardware) {
    const fs::path path = used_path(root, bdf);
    try {
        return read_used(path, hardware);
    } catch (const std::exception& e) {
        throw Refused("BC_CFG_ADMISSION_REFUSED (RAD_E_NOMEM): measuring helper " + bdf +
                      ": cannot read VRAM usage from " + path.string() + ": " + e.what());
    }
}
} // namespace
Baseline snapshot(const std::vector<std::string>& bdfs, const fs::path& root, const Hardware& hardware) {
    Baseline result;
    for (const auto& bdf : bdfs) {
        // Three pre-start observations bound the actual idle counter noise for this BDF.
        uint64_t low = sample(root, bdf, hardware);
        uint64_t high = low;
        for (int i = 0; i < 2; ++i) {
            const auto used = sample(root, bdf, hardware);
            low = std::min(low, used);
            high = std::max(high, used);
        }
        if (!result.emplace(bdf, Observation{high, high - low}).second)
            throw Refused("BC_CFG_ADMISSION_REFUSED (RAD_E_NOMEM): duplicate helper PCI BDF " + bdf);
    }
    return result;
}
void await_release(const Baseline& baseline, const fs::path& root, std::chrono::milliseconds stop_timeout,
                   const Hardware& hardware) {
    const auto deadline = Clock::now() + stop_timeout;
    while (true) {
        std::ostringstream outstanding;
        for (const auto& [bdf, observed] : baseline) {
            const auto used = sample(root, bdf, hardware);
            const uint64_t ceiling =
                observed.baseline_bytes > std::numeric_limits<uint64_t>::max() - observed.tolerance_bytes
                    ? std::numeric_limits<uint64_t>::max()
                    : observed.baseline_bytes + observed.tolerance_bytes;
            if (used > ceiling) {
                outstanding << " BDF " << bdf << " used=" << used << " baseline=" << observed.baseline_bytes
                            << " observed_noise=" << observed.tolerance_bytes << " ceiling=" << ceiling
                            << " bytes;";
            }
        }
        if (outstanding.str().empty())
            return;
        const auto now = Clock::now();
        if (now >= deadline)
            throw Refused("BC_CFG_ADMISSION_REFUSED (RAD_E_NOMEM): measuring helper VRAM not "
                          "released after verified child exit (timeout " +
                          std::to_string(stop_timeout.count()) + " ms):" + outstanding.str());
        const auto remaining = std::chrono::duration_cast<std::chrono::milliseconds>(deadline - now);
        const auto delay = std::min(100ms, remaining);
        if (hardware.wait) {
            if (!hardware.wait.require("helper_vram.hardware.wait", delay))
                throw Refused("BC_CFG_ADMISSION_REFUSED (RAD_E_NOMEM): stop requested while "
                              "waiting for measuring helper VRAM:" +
                              outstanding.str());
        } else {
            std::this_thread::sleep_for(delay);
        }
    }
}
} // namespace bc::serving::session::vram
