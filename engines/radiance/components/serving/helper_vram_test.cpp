// Card-free tests for the native session owner's measuring-helper VRAM barrier.
#ifdef NDEBUG
#undef NDEBUG
#endif
#include "helper_vram.h"
#include <cassert>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>
namespace fs = std::filesystem;
using namespace std::chrono_literals;
namespace hv = bc::serving::session::vram;
int main() {
    const fs::path root = fs::current_path() / "bc-helper-vram-fixture";
    fs::create_directories(root);
    const std::string bdf = "0000:03:00.0";
    const fs::path usage = root / "sys/bus/pci/devices" / bdf / "mem_info_vram_used";
    fs::create_directories(usage.parent_path());
    auto write = [&](uint64_t bytes) { std::ofstream(usage, std::ios::trunc) << bytes << "\n"; };
    const uint64_t mib = uint64_t{1024} * 1024;
    write(128 * mib);
    const auto baseline = hv::snapshot({bdf}, root, {});
    assert(baseline.at(bdf).baseline_bytes == 128 * mib);
    assert(baseline.at(bdf).tolerance_bytes == 0);
    hv::await_release(baseline, root, 10ms, {});
    write(256 * mib);
    int polls = 0;
    hv::Hardware later;
    later.wait = [&](std::chrono::milliseconds) {
        if (++polls == 2)
            write(128 * mib);
        return true;
    };
    hv::await_release(baseline, root, 100ms, later);
    assert(polls == 2);
    write(256 * mib);
    bool timed_out = false;
    try {
        hv::await_release(baseline, root, 2ms, {});
    } catch (const hv::Refused& e) {
        const std::string why = e.what();
        timed_out = why.find("BC_CFG_ADMISSION_REFUSED") != std::string::npos &&
                    why.find(bdf) != std::string::npos && why.find("268435456") != std::string::npos &&
                    why.find("134217728") != std::string::npos;
    }
    assert(timed_out);
    fs::remove(usage);
    bool unreadable_baseline = false;
    try {
        (void)hv::snapshot({bdf}, root, {});
    } catch (const hv::Refused& e) {
        unreadable_baseline = std::string(e.what()).find(bdf) != std::string::npos;
    }
    assert(unreadable_baseline);
    write(128 * mib);
    const auto prior = hv::snapshot({bdf}, root, {});
    fs::remove(usage);
    bool unreadable_after = false;
    try {
        hv::await_release(prior, root, 10ms, {});
    } catch (const hv::Refused& e) {
        unreadable_after = std::string(e.what()).find(bdf) != std::string::npos;
    }
    assert(unreadable_after);
    // Noise is observed, not a hand-picked MiB allowance.
    int samples = 0;
    hv::Hardware noisy;
    noisy.read_used = [&](const fs::path&) { return (128 + (samples++ % 2)) * mib; };
    const auto observed = hv::snapshot({bdf}, root, noisy);
    assert(observed.at(bdf).tolerance_bytes == mib);
    bool invalid_bdf = false;
    try {
        (void)hv::snapshot({"../proc"}, root, {});
    } catch (const hv::Refused&) {
        invalid_bdf = true;
    }
    assert(invalid_bdf);
    fs::remove_all(root);
}
