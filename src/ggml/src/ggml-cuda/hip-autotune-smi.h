// Device clock / power / thermal state, for falsification (HI52 part 2).
//
// This is NOT a ranking axis and must never become one. Its purpose is to
// answer a question the tuner currently cannot: "was this GPU throttled, or
// running at a different clock, during the window in which these candidates
// were compared?" HI46 lists `device_clock_power_state` in its
// `unchecked_dimensions` precisely because a drift check that cannot see clock
// state has no way to tell "the winner changed because the hardware got slower"
// from "the winner changed because the code changed".
//
// Loaded with dlopen, never linked. A hard dependency on librocm_smi64 would
// make the whole HIP backend unloadable anywhere the library is absent --
// including every Windows build, where it does not exist at all. Every accessor
// degrades to `valid == false` rather than failing.
#pragma once

#include <cstdint>

struct ggml_hip_device_state {
    bool     valid            = false;
    bool     identity_valid   = false;
    int      hip_device       = -1;
    uint32_t pci_domain       = 0;
    uint32_t pci_bus          = 0;
    uint32_t pci_device       = 0;

    // `valid` covers device identity only (library loaded, symbols resolved,
    // PCI-BDF matched) -- it does NOT guarantee any individual metric below
    // was actually read. Each metric has its own `*_valid` flag because RSMI
    // calls fail independently (e.g. clocks readable, power unsupported on
    // this ASIC); a metric whose `*_valid` is false must be treated as
    // absent, never as a genuine zero reading. See device_state_json() in
    // hip-autotune-tuner.cu, which must serialise an invalid metric as null,
    // not as the numeric default below.
    bool     sclk_valid          = false;
    uint64_t sclk_mhz            = 0;   // shader/system clock, current
    bool     mclk_valid          = false;
    uint64_t mclk_mhz            = 0;   // memory clock, current
    bool     edge_temp_valid     = false;
    uint64_t edge_temp_mc        = 0;   // millidegrees C
    bool     junction_temp_valid = false;
    uint64_t junction_temp_mc    = 0;   // millidegrees C (hotspot)
    bool     power_valid         = false;
    uint64_t socket_power_uw     = 0;   // microwatts
    bool     busy_valid          = false;
    uint32_t busy_percent        = 0;
};

// Returns valid == false on: non-Linux, library absent, any symbol missing,
// rsmi_init failure, an RSMI error, or -- critically -- no PCI-BDF match for
// the requested HIP device. Never guesses a device index. `identity_valid` is
// separate from `valid`: telemetry must not be compared across unknown GPUs.
ggml_hip_device_state ggml_hip_query_device_state(int hip_device);

// True when the capture is both compiled in and enabled by the environment.
// Off by default (GGML_HIP_TUNE_SMI=1 to enable) until its per-call overhead is
// measured and published in HI52 -- choosing a sampling cadence before knowing
// the cost would be picking a number out of the air.
bool ggml_hip_smi_enabled();

// Median wall-clock microseconds for one ggml_hip_query_device_state call, over
// `iterations` calls. Exists so the overhead question is answered with a
// measurement rather than an assumption; returns -1.0 when unavailable.
double ggml_hip_smi_measure_overhead_us(int hip_device, int iterations);
