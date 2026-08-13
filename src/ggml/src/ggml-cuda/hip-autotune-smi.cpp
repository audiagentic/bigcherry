// See hip-autotune-smi.h for why this is dlopen'd rather than linked.
#include "hip-autotune-smi.h"

#include <cstdlib>
#include <cstring>

#if defined(__linux__)

#include <dlfcn.h>

#include <algorithm>
#include <mutex>
#include <vector>

#include <hip/hip_runtime.h>

#include "ggml.h"   // ggml_time_us, for the overhead measurement below

namespace {

// ABI constants, taken from /opt/rocm/include/rocm_smi/rocm_smi.h and verified
// against ROCm 7.2.4 rather than assumed. They must be spelled out because the
// header is deliberately not included: including it would not create a link
// dependency, but it would make the build fail on machines without the ROCm SMI
// development package, which is exactly the portability this file exists to
// avoid.
constexpr int      RSMI_STATUS_SUCCESS_    = 0;
constexpr uint32_t RSMI_CLK_TYPE_SYS_      = 0;
constexpr uint32_t RSMI_CLK_TYPE_MEM_      = 4;
constexpr uint32_t RSMI_TEMP_TYPE_EDGE_    = 0;
constexpr uint32_t RSMI_TEMP_TYPE_JUNCTION_ = 1;
constexpr uint32_t RSMI_TEMP_CURRENT_      = 0;
constexpr uint32_t RSMI_MAX_NUM_FREQUENCIES_ = 33;

// Layout of rsmi_frequencies_t. `current` is an INDEX into `frequency[]`, not a
// frequency -- reading it as a value is the obvious mistake here and would
// yield a plausible small integer rather than an obvious error.
struct rsmi_frequencies_compat {
    bool     has_deep_sleep;
    uint32_t num_supported;
    uint32_t current;
    uint64_t frequency[RSMI_MAX_NUM_FREQUENCIES_];
};

struct Rsmi {
    void * handle = nullptr;
    bool   ready  = false;

    int (*init)(uint64_t)                              = nullptr;
    int (*num_monitor_devices)(uint32_t *)             = nullptr;
    int (*pci_id_get)(uint32_t, uint64_t *)            = nullptr;
    int (*gpu_clk_freq_get)(uint32_t, uint32_t, void *) = nullptr;
    int (*temp_metric_get)(uint32_t, uint32_t, uint32_t, int64_t *) = nullptr;
    int (*socket_power_get)(uint32_t, uint64_t *)      = nullptr;
    int (*busy_percent_get)(uint32_t, uint32_t *)      = nullptr;
};

Rsmi & rsmi() {
    static Rsmi r;
    static std::once_flag once;
    std::call_once(once, [] {
        // The versioned soname, not the bare .so: the unversioned link is part
        // of the -dev package and is frequently absent on a runtime-only host.
        r.handle = dlopen("librocm_smi64.so.1", RTLD_LAZY | RTLD_LOCAL);
        if (r.handle == nullptr) {
            return;
        }
        auto sym = [](void * h, const char * n) { return dlsym(h, n); };
        r.init                = (decltype(r.init))                sym(r.handle, "rsmi_init");
        r.num_monitor_devices = (decltype(r.num_monitor_devices)) sym(r.handle, "rsmi_num_monitor_devices");
        r.pci_id_get          = (decltype(r.pci_id_get))          sym(r.handle, "rsmi_dev_pci_id_get");
        r.gpu_clk_freq_get    = (decltype(r.gpu_clk_freq_get))    sym(r.handle, "rsmi_dev_gpu_clk_freq_get");
        r.temp_metric_get     = (decltype(r.temp_metric_get))     sym(r.handle, "rsmi_dev_temp_metric_get");
        r.socket_power_get    = (decltype(r.socket_power_get))    sym(r.handle, "rsmi_dev_current_socket_power_get");
        r.busy_percent_get    = (decltype(r.busy_percent_get))    sym(r.handle, "rsmi_dev_busy_percent_get");

        // All-or-nothing. A partially resolved struct would emit some fields as
        // real readings and the rest as zeros, and nothing downstream could tell
        // the two apart -- the same class of silently-wrong data that sank the
        // first VRAM attempt.
        const bool all = r.init && r.num_monitor_devices && r.pci_id_get &&
                         r.gpu_clk_freq_get && r.temp_metric_get &&
                         r.socket_power_get && r.busy_percent_get;
        r.ready = all && r.init(0) == RSMI_STATUS_SUCCESS_;
    });
    return r;
}

// RSMI indices enumerate every AMD device the kernel driver knows about, in
// driver order, and they ignore HIP_VISIBLE_DEVICES -- which every campaign
// phase sets. The two orderings therefore diverge in exactly the configuration
// this is most likely to run in. Match on PCI BDF or report nothing; a fallback
// to index 0 would silently report GPU0's clocks for all four cards.
int rsmi_index_for_hip_device(int hip_device) {
    hipDeviceProp_t prop{};
    if (hipGetDeviceProperties(&prop, hip_device) != hipSuccess) {
        return -1;
    }
    uint32_t count = 0;
    if (rsmi().num_monitor_devices(&count) != RSMI_STATUS_SUCCESS_) {
        return -1;
    }
    for (uint32_t i = 0; i < count; ++i) {
        uint64_t bdfid = 0;
        if (rsmi().pci_id_get(i, &bdfid) != RSMI_STATUS_SUCCESS_) {
            continue;
        }
        // BDFID layout (rocm_smi.h): domain 63:32, bus 15:8, device 7:3,
        // function 2:0.
        const int domain = (int) (uint32_t) (bdfid >> 32);
        const int bus    = (int) (uint32_t) ((bdfid >> 8) & 0xffu);
        const int dev    = (int) (uint32_t) ((bdfid >> 3) & 0x1fu);
        if (domain == prop.pciDomainID && bus == prop.pciBusID &&
                dev == prop.pciDeviceID) {
            return (int) i;
        }
    }
    return -1;
}

uint64_t current_clock_mhz(uint32_t index, uint32_t clk_type) {
    rsmi_frequencies_compat freq{};
    if (rsmi().gpu_clk_freq_get(index, clk_type, &freq) != RSMI_STATUS_SUCCESS_) {
        return 0;
    }
    if (freq.num_supported == 0 || freq.current >= freq.num_supported ||
            freq.num_supported > RSMI_MAX_NUM_FREQUENCIES_) {
        return 0;
    }
    return freq.frequency[freq.current] / 1000000ull;  // Hz -> MHz
}

}  // namespace

ggml_hip_device_state ggml_hip_query_device_state(int hip_device) {
    ggml_hip_device_state out;
    if (!rsmi().ready) {
        return out;
    }
    const int index = rsmi_index_for_hip_device(hip_device);
    if (index < 0) {
        return out;
    }
    const uint32_t dv = (uint32_t) index;

    hipDeviceProp_t prop{};
    if (hipGetDeviceProperties(&prop, hip_device) != hipSuccess) {
        return out;
    }
    out.hip_device = hip_device;
    out.pci_domain = (uint32_t) prop.pciDomainID;
    out.pci_bus = (uint32_t) prop.pciBusID;
    out.pci_device = (uint32_t) prop.pciDeviceID;
    out.identity_valid = true;

    out.sclk_mhz = current_clock_mhz(dv, RSMI_CLK_TYPE_SYS_);
    out.mclk_mhz = current_clock_mhz(dv, RSMI_CLK_TYPE_MEM_);

    int64_t temp = 0;
    if (rsmi().temp_metric_get(dv, RSMI_TEMP_TYPE_EDGE_, RSMI_TEMP_CURRENT_,
                               &temp) == RSMI_STATUS_SUCCESS_) {
        out.edge_temp_mc = (uint64_t) std::max<int64_t>(0, temp);
    }
    temp = 0;
    if (rsmi().temp_metric_get(dv, RSMI_TEMP_TYPE_JUNCTION_, RSMI_TEMP_CURRENT_,
                               &temp) == RSMI_STATUS_SUCCESS_) {
        out.junction_temp_mc = (uint64_t) std::max<int64_t>(0, temp);
    }

    uint64_t power = 0;
    if (rsmi().socket_power_get(dv, &power) == RSMI_STATUS_SUCCESS_) {
        out.socket_power_uw = power;
    }
    uint32_t busy = 0;
    if (rsmi().busy_percent_get(dv, &busy) == RSMI_STATUS_SUCCESS_) {
        out.busy_percent = busy;
    }

    // Valid means "this really is the requested device's state". Individual
    // metrics may still be 0 where the ASIC does not expose them; that is a
    // different claim from "we could not read this device at all".
    out.valid = true;
    return out;
}

double ggml_hip_smi_measure_overhead_us(int hip_device, int iterations) {
    if (!rsmi().ready || iterations < 1) {
        return -1.0;
    }
    std::vector<double> samples;
    samples.reserve((size_t) iterations);
    for (int i = 0; i < iterations; ++i) {
        const int64_t t0 = ggml_time_us();
        const ggml_hip_device_state s = ggml_hip_query_device_state(hip_device);
        const int64_t t1 = ggml_time_us();
        if (!s.valid) {
            return -1.0;
        }
        samples.push_back((double) (t1 - t0));
    }
    std::sort(samples.begin(), samples.end());
    return samples[samples.size() / 2];
}

#else  // !__linux__

ggml_hip_device_state ggml_hip_query_device_state(int) {
    return ggml_hip_device_state{};
}

double ggml_hip_smi_measure_overhead_us(int, int) {
    return -1.0;
}

#endif  // __linux__

bool ggml_hip_smi_enabled() {
    static const bool enabled = [] {
        const char * value = getenv("GGML_HIP_TUNE_SMI");
        return value != nullptr && strcmp(value, "0") != 0 && value[0] != '\0';
    }();
    return enabled;
}
