#pragma once
/* One helper process's settings: a flat, immutable projection of the manifest the shared v1 reader verified
 * (v1::load_manifest, then v1::helper_view: helper_common, the role's patch and the lab experiment's shape, spec §2.2).
 * Not a second schema: every value is read from a v1 field and nothing is defaulted here except the diagnostics that are
 * absent from a production manifest (their off state, which is the helper's unset behaviour). r3_moe_worker --manifest
 * FILE --role helperN loads it before any GPU work; a failure is a v1::ConfigError. */
#include "manifest.h"

#include <cstdint>
#include <filesystem>
#include <string>
#include <string_view>
#include <vector>

namespace bc::serving {

struct WorkerHelper {
    uint32_t count = 0, index = 0, engine_ranks = 0;
    bool cpu = false, power_save = true, queue_warm = false, combine = false;
    uint32_t combine_min_rows = 256, combine_threads = 8;
    bool cpu_set_auto = true;
};
struct WorkerHandoff {
    std::string shm, heat, histogram, placement;
    uint32_t max_rows = 0, small_rows = 0, gemv_large_rows = 0;
    uint32_t histogram_period_s = 0;
    uint32_t moe_down_tile = 0;
    bool skip_owned = true;
};
struct HelperSettings {
    WorkerHelper helper;
    WorkerHandoff handoff;
    std::string role, content_sha256, model, radiance_home;
    std::string plugin, plugin_file, formats;
    std::vector<uint64_t> budgets_mib, shares;
    std::vector<double> balance_speeds;
    std::vector<uint32_t> small_lanes;
    uint64_t budget_mib = 0;
    uint32_t gemv = 1, gemv_h2 = 2, moe_form = 0, moe_long_rows = 0;
    bool gemv_wide = true, gemv_w4h2 = true, gemv_mr = false;
    bool rows = false, rows_small = false, rows_kernel = false;
    bool fuseroute = true, vecio = true, sparse_act = true, lane_quant = true;
    bool native = true, fp16 = true, ahalf = true;
    bool balance = false;
    std::string capture, poison = "0", test_stop_at;
    uint32_t capture_every = 37, capture_max = 16, trace = 0;
    uint32_t moe_probe = 0;
    bool refusal_trace = false;
    bool sweep_fast = false;
};

/* The settings of the helper with `role`, from a manifest that read_manifest accepted. BC_CFG_UNRESOLVED_REFERENCE when
 * no helper has the role. */
v1::Result<HelperSettings> helper_settings(const v1::Manifest& m, std::string_view role);
/* load_manifest, then helper_settings. */
v1::Result<HelperSettings> load_helper_settings(const std::filesystem::path& path, std::string_view role);

}  // namespace bc::serving
