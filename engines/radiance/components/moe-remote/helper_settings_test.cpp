#ifdef NDEBUG
#undef NDEBUG
#endif
// Card-free: the helper's settings are a projection of what the shared strict v1 reader accepted. The two contract
// examples are real, digest-verified manifests, so every value below comes through the production read path.
#include "../serving/v1/helper_settings.h"
#include <cassert>
#include <cstdio>
#include <iostream>

namespace {
using bc::serving::HelperSettings;
HelperSettings load(const char* example, const char* role) {
    const auto r = bc::serving::load_helper_settings(std::string(BC_CONTRACT_DIR) + "/" + example, role,
                                                  bc::serving::v1::PlanUse::serving);
    if (!r) std::cerr << bc::serving::v1::render_human("helper_settings_test", r.error());
    assert(r);
    return r.value();
}
}  // namespace

int main() {
    {   // production example: three helpers, helper1 patches gemv.h2, helper2 is the rows-kernel tail card
        const HelperSettings h0 = load("example.flashnext-4gpu.json", "helper0");
        assert(h0.role == "helper0" && h0.content_sha256.rfind("sha256:", 0) == 0);
        assert(h0.helper.count == 3 && h0.helper.index == 0 && h0.helper.engine_ranks == 1);
        assert(h0.helper.combine && h0.helper.combine_min_rows == 256 && h0.helper.combine_threads == 8);
        assert(h0.helper.cpu_set_auto && h0.helper.power_save && !h0.helper.queue_warm && !h0.helper.cpu);
        assert(h0.handoff.shm == "/r3-moe-helper-example-001" && h0.handoff.max_rows == 4096 && h0.handoff.small_rows == 8);
        assert(h0.handoff.gemv_large_rows == 32 && h0.handoff.moe_down_tile == 1 && h0.handoff.skip_owned);
        assert(h0.handoff.placement == "tail" && h0.handoff.heat == "/models/flash-next.heat");
        assert(h0.handoff.histogram == "/models/flash-next.hist" && h0.handoff.histogram_period_s == 60);
        assert(h0.model == "/models/flash-next.rad");
        assert(h0.plugin_file == "libr3.so" && h0.formats == "w4nl64,fp8_e4m3_block" && h0.plugin.empty());
        assert((h0.budgets_mib == std::vector<uint64_t>{22528, 22528, 14336}) && h0.budget_mib == 22528);
        assert((h0.shares == std::vector<uint64_t>{1, 1, 0}));
        assert(h0.gemv == 1 && h0.gemv_h2 == 2 && h0.gemv_wide && h0.gemv_w4h2 && !h0.gemv_mr);
        assert(!h0.rows && !h0.rows_kernel && !h0.balance && h0.small_lanes.empty());
        assert(h0.native && h0.fp16 && h0.ahalf && h0.lane_quant && h0.vecio && h0.sparse_act && h0.fuseroute);
        // a production manifest carries no diagnostics: all off
        assert(h0.capture.empty() && h0.poison == "0" && h0.test_stop_at.empty() && h0.trace == 0 && h0.moe_probe == 0);
        assert(!h0.refusal_trace && !h0.sweep_fast);

        assert(load("example.flashnext-4gpu.json", "helper1").gemv_h2 == 4);
        assert(load("example.flashnext-4gpu.json", "helper1").helper.index == 1);
        const HelperSettings h2 = load("example.flashnext-4gpu.json", "helper2");
        assert(h2.rows && h2.plugin_file == "libr2.so" && h2.formats == "w4nl64" && h2.budget_mib == 14336);
    }
    {   // lab example: experiment shape overlays (common, then helper2) and typed diagnostics
        const HelperSettings h0 = load("example.lab-experiment.json", "helper0");
        assert(h0.gemv_h2 == 4 && !h0.lane_quant && h0.gemv == 1);
        const HelperSettings h2 = load("example.lab-experiment.json", "helper2");
        assert(h2.gemv == 0 && h2.gemv_h2 == 4 && (h2.small_lanes == std::vector<uint32_t>{4, 8}) && h2.rows);
        assert(h2.moe_probe == 1 && h2.refusal_trace && h2.trace == 2 && h2.sweep_fast);
        assert(h2.capture == "/tmp/bigcherry/capture.lab" && h2.capture_every == 37 && h2.capture_max == 16);
        assert(h2.poison == "0");
    }
    {   // a role the manifest does not define is refused by name, before any GPU work
        const auto r = bc::serving::load_helper_settings(
            std::string(BC_CONTRACT_DIR) + "/example.flashnext-4gpu.json", "helper7", bc::serving::v1::PlanUse::serving);
        assert(!r && r.error().code == bc::serving::v1::ConfigErrorCode::unresolved_reference);
        const auto missing = bc::serving::load_helper_settings("/nonexistent/manifest.json", "helper0",
                                                                 bc::serving::v1::PlanUse::serving);
        assert(!missing && missing.error().code == bc::serving::v1::ConfigErrorCode::io_failure);
    }
    std::puts("helper_settings_test: ok");
    return 0;
}
