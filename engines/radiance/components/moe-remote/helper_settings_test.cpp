#ifdef NDEBUG
#undef NDEBUG
#endif
// Card-free: the helper's settings are a projection of what the shared strict v1 reader accepted. The two contract
// examples are real, digest-verified manifests, so every value below comes through the production read path.
#include "../serving/v2/helper_settings.h"
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>

#include <cassert>
#ifdef NDEBUG
#error "assert-based test built with NDEBUG: its assertions would not run"
#endif
#include <cstdio>
#include <iostream>

namespace {
using bc::serving::HelperSettings;
HelperSettings load(const char* example, const char* role) {
    const auto r = bc::serving::load_helper_settings(std::string(BC_CONTRACT_DIR) + "/" + example, role,
                                                  bc::serving::v2::PlanUse::serving);
    if (!r) std::cerr << bc::serving::v2::render_human("helper_settings_test", r.error());
    assert(r);
    return r.value();
}
}  // namespace


namespace {
using bc::serving::HelperSettings;
using Json = nlohmann::json;
using bc::serving::v2::ConfigErrorCode;

// Every member set to a value that is not its default, so a member the wire forgets cannot round-trip by accident.
HelperSettings populated() {
    HelperSettings s = load("example.flashnext-4gpu.json", "helper1");   // a real live_policy and real budgets
    s.measuring = true;
    s.helper = {4, 1, 2, true, false, true, true, 77, 5, false};
    s.handoff = {"/r3-serve-roundtrip", "/h/heat", "/h/hist", "tail", 123, 9, 33, 17, 2, false};
    s.model = "/m/model.rad";
    s.radiance_home = "/runtime";
    s.plan_cache_key = "key-1";
    s.layout = "layout-1";
    s.objective = "latency";
    s.lab_profile = true;
    s.plugin = "libx.so";
    s.plugin_file = "libr4d.so";
    s.formats = "w4nl64";
    s.budgets_mib = {1, 2, 3, 4};
    s.shares = {4, 3, 2, 1};
    s.budget_mib = 2;
    s.balance_speeds = {0.5, 2.25};
    s.small_lanes = {3, 6};
    s.gemv = 2; s.gemv_h2 = 8; s.moe_form = 3; s.moe_long_rows = 11;
    s.gemv_wide = false; s.gemv_w4h2 = false; s.gemv_mr = true;
    s.rows = true; s.rows_small = true; s.rows_kernel = true;
    s.fuseroute = false; s.vecio = false; s.sparse_act = false; s.lane_quant = false;
    s.native = false; s.fp16 = false; s.ahalf = false; s.balance = true;
    s.capture = "/c"; s.poison = "one"; s.test_stop_at = "stop";
    s.capture_every = 5; s.capture_max = 6; s.trace = 7; s.moe_probe = 8;
    s.refusal_trace = true; s.sweep_fast = true;
    assert(s.live_policy);   // from the example
    s.live_calibration = bc::serving::v2::LiveCalibration{3, 500, 9000};
    bc::serving::v2::ForcedChurn forced;
    forced.repeat = 2;
    forced.pace_requests = 64;
    bc::serving::v2::ForcedStep step;
    step.kind = bc::serving::v2::ForcedKind::kMove;
    step.a_layer = 1; step.a_expert = 2; step.destination = 0;
    forced.steps = {step};
    s.forced = forced;
    s.epoch_cycle_s = 30;
    return s;
}

void expect_refused(const Json& j, ConfigErrorCode code, const char* field) {
    const auto r = bc::serving::helper_settings_from_json(j);
    assert(!r && r.error().code == code);
    assert(r.error().path.find(field) != std::string::npos);
}

void wire_tests() {
    const HelperSettings s = populated();
    // The size is a tripwire: a member added to HelperSettings changes it, and this test then asks for the wire
    // (helper_settings_to_json / helper_settings_from_json) and for populated() above to learn the new member.
    static_assert(sizeof(HelperSettings) == 992, "HelperSettings changed: extend the wire and populated()");
    const Json j = bc::serving::helper_settings_to_json(s);
    const auto back = bc::serving::helper_settings_from_json(j);
    if (!back) std::cerr << bc::serving::v2::render_human("wire", back.error());
    assert(back);
    assert(bc::serving::helper_settings_to_json(back.value()) == j);   // nothing dropped, nothing invented
    assert(back.value().measuring && back.value().live_calibration && back.value().forced && back.value().live_policy);

    Json unknown = j; unknown["surprise"] = 1;
    expect_refused(unknown, ConfigErrorCode::unknown_field, "surprise");
    Json inner = j; inner["handoff"]["extra"] = true;
    expect_refused(inner, ConfigErrorCode::unknown_field, "handoff.extra");
    Json missing = j; missing.erase("budgets_mib");
    expect_refused(missing, ConfigErrorCode::missing_field, "budgets_mib");
    Json inner_missing = j; inner_missing["helper"].erase("index");
    expect_refused(inner_missing, ConfigErrorCode::missing_field, "helper.index");
    Json wrong = j; wrong["gemv"] = "two";
    expect_refused(wrong, ConfigErrorCode::invalid_type, "gemv");
    Json wrong_inner = j; wrong_inner["handoff"]["skip_owned"] = 1;
    expect_refused(wrong_inner, ConfigErrorCode::invalid_type, "handoff.skip_owned");
    Json negative = j; negative["capture_max"] = -1;
    expect_refused(negative, ConfigErrorCode::invalid_type, "capture_max");

    // what the process was started as must agree with what the owner promised, before any device is touched
    Json bad_role = j; bad_role["role"] = "engine";
    expect_refused(bad_role, ConfigErrorCode::invalid_value, "role");
    Json other_role = j; other_role["role"] = "helper2";
    expect_refused(other_role, ConfigErrorCode::ownership_mismatch, "helper");
    Json past_count = j; past_count["helper"]["index"] = 4u; past_count["role"] = "helper4";
    expect_refused(past_count, ConfigErrorCode::ownership_mismatch, "helper");
    Json budgets = j; budgets["budget_mib"] = 99u;
    expect_refused(budgets, ConfigErrorCode::ownership_mismatch, "budgets_mib");
    Json segment = j; segment["handoff"]["shm"] = "no-slash";
    expect_refused(segment, ConfigErrorCode::invalid_value, "shm");

    // the descriptor: a sealed memfd is read, an unsealed one is refused
    auto memfd = [](const std::string& bytes, bool seal) {
        const int fd = ::memfd_create("settings-test", MFD_CLOEXEC | MFD_ALLOW_SEALING);
        assert(fd >= 3 && ::write(fd, bytes.data(), bytes.size()) == static_cast<ssize_t>(bytes.size()));
        if (seal) assert(::fcntl(fd, F_ADD_SEALS, F_SEAL_WRITE | F_SEAL_GROW | F_SEAL_SHRINK | F_SEAL_SEAL) == 0);
        return fd;
    };
    const int sealed = memfd(j.dump(), true);
    const auto read = bc::serving::read_helper_settings(sealed);
    if (!read) std::cerr << bc::serving::v2::render_human("sealed", read.error());
    assert(read && bc::serving::helper_settings_to_json(read.value()) == j);
    ::close(sealed);
    const int open_fd = memfd(j.dump(), false);
    assert(!bc::serving::read_helper_settings(open_fd));
    ::close(open_fd);
    const int junk = memfd("{not json", true);
    assert(!bc::serving::read_helper_settings(junk));
    ::close(junk);
}
}  // namespace

int main() {
    {   // production example: three helpers, helper1 patches gemv.h2, helper2 is the rows-kernel tail card
        const HelperSettings h0 = load("example.flashnext-4gpu.json", "helper0");
        assert(h0.role == "helper0" && !h0.measuring);
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
            std::string(BC_CONTRACT_DIR) + "/example.flashnext-4gpu.json", "helper7", bc::serving::v2::PlanUse::serving);
        assert(!r && r.error().code == bc::serving::v2::ConfigErrorCode::unresolved_reference);
        const auto missing = bc::serving::load_helper_settings("/nonexistent/manifest.json", "helper0",
                                                                 bc::serving::v2::PlanUse::serving);
        assert(!missing && missing.error().code == bc::serving::v2::ConfigErrorCode::io_failure);
    }
    wire_tests();
    std::puts("helper_settings_test: ok");
    return 0;
}
