/* bc_serving_engine.cpp -- what the engine keeps of the owner's admission (BigCherry managed patch rad0070, RSC02): the
 * expected memory plan and KV floor rad0011 checks, the ordered rank BDFs, the typed engine/remote settings plugins read
 * (bc_engine_settings.h) and the step-phase directory, all installed once by the managed flag step
 * (bc_serving_managed_take_flag), plus the engine's one entry: the stock `--config FILE` YAML entry, run after the
 * owner's flags are taken. Nothing here reads a manifest. */
#include "bc_serving_engine.h"
#include "bc_engine_settings.h"
#include "bc_serving_managed.h"
#include "engine_admission.h"
#include "serving_codec.h"

#ifdef BC_RAD_NATIVE_FLEET_CONFIG
#include "native_fleet_config.h"
#include "bc_plan_memory.h"
#include <iostream>
#endif

#include <nlohmann/json.hpp>

#include <string>
#include <vector>

namespace v2 = bc::serving::v2;

namespace {
v2::EnginePlan g_expected;          /* a serving start's expected plan, from the owner's admission */
int64_t g_floor_ppm = 0;
bool g_have_expected = false;
std::vector<std::string> g_engine_bdfs;
BcEngineSettings g_engine_settings{};
std::string g_shm;
std::string g_histogram;
std::string g_heat_launch_nonce;
std::string g_step_phase_dir; // owner-only serving diagnostic; never part of the plan
}  // namespace

extern "C" const char* bc_serving_step_phase_dir() {
    return g_step_phase_dir.empty() ? nullptr : g_step_phase_dir.c_str();
}

extern "C" int bc_serving_engine_settings(BcEngineSettings* out) {
    if (!out || out->struct_size != sizeof(BcEngineSettings) || !g_engine_settings.manifest) return 0;
    *out = g_engine_settings;
    // The normal native owner turns on existing per-helper request stamps only
    // when the ephemeral phase trace has been explicitly selected.
    if (!g_step_phase_dir.empty()) out->mailbox_trace_every = 1;
    return 1;
}

// Engine-core remote policy consumed by rad0002's declaration and placement
// gates; never uses environment variables, including before plugin registration.
extern "C" int bc_serving_remote_settings(int64_t* rows, uint32_t* skip_owned,
                                          uint32_t* shadow, uint32_t* mask_only) {
    if (!rows || !skip_owned || !shadow || !mask_only || !g_engine_settings.manifest) return 0;
    *rows = g_engine_settings.handoff ? g_engine_settings.max_rows : 0;
    *skip_owned = g_engine_settings.skip_owned;
    *shadow = g_engine_settings.shadow;
    *mask_only = g_engine_settings.mask_only;
    return 1;
}

void bc_serving_install_engine_state(const bc::serving::EngineAdmission& a, const std::string& heat_nonce,
                                     const std::string& step_phase_dir) {
    g_have_expected = false;
    g_floor_ppm = a.kv_backing_floor_ppm;
    g_engine_bdfs = a.bdfs;
    g_step_phase_dir = step_phase_dir;
    g_heat_launch_nonce = heat_nonce;
    if (a.expected) {
        v2::EnginePlan plan;   // already decoded once by the admission reader
        if (!v2::decode(*a.expected, "", &plan)) {
            g_expected = std::move(plan);
            g_have_expected = true;
        }
    }
    BcEngineSettings settings{};
    settings.manifest = 1;   // "typed settings present": the DSO getter's marker, not a manifest
    if (a.handoff) {
        settings.handoff = 1;
        settings.max_rows = a.max_rows;
        settings.skip_owned = a.skip_owned ? 1u : 0u;
        g_shm = a.shm;
        settings.shm = g_shm.c_str();
        if (a.histogram_file) {
            g_histogram = *a.histogram_file;
            settings.histogram_file = g_histogram.c_str();
        }
        settings.histogram_period_s = a.histogram_period_s;
        settings.combine = a.combine ? 1u : 0u;
    }
    settings.moe_native = a.moe_native ? 1u : 0u;
    settings.moe_fp16 = a.moe_fp16 ? 1u : 0u;
    settings.moe_ahalf = a.moe_ahalf ? 1u : 0u;
    settings.gemv = a.gemv ? 1u : 0u;
    settings.dec11 = a.dec11 ? 1u : 0u;
    settings.fuse_stage = a.fuse_stage ? 1u : 0u;
    settings.attn_native = a.attn_native ? 1u : 0u;
    settings.all_reduce_wire4 = a.all_reduce_wire4 ? 1u : 0u;
    settings.shadow = a.shadow ? 1u : 0u;
    settings.mask_only = a.mask_only;
    settings.mailbox_trace_every = a.mailbox_trace_every;
    if (!g_heat_launch_nonce.empty()) settings.heat_launch_nonce = g_heat_launch_nonce.c_str();
    g_engine_settings = settings;
}

bool bc_serving_expected_plan(v2::EnginePlan* expected, int64_t* floor_ppm) {
    if (!g_have_expected) return false;
    *expected = g_expected;
    *floor_ppm = g_floor_ppm;
    return true;
}

int64_t bc_serving_kv_backing_floor_ppm() { return g_floor_ppm; }

std::vector<std::string> bc_serving_engine_bdfs() { return g_engine_bdfs; }

// The stock YAML entry is kept in upstream core/main.cpp as a callable. The owner's flags are taken first (they install
// what rad0011's flag step reads), then rad0011's, then the stock entry runs on what is left: `--config FILE`.
#ifdef BC_RAD_NATIVE_FLEET_CONFIG
int bc_serving_engine_entry(int argc, char** argv, const bc::serving::callback::Guarded<int(int, char**)>& run_stock) {
    if (const auto rc = bc::serving::inspect_native_fleet_config_command(argc, argv); rc.has_value())
        return *rc;
    if (int rc = bc_serving_managed_take_flag(&argc, &argv)) return rc;
    if (int rc = bc_plan_memory_take_flag(&argc, &argv)) return rc;
    const auto stock = run_stock.invoke("stock YAML entry", argc, argv);
    if (!stock) {
        std::cerr << "radiance: " << stock.error().message << "\n";
        return 2;
    }
    return stock.value();
}
#endif
