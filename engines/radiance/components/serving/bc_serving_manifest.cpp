/* bc_serving_manifest.cpp -- policy behind bc_serving_manifest.h: a serving manifest v1, read by the shared strict reader,
 * to radiance's own flags through the generated registry, and the report source label. */
#include "bc_serving_manifest.h"
#include "bc_engine_settings.h"
#include "bc_serving_managed.h"
#include "engine_admission.h"
#include "serving_codec.h"

#include "document_bytes.h"
#include "manifest.h"
#include "serving_types.h"
#ifdef BC_RAD_NATIVE_FLEET_CONFIG
#include "native_fleet_config.h"
#include "fleet/fleet.h"
#include "engine.h"
#include "server/front.h"
#include "util/stallwatch.h"
#include "bc_plan_memory.h"
#include "hf_model.h"
#include <iostream>
#include <atomic>
#include <chrono>
#include <csignal>
#include <thread>
#endif

#include <nlohmann/json.hpp>

#include <algorithm>
#include <charconv>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <limits>
#include <optional>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include <sys/stat.h>
#include <unistd.h>

namespace rad {
/* radiance's own entry point, declared here so this file needs no engine header; a signature that drifts fails
 * at link time. */
namespace diag {
std::vector<std::string> config_option_names();
}
}  // namespace rad

namespace {
namespace v2 = bc::serving::v2;

std::vector<std::string> g_store;   /* the synthetic argv, kept for the life of the process */
std::vector<char*> g_ptrs;
std::string g_label;                /* "manifest <12 hex>" once a manifest was loaded, else empty */
v2::EnginePlan g_expected;          /* a serving start's expected plan, from the manifest's measured plan */
int64_t g_floor_ppm = 0;
bool g_have_expected = false;
std::vector<std::string> g_engine_bdfs;
BcEngineSettings g_engine_settings{};
std::string g_shm;
std::string g_histogram;
std::string g_heat_launch_nonce;
std::string g_step_phase_dir; // owner-only serving diagnostic; never part of the immutable plan
struct NativeEngineInputs {
    std::string yaml_utf8;
    std::string sha256;
    std::string resolution_root;
    std::string model;
    std::string radiance_home;
    std::vector<std::string> kernels;
    std::vector<std::string> engine_bdfs;
    std::optional<std::vector<std::string>> rank_devices;
    int64_t tp = 0;
    int64_t port = 0;
    int64_t max_context = 0;
    int64_t max_sequences = 0;
    int64_t host_pool_mib_per_rank = 0;
    int64_t host_kv_mib = 0;
    int64_t host_reserve_mib = 0;
    std::optional<int64_t> checkpoint_slots;
    bool offline_quality = false;
};
std::optional<NativeEngineInputs> g_native_engine;

NativeEngineInputs native_engine_inputs(const v2::Spec& spec, const v2::NativeConfigView& snapshot) {
    NativeEngineInputs admitted;
    admitted.yaml_utf8.assign(snapshot.yaml_utf8);
    admitted.sha256.assign(snapshot.identity.sha256);
    admitted.resolution_root.assign(snapshot.identity.resolution_root);
    admitted.model = spec.intent.model;
    admitted.radiance_home = spec.processes.engine.runtime.radiance_home;
    admitted.kernels = spec.processes.engine.runtime.kernels;
    admitted.engine_bdfs = spec.processes.engine.device.gpus;
    admitted.rank_devices = spec.processes.engine.device.rank_devices;
    admitted.tp = spec.processes.engine.scheduler.tp;
    admitted.port = spec.processes.engine.server.port;
    admitted.max_context = spec.intent.service.max_context_tokens;
    admitted.max_sequences = spec.intent.service.max_sessions;
    admitted.host_pool_mib_per_rank = spec.resources.host.pool_mib;
    admitted.host_kv_mib = spec.resources.host.kv_mib;
    admitted.host_reserve_mib = spec.resources.host.reserve_mib;
    admitted.checkpoint_slots = spec.resources.host.checkpoint_slots;
    return admitted;
}

bool complete_native_engine_inputs(const NativeEngineInputs& input) {
    return input.tp > 0 && input.engine_bdfs.size() == static_cast<size_t>(input.tp) &&
           input.max_context > 0 && input.max_sequences > 0 && input.host_pool_mib_per_rank >= 0;
}

bool fail(const std::string& why) {
    std::fprintf(stderr, "radiance: --manifest: %s\n", why.c_str());
    return false;
}

std::string read_native_serving_manifest(std::string_view path, int& sealed_fd) {
    constexpr std::string_view prefix = "/proc/self/fd/";
    if (path.substr(0, prefix.size()) != prefix)
        throw std::invalid_argument("v3 serving requires an inherited sealed manifest descriptor");
    path.remove_prefix(prefix.size());
    int fd = -1;
    const auto [end, err] = std::from_chars(path.data(), path.data() + path.size(), fd);
    if (err != std::errc{} || end != path.data() + path.size())
        throw std::invalid_argument("v3 serving manifest descriptor is invalid");
    std::string text = v2::read_sealed_document(fd, "v3 serving manifest descriptor");
    sealed_fd = fd;
    return text;
}

bool matched_quality_owner(const std::string& home, int manifest_fd) {
    namespace fs = std::filesystem;
    const pid_t owner = ::getppid();
    if (owner <= 1) return false;
    const fs::path expected = fs::path(home) / "bin/bc-serve";
    const fs::path parent = fs::path("/proc") / std::to_string(owner);
    std::error_code error;
    if (!expected.is_absolute() || fs::canonical(expected, error) != expected || error)
        return false;
    if (fs::read_symlink(parent / "exe", error) != expected || error)
        return false;
    struct stat local{}, retained{}, runtime{}, executable{};
    return ::fstat(manifest_fd, &local) == 0 &&
           ::stat((parent / "fd" / std::to_string(manifest_fd)).c_str(), &retained) == 0 &&
           local.st_dev == retained.st_dev && local.st_ino == retained.st_ino &&
           ::lstat(expected.c_str(), &runtime) == 0 && S_ISREG(runtime.st_mode) &&
           ::stat((parent / "exe").c_str(), &executable) == 0 &&
           runtime.st_dev == executable.st_dev && runtime.st_ino == executable.st_ino &&
           ::getppid() == owner;
}

}  // namespace

bool bc_serving_engine_argv(int* argc, char*** argv) {
    g_native_engine.reset();
    g_engine_bdfs.clear();
    g_have_expected = false;   /* each call starts from nothing: a start without --manifest holds no plan */
    g_floor_ppm = 0;
    g_engine_settings = BcEngineSettings{};
    g_shm.clear();
    g_histogram.clear();
    g_heat_launch_nonce.clear();
    g_step_phase_dir.clear();
    bool manifest_mode = false;
    for (int i = 1; i < *argc; ++i)
        if (!std::strcmp((*argv)[i], "--manifest")) manifest_mode = true;
    if (!manifest_mode) return true;

    std::string path, role, plan_memory, rank_out, heat_nonce, step_phase_dir;
    std::vector<std::string> verbosity;
    /* Offline quality runs (the KL mode): which run to do and where its files are, not configuration. They name no manifest
     * field, so they pass through to config_parse unchanged beside the manifest's own flags. */
    static const std::set<std::string> kRunFlags = {"--kld-record", "--kld-corpus", "--kld-ref", "--kld-out"};
    std::vector<std::pair<std::string, std::string>> run_flags;
    for (int i = 1; i < *argc; ++i) {
        const std::string a = (*argv)[i];
        if (a == "--manifest") {
            if (!path.empty()) return fail("given twice");
            if (i + 1 >= *argc || !*(*argv)[i + 1]) return fail("needs a path");
            path = (*argv)[++i];
        } else if (a == "--role") {
            if (i + 1 >= *argc) return fail("--role needs an argument");
            role = (*argv)[++i];
            if (role != "engine") return fail("--role must be engine, not '" + role + "'");
        } else if (a == "--bc-heat-launch-nonce") {
            // The launch owner creates a one-time token after manifest planning.
            // It is never stored in the immutable manifest or forwarded to core.
            if (!heat_nonce.empty()) return fail("--bc-heat-launch-nonce given twice");
            if (i + 1 >= *argc || !*(*argv)[i + 1])
                return fail("--bc-heat-launch-nonce needs a value");
            heat_nonce = (*argv)[++i];
            if (heat_nonce.size() != 32 ||
                !std::all_of(heat_nonce.begin(), heat_nonce.end(), [](const char c) {
                    return (c >= '0' && c <= '9') || (c >= 'a' && c <= 'f');
                }))
                return fail("--bc-heat-launch-nonce must be exactly 128 bits of lowercase hex");
        } else if (a == "--bc-step-phase-dir") {
            if (!step_phase_dir.empty()) return fail("--bc-step-phase-dir given twice");
            if (i + 1 >= *argc || !*(*argv)[i + 1])
                return fail("--bc-step-phase-dir needs an absolute output directory");
            step_phase_dir = (*argv)[++i];
            if (!std::filesystem::path(step_phase_dir).is_absolute() ||
                !std::filesystem::is_directory(step_phase_dir) ||
                std::filesystem::is_symlink(std::filesystem::symlink_status(step_phase_dir)))
                return fail("--bc-step-phase-dir must name an existing safe absolute directory");
        } else if (a == "--plan-memory") {
            /* The measuring start: radiance writes its memory plan to FILE and stops (rad0011). It is the one flag that
             * lets the engine take an unmeasured manifest; every other start takes only a measured one. */
            if (!plan_memory.empty()) return fail("--plan-memory given twice");
            if (i + 1 >= *argc || !*(*argv)[i + 1]) return fail("--plan-memory needs a path");
            plan_memory = (*argv)[++i];
        } else if (a == "--plan-memory-ranks") {
            if (!rank_out.empty()) return fail("--plan-memory-ranks given twice");
            if (i + 1 >= *argc || !*(*argv)[i + 1]) return fail("--plan-memory-ranks needs a path");
            rank_out = (*argv)[++i];
        } else if (a == "-v" || a == "-vv") {
            verbosity.push_back(a);
        } else if (kRunFlags.count(a)) {
            if (i + 1 >= *argc || !*(*argv)[i + 1]) return fail(a + " needs a path");
            run_flags.emplace_back(a, (*argv)[++i]);
        } else {
            return fail("'" + a + "' cannot be combined with --manifest (only --role engine, --plan-memory FILE, launch-only --bc-heat-launch-nonce, -v, -vv and the quality-run flags may)");
        }
    }

    if (!step_phase_dir.empty() && (!plan_memory.empty() || !run_flags.empty() || role != "engine"))
        return fail("--bc-step-phase-dir is valid only for a managed serving engine");
    if (!rank_out.empty() && plan_memory.empty()) return fail("--plan-memory-ranks requires --plan-memory");
    if (!heat_nonce.empty() && (!plan_memory.empty() || !rank_out.empty() || !run_flags.empty()))
        return fail("--bc-heat-launch-nonce is valid only for serving, not measurement or quality runs");
    if (!run_flags.empty() && !plan_memory.empty())
        return fail("v3 native offline quality cannot be combined with memory measurement");
    std::string text;
    int sealed_fd = -1;
    if (plan_memory.empty()) {
        if (role != "engine") return fail("v3 serving requires --role engine");
        try {
            text = read_native_serving_manifest(path, sealed_fd);
        } catch (const std::exception& error) {
            return fail(error.what());
        }
    } else {
        std::ifstream in(path, std::ios::binary);
        if (!in) return fail("cannot read '" + path + "'");
        text.assign(std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>());
    }
    /* The shared strict reader: size, UTF-8, duplicate keys, integers, the contract identity, the content digest, the
     * generated structural decode, provenance and every semantic rule, with the same errors bc-serve gives. */
    const v2::Result<v2::Manifest> manifest =
        v2::read_manifest(text, path, plan_memory.empty() ? v2::PlanUse::serving : v2::PlanUse::measuring);
    if (!manifest.ok()) {
        std::fputs(v2::render_human("radiance", manifest.error()).c_str(), stderr);
        return false;
    }
    if (!heat_nonce.empty() &&
        (!manifest.value().spec.handoff || !manifest.value().spec.handoff->telemetry.histogram_file ||
         manifest.value().spec.handoff->telemetry.histogram_file->empty()))
        return fail("--bc-heat-launch-nonce requires an authenticated histogram publisher");
    g_step_phase_dir = step_phase_dir; // the strict v3 manifest was authenticated
    const nlohmann::json doc = nlohmann::json::parse(text, nullptr, false);
    if (doc.is_discarded()) return fail("'" + path + "' is not JSON");   /* unreachable after the strict reader */
    if (!plan_memory.empty() && rank_out.empty())
        return fail("v3 native measurement requires --plan-memory-ranks");
    const auto& spec = manifest.value().spec;
    // The measuring manifest already carries the selected calibration, even
    // though its expected engine plan is not populated until measurement.
    if (!plan_memory.empty())
        for (const v2::CalibrationEntry& entry : spec.calibration)
            if (entry.key == v2::CalibrationKey::kKvBackingFloorFractionPpm)
                g_floor_ppm = std::get<int64_t>(entry.value);
    if (!run_flags.empty() && (spec.handoff || !spec.processes.helpers.empty())) {
        const auto flag = [&](const char* name) {
            return std::count_if(run_flags.begin(), run_flags.end(), [&](const auto& arg) {
                return arg.first == name && std::filesystem::path(arg.second).is_absolute();
            }) == 1;
        };
        if (run_flags.size() != 2 || !flag("--kld-ref") || !flag("--kld-out") ||
            !matched_quality_owner(spec.processes.engine.runtime.radiance_home, sealed_fd))
            return fail("v3 native helper KL scoring requires its matched owner and retained manifest");
    }
    const auto view = v2::native_config_view(manifest.value());
    if (!view) return fail("v3 manifest has no admitted native YAML snapshot");
    auto admitted = native_engine_inputs(spec, *view);
    if (!complete_native_engine_inputs(admitted))
        return fail("v3 manifest has incomplete native engine admission inputs");
    admitted.offline_quality = !run_flags.empty();
    g_native_engine = std::move(admitted);
    g_have_expected = false;
    /* A quality run (the KL mode) scores the trunk's own rows and never serves: it holds no checkpoint slots and a smaller
     * static VRAM set, so the serving plan's admission check does not describe it and none is armed. */
    /* Rank identities belong to the manifest even for a measuring start. */
    if (!rank_out.empty()) {
        g_engine_bdfs = doc.at("spec").at("processes").at("engine")
                        .at("device").at("gpus").get<std::vector<std::string>>();
    }
    if (plan_memory.empty() && run_flags.empty()) {
        /* a serving start: the plan the manifest carries is the plan this engine must admit (rad0011 checks it) */
        const v2::Result<v2::EngineView> view = v2::engine_view(manifest.value());
        if (!view.ok()) {
            std::fputs(v2::render_human("radiance", view.error()).c_str(), stderr);
            return false;
        }
        g_expected = view.value().memory.expected;
        g_engine_bdfs = view.value().engine.device.gpus;
        g_floor_ppm = view.value().kv_backing_floor_ppm;
        g_have_expected = true;
    }

    g_store.clear();
    g_store.push_back((*argv)[0]);
    for (const std::string& v : verbosity) g_store.push_back(v);
    if (!plan_memory.empty()) {
        g_store.push_back("--plan-memory");
        g_store.push_back(plan_memory);
    }
    if (!rank_out.empty()) {
        g_store.push_back("--plan-memory-ranks");
        g_store.push_back(rank_out);
    }
    for (const auto& kv : run_flags) {
        g_store.push_back(kv.first);
        g_store.push_back(kv.second);
    }
    g_ptrs.clear();
    for (std::string& s : g_store) g_ptrs.push_back(&s[0]);
    g_ptrs.push_back(nullptr);
    *argc = (int)g_store.size();
    *argv = g_ptrs.data();
    // Fill one immutable typed snapshot from the strict reader's decoded manifest.
    // Dynamic plugins consume this via bc_serving_engine_settings, not getenv.
    BcEngineSettings settings{};
    settings.manifest = 1;
    if (spec.handoff) {
        const auto& h = *spec.handoff;
        settings.handoff = 1;
        settings.max_rows = h.transport.max_rows;
        settings.skip_owned = h.transport.skip_owned ? 1u : 0u;
        g_shm = h.transport.shm;
        settings.shm = g_shm.c_str();
        if (h.telemetry.histogram_file) {
            g_histogram = *h.telemetry.histogram_file;
            settings.histogram_file = g_histogram.c_str();
        }
        settings.histogram_period_s = h.telemetry.period_s;
        settings.combine = spec.processes.helper_common->combine.enabled ? 1u : 0u;
    }
    if (spec.processes.engine.kernels) {
        const auto& k = *spec.processes.engine.kernels;
        settings.moe_native = k.moe_native.value_or(true) ? 1u : 0u;
        settings.moe_fp16 = k.moe_fp16.value_or(true) ? 1u : 0u;
        settings.moe_ahalf = k.moe_ahalf.value_or(true) ? 1u : 0u;
        settings.gemv = k.gemv.value_or(false) ? 1u : 0u;
        settings.dec11 = k.dec11.value_or(false) ? 1u : 0u;
        settings.fuse_stage = k.fuse_stage.value_or(true) ? 1u : 0u;
        settings.attn_native = k.attn_native.value_or(true) ? 1u : 0u;
        settings.all_reduce_wire4 = k.all_reduce_wire4.value_or(false) ? 1u : 0u;
    }
    if (spec.experiment && spec.experiment->engine) {
        const auto& e = *spec.experiment->engine;
        settings.shadow = e.shadow.value_or(false) ? 1u : 0u;
        settings.mask_only = static_cast<uint32_t>(e.mask_only.value_or(0));
    }
    if (spec.experiment && spec.experiment->diagnostics &&
        spec.experiment->diagnostics->lanes &&
        spec.experiment->diagnostics->lanes->trace_level) {
        settings.mailbox_trace_every = static_cast<uint32_t>(
            *spec.experiment->diagnostics->lanes->trace_level);
    }
    if (!heat_nonce.empty()) {
        g_heat_launch_nonce = std::move(heat_nonce);
        settings.heat_launch_nonce = g_heat_launch_nonce.c_str();
    }
    g_engine_settings = settings;
    // The remaining non-migrated engine switches retain their original bridge
    // pending RSC11's separately-owned consumers (RAD_* and BC_GFX12_*).
    for (const bc::serving::v2::EnvBinding& b : bc::serving::v2::kEnvBindings) {
        if (b.env == "R3_MOE_NATIVE" || b.env == "R3_MOE_FP16" || b.env == "R3_MOE_AHALF" ||
            b.env == "R3_GEMV" || b.env == "R3_DEC11" || b.env == "R3_MOE_WORKER_FUSESTAGE" ||
            b.env == "R3_MOE_WORKER_MASK_ONLY" || b.env == "R3_ATTN_NATIVE" ||
            b.env == "R3_AR_WIRE4") continue;
        const nlohmann::json::json_pointer ptr{std::string(b.path)};
        if (!doc.contains(ptr)) continue;
        const nlohmann::json& v = doc.at(ptr);
        if (b.kind == "boolean" && v.is_boolean()) setenv(std::string(b.env).c_str(), v.get<bool>() ? "1" : "0", 1);
        else if ((b.kind == "integer" || b.kind == "int_enum") && v.is_number_integer())
            setenv(std::string(b.env).c_str(), std::to_string(v.get<long long>()).c_str(), 1);
    }
    const std::string digest = doc.at("metadata").at("content_digest").get<std::string>();   /* "sha256:<hex>" */
    g_label = "manifest " + digest.substr(digest.find(':') + 1, 12);
    return true;
}

namespace {
#ifdef BC_RAD_NATIVE_FLEET_CONFIG
const rad::fleet::FleetConfig k_default_fleet;
int64_t g_front_load_ms = k_default_fleet.load_timeout_ms;
int64_t g_front_drain_ms = k_default_fleet.drain_timeout_ms;
int64_t g_front_keepalive_ms = k_default_fleet.keepalive_ms;
std::vector<std::string> g_front_aliases;
static_assert(std::atomic<bool>::is_always_lock_free);
std::atomic<bool> g_front_stop{false};
void stop_managed_front(int) { g_front_stop.store(true, std::memory_order_relaxed); }
#endif

int prepare_native_engine_config(const NativeEngineInputs& admitted, rad::Config* out) {
    if (!out) return fail("native Config output is null"), -1;
#ifndef BC_RAD_NATIVE_FLEET_CONFIG
    fail("this Radiance source has no native fleet parser for v3 Config replay");
    return -1;
#else
    namespace fs = std::filesystem;
    const auto reject_unless = [](bool condition, const char* message) {
        if (condition) return false;
        std::fprintf(stderr, "radiance: --manifest: native Config: %s\n", message);
        return true;
    };
    const auto canonical_under_root = [&](const std::string& value) {
        fs::path path(value);
        if (path.is_relative()) path = fs::path(admitted.resolution_root) / path;
        std::error_code error;
        const fs::path canonical = fs::weakly_canonical(path, error);
        if (error) throw std::runtime_error("cannot resolve " + path.string());
        return canonical;
    };
    try {
        bc::serving::NativeConfigSnapshot snapshot = bc::serving::NativeConfigSnapshot::admitted(
            admitted.yaml_utf8, admitted.sha256, admitted.resolution_root);
        bc::serving::NativeFleetConfig native = bc::serving::NativeFleetConfig::replay(std::move(snapshot));
        const rad::fleet::FleetConfig& fleet = native.fleet();
        if (fleet.models.size() != 1) return fail("native Config requires exactly one model"), -1;
        rad::Config cfg = fleet.models.front().cfg;
        std::vector<int> expected_cards;
        expected_cards.reserve(admitted.engine_bdfs.size());
        for (size_t i = 0; i < admitted.engine_bdfs.size(); ++i)
            expected_cards.push_back(static_cast<int>(i));
        // rank-devices is a machine fact: a YAML that omits it takes the admitted ordered BDF selection (sealed in the
        // manifest); one that states it must match that selection exactly (checked below).
        if (cfg.given.find("--rank-devices") == cfg.given.end() && admitted.rank_devices) {
            cfg.rank_devices = *admitted.rank_devices;
            cfg.given.insert_or_assign("--rank-devices", "manifest spec.processes.engine.device.rank_devices");
        }
        if (reject_unless(fleet.cards == expected_cards, "YAML card ordinals differ from selected rank order") ||
            reject_unless(cfg.tp == admitted.tp, "YAML TP differs from admitted engine TP") ||
            reject_unless(cfg.host == "127.0.0.1", "only the managed loopback listener is supported") ||
            reject_unless(cfg.port == admitted.port, "YAML listener port differs from admitted port") ||
            reject_unless(cfg.max_ctx == admitted.max_context,
                          "YAML max-model-len differs from admitted context limit") ||
            reject_unless(cfg.max_seqs == admitted.max_sequences,
                          "YAML max-num-seqs differs from admitted session limit") ||
            reject_unless(cfg.kernel_hierarchy == admitted.kernels,
                          "YAML kernel hierarchy differs from admitted runtime kernels") ||
            reject_unless(cfg.rank_devices == admitted.rank_devices.value_or(std::vector<std::string>{}),
                          "YAML rank-devices differs from admitted ordered BDF selection") ||
            reject_unless(canonical_under_root(cfg.model) == canonical_under_root(admitted.model),
                          "YAML model differs from admitted local model") ||
            reject_unless(canonical_under_root(cfg.radiance_home) == canonical_under_root(admitted.radiance_home),
                          "YAML runtime home differs from matched build") ||
            reject_unless(!cfg.kld() && !cfg.calib() && !cfg.diagnose_mode() && !cfg.debug_graph,
                          "native YAML cannot select an offline or debug mode in a managed manifest"))
            return -1;
        const auto bind_host = [&](const char* option, int64_t& value, int64_t measured,
                                   const char* provenance, bool allow_auto = false) {
            const auto authored = cfg.given.find(option);
            if (authored != cfg.given.end() && !(allow_auto && value == -1)) {
                if (value == measured) return true;
                return fail(std::string("native YAML ") + option + " differs from " + provenance);
            }
            value = measured;
            cfg.given.insert_or_assign(option, provenance);
            return true;
        };
        // The manifest and native Config both use per-rank pool MiB. Only the
        // upstream argv/YAML boundary consumes aggregate pool MiB.
        if (!bind_host("--host-pool-mib", cfg.host_pool_mib, admitted.host_pool_mib_per_rank,
                       "manifest spec.resources.host.pool_mib"))
            return -1;
        if (!admitted.offline_quality) {
            // The offline scorer runs with its original model settings and
            // does not open the managed serving cache or checkpoint store.
            if (admitted.host_kv_mib > std::numeric_limits<int32_t>::max())
                return fail("admitted host KV tier exceeds native prefix-cache-mib range"), -1;
            if (!bind_host("--prefix-cache-mib", cfg.prefix_cache_mib, admitted.host_kv_mib,
                           "manifest spec.resources.host.kv_mib") ||
                !bind_host("--host-headroom-mib", cfg.host_headroom_mib, admitted.host_reserve_mib,
                           "manifest spec.resources.host.reserve_mib"))
                return -1;
            if (admitted.checkpoint_slots &&
                !bind_host("--checkpoint-slots", cfg.checkpoint_slots, *admitted.checkpoint_slots,
                           "manifest spec.resources.host.checkpoint_slots", true))
                return -1;
        }
        if (fleet.log_set) rad::log_set_level(fleet.log);
        // The manifest bridge already stripped every option except verbosity
        // and KL run paths. Use upstream's own CLI checks for that overlay.
        if (admitted.offline_quality &&
            (rad::config_parse(static_cast<int>(g_ptrs.size() - 1), g_ptrs.data(), &cfg) != RAD_OK || !cfg.kld()))
            return fail("native offline run requires a valid KL record or score invocation"), -1;
        // Fleet card ordinals describe process-visible GPUs; rank_devices remains
        // the native BDF list used by rad0016 when the layout requires it.
        cfg.devices = fleet.cards;
        g_front_load_ms = fleet.load_timeout_ms;
        g_front_drain_ms = fleet.drain_timeout_ms;
        g_front_keepalive_ms = fleet.keepalive_ms;
        g_front_aliases = fleet.models.front().aliases;
        *out = std::move(cfg);
        return 1;
    } catch (const std::exception& error) {
        fail(std::string("native YAML replay failed: ") + error.what());
        return -1;
    }
#endif
}
} // namespace

#ifdef BC_RAD_NATIVE_FLEET_CONFIG
int bc_serving_managed_front(rad::Engine& engine, std::unique_ptr<rad::StallWatch> watch) {
    using namespace std::chrono_literals;
    const rad::Config& cfg = engine.config();
    if (cfg.kld() || cfg.calib() || cfg.diagnose_mode()) return engine.serve(std::move(watch));

    // The stock fleet's transport policy for one model, using its admitted server keys.
    rad::server::TransportOptions transport;
    transport.host = cfg.host;
    transport.port = cfg.port;
    transport.n_threads = rad::server::idle_workers(cfg.http_threads);
    transport.api_key = cfg.api_key;
    transport.cors = cfg.cors;
    transport.read_timeout_s = cfg.http_read_timeout_s;
    transport.write_timeout_s = cfg.http_write_timeout_s;
    transport.keep_alive_timeout_s = cfg.http_keep_alive_timeout_s;
    transport.max_body_bytes = static_cast<size_t>(cfg.http_max_body_mib) << 20;
    transport.trusted_proxies = cfg.trusted_proxies;
    const int64_t queue_depth = cfg.max_queued_requests > 0 ? cfg.max_queued_requests : cfg.max_seqs * 8;
    transport.max_threads = queue_depth + 64;
    transport.max_queued = 64;

    rad::server::Front front(std::move(transport), g_front_keepalive_ms);
    auto aliases = g_front_aliases;
    // Older managed clients send an explicit model "default".
    if (cfg.served_model_name != "default" &&
        std::find(aliases.begin(), aliases.end(), "default") == aliases.end())
        aliases.emplace_back("default");
    front.add(cfg.served_model_name, aliases, g_front_load_ms, true);
    front.set_operator_rule([&cfg](const rad::server::HttpRequest& req) {
        return !cfg.api_key.empty() || rad::server::local_operator(req, cfg.host, cfg.allow_hosts);
    });
    struct sigaction action {};
    action.sa_handler = stop_managed_front;
    sigemptyset(&action.sa_mask);
    action.sa_flags = SA_RESETHAND;
    if (sigaction(SIGINT, &action, nullptr) != 0 || sigaction(SIGTERM, &action, nullptr) != 0) {
        RAD_ERR("managed front: cannot install stop handlers");
        return RAD_E_IO;
    }

    engine.set_front(&front, std::move(aliases));
    std::atomic<bool> served{false};
    int serve_rc = RAD_OK;
    std::thread model([&] {
        serve_rc = engine.serve(std::move(watch));
        served.store(true);
    });
    while (!engine.serving() && !served.load() && !g_front_stop.load())
        std::this_thread::sleep_for(20ms);

    int listen_rc = RAD_OK;
    if (!served.load() && !g_front_stop.load()) {
        std::atomic<bool> listener_done{false};
        std::thread listener([&] {
            listen_rc = front.run();
            listener_done.store(true);
        });
        while (!served.load() && !listener_done.load() && !g_front_stop.load())
            std::this_thread::sleep_for(20ms);
        engine.request_stop(g_front_drain_ms);
        model.join();
        // Stop only after the transport is running or run() has already returned.
        while (front.port() == 0 && !listener_done.load())
            std::this_thread::sleep_for(20ms);
        front.stop();
        listener.join();
    } else {
        engine.request_stop(g_front_drain_ms);
        model.join();
    }
    engine.set_front(nullptr, {});
    return serve_rc < 0 ? serve_rc : listen_rc;
}
#endif

int bc_serving_take_native_config(rad::Config* out) {
    if (!g_native_engine) return 0;
    const int result = prepare_native_engine_config(*g_native_engine, out);
    g_native_engine.reset();
    if (result < 0) {
        g_have_expected = false;
        g_engine_bdfs.clear();
    }
    return result;
}

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

bool bc_serving_mark_given(std::map<std::string, std::string>* given) {
    if (g_label.empty()) return true;
    for (auto& kv : *given)
        if (kv.second == "command line" && kv.first != "-v" && kv.first != "-vv" && kv.first != "--plan-memory" &&
            kv.first != "--kld-record" && kv.first != "--kld-corpus" && kv.first != "--kld-ref" && kv.first != "--kld-out") kv.second = g_label;
    return true;
}


// The stock YAML entry is kept in upstream core/main.cpp as a callable;
// managed native serving and quality run admission lives here. Order matters:
// native inspection precedes --manifest, and plan flags are extracted only
// after the strict manifest reader sees measuring vs serving.
#ifdef BC_RAD_NATIVE_FLEET_CONFIG
int bc_serving_engine_entry(int argc, char** argv, const bc::serving::callback::Guarded<int(int, char**)>& run_stock) {
    if (const auto rc = bc::serving::inspect_native_fleet_config_command(argc, argv); rc.has_value())
        return *rc;
    char** const original_argv = argv;
    if (!bc_serving_engine_argv(&argc, &argv)) return 2;
    // The managed native start (RSC02): installs what rad0011's flag step below reads, so it comes first.
    if (int rc = bc_serving_managed_take_flag(&argc, &argv)) return rc;
    if (int rc = bc_plan_memory_take_flag(&argc, &argv)) return rc;

    rad::Config cfg;
    const int native_config = bc_serving_take_native_config(&cfg);
    if (native_config < 0) return 2;
    if (native_config == 0 && argv == original_argv) {
        const auto stock = run_stock.invoke("stock YAML entry", argc, argv);
        if (!stock) {
            std::cerr << "radiance: " << stock.error().message << "\n";
            return 2;
        }
        return stock.value();
    }
    if (native_config == 0 &&
        (rad::config_parse(argc, argv, &cfg) != RAD_OK || !bc_serving_mark_given(&cfg.given)))
        return 2;

    if (cfg.proceed_incompatible) rad::Engine::print_unsupported_banner();
    if (!rad::resolve_hf_model(cfg)) return 1;

    rad::Engine engine(std::move(cfg));
    auto watch = std::make_unique<rad::StallWatch>("startup", 120000);
    int s = engine.start();
    if (s < 0 || engine.config().debug_graph) watch.reset();
    if (s < 0) {
        engine.print_startup_report(s);
        return 1;
    }
    if (engine.config().debug_graph) return engine.dump();
    s = bc_serving_managed_front(engine, std::move(watch));
    if (s < 0) {
        if (!engine.startup_report_printed()) engine.print_startup_report(s);
        return 1;
    }
    return 0;
}
#endif
