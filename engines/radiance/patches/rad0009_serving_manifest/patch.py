"""Radiance 1.14.0: start the engine from a serving manifest v1 (RSC07, RSC08 S2b), as thin hooks into one
BigCherry-owned extension point.

All logic lives in the two files this patch creates (core/bc_serving_manifest.h, core/bc_serving_manifest.cpp):
``--manifest PATH`` is read by the shared strict v1 reader (bc_serving_v2: the digest, the contract and every semantic
rule are checked exactly as bc-serve checks them), the fields the registry binds to a radiance option become the
equivalent flags before config_parse, so radiance's own parser and validation still run, and the startup report's
option sources are relabelled afterwards. core/main.cpp gains one include and two calls; core/CMakeLists.txt links the
shared library. config.cpp is not touched: --manifest is handled before config_parse and is deliberately not a row of
the option table (a row is not a call into the extension point, so it cannot be a thin edit).
"""

from bigcherry.patcher import Edit, FilePatch

_BC_SERVING_MANIFEST_H = r"""/* bc_serving_manifest.h -- BigCherry's start-up extension point: the engine starts from a serving manifest v1 (the
 * document bc-serve --write-manifest writes) instead of command-line flags.
 *
 *   engine --manifest PATH [--role engine] [-v|-vv] [--bc-heat-launch-nonce HEX32]
 *
 * main() calls bc_serving_engine_argv before config_parse, so radiance's own parser and its own validation and
 * errors see the manifest's engine-bound values as the equivalent flags; no other flag may accompany --manifest.
 * After a successful parse, bc_serving_mark_given relabels the startup report's option sources. All policy
 * lives in core/bc_serving_manifest.cpp; --manifest is deliberately not a row of radiance's option table. */
#ifndef BC_SERVING_MANIFEST_H
#define BC_SERVING_MANIFEST_H

#include <cstdint>
#include <map>
#include <string>
#include <vector>

namespace bc::serving::v2 { struct EnginePlan; }

/* The argv radiance should parse. Without --manifest, argv is left unchanged and true is returned. With it, *argc
 * and *argv are replaced by a synthetic argv (storage lives for the process). On a malformed manifest or a stray
 * flag a message is printed to stderr and false is returned: the caller exits 2, as for any config error. */
bool bc_serving_engine_argv(int* argc, char*** argv);

/* After a successful parse: every option the manifest produced must be recorded in `given` as "command line" (an
 * option another option swallowed as its argument is not), else a message is printed and false is returned: the
 * caller exits 2. Then those sources are relabelled "manifest <first 12 hex of metadata.content_digest>". Returns true
 * and does nothing when the engine was not started from a manifest. */
bool bc_serving_mark_given(std::map<std::string, std::string>* given);

/* The expected memory plan the engine must admit at start-up, and the kv_backing_floor_ppm that goes with it, taken from the
 * manifest's measured plan (engine_view; spec §5.2). True only when the engine was started from a measured manifest (a
 * serving start); false for a start without --manifest and for the measuring start (--plan-memory), which has no plan to
 * check. A serving start with no plan cannot get here: the strict reader refuses an unmeasured manifest for it. */
bool bc_serving_expected_plan(bc::serving::v2::EnginePlan* expected, int64_t* floor_ppm);
/* BDFs are from the same strictly validated engine view, not device ordinals. */
std::vector<std::string> bc_serving_engine_bdfs();

/* Convert the manifest's per-rank host-pool budget to Radiance 2.1's aggregate CLI value. */
bool bc_serving_manifest_host_pool_total(int64_t per_rank_mib, int64_t tp, int64_t* total_mib);

#endif /* BC_SERVING_MANIFEST_H */
"""

_BC_SERVING_MANIFEST_CPP = r"""/* bc_serving_manifest.cpp -- policy behind bc_serving_manifest.h: a serving manifest v1, read by the shared strict reader,
 * to radiance's own flags through the generated registry, and the report source label. */
#include "bc_serving_manifest.h"
#include "bc_engine_settings.h"

#include "manifest.h"
#include "serving_types.h"
#include "native_option_projection.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iterator>
#include <limits>
#include <set>
#include <string>
#include <utility>
#include <vector>

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
std::vector<std::string> g_emitted;  /* the options the manifest produced, for bc_serving_mark_given */
std::string g_label;                /* "manifest <12 hex>" once a manifest was loaded, else empty */
v2::EnginePlan g_expected;          /* a serving start's expected plan, from the manifest's measured plan */
int64_t g_floor_ppm = 0;
bool g_have_expected = false;
std::vector<std::string> g_engine_bdfs;
BcEngineSettings g_engine_settings{};
std::string g_shm;
std::string g_histogram;
std::string g_heat_launch_nonce;

bool fail(const std::string& why) {
    std::fprintf(stderr, "radiance: --manifest: %s\n", why.c_str());
    return false;
}

}  // namespace

bool bc_serving_manifest_host_pool_total(int64_t per_rank_mib, int64_t tp, int64_t* total_mib) {
    if (!total_mib || tp <= 0 || per_rank_mib < 0 || per_rank_mib > std::numeric_limits<int64_t>::max() / tp)
        return false;
    *total_mib = per_rank_mib * tp;
    return true;
}

bool bc_serving_engine_argv(int* argc, char*** argv) {
    g_engine_bdfs.clear();
    g_have_expected = false;   /* each call starts from nothing: a start without --manifest holds no plan */
    g_engine_settings = BcEngineSettings{};
    g_shm.clear();
    g_histogram.clear();
    g_heat_launch_nonce.clear();
    bool manifest_mode = false;
    for (int i = 1; i < *argc; ++i)
        if (!std::strcmp((*argv)[i], "--manifest")) manifest_mode = true;
    if (!manifest_mode) return true;

    std::string path, role, plan_memory, rank_out, heat_nonce;
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

    if (!rank_out.empty() && plan_memory.empty()) return fail("--plan-memory-ranks requires --plan-memory");
    if (!heat_nonce.empty() && (!plan_memory.empty() || !rank_out.empty() || !run_flags.empty()))
        return fail("--bc-heat-launch-nonce is valid only for serving, not measurement or quality runs");
    std::ifstream in(path, std::ios::binary);
    if (!in) return fail("cannot read '" + path + "'");
    const std::string text((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
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
    const nlohmann::json doc = nlohmann::json::parse(text, nullptr, false);
    if (doc.is_discarded()) return fail("'" + path + "' is not JSON");   /* unreachable after the strict reader */
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

    std::set<std::string> known;
    for (const std::string& n : rad::diag::config_option_names()) known.insert(n);

    /* Every registry field bound to a radiance option, in registry order. A field the manifest leaves out is left to
     * radiance's own default. The argv shape follows the JSON type alone; a shape the option does not accept is left
     * for radiance's own config_parse to reject, which makes main exit 2. */
    // Exactly one projection: the generated serving-v2 registry supplies ALL
    // Radiance option names. This narrow hook is replaced by RUM04's immutable
    // upstream runtime bundle; neither the manifest writer nor the engine
    // maintains an independent option-name map.
    auto projected = v2::project_native_options(doc, v2::kFields, known);
    if (!projected.error.empty()) return fail(projected.error);
    auto& flags = projected.flags;
#ifdef BC_RAD_HOST_POOL_MIB_AGGREGATE
    /* The manifest records the admitted host-pool budget per rank, while Radiance 2.1's --host-pool-mib
     * parser accepts the aggregate and divides it by TP. Convert at this adapter boundary so config_parse
     * reconstructs the exact per-rank value that BigCherry admitted. */
    const auto& typed_spec = manifest.value().spec;
    const int64_t tp = typed_spec.processes.engine.scheduler.tp;
    const int64_t host_pool_per_rank = typed_spec.resources.host.pool_mib;
    int64_t host_pool_total = 0;
    if (!bc_serving_manifest_host_pool_total(host_pool_per_rank, tp, &host_pool_total))
        return fail("resources.host.pool_mib x engine.tp exceeds Radiance's signed MiB limit");
    const auto host_pool_flag = std::find_if(flags.begin(), flags.end(), [](const auto& flag) {
        return flag.first == "--host-pool-mib";
    });
    if (host_pool_flag == flags.end()) return fail("resources.host.pool_mib has no --host-pool-mib projection");
    host_pool_flag->second = std::to_string(host_pool_total);
#endif
    for (const char* required : {"--radiance-home", "--kernels"}) {
        bool found = false;
        for (const auto& kv : flags)
            if (kv.first == required && !kv.second.empty()) found = true;
        if (!found)
            return fail(std::string(required) + " must be present and non-empty: it is required so that " +
                        (std::strcmp(required, "--kernels") == 0 ? "RADIANCE_KERNELS" : "RADIANCE_HOME") +
                        " cannot silently supply it");
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
    g_emitted.clear();
    for (const auto& kv : flags) g_emitted.push_back(kv.first);
    const auto native_argv = projected.argv();
    g_store.insert(g_store.end(), native_argv.begin(), native_argv.end());
    g_ptrs.clear();
    for (std::string& s : g_store) g_ptrs.push_back(&s[0]);
    g_ptrs.push_back(nullptr);
    *argc = (int)g_store.size();
    *argv = g_ptrs.data();
    // Fill one immutable typed snapshot from the strict reader's decoded manifest.
    // Dynamic plugins consume this via bc_serving_engine_settings, not getenv.
    const auto& spec = manifest.value().spec;
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

extern "C" int bc_serving_engine_settings(BcEngineSettings* out) {
    if (!out || out->struct_size != sizeof(BcEngineSettings) || !g_engine_settings.manifest) return 0;
    *out = g_engine_settings;
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

bool bc_serving_expected_plan(v2::EnginePlan* expected, int64_t* floor_ppm) {
    if (!g_have_expected) return false;
    *expected = g_expected;
    *floor_ppm = g_floor_ppm;
    return true;
}

std::vector<std::string> bc_serving_engine_bdfs() { return g_engine_bdfs; }

bool bc_serving_mark_given(std::map<std::string, std::string>* given) {
    if (g_label.empty()) return true;
    /* The argv shape came from the JSON types alone, so an option the parser took as another option's argument never
     * reaches `given`: the parse was not the manifest's. */
    for (const std::string& opt : g_emitted) {
        const auto it = given->find(opt);
        if (it == given->end() || it->second != "command line") {
            std::fprintf(stderr, "radiance: --manifest: %s was not parsed as given\n", opt.c_str());
            return false;
        }
    }
    for (auto& kv : *given)
        if (kv.second == "command line" && kv.first != "-v" && kv.first != "-vv" && kv.first != "--plan-memory" &&
            kv.first != "--kld-record" && kv.first != "--kld-corpus" && kv.first != "--kld-ref" && kv.first != "--kld-out") kv.second = g_label;
    return true;
}
"""

# The shared library comes from the BigCherry tree, never a copy: the lane passes its .cmake (components/serving).
_CORE_LINK = r"""
# BigCherry rad0009 (RSC08 S2b): the engine reads its manifest and (rad0011) checks its plan through the shared serving manifest v1 library, built here from
# its one source list (BigCherry's components/serving/bc_serving_v2.cmake) against this tree's ABI.
if(NOT DEFINED BC_SERVING_V2_CMAKE)
  message(FATAL_ERROR "rad0009: -DBC_SERVING_V2_CMAKE=<bigcherry>/engines/radiance/components/serving/bc_serving_v2.cmake "
                      "is required (the shared serving manifest v1 library); the BigCherry lane passes it")
endif()
include("${BC_SERVING_V2_CMAKE}")
target_link_libraries(rad_core PUBLIC bc_serving_v2)
if(PROJECT_VERSION VERSION_GREATER_EQUAL "2.0")
  target_compile_definitions(rad_core PRIVATE BC_RAD_HOST_POOL_MIB_AGGREGATE=1)
endif()

if(RAD_BUILD_TESTS AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/fleet/config.h")
  get_filename_component(_bc_serving "${BC_SERVING_V2_CMAKE}" DIRECTORY)
  add_executable(bc_native_fleet_config_test
      "${_bc_serving}/native_fleet_config.cpp"
      "${_bc_serving}/native_fleet_config_test.cpp")
  target_link_libraries(bc_native_fleet_config_test PRIVATE rad_core bc_jcs)
  target_compile_definitions(bc_native_fleet_config_test PRIVATE
      BC_CONTRACT_DIR="${_bc_serving}/../../../../docs/design/serving-manifest-v2")
endif()
"""

_CORE_TEST_REGISTRATION = r"""
add_test(NAME bc_native_fleet_config_test COMMAND bc_native_fleet_config_test)
"""



PATCHES = (
    FilePatch(
        path="core/bc_serving_manifest.h",
        create=True,
        description="Create the BigCherry serving-manifest extension-point header.",
        edits=(Edit(
            id="create-bc-serving-manifest-h",
            anchor=r"\A",
            text=_BC_SERVING_MANIFEST_H,
            guard=r"^bool bc_serving_engine_argv\(int\* argc, char\*\*\* argv\);$",
            expect_matches=1,
            rationale="Created files are anchored at the empty upstream file; create=True rejects any upstream collision.",
        ),),
    ),
    FilePatch(
        path="core/bc_serving_manifest.cpp",
        create=True,
        description="Create the manifest-to-argv policy, the RADIANCE_HOME/RADIANCE_KERNELS guard and the report-source relabel.",
        edits=(Edit(
            id="create-bc-serving-manifest-cpp",
            anchor=r"\A",
            text=_BC_SERVING_MANIFEST_CPP,
            guard=r"^bool bc_serving_engine_argv\(int\* argc, char\*\*\* argv\) \{$",
            expect_matches=1,
            rationale="Created files are anchored at the empty upstream file; create=True rejects any upstream collision.",
        ),),
    ),
    FilePatch(
        path="core/CMakeLists.txt",
        description="Link the shared serving manifest v1 library, built from BigCherry's source list, into rad_core.",
        edits=(Edit(
            id="link-bc-serving-v2",
            anchor=r"^target_link_libraries\(rad_core PUBLIC rad_abi \$\{CMAKE_DL_LIBS\} Threads::Threads\)$",
            text=_CORE_LINK,
            guard=r"include\(\"\$\{BC_SERVING_V2_CMAKE\}\"\)",
            max_span_lines=1,
            expect_matches=1,
            rationale="rad_core's unique link line; the shared library is included and linked right after it, failing "
                      "closed when the lane did not pass its .cmake.",
        ),),
    ),
    FilePatch(
        path="tests/CMakeLists.txt",
        description="Register the native fleet config test after the upstream test harness enables CTest.",
        edits=(Edit(
            id="register-bc-native-fleet-config-test",
            anchor=r"^get_property\(RAD_ALL_TESTS DIRECTORY PROPERTY TESTS\)$",
            text=_CORE_TEST_REGISTRATION,
            mode="insert_before",
            guard=r"^add_test\(NAME bc_native_fleet_config_test COMMAND bc_native_fleet_config_test\)$",
            expect_matches=1,
            rationale="tests/CMakeLists is included after enable_testing; register before the suite's final test-label pass.",
        ),),
    ),
    FilePatch(
        path="core/main.cpp",
        description="Load the engine's configuration from a serving manifest before config_parse and relabel its sources after.",
        edits=(
            Edit(
                id="include-bc-serving-manifest",
                anchor=r'^#include "util/stallwatch\.h"$',
                text='\n#include "bc_serving_manifest.h"',
                guard=r'^#include "bc_serving_manifest\.h"$',
                expect_matches=1,
                rationale="util/stallwatch.h is main.cpp's unique last include line.",
            ),
            Edit(
                id="bc-serving-engine-argv",
                anchor=r"^int main\(int argc, char\*\* argv\) \{$",
                text="\n    if (!bc_serving_engine_argv(&argc, &argv)) return 2;",
                guard=r"bc_serving_engine_argv\(&argc, &argv\)",
                expect_matches=1,
                rationale="main's signature is unique; the synthetic argv must be in place before config_parse reads argc/argv.",
            ),
            Edit(
                id="bc-managed-single-engine-marker",
                anchor=r"^int main\(int argc, char\*\* argv\) \{$",
                text="\n    char** const bc_original_argv = argv;",
                guard=r"bc_original_argv = argv",
                applies_if=r'#include "fleet/config\.h"',
                expect_matches=1,
                rationale="Identify an admitted manifest by the argv replacement, before the flag extractor runs.",
            ),
            Edit(
                id="bc-skip-native-cli-for-measured-manifest",
                anchor=r"^    std::string path;$",
                text="    rad::Config cfg;\n    if (argv == bc_original_argv) {\n",
                mode="insert_before",
                guard=r"if \(argv == bc_original_argv\)",
                applies_if=r'#include "fleet/config\.h"',
                expect_matches=1,
                rationale="Managed admission already validated the input and extracted its launch flags.",
            ),
            Edit(
                id="bc-use-single-engine-for-measured-manifest",
                anchor=r"^    rad::Config cfg = std::move\(fleet\.models\[0\]\.cfg\);$",
                text=(
                    "        cfg = std::move(fleet.models[0].cfg);\n"
                    "    } else {\n"
                    "        if (rad::config_parse(argc, argv, &cfg) != RAD_OK ||\n"
                    "            !bc_serving_mark_given(&cfg.given)) return 2;\n"
                    "    }"
                ),
                mode="replace",
                guard=r"^        cfg = std::move\(fleet\.models\[0\]\.cfg\);$",
                applies_if=r'#include "fleet/config\.h"',
                expect_matches=1,
                rationale="The temporary measured-manifest lane uses the prior single-engine lifecycle; native YAML keeps the fleet.",
            ),
            Edit(
                id="bc-serving-mark-given",
                applies_if=r"rad::config_usage\(argv\[0\]\)",
                anchor=r"^    if \(s < 0\)  \{ rad::config_usage\(argv\[0\]\); return 2; \}$",
                text="\n    if (!bc_serving_mark_given(&cfg.given)) return 2;",
                guard=r"if \(!bc_serving_mark_given\(&cfg\.given\)\) return 2;",
                expect_matches=1,
                rationale="The line after the config_parse error exit: reached only after a successful parse (--help returned above), so the check and relabel never run on a failed parse.",
            ),
        ),
    ),
)
