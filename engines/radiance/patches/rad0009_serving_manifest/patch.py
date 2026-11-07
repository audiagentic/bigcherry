"""Radiance 1.14.0: start the engine from a serving manifest v1 (RSC07, RSC08 S2b), as thin hooks into one
BigCherry-owned extension point.

All logic lives in the two files this patch creates (core/bc_serving_manifest.h, core/bc_serving_manifest.cpp):
``--manifest PATH`` is read by the shared strict v1 reader (bc_serving_v1: the digest, the contract and every semantic
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
 *   engine --manifest PATH [--role engine] [-v|-vv]
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

namespace bc::serving::v1 { struct EnginePlan; }

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
bool bc_serving_expected_plan(bc::serving::v1::EnginePlan* expected, int64_t* floor_ppm);

#endif /* BC_SERVING_MANIFEST_H */
"""

_BC_SERVING_MANIFEST_CPP = r"""/* bc_serving_manifest.cpp -- policy behind bc_serving_manifest.h: a serving manifest v1, read by the shared strict reader,
 * to radiance's own flags through the generated registry, and the report source label. */
#include "bc_serving_manifest.h"

#include "manifest.h"
#include "serving_types.h"

#include <nlohmann/json.hpp>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iterator>
#include <set>
#include <string>
#include <vector>

namespace rad {
/* radiance's own entry point, declared here so this file needs no engine header; a signature that drifts fails
 * at link time. */
namespace diag {
std::vector<std::string> config_option_names();
}
}  // namespace rad

namespace {
namespace v1 = bc::serving::v1;

std::vector<std::string> g_store;   /* the synthetic argv, kept for the life of the process */
std::vector<char*> g_ptrs;
std::vector<std::string> g_emitted;  /* the options the manifest produced, for bc_serving_mark_given */
std::string g_label;                /* "manifest <12 hex>" once a manifest was loaded, else empty */
v1::EnginePlan g_expected;          /* a serving start's expected plan, from the manifest's measured plan */
int64_t g_floor_ppm = 0;
bool g_have_expected = false;

bool fail(const std::string& why) {
    std::fprintf(stderr, "radiance: --manifest: %s\n", why.c_str());
    return false;
}

/* One field's value as flag text: a string or integer as is, an array of strings joined with commas (--kernels). */
bool flag_text(const nlohmann::json& v, std::string* out) {
    if (v.is_string()) { *out = v.get<std::string>(); return true; }
    if (v.is_number_integer()) { *out = v.dump(); return true; }
    if (v.is_array()) {
        std::string joined;
        for (const auto& e : v) {
            if (!e.is_string()) return false;
            if (!joined.empty()) joined += ',';
            joined += e.get<std::string>();
        }
        *out = joined;
        return true;
    }
    return false;
}

}  // namespace

bool bc_serving_engine_argv(int* argc, char*** argv) {
    g_have_expected = false;   /* each call starts from nothing: a start without --manifest holds no plan */
    bool manifest_mode = false;
    for (int i = 1; i < *argc; ++i)
        if (!std::strcmp((*argv)[i], "--manifest")) manifest_mode = true;
    if (!manifest_mode) return true;

    std::string path, role, plan_memory;
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
        } else if (a == "--plan-memory") {
            /* The measuring start: radiance writes its memory plan to FILE and stops (rad0011). It is the one flag that
             * lets the engine take an unmeasured manifest; every other start takes only a measured one. */
            if (!plan_memory.empty()) return fail("--plan-memory given twice");
            if (i + 1 >= *argc || !*(*argv)[i + 1]) return fail("--plan-memory needs a path");
            plan_memory = (*argv)[++i];
        } else if (a == "-v" || a == "-vv") {
            verbosity.push_back(a);
        } else if (kRunFlags.count(a)) {
            if (i + 1 >= *argc || !*(*argv)[i + 1]) return fail(a + " needs a path");
            run_flags.emplace_back(a, (*argv)[++i]);
        } else {
            return fail("'" + a + "' cannot be combined with --manifest (only --role engine, --plan-memory FILE, -v, -vv and the quality-run flags --kld-record, --kld-corpus, --kld-ref and --kld-out may)");
        }
    }

    std::ifstream in(path, std::ios::binary);
    if (!in) return fail("cannot read '" + path + "'");
    const std::string text((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    /* The shared strict reader: size, UTF-8, duplicate keys, integers, the contract identity, the content digest, the
     * generated structural decode, provenance and every semantic rule, with the same errors bc-serve gives. */
    const v1::Result<v1::Manifest> manifest =
        v1::read_manifest(text, path, plan_memory.empty() ? v1::PlanUse::serving : v1::PlanUse::measuring);
    if (!manifest.ok()) {
        std::fputs(v1::render_human("radiance", manifest.error()).c_str(), stderr);
        return false;
    }
    const nlohmann::json doc = nlohmann::json::parse(text, nullptr, false);
    if (doc.is_discarded()) return fail("'" + path + "' is not JSON");   /* unreachable after the strict reader */
    g_have_expected = false;
    /* A quality run (the KL mode) scores the trunk's own rows and never serves: it holds no checkpoint slots and a smaller
     * static VRAM set, so the serving plan's admission check does not describe it and none is armed. */
    if (plan_memory.empty() && run_flags.empty()) {
        /* a serving start: the plan the manifest carries is the plan this engine must admit (rad0011 checks it) */
        const v1::Result<v1::EngineView> view = v1::engine_view(manifest.value());
        if (!view.ok()) {
            std::fputs(v1::render_human("radiance", view.error()).c_str(), stderr);
            return false;
        }
        g_expected = view.value().memory.expected;
        g_floor_ppm = view.value().kv_backing_floor_ppm;
        g_have_expected = true;
    }

    std::set<std::string> known;
    for (const std::string& n : rad::diag::config_option_names()) known.insert(n);

    /* Every registry field bound to a radiance option, in registry order. A field the manifest leaves out is left to
     * radiance's own default. The argv shape follows the JSON type alone; a shape the option does not accept is left
     * for radiance's own config_parse to reject, which makes main exit 2. */
    std::vector<std::pair<std::string, std::string>> flags;
    std::set<std::string> switches;   /* the flags above that take no value */
    for (const v1::FieldInfo& f : v1::kFields) {
        if (f.radiance_option.empty() || f.path.rfind("/spec/", 0) != 0) continue;
        const nlohmann::json::json_pointer ptr{std::string(f.path)};
        if (!doc.contains(ptr)) continue;
        const std::string opt(f.radiance_option);
        if (!known.count(opt)) return fail("\"" + std::string(f.path) + "\" is bound to " + opt + ", which is not a radiance option");
        const nlohmann::json& v = doc.at(ptr);
        if (f.radiance_switch) {   /* a flag that takes no value: its presence in the manifest is the flag (the registry makes it the one value "on") */
            flags.emplace_back(opt, std::string());
            switches.insert(opt);
            continue;
        }
        std::string value;
        if (!flag_text(v, &value)) return fail("\"" + std::string(f.path) + "\" has a value that is not a string, an integer or an array of strings");
        flags.emplace_back(opt, value);
    }
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
    for (const auto& kv : run_flags) {
        g_store.push_back(kv.first);
        g_store.push_back(kv.second);
    }
    g_emitted.clear();
    for (const auto& kv : flags) {
        g_store.push_back(kv.first);
        if (!switches.count(kv.first)) g_store.push_back(kv.second);
        g_emitted.push_back(kv.first);
    }
    g_ptrs.clear();
    for (std::string& s : g_store) g_ptrs.push_back(&s[0]);
    g_ptrs.push_back(nullptr);
    *argc = (int)g_store.size();
    *argv = g_ptrs.data();
    /* The hand-off settings the engine's MoE hook reads from its environment (libr3, R3_MOE_WORKER*). The manifest is the one
     * place they are configured, so they are set from it here, before the engine initialises, and win over a value the
     * launcher left in the environment: R3_MOE_WORKER (the hand-off is on when the manifest has helpers), the segment name,
     * the row cap, whether the engine skips helper-owned experts, and the route-histogram file and period. */
    if (doc.at("spec").contains("handoff")) {
        const nlohmann::json& h = doc.at("spec").at("handoff");
        setenv("R3_MOE_WORKER", "1", 1);
        if (h.contains("transport")) {
            const nlohmann::json& tr = h.at("transport");
            if (tr.contains("shm") && tr.at("shm").is_string()) setenv("R3_MOE_WORKER_SHM", tr.at("shm").get<std::string>().c_str(), 1);
            if (tr.contains("max_rows") && tr.at("max_rows").is_number_integer())
                setenv("R3_MOE_WORKER_MAX_ROWS", std::to_string(tr.at("max_rows").get<long long>()).c_str(), 1);
            if (tr.contains("skip_owned") && tr.at("skip_owned").is_boolean())
                setenv("R3_MOE_WORKER_SKIP_OWNED", tr.at("skip_owned").get<bool>() ? "1" : "0", 1);
        }
        if (h.contains("telemetry")) {
            const nlohmann::json& te = h.at("telemetry");
            if (te.contains("histogram_file") && te.at("histogram_file").is_string())
                setenv("R3_MOE_WORKER_HISTOGRAM", te.at("histogram_file").get<std::string>().c_str(), 1);
            if (te.contains("period_s") && te.at("period_s").is_number_integer())
                setenv("R3_MOE_WORKER_HISTOGRAM_PERIOD_S", std::to_string(te.at("period_s").get<long long>()).c_str(), 1);
        }
    }
    /* Every field the contract binds to an environment variable (kEnvBindings, generated from the registry's env_setting):
     * the variable is set from the manifest, so the manifest is the one place the setting is configured, and it wins over a
     * value the launcher left in the environment. An absent field leaves the variable alone (the kernel's own default). */
    for (const bc::serving::v1::EnvBinding& b : bc::serving::v1::kEnvBindings) {
        const nlohmann::json::json_pointer ptr{std::string(b.path)};
        if (!doc.contains(ptr)) continue;
        const nlohmann::json& v = doc.at(ptr);
        if (b.kind == "boolean" && v.is_boolean()) setenv(std::string(b.env).c_str(), v.get<bool>() ? "1" : "0", 1);
        else if (b.kind == "integer" && v.is_number_integer())
            setenv(std::string(b.env).c_str(), std::to_string(v.get<long long>()).c_str(), 1);
    }
    const std::string digest = doc.at("metadata").at("content_digest").get<std::string>();   /* "sha256:<hex>" */
    g_label = "manifest " + digest.substr(digest.find(':') + 1, 12);
    return true;
}

bool bc_serving_expected_plan(v1::EnginePlan* expected, int64_t* floor_ppm) {
    if (!g_have_expected) return false;
    *expected = g_expected;
    *floor_ppm = g_floor_ppm;
    return true;
}

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
# its one source list (BigCherry's components/serving/bc_serving_v1.cmake) with this tree's abi and vendored JSON.
if(NOT DEFINED BC_SERVING_V1_CMAKE)
  message(FATAL_ERROR "rad0009: -DBC_SERVING_V1_CMAKE=<bigcherry>/engines/radiance/components/serving/bc_serving_v1.cmake "
                      "is required (the shared serving manifest v1 library); the BigCherry lane passes it")
endif()
set(BC_SERVING_V1_NLOHMANN_INCLUDE "${CMAKE_SOURCE_DIR}/vendor")
include("${BC_SERVING_V1_CMAKE}")
target_link_libraries(rad_core PUBLIC bc_serving_v1)"""



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
            id="link-bc-serving-v1",
            anchor=r"^target_link_libraries\(rad_core PUBLIC rad_abi \$\{CMAKE_DL_LIBS\} Threads::Threads\)$",
            text=_CORE_LINK,
            guard=r"include\(\"\$\{BC_SERVING_V1_CMAKE\}\"\)",
            max_span_lines=1,
            expect_matches=1,
            rationale="rad_core's unique link line; the shared library is included and linked right after it, failing "
                      "closed when the lane did not pass its .cmake.",
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
                id="bc-serving-mark-given",
                anchor=r"^    if \(s < 0\)  \{ rad::config_usage\(argv\[0\]\); return 2; \}$",
                text="\n    if (!bc_serving_mark_given(&cfg.given)) return 2;",
                guard=r"if \(!bc_serving_mark_given\(&cfg\.given\)\) return 2;",
                expect_matches=1,
                rationale="The line after the config_parse error exit: reached only after a successful parse (--help returned above), so the check and relabel never run on a failed parse.",
            ),
        ),
    ),
)
