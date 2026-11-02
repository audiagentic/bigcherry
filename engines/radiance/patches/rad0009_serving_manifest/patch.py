"""Radiance 1.14.0: start the engine from a resolved serving manifest (RSC07), as thin hooks into one BigCherry-owned
extension point.

All logic lives in the two files this patch creates (core/bc_serving_manifest.h, core/bc_serving_manifest.cpp):
``--manifest PATH`` is turned into the equivalent radiance flags before config_parse, so radiance's own parser and
validation still run, and the startup report's option sources are relabelled afterwards. core/main.cpp gains one
include and two calls. config.cpp is not touched: --manifest is handled before config_parse and is deliberately not a
row of the option table (a row is not a call into the extension point, so it cannot be a thin edit).
"""

from bigcherry.patcher import Edit, FilePatch

_BC_SERVING_MANIFEST_H = r"""/* bc_serving_manifest.h -- BigCherry's start-up extension point: the engine starts from a resolved serving
 * manifest (the JSON bc-serve resolves) instead of command-line flags.
 *
 *   engine --manifest PATH [--role engine] [-v|-vv]
 *
 * main() calls bc_serving_engine_argv before config_parse, so radiance's own parser and its own validation and
 * errors see the manifest's "engine.*" values as the equivalent flags; no other flag may accompany --manifest.
 * After a successful parse, bc_serving_mark_given relabels the startup report's option sources. All policy
 * lives in core/bc_serving_manifest.cpp; --manifest is deliberately not a row of radiance's option table. */
#ifndef BC_SERVING_MANIFEST_H
#define BC_SERVING_MANIFEST_H

#include <map>
#include <string>

/* The argv radiance should parse. Without --manifest, argv is left unchanged and true is returned. With it, *argc
 * and *argv are replaced by a synthetic argv (storage lives for the process). On a malformed manifest or a stray
 * flag a message is printed to stderr and false is returned: the caller exits 2, as for any config error. */
bool bc_serving_engine_argv(int* argc, char*** argv);

/* After a successful parse: every option the manifest produced must be recorded in `given` as "command line" (an
 * option another option swallowed as its argument is not), else a message is printed and false is returned: the
 * caller exits 2. Then those sources are relabelled "manifest <first 12 chars of content_sha256>". Returns true
 * and does nothing when the engine was not started from a manifest. */
bool bc_serving_mark_given(std::map<std::string, std::string>* given);

#endif /* BC_SERVING_MANIFEST_H */
"""

_BC_SERVING_MANIFEST_CPP = r"""/* bc_serving_manifest.cpp -- policy behind bc_serving_manifest.h: manifest "engine.*" values to radiance's own
 * flags, the checks that make the manifest the only source of configuration, and the report source label. */
#include "bc_serving_manifest.h"

#include <nlohmann/json.hpp>

#include <cmath>
#include <cstdio>
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

std::vector<std::string> g_store;   /* the synthetic argv, kept for the life of the process */
std::vector<char*> g_ptrs;
std::vector<std::string> g_emitted;  /* the options the manifest produced, for bc_serving_mark_given */
std::string g_label;                /* "manifest <12 hex>" once a manifest was loaded, else empty */

bool fail(const std::string& why) {
    std::fprintf(stderr, "radiance: --manifest: %s\n", why.c_str());
    return false;
}

bool string_nonempty(const nlohmann::json& values, const char* key, const char* env, std::string* why) {
    const auto it = values.find(key);
    if (it == values.end() || !it->is_string() || it->get_ref<const std::string&>().empty()) {
        *why = std::string("\"") + key + "\" must be a non-empty string: it is required so that " + env +
               " cannot silently supply it";
        return false;
    }
    return true;
}

}  // namespace

bool bc_serving_engine_argv(int* argc, char*** argv) {
    bool manifest_mode = false;
    for (int i = 1; i < *argc; ++i)
        if (!std::strcmp((*argv)[i], "--manifest")) manifest_mode = true;
    if (!manifest_mode) return true;

    std::string path, role;
    std::vector<std::string> verbosity;
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
        } else if (a == "-v" || a == "-vv") {
            verbosity.push_back(a);
        } else {
            return fail("'" + a + "' cannot be combined with --manifest (only --role engine, -v and -vv may)");
        }
    }

    std::ifstream in(path, std::ios::binary);
    if (!in) return fail("cannot read '" + path + "'");
    const std::string text((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    const nlohmann::json doc = nlohmann::json::parse(text, nullptr, false);
    if (doc.is_discarded() || !doc.is_object()) return fail("'" + path + "' is not a JSON object");
    const auto ver = doc.find("schema_version");
    if (ver == doc.end() || !ver->is_number_integer() || ver->get<long long>() != 1)
        return fail("schema_version must be 1");
    const auto vals = doc.find("values");
    if (vals == doc.end() || !vals->is_object()) return fail("no \"values\" object");
    const auto alg = doc.find("hash_algorithm");
    if (alg == doc.end() || !alg->is_string() || alg->get<std::string>() != "sha256")
        return fail("hash_algorithm must be \"sha256\"");
    const auto sha = doc.find("content_sha256");
    if (sha == doc.end() || !sha->is_string() || sha->get_ref<const std::string&>().size() < 12)
        return fail("content_sha256 is missing");
    const nlohmann::json& values = *vals;

    std::string why;
    if (!string_nonempty(values, "engine.radiance_home", "RADIANCE_HOME", &why) ||
        !string_nonempty(values, "engine.kernels", "RADIANCE_KERNELS", &why))
        return fail(why);

    std::set<std::string> known;
    for (const std::string& n : rad::diag::config_option_names()) known.insert(n);

    /* The argv shape follows the JSON type alone: true is a bare flag, false is no flag, a string or number is a
     * flag with a value. A shape the option does not accept (a value for a boolean flag, a bare flag that needs an
     * argument) is left for radiance's own config_parse to reject, which makes main exit 2. */
    std::map<std::string, std::string> keys;   /* sorted: option -> argument */
    std::set<std::string> flags;               /* sorted bare flags */
    for (auto it = values.begin(); it != values.end(); ++it) {
        const std::string& key = it.key();
        if (key.rfind("engine.", 0) != 0) continue;
        std::string opt = "--" + key.substr(7);
        for (char& c : opt)
            if (c == '_') c = '-';
        if (opt == "--" || !known.count(opt))
            return fail("\"" + key + "\" is not a radiance option (no " + opt + ")");
        const nlohmann::json& v = it.value();
        if (v.is_boolean()) {
            if (v.get<bool>()) flags.insert(opt);   /* false: the flag is not given, so the default stands */
        } else if (v.is_string()) {
            keys[opt] = v.get<std::string>();
        } else if (v.is_number_integer()) {
            keys[opt] = v.dump();
        } else if (v.is_number_float() && std::isfinite(v.get<double>())) {
            keys[opt] = v.dump();
        } else {
            return fail("\"" + key + "\" must be a string, a finite number or a boolean");
        }
    }

    g_store.clear();
    g_store.push_back((*argv)[0]);
    for (const std::string& v : verbosity) g_store.push_back(v);
    for (const auto& kv : keys) {
        g_store.push_back(kv.first);
        g_store.push_back(kv.second);
    }
    for (const std::string& f : flags) g_store.push_back(f);
    g_emitted.clear();
    for (const auto& kv : keys) g_emitted.push_back(kv.first);
    for (const std::string& f : flags) g_emitted.push_back(f);
    g_ptrs.clear();
    for (std::string& s : g_store) g_ptrs.push_back(&s[0]);
    g_ptrs.push_back(nullptr);
    *argc = (int)g_store.size();
    *argv = g_ptrs.data();
    g_label = "manifest " + sha->get<std::string>().substr(0, 12);
    return true;
}

bool bc_serving_mark_given(std::map<std::string, std::string>* given) {
    if (g_label.empty()) return true;
    /* The argv shape came from the JSON types alone, so an option the parser took as another option's argument
     * (a bare flag for an option that needs one) never reaches `given`: the parse was not the manifest's. */
    for (const std::string& opt : g_emitted) {
        const auto it = given->find(opt);
        if (it == given->end() || it->second != "command line") {
            std::fprintf(stderr, "radiance: --manifest: %s was not parsed as given\n", opt.c_str());
            return false;
        }
    }
    for (auto& kv : *given)
        if (kv.second == "command line" && kv.first != "-v" && kv.first != "-vv") kv.second = g_label;
    return true;
}
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
