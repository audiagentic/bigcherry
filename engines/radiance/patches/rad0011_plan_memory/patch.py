"""RSC03 / RSC08 A3: the engine's memory plan as the serving manifest v1 EnginePlan, measured (`--plan-memory FILE`) or
checked at start-up against the expected one (`--plan-check FILE --plan-check-floor-ppm N`), through the shared v1
library bc_serving_v1 (components/serving), which radiance's core links."""
from bigcherry.patcher import Edit, FilePatch

_BC_PLAN_MEMORY_H = r"""/* bc_plan_memory.h -- BigCherry managed patch rad0011 (RSC03, RSC08 A3): the engine's memory plan.
 *
 * Engine::plan() records its sizes here as a serving manifest v1 EnginePlan (components/serving, bc_serving_v1).
 *   --plan-memory FILE                          start-up stops after plan(): FILE gets the plan (v1 JSON, written by
 *                                               the generated encoder) and the process exits before any weight loads.
 *   --plan-check FILE --plan-check-floor-ppm N  FILE is the expected plan (the one bc-serve --plan-memory measured);
 *                                               right after the VRAM split, before any pool is configured, the live
 *                                               plan is checked against it (spec section 5.2) and the result printed
 *                                               once; a refusal exits with its exit code.
 * The two --plan-check flags are interim: serving manifest v1's engine_view replaces them at the slice 6 cutover
 * (RSC08). Without these flags every call here does nothing. */
#pragma once
#include <cstdint>
#include <vector>

namespace rad {
struct Config;
struct VramFacts;
struct VramBudget;
struct WeightFootprint;
class Program;
}

/* Takes the flags above off the command line (before rad0009's manifest reader sees them) and reads the expected plan.
 * 0, or the exit code of a usage, input or decode fault, which has been printed once. */
int bc_plan_memory_take_flag(int* argc, char*** argv);

/* Records what plan() sized, and with --plan-check checks it. Called once, right after its VRAM split resolves, with
 * plan()'s own values. A refused check exits here. */
void bc_plan_memory_note(const rad::Config& cfg, const std::vector<rad::VramFacts>& facts, const rad::VramBudget& vb,
                         const std::vector<int64_t>& emb_bytes, const std::vector<rad::WeightFootprint>& wfs,
                         int64_t ngram_bytes, int64_t ck_slot, int64_t host_arenas, int64_t kv_one, int64_t ctx,
                         const rad::Program& prog);

/* After plan(): with --plan-memory, writes the plan and exits (status 0, or 2 when nothing was recorded or FILE cannot be
 * written). Otherwise returns 0 and start-up goes on. */
int bc_plan_memory_done();
"""

_BC_PLAN_CHECK_H = r"""/* bc_plan_check.h -- BigCherry managed patch rad0011 (RSC08 A3): the engine-independent half of the plan flags. No
 * Radiance type is used, so it is tested card-free (tools/tests/patch/test_radiance_patch_packages.py). */
#pragma once
#include "serving_types.h"

#include <string>

namespace bc_plan {

enum class Mode { none, measure, check };

/* Takes --plan-memory FILE, or --plan-check FILE with --plan-check-floor-ppm N (both or neither; not with
 * --plan-memory), off argv. The floor is checked by the shared validator and FILE is read and decoded as an EnginePlan
 * here, before any device work. 0, or the exit code of the fault (errors.md section 5), printed once as
 * "radiance: config: ...". */
int take_flags(int* argc, char*** argv);
Mode mode();
const std::string& measure_out();

/* With --plan-check: checks `live` against the expected plan (bc::serving::v1::check_engine_plan) and prints the one
 * start-up line, admitted with its margins or the refusal. 0 when admitted, else the refusal's exit code. */
int check(const bc::serving::v1::EnginePlan& live);

}  // namespace bc_plan
"""

_BC_PLAN_CHECK_CPP = r"""/* bc_plan_check.cpp -- see bc_plan_check.h (BigCherry managed patch rad0011, RSC08 A3). */
#include "bc_plan_check.h"

#include "admission.h"
#include "serving_codec.h"
#include "strict_json.h"

#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>

namespace bc_plan {
namespace {
namespace v1 = bc::serving::v1;
Mode g_mode = Mode::none;
std::string g_out;
v1::EnginePlan g_expected;
int64_t g_ppm = 0;

int report(const v1::ConfigError& e) {
    std::fputs(v1::render_human("radiance", e).c_str(), stderr);
    std::fflush(stderr);
    return v1::exit_code_of(e);
}

int usage(const std::string& flag, const std::string& message) {
    return report(v1::make_error(v1::ConfigErrorCode::invalid_value, flag, message,
                                 "--plan-check FILE --plan-check-floor-ppm N go together; --plan-memory FILE alone"));
}

int read_expected(const std::string& file) {
    std::ifstream in(file, std::ios::binary);
    if (!in) {
        v1::ConfigError e = v1::make_error(v1::ConfigErrorCode::io_failure, "--plan-check",
                                           "cannot read the expected plan " + file,
                                           "measure it first: bc-serve --plan-memory --plan-cache DIR");
        e.source_file = file;
        return report(e);
    }
    std::ostringstream text;
    text << in.rdbuf();
    v1::Result<nlohmann::json> doc = v1::parse_strict_json(text.str(), file);
    if (!doc) return report(doc.error());
    if (v1::Status e = v1::decode(doc.value(), "", &g_expected)) {
        e->source_file = file;
        return report(*e);
    }
    return 0;
}
}  // namespace

int take_flags(int* argc, char*** argv) {
    g_mode = Mode::none;   // each call starts from nothing, so the card-free test can run many scenarios in one process
    g_out.clear();
    g_expected = v1::EnginePlan{};
    g_ppm = 0;
    std::string check_file, ppm_text;
    bool have_ppm = false;
    int w = 1;
    for (int i = 1; i < *argc; ++i) {
        const char* a = (*argv)[i];
        std::string* into = std::strcmp(a, "--plan-memory") == 0 ? &g_out
                            : std::strcmp(a, "--plan-check") == 0 ? &check_file
                            : std::strcmp(a, "--plan-check-floor-ppm") == 0 ? &ppm_text
                            : nullptr;
        if (!into) {
            (*argv)[w++] = (*argv)[i];
            continue;
        }
        if (!into->empty() || (into == &ppm_text && have_ppm)) return usage(a, std::string(a) + " given twice");
        if (i + 1 >= *argc || !*(*argv)[i + 1]) return usage(a, std::string(a) + " needs a value");
        *into = (*argv)[++i];
        if (into == &ppm_text) have_ppm = true;
    }
    (*argv)[w] = nullptr;
    *argc = w;
    if (!g_out.empty() && (!check_file.empty() || have_ppm))
        return usage("--plan-memory", "--plan-memory measures a plan; it does not go with --plan-check");
    if (!check_file.empty() != have_ppm)
        return usage(have_ppm ? "--plan-check-floor-ppm" : "--plan-check",
                     "--plan-check and --plan-check-floor-ppm go together");
    if (!g_out.empty()) g_mode = Mode::measure;
    if (check_file.empty()) return 0;
    char* end = nullptr;
    errno = 0;
    const long long ppm = std::strtoll(ppm_text.c_str(), &end, 10);
    if (errno || end == ppm_text.c_str() || *end)
        return usage("--plan-check-floor-ppm", "--plan-check-floor-ppm must be an integer, is \"" + ppm_text + "\"");
    if (v1::Status e = v1::check_kv_backing_floor_ppm(ppm)) return report(*e);
    g_ppm = ppm;
    if (int rc = read_expected(check_file)) return rc;
    g_mode = Mode::check;
    return 0;
}

Mode mode() { return g_mode; }
const std::string& measure_out() { return g_out; }

int check(const v1::EnginePlan& live) {
    if (v1::Status e = v1::check_engine_plan(g_expected, live, g_ppm)) return report(*e);
    const v1::AdmissionMargins m = v1::admission_margins(g_expected, live, g_ppm);
    std::fprintf(stderr,
                 "radiance: plan check: admitted: claimable %lld >= required %lld bytes (margin %lld); resolved KV %lld "
                 ">= kv_required %lld bytes (margin %lld)\n",
                 (long long)m.claimable, (long long)m.required_total, (long long)(m.claimable - m.required_total),
                 (long long)m.kv_resolved, (long long)m.kv_required, (long long)(m.kv_resolved - m.kv_required));
    std::fflush(stderr);
    return 0;
}

}  // namespace bc_plan
"""

_BC_PLAN_MEMORY_CPP = r"""/* bc_plan_memory.cpp -- see bc_plan_memory.h (BigCherry managed patch rad0011, RSC03, RSC08 A3). */
#include "bc_plan_memory.h"

#include "bc_plan_check.h"
#include "serving_codec.h"

#include "mem/prefix.h"
#include "mem/vram_budget.h"
#include "place/planner.h"
#include "rad_core.h"
#include "sched/geometry.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <memory>
#include <string>

namespace {
namespace v1 = bc::serving::v1;
std::string g_error;
v1::EnginePlan g_plan;
bool g_noted = false;

v1::PlanHostConsumer consumer(const char* name, int64_t bytes, const char* from, const char* remedy) {
    return {name, bytes, v1::Residence::kPinned, from, remedy};
}
}  // namespace

int bc_plan_memory_take_flag(int* argc, char*** argv) { return bc_plan::take_flags(argc, argv); }

void bc_plan_memory_note(const rad::Config& cfg, const std::vector<rad::VramFacts>& facts, const rad::VramBudget& vb,
                         const std::vector<int64_t>& emb_bytes, const std::vector<rad::WeightFootprint>& wfs,
                         int64_t ngram_bytes, int64_t ck_slot, int64_t host_arenas, int64_t kv_one, int64_t ctx,
                         const rad::Program& prog) {
    if (bc_plan::mode() == bc_plan::Mode::none) return;
    /* The checkpoint interval the store is sized with is the resolved chunk geometry's (configure_kv resolves it after
     * plan(); the same function is called here), not the raw flag: it rounds up to the step quantum. */
    rad::ChunkGeometry geo;
    if (rad::chunk_geometry_resolve(prog, cfg, &geo) < 0) {
        g_error = "the chunk geometry could not be resolved (chunk_geometry_resolve)";
        if (bc_plan::mode() == bc_plan::Mode::check) {
            std::fprintf(stderr, "radiance: plan check: %s\n", g_error.c_str());
            std::fflush(nullptr);
            std::_Exit(5);
        }
        return;
    }
    const int64_t checkpoint_interval = geo.checkpoint_interval;
    using rad::TablePlacement;
    v1::EnginePlan p;
    p.plan_version = 1;
    p.producer = "radiance --plan-memory (rad0011)";
    if (cfg.ngram_resolved == TablePlacement::Ram && ngram_bytes > 0)
        p.host.push_back(consumer("ngram_table", ngram_bytes, "engine plan (rad0011: weight_footprint, resolve_table_placement)",
                                  "set engine.ngram_placement = \"disk\""));
    int64_t emb = 0;
    for (int64_t b : emb_bytes) emb += b;
    if (cfg.embedding_resolved == TablePlacement::Ram && emb > 0)
        p.host.push_back(consumer("embedding", emb, "engine plan (rad0011: weight_footprint, resolve_table_placement)", ""));
    if (host_arenas > 0)
        p.host.push_back(consumer("host_arenas", host_arenas, "engine plan (rad0011: arena_host_bytes)", ""));
    int64_t want = 0, floor = 0;
    if (cfg.prefix_cache && ck_slot > 0) {
        const std::unique_ptr<rad::CheckpointPolicy> pol =
            rad::make_checkpoint_policy(cfg.checkpoint_policy, checkpoint_interval);
        want = rad::checkpoint_slots_want(*pol, cfg.max_seqs, ctx, checkpoint_interval);
        floor = std::min<int64_t>(want, rad::kCheckpointSlotsFloor);
    }
    p.checkpoints = {ck_slot, floor, want,
                     "engine plan (rad0011: KVManager::checkpoint_bytes, chunk_geometry_resolve, checkpoint_slots_want)"};
    p.kv_one_session = {kv_one, "engine plan (rad0011: VramFacts::kv_one)"};
    p.vram.claimable = vb.claimable;
    p.vram.statics = vb.statics;
    p.vram.elastic = vb.elastic;
    p.vram.experts = vb.experts;
    p.vram.kv = vb.kv;
    p.vram.stage = vb.stage;
    p.vram.kv_resolved = cfg.vram_kv_mib * int64_t(1024 * 1024);
    p.vram.expert_floor = vb.expert_floor;
    p.vram.kv_ceiling = vb.kv_ceiling;
    p.vram.expert_ceiling = vb.expert_ceiling;
    p.vram.from = "engine plan (rad0011: vram_budget_resolve)";
    for (size_t i = 0; i < facts.size(); ++i) {
        const rad::WeightFootprint& wf = wfs.at(i);
        p.vram.cards.push_back({facts[i].capacity, facts[i].free, wf.experts, wf.expert_unit_max});
    }
    p.host_pool_mib = cfg.host_pool_mib;
    if (bc_plan::mode() == bc_plan::Mode::check) {
        // before any pool is configured (Engine::plan() configures them after this call): RSC08 S4
        if (int rc = bc_plan::check(p)) {
            std::fflush(nullptr);
            std::_Exit(rc);
        }
        return;
    }
    g_plan = std::move(p);
    g_noted = true;
}

int bc_plan_memory_done() {
    if (bc_plan::mode() != bc_plan::Mode::measure) return 0;
    const std::string& out_path = bc_plan::measure_out();
    if (!g_noted) {
        std::fprintf(stderr, "radiance: --plan-memory: the memory plan recorded nothing%s%s\n",
                     g_error.empty() ? "" : ": ", g_error.c_str());
        std::fflush(nullptr);
        std::_Exit(2);
    }
    const std::string tmp = out_path + ".tmp";
    {
        std::ofstream out(tmp, std::ios::binary | std::ios::trunc);
        out << v1::encode(g_plan).dump(2) << '\n';
        if (!out) {
            std::fprintf(stderr, "radiance: --plan-memory: cannot write %s\n", tmp.c_str());
            std::fflush(nullptr);
            std::_Exit(2);
        }
    }
    if (std::rename(tmp.c_str(), out_path.c_str()) != 0) {
        std::fprintf(stderr, "radiance: --plan-memory: cannot replace %s\n", out_path.c_str());
        std::fflush(nullptr);
        std::_Exit(2);
    }
    std::fprintf(stderr, "radiance: --plan-memory: wrote %s; nothing loaded or allocated past the plan\n", out_path.c_str());
    std::fflush(nullptr);
    /* Before weights load and the cache is allocated: nothing past plan() needs tearing down, and the device runtime's
     * exit handlers are not run against a half-built engine. */
    std::_Exit(0);
}
"""

# The shared library comes from the BigCherry tree, never a copy: the lane passes its .cmake (components/serving).
_CORE_LINK = r"""
# BigCherry rad0011 (RSC08 A3): the engine's plan check runs the shared serving manifest v1 library, built here from
# its one source list (BigCherry's components/serving/bc_serving_v1.cmake) with this tree's abi and vendored JSON.
if(NOT DEFINED BC_SERVING_V1_CMAKE)
  message(FATAL_ERROR "rad0011: -DBC_SERVING_V1_CMAKE=<bigcherry>/engines/radiance/components/serving/bc_serving_v1.cmake "
                      "is required (the shared serving manifest v1 library); the BigCherry lane passes it")
endif()
set(BC_SERVING_V1_NLOHMANN_INCLUDE "${CMAKE_SOURCE_DIR}/vendor")
include("${BC_SERVING_V1_CMAKE}")
target_link_libraries(rad_core PUBLIC bc_serving_v1)"""


def _created(path: str, text: str, guard: str, description: str) -> FilePatch:
    return FilePatch(
        path=path,
        create=True,
        description=description,
        edits=(Edit(
            id="create-" + path.split("/")[-1].replace(".", "-").replace("_", "-"),
            anchor=r"\A",
            text=text,
            guard=guard,
            max_span_lines=1,
            expect_matches=1,
            rationale="Created files are anchored at the empty upstream file; create=True rejects any upstream collision.",
        ),),
    )


PATCHES = (
    _created("core/bc_plan_memory.h", _BC_PLAN_MEMORY_H, r"^int bc_plan_memory_done\(\);$",
             "Create the plan-memory extension-point header."),
    _created("core/bc_plan_memory.cpp", _BC_PLAN_MEMORY_CPP, r"^int bc_plan_memory_done\(\) \{$",
             "Create the record of plan()'s sizes as a v1 EnginePlan, the writer and the start-up check call."),
    _created("core/bc_plan_check.h", _BC_PLAN_CHECK_H, r"^int check\(const bc::serving::v1::EnginePlan& live\);$",
             "Create the engine-independent plan flags and check."),
    _created("core/bc_plan_check.cpp", _BC_PLAN_CHECK_CPP, r"^int take_flags\(int\* argc, char\*\*\* argv\) \{$",
             "Create the plan flags, the expected-plan read and the check through bc_serving_v1."),
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
        description="Take the plan flags off the command line before rad0009's manifest reader sees them.",
        edits=(
            Edit(
                id="include-bc-plan-memory",
                anchor=r'^#include "bc_serving_manifest\.h"$',
                text='\n#include "bc_plan_memory.h"',
                guard=r'^#include "bc_plan_memory\.h"$',
                max_span_lines=1,
                expect_matches=1,
                rationale="rad0009's include is main.cpp's unique last include once it applies (rad0011 requires rad0009).",
            ),
            Edit(
                id="bc-plan-memory-take-flag",
                anchor=r"^int main\(int argc, char\*\* argv\) \{$",
                text="\n    if (int rc = bc_plan_memory_take_flag(&argc, &argv)) return rc;",
                guard=r"bc_plan_memory_take_flag\(&argc, &argv\)",
                max_span_lines=1,
                expect_matches=1,
                rationale="Inserted right after main's unique signature, so it runs before rad0009's bc_serving_engine_argv, "
                          "which refuses any argument it does not know.",
            ),
        ),
    ),
    FilePatch(
        path="core/engine_bringup.cpp",
        description="Record (and with --plan-check, check) plan()'s sizes at the end of its VRAM split; with "
                    "--plan-memory, write them and exit after plan().",
        edits=(
            Edit(
                id="include-bc-plan-memory",
                anchor=r'^#include "mem/vram_budget\.h"$',
                text='\n#include "bc_plan_memory.h"',
                guard=r'^#include "bc_plan_memory\.h"$',
                max_span_lines=1,
                expect_matches=1,
                rationale="The VRAM budget header is included once; the plan record sits beside it.",
            ),
            Edit(
                id="bc-plan-memory-note",
                anchor=r"^        RAD_TRY\(vram_budget_resolve\(cfg_, facts, &vb\)\);$",
                text="\n        bc_plan_memory_note(cfg_, facts, vb, emb_bytes, wfs, ngram_bytes, ck_slot, host_arenas, kv_floor, ctx, ranks_[0]->program);",
                guard=r"bc_plan_memory_note\(cfg_, facts, vb,",
                max_span_lines=1,
                expect_matches=1,
                rationale="plan()'s only vram_budget_resolve call: every size plan() computes is final once it returns, "
                          "and the pools are configured after it (r->pools.configure), so the check precedes them.",
            ),
            Edit(
                id="bc-plan-memory-done",
                anchor=r"^    RAD_TRY\(plan\(\)\);$",
                text="\n    RAD_TRY(bc_plan_memory_done());",
                guard=r"RAD_TRY\(bc_plan_memory_done\(\)\);",
                max_span_lines=1,
                expect_matches=1,
                rationale="start()'s only plan() call, before load_weights: with --plan-memory nothing past the plan runs.",
            ),
        ),
    ),
)
