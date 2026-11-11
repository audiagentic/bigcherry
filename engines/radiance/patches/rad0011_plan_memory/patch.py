"""RSC03 / RSC08 A3: the engine's memory plan as the serving manifest v1 EnginePlan, measured (`--plan-memory FILE`) or
checked at start-up against the expected one in the manifest the engine was started from (rad0009's
bc_serving_expected_plan, from engine_view), through the shared v1 library bc_serving_v2 (components/serving), which
radiance's core links."""
from bigcherry.patcher import Edit, FilePatch

_BC_PLAN_MEMORY_H = r"""/* bc_plan_memory.h -- BigCherry managed patch rad0011 (RSC03, RSC08 A3): the engine's memory plan.
 *
 * Engine::plan() records its sizes here as a serving manifest v1 EnginePlan (components/serving, bc_serving_v2).
 *   --plan-memory FILE                          start-up stops after plan(): FILE gets the plan (v1 JSON, written by
 *                                               the generated encoder) and the process exits before any weight loads.
 *   (a start from a measured manifest)         the manifest's own expected plan and kv_backing_floor_ppm (engine_view,
 *                                               rad0009) are the check: right after the VRAM split, before any pool
 *                                               is configured, the live plan is checked against them (spec section
 *                                               5.2) and the result printed once; a refusal exits with its exit code.
 * There is no flag that supplies an expected plan: only the manifest does. A start without --manifest or --plan-memory
 * does nothing here. */
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

/* Takes --plan-memory off the command line (after rad0009's manifest reader has run: it hands --plan-memory on) and arms the
 * check when rad0009 holds an expected plan. 0, or the exit code of a usage fault, which has been printed once. */
int bc_plan_memory_take_flag(int* argc, char*** argv);

/* Records what plan() sized, and started from a measured manifest checks it. Called once, right after its VRAM split resolves, with
 * plan()'s own values. A refused check exits here. */
void bc_plan_memory_note(const rad::Config& cfg, const std::vector<rad::VramFacts>& facts, const rad::VramBudget& vb,
                         const std::vector<int64_t>& emb_bytes, const std::vector<rad::WeightFootprint>& wfs,
                         int64_t ngram_bytes, int64_t ck_slot, int64_t host_arenas, int64_t kv_one, int64_t ctx,
                         const rad::Program& prog);
/* Publish rank demand only after the rank split, before configuring pools. */
void bc_plan_memory_note_ranks(const std::vector<rad::Config>& rank_cfg);

/* After plan(): with --plan-memory, writes the plan and exits (status 0, or 2 when nothing was recorded or FILE cannot be
 * written). Otherwise returns 0 and start-up goes on. */
int bc_plan_memory_done();
"""

_BC_PLAN_CHECK_H = r"""/* bc_plan_check.h -- BigCherry managed patch rad0011 (RSC08 A3): the engine-independent half of the plan flags. No
 * Radiance type is used, so it is tested card-free (tools/tests/patch/test_radiance_patch_packages.py). */
#pragma once
#include "serving_types.h"

#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace bc_plan {

enum class Mode { none, measure, check };

/* Takes --plan-memory FILE off argv, and arms the check with rad0009's expected plan and floor when the engine was started
 * from a measured manifest (the floor is checked by the shared validator). A measuring start and a start from a measured
 * manifest are different modes; rad0009 has already refused a measuring start with a measured manifest and a serving
 * start with an unmeasured one. 0, or the exit code of the fault (errors.md section 5), printed once as
 * "radiance: config: ...". */
int take_flags(int* argc, char*** argv);
/* Arms the check with an expected plan and floor (take_flags does it from rad0009's; the card-free test from the admission vectors). */
int arm(const bc::serving::v2::EnginePlan& expected, int64_t floor_ppm);
/* MemAvailable in kB (/proc/meminfo; 0 when unreadable): at take_flags() (the process's first line) and now. */
long long mem_available_kb_at_start();
long long mem_available_kb_now();
Mode mode();
const std::string& measure_out();
const std::string& rank_measure_out();

/* With a manifest's expected plan: checks `live` against the expected plan (bc::serving::v2::check_engine_plan) and prints the one
 * start-up line, admitted with its margins or the refusal. 0 when admitted, else the refusal's exit code. */
int check(const bc::serving::v2::EnginePlan& live);

/* Re-evaluate live host room only at a terminal pre-weight admission.
 * Driver-only failures are not classified as a capacity shortage. */
int report_host_room_if_short(std::string_view resource, int64_t headroom_bytes,
                             const std::vector<std::pair<std::string, int64_t>>& components);
int report_exact_pool_shortfall(int64_t available_bytes, int64_t required_bytes);

}  // namespace bc_plan
"""

_BC_PLAN_CHECK_CPP = r"""/* bc_plan_check.cpp -- see bc_plan_check.h (BigCherry managed patch rad0011, RSC08 A3). */
#include "bc_plan_check.h"

#include "admission.h"
#include "bc_serving_manifest.h"
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
namespace v2 = bc::serving::v2;
Mode g_mode = Mode::none;
std::string g_out;
std::string g_rank_out;
v2::EnginePlan g_expected;
int64_t g_ppm = 0;
std::vector<std::string> g_bdfs;

int report(const v2::ConfigError& e) {
    if (e.room_refusal)
        std::fprintf(stderr, "BC_SERVING_ADMISSION_REFUSAL=%s\n",
                     v2::room_refusal_json(*e.room_refusal).c_str());
    std::fputs(v2::render_human("radiance", e).c_str(), stderr);
    std::fflush(stderr);
    return v2::exit_code_of(e);
}

int usage(const std::string& flag, const std::string& message) {
    return report(v2::make_error(v2::ConfigErrorCode::invalid_value, flag, message, "--plan-memory FILE alone"));
}
}  // namespace

namespace {
long long g_available_at_start_kb = 0;
}  // namespace

long long mem_available_kb_now() {
    std::ifstream meminfo("/proc/meminfo");
    for (std::string line; std::getline(meminfo, line);)
        if (line.rfind("MemAvailable:", 0) == 0) return std::atoll(line.c_str() + 13);
    return 0;
}

long long mem_available_kb_at_start() { return g_available_at_start_kb; }

int arm(const v2::EnginePlan& expected, int64_t floor_ppm) {
    if (v2::Status e = v2::check_kv_backing_floor_ppm(floor_ppm)) return report(*e);
    g_expected = expected;
    g_ppm = floor_ppm;
    g_bdfs = bc_serving_engine_bdfs();
    g_mode = Mode::check;
    return 0;
}

int take_flags(int* argc, char*** argv) {
    g_mode = Mode::none;   // each call starts from nothing, so the card-free test can run many scenarios in one process
    g_available_at_start_kb = mem_available_kb_now();
    g_out.clear();
    g_rank_out.clear();
    g_expected = v2::EnginePlan{};
    g_ppm = 0;
    g_bdfs.clear();
    int w = 1;
    for (int i = 1; i < *argc; ++i) {
        const char* a = (*argv)[i];
        if (std::strcmp(a, "--plan-memory-ranks") == 0) {
            if (!g_rank_out.empty()) return usage(a, "given twice");
            if (i + 1 >= *argc || !*(*argv)[i + 1]) return usage(a, "needs a value");
            g_rank_out = (*argv)[++i];
            continue;
        }
        if (std::strcmp(a, "--plan-memory") != 0) {
            (*argv)[w++] = (*argv)[i];
            continue;
        }
        if (!g_out.empty()) return usage(a, std::string(a) + " given twice");
        if (i + 1 >= *argc || !*(*argv)[i + 1]) return usage(a, std::string(a) + " needs a value");
        g_out = (*argv)[++i];
    }
    (*argv)[w] = nullptr;
    *argc = w;
    if (!g_rank_out.empty() && g_out.empty()) return usage("--plan-memory-ranks", "requires --plan-memory");
    if (!g_out.empty()) {
        g_mode = Mode::measure;
        return 0;
    }
    // a start from a measured manifest: its plan is the one to admit (rad0009 holds it; nothing else supplies one)
    v2::EnginePlan expected;
    int64_t floor_ppm = 0;
    if (bc_serving_expected_plan(&expected, &floor_ppm)) return arm(expected, floor_ppm);
    return 0;
}

Mode mode() { return g_mode; }
const std::string& measure_out() { return g_out; }
const std::string& rank_measure_out() { return g_rank_out; }

int report_host_room_if_short(std::string_view resource, int64_t headroom_bytes,
                             const std::vector<std::pair<std::string, int64_t>>& components) {
    if (g_mode != Mode::check) return 0;
    const long long kb = mem_available_kb_now();
    if (kb <= 0 || kb > (9007199254740991LL / 1024)) return 0;
    if (v2::Status e = v2::check_host_room(resource, kb * 1024, headroom_bytes, components))
        if (e->room_refusal) return report(*e);
    return 0;
}

int report_exact_pool_shortfall(int64_t available_bytes, int64_t required_bytes) {
    if (g_mode != Mode::check || available_bytes < 0 || required_bytes <= available_bytes) return 0;
    if (v2::Status e = v2::check_host_room("host_pool", available_bytes, 0,
                                          {{"host_pool", required_bytes}}))
        if (e->room_refusal) return report(*e);
    return 0;
}

int check(const v2::EnginePlan& live) {
    std::string binding_bdf;
    if (g_bdfs.size() == 1) {
        binding_bdf = g_bdfs.front();
    } else if (g_bdfs.size() == live.vram.cards.size() && !g_bdfs.empty()) {
        size_t limiting = 0;
        for (size_t i = 1; i < live.vram.cards.size(); ++i)
            if (live.vram.cards[i].free < live.vram.cards[limiting].free) limiting = i;
        binding_bdf = g_bdfs[limiting];
    }
    if (v2::Status e = v2::check_engine_plan(g_expected, live, g_ppm, binding_bdf)) return report(*e);
    const v2::AdmissionMargins m = v2::admission_margins(g_expected, live, g_ppm);
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
#include "bc_serving_manifest.h"
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
namespace v2 = bc::serving::v2;
std::string g_error;
v2::EnginePlan g_plan;
bool g_noted = false;
nlohmann::json g_rank_gpu = nlohmann::json::array();
bool g_rank_noted = false;
std::vector<rad::VramFacts> g_rank_facts;
int64_t g_rank_stage = 0;
long long g_taken_kb = 0;   /* MemAvailable fallen since this process's first line, at the end of the VRAM split (before any pool) */

v2::PlanHostConsumer consumer(const char* name, int64_t bytes, const char* from, const char* remedy) {
    return {name, bytes, v2::Residence::kPinned, from, remedy};
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
    v2::EnginePlan p;
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
    if (!bc_plan::rank_measure_out().empty()) {
        g_rank_facts = facts;
        g_rank_stage = vb.stage;
    }
    if (bc_plan::mode() == bc_plan::Mode::check) {
        // before any pool is configured (Engine::plan() configures them after this call): RSC08 S4
        if (int rc = bc_plan::check(p)) {
            std::fflush(nullptr);
            std::_Exit(rc);
        }
        return;
    }
    /* What start-up took from the host up to here, before the pools, the n-gram table and the checkpoint store are allocated (the
     * same point the engine's own host-room checks run at): the fall of MemAvailable since this process's first line. It counts
     * only what the host could not give back (anonymous, pinned and shared memory; page cache and mapped files are reclaimable
     * and do not move it). The host ledger counts it against MemAvailable beside the helpers' and the pools' (bc-serve
     * --measured-plan --engine-resident-mib). */
    g_taken_kb = bc_plan::mem_available_kb_at_start() - bc_plan::mem_available_kb_now();
    g_plan = std::move(p);
    g_noted = true;
}

void bc_plan_memory_note_ranks(const std::vector<rad::Config>& rank_cfg) {
    /* Capture each rank's split only after vram_budget_rank_split. */
    if (!bc_plan::rank_measure_out().empty()) {
        const auto bdfs = bc_serving_engine_bdfs();
        if (bdfs.size() != g_rank_facts.size() || rank_cfg.size() != g_rank_facts.size() ||
            g_rank_facts.empty()) {
            g_error = "P1 rank/BDF/VRAM g_rank_facts do not match the measured layout";
            return;
        }
        constexpr int64_t wire_max = 9007199254740991LL; /* I-JSON integer limit */
        constexpr int64_t mib = int64_t(1) << 20;
        auto safe = [wire_max](__int128 n) { return n >= 0 && n <= wire_max; };
        nlohmann::json gpu = nlohmann::json::array();
        for (size_t i = 0; i < g_rank_facts.size(); ++i) {
            const auto& f = g_rank_facts[i];
            const auto& c = rank_cfg[i];
            const __int128 overhead = (__int128)f.arena + f.staging +
                (__int128)f.n_queues * f.queue_bytes + g_rank_stage;
            const __int128 fixed = overhead + f.statics;
            const __int128 required = overhead + (__int128)mib *
                (c.vram_weights_mib + (__int128)c.vram_kv_mib);
            if (bdfs[i].empty() || !safe(overhead) || !safe(fixed) ||
                !safe(required) || required <= 0 || fixed > required || !safe(f.free)) {
                g_error = "P1 rank demand is missing, overflows or disagrees with fixed VRAM";
                return;
            }
            gpu.push_back({{"bdf", bdfs[i]}, {"required_bytes", (int64_t)required},
                           {"fixed_bytes", (int64_t)fixed},
                           {"claimable_bytes", f.free},
                           {"evidence", "rad0011 rank_cfg and VramFacts after rank split"}});
        }
        g_rank_gpu = std::move(gpu);
        g_rank_noted = true;
    }
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
        out << v2::encode(g_plan).dump(2) << '\n';
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
    if (!bc_plan::rank_measure_out().empty()) {
        if (!g_rank_noted || g_taken_kb < 0 || g_taken_kb > 9007199254740991LL / 1024) {
            std::fprintf(stderr, "radiance: --plan-memory-ranks: %s\n",
                         g_error.empty() ? "rank measurement missing" : g_error.c_str());
            std::fflush(nullptr);
            std::_Exit(2);
        }
        const std::string rank_path = bc_plan::rank_measure_out();
        const std::string rank_tmp = rank_path + ".tmp";
        std::ofstream out(rank_tmp, std::ios::binary | std::ios::trunc);
        out << nlohmann::json{{"schema_version", 1}, {"kind", "radiance_p1_rank_demand"},
                              {"gpu", g_rank_gpu},
                              {"resident_other_bytes", g_taken_kb * 1024}}.dump(2) << '\n';
        out.close();
        if (!out || std::rename(rank_tmp.c_str(), rank_path.c_str()) != 0) {
            std::fprintf(stderr, "radiance: --plan-memory-ranks: cannot publish %s\n", rank_path.c_str());
            std::fflush(nullptr);
            std::_Exit(2);
        }
    }
    const long long taken_kb = g_taken_kb;   /* at least 1 MiB below */
    std::fprintf(stderr, "radiance: --plan-memory: wrote %s; engine start-up took %lld MiB of MemAvailable; nothing loaded or allocated past the plan\n",
                 out_path.c_str(), std::max<long long>(1, (taken_kb + 1023) / 1024));
    std::fflush(nullptr);
    /* Before weights load and the cache is allocated: nothing past plan() needs tearing down, and the device runtime's
     * exit handlers are not run against a half-built engine. */
    std::_Exit(0);
}
"""

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
    _created("core/bc_plan_check.h", _BC_PLAN_CHECK_H, r"^int check\(const bc::serving::v2::EnginePlan& live\);$",
             "Create the engine-independent plan flags and check."),
    _created("core/bc_plan_check.cpp", _BC_PLAN_CHECK_CPP, r"^int take_flags\(int\* argc, char\*\*\* argv\) \{$",
             "Create the plan flags, the expected-plan read and the check through bc_serving_v2."),
    FilePatch(
        path="core/main.cpp",
        description="Take --plan-memory off the command line after rad0009's manifest reader has run and arm the check.",
        edits=(
            Edit(
                id="include-bc-plan-memory",
                anchor=r'^#include "bc_serving_manifest\.h"$',
                text='\n#include "bc_plan_memory.h"\n#include "bc_plan_check.h"',
                guard=r'^#include "bc_plan_memory\.h"$',
                max_span_lines=1,
                expect_matches=1,
                rationale="rad0009's include is main.cpp's unique last include once it applies (rad0011 requires rad0009).",
            ),
            Edit(
                id="bc-plan-memory-take-flag",
                anchor=r"^    if \(!bc_serving_engine_argv\(&argc, &argv\)\) return 2;$",
                text="\n    if (int rc = bc_plan_memory_take_flag(&argc, &argv)) return rc;",
                guard=r"bc_plan_memory_take_flag\(&argc, &argv\)",
                max_span_lines=1,
                expect_matches=1,
                rationale="Inserted right after rad0009's manifest call (rad0011 requires rad0009): the manifest reader decides "
                          "measuring or serving from --plan-memory and hands the flag on, and this call then takes it off and "
                          "arms the check with the manifest's plan. Run before it, it would strip --plan-memory first and a "
                          "measuring start would read its own unmeasured manifest as a serving start.",
            ),
        ),
    ),
    FilePatch(
        path="core/engine_bringup.cpp",
        description="Record (and, started from a measured manifest, check) plan()'s sizes at the end of its VRAM split; with "
                    "--plan-memory, write them and exit after plan().",
        edits=(
            Edit(
                id="include-bc-plan-memory",
                anchor=r'^#include "mem/vram_budget\.h"$',
                text='\n#include "bc_plan_memory.h"\n#include "bc_plan_check.h"',
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
                rationale="Record the v1 plan while its measured facts are in scope, before the rank split and before any pool.",
            ),
            Edit(
                id="bc-plan-memory-ranks-post-split",
                anchor=r"^        const Config& rc = rank_cfg_\[ri\];$",
                text="\n        if (ri == 0) bc_plan_memory_note_ranks(rank_cfg_);",
                guard=r"if \(ri == 0\) bc_plan_memory_note_ranks\(rank_cfg_\);",
                max_span_lines=1,
                expect_matches=1,
                rationale="Unique rank config access occurs after the rank split and before any per-rank pool is configured.",
            ),
            Edit(
                id="s5-exact-host-pool-shortfall",
                anchor=r"^    for \(const Shortfall& f : pools.shortfalls\(\)\).*$",
                text=("\n    for (const Shortfall& f : pools.shortfalls()) {"
                      "\n        if (f.pool != \"host_pool\" || !f.refused.empty()) continue;"
                      "\n        if (int rc = bc_plan::report_exact_pool_shortfall(f.available, f.wanted)) {"
                      "\n            std::fflush(nullptr);"
                      "\n            std::_Exit(rc);"
                      "\n        }"
                      "\n    }"),
                guard=r'^    for \(const Shortfall& f : pools.shortfalls\(\)\) \{$',
                max_span_lines=1, expect_matches=1,
                rationale="A measured pool reservation has exactly wanted and available, unlike an opaque driver refusal.",
            ),
            Edit(
                id="s5-embedding-terminal-host-room",
                anchor=r"^            if \(emb_asked\) \{$",
                text=('\n                if (int rc = bc_plan::report_host_room_if_short("host_ram", 0,'
     '\n                    {{"pending_pools_and_host_kv", pools_still}, {"embedding", emb_pin}})) {'
     ' std::fflush(nullptr); std::_Exit(rc); }'),
                guard=r"pools_still\}, \{[^\n}]+, emb_pin\}\}",
                max_span_lines=1, expect_matches=1,
                rationale="Only an explicit RAM embedding that cannot fit is terminal; default placement may fall back.",
            ),
            Edit(
                id="s5-ngram-terminal-host-room",
                anchor=r"^            if \(!room\) \{$",
                text=('\n                if (int rc = bc_plan::report_host_room_if_short("host_ram", 0,'
     '\n                    {{"pending_pools_and_host_kv", still}, {"ngram", need}})) {'
     ' std::fflush(nullptr); std::_Exit(rc); }'),
                guard=r"still\}, \{[^\n}]+, need\}\}",
                max_span_lines=1, expect_matches=1,
                rationale="An explicit RAM n-gram refusal reports a measured shortage, not a guessed RLIMIT/copy failure.",
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
