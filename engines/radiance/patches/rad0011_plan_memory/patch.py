"""RSC03: `--plan-memory FILE` writes the engine's own memory plan (Engine::plan()) as strict JSON and exits."""
from bigcherry.patcher import Edit, FilePatch

_BC_PLAN_MEMORY_H = r"""/* bc_plan_memory.h -- BigCherry managed patch rad0011 (RSC03): `--plan-memory FILE`.
 *
 * Start-up runs as far as Engine::plan(), whose sizes are recorded here and written to FILE as strict versioned JSON;
 * the process then exits before any weight loads or the KV cache and arenas are allocated. bc-serve reads the file as a
 * measured fact (components/serving/ledger.h, parse_engine_plan). Without the flag every call here does nothing. */
#pragma once
#include <cstdint>
#include <vector>

namespace rad {
struct Config;
struct VramFacts;
struct VramBudget;
struct WeightFootprint;
}

/* Takes `--plan-memory FILE` off the command line (before rad0009's manifest reader sees it). False, with a message, on
 * a missing or repeated FILE. */
bool bc_plan_memory_take_flag(int* argc, char*** argv);

/* Records what plan() sized. Called once, right after its VRAM split resolves, with plan()'s own values. */
void bc_plan_memory_note(const rad::Config& cfg, const std::vector<rad::VramFacts>& facts, const rad::VramBudget& vb,
                         const std::vector<int64_t>& emb_bytes, const std::vector<rad::WeightFootprint>& wfs,
                         int64_t ngram_bytes, int64_t ck_slot, int64_t host_arenas, int64_t kv_one, int64_t ctx,
                         int64_t checkpoint_interval);

/* After plan(): with --plan-memory, writes the plan and exits (status 0, or 2 when nothing was recorded or FILE cannot be
 * written). Otherwise returns 0 and start-up goes on. */
int bc_plan_memory_done();
"""

_BC_PLAN_MEMORY_CPP = r"""/* bc_plan_memory.cpp -- see bc_plan_memory.h (BigCherry managed patch rad0011, RSC03). */
#include "bc_plan_memory.h"

#include "mem/prefix.h"
#include "mem/vram_budget.h"
#include "place/planner.h"
#include "rad_core.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <memory>
#include <string>

namespace {
std::string g_out;
nlohmann::json g_plan;
bool g_noted = false;

nlohmann::json consumer(const char* name, int64_t bytes, const char* from, const char* remedy) {
    return {{"name", name}, {"bytes", bytes}, {"residence", "pinned"}, {"from", from}, {"remedy", remedy}};
}
}  // namespace

bool bc_plan_memory_take_flag(int* argc, char*** argv) {
    int w = 1;
    for (int i = 1; i < *argc; ++i) {
        if (std::strcmp((*argv)[i], "--plan-memory") != 0) {
            (*argv)[w++] = (*argv)[i];
            continue;
        }
        if (!g_out.empty()) {
            std::fprintf(stderr, "radiance: --plan-memory given twice\n");
            return false;
        }
        if (i + 1 >= *argc || !*(*argv)[i + 1] || (*argv)[i + 1][0] == '-') {
            std::fprintf(stderr, "radiance: --plan-memory needs the file to write the plan to\n");
            return false;
        }
        g_out = (*argv)[++i];
    }
    (*argv)[w] = nullptr;
    *argc = w;
    return true;
}

void bc_plan_memory_note(const rad::Config& cfg, const std::vector<rad::VramFacts>& facts, const rad::VramBudget& vb,
                         const std::vector<int64_t>& emb_bytes, const std::vector<rad::WeightFootprint>& wfs,
                         int64_t ngram_bytes, int64_t ck_slot, int64_t host_arenas, int64_t kv_one, int64_t ctx,
                         int64_t checkpoint_interval) {
    if (g_out.empty()) return;
    using rad::TablePlacement;
    nlohmann::json host = nlohmann::json::array();
    if (cfg.ngram_resolved == TablePlacement::Ram && ngram_bytes > 0)
        host.push_back(consumer("ngram_table", ngram_bytes, "engine plan (rad0011: weight_footprint, resolve_table_placement)",
                                "set engine.ngram_placement = \"disk\""));
    int64_t emb = 0;
    for (int64_t b : emb_bytes) emb += b;
    if (cfg.embedding_resolved == TablePlacement::Ram && emb > 0)
        host.push_back(consumer("embedding", emb, "engine plan (rad0011: weight_footprint, resolve_table_placement)", ""));
    if (host_arenas > 0)
        host.push_back(consumer("host_arenas", host_arenas, "engine plan (rad0011: arena_host_bytes)", ""));

    int64_t want = 0, floor = 0;
    if (cfg.prefix_cache && ck_slot > 0) {
        const std::unique_ptr<rad::CheckpointPolicy> pol =
            rad::make_checkpoint_policy(cfg.checkpoint_policy, checkpoint_interval);
        want = rad::checkpoint_slots_want(*pol, cfg.max_seqs, ctx, checkpoint_interval);
        floor = std::min<int64_t>(want, rad::kCheckpointSlotsFloor);
    }
    nlohmann::json cards = nlohmann::json::array();
    for (size_t i = 0; i < facts.size(); ++i) {
        const rad::WeightFootprint& wf = wfs.at(i);
        cards.push_back({{"capacity", facts[i].capacity}, {"free", facts[i].free},
                         {"engine_expert_plane", wf.experts}, {"expert_unit_max", wf.expert_unit_max}});
    }
    g_plan = {
        {"plan_version", 1},
        {"producer", "radiance --plan-memory (rad0011)"},
        {"host", host},
        {"checkpoints", {{"slot_bytes", ck_slot}, {"floor_slots", floor}, {"want_slots", want},
                         {"from", "engine plan (rad0011: KVManager::checkpoint_bytes, checkpoint_slots_want)"}}},
        {"kv_one_session", {{"bytes", kv_one}, {"from", "engine plan (rad0011: VramFacts::kv_one)"}}},
        {"vram", {{"claimable", vb.claimable}, {"statics", vb.statics}, {"elastic", vb.elastic},
                  {"experts", vb.experts}, {"kv", vb.kv}, {"stage", vb.stage}, {"expert_floor", vb.expert_floor},
                  {"kv_ceiling", vb.kv_ceiling}, {"expert_ceiling", vb.expert_ceiling},
                  {"from", "engine plan (rad0011: vram_budget_resolve)"}, {"cards", cards}}},
        {"host_pool_mib", cfg.host_pool_mib},
    };
    g_noted = true;
}

int bc_plan_memory_done() {
    if (g_out.empty()) return 0;
    if (!g_noted) {
        std::fprintf(stderr, "radiance: --plan-memory: the memory plan recorded nothing\n");
        std::fflush(nullptr);
        std::_Exit(2);
    }
    const std::string tmp = g_out + ".tmp";
    {
        std::ofstream out(tmp, std::ios::binary | std::ios::trunc);
        out << g_plan.dump(2) << '\n';
        if (!out) {
            std::fprintf(stderr, "radiance: --plan-memory: cannot write %s\n", tmp.c_str());
            std::fflush(nullptr);
            std::_Exit(2);
        }
    }
    if (std::rename(tmp.c_str(), g_out.c_str()) != 0) {
        std::fprintf(stderr, "radiance: --plan-memory: cannot replace %s\n", g_out.c_str());
        std::fflush(nullptr);
        std::_Exit(2);
    }
    std::fprintf(stderr, "radiance: --plan-memory: wrote %s; nothing loaded or allocated past the plan\n", g_out.c_str());
    std::fflush(nullptr);
    /* Before weights load and the cache is allocated: nothing past plan() needs tearing down, and the device runtime's
     * exit handlers are not run against a half-built engine. */
    std::_Exit(0);
}
"""

PATCHES = (
    FilePatch(
        path="core/bc_plan_memory.h",
        create=True,
        description="Create the --plan-memory extension-point header.",
        edits=(Edit(
            id="create-bc-plan-memory-h",
            anchor=r"\A",
            text=_BC_PLAN_MEMORY_H,
            guard=r"^int bc_plan_memory_done\(\);$",
            max_span_lines=1,
            expect_matches=1,
            rationale="Created files are anchored at the empty upstream file; create=True rejects any upstream collision.",
        ),),
    ),
    FilePatch(
        path="core/bc_plan_memory.cpp",
        create=True,
        description="Create the --plan-memory flag, the record of plan()'s sizes and the JSON writer.",
        edits=(Edit(
            id="create-bc-plan-memory-cpp",
            anchor=r"\A",
            text=_BC_PLAN_MEMORY_CPP,
            guard=r"^int bc_plan_memory_done\(\) \{$",
            max_span_lines=1,
            expect_matches=1,
            rationale="Created files are anchored at the empty upstream file; create=True rejects any upstream collision.",
        ),),
    ),
    FilePatch(
        path="core/main.cpp",
        description="Take --plan-memory FILE off the command line before rad0009's manifest reader sees it.",
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
                text="\n    if (!bc_plan_memory_take_flag(&argc, &argv)) return 2;",
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
        description="Record plan()'s sizes at the end of its VRAM split, and write them and exit after plan().",
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
                text="\n        bc_plan_memory_note(cfg_, facts, vb, emb_bytes, wfs, ngram_bytes, ck_slot, host_arenas, kv_floor, ctx, geo_.checkpoint_interval);",
                guard=r"bc_plan_memory_note\(cfg_, facts, vb,",
                max_span_lines=1,
                expect_matches=1,
                rationale="plan()'s only vram_budget_resolve call: every size plan() computes is final once it returns.",
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
