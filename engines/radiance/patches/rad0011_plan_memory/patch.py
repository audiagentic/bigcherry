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
class Program;
}

/* Takes `--plan-memory FILE` off the command line (before rad0009's manifest reader sees it). False, with a message, on
 * a missing or repeated FILE. */
bool bc_plan_memory_take_flag(int* argc, char*** argv);

/* Records what plan() sized. Called once, right after its VRAM split resolves, with plan()'s own values. */
void bc_plan_memory_note(const rad::Config& cfg, const std::vector<rad::VramFacts>& facts, const rad::VramBudget& vb,
                         const std::vector<int64_t>& emb_bytes, const std::vector<rad::WeightFootprint>& wfs,
                         int64_t ngram_bytes, int64_t ck_slot, int64_t host_arenas, int64_t kv_one, int64_t ctx,
                         const rad::Program& prog);

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
#include "sched/geometry.h"

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
std::string g_error;
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
                         const rad::Program& prog) {
    if (g_out.empty()) return;
    /* The checkpoint interval the store is sized with is the resolved chunk geometry's (configure_kv resolves it after
     * plan(); the same function is called here), not the raw flag: it rounds up to the step quantum. */
    rad::ChunkGeometry geo;
    if (rad::chunk_geometry_resolve(prog, cfg, &geo) < 0) {
        g_error = "the chunk geometry could not be resolved (chunk_geometry_resolve)";
        return;
    }
    const int64_t checkpoint_interval = geo.checkpoint_interval;
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
                         {"from", "engine plan (rad0011: KVManager::checkpoint_bytes, chunk_geometry_resolve, checkpoint_slots_want)"}}},
        {"kv_one_session", {{"bytes", kv_one}, {"from", "engine plan (rad0011: VramFacts::kv_one)"}}},
        {"vram", {{"claimable", vb.claimable}, {"statics", vb.statics}, {"elastic", vb.elastic},
                  {"experts", vb.experts}, {"kv", vb.kv}, {"stage", vb.stage},
                  {"kv_resolved", cfg.vram_kv_mib * int64_t(1024 * 1024)}, {"expert_floor", vb.expert_floor},
                  {"kv_ceiling", vb.kv_ceiling}, {"expert_ceiling", vb.expert_ceiling},
                  {"from", "engine plan (rad0011: vram_budget_resolve)"}, {"cards", cards}}},
        {"host_pool_mib", cfg.host_pool_mib},
    };
    g_noted = true;
}

int bc_plan_memory_done() {
    if (g_out.empty()) return 0;
    if (!g_noted) {
        std::fprintf(stderr, "radiance: --plan-memory: the memory plan recorded nothing%s%s\n",
                     g_error.empty() ? "" : ": ", g_error.c_str());
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

_BC_PLAN_MEMORY_CHECK_H = r"""/* bc_plan_memory_check.h -- BigCherry managed patch rad0011 (RSC03): the start-up check of a live memory plan
 * against the plan the serving manifest was derived from. A pure function of two plan documents (rad0011's JSON,
 * plan_version 1), so the manifest only has to supply the expected one. */
#pragma once
#include <cstdint>
#include <string>

#include <nlohmann/json.hpp>

/* True when nothing the live plan needs exceeds what the expected plan provided for. Otherwise false, with `why` naming
 * the first consumer that grew, the two byte counts, where the live number came from and what to do. Checked: every
 * host consumer (one the expected plan does not have counts as grown from 0), the checkpoint slot and floor, one
 * full-length session's KV, the static weights, the prefill staging, and the card's capacity (which must not shrink).
 * VRAM (serving manifest v1 spec §5.2, amendment 5): the stable facts must equal the plan's (statics, expert floor and
 * ceiling, KV ceiling, each card's expert-plane geometry; a card's capacity must not shrink). Claimable, expert
 * residency, the staging region and the KV are live terms: the engine derives them from what is claimable now, so they
 * are never compared with the measured plan. Admission then requires, in order: live expert residency >= the expert
 * floor; live claimable >= statics + live experts + live stage + kv_required; and the resolved KV the engine allocates
 * (kv_resolved: vram_kv_mib after the staging split and MiB rounding) >= kv_required, where kv_required =
 * max(one full-length session's KV, ceil(kv_ceiling * kv_backing_floor_ppm / 1000000)) and `kv_backing_floor_ppm` is the
 * selected calibration entry kv_backing_floor.fraction_ppm (0 when none is selected; [0, 1000000) or refused). Every
 * VRAM term read must be a byte count in [0, 2^53 - 1]. docs/design/serving-manifest-v1/admission-vectors.json holds the
 * cases this function must reproduce. */
bool bc_plan_memory_check(const nlohmann::json& expected, const nlohmann::json& live, int64_t kv_backing_floor_ppm,
                          std::string* why);
"""

_BC_PLAN_MEMORY_CHECK_CPP = r"""/* bc_plan_memory_check.cpp -- see bc_plan_memory_check.h (BigCherry managed patch rad0011, RSC03). */
#include "bc_plan_memory_check.h"

#include <algorithm>
#include <cstdint>
#include <string>

namespace {
constexpr const char* kMeasure = "measure the plan again (bc-serve --plan-memory) and resolve a new manifest";

int64_t num(const nlohmann::json& j, const char* key) {
    const auto it = j.find(key);
    return it != j.end() && it->is_number_integer() ? it->get<int64_t>() : -1;
}

/* `verb` names the comparison and its basis, e.g. "exceed the planned" or "fall short of the required". */
bool fail(std::string* why, const std::string& what, int64_t live, int64_t bound, const std::string& from,
          const std::string& remedy, const char* verb) {
    if (why)
        *why = what + ": live " + std::to_string(live) + " bytes " + verb + " " + std::to_string(bound) +
               " bytes (" + from + "); " + (remedy.empty() ? std::string(kMeasure) : remedy);
    return false;
}
}  // namespace

bool bc_plan_memory_check(const nlohmann::json& expected, const nlohmann::json& live, int64_t kv_backing_floor_ppm,
                          std::string* why) {
    for (const auto* doc : {&expected, &live})
        if (!doc->is_object() || num(*doc, "plan_version") != 1) {
            if (why) *why = "plan check: both plans must be plan_version 1 documents";
            return false;
        }
    for (const auto& c : live.at("host")) {
        const std::string name = c.at("name").get<std::string>();
        int64_t planned = 0;
        for (const auto& e : expected.at("host"))
            if (e.at("name") == name) planned = num(e, "bytes");
        if (num(c, "bytes") > planned)
            return fail(why, "host consumer " + name, num(c, "bytes"), planned, c.at("from").get<std::string>(),
                        c.at("remedy").get<std::string>(), "exceed the planned");
    }
    const auto& lc = live.at("checkpoints");
    const auto& ec = expected.at("checkpoints");
    if (num(lc, "slot_bytes") > num(ec, "slot_bytes"))
        return fail(why, "checkpoint slot", num(lc, "slot_bytes"), num(ec, "slot_bytes"), lc.at("from").get<std::string>(), "", "exceed the planned");
    if (num(lc, "floor_slots") * num(lc, "slot_bytes") > num(ec, "floor_slots") * num(ec, "slot_bytes"))
        return fail(why, "checkpoint floor", num(lc, "floor_slots") * num(lc, "slot_bytes"),
                    num(ec, "floor_slots") * num(ec, "slot_bytes"), lc.at("from").get<std::string>(), "", "exceed the planned");
    const auto& lk = live.at("kv_one_session");
    if (num(lk, "bytes") > num(expected.at("kv_one_session"), "bytes"))
        return fail(why, "one full-length session's KV", num(lk, "bytes"), num(expected.at("kv_one_session"), "bytes"),
                    lk.at("from").get<std::string>(), "", "exceed the planned");
    const auto& lv = live.at("vram");
    const auto& ev = expected.at("vram");
    if (kv_backing_floor_ppm < 0 || kv_backing_floor_ppm >= 1000000) {
        if (why) *why = "plan check: kv_backing_floor_ppm must be in [0, 1000000)";
        return false;
    }
    /* Every term is a v1 byte count in [0, 2^53 - 1] (a missing one reads -1): the ceiling below is taken by quotient
     * and remainder so no intermediate leaves int64 (q * ppm < 2^53, r * ppm < 10^12), and a sum of four terms stays
     * below 2^55. */
    constexpr int64_t kMaxBytes = (int64_t(1) << 53) - 1;
    const int64_t kv_one = num(expected.at("kv_one_session"), "bytes");
    bool in_range = kv_one >= 0 && kv_one <= kMaxBytes;
    for (const char* k : {"statics", "experts", "stage", "kv_ceiling", "expert_floor", "expert_ceiling"})
        in_range = in_range && num(ev, k) >= 0 && num(ev, k) <= kMaxBytes;
    for (const char* k : {"claimable", "statics", "experts", "stage", "kv_resolved", "kv_ceiling", "expert_floor",
                          "expert_ceiling"})
        in_range = in_range && num(lv, k) >= 0 && num(lv, k) <= kMaxBytes;
    if (!in_range) {
        if (why) *why = "plan check: the VRAM terms must be byte counts in [0, 2^53 - 1]";
        return false;
    }
    /* Stable facts: what the weights and the model's geometry fix, independent of what is claimable. */
    for (const char* k : {"statics", "expert_floor", "expert_ceiling", "kv_ceiling"})
        if (num(lv, k) != num(ev, k))
            return fail(why, std::string("VRAM ") + k, num(lv, k), num(ev, k), lv.at("from").get<std::string>(), "",
                        "differ from the planned");
    const auto& lcards = lv.at("cards");
    const auto& ecards = ev.at("cards");
    if (lcards.size() != ecards.size()) {
        if (why) *why = "plan check: the live plan has " + std::to_string(lcards.size()) + " card(s), the manifest's " +
                        std::to_string(ecards.size()) + "; " + kMeasure;
        return false;
    }
    for (size_t i = 0; i < lcards.size(); ++i) {
        const std::string card = "card " + std::to_string(i) + " ";
        if (num(lcards[i], "capacity") < num(ecards[i], "capacity"))
            return fail(why, card + "capacity", num(lcards[i], "capacity"), num(ecards[i], "capacity"),
                        lv.at("from").get<std::string>(), "", "fall short of the planned");
        for (const char* k : {"engine_expert_plane", "expert_unit_max"})
            if (num(lcards[i], k) != num(ecards[i], k))
                return fail(why, card + k, num(lcards[i], k), num(ecards[i], k), lv.at("from").get<std::string>(), "",
                            "differ from the planned");
    }
    /* Admission, on the live split: expert residency, claimable and the resolved KV (amendment 5). The measured kv share
     * is never an input: it is the remainder of the elastic plane, so the measured split equals claimable with no
     * margin, and a reading 8 KiB lower would refuse the start. */
    const int64_t kv_ceiling = num(ev, "kv_ceiling");
    const int64_t floor = kv_ceiling / 1000000 * kv_backing_floor_ppm +
                          (kv_ceiling % 1000000 * kv_backing_floor_ppm + 999999) / 1000000;
    const int64_t kv_required = std::max<int64_t>(kv_one, floor);
    const std::string from = lv.at("from").get<std::string>();
    const std::string kRemedy = "free the engine card, or choose a smaller service intent, and resolve a new manifest";
    if (num(lv, "experts") < num(ev, "expert_floor"))
        return fail(why, "admission: expert residency", num(lv, "experts"), num(ev, "expert_floor"),
                    from + "; the experts the pinned host pool cannot hold", kRemedy + " (or raise the host pool)",
                    "fall short of the required");
    const int64_t total = num(ev, "statics") + num(lv, "experts") + num(lv, "stage") + kv_required;
    if (num(lv, "claimable") < total)
        return fail(why, "admission: claimable VRAM", num(lv, "claimable"), total,
                    from + "; statics " + std::to_string(num(ev, "statics")) + " + experts " +
                        std::to_string(num(lv, "experts")) + " + stage " + std::to_string(num(lv, "stage")) +
                        " + kv_required " + std::to_string(kv_required),
                    kRemedy, "fall short of the required");
    if (num(lv, "kv_resolved") < kv_required)
        return fail(why, "admission: resolved KV", num(lv, "kv_resolved"), kv_required,
                    from + "; kv_required " + std::to_string(kv_required) +
                        ", the cache the engine allocates after the staging split and MiB rounding",
                    kRemedy, "fall short of the required");
    return true;
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
        path="core/bc_plan_memory_check.h",
        create=True,
        description="Create the start-up check of a live memory plan against the manifest's expected plan.",
        edits=(Edit(
            id="create-bc-plan-memory-check-h",
            anchor=r"\A",
            text=_BC_PLAN_MEMORY_CHECK_H,
            guard=r"^bool bc_plan_memory_check\(const nlohmann::json& expected, const nlohmann::json& live, int64_t kv_backing_floor_ppm,$",
            max_span_lines=1,
            expect_matches=1,
            rationale="Created files are anchored at the empty upstream file; create=True rejects any upstream collision.",
        ),),
    ),
    FilePatch(
        path="core/bc_plan_memory_check.cpp",
        create=True,
        description="Create the plan check: a pure function of two plan documents.",
        edits=(Edit(
            id="create-bc-plan-memory-check-cpp",
            anchor=r"\A",
            text=_BC_PLAN_MEMORY_CHECK_CPP,
            guard=r"^bool bc_plan_memory_check\(const nlohmann::json& expected, const nlohmann::json& live, int64_t kv_backing_floor_ppm,$",
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
                text="\n        bc_plan_memory_note(cfg_, facts, vb, emb_bytes, wfs, ngram_bytes, ck_slot, host_arenas, kv_floor, ctx, ranks_[0]->program);",
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
