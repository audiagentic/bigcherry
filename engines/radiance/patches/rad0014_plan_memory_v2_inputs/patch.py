"""RSC11 S13 P1: publish the engine's final per-rank GPU required/fixed byte ledger.

Inert and opt-in: `radiance ... --plan-memory-v2-inputs FILE` runs the real
start-up through plan(), writes the per-rank ledger derived from rank_cfg_ and
VramFacts after the uneven-rank resplit, and exits before any weight loads.
Without the flag nothing changes. rad0011's v1 EnginePlan is untouched."""
from bigcherry.patcher import Edit, FilePatch

_HEADER = r"""/* bc_plan_memory_v2_inputs.h -- BigCherry managed patch rad0014 (RSC11 S13 P1): final per-rank GPU
 * required/fixed byte ledger, published after the uneven-rank resplit. Inert unless
 * --plan-memory-v2-inputs FILE is given. */
#pragma once
#include <string>
#include <vector>

namespace rad {
struct Config;
struct VramFacts;
}

/* Takes --plan-memory-v2-inputs off argv. 0, or a usage fault's exit code. */
int bc_plan_memory_v2_inputs_take_flag(int* argc, char*** argv);

/* Records the final per-rank ledger from the real rank_cfg_ and VramFacts, after
 * the uneven-rank resplit. No-op unless measuring. */
void bc_plan_memory_v2_inputs_note(const std::vector<rad::VramFacts>& facts,
                                   const std::vector<rad::Config>& rank_cfg,
                                   const std::vector<std::string>& bdfs);

/* Writes the sidecar (if measuring) and delegates to bc_plan_memory_done() for
 * the v1 plan. Exits before weights load when measuring. */
int bc_plan_memory_v2_inputs_done();
"""

_CPP = r"""/* bc_plan_memory_v2_inputs.cpp -- see bc_plan_memory_v2_inputs.h (rad0014). */
#include "bc_plan_memory_v2_inputs.h"

#include "bc_plan_check.h"
#include "bc_plan_memory.h"
#include "mem/vram_budget.h"
#include "rad_core.h"
#include "nlohmann/json.hpp"

#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>

namespace {
enum class Mode { none, measure };
Mode g_mode = Mode::none;
std::string g_out;
struct Ledger {
    std::string bdf;
    int64_t required_bytes = 0;
    int64_t fixed_bytes = 0;
    int64_t vram_weights_mib = 0;
    int64_t vram_kv_mib = 0;
    int64_t statics_bytes = 0;
    int64_t expert_floor_bytes = 0;
    int64_t capacity_bytes = 0;
    int64_t free_bytes = 0;
};
std::vector<Ledger> g_ledger;
bool g_noted = false;
std::string g_error;
constexpr int64_t kMiB = 1024 * 1024;
}  // namespace

int bc_plan_memory_v2_inputs_take_flag(int* argc, char*** argv) {
    g_mode = Mode::none;
    g_out.clear();
    g_ledger.clear();
    g_error.clear();
    g_noted = false;
    int w = 1;
    for (int i = 1; i < *argc; ++i) {
        const char* a = (*argv)[i];
        if (std::strcmp(a, "--plan-memory-v2-inputs") != 0) {
            (*argv)[w++] = (*argv)[i];
            continue;
        }
        if (!g_out.empty()) {
            std::fprintf(stderr, "radiance: config: --plan-memory-v2-inputs given twice\n");
            std::fflush(nullptr);
            return 2;
        }
        if (i + 1 >= *argc || !*(*argv)[i + 1]) {
            std::fprintf(stderr, "radiance: config: --plan-memory-v2-inputs needs a value\n");
            std::fflush(nullptr);
            return 2;
        }
        g_out = (*argv)[++i];
    }
    (*argv)[w] = nullptr;
    *argc = w;
    if (!g_out.empty()) g_mode = Mode::measure;
    return 0;
}

void bc_plan_memory_v2_inputs_note(const std::vector<rad::VramFacts>& facts,
                                   const std::vector<rad::Config>& rank_cfg,
                                   const std::vector<std::string>& bdfs) {
    if (g_mode != Mode::measure) return;
    if (facts.empty() || facts.size() != rank_cfg.size() || facts.size() != bdfs.size()) {
        g_error = "rank_cfg_/VramFacts/BDF counts disagree";
        g_noted = false;
        return;
    }
    g_ledger.clear();
    for (size_t i = 0; i < facts.size(); ++i) {
        if (bdfs[i].empty()) {
            g_error = "missing BDF identity for rank " + std::to_string(i);
            g_ledger.clear();
            g_noted = false;
            return;
        }
        const int64_t required = rank_cfg[i].vram_weights_mib * kMiB +
                                 rank_cfg[i].vram_kv_mib * kMiB;
        const int64_t fixed = facts[i].statics;
        if (required <= 0 || fixed < 0 || fixed > required) {
            g_error = "non-authoritative or inconsistent per-rank VRAM ledger for rank " +
                      std::to_string(i);
            g_ledger.clear();
            g_noted = false;
            return;
        }
        Ledger l;
        l.bdf = bdfs[i];
        l.required_bytes = required;
        l.fixed_bytes = fixed;
        l.vram_weights_mib = rank_cfg[i].vram_weights_mib;
        l.vram_kv_mib = rank_cfg[i].vram_kv_mib;
        l.statics_bytes = facts[i].statics;
        l.expert_floor_bytes = facts[i].expert_floor;
        l.capacity_bytes = facts[i].capacity;
        l.free_bytes = facts[i].free;
        g_ledger.push_back(std::move(l));
    }
    g_noted = true;
}

int bc_plan_memory_v2_inputs_done() {
    if (g_mode != Mode::measure) return bc_plan_memory_done();
    if (!g_noted) {
        std::fprintf(stderr, "radiance: --plan-memory-v2-inputs: %s\n",
                     g_error.empty() ? "nothing was recorded" : g_error.c_str());
        std::fflush(nullptr);
        std::_Exit(2);
    }
    nlohmann::json ranks = nlohmann::json::array();
    nlohmann::json gpu = nlohmann::json::array();
    for (const auto& l : g_ledger) {
        ranks.push_back({
            {"bdf", l.bdf},
            {"required_bytes", l.required_bytes},
            {"fixed_bytes", l.fixed_bytes},
            {"vram_weights_mib", l.vram_weights_mib},
            {"vram_kv_mib", l.vram_kv_mib},
            {"statics_bytes", l.statics_bytes},
            {"expert_floor_bytes", l.expert_floor_bytes},
            {"capacity_bytes", l.capacity_bytes},
            {"free_bytes", l.free_bytes},
        });
        gpu.push_back({
            {"bdf", l.bdf},
            {"required_bytes", l.required_bytes},
            {"fixed_bytes", l.fixed_bytes},
        });
    }
    nlohmann::json doc = {
        {"schema_version", 1},
        {"kind", "radiance_engine_gpu_inputs_v2"},
        {"producer", "radiance --plan-memory-v2-inputs (rad0014)"},
        {"provenance", "rad0014: final per-rank required/fixed ledger from rank_cfg_ "
                       "and VramFacts after the uneven-rank resplit; BDFs from identify_devices"},
        {"gpu", gpu},
        {"ranks", ranks},
    };
    const std::string tmp = g_out + ".tmp";
    {
        std::ofstream out(tmp, std::ios::binary | std::ios::trunc);
        out << doc.dump(2) << '\n';
        if (!out) {
            std::fprintf(stderr, "radiance: --plan-memory-v2-inputs: cannot write %s\n", tmp.c_str());
            std::fflush(nullptr);
            std::_Exit(2);
        }
    }
    if (std::rename(tmp.c_str(), g_out.c_str()) != 0) {
        std::fprintf(stderr, "radiance: --plan-memory-v2-inputs: cannot replace %s\n", g_out.c_str());
        std::fflush(nullptr);
        std::_Exit(2);
    }
    std::fprintf(stderr, "radiance: --plan-memory-v2-inputs: wrote %s\n", g_out.c_str());
    std::fflush(nullptr);
    int rc = bc_plan_memory_done();
    if (bc_plan::mode() != bc_plan::Mode::measure) std::_Exit(0);
    return rc;
}
"""

def _created(path, text, guard, description):
    from bigcherry.patcher import Edit, FilePatch
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
            rationale="Created file anchored at empty upstream file.",
        ),),
    )

PATCHES = (
    _created("core/bc_plan_memory_v2_inputs.h", _HEADER,
             r"^int bc_plan_memory_v2_inputs_done\(\);$",
             "Create the v2-inputs sidecar header."),
    _created("core/bc_plan_memory_v2_inputs.cpp", _CPP,
             r"^int bc_plan_memory_v2_inputs_done\(\) \{$",
             "Create the v2-inputs sidecar writer."),
    FilePatch(
        path="core/main.cpp",
        description="Take the new flag after rad0011's.",
        edits=(
            Edit(
                id="include-v2-inputs",
                anchor=r'^#include "bc_plan_memory\.h"$',
                text='\n#include "bc_plan_memory_v2_inputs.h"',
                guard=r'^#include "bc_plan_memory_v2_inputs\.h"$',
                max_span_lines=1,
                expect_matches=1,
                rationale="Beside rad0011's include.",
            ),
            Edit(
                id="take-v2-inputs-flag",
                anchor=r"bc_plan_memory_take_flag\(&argc, &argv\)",
                text="\n    if (int rc = bc_plan_memory_v2_inputs_take_flag(&argc, &argv)) return rc;",
                guard=r"bc_plan_memory_v2_inputs_take_flag\(&argc, &argv\)",
                max_span_lines=1,
                expect_matches=1,
                rationale="After rad0011 takes --plan-memory.",
            ),
        ),
    ),
    FilePatch(
        path="core/engine_bringup.cpp",
        description="Record the final ledger after the uneven-rank resplit; replace the done call.",
        edits=(
            Edit(
                id="include-v2-inputs",
                anchor=r'^#include "bc_plan_memory\.h"$',
                text='\n#include "bc_plan_memory_v2_inputs.h"',
                guard=r'^#include "bc_plan_memory_v2_inputs\.h"$',
                max_span_lines=1,
                expect_matches=1,
                rationale="Beside rad0011's include.",
            ),
            Edit(
                id="note-final-ledger",
                anchor=r"^    for \(size_t ri = 0; ri < ranks_\.size\(\); \+\+ri\) \{",
                expect_matches=2,
                occurrence=0,
                mode="insert_before",
                text=("\n    {\n        std::vector<std::string> _bc_v2_bdfs;\n"
                      "        for (auto& _r : ranks_) {\n"
                      "            std::string _bdf;\n"
                      "            for (const auto& _id : dev_ids_) if (_id.device == _r->device) { _bdf = _id.bdf; break; }\n"
                      "            _bc_v2_bdfs.push_back(_bdf);\n"
                      "        }\n"
                      "        bc_plan_memory_v2_inputs_note(facts, rank_cfg_, _bc_v2_bdfs);\n"
                      "    }\n"),
                guard=r"bc_plan_memory_v2_inputs_note\(facts, rank_cfg_,",
                max_span_lines=12,
                rationale="After the uneven-rank resplit, before pools are configured.",
            ),
            Edit(
                id="done-v2-inputs",
                anchor=r"RAD_TRY\(bc_plan_memory_done\(\)\);",
                text="RAD_TRY(bc_plan_memory_v2_inputs_done());",
                guard=r"RAD_TRY\(bc_plan_memory_v2_inputs_done\(\)\);",
                mode="replace",
                max_span_lines=1,
                expect_matches=1,
                rationale="Delegate to the sidecar writer, which calls bc_plan_memory_done().",
            ),
        ),
    ),
)
