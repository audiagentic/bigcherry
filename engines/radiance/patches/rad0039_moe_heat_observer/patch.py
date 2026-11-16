"""RHP02-A: inactive, versioned CPU observer at Radiance's existing credit seam.

All source changes use code-line anchors (comments are stripped by the patcher).
The original histogram producer, credit, heat and xstats statements remain intact.
"""
from pathlib import Path

from bigcherry.patcher import Edit, FilePatch

_ROOT = Path(__file__).parent


def _created(path: str, anchor: str):
    return FilePatch(
        path=path,
        create=True,
        description="Create an observer-owned ABI or implementation payload.",
        edits=(Edit(
            id="create-" + Path(path).name,
            anchor=r"\A",
            text=(_ROOT / Path(path).name).read_text(encoding="utf-8"),
            guard=anchor,
            expect_matches=1,
            rationale="long-anchor: entire new package-owned payload; refuse a pre-existing file.",
        ),),
    )


PATCHES = (
    _created("abi/rad_moe_heat.h", r"^typedef struct RadMoeHeatRowV1 \{$"),
    _created("core/runtime/bc_moe_heat_sidecar.h", r"^struct BcMoeHeatStamp \{$"),
    _created("core/runtime/bc_moe_heat_registry.cpp", r"^extern \"C\" int rad_moe_heat_register_v1\("),
    _created("core/runtime/bc_moe_heat_observer.cpp", r"^void Ctx::heat_credit\("),
    FilePatch(
        path="core/runtime/ctx.h",
        description="Add source-generation sidecar and direct Ctx credit methods.",
        edits=(
            Edit(
                id="include-heat-sidecar",
                anchor=r'^#include "stager\.h"$',
                text='\n#include "bc_moe_heat_sidecar.h"',
                guard=r'^#include "bc_moe_heat_sidecar\.h"$',
                expect_matches=1,
                rationale="Ctx owns its histogram banks; sidecar adds only CPU phase metadata.",
            ),
            Edit(
                id="stamp-route-bank",
                anchor=r'^    int        reports\[kRouteBanks\] = \{\};$',
                text='\n    BcMoeHeatStamp heat[kRouteBanks] = {};',
                guard=r'^    BcMoeHeatStamp heat\[kRouteBanks\] = \{\};$',
                expect_matches=1,
                rationale="One source-generation stamp per existing layer and bank, no new histogram.",
            ),
            Edit(
                id="publish-credit-declarations",
                anchor=r'^    void              mark_credited\(int layer, int bank\);$',
                text='\n    void              heat_note_issue(RouteSlot& slot, int bank, bool again, int64_t n);\n    void              heat_credit(int layer, int bank);',
                guard=r'^    void              heat_credit\(int layer, int bank\);$',
                expect_matches=1,
                rationale="Engine-owned metadata and callback bridge invoked only from original Ctx seams.",
            ),
            Edit(
                id="run-and-bank-generation",
                anchor=r'^    uint64_t route_dropped_ = 0;$',
                text='\n    uint64_t route_generation_ = 0, run_generation_ = 0;',
                guard=r'^    uint64_t route_generation_ = 0, run_generation_ = 0;$',
                expect_matches=1,
                rationale="Never reuse an old ring generation or infer source phase at credit time.",
            ),
        ),
    ),
    FilePatch(
        path="core/runtime/ctx.cpp",
        description="Capture issue-time provenance and emit once after each true bank credit.",
        edits=(
            Edit(
                id="open-run-generation",
                anchor=r'^    route_\.assign\(\(size_t\)n_layer, RouteSlot\{\}\);$',
                text='\n    run_generation_ = bc_moe_heat_new_run();',
                guard=r'^    run_generation_ = bc_moe_heat_new_run\(\);$',
                expect_matches=1,
                rationale="Lifecycle identity is set once when the routed model binds.",
            ),
            Edit(
                id="issue-phase-stamp",
                anchor=r'^            s\.step\[bank\] = step;$',
                text='\n            heat_note_issue(s, bank, again, n);',
                guard=r'^            heat_note_issue\(s, bank, again, n\);$',
                expect_matches=1,
                rationale="Capture batch phase at issue, including coalesced same-step MTP passes.",
            ),
            Edit(
                id="once-per-true-credit",
                anchor=r'^    route_\[\(size_t\)layer\]\.credited\[bank\] = true;$',
                text='\n    heat_credit(layer, bank);',
                guard=r'^    heat_credit\(layer, bank\);$',
                expect_matches=1,
                rationale="The one true credit seam serves both HeatEngine and xstats without source duplication.",
            ),
        ),
    ),
    FilePatch(
        path="core/engine.cpp",
        description="Read-only event-ready drain when both placement heat and xstats are off.",
        edits=(
            Edit(
                id="include-heat-drain",
                anchor=r'^#include "engine_priv\.h"$',
                text='\n#include "runtime/bc_moe_heat_sidecar.h"',
                guard=r'^#include "runtime/bc_moe_heat_sidecar\.h"$',
                expect_matches=1,
                rationale="Standalone read-only drain reuses original next_histogram/mark_credited.",
            ),
            Edit(
                id="disabled-static-drain",
                anchor=r'^void Engine::tiering_dispatch\(Rank& r, int step\) \{$',
                text='\n    if (bc_moe_heat_static_drain(r.ctx, r.heat_seen, step,\n            !r.heat.enabled() && (!xstats_ || !xstats_->tally(r.index)),\n            &r.heat_credited)) return;',
                guard=r'bc_moe_heat_static_drain\(r\.ctx, r\.heat_seen, step,',
                expect_matches=1,
                rationale="Run before the independent RHP01 opt-in static drain; never turn placement on.",
            ),
        ),
    ),
)
