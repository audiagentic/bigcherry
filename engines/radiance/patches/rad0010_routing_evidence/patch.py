"""RHP01 opt-in host-side accounting at Radiance's existing histogram seam.

Every upstream addition is a thin call. State, guards and exit printing live in the
package-owned core/runtime/bc_moe_remote_evidence.h.
"""
from pathlib import Path
from bigcherry.patcher import Edit, FilePatch

_HEADER = Path(__file__).with_name("bc_moe_remote_evidence.h").read_text(encoding="utf-8")

PATCHES = (
    FilePatch(
        path="core/runtime/bc_moe_remote_evidence.h", create=True,
        description="Create the diagnostic ring lifecycle/accounting extension.",
        edits=(Edit(
            id="create-routing-evidence", anchor=r"\A", text=_HEADER,
            guard=r"^struct BcMoeRemoteEvidence \{$", expect_matches=1,
            rationale="Create the extension once; reject pre-existing upstream files.",
        ),),
    ),
    FilePatch(
        path="core/runtime/ctx.cpp",
        description="Observe report, flush, event, credit, and teardown at existing seams.",
        edits=(
            Edit(
                id="include-routing-evidence",
                anchor=r'^#include "ctx\.h"$',
                text='\n#include "bc_moe_remote_evidence.h"',
                guard=r'^#include "bc_moe_remote_evidence\.h"$',
                expect_matches=1,
                rationale="The Ctx translation unit owns histogram bank lifecycle.",
            ),
            Edit(
                id="route-ring-open",
                anchor=r'^    for \(RadEvent& e : route_done_\) RAD_TRY\(rad_event_create_local\(&e\)\);$',
                text='\n    bc_moe_remote_evidence_open(this, (int)route_.size(), kRouteBanks);',
                guard=r'bc_moe_remote_evidence_open\(this, \(int\)route_\.size\(\), kRouteBanks\)',
                expect_matches=1,
                rationale="Allocate counters from the engine's true number of layers and banks.",
            ),
            Edit(
                id="route-report-issue",
                anchor=r'^            const bool again = s\.step\[bank\] == step;$',
                text='\n            bc_moe_remote_evidence_issue(this, layer, bank, step, batch_ ? batch_->n_seq : 0, batch_ ? batch_->n_seq_decode : 0, batch_ ? batch_->draft_pass : 0);',
                guard=r'bc_moe_remote_evidence_issue\(this, layer, bank, step,',
                expect_matches=1,
                rationale="Observe one real expert-count producer report before bank reuse.",
            ),
            Edit(
                id="route-flush-record",
                anchor=r'^    RAD_TRY\(rad_event_record\(route_done_\[route_bank_\], stream_\)\);$',
                text='\n    bc_moe_remote_evidence_flush(this, route_bank_, route_lo_, route_hi_, route_[(size_t)route_lo_].step[route_bank_]);',
                guard=r'bc_moe_remote_evidence_flush\(this, route_bank_, route_lo_, route_hi_,',
                expect_matches=1,
                rationale="Classify touched rows only after stream-ordered DMA and event enqueue.",
            ),
            Edit(
                id="route-query-guard",
                anchor=r'^        if \(!ok\) continue;$',
                mode="replace",
                text='        if (!bc_moe_remote_evidence_landed(this, layer, b, s.step[b], ok)) continue;',
                guard=r'bc_moe_remote_evidence_landed\(this, layer, b, s\.step\[b\], ok\)',
                expect_matches=1,
                rationale="Do not select stale, duplicate, unlanded or out-of-order rows.",
            ),
            Edit(
                id="route-credit-guard",
                anchor=r'^    route_\[\(size_t\)layer\]\.credited\[bank\] = true;$',
                mode="replace",
                text='    if (bc_moe_remote_evidence_credit(this, layer, bank, route_[(size_t)layer].step[bank], route_[(size_t)layer].reports[bank])) route_[(size_t)layer].credited[bank] = true;',
                guard=r'bc_moe_remote_evidence_credit\(this, layer, bank,',
                expect_matches=1,
                rationale="One credit per landed, current-generation bank, with actual pass weight.",
            ),
            Edit(
                id="teardown-evidence-exit",
                anchor=r'^    if \(lane1_\) rad_stream_sync\(lane1_\);$',
                text='\n    bc_moe_remote_evidence_exit(this, rank_, route_dropped_, route_done_);',
                guard=r'bc_moe_remote_evidence_exit\(this, rank_, route_dropped_, route_done_\)',
                expect_matches=1,
                rationale="Print completion after the engine's existing shutdown stream drain, no new wait.",
            ),
            Edit(
                id="teardown-evidence-close",
                anchor=r'^    route_\.clear\(\);$',
                text='    bc_moe_remote_evidence_close(this);\n',
                guard=r'bc_moe_remote_evidence_close\(this\)',
                expect_matches=1,
                rationale="Avoid retaining counters across context teardown or restart.",
            ),
        ),
    ),
    FilePatch(
        path="core/engine.cpp",
        description="Drain the existing histogram reader in an enabled-only static-plan mode.",
        edits=(
            Edit(
                id="include-engine-evidence",
                anchor=r'^#include "engine_priv\.h"$',
                text='\n#include "runtime/bc_moe_remote_evidence.h"',
                guard=r'^#include "runtime/bc_moe_remote_evidence\.h"$',
                expect_matches=1,
                rationale="Use the same host-side lifecycle tracker in engine dispatch.",
            ),
            Edit(
                id="static-heat-evidence",
                anchor=r'^    XsTally\* const xs = xstats_ \? xstats_->tally\(r\.index\) : nullptr;$',
                text='\n    if (bc_moe_remote_evidence_static_drain(r.ctx, r.heat_seen, step, !r.heat.enabled() && !xs)) return;',
                guard=r'bc_moe_remote_evidence_static_drain\(r\.ctx, r\.heat_seen,',
                expect_matches=1,
                rationale="Observe landed banks without enabling movement in deterministic/all-resident plans.",
            ),
        ),
    ),
)
