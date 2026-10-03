"""1292: kpool layout survives a tail edit -- truncate instead of rebuilding the whole sequence.

Qwen4Exp's indexer keeps, per sequence, the (position, cell) list of every cached token plus the pool
starts (llama_memory_hybrid_idx::kpool_layout). Any sequence edit marks the sequence stale from the
edited position on, and kpool_layout_update() then rebuilt the list from scratch: an O(n_ctx) walk of
the cell map's std::map plus a pool rescan. MTP speculative decoding removes the rejected draft tail
(seq_rm [p, inf)) on almost every step, so at 80K context every decode step paid a full rebuild
(perf: std::_Rb_tree_increment / vector::_M_assign_aux from that map were the top libllama symbols).

The stale contract already says nothing before the stale position changed, so the layout is truncated
to the cells before it and the pools that end before the cut, and the normal append path re-reads the
tail from the cell map. That is the same result as the full rebuild: the pool scan is a greedy
left-to-right walk whose only state is its index, so resuming at the end of the last kept pool
reproduces the pools a scan from zero would find. Sequences with shared cells keep the full rebuild
(their sharing flag is re-derived there).
"""

import re as _re

from bigcherry.patcher import Edit, FilePatch

GROUP = "core"
STATE = "untested"

_TRUNCATE = """        // BigCherry 1292: a stale position after the first cell is a tail edit (e.g. rejected MTP drafts
        // removed with seq_rm [p, inf)); everything before it is unchanged, so cut the layout there and let
        // the append path below re-read the tail, instead of rebuilding the whole sequence every step.
        bool bc_tail_cut = false;
        if (mem_idx_stale[s] != POS_CLEAN && mem_idx_stale[s] > sq.pos_min && !sq.shared &&
                !sq.cells.empty() && !sp.empty() && sq.pos_min == sp.begin()->first) {
            const auto cut = std::lower_bound(sq.cells.begin(), sq.cells.end(),
                    std::make_pair(mem_idx_stale[s], (uint32_t) 0));
            const size_t n_cut = (size_t) (cut - sq.cells.begin());
            if (n_cut > 0) {
                sq.cells.resize(n_cut);
                while (!sq.pools.empty() && (size_t) sq.pools.back() + kpool > n_cut) {
                    sq.pools.pop_back();
                }
                sq.j_next   = sq.pools.empty() ? 0 : (size_t) sq.pools.back() + kpool;
                bc_tail_cut = true;
            }
        }

"""

HYBRID_IDX = FilePatch(
    path="src/llama-memory-hybrid-idx.cpp",
    language="none",
    description="kpool layout: truncate on a tail edit instead of an O(n_ctx) rebuild every MTP step.",
    edits=(
        Edit(
            id="kpool-tail-cut",
            anchor=r"(?m)^        size_t n_kept = 0;\n        if \(mem_idx_stale\[s\] == POS_CLEAN && !sq\.cells\.empty\(\) && !sp\.empty\(\) &&\n",
            text=_TRUNCATE,
            mode="insert_before",
            guard=r"bool bc_tail_cut = false;",
            expect_matches=1,
            rationale="Only kpool_layout_update() starts its per-sequence append check with this exact pair.",
        ),
        Edit(
            id="kpool-tail-append",
            anchor=_re.escape("        if (mem_idx_stale[s] == POS_CLEAN && !sq.cells.empty() && !sp.empty() &&\n"),
            text="        if ((mem_idx_stale[s] == POS_CLEAN || bc_tail_cut) && !sq.cells.empty() && !sp.empty() &&\n",
            mode="replace",
            guard=_re.escape("(mem_idx_stale[s] == POS_CLEAN || bc_tail_cut)"),
            expect_matches=1,
            rationale="A cut layout takes the clean append path.",
        ),
        Edit(
            id="kpool-tail-no-rebuild",
            anchor=_re.escape("        if (sq.cells.size() != sp.size() || mem_idx_stale[s] != POS_CLEAN) {\n"),
            text="        if (sq.cells.size() != sp.size() || (mem_idx_stale[s] != POS_CLEAN && !bc_tail_cut)) {\n",
            mode="replace",
            guard=_re.escape("(mem_idx_stale[s] != POS_CLEAN && !bc_tail_cut)"),
            expect_matches=1,
            rationale="A cut layout skips the full rebuild unless the cell counts disagree.",
        ),
    ),
)

PATCHES = [HYBRID_IDX]
