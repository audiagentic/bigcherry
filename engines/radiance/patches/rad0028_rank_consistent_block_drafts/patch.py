"""rad0028 (VTP31): every rank verifies the same DFlash2 draft.

Each rank runs the DFlash2 selector itself. Its candidates are merged identically on every rank, but its transition
scores (`hp`) come from the rank's own drafter state. Card architectures round differently, so on a near tie a rank of
another architecture can select another token. The ranks then verify different drafts, accept different counts, and
their contexts drift apart: the step mixes partial sums of different tokens, and the drifted rank's attention reads
past the host's block table into its -1 padding (a GPU memory fault). VTP31 measured this at TP3 on two gfx1100 and
one gfx1201 with the int8 (w8a8) model: rank 2 disagreed with rank 0 on 166 rows, rank 1 never.

With more than one rank and a chain proposal, the selected tokens are all-gathered after the selector (and the coupled
walk, which may replace them), and the engine reads its proposal from the gathered buffer, whose first slice is rank
0's. Every rank then verifies and accepts rank 0's draft. The gather moves the tokens' bytes; the tokens buffer gets
rows enough for whole 16-byte words. A tree proposal (whose parents the engine reads too) is left as it was.
Source-pinned to radiance v1.17.0 (19d4e8fe), after rad0027.
"""
from __future__ import annotations

import re

from bigcherry.patcher import Edit, FilePatch


def _edit(id: str, before: str, after: str, *, guard: str) -> Edit:
    return Edit(
        id=id,
        anchor=re.escape(before),
        mode="replace",
        text=after,
        guard=re.escape(guard),
        expect_matches=1,
        max_span_lines=max(2, before.count("\n") + 1),
        rationale="VTP31: every rank verifies rank 0's block draft",
    )


BLOCK = FilePatch(path="arch/common/rad_block_dflash2.h", language="none", description="VTP31 rank-consistent drafts",
                  edits=(
    _edit(
        "rcd-wire",
        """        rad_buf tokens = 0; /* [max_seqs, steps] i32  the draft; the engine reads it by name */
""",
        """        rad_buf tokens = 0; /* [max_seqs, steps] i32  the draft; the engine reads it by name */
        /* BC_VTP31_RCD_WIRE (rad0028): [world * tok_rows, steps] i32, every rank's `tokens` (of
         * tok_rows rows) gathered; rank 0's first. Declared by the plugin when there are ranks to
         * agree; the engine then proposes from it. */
        rad_buf tokg = 0;
        int64_t tok_rows = 0;
""",
        guard="BC_VTP31_RCD_WIRE",
    ),
    _edit(
        "rcd-op",
        """    rad_op op_drerank = 0;                  /* the candidates rescored against the target's head */
""",
        """    rad_op op_drerank = 0;                  /* the candidates rescored against the target's head */
    rad_op op_tgather = 0;                  /* BC_VTP31_RCD_OP (rad0028): every rank's draft */
""",
        guard="BC_VTP31_RCD_OP",
    ),
    _edit(
        "rcd-decl",
        """    /* The reads beside the all-reduces, after every weight they read is declared. */
""",
        """    /* BC_VTP31_RCD_DECL (rad0028): the selected tokens, all-gathered after every op that writes
     * them, so every rank can verify rank 0's draft. */
    if (w.tokg && !cfg.tree) {
        if (g.world <= 1 || w.tok_rows <= 0 || (w.tok_rows * cfg.steps()) % 4) return RAD_E_INVAL;
        op_tgather = rw(b, RAD_OP(b, "all_gather",
                           RAD_PARAMS(RAD_INT("world_size", g.world),
                                      RAD_INT("numel", w.tok_rows * cfg.steps()),
                                      RAD_STR("dtype", "f32")),
                           RAD_NOWEIGHTS),
                       {w.tokens}, {w.tokg});
        if (!op_tgather) return RAD_E_INVAL;
    }

    /* The reads beside the all-reduces, after every weight they read is declared. */
""",
        guard="BC_VTP31_RCD_DECL",
    ),
    _edit(
        "rcd-issue",
        """                    RAD_B(w.couple_state));
}
""",
        """                    RAD_B(w.couple_state));
    if (op_tgather) RAD_ISSUE(c, op_tgather, RAD_B(w.tokens), RAD_B(w.tokg)); /* BC_VTP31_RCD_ISSUE (rad0028) */
}
""",
        guard="BC_VTP31_RCD_ISSUE",
    ),
))

PLUGIN = FilePatch(path="arch/qwen35_fp8/qwen35_fp8.cpp", language="none", description="VTP31 rank-consistent drafts",
                   edits=(
    _edit(
        "rcd-buf",
        """    dw.tokens  = decl_b(b, nm.f("dflash_tokens"), RAD_I32, {dg.max_seqs, dc.steps()});
""",
        """    /* BC_VTP31_RCD_BUF (rad0028): with ranks to agree, the tokens buffer holds whole 16-byte
     * words for the gather, and every rank's copy lands in `dflash_tokens_ranks`. */
    const bool rcd = dg.world > 1 && !dc.tree;
    int64_t tok_rows = dg.max_seqs;
    while (rcd && (tok_rows * dc.steps()) % 4) ++tok_rows;
    dw.tokens  = decl_b(b, nm.f("dflash_tokens"), RAD_I32, {tok_rows, dc.steps()});
    if (rcd) {
        dw.tok_rows = tok_rows;
        dw.tokg = decl_b(b, nm.f("dflash_tokens_ranks"), RAD_I32,
                         {(int64_t)dg.world * tok_rows, dc.steps()});
        if (!dw.tokg) return RAD_E_INVAL;
    }
""",
        guard="BC_VTP31_RCD_BUF",
    ),
    _edit(
        "rcd-prop",
        """    drf.proposal       = dw.tokens;
""",
        """    drf.proposal       = dw.tokg ? dw.tokg : dw.tokens; /* BC_VTP31_RCD_PROP (rad0028): rank 0's */
""",
        guard="BC_VTP31_RCD_PROP",
    ),
))

PATCHES = (BLOCK, PLUGIN)
