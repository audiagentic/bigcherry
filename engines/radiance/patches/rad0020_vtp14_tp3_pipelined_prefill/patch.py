"""rad0020 (VTP14): the fp8 Qwen plugin's pipelined prefill at three ranks under --tp-ownership.

qwen35_fp8 already pipelines a wide prefill step (step_pipelined): the step's rows run as two pieces, and each block's
all-reduce for one piece runs on the second lane beside the other piece's work, bit for bit the serial step. Upstream
enables it only at two ranks ("the only rank count where the step has been timed against the serial one"), so the
three-rank ownership (rad0018/rad0019) prefills serially while the two-rank baseline it is compared with pipelines.
Every all-reduce in the pipelined step is the collective the serial step declares, at `overlap` 1, and selection picks
its kernel by world, size and peer access as for the serial one. Nothing in the step depends on the rank count.

This patch turns it on at three ranks when an ownership profile is in force (Geom::tp_owned), and names three ranks in
the note an int8 trunk prints when it stays serial. An even three-rank split without a profile is unchanged.
Source-pinned to radiance v1.17.0 (19d4e8fe), applied after rad0019.
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
        rationale="VTP14: pipelined prefill at three ranks",
    )


QWEN = FilePatch(path="arch/qwen35_fp8/qwen35_fp8.cpp", language="none", description="VTP14 TP3 pipelined prefill", edits=(
    _edit(
        "pipe3-on",
        "        m.pipe_on = g.world == 2 && g.max_tok >= kPipeMinRows;\n",
        "        /* BC_VTP14_PIPE3_ON (rad0020): also at three ranks under --tp-ownership, so the TP3 ownership\n"
        "         * prefills like the two-rank split it is compared with. */\n"
        "        m.pipe_on = (g.world == 2 || (g.world == 3 && g.tp_owned)) && g.max_tok >= kPipeMinRows;\n",
        guard="BC_VTP14_PIPE3_ON",
    ),
    _edit(
        "pipe3-note",
        "        if (g.world == 2 && g.max_tok >= kPipeMinRows && ctx->rank == 0)\n",
        "        if ((g.world == 2 || (g.world == 3 && g.tp_owned)) && /* BC_VTP14_PIPE3_NOTE */\n"
        "            g.max_tok >= kPipeMinRows && ctx->rank == 0)\n",
        guard="BC_VTP14_PIPE3_NOTE",
    ),
))

PATCHES = (QWEN,)
