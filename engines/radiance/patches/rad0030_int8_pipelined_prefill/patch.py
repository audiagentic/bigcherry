"""rad0030 (VTP37): the pipelined prefill on a trunk whose norms write rotated int8 codes, under --pipelined-prefill on.

qwen35_fp8's pipelined prefill (step_pipelined, rad0020 and rad0025-rad0027) cuts a wide prefill step into two pieces
and runs every block's all-reduce and the next block's norm on the second lane, beside the other piece's work. Upstream
keeps it off on a trunk whose norms write rotated int8 codes (qwen35_w8a8 where the kernel library serves the
`_had_quant_i8g` forms, ActFP8::q8_fed): the boundary norm on the second lane, NormQuantFP8::step_plain, issues its op
with the E4M3 code pair as outputs, while the op declared on such a trunk is `rmsnorm_had_quant_i8g`, which writes the
int8 pair. Nothing else in the pipelined step depends on the code type: every block's piece path reads the fed pair
(ActFP8::cq / cs), and the second lane's other ops (the all-reduce, the add and the final norm) touch only bf16 planes.

This patch:

- makes step_plain write the pair step() writes: the int8 pair on a rotated-int8 trunk, the E4M3 pair otherwise;
- names the pair the norms write (ActFP8::cq / cs) to rad_buf_concurrent, so the second lane's int8 codes leave the
  sharing pool (the E4M3 pair of such a trunk is touched by no op and stays in it);
- pipelines such a trunk only under `--pipelined-prefill on`. Under `auto` (the default) it prefills serially as
  before, and the note it prints names the option. A trunk fed int8 codes that are not rotated stays serial under any
  mode: its norm op writes E4M3 codes no linear reads.

Without `--pipelined-prefill on` nothing changes. An E4M3 trunk is unchanged under every mode.
Source-pinned to radiance v1.17.0 (19d4e8fe), after rad0029.
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
        rationale="VTP37: pipelined prefill on a rotated-int8 trunk",
    )


NORM = FilePatch(path="arch/common/rad_fp8.h", language="none", description="VTP37 plain norm writes the fed pair", edits=(
    _edit(
        "plain-norm-fed-pair",
        "                brow_slice(a.q, r0, rows, n),\n"
        "                brow_slice(a.s, r0, rows, n / RAD_FP8_BLOCK),\n",
        "                /* BC_VTP37_PLAIN_FED (rad0030): the pair step() writes, int8 on a rotated-int8 trunk */\n"
        "                brow_slice(hi8 ? a.q8 : a.q, r0, rows, n),\n"
        "                brow_slice(hi8 ? a.s8 : a.s, r0, rows, n / RAD_FP8_BLOCK),\n",
        guard="BC_VTP37_PLAIN_FED",
    ),
))

PLUGIN = FilePatch(path="arch/qwen35_fp8/qwen35_fp8.cpp", language="none", description="VTP37 int8 trunk pipelines under on", edits=(
    _edit(
        "pipe-i8-mode",
        "    if (m.a_h.q8_fed) {\n"
        "        if (pipe_want && g.max_tok >= m.pipe_min_rows && ctx->rank == 0)\n",
        "    /* BC_VTP37_PIPE_I8 (rad0030): a rotated-int8 trunk pipelines under --pipelined-prefill on (its\n"
        "     * boundary norms write the int8 pair, NormQuantFP8::step_plain); under auto it stays serial,\n"
        "     * as does a trunk fed int8 codes that are not rotated under any mode. */\n"
        "    if (m.a_h.q8_fed && (ctx->pipelined_prefill != BC_PIPE_ON || !m.a_h.q8_rot)) {\n"
        "        if (pipe_want && g.max_tok >= m.pipe_min_rows && ctx->rank == 0)\n",
        guard="BC_VTP37_PIPE_I8",
    ),
    _edit(
        "pipe-i8-note",
        "            fprintf(stderr, \"radiance: qwen35: the pipelined prefill stays off on this trunk: its norms \"\n"
        "                            \"write rotated int8 codes, which the pipeline's boundary norms do not\\n\");\n",
        "            fprintf(stderr, \"radiance: qwen35: the pipelined prefill stays off on this trunk: its norms \"\n"
        "                            \"write int8 codes, which it pipelines only when they are rotated and only \"\n"
        "                            \"under --pipelined-prefill on\\n\"); /* BC_VTP37_PIPE_NOTE */\n",
        guard="BC_VTP37_PIPE_NOTE",
    ),
    _edit(
        "pipe-i8-concurrent",
        "        RAD_ARCH_TRY(rad_buf_concurrent(b, m.a_h.q));\n"
        "        RAD_ARCH_TRY(rad_buf_concurrent(b, m.a_h.s));\n",
        "        /* BC_VTP37_PIPE_BUFS (rad0030): the pair the norms write (ActFP8::cq / cs) */\n"
        "        RAD_ARCH_TRY(rad_buf_concurrent(b, m.a_h.cq()));\n"
        "        RAD_ARCH_TRY(rad_buf_concurrent(b, m.a_h.cs()));\n",
        guard="BC_VTP37_PIPE_BUFS",
    ),
))

PATCHES = (NORM, PLUGIN)
