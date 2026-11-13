"""rad0033 (VTP36): the host-staged three-rank exact two-shot writes its reduced shard once.

In the two-shot's second hop a rank stores every reduced word of its shard once per peer, into its slot in that
peer's scratch (region B). Over peer access those are different cards' memories, so the copies are needed. Through
the host mailbox every rank's scratch is pinned host memory that every card maps, so the copies are the same bytes
written twice across the writer's own link: at three ranks a rank that owns 4/9 of the message sends 8/9 of it in
the second hop where 4/9 would do.

Under the new axis `ts_once` (0, 1; default 0) of the all-reduce rows, on a host-mailbox instance, a rank writes its
reduced shard once, into region B of its own slot in its own scratch (the slot of a rank's own source in its own
scratch is otherwise never used by the wide family: a rank's own contribution comes from its input), and each peer
assembles that shard from the owner's scratch instead of its own. The handshake is unchanged: the owner publishes
the step's flag after its stores have drained, and peers read only after seeing it. The slot's parity alternates per
launch as before, and a rank reaches launch k + 2 (the next writer of the same parity) only after every rank has
finished launch k, so no reader meets a later launch's bytes. The reduced words are the same, so the bits are too.

Selected per run with `--tune-pin` (rad0022), e.g. `all_reduce:nblocks=4,nthreads=256,ts_once=1`. Over peer access
the axis is ignored (reading a peer's VRAM would replace a write by a slower read).

Source-pinned to radiance v1.17.0 (19d4e8fe), after rad0032.
"""
from __future__ import annotations

import re

from bigcherry.patcher import Edit, FilePatch


def _edit(id: str, before: str, after: str, *, guard: str, max_span: int | None = None) -> Edit:
    return Edit(
        id=id,
        anchor=re.escape(before),
        mode="replace",
        text=after,
        guard=re.escape(guard),
        expect_matches=1,
        max_span_lines=max_span or max(6, before.count("\n") + 1),
        rationale="VTP36: host-staged three-rank two-shot writes its reduced shard once",
    )


def _axis(tag: str) -> Edit:
    head = f"  /* BC_VTP36_TS3P_AXES_{tag.upper()} (rad0032): read by the three-rank exact two-shot only */\n"
    return _edit(
        f"ts3o-axis-{tag}",
        head,
        head +
        f'  /* BC_VTP36_TS3O_AXIS_{tag.upper()} (rad0033) */\n'
        '  { "ts_once", kArTsDuplex, NELEM(kArTsDuplex), 0, "host-staged three-rank exact two-shot: 1 = a rank writes its reduced shard once, in its own scratch, and peers assemble from there" },\n',
        guard=f"BC_VTP36_TS3O_AXIS_{tag.upper()} ",
    )


ROWS = FilePatch(path="libr4d/r4d_rows.cpp", language="none", description="VTP36 write-once axis", edits=(
    _axis("ar"),
    _axis("ts4lane"),
    _axis("lane"),
    _edit(
        "ts3o-cfg",
        'extern "C" int r4d_cfg_ar_ts_stagger(const RadArgs* a) {\n'
        '    return TUNE(a, tAr, "ts_stagger");\n'
        "}\n",
        'extern "C" int r4d_cfg_ar_ts_stagger(const RadArgs* a) {\n'
        '    return TUNE(a, tAr, "ts_stagger");\n'
        "}\n"
        "/* BC_VTP36_TS3O_CFG (rad0033) */\n"
        'extern "C" int r4d_cfg_ar_ts_once(const RadArgs* a) {\n'
        '    return TUNE(a, tAr, "ts_once");\n'
        "}\n",
        guard="BC_VTP36_TS3O_CFG",
    ),
))

ARGS = FilePatch(path="libr4d/r4d_args.h", language="none", description="VTP36 write-once axis", edits=(
    _edit(
        "ts3o-cfg-decl",
        "int  r4d_cfg_ar_ts_stagger(const RadArgs*);\n",
        "int  r4d_cfg_ar_ts_stagger(const RadArgs*);\n"
        "int  r4d_cfg_ar_ts_once(const RadArgs*);   /* BC_VTP36_TS3O_ARGDECL (rad0033): the `ts_once` axis */\n",
        guard="BC_VTP36_TS3O_ARGDECL",
    ),
))

ENTRY = FilePatch(path="libr4d/r4d_ar_entry.hip", language="none", description="VTP36 write-once second hop", edits=(
    _edit(
        "ts3o-launch",
        "    if (ts_pieces > 1 && r4d_cfg_ar_ts_stagger(a) != 0) ts_opts |= R4D_TS_OPT_STAGGER;\n",
        "    if (ts_pieces > 1 && r4d_cfg_ar_ts_stagger(a) != 0) ts_opts |= R4D_TS_OPT_STAGGER;\n"
        "    /* BC_VTP36_TS3O_LAUNCH (rad0033): through the host mailbox every card maps every rank's scratch, so\n"
        "     * the reduced shard is written once and read from its owner's scratch. Not over peer access. */\n"
        "    if (in->host && r4d_cfg_ar_ts_once(a) != 0) ts_opts |= R4D_TS_OPT_ONCE;\n",
        guard="BC_VTP36_TS3O_LAUNCH",
    ),
))

KERNEL = FilePatch(path="libr4d/r4d_ar_twoshot_Nrank_exact.hip", language="none", description="VTP36 write-once second hop", edits=(
    _edit(
        "ts3o-kself",
        "  int4* my_out = out4 + (size_t)R4D_TS_OFF(rank);\n",
        "  int4* my_out = out4 + (size_t)R4D_TS_OFF(rank);\n"
        "  // BC_VTP36_TS3O_KSELF (rad0033): written once, the reduced shard goes to region B of this rank's own slot\n"
        "  // in its own scratch (a slot nothing else uses: a rank's own contribution comes from its input), which a\n"
        "  // host-mailbox peer maps and assembles from.\n"
        "  const bool ts_once = (ts_opts & R4D_TS_OPT_ONCE) != 0;\n"
        "  int4* const ts_self_b = const_cast<int4*>(my_base) + ((size_t)parity * WS + rank) * (size_t)slot_stride16 +\n"
        "                          sh_max;\n",
        guard="BC_VTP36_TS3O_KSELF",
    ),
    _edit(
        "ts3o-kstore",
        "      my_out[k] = o;\n"
        "#pragma unroll\n"
        "      for (int p = 0; p < NPEERS; ++p) pa[p][sh_max + k] = o;  // region B\n",
        "      my_out[k] = o;\n"
        "      if (ts_once) {  /* BC_VTP36_TS3O_KSTORE */\n"
        "        ts_self_b[k] = o;\n"
        "      } else {\n"
        "#pragma unroll\n"
        "        for (int p = 0; p < NPEERS; ++p) pa[p][sh_max + k] = o;  // region B\n"
        "      }\n",
        guard="BC_VTP36_TS3O_KSTORE",
    ),
    _edit(
        "ts3o-kload",
        "      const int4* srcB = my_base + ((size_t)parity * WS + owner) * (size_t)slot_stride16 + sh_max;\n",
        "      /* BC_VTP36_TS3O_KLOAD: the owner's own slot in the owner's scratch, or its slot in mine */\n"
        "      const int4* srcB = (ts_once ? (const int4*)peers.scratch[p] : my_base) +\n"
        "                         ((size_t)parity * WS + owner) * (size_t)slot_stride16 + sh_max;\n",
        guard="BC_VTP36_TS3O_KLOAD",
    ),
))

HEADER = FilePatch(path="libr4d/r4d.h", language="none", description="VTP36 write-once second hop", edits=(
    _edit(
        "ts3o-opt",
        "enum { R4D_TS_OPT_STAGGER = 1 };\n",
        "/* BC_VTP36_TS3O_OPT (rad0033): R4D_TS_OPT_ONCE, host mailbox only: the reduced shard is written once. */\n"
        "enum { R4D_TS_OPT_STAGGER = 1, R4D_TS_OPT_ONCE = 2 };\n",
        guard="BC_VTP36_TS3O_OPT",
    ),
))

PATCHES = (ROWS, ARGS, ENTRY, KERNEL, HEADER)
