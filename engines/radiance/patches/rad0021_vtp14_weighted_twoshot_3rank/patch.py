"""rad0021 (VTP14): link-weighted shards for the exact three-rank two-shot all-reduce.

libr4d's exact two-shot (r4d_ar_twoshot_Nrank_exact.hip) takes a shard table: rank s reduces words [off[s], off[s+1]).
At eight ranks the engine already weights it by each card's slowest PCIe hop (ar8_settle, r4d_ar_weights_from_rates in
r4d_ar_links.h): a rank behind a slow link owns a smaller shard, or none, and so moves fewer bytes over it (each
direction carries (1 + (ws - 2) x_r) N for a shard fraction x_r). Who owns a word changes only the link it crosses,
never its fold (fp32, ascending rank), so every table gives the same bits as the even split and the one-shot.

At three ranks the same kernel is compiled but always launched with the even split. A world whose cards sit on
unequal links (two RX 7900 XTX at x16 and an R9700 behind a Gen4 x4 port) then pushes a third of every prefill
all-reduce through the slow link twice. This patch settles the weights at three ranks as at eight (the same rule, the
same rendezvous, the same log line), sizes the scratch for the world rather than for eight, and passes the table to the
three-rank launch. Equal links give equal weights and keep the even split exactly; messages under the weights' floor
(R4D_AR8_MIN_ELEMS_DEFAULT) stay even, so decode is unchanged.

libr3 (gfx1100) is generated from this libr4d source, so both libraries of a mixed world derive the one table; the
weights come from the shared rendezvous (rad0017) after a barrier. Source-pinned to radiance v1.17.0 (19d4e8fe).
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
        max_span_lines=max_span or max(2, before.count("\n") + 1),
        rationale="VTP14: link-weighted three-rank two-shot",
    )


ENTRY = FilePatch(path="libr4d/r4d_ar_entry.hip", language="none", description="VTP14 weighted 3-rank two-shot", edits=(
    _edit(
        "ts3w-settle",
        "    if (world == 8) {\n"
        "        const int rc8 = ar8_settle(in, rank, world);\n",
        "    /* BC_VTP14_TS3W_SETTLE (rad0021): the link weights at three ranks as at eight; the three-rank two-shot\n"
        "     * takes the table too. Equal links settle to the even split. */\n"
        "    if (world == 8 || world == 3) {\n"
        "        const int rc8 = ar8_settle(in, rank, world);\n",
        guard="BC_VTP14_TS3W_SETTLE",
    ),
    _edit(
        "ts3w-size",
        "            2 * r4d_ar_shard_max_bound((long)max_numel * in->esize / 16, 8, in->shard_w);\n",
        "            2 * r4d_ar_shard_max_bound((long)max_numel * in->esize / 16, world, in->shard_w); /* BC_VTP14_TS3W_SIZE */\n",
        guard="BC_VTP14_TS3W_SIZE",
    ),
    _edit(
        "ts3w-launch",
        "    if (c.in->world != 3) return RAD_E_SHAPE;\n"
        "    if (c.numel > R4D_AR_TWOSHOT_WIDE_MAX_ELEMS) return RAD_E_SHAPE;\n"
        "    const long words16 = c.numel * c.in->esize / 16;\n"
        "    if (words16 < c.in->world) return RAD_E_SHAPE;\n"
        "    const long sh_max = (words16 + c.in->world - 1) / c.in->world;\n"
        "    if (2 * sh_max * 16 > c.in->slot_bytes) return RAD_E_SCRATCH;\n"
        "    const ArInst* in = c.in;\n"
        "    R4D_TRY(\"ar_twoshot_3rank_exact\",\n"
        "        r4d_ar_twoshot_3rank_exact(in->peer_scratch[0], in->peer_scratch[1],\n",
        "    if (c.in->world != 3) return RAD_E_SHAPE;\n"
        "    if (c.numel > R4D_AR_TWOSHOT_WIDE_MAX_ELEMS) return RAD_E_SHAPE;\n"
        "    /* BC_VTP14_TS3W_LAUNCH (rad0021): the settled link weights, or the even split. */\n"
        "    const int* weights = ar8_weights_for(c.in, c.numel);\n"
        "    const long words16 = c.numel * c.in->esize / 16;\n"
        "    if (!weights && words16 < c.in->world) return RAD_E_SHAPE;\n"
        "    long off[R4D_MAX_RANKS + 1];\n"
        "    const long sh_max = r4d_ar_shard_table(words16, c.in->world, weights, off);\n"
        "    if (2 * sh_max * 16 > c.in->slot_bytes) return RAD_E_SCRATCH;\n"
        "    const ArInst* in = c.in;\n"
        "    R4D_TRY(\"ar_twoshot_3rank_exact\",\n"
        "        r4d_ar_twoshot_3rank_exact_w(weights, in->peer_scratch[0], in->peer_scratch[1],\n",
        guard="BC_VTP14_TS3W_LAUNCH",
    ),
    _edit(
        "ts3w-log",
        "            std::fprintf(stderr, \"[r4d] eight-rank two-shot shard weights from the slowest PCIe \"\n"
        "                         \"hop of each card:%s\\n\", l.c_str());\n",
        "            std::fprintf(stderr, \"[r4d] %d-rank two-shot shard weights from the slowest PCIe \" /* BC_VTP14_TS3W_LOG */\n"
        "                         \"hop of each card:%s\\n\", world, l.c_str());\n",
        guard="BC_VTP14_TS3W_LOG",
    ),
))

KERNEL = FilePatch(path="libr4d/r4d_ar_twoshot_Nrank_exact.hip", language="none", description="VTP14 weighted 3-rank two-shot", edits=(
    _edit(
        "ts3w-wrapper",
        "void r4d_ar_twoshot_4rank_exact(long peer_scratch0, long peer_scratch1, long peer_scratch2,\n"
        "                                long peer_flags0, long peer_flags1, long peer_flags2,\n",
        "/* BC_VTP14_TS3W_WRAPPER (rad0021): the three-rank two-shot with a shard table (null: the even split). */\n"
        "void r4d_ar_twoshot_3rank_exact_w(const int* shard_weights, long peer_scratch0, long peer_scratch1,\n"
        "                                  long peer_flags0, long peer_flags1, long my_scratch, long my_flags,\n"
        "                                  long seq_ctrs, long slot_stride16, long inp, long out, long n_elem,\n"
        "                                  long dtype, long rank, long stream, long nblocks, long nthreads,\n"
        "                                  long drain, long acq, long pub) {\n"
        "  const long ps[2] = {peer_scratch0, peer_scratch1};\n"
        "  const long pf[2] = {peer_flags0, peer_flags1};\n"
        "  ar_twoshot_launch<3>(ps, pf, my_scratch, my_flags, seq_ctrs, slot_stride16, inp, out, n_elem,\n"
        "                       dtype, rank, stream, nblocks, nthreads, drain, acq, pub, 0L, 0L, shard_weights,\n"
        "                       1L, 0L, 0L);\n"
        "}\n"
        "\n"
        "void r4d_ar_twoshot_4rank_exact(long peer_scratch0, long peer_scratch1, long peer_scratch2,\n"
        "                                long peer_flags0, long peer_flags1, long peer_flags2,\n",
        guard="BC_VTP14_TS3W_WRAPPER",
    ),
))

HEADER = FilePatch(path="libr4d/r4d.h", language="none", description="VTP14 weighted 3-rank two-shot", edits=(
    _edit(
        "ts3w-decl",
        "                                long rank, long stream, long nblocks, long nthreads, long drain,\n"
        "                                long acq, long pub);\n"
        "void r4d_ar_twoshot_4rank_exact(long peer_scratch0, long peer_scratch1, long peer_scratch2,\n",
        "                                long rank, long stream, long nblocks, long nthreads, long drain,\n"
        "                                long acq, long pub);\n"
        "/* BC_VTP14_TS3W_DECL (rad0021): the same with a shard table (r4d_ar_shards.h); null is the even split. */\n"
        "void r4d_ar_twoshot_3rank_exact_w(const int* shard_weights, long peer_scratch0, long peer_scratch1,\n"
        "                                  long peer_flags0, long peer_flags1, long my_scratch, long my_flags,\n"
        "                                  long seq_ctrs, long slot_stride16, long inp, long out, long n_elem,\n"
        "                                  long dtype, long rank, long stream, long nblocks, long nthreads,\n"
        "                                  long drain, long acq, long pub);\n"
        "void r4d_ar_twoshot_4rank_exact(long peer_scratch0, long peer_scratch1, long peer_scratch2,\n",
        guard="BC_VTP14_TS3W_DECL",
    ),
))

PATCHES = (ENTRY, KERNEL, HEADER)
