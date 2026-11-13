"""rad0032 (VTP36): the three-rank exact two-shot can run in pieces and with the duplex shard weights.

A host-staged three-rank world with one rank behind a narrower link is link-bound in prefill: with the settled
quadratic weights (rad0021) the rank behind the narrow link sends nearly its whole message in the two-shot's first
hop and receives nearly the whole result in its second, so in the lockstep form its link carries one direction at a
time and a call costs about twice the link's one-way time. Upstream already has both remedies, for eight ranks only:

* pieces (`r4d_ar_twoshot_wide_exact_body`, PIECED): each block's words run as a pipeline of K pieces, piece j+1's
  first hop sharing the link with piece j's second, so both directions carry bytes at once;
* the duplex weights (`r4d_ar_weights_duplex`, r4d_ar_links.h): the table that minimises the link time when both
  directions run at once; the rank behind a link of half the rate then owns no shard.

This patch offers both to the three-rank two-shot as two axes of the all-reduce rows, pinned per run with
`--tune-pin` (rad0022), e.g. `all_reduce:nblocks=4,nthreads=256,ts_pieces=4,ts_duplex=1`:

* `ts_pieces` (1, 2, 4, 8; default 1): the piece count of a three-rank two-shot of at least the instance's piece
  floor (upstream's 262144 elements); smaller messages keep one piece.
* `ts_duplex` (0, 1; default 0): where the settled link weights apply, use the duplex table instead, if the shard
  pair it gives fits the slot (else the settled table).

Which rank owns a word and how a block's words are cut into pieces change only when a word crosses which link, never
its fold (fp32, ascending rank), so every setting gives the bits of the default. With both axes at their defaults the
launch is the one rad0021/rad0024 make. The axes are declared on every all-reduce tunable table, as `ts_min` is, so
one pin is legal for every row the op resolves to; other world sizes do not read them.

Source-pinned to radiance v1.17.0 (19d4e8fe), after rad0021, rad0024 and rad0031.
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
        rationale=("long-anchor: the anchor is one whole launch statement; " if (max_span or 0) > 6 else "")
        + "VTP36: three-rank two-shot pieces and duplex weights",
    )


def _axes(table: str, tag: str) -> Edit:
    head = f"static const RadTunable {table}[] = {{\n"
    return _edit(
        f"ts3p-axes-{tag}",
        head,
        head +
        f'  /* BC_VTP36_TS3P_AXES_{tag.upper()} (rad0032): read by the three-rank exact two-shot only */\n'
        '  { "ts_pieces", kArTsPieces, NELEM(kArTsPieces), 1, "three-rank exact two-shot: pieces a block runs as a pipeline (both link directions at once); 1 = lockstep" },\n'
        '  { "ts_duplex", kArTsDuplex, NELEM(kArTsDuplex), 0, "three-rank exact two-shot: 1 = the duplex link weights where the settled ones apply" },\n',
        guard=f"BC_VTP36_TS3P_AXES_{tag.upper()} ",
    )


ROWS = FilePatch(path="libr4d/r4d_rows.cpp", language="none", description="VTP36 three-rank two-shot axes", edits=(
    _edit(
        "ts3p-values",
        "static const int64_t kArTsMin[]    = { 0, 16384, 32768, 65536, 131072, 262144 };\n",
        "static const int64_t kArTsMin[]    = { 0, 16384, 32768, 65536, 131072, 262144 };\n"
        "/* BC_VTP36_TS3P_VALUES (rad0032): the three-rank exact two-shot's piece count and weight rule. */\n"
        "static const int64_t kArTsPieces[] = { 1, 2, 4, 8 };\n"
        "static const int64_t kArTsDuplex[] = { 0, 1 };\n",
        guard="BC_VTP36_TS3P_VALUES",
    ),
    _axes("tAr", "ar"),
    _axes("tArTs4Lane", "ts4lane"),
    _axes("tArLane", "lane"),
    _edit(
        "ts3p-cfg",
        'extern "C" long r4d_cfg_ar_ts_min(const RadArgs* a) {\n'
        '    return (long)tune_i(a, tAr, NELEM(tAr), "ts_min");\n'
        "}\n",
        'extern "C" long r4d_cfg_ar_ts_min(const RadArgs* a) {\n'
        '    return (long)tune_i(a, tAr, NELEM(tAr), "ts_min");\n'
        "}\n"
        "/* BC_VTP36_TS3P_CFG (rad0032) */\n"
        'extern "C" int r4d_cfg_ar_ts_pieces(const RadArgs* a) {\n'
        '    return TUNE(a, tAr, "ts_pieces");\n'
        "}\n"
        'extern "C" int r4d_cfg_ar_ts_duplex(const RadArgs* a) {\n'
        '    return TUNE(a, tAr, "ts_duplex");\n'
        "}\n",
        guard="BC_VTP36_TS3P_CFG",
    ),
))

ARGS = FilePatch(path="libr4d/r4d_args.h", language="none", description="VTP36 three-rank two-shot axes", edits=(
    _edit(
        "ts3p-cfg-decl",
        "long r4d_cfg_ar_ts_min(const RadArgs*);\n",
        "long r4d_cfg_ar_ts_min(const RadArgs*);\n"
        "/* BC_VTP36_TS3P_ARGDECL (rad0032): the three-rank exact two-shot's `ts_pieces` and `ts_duplex` axes. */\n"
        "int  r4d_cfg_ar_ts_pieces(const RadArgs*);\n"
        "int  r4d_cfg_ar_ts_duplex(const RadArgs*);\n",
        guard="BC_VTP36_TS3P_ARGDECL",
    ),
))

ENTRY = FilePatch(path="libr4d/r4d_ar_entry.hip", language="none", description="VTP36 three-rank two-shot pieces and duplex weights", edits=(
    _edit(
        "ts3p-inst",
        "    int  shard_weighted = 0;\n"
        "    int  shard_w[R4D_MAX_RANKS] = { 0 };\n",
        "    int  shard_weighted = 0;\n"
        "    int  shard_w[R4D_MAX_RANKS] = { 0 };\n"
        "    /* BC_VTP36_TS3P_INST (rad0032): the duplex table for the same links (r4d_ar_weights_duplex), settled\n"
        "     * beside shard_w at three ranks; a launch takes it under the `ts_duplex` axis. */\n"
        "    int  shard_wd_ok = 0;\n"
        "    int  shard_wd[R4D_MAX_RANKS] = { 0 };\n",
        guard="BC_VTP36_TS3P_INST",
    ),
    _edit(
        "ts3p-settle",
        "        bool same = true;\n"
        "        for (int r = 1; r < world; ++r) same = same && w[r] == w[0];\n"
        "        if (same) return RAD_OK;   /* equal links: the even split, on the unweighted path */\n",
        "        bool same = true;\n"
        "        for (int r = 1; r < world; ++r) same = same && w[r] == w[0];\n"
        "        if (same) return RAD_OK;   /* equal links: the even split, on the unweighted path */\n"
        "        /* BC_VTP36_TS3P_SETTLE (rad0032): every rank derives the same duplex table from the same rates. */\n"
        "        if (world == 3) {\n"
        "            int wd[R4D_MAX_RANKS] = { 0 };\n"
        "            if (r4d_ar_weights_duplex(gbs, world, wd)) {\n"
        "                in->shard_wd_ok = 1;\n"
        "                std::memcpy(in->shard_wd, wd, sizeof(in->shard_wd));\n"
        "                static std::once_flag said_duplex;\n"
        "                std::call_once(said_duplex, [&wd] {\n"
        '                    std::fprintf(stderr, "[r4d] 3-rank two-shot duplex shard weights (axis ts_duplex): "\n'
        '                                 "%d,%d,%d\\n", wd[0], wd[1], wd[2]);\n'
        "                });\n"
        "            }\n"
        "        }\n",
        guard="BC_VTP36_TS3P_SETTLE",
    ),
    _edit(
        "ts3p-launch",
        "    long off[R4D_MAX_RANKS + 1];\n"
        "    const long sh_max = r4d_ar_shard_table(words16, c.in->world, weights, off);\n"
        "    if (2 * sh_max * 16 > c.in->slot_bytes) return RAD_E_SCRATCH;\n"
        "    const ArInst* in = c.in;\n"
        '    R4D_TRY("ar_twoshot_3rank_exact",\n'
        "        r4d_ar_twoshot_3rank_exact_w(weights, in->peer_scratch[0], in->peer_scratch[1],\n"
        "                                   in->peer_flags[0], in->peer_flags[1],\n"
        "                                   in->my_scratch, in->my_flags, in->my_seq, in->slot16,\n"
        "                                   c.inp, c.out, c.numel, in->dtype, in->rank,\n"
        "                                   (long)(uintptr_t)s, c.nblocks, c.nthreads, c.drain, c.acq,\n"
        "                                   c.pub));\n",
        "    long off[R4D_MAX_RANKS + 1];\n"
        "    /* BC_VTP36_TS3P_DUPLEX (rad0032): under `ts_duplex`, the duplex table where the settled one applies and\n"
        "     * its shard pair fits the slot. A function of the message size and the instance, the same on every rank. */\n"
        "    if (weights && c.in->shard_wd_ok && r4d_cfg_ar_ts_duplex(a) != 0 &&\n"
        "        2 * r4d_ar_shard_table(words16, c.in->world, c.in->shard_wd, off) * 16 <= c.in->slot_bytes)\n"
        "        weights = c.in->shard_wd;\n"
        "    const long sh_max = r4d_ar_shard_table(words16, c.in->world, weights, off);\n"
        "    if (2 * sh_max * 16 > c.in->slot_bytes) return RAD_E_SCRATCH;\n"
        "    const ArInst* in = c.in;\n"
        "    /* BC_VTP36_TS3P_PIECES (rad0032): `ts_pieces` from the instance's piece floor, as the eight-rank row. */\n"
        "    const long ts_pieces = c.numel >= in->pieces_min ? (long)r4d_cfg_ar_ts_pieces(a) : 1L;\n"
        "    if (ts_pieces < 1 || ts_pieces > R4D_AR_TS_MAX_PIECES) return RAD_E_INVAL;\n"
        '    R4D_TRY("ar_twoshot_3rank_exact",\n'
        "        r4d_ar_twoshot_3rank_exact_wp(weights, ts_pieces, in->peer_scratch[0], in->peer_scratch[1],\n"
        "                                   in->peer_flags[0], in->peer_flags[1],\n"
        "                                   in->my_scratch, in->my_flags, in->my_seq, in->slot16,\n"
        "                                   c.inp, c.out, c.numel, in->dtype, in->rank,\n"
        "                                   (long)(uintptr_t)s, c.nblocks, c.nthreads, c.drain, c.acq,\n"
        "                                   c.pub));\n",
        guard="BC_VTP36_TS3P_DUPLEX",
        max_span=14,
    ),
))

KERNEL = FilePatch(path="libr4d/r4d_ar_twoshot_Nrank_exact.hip", language="none", description="VTP36 three-rank two-shot in pieces", edits=(
    _edit(
        "ts3p-wrapper",
        "/* BC_VTP14_TS3W_WRAPPER (rad0021): the three-rank two-shot with a shard table (null: the even split). */\n",
        "/* BC_VTP36_TS3P_WRAPPER (rad0032): the three-rank two-shot with a shard table and a piece count. */\n"
        "void r4d_ar_twoshot_3rank_exact_wp(const int* shard_weights, long pieces, long peer_scratch0,\n"
        "                                   long peer_scratch1, long peer_flags0, long peer_flags1, long my_scratch,\n"
        "                                   long my_flags, long seq_ctrs, long slot_stride16, long inp, long out,\n"
        "                                   long n_elem, long dtype, long rank, long stream, long nblocks,\n"
        "                                   long nthreads, long drain, long acq, long pub) {\n"
        "  const long ps[2] = {peer_scratch0, peer_scratch1};\n"
        "  const long pf[2] = {peer_flags0, peer_flags1};\n"
        "  ar_twoshot_launch<3>(ps, pf, my_scratch, my_flags, seq_ctrs, slot_stride16, inp, out, n_elem,\n"
        "                       dtype, rank, stream, nblocks, nthreads, drain, acq, pub, 0L, 0L, shard_weights,\n"
        "                       pieces, 0L, 0L);\n"
        "}\n"
        "\n"
        "/* BC_VTP14_TS3W_WRAPPER (rad0021): the three-rank two-shot with a shard table (null: the even split). */\n",
        guard="BC_VTP36_TS3P_WRAPPER",
    ),
))

HEADER = FilePatch(path="libr4d/r4d.h", language="none", description="VTP36 three-rank two-shot in pieces", edits=(
    _edit(
        "ts3p-decl",
        "/* BC_VTP14_TS3W_DECL (rad0021): the same with a shard table (r4d_ar_shards.h); null is the even split. */\n",
        "/* BC_VTP36_TS3P_DECL (rad0032): the same with a shard table and a piece count (1 to R4D_AR_TS_MAX_PIECES). */\n"
        "void r4d_ar_twoshot_3rank_exact_wp(const int* shard_weights, long pieces, long peer_scratch0,\n"
        "                                   long peer_scratch1, long peer_flags0, long peer_flags1, long my_scratch,\n"
        "                                   long my_flags, long seq_ctrs, long slot_stride16, long inp, long out,\n"
        "                                   long n_elem, long dtype, long rank, long stream, long nblocks,\n"
        "                                   long nthreads, long drain, long acq, long pub);\n"
        "/* BC_VTP14_TS3W_DECL (rad0021): the same with a shard table (r4d_ar_shards.h); null is the even split. */\n",
        guard="BC_VTP36_TS3P_DECL",
    ),
))

PATCHES = (ROWS, ARGS, ENTRY, KERNEL, HEADER)
