"""rad0024 (VTP31): the three-rank host-mailbox one-shot hands decode-sized messages to the weighted two-shot.

Upstream's four-rank exact one-shot already hands a message to the exact two-shot from the all-reduce row's `ts_min`
axis (r4d_rad_ar_oneshot_4rank_exact): the same fold per element, so the same bits, for fewer bytes a rank. The
three-rank one-shot has no such handover, and its two-shot row starts above R4D_AR_WIDE_ONESHOT_MAX_ELEMS, so a
decode all-reduce always takes the one-shot. Through the host mailbox that makes every rank read both peers' whole
messages; a card behind a narrow link (an R9700 at Gen4 x4 beside two x16 cards) spends most of the call there
(VTP31 experiment 2: 120 us of a 218 us call to read 640 KiB at eight streams).

This patch gives the three-rank one-shot the four-rank rule on a host-mailbox instance whose link weights are settled
(rad0021): from `ts_min` elements it runs the exact three-rank two-shot with the settled table at any size, rather
than only above the weights' floor, so the slow-link rank owns the smallest shard and reads about one message instead
of two. Who owns a word changes only the link it crosses, never its fold (fp32, ascending rank), so the result is the
same bits as the one-shot. Below `ts_min`, over peer access, at equal weights, or when the shard pair does not fit the
slot, the one-shot runs unchanged. `--tune-pin all_reduce:ts_min=0` turns it off.

Source-pinned to radiance v1.17.0 (19d4e8fe), after rad0021.
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
        rationale="VTP31: weighted two-shot for three-rank host decode",
    )


ENTRY = FilePatch(path="libr4d/r4d_ar_entry.hip", language="none", description="VTP31 3-rank decode two-shot", edits=(
    _edit(
        "ts3d-handoff",
        'extern "C" int r4d_rad_ar_oneshot_3rank_exact(const RadArgs* a, RadStream s) {\n'
        "    ArCall c;\n"
        "    int rc = ar_setup(a, &c, 0);\n"
        "    if (rc != RAD_OK) return rc;\n"
        "    if (c.in->world != 3) return RAD_E_SHAPE;\n"
        "    const ArInst* in = c.in;\n",
        "/* BC_VTP31_TS3D_HANDOFF (rad0024): the exact three-rank two-shot (below). */\n"
        "static int ts3_exact(const RadArgs* a, RadStream s, bool any_size);\n"
        'extern "C" int r4d_rad_ar_oneshot_3rank_exact(const RadArgs* a, RadStream s) {\n'
        "    ArCall c;\n"
        "    int rc = ar_setup(a, &c, 0);\n"
        "    if (rc != RAD_OK) return rc;\n"
        "    if (c.in->world != 3) return RAD_E_SHAPE;\n"
        "    const ArInst* in = c.in;\n"
        "    /* From `ts_min` elements, as the four-rank one-shot does: on a host mailbox with settled link weights,\n"
        "     * the weighted two-shot at any size, so the rank behind the slowest link reads about one message\n"
        "     * instead of two. Same fold, same bits; a shape or slot it cannot take falls through to the one-shot. */\n"
        "    if (in->host && in->shard_weighted) {\n"
        "        const long ts_min = r4d_cfg_ar_ts_min(a);\n"
        "        if (ts_min > 0 && c.numel >= ts_min) {\n"
        "            const int rc2 = ts3_exact(a, s, true);\n"
        "            if (rc2 != RAD_E_SHAPE && rc2 != RAD_E_SCRATCH) return rc2;\n"
        "        }\n"
        "    }\n",
        guard="BC_VTP31_TS3D_HANDOFF",
    ),
    _edit(
        "ts3d-split",
        'extern "C" int r4d_rad_ar_twoshot_3rank_exact(const RadArgs* a, RadStream s) {\n'
        "    ArCall c;\n"
        "    int rc = ar_setup(a, &c, 1);\n",
        "/* BC_VTP31_TS3D_SPLIT (rad0024): the row's entry is the body with the weights' floor; the one-shot's\n"
        " * handover takes it without the floor. */\n"
        'extern "C" int r4d_rad_ar_twoshot_3rank_exact(const RadArgs* a, RadStream s) {\n'
        "    return ts3_exact(a, s, false);\n"
        "}\n"
        "static int ts3_exact(const RadArgs* a, RadStream s, bool any_size) {\n"
        "    ArCall c;\n"
        "    int rc = ar_setup(a, &c, 1);\n",
        guard="BC_VTP31_TS3D_SPLIT",
    ),
    _edit(
        "ts3d-weights",
        "    const int* weights = ar8_weights_for(c.in, c.numel);\n"
        "    const long words16 = c.numel * c.in->esize / 16;\n"
        "    if (!weights && words16 < c.in->world) return RAD_E_SHAPE;\n",
        "    const int* weights = (any_size && c.in->shard_weighted) ? c.in->shard_w  /* BC_VTP31_TS3D_WEIGHTS */\n"
        "                                                            : ar8_weights_for(c.in, c.numel);\n"
        "    const long words16 = c.numel * c.in->esize / 16;\n"
        "    if (!weights && words16 < c.in->world) return RAD_E_SHAPE;\n",
        guard="BC_VTP31_TS3D_WEIGHTS",
    ),
))

PATCHES = (ENTRY,)
