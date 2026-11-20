"""RUM06: restore the original plain OCP-MXFP4 weight path on Radiance 2.2.

The 2.2 retirement classifier rejects MXFP4 before LinearFP8::declare can reach its
still-present MXFP4 GEMM, including GDN's q|k|v|z row-sharded in-projection.
Leave M2XFP4, rotated MXFP4, ParoQuant and block-FP8 retirements intact.
The existing VTP14 row-part and uneven-span rules remain authoritative.
"""
from __future__ import annotations

import re

from bigcherry.patcher import Edit, FilePatch

_BEFORE = '''    if (linear_is_m2xfp4(e))      return "M2XFP4";
    if (linear_is_mxfp4(e))       return "MXFP4";
    const RadEncPlane* sc = rad_enc_plane(&e, "scale");'''

_AFTER = '''    if (linear_is_m2xfp4(e))      return "M2XFP4";
    /* BC_RUM06_RETAIN_PLAIN_MXFP4: plain OCP-MXFP4 has an existing LinearFP8
     * GEMM and per-part row shard (rad0019). Keep the other retired formats
     * refused; no change to rotations, layouts or TP ownership. */
    const RadEncPlane* sc = rad_enc_plane(&e, "scale");'''

PATCH = FilePatch(
    path="arch/common/rad_fp8.h",
    language="none",
    description="Restore only plain MXFP4 admission for Qwen35's existing linears",
    edits=(
        Edit(
            id="rum06-retain-plain-mxfp4",
            anchor=re.escape(_BEFORE),
            mode="replace",
            text=_AFTER,
            guard=re.escape("BC_RUM06_RETAIN_PLAIN_MXFP4"),
            expect_matches=1,
            max_span_lines=4,
            rationale="ABI44 still declares plain MXFP4 in LinearFP8/DenseFP8; v1.17 TP2 used it",
        ),
    ),
)
