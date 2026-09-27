"""RD30B: extend 1237's compact MoE MMQ launch grid to RDNA4 and RDNA2.

1237 (RD30) is runtime-gated to gfx1100 exactly (cc == GGML_CUDA_CC_RDNA3)
because only gfx1100 had evidence. The compaction is architecture-neutral
host-side launch geometry (the kernel body, accumulation order and routing
are unchanged), so this patch widens the gate to RDNA4 (gfx1201) and RDNA2
(gfx1030) for measurement on those cards. RDNA3.5 stays excluded (no
hardware here). 1237's own gfx1100 evidence is untouched: this is a separate
package with its own contract (RD30B-MOE-MMQ-COMPACT-GRID-RDNA4-RDNA2).
"""

import re as _re

from bigcherry.patcher import Edit, FilePatch

GATE = FilePatch(
    path="ggml/src/ggml-cuda/mmq.cuh",
    description="RD30B: admit exactly gfx1201 and gfx1030 to the compact MoE MMQ grid",
    edits=(
        Edit(
            id="rd30b-gate",
            anchor=_re.escape("    return cc == GGML_CUDA_CC_RDNA3;\n"),
            # GPT code review 2026-09-27 (req_6c90e1ebba83464a): the first version
            # of this gate used GGML_CUDA_CC_IS_RDNA4(cc)/IS_RDNA2(cc), which are
            # OPEN family ranges -- IS_RDNA4 has no upper bound at all (matches
            # every future arch with cc >= the RDNA4 threshold) and IS_RDNA2
            # matches every RDNA2 sibling (gfx1031..gfx1036), not only gfx1030.
            # Only gfx1201 and gfx1030 were ever measured, so the gate compares
            # cc against their exact numeric IDs (ggml_cuda_parse_id() maps
            # "gfxNNNN" to GGML_CUDA_CC_OFFSET_AMD + 0xNNNN; gfx1030's ID is
            # already the named GGML_CUDA_CC_RDNA2 threshold itself).
            text="    // RD30B (1265): also gfx1201 (RDNA4) and gfx1030 (RDNA2) exactly --\n"
                 "    // no untested sibling in either family; RDNA3.5 stays excluded.\n"
                 "    return cc == GGML_CUDA_CC_RDNA3 || cc == GGML_CUDA_CC_OFFSET_AMD + 0x1201 ||\n"
                 "           cc == GGML_CUDA_CC_RDNA2;\n",
            mode="replace",
            guard=_re.escape("cc == GGML_CUDA_CC_OFFSET_AMD + 0x1201"),
            rationale="1237's gate function body is its only code line; replaced whole.",
            max_span_lines=3,
        ),
    ),
)

PATCHES = [GATE]
