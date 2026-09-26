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
    description="RD30B: admit RDNA4 and RDNA2 to the compact MoE MMQ grid",
    edits=(
        Edit(
            id="rd30b-gate",
            anchor=_re.escape("    return cc == GGML_CUDA_CC_RDNA3;\n"),
            text="    // RD30B (1265): also RDNA4 (gfx1201) and RDNA2 (gfx1030); RDNA3.5 still excluded.\n"
                 "    return cc == GGML_CUDA_CC_RDNA3 || GGML_CUDA_CC_IS_RDNA4(cc) || GGML_CUDA_CC_IS_RDNA2(cc);\n",
            mode="replace",
            guard=_re.escape("return cc == GGML_CUDA_CC_RDNA3 || GGML_CUDA_CC_IS_RDNA4(cc) || GGML_CUDA_CC_IS_RDNA2(cc);"),
            rationale="1237's gate function body is its only code line; replaced whole.",
            max_span_lines=2,
        ),
    ),
)

PATCHES = [GATE]
