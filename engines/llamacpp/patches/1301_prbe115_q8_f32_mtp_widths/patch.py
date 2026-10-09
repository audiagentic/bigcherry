"""1301 (PRBE115): Q8_0 F32-activation MMVQ at MTP verify widths, opt-in RDNA4.

1241 routes dense Q8_0 matvec through the raw-F32 activation kernel (no
quantize_row_q8_1 launch) only for ncols_dst == 1 on RDNA3. Under MTP the
target verifies n_draft+1 tokens per step, so that path almost never fires.
The f32_act kernel branch is already column-generic (y_f32 + j*stride_col_y),
so this patch only widens the host gate:

  BIGCHERRY_Q8_F32_MAXCOLS=N  (1..8, default 1 = 1241 behaviour) -- widths 1..N
  BIGCHERRY_Q8_F32_RDNA4=1    -- also enable on RDNA4 (gfx1201)

Both are read once per process. Activation evidence: the existing 1241
BIGCHERRY_PATCH_HIT line, already keyed per ncols_dst.
"""

from __future__ import annotations

import re

from bigcherry.patcher import Edit, FilePatch

GROUP = "rdna-boosts"
STATE = "rejected"

_GATE_OLD = (
    "    if (!ids && src0->type == GGML_TYPE_Q8_0 && ne1 == 1 && !forced.requested()) {\n"
    "        const int rd33_cc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;\n"
    "        if (GGML_CUDA_CC_IS_RDNA3_0(rd33_cc)) {\n"
)

_GATE_NEW = """\
    // bigcherry 1301 (PRBE115): widen 1241's Q8_0 F32-activation gate to MTP verify widths.
    static const int prbe115_maxcols = [] {
        const char * s = getenv("BIGCHERRY_Q8_F32_MAXCOLS");
        const int v = s ? atoi(s) : 1;
        return v < 1 ? 1 : (v > 8 ? 8 : v);
    }();
    static const bool prbe115_rdna4 = [] {
        const char * s = getenv("BIGCHERRY_Q8_F32_RDNA4");
        return s != nullptr && atoi(s) != 0;
    }();
    if (!ids && src0->type == GGML_TYPE_Q8_0 && ne1 >= 1 && ne1 <= prbe115_maxcols && !forced.requested()) {
        const int rd33_cc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;
        if (GGML_CUDA_CC_IS_RDNA3_0(rd33_cc) || (prbe115_rdna4 && GGML_CUDA_CC_IS_RDNA4(rd33_cc))) {
"""

_LAUNCH_OLD = "            rd33_launch(std::integral_constant<int, 1>{});\n"

_LAUNCH_NEW = """\
            switch (ne1) {
                case 1: rd33_launch(std::integral_constant<int, 1>{}); break;
                case 2: rd33_launch(std::integral_constant<int, 2>{}); break;
                case 3: rd33_launch(std::integral_constant<int, 3>{}); break;
                case 4: rd33_launch(std::integral_constant<int, 4>{}); break;
                case 5: rd33_launch(std::integral_constant<int, 5>{}); break;
                case 6: rd33_launch(std::integral_constant<int, 6>{}); break;
                case 7: rd33_launch(std::integral_constant<int, 7>{}); break;
                case 8: rd33_launch(std::integral_constant<int, 8>{}); break;
                default: GGML_ABORT("1301: unsupported ncols_dst %d", (int) ne1);
            }
"""

PATCHES = [
    FilePatch(
        path="ggml/src/ggml-cuda/mmvq.cu",
        description="1301 (PRBE115): Q8_0 F32-act MMVQ at MTP verify widths, opt-in RDNA4",
        language="none",
        edits=(
            Edit(
                id="prbe115-gate",
                anchor=re.escape(_GATE_OLD),
                mode="replace",
                text=_GATE_NEW,
                guard=r"bigcherry 1301 \(PRBE115\)",
                rationale="1241's Q8_0 F32-activation eligibility gate (ne1 == 1, RDNA3 only); widened to env-selected widths and optional RDNA4.",
                expect_matches=1,
                max_span_lines=4,
            ),
            Edit(
                id="prbe115-launch",
                anchor=re.escape(_LAUNCH_OLD),
                mode="replace",
                text=_LAUNCH_NEW,
                guard=r"default: GGML_ABORT\(\"1301: unsupported ncols_dst",
                rationale="1241's single width-1 launch of the rd33 lambda; dispatch each admitted width to its own compile-time instantiation.",
                expect_matches=1,
                max_span_lines=2,
            ),
        ),
    ),
]
