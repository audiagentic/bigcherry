"""1280: keep the qwen4exp k-pool k_idxs input allocated (MTP draft under -sm tensor, QFN01).

build_inp_kpool forward-expands every k-pool input except k_idxs, which only a
QSA layer's indexer cpy_k reads. The MTP draft graph has no QSA layer, so under
the meta (tensor-split) backend k_idxs gets no buffer and set_input_k_idxs hits
GGML_ASSERT(buffer) during server load.
"""

import re as _re

from bigcherry.patcher import Edit, FilePatch

GROUP = "upstream-fixes"
STATE = "untested"

_ANCHOR = "    ggml_build_forward_expand(gf, inp->pool_cells);\n"

PATCHES = [
    FilePatch(
        path="src/models/qwen4exp.cpp",
        language="none",
        description="Forward-expand k_idxs with the other k-pool inputs.",
        edits=(
            Edit(
                id="kpool-k-idxs-expand",
                anchor=_re.escape(_ANCHOR),
                text="    ggml_build_forward_expand(gf, inp->k_idxs); // BigCherry 1280: set_input fills it even without a QSA layer\n",
                mode="insert_before",
                guard=r"BigCherry 1280",
                expect_matches=1,
                rationale="set_input always writes k_idxs; the MTP graph has no reader, so it must be kept allocated.",
            ),
        ),
    ),
]
