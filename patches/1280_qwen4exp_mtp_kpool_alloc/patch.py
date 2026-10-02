"""1280: keep the qwen4exp k-pool inputs allocated (MTP draft under -sm tensor, QFN01).

build_inp_kpool forward-expands pool_cells/idxs/mask/tail but not k_idxs or the
new_pool_* re-pool inputs, which only a QSA layer reads. The MTP draft graph has
no QSA layer, so under the meta (tensor-split) backend they get no buffer and
set_input hits GGML_ASSERT(buffer) during server load.
"""

import re as _re

from bigcherry.patcher import Edit, FilePatch

GROUP = "upstream-fixes"
STATE = "rejected"

_NEW_POOL = (
    "    // BigCherry 1280: set_input fills the re-pool inputs too, so keep them allocated\n"
    "    ggml_build_forward_expand(gf, inp->new_pool_idxs);\n"
    "    if (inp->new_pool_rep) {\n"
    "        ggml_build_forward_expand(gf, inp->new_pool_rep);\n"
    "    }\n"
    "    ggml_build_forward_expand(gf, inp->new_pool_pos);\n"
)

PATCHES = [
    FilePatch(
        path="src/models/qwen4exp.cpp",
        language="none",
        description="Forward-expand k_idxs and the re-pool inputs with the other k-pool inputs.",
        edits=(
            Edit(
                id="kpool-k-idxs-expand",
                anchor=_re.escape("    ggml_build_forward_expand(gf, inp->pool_cells);\n"),
                text="    ggml_build_forward_expand(gf, inp->k_idxs); // BigCherry 1280: set_input fills it even without a QSA layer\n",
                mode="insert_before",
                guard=r"inp->k_idxs\); // BigCherry 1280",
                expect_matches=1,
                rationale="set_input always writes k_idxs; the MTP graph has no reader, so it must be kept allocated.",
            ),
            Edit(
                id="kpool-new-pool-expand",
                anchor=_re.escape("    ggml_set_input(inp->new_pool_pos);\n"),
                text=_NEW_POOL,
                mode="insert_after",
                guard=r"BigCherry 1280: set_input fills the re-pool inputs",
                expect_matches=1,
                rationale="new_pool_idxs/rep/pos are written by set_input_kpool; the MTP graph never reads them.",
            ),
        ),
    ),
]
