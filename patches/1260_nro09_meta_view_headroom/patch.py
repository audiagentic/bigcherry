"""PNRO09: Increase Meta compute-container view headroom for recurrent/MTP graphs.

The current compute_headroom=16 in ggml-backend-meta.cpp sizes the compute
buffer as 16x the static buffer size. For recurrent+MTP graphs with
n_rs_seq>0, the GDN output tensor creates 2 views per snapshot slot
(output + src), giving 2*(n_rs_seq+1) views per recurrent layer.

For n_rs_seq=8 with 4 recurrent layers: 2*9*4 = 72 views, exceeding 16.

This patch increases compute_headroom to 80 to cover the maximum supported
recurrent+MTP configuration (n_rs_seq=8, 4 recurrent layers = 72 views)
with a small margin.

The change is correctness-scoped: it only increases the compute buffer
size, which affects memory allocation but not numerical behavior.
"""

from bigcherry.patcher import Edit, FilePatch

PATCHES = [
    FilePatch(
        path="vendor/llama.cpp/ggml/src/ggml-backend-meta.cpp",
        edits=(
            Edit(
                id="pnro09_increase_headroom",
                anchor="constexpr size_t compute_headroom = 16;",
                text=(
                    "constexpr size_t compute_headroom = 80; // PNRO09: increased from 16 to cover recurrent+MTP graphs.\n"
                    "    // 2*(n_rs_seq+1) views per recurrent layer; n_rs_seq=8, 4 layers = 72 views.\n"
                    "    // The 16-view bound was insufficient for recurrent+MTP configurations with\n"
                    "    // n_rs_seq >= 3 and multiple recurrent layers."
                ),
                mode="replace",
                guard="constexpr size_t compute_headroom = 80;",
                rationale="Increase compute_headroom from 16 to 80 to cover recurrent+MTP graphs with n_rs_seq up to 8 and up to 4 recurrent layers.",
            ),
        ),
    ),
]
