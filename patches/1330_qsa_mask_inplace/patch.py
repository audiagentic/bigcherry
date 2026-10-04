"""1330 (QFP17): build the QSA attention mask in one dense buffer instead of two.

build_qsa_sel creates mask_all = repeat(fill(-inf)) [n_kv + n_sel, T], scatters zeros into it with ggml_set_rows (which
returns a view of mask_all - in place), views the first n_kv rows and then adds the causal kq_mask with an
OUT-OF-PLACE ggml_add, which allocates a second dense [n_kv, T] F16 tensor. The 1329 attribution showed these dense
masks own the compute-buffer growth with -ub: ~242 MiB per tensor at ub512 and ~484 MiB at ub1024 at 240K context,
mirrored on every Meta rank (GPT req_2e524202a01a4acc: "two-mask live pair"). With BIGCHERRY_QSA_MASK_INPLACE=1 the
final add writes into the mask_all view (ggml_add_inplace): the values are identical (same F16 add), and one dense
mask per QSA layer is never allocated - ~484 MiB per GPU at ub1024.
"""

from __future__ import annotations

import re

from bigcherry.patcher import Edit, FilePatch

GROUP = "rdna-boosts"
STATE = "untested"

_A = ("    sel = ggml_add(ctx0, sel, kq_mask);\n"
      "    cb(sel, \"indexer_sel\", il);\n")
_N = ("    {   // bigcherry 1330: add the causal mask in place (one dense mask instead of two)\n"
      "        static const bool bc_inplace = getenv(\"BIGCHERRY_QSA_MASK_INPLACE\") != nullptr &&\n"
      "                                       atoi(getenv(\"BIGCHERRY_QSA_MASK_INPLACE\")) != 0;\n"
      "        sel = bc_inplace ? ggml_add_inplace(ctx0, sel, kq_mask) : ggml_add(ctx0, sel, kq_mask);\n"
      "    }\n"
      "    cb(sel, \"indexer_sel\", il);\n")

_A2 = "    ggml_tensor * mask    = ggml_reshape_4d(ctx0, sel, kq_mask->ne[0], kq_mask->ne[1], kq_mask->ne[2], kq_mask->ne[3]);\n"
_N2 = ("    // bigcherry 1330: the in-place selection mask is a strided view of [n_kv + n_sel, T] already shaped like kq_mask;\n"
       "    // flash attention reads the mask by its own row stride, so pass it through instead of reshaping (needs contiguity)\n"
       "    ggml_tensor * mask    = ggml_are_same_shape(sel, kq_mask) ? sel\n"
       "        : ggml_reshape_4d(ctx0, sel, kq_mask->ne[0], kq_mask->ne[1], kq_mask->ne[2], kq_mask->ne[3]);\n")

_A3 = ("    ggml_tensor * mask_all = ggml_new_tensor_4d(ctx0, kq_mask->type, n_kv + n_sel, 1, 1, 1);\n"
       "    mask_all = ggml_fill(ctx0, mask_all, -INFINITY);\n"
       "    mask_all = ggml_repeat_4d(ctx0, mask_all, n_kv + n_sel, n_tokens, 1, 1);\n"
       "    mask_all = ggml_reshape_3d(ctx0, mask_all, 1, n_kv + n_sel, n_tokens);\n")
_N3 = ("    // bigcherry 1330: with the in-place mask the attention mask is a strided view of this buffer, so pad its row to\n"
       "    // 256 elements (aligned half2 / vector mask loads); the extra rows stay -inf and are never selected\n"
       "    static const bool bc_mask_inplace = getenv(\"BIGCHERRY_QSA_MASK_INPLACE\") != nullptr &&\n"
       "                                        atoi(getenv(\"BIGCHERRY_QSA_MASK_INPLACE\")) != 0;\n"
       "    const int64_t bc_mask_rows = bc_mask_inplace ? GGML_PAD(n_kv + n_sel, 256) : n_kv + n_sel;\n"
       "    ggml_tensor * mask_all = ggml_new_tensor_4d(ctx0, kq_mask->type, bc_mask_rows, 1, 1, 1);\n"
       "    mask_all = ggml_fill(ctx0, mask_all, -INFINITY);\n"
       "    mask_all = ggml_repeat_4d(ctx0, mask_all, bc_mask_rows, n_tokens, 1, 1);\n"
       "    mask_all = ggml_reshape_3d(ctx0, mask_all, 1, bc_mask_rows, n_tokens);\n")

_A4 = ("        GGML_ASSERT(mask->type == GGML_TYPE_F16);\n"
       "        GGML_ASSERT(ggml_is_contiguous(mask));\n"
       "        //GGML_ASSERT(ggml_can_repeat_rows(mask, qk));\n")
_N4 = ("        GGML_ASSERT(mask->type == GGML_TYPE_F16);\n"
       "        // bigcherry 1330: rows must be contiguous; backends index mask rows by nb[1..3] (CPU ops.cpp, CUDA/HIP\n"
       "        // fattn-common), so a strided row view (the in-place QSA mask) is valid\n"
       "        GGML_ASSERT(mask->nb[0] == ggml_type_size(mask->type));\n"
       "        //GGML_ASSERT(ggml_can_repeat_rows(mask, qk));\n")

PATCHES = [
    FilePatch(
        path="src/models/qwen4exp.cpp",
        description="1330: in-place causal-mask add in build_qsa_sel (BIGCHERRY_QSA_MASK_INPLACE=1)",
        language="none",
        edits=(
            Edit(id="qsa-mask-inplace-include", anchor=re.escape("#include <algorithm>\n"), mode="insert_after",
                 text="#include <cstdlib>  // bigcherry 1330: getenv/atoi\n", guard=r"#include <cstdlib>  // bigcherry 1330",
                 rationale="Standard include block of qwen4exp.cpp.", expect_matches=1, max_span_lines=2),
            Edit(id="qsa-mask-inplace", anchor=re.escape(_A), mode="replace", text=_N,
                 guard=r"bigcherry 1330: add the causal mask in place", rationale="Final mask add in build_qsa_sel.",
                 expect_matches=1, max_span_lines=3),
            Edit(id="qsa-mask-padded-rows", anchor=re.escape(_A3), mode="replace", text=_N3,
                 guard=r"bigcherry 1330: with the in-place mask the attention mask is a strided view", rationale="mask_all construction.",
                 expect_matches=1, max_span_lines=5),
            Edit(id="qsa-mask-passthrough", anchor=re.escape(_A2), mode="replace", text=_N2,
                 guard=r"bigcherry 1330: the in-place selection mask is a strided view", rationale="build_attn_qsa mask.",
                 expect_matches=1, max_span_lines=2),
        ),
    ),
    FilePatch(
        path="ggml/src/ggml.c",
        description="1330: ggml_flash_attn_ext accepts a mask with contiguous rows and any row stride",
        language="none",
        edits=(
            Edit(id="fa-mask-row-contiguous", anchor=re.escape(_A4), mode="replace", text=_N4,
                 guard=r"bigcherry 1330: rows must be contiguous", rationale="ggml_flash_attn_ext mask checks.",
                 expect_matches=1, max_span_lines=4),
        ),
    ),
]
