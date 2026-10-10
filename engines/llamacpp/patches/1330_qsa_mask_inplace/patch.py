"""1330 (QFP17): build the dense-fallback QSA attention mask in one buffer instead of two, after 1332.

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

from bigcherry.patcher import Edit, EnvDoc, FilePatch

GROUP = "rdna-boosts"
STATE = "untested"

_A = ("    sel = ggml_add(ctx0, sel, kq_mask);\n"
      "    cb(sel, \"indexer_sel\", il);\n")
_N = ("    {   // bigcherry 1330: large prompt batches only; decode/MTP <=8 keep the exact dense fallback\n"
      "        const bool bc_inplace = n_tokens > 8 && getenv(\"BIGCHERRY_QSA_MASK_INPLACE\") != nullptr &&\n"
      "                                               atoi(getenv(\"BIGCHERRY_QSA_MASK_INPLACE\")) != 0;\n"
      "        sel = bc_inplace ? ggml_add_inplace(ctx0, sel, kq_mask) : ggml_add(ctx0, sel, kq_mask);\n"
      "    }\n"
      "    cb(sel, \"indexer_sel\", il);\n")

_A2 = "        ggml_tensor * mask = ggml_reshape_4d(ctx0, sel, kq_mask->ne[0], kq_mask->ne[1], kq_mask->ne[2], kq_mask->ne[3]);\n"
_N2 = ("        // bigcherry 1330: only a >8-token in-place dense fallback can pass its strided mask through directly\n"
       "        // (BIGCHERRY_QSA_MASK_INPLACE=2: diagnostic, make it contiguous first to isolate the strided-mask read)\n"
       "        static const int bc_mask_mode = getenv(\"BIGCHERRY_QSA_MASK_INPLACE\") ? atoi(getenv(\"BIGCHERRY_QSA_MASK_INPLACE\")) : 0;\n"
       "        const bool bc_mask_inplace_batch = bc_mask_mode != 0 && q_cur->ne[2] > 8;\n"
       "        ggml_tensor * mask = bc_mask_inplace_batch && ggml_are_same_shape(sel, kq_mask)\n"
       "            ? (bc_mask_mode == 2 ? ggml_cont(ctx0, sel) : sel)\n"
       "            : ggml_reshape_4d(ctx0, sel, kq_mask->ne[0], kq_mask->ne[1], kq_mask->ne[2], kq_mask->ne[3]);\n")

_A3 = ("    ggml_tensor * mask_all = ggml_new_tensor_4d(ctx0, kq_mask->type, n_kv + n_sel, 1, 1, 1);\n"
       "    mask_all = ggml_fill(ctx0, mask_all, -INFINITY);\n"
       "    mask_all = ggml_repeat_4d(ctx0, mask_all, n_kv + n_sel, n_tokens, 1, 1);\n"
       "    mask_all = ggml_reshape_3d(ctx0, mask_all, 1, n_kv + n_sel, n_tokens);\n")
_N3 = ("    // bigcherry 1330: only >8-token prompt batches use the in-place strided mask; decode/MTP retains\n"
       "    // the exact production allocation/reshape path. Pad prompt rows to 256 for aligned vector mask loads.\n"
       "    const bool bc_mask_inplace = n_tokens > 8 && getenv(\"BIGCHERRY_QSA_MASK_INPLACE\") != nullptr &&\n"
       "                                                    atoi(getenv(\"BIGCHERRY_QSA_MASK_INPLACE\")) != 0;\n"
       "    const int64_t bc_mask_rows = bc_mask_inplace ? GGML_PAD(n_kv + n_sel, 256) : n_kv + n_sel;\n"
       "    ggml_tensor * mask_all = ggml_new_tensor_4d(ctx0, kq_mask->type, bc_mask_rows, 1, 1, 1);\n"
       "    mask_all = ggml_fill(ctx0, mask_all, -INFINITY);\n"
       "    mask_all = ggml_repeat_4d(ctx0, mask_all, bc_mask_rows, n_tokens, 1, 1);\n"
       "    mask_all = ggml_reshape_3d(ctx0, mask_all, 1, bc_mask_rows, n_tokens);\n")

PATCHES = [
    FilePatch(
        path="src/models/qwen4exp.cpp",
        description="1330: in-place causal-mask add in build_qsa_sel (BIGCHERRY_QSA_MASK_INPLACE=1)",
        language="none",
        edits=(
            Edit(id="qsa-mask-inplace", anchor=re.escape(_A), mode="replace", text=_N,
                 guard=r"bigcherry 1330: large prompt batches only", rationale="Final mask add in build_qsa_sel.",
                 expect_matches=1, max_span_lines=3),
            Edit(id="qsa-mask-padded-rows", anchor=re.escape(_A3), mode="replace", text=_N3,
                 guard=r"bigcherry 1330: only >8-token prompt batches use the in-place strided mask", rationale="mask_all construction.",
                 expect_matches=1, max_span_lines=5),
            Edit(id="qsa-mask-passthrough", anchor=re.escape(_A2), mode="replace", text=_N2,
                 guard=r"bigcherry 1330: only a >8-token in-place dense fallback can pass its strided mask", rationale="build_attn_qsa dense fallback mask.",
                 expect_matches=1, max_span_lines=2),
        ),
    ),
]

ENV_DOCS = (
    EnvDoc('BIGCHERRY_QSA_MASK_INPLACE', '0|1|2', '0',
           'experimental: in-place QSA causal-mask add for prompt batches >8 tokens; decode/MTP keeps production path (2 = contiguous copy before FA, diagnostic)'),
)
