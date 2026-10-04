"""1327 (QFP13 post-bump): compute the QSA dead-slot remap terms that only depend on host data on the host.

Upstream #29819 (in pin 0504396) made build_qsa_sel remap every dead selection slot to its own dump row:
    live_tail = clamp(scale_bias(cast(tail_idxs), -1, n_kv), 0, 1)          // per QSA layer: CAST, SCALE, CLAMP
    dump      = scale_bias(cumsum(fill(live, 1)), 1, n_kv - 1)              // per QSA layer: FILL, CUMSUM, SCALE
    idx       = dump + live*(idx - dump)
The kernel census after the bump showed +~54 kernels per token per XTX (elementwise +37, get/set_rows +9) and decode
-2.3% at ~8K. tail_idxs is a host-set input and dump is n_kv + slot, so by default (BIGCHERRY_QSA_HOST_REMAP=0 disables) both are
computed once per graph on the host in the kpool input's set_input (bit-identical values: t < n_kv ? 1 : 0, and
n_kv + s exactly representable in F32 for n_kv < 2^24) and fed to every QSA layer as inputs, removing six ops per QSA
layer. The live_pool part (from the device top_k scores) and the final remap stay on the device unchanged.
Requires nothing; anchors on the upstream #29819 code.
"""

from __future__ import annotations

import re

from bigcherry.patcher import Edit, EnvDoc, FilePatch

GROUP = "rdna-boosts"
STATE = "validated"

_A_MEMBERS = "    ggml_tensor * new_pool_pos  = nullptr; // I32 [4*n_new]        M-RoPE position of each new block's first member\n"
_N_MEMBERS = _A_MEMBERS + ("    ggml_tensor * bc_live_tail  = nullptr; // bigcherry 1327: F32 [kpool - 1, n_tokens] tail cell live (1) / sentinel (0)\n"
                           "    ggml_tensor * bc_dump       = nullptr; // bigcherry 1327: F32 [n_sel, n_tokens]     dump row n_kv + slot\n")

_A_SET = ("        mctx->set_input_kpool(pool_cells, pool_idxs, pool_mask, tail_idxs, nullptr, false, new_pool_idxs, new_pool_rep,\n"
          "                              ubatch, new_pool_pos);\n"
          "    }\n")
_N_SET = ("        mctx->set_input_kpool(pool_cells, pool_idxs, pool_mask, tail_idxs, nullptr, false, new_pool_idxs, new_pool_rep,\n"
          "                              ubatch, new_pool_pos);\n"
          "        if (bc_live_tail != nullptr) {  // bigcherry 1327: host-computed remap terms (same values as the device ops)\n"
          "            GGML_ASSERT(ggml_backend_buffer_is_host(bc_live_tail->buffer) && ggml_backend_buffer_is_host(bc_dump->buffer));\n"
          "            const int32_t * t = (const int32_t *) tail_idxs->data;\n"
          "            float * lt = (float *) bc_live_tail->data;\n"
          "            for (int64_t i = 0; i < ggml_nelements(bc_live_tail); ++i) {\n"
          "                lt[i] = t[i] < (int32_t) n_kv ? 1.0f : 0.0f;  // == clamp(n_kv - t, 0, 1), t <= n_kv\n"
          "            }\n"
          "            float * d = (float *) bc_dump->data;\n"
          "            for (int64_t j = 0; j < bc_dump->ne[1]; ++j) {\n"
          "                for (int64_t s = 0; s < bc_dump->ne[0]; ++s) {\n"
          "                    d[j*bc_dump->ne[0] + s] = (float) ((int64_t) n_kv + s);\n"
          "                }\n"
          "            }\n"
          "        }\n"
          "    }\n")

_A_BUILD = ("    inp->n_sel      = kpool*std::min<uint32_t>(n_pool, hparams.indexer_top_k / kpool) + kpool - 1;\n")
_N_BUILD = _A_BUILD + (
    "    {   // bigcherry 1327: host-side QSA remap inputs (bit-identical, on by default; BIGCHERRY_QSA_HOST_REMAP=0 disables)\n"
    "        static const bool bc_host_remap = getenv(\"BIGCHERRY_QSA_HOST_REMAP\") == nullptr || atoi(getenv(\"BIGCHERRY_QSA_HOST_REMAP\")) != 0;\n"
    "        if (bc_host_remap) {\n"
    "            inp->bc_live_tail = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, kpool - 1, n_tokens);\n"
    "            inp->bc_dump      = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, inp->n_sel, n_tokens);\n"
    "            ggml_set_input(inp->bc_live_tail);\n"
    "            ggml_set_input(inp->bc_dump);\n"
    "            ggml_build_forward_expand(gf, inp->bc_live_tail);\n"
    "            ggml_build_forward_expand(gf, inp->bc_dump);\n"
    "        }\n"
    "    }\n")

_A_REMAP = ("    ggml_tensor * live_tail = ggml_cast(ctx0, inp_kpool->tail_idxs, GGML_TYPE_F32);\n"
            "    live_tail = ggml_clamp(ctx0, ggml_scale_bias(ctx0, live_tail, -1.0f, (float) n_kv), 0.0f, 1.0f);\n"
            "    ggml_tensor * live = ggml_concat(ctx0, live_pool, live_tail, 0); // [n_sel, n_tokens]\n"
            "\n"
            "    // dump rows n_kv + slot as a cumulative sum: the meta backend cannot split an arange, which has no source\n"
            "    ggml_tensor * dump  = ggml_scale_bias(ctx0, ggml_cumsum(ctx0, ggml_fill(ctx0, live, 1.0f)), 1.0f, (float) (n_kv - 1));\n")
_N_REMAP = ("    ggml_tensor * live_tail = inp_kpool->bc_live_tail;  // bigcherry 1327: host-computed when enabled\n"
            "    if (live_tail == nullptr) {\n"
            "        live_tail = ggml_cast(ctx0, inp_kpool->tail_idxs, GGML_TYPE_F32);\n"
            "        live_tail = ggml_clamp(ctx0, ggml_scale_bias(ctx0, live_tail, -1.0f, (float) n_kv), 0.0f, 1.0f);\n"
            "    }\n"
            "    ggml_tensor * live = ggml_concat(ctx0, live_pool, live_tail, 0); // [n_sel, n_tokens]\n"
            "\n"
            "    // dump rows n_kv + slot as a cumulative sum: the meta backend cannot split an arange, which has no source\n"
            "    ggml_tensor * dump = inp_kpool->bc_dump;  // bigcherry 1327: host input n_kv + slot when enabled\n"
            "    if (dump == nullptr) {\n"
            "        dump = ggml_scale_bias(ctx0, ggml_cumsum(ctx0, ggml_fill(ctx0, live, 1.0f)), 1.0f, (float) (n_kv - 1));\n"
            "    }\n")

PATCHES = [
    FilePatch(
        path="src/models/qwen4exp.cpp",
        description="1327: host-computed QSA dead-slot remap terms (on by default; BIGCHERRY_QSA_HOST_REMAP=0 disables)",
        language="none",
        edits=(
            Edit(id="qsa-remap-include", anchor=re.escape("#include <algorithm>\n"), mode="insert_after",
                 text="#include <cstdlib>  // bigcherry 1327: getenv/atoi\n", guard=r"#include <cstdlib>  // bigcherry 1327",
                 rationale="Standard include block of qwen4exp.cpp.", expect_matches=1, max_span_lines=2),
            Edit(id="qsa-remap-members", anchor=re.escape(_A_MEMBERS), mode="replace", text=_N_MEMBERS,
                 guard=r"bigcherry 1327: F32 \[kpool - 1, n_tokens\]", rationale="llm_graph_input_kpool member list.",
                 expect_matches=1, max_span_lines=2),
            Edit(id="qsa-remap-set-input", anchor=re.escape(_A_SET), mode="replace", text=_N_SET,
                 guard=r"bigcherry 1327: host-computed remap terms", rationale="kpool set_input after set_input_kpool.",
                 expect_matches=1, max_span_lines=4),
            Edit(id="qsa-remap-build", anchor=re.escape(_A_BUILD), mode="replace", text=_N_BUILD,
                 guard=r"bigcherry 1327: host-side QSA remap inputs", rationale="build_inp_kpool once n_sel is known.",
                 expect_matches=1, max_span_lines=2),
            Edit(id="qsa-remap-use", anchor=re.escape(_A_REMAP), mode="replace", text=_N_REMAP,
                 guard=r"bigcherry 1327: host-computed when enabled", rationale="build_qsa_sel #29819 remap terms.",
                 expect_matches=1, max_span_lines=7),
        ),
    ),
]

ENV_DOCS = (
    EnvDoc('BIGCHERRY_QSA_HOST_REMAP', '0|1', '1 (on)',
           'compute QSA dead-slot remap terms on the host (removes six ops per QSA layer)'),
)
