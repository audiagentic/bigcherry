"""1297: Qwen4Exp MTP draft computes its logits over a trimmed vocabulary (opt-in, BIGCHERRY_DRAFT_VOCAB_N).

The MTP draft GGUF carries its own output.weight (Q8_0, 2560 x 248320, ~675 MB), read in full for every
draft token: ~31% of the draft GPU's kernel time per MTP step at 80K on the 6900 (3 x ~1.35 ms). Drafts only
need to propose likely tokens; the target verifies every one, so output cannot change - only acceptance.

With BIGCHERRY_DRAFT_VOCAB_N=N, the MTP-only Qwen4Exp context copies output rows [0, N) - byte-level BPE ids
are ordered roughly by merge frequency - plus every control / user-defined / end-of-generation token into a
draft-local tensor of the same type at context creation. The MTP graph multiplies by that tensor and
scatters the result into full-vocabulary logits filled with -inf, so sampling and verification see the
usual [n_vocab, n_out] logits. Unset (default): unchanged. A model with output_s or no output is unchanged.
"""

import re as _re

from bigcherry.patcher import Edit, FilePatch

GROUP = "core"
STATE = "untested"

_CPARAMS = FilePatch(
    path="src/llama-cparams.h",
    language="none",
    description="Draft trimmed-vocabulary head tensors.",
    edits=(
        Edit(
            id="cparams-draft-vocab-trim",
            anchor=_re.escape("    llama_context * ctx_other;\n"),
            text=(
                "    // BigCherry 1297: trimmed draft head [n_embd, n_trim] and its token ids [n_trim] (MTP drafts)\n"
                "    ggml_tensor * bc_out_trim     = nullptr;\n"
                "    ggml_tensor * bc_out_trim_ids = nullptr;\n"
            ),
            mode="insert_after",
            guard=r"BigCherry 1297: trimmed draft head",
            expect_matches=1,
            rationale="Graph builders read cparams; the owning context sets these.",
        ),
    ),
)

_CONTEXT_H = FilePatch(
    path="src/llama-context.h",
    language="none",
    description="Own the trimmed draft head.",
    edits=(
        Edit(
            id="context-draft-vocab-trim-buffers",
            anchor=_re.escape("    ggml_backend_buffer_ptr buf_output;\n"),
            text=(
                "\n    // BigCherry 1297: trimmed MTP draft head\n"
                "    ggml_context_ptr        bc_trim_ctx;\n"
                "    ggml_backend_buffer_ptr bc_trim_buf;\n"
            ),
            mode="insert_after",
            guard=r"bc_trim_ctx;",
            expect_matches=1,
            rationale="The trimmed head must live as long as the draft context.",
        ),
    ),
)

_BUILD = r"""    // BigCherry 1297: MTP-only Qwen4Exp draft - optional trimmed-vocabulary head (BIGCHERRY_DRAFT_VOCAB_N)
    if (model.arch == LLM_ARCH_QWEN4EXP && model.output != nullptr && model.output_s == nullptr &&
            model.output->buffer != nullptr && !model.layers.empty() && model.layers[0].hc_attn_norm == nullptr) {
        const char * e = getenv("BIGCHERRY_DRAFT_VOCAB_N");
        const int64_t n_vocab = model.output->ne[1];
        const int64_t n_keep  = e != nullptr ? (int64_t) atoll(e) : 0;
        if (n_keep > 0 && n_keep < n_vocab) {
            const llama_vocab & vocab = model.vocab;
            std::vector<int32_t> ids;
            ids.reserve(n_keep + 512);
            for (int64_t i = 0; i < n_keep; ++i) {
                ids.push_back((int32_t) i);
            }
            for (int64_t i = n_keep; i < n_vocab && i < (int64_t) vocab.n_tokens(); ++i) {
                if (vocab.is_control((llama_token) i) || vocab.is_user_defined((llama_token) i) || vocab.is_eog((llama_token) i)) {
                    ids.push_back((int32_t) i);
                }
            }
            const int64_t n_trim = (int64_t) ids.size();
            ggml_init_params ip = { 2*ggml_tensor_overhead(), nullptr, true };
            bc_trim_ctx.reset(ggml_init(ip));
            ggml_tensor * w = ggml_new_tensor_2d(bc_trim_ctx.get(), model.output->type, model.output->ne[0], n_trim);
            ggml_tensor * t = ggml_new_tensor_1d(bc_trim_ctx.get(), GGML_TYPE_I32, n_trim);
            ggml_set_name(w, "output_trim.weight");
            ggml_set_name(t, "output_trim.ids");
            bc_trim_buf.reset(ggml_backend_alloc_ctx_tensors_from_buft(bc_trim_ctx.get(),
                ggml_backend_buffer_get_type(model.output->buffer)));
            if (!bc_trim_buf) {
                throw std::runtime_error("BigCherry 1297: failed to allocate the trimmed draft head");
            }
            const size_t row = model.output->nb[1];
            std::vector<uint8_t> full(ggml_nbytes(model.output));
            ggml_backend_tensor_get(model.output, full.data(), 0, full.size());
            std::vector<uint8_t> rows((size_t) n_trim*row);
            for (int64_t j = 0; j < n_trim; ++j) {
                memcpy(rows.data() + (size_t) j*row, full.data() + (size_t) ids[j]*row, row);
            }
            ggml_backend_tensor_set(w, rows.data(), 0, rows.size());
            ggml_backend_tensor_set(t, ids.data(), 0, ids.size()*sizeof(int32_t));
            cparams.bc_out_trim     = w;
            cparams.bc_out_trim_ids = t;
            LLAMA_LOG_INFO("%s: BigCherry 1297: draft head trimmed to %lld of %lld tokens (%.1f MiB)\n",
                __func__, (long long) n_trim, (long long) n_vocab, rows.size()/1048576.0);
        }
    }

"""

_CONTEXT_CPP = FilePatch(
    path="src/llama-context.cpp",
    language="none",
    description="Build the trimmed MTP draft head at context creation.",
    edits=(
        Edit(
            id="context-draft-vocab-trim",
            anchor=_re.escape("    if (cparams.rope_scaling_type == LLAMA_ROPE_SCALING_TYPE_UNSPECIFIED) {\n"),
            text=_BUILD,
            mode="insert_before",
            guard=r"BigCherry 1297: MTP-only Qwen4Exp draft",
            expect_matches=1,
            rationale="In the context constructor after cparams setup; the model is fully loaded here.",
        ),
    ),
)

_HEAD = """            nullptr, nullptr, il);
    cb(cur, "result_norm", -1);
    res->t_embd = cur;

    if (cparams.bc_out_trim != nullptr && ggml_nelements(cur) > 0) {
        // BigCherry 1297 (batches with no output rows - MTP prompt replay - keep the plain head): draft logits over the trimmed vocabulary, scattered into -inf full-vocabulary logits
        // flatten whatever leading layout the head input has to [n_embd, n_out]
        const int64_t n_out   = ggml_nelements(cur) / cur->ne[0];
        const int64_t n_trim  = cparams.bc_out_trim->ne[1];
        const int64_t n_vocab = model.output->ne[1];
        cur = ggml_reshape_2d(ctx0, ggml_cont(ctx0, cur), cur->ne[0], n_out);
        ggml_tensor * small = ggml_mul_mat(ctx0, cparams.bc_out_trim, cur); // [n_trim, n_out]
        ggml_tensor * idx = cparams.bc_out_trim_ids;
        for (int64_t t = 1; t < n_out; ++t) {
            idx = ggml_concat(ctx0, idx, cparams.bc_out_trim_ids, 0);
        }
        ggml_tensor * full = ggml_fill(ctx0, ggml_new_tensor_3d(ctx0, GGML_TYPE_F32, 1, n_vocab, n_out), -INFINITY);
        full = ggml_set_rows(ctx0, full, ggml_reshape_3d(ctx0, small, 1, n_trim, n_out),
                ggml_reshape_3d(ctx0, idx, n_trim, n_out, 1));
        cur = ggml_reshape_2d(ctx0, full, n_vocab, n_out);
    } else {
        cur = build_lora_mm(model.output, cur, model.output_s);
    }
"""

_MODEL = FilePatch(
    path="src/models/qwen4exp.cpp",
    language="none",
    description="MTP graph: trimmed-vocabulary head when the context provides one.",
    edits=(
        Edit(
            id="mtp-trimmed-head",
            anchor=_re.escape(
                "            nullptr, nullptr, il);\n"
                "    cb(cur, \"result_norm\", -1);\n"
                "    res->t_embd = cur;\n"
                "\n"
                "    cur = build_lora_mm(model.output, cur, model.output_s);\n"
            ),
            text=_HEAD,
            mode="replace",
            guard=r"BigCherry 1297: draft logits over the trimmed vocabulary",
            expect_matches=1,
            rationale="The MTP graph's head (its hc mix ends with il, the target graph's with -1).",
        ),
    ),
)

PATCHES = [_CPARAMS, _CONTEXT_H, _CONTEXT_CPP, _MODEL]
