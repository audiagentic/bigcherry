"""1286: give a DFlash/DSpark draft its own copy of target tensors that live in a tensor-split buffer.

DFlash and DSpark drafts borrow the target model's tok_embd / output / output_s through ctx_other. With
the target on -sm tensor those tensors sit in the meta (tensor-split) buffer, and a single-device draft
graph aborts: "pre-allocated tensor (output.weight) in a buffer (Meta()) that cannot run the operation".
At draft-context creation, each borrowed tensor that lives in a meta buffer is copied (gathered with
ggml_backend_tensor_get) into a buffer on the draft's own device; the graph builders prefer the copies.
Layer-split and single-device targets are unchanged (nothing is copied).
"""

import re as _re

from bigcherry.patcher import Edit, FilePatch

GROUP = "core"
STATE = "untested"

_CPARAMS = FilePatch(
    path="src/llama-cparams.h",
    language="none",
    description="Draft-local copies of borrowed target tensors.",
    edits=(
        Edit(
            id="cparams-other-copies",
            anchor=_re.escape("    llama_context * ctx_other;\n"),
            text=(
                "    // BigCherry 1286: draft-local copies of target tensors that live in a tensor-split buffer\n"
                "    ggml_tensor * other_tok_embd = nullptr;\n"
                "    ggml_tensor * other_output   = nullptr;\n"
                "    ggml_tensor * other_output_s = nullptr;\n"
            ),
            mode="insert_after",
            guard=r"BigCherry 1286: draft-local copies",
            expect_matches=1,
            rationale="The graph builders read cparams; the copies are set by the context that owns them.",
        ),
    ),
)

_CONTEXT_H = FilePatch(
    path="src/llama-context.h",
    language="none",
    description="Own the draft-local copies.",
    edits=(
        Edit(
            id="context-other-copy-buffers",
            anchor=_re.escape("    ggml_backend_buffer_ptr buf_output;\n"),
            text=(
                "\n    // BigCherry 1286: draft-local copies of target tensors that live in a tensor-split buffer\n"
                "    ggml_context_ptr        bc_other_ctx;\n"
                "    ggml_backend_buffer_ptr bc_other_buf;\n"
            ),
            mode="insert_after",
            guard=r"bc_other_ctx;",
            expect_matches=1,
            rationale="The copies must live as long as the draft context.",
        ),
    ),
)

_COPY = r"""
    // BigCherry 1286: a single-device DFlash/DSpark draft cannot read target tensors that live in the
    // target's tensor-split (meta) buffer; copy each such borrowed tensor onto the draft's device.
    if (cparams.ctx_other != nullptr && model.arch == LLM_ARCH_DFLASH && !model.devices.empty() && !model.devices[0].is_meta) {
        const llama_model * other = llama_get_model(cparams.ctx_other);
        auto in_meta = [](const ggml_tensor * t) {
            if (t == nullptr || t->buffer == nullptr) {
                return false;
            }
            ggml_backend_dev_t d = ggml_backend_buft_get_device(ggml_backend_buffer_get_type(t->buffer));
            return d != nullptr && ggml_backend_dev_type(d) == GGML_BACKEND_DEVICE_TYPE_META;
        };
        ggml_tensor * src[3] = {
            model.tok_embd == nullptr ? other->tok_embd : nullptr,
            model.output   == nullptr ? other->output   : nullptr,
            model.output   == nullptr ? other->output_s : nullptr,
        };
        ggml_tensor ** dst[3] = { &cparams.other_tok_embd, &cparams.other_output, &cparams.other_output_s };
        bool any = false;
        for (ggml_tensor * t : src) {
            any = any || in_meta(t);
        }
        if (any) {
            ggml_init_params ip = { 3*ggml_tensor_overhead(), nullptr, true };
            bc_other_ctx.reset(ggml_init(ip));
            for (int i = 0; i < 3; ++i) {
                if (in_meta(src[i])) {
                    *dst[i] = ggml_dup_tensor(bc_other_ctx.get(), src[i]);
                    ggml_set_name(*dst[i], src[i]->name);
                }
            }
            bc_other_buf.reset(ggml_backend_alloc_ctx_tensors_from_buft(
                bc_other_ctx.get(), ggml_backend_dev_buffer_type(model.devices[0].dev)));
            if (!bc_other_buf) {
                throw std::runtime_error("BigCherry 1286: failed to allocate draft-local copies of target tensors");
            }
            std::vector<uint8_t> tmp;
            for (int i = 0; i < 3; ++i) {
                if (*dst[i] == nullptr) {
                    continue;
                }
                tmp.resize(ggml_nbytes(src[i]));
                ggml_backend_tensor_get(src[i], tmp.data(), 0, tmp.size());
                ggml_backend_tensor_set(*dst[i], tmp.data(), 0, tmp.size());
                LLAMA_LOG_INFO("%s: BigCherry 1286: copied target %s (%.2f MiB) from the tensor-split buffer to %s\n",
                    __func__, src[i]->name, tmp.size()/1048576.0, ggml_backend_dev_name(model.devices[0].dev));
            }
        }
    }
"""

_CONTEXT_CPP = FilePatch(
    path="src/llama-context.cpp",
    language="none",
    description="Copy borrowed target tensors out of the tensor-split buffer for single-device drafts.",
    edits=(
        Edit(
            id="context-copy-other-tensors",
            anchor=_re.escape(
                "                throw std::runtime_error(model.arch_name() + \" requires ctx_other to be set (this warning is normal during memory fitting)\");\n"
                "            }\n"
                "            cparams.ctx_other = params.ctx_other;\n"
                "        }\n"
                "    }\n"
            ),
            text=_COPY,
            mode="insert_after",
            guard=r"BigCherry 1286: a single-device DFlash/DSpark draft",
            expect_matches=1,
            rationale="Right after ctx_other is bound for EAGLE3/DFlash drafts; model and target are fully loaded here.",
        ),
    ),
)

_DFLASH = FilePatch(
    path="src/models/dflash.cpp",
    language="none",
    description="Prefer the draft-local copies of tok_embd / output / output_s.",
    edits=(
        Edit(
            id="dflash-tok-embd-copy",
            anchor=_re.escape("        tok_embd = model_other->tok_embd;\n"),
            text="        tok_embd = cparams.other_tok_embd ? cparams.other_tok_embd : model_other->tok_embd; // BigCherry 1286\n",
            mode="replace_all",
            guard=r"cparams\.other_tok_embd \? cparams\.other_tok_embd",
            expect_matches=2,
            rationale="DFlash and DSpark builders both borrow tok_embd.",
        ),
        Edit(
            id="dflash-output-copy",
            anchor=_re.escape("        output   = model_other->output;\n        output_s = model_other->output_s;\n"),
            text=(
                "        output   = cparams.other_output ? cparams.other_output   : model_other->output;   // BigCherry 1286\n"
                "        output_s = cparams.other_output ? cparams.other_output_s : model_other->output_s;\n"
            ),
            mode="replace_all",
            guard=r"cparams\.other_output \? cparams\.other_output ",
            expect_matches=2,
            rationale="DFlash and DSpark builders both borrow output/output_s.",
        ),
    ),
)

PATCHES = [_CPARAMS, _CONTEXT_H, _CONTEXT_CPP, _DFLASH]
