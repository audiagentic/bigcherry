"""1359 (QFP50): keep one prompt batch queued ahead of the one the cards are computing.

With an MTP drafter, the server calls the drafter hook after every target prompt batch, and the hook (1348) waits for
that batch in llama_get_embeddings_nextn before it returns. So batch k+1 is only built, allocated and submitted after
batch k has finished, and each target card has nothing queued for that long: about 34 ms a 512-token batch on
Flash-Next (kernel trace dp1), where the same prefill without a drafter is 14.6% faster (run nomtp1).

Nothing else in the batch path waits for the previous batch, so the change is confined to the hidden-state hand-off:

- llama_context::decode copies the batch's hidden states a second time into one of two pinned host buffers, by
  parity, and records a ggml backend event behind that copy on every device of the tensor-split backend. A stream
  synchronise would also wait for the next batch; the event only waits for this one.
- 1348's hook, called after batch k is submitted, collects batch k-1 through that event, runs its catch-up while k
  computes, and leaves k outstanding. Every other entry (flush at the prompt boundary, the non-deferred path, reset)
  collects or drops the outstanding batch first, through flush_deferred.

Default off: BIGCHERRY_PREFILL_PIPELINE=1. No new host thread; the order of work on each card's stream is unchanged.
"""

from __future__ import annotations

import re

from bigcherry.patcher import Edit, EnvDoc, FilePatch

GROUP = "rdna-boosts"
STATE = "validated"

# ---------------------------------------------------------------------------------------------- llama-context.h
_A_H_METHOD = "    float * get_embeddings_nextn();\n"
_N_H_METHOD = r"""    float * get_embeddings_nextn();

    // BigCherry 1359 (QFP50): the hidden states of the latest prompt batch, readable without waiting for a later
    // batch. bc_nextn_fence_slot() names the buffer the latest decode call wrote (-1 if it wrote none);
    // bc_nextn_fenced(slot) waits for that batch only and returns its rows.
    int32_t bc_nextn_fence_slot() const { return bc_nextn_slot_valid ? bc_nextn_cur : -1; }
    float * bc_nextn_fenced(int32_t slot);
    void    bc_nextn_fence_wait(int slot);
"""

_A_H_FIELD = "    buffer_view<float> embd_nextn = {nullptr, 0};\n"
_N_H_FIELD = r"""    buffer_view<float> embd_nextn = {nullptr, 0};

    // BigCherry 1359 (QFP50): two pinned host copies of the nextn hidden states, by decode-call parity, each with
    // one backend event a device recorded behind its copy
    ggml_backend_buffer_ptr           bc_nextn_buf[2];
    std::vector<ggml_backend_event_t> bc_nextn_events[2];
    int  bc_nextn_cur        = 0;
    bool bc_nextn_slot_valid = false;
"""

# -------------------------------------------------------------------------------------------- llama-context.cpp
_A_C_INCLUDE = '#include "llama-sampler.h"\n#include "llama.h"\n'
_N_C_INCLUDE = r"""#include "llama-sampler.h"
#include "llama.h"

// BigCherry 1359 (QFP50): the devices behind the tensor-split backend (ggml-backend-impl.h, exported by ggml-base)
extern "C" {
    GGML_API bool           ggml_backend_is_meta(ggml_backend_t backend);
    GGML_API size_t         ggml_backend_meta_n_backends(ggml_backend_t meta_backend);
    GGML_API ggml_backend_t ggml_backend_meta_simple_backend(ggml_backend_t meta_backend, size_t index);
}
"""

_A_C_DTOR = ("llama_context::~llama_context() {\n"
             "    // wait for any pending asynchronous copies into the output buffers before they are freed\n"
             "    synchronize();\n")
_N_C_DTOR = r"""llama_context::~llama_context() {
    // wait for any pending asynchronous copies into the output buffers before they are freed
    synchronize();

    for (auto & bc_events : bc_nextn_events) {  // BigCherry 1359
        for (ggml_backend_event_t bc_event : bc_events) {
            ggml_backend_event_free(bc_event);
        }
        bc_events.clear();
    }
"""

_A_C_EXTRACT = ("                ggml_backend_tensor_get_async(backend_h, t_h_nextn, embd_nextn_out, 0, n_rows*n_embd*sizeof(float));\n"
                "                extract_all_idxs = extract_all_idxs || !masked;\n")
_N_C_EXTRACT = r"""                ggml_backend_tensor_get_async(backend_h, t_h_nextn, embd_nextn_out, 0, n_rows*n_embd*sizeof(float));
                extract_all_idxs = extract_all_idxs || !masked;

                // BigCherry 1359 (QFP50): a second copy of an unmasked batch's rows into the other of two pinned
                // buffers, with an event behind it on every device, so the drafter hook can take this batch's
                // hidden states after the NEXT batch has been submitted. The copy above is unchanged.
                {
                    static const bool bc_1359_on = [] {
                        const char * s = getenv("BIGCHERRY_PREFILL_PIPELINE");
                        return s != nullptr && atoi(s) != 0;
                    }();
                    bc_nextn_slot_valid = false;
                    if (bc_1359_on && !masked && offset == 0) {
                        const int bc_slot = bc_nextn_cur ^ 1;
                        // the buffer about to be reused: its last copy must have landed (this also keeps the
                        // pipeline two batches deep and no deeper)
                        bc_nextn_fence_wait(bc_slot);

                        std::vector<ggml_backend_t> bc_backends;
                        if (ggml_backend_is_meta(backend_h)) {
                            for (size_t j = 0; j < ggml_backend_meta_n_backends(backend_h); ++j) {
                                bc_backends.push_back(ggml_backend_meta_simple_backend(backend_h, j));
                            }
                        } else {
                            bc_backends.push_back(backend_h);
                        }

                        auto & bc_events = bc_nextn_events[bc_slot];
                        bool bc_ok = true;
                        if (bc_events.size() != bc_backends.size()) {
                            for (ggml_backend_event_t bc_event : bc_events) {
                                ggml_backend_event_free(bc_event);
                            }
                            bc_events.clear();
                            for (ggml_backend_t bc_backend : bc_backends) {
                                ggml_backend_event_t bc_event = ggml_backend_event_new(ggml_backend_get_device(bc_backend));
                                if (bc_event == nullptr) {  // a backend without events: no pipeline, the hook waits as before
                                    bc_ok = false;
                                    break;
                                }
                                bc_events.push_back(bc_event);
                            }
                            if (!bc_ok) {
                                for (ggml_backend_event_t bc_event : bc_events) {
                                    ggml_backend_event_free(bc_event);
                                }
                                bc_events.clear();
                            }
                        }

                        const size_t bc_bytes = (size_t) n_rows * n_embd * sizeof(float);
                        if (bc_ok && (!bc_nextn_buf[bc_slot] || ggml_backend_buffer_get_size(bc_nextn_buf[bc_slot].get()) < bc_bytes)) {
                            auto * bc_buft = ggml_backend_cpu_buffer_type();
                            auto * bc_dev  = model.dev_output();
                            auto * bc_host = bc_dev ? ggml_backend_dev_host_buffer_type(bc_dev) : nullptr;
                            if (bc_host) {
                                bc_buft = bc_host;  // pinned, as the output buffer is: a pageable target would stall the copy
                            }
                            const size_t bc_cap = std::max(bc_bytes, (size_t) n_embd * cparams.n_batch * sizeof(float));
                            bc_nextn_buf[bc_slot].reset(ggml_backend_buft_alloc_buffer(bc_buft, bc_cap));
                            bc_ok = bc_nextn_buf[bc_slot] != nullptr;
                        }

                        if (bc_ok) {
                            float * bc_dst = (float *) ggml_backend_buffer_get_base(bc_nextn_buf[bc_slot].get());
                            ggml_backend_tensor_get_async(backend_h, t_h_nextn, bc_dst, 0, bc_bytes);
                            for (size_t j = 0; j < bc_events.size(); ++j) {
                                ggml_backend_event_record(bc_events[j], bc_backends[j]);
                            }
                            bc_nextn_cur        = bc_slot;
                            bc_nextn_slot_valid = true;
                        }
                    }
                }
"""

_A_C_API = "float * llama_get_embeddings_nextn(llama_context * ctx) {\n"
_N_C_API = r"""// BigCherry 1359 (QFP50)
void llama_context::bc_nextn_fence_wait(int slot) {
    for (ggml_backend_event_t bc_event : bc_nextn_events[slot]) {
        ggml_backend_event_synchronize(bc_event);
    }
}

float * llama_context::bc_nextn_fenced(int32_t slot) {
    if (slot < 0 || slot > 1 || !bc_nextn_buf[slot]) {
        return nullptr;
    }
    bc_nextn_fence_wait(slot);
    return (float *) ggml_backend_buffer_get_base(bc_nextn_buf[slot].get());
}

int32_t llama_nextn_fence_slot(llama_context * ctx) {
    return ctx->bc_nextn_fence_slot();
}

float * llama_get_embeddings_nextn_fenced(llama_context * ctx, int32_t slot) {
    return ctx->bc_nextn_fenced(slot);
}

"""

# -------------------------------------------------------------------------------------------------- llama-ext.h
_A_EXT = "LLAMA_API float * llama_get_embeddings_nextn(struct llama_context * ctx);\n"
_N_EXT = r"""LLAMA_API float * llama_get_embeddings_nextn(struct llama_context * ctx);

// BigCherry 1359 (QFP50): llama_nextn_fence_slot names the buffer the latest llama_decode call copied its unmasked
// nextn rows into (-1: none, or BIGCHERRY_PREFILL_PIPELINE is off). llama_get_embeddings_nextn_fenced waits for
// THAT call's copy only - not for anything submitted after it - and returns the rows. A slot stays readable until
// the second decode call after the one that wrote it.
LLAMA_API int32_t llama_nextn_fence_slot(struct llama_context * ctx);
LLAMA_API float * llama_get_embeddings_nextn_fenced(struct llama_context * ctx, int32_t slot);
"""

# ------------------------------------------------------------------------------------- common/speculative.cpp
_A_S_FIELD = "    std::vector<bool> bc_poisoned;\n"
_N_S_FIELD = r"""    std::vector<bool> bc_poisoned;

    // BigCherry 1359 (QFP50): the prompt batch that is on the cards and has not been snapshotted yet
    bool bc_pipeline = false;
    bool bc_pipeline_logged = false;
    int32_t bc_out_slot = -1;
    std::vector<common_batch::token> bc_out_tokens;
"""

_A_S_INIT = "        bc_deferred_enabled = !is_mem_shared && (bc_defer == nullptr || std::atoi(bc_defer) != 0);\n"
_N_S_INIT = r"""        bc_deferred_enabled = !is_mem_shared && (bc_defer == nullptr || std::atoi(bc_defer) != 0);
        {
            const char * bc_pipe = std::getenv("BIGCHERRY_PREFILL_PIPELINE");
            bc_pipeline = bc_deferred_enabled && bc_pipe != nullptr && std::atoi(bc_pipe) != 0;  // BigCherry 1359
        }
"""

_A_S_FLUSH = ("    bool flush_deferred() override {\n"
              "        if (!bc_deferred_enabled || bc_pending < 0) {\n"
              "            return true;\n"
              "        }\n")
_N_S_FLUSH = r"""    // BigCherry 1359 (QFP50): take the outstanding batch's hidden states - waiting for that batch only - snapshot
    // them and run its catch-up. Called through flush_deferred, so every path that flushes collects first.
    bool bc_collect_outstanding() {
        if (bc_out_slot < 0) {
            return true;
        }
        const int32_t slot = bc_out_slot;
        bc_out_slot = -1;
        if (!flush_deferred()) {  // a snapshot taken earlier goes first; there is none in steady state
            return false;
        }
        const float * h_tgt = llama_get_embeddings_nextn_fenced(this->params.ctx_tgt, slot);
        if (h_tgt == nullptr) {
            return false;
        }
        auto & dst = bc_chunks[bc_write];
        dst.tokens = std::move(bc_out_tokens);
        bc_out_tokens.clear();
        dst.h_nextn.resize(dst.tokens.size() * (size_t) n_embd);
        std::memcpy(dst.h_nextn.data(), h_tgt, dst.h_nextn.size() * sizeof(float));
        dst.valid = true;
        bc_pending = bc_write;
        bc_write ^= 1;
        return flush_deferred();  // its catch-up now, while the next batch computes
    }

    bool flush_deferred() override {
        if (bc_out_slot >= 0 && !bc_collect_outstanding()) {  // BigCherry 1359: the batch still on the cards first
            return false;
        }
        if (!bc_deferred_enabled || bc_pending < 0) {
            return true;
        }
"""

_A_S_RESET = "        bc_poisoned[seq_id] = poison;\n"
_N_S_RESET = r"""        bc_poisoned[seq_id] = poison;

        if (bc_out_slot >= 0) {  // BigCherry 1359: the outstanding batch is dropped by the rule of a pending snapshot
            bool bc_touches = false;
            for (const auto & tok : bc_out_tokens) {
                if (tok.seq_id == seq_id) {
                    bc_touches = true;
                    break;
                }
            }
            if (bc_touches) {
                for (const auto & tok : bc_out_tokens) {
                    if (tok.seq_id >= 0 && tok.seq_id < (llama_seq_id) n_seq && tok.seq_id != seq_id) {
                        bc_poisoned[tok.seq_id] = true;
                    }
                }
                bc_out_slot = -1;
                bc_out_tokens.clear();
            }
        }
"""

_A_S_HOOK = ("        // The server submits target chunk k+1 before entering here. "
             "Catch-up k therefore runs while k+1 is on GPU.\n")
_N_S_HOOK = r"""        if (bc_pipeline) {
            // BigCherry 1359 (QFP50): this batch (k) is already on the cards. Collect batch k-1 - waiting for ITS
            // hidden states only - run its catch-up while k computes, and leave k outstanding for the next call,
            // so the server goes straight on to build and submit k+1.
            if (!flush_deferred()) {
                return false;
            }
            const int32_t bc_slot = llama_nextn_fence_slot(this->params.ctx_tgt);
            if (bc_slot >= 0) {
                bc_out_slot = bc_slot;
                bc_out_tokens = batch_in.tokens;
                if (!bc_pipeline_logged) {
                    bc_pipeline_logged = true;
                    std::fprintf(stderr, "BIGCHERRY_PATCH_HIT patch=1359_prefill_pipeline\n");
                }
                return true;
            }
            // no fence for this batch (masked rows, or a backend without events): the synchronous snapshot below
        }

"""

PATCHES = [
    FilePatch(
        path="src/llama-context.h",
        description="1359: two fenced host copies of the nextn hidden states",
        language="none",
        edits=(
            Edit(id="pipeline-nextn-methods", anchor=re.escape(_A_H_METHOD), mode="replace", text=_N_H_METHOD,
                 guard=r"int32_t bc_nextn_fence_slot\(\) const", rationale="Beside the public nextn getter.",
                 expect_matches=1, max_span_lines=2),
            Edit(id="pipeline-nextn-fields", anchor=re.escape(_A_H_FIELD), mode="replace", text=_N_H_FIELD,
                 guard=r"ggml_backend_buffer_ptr +bc_nextn_buf\[2\];", rationale="Beside the nextn output view.",
                 expect_matches=1, max_span_lines=2),
        ),
    ),
    FilePatch(
        path="src/llama-context.cpp",
        description="1359: second copy of an unmasked batch's nextn rows with a backend event behind it",
        language="none",
        edits=(
            Edit(id="pipeline-meta-decls", anchor=re.escape(_A_C_INCLUDE), mode="replace", text=_N_C_INCLUDE,
                 guard=r"BigCherry 1359 \(QFP50\): the devices behind the tensor-split backend",
                 rationale="The two adjacent includes that end llama-context.cpp's project includes.",
                 expect_matches=1, max_span_lines=3),
            Edit(id="pipeline-event-free", anchor=re.escape(_A_C_DTOR), mode="replace", text=_N_C_DTOR,
                 guard=r"for \(auto & bc_events : bc_nextn_events\) \{  // BigCherry 1359",
                 rationale="The context destructor, after its synchronise.", expect_matches=1, max_span_lines=4),
            Edit(id="pipeline-nextn-copy", anchor=re.escape(_A_C_EXTRACT), mode="replace", text=_N_C_EXTRACT,
                 guard=r"static const bool bc_1359_on = \[\]",
                 rationale="The nextn extraction in decode (the encode one reads embd_nextn.data directly).",
                 expect_matches=1, max_span_lines=3),
            Edit(id="pipeline-nextn-api", anchor=re.escape(_A_C_API), mode="insert_before", text=_N_C_API,
                 guard=r"float \* llama_get_embeddings_nextn_fenced\(llama_context \* ctx, int32_t slot\) \{",
                 rationale="Before the synchronising getter it sits beside.", expect_matches=1, max_span_lines=2),
        ),
    ),
    FilePatch(
        path="src/llama-ext.h",
        description="1359: fenced nextn getter",
        language="none",
        edits=(
            Edit(id="pipeline-ext-api", anchor=re.escape(_A_EXT), mode="replace", text=_N_EXT,
                 guard=r"LLAMA_API int32_t llama_nextn_fence_slot\(struct llama_context \* ctx\);",
                 rationale="Beside the getter it complements.", expect_matches=1, max_span_lines=2),
        ),
    ),
    FilePatch(
        path="common/speculative.cpp",
        description="1359: the MTP deferred hook leaves the newest prompt batch outstanding (after 1348 and 1346)",
        language="none",
        edits=(
            Edit(id="pipeline-mtp-fields", anchor=re.escape(_A_S_FIELD), mode="replace", text=_N_S_FIELD,
                 guard=r"int32_t bc_out_slot = -1;", rationale="1348's last deferred field.",
                 expect_matches=1, max_span_lines=2),
            Edit(id="pipeline-mtp-init", anchor=re.escape(_A_S_INIT), mode="replace", text=_N_S_INIT,
                 guard=r'std::getenv\("BIGCHERRY_PREFILL_PIPELINE"\)', rationale="Where 1348 reads its own switch.",
                 expect_matches=1, max_span_lines=2),
            Edit(id="pipeline-mtp-collect", anchor=re.escape(_A_S_FLUSH), mode="replace", text=_N_S_FLUSH,
                 guard=r"bool bc_collect_outstanding\(\) \{",
                 rationale="The head of 1348's flush_deferred, which 1346 leaves as it is.",
                 expect_matches=1, max_span_lines=5),
            Edit(id="pipeline-mtp-reset", anchor=re.escape(_A_S_RESET), mode="replace", text=_N_S_RESET,
                 guard=r"BigCherry 1359: the outstanding batch is dropped by the rule of a pending snapshot",
                 rationale="1348's reset_deferred, after it records the poison.", expect_matches=1, max_span_lines=2),
            Edit(id="pipeline-mtp-hook", anchor=re.escape(_A_S_HOOK), mode="insert_before", text=_N_S_HOOK,
                 guard=r"BIGCHERRY_PATCH_HIT patch=1359_prefill_pipeline",
                 rationale="In front of 1348's own catch-up-then-wait sequence in process_deferred.",
                 expect_matches=1, max_span_lines=2),
        ),
    ),
]

ENV_DOCS = (
    EnvDoc(
        "BIGCHERRY_PREFILL_PIPELINE",
        "0|1",
        "0 (off)",
        "experimental: with an MTP drafter, submit prompt batch k+1 before collecting batch k's hidden states "
        "(waits on a backend event for batch k only); needs BIGCHERRY_MTP_DEFERRED_CATCHUP on",
    ),
)
