"""1348: defer MTP prompt catch-up until after the next target chunk is submitted (QFP31).

During prompt processing, draft-MTP process() synchronizes the target to read h_nextn and then immediately
re-evaluates the same prompt chunk on the draft. That serializes ~83 ms of draft work between target chunks.
This patch snapshots the synchronized target h_nextn rows into one of two MTP-owned host buffers, returns without
running the draft catch-up, and on the next prompt-only call runs the previous catch-up first. The server has already
submitted the next target chunk before common_speculative_process(), so that draft work overlaps target GPU work.

The final pending catch-up is flushed before common_speculative_begin()/sampling. Non-prompt or embedding batches
flush and use the native path. Slot reset/cache clear/context shift and failed target decode invalidate pending work;
unsafe sequences are poisoned for the remainder of that request, which disables drafting rather than using stale
draft state. BIGCHERRY_MTP_DEFERRED_CATCHUP=0 restores the native synchronous process() path.
"""

from __future__ import annotations

import re

from bigcherry.patcher import Edit, EnvDoc, FilePatch

GROUP = "rdna-boosts"
STATE = "validated"

_A_SPEC_H = """\
bool common_speculative_process(common_speculative * spec, const common_batch & batch);

// generate drafts for the sequences specified with `common_speculative_get_draft_params`
"""
_N_SPEC_H = r"""bool common_speculative_process(common_speculative * spec, const common_batch & batch);

// BigCherry 1348 (QFP31): prompt-only MTP may snapshot this target batch and defer its draft catch-up until
// the next target batch has already been submitted. Other speculative implementations use process() unchanged.
bool common_speculative_process_deferred(common_speculative * spec, const common_batch & batch, bool allow_defer);

// Run the last deferred MTP catch-up before sampling/state transitions.
bool common_speculative_flush_deferred(common_speculative * spec);

// Invalidate a deferred chunk for seq_id. poison=true also disables MTP drafting for the rest of this request.
void common_speculative_reset_deferred(common_speculative * spec, llama_seq_id seq_id, bool poison);

// generate drafts for the sequences specified with common_speculative_get_draft_params
"""

_A_INCLUDES = """\
#include <map>
#include <cinttypes>
"""
_N_INCLUDES = r"""#include <map>
#include <cinttypes>
#include <cstdio>
#include <cstdlib>
"""

_A_BASE_VIRTUAL = """\
    virtual bool process(const common_batch & batch) = 0;

    virtual void draft(common_speculative_draft_params_vec & dparams) = 0;
"""
_N_BASE_VIRTUAL = r"""    virtual bool process(const common_batch & batch) = 0;

    // BigCherry 1348: opt-in seam for same-thread prompt catch-up deferral.
    virtual bool process_deferred(const common_batch & batch, bool /*allow_defer*/) { return process(batch); }
    virtual bool flush_deferred() { return true; }
    virtual void reset_deferred(llama_seq_id /*seq_id*/, bool /*poison*/) {}

    virtual void draft(common_speculative_draft_params_vec & dparams) = 0;
"""

_A_MTP_FIELDS = """\
    std::vector<std::vector<float>> verify_h;
    std::vector<int32_t> verify_h_rows;

    std::vector<int>                i_last;
"""
_N_MTP_FIELDS = r"""    std::vector<std::vector<float>> verify_h;
    std::vector<int32_t> verify_h_rows;

    // BigCherry 1348 (QFP31): two host snapshots let target chunk k+1 be submitted before draft catch-up k.
    struct bc_deferred_chunk {
        bool valid = false;
        std::vector<common_batch::token> tokens;
        std::vector<float> h_nextn;
    };
    std::array<bc_deferred_chunk, 2> bc_chunks;
    int bc_pending = -1;
    int bc_write = 0;
    bool bc_deferred_enabled = false;
    bool bc_hit_logged = false;
    std::vector<bool> bc_poisoned;

    std::vector<int>                i_last;
"""

_A_MTP_CTOR_END = """\
        verify_h.assign(n_seq, {});
        verify_h_rows.assign(n_seq, 0);
    }

    ~common_speculative_impl_draft_mtp() override {
"""
_N_MTP_CTOR_END = r"""        verify_h.assign(n_seq, {});
        verify_h_rows.assign(n_seq, 0);

        // Default-on qualification switch. Shared-KV MTP has no catch-up decode to overlap.
        const char * bc_defer = std::getenv("BIGCHERRY_MTP_DEFERRED_CATCHUP");
        bc_deferred_enabled = !is_mem_shared && (bc_defer == nullptr || std::atoi(bc_defer) != 0);
        bc_poisoned.assign(n_seq, false);
    }

    ~common_speculative_impl_draft_mtp() override {
"""

_A_MTP_PROCESS = """\
        if (pos_max < N - 1 && !is_mem_shared) {
            SPC_WRN("ctx_dft pos_max=%d < N-1=%d - "
                    "process() hook may not have run on every prefill ubatch "
                    "(need_embd / output flag on every prompt position?). "
                    "Drafts may degrade.\\n",
                    (int) pos_max, N - 1);
        }
    }

    bool process(const common_batch & batch_in) override {
"""
_N_MTP_PROCESS = r"""        if (pos_max < N - 1 && !is_mem_shared) {
            SPC_WRN("ctx_dft pos_max=%d < N-1=%d - "
                    "process() hook may not have run on every prefill ubatch "
                    "(need_embd / output flag on every prompt position?). "
                    "Drafts may degrade.\n",
                    (int) pos_max, N - 1);
        }
    }

    bool bc_process_snapshot(const bc_deferred_chunk & chunk) {
        if (!chunk.valid || chunk.tokens.empty()) {
            return true;
        }

        const int32_t n_tokens = (int32_t) chunk.tokens.size();
        GGML_ASSERT(chunk.h_nextn.size() == (size_t) n_tokens * n_embd);

        std::fill(i_batch_beg.begin(), i_batch_beg.end(), -1);
        std::fill(i_batch_end.begin(), i_batch_end.end(), -1);

        for (int k = 0; k < n_tokens; ++k) {
            const llama_seq_id seq_id = chunk.tokens[k].seq_id;
            if (seq_id < 0 || seq_id >= (llama_seq_id) n_seq || bc_poisoned[seq_id]) {
                continue;
            }
            i_batch_end[seq_id] = k;
            if (i_batch_beg[seq_id] < 0) {
                i_batch_beg[seq_id] = k;
            }
        }

        auto * ctx_dft = this->params.ctx_dft;
        const float * h_tgt = chunk.h_nextn.data();
        const size_t row_bytes = (size_t) n_embd * sizeof(float);

        batch.clear();

        for (int k = 0; k < n_tokens; ++k) {
            const auto & tok = chunk.tokens[k];
            const llama_seq_id seq_id = tok.seq_id;
            if (seq_id < 0 || seq_id >= (llama_seq_id) n_seq || bc_poisoned[seq_id]) {
                continue;
            }

            const int32_t idx = batch.add(tok.id, tok.pos[0], seq_id, false);
            for (llama_seq_id seq_extra : tok.seq_ids_extra) {
                batch.add_seq(idx, seq_extra);
            }

            const float * h_row = k == i_batch_beg[seq_id]
                ? pending_h[seq_id].data()
                : h_tgt + (size_t) (k - 1) * n_embd;

            batch.set_embd(idx, { h_row, 1, (size_t) n_embd });
        }

        auto * mem_dft = llama_get_memory(ctx_dft);

        bool ok = true;
        if (batch.size() > 0) {
            for (int head = 0; head < n_mtp_layers; ++head) {
                if (chain_heads) {
                    for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
                        if (i_batch_beg[seq_id] < 0 || bc_poisoned[seq_id]) {
                            continue;
                        }
                        llama_memory_seq_rm(mem_dft, seq_id, chunk.tokens[i_batch_beg[seq_id]].pos[0], -1);
                    }
                    llama_set_nextn_layer_offset(ctx_dft, head);
                }

                const int32_t rc = llama_process(ctx_dft, LLAMA_PROCESS_TYPE_DECODE, batch.get());
                if (rc != 0) {
                    SPC_ERR("llama_process(ctx_dft) deferred head=%d failed rc=%d (pos=%d)\n",
                            head, (int) rc, (int) chunk.tokens[0].pos[0]);
                    ok = false;
                    break;
                }
            }
        }

        if (chain_heads) {
            llama_set_nextn_layer_offset(ctx_dft, 0);
        }
        if (!ok) {
            return false;
        }

        for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
            if (i_batch_end[seq_id] < 0 || bc_poisoned[seq_id]) {
                continue;
            }

            const int32_t n_rows = i_batch_end[seq_id] - i_batch_beg[seq_id] + 1;
            verify_h_rows[seq_id] = n_rows;
            verify_h[seq_id].resize((size_t) n_rows * n_embd);

            for (int32_t i = 0; i < n_rows; ++i) {
                const int32_t k = i_batch_beg[seq_id] + i;
                std::memcpy(verify_h[seq_id].data() + (size_t) i * n_embd,
                        h_tgt + (size_t) k * n_embd, row_bytes);
            }

            std::memcpy(pending_h[seq_id].data(),
                    verify_h[seq_id].data() + (size_t) (n_rows - 1) * n_embd, row_bytes);
        }

        return true;
    }

    bool flush_deferred() override {
        if (!bc_deferred_enabled || bc_pending < 0) {
            return true;
        }

        const int pending = bc_pending;
        bc_pending = -1;
        const bool ok = bc_process_snapshot(bc_chunks[pending]);
        bc_chunks[pending].valid = false;
        return ok;
    }

    void reset_deferred(llama_seq_id seq_id, bool poison) override {
        if (seq_id < 0 || seq_id >= (llama_seq_id) n_seq) {
            return;
        }

        bc_poisoned[seq_id] = poison;

        if (bc_pending < 0) {
            return;
        }

        bool touches = false;
        for (const auto & tok : bc_chunks[bc_pending].tokens) {
            if (tok.seq_id == seq_id) {
                touches = true;
                break;
            }
        }
        if (!touches) {
            return;
        }

        // A snapshot can contain several sequences. Dropping it for one invalidated sequence means the others
        // cannot safely draft from this batch either; keep target generation correct by disabling their MTP.
        for (const auto & tok : bc_chunks[bc_pending].tokens) {
            if (tok.seq_id >= 0 && tok.seq_id < (llama_seq_id) n_seq && tok.seq_id != seq_id) {
                bc_poisoned[tok.seq_id] = true;
            }
        }
        bc_chunks[bc_pending].valid = false;
        bc_pending = -1;
    }

    bool process_deferred(const common_batch & batch_in, bool allow_defer) override {
        if (!bc_deferred_enabled || !allow_defer || batch_in.size() <= 0 ||
                !batch_in.has_token() || batch_in.has_embd()) {
            if (!flush_deferred()) {
                return false;
            }
            return process(batch_in);
        }

        for (const auto & tok : batch_in.tokens) {
            if (tok.seq_id < 0 || tok.seq_id >= (llama_seq_id) n_seq || bc_poisoned[tok.seq_id]) {
                // A failed/cancelled/shifted sequence is target-only for the rest of this request.
                return true;
            }
        }

        // The server submits target chunk k+1 before entering here. Catch-up k therefore runs while k+1 is on GPU.
        if (!flush_deferred()) {
            return false;
        }

        auto * ctx_tgt = this->params.ctx_tgt;
        const float * h_tgt = llama_get_embeddings_nextn(ctx_tgt); // sync current target only after prior catch-up ran
        if (h_tgt == nullptr) {
            return false;
        }

        auto & dst = bc_chunks[bc_write];
        dst.tokens = batch_in.tokens;
        dst.h_nextn.resize((size_t) batch_in.size() * n_embd);
        std::memcpy(dst.h_nextn.data(), h_tgt, dst.h_nextn.size() * sizeof(float));
        dst.valid = true;

        bc_pending = bc_write;
        bc_write ^= 1;

        if (!bc_hit_logged) {
            bc_hit_logged = true;
            std::fprintf(stderr, "BIGCHERRY_PATCH_HIT patch=1348_mtp_deferred_catchup\n");
        }

        return true;
    }

    bool process(const common_batch & batch_in) override {
"""

_A_MTP_DRAFT_PREFIX = """\
            std::memcpy(pending_h[seq_id].data(),
                    verify_h[seq_id].data() + (size_t) (n_rows - 1) * n_embd, row_bytes);
        }

        return true;
    }

    void draft(common_speculative_draft_params_vec & dparams) override {
        auto & ctx_dft = params.ctx_dft;

        batch.clear();

        // keep track of which sequences are still drafting
        int n_drafting = 0;
        std::vector<bool> drafting(n_seq);

        for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
            auto & dp = dparams[seq_id];

            if (!dp.drafting) {
"""
_N_MTP_DRAFT_PREFIX = r"""            std::memcpy(pending_h[seq_id].data(),
                    verify_h[seq_id].data() + (size_t) (n_rows - 1) * n_embd, row_bytes);
        }

        return true;
    }

    void draft(common_speculative_draft_params_vec & dparams) override {
        auto & ctx_dft = params.ctx_dft;

        batch.clear();

        // keep track of which sequences are still drafting
        int n_drafting = 0;
        std::vector<bool> drafting(n_seq);

        for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
            auto & dp = dparams[seq_id];

            // BigCherry 1348: after a dropped deferred catch-up, never generate from stale draft state.
            if (bc_poisoned[seq_id]) {
                dp.drafting = false;
                continue;
            }

            if (!dp.drafting) {
"""

_A_PUBLIC_PROCESS = """\
bool common_speculative_process(common_speculative * spec, const common_batch & batch) {
    bool result = true;

    if (spec == nullptr) {
        return result;
    }

    for (auto & impl : spec->impls) {
        result = result && impl->process(batch);
    }

    return result;
}

void common_speculative_draft(common_speculative * spec) {
"""
_N_PUBLIC_PROCESS = r"""bool common_speculative_process(common_speculative * spec, const common_batch & batch) {
    bool result = true;

    if (spec == nullptr) {
        return result;
    }

    for (auto & impl : spec->impls) {
        result = result && impl->process(batch);
    }

    return result;
}

bool common_speculative_process_deferred(common_speculative * spec, const common_batch & batch, bool allow_defer) {
    bool result = true;

    if (spec == nullptr) {
        return result;
    }

    for (auto & impl : spec->impls) {
        result = result && impl->process_deferred(batch, allow_defer);
    }

    return result;
}

bool common_speculative_flush_deferred(common_speculative * spec) {
    bool result = true;

    if (spec == nullptr) {
        return result;
    }

    for (auto & impl : spec->impls) {
        result = result && impl->flush_deferred();
    }

    return result;
}

void common_speculative_reset_deferred(common_speculative * spec, llama_seq_id seq_id, bool poison) {
    if (spec == nullptr) {
        return;
    }

    for (auto & impl : spec->impls) {
        impl->reset_deferred(seq_id, poison);
    }
}

void common_speculative_draft(common_speculative * spec) {
"""

_A_PROMPT_LOAD_CLEAR = """\
    bool prompt_load(server_prompt_cache & prompt_cache, const server_tokens & tokens) {
        bool res = prompt_cache.load(prompt, tokens, ctx_tgt, ctx_dft, id);
        if (!res) {
            SLT_WRN(*this, "%s", "failed to load prompt from cache\\n");
        }

        return res;
    }

    void prompt_clear() {
        SLT_TRC(*this, "clearing prompt with %zu tokens\\n", prompt.tokens.size());

        mem.seq_rm(id, -1, -1);

        prompt.clear();
    }
"""
_N_PROMPT_LOAD_CLEAR = r"""    bool prompt_load(server_prompt_cache & prompt_cache, const server_tokens & tokens) {
        // BigCherry 1348: cached target/draft state supersedes any in-flight deferred prompt snapshot.
        if (spec) {
            common_speculative_reset_deferred(spec, id, false);
        }

        bool res = prompt_cache.load(prompt, tokens, ctx_tgt, ctx_dft, id);
        if (!res) {
            SLT_WRN(*this, "%s", "failed to load prompt from cache\n");
        }

        return res;
    }

    void prompt_clear() {
        SLT_TRC(*this, "clearing prompt with %zu tokens\n", prompt.tokens.size());

        if (spec) {
            common_speculative_reset_deferred(spec, id, false);
        }
        mem.seq_rm(id, -1, -1);

        prompt.clear();
    }
"""

_A_RELEASE = """\
    void release() {
        if (is_processing()) {
            GGML_ASSERT(task);

            SLT_INF(*this, "stop processing: n_tokens = %d, truncated = %d\\n", prompt.n_tokens(), truncated);
"""
_N_RELEASE = r"""    void release() {
        if (is_processing()) {
            GGML_ASSERT(task);

            // BigCherry 1348: cancellation/normal release must not leave a prompt catch-up for a reused slot.
            if (spec) {
                common_speculative_reset_deferred(spec, id, false);
            }

            SLT_INF(*this, "stop processing: n_tokens = %d, truncated = %d\n", prompt.n_tokens(), truncated);
"""

_A_CONTEXT_SHIFT = """\
                SLT_WRN(slot, "slot context shift, n_keep = %d, n_left = %d, n_discard = %d\\n", n_keep, n_left, n_discard);

                slot.mem.seq_rm (slot.id, n_keep            , n_keep + n_discard);
"""
_N_CONTEXT_SHIFT = r"""                SLT_WRN(slot, "slot context shift, n_keep = %d, n_left = %d, n_discard = %d\n", n_keep, n_left, n_discard);

                // BigCherry 1348: a pending snapshot was captured against the pre-shift position mapping.
                // Drop it and disable MTP for this request; target generation remains authoritative.
                if (spec) {
                    common_speculative_reset_deferred(spec.get(), slot.id, true);
                }

                slot.mem.seq_rm (slot.id, n_keep            , n_keep + n_discard);
"""

_A_HAS_OUTPUT = """\
        bool has_output = false;
        for (int i = off; i < off + batch.view.size(); ++i) {
            has_output |= batch.tokens[i].output;
        }

        // yield to the queue, so we can still handle metrics tasks while decoding
"""
_N_HAS_OUTPUT = r"""        bool has_output = false;
        bool bc_prompt_only = spec != nullptr && !batch.has_embd();
        for (int i = off; i < off + batch.view.size(); ++i) {
            has_output |= batch.tokens[i].output;
            bc_prompt_only = bc_prompt_only && batch.tokens[i].is_prompt && batch.tokens[i].i_embd < 0;
        }

        // yield to the queue, so we can still handle metrics tasks while decoding
"""

_A_AHEAD_GATE = """\
            if (ret == 0 && spec && ctx_dft && bc_mtp_ahead_on() && ctx_dft_seq_rm_type == COMMON_CONTEXT_SEQ_RM_TYPE_PART) {
"""
_N_AHEAD_GATE = r"""            if (ret == 0 && !bc_prompt_only && spec && ctx_dft && bc_mtp_ahead_on() && ctx_dft_seq_rm_type == COMMON_CONTEXT_SEQ_RM_TYPE_PART) {
                // BigCherry 1348: prompt catch-up and generation look-ahead are separate phases. Drain any final
                // deferred prompt chunk before 1322 mutates the draft context. DONE_PROMPT normally drained it already.
                bool bc_boundary_ok = common_speculative_flush_deferred(spec.get());
                if (!bc_boundary_ok) {
                    throw std::runtime_error("failed to flush deferred MTP prompt catch-up before ahead draft");
                }
"""

_A_TARGET_SUBMIT = """\
            if (ret == 0 && has_output) {
                llama_synchronize(ctx_tgt);
            }
"""
_N_TARGET_SUBMIT = r"""            // BigCherry 1348: prompt-only deferred catch-up must observe target submission before the
            // target sync; process_deferred() synchronizes when it snapshots the current NextN rows.
            if (ret == 0 && has_output && !bc_prompt_only) {
                llama_synchronize(ctx_tgt);
            }
"""

_A_TARGET_FAIL = """\
        if (ret != 0) {
            {
                std::string err;
"""
_N_TARGET_FAIL = r"""        if (ret != 0) {
            // BigCherry 1348: the target chunk did not commit; pending catch-up must not be applied across retry/error.
            if (spec) {
                for (const auto & tok : batch.view.tokens) {
                    common_speculative_reset_deferred(spec.get(), tok.seq_id, true);
                }
            }

            {
                std::string err;
"""

_A_SPEC_PROCESS = """\
        if (spec) {
            bool ok = true;
            queue_tasks.yield_to_queue([&]() {
                const int64_t bc_t0 = bc_spec_timing_on() ? ggml_time_us() : 0;  // bigcherry 1317
                ok = common_speculative_process(spec.get(), batch.view);
                if (bc_spec_timing_on()) {
                    bc_spec_t().process_us += ggml_time_us() - bc_t0;
                }
            });

            if (!ok) {
                SRV_ERR("%s", "failed to process speculative batch\\n");

                // TODO: handle error
                throw std::runtime_error("failed to process speculative batch");
            }
        }

        // handle `n_cmpl > 1` tasks - when the main prompt is processed, activate all child tasks too
"""
_N_SPEC_PROCESS = r"""        if (spec) {
            bool ok = true;
            queue_tasks.yield_to_queue([&]() {
                const int64_t bc_t0 = bc_spec_timing_on() ? ggml_time_us() : 0;  // bigcherry 1317
                ok = common_speculative_process_deferred(spec.get(), batch.view, bc_prompt_only);
                if (bc_spec_timing_on()) {
                    bc_spec_t().process_us += ggml_time_us() - bc_t0;
                }
            });

            if (!ok) {
                SRV_ERR("%s", "failed to process speculative batch\n");

                // TODO: handle error
                throw std::runtime_error("failed to process speculative batch");
            }
        }

        // The standard has_output contract is preserved. Eligible deferred MTP has already synchronized while
        // snapshotting NextN, so this is normally a no-op; it also protects non-MTP speculative combinations.
        if (has_output && bc_prompt_only) {
            queue_tasks.yield_to_queue([&]() {
                llama_synchronize(ctx_tgt);
            });
        }

        // handle n_cmpl > 1 tasks - when the main prompt is processed, activate all child tasks too
"""

_A_FINAL_FLUSH = """\
                if (slot.can_speculate()) {
                    common_speculative_begin(spec.get(), slot.id, slot.prompt.tokens.get_text_tokens());
                }
"""
_N_FINAL_FLUSH = r"""                if (slot.can_speculate()) {
                    // BigCherry 1348: the final prompt chunk has no k+1 to hide behind. Catch it up now, before
                    // begin()/sampling observes the draft state.
                    bool bc_ok = true;
                    queue_tasks.yield_to_queue([&]() {
                        bc_ok = common_speculative_flush_deferred(spec.get());
                    });
                    if (!bc_ok) {
                        throw std::runtime_error("failed to flush deferred MTP prompt catch-up");
                    }
                    common_speculative_begin(spec.get(), slot.id, slot.prompt.tokens.get_text_tokens());
                }
"""

PATCHES = [
    FilePatch(
        path="common/speculative.h",
        description="1348: expose deferred prompt catch-up lifecycle hooks",
        language="none",
        edits=(
            Edit(
                id="mtp-deferred-api",
                anchor=re.escape(_A_SPEC_H),
                mode="replace",
                text=_N_SPEC_H,
                guard=r"common_speculative_process_deferred\(",
                rationale="Immediately after the existing process API, before draft generation.",
                expect_matches=1,
                max_span_lines=4,
            ),
        ),
    ),
    FilePatch(
        path="common/speculative.cpp",
        description="1348: double-buffer target NextN snapshots and defer MTP draft catch-up by one prompt chunk",
        language="none",
        edits=(
            Edit(
                id="mtp-deferred-includes",
                anchor=re.escape(_A_INCLUDES),
                mode="replace",
                text=_N_INCLUDES,
                guard=r"#include <cstdio>\n#include <cstdlib>",
                rationale="Standard includes used by the env switch and first-use marker.",
                expect_matches=1,
                max_span_lines=3,
            ),
            Edit(
                id="mtp-deferred-virtual-api",
                anchor=re.escape(_A_BASE_VIRTUAL),
                mode="replace",
                text=_N_BASE_VIRTUAL,
                guard=r"virtual bool process_deferred\(",
                rationale="Base speculative implementation API beside process(); defaults preserve every non-MTP implementation.",
                expect_matches=1,
                max_span_lines=4,
            ),
            Edit(
                id="mtp-deferred-state",
                anchor=re.escape(_A_MTP_FIELDS),
                mode="replace",
                text=_N_MTP_FIELDS,
                guard=r"struct bc_deferred_chunk",
                rationale="MTP-only state beside verify_h/pending_h.",
                expect_matches=1,
                max_span_lines=5,
            ),
            Edit(
                id="mtp-deferred-enable",
                anchor=re.escape(_A_MTP_CTOR_END),
                mode="replace",
                text=_N_MTP_CTOR_END,
                guard=r"BIGCHERRY_MTP_DEFERRED_CATCHUP",
                rationale="MTP constructor after per-sequence storage is initialized.",
                expect_matches=1,
                max_span_lines=6,
            ),
            Edit(
                id="mtp-deferred-process",
                anchor=re.escape(_A_MTP_PROCESS),
                mode="replace",
                text=_N_MTP_PROCESS,
                guard=r"BIGCHERRY_PATCH_HIT patch=1348_mtp_deferred_catchup",
                rationale="Tail of MTP begin(), immediately before native process(); native process remains intact for the off switch.",
                expect_matches=1,
                max_span_lines=12,
            ),
            Edit(
                id="mtp-deferred-poison-draft",
                anchor=re.escape(_A_MTP_DRAFT_PREFIX),
                mode="replace",
                text=_N_MTP_DRAFT_PREFIX,
                guard=r"after a dropped deferred catch-up, never generate from stale draft state",
                rationale="Tail of native MTP process() through the draft() admission point.",
                expect_matches=1,
                max_span_lines=30,
            ),
            Edit(
                id="mtp-deferred-public-api",
                anchor=re.escape(_A_PUBLIC_PROCESS),
                mode="replace",
                text=_N_PUBLIC_PROCESS,
                guard=r"bool common_speculative_flush_deferred\(",
                rationale="Public process wrapper; deferred/flush/reset variants belong beside it.",
                expect_matches=1,
                max_span_lines=16,
            ),
        ),
    ),
    FilePatch(
        path="tools/server/server-context.cpp",
        description="1348: submit next target prompt chunk before prior MTP catch-up and flush/invalidate at lifecycle boundaries",
        language="none",
        edits=(
            Edit(
                id="mtp-deferred-prompt-cache-reset",
                anchor=re.escape(_A_PROMPT_LOAD_CLEAR),
                mode="replace",
                text=_N_PROMPT_LOAD_CLEAR,
                guard=r"cached target/draft state supersedes any in-flight deferred prompt snapshot",
                rationale="Slot prompt cache load/clear are seq-state replacement/removal boundaries.",
                expect_matches=1,
                max_span_lines=19,
            ),
            Edit(
                id="mtp-deferred-release-reset",
                anchor=re.escape(_A_RELEASE),
                mode="replace",
                text=_N_RELEASE,
                guard=r"cancellation/normal release must not leave a prompt catch-up",
                rationale="Slot release is the definitive cancellation/reuse boundary.",
                expect_matches=1,
                max_span_lines=7,
            ),
            Edit(
                id="mtp-deferred-context-shift",
                anchor=re.escape(_A_CONTEXT_SHIFT),
                mode="replace",
                text=_N_CONTEXT_SHIFT,
                guard=r"pending snapshot was captured against the pre-shift position mapping",
                rationale="Immediately before seq_rm/seq_add changes token positions.",
                expect_matches=1,
                max_span_lines=4,
            ),
            Edit(
                id="mtp-deferred-prompt-gate",
                anchor=re.escape(_A_HAS_OUTPUT),
                mode="replace",
                text=_N_HAS_OUTPUT,
                guard=r"bool bc_prompt_only = spec != nullptr",
                rationale="Decode already scans the rendered sub-batch for output; derive strict text-prompt eligibility there.",
                expect_matches=1,
                max_span_lines=7,
            ),
            Edit(
                id="mtp-deferred-ahead-boundary",
                anchor=re.escape(_A_AHEAD_GATE),
                mode="replace",
                text=_N_AHEAD_GATE,
                guard=r"failed to flush deferred MTP prompt catch-up before ahead draft",
                rationale="1322 generation look-ahead must never run until the final deferred prompt catch-up is drained.",
                expect_matches=1,
                max_span_lines=2,
            ),
            Edit(
                id="mtp-deferred-target-submit",
                anchor=re.escape(_A_TARGET_SUBMIT),
                mode="replace",
                text=_N_TARGET_SUBMIT,
                guard=r"prompt-only deferred catch-up must observe target submission",
                rationale="After 1317/1322 composition, only the sync seam changes: prompt batches defer it until the NextN snapshot.",
                expect_matches=1,
                max_span_lines=4,
            ),
            Edit(
                id="mtp-deferred-target-fail",
                anchor=re.escape(_A_TARGET_FAIL),
                mode="replace",
                text=_N_TARGET_FAIL,
                guard=r"target chunk did not commit; pending catch-up must not be applied",
                rationale="Before retry/fatal handling mutates slots or target context.",
                expect_matches=1,
                max_span_lines=4,
            ),
            Edit(
                id="mtp-deferred-server-process",
                anchor=re.escape(_A_SPEC_PROCESS),
                mode="replace",
                text=_N_SPEC_PROCESS,
                guard=r"common_speculative_process_deferred\(spec.get\(\), batch.view, bc_prompt_only\)",
                rationale="Preserve 1317 timing around the post-target hook while routing prompt-only batches through deferred catch-up.",
                expect_matches=1,
                max_span_lines=22,
            ),
            Edit(
                id="mtp-deferred-final-flush",
                anchor=re.escape(_A_FINAL_FLUSH),
                mode="replace",
                text=_N_FINAL_FLUSH,
                guard=r"failed to flush deferred MTP prompt catch-up",
                rationale="DONE_PROMPT transition immediately before speculative begin/sampling.",
                expect_matches=1,
                max_span_lines=4,
            ),
        ),
    ),
]

ENV_DOCS = (
    EnvDoc(
        "BIGCHERRY_MTP_DEFERRED_CATCHUP",
        "0|1",
        "1 (on)",
        "defer prompt-only MTP draft catch-up by one target chunk so catch-up k runs after target k+1 is submitted; "
        "0 restores native synchronous common_speculative_process behavior",
    ),
)
