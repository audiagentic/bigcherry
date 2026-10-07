"""1346 (QFP31 chunk 2b): bounded MTP prompt-window replay plus timing.

BIGCHERRY_MTP_PROMPT_TIMING=1 adds prompt/target lifecycle hooks and times the existing
MTP prompt catch-up without changing model work or ordering. At prompt end it prints one line:
BIGCHERRY_MTP_PROMPT_TIMING target_nextn_ms= target_sync_ms= target_fetch_ms= draft_process_ms= draft_decode_ms= host_gap_ms= chunks= tokens=

The diagnostic is intentionally single-sequence attribution: mixed-sequence server batches are
not accumulated, because their MTP process wall cannot be assigned to one prompt honestly.
"""

from __future__ import annotations

import re

from bigcherry.patcher import Edit, EnvDoc, FilePatch

GROUP = "rdna-boosts"
STATE = "untested"

_API_ANCHOR = "common_speculative_draft_params & common_speculative_get_draft_params(common_speculative * spec, llama_seq_id seq_id);\n"
_API_TEXT = """\

// bigcherry 1346 (QFP31): called after prompt cache/checkpoint resolution and before the first prompt batch.
void common_speculative_prefill_begin(
        common_speculative * spec, llama_seq_id seq_id,
        int32_t n_prompt, int32_t n_cached, bool fresh_text);

// bigcherry 1346 diagnostic seams: called immediately before/after target llama_process().
void common_speculative_target_process_begin(common_speculative * spec, const common_batch & batch);
void common_speculative_target_process_end(common_speculative * spec, const common_batch & batch);
"""

_INCLUDE_ANCHOR = "#include <cinttypes>\n"
_INCLUDE_TEXT = """\
#include <cstdio>   // bigcherry 1346: stderr prompt timing
#include <cstdlib>  // bigcherry 1346: std::getenv / std::strtol
#include <limits>   // bigcherry 1346: bounded prompt-window allocation
"""

_VIRTUAL_ANCHOR = "    virtual ~common_speculative_impl() = default;\n"
_VIRTUAL_TEXT = """\

    // bigcherry 1346 (QFP31): optional prompt lifecycle boundaries; no-op unless an implementation uses them.
    virtual void prefill_begin(
            llama_seq_id /*seq_id*/, int32_t /*n_prompt*/, int32_t /*n_cached*/, bool /*fresh_text*/) {}
    virtual void target_process_begin(const common_batch & /*batch*/) {}
    virtual void target_process_end(const common_batch & /*batch*/) {}
"""

_STATE_ANCHOR = "    std::vector<std::vector<float>> pending_h;   // [n_seq][n_embd]\n"
_STATE_TEXT = """\

    // bigcherry 1346 (QFP31): diagnostic state.
    struct bc_mtp_prompt_timing_state {
        bool collecting = false;
        int64_t target_nextn_us = 0;
        int64_t target_sync_us = 0;
        int64_t target_fetch_us = 0;
        int64_t process_us = 0;
        int64_t draft_decode_us = 0;
        int64_t host_gap_us = 0;
        int64_t last_target_return_us = 0;
        uint64_t chunks = 0;
        uint64_t tokens = 0;
    };
    bool bc_mtp_prompt_timing_on = false;
    std::vector<bc_mtp_prompt_timing_state> bc_mtp_prompt_timing;

    // WINDOW is qualified only for a fresh, single-head, non-shared text prompt.
    // The collector is pre-sized to exactly N rows, so host storage is O(N*n_embd), never O(prompt).
    struct bc_mtp_prompt_window_state {
        bool armed = false;
        bool active = false;
        int32_t prompt_tokens = 0;
        int32_t replay_first = 0;
        int32_t count = 0;
        llama_pos expected_pos = 0;
        std::vector<llama_token> ids;
        std::vector<llama_pos> pos;
        std::vector<float> h_prev;
    };
    int32_t bc_mtp_prompt_window = 0;
    std::vector<bc_mtp_prompt_window_state> bc_mtp_prompt_windows;
"""

_CTOR_ANCHOR = "        pending_h.assign(n_seq, std::vector<float>(n_embd, 0.0f));\n"
_CTOR_TEXT = """\
        if (const char * value = std::getenv("BIGCHERRY_MTP_PROMPT_TIMING")) {
            bc_mtp_prompt_timing_on = std::atoi(value) != 0;
        }
        bc_mtp_prompt_timing.resize(n_seq);

        if (const char * value = std::getenv("BIGCHERRY_MTP_PROMPT_WINDOW")) {
            char * end = nullptr;
            const long parsed = std::strtol(value, &end, 10);
            if (end != value && *end == '\0' && parsed > 0 && parsed <= 0x7fffffffL) {
                bc_mtp_prompt_window = (int32_t) parsed;
            }
        }
        bc_mtp_prompt_windows.resize(n_seq);
"""

_MTP_BEGIN_ANCHOR = """\
    void begin(llama_seq_id seq_id, const llama_tokens & prompt) override {
        // reset here rather than per round, or two identical requests differ
        common_sampler_reset(smpls[seq_id].get());
"""
_MTP_PREFILL_TEXT = """\
    bc_mtp_prompt_timing_state * bc_prompt_timing_for_batch(const common_batch & batch_in) {
        if (!bc_mtp_prompt_timing_on || batch_in.size() <= 0) {
            return nullptr;
        }

        llama_seq_id seq_id = -1;
        for (int k = 0; k < batch_in.size(); ++k) {
            const llama_seq_id cur = batch_in.tokens[k].seq_id;
            if (cur < 0 || cur >= (llama_seq_id) bc_mtp_prompt_timing.size()) {
                return nullptr;
            }
            if (seq_id < 0) {
                seq_id = cur;
            } else if (seq_id != cur) {
                return nullptr;
            }
        }

        auto & timing = bc_mtp_prompt_timing[seq_id];
        return timing.collecting ? &timing : nullptr;
    }

    void target_process_begin(const common_batch & batch_in) override {
        auto * timing = bc_prompt_timing_for_batch(batch_in);
        if (timing != nullptr && timing->last_target_return_us != 0) {
            timing->host_gap_us += ggml_time_us() - timing->last_target_return_us;
            timing->last_target_return_us = 0;
        }
    }

    void target_process_end(const common_batch & batch_in) override {
        auto * timing = bc_prompt_timing_for_batch(batch_in);
        if (timing != nullptr) {
            timing->last_target_return_us = ggml_time_us();
        }
    }

    void prefill_begin(
            llama_seq_id seq_id, int32_t n_prompt, int32_t n_cached, bool fresh_text) override {
        if (seq_id < 0 || seq_id >= (llama_seq_id) n_seq) {
            return;
        }

        if (bc_mtp_prompt_timing_on) {
            bc_mtp_prompt_timing[seq_id] = {};
            bc_mtp_prompt_timing[seq_id].collecting = true;
        }

        auto & window = bc_mtp_prompt_windows[seq_id];
        window = {};

        // Invariant WINDOW-ARM: do not truncate an existing/restored draft context.
        // V1 only arms before a completely fresh text prompt whose absolute positions start at zero.
        if (bc_mtp_prompt_window <= 0 ||
                n_prompt <= bc_mtp_prompt_window ||
                n_cached != 0 ||
                !fresh_text ||
                is_mem_shared ||
                chain_heads ||
                n_mtp_layers != 1 ||
                bc_mtp_prompt_window > (int32_t) llama_n_ctx(params.ctx_dft)) {
            return;
        }

        const size_t n_window = (size_t) bc_mtp_prompt_window;
        if ((size_t) n_embd > 0 && n_window > std::numeric_limits<size_t>::max() / (size_t) n_embd) {
            return;
        }

        window.armed = true;
        window.prompt_tokens = n_prompt;
        window.replay_first = n_prompt - bc_mtp_prompt_window;
        window.expected_pos = 0;
        window.ids.resize(n_window);
        window.pos.resize(n_window);
        window.h_prev.resize(n_window * (size_t) n_embd);
    }

"""
_MTP_BEGIN_TIMING_TEXT = """\

        if (bc_mtp_prompt_timing_on && seq_id >= 0 && seq_id < (llama_seq_id) bc_mtp_prompt_timing.size()) {
            auto & timing = bc_mtp_prompt_timing[seq_id];
            if (timing.collecting) {
                std::fprintf(stderr,
                        "BIGCHERRY_MTP_PROMPT_TIMING target_nextn_ms=%.3f target_sync_ms=%.3f target_fetch_ms=%.3f draft_process_ms=%.3f draft_decode_ms=%.3f host_gap_ms=%.3f chunks=%llu tokens=%llu\\n",
                        timing.target_nextn_us / 1000.0, timing.target_sync_us / 1000.0, timing.target_fetch_us / 1000.0,
                        timing.process_us / 1000.0, timing.draft_decode_us / 1000.0, timing.host_gap_us / 1000.0,
                        (unsigned long long) timing.chunks, (unsigned long long) timing.tokens);
                timing.collecting = false;
            }
        }

        auto & bc_window = bc_mtp_prompt_windows[seq_id];
        if (bc_window.armed) {
            // Invariant WINDOW-REPLAY: collector rows are exactly the final N absolute prompt positions,
            // and row p carries token[p] with the exact native MTP input hidden h[p-1].
            const int32_t N = (int32_t) prompt.size();
            const int32_t n_replay = bc_mtp_prompt_window;
            const bool complete =
                N == bc_window.prompt_tokens &&
                bc_window.expected_pos == N &&
                bc_window.count == n_replay &&
                !bc_window.pos.empty() &&
                bc_window.pos.front() == bc_window.replay_first &&
                bc_window.pos.back() == N - 1;
            if (!complete) {
                throw std::runtime_error("BIGCHERRY_MTP_PROMPT_WINDOW collector invariant failed");
            }

            auto * ctx_dft = this->params.ctx_dft;
            auto * mem_dft = llama_get_memory(ctx_dft);
            llama_memory_seq_rm(mem_dft, seq_id, -1, -1);

            const int32_t n_batch_dft = (int32_t) llama_n_batch(ctx_dft);
            for (int32_t off = 0; off < n_replay; off += n_batch_dft) {
                const int32_t n_cur = std::min(n_batch_dft, n_replay - off);
                batch.clear();
                for (int32_t i = 0; i < n_cur; ++i) {
                    const int32_t row = off + i;
                    const int32_t idx = batch.add(bc_window.ids[row], bc_window.pos[row], seq_id, false);
                    batch.set_embd(idx, { bc_window.h_prev.data() + (size_t) row * n_embd, 1, (size_t) n_embd });
                }

                const int32_t rc = llama_process(ctx_dft, LLAMA_PROCESS_TYPE_DECODE, batch.get());
                if (rc != 0) {
                    throw std::runtime_error("BIGCHERRY_MTP_PROMPT_WINDOW replay decode failed");
                }
            }

            // Async host inputs may still reference the bounded collector. Join before freeing it.
            llama_synchronize(ctx_dft);

            const llama_pos replay_pos_max = llama_memory_seq_pos_max(mem_dft, seq_id);
            if (replay_pos_max != N - 1) {
                throw std::runtime_error("BIGCHERRY_MTP_PROMPT_WINDOW replay position invariant failed");
            }

            bc_window.armed = false;
            bc_window.active = true;
            bc_window.ids.clear();
            bc_window.pos.clear();
            bc_window.h_prev.clear();
        }
"""

_PROCESS_START_OLD = """\
        const int32_t n_tokens = batch_in.size();

        // remember the first and last batch index for each sequence
"""
_PROCESS_START_NEW = """\
        const int32_t n_tokens = batch_in.size();

        // bigcherry 1346: attribute the diagnostic only when this process() call belongs to one sequence.
        bc_mtp_prompt_timing_state * bc_pt_state = bc_prompt_timing_for_batch(batch_in);
        const int64_t bc_pt_process_t0 = bc_pt_state ? ggml_time_us() : 0;
        int64_t bc_pt_target_sync_us = 0;
        int64_t bc_pt_target_fetch_us = 0;
        int64_t bc_pt_draft_decode_us = 0;

        // remember the first and last batch index for each sequence
"""

_PROCESS_NATIVE_OLD = """\
        const size_t row_bytes = (size_t) n_embd * sizeof(float);

        // if kv is shared with target (e.g Gemma4), then we can skip this catch-up decode
        if (!is_mem_shared) {
            batch.clear();

            // pair each token with the tgt embedding shifted right by one position, and
            // the first token of each sequence with the pending embedding from a previous run
            // assumes that the tokens in the batch are sequential for each sequence
            // i.e. we cannot have seq_id like this: [0, 0, 0, 1, 1, 0, 1, 1]
            //                                                       ^--- this is a problem
            // TODO:this is generally true, but would be nice to assert it
            const float * h_tgt = llama_get_embeddings_nextn(ctx_tgt);

            for (int k = 0; k < n_tokens; ++k) {
                const llama_seq_id seq_id = batch_in.tokens[k].seq_id;

                const int32_t idx = batch.add(batch_in.tokens[k].id, batch_in.tokens[k].pos[0], seq_id, false);

                const float * h_row = k == i_batch_beg[seq_id]
                    ? pending_h[seq_id].data()
                    : h_tgt + (size_t) (k - 1) * n_embd;

                batch.set_embd(idx, { h_row, 1, (size_t) n_embd });
            }

            auto * mem_dft = llama_get_memory(ctx_dft);

            bool ok = true;
            for (int head = 0; head < n_mtp_layers; ++head) {
                if (chain_heads) {
                    // ref: https://github.com/ggml-org/llama.cpp/pull/24340/changes#r3413498544
                    for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
                        if (i_batch_beg[seq_id] < 0) {
                            continue;
                        }
                        llama_memory_seq_rm(mem_dft, seq_id, batch_in.tokens[i_batch_beg[seq_id]].pos[0], -1);
                    }
                    llama_set_nextn_layer_offset(ctx_dft, head);
                }

                const int32_t rc = llama_process(ctx_dft, LLAMA_PROCESS_TYPE_DECODE, batch.get());
                if (rc != 0) {
                    SPC_ERR("llama_process(ctx_dft) head=%d failed rc=%d (pos=%d)\n",
                            head, (int) rc, (int) batch_in.tokens[0].pos[0]);
                    ok = false;
                    break;
                }
            }

            if (chain_heads) {
                llama_set_nextn_layer_offset(ctx_dft, 0); // restore default for non-draft decodes
            }
            if (!ok) {
                return false;
            }
        }

        for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
            if (i_batch_end[seq_id] < 0) {
                continue;
            }

            const int32_t n_rows = i_batch_end[seq_id] - i_batch_beg[seq_id] + 1;
            verify_h_rows[seq_id] = n_rows;
            verify_h[seq_id].resize((size_t) n_rows * n_embd);

            for (int32_t i = 0; i < n_rows; ++i) {
                const float * h = llama_get_embeddings_nextn_ith(ctx_tgt, i_batch_beg[seq_id] + i);
                std::memcpy(verify_h[seq_id].data() + (size_t) i * n_embd, h, row_bytes);
            }

            std::memcpy(pending_h[seq_id].data(),
                    verify_h[seq_id].data() + (size_t) (n_rows - 1) * n_embd, row_bytes);
        }

        return true;
"""
_PROCESS_NATIVE_NEW = """\
        const size_t row_bytes = (size_t) n_embd * sizeof(float);

        std::vector<uint8_t> bc_window_fetch(n_seq, 0);
        bool bc_need_target_nextn = false;

        for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
            const int32_t beg = i_batch_beg[seq_id];
            const int32_t end = i_batch_end[seq_id];
            if (beg < 0 || end < 0) {
                continue;
            }

            auto & window = bc_mtp_prompt_windows[seq_id];
            if (!window.armed) {
                bc_need_target_nextn = true;
                continue;
            }

            // Invariant WINDOW-POS: once armed, every prompt row is fresh text at the next absolute position.
            for (int32_t k = beg; k <= end; ++k) {
                if (batch_in.tokens[k].seq_id != seq_id ||
                        batch_in.tokens[k].pos[0] != window.expected_pos) {
                    SPC_ERR("BIGCHERRY_MTP_PROMPT_WINDOW position mismatch seq=%d expected=%d got=%d\n",
                            (int) seq_id, (int) window.expected_pos, (int) batch_in.tokens[k].pos[0]);
                    return false;
                }
                window.expected_pos++;
            }

            // Replay token[P-N] consumes h[P-N-1]. Fetch that one predecessor boundary,
            // but do not draft-decode it; every earlier chunk remains free of the MTP join.
            const llama_pos predecessor = (llama_pos) window.replay_first - 1;
            if (batch_in.tokens[end].pos[0] >= predecessor) {
                bc_window_fetch[seq_id] = 1;
                bc_need_target_nextn = true;
            }
        }

        const float * h_tgt = nullptr;
        bool bc_pt_target_fetched = false;

        // if kv is shared with target (e.g Gemma4), then we can skip this catch-up decode
        if (!is_mem_shared && bc_need_target_nextn) {
            if (bc_pt_state) {
                const int64_t bc_pt_sync_t0 = ggml_time_us();
                llama_synchronize(ctx_tgt);
                bc_pt_target_sync_us += ggml_time_us() - bc_pt_sync_t0;
            }

            const int64_t bc_pt_fetch_t0 = bc_pt_state ? ggml_time_us() : 0;
            h_tgt = llama_get_embeddings_nextn(ctx_tgt);
            if (bc_pt_state) {
                bc_pt_target_fetch_us += ggml_time_us() - bc_pt_fetch_t0;
                bc_pt_target_fetched = true;
            }

            batch.clear();

            for (int k = 0; k < n_tokens; ++k) {
                const llama_seq_id seq_id = batch_in.tokens[k].seq_id;
                auto & window = bc_mtp_prompt_windows[seq_id];

                if (window.armed) {
                    if (!bc_window_fetch[seq_id]) {
                        continue;
                    }

                    const llama_pos pos = batch_in.tokens[k].pos[0];
                    if (pos < window.replay_first) {
                        continue;
                    }

                    const int32_t row = (int32_t) (pos - window.replay_first);
                    if (row != window.count || row < 0 || row >= bc_mtp_prompt_window) {
                        SPC_ERR("BIGCHERRY_MTP_PROMPT_WINDOW collector mismatch seq=%d row=%d count=%d\n",
                                (int) seq_id, row, window.count);
                        return false;
                    }

                    const float * h_row = k == i_batch_beg[seq_id]
                        ? pending_h[seq_id].data()
                        : h_tgt + (size_t) (k - 1) * n_embd;

                    window.ids[row] = batch_in.tokens[k].id;
                    window.pos[row] = pos;
                    std::memcpy(window.h_prev.data() + (size_t) row * n_embd, h_row, row_bytes);
                    window.count++;
                    continue;
                }

                const int32_t idx = batch.add(batch_in.tokens[k].id, batch_in.tokens[k].pos[0], seq_id, false);
                const float * h_row = k == i_batch_beg[seq_id]
                    ? pending_h[seq_id].data()
                    : h_tgt + (size_t) (k - 1) * n_embd;
                batch.set_embd(idx, { h_row, 1, (size_t) n_embd });
            }

            if (batch.size() > 0) {
                auto * mem_dft = llama_get_memory(ctx_dft);

                bool ok = true;
                for (int head = 0; head < n_mtp_layers; ++head) {
                    if (chain_heads) {
                        for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
                            if (i_batch_beg[seq_id] < 0) {
                                continue;
                            }
                            llama_memory_seq_rm(mem_dft, seq_id, batch_in.tokens[i_batch_beg[seq_id]].pos[0], -1);
                        }
                        llama_set_nextn_layer_offset(ctx_dft, head);
                    }

                    const int64_t bc_pt_draft_t0 = bc_pt_state ? ggml_time_us() : 0;
                    const int32_t rc = llama_process(ctx_dft, LLAMA_PROCESS_TYPE_DECODE, batch.get());
                    if (bc_pt_state) {
                        bc_pt_draft_decode_us += ggml_time_us() - bc_pt_draft_t0;
                    }
                    if (rc != 0) {
                        SPC_ERR("llama_process(ctx_dft) head=%d failed rc=%d (pos=%d)\n",
                                head, (int) rc, (int) batch_in.tokens[0].pos[0]);
                        ok = false;
                        break;
                    }
                }

                if (chain_heads) {
                    llama_set_nextn_layer_offset(ctx_dft, 0);
                }
                if (!ok) {
                    return false;
                }
            }
        }

        const int64_t bc_pt_verify_t0 = bc_pt_state ? ggml_time_us() : 0;

        for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
            if (i_batch_end[seq_id] < 0) {
                continue;
            }

            auto & window = bc_mtp_prompt_windows[seq_id];
            if (window.armed && !bc_window_fetch[seq_id]) {
                continue;
            }

            const int32_t n_rows = i_batch_end[seq_id] - i_batch_beg[seq_id] + 1;
            verify_h_rows[seq_id] = n_rows;
            verify_h[seq_id].resize((size_t) n_rows * n_embd);

            for (int32_t i = 0; i < n_rows; ++i) {
                const float * h = llama_get_embeddings_nextn_ith(ctx_tgt, i_batch_beg[seq_id] + i);
                std::memcpy(verify_h[seq_id].data() + (size_t) i * n_embd, h, row_bytes);
            }

            std::memcpy(pending_h[seq_id].data(),
                    verify_h[seq_id].data() + (size_t) (n_rows - 1) * n_embd, row_bytes);
        }

        if (bc_pt_state) {
            if (bc_pt_target_fetched) {
                bc_pt_target_fetch_us += ggml_time_us() - bc_pt_verify_t0;
            }
            bc_pt_state->target_sync_us += bc_pt_target_sync_us;
            bc_pt_state->target_fetch_us += bc_pt_target_fetch_us;
            bc_pt_state->target_nextn_us += bc_pt_target_sync_us + bc_pt_target_fetch_us;
            bc_pt_state->draft_decode_us += bc_pt_draft_decode_us;
            bc_pt_state->process_us += ggml_time_us() - bc_pt_process_t0;
            bc_pt_state->chunks++;
            bc_pt_state->tokens += (uint64_t) n_tokens;
        }

        return true;
"""

_WRAPPER_ANCHOR = "void common_speculative_begin(common_speculative * spec, llama_seq_id seq_id, const llama_tokens & prompt) {\n"
_WRAPPER_TEXT = """\
void common_speculative_prefill_begin(
        common_speculative * spec, llama_seq_id seq_id,
        int32_t n_prompt, int32_t n_cached, bool fresh_text) {
    if (spec == nullptr) {
        return;
    }

    for (auto & impl : spec->impls) {
        impl->prefill_begin(seq_id, n_prompt, n_cached, fresh_text);
    }
}

void common_speculative_target_process_begin(common_speculative * spec, const common_batch & batch) {
    if (spec == nullptr) {
        return;
    }

    for (auto & impl : spec->impls) {
        impl->target_process_begin(batch);
    }
}

void common_speculative_target_process_end(common_speculative * spec, const common_batch & batch) {
    if (spec == nullptr) {
        return;
    }

    for (auto & impl : spec->impls) {
        impl->target_process_end(batch);
    }
}

"""

_SERVER_ANCHOR = "                        slot.prompt.tokens.keep_first(n_past);\n"
_TARGET_PROCESS_OLD = """\
        queue_tasks.yield_to_queue([&]() {
            ret = llama_process(ctx_tgt, LLAMA_PROCESS_TYPE_DECODE, batch.view.get());
            if (ret == 0 && has_output) {
"""
_TARGET_PROCESS_NEW = """\
        queue_tasks.yield_to_queue([&]() {
            if (spec) {
                common_speculative_target_process_begin(spec.get(), batch.view);
            }
            ret = llama_process(ctx_tgt, LLAMA_PROCESS_TYPE_DECODE, batch.view.get());
            if (spec) {
                common_speculative_target_process_end(spec.get(), batch.view);
            }
            if (ret == 0 && has_output) {
"""
_SERVER_TEXT = """\

                        // bigcherry 1346 (QFP31): resolved prompt-start contract. WINDOW only arms for a
                        // fresh text prompt: no cached prefix, no MTMD and no shared-prefix child.
                        common_speculative_prefill_begin(
                                spec.get(), slot.id, (int32_t) slot.task->n_tokens(), n_past,
                                mctx == nullptr && slot.task->n_tokens_shared == 0);
"""

PATCHES = [
    FilePatch(
        path="common/speculative.h",
        description="1346: expose resolved prompt-start and target-process lifecycle hooks for MTP WINDOW/timing",
        language="none",
        edits=(
            Edit(
                id="mtp-prompt-prefill-api",
                anchor=re.escape(_API_ANCHOR),
                mode="insert_after",
                text=_API_TEXT,
                guard=r"int32_t n_prompt, int32_t n_cached, bool fresh_text\);",
                rationale="Stable public speculative-driver seam immediately before the existing generation-begin hook.",
                expect_matches=1,
                max_span_lines=2,
            ),
        ),
    ),
    FilePatch(
        path="common/speculative.cpp",
        description="1346: default-off bounded MTP prompt WINDOW plus target-sync/catch-up timing",
        language="none",
        edits=(
            Edit(
                id="mtp-prompt-timing-includes",
                anchor=re.escape(_INCLUDE_ANCHOR),
                mode="insert_after",
                text=_INCLUDE_TEXT,
                guard=r"#include <cstdio>   // bigcherry 1346: stderr prompt timing",
                rationale="Existing standard-library include block; host-only fprintf/getenv helpers.",
                expect_matches=1,
                max_span_lines=2,
            ),
            Edit(
                id="mtp-prompt-prefill-virtual",
                anchor=re.escape(_VIRTUAL_ANCHOR),
                mode="insert_after",
                text=_VIRTUAL_TEXT,
                guard=r"int32_t /\*n_prompt\*/, int32_t /\*n_cached\*/, bool /\*fresh_text\*/\) \{\}",
                rationale="Optional lifecycle method on the speculative implementation base; existing implementations stay no-op.",
                expect_matches=1,
                max_span_lines=2,
            ),
            Edit(
                id="mtp-prompt-timing-state",
                anchor=re.escape(_STATE_ANCHOR),
                mode="insert_after",
                text=_STATE_TEXT,
                guard=r"struct bc_mtp_prompt_timing_state \{",
                rationale="MTP-owned per-sequence diagnostic state adjacent to its existing cross-batch hidden carry.",
                expect_matches=1,
                max_span_lines=2,
            ),
            Edit(
                id="mtp-prompt-timing-ctor",
                anchor=re.escape(_CTOR_ANCHOR),
                mode="insert_after",
                text=_CTOR_TEXT,
                guard=r"bc_mtp_prompt_timing\.resize\(n_seq\);",
                rationale="Size diagnostic state with the existing per-sequence MTP state; flag defaults off.",
                expect_matches=1,
                max_span_lines=2,
            ),
            Edit(
                id="mtp-prompt-prefill-override",
                anchor=re.escape(_MTP_BEGIN_ANCHOR),
                mode="insert_before",
                text=_MTP_PREFILL_TEXT,
                guard=r"bc_mtp_prompt_timing\[seq_id\] = \{\};",
                rationale="MTP-specific prompt-start reset immediately before its existing generation-begin method.",
                expect_matches=1,
                max_span_lines=4,
            ),
            Edit(
                id="mtp-prompt-timing-report",
                anchor=re.escape(_MTP_BEGIN_ANCHOR),
                mode="insert_after",
                text=_MTP_BEGIN_TIMING_TEXT,
                guard=r"BIGCHERRY_MTP_PROMPT_TIMING target_nextn_ms=",
                rationale="Existing begin() is the server's prompt-end boundary; report once before generation reads draft state.",
                expect_matches=1,
                max_span_lines=4,
            ),
            Edit(
                id="mtp-prompt-timing-process-start",
                anchor=re.escape(_PROCESS_START_OLD),
                mode="replace",
                text=_PROCESS_START_NEW,
                guard=r"bc_mtp_prompt_timing_state \* bc_pt_state = bc_prompt_timing_for_batch\(batch_in\);",
                rationale="MTP process() token-count seam; establish single-sequence attribution and whole-process wall timer.",
                expect_matches=1,
                max_span_lines=4,
            ),
            Edit(
                id="mtp-prompt-window-process",
                anchor=re.escape(_PROCESS_NATIVE_OLD),
                mode="replace",
                text=_PROCESS_NATIVE_NEW,
                guard=r"std::vector<uint8_t> bc_window_fetch\(n_seq, 0\);",
                rationale="Native MTP catch-up unit: qualified early WINDOW chunks skip target getters and draft decode; final N rows are collected exactly.",
                expect_matches=1,
                max_span_lines=100,
            ),
            Edit(
                id="mtp-prompt-prefill-wrapper",
                anchor=re.escape(_WRAPPER_ANCHOR),
                mode="insert_before",
                text=_WRAPPER_TEXT,
                guard=r"int32_t n_prompt, int32_t n_cached, bool fresh_text\) \{",
                rationale="Public wrapper next to the existing common_speculative_begin wrapper.",
                expect_matches=1,
                max_span_lines=2,
            ),
        ),
    ),
    FilePatch(
        path="tools/server/server-context.cpp",
        description="1346: signal resolved prompt start after cache/checkpoint decisions",
        language="none",
        edits=(
            Edit(
                id="mtp-prompt-server-prefill-begin",
                anchor=re.escape(_SERVER_ANCHOR),
                mode="insert_after",
                text=_SERVER_TEXT,
                guard=r"mctx == nullptr && slot\.task->n_tokens_shared == 0\);",
                rationale="SLOT_STATE_STARTED has finalized n_past and restored any cache/checkpoint state at this line.",
                expect_matches=1,
                max_span_lines=2,
            ),
            Edit(
                id="mtp-prompt-target-process-gap",
                anchor=re.escape(_TARGET_PROCESS_OLD),
                mode="replace",
                text=_TARGET_PROCESS_NEW,
                guard=r"common_speculative_target_process_begin\(spec\.get\(\), batch\.view\);",
                rationale="Target decode lambda gives exact process-call entry/return timestamps around llama_process().",
                expect_matches=1,
                max_span_lines=5,
            ),
        ),
    ),
]

ENV_DOCS = (
    EnvDoc(
        "BIGCHERRY_MTP_PROMPT_TIMING",
        "0|1",
        "0",
        "diagnostic: split MTP prompt wall into explicit target sync, NextN fetch/copy, draft catch-up decode and inter-submit host gap; single-slot attribution",
    ),
    EnvDoc(
        "BIGCHERRY_MTP_PROMPT_WINDOW",
        "tokens",
        "0",
        "optimization: on qualified fresh single-head MTP prompts, keep/replay only the final N draft-context prompt rows; 0 disables",
    ),
)
