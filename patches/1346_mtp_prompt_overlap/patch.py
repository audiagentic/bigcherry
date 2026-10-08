"""1346: MTP prompt timing diagnostic (QFP31).

BIGCHERRY_MTP_PROMPT_TIMING=1 times the existing MTP prompt path without changing model work or ordering.\nAt prompt end it reports deferred on/off, target blocking, draft catch-up, total/per-chunk host gap, chunks and tokens.

The diagnostic uses a prompt-start hook only to reset attribution and target-process hooks to measure the
inter-submit host gap. Mixed-sequence batches are not accumulated because their MTP process wall cannot be
assigned to one prompt honestly. The former BIGCHERRY_MTP_PROMPT_WINDOW mechanism is intentionally absent.
"""

from __future__ import annotations

import re

from bigcherry.patcher import Edit, EnvDoc, FilePatch

GROUP = "rdna-boosts"
STATE = "validated"

_API_ANCHOR = "common_speculative_draft_params & common_speculative_get_draft_params(common_speculative * spec, llama_seq_id seq_id);\n"

_INCLUDE_ANCHOR = "#include <cinttypes>\n"

_VIRTUAL_ANCHOR = "    virtual ~common_speculative_impl() = default;\n"

_STATE_ANCHOR = "    std::vector<std::vector<float>> pending_h;   // [n_seq][n_embd]\n"

_CTOR_ANCHOR = "        pending_h.assign(n_seq, std::vector<float>(n_embd, 0.0f));\n"

_MTP_BEGIN_ANCHOR = r"""    void begin(llama_seq_id seq_id, const llama_tokens & prompt) override {
        // reset here rather than per round, or two identical requests differ
        common_sampler_reset(smpls[seq_id].get());
"""

_PROCESS_START_OLD = r"""        const int32_t n_tokens = batch_in.size();

        // remember the first and last batch index for each sequence
"""

_PROCESS_NATIVE_OLD = r"""        const size_t row_bytes = (size_t) n_embd * sizeof(float);

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

_WRAPPER_ANCHOR = "void common_speculative_begin(common_speculative * spec, llama_seq_id seq_id, const llama_tokens & prompt) {\n"

_SERVER_ANCHOR = "                        slot.prompt.tokens.keep_first(n_past);\n"

_TARGET_PROCESS_OLD = r"""            const int64_t bc_t0 = bc_spec_timing_on() ? ggml_time_us() : 0;  // bigcherry 1317
            ret = llama_process(ctx_tgt, LLAMA_PROCESS_TYPE_DECODE, batch.view.get());
            const int64_t bc_t1 = bc_spec_timing_on() ? ggml_time_us() : 0;
"""

_TARGET_PROCESS_CALL = r"""            ret = llama_process(ctx_tgt, LLAMA_PROCESS_TYPE_DECODE, batch.view.get());
"""
_TARGET_PROCESS_BEGIN_TEXT = r"""            if (spec) {
                common_speculative_target_process_begin(spec.get(), batch.view);
            }
"""
_TARGET_PROCESS_END_TEXT = r"""            if (spec) {
                common_speculative_target_process_end(spec.get(), batch.view);
            }
"""

_API_TEXT = r"""
// bigcherry 1346 (QFP31): reset prompt-timing attribution at the resolved prompt start.
void common_speculative_prefill_begin(common_speculative * spec, llama_seq_id seq_id);

// bigcherry 1346 diagnostic seams: called immediately before/after target llama_process().
void common_speculative_target_process_begin(common_speculative * spec, const common_batch & batch);
void common_speculative_target_process_end(common_speculative * spec, const common_batch & batch);
"""

_INCLUDE_TEXT = r"""#include <cstdio>   // bigcherry 1346: stderr prompt timing
#include <cstdlib>  // bigcherry 1346: std::getenv
"""

_VIRTUAL_TEXT = r"""
    // bigcherry 1346 (QFP31): optional timing lifecycle boundaries; no-op for other implementations.
    virtual void prefill_begin(llama_seq_id /*seq_id*/) {}
    virtual void target_process_begin(const common_batch & /*batch*/) {}
    virtual void target_process_end(const common_batch & /*batch*/) {}
"""

_STATE_TEXT = r"""
    // bigcherry 1346 (QFP42 step 0): per-sequence prompt timing; diagnostic only.
    struct bc_mtp_prompt_timing_state {
        bool collecting = false;
        bool deferred = false;
        int64_t target_nextn_us = 0;
        int64_t target_sync_us = 0;
        int64_t target_fetch_us = 0;
        int64_t process_us = 0;
        int64_t draft_decode_us = 0;
        int64_t target_block_us = 0;
        int64_t draft_catchup_us = 0;
        int64_t host_gap_us = 0;
        int64_t last_target_return_us = 0;
        uint64_t chunks = 0;
        uint64_t tokens = 0;
    };
    bool bc_mtp_prompt_timing_on = false;
    std::vector<bc_mtp_prompt_timing_state> bc_mtp_prompt_timing;
"""

_CTOR_TEXT = r"""        if (const char * value = std::getenv("BIGCHERRY_MTP_PROMPT_TIMING")) {
            bc_mtp_prompt_timing_on = std::atoi(value) != 0;
        }
        bc_mtp_prompt_timing.resize(n_seq);
"""

_MTP_PREFILL_TEXT = r"""    bc_mtp_prompt_timing_state * bc_prompt_timing_for_batch(const common_batch & batch_in) {
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

    bc_mtp_prompt_timing_state * bc_prompt_timing_for_tokens(const std::vector<common_batch::token> & tokens) {
        if (!bc_mtp_prompt_timing_on || tokens.empty()) {
            return nullptr;
        }

        llama_seq_id seq_id = -1;
        for (const auto & tok : tokens) {
            const llama_seq_id cur = tok.seq_id;
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

    void prefill_begin(llama_seq_id seq_id) override {
        if (!bc_mtp_prompt_timing_on || seq_id < 0 || seq_id >= (llama_seq_id) bc_mtp_prompt_timing.size()) {
            return;
        }
        bc_mtp_prompt_timing[seq_id] = {};
        bc_mtp_prompt_timing[seq_id].collecting = true;
        bc_mtp_prompt_timing[seq_id].deferred = bc_deferred_enabled;
    }

"""

_MTP_BEGIN_TIMING_TEXT = r"""
        if (bc_mtp_prompt_timing_on && seq_id >= 0 && seq_id < (llama_seq_id) bc_mtp_prompt_timing.size()) {
            auto & timing = bc_mtp_prompt_timing[seq_id];
            if (timing.collecting) {
                if (timing.last_target_return_us != 0) {
                    timing.host_gap_us += ggml_time_us() - timing.last_target_return_us;
                    timing.last_target_return_us = 0;
                }
                const double host_gap_per_chunk_ms = timing.chunks > 0
                    ? timing.host_gap_us / 1000.0 / (double) timing.chunks
                    : 0.0;
                std::fprintf(stderr,
                        "BIGCHERRY_MTP_PROMPT_TIMING deferred=%d target_block_ms=%.3f draft_catchup_ms=%.3f host_gap_ms=%.3f host_gap_per_chunk_ms=%.3f chunks=%llu tokens=%llu\n",
                        timing.deferred ? 1 : 0,
                        timing.target_block_us / 1000.0,
                        timing.draft_catchup_us / 1000.0,
                        timing.host_gap_us / 1000.0,
                        host_gap_per_chunk_ms,
                        (unsigned long long) timing.chunks,
                        (unsigned long long) timing.tokens);
                timing.collecting = false;
            }
        }
"""

_PROCESS_START_NEW = r"""        const int32_t n_tokens = batch_in.size();

        // bigcherry 1346: attribute the diagnostic only when this process() call belongs to one sequence.
        bc_mtp_prompt_timing_state * bc_pt_state = bc_prompt_timing_for_batch(batch_in);
        const int64_t bc_pt_process_t0 = bc_pt_state ? ggml_time_us() : 0;
        int64_t bc_pt_target_sync_us = 0;
        int64_t bc_pt_target_fetch_us = 0;
        int64_t bc_pt_draft_decode_us = 0;

        // remember the first and last batch index for each sequence
"""

_PROCESS_NATIVE_NEW = r"""        const size_t row_bytes = (size_t) n_embd * sizeof(float);

        bool bc_pt_target_fetched = false;

        // if kv is shared with target (e.g Gemma4), then we can skip this catch-up decode
        if (!is_mem_shared) {
            if (bc_pt_state) {
                const int64_t bc_pt_sync_t0 = ggml_time_us();
                llama_synchronize(ctx_tgt);
                bc_pt_target_sync_us += ggml_time_us() - bc_pt_sync_t0;
            }

            const int64_t bc_pt_fetch_t0 = bc_pt_state ? ggml_time_us() : 0;
            const float * h_tgt = llama_get_embeddings_nextn(ctx_tgt);
            if (bc_pt_state) {
                bc_pt_target_fetch_us += ggml_time_us() - bc_pt_fetch_t0;
                bc_pt_target_fetched = true;
            }

            batch.clear();

            // pair each token with the tgt embedding shifted right by one position, and
            // the first token of each sequence with the pending embedding from a previous run
            // assumes that the tokens in the batch are sequential for each sequence
            // i.e. we cannot have seq_id like this: [0, 0, 0, 1, 1, 0, 1, 1]
            //                                                       ^--- this is a problem
            // TODO:this is generally true, but would be nice to assert it
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
                llama_set_nextn_layer_offset(ctx_dft, 0); // restore default for non-draft decodes
            }
            if (!ok) {
                return false;
            }
        }

        const int64_t bc_pt_verify_t0 = bc_pt_state ? ggml_time_us() : 0;

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

        if (bc_pt_state) {
            if (bc_pt_target_fetched) {
                bc_pt_target_fetch_us += ggml_time_us() - bc_pt_verify_t0;
            }
            bc_pt_state->target_sync_us += bc_pt_target_sync_us;
            bc_pt_state->target_fetch_us += bc_pt_target_fetch_us;
            bc_pt_state->target_nextn_us += bc_pt_target_sync_us + bc_pt_target_fetch_us;
            bc_pt_state->draft_decode_us += bc_pt_draft_decode_us;
            const int64_t bc_pt_process_us = ggml_time_us() - bc_pt_process_t0;
            bc_pt_state->process_us += bc_pt_process_us;
            bc_pt_state->target_block_us += bc_pt_target_sync_us + bc_pt_target_fetch_us;
            const int64_t bc_pt_catchup_us = bc_pt_process_us - bc_pt_target_sync_us - bc_pt_target_fetch_us;
            if (bc_pt_catchup_us > 0) {
                bc_pt_state->draft_catchup_us += bc_pt_catchup_us;
            }
            bc_pt_state->chunks++;
            bc_pt_state->tokens += (uint64_t) n_tokens;
        }

        return true;
"""

_DEFERRED_FLUSH_OLD = r"""    bool flush_deferred() override {
        if (!bc_deferred_enabled || bc_pending < 0) {
            return true;
        }

        const int pending = bc_pending;
        bc_pending = -1;
        const bool ok = bc_process_snapshot(bc_chunks[pending]);
        bc_chunks[pending].valid = false;
        return ok;
    }
"""

_DEFERRED_FLUSH_NEW = r"""    bool flush_deferred() override {
        if (!bc_deferred_enabled || bc_pending < 0) {
            return true;
        }

        const int pending = bc_pending;
        bc_pending = -1;
        auto * bc_pt_state = bc_prompt_timing_for_tokens(bc_chunks[pending].tokens);
        const int64_t bc_pt_catchup_t0 = bc_pt_state ? ggml_time_us() : 0;
        const bool ok = bc_process_snapshot(bc_chunks[pending]);
        if (bc_pt_state) {
            bc_pt_state->draft_catchup_us += ggml_time_us() - bc_pt_catchup_t0;
        }
        bc_chunks[pending].valid = false;
        return ok;
    }
"""

_DEFERRED_NEXTN_OLD = r"""        auto * ctx_tgt = this->params.ctx_tgt;
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
"""

_DEFERRED_NEXTN_NEW = r"""        auto * bc_pt_state = bc_prompt_timing_for_batch(batch_in);
        auto * ctx_tgt = this->params.ctx_tgt;
        const int64_t bc_pt_block_t0 = bc_pt_state ? ggml_time_us() : 0;
        const float * h_tgt = llama_get_embeddings_nextn(ctx_tgt); // blocks for current target after prior catch-up ran
        if (bc_pt_state) {
            const int64_t bc_pt_block_us = ggml_time_us() - bc_pt_block_t0;
            bc_pt_state->target_block_us += bc_pt_block_us;
            bc_pt_state->target_nextn_us += bc_pt_block_us;
            bc_pt_state->target_fetch_us += bc_pt_block_us;
        }
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
        if (bc_pt_state) {
            bc_pt_state->chunks++;
            bc_pt_state->tokens += (uint64_t) batch_in.size();
        }
"""

_WRAPPER_TEXT = r"""void common_speculative_prefill_begin(common_speculative * spec, llama_seq_id seq_id) {
    if (spec == nullptr) {
        return;
    }

    for (auto & impl : spec->impls) {
        impl->prefill_begin(seq_id);
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

_TARGET_PROCESS_NEW = r"""            if (spec) {
                common_speculative_target_process_begin(spec.get(), batch.view);
            }
            const int64_t bc_t0 = bc_spec_timing_on() ? ggml_time_us() : 0;  // bigcherry 1317
            ret = llama_process(ctx_tgt, LLAMA_PROCESS_TYPE_DECODE, batch.view.get());
            const int64_t bc_t1 = bc_spec_timing_on() ? ggml_time_us() : 0;
            if (spec) {
                common_speculative_target_process_end(spec.get(), batch.view);
            }
"""

_SERVER_TEXT = r"""
                        // bigcherry 1346 (QFP31): resolved prompt-start boundary for timing attribution.
                        common_speculative_prefill_begin(spec.get(), slot.id);
"""

PATCHES = [
    FilePatch(
        path="common/speculative.h",
        description="1346: expose MTP prompt-timing lifecycle hooks",
        language="none",
        edits=(
            Edit(
                id="mtp-prompt-timing-api",
                anchor=re.escape(_API_ANCHOR),
                mode="insert_after",
                text=_API_TEXT,
                guard=r"void common_speculative_target_process_begin\(common_speculative \* spec, const common_batch & batch\);",
                rationale="Narrow prompt-start and target-process seams required only for timing attribution.",
                expect_matches=1,
                max_span_lines=2,
            ),
        ),
    ),
    FilePatch(
        path="common/speculative.cpp",
        description="1346: default-off target-sync and MTP catch-up timing",
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
                id="mtp-prompt-timing-virtual",
                anchor=re.escape(_VIRTUAL_ANCHOR),
                mode="insert_after",
                text=_VIRTUAL_TEXT,
                guard=r"virtual void prefill_begin\(llama_seq_id /\*seq_id\*/\) \{\}",
                rationale="Optional timing lifecycle methods; non-MTP implementations remain no-op.",
                expect_matches=1,
                max_span_lines=2,
            ),
            Edit(
                id="mtp-prompt-timing-state",
                anchor=re.escape(_STATE_ANCHOR),
                mode="insert_after",
                text=_STATE_TEXT,
                guard=r"struct bc_mtp_prompt_timing_state \{",
                rationale="MTP-owned per-sequence timing state adjacent to its cross-batch hidden carry.",
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
                id="mtp-prompt-timing-prefill",
                anchor=re.escape(_MTP_BEGIN_ANCHOR),
                mode="insert_before",
                text=_MTP_PREFILL_TEXT,
                guard=r"bc_mtp_prompt_timing\[seq_id\]\.collecting = true;",
                rationale="Reset timing at the resolved prompt-start boundary before generation begin.",
                expect_matches=1,
                max_span_lines=4,
            ),
            Edit(
                id="mtp-prompt-timing-report",
                anchor=re.escape(_MTP_BEGIN_ANCHOR),
                mode="insert_after",
                text=_MTP_BEGIN_TIMING_TEXT,
                guard=r"BIGCHERRY_MTP_PROMPT_TIMING target_nextn_ms=",
                rationale="Existing begin() is the prompt-end boundary; report once before generation.",
                expect_matches=1,
                max_span_lines=4,
            ),
            Edit(
                id="mtp-prompt-timing-process-start",
                anchor=re.escape(_PROCESS_START_OLD),
                mode="replace",
                text=_PROCESS_START_NEW,
                guard=r"bc_mtp_prompt_timing_state \* bc_pt_state = bc_prompt_timing_for_batch\(batch_in\);",
                rationale="Establish single-sequence attribution and whole-process wall timer.",
                expect_matches=1,
                max_span_lines=4,
            ),
            Edit(
                id="mtp-prompt-timing-process",
                anchor=re.escape(_PROCESS_NATIVE_OLD),
                mode="replace",
                text=_PROCESS_NATIVE_NEW,
                guard=r"bc_pt_state->target_sync_us \+= bc_pt_target_sync_us;",
                rationale="Time the native target synchronization/fetch and draft catch-up without changing work or order.",
                expect_matches=1,
                max_span_lines=100,
            ),
            Edit(
                id="mtp-prompt-timing-deferred-flush",
                anchor=re.escape(_DEFERRED_FLUSH_OLD),
                mode="replace",
                text=_DEFERRED_FLUSH_NEW,
                guard=r"bc_pt_state->draft_catchup_us \+=",
                rationale="After 1348, attribute deferred snapshot catch-up wall to the prompt owning that snapshot.",
                expect_matches=1,
                max_span_lines=12,
            ),
            Edit(
                id="mtp-prompt-timing-deferred-nextn",
                anchor=re.escape(_DEFERRED_NEXTN_OLD),
                mode="replace",
                text=_DEFERRED_NEXTN_NEW,
                guard=r"bc_pt_state->target_block_us \+=",
                rationale="After 1348, measure the blocking NextN snapshot and count deferred prompt chunks.",
                expect_matches=1,
                max_span_lines=20,
            ),
            Edit(
                id="mtp-prompt-timing-wrapper",
                anchor=re.escape(_WRAPPER_ANCHOR),
                mode="insert_before",
                text=_WRAPPER_TEXT,
                guard=r"void common_speculative_prefill_begin\(common_speculative \* spec, llama_seq_id seq_id\) \{",
                rationale="Public timing wrappers next to common_speculative_begin.",
                expect_matches=1,
                max_span_lines=2,
            ),
        ),
    ),
    FilePatch(
        path="tools/server/server-context.cpp",
        description="1346: mark prompt start and target submit/return for timing",
        language="none",
        edits=(
            Edit(
                id="mtp-prompt-timing-server-prefill",
                anchor=re.escape(_SERVER_ANCHOR),
                mode="insert_after",
                text=_SERVER_TEXT,
                guard=r"common_speculative_prefill_begin\(spec\.get\(\), slot\.id\);",
                rationale="Resolved prompt-start point after cache/checkpoint decisions.",
                expect_matches=1,
                max_span_lines=2,
            ),
            Edit(
                id="mtp-prompt-target-process-gap-begin",
                anchor=re.escape(_TARGET_PROCESS_CALL),
                mode="insert_before",
                text=_TARGET_PROCESS_BEGIN_TEXT,
                guard=r"common_speculative_target_process_begin\(spec\.get\(\), batch\.view\);",
                rationale="Post-1348 target submit seam; begin immediately before target submission regardless of 1322 code around it.",
                expect_matches=1,
                max_span_lines=2,
            ),
            Edit(
                id="mtp-prompt-target-process-gap-end",
                anchor=re.escape(_TARGET_PROCESS_CALL),
                mode="insert_after",
                text=_TARGET_PROCESS_END_TEXT,
                guard=r"common_speculative_target_process_end\(spec\.get\(\), batch\.view\);",
                rationale="Record target-submit return before 1322 ahead work so host gap includes all work until the next target submission.",
                expect_matches=1,
                max_span_lines=2,
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
)
