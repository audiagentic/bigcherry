"""1346 (QFP31 chunk 1): split MTP prompt overhead before implementing prompt overlap.

BIGCHERRY_MTP_PROMPT_TIMING=1 adds a prompt-start lifecycle hook and times the existing
MTP prompt catch-up without changing its work or ordering. At prompt end it prints one line:
BIGCHERRY_MTP_PROMPT_TIMING target_nextn_ms= draft_process_ms= draft_decode_ms= chunks= tokens=

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
void common_speculative_prefill_begin(common_speculative * spec, llama_seq_id seq_id);
"""

_INCLUDE_ANCHOR = "#include <cinttypes>\n"
_INCLUDE_TEXT = """\
#include <cstdio>   // bigcherry 1346: stderr prompt timing
#include <cstdlib>  // bigcherry 1346: std::getenv / std::atoi
"""

_VIRTUAL_ANCHOR = "    virtual ~common_speculative_impl() = default;\n"
_VIRTUAL_TEXT = """\

    // bigcherry 1346 (QFP31): optional prompt-start lifecycle boundary; no-op unless an implementation uses it.
    virtual void prefill_begin(llama_seq_id /*seq_id*/) {}
"""

_STATE_ANCHOR = "    std::vector<std::vector<float>> pending_h;   // [n_seq][n_embd]\n"
_STATE_TEXT = """\

    // bigcherry 1346 (QFP31 chunk 1): diagnostic state only; no prompt work is skipped or reordered.
    struct bc_mtp_prompt_timing_state {
        bool collecting = false;
        int64_t target_nextn_us = 0;
        int64_t process_us = 0;
        int64_t draft_decode_us = 0;
        uint64_t chunks = 0;
        uint64_t tokens = 0;
    };
    bool bc_mtp_prompt_timing_on = false;
    std::vector<bc_mtp_prompt_timing_state> bc_mtp_prompt_timing;
"""

_CTOR_ANCHOR = "        pending_h.assign(n_seq, std::vector<float>(n_embd, 0.0f));\n"
_CTOR_TEXT = """\
        if (const char * value = std::getenv("BIGCHERRY_MTP_PROMPT_TIMING")) {
            bc_mtp_prompt_timing_on = std::atoi(value) != 0;
        }
        bc_mtp_prompt_timing.resize(n_seq);
"""

_MTP_BEGIN_ANCHOR = """\
    void begin(llama_seq_id seq_id, const llama_tokens & prompt) override {
        // reset here rather than per round, or two identical requests differ
        common_sampler_reset(smpls[seq_id].get());
"""
_MTP_PREFILL_TEXT = """\
    void prefill_begin(llama_seq_id seq_id) override {
        if (!bc_mtp_prompt_timing_on || seq_id < 0 || seq_id >= (llama_seq_id) bc_mtp_prompt_timing.size()) {
            return;
        }

        bc_mtp_prompt_timing[seq_id] = {};
        bc_mtp_prompt_timing[seq_id].collecting = true;
    }

"""
_MTP_BEGIN_TIMING_TEXT = """\

        if (bc_mtp_prompt_timing_on && seq_id >= 0 && seq_id < (llama_seq_id) bc_mtp_prompt_timing.size()) {
            auto & timing = bc_mtp_prompt_timing[seq_id];
            if (timing.collecting) {
                std::fprintf(stderr,
                        "BIGCHERRY_MTP_PROMPT_TIMING target_nextn_ms=%.3f draft_process_ms=%.3f draft_decode_ms=%.3f chunks=%llu tokens=%llu\\n",
                        timing.target_nextn_us / 1000.0, timing.process_us / 1000.0, timing.draft_decode_us / 1000.0,
                        (unsigned long long) timing.chunks, (unsigned long long) timing.tokens);
                timing.collecting = false;
            }
        }
"""

_PROCESS_START_OLD = """\
        const int32_t n_tokens = batch_in.size();

        // remember the first and last batch index for each sequence
"""
_PROCESS_START_NEW = """\
        const int32_t n_tokens = batch_in.size();

        // bigcherry 1346: attribute the diagnostic only when this process() call belongs to one sequence.
        bc_mtp_prompt_timing_state * bc_pt_state = nullptr;
        if (bc_mtp_prompt_timing_on) {
            llama_seq_id bc_pt_seq = -1;
            bool bc_pt_single_seq = true;
            for (int k = 0; k < n_tokens; ++k) {
                const llama_seq_id seq_id = batch_in.tokens[k].seq_id;
                if (seq_id < 0 || seq_id >= (llama_seq_id) n_seq) {
                    bc_pt_single_seq = false;
                    break;
                }
                if (bc_pt_seq < 0) {
                    bc_pt_seq = seq_id;
                } else if (bc_pt_seq != seq_id) {
                    bc_pt_single_seq = false;
                    break;
                }
            }
            if (bc_pt_single_seq && bc_pt_seq >= 0) {
                auto & timing = bc_mtp_prompt_timing[bc_pt_seq];
                if (timing.collecting) {
                    bc_pt_state = &timing;
                }
            }
        }
        const int64_t bc_pt_process_t0 = bc_pt_state ? ggml_time_us() : 0;
        int64_t bc_pt_target_nextn_us = 0;
        int64_t bc_pt_draft_decode_us = 0;

        // remember the first and last batch index for each sequence
"""

_NEXTN_OLD = "            const float * h_tgt = llama_get_embeddings_nextn(ctx_tgt);\n"
_NEXTN_NEW = """\
            const int64_t bc_pt_nextn_t0 = bc_pt_state ? ggml_time_us() : 0;  // bigcherry 1346
            const float * h_tgt = llama_get_embeddings_nextn(ctx_tgt);
            if (bc_pt_state) {
                bc_pt_target_nextn_us += ggml_time_us() - bc_pt_nextn_t0;
            }
"""

_DRAFT_DECODE_OLD = "                const int32_t rc = llama_process(ctx_dft, LLAMA_PROCESS_TYPE_DECODE, batch.get());\n"
_DRAFT_DECODE_NEW = """\
                const int64_t bc_pt_draft_t0 = bc_pt_state ? ggml_time_us() : 0;  // bigcherry 1346
                const int32_t rc = llama_process(ctx_dft, LLAMA_PROCESS_TYPE_DECODE, batch.get());
                if (bc_pt_state) {
                    bc_pt_draft_decode_us += ggml_time_us() - bc_pt_draft_t0;
                }
"""

_VERIFY_ANCHOR = """\
        for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
            if (i_batch_end[seq_id] < 0) {
                continue;
            }
"""
_VERIFY_TEXT = """\
        const int64_t bc_pt_verify_t0 = bc_pt_state ? ggml_time_us() : 0;  // bigcherry 1346: getters + host copies

"""

_PROCESS_TAIL_OLD = """\
            std::memcpy(pending_h[seq_id].data(),
                    verify_h[seq_id].data() + (size_t) (n_rows - 1) * n_embd, row_bytes);
        }

        return true;
"""
_PROCESS_TAIL_NEW = """\
            std::memcpy(pending_h[seq_id].data(),
                    verify_h[seq_id].data() + (size_t) (n_rows - 1) * n_embd, row_bytes);
        }

        if (bc_pt_state) {
            bc_pt_target_nextn_us += ggml_time_us() - bc_pt_verify_t0;
            bc_pt_state->target_nextn_us += bc_pt_target_nextn_us;
            bc_pt_state->draft_decode_us += bc_pt_draft_decode_us;
            bc_pt_state->process_us += ggml_time_us() - bc_pt_process_t0;
            bc_pt_state->chunks++;
            bc_pt_state->tokens += (uint64_t) n_tokens;
        }

        return true;
"""

_WRAPPER_ANCHOR = "void common_speculative_begin(common_speculative * spec, llama_seq_id seq_id, const llama_tokens & prompt) {\n"
_WRAPPER_TEXT = """\
void common_speculative_prefill_begin(common_speculative * spec, llama_seq_id seq_id) {
    if (spec == nullptr) {
        return;
    }

    for (auto & impl : spec->impls) {
        impl->prefill_begin(seq_id);
    }
}

"""

_SERVER_ANCHOR = "                        slot.prompt.tokens.keep_first(n_past);\n"
_SERVER_TEXT = """\

                        // bigcherry 1346 (QFP31): resolved prompt-start boundary. Chunk 1 is diagnostic-only;
                        // later overlap work can reuse this contract without guessing prompt/cache state in process().
                        common_speculative_prefill_begin(spec.get(), slot.id);
"""

PATCHES = [
    FilePatch(
        path="common/speculative.h",
        description="1346: expose a prompt-start lifecycle hook for MTP prompt diagnostics/overlap",
        language="none",
        edits=(
            Edit(
                id="mtp-prompt-prefill-api",
                anchor=re.escape(_API_ANCHOR),
                mode="insert_after",
                text=_API_TEXT,
                guard=r"void common_speculative_prefill_begin\(common_speculative \* spec, llama_seq_id seq_id\);",
                rationale="Stable public speculative-driver seam immediately before the existing generation-begin hook.",
                expect_matches=1,
                max_span_lines=2,
            ),
        ),
    ),
    FilePatch(
        path="common/speculative.cpp",
        description="1346: default-off per-prompt MTP target-NextN / catch-up timing split",
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
                guard=r"virtual void prefill_begin\(llama_seq_id /\*seq_id\*/\) \{\}",
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
                guard=r"bc_mtp_prompt_timing_state \* bc_pt_state = nullptr;",
                rationale="MTP process() token-count seam; establish single-sequence attribution and whole-process wall timer.",
                expect_matches=1,
                max_span_lines=4,
            ),
            Edit(
                id="mtp-prompt-timing-nextn",
                anchor=re.escape(_NEXTN_OLD),
                mode="replace",
                text=_NEXTN_NEW,
                guard=r"const int64_t bc_pt_nextn_t0 = bc_pt_state \? ggml_time_us\(\) : 0;",
                rationale="Exact host call that obtains the target NextN buffer and may synchronize it.",
                expect_matches=1,
                max_span_lines=2,
            ),
            Edit(
                id="mtp-prompt-timing-draft-decode",
                anchor=re.escape(_DRAFT_DECODE_OLD),
                mode="replace",
                text=_DRAFT_DECODE_NEW,
                guard=r"const int64_t bc_pt_draft_t0 = bc_pt_state \? ggml_time_us\(\) : 0;",
                rationale="Exact draft-context catch-up decode submission, summed across MTP heads.",
                expect_matches=1,
                max_span_lines=2,
            ),
            Edit(
                id="mtp-prompt-timing-verify-start",
                anchor=re.escape(_VERIFY_ANCHOR),
                mode="insert_before",
                text=_VERIFY_TEXT,
                guard=r"bc_pt_verify_t0 = bc_pt_state \? ggml_time_us\(\) : 0;",
                rationale="Target NextN row getters and host copies are contiguous immediately before the native verify loop.",
                expect_matches=1,
                max_span_lines=5,
            ),
            Edit(
                id="mtp-prompt-timing-process-end",
                anchor=re.escape(_PROCESS_TAIL_OLD),
                mode="replace",
                text=_PROCESS_TAIL_NEW,
                guard=r"bc_pt_state->target_nextn_us \+= bc_pt_target_nextn_us;",
                rationale="Native pending_h update is the end of MTP process bookkeeping; accumulate the completed chunk once.",
                expect_matches=1,
                max_span_lines=6,
            ),
            Edit(
                id="mtp-prompt-prefill-wrapper",
                anchor=re.escape(_WRAPPER_ANCHOR),
                mode="insert_before",
                text=_WRAPPER_TEXT,
                guard=r"void common_speculative_prefill_begin\(common_speculative \* spec, llama_seq_id seq_id\) \{",
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
                guard=r"common_speculative_prefill_begin\(spec\.get\(\), slot\.id\);",
                rationale="SLOT_STATE_STARTED has finalized n_past and restored any cache/checkpoint state at this line.",
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
        "diagnostic: split MTP prompt process wall into target NextN fetch/copy and draft catch-up decode time; single-slot attribution",
    ),
)
