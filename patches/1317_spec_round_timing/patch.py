"""1317 (FMTP01 Gate 0): per-round speculative timing in the server.

With BIGCHERRY_SPEC_TIMING=1 the server times the serial phases of each speculative round with ggml_time_us() and
logs one `BIGCHERRY_SPEC_TIMING` line per accepted round:
  draft_us   - common_speculative_draft (fresh MTP front on the draft GPU)
  submit_us  - llama_process(ctx_tgt) host time (graph build + async submit)
  sync_us    - llama_synchronize(ctx_tgt) wait (remaining target device work + output copies)
  process_us - common_speculative_process (draft-context catch-up/reseed on the verified batch)
  sample_us  - target sample-and-accept over the verify rows (CPU greedy + logits reads)
  n_draft / n_acc
The sync is not moved and no synchronization is added, so the schedule is unchanged. submit_us + sync_us is the
target critical path; a large sync_us with small submit_us shows the async window FMTP03 needs (prove kernel
occupancy separately with rocprof). Diagnostic; single-slot interpretation (accumulates across slots per round).
"""

from __future__ import annotations

import re

from bigcherry.patcher import Edit, EnvDoc, FilePatch

GROUP = "rdna-boosts"
STATE = "untested"

_HELPER_ANCHOR = "// note: this is not a member of server_slot because we want to run it inside yield_to_queue\n"
_HELPER = r"""// bigcherry 1317: per-round speculative timing (BIGCHERRY_SPEC_TIMING=1)
struct bc_spec_timing {
    int64_t draft_us = 0, submit_us = 0, sync_us = 0, process_us = 0, sample_us = 0;
};
static bool bc_spec_timing_on() {
    static const bool on = getenv("BIGCHERRY_SPEC_TIMING") != nullptr && atoi(getenv("BIGCHERRY_SPEC_TIMING")) != 0;
    return on;
}
static bc_spec_timing & bc_spec_t() {
    static bc_spec_timing t;
    return t;
}

"""

_DRAFT_OLD = """            queue_tasks.yield_to_queue([&]() {
                common_speculative_draft(spec.get());
            });
"""
_DRAFT_NEW = """            queue_tasks.yield_to_queue([&]() {
                const int64_t bc_t0 = bc_spec_timing_on() ? ggml_time_us() : 0;  // bigcherry 1317
                common_speculative_draft(spec.get());
                if (bc_spec_timing_on()) {
                    bc_spec_t().draft_us += ggml_time_us() - bc_t0;
                }
            });
"""

_DECODE_OLD = """            ret = llama_process(ctx_tgt, LLAMA_PROCESS_TYPE_DECODE, batch.view.get());
            if (ret == 0 && has_output) {
                llama_synchronize(ctx_tgt);
            }
"""
_DECODE_NEW = """            const int64_t bc_t0 = bc_spec_timing_on() ? ggml_time_us() : 0;  // bigcherry 1317
            ret = llama_process(ctx_tgt, LLAMA_PROCESS_TYPE_DECODE, batch.view.get());
            const int64_t bc_t1 = bc_spec_timing_on() ? ggml_time_us() : 0;
            if (ret == 0 && has_output) {
                llama_synchronize(ctx_tgt);
            }
            if (bc_spec_timing_on()) {
                bc_spec_t().submit_us += bc_t1 - bc_t0;
                bc_spec_t().sync_us   += ggml_time_us() - bc_t1;
            }
"""

_PROCESS_OLD = """            queue_tasks.yield_to_queue([&]() {
                ok = common_speculative_process(spec.get(), batch.view);
            });
"""
_PROCESS_NEW = """            queue_tasks.yield_to_queue([&]() {
                const int64_t bc_t0 = bc_spec_timing_on() ? ggml_time_us() : 0;  // bigcherry 1317
                ok = common_speculative_process(spec.get(), batch.view);
                if (bc_spec_timing_on()) {
                    bc_spec_t().process_us += ggml_time_us() - bc_t0;
                }
            });
"""

_SAMPLE_OLD = """                const auto & synth_probs = common_speculative_get_synth_probs(spec.get());
"""
_SAMPLE_NEW = """                const auto & synth_probs = common_speculative_get_synth_probs(spec.get());
                const int64_t bc_ts0 = bc_spec_timing_on() ? ggml_time_us() : 0;  // bigcherry 1317
"""

_ACCEPT_OLD = """                common_speculative_accept(spec.get(), slot.id, accepted.size() - 1);
"""
_ACCEPT_NEW = """                common_speculative_accept(spec.get(), slot.id, accepted.size() - 1);
                if (bc_spec_timing_on()) {  // bigcherry 1317: one line per round, then reset
                    bc_spec_timing & bt = bc_spec_t();
                    bt.sample_us += ggml_time_us() - bc_ts0;
                    SLT_WRN(slot, "BIGCHERRY_SPEC_TIMING draft_us=%lld submit_us=%lld sync_us=%lld process_us=%lld sample_us=%lld n_draft=%zu n_acc=%zu\\n",
                        (long long) bt.draft_us, (long long) bt.submit_us, (long long) bt.sync_us, (long long) bt.process_us,
                        (long long) bt.sample_us, n_draft, accepted.size() - 1);
                    bc_spec_t() = bc_spec_timing();
                }
"""

PATCHES = [
    FilePatch(
        path="tools/server/server-context.cpp",
        description="1317: BIGCHERRY_SPEC_TIMING per-round speculative phase timing",
        language="none",
        edits=(
            Edit(id="spec-timing-helper", anchor=re.escape(_HELPER_ANCHOR), mode="insert_before", text=_HELPER,
                 guard=r"struct bc_spec_timing \{", rationale="File-static helpers before the yield_to_queue helper.",
                 expect_matches=1, max_span_lines=2),
            Edit(id="spec-timing-draft", anchor=re.escape(_DRAFT_OLD), mode="replace", text=_DRAFT_NEW,
                 guard=r"bc_spec_t\(\)\.draft_us \+=", rationale="Serial fresh draft.", expect_matches=1, max_span_lines=4),
            Edit(id="spec-timing-decode", anchor=re.escape(_DECODE_OLD), mode="replace", text=_DECODE_NEW,
                 guard=r"bc_spec_t\(\)\.sync_us   \+=", rationale="Target submit and explicit sync (split, not moved).",
                 expect_matches=1, max_span_lines=5),
            Edit(id="spec-timing-process", anchor=re.escape(_PROCESS_OLD), mode="replace", text=_PROCESS_NEW,
                 guard=r"bc_spec_t\(\)\.process_us \+=", rationale="Draft-context catch-up/reseed.",
                 expect_matches=1, max_span_lines=4),
            Edit(id="spec-timing-sample", anchor=re.escape(_SAMPLE_OLD), mode="replace", text=_SAMPLE_NEW,
                 guard=r"const int64_t bc_ts0 = bc_spec_timing_on\(\) \? ggml_time_us\(\) : 0;", rationale="Start of target sample-and-accept.",
                 expect_matches=1, max_span_lines=3),
            Edit(id="spec-timing-accept", anchor=re.escape(_ACCEPT_OLD), mode="replace", text=_ACCEPT_NEW,
                 guard=r"BIGCHERRY_SPEC_TIMING draft_us=", rationale="Round end: log and reset.",
                 expect_matches=1, max_span_lines=2),
        ),
    ),
]

ENV_DOCS = (
    EnvDoc('BIGCHERRY_SPEC_TIMING', '0|1', '0',
           'diagnostic: per speculative round draft/submit/sync/process/sample times'),
)
