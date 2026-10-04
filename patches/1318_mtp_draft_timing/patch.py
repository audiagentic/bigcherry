"""1318 (FMTP01 / QFP08): per-call timing of the single-head MTP draft loop on the draft GPU.

1317 showed the serial fresh draft is 6.4 ms (~10K) / 8.7 ms (~80K) per round, while the 6900 is busy only ~1.1 ms per
generated token (~3.2 ms per round incl. the reseed): roughly half of drafting is host-side. With
BIGCHERRY_SPEC_TIMING=1 each draft call logs `BIGCHERRY_DRAFT_TIMING steps submit_us sync_us rest_us total_us`:
  submit_us - llama_process(ctx_dft) host time over all steps (graph build/reuse + async submit)
  sync_us   - llama_synchronize(ctx_dft) right after each submit: the draft GPU's remaining work
              (only when timing is on; the sampler's logits read would wait for it anyway, so the schedule is unchanged)
  rest_us   - everything else in the loop: sampling (logits read + top-k over the trimmed vocab), nextn hidden read,
              batch rebuild
Diagnostic only.
"""

from __future__ import annotations

import re

from bigcherry.patcher import Edit, EnvDoc, FilePatch

GROUP = "rdna-boosts"
STATE = "untested"

_START_ANCHOR = """        int i = 0;

        while (n_drafting > 0) {
            // each step decodes under a different head"""
_START_NEW = """        // bigcherry 1318: draft-loop timing (BIGCHERRY_SPEC_TIMING=1)
        static const bool bc_dt_on = getenv("BIGCHERRY_SPEC_TIMING") != nullptr && atoi(getenv("BIGCHERRY_SPEC_TIMING")) != 0;
        const int64_t bc_dt_t0 = bc_dt_on ? ggml_time_us() : 0;
        int64_t bc_dt_submit = 0, bc_dt_sync = 0;

        int i = 0;

        while (n_drafting > 0) {
            // each step decodes under a different head"""

_PROC_OLD = "            int ret = llama_process(ctx_dft, LLAMA_PROCESS_TYPE_DECODE, batch.get());\n"
_PROC_NEW = """            const int64_t bc_dt_s0 = bc_dt_on ? ggml_time_us() : 0;  // bigcherry 1318
            int ret = llama_process(ctx_dft, LLAMA_PROCESS_TYPE_DECODE, batch.get());
            if (bc_dt_on) {
                const int64_t bc_dt_s1 = ggml_time_us();
                llama_synchronize(ctx_dft);
                bc_dt_submit += bc_dt_s1 - bc_dt_s0;
                bc_dt_sync   += ggml_time_us() - bc_dt_s1;
            }
"""

_END_OLD = """            if (batch.size() == 0) {
                break;
            }

            ++i;
        }

        if (chain_heads) {
            llama_set_nextn_layer_offset(ctx_dft, 0); // restore default for non-draft decodes
        }
"""
_END_NEW = """            if (batch.size() == 0) {
                break;
            }

            ++i;
        }

        if (bc_dt_on) {  // bigcherry 1318
            const int64_t bc_dt_total = ggml_time_us() - bc_dt_t0;
            LOG_WRN("BIGCHERRY_DRAFT_TIMING steps=%d submit_us=%lld sync_us=%lld rest_us=%lld total_us=%lld\\n", i + 1,
                    (long long) bc_dt_submit, (long long) bc_dt_sync,
                    (long long) (bc_dt_total - bc_dt_submit - bc_dt_sync), (long long) bc_dt_total);
        }

        if (chain_heads) {
            llama_set_nextn_layer_offset(ctx_dft, 0); // restore default for non-draft decodes
        }
"""

PATCHES = [
    FilePatch(
        path="common/speculative.cpp",
        description="1318: BIGCHERRY_SPEC_TIMING per-call MTP draft-loop timing",
        language="none",
        edits=(
            Edit(id="draft-timing-start", anchor=re.escape(_START_ANCHOR), mode="replace", text=_START_NEW,
                 guard=r"bigcherry 1318: draft-loop timing", rationale="Single-head MTP draft loop entry.",
                 expect_matches=1, max_span_lines=5),
            Edit(id="draft-timing-process", anchor=re.escape(_PROC_OLD), mode="replace", text=_PROC_NEW,
                 guard=r"bc_dt_submit \+= bc_dt_s1 - bc_dt_s0;", rationale="Per-step draft decode submit.",
                 expect_matches=1, max_span_lines=2),
            Edit(id="draft-timing-end", anchor=re.escape(_END_OLD), mode="replace", text=_END_NEW,
                 guard=r"BIGCHERRY_DRAFT_TIMING steps=", rationale="Draft loop exit, before the chain-heads restore.",
                 expect_matches=1, max_span_lines=11),
        ),
    ),
]

ENV_DOCS = (
    EnvDoc('BIGCHERRY_DRAFT_TIMING', '0|1', '0',
           'diagnostic: MTP draft step timing (with BIGCHERRY_SPEC_TIMING)'),
)
