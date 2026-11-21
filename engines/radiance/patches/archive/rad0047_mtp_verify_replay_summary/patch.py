"""Do not record wide Qwen4-Exp verify without a proven full-KV summary."""

import re

from bigcherry.patcher import Edit, FilePatch

OLD = """    uint64_t sel = 0;
    if (used_max_h && batch->kv && m.kv_full && m.qsacfg.budget > 0 && m.qsacfg.ratio > 0)
        for (int i = 0; i < batch->n_kv_groups; ++i)
            if (batch->kv[i].group == m.kv_full) {
                if (used_max_h[i] > 0 &&
                    used_max_h[i] <= m.qsacfg.budget + m.qsacfg.ratio - 1) sel = 2;
                break;
            }"""

NEW = """    uint64_t sel = 0;
    int full_group = -1;
    if (batch->kv && m.kv_full)
        for (int i = 0; i < batch->n_kv_groups; ++i)
            if (batch->kv[i].group == m.kv_full) {
                full_group = i;
                break;
            }
    const bool has_full_summary = used_max_h && full_group >= 0 &&
                                  used_max_h[full_group] > 0;
    /* A wide dense verify can select the prefill attention row. Its split count
     * depends on the proven host KV summary, while issue-time fallback keeps the
     * model bound for an unknown summary. Do not let those graphs share a key. */
    if (batch->phase == RAD_PHASE_DECODE && batch->draft_pass == 0 &&
        batch->n_spec > 0 && batch->n_tok > batch->n_seq &&
        m.kv_full && !has_full_summary)
        return 0;
    if (has_full_summary && m.qsacfg.budget > 0 && m.qsacfg.ratio > 0 &&
        used_max_h[full_group] <= m.qsacfg.budget + m.qsacfg.ratio - 1)
        sel = 2;"""

PATCH = FilePatch(
    language="none",
    path="arch/qwen4exp_fp8/qwen4exp_fp8.cpp",
    description="Refuse wide speculative replay when full-KV split geometry is unknown.",
    edits=(Edit(
        id="unknown-wide-verify-summary",
        anchor=re.escape(OLD),
        mode="replace",
        text=NEW,
        guard="    int full_group = -1;",
        expect_matches=1,
        max_span_lines=9,
        rationale="Nonpositive summary uses model-bound split geometry, which can differ from the positive keyed bound under the same pass key.",
    ),),
)
