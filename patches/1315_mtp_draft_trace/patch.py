"""1315 (QFP15): bit-exact MTP draft trace for locating run-to-run nondeterminism.

Same build, same prompt, temperature 0: greedy target output is identical across runs, but drafted/accepted counts
differ (e.g. 252 vs 255 drafted on two baseline arms). The target and the MTP draft sampler are both greedy (draft =
top-1 after top-k 10, no RNG), so some GPU result differs bitwise between runs and flips a near-tie or the p_min
cutoff. With BIGCHERRY_DRAFT_TRACE=1 the single-head MTP path logs, per draft step, the top token, its probability
as an exact hex float (%a) and an FNV-1a hash of the draft hidden row it produced; and on every accept() the
accepted count and the FNV-1a hash of the TARGET hidden row (pending_h) that seeds the next round. Diff two runs
(drop the timestamp prefix): the first differing BIGCHERRY_DRAFT_TRACE line says whether the target (accept hash)
or the draft (step p/hash) diverged first. Trace only; no behaviour change. Not for production.
"""

from __future__ import annotations

import re

from bigcherry.patcher import Edit, FilePatch

GROUP = "rdna-boosts"
STATE = "untested"

_HELPER_ANCHOR = '#define SPC_CNT(fmt, ...) LOG_CNT(""              fmt,               __VA_ARGS__)\n'
_HELPER = r"""
// bigcherry 1315: bit-exact draft trace (BIGCHERRY_DRAFT_TRACE=1)
static bool bc_draft_trace_on() {
    static const bool on = getenv("BIGCHERRY_DRAFT_TRACE") != nullptr && atoi(getenv("BIGCHERRY_DRAFT_TRACE")) != 0;
    return on;
}

static uint64_t bc_fnv1a(const void * data, size_t n) {
    const unsigned char * p = (const unsigned char *) data;
    uint64_t h = 1469598103934665603ull;
    for (size_t i = 0; i < n; ++i) {
        h = (h ^ p[i]) * 1099511628211ull;
    }
    return h;
}
"""

_STEP_ANCHOR = ("                common_sampler_sample(smpl, ctx_dft, i_last[seq_id], true);\n"
                "                const float * h_row = llama_get_embeddings_nextn_ith(ctx_dft, i_last[seq_id]);\n")
_STEP = r"""                if (bc_draft_trace_on()) {  // bigcherry 1315
                    const auto * bc_cur = common_sampler_get_candidates(smpl, true);
                    LOG_WRN("BIGCHERRY_DRAFT_TRACE step seq=%d i=%d id=%d p=%a h=%016llx\n", (int) seq_id, i,
                            bc_cur->data[0].id, bc_cur->data[0].p,
                            (unsigned long long) bc_fnv1a(h_row, (size_t) n_embd * sizeof(float)));
                }
"""

_ACCEPT_ANCHOR = "        std::memcpy(pending_h[seq_id].data(), verify_h[seq_id].data() + (size_t) i_h * n_embd, row_bytes);\n"
_ACCEPT = r"""        if (bc_draft_trace_on()) {  // bigcherry 1315
            LOG_WRN("BIGCHERRY_DRAFT_TRACE accept seq=%d n_accepted=%d target_h=%016llx\n", (int) seq_id, (int) n_accepted,
                    (unsigned long long) bc_fnv1a(pending_h[seq_id].data(), row_bytes));
        }
"""

PATCHES = [
    FilePatch(
        path="common/speculative.cpp",
        description="1315: BIGCHERRY_DRAFT_TRACE bit-exact MTP draft/accept trace",
        language="none",
        edits=(
            Edit(
                id="draft-trace-helper",
                anchor=re.escape(_HELPER_ANCHOR),
                mode="insert_after",
                text=_HELPER,
                guard=r"static uint64_t bc_fnv1a\(",
                rationale="After the SPC_* logging macros at the top of speculative.cpp.",
                expect_matches=1,
                max_span_lines=2,
            ),
            Edit(
                id="draft-trace-step",
                anchor=re.escape(_STEP_ANCHOR),
                mode="insert_after",
                text=_STEP,
                guard=r"BIGCHERRY_DRAFT_TRACE step",
                rationale="Single-head MTP draft step: candidates sampled and the draft hidden row read.",
                expect_matches=1,
                max_span_lines=3,
            ),
            Edit(
                id="draft-trace-accept",
                anchor=re.escape(_ACCEPT_ANCHOR),
                mode="insert_after",
                text=_ACCEPT,
                guard=r"BIGCHERRY_DRAFT_TRACE accept",
                rationale="MTP accept(): pending_h now holds the target hidden row that seeds the next draft round.",
                expect_matches=1,
                max_span_lines=2,
            ),
        ),
    ),
]
