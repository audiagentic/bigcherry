"""1343: reserve the scheduler again when the NextN (MTP) outputs are switched on or off (QFP32).

llama_context reserves its worst-case graph in the constructor, with cparams.embeddings_nextn = false. The MTP driver
(common_speculative_impl_draft_mtp) switches the NextN outputs on afterwards through llama_set_embeddings_nextn, and
the setter only assigns the two fields: the graphs that then run have another shape (other node count) than the one
that was reserved. The first of them is planned from its own, current sizes, and from then on every input that is sized
by the filled context no longer fits that plan when it grows - one allocator re-plan (with a synchronize) per prefill
chunk for the whole prompt. Measured on Flash-Next at ctx 245760 with 1340's counters: the first graph that runs has
"no reserve plan nodes=7204", then 11 re-plans per device for a 2K fill and 246 for a 98K fill.

By default (BIGCHERRY_MTP_RERESERVE=0 disables) the setter marks the scheduler for a reserve when the mode really changes, as
set_embeddings_layer_inp already does for its graph-visible change. The next process() then reserves the worst-case
graph of the shape that will run, once, and later graphs bind into that plan. Allocation only: the output must be
identical.
"""

from __future__ import annotations

import re

from bigcherry.patcher import Edit, EnvDoc, FilePatch

GROUP = "rdna-boosts"
STATE = "validated"

_OLD = """\
void llama_context::set_embeddings_nextn(bool value, bool masked) {
    LLAMA_LOG_DEBUG("%s: value = %d, masked = %d\\n", __func__, value, masked);

    cparams.embeddings_nextn        = value;
    cparams.embeddings_nextn_masked = masked;
}
"""
_NEW = """\
void llama_context::set_embeddings_nextn(bool value, bool masked) {
    LLAMA_LOG_DEBUG("%s: value = %d, masked = %d\\n", __func__, value, masked);

    // bigcherry 1343 (QFP32): the NextN outputs change the graph's shape, so the worst-case graph that was reserved
    // before the switch is not the one that runs. Ask for a reserve when the mode really changes (the next process()
    // does it, once), as set_embeddings_layer_inp does. On by default, BIGCHERRY_MTP_RERESERVE=0 disables.
    static const bool bigcherry_mtp_rereserve = [] {
        const char * s = std::getenv("BIGCHERRY_MTP_RERESERVE");
        return s == nullptr || std::atoi(s) != 0;
    }();
    if (bigcherry_mtp_rereserve && (cparams.embeddings_nextn != value || cparams.embeddings_nextn_masked != masked)) {
        sched_need_reserve = true;
        if (std::getenv("BIGCHERRY_PATCH_TRACE") != nullptr) {
            fprintf(stderr, "BIGCHERRY_PATCH_HIT patch=1343_mtp_nextn_rereserve value=%d masked=%d\\n", value ? 1 : 0, masked ? 1 : 0);
        }
    }

    cparams.embeddings_nextn        = value;
    cparams.embeddings_nextn_masked = masked;
}
"""

PATCHES = [
    FilePatch(
        path="src/llama-context.cpp",
        description="1343: a NextN output mode change asks for a scheduler reserve (on by default)",
        language="none",
        edits=(
            Edit(
                id="mtp-rereserve-include",
                anchor=re.escape("#include <cinttypes>\n"),
                mode="insert_before",
                text="#include <cstdio>   // bigcherry 1343: fprintf\n#include <cstdlib>  // bigcherry 1343: std::getenv / std::atoi\n",
                guard=r"#include <cstdlib>  // bigcherry 1343",
                rationale="First standard-library include of llama-context.cpp.",
                expect_matches=1,
                max_span_lines=2,
            ),
            Edit(
                id="mtp-rereserve-setter",
                anchor=re.escape(_OLD),
                mode="replace",
                text=_NEW,
                guard=r"bigcherry 1343 \(QFP32\): the NextN outputs change the graph's shape",
                rationale="The whole NextN setter: it assigns the two fields and nothing else at this pin.",
                expect_matches=1,
                max_span_lines=7,
            ),
        ),
    ),
]

ENV_DOCS = (
    EnvDoc("BIGCHERRY_MTP_RERESERVE", "0|1", "1 (on)",
           "reserve the scheduler's worst-case graph again when the NextN (MTP) outputs are switched on or off, so "
           "later graphs bind into that plan instead of re-planning per prefill chunk; 0 disables"),
)
