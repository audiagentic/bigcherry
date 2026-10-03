"""1293: ggml_backend_sched synchronizes a split backend once before copying user inputs, not once per input.

When the split backend has no events (the meta backend used by -sm tensor has none), the scheduler called
ggml_backend_synchronize(split_backend) before EVERY user input copy of every split, so the previous use of
the input buffers is finished before they are overwritten. The meta backend's synchronize is a sync of every
rank, and llm_graph sets ~18 inputs per decode, so a tensor-split MTP decode step did ~54 rank syncs per
llama_decode call from this site alone (LD_PRELOAD hipStreamSynchronize call-site trace, Flash-Next, ~485
syncs per MTP step in total).

One synchronize per split is enough: after it the split backend has finished all earlier work, and the only
work enqueued afterwards within this input loop is copies into other input tensors of the same split, which
cannot touch a user input copied later. The per-input event path (backends with events) is unchanged.
"""

import re as _re

from bigcherry.patcher import Edit, FilePatch

GROUP = "core"
STATE = "untested"

SCHED = FilePatch(
    path="ggml/src/ggml-backend.cpp",
    language="none",
    description="Sync the split backend once per split before user-input copies (event-less backends).",
    edits=(
        Edit(
            id="sched-input-sync-flag",
            anchor=_re.escape("        // copy the input tensors to the split backend\n"),
            text=(
                "        // copy the input tensors to the split backend\n"
                "        // BigCherry 1293: without events, one synchronize before the first user input is enough\n"
                "        bool bc_user_inputs_synced = false;\n"
            ),
            mode="replace",
            guard=r"bool bc_user_inputs_synced = false;",
            expect_matches=1,
            rationale="The comment heads the per-split input copy loop in ggml_backend_sched_compute_splits.",
        ),
        Edit(
            id="sched-input-sync-once",
            anchor=_re.escape(
                "                // inputs from the user must be copied immediately to prevent the user overwriting the data before the copy is done\n"
                "                if (sched->events[split_backend_id][sched->cur_copy] != NULL) {\n"
                "                    ggml_backend_event_synchronize(sched->events[split_backend_id][sched->cur_copy]);\n"
                "                } else {\n"
                "                    ggml_backend_synchronize(split_backend);\n"
                "                }\n"
            ),
            text=(
                "                // inputs from the user must be copied immediately to prevent the user overwriting the data before the copy is done\n"
                "                if (sched->events[split_backend_id][sched->cur_copy] != NULL) {\n"
                "                    ggml_backend_event_synchronize(sched->events[split_backend_id][sched->cur_copy]);\n"
                "                } else if (!bc_user_inputs_synced) {\n"
                "                    ggml_backend_synchronize(split_backend);\n"
                "                    bc_user_inputs_synced = true;\n"
                "                }\n"
            ),
            mode="replace",
            guard=r"\} else if \(!bc_user_inputs_synced\) \{",
            expect_matches=1,
            rationale="Only the user-input branch of the copy loop has this comment + event/sync pair.",
        ),
    ),
)

PATCHES = [SCHED]
