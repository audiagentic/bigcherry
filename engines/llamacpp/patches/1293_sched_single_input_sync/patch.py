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

b11474 extracted the per-input copy into ggml_backend_sched_copy_input. Keep the once-per-split state in
ggml_backend_sched_compute_splits and pass it into the helper so the original lifetime remains unchanged.
"""

from __future__ import annotations

import re as _re

from bigcherry.patcher import Edit, FilePatch

GROUP = "core"
STATE = "untested"

_A_FLAG = r"""        // copy the input tensors to the split backend
        // the weights in host memory are copied last, so that the copy callback can read the other inputs of the split
        for (int input_id = 0; input_id < split->n_inputs; input_id++) {
            if (!ggml_backend_sched_is_host_weight(split->inputs[input_id])) {
                ggml_backend_sched_copy_input(sched, split, split->inputs[input_id]);
            }
        }
        for (int input_id = 0; input_id < split->n_inputs; input_id++) {
            if (ggml_backend_sched_is_host_weight(split->inputs[input_id])) {
                ggml_backend_sched_copy_input(sched, split, split->inputs[input_id]);
            }
        }
"""

_N_FLAG = r"""        // copy the input tensors to the split backend
        // BigCherry 1293: without events, one synchronize before the first user input is enough.
        bool bc_user_inputs_synced = false;
        // the weights in host memory are copied last, so that the copy callback can read the other inputs of the split
        for (int input_id = 0; input_id < split->n_inputs; input_id++) {
            if (!ggml_backend_sched_is_host_weight(split->inputs[input_id])) {
                ggml_backend_sched_copy_input(sched, split, split->inputs[input_id], &bc_user_inputs_synced);
            }
        }
        for (int input_id = 0; input_id < split->n_inputs; input_id++) {
            if (ggml_backend_sched_is_host_weight(split->inputs[input_id])) {
                ggml_backend_sched_copy_input(sched, split, split->inputs[input_id], &bc_user_inputs_synced);
            }
        }
"""

_A_ONCE = r"""static void ggml_backend_sched_copy_input(ggml_backend_sched_t sched, struct ggml_backend_sched_split * split, struct ggml_tensor * input) {
    const int split_backend_id = split->backend_id;
    ggml_backend_t split_backend = sched->backends[split_backend_id];
    ggml_backend_t input_backend = ggml_backend_sched_get_tensor_backend(sched, input);
    struct ggml_tensor * input_cpy = tensor_copy(input, split_backend_id, sched->cur_copy);

    if (input->flags & GGML_TENSOR_FLAG_INPUT) {
        // inputs from the user must be copied immediately to prevent the user overwriting the data before the copy is done
        if (sched->events[split_backend_id][sched->cur_copy] != NULL) {
            ggml_backend_event_synchronize(sched->events[split_backend_id][sched->cur_copy]);
        } else {
            ggml_backend_synchronize(split_backend);
        }
        ggml_backend_tensor_copy(input, input_cpy);
        return;
    }
"""

_N_ONCE = r"""static void ggml_backend_sched_copy_input(ggml_backend_sched_t sched, struct ggml_backend_sched_split * split, struct ggml_tensor * input, bool * user_inputs_synced) {
    const int split_backend_id = split->backend_id;
    ggml_backend_t split_backend = sched->backends[split_backend_id];
    ggml_backend_t input_backend = ggml_backend_sched_get_tensor_backend(sched, input);
    struct ggml_tensor * input_cpy = tensor_copy(input, split_backend_id, sched->cur_copy);

    if (input->flags & GGML_TENSOR_FLAG_INPUT) {
        // inputs from the user must be copied immediately to prevent the user overwriting the data before the copy is done
        if (sched->events[split_backend_id][sched->cur_copy] != NULL) {
            ggml_backend_event_synchronize(sched->events[split_backend_id][sched->cur_copy]);
        } else if (!*user_inputs_synced) {
            ggml_backend_synchronize(split_backend);
            *user_inputs_synced = true;
        }
        ggml_backend_tensor_copy(input, input_cpy);
        return;
    }
"""

SCHED = FilePatch(
    path="ggml/src/ggml-backend.cpp",
    language="none",
    description="Sync the split backend once per split before user-input copies (event-less backends).",
    edits=(
        Edit(
            id="sched-input-sync-flag",
            anchor=_re.escape(_A_FLAG),
            text=_N_FLAG,
            mode="replace",
            guard=r"ggml_backend_sched_copy_input\(sched, split, split->inputs\[input_id\], &bc_user_inputs_synced\)",
            expect_matches=1,
            rationale="b11474 moved per-input copying to a helper; retain one state bit in the per-split loop and pass it to both helper call sites.",
            max_span_lines=14,
        ),
        Edit(
            id="sched-input-sync-once",
            anchor=_re.escape(_A_ONCE),
            text=_N_ONCE,
            mode="replace",
            guard=r"struct ggml_tensor \* input, bool \* user_inputs_synced\)",
            expect_matches=1,
            rationale="The extracted helper now owns the user-input event/synchronize branch, so gate only its no-event synchronize with the per-split state.",
            max_span_lines=18,
        ),
    ),
)

PATCHES = [SCHED]
