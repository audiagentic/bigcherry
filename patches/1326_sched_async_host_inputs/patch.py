"""1326 (QFP16): asynchronous host->device split-input copies in the scheduler.

1325 showed ~2.6 ms per target verify round (28 inputs into the -sm tensor meta split) and ~0.4 ms per draft call
(15 inputs into the ROCm3 split) spent in ggml_backend_sched_compute_splits input handling: for every input the
scheduler synchronizes the split backend (no events without pipeline parallelism) and then copies synchronously,
because the CUDA/HIP backend's cpy_tensor_async refuses a CPU source (and the meta backend has none).

With BIGCHERRY_SCHED_ASYNC_INPUTS=1, an input whose source buffer is host-resident (not a weights buffer), contiguous,
and whose split backend implements set_tensor_async, is enqueued with ggml_backend_tensor_set_async on the split
backend instead (the meta backend fans it out to every device stream). Ordering: the copy is stream-ordered before the
split's compute and after the previous use of the same input copy on that stream, so the split-backend synchronize is
not needed; the source producer (CPU split, or the user for graph inputs) is synchronized first as before. On HIP an
async H2D copy from pageable memory has consumed the host data when the call returns, so the user may overwrite
inputs afterwards (the upstream reason for copying user inputs synchronously). Anything else takes the upstream path.
"""

from __future__ import annotations

import re

from bigcherry.patcher import Edit, EnvDoc, FilePatch

GROUP = "rdna-boosts"
STATE = "untested"

_A = ("            struct ggml_tensor * input_cpy = tensor_copy(input, split_backend_id, sched->cur_copy);\n"
      "\n"
      "            if (input->flags & GGML_TENSOR_FLAG_INPUT) {\n")
_N = ("            struct ggml_tensor * input_cpy = tensor_copy(input, split_backend_id, sched->cur_copy);\n"
      "\n"
      "            // bigcherry 1326: async host->device input copy (BIGCHERRY_SCHED_ASYNC_INPUTS=1)\n"
      "            static const bool bc_async_inputs = getenv(\"BIGCHERRY_SCHED_ASYNC_INPUTS\") != nullptr &&\n"
      "                                                atoi(getenv(\"BIGCHERRY_SCHED_ASYNC_INPUTS\")) != 0;\n"
      "            if (bc_async_inputs && split_backend->iface.set_tensor_async != nullptr && input->buffer != nullptr &&\n"
      "                    ggml_backend_buffer_is_host(input->buffer) &&\n"
      "                    ggml_backend_buffer_get_usage(input->buffer) != GGML_BACKEND_BUFFER_USAGE_WEIGHTS &&\n"
      "                    ggml_is_contiguous(input) && ggml_is_contiguous(input_cpy) &&\n"
      "                    ggml_nbytes(input) == ggml_nbytes(input_cpy) &&\n"
      "                    // decode-sized inputs only: staging a large prefill input (KQ mask ~10 MB per ubatch) through\n"
      "                    // pageable memory costs more than upstream's synchronous copy (v5c: prefill -2%)\n"
      "                    ggml_nbytes(input) <= ((size_t) 4 << 20)) {\n"
      "                ggml_backend_synchronize(input_backend);  // the producer (CPU split) has finished writing\n"
      "                // Lifetime (review req_3b37d17e61374a6a): the source may be a pinned host buffer, whose DMA is truly\n"
      "                // asynchronous, and the caller rewrites graph inputs on the next set_inputs (e.g. the next prefill\n"
      "                // ubatch). Stage through scheduler-owned pageable memory: the caller's buffer is free as soon as the\n"
      "                // memcpy returns, and a pageable H2D async copy consumes the staging data before it returns on HIP.\n"
      "                static thread_local std::unordered_map<const ggml_tensor *, std::vector<uint8_t>> bc_staging;\n"
      "                std::vector<uint8_t> & bc_stage = bc_staging[input_cpy];\n"
      "                bc_stage.resize(ggml_nbytes(input));\n"
      "                memcpy(bc_stage.data(), input->data, ggml_nbytes(input));\n"
      "                ggml_backend_tensor_set_async(split_backend, input_cpy, bc_stage.data(), 0, ggml_nbytes(input));\n"
      "                continue;\n"
      "            }\n"
      "\n"
      "            if (input->flags & GGML_TENSOR_FLAG_INPUT) {\n")

_META_A = """static void ggml_backend_meta_set_tensor_async(ggml_backend_t backend, ggml_tensor * tensor, const void * data, size_t offset, size_t size) {
    const size_t n_backends = ggml_backend_meta_n_backends(backend);
    GGML_ASSERT(offset == 0);
    GGML_ASSERT(ggml_is_contiguous(tensor));

    const ggml_backend_meta_split_state split_state = ggml_backend_meta_get_split_state(tensor, /*assume_sync =*/ false);
    GGML_ASSERT(split_state.n_segments == 1);
    GGML_ASSERT(split_state.nr[0]      == 1);
"""

_META_N = """static void ggml_backend_meta_set_tensor_async(ggml_backend_t backend, ggml_tensor * tensor, const void * data, size_t offset, size_t size) {
    const size_t n_backends = ggml_backend_meta_n_backends(backend);
    GGML_ASSERT(offset == 0);
    GGML_ASSERT(ggml_is_contiguous(tensor));

    const ggml_backend_meta_split_state split_state = ggml_backend_meta_get_split_state(tensor, /*assume_sync =*/ false);
    // bigcherry 1326: states the async splice cannot express (multi-segment, nr != 1, other axes) take the
    // synchronous buffer path, which handles them, instead of aborting (1326 sends scheduler inputs here).
    if (split_state.n_segments != 1 || split_state.nr[0] != 1 || !(split_state.axis == GGML_BACKEND_SPLIT_AXIS_0 ||
            split_state.axis == GGML_BACKEND_SPLIT_AXIS_1 || split_state.axis == GGML_BACKEND_SPLIT_AXIS_2 ||
            split_state.axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED)) {
        ggml_backend_synchronize(backend);
        ggml_backend_tensor_set(tensor, data, offset, size);
        return;
    }
"""

PATCHES = [
    FilePatch(
        path="ggml/src/ggml-backend.cpp",
        description="1326: async host->device split-input copies (BIGCHERRY_SCHED_ASYNC_INPUTS=1)",
        language="none",
        edits=(
            Edit(id="sched-async-inputs", anchor=re.escape(_A), mode="replace", text=_N,
                 guard=r"bigcherry 1326: async host->device input copy",
                 rationale="Top of the per-input copy loop in ggml_backend_sched_compute_splits, before the upstream "
                           "user-input and inter-split branches.",
                 expect_matches=1, max_span_lines=4),
        ),
    ),
    FilePatch(
        path="ggml/src/ggml-backend-meta.cpp",
        description="1326: meta set_tensor_async falls back to the synchronous buffer path for unsupported split states",
        language="none",
        edits=(
            Edit(id="meta-set-async-fallback", anchor=re.escape(_META_A), mode="replace", text=_META_N,
                 guard=r"bigcherry 1326: states the async splice cannot express",
                 rationale="ggml_backend_meta_set_tensor_async (get_tensor_async has the same prologue but a different "
                           "splice comment, so the anchor including the set comment is unique).",
                 expect_matches=1, max_span_lines=9),
        ),
    ),
]

ENV_DOCS = (
    EnvDoc('BIGCHERRY_SCHED_ASYNC_INPUTS', '0|1', '0',
           'scheduler stages small host inputs asynchronously (capped 4 MiB)'),
)
