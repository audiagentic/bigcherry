"""1336 (MET01): backport of upstream #29943 - selective MoE expert copying moves from the scheduler to a copy callback.

Upstream PR #29943 (not merged at the b11402 pin). The scheduler's compute loop contained the MUL_MAT_ID special case
that copies only the used experts of a host-resident weight into a GPU split. The PR adds a public
ggml_backend_sched_copy_callback, has the scheduler copy host weights last (so the callback can read the split's
other inputs, e.g. the expert ids) and moves the selection to llama_context::sched_copy_experts. This is the seam
MET01 requires before any expert cache (#29887): policy lives in user code, not in ggml-backend.cpp.

Form: the callback API, the two-pass ordering and sched_copy_experts are upstream's. The scheduler loop is edited in
place (a two-pass loop and the callback call replacing the embedded block) instead of upstream's extraction into
ggml_backend_sched_copy_input, so that it composes with 1326, which edits the same loop.

BigCherry additions, all inside sched_copy_experts:
  - BIGCHERRY_MOE_COPY=0: observation-only control - the callback counts and returns false, the scheduler copies every
    host weight whole (MET01 step 3: semantic transparency of the seam);
  - BIGCHERRY_MOE_COPY_DENSE_PCT (default 90): when at least that share of the experts is used (large prefill
    batches touch ~496 of 512), one whole copy replaces hundreds of range copies; 0 disables;
  - counters printed at exit under BIGCHERRY_PATCH_TRACE (calls, bytes, experts used / total, dense shortcuts).

Superseded when the pin reaches a release containing #29943.
"""
import re as _re

from bigcherry.patcher import Edit, EnvDoc, FilePatch

GROUP = "core"
STATE = "untested"

_A_H_TYPE = (
    '    typedef bool (*ggml_backend_sched_eval_callback)(struct ggml_tensor * t, bool ask, void * user_data);\n'
)

_N_H_TYPE = (
    '    typedef bool (*ggml_backend_sched_eval_callback)(struct ggml_tensor * t, bool ask, void * user_data);\n'
    '\n'
    '    // BigCherry 1336 (upstream #29943): callback while copying input weights for a graph split\n'
    '    // the callback is called for weight inputs in host memory after all other inputs have been copied\n'
    '    // if the callback returns false the scheduler copies the entire input\n'
    '    // `src` is the tensor in the previous split\n'
    '    // `dst` is the copy of `src` in the split\n'
    '    // `graph` is the compute graph of the split\n'
    '    typedef bool (*ggml_backend_sched_copy_callback)(ggml_backend_t backend, const struct ggml_tensor * src, struct ggml_tensor * dst, struct ggml_cgraph * graph, void * user_data);\n'
)

_A_H_SET = (
    '    GGML_API void                 ggml_backend_sched_set_eval_callback(ggml_backend_sched_t sched, ggml_backend_sched_eval_callback callback, void * user_data);\n'
)

_N_H_SET = (
    '    GGML_API void                 ggml_backend_sched_set_eval_callback(ggml_backend_sched_t sched, ggml_backend_sched_eval_callback callback, void * user_data);\n'
    '\n'
    '    // Set a callback to be called when inputs weights are being copied\n'
    '    GGML_API void                 ggml_backend_sched_set_copy_callback(ggml_backend_sched_t sched, ggml_backend_sched_copy_callback callback, void * user_data);\n'
)

_A_BE_FIELDS = (
    '    ggml_backend_sched_eval_callback callback_eval;\n'
    '    void * callback_eval_user_data;\n'
)

_N_BE_FIELDS = (
    '    ggml_backend_sched_eval_callback callback_eval;\n'
    '    void * callback_eval_user_data;\n'
    '\n'
    '    ggml_backend_sched_copy_callback callback_copy;  // BigCherry 1336 (upstream #29943)\n'
    '    void * callback_copy_user_data;\n'
)

_A_BE_HELPER = (
    'static enum ggml_status ggml_backend_sched_compute_splits(ggml_backend_sched_t sched) {\n'
)

_N_BE_HELPER = (
    '// BigCherry 1336 (upstream #29943)\n'
    'static bool ggml_backend_sched_is_host_weight(const struct ggml_tensor * t) {\n'
    '    return t->buffer != NULL &&\n'
    '        ggml_backend_buffer_get_usage(t->buffer) == GGML_BACKEND_BUFFER_USAGE_WEIGHTS &&\n'
    '        ggml_backend_buffer_is_host(t->buffer);\n'
    '}\n'
    '\n'
    'static enum ggml_status ggml_backend_sched_compute_splits(ggml_backend_sched_t sched) {\n'
)

_A_BE_STATE = (
    '    ggml_tensor * prev_ids_tensor = nullptr;\n'
    '    std::vector<int32_t> ids;\n'
    '    std::vector<ggml_bitset_t> used_ids;\n'
    '\n'
)

_N_BE_STATE = (
    "    // BigCherry 1336 (upstream #29943): the expert selection state lives in the copy callback's owner\n"
    '\n'
)

_A_BE_LOOP = (
    '        for (int input_id = 0; input_id < split->n_inputs; input_id++) {\n'
    '            ggml_backend_t input_backend = ggml_backend_sched_get_tensor_backend(sched, split->inputs[input_id]);\n'
    '            struct ggml_tensor * input = split->inputs[input_id];\n'
)

_N_BE_LOOP = (
    '        // BigCherry 1336 (upstream #29943): the weights in host memory are copied last, so that the copy callback\n'
    '        // can read the other inputs of the split\n'
    '        for (int bc_copy_pass = 0; bc_copy_pass < 2; bc_copy_pass++)\n'
    '        for (int input_id = 0; input_id < split->n_inputs; input_id++) {\n'
    '            if (ggml_backend_sched_is_host_weight(split->inputs[input_id]) != (bc_copy_pass == 1)) {\n'
    '                continue;\n'
    '            }\n'
    '            ggml_backend_t input_backend = ggml_backend_sched_get_tensor_backend(sched, split->inputs[input_id]);\n'
    '            struct ggml_tensor * input = split->inputs[input_id];\n'
)

_A_BE_BLOCK = (
    '                // when offloading MoE weights, we can reduce the amount of data copied by copying only the experts that are used\n'
    '                ggml_tensor * node = split->graph.nodes[0];\n'
    '                if (split->graph.n_nodes > 0 &&\n'
    '                    ggml_backend_buffer_get_usage(input->buffer) == GGML_BACKEND_BUFFER_USAGE_WEIGHTS &&\n'
    '                    ggml_backend_buffer_is_host(input->buffer) && (\n'
    '                    (node->src[0] == input_cpy && node->op == GGML_OP_MUL_MAT_ID)\n'
    '                    //|| (node->src[1] == input_cpy && node->op == GGML_OP_ADD_ID) /* GGML_OP_ADD_ID weights are small and not worth splitting */\n'
    '                    )) {\n'
    '\n'
    '                    const int64_t n_expert   = node->op == GGML_OP_MUL_MAT_ID ? input->ne[2] : input->ne[1];\n'
    '                    const size_t expert_size = node->op == GGML_OP_MUL_MAT_ID ? input->nb[2] : input->nb[1];\n'
    '\n'
    '                    ggml_backend_synchronize(input_backend);\n'
    '\n'
    '                    // get the ids\n'
    '                    ggml_tensor * ids_tensor = node->src[2];\n'
    '                    ggml_backend_t ids_backend = split_backend;\n'
    '\n'
    '                    if (ggml_nelements(ids_tensor) == 0) {\n'
    '                        continue;\n'
    '                    }\n'
    '\n'
    '                    // if the ids tensor is also an input of the split, it may not have been copied yet to the split backend\n'
    '                    // in that case, we use the original ids tensor\n'
    '                    for (int i = input_id + 1; i < split->n_inputs; i++) {\n'
    '                        if (ids_tensor == tensor_copy(split->inputs[i], split_backend_id, sched->cur_copy)) {\n'
    '                            ids_tensor = split->inputs[i];\n'
    '                            ids_backend = ggml_backend_sched_get_tensor_backend(sched, split->inputs[i]);\n'
    '                            break;\n'
    '                        }\n'
    '                    }\n'
    '\n'
    '                    if (ids_tensor != prev_ids_tensor) {\n'
    '                        ids.resize(ggml_nbytes(ids_tensor) / sizeof(int32_t));\n'
    '                        ggml_backend_tensor_get_async(ids_backend, ids_tensor, ids.data(), 0, ggml_nbytes(ids_tensor));\n'
    '                        ggml_backend_synchronize(ids_backend);\n'
    '\n'
    '                        // find the used experts\n'
    '                        used_ids.clear();\n'
    '                        used_ids.resize(ggml_bitset_size(n_expert));\n'
    '                        for (int64_t i1 = 0; i1 < ids_tensor->ne[1]; i1++) {\n'
    '                            for (int64_t i0 = 0; i0 < ids_tensor->ne[0]; i0++) {\n'
    '                                int32_t id = ids[i1 * ids_tensor->nb[1]/sizeof(int32_t) + i0 * ids_tensor->nb[0]/sizeof(int32_t)];\n'
    '                                GGML_ASSERT(id >= 0 && id < n_expert);\n'
    '                                ggml_bitset_set(used_ids.data(), id);\n'
    '                            }\n'
    '                        }\n'
    '\n'
    '                        prev_ids_tensor = ids_tensor;\n'
    '                    }\n'
    '\n'
    '                    // group consecutive experts and copy them together\n'
    '                    auto copy_experts = [&](int32_t first_id, int32_t last_id) {\n'
    '                        const size_t expert_offset = first_id * expert_size;\n'
    '                        const size_t expert_size_copy =  (last_id - first_id + 1) * expert_size;\n'
    '                        const size_t padding = std::min<size_t>(expert_size, 512);\n'
    '                        const size_t padding_end = last_id < n_expert - 1 ? padding : 0;\n'
    '\n'
    '                        ggml_backend_tensor_set_async(split_backend,\n'
    '                            input_cpy,\n'
    '                            (const uint8_t *)input->data + expert_offset, expert_offset,\n'
    '                            // copy a bit extra at the to ensure there are no NaNs in the padding of the last expert\n'
    '                            // this is necessary for MMQ in the CUDA backend\n'
    '                            expert_size_copy + padding_end);\n'
    '                    };\n'
    '\n'
    '                    int id = 0;\n'
    '                    while (!ggml_bitset_get(used_ids.data(), id)) {\n'
    '                        id++;\n'
    '                    }\n'
    '                    int32_t first_id = id;\n'
    '                    int32_t last_id = first_id;\n'
    '\n'
    '                    for (++id; id < n_expert; ++id) {\n'
    '                        if (!ggml_bitset_get(used_ids.data(), id)) {\n'
    '                            continue;\n'
    '                        }\n'
    '\n'
    '                        if (id == last_id + 1) {\n'
    '                            last_id = id;\n'
    '                            continue;\n'
    '                        }\n'
    '\n'
    '                        copy_experts(first_id, last_id);\n'
    '\n'
    '                        first_id = id;\n'
    '                        last_id = id;\n'
    '                    }\n'
    '                    copy_experts(first_id, last_id);\n'
    '                } else {\n'
)

_N_BE_BLOCK = (
    '                // BigCherry 1336 (upstream #29943): selective copying of host weights belongs to the copy callback\n'
    '                if (sched->callback_copy != NULL && ggml_backend_sched_is_host_weight(input) &&\n'
    '                    sched->callback_copy(split_backend, input, input_cpy, &split->graph, sched->callback_copy_user_data)) {\n'
    '                    // copied by the callback\n'
    '                } else {\n'
)

_A_BE_SET = (
    'void ggml_backend_sched_set_eval_callback(ggml_backend_sched_t sched, ggml_backend_sched_eval_callback callback, void * user_data) {\n'
    '    GGML_ASSERT(sched);\n'
    '    sched->callback_eval = callback;\n'
    '    sched->callback_eval_user_data = user_data;\n'
    '}\n'
)

_N_BE_SET = (
    'void ggml_backend_sched_set_eval_callback(ggml_backend_sched_t sched, ggml_backend_sched_eval_callback callback, void * user_data) {\n'
    '    GGML_ASSERT(sched);\n'
    '    sched->callback_eval = callback;\n'
    '    sched->callback_eval_user_data = user_data;\n'
    '}\n'
    '\n'
    'void ggml_backend_sched_set_copy_callback(ggml_backend_sched_t sched, ggml_backend_sched_copy_callback callback, void * user_data) {\n'
    '    GGML_ASSERT(sched);\n'
    '    sched->callback_copy = callback;\n'
    '    sched->callback_copy_user_data = user_data;\n'
    '}\n'
)

_A_CH_DECL = (
    '    llm_graph_cb graph_get_cb() const;\n'
)

_N_CH_DECL = (
    '    llm_graph_cb graph_get_cb() const;\n'
    '\n'
    '    // BigCherry 1336 (upstream #29943): ggml_backend_sched copy callback, copies only the experts used by MUL_MAT_ID\n'
    '    static bool sched_copy_experts(ggml_backend_t backend, const ggml_tensor * src, ggml_tensor * dst, ggml_cgraph * graph, void * user_data);\n'
)

_A_CH_STATE = (
    '    bool sched_need_reserve = true;\n'
)

_N_CH_STATE = (
    '    bool sched_need_reserve = true;\n'
    '\n'
    '    // state of sched_copy_experts, reset before each graph compute\n'
    '    struct copy_experts_info {\n'
    '        const ggml_tensor *  ids = nullptr;\n'
    '        std::vector<int32_t> ids_data;\n'
    '        std::vector<bool>    used;\n'
    '\n'
    '        void reset() {\n'
    '            ids = nullptr;\n'
    '            ids_data.clear();\n'
    '            used.clear();\n'
    '        }\n'
    '    };\n'
    '\n'
    '    copy_experts_info copy_experts;\n'
)

_A_C_NEW1 = (
    '    sched.reset(ggml_backend_sched_new(backend_ptrs.data(), backend_buft.data(), backend_ptrs.size(), max_nodes, cparams.pipeline_parallel, cparams.op_offload));\n'
)

_N_C_NEW1 = (
    '    sched.reset(ggml_backend_sched_new(backend_ptrs.data(), backend_buft.data(), backend_ptrs.size(), max_nodes, cparams.pipeline_parallel, cparams.op_offload));\n'
    '    ggml_backend_sched_set_copy_callback(sched.get(), sched_copy_experts, this);  // BigCherry 1336 (upstream #29943)\n'
)

_A_C_NEW2 = (
    '                sched.reset(ggml_backend_sched_new(backend_ptrs.data(), backend_buft.data(), backend_ptrs.size(), max_nodes, false, cparams.op_offload));\n'
)

_N_C_NEW2 = (
    '                sched.reset(ggml_backend_sched_new(backend_ptrs.data(), backend_buft.data(), backend_ptrs.size(), max_nodes, false, cparams.op_offload));\n'
    '                ggml_backend_sched_set_copy_callback(sched.get(), sched_copy_experts, this);  // BigCherry 1336 (upstream #29943)\n'
)

_A_C_RESET = (
    '    auto status = ggml_backend_sched_graph_compute_async(sched.get(), gf);\n'
)

_N_C_RESET = (
    '    copy_experts.reset();  // BigCherry 1336 (upstream #29943)\n'
    '\n'
    '    auto status = ggml_backend_sched_graph_compute_async(sched.get(), gf);\n'
)

_A_C_RESET_OPT = (
    '            ggml_opt_eval(opt_ctx, result);\n'
)

_N_C_RESET_OPT = (
    '            copy_experts.reset();  // BigCherry 1336 (upstream #29943)\n'
    '            ggml_opt_eval(opt_ctx, result);\n'
)

_A_C_FUNC = (
    'llm_graph_cb llama_context::graph_get_cb() const {\n'
)

_N_C_FUNC = (
    '// BigCherry 1336: counters of the host-weight copy callback, printed at exit under BIGCHERRY_PATCH_TRACE\n'
    'namespace {\n'
    'struct bc_moe_copy_stats_t {\n'
    '    uint64_t calls = 0, host_weight_bytes = 0, selective_calls = 0, dense_calls = 0, experts_total = 0, experts_used = 0, copied_bytes = 0;\n'
    '    ~bc_moe_copy_stats_t() {\n'
    '        if (calls > 0 && getenv("BIGCHERRY_PATCH_TRACE") != nullptr) {\n'
    '            fprintf(stderr, "BIGCHERRY_PATCH_HIT patch=1336_sched_copy_callback calls=%llu host_weight_bytes=%llu selective_calls=%llu "\n'
    '                    "dense_calls=%llu experts_used=%llu experts_total=%llu copied_bytes=%llu\\n",\n'
    '                    (unsigned long long) calls, (unsigned long long) host_weight_bytes, (unsigned long long) selective_calls,\n'
    '                    (unsigned long long) dense_calls, (unsigned long long) experts_used, (unsigned long long) experts_total,\n'
    '                    (unsigned long long) copied_bytes);\n'
    '        }\n'
    '    }\n'
    '};\n'
    'bc_moe_copy_stats_t bc_moe_copy_stats;\n'
    '}\n'
    '\n'
    'bool llama_context::sched_copy_experts(ggml_backend_t backend, const ggml_tensor * src, ggml_tensor * dst, ggml_cgraph * graph, void * user_data) {\n'
    '    auto & st = static_cast<llama_context *>(user_data)->copy_experts;\n'
    '\n'
    '    // BigCherry 1336: BIGCHERRY_MOE_COPY=0 observes only (the scheduler copies every host weight whole, MET01 control),\n'
    '    // 1 (default) copies the used experts, as upstream. BIGCHERRY_MOE_COPY_DENSE_PCT (default 90): when at least that\n'
    '    // share of the experts is used, one whole copy replaces the per-range copies (0 disables the shortcut).\n'
    '    static const int bc_mode = getenv("BIGCHERRY_MOE_COPY") != nullptr ? atoi(getenv("BIGCHERRY_MOE_COPY")) : 1;\n'
    '    static const int bc_dense_pct = getenv("BIGCHERRY_MOE_COPY_DENSE_PCT") != nullptr ? atoi(getenv("BIGCHERRY_MOE_COPY_DENSE_PCT")) : 90;\n'
    '    bc_moe_copy_stats.calls++;\n'
    '    bc_moe_copy_stats.host_weight_bytes += ggml_nbytes(src);\n'
    '    if (bc_mode == 0) {\n'
    '        return false;\n'
    '    }\n'
    '\n'
    '    // the ids must be computed before the split starts, so only the first node of the split is considered\n'
    '    if (ggml_graph_n_nodes(graph) == 0) {\n'
    '        return false;\n'
    '    }\n'
    '    const ggml_tensor * node = ggml_graph_node(graph, 0);\n'
    '    if (node->op != GGML_OP_MUL_MAT_ID || node->src[0] != dst) {\n'
    '        return false;\n'
    '    }\n'
    '\n'
    '    const ggml_tensor * ids = node->src[2];\n'
    '    if (ggml_nelements(ids) == 0) {\n'
    '        return true;\n'
    '    }\n'
    '\n'
    '    const int64_t n_expert    = src->ne[2];\n'
    '    const size_t  expert_size = src->nb[2];\n'
    '\n'
    '    if (ids != st.ids || (int64_t) st.used.size() != n_expert) {\n'
    '        st.ids_data.resize(ggml_nbytes(ids)/sizeof(int32_t));\n'
    '        ggml_backend_tensor_get_async(backend, ids, st.ids_data.data(), 0, ggml_nbytes(ids));\n'
    '        ggml_backend_synchronize(backend);\n'
    '\n'
    '        st.used.assign(n_expert, false);\n'
    '        for (int64_t i1 = 0; i1 < ids->ne[1]; i1++) {\n'
    '            for (int64_t i0 = 0; i0 < ids->ne[0]; i0++) {\n'
    '                const int32_t id = st.ids_data[i1*ids->nb[1]/sizeof(int32_t) + i0*ids->nb[0]/sizeof(int32_t)];\n'
    '                GGML_ASSERT(id >= 0 && id < n_expert);\n'
    '                st.used[id] = true;\n'
    '            }\n'
    '        }\n'
    '\n'
    '        st.ids = ids;\n'
    '    }\n'
    '\n'
    '    int64_t bc_n_used = 0;\n'
    '    for (int64_t i = 0; i < n_expert; i++) {\n'
    '        bc_n_used += st.used[i] ? 1 : 0;\n'
    '    }\n'
    '    bc_moe_copy_stats.selective_calls++;\n'
    '    bc_moe_copy_stats.experts_total += (uint64_t) n_expert;\n'
    '    bc_moe_copy_stats.experts_used  += (uint64_t) bc_n_used;\n'
    '    if (bc_dense_pct > 0 && bc_n_used*100 >= n_expert*bc_dense_pct) {\n'
    '        // nearly every expert is needed (large prefill batches): one contiguous copy by the scheduler\n'
    '        bc_moe_copy_stats.dense_calls++;\n'
    '        return false;\n'
    '    }\n'
    '\n'
    '    // group consecutive experts and copy them together\n'
    '    for (int64_t first = 0; first < n_expert; ) {\n'
    '        if (!st.used[first]) {\n'
    '            first++;\n'
    '            continue;\n'
    '        }\n'
    '        int64_t last = first;\n'
    '        while (last + 1 < n_expert && st.used[last + 1]) {\n'
    '            last++;\n'
    '        }\n'
    '\n'
    '        // copy a bit extra to ensure there are no NaNs in the padding of the last expert, this is necessary for MMQ in the CUDA backend\n'
    '        const size_t offset  = first*expert_size;\n'
    '        const size_t padding = last < n_expert - 1 ? std::min<size_t>(expert_size, 512) : 0;\n'
    '        ggml_backend_tensor_set_async(backend, dst, (const uint8_t *) src->data + offset, offset, (last - first + 1)*expert_size + padding);\n'
    '        bc_moe_copy_stats.copied_bytes += (last - first + 1)*expert_size + padding;\n'
    '\n'
    '        first = last + 1;\n'
    '    }\n'
    '\n'
    '    return true;\n'
    '}\n'
    '\n'
    'llm_graph_cb llama_context::graph_get_cb() const {\n'
)

PATCHES = [
    FilePatch(
        path="ggml/include/ggml-backend.h",
        description="1336: upstream #29943 scheduler copy callback (API)",
        language="none",
        edits=(
            Edit(id="copy-cb-typedef", anchor=_re.escape(_A_H_TYPE), mode="replace", text=_N_H_TYPE,
                 guard=r"typedef bool \(\*ggml_backend_sched_copy_callback\)",
                 rationale="After the eval callback typedef.", expect_matches=1, max_span_lines=2),
            Edit(id="copy-cb-setter-decl", anchor=_re.escape(_A_H_SET), mode="replace", text=_N_H_SET,
                 guard=r"ggml_backend_sched_set_copy_callback\(ggml_backend_sched_t sched",
                 rationale="After the eval callback setter.", expect_matches=1, max_span_lines=2),
        ),
    ),
    FilePatch(
        path="ggml/src/ggml-backend.cpp",
        description="1336: upstream #29943 scheduler copy callback (host weights copied last, selection in the callback)",
        language="none",
        edits=(
            Edit(id="copy-cb-fields", anchor=_re.escape(_A_BE_FIELDS), mode="replace", text=_N_BE_FIELDS,
                 guard=r"ggml_backend_sched_copy_callback callback_copy;",
                 rationale="Scheduler struct, after the eval callback.", expect_matches=1, max_span_lines=3),
            Edit(id="copy-cb-helper", anchor=_re.escape(_A_BE_HELPER), mode="replace", text=_N_BE_HELPER,
                 guard=r"static bool ggml_backend_sched_is_host_weight\(",
                 rationale="Before ggml_backend_sched_compute_splits.", expect_matches=1, max_span_lines=2),
            Edit(id="copy-cb-state", anchor=_re.escape(_A_BE_STATE), mode="replace", text=_N_BE_STATE,
                 guard=r"the expert selection state lives in the copy callback",
                 rationale="Locals of the embedded expert selection.", expect_matches=1, max_span_lines=5),
            Edit(id="copy-cb-two-pass", anchor=_re.escape(_A_BE_LOOP), mode="replace", text=_N_BE_LOOP,
                 guard=r"for \(int bc_copy_pass = 0; bc_copy_pass < 2; bc_copy_pass\+\+\)",
                 rationale="Head of the split input copy loop.", expect_matches=1, max_span_lines=4),
            Edit(id="copy-cb-call", anchor=_re.escape(_A_BE_BLOCK), mode="replace", text=_N_BE_BLOCK,
                 guard=r"sched->callback_copy\(split_backend, input, input_cpy, &split->graph",
                 rationale="The embedded MUL_MAT_ID expert selection, whole block.", expect_matches=1, max_span_lines=91),
            Edit(id="copy-cb-setter", anchor=_re.escape(_A_BE_SET), mode="replace", text=_N_BE_SET,
                 guard=r"void ggml_backend_sched_set_copy_callback\(",
                 rationale="After ggml_backend_sched_set_eval_callback.", expect_matches=1, max_span_lines=6),
        ),
    ),
    FilePatch(
        path="src/llama-context.h",
        description="1336: upstream #29943 sched_copy_experts declaration and state",
        language="none",
        edits=(
            Edit(id="copy-experts-decl", anchor=_re.escape(_A_CH_DECL), mode="replace", text=_N_CH_DECL,
                 guard=r"static bool sched_copy_experts\(",
                 rationale="After graph_get_cb.", expect_matches=1, max_span_lines=2),
            Edit(id="copy-experts-state", anchor=_re.escape(_A_CH_STATE), mode="replace", text=_N_CH_STATE,
                 guard=r"copy_experts_info copy_experts;",
                 rationale="After sched_need_reserve.", expect_matches=1, max_span_lines=2),
        ),
    ),
    FilePatch(
        path="src/llama-context.cpp",
        description="1336: upstream #29943 sched_copy_experts (with observation mode, dense shortcut and counters)",
        language="none",
        edits=(
            Edit(id="copy-experts-install", anchor=_re.escape(_A_C_NEW1), mode="replace", text=_N_C_NEW1,
                 guard=r"pipeline_parallel, cparams.op_offload\)\);\n    ggml_backend_sched_set_copy_callback",
                 rationale="First scheduler creation in sched_reserve.", expect_matches=1, max_span_lines=2),
            Edit(id="copy-experts-install-retry", anchor=_re.escape(_A_C_NEW2), mode="replace", text=_N_C_NEW2,
                 guard=r"max_nodes, false, cparams.op_offload\)\);\n                ggml_backend_sched_set_copy_callback",
                 rationale="Scheduler re-creation without pipeline parallelism.", expect_matches=1, max_span_lines=2),
            Edit(id="copy-experts-reset", anchor=_re.escape(_A_C_RESET), mode="replace", text=_N_C_RESET,
                 guard=r"copy_experts.reset\(\);  // BigCherry 1336 \(upstream #29943\)\n\n    auto status",
                 rationale="graph_compute, before the scheduler runs.", expect_matches=1, max_span_lines=2),
            Edit(id="copy-experts-reset-opt", anchor=_re.escape(_A_C_RESET_OPT), mode="replace", text=_N_C_RESET_OPT,
                 guard=r"copy_experts.reset\(\);  // BigCherry 1336 \(upstream #29943\)\n            ggml_opt_eval",
                 rationale="opt_epoch_iter, before the eval.", expect_matches=1, max_span_lines=2),
            Edit(id="copy-experts-func", anchor=_re.escape(_A_C_FUNC), mode="replace", text=_N_C_FUNC,
                 guard=r"bool llama_context::sched_copy_experts\(",
                 rationale="Before graph_get_cb, where upstream defines it.", expect_matches=1, max_span_lines=2),
        ),
    ),
]

ENV_DOCS = (
    EnvDoc("BIGCHERRY_MOE_COPY", "0|1", "1",
           "host-resident MoE experts: 1 copies only the experts a split uses (upstream behaviour, now in the copy "
           "callback); 0 is the observation-only control, every host weight is copied whole"),
    EnvDoc("BIGCHERRY_MOE_COPY_DENSE_PCT", "0..100", "90",
           "when at least this share of a layer's experts is used, copy the weight whole in one transfer instead of "
           "per-range copies; 0 disables"),
)
