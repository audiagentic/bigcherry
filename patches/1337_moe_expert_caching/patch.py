"""1337 (MET01): backport of upstream #29887 - a GPU cache for MoE experts kept in host memory.

Upstream PR #29887 (not merged at the b11402 pin), commit 6b7b03aab cherry-picked onto the pin. With
--moe-cache-mib N (LLAMA_ARG_MOE_CACHE_MIB), MUL_MAT_ID ops whose expert weights are host-resident (--n-cpu-moe) run on
the GPU against a persistent device buffer: per layer layout a bank of cache slots, the experts a batch selects are
uploaded on a miss (least recently used slot evicted), and the op reads a remapped ids tensor that names cache slots.
Without the flag nothing changes. Single device only, no pipeline parallelism (upstream's own limits).

The cache implementation (src/llama-moe-cache.cpp/.h) is carried as overlay source; this patch wires it in: the
scheduler hooks (ggml_backend_sched_set_moe_cache: resolve / begin / prepare), the split-graph changes that keep a
cached MUL_MAT_ID on the cache backend and substitute the remapped ids, context and common parameter plumbing, and the
CLI flag. Every edit is one hunk of the upstream commit, anchored on its unchanged context.

b11474 already contains #29943's native scheduler copy callback. This rebase prepares cache-owned host weights in
the scheduler's host-weight pass, while ordinary host weights continue through ggml_backend_sched_copy_input and the
native selective expert-copy callback. Scheduler creation installs both callbacks on every scheduler instance.
Superseded when the pin reaches a release containing #29887.
"""
import re as _re

from bigcherry.patcher import Edit, FilePatch

GROUP = "core"
STATE = "untested"

_A_ARG_CPP_0 = (
    '    ).set_env("LLAMA_ARG_N_CPU_MOE"));\n'
    '    add_opt(common_arg(\n'
)

_N_ARG_CPP_0 = (
    '    ).set_env("LLAMA_ARG_N_CPU_MOE"));\n'
    '    add_opt(common_arg(\n'
    '        {"--moe-cache-mib"}, "N",\n'
    '        "GPU cache size in MiB for the MoE experts kept in the CPU (default: 0, disabled)",\n'
    '        [](common_params & params, int value) {\n'
    '            if (value < 0) {\n'
    '                throw std::invalid_argument("invalid value");\n'
    '            }\n'
    '            params.moe_cache_size = (size_t) value*1024*1024;\n'
    '        }\n'
    '    ).set_env("LLAMA_ARG_MOE_CACHE_MIB"));\n'
    '    add_opt(common_arg(\n'
)

_A_COMMON_CPP_0 = (
    '\n'
    '    return cparams;\n'
)

_N_COMMON_CPP_0 = (
    '\n'
    '    cparams.moe_cache_size = params.moe_cache_size;\n'
    '\n'
    '    return cparams;\n'
)

_A_COMMON_H_0 = (
    '\n'
    '    common_conversation_mode conversation_mode = COMMON_CONVERSATION_MODE_AUTO;\n'
)

_N_COMMON_H_0 = (
    '\n'
    '    size_t moe_cache_size = 0; // GPU cache size in bytes for the MoE experts kept in the CPU\n'
    '\n'
    '    common_conversation_mode conversation_mode = COMMON_CONVERSATION_MODE_AUTO;\n'
)

_A_SPECULATIVE_CPP_0 = (
    '\n'
    '    // dflash/dspark decode the whole noise block in a single pass and sample every block position on the backend\n'
)

_N_SPECULATIVE_CPP_0 = (
    '\n'
    '    // the MoE cache is only used by the target context\n'
    '    result.moe_cache_size = 0;\n'
    '\n'
    '    // dflash/dspark decode the whole noise block in a single pass and sample every block position on the backend\n'
)

_A_GGML_BACKEND_H_0 = (
    '    // Initialize a backend scheduler, backends with low index are given priority over backends with high index\n'
)

_N_GGML_BACKEND_H_0 = (
    '    // MoE expert cache callbacks (set with ggml_backend_sched_set_moe_cache)\n'
    '    // resolve: return true if the MUL_MAT_ID node should read its host experts from the cache on backend, and the cache tensor to use instead\n'
    '    // begin:   called once at the start of each graph compute\n'
    '    // prepare: upload the missing experts and return the cache slot of each id, the array must stay valid until the next graph compute\n'
    '    typedef bool (*ggml_backend_sched_moe_cache_resolve_callback)(void * user_data, const struct ggml_tensor * node, ggml_backend_t backend, struct ggml_tensor ** cached_weight, void ** cache_entry);\n'
    '    typedef void (*ggml_backend_sched_moe_cache_begin_callback)(void * user_data);\n'
    '    typedef bool (*ggml_backend_sched_moe_cache_prepare_callback)(void * user_data, void * cache_entry, const int32_t * ids, size_t n_ids, const int32_t ** remapped_ids);\n'
    '\n'
    '    // Initialize a backend scheduler, backends with low index are given priority over backends with high index\n'
)

_A_GGML_BACKEND_H_1 = (
    '    //\n'
    '    // Meta backend\n'
    '    //\n'
)

_N_GGML_BACKEND_H_1 = (
    '    // Run MUL_MAT_ID ops with host expert weights on backend, reading the experts from a persistent cache\n'
    '    GGML_API void                 ggml_backend_sched_set_moe_cache(\n'
    '            ggml_backend_sched_t                          sched,\n'
    '            ggml_backend_t                                backend,\n'
    '            ggml_backend_sched_moe_cache_resolve_callback resolve,\n'
    '            ggml_backend_sched_moe_cache_begin_callback   begin,\n'
    '            ggml_backend_sched_moe_cache_prepare_callback prepare,\n'
    '            void *                                        user_data);\n'
    '\n'
    '    //\n'
    '    // Meta backend\n'
    '    //\n'
)

_A_GGML_BACKEND_CPP_0 = (
    '};\n'
    '\n'
    'struct ggml_backend_sched {\n'
    '    bool is_reset; // true if the scheduler has been reset since the last graph split\n'
)

_N_GGML_BACKEND_CPP_0 = (
    '};\n'
    '\n'
    'struct ggml_backend_sched_moe_cache_entry {\n'
    '    int split_id;\n'
    '    struct ggml_tensor * input;\n'
    '    struct ggml_tensor * ids;\n'
    '    struct ggml_tensor * ids_copy;\n'
    '    void * handle;\n'
    '    bool shared; // ids_copy belongs to a previous entry of the split\n'
    '};\n'
    '\n'
    'struct ggml_backend_sched {\n'
    '    bool is_reset; // true if the scheduler has been reset since the last graph split\n'
)

_A_GGML_BACKEND_CPP_1 = (
    '    int splits_capacity;\n'
    '\n'
    '    // pipeline parallelism support\n'
    '    int n_copies;\n'
)

_N_GGML_BACKEND_CPP_1 = (
    '    int splits_capacity;\n'
    '\n'
    '    struct ggml_backend_sched_moe_cache_entry * moe_cache_entries;\n'
    '    int n_moe_cache_entries;\n'
    '    int moe_cache_entries_capacity;\n'
    '\n'
    '    // pipeline parallelism support\n'
    '    int n_copies;\n'
)

_A_GGML_BACKEND_CPP_2 = (
    '    char * context_buffer;\n'
    '    size_t context_buffer_size;\n'
)

_N_GGML_BACKEND_CPP_2 = (
    '    ggml_backend_sched_moe_cache_resolve_callback callback_moe_cache_resolve;\n'
    '    ggml_backend_sched_moe_cache_begin_callback callback_moe_cache_begin;\n'
    '    ggml_backend_sched_moe_cache_prepare_callback callback_moe_cache_prepare;\n'
    '    void * callback_moe_cache_user_data;\n'
    '    ggml_backend_t moe_cache_backend;\n'
    '\n'
    '    char * context_buffer;\n'
    '    size_t context_buffer_size;\n'
)

_A_GGML_BACKEND_CPP_3 = (
    '}\n'
    '\n'
    '// returns the priority of the backend, lower id is higher priority\n'
    'static int ggml_backend_sched_backend_id(ggml_backend_sched_t sched, ggml_backend_t backend) {\n'
)

_N_GGML_BACKEND_CPP_3 = (
    '}\n'
    '\n'
    'static struct ggml_backend_sched_moe_cache_entry * ggml_backend_sched_moe_cache_entry_add(ggml_backend_sched_t sched) {\n'
    '    if (sched->n_moe_cache_entries >= sched->moe_cache_entries_capacity) {\n'
    '        const int new_cap = sched->moe_cache_entries_capacity > 0 ? 2*sched->moe_cache_entries_capacity : 16;\n'
    '        auto * pnew = (struct ggml_backend_sched_moe_cache_entry *) realloc(\n'
    '            sched->moe_cache_entries, new_cap * sizeof(struct ggml_backend_sched_moe_cache_entry));\n'
    '        if (pnew == NULL) {\n'
    '            GGML_ABORT("failed to grow MoE cache container");\n'
    '        }\n'
    '        sched->moe_cache_entries = pnew;\n'
    '        sched->moe_cache_entries_capacity = new_cap;\n'
    '    }\n'
    '\n'
    '    struct ggml_backend_sched_moe_cache_entry * entry = &sched->moe_cache_entries[sched->n_moe_cache_entries++];\n'
    '    memset(entry, 0, sizeof(*entry));\n'
    '    return entry;\n'
    '}\n'
    '\n'
    'static struct ggml_backend_sched_moe_cache_entry * ggml_backend_sched_moe_cache_entry_find(\n'
    '        ggml_backend_sched_t sched, int split_id, const struct ggml_tensor * input, const struct ggml_tensor * ids) {\n'
    '    for (int i = 0; i < sched->n_moe_cache_entries; ++i) {\n'
    '        struct ggml_backend_sched_moe_cache_entry * entry = &sched->moe_cache_entries[i];\n'
    '        if (entry->split_id == split_id && (entry->input == input || (ids != NULL && entry->ids == ids))) {\n'
    '            return entry;\n'
    '        }\n'
    '    }\n'
    '    return NULL;\n'
    '}\n'
    '\n'
    '// returns the priority of the backend, lower id is higher priority\n'
    'static int ggml_backend_sched_backend_id(ggml_backend_sched_t sched, ggml_backend_t backend) {\n'
)

_A_GGML_BACKEND_CPP_4 = (
    '                // check if a backend with higher prio wants to offload the op\n'
    '                if (sched->op_offload && src_backend_id == sched->n_backends - 1 && ggml_backend_buffer_is_host(src->buffer)) {\n'
    '                    for (int b = 0; b < src_backend_id; b++) {\n'
    '                        if (ggml_backend_supports_op(sched->backends[b], tensor) && ggml_backend_offload_op(sched->backends[b], tensor)) {\n'
)

_N_GGML_BACKEND_CPP_4 = (
    '                // check if a backend with higher prio wants to offload the op\n'
    '                if (sched->op_offload && src_backend_id == sched->n_backends - 1 && ggml_backend_buffer_is_host(src->buffer)) {\n'
    '                    // experts held by the MoE cache run on the cache backend\n'
    '                    if (i == 0 && tensor->op == GGML_OP_MUL_MAT_ID && sched->moe_cache_backend != NULL) {\n'
    '                        ggml_tensor * cached = NULL;\n'
    '                        void * handle = NULL;\n'
    '                        if (sched->callback_moe_cache_resolve(sched->callback_moe_cache_user_data, tensor, sched->moe_cache_backend, &cached, &handle) &&\n'
    '                            ggml_backend_supports_op(sched->moe_cache_backend, tensor)) {\n'
    '                            SET_CAUSE(tensor, "1.moe");\n'
    '                            return ggml_backend_sched_backend_id(sched, sched->moe_cache_backend);\n'
    '                        }\n'
    '                    }\n'
    '                    for (int b = 0; b < src_backend_id; b++) {\n'
    '                        if (ggml_backend_supports_op(sched->backends[b], tensor) && ggml_backend_offload_op(sched->backends[b], tensor)) {\n'
)

_A_GGML_BACKEND_CPP_5 = (
    '    sched->n_splits = 0;\n'
    '    sched->n_graph_inputs = 0;\n'
    '    sched->is_reset = false;\n'
    '\n'
)

_N_GGML_BACKEND_CPP_5 = (
    '    sched->n_splits = 0;\n'
    '    sched->n_graph_inputs = 0;\n'
    '    sched->n_moe_cache_entries = 0;\n'
    '    sched->is_reset = false;\n'
    '\n'
)

_A_GGML_BACKEND_CPP_6 = (
    '                        int src_backend_id = tensor_backend_id(src);\n'
    '                        if (src_backend_id != cur_backend_id && !ggml_backend_sched_buffer_supported(sched, src, cur_backend_id)) {\n'
    '                            need_new_split = true;\n'
    '                            break;\n'
)

_N_GGML_BACKEND_CPP_6 = (
    '                        int src_backend_id = tensor_backend_id(src);\n'
    '                        if (src_backend_id != cur_backend_id && !ggml_backend_sched_buffer_supported(sched, src, cur_backend_id)) {\n'
    '                            // cached experts selected by ids that are already remapped in this split can stay in it\n'
    '                            struct ggml_tensor * cached = NULL;\n'
    '                            void * handle = NULL;\n'
    '                            if (j == 0 && node->op == GGML_OP_MUL_MAT_ID && sched->callback_moe_cache_resolve != NULL &&\n'
    '                                ggml_backend_sched_moe_cache_entry_find(sched, i_split, NULL, node->src[2]) != NULL &&\n'
    '                                sched->callback_moe_cache_resolve(sched->callback_moe_cache_user_data, node, sched->backends[cur_backend_id], &cached, &handle)) {\n'
    '                                continue;\n'
    '                            }\n'
    '                            need_new_split = true;\n'
    '                            break;\n'
)

_A_GGML_BACKEND_CPP_7 = (
    '\n'
    '            // find inputs that are not on the same backend\n'
    '            for (int j = 0; j < GGML_MAX_SRC; j++) {\n'
    '                struct ggml_tensor * src = node->src[j];\n'
)

_N_GGML_BACKEND_CPP_7 = (
    '\n'
    '            // find inputs that are not on the same backend\n'
    '            struct ggml_backend_sched_moe_cache_entry * cache_entry = NULL;\n'
    '            for (int j = 0; j < GGML_MAX_SRC; j++) {\n'
    '                struct ggml_tensor * src = node->src[j];\n'
)

_A_GGML_BACKEND_CPP_8 = (
    '                if (src_backend_id != cur_backend_id && !ggml_backend_sched_buffer_supported(sched, src, cur_backend_id)) {\n'
    "                    // create a copy of the input in the split's backend\n"
    '                    if (tensor_id_copy(src_id, cur_backend_id, 0) == NULL) {\n'
    '                        ggml_backend_t backend = sched->backends[cur_backend_id];\n'
    '                        for (int c = 0; c < sched->n_copies; c++) {\n'
    '                            struct ggml_tensor * tensor_copy = ggml_dup_tensor_layout(sched->ctx, src);\n'
    '                            ggml_format_name(tensor_copy, "%s#%s#%d", ggml_backend_name(backend), src->name, c);\n'
    '                            if (sched->n_copies > 1) {\n'
    '                                ggml_set_input(tensor_copy);\n'
    '                                ggml_set_output(tensor_copy); // prevent ggml-alloc from overwriting the tensor\n'
    '                            }\n'
    '                            tensor_id_copy(src_id, cur_backend_id, c) = tensor_copy;\n'
    '                            SET_CAUSE(tensor_copy, "4.cpy");\n'
    '                        }\n'
    '                        int n_inputs = split->n_inputs++;\n'
)

_N_GGML_BACKEND_CPP_8 = (
    '                if (src_backend_id != cur_backend_id && !ggml_backend_sched_buffer_supported(sched, src, cur_backend_id)) {\n'
    "                    // create a copy of the input in the split's backend\n"
    '                    ggml_backend_t backend = sched->backends[cur_backend_id];\n'
    '                    struct ggml_tensor * cached_tensor = NULL;\n'
    '                    void * cache_handle = NULL;\n'
    '                    const bool cache_active =\n'
    '                        j == 0 &&\n'
    '                        node->op == GGML_OP_MUL_MAT_ID &&\n'
    '                        sched->n_copies == 1 &&\n'
    '                        sched->callback_moe_cache_resolve != NULL &&\n'
    '                        sched->callback_moe_cache_resolve(sched->callback_moe_cache_user_data, node, backend, &cached_tensor, &cache_handle);\n'
    '\n'
    '                    GGML_ASSERT(!cache_active || (cached_tensor->buffer != NULL && cached_tensor->ne[0] == src->ne[0] && cached_tensor->ne[1] == src->ne[1]));\n'
    '                    GGML_ASSERT(!cache_active || tensor_id_copy(src_id, cur_backend_id, 0) == NULL);\n'
    '                    if (tensor_id_copy(src_id, cur_backend_id, 0) == NULL) {\n'
    '                        if (cache_active) {\n'
    '                            tensor_id_copy(src_id, cur_backend_id, 0) = cached_tensor;\n'
    '                            SET_CAUSE(cached_tensor, "4.moe");\n'
    '                        } else {\n'
    '                            for (int c = 0; c < sched->n_copies; c++) {\n'
    '                                struct ggml_tensor * tensor_copy = ggml_dup_tensor_layout(sched->ctx, src);\n'
    '                                ggml_format_name(tensor_copy, "%s#%s#%d", ggml_backend_name(backend), src->name, c);\n'
    '                                if (sched->n_copies > 1) {\n'
    '                                    ggml_set_input(tensor_copy);\n'
    '                                    ggml_set_output(tensor_copy); // prevent ggml-alloc from overwriting the tensor\n'
    '                                }\n'
    '                                tensor_id_copy(src_id, cur_backend_id, c) = tensor_copy;\n'
    '                                SET_CAUSE(tensor_copy, "4.cpy");\n'
    '                            }\n'
    '                        }\n'
    '                        int n_inputs = split->n_inputs++;\n'
)

_A_GGML_BACKEND_CPP_9 = (
    '                        split->inputs[n_inputs] = src;\n'
    '                    }\n'
    '                    node->src[j] = tensor_id_copy(src_id, cur_backend_id, sched->cur_copy);\n'
    '                }\n'
    '            }\n'
    '        }\n'
    '        split->i_end = graph->n_nodes;\n'
)

_N_GGML_BACKEND_CPP_9 = (
    '                        split->inputs[n_inputs] = src;\n'
    '                    }\n'
    '                    if (cache_active) {\n'
    '                        // the cache slots of the experts are written to ids_copy before the split runs\n'
    '                        // the other projections of the layer use the same ids and share ids_copy\n'
    '                        const struct ggml_backend_sched_moe_cache_entry * prev = ggml_backend_sched_moe_cache_entry_find(sched, i_split, NULL, node->src[2]);\n'
    '                        struct ggml_tensor * ids_copy = prev ? prev->ids_copy : NULL;\n'
    '                        cache_entry = ggml_backend_sched_moe_cache_entry_add(sched);\n'
    '                        cache_entry->split_id = i_split;\n'
    '                        cache_entry->input    = src;\n'
    '                        cache_entry->ids      = node->src[2];\n'
    '                        cache_entry->handle   = cache_handle;\n'
    '                        cache_entry->shared   = ids_copy != NULL;\n'
    '                        if (ids_copy == NULL) {\n'
    '                            ids_copy = ggml_new_tensor_2d(sched->ctx, GGML_TYPE_I32, node->src[2]->ne[0], node->src[2]->ne[1]);\n'
    '                            ggml_format_name(ids_copy, "%s#%s#moe_cache", ggml_backend_name(backend), node->src[2]->name);\n'
    '                            ggml_set_output(node->src[2]); // keep the expert ids alive until they are remapped\n'
    '                        }\n'
    '                        cache_entry->ids_copy = ids_copy;\n'
    '                    }\n'
    '                    node->src[j] = tensor_id_copy(src_id, cur_backend_id, sched->cur_copy);\n'
    '                }\n'
    '            }\n'
    '            if (cache_entry != NULL) {\n'
    '                node->src[2] = cache_entry->ids_copy;\n'
    '            }\n'
    '        }\n'
    '        split->i_end = graph->n_nodes;\n'
)

_A_GGML_BACKEND_CPP_10 = (
    '        total_inputs += sched->splits[i].n_inputs;\n'
    '    }\n'
    '    int graph_size = std::max(graph->n_nodes, graph->n_leafs) + total_inputs * 2 * sched->n_copies + n_dep_nodes;\n'
    '\n'
    '    // remember the actual graph_size for performing reallocation checks later [GGML_SCHED_DEBUG_REALLOC]\n'
)

_N_GGML_BACKEND_CPP_10 = (
    '        total_inputs += sched->splits[i].n_inputs;\n'
    '    }\n'
    '    int graph_size = std::max(graph->n_nodes, graph->n_leafs) + total_inputs * 2 * sched->n_copies + n_dep_nodes + sched->n_moe_cache_entries;\n'
    '\n'
    '    // remember the actual graph_size for performing reallocation checks later [GGML_SCHED_DEBUG_REALLOC]\n'
)

_A_GGML_BACKEND_CPP_11 = (
    '        struct ggml_backend_sched_split * split = &sched->splits[i];\n'
    '\n'
    '        // add inputs to the graph copy so that they are allocated by ggml-alloc at the start of the split\n'
    '        for (int j = 0; j < split->n_inputs; j++) {\n'
)

_N_GGML_BACKEND_CPP_11 = (
    '        struct ggml_backend_sched_split * split = &sched->splits[i];\n'
    '\n'
    '        for (int j = 0; j < sched->n_moe_cache_entries; ++j) {\n'
    '            struct ggml_backend_sched_moe_cache_entry * entry = &sched->moe_cache_entries[j];\n'
    '            if (entry->split_id != i || entry->shared) {\n'
    '                continue;\n'
    '            }\n'
    '            sched->node_backend_ids[graph_copy->n_nodes] = split->backend_id;\n'
    '            graph_copy->nodes[graph_copy->n_nodes++] = entry->ids_copy;\n'
    '        }\n'
    '\n'
    '        // add inputs to the graph copy so that they are allocated by ggml-alloc at the start of the split\n'
    '        for (int j = 0; j < split->n_inputs; j++) {\n'
)

_A_GGML_BACKEND_CPP_12 = (
    '    int prev_backend_id = -1;\n'
)

_N_GGML_BACKEND_CPP_12 = (
    '    std::vector<int32_t> selected_ids;\n'
    '    std::vector<int32_t> cache_ids;\n'
    '\n'
    '    int prev_backend_id = -1;\n'
    '\n'
    '    if (sched->callback_moe_cache_begin != NULL) {\n'
    '        sched->callback_moe_cache_begin(sched->callback_moe_cache_user_data);\n'
    '    }\n'
)

_A_GGML_BACKEND_CPP_13 = r'''        for (int input_id = 0; input_id < split->n_inputs; input_id++) {
            if (ggml_backend_sched_is_host_weight(split->inputs[input_id])) {
                ggml_backend_sched_copy_input(sched, split, split->inputs[input_id]);
            }
        }
'''

_N_GGML_BACKEND_CPP_13 = r'''        for (int input_id = 0; input_id < split->n_inputs; input_id++) {
            if (!ggml_backend_sched_is_host_weight(split->inputs[input_id])) {
                continue;
            }

            struct ggml_tensor * input = split->inputs[input_id];
            struct ggml_backend_sched_moe_cache_entry * cache_entry =
                ggml_backend_sched_moe_cache_entry_find(sched, split_id, input, NULL);
            if (cache_entry == NULL) {
                ggml_backend_sched_copy_input(sched, split, input);
                continue;
            }

            // The cache owns this host expert input. Prepare it instead of invoking the ordinary
            // selective-copy callback; other host weights still use ggml_backend_sched_copy_input.
            if (cache_entry->shared) {
                continue;
            }
            ggml_tensor * ids_tensor = cache_entry->ids;
            ggml_backend_t ids_backend = ggml_backend_sched_get_tensor_backend(sched, ids_tensor);
            const int64_t n_ids = ggml_nelements(ids_tensor);
            if (n_ids == 0) {
                continue;
            }

            cache_ids.resize(ggml_nbytes(ids_tensor) / sizeof(int32_t));
            ggml_backend_tensor_get_async(ids_backend, ids_tensor, cache_ids.data(), 0, ggml_nbytes(ids_tensor));
            ggml_backend_synchronize(ids_backend);

            selected_ids.resize(n_ids);
            for (int64_t i1 = 0; i1 < ids_tensor->ne[1]; i1++) {
                for (int64_t i0 = 0; i0 < ids_tensor->ne[0]; i0++) {
                    selected_ids[i1*ids_tensor->ne[0] + i0] =
                        cache_ids[i1 * ids_tensor->nb[1]/sizeof(int32_t) + i0 * ids_tensor->nb[0]/sizeof(int32_t)];
                }
            }

            const int32_t * remapped_ids = NULL;
            if (!sched->callback_moe_cache_prepare(
                    sched->callback_moe_cache_user_data, cache_entry->handle, selected_ids.data(), n_ids, &remapped_ids)) {
                GGML_LOG_ERROR("%s: failed to prepare the MoE cache\n", __func__);
                return GGML_STATUS_FAILED;
            }
            ggml_backend_tensor_set_async(
                split_backend, cache_entry->ids_copy, remapped_ids, 0, ggml_nbytes(cache_entry->ids_copy));
        }
'''

_A_GGML_BACKEND_CPP_15 = (
    '    }\n'
    '    free(sched->splits);\n'
    '    free(sched->graph_inputs);\n'
    '    free(sched->hv_tensor_backend_ids);\n'
)

_N_GGML_BACKEND_CPP_15 = (
    '    }\n'
    '    free(sched->splits);\n'
    '    free(sched->moe_cache_entries);\n'
    '    free(sched->graph_inputs);\n'
    '    free(sched->hv_tensor_backend_ids);\n'
)

_A_GGML_BACKEND_CPP_16 = r'''void ggml_backend_sched_set_copy_callback(ggml_backend_sched_t sched, ggml_backend_sched_copy_callback callback, void * user_data) {
    GGML_ASSERT(sched);
    sched->callback_copy = callback;
    sched->callback_copy_user_data = user_data;
}

int ggml_backend_sched_get_n_splits(ggml_backend_sched_t sched) {
    GGML_ASSERT(sched);
'''

_N_GGML_BACKEND_CPP_16 = r'''void ggml_backend_sched_set_copy_callback(ggml_backend_sched_t sched, ggml_backend_sched_copy_callback callback, void * user_data) {
    GGML_ASSERT(sched);
    sched->callback_copy = callback;
    sched->callback_copy_user_data = user_data;
}

void ggml_backend_sched_set_moe_cache(
        ggml_backend_sched_t                          sched,
        ggml_backend_t                                backend,
        ggml_backend_sched_moe_cache_resolve_callback resolve,
        ggml_backend_sched_moe_cache_begin_callback   begin,
        ggml_backend_sched_moe_cache_prepare_callback prepare,
        void *                                        user_data) {
    GGML_ASSERT(sched);
    GGML_ASSERT((backend == NULL && resolve == NULL && begin == NULL && prepare == NULL) ||
                (backend != NULL && resolve != NULL && begin != NULL && prepare != NULL));
    const int backend_id = backend == NULL ? -1 : ggml_backend_sched_backend_id(sched, backend);
    GGML_ASSERT(backend == NULL || (backend_id >= 0 && backend_id < sched->n_backends - 1));
    sched->moe_cache_backend            = backend;
    sched->callback_moe_cache_resolve   = resolve;
    sched->callback_moe_cache_begin     = begin;
    sched->callback_moe_cache_prepare   = prepare;
    sched->callback_moe_cache_user_data = user_data;
}

int ggml_backend_sched_get_n_splits(ggml_backend_sched_t sched) {
    GGML_ASSERT(sched);
'''

_A_LLAMA_H_0 = (
    '\n'
    '        // Abort callback\n'
)

_N_LLAMA_H_0 = (
    '\n'
    '        size_t moe_cache_size; // device cache in bytes for the experts kept in host memory, 0 = disabled [EXPERIMENTAL]\n'
    '\n'
    '        // Abort callback\n'
)

_A_CMAKELISTS_TXT_0 = (
    '    llama-memory.cpp\n'
    '    llama-memory-hybrid.cpp\n'
)

_N_CMAKELISTS_TXT_0 = (
    '    llama-memory.cpp\n'
    '    llama-moe-cache.cpp\n'
    '    llama-memory-hybrid.cpp\n'
)

_A_LLAMA_CONTEXT_CPP_0 = (
    '#include "llama-model.h"\n'
    '#include "llama-ext.h"\n'
)

_N_LLAMA_CONTEXT_CPP_0 = (
    '#include "llama-model.h"\n'
    '#include "llama-moe-cache.h"\n'
    '#include "llama-ext.h"\n'
)

_A_LLAMA_CONTEXT_CPP_1 = (
    '\n'
    '    cparams.op_offload = params.op_offload;\n'
    '    cparams.kv_unified = params.kv_unified;\n'
    '\n'
)

_N_LLAMA_CONTEXT_CPP_1 = (
    '\n'
    '    cparams.op_offload     = params.op_offload;\n'
    '    cparams.kv_unified     = params.kv_unified;\n'
    '    cparams.moe_cache_size = params.moe_cache_size;\n'
    '\n'
)

_A_LLAMA_CONTEXT_CPP_2 = (
    '\n'
    '        sched_reserve();\n'
)

_N_LLAMA_CONTEXT_CPP_2 = (
    '\n'
    '        if (cparams.moe_cache_size > 0) {\n'
    '            if (!cparams.op_offload) {\n'
    '                throw std::runtime_error("MoE cache requires op offload");\n'
    '            }\n'
    '            if (cparams.pipeline_parallel || model.n_devices() > 1) {\n'
    '                throw std::runtime_error("MoE cache does not support multiple devices");\n'
    '            }\n'
    '            for (size_t i = 0; i < backend_ptrs.size(); ++i) {\n'
    '                const auto type = ggml_backend_dev_type(ggml_backend_get_device(backend_ptrs[i]));\n'
    '                if (type == GGML_BACKEND_DEVICE_TYPE_GPU || type == GGML_BACKEND_DEVICE_TYPE_IGPU) {\n'
    '                    moe_cache = std::make_unique<llama_moe_cache>(model, backend_ptrs[i], backend_buft[i], cparams.moe_cache_size);\n'
    '                    break;\n'
    '                }\n'
    '            }\n'
    '            if (!moe_cache) {\n'
    '                throw std::runtime_error("MoE cache requires a GPU backend");\n'
    '            }\n'
    '        }\n'
    '\n'
    '        sched_reserve();\n'
)

_A_LLAMA_CONTEXT_CPP_3 = r'''    sched.reset(ggml_backend_sched_new(backend_ptrs.data(), backend_buft.data(), backend_ptrs.size(), max_nodes, cparams.pipeline_parallel, cparams.op_offload));
    ggml_backend_sched_set_copy_callback(sched.get(), sched_copy_experts, this);
'''

_N_LLAMA_CONTEXT_CPP_3 = r'''    auto create_sched = [&](bool parallel) {
        sched.reset(ggml_backend_sched_new(backend_ptrs.data(), backend_buft.data(), backend_ptrs.size(), max_nodes, parallel, cparams.op_offload));
        ggml_backend_sched_set_copy_callback(sched.get(), sched_copy_experts, this);
        if (moe_cache) {
            ggml_backend_sched_set_moe_cache(sched.get(), moe_cache->backend(),
                llama_moe_cache::sched_resolve, llama_moe_cache::sched_begin, llama_moe_cache::sched_prepare, moe_cache.get());
        }
    };
    create_sched(cparams.pipeline_parallel);
'''

_A_LLAMA_CONTEXT_CPP_4 = r'''                sched.reset(ggml_backend_sched_new(backend_ptrs.data(), backend_buft.data(), backend_ptrs.size(), max_nodes, false, cparams.op_offload));
                ggml_backend_sched_set_copy_callback(sched.get(), sched_copy_experts, this);
'''

_N_LLAMA_CONTEXT_CPP_4 = r'''                create_sched(false);
'''

_A_LLAMA_CONTEXT_CPP_5 = (
    '    }\n'
    '    if (model.hparams.no_alloc) {\n'
)

_N_LLAMA_CONTEXT_CPP_5 = (
    '    }\n'
    '    if (moe_cache) {\n'
    '        for (const auto & [buft, size] : moe_cache->memory_breakdown()) {\n'
    '            ret[buft].context += size;\n'
    '        }\n'
    '    }\n'
    '    if (model.hparams.no_alloc) {\n'
)

_A_LLAMA_CONTEXT_CPP_6 = (
    '        /*.type_v                      =*/ GGML_TYPE_F16,\n'
    '        /*.abort_callback              =*/ nullptr,\n'
)

_N_LLAMA_CONTEXT_CPP_6 = (
    '        /*.type_v                      =*/ GGML_TYPE_F16,\n'
    '        /*.moe_cache_size              =*/ 0,\n'
    '        /*.abort_callback              =*/ nullptr,\n'
)

_A_LLAMA_CONTEXT_H_0 = (
    'class llama_batch_allocr;\n'
    '\n'
)

_N_LLAMA_CONTEXT_H_0 = (
    'class llama_batch_allocr;\n'
    'class llama_moe_cache;\n'
    '\n'
)

_A_LLAMA_CONTEXT_H_1 = (
    '    llama_memory_ptr memory;\n'
    '\n'
)

_N_LLAMA_CONTEXT_H_1 = (
    '    llama_memory_ptr memory;\n'
    '    std::unique_ptr<llama_moe_cache> moe_cache;\n'
    '\n'
)

_A_LLAMA_CPARAMS_H_0 = (
    '\n'
    '    std::vector<bool> embeddings_layer_inp; // [n_layer()] extract input embeddings for layer\n'
)

_N_LLAMA_CPARAMS_H_0 = (
    '\n'
    '    size_t moe_cache_size;\n'
    '\n'
    '    std::vector<bool> embeddings_layer_inp; // [n_layer()] extract input embeddings for layer\n'
)

_NEW_LLAMA_MOE_CACHE_CPP = (
    '#include "llama-moe-cache.h"\n'
    '\n'
    '#include "llama-impl.h"\n'
    '#include "llama-model.h"\n'
    '\n'
    '#include "ggml-cpp.h"\n'
    '\n'
    '#include <algorithm>\n'
    '#include <cstdlib>\n'
    '#include <cstring>\n'
    '#include <stdexcept>\n'
    '#include <unordered_map>\n'
    '#include <vector>\n'
    '\n'
    'namespace {\n'
    '\n'
    '// LRU map from (layer, expert) to a cache slot\n'
    'struct moe_cache_lru {\n'
    '    int32_t n_expert = 0;\n'
    '    int32_t n_slots  = 0;\n'
    '\n'
    '    std::vector<int32_t> slot_of; // [n_layer*n_expert], -1 if not cached\n'
    '    std::vector<int32_t> key_of;  // [n_slots], -1 if empty\n'
    '\n'
    '    // doubly linked list of the slots, head is the least recently used\n'
    '    std::vector<int32_t> prev;\n'
    '    std::vector<int32_t> next;\n'
    '    int32_t head = -1;\n'
    '    int32_t tail = -1;\n'
    '\n'
    '    std::vector<uint32_t> seen; // [n_expert]\n'
    '    uint32_t seen_gen = 0;\n'
    '    std::vector<int32_t> uniq;\n'
    '\n'
    '    void init(int32_t n_layer, int32_t n_expert, int32_t n_slots) {\n'
    '        this->n_expert = n_expert;\n'
    '        this->n_slots  = n_slots;\n'
    '        slot_of.assign((size_t) n_layer*n_expert, -1);\n'
    '        key_of.assign(n_slots, -1);\n'
    '        prev.resize(n_slots);\n'
    '        next.resize(n_slots);\n'
    '        for (int32_t s = 0; s < n_slots; ++s) {\n'
    '            prev[s] = s - 1;\n'
    '            next[s] = s + 1 < n_slots ? s + 1 : -1;\n'
    '        }\n'
    '        head = 0;\n'
    '        tail = n_slots - 1;\n'
    '        seen.assign(n_expert, 0);\n'
    '    }\n'
    '\n'
    '    // move slot s to the tail (most recently used)\n'
    '    void touch(int32_t s) {\n'
    '        if (s == tail) {\n'
    '            return;\n'
    '        }\n'
    '        if (prev[s] >= 0) {\n'
    '            next[prev[s]] = next[s];\n'
    '        } else {\n'
    '            head = next[s];\n'
    '        }\n'
    '        prev[next[s]] = prev[s];\n'
    '\n'
    '        prev[s] = tail;\n'
    '        next[s] = -1;\n'
    '        next[tail] = s;\n'
    '        tail = s;\n'
    '    }\n'
    '\n'
    '    struct fill {\n'
    '        int32_t expert;\n'
    '        int32_t slot;\n'
    '    };\n'
    '\n'
    '    // returns false if the ids select more distinct experts than there are slots\n'
    '    bool plan(int32_t il, const int32_t * ids, size_t n_ids, int32_t * remapped_ids, std::vector<fill> & fills, size_t & n_hit) {\n'
    '        fills.clear();\n'
    '        n_hit = 0;\n'
    '\n'
    '        if (++seen_gen == 0) {\n'
    '            std::fill(seen.begin(), seen.end(), 0);\n'
    '            seen_gen = 1;\n'
    '        }\n'
    '        uniq.clear();\n'
    '        for (size_t i = 0; i < n_ids; ++i) {\n'
    '            GGML_ASSERT(ids[i] >= 0 && ids[i] < n_expert);\n'
    '            if (seen[ids[i]] != seen_gen) {\n'
    '                seen[ids[i]] = seen_gen;\n'
    '                uniq.push_back(ids[i]);\n'
    '            }\n'
    '        }\n'
    '        if (uniq.size() > (size_t) n_slots) {\n'
    '            return false;\n'
    '        }\n'
    '\n'
    '        const size_t base = (size_t) il*n_expert;\n'
    '\n'
    '        // hits go to the tail first, so the head can be evicted below\n'
    '        for (int32_t e : uniq) {\n'
    '            if (slot_of[base + e] >= 0) {\n'
    '                touch(slot_of[base + e]);\n'
    '                n_hit++;\n'
    '            }\n'
    '        }\n'
    '        // sorted misses usually get consecutive slots, so the uploads can be merged\n'
    '        std::sort(uniq.begin(), uniq.end());\n'
    '        for (int32_t e : uniq) {\n'
    '            if (slot_of[base + e] >= 0) {\n'
    '                continue;\n'
    '            }\n'
    '            const int32_t s = head;\n'
    '            if (key_of[s] >= 0) {\n'
    '                slot_of[key_of[s]] = -1;\n'
    '            }\n'
    '            key_of[s] = base + e;\n'
    '            slot_of[base + e] = s;\n'
    '            touch(s);\n'
    '            fills.push_back({ e, s });\n'
    '        }\n'
    '\n'
    '        for (size_t i = 0; i < n_ids; ++i) {\n'
    '            remapped_ids[i] = slot_of[base + ids[i]];\n'
    '        }\n'
    '        return true;\n'
    '    }\n'
    '};\n'
    '\n'
    '// gate, up, down or gate_up, down\n'
    'static std::vector<ggml_tensor *> llama_moe_cache_layer_experts(const llama_layer & layer) {\n'
    '    std::vector<ggml_tensor *> res;\n'
    '    for (ggml_tensor * t : { layer.ffn_gate_up_exps, layer.ffn_gate_exps, layer.ffn_up_exps, layer.ffn_down_exps }) {\n'
    '        if (t != nullptr) {\n'
    '            res.push_back(t);\n'
    '        }\n'
    '    }\n'
    '    return res;\n'
    '}\n'
    '\n'
    'static bool llama_moe_cache_same_layout(const std::vector<ggml_tensor *> & a, const std::vector<ggml_tensor *> & b) {\n'
    '    if (a.size() != b.size()) {\n'
    '        return false;\n'
    '    }\n'
    '    for (size_t i = 0; i < a.size(); ++i) {\n'
    '        if (a[i]->type != b[i]->type || !ggml_are_same_shape(a[i], b[i]) || a[i]->nb[2] != b[i]->nb[2]) {\n'
    '            return false;\n'
    '        }\n'
    '    }\n'
    '    return true;\n'
    '}\n'
    '\n'
    'static bool llama_moe_cache_is_host_weight(const ggml_tensor * t) {\n'
    '    return t->buffer != nullptr &&\n'
    '        ggml_backend_buffer_get_usage(t->buffer) == GGML_BACKEND_BUFFER_USAGE_WEIGHTS &&\n'
    '        ggml_backend_buffer_is_host(t->buffer);\n'
    '}\n'
    '\n'
    '}\n'
    '\n'
    'struct llama_moe_cache::impl {\n'
    '    // layers with the same expert tensor layout share the banks and the LRU of a group\n'
    '    struct group {\n'
    '        std::vector<ggml_tensor *> ref; // expert tensors of the first layer\n'
    '        std::vector<int32_t> layers;\n'
    '        std::vector<ggml_tensor *> banks;\n'
    '        size_t host_bytes = 0;\n'
    '        int32_t n_slots = 0;\n'
    '        moe_cache_lru lru;\n'
    '    };\n'
    '\n'
    '    struct binding {\n'
    '        int32_t il;\n'
    '        int32_t ig;\n'
    '        ggml_tensor * src;    // host expert tensor\n'
    '        ggml_tensor * bank;   // device storage of all slots\n'
    '        ggml_tensor * cached; // view of the bank used in place of src\n'
    '    };\n'
    '\n'
    '    struct layer_state {\n'
    '        std::vector<int32_t> bindings;\n'
    '        uint64_t planned_epoch = 0;\n'
    '        std::vector<int32_t> planned_ids;\n'
    '        std::vector<int32_t> remapped_ids;\n'
    '    };\n'
    '\n'
    '    struct stats {\n'
    '        size_t hits   = 0;\n'
    '        size_t misses = 0;\n'
    '        size_t bytes  = 0;\n'
    '    };\n'
    '\n'
    '    static constexpr int64_t max_batch = 32;\n'
    '\n'
    '    ggml_backend_t backend;\n'
    '    bool no_alloc;\n'
    '    int32_t n_expert_used;\n'
    '\n'
    '    uint64_t epoch = 0;\n'
    '    stats stats_small; // up to 8 tokens per ubatch\n'
    '    stats stats_large;\n'
    '\n'
    '    std::vector<group> groups;\n'
    '    std::vector<moe_cache_lru::fill> fills;\n'
    '\n'
    '    std::vector<binding> bindings;\n'
    '    std::vector<layer_state> layers;\n'
    '    std::unordered_map<const ggml_tensor *, int32_t> binding_of;\n'
    '\n'
    '    ggml_context_ptr ctx;\n'
    '    ggml_backend_buffer_ptr buf;\n'
    '    size_t buf_size = 0;\n'
    '\n'
    '    impl(const llama_model & model, ggml_backend_t backend, ggml_backend_buffer_type_t buft, size_t size) :\n'
    '            backend(backend), no_alloc(model.hparams.no_alloc), n_expert_used(model.hparams.n_expert_used_max()), layers(model.layers.size()) {\n'
    '        ggml_backend_dev_t dev = ggml_backend_get_device(backend);\n'
    '        const auto dev_type = ggml_backend_dev_type(dev);\n'
    '        if (dev_type != GGML_BACKEND_DEVICE_TYPE_GPU && dev_type != GGML_BACKEND_DEVICE_TYPE_IGPU) {\n'
    '            throw std::runtime_error("MoE cache requires a GPU backend");\n'
    '        }\n'
    '        if (model.split_mode() == LLAMA_SPLIT_MODE_TENSOR) {\n'
    '            throw std::runtime_error("MoE cache does not support tensor parallelism");\n'
    '        }\n'
    '        if (model.hparams.n_expert == 0 || n_expert_used == 0) {\n'
    '            throw std::runtime_error("MoE cache requires a MoE model");\n'
    '        }\n'
    '        const int32_t n_expert = model.hparams.n_expert;\n'
    '\n'
    '        // only cache layers that keep all of their experts in host memory\n'
    '        size_t host_bytes = 0;\n'
    '        for (size_t il = 0; il < model.layers.size(); ++il) {\n'
    '            auto experts = llama_moe_cache_layer_experts(model.layers[il]);\n'
    '            if (experts.empty() || model.dev_layer(il) != dev ||\n'
    '                !std::all_of(experts.begin(), experts.end(), llama_moe_cache_is_host_weight)) {\n'
    '                continue;\n'
    '            }\n'
    '            auto it = std::find_if(groups.begin(), groups.end(), [&](const group & g) { return llama_moe_cache_same_layout(g.ref, experts); });\n'
    '            if (it == groups.end()) {\n'
    '                groups.emplace_back();\n'
    '                it = groups.end() - 1;\n'
    '                it->ref = experts;\n'
    '            }\n'
    '            it->layers.push_back(il);\n'
    '            for (const ggml_tensor * t : experts) {\n'
    '                it->host_bytes += ggml_nbytes(t);\n'
    '                host_bytes     += ggml_nbytes(t);\n'
    '            }\n'
    '        }\n'
    '        if (groups.empty()) {\n'
    '            LLAMA_LOG_WARN("%s: no layer has all of its experts in host memory, MoE cache is disabled\\n", __func__);\n'
    '            return;\n'
    '        }\n'
    '\n'
    '        // one extra slot at the end, CUDA MMQ can read past the last expert\n'
    '        const size_t alignment = ggml_backend_buft_get_alignment(buft);\n'
    '        auto alloc_size = [&](const group & g, int32_t n_slots) {\n'
    '            size_t res = 0;\n'
    '            for (const ggml_tensor * t : g.ref) {\n'
    '                res += GGML_PAD(t->nb[2]*(n_slots + 1), alignment);\n'
    '            }\n'
    '            return res;\n'
    '        };\n'
    '\n'
    '        // split the budget by the size of the experts, so each group caches the same fraction of its experts\n'
    '        size_t n_tensors = 0;\n'
    '        for (group & g : groups) {\n'
    '            const size_t budget = (size_t) ((double) size*g.host_bytes/host_bytes);\n'
    '            const int32_t max_slots = g.layers.size()*n_expert;\n'
    '            while (g.n_slots < max_slots && alloc_size(g, g.n_slots + 1) <= budget) {\n'
    '                g.n_slots++;\n'
    '            }\n'
    '            if (g.n_slots < n_expert_used) {\n'
    '                LLAMA_LOG_WARN("%s: MoE cache budget is too small for %zu layers, they are not cached\\n", __func__, g.layers.size());\n'
    '                g.n_slots = 0;\n'
    '                continue;\n'
    '            }\n'
    '            g.lru.init(model.layers.size(), n_expert, g.n_slots);\n'
    '            n_tensors += g.ref.size()*(1 + g.layers.size());\n'
    '        }\n'
    '        if (n_tensors == 0) {\n'
    '            throw std::runtime_error("MoE cache is too small to hold the experts of one token");\n'
    '        }\n'
    '\n'
    '        ggml_init_params params = {\n'
    '            /*.mem_size   =*/ n_tensors*ggml_tensor_overhead(),\n'
    '            /*.mem_buffer =*/ nullptr,\n'
    '            /*.no_alloc   =*/ true,\n'
    '        };\n'
    '        ctx.reset(ggml_init(params));\n'
    '        if (!ctx) {\n'
    '            throw std::runtime_error("failed to create the MoE cache context");\n'
    '        }\n'
    '\n'
    '        for (size_t ig = 0; ig < groups.size(); ++ig) {\n'
    '            group & g = groups[ig];\n'
    '            if (g.n_slots == 0) {\n'
    '                continue;\n'
    '            }\n'
    '            for (const ggml_tensor * t : g.ref) {\n'
    '                ggml_tensor * bank = ggml_new_tensor_3d(ctx.get(), t->type, t->ne[0], t->ne[1], g.n_slots + 1);\n'
    '                GGML_ASSERT(bank->nb[2] == t->nb[2]);\n'
    '                ggml_format_name(bank, "moe_cache.%zu.%s", ig, t->name);\n'
    '                g.banks.push_back(bank);\n'
    '            }\n'
    '            for (int32_t il : g.layers) {\n'
    '                const auto experts = llama_moe_cache_layer_experts(model.layers[il]);\n'
    '                for (size_t ip = 0; ip < experts.size(); ++ip) {\n'
    '                    ggml_tensor * bank   = g.banks[ip];\n'
    '                    ggml_tensor * cached = ggml_view_3d(ctx.get(), bank, bank->ne[0], bank->ne[1], g.n_slots, bank->nb[1], bank->nb[2], 0);\n'
    '                    ggml_format_name(cached, "moe_cache.%s", experts[ip]->name);\n'
    '                    binding_of[experts[ip]] = bindings.size();\n'
    '                    layers[il].bindings.push_back(bindings.size());\n'
    '                    bindings.push_back({ il, (int32_t) ig, experts[ip], bank, cached });\n'
    '                }\n'
    '            }\n'
    '            buf_size += alloc_size(g, g.n_slots);\n'
    '        }\n'
    '\n'
    '        if (no_alloc) {\n'
    '            // only used to measure the memory use, see llama_context::memory_breakdown\n'
    '            buf.reset(ggml_backend_buft_alloc_buffer(buft, 0));\n'
    '            for (ggml_tensor * t = ggml_get_first_tensor(ctx.get()); t != nullptr; t = ggml_get_next_tensor(ctx.get(), t)) {\n'
    '                t->buffer = buf.get();\n'
    '            }\n'
    '        } else {\n'
    '            buf.reset(ggml_backend_alloc_ctx_tensors_from_buft(ctx.get(), buft));\n'
    '            if (!buf) {\n'
    '                throw std::runtime_error("failed to allocate the MoE cache buffer");\n'
    '            }\n'
    '            ggml_backend_buffer_clear(buf.get(), 0);\n'
    '            buf_size = ggml_backend_buffer_get_size(buf.get());\n'
    '        }\n'
    '\n'
    '        LLAMA_LOG_INFO("%s: %10s MoE cache size = %8.2f MiB for %.2f MiB of host experts\\n", __func__,\n'
    '            ggml_backend_buft_name(buft), buf_size/1024.0/1024.0, host_bytes/1024.0/1024.0);\n'
    '        for (const group & g : groups) {\n'
    '            LLAMA_LOG_INFO("%s: %2zu layers, %s: %5d slots (%.1f%%)\\n", __func__,\n'
    '                g.layers.size(), ggml_type_name(g.ref.back()->type), g.n_slots, 100.0*g.n_slots/(g.layers.size()*n_expert));\n'
    '        }\n'
    '    }\n'
    '\n'
    '    ~impl() {\n'
    '        log_stats();\n'
    '    }\n'
    '\n'
    '    bool resolve(const ggml_tensor * node, ggml_backend_t target, ggml_tensor ** cached_weight, void ** cache_entry) {\n'
    '        if (target != backend) {\n'
    '            return false;\n'
    '        }\n'
    '        const auto it = binding_of.find(node->src[0]);\n'
    '        if (it == binding_of.end()) {\n'
    '            return false;\n'
    '        }\n'
    '        binding & b = bindings[it->second];\n'
    '\n'
    '        // large batches use most experts of a layer, so they gain little from the cache and would evict the experts used in generation\n'
    '        const int64_t n_tokens = node->src[2]->ne[1];\n'
    '        if (n_tokens > max_batch || std::min(n_tokens*node->src[2]->ne[0], b.src->ne[2]) > groups[b.ig].n_slots) {\n'
    '            return false;\n'
    '        }\n'
    '\n'
    '        *cached_weight = b.cached;\n'
    '        *cache_entry   = &b;\n'
    '        return true;\n'
    '    }\n'
    '\n'
    '    void begin() {\n'
    '        if (++epoch == 0) {\n'
    '            epoch = 1;\n'
    '            for (auto & l : layers) {\n'
    '                l.planned_epoch = 0;\n'
    '            }\n'
    '        }\n'
    '    }\n'
    '\n'
    '    bool prepare(const binding * entry, const int32_t * ids, size_t n_ids, const int32_t ** remapped_ids) {\n'
    '        layer_state & l = layers[entry->il];\n'
    '\n'
    '        // the experts of all projections of the layer are uploaded on the first call, the others reuse the plan\n'
    '        if (l.planned_epoch == epoch) {\n'
    '            if (l.planned_ids.size() != n_ids || !std::equal(l.planned_ids.begin(), l.planned_ids.end(), ids)) {\n'
    '                return false;\n'
    '            }\n'
    '            *remapped_ids = l.remapped_ids.data();\n'
    '            return true;\n'
    '        }\n'
    '\n'
    '        l.planned_ids.assign(ids, ids + n_ids);\n'
    '        l.remapped_ids.resize(n_ids);\n'
    '\n'
    '        size_t n_hit = 0;\n'
    '        if (!groups[entry->ig].lru.plan(entry->il, ids, n_ids, l.remapped_ids.data(), fills, n_hit)) {\n'
    '            return false;\n'
    '        }\n'
    '\n'
    '        size_t bytes = 0;\n'
    '        for (int32_t ib : l.bindings) {\n'
    '            const binding & b = bindings[ib];\n'
    '            const size_t expert_size = b.src->nb[2];\n'
    '            for (size_t i = 0; i < fills.size();) {\n'
    '                size_t n = 1;\n'
    '                while (i + n < fills.size() && fills[i + n].expert == fills[i].expert + (int32_t) n && fills[i + n].slot == fills[i].slot + (int32_t) n) {\n'
    '                    n++;\n'
    '                }\n'
    '                ggml_backend_tensor_set_async(backend, b.bank, (const uint8_t *) b.src->data + fills[i].expert*expert_size, fills[i].slot*expert_size, n*expert_size);\n'
    '                bytes += n*expert_size;\n'
    '                i += n;\n'
    '            }\n'
    '        }\n'
    '\n'
    '        stats & st = n_ids <= (size_t) 8*n_expert_used ? stats_small : stats_large;\n'
    '        st.hits   += n_hit;\n'
    '        st.misses += fills.size();\n'
    '        st.bytes  += bytes;\n'
    '\n'
    '        l.planned_epoch = epoch;\n'
    '        *remapped_ids = l.remapped_ids.data();\n'
    '        return true;\n'
    '    }\n'
    '\n'
    '    void log_stats() const {\n'
    '        auto log = [](const char * name, const stats & st) {\n'
    '            const size_t n = st.hits + st.misses;\n'
    '            if (n == 0) {\n'
    '                return;\n'
    '            }\n'
    '            LLAMA_LOG_INFO("llama_moe_cache: %s: hits = %zu, misses = %zu, hit rate = %.2f%%, uploaded = %.2f MiB\\n",\n'
    '                name, st.hits, st.misses, 100.0*st.hits/n, st.bytes/1024.0/1024.0);\n'
    '        };\n'
    '        log("ubatch <= 8", stats_small);\n'
    '        log("ubatch  > 8", stats_large);\n'
    '        if (getenv("BIGCHERRY_PATCH_TRACE") != nullptr) {\n'
    '            LLAMA_LOG_INFO("BIGCHERRY_PATCH_HIT patch=1337_moe_expert_caching small_hits=%zu small_misses=%zu small_uploaded_mib=%.3f large_hits=%zu large_misses=%zu large_uploaded_mib=%.3f\\n",\n'
    '                stats_small.hits, stats_small.misses, stats_small.bytes/1024.0/1024.0,\n'
    '                stats_large.hits, stats_large.misses, stats_large.bytes/1024.0/1024.0);\n'
    '        }\n'
    '    }\n'
    '};\n'
    '\n'
    'llama_moe_cache::llama_moe_cache(const llama_model & model, ggml_backend_t backend, ggml_backend_buffer_type_t buft, size_t size) :\n'
    '    pimpl(new impl(model, backend, buft, size)) {\n'
    '}\n'
    '\n'
    'llama_moe_cache::~llama_moe_cache() = default;\n'
    '\n'
    'ggml_backend_t llama_moe_cache::backend() const {\n'
    '    return pimpl->backend;\n'
    '}\n'
    '\n'
    'std::map<ggml_backend_buffer_type_t, size_t> llama_moe_cache::memory_breakdown() const {\n'
    '    std::map<ggml_backend_buffer_type_t, size_t> res;\n'
    '    if (pimpl->buf) {\n'
    '        res[ggml_backend_buffer_get_type(pimpl->buf.get())] = pimpl->buf_size;\n'
    '    }\n'
    '    return res;\n'
    '}\n'
    '\n'
    'bool llama_moe_cache::sched_resolve(void * user_data, const ggml_tensor * node, ggml_backend_t backend, ggml_tensor ** cached_weight, void ** cache_entry) {\n'
    '    return static_cast<llama_moe_cache *>(user_data)->pimpl->resolve(node, backend, cached_weight, cache_entry);\n'
    '}\n'
    '\n'
    'void llama_moe_cache::sched_begin(void * user_data) {\n'
    '    static_cast<llama_moe_cache *>(user_data)->pimpl->begin();\n'
    '}\n'
    '\n'
    'bool llama_moe_cache::sched_prepare(void * user_data, void * cache_entry, const int32_t * ids, size_t n_ids, const int32_t ** remapped_ids) {\n'
    '    return static_cast<llama_moe_cache *>(user_data)->pimpl->prepare(static_cast<const impl::binding *>(cache_entry), ids, n_ids, remapped_ids);\n'
    '}\n'
)

_NEW_LLAMA_MOE_CACHE_H = (
    '#pragma once\n'
    '\n'
    '#include "ggml-backend.h"\n'
    '\n'
    '#include <map>\n'
    '#include <memory>\n'
    '\n'
    'struct llama_model;\n'
    '\n'
    '// keeps the most recently used experts of host-resident MoE layers in a device buffer\n'
    '// MUL_MAT_ID ops on these experts run on the device and only the cache misses are uploaded\n'
    'class llama_moe_cache {\n'
    'public:\n'
    '    llama_moe_cache(const llama_model & model, ggml_backend_t backend, ggml_backend_buffer_type_t buft, size_t size);\n'
    '    ~llama_moe_cache();\n'
    '\n'
    '    ggml_backend_t backend() const;\n'
    '\n'
    '    std::map<ggml_backend_buffer_type_t, size_t> memory_breakdown() const;\n'
    '\n'
    '    // ggml_backend_sched callbacks, user_data is the llama_moe_cache\n'
    '    static bool sched_resolve(void * user_data, const ggml_tensor * node, ggml_backend_t backend, ggml_tensor ** cached_weight, void ** cache_entry);\n'
    '    static void sched_begin(void * user_data);\n'
    '    static bool sched_prepare(void * user_data, void * cache_entry, const int32_t * ids, size_t n_ids, const int32_t ** remapped_ids);\n'
    '\n'
    'private:\n'
    '    struct impl;\n'
    '    std::unique_ptr<impl> pimpl;\n'
    '};\n'
)

PATCHES = [
    FilePatch(
        path="common/arg.cpp",
        description="1337: upstream #29887 MoE expert cache (arg.cpp)",
        language="none",
        edits=(
            Edit(id="moe-cache-arg-cpp-0", anchor=_re.escape(_A_ARG_CPP_0), mode="replace", text=_N_ARG_CPP_0,
                 guard=_re.escape('{"--moe-cache-mib"}, "N",'),
                 rationale="upstream #29887 hunk 1 of common/arg.cpp.", expect_matches=1, max_span_lines=3),
        ),
    ),
    FilePatch(
        path="common/common.cpp",
        description="1337: upstream #29887 MoE expert cache (common.cpp)",
        language="none",
        edits=(
            Edit(id="moe-cache-common-cpp-0", anchor=_re.escape(_A_COMMON_CPP_0), mode="replace", text=_N_COMMON_CPP_0,
                 guard=_re.escape('cparams.moe_cache_size = params.moe_cache_size;'),
                 rationale="upstream #29887 hunk 1 of common/common.cpp.", expect_matches=1, max_span_lines=3),
        ),
    ),
    FilePatch(
        path="common/common.h",
        description="1337: upstream #29887 MoE expert cache (common.h)",
        language="none",
        edits=(
            Edit(id="moe-cache-common-h-0", anchor=_re.escape(_A_COMMON_H_0), mode="replace", text=_N_COMMON_H_0,
                 guard=_re.escape('size_t moe_cache_size = 0; // GPU cache size in bytes for the MoE experts kept in the CPU'),
                 rationale="upstream #29887 hunk 1 of common/common.h.", expect_matches=1, max_span_lines=3),
        ),
    ),
    FilePatch(
        path="common/speculative.cpp",
        description="1337: upstream #29887 MoE expert cache (speculative.cpp)",
        language="none",
        edits=(
            Edit(id="moe-cache-speculative-cpp-0", anchor=_re.escape(_A_SPECULATIVE_CPP_0), mode="replace", text=_N_SPECULATIVE_CPP_0,
                 guard=_re.escape('// the MoE cache is only used by the target context'),
                 rationale="upstream #29887 hunk 1 of common/speculative.cpp.", expect_matches=1, max_span_lines=3),
        ),
    ),
    FilePatch(
        path="ggml/include/ggml-backend.h",
        description="1337: upstream #29887 MoE expert cache (ggml-backend.h)",
        language="none",
        edits=(
            Edit(id="moe-cache-ggml-backend-h-0", anchor=_re.escape(_A_GGML_BACKEND_H_0), mode="replace", text=_N_GGML_BACKEND_H_0,
                 guard=_re.escape('// MoE expert cache callbacks (set with ggml_backend_sched_set_moe_cache)'),
                 rationale="upstream #29887 hunk 1 of ggml/include/ggml-backend.h.", expect_matches=1, max_span_lines=2),
            Edit(id="moe-cache-ggml-backend-h-1", anchor=_re.escape(_A_GGML_BACKEND_H_1), mode="replace", text=_N_GGML_BACKEND_H_1,
                 guard=_re.escape('// Run MUL_MAT_ID ops with host expert weights on backend, reading the experts from a persistent cache'),
                 rationale="upstream #29887 hunk 2 of ggml/include/ggml-backend.h.", expect_matches=1, max_span_lines=4),
        ),
    ),
    FilePatch(
        path="ggml/src/ggml-backend.cpp",
        description="1337: upstream #29887 MoE expert cache (ggml-backend.cpp)",
        language="none",
        edits=(
            Edit(id="moe-cache-ggml-backend-cpp-0", anchor=_re.escape(_A_GGML_BACKEND_CPP_0), mode="replace", text=_N_GGML_BACKEND_CPP_0,
                 guard=_re.escape('struct ggml_backend_sched_moe_cache_entry {'),
                 rationale="upstream #29887 hunk 1 of ggml/src/ggml-backend.cpp.", expect_matches=1, max_span_lines=5),
            Edit(id="moe-cache-ggml-backend-cpp-1", anchor=_re.escape(_A_GGML_BACKEND_CPP_1), mode="replace", text=_N_GGML_BACKEND_CPP_1,
                 guard=_re.escape('struct ggml_backend_sched_moe_cache_entry * moe_cache_entries;'),
                 rationale="upstream #29887 hunk 2 of ggml/src/ggml-backend.cpp.", expect_matches=1, max_span_lines=5),
            Edit(id="moe-cache-ggml-backend-cpp-2", anchor=_re.escape(_A_GGML_BACKEND_CPP_2), mode="replace", text=_N_GGML_BACKEND_CPP_2,
                 guard=_re.escape('ggml_backend_sched_moe_cache_resolve_callback callback_moe_cache_resolve;'),
                 rationale="upstream #29887 hunk 3 of ggml/src/ggml-backend.cpp.", expect_matches=1, max_span_lines=3),
            Edit(id="moe-cache-ggml-backend-cpp-3", anchor=_re.escape(_A_GGML_BACKEND_CPP_3), mode="replace", text=_N_GGML_BACKEND_CPP_3,
                 guard=_re.escape('static struct ggml_backend_sched_moe_cache_entry * ggml_backend_sched_moe_cache_entry_add(ggml_backend_sched_t sched) {'),
                 rationale="upstream #29887 hunk 4 of ggml/src/ggml-backend.cpp.", expect_matches=1, max_span_lines=5),
            Edit(id="moe-cache-ggml-backend-cpp-4", anchor=_re.escape(_A_GGML_BACKEND_CPP_4), mode="replace", text=_N_GGML_BACKEND_CPP_4,
                 guard=_re.escape('// experts held by the MoE cache run on the cache backend'),
                 rationale="upstream #29887 hunk 5 of ggml/src/ggml-backend.cpp.", expect_matches=1, max_span_lines=5),
            Edit(id="moe-cache-ggml-backend-cpp-5", anchor=_re.escape(_A_GGML_BACKEND_CPP_5), mode="replace", text=_N_GGML_BACKEND_CPP_5,
                 guard=_re.escape('sched->n_moe_cache_entries = 0;'),
                 rationale="upstream #29887 hunk 6 of ggml/src/ggml-backend.cpp.", expect_matches=1, max_span_lines=5),
            Edit(id="moe-cache-ggml-backend-cpp-6", anchor=_re.escape(_A_GGML_BACKEND_CPP_6), mode="replace", text=_N_GGML_BACKEND_CPP_6,
                 guard=_re.escape('// cached experts selected by ids that are already remapped in this split can stay in it'),
                 rationale="upstream #29887 hunk 7 of ggml/src/ggml-backend.cpp.", expect_matches=1, max_span_lines=5),
            Edit(id="moe-cache-ggml-backend-cpp-7", anchor=_re.escape(_A_GGML_BACKEND_CPP_7), mode="replace", text=_N_GGML_BACKEND_CPP_7,
                 guard=_re.escape('struct ggml_backend_sched_moe_cache_entry * cache_entry = NULL;'),
                 rationale="upstream #29887 hunk 8 of ggml/src/ggml-backend.cpp.", expect_matches=1, max_span_lines=5),
            Edit(id="moe-cache-ggml-backend-cpp-8", anchor=_re.escape(_A_GGML_BACKEND_CPP_8), mode="replace", text=_N_GGML_BACKEND_CPP_8,
                 guard=_re.escape('struct ggml_tensor * cached_tensor = NULL;'),
                 rationale="upstream #29887 hunk 9 of ggml/src/ggml-backend.cpp.", expect_matches=1, max_span_lines=16),
            Edit(id="moe-cache-ggml-backend-cpp-9", anchor=_re.escape(_A_GGML_BACKEND_CPP_9), mode="replace", text=_N_GGML_BACKEND_CPP_9,
                 guard=_re.escape('// the cache slots of the experts are written to ids_copy before the split runs'),
                 rationale="upstream #29887 hunk 10 of ggml/src/ggml-backend.cpp.", expect_matches=1, max_span_lines=8),
            Edit(id="moe-cache-ggml-backend-cpp-10", anchor=_re.escape(_A_GGML_BACKEND_CPP_10), mode="replace", text=_N_GGML_BACKEND_CPP_10,
                 guard=_re.escape('int graph_size = std::max(graph->n_nodes, graph->n_leafs) + total_inputs * 2 * sched->n_copies + n_dep_nodes + sched->n_moe_cache_entries;'),
                 rationale="upstream #29887 hunk 11 of ggml/src/ggml-backend.cpp.", expect_matches=1, max_span_lines=6),
            Edit(id="moe-cache-ggml-backend-cpp-11", anchor=_re.escape(_A_GGML_BACKEND_CPP_11), mode="replace", text=_N_GGML_BACKEND_CPP_11,
                 guard=_re.escape('for (int j = 0; j < sched->n_moe_cache_entries; ++j) {'),
                 rationale="upstream #29887 hunk 12 of ggml/src/ggml-backend.cpp.", expect_matches=1, max_span_lines=5),
            Edit(id="moe-cache-ggml-backend-cpp-12", anchor=_re.escape(_A_GGML_BACKEND_CPP_12), mode="replace", text=_N_GGML_BACKEND_CPP_12,
                 guard=_re.escape('std::vector<int32_t> selected_ids;'),
                 rationale="upstream #29887 hunk 13 of ggml/src/ggml-backend.cpp.", expect_matches=1, max_span_lines=2),
            Edit(id="moe-cache-ggml-backend-cpp-13", anchor=_re.escape(_A_GGML_BACKEND_CPP_13), mode="replace", text=_N_GGML_BACKEND_CPP_13,
                 guard=_re.escape('The cache owns this host expert input. Prepare it instead of invoking the ordinary'),
                 rationale="b11474 #29943 extracted ordinary host-weight copying; prepare cache-owned weights in the host-weight pass and delegate all others to the native helper.",
                 expect_matches=1, max_span_lines=6),
            Edit(id="moe-cache-ggml-backend-cpp-15", anchor=_re.escape(_A_GGML_BACKEND_CPP_15), mode="replace", text=_N_GGML_BACKEND_CPP_15,
                 guard=_re.escape('free(sched->moe_cache_entries);'),
                 rationale="upstream #29887 hunk 16 of ggml/src/ggml-backend.cpp.", expect_matches=1, max_span_lines=5),
            Edit(id="moe-cache-ggml-backend-cpp-16", anchor=_re.escape(_A_GGML_BACKEND_CPP_16), mode="replace", text=_N_GGML_BACKEND_CPP_16,
                 guard=_re.escape('void ggml_backend_sched_set_moe_cache('),
                 rationale="Install the MoE-cache setter immediately after b11474\'s native #29943 copy-callback setter.", expect_matches=1, max_span_lines=9),
        ),
    ),
    FilePatch(
        path="include/llama.h",
        description="1337: upstream #29887 MoE expert cache (llama.h)",
        language="none",
        edits=(
            Edit(id="moe-cache-llama-h-0", anchor=_re.escape(_A_LLAMA_H_0), mode="replace", text=_N_LLAMA_H_0,
                 guard=_re.escape('size_t moe_cache_size; // device cache in bytes for the experts kept in host memory, 0 = disabled [EXPERIMENTAL]'),
                 rationale="upstream #29887 hunk 1 of include/llama.h.", expect_matches=1, max_span_lines=3),
        ),
    ),
    FilePatch(
        path="src/CMakeLists.txt",
        description="1337: upstream #29887 MoE expert cache (CMakeLists.txt)",
        language="none",
        edits=(
            Edit(id="moe-cache-cmakelists-txt-0", anchor=_re.escape(_A_CMAKELISTS_TXT_0), mode="replace", text=_N_CMAKELISTS_TXT_0,
                 guard=_re.escape('llama-moe-cache.cpp'),
                 rationale="upstream #29887 hunk 1 of src/CMakeLists.txt.", expect_matches=1, max_span_lines=3),
        ),
    ),
    FilePatch(
        path="src/llama-context.cpp",
        description="1337: upstream #29887 MoE expert cache (llama-context.cpp)",
        language="none",
        edits=(
            Edit(id="moe-cache-llama-context-cpp-0", anchor=_re.escape(_A_LLAMA_CONTEXT_CPP_0), mode="replace", text=_N_LLAMA_CONTEXT_CPP_0,
                 guard=_re.escape('#include "llama-moe-cache.h"'),
                 rationale="upstream #29887 hunk 1 of src/llama-context.cpp.", expect_matches=1, max_span_lines=3),
            Edit(id="moe-cache-llama-context-cpp-1", anchor=_re.escape(_A_LLAMA_CONTEXT_CPP_1), mode="replace", text=_N_LLAMA_CONTEXT_CPP_1,
                 guard=_re.escape('cparams.op_offload     = params.op_offload;'),
                 rationale="upstream #29887 hunk 2 of src/llama-context.cpp.", expect_matches=1, max_span_lines=5),
            Edit(id="moe-cache-llama-context-cpp-2", anchor=_re.escape(_A_LLAMA_CONTEXT_CPP_2), mode="replace", text=_N_LLAMA_CONTEXT_CPP_2,
                 guard=_re.escape('if (cparams.moe_cache_size > 0) {'),
                 rationale="upstream #29887 hunk 3 of src/llama-context.cpp.", expect_matches=1, max_span_lines=3),
            Edit(id="moe-cache-llama-context-cpp-3", anchor=_re.escape(_A_LLAMA_CONTEXT_CPP_3), mode="replace", text=_N_LLAMA_CONTEXT_CPP_3,
                 guard=_re.escape('auto create_sched = [&](bool parallel) {'),
                 rationale="Wrap b11474\'s scheduler creation so every instance retains the native selective-copy callback and installs the MoE-cache callback.", expect_matches=1, max_span_lines=3),
            Edit(id="moe-cache-llama-context-cpp-4", anchor=_re.escape(_A_LLAMA_CONTEXT_CPP_4), mode="replace", text=_N_LLAMA_CONTEXT_CPP_4,
                 guard=_re.escape('create_sched(false);'),
                 rationale="The pipeline-parallel fallback must use the same callback-installing scheduler factory.", expect_matches=1, max_span_lines=3),
            Edit(id="moe-cache-llama-context-cpp-5", anchor=_re.escape(_A_LLAMA_CONTEXT_CPP_5), mode="replace", text=_N_LLAMA_CONTEXT_CPP_5,
                 guard=_re.escape('for (const auto & [buft, size] : moe_cache->memory_breakdown()) {'),
                 rationale="upstream #29887 hunk 6 of src/llama-context.cpp.", expect_matches=1, max_span_lines=3),
            Edit(id="moe-cache-llama-context-cpp-6", anchor=_re.escape(_A_LLAMA_CONTEXT_CPP_6), mode="replace", text=_N_LLAMA_CONTEXT_CPP_6,
                 guard=_re.escape('/*.moe_cache_size              =*/ 0,'),
                 rationale="upstream #29887 hunk 7 of src/llama-context.cpp.", expect_matches=1, max_span_lines=3),
        ),
    ),
    FilePatch(
        path="src/llama-context.h",
        description="1337: upstream #29887 MoE expert cache (llama-context.h)",
        language="none",
        edits=(
            Edit(id="moe-cache-llama-context-h-0", anchor=_re.escape(_A_LLAMA_CONTEXT_H_0), mode="replace", text=_N_LLAMA_CONTEXT_H_0,
                 guard=_re.escape('class llama_moe_cache;'),
                 rationale="upstream #29887 hunk 1 of src/llama-context.h.", expect_matches=1, max_span_lines=3),
            Edit(id="moe-cache-llama-context-h-1", anchor=_re.escape(_A_LLAMA_CONTEXT_H_1), mode="replace", text=_N_LLAMA_CONTEXT_H_1,
                 guard=_re.escape('std::unique_ptr<llama_moe_cache> moe_cache;'),
                 rationale="upstream #29887 hunk 2 of src/llama-context.h.", expect_matches=1, max_span_lines=3),
        ),
    ),
    FilePatch(
        path="src/llama-cparams.h",
        description="1337: upstream #29887 MoE expert cache (llama-cparams.h)",
        language="none",
        edits=(
            Edit(id="moe-cache-llama-cparams-h-0", anchor=_re.escape(_A_LLAMA_CPARAMS_H_0), mode="replace", text=_N_LLAMA_CPARAMS_H_0,
                 guard=_re.escape('size_t moe_cache_size;'),
                 rationale="upstream #29887 hunk 1 of src/llama-cparams.h.", expect_matches=1, max_span_lines=3),
        ),
    ),
    FilePatch(
        path="src/llama-moe-cache.cpp",
        description="1337: upstream #29887 MoE expert cache (new file llama-moe-cache.cpp)",
        language="none",
        create=True,
        edits=(
            Edit(id="moe-cache-create-llama-moe-cache-cpp", anchor='\\A', mode="insert_after", text=_NEW_LLAMA_MOE_CACHE_CPP,
                 guard=_re.escape('// LRU map from (layer, expert) to a cache slot'),
                 rationale="upstream #29887: new file src/llama-moe-cache.cpp.", max_span_lines=1),
        ),
    ),
    FilePatch(
        path="src/llama-moe-cache.h",
        description="1337: upstream #29887 MoE expert cache (new file llama-moe-cache.h)",
        language="none",
        create=True,
        edits=(
            Edit(id="moe-cache-create-llama-moe-cache-h", anchor='\\A', mode="insert_after", text=_NEW_LLAMA_MOE_CACHE_H,
                 guard=_re.escape('// keeps the most recently used experts of host-resident MoE layers in a device buffer'),
                 rationale="upstream #29887: new file src/llama-moe-cache.h.", max_span_lines=1),
        ),
    ),
]
