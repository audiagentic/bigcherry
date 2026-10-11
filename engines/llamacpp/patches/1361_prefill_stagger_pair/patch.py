"""BigCherry 1361: stagger two prompt ubatches across Meta subgraphs."""

from __future__ import annotations

import re

from bigcherry.patcher import Edit, EnvDoc, FilePatch

GROUP = "rdna-boosts"


def _edit(*, id: str, anchor: str, text: str, mode: str, guard: str, rationale: str) -> Edit:
    return Edit(
        id=id,
        anchor=re.escape(anchor),
        mode=mode,
        text=text,
        guard=guard,
        rationale=rationale,
        expect_matches=1,
        max_span_lines=len(anchor.splitlines()) + 1,
    )


# ------------------------------------------------------------------------------------------------- ggml-cuda
_A_CUDA_DISPATCH = r'''// Top-level dispatch -- calls the function pointer chosen by comm_init.
// Returns false to let the meta-backend's butterfly run.
static bool ggml_backend_cuda_comm_allreduce_tensor(void * comm_ctx_v, struct ggml_tensor ** tensors) {
    if (comm_ctx_v == nullptr) {
        return false;
    }
    auto * comm_ctx = static_cast<ggml_backend_cuda_comm_context *>(comm_ctx_v);
    return comm_ctx->try_allreduce(comm_ctx, tensors);
}
'''
_N_CUDA_DISPATCH = r'''// BigCherry 1361 (QFP49): the stagger pair is deliberately RCCL-only.  Admission duplicates the
// production provider decision before either graph starts, so internal/cpu-root/fallback routes never become a
// half-staggered graph.  State is caller-owned because A and B can each have one reduction in flight.
struct bc_1361_ar_state {
#ifdef GGML_USE_NCCL
    bc_ar_transfer_state transfer;
#endif
};

static bool bc_1361_ar_route_rccl(
        ggml_backend_cuda_comm_context * comm_ctx, struct ggml_tensor ** tensors) {
#ifdef GGML_USE_NCCL
    if (comm_ctx == nullptr || tensors == nullptr || tensors[0] == nullptr || comm_ctx->comms.empty()) {
        return false;
    }
    const int64_t ne = ggml_nelements(tensors[0]);
    for (size_t i = 0; i < comm_ctx->backends.size(); ++i) {
        if (tensors[i] == nullptr || tensors[i]->type != GGML_TYPE_F32 ||
                ggml_nelements(tensors[i]) != ne || !ggml_is_contiguously_allocated(tensors[i])) {
            return false;
        }
    }
    if (comm_ctx->try_allreduce == ggml_backend_cuda_comm_try_allreduce_nccl) {
        return true;
    }
#if defined(GGML_USE_HIP)
    if (comm_ctx->try_allreduce == ggml_backend_cuda_comm_try_allreduce_hybrid) {
        const size_t reduction_bytes = ggml_nbytes(tensors[0]);
        const bool have_internal = comm_ctx->ar_pipeline != nullptr;
        const bool prefer_internal = have_internal &&
            (reduction_bytes < comm_ctx->adaptive_switch_bytes || comm_ctx->comms.empty());
        return !prefer_internal;
    }
    if (comm_ctx->try_allreduce == ggml_backend_cuda_comm_try_allreduce_cpu_root) {
        bc_cpu_root * cr = comm_ctx->cpu_root;
        bool small = cr != nullptr && tensors[0]->type == GGML_TYPE_F32 &&
            (size_t) ne <= cr->max_elems;
        for (size_t i = 0; small && i < comm_ctx->backends.size(); ++i) {
            small = tensors[i] != nullptr && ggml_is_contiguous(tensors[i]) &&
                ggml_nelements(tensors[i]) == ne && tensors[i]->type == GGML_TYPE_F32;
        }
        bool large = !small && cr != nullptr && cr->l_max_elems > 0 && tensors[0]->type == GGML_TYPE_F32 &&
            (size_t) ne <= cr->l_max_elems && ne > 0 && (int) comm_ctx->backends.size() == cr->n;
        for (size_t i = 0; large && i < comm_ctx->backends.size(); ++i) {
            large = tensors[i] != nullptr && ggml_is_contiguous(tensors[i]) &&
                ggml_nelements(tensors[i]) == ne && tensors[i]->type == GGML_TYPE_F32;
        }
        return !small && !large;
    }
#endif // GGML_USE_HIP
    return false;
#else
    GGML_UNUSED(comm_ctx);
    GGML_UNUSED(tensors);
    return false;
#endif // GGML_USE_NCCL
}

static void * bc_1361_ar_state_new() {
    return new bc_1361_ar_state;
}

static void bc_1361_ar_state_free(void * state_v) {
    delete static_cast<bc_1361_ar_state *>(state_v);
}

static bool bc_1361_ar_can_begin(void * comm_ctx_v, struct ggml_tensor ** tensors) {
#ifdef GGML_USE_NCCL
    auto * comm_ctx = static_cast<ggml_backend_cuda_comm_context *>(comm_ctx_v);
    if (!bc_1361_ar_route_rccl(comm_ctx, tensors)) {
        return false;
    }
    int free_slots = 0;
    for (int i = 0; i < 2; ++i) {
        free_slots += !comm_ctx->bc_ar_transfer_busy[i];
    }
    return ggml_nelements(tensors[0]) == 0 || free_slots >= 2;
#else
    GGML_UNUSED(comm_ctx_v);
    GGML_UNUSED(tensors);
    return false;
#endif
}

static bool bc_1361_ar_begin(void * comm_ctx_v, struct ggml_tensor ** tensors, void * state_v) {
#ifdef GGML_USE_NCCL
    auto * comm_ctx = static_cast<ggml_backend_cuda_comm_context *>(comm_ctx_v);
    auto * state = static_cast<bc_1361_ar_state *>(state_v);
    return state != nullptr && bc_1361_ar_route_rccl(comm_ctx, tensors) &&
        bc_1360_ar_transfer_begin(comm_ctx, tensors, state->transfer);
#else
    GGML_UNUSED(comm_ctx_v);
    GGML_UNUSED(tensors);
    GGML_UNUSED(state_v);
    return false;
#endif
}

static void bc_1361_ar_end(void * state_v) {
#ifdef GGML_USE_NCCL
    auto * state = static_cast<bc_1361_ar_state *>(state_v);
    GGML_ASSERT(state != nullptr);
    bc_1360_ar_transfer_end(state->transfer);
#else
    GGML_UNUSED(state_v);
#endif
}

// Top-level dispatch -- calls the function pointer chosen by comm_init.
// Returns false to let the meta-backend's butterfly run.
static bool ggml_backend_cuda_comm_allreduce_tensor(void * comm_ctx_v, struct ggml_tensor ** tensors) {
    if (comm_ctx_v == nullptr) {
        return false;
    }
    auto * comm_ctx = static_cast<ggml_backend_cuda_comm_context *>(comm_ctx_v);
    return comm_ctx->try_allreduce(comm_ctx, tensors);
}
'''

_A_CUDA_REG = r'''    if (strcmp(name, "ggml_backend_comm_allreduce_tensor") == 0) {
        return (void *)ggml_backend_cuda_comm_allreduce_tensor;
    }
'''
_N_CUDA_REG = _A_CUDA_REG + r'''    if (strcmp(name, "ggml_backend_comm_allreduce_stagger_state_new") == 0) {
        return (void *)bc_1361_ar_state_new;
    }
    if (strcmp(name, "ggml_backend_comm_allreduce_stagger_state_free") == 0) {
        return (void *)bc_1361_ar_state_free;
    }
    if (strcmp(name, "ggml_backend_comm_allreduce_stagger_can_begin") == 0) {
        return (void *)bc_1361_ar_can_begin;
    }
    if (strcmp(name, "ggml_backend_comm_allreduce_stagger_begin") == 0) {
        return (void *)bc_1361_ar_begin;
    }
    if (strcmp(name, "ggml_backend_comm_allreduce_stagger_end") == 0) {
        return (void *)bc_1361_ar_end;
    }
'''


# --------------------------------------------------------------------------------------------- Meta backend
_A_META_CTX_DECL = r'''struct ggml_backend_meta_context {
'''
_N_META_CTX_DECL = r'''struct bc_1361_graph_set;

struct ggml_backend_meta_context {
'''

_A_META_CONTEXT = r'''    uint64_t                    uid           = 0;

    void *                               comm_ctx       = nullptr;
'''
_N_META_CONTEXT = r'''    uint64_t                    uid           = 0;

    // BigCherry 1361 (QFP49): transient target used only while materialising one half into its own graph metadata.
    struct bc_1361_graph_set *           bc_pair_capture = nullptr;

    void *                               comm_ctx       = nullptr;
'''


_A_META_REBUILD = r'''    const bool needs_rebuild = (cgraph->uid == 0) || (cgraph->uid != backend_ctx->uid);
'''
_N_META_REBUILD = r'''    const bool needs_rebuild = backend_ctx->bc_pair_capture != nullptr ||
        (cgraph->uid == 0) || (cgraph->uid != backend_ctx->uid);
'''

_A_META_GRAPH_FN = r'''static enum ggml_status ggml_backend_meta_graph_compute(ggml_backend_t backend, struct ggml_cgraph * cgraph) {
    GGML_ASSERT(cgraph->grads == nullptr);
'''
_N_META_GRAPH_FN = r'''struct bc_1361_graph_set {
    ggml_context_ptr ctx;
    std::vector<std::vector<ggml_cgraph *>> cgraphs;
    // Dedicated per-half fallback scratch. Accepted pairs never execute it because every reduction preflights RCCL.
    std::vector<std::vector<ggml_tensor *>> nodes_aux;
    std::vector<std::vector<ggml_cgraph *>> cgraphs_aux;
    std::vector<std::vector<ggml_backend_buffer_ptr>> tmp_bufs;
    size_t n_subgraphs = 0;
};

static enum ggml_status ggml_backend_meta_graph_compute(ggml_backend_t backend, struct ggml_cgraph * cgraph) {
    GGML_ASSERT(cgraph->grads == nullptr);
'''

_A_META_AUX = r'''    size_t iga = 0; // i graph aux
    size_t ina = 0; // i node aux
'''
_N_META_AUX = r'''    // BigCherry 1361: pair capture stops before aux/fallback execution.  Graph metadata is duplicated because
    // backend_ctx->cgraphs is the single normal working set and B's rebuild would otherwise overwrite A's views.
    if (backend_ctx->bc_pair_capture != nullptr) {
        auto & out = *backend_ctx->bc_pair_capture;
        out.n_subgraphs = backend_ctx->n_subgraphs;
        const size_t mem_size = std::max<size_t>(4096,
            n_backends * out.n_subgraphs * ggml_graph_overhead_custom(std::max(1, backend_ctx->max_nnodes), false) + 4096);
        const ggml_init_params params = {
            /*.mem_size   =*/ mem_size,
            /*.mem_buffer =*/ nullptr,
            /*.no_alloc   =*/ true,
        };
        out.ctx.reset(ggml_init(params));
        if (!out.ctx) {
            return GGML_STATUS_ALLOC_FAILED;
        }
        out.cgraphs.assign(n_backends, {});
        out.nodes_aux.assign(n_backends, {});
        out.cgraphs_aux.assign(n_backends, {});
        out.tmp_bufs.resize(n_backends);
        for (size_t j = 0; j < n_backends; ++j) {
            out.cgraphs[j].reserve(out.n_subgraphs);
            for (size_t i = 0; i < out.n_subgraphs; ++i) {
                out.cgraphs[j].push_back(ggml_graph_dup(out.ctx.get(), backend_ctx->backend_configs[j].cgraphs[i].cgraph_main, false));
            }
        }
        return GGML_STATUS_SUCCESS;
    }

    size_t iga = 0; // i graph aux
    size_t ina = 0; // i node aux
'''

_A_META_IFACE = r'''static const ggml_backend_i ggml_backend_meta_i = {
'''
_N_META_IFACE = r'''// BigCherry 1361 (QFP49): materialise A and B into independent graph metadata, preflight every reduction,
// then issue the proven A-compute/A-begin/B-compute/B-begin schedule.  false means unsupported before model compute.
bool ggml_backend_meta_graph_compute_pair(
        ggml_backend_t backend, struct ggml_cgraph * cgraph_a, struct ggml_cgraph * cgraph_b, ggml_status * status_out,
        ggml_status (*prepare_half)(void *, int), void * prepare_ctx,
        int64_t * host_capture_us, int64_t * host_rehome_us, int64_t * host_submit_us, size_t * host_sums) {
    GGML_ASSERT(ggml_backend_is_meta(backend));
    if (host_capture_us) *host_capture_us = 0;
    if (host_rehome_us) *host_rehome_us = 0;
    if (host_submit_us)  *host_submit_us  = 0;
    if (host_sums)       *host_sums       = 0;
    auto * backend_ctx = static_cast<ggml_backend_meta_context *>(backend->context);
    const size_t n_backends = backend_ctx->backend_configs.size();
    if (status_out) {
        *status_out = GGML_STATUS_SUCCESS;
    }
    auto decline = [](const char * why) {
        static std::set<std::string> seen;
        if (seen.insert(why).second) GGML_LOG_INFO("BIGCHERRY_PREFILL_STAGGER declined: %s\n", why);
        return false;
    };
    if (n_backends < 2 || backend_ctx->comm_ctx == nullptr) return decline("Meta communication unavailable");

    bc_1361_graph_set sets[2];

    // Both schedulers allocate their views before either capture.  Capturing A rotates shared static Meta
    // buffers to the other compute container, so B's preallocated views are no longer in the current one.
    // Re-home only those external views into A's current container before B rotates away from it.  This
    // preserves A's captured mappings while giving B a valid source set for its rebuild.
    auto rehome_static_views = [](ggml_cgraph * graph) -> ggml_status {
        auto rehome_one = [](ggml_tensor * tensor) -> ggml_status {
            if (tensor == nullptr || tensor->view_src == nullptr || !ggml_backend_buffer_is_meta(tensor->buffer)) {
                return GGML_STATUS_SUCCESS;
            }
            auto * buf_ctx = static_cast<ggml_backend_meta_buffer_context *>(tensor->buffer->context);
            if (buf_ctx->stc_static.simple_tensors.empty() ||
                    buf_ctx->stc_static.simple_tensors.find(tensor) != buf_ctx->stc_static.simple_tensors.end()) {
                return GGML_STATUS_SUCCESS;
            }
            auto & stc = buf_ctx->stc_compute[buf_ctx->stc_compute_index];
            if (stc.simple_tensors.find(tensor) != stc.simple_tensors.end()) {
                return GGML_STATUS_SUCCESS;
            }

            std::vector<ggml_tensor *> chain;
            ggml_tensor * cur = tensor;
            while (cur != nullptr && cur->buffer == tensor->buffer && cur->view_src != nullptr &&
                    buf_ctx->stc_static.simple_tensors.find(cur) == buf_ctx->stc_static.simple_tensors.end() &&
                    stc.simple_tensors.find(cur) == stc.simple_tensors.end()) {
                chain.push_back(cur);
                cur = cur->view_src;
            }
            if (cur != nullptr && cur->buffer == tensor->buffer &&
                    buf_ctx->stc_static.simple_tensors.find(cur) == buf_ctx->stc_static.simple_tensors.end() &&
                    stc.simple_tensors.find(cur) == stc.simple_tensors.end()) {
                return GGML_STATUS_FAILED;
            }
            for (auto it = chain.rbegin(); it != chain.rend(); ++it) {
                const ggml_status st = ggml_backend_meta_buffer_init_tensor_impl(stc, *it);
                if (st != GGML_STATUS_SUCCESS) {
                    return st;
                }
            }
            return GGML_STATUS_SUCCESS;
        };

        for (int i = 0; i < graph->n_leafs; ++i) {
            const ggml_status st = rehome_one(graph->leafs[i]);
            if (st != GGML_STATUS_SUCCESS) return st;
        }
        for (int i = 0; i < graph->n_nodes; ++i) {
            const ggml_status st = rehome_one(graph->nodes[i]);
            if (st != GGML_STATUS_SUCCESS) return st;
        }
        return GGML_STATUS_SUCCESS;
    };

    const int64_t bc_capture_a_t0 = host_capture_us ? ggml_time_us() : 0;
    backend_ctx->bc_pair_capture = &sets[0];
    ggml_status status = ggml_backend_meta_graph_compute(backend, cgraph_a);
    backend_ctx->bc_pair_capture = nullptr;
    if (host_capture_us) *host_capture_us += ggml_time_us() - bc_capture_a_t0;
    if (status != GGML_STATUS_SUCCESS) {
        if (status_out) *status_out = status;
        return false;
    }

    // Scheduler copies are graph-external operations. Meta's rotating compute tensor containers only guarantee
    // that such an operation resolves the simple tensors for the graph that is current now. Snapshot A while
    // A's just-captured mapping is current; waiting until after B capture can resolve A through a later mapping.
    if (prepare_half != nullptr) {
        status = prepare_half(prepare_ctx, 0);
        if (status != GGML_STATUS_SUCCESS) {
            if (status_out) *status_out = status;
            return true;
        }
    }

    const int64_t bc_rehome_t0 = host_rehome_us ? ggml_time_us() : 0;
    status = rehome_static_views(cgraph_b);
    if (host_rehome_us) *host_rehome_us += ggml_time_us() - bc_rehome_t0;
    if (status != GGML_STATUS_SUCCESS) {
        if (status_out) *status_out = status;
        return false;
    }

    const int64_t bc_capture_b_t0 = host_capture_us ? ggml_time_us() : 0;
    backend_ctx->bc_pair_capture = &sets[1];
    status = ggml_backend_meta_graph_compute(backend, cgraph_b);
    backend_ctx->bc_pair_capture = nullptr;
    if (host_capture_us) *host_capture_us += ggml_time_us() - bc_capture_b_t0;
    if (status != GGML_STATUS_SUCCESS) {
        if (status_out) *status_out = status;
        return false;
    }
    if (prepare_half != nullptr) {
        status = prepare_half(prepare_ctx, 1);
        if (status != GGML_STATUS_SUCCESS) {
            if (status_out) *status_out = status;
            return true;
        }
    }
    if (sets[0].n_subgraphs == 0 || sets[1].n_subgraphs == 0) {
        return decline("Meta subgraphs missing");
    }

    // A prompt pair is intentionally asymmetric at the output end: A usually requests no logits while B carries
    // the last-token logits head. Pair only the matching reduction-producing prefix, then execute each half's tail
    // A-before-B. n_kv/mask dimensions may differ. Per call we compare only subgraph node counts and boundary ops:
    // scanning every internal node was redundant host work because the cross-half contract is at reduction boundaries.
    auto subgraph_ops_equal = [&](size_t i, size_t * diff_backend, size_t * diff_node) {
        for (size_t j = 0; j < n_backends; ++j) {
            ggml_cgraph * ga = sets[0].cgraphs[j][i];
            ggml_cgraph * gb = sets[1].cgraphs[j][i];
            if (ga->n_nodes != gb->n_nodes) {
                if (diff_backend) *diff_backend = j;
                if (diff_node) *diff_node = (size_t) std::min(ga->n_nodes, gb->n_nodes);
                return false;
            }
            if (ga->n_nodes > 0 && ga->nodes[ga->n_nodes - 1]->op != gb->nodes[gb->n_nodes - 1]->op) {
                if (diff_backend) *diff_backend = j;
                if (diff_node) *diff_node = (size_t) ga->n_nodes - 1;
                return false;
            }
        }
        return true;
    };

    const size_t common_subgraphs = std::min(sets[0].n_subgraphs, sets[1].n_subgraphs);
    size_t first_diff = common_subgraphs;
    size_t diff_backend = 0;
    size_t diff_node = 0;
    for (size_t i = 0; i < common_subgraphs; ++i) {
        size_t db = 0, dn = 0;
        if (!subgraph_ops_equal(i, &db, &dn)) {
            first_diff = i;
            diff_backend = db;
            diff_node = dn;
            break;
        }
    }
    if (first_diff == common_subgraphs && sets[0].n_subgraphs != sets[1].n_subgraphs) {
        first_diff = common_subgraphs;
    }

    const size_t reductions_a = sets[0].n_subgraphs - 1;
    const size_t reductions_b = sets[1].n_subgraphs - 1;
    const size_t max_pair_reductions = std::min(reductions_a, reductions_b);
    const size_t paired_reductions = std::min(first_diff, max_pair_reductions);
    if (host_sums) *host_sums = reductions_a + reductions_b;

    auto subgraph_desc = [&](int h, size_t i) {
        if (i >= sets[h].n_subgraphs) return std::string("none");
        ggml_cgraph * g = sets[h].cgraphs[std::min(diff_backend, n_backends - 1)][i];
        std::string out = "nodes=" + std::to_string(g->n_nodes);
        if (g->n_nodes > 0) {
            const size_t k = std::min(diff_node, (size_t) g->n_nodes - 1);
            out += " op[" + std::to_string(k) + "]=" + ggml_op_name(g->nodes[k]->op);
        }
        return out;
    };
    if (first_diff < common_subgraphs || sets[0].n_subgraphs != sets[1].n_subgraphs) {
        const std::string why =
            "subgraphs A=" + std::to_string(sets[0].n_subgraphs) +
            " B=" + std::to_string(sets[1].n_subgraphs) +
            " first_diff=" + std::to_string(first_diff) +
            " A(" + subgraph_desc(0, first_diff) + ")" +
            " B(" + subgraph_desc(1, first_diff) + ")" +
            " paired_reductions=" + std::to_string(paired_reductions);
        static std::set<std::string> shape_seen;
        if (shape_seen.insert(why).second) {
            GGML_LOG_INFO("BIGCHERRY_PREFILL_STAGGER shape-tail: %s\n", why.c_str());
        }
        if (paired_reductions == 0) {
            return decline(why.c_str());
        }
    }
    if (paired_reductions == 0) {
        return decline("no common reduction-producing Meta prefix");
    }

    ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(ggml_backend_get_device(backend_ctx->backend_configs[0].backend));
    using state_new_t  = void *(*)();
    using state_free_t = void (*)(void *);
    using can_begin_t  = bool (*)(void *, ggml_tensor **);
    using begin_t      = bool (*)(void *, ggml_tensor **, void *);
    using end_t        = void (*)(void *);
    auto state_new  = (state_new_t)  ggml_backend_reg_get_proc_address(reg, "ggml_backend_comm_allreduce_stagger_state_new");
    auto state_free = (state_free_t) ggml_backend_reg_get_proc_address(reg, "ggml_backend_comm_allreduce_stagger_state_free");
    auto can_begin  = (can_begin_t)  ggml_backend_reg_get_proc_address(reg, "ggml_backend_comm_allreduce_stagger_can_begin");
    auto begin      = (begin_t)      ggml_backend_reg_get_proc_address(reg, "ggml_backend_comm_allreduce_stagger_begin");
    auto end        = (end_t)        ggml_backend_reg_get_proc_address(reg, "ggml_backend_comm_allreduce_stagger_end");
    if (!state_new || !state_free || !can_begin || !begin || !end) {
        return decline("RCCL stagger procs unavailable");
    }

    auto nodes_for = [&](bc_1361_graph_set & set, size_t i) {
        std::vector<ggml_tensor *> nodes;
        nodes.reserve(n_backends);
        for (size_t j = 0; j < n_backends; ++j) {
            ggml_cgraph * graph = set.cgraphs[j][i];
            GGML_ASSERT(graph->n_nodes > 0);
            nodes.push_back(graph->nodes[graph->n_nodes - 1]);
        }
        return nodes;
    };

    // Provider/shape admission for every reduction happens before either half executes.
    for (int h = 0; h < 2; ++h) {
        for (size_t i = 0; i + 1 < sets[h].n_subgraphs; ++i) {
            auto nodes = nodes_for(sets[h], i);
            if (!can_begin(backend_ctx->comm_ctx, nodes.data())) {
                static std::set<std::string> reductions;
                const std::string why = "reduction=" + std::to_string(i) +
                    " half=" + std::to_string(h) + " route=RCCL-unavailable";
                if (reductions.insert(why).second)
                    GGML_LOG_INFO("BIGCHERRY_PREFILL_STAGGER declined: %s\n", why.c_str());
                return false;
            }
        }
    }

    const int64_t bc_submit_t0 = host_submit_us ? ggml_time_us() : 0;
    void * pending[2] = {nullptr, nullptr};
    auto finish = [&](int h) {
        if (pending[h] != nullptr) {
            end(pending[h]);
            state_free(pending[h]);
            pending[h] = nullptr;
        }
    };
    auto cleanup = [&]() { finish(0); finish(1); };

    auto compute_subgraph = [&](int h, size_t i) {
        for (size_t j = 0; j < n_backends; ++j) {
            status = ggml_backend_graph_compute_async(backend_ctx->backend_configs[j].backend, sets[h].cgraphs[j][i]);
            if (status != GGML_STATUS_SUCCESS) {
                if (status_out) *status_out = status;
                return false;
            }
        }
        return true;
    };
    auto begin_reduction = [&](int h, size_t i, void ** state_out) {
        auto nodes = nodes_for(sets[h], i);
        *state_out = state_new();
        if (*state_out == nullptr || !begin(backend_ctx->comm_ctx, nodes.data(), *state_out)) {
            if (*state_out != nullptr) {
                state_free(*state_out);
                *state_out = nullptr;
            }
            if (status_out) *status_out = GGML_STATUS_FAILED;
            return false;
        }
        return true;
    };

    // Only reduction-producing subgraphs are interleaved. The final/common-output tails are deliberately left for
    // the A-then-B sequential tail below; this admits B's logits head without pretending the two graphs are equal.
    for (size_t i = 0; i < paired_reductions; ++i) {
        for (int h = 0; h < 2; ++h) {
            finish(h);
            if (!compute_subgraph(h, i) || !begin_reduction(h, i, &pending[h])) {
                cleanup();
                return true;
            }
        }
    }
    cleanup();

    // Finish whatever topology is left after the comparable prefix. Reductions here use the same begin/end seam
    // back-to-back: correct but intentionally not overlapped. A's complete tail precedes B's to preserve shared
    // KV/recurrent-state ordering even when the output/logits tail exists only on B.
    auto run_tail = [&](int h) {
        for (size_t i = paired_reductions; i < sets[h].n_subgraphs; ++i) {
            if (!compute_subgraph(h, i)) {
                return false;
            }
            if (i + 1 < sets[h].n_subgraphs) {
                void * state = nullptr;
                if (!begin_reduction(h, i, &state)) {
                    return false;
                }
                end(state);
                state_free(state);
            }
        }
        return true;
    };
    if (!run_tail(0) || !run_tail(1)) {
        if (host_submit_us) *host_submit_us += ggml_time_us() - bc_submit_t0;
        return true;
    }
    if (host_submit_us) *host_submit_us += ggml_time_us() - bc_submit_t0;

    if (status_out) *status_out = GGML_STATUS_SUCCESS;
    return true;
}

static const ggml_backend_i ggml_backend_meta_i = {
'''


# ------------------------------------------------------------------------------------------- second 1340 arena
_A_ARENA_FIELDS = r'''        ggml_gallocr_ptr                    arena_galloc; // BigCherry 1340 (MSM02): sole physical owner
        std::vector<arena_plan_t>            arena_plans; // BigCherry 1340 (MSM02)
        std::vector<ggml_tensor *>           arena_nodes; // persistent: resized/reused for each transformed graph
'''
_N_ARENA_FIELDS = r'''        ggml_gallocr_ptr                    arena_galloc; // BigCherry 1340 (MSM02): sole physical owner
        std::vector<arena_plan_t>            arena_plans; // BigCherry 1340 (MSM02)
        // BigCherry 1361: scheduler B gets an independent physical owner/shape plans while both halves coexist.
        ggml_gallocr_ptr                    arena_galloc_pair;
        std::vector<arena_plan_t>            arena_plans_pair;
        uint64_t                             arena_pair_replans = 0;
        uint64_t                             arena_pair_grows   = 0;
        std::vector<ggml_tensor *>           arena_nodes; // persistent: resized/reused for each transformed graph
'''

_A_ARENA_DTOR = r'''            bc.arena_plans.clear();
            bc.arena_galloc.reset();
            ggml_backend_free(bc.backend);
'''
_N_ARENA_DTOR = r'''            bc.arena_plans.clear();
            bc.arena_galloc.reset();
            bc.arena_plans_pair.clear();
            bc.arena_galloc_pair.reset();
            ggml_backend_free(bc.backend);
'''

# A static Meta buffer (KV cache, recurrent state) keeps the views a graph makes of its tensors in one container until
# the next compute rebuild moves on to the other. A pair allocates graph A and graph B before either is computed, so
# both graphs' views land in the same container: twice the views between evals.
_A_META_HEADROOM = r'''    constexpr size_t compute_headroom = 16; // Maximum number of views per statically allocated tensor that can be created between evals.
'''
_N_META_HEADROOM = r'''    // BigCherry 1361: two graphs need twice the view capacity; flag-off keeps upstream's exact 16.
    const char * bc_stagger_env = getenv("BIGCHERRY_PREFILL_STAGGER");
    const size_t compute_headroom = bc_stagger_env != nullptr && strtoul(bc_stagger_env, nullptr, 10) > 0
        ? 32 : 16; // Maximum views per statically allocated tensor between evals.
'''

_A_ARENA_ROTATE = r'''// BigCherry 1340 (MSM02): what graph_compute does when it rebuilds - move the Meta buffers a graph uses on to their
'''
_N_ARENA_ROTATE = r'''// BigCherry 1361 (QFP49): reserve/bind scheduler B into a second 1340 physical arena.  Mapping scratch vectors
// are shared because A/B reserve and allocation run serially on the existing host thread; physical owners and plans are not.
static size_t bc_1361_arena_find_plan(
        const std::vector<ggml_backend_meta_context::backend_config::arena_plan_t> & plans,
        const struct ggml_cgraph & cgraph) {
    const uint64_t out_sig = ggml_backend_meta_arena_out_sig(cgraph);
    size_t i = 0;
    while (i < plans.size() && (plans[i].n_nodes != cgraph.n_nodes || plans[i].n_leafs != cgraph.n_leafs ||
            plans[i].out_sig != out_sig)) {
        ++i;
    }
    return i;
}

bool ggml_backend_meta_reserve_graph_pair(ggml_backend_t meta_backend, struct ggml_cgraph * cgraph) {
    GGML_ASSERT(ggml_backend_is_meta(meta_backend));
    if (!ggml_backend_meta_per_device_arena_enabled()) return true;
    ggml_backend_meta_arena_map_graph(meta_backend, cgraph);
    auto * backend_ctx = static_cast<ggml_backend_meta_context *>(meta_backend->context);
    for (size_t j = 0; j < backend_ctx->backend_configs.size(); ++j) {
        auto & bc = backend_ctx->backend_configs[j];
        ggml_cgraph simple = *cgraph;
        simple.nodes = bc.arena_nodes.data();
        simple.leafs = bc.arena_leafs.data();
        size_t p = bc_1361_arena_find_plan(bc.arena_plans_pair, simple);
        if (p == bc.arena_plans_pair.size()) {
            bc.arena_plans_pair.emplace_back();
            p = bc.arena_plans_pair.size() - 1;
            bc.arena_plans_pair[p].n_nodes = simple.n_nodes;
            bc.arena_plans_pair[p].n_leafs = simple.n_leafs;
            bc.arena_plans_pair[p].out_sig = ggml_backend_meta_arena_out_sig(simple);
            bc.arena_plans_pair[p].galloc.reset(ggml_gallocr_new(ggml_backend_get_default_buffer_type(bc.backend)));
        }
        if (ggml_gallocr_needs_realloc(bc.arena_plans_pair[p].galloc.get(), &simple)) {
            size_t planned = 0;
            ggml_gallocr_reserve_n_size(bc.arena_plans_pair[p].galloc.get(), &simple, nullptr, nullptr, &planned);
            (void) planned;
        }
        if (!bc.arena_galloc_pair) {
            bc.arena_galloc_pair.reset(ggml_gallocr_new(ggml_backend_get_default_buffer_type(bc.backend)));
        }
        if (!ggml_gallocr_reserve_grow(bc.arena_galloc_pair.get(), &simple)) return false;
        GGML_LOG_INFO("BIGCHERRY_PREFILL_STAGGER reserve_b dev=%zu size_mib=%.2f\n", j,
            ggml_gallocr_get_buffer_size(bc.arena_galloc_pair.get(), 0) / 1024.0 / 1024.0);
    }
    return true;
}

bool ggml_backend_meta_alloc_graph_pair(ggml_backend_t meta_backend, struct ggml_cgraph * cgraph) {
    GGML_ASSERT(ggml_backend_is_meta(meta_backend));
    if (!ggml_backend_meta_per_device_arena_enabled()) return true;
    ggml_backend_meta_arena_map_graph(meta_backend, cgraph);
    auto * backend_ctx = static_cast<ggml_backend_meta_context *>(meta_backend->context);
    for (size_t j = 0; j < backend_ctx->backend_configs.size(); ++j) {
        auto & bc = backend_ctx->backend_configs[j];
        ggml_cgraph simple = *cgraph;
        simple.nodes = bc.arena_nodes.data();
        simple.leafs = bc.arena_leafs.data();
        size_t p = bc_1361_arena_find_plan(bc.arena_plans_pair, simple);
        bool nonfit = p == bc.arena_plans_pair.size() || !bc.arena_galloc_pair;
        if (!nonfit && ggml_gallocr_needs_realloc(bc.arena_plans_pair[p].galloc.get(), &simple)) nonfit = true;
        if (nonfit) {
            const size_t before = bc.arena_galloc_pair ? ggml_gallocr_get_buffer_size(bc.arena_galloc_pair.get(), 0) : 0;
            bc.arena_pair_replans++;
            if (p == bc.arena_plans_pair.size()) {
                bc.arena_plans_pair.emplace_back();
                p = bc.arena_plans_pair.size() - 1;
                bc.arena_plans_pair[p].n_nodes = simple.n_nodes;
                bc.arena_plans_pair[p].n_leafs = simple.n_leafs;
                bc.arena_plans_pair[p].out_sig = ggml_backend_meta_arena_out_sig(simple);
                bc.arena_plans_pair[p].galloc.reset(ggml_gallocr_new(ggml_backend_get_default_buffer_type(bc.backend)));
            }
            size_t planned = 0;
            ggml_gallocr_reserve_n_size(bc.arena_plans_pair[p].galloc.get(), &simple, nullptr, nullptr, &planned);
            (void) planned;
            if (!bc.arena_galloc_pair) {
                bc.arena_galloc_pair.reset(ggml_gallocr_new(ggml_backend_get_default_buffer_type(bc.backend)));
            }
            if (!ggml_gallocr_reserve_grow(bc.arena_galloc_pair.get(), &simple)) return false;
            const size_t after = ggml_gallocr_get_buffer_size(bc.arena_galloc_pair.get(), 0);
            if (after > before) bc.arena_pair_grows++;
        }
        if (!ggml_gallocr_alloc_graph_reuse_from(
                bc.arena_plans_pair[p].galloc.get(), bc.arena_galloc_pair.get(), &simple)) return false;
    }
    return true;
}

// BigCherry 1340 (MSM02): what graph_compute does when it rebuilds - move the Meta buffers a graph uses on to their
'''

# ----------------------------------------------------------------------------------------------- scheduler
_A_SCHED_TYPE = r'''struct ggml_backend_sched {
'''
_N_SCHED_TYPE = r'''struct bc_1361_input_stages;

struct ggml_backend_sched {
'''

_A_SCHED_STRUCT = r'''    bool op_offload;

    int debug;
'''
_N_SCHED_STRUCT = r'''    bool op_offload;
    bool bc_stagger_pair_arena = false; // BigCherry 1361: scheduler B selects 1340's second physical owner.
    // 0 = normal copy path, otherwise final-split input index + 1 while 1361 calls the existing 1326 copier.
    int bc_stagger_input_slot = 0;
    bc_1361_input_stages * bc_stagger_input_stages = nullptr;

    int debug;
'''

_A_BACKEND_DECLS = r'''bool ggml_backend_meta_alloc_graph(ggml_backend_t meta_backend, struct ggml_cgraph * cgraph);
void ggml_backend_meta_rotate_graph_containers(struct ggml_cgraph * cgraph);
'''
_N_BACKEND_DECLS = r'''bool ggml_backend_meta_alloc_graph(ggml_backend_t meta_backend, struct ggml_cgraph * cgraph);
bool ggml_backend_meta_reserve_graph_pair(ggml_backend_t meta_backend, struct ggml_cgraph * cgraph);
bool ggml_backend_meta_alloc_graph_pair(ggml_backend_t meta_backend, struct ggml_cgraph * cgraph);
bool ggml_backend_meta_graph_compute_pair(
    ggml_backend_t backend, struct ggml_cgraph * cgraph_a, struct ggml_cgraph * cgraph_b,
    ggml_status * status_out, ggml_status (*prepare_half)(void *, int), void * prepare_ctx,
    int64_t * host_capture_us, int64_t * host_rehome_us, int64_t * host_submit_us, size_t * host_sums);
void ggml_backend_meta_rotate_graph_containers(struct ggml_cgraph * cgraph);
'''

_A_SCHED_ALLOC = r'''            if (ggml_backend_is_meta(sched->backends[i]) &&
                    !ggml_backend_meta_alloc_graph(sched->backends[i], &sched->graph)) {
                GGML_LOG_ERROR("%s: failed to allocate per-device Meta arena\n", __func__);
                return false;
            }
'''
_N_SCHED_ALLOC = r'''            if (ggml_backend_is_meta(sched->backends[i])) {
                const bool ok = sched->bc_stagger_pair_arena
                    ? ggml_backend_meta_alloc_graph_pair(sched->backends[i], &sched->graph)
                    : ggml_backend_meta_alloc_graph(sched->backends[i], &sched->graph);
                if (!ok) {
                    GGML_LOG_ERROR("%s: failed to allocate per-device Meta arena%s\n", __func__,
                        sched->bc_stagger_pair_arena ? " B" : "");
                    return false;
                }
            }
'''

_A_SCHED_RESERVE = r'''            if (ggml_backend_is_meta(sched->backends[i]) &&
                    !ggml_backend_meta_reserve_graph(sched->backends[i], &sched->graph)) {
                GGML_LOG_ERROR("%s: failed to reserve per-device Meta arena\n", __func__);
                return false;
            }
'''
_N_SCHED_RESERVE = r'''            if (ggml_backend_is_meta(sched->backends[i])) {
                const bool ok = sched->bc_stagger_pair_arena
                    ? ggml_backend_meta_reserve_graph_pair(sched->backends[i], &sched->graph)
                    : ggml_backend_meta_reserve_graph(sched->backends[i], &sched->graph);
                if (!ok) {
                    GGML_LOG_ERROR("%s: failed to reserve per-device Meta arena%s\n", __func__,
                        sched->bc_stagger_pair_arena ? " B" : "");
                    return false;
                }
            }
'''

_A_SCHED_INCLUDES = r'''#include <algorithm>
#include <unordered_map>
#include <vector>
'''
_N_SCHED_INCLUDES = r'''#include <algorithm>
#include <set>
#include <string>
#include <unordered_map>
#include <vector>
'''
# Two edits, not one: production's 1325 (split timing) sits between the function head and its loop.

_A_SCHED_COPY_INPUT = r'''static void ggml_backend_sched_copy_input(ggml_backend_sched_t sched, struct ggml_backend_sched_split * split, struct ggml_tensor * input) {
'''
_N_SCHED_COPY_INPUT = r'''// BigCherry 1361: pinned, two-slot staging for the existing 1326 host-input path. Each slot's events are
// recorded immediately after its H2D copies, before any pair model work is submitted. Reusing a slot can therefore
// wait only for that old H2D, never for a later compute kernel or RCCL sum on the same stream.
struct bc_1361_input_stage_slot {
    ggml_backend_buffer_t buffer = nullptr;
    std::vector<ggml_backend_event_t> done;
    bool pending = false;
};

struct bc_1361_input_stage_entry {
    size_t current_bytes = 0;
    size_t capacity = 0;
    int next = 0;
    bc_1361_input_stage_slot slot[2];
};

struct bc_1361_input_stages {
    std::vector<bc_1361_input_stage_entry> inputs;

    ~bc_1361_input_stages() {
        for (auto & input : inputs) {
            for (auto & slot : input.slot) {
                for (ggml_backend_event_t event : slot.done) {
                    ggml_backend_event_free(event);
                }
                if (slot.buffer != nullptr) {
                    ggml_backend_buffer_free(slot.buffer);
                }
            }
        }
    }
};

static bool bc_1361_stage_eligible(
        ggml_backend_sched_t sched, ggml_backend_sched_split * split,
        ggml_tensor * input, ggml_tensor * input_cpy) {
    ggml_backend_t split_backend = sched->backends[split->backend_id];
    return split_backend->iface.set_tensor_async != nullptr &&
        input->buffer != nullptr && ggml_backend_buffer_is_host(input->buffer) &&
        ggml_backend_buffer_get_usage(input->buffer) != GGML_BACKEND_BUFFER_USAGE_WEIGHTS &&
        ggml_is_contiguous(input) && ggml_is_contiguous(input_cpy) &&
        ggml_nbytes(input) == ggml_nbytes(input_cpy);
}

static std::string bc_1361_stage_reason(
        int input_id, ggml_tensor * input, size_t bytes_now, size_t bytes_before, const char * reason) {
    const char * name = input != nullptr ? ggml_get_name(input) : nullptr;
    return "input=" + std::to_string(input_id) +
        " name=" + std::string(name != nullptr && name[0] != '\0' ? name : "<unnamed>") +
        " bytes_now=" + std::to_string(bytes_now) +
        " bytes_before=" + std::to_string(bytes_before) +
        " reason=" + reason;
}

// Called before Meta pair admission. A growing mask/index input may resize a slot only after that slot's previous
// H2D completion event has been waited. Capacity grows geometrically; there is deliberately no fixed byte ceiling.
static bool bc_1361_prepare_input_stages(
        ggml_backend_sched_t sched, ggml_backend_sched_split * split, std::string & why) {
    ggml_backend_t split_backend = sched->backends[split->backend_id];
    if (!ggml_backend_is_meta(split_backend)) {
        why = "final split is not Meta";
        return false;
    }
    ggml_backend_buffer_type_t host_buft =
        ggml_backend_dev_host_buffer_type(ggml_backend_get_device(split_backend));
    if (host_buft == nullptr) {
        why = "Meta host buffer type unavailable";
        return false;
    }
    if (sched->bc_stagger_input_stages == nullptr) {
        sched->bc_stagger_input_stages = new bc_1361_input_stages();
    }
    auto & stages = sched->bc_stagger_input_stages->inputs;
    if (stages.size() < (size_t) split->n_inputs) {
        stages.resize(split->n_inputs);
    }

    const size_t n_children = ggml_backend_meta_n_backends(split_backend);
    if (n_children == 0) {
        why = "Meta child backends unavailable";
        return false;
    }
    for (int input_id = 0; input_id < split->n_inputs; ++input_id) {
        ggml_tensor * input = split->inputs[input_id];
        auto & entry = stages[input_id];
        const size_t before = entry.current_bytes;
        const size_t bytes = ggml_nbytes(input);
        ggml_tensor * input_cpy = tensor_copy(input, split->backend_id, sched->cur_copy);
        if (input_cpy == nullptr) {
            why = bc_1361_stage_reason(input_id, input, bytes, before, "missing-copy");
            return false;
        }

        const bool host_nonweight = input->buffer != nullptr && ggml_backend_buffer_is_host(input->buffer) &&
            ggml_backend_buffer_get_usage(input->buffer) != GGML_BACKEND_BUFFER_USAGE_WEIGHTS;
        if (!host_nonweight) {
            continue;
        }
        if (!bc_1361_stage_eligible(sched, split, input, input_cpy)) {
            why = bc_1361_stage_reason(input_id, input, bytes, before, "not-async-eligible");
            return false;
        }

        for (auto & slot : entry.slot) {
            if (slot.done.empty()) {
                for (size_t j = 0; j < n_children; ++j) {
                    ggml_backend_t child = ggml_backend_meta_simple_backend(split_backend, j);
                    ggml_backend_event_t event = ggml_backend_event_new(ggml_backend_get_device(child));
                    if (event == nullptr) {
                        why = bc_1361_stage_reason(input_id, input, bytes, before, "event-allocation");
                        return false;
                    }
                    slot.done.push_back(event);
                }
            }
            if (slot.done.size() != n_children) {
                why = bc_1361_stage_reason(input_id, input, bytes, before, "child-count-changed");
                return false;
            }
        }

        if (bytes > entry.capacity) {
            size_t new_capacity = entry.capacity == 0 ? bytes : entry.capacity;
            while (new_capacity < bytes) {
                new_capacity = new_capacity > bytes / 2 ? bytes : 2 * new_capacity;
            }

            for (auto & slot : entry.slot) {
                if (slot.pending) {
                    for (ggml_backend_event_t event : slot.done) {
                        ggml_backend_event_synchronize(event);
                    }
                    slot.pending = false;
                }
                if (slot.buffer != nullptr) {
                    ggml_backend_buffer_free(slot.buffer);
                    slot.buffer = nullptr;
                }
                slot.buffer = ggml_backend_buft_alloc_buffer(host_buft, new_capacity);
                if (slot.buffer == nullptr) {
                    why = bc_1361_stage_reason(input_id, input, bytes, before, "host-stage-allocation");
                    return false;
                }
            }
            entry.capacity = new_capacity;
        }
        entry.current_bytes = bytes;
    }
    return true;
}

static void bc_1361_copy_input_pinned(
        ggml_backend_sched_t sched, ggml_backend_sched_split * split,
        ggml_tensor * input, ggml_tensor * input_cpy) {
    GGML_ASSERT(sched->bc_stagger_input_slot > 0);
    const int input_id = sched->bc_stagger_input_slot - 1;
    GGML_ASSERT(sched->bc_stagger_input_stages != nullptr);
    GGML_ASSERT((size_t) input_id < sched->bc_stagger_input_stages->inputs.size());

    auto & entry = sched->bc_stagger_input_stages->inputs[input_id];
    const size_t bytes = ggml_nbytes(input);
    GGML_ASSERT(entry.current_bytes == bytes);
    auto & slot = entry.slot[entry.next];
    GGML_ASSERT(slot.buffer != nullptr && ggml_backend_buffer_get_size(slot.buffer) >= bytes);
    entry.next ^= 1;

    // This fence is behind the old H2D and ahead of that call's model compute. It cannot wait for an old pair sum.
    if (slot.pending) {
        for (ggml_backend_event_t event : slot.done) {
            ggml_backend_event_synchronize(event);
        }
    }

    void * stage = ggml_backend_buffer_get_base(slot.buffer);
    memcpy(stage, input->data, bytes);
    ggml_backend_t split_backend = sched->backends[split->backend_id];
    ggml_backend_tensor_set_async(split_backend, input_cpy, stage, 0, bytes);

    GGML_ASSERT(ggml_backend_is_meta(split_backend));
    GGML_ASSERT(slot.done.size() == ggml_backend_meta_n_backends(split_backend));
    for (size_t j = 0; j < slot.done.size(); ++j) {
        ggml_backend_t child = ggml_backend_meta_simple_backend(split_backend, j);
        ggml_backend_event_record(slot.done[j], child);
    }
    slot.pending = true;
}

static void ggml_backend_sched_copy_input(ggml_backend_sched_t sched, struct ggml_backend_sched_split * split, struct ggml_tensor * input) {
'''

_A_1326_GATE = r'''    if (bc_async_inputs && split_backend->iface.set_tensor_async != nullptr && input->buffer != nullptr &&
            ggml_backend_buffer_is_host(input->buffer) &&
            ggml_backend_buffer_get_usage(input->buffer) != GGML_BACKEND_BUFFER_USAGE_WEIGHTS &&
            ggml_is_contiguous(input) && ggml_is_contiguous(input_cpy) &&
            ggml_nbytes(input) == ggml_nbytes(input_cpy) &&
            // decode-sized inputs only: staging a large prefill input (KQ mask ~10 MB per ubatch) through
            // pageable memory costs more than upstream's synchronous copy (v5c: prefill -2%)
            ggml_nbytes(input) <= ((size_t) 4 << 20)) {
'''
_N_1326_GATE = r'''    if ((sched->bc_stagger_input_slot > 0 || bc_async_inputs) &&
            split_backend->iface.set_tensor_async != nullptr && input->buffer != nullptr &&
            ggml_backend_buffer_is_host(input->buffer) &&
            ggml_backend_buffer_get_usage(input->buffer) != GGML_BACKEND_BUFFER_USAGE_WEIGHTS &&
            ggml_is_contiguous(input) && ggml_is_contiguous(input_cpy) &&
            ggml_nbytes(input) == ggml_nbytes(input_cpy) &&
            // Keep 1326's 4 MiB policy exactly when not in a 1361 pair. Pair mode owns pinned lifetime/fences
            // and may stage the growing KQ/QSA/indexer inputs at their real size.
            (sched->bc_stagger_input_slot > 0 || ggml_nbytes(input) <= ((size_t) 4 << 20))) {
'''

_A_1326_STAGE = r'''        static thread_local std::unordered_map<const ggml_tensor *, std::vector<uint8_t>> bc_staging;
        std::vector<uint8_t> & bc_stage = bc_staging[input_cpy];
        bc_stage.resize(ggml_nbytes(input));
        memcpy(bc_stage.data(), input->data, ggml_nbytes(input));
        ggml_backend_tensor_set_async(split_backend, input_cpy, bc_stage.data(), 0, ggml_nbytes(input));
        return;
'''
_N_1326_STAGE = r'''        if (sched->bc_stagger_input_slot > 0) {
            // 1361 pair path: pinned two-slot staging; lifetime is fenced by events immediately after H2D.
            bc_1361_copy_input_pinned(sched, split, input, input_cpy);
            return;
        }
        static thread_local std::unordered_map<const ggml_tensor *, std::vector<uint8_t>> bc_staging;
        std::vector<uint8_t> & bc_stage = bc_staging[input_cpy];
        bc_stage.resize(ggml_nbytes(input));
        memcpy(bc_stage.data(), input->data, ggml_nbytes(input));
        ggml_backend_tensor_set_async(split_backend, input_cpy, bc_stage.data(), 0, ggml_nbytes(input));
        return;
'''

_A_SCHED_FREE = r'''    free(sched->graph.leafs);
    free(sched);
}
'''
_N_SCHED_FREE = r'''    free(sched->graph.leafs);
    delete sched->bc_stagger_input_stages;
    sched->bc_stagger_input_stages = nullptr;
    free(sched);
}
'''

_A_SCHED_PREFIX_START = r'''static enum ggml_status ggml_backend_sched_compute_splits(ggml_backend_sched_t sched) {
    GGML_ASSERT(sched);
'''
_N_SCHED_PREFIX_START = r'''static enum ggml_status ggml_backend_sched_compute_splits_range(ggml_backend_sched_t sched, int end_split) {
    GGML_ASSERT(sched);
'''
_A_SCHED_PREFIX_LOOP = r'''    for (int split_id = 0; split_id < sched->n_splits; split_id++) {
        struct ggml_backend_sched_split * split = &splits[split_id];
'''
_N_SCHED_PREFIX_LOOP = r'''    for (int split_id = 0; split_id < end_split; split_id++) {  // BigCherry 1361: the pair runs every split but the last here
        struct ggml_backend_sched_split * split = &splits[split_id];
'''
_A_SCHED_PREFIX_END = r'''    return GGML_STATUS_SUCCESS;
}

ggml_backend_sched_t ggml_backend_sched_new(
'''
_N_SCHED_PREFIX_END = r'''    return GGML_STATUS_SUCCESS;
}
static enum ggml_status ggml_backend_sched_compute_splits(ggml_backend_sched_t sched) {
    return ggml_backend_sched_compute_splits_range(sched, sched->n_splits);
}
static enum ggml_status ggml_backend_sched_compute_splits_prefix(ggml_backend_sched_t sched) {
    return ggml_backend_sched_compute_splits_range(sched, sched->n_splits - 1);
}

ggml_backend_sched_t ggml_backend_sched_new(
'''

_A_SCHED_API_SITE = r'''void ggml_backend_sched_set_eval_callback(ggml_backend_sched_t sched, ggml_backend_sched_eval_callback callback, void * user_data) {
'''
_N_SCHED_API_SITE = r'''void ggml_backend_sched_set_stagger_pair_arena(ggml_backend_sched_t sched, bool enabled) {
    GGML_ASSERT(sched);
    GGML_ASSERT(!sched->is_alloc);
    sched->bc_stagger_pair_arena = enabled;
}

enum ggml_status ggml_backend_sched_graph_compute_async_pair(
        ggml_backend_sched_t sched_a, struct ggml_cgraph * graph_a,
        ggml_backend_sched_t sched_b, struct ggml_cgraph * graph_b,
        bool * supported) {
    GGML_ASSERT(sched_a && sched_b && supported);
    *supported = false;
    GGML_UNUSED(graph_a);
    GGML_UNUSED(graph_b);

    static const bool bc_timing_on = getenv("BIGCHERRY_PREFILL_STAGGER_TIMING") != nullptr &&
        atoi(getenv("BIGCHERRY_PREFILL_STAGGER_TIMING")) != 0;
    const int64_t bc_total_t0 = bc_timing_on ? ggml_time_us() : 0;
    int64_t bc_prefix_us = 0;
    int64_t bc_stage_us = 0;
    int64_t bc_capture_us = 0;
    int64_t bc_rehome_us = 0;
    int64_t bc_submit_us = 0;
    size_t bc_staged_bytes = 0;
    size_t bc_sums = 0;

    auto decline = [](const char * why) {
        static std::set<std::string> seen;
        if (seen.insert(why).second) GGML_LOG_INFO("BIGCHERRY_PREFILL_STAGGER declined: %s\n", why);
        return GGML_STATUS_SUCCESS;
    };
    if (!sched_a->is_alloc || !sched_b->is_alloc) return decline("scheduler unallocated");
    if (sched_a->callback_eval || sched_b->callback_eval) return decline("evaluation callback");
    if (sched_a->n_splits < 1 || sched_a->n_splits != sched_b->n_splits) return decline("different split counts");
    for (int i = 0; i < sched_a->n_splits; ++i) {
        const auto * sa = &sched_a->splits[i];
        const auto * sb = &sched_b->splits[i];
        const bool ma = ggml_backend_is_meta(sched_a->backends[sa->backend_id]);
        const bool mb = ggml_backend_is_meta(sched_b->backends[sb->backend_id]);
        if (ma != mb || sa->backend_id != sb->backend_id) return decline("different split shapes");
        if (ma && i != sched_a->n_splits - 1) return decline("Meta split is not last");
    }
    auto * a = &sched_a->splits[sched_a->n_splits - 1];
    auto * b = &sched_b->splits[sched_b->n_splits - 1];
    if (a->backend_id != b->backend_id ||
            sched_a->backends[a->backend_id] != sched_b->backends[b->backend_id]) {
        return decline("different final Meta backend");
    }
    ggml_backend_t backend = sched_a->backends[a->backend_id];
    if (!ggml_backend_is_meta(backend)) {
        return decline("last split not Meta");
    }

    auto copy_inputs = [](ggml_backend_sched_t sched, ggml_backend_sched_split * split) -> size_t {
        size_t staged = 0;
        for (int input_id = 0; input_id < split->n_inputs; ++input_id) {
            ggml_tensor * input = split->inputs[input_id];
            if (!ggml_backend_sched_is_host_weight(input)) {
                if (input->buffer != nullptr && ggml_backend_buffer_is_host(input->buffer) &&
                        ggml_backend_buffer_get_usage(input->buffer) != GGML_BACKEND_BUFFER_USAGE_WEIGHTS) {
                    staged += ggml_nbytes(input);
                }
                sched->bc_stagger_input_slot = input_id + 1;
                ggml_backend_sched_copy_input(sched, split, input);
                sched->bc_stagger_input_slot = 0;
            }
        }
        for (int input_id = 0; input_id < split->n_inputs; ++input_id) {
            if (ggml_backend_sched_is_host_weight(split->inputs[input_id])) {
                ggml_backend_sched_copy_input(sched, split, split->inputs[input_id]);
            }
        }
        return staged;
    };
    // tensor_copy() is a macro over a variable named `sched`; this function has two schedulers
    auto copy_of = [](ggml_backend_sched_t sched, ggml_tensor * t, int backend_id, int copy_id) {
        return tensor_copy(t, backend_id, copy_id);
    };
    // The two schedulers must not land their final split inputs in the same allocation. If they do, B can
    // overwrite a device input that A has not consumed yet; fail closed instead of depending on enqueue timing.
    for (int ia = 0; ia < a->n_inputs; ++ia) {
        ggml_tensor * ca = copy_of(sched_a, a->inputs[ia], a->backend_id, sched_a->cur_copy);
        if (!ca) return decline("missing A final-split input copy");
        for (int ib = 0; ib < b->n_inputs; ++ib) {
            ggml_tensor * cb = copy_of(sched_b, b->inputs[ib], b->backend_id, sched_b->cur_copy);
            if (!cb) return decline("missing B final-split input copy");
            if (ca->buffer != nullptr && ca->buffer == cb->buffer) {
                return decline("paired final-split input copies share a buffer");
            }
        }
    }

    // Preallocate 1326's pair-safe pinned staging before Meta admits either graph. Any host input that would take
    // the blocking pageable/synchronous route makes the pair decline instead.
    std::string bc_stage_why;
    const int64_t bc_stage_prep_t0 = bc_timing_on ? ggml_time_us() : 0;
    if (!bc_1361_prepare_input_stages(sched_a, a, bc_stage_why)) {
        const std::string why = "A staging: " + bc_stage_why;
        return decline(why.c_str());
    }
    bc_stage_why.clear();
    if (!bc_1361_prepare_input_stages(sched_b, b, bc_stage_why)) {
        const std::string why = "B staging: " + bc_stage_why;
        return decline(why.c_str());
    }
    if (bc_timing_on) bc_stage_us += ggml_time_us() - bc_stage_prep_t0;

    // Prefixes must be host-only: prepare_half runs after each half is captured but before provider admission.
    // That makes the final H2D graph-external copy target the exact Meta mapping just captured without launching
    // any final Meta model work before the pair is known to be admissible.
    for (int split_id = 0; split_id + 1 < sched_a->n_splits; ++split_id) {
        ggml_backend_t ba = sched_a->backends[sched_a->splits[split_id].backend_id];
        ggml_backend_t bb = sched_b->backends[sched_b->splits[split_id].backend_id];
        if (ggml_backend_dev_type(ggml_backend_get_device(ba)) != GGML_BACKEND_DEVICE_TYPE_CPU ||
                ggml_backend_dev_type(ggml_backend_get_device(bb)) != GGML_BACKEND_DEVICE_TYPE_CPU) {
            return decline("non-host prefix split");
        }
    }

    auto prepare_half = [&](int half) -> ggml_status {
        ggml_backend_sched_t sched = half == 0 ? sched_a : sched_b;
        ggml_backend_sched_split * split = half == 0 ? a : b;
        const int64_t prefix_t0 = bc_timing_on ? ggml_time_us() : 0;
        ggml_status ec = ggml_backend_sched_compute_splits_prefix(sched);
        if (bc_timing_on) bc_prefix_us += ggml_time_us() - prefix_t0;
        if (ec != GGML_STATUS_SUCCESS) return ec;
        const int64_t stage_t0 = bc_timing_on ? ggml_time_us() : 0;
        bc_staged_bytes += copy_inputs(sched, split);
        if (bc_timing_on) bc_stage_us += ggml_time_us() - stage_t0;
        return GGML_STATUS_SUCCESS;
    };
    using prepare_type = decltype(prepare_half);
    auto cb = [](void * p, int half) -> ggml_status {
        return (*static_cast<prepare_type *>(p))(half);
    };
    ggml_status status = GGML_STATUS_SUCCESS;
    const bool ok = ggml_backend_meta_graph_compute_pair(
        backend, &a->graph, &b->graph, &status, cb, &prepare_half,
        bc_timing_on ? &bc_capture_us : nullptr,
        bc_timing_on ? &bc_rehome_us : nullptr,
        bc_timing_on ? &bc_submit_us : nullptr,
        bc_timing_on ? &bc_sums : nullptr);
    if (!ok) {
        return status; // unsupported was decided before model compute; caller runs today's sequential order.
    }
    *supported = true;
    if (status != GGML_STATUS_SUCCESS) {
        return status;
    }
    if (sched_a->events[a->backend_id][sched_a->cur_copy] != nullptr) {
        ggml_backend_event_record(sched_a->events[a->backend_id][sched_a->cur_copy], backend);
    }
    if (sched_b->events[b->backend_id][sched_b->cur_copy] != nullptr) {
        ggml_backend_event_record(sched_b->events[b->backend_id][sched_b->cur_copy], backend);
    }
    if (bc_timing_on) {
        GGML_LOG_INFO(
            "BIGCHERRY_PREFILL_STAGGER_HOST prefix_us=%lld staging_us=%lld capture_us=%lld rehome_us=%lld "
            "submit_us=%lld total_us=%lld staged_bytes=%zu sums=%zu\n",
            (long long) bc_prefix_us, (long long) bc_stage_us, (long long) bc_capture_us,
            (long long) bc_rehome_us, (long long) bc_submit_us, (long long) (ggml_time_us() - bc_total_t0),
            bc_staged_bytes, bc_sums);
    }
    return GGML_STATUS_SUCCESS;
}

void ggml_backend_sched_set_eval_callback(ggml_backend_sched_t sched, ggml_backend_sched_eval_callback callback, void * user_data) {
'''

_A_HEADER_SCHED = r'''    GGML_API enum ggml_status     ggml_backend_sched_graph_compute_async(ggml_backend_sched_t sched, struct ggml_cgraph * graph);
    GGML_API void                 ggml_backend_sched_synchronize(ggml_backend_sched_t sched);
'''
_N_HEADER_SCHED = r'''    GGML_API enum ggml_status     ggml_backend_sched_graph_compute_async(ggml_backend_sched_t sched, struct ggml_cgraph * graph);
    // BigCherry 1361 (QFP49): narrow two-scheduler Meta path; supported=false means no model work was launched.
    GGML_API enum ggml_status     ggml_backend_sched_graph_compute_async_pair(
        ggml_backend_sched_t sched_a, struct ggml_cgraph * graph_a,
        ggml_backend_sched_t sched_b, struct ggml_cgraph * graph_b,
        bool * supported);
    GGML_API void                 ggml_backend_sched_set_stagger_pair_arena(ggml_backend_sched_t sched, bool enabled);
    GGML_API void                 ggml_backend_sched_synchronize(ggml_backend_sched_t sched);
'''


# ------------------------------------------------------------------------------------------------ llama context
# <cstdlib> and <mutex> are already included there (1316, 1319)
_A_LLAMA_INCLUDES = r'''#include <cinttypes>
'''
_N_LLAMA_INCLUDES = r'''#include <algorithm>
#include <cinttypes>
'''

_A_CTX_FIELDS = r'''    ggml_backend_sched_ptr sched;

    bool sched_need_reserve = true;
'''
_N_CTX_FIELDS = r'''    ggml_backend_sched_ptr sched;

    // BigCherry 1361 (QFP49): B has a distinct scheduler/result and therefore distinct logical Meta compute buffers.
    ggml_backend_sched_ptr bc_1361_sched_b;
    llm_graph_result_ptr   bc_1361_res_b;
    uint32_t               bc_1361_half = 0;
    bool                   bc_1361_declined = false;
    bool                   bc_1361_nextn_pair = false;
    bool                   bc_1361_nextn_pair_last = false;
    int                    bc_1361_nextn_pair_slot = -1;

    bool sched_need_reserve = true;
'''

_A_RESERVE_END = r'''    LLAMA_LOG_INFO("%s: reserve took %.2f ms, sched copies = %d\n",
            __func__, (t_end_us - t_start_us)/1000.0, ggml_backend_sched_get_n_copies(sched.get()));
}
'''
_N_RESERVE_END = r'''    LLAMA_LOG_INFO("%s: reserve took %.2f ms, sched copies = %d\n",
            __func__, (t_end_us - t_start_us)/1000.0, ggml_backend_sched_get_n_copies(sched.get()));

    // BigCherry 1361: reserve B only when explicitly requested.  The public batch/ubatch defaults are untouched.
    const char * bc_env = getenv("BIGCHERRY_PREFILL_STAGGER");
    const unsigned long bc_half_ul = bc_env ? strtoul(bc_env, nullptr, 10) : 0;
    bc_1361_half = bc_half_ul <= std::numeric_limits<uint32_t>::max() ? (uint32_t) bc_half_ul : 0;
    bc_1361_declined = false;
    if (bc_1361_half > 1 && cparams.n_ubatch == bc_1361_half && cparams.n_batch >= 2*bc_1361_half &&
            !model.hparams.no_alloc) {
        const size_t bc_max_nodes = graph_max_nodes(bc_1361_half);
        bc_1361_sched_b.reset(ggml_backend_sched_new(
            backend_ptrs.data(), backend_buft.data(), backend_ptrs.size(), bc_max_nodes,
            cparams.pipeline_parallel, cparams.op_offload));
        ggml_backend_sched_set_copy_callback(bc_1361_sched_b.get(), sched_copy_experts, this);
        ggml_backend_sched_set_stagger_pair_arena(bc_1361_sched_b.get(), true);
        bc_1361_res_b.reset(new llm_graph_result(bc_max_nodes));

        uint32_t bc_n_seqs = std::min<uint32_t>(cparams.n_seq_max, bc_1361_half);
        while (bc_n_seqs > 1 && bc_1361_half % bc_n_seqs != 0) --bc_n_seqs;
        const uint32_t bc_outputs = std::min<uint32_t>(bc_1361_half, cparams.n_outputs_max);

        auto * bc_saved_active = gf_res_prev_active;
        std::swap(sched, bc_1361_sched_b);
        std::swap(gf_res_reserve, bc_1361_res_b);
        ggml_cgraph * bc_gf = graph_reserve(bc_1361_half, bc_n_seqs, bc_outputs, mctx.get());
        std::swap(gf_res_reserve, bc_1361_res_b);
        std::swap(sched, bc_1361_sched_b);
        gf_res_prev_active = bc_saved_active;
        if (bc_gf == nullptr) {
            LLAMA_LOG_WARN("BIGCHERRY_PREFILL_STAGGER disabled: failed to reserve B scheduler for half=%u\n", bc_1361_half);
            bc_1361_sched_b.reset();
            bc_1361_res_b.reset();
            bc_1361_half = 0;
        } else {
            for (size_t i = 0; i < backend_ptrs.size(); ++i) {
                const size_t bytes = ggml_backend_sched_get_buffer_size(bc_1361_sched_b.get(), backend_ptrs[i]);
                if (bytes > 1) {
                    LLAMA_LOG_INFO("BIGCHERRY_PREFILL_STAGGER reserve_b half=%u backend=%s size_mib=%.2f\n",
                        bc_1361_half, ggml_backend_name(backend_ptrs[i]), bytes / 1024.0 / 1024.0);
                }
            }
        }
    } else {
        bc_1361_sched_b.reset();
        bc_1361_res_b.reset();
        if (bc_1361_half != 0) {
            LLAMA_LOG_WARN("BIGCHERRY_PREFILL_STAGGER disabled: half=%u requires -b >= %u -ub %u and allocated compute buffers\n",
                bc_1361_half, 2*bc_1361_half, bc_1361_half);
        }
        bc_1361_half = 0;
    }
}
'''

# Insert preparation just before the normal do/while.  Existing output extraction remains the single implementation.
_A_DECODE_LOOP = r'''    int64_t n_outputs_prev = 0;
    int64_t n_tokens_prev  = 0;

    do {
        const auto & ubatch = mctx->get_ubatch();
'''
_N_DECODE_LOOP = r'''    int64_t n_outputs_prev = 0;
    int64_t n_tokens_prev  = 0;

    llama_ubatch bc_pair_ubatch[2] = {};
    llm_graph_result * bc_pair_res[2] = {nullptr, nullptr};
    int bc_pair_count = 0;
    int bc_pair_index = 0;
    bool bc_pair_mctx_is_b = false;

    auto bc_count_outputs = [&](const llama_ubatch & ub) {
        if (n_outputs_all == n_tokens_all) return (int32_t) ub.n_tokens;
        int32_t ret = 0;
        for (uint32_t i = 0; i < ub.n_tokens; ++i) ret += (int32_t) (ub.output[i] != 0);
        return ret;
    };

    auto bc_prepare = [&](const llama_ubatch & ub, bool use_b, llm_graph_result * res, ggml_status & ret) -> bool {
        if (!mctx->apply()) {
            ret = GGML_STATUS_FAILED;
            return false;
        }
        n_outputs = bc_count_outputs(ub);
        ggml_backend_sched_ptr * target = use_b ? &bc_1361_sched_b : &sched;
        if (!*target || !res) {
            ret = GGML_STATUS_FAILED;
            return false;
        }
        bool swapped = use_b;
        if (swapped) std::swap(sched, bc_1361_sched_b);
        if (!use_b) gf_res_prev_active = nullptr;
        res->reset();
        ggml_backend_sched_reset(sched.get());
        ggml_backend_sched_set_eval_callback(sched.get(), nullptr, nullptr);
        const auto gparams = graph_params(res, ub, mctx.get(), ctx_type_to_graph_type(cparams.ctx_type));
        ggml_cgraph * gf = model.build_graph(gparams);
        bool ok = gf != nullptr && ggml_backend_sched_alloc_graph(sched.get(), gf);
        if (ok) res->set_inputs(&ub);
        if (swapped) std::swap(sched, bc_1361_sched_b);
        if (!ok) {
            ret = gf ? GGML_STATUS_ALLOC_FAILED : GGML_STATUS_FAILED;
            return false;
        }
        ret = GGML_STATUS_SUCCESS;
        return true;
    };

    const bool bc_pair_candidate = !bc_1361_declined && bc_1361_half > 1 && bc_1361_sched_b && bc_1361_res_b &&
        n_tokens_all == 2*bc_1361_half && cparams.n_ubatch == bc_1361_half && cparams.causal_attn &&
        cparams.cb_eval == nullptr && !cparams.embeddings && cparams.pooling_type == LLAMA_POOLING_TYPE_NONE &&
        std::none_of(cparams.embeddings_layer_inp.begin(), cparams.embeddings_layer_inp.end(), [](bool v) { return v; }) &&
        mctx->get_ubatch().n_tokens == bc_1361_half;

    if (bc_pair_candidate) {
        ggml_status bc_status = GGML_STATUS_SUCCESS;
        bc_pair_ubatch[0] = mctx->get_ubatch();
        auto * bc_res_a = get_gf_res_prev();
        if (bc_prepare(bc_pair_ubatch[0], false, bc_res_a, bc_status)) {
            bc_pair_res[0] = bc_res_a;
            bc_pair_count = 1;
            if (mctx->next()) {
                bc_pair_mctx_is_b = true;
                bc_pair_ubatch[1] = mctx->get_ubatch();
                if (bc_pair_ubatch[1].n_tokens == bc_1361_half &&
                        bc_prepare(bc_pair_ubatch[1], true, bc_1361_res_b.get(), bc_status)) {
                    bc_pair_res[1] = bc_1361_res_b.get();
                    bc_pair_count = 2;
                    bool bc_supported = false;
                    bc_status = ggml_backend_sched_graph_compute_async_pair(
                        sched.get(), bc_pair_res[0]->get_gf(),
                        bc_1361_sched_b.get(), bc_pair_res[1]->get_gf(), &bc_supported);
                    if (bc_supported) {
                        if (bc_status != GGML_STATUS_SUCCESS) {
                            LLAMA_LOG_ERROR("%s: stagger pair failed with status %d\n", __func__, bc_status);
                            return -3;
                        }
                        static std::once_flag bc_1361_logged;
                        std::call_once(bc_1361_logged, [] {
                            LLAMA_LOG_WARN("BIGCHERRY_PATCH_HIT patch=1361_prefill_stagger_pair\n");
                        });
                    } else {
                        if (bc_status == GGML_STATUS_SUCCESS) {
                            bc_1361_declined = true;
                            gf_res_prev_active = nullptr;
                        }
                        if (bc_status != GGML_STATUS_SUCCESS) {
                            // Meta preparation failed (not a provider rejection); never execute partially prepared graphs.
                            return bc_status == GGML_STATUS_ALLOC_FAILED ? -2 : -3;
                        }
                        // Pair rejection happens before Meta model compute.  Preserve A-before-B sequential compute.
                        bc_status = graph_compute(bc_pair_res[0]->get_gf(), true);
                        if (bc_status == GGML_STATUS_SUCCESS) {
                            std::swap(sched, bc_1361_sched_b);
                            bc_status = graph_compute(bc_pair_res[1]->get_gf(), true);
                            std::swap(sched, bc_1361_sched_b);
                        }
                        if (bc_status != GGML_STATUS_SUCCESS) return -3;
                    }
                } else {
                    if (bc_pair_ubatch[1].n_tokens == bc_1361_half) {
                        // B's apply/build failed; retrying would apply memory twice and invalidate A's state.
                        return bc_status == GGML_STATUS_ALLOC_FAILED ? -2 : -3;
                    }
                    // The split was not exactly two halves.  A is already prepared but no pair work started.
                    bc_status = graph_compute(bc_pair_res[0]->get_gf(), true);
                    if (bc_status != GGML_STATUS_SUCCESS) return -3;
                }
            } else {
                bc_status = graph_compute(bc_pair_res[0]->get_gf(), true);
                if (bc_status != GGML_STATUS_SUCCESS) return -3;
            }
        } else {
            return bc_status == GGML_STATUS_ALLOC_FAILED ? -2 : -3;
        }
    }

    bool bc_loop_done = false;
    while (!bc_loop_done) {
        const bool bc_cached = bc_pair_index < bc_pair_count;
        const auto & ubatch = bc_cached ? bc_pair_ubatch[bc_pair_index] : mctx->get_ubatch();
'''

_A_PROCESS = r'''        ggml_status status;

        const auto * res = process_ubatch(ubatch, ctx_type_to_graph_type(cparams.ctx_type), mctx.get(), status);
'''
_N_PROCESS = r'''        ggml_status status = GGML_STATUS_SUCCESS;

        const llm_graph_result * res = bc_cached ? bc_pair_res[bc_pair_index]
            : process_ubatch(ubatch, ctx_type_to_graph_type(cparams.ctx_type), mctx.get(), status);
        const bool bc_using_b = bc_cached && bc_pair_index == 1;
        if (bc_using_b) {
            // Existing extraction code uses sched.get(); point it at B only while extracting B's outputs.
            std::swap(sched, bc_1361_sched_b);
        }
'''

_A_LOOP_TAIL = r'''        n_outputs_prev += n_outputs;
        n_tokens_prev  += ubatch.n_tokens;
    } while (mctx->next());
'''
_N_LOOP_TAIL = r'''        n_outputs_prev += n_outputs;
        n_tokens_prev  += ubatch.n_tokens;

        if (bc_using_b) {
            std::swap(sched, bc_1361_sched_b);
        }
        if (bc_cached) {
            ++bc_pair_index;
            if (bc_pair_index < bc_pair_count) {
                continue;
            }
            // If exactly two cached halves were prepared, mctx is already on B and the logical batch is complete.
            if (bc_pair_count == 2) {
                bc_loop_done = true;
                continue;
            }
            // A-only fallback: mctx may already point at an unprocessed tail B. Do not advance it here.
            if (bc_pair_mctx_is_b) {
                bc_pair_count = 0;
                continue;
            }
        }
        bc_loop_done = !mctx->next();
    }
'''

# 1359 fenced NextN copy: pair halves share one slot; B records the event after its copy.
_A_1359_GATE = r'''                    if (bc_1359_on && !masked && offset == 0) {
                        const int bc_slot = bc_nextn_cur ^ 1;
                        // the buffer about to be reused: its last copy must have landed (this also keeps the
                        // pipeline two batches deep and no deeper)
                        bc_nextn_fence_wait(bc_slot);
'''
_N_1359_GATE = r'''                    if (bc_1359_on && !masked && (offset == 0 || bc_1361_nextn_pair)) {
                        int bc_slot = bc_nextn_cur ^ 1;
                        if (bc_1361_nextn_pair) {
                            if (bc_1361_nextn_pair_slot < 0) bc_1361_nextn_pair_slot = bc_slot;
                            bc_slot = bc_1361_nextn_pair_slot;
                        }
                        // A waits before reusing the slot; B reuses A's slot and must not wait for its own in-flight copy.
                        if (!bc_1361_nextn_pair || offset == 0) {
                            bc_nextn_fence_wait(bc_slot);
                        }
'''

_A_1359_BYTES = r'''                        const size_t bc_bytes = (size_t) n_rows * n_embd * sizeof(float);
                        if (bc_ok && (!bc_nextn_buf[bc_slot] || ggml_backend_buffer_get_size(bc_nextn_buf[bc_slot].get()) < bc_bytes)) {
'''
_N_1359_BYTES = r'''                        const size_t bc_bytes = (size_t) n_rows * n_embd * sizeof(float);
                        const size_t bc_off_bytes = (size_t) offset * n_embd * sizeof(float);
                        const size_t bc_need_bytes = bc_off_bytes + bc_bytes;
                        if (bc_ok && (!bc_nextn_buf[bc_slot] || ggml_backend_buffer_get_size(bc_nextn_buf[bc_slot].get()) < bc_need_bytes)) {
'''

_A_1359_CAP = r'''                            const size_t bc_cap = std::max(bc_bytes, (size_t) n_embd * cparams.n_batch * sizeof(float));
'''
_N_1359_CAP = r'''                            const size_t bc_cap = std::max(bc_need_bytes, (size_t) n_embd * cparams.n_batch * sizeof(float));
'''

_A_1359_COPY = r'''                            float * bc_dst = (float *) ggml_backend_buffer_get_base(bc_nextn_buf[bc_slot].get());
                            ggml_backend_tensor_get_async(backend_h, t_h_nextn, bc_dst, 0, bc_bytes);
                            for (size_t j = 0; j < bc_events.size(); ++j) {
                                ggml_backend_event_record(bc_events[j], bc_backends[j]);
                            }
                            bc_nextn_cur        = bc_slot;
                            bc_nextn_slot_valid = true;
'''
_N_1359_COPY = r'''                            float * bc_dst = (float *) ggml_backend_buffer_get_base(bc_nextn_buf[bc_slot].get()) + offset*n_embd;
                            ggml_backend_tensor_get_async(backend_h, t_h_nextn, bc_dst, 0, bc_bytes);
                            if (!bc_1361_nextn_pair || bc_1361_nextn_pair_last) {
                                // For a stagger pair, B's event fences both A and B copies in this slot.
                                for (size_t j = 0; j < bc_events.size(); ++j) {
                                    ggml_backend_event_record(bc_events[j], bc_backends[j]);
                                }
                                bc_nextn_cur        = bc_slot;
                                bc_nextn_slot_valid = true;
                                bc_1361_nextn_pair_slot = -1;
                            }
'''

# Set the 1359 pair flags around the existing extraction block without changing flag-off behavior.
_A_NEXTN_BLOCK = r'''        // extract nextn embeddings before
        // only meaningful in LLAMA_POOLING_TYPE_NONE (per-token); other pooling modes are ignored.
        {
'''
_N_NEXTN_BLOCK = r'''        // extract nextn embeddings before
        // only meaningful in LLAMA_POOLING_TYPE_NONE (per-token); other pooling modes are ignored.
        bc_1361_nextn_pair = bc_pair_count == 2;
        bc_1361_nextn_pair_last = bc_1361_nextn_pair && bc_pair_index == 1;
        {
'''

_A_NEXTN_AFTER = r'''        if (extract_all_idxs) {
'''
_N_NEXTN_AFTER = r'''        bc_1361_nextn_pair = false;
        bc_1361_nextn_pair_last = false;

        if (extract_all_idxs) {
'''

PATCHES = [
    FilePatch(
        path="ggml/src/ggml-cuda/ggml-cuda.cu",
        description="1361: expose RCCL-only transfer-stream begin/end through backend registry",
        language="none",
        edits=(
            _edit(id="stagger-cuda-route", anchor=_A_CUDA_DISPATCH, mode="replace", text=_N_CUDA_DISPATCH,
                  guard=r"struct bc_1361_ar_state", rationale="Beside the communication dispatch after 1360/provider helpers."),
            _edit(id="stagger-cuda-procs", anchor=_A_CUDA_REG, mode="replace", text=_N_CUDA_REG,
                  guard=r"ggml_backend_comm_allreduce_stagger_begin", rationale="CUDA backend registry communication proc-address table."),
        ),
    ),
    FilePatch(
        path="ggml/src/ggml-backend-meta.cpp",
        description="1361: independent A/B Meta graph sets, second arena owner, and stagger pair execution",
        language="none",
        edits=(
            _edit(id="stagger-meta-forward", anchor=_A_META_CTX_DECL, mode="replace", text=_N_META_CTX_DECL,
                  guard=r"struct bc_1361_graph_set;", rationale="Forward declaration for the transient capture pointer."),
            _edit(id="stagger-meta-context", anchor=_A_META_CONTEXT, mode="replace", text=_N_META_CONTEXT,
                  guard=r"bc_pair_capture", rationale="Meta context beside its normal UID/communication state."),
            _edit(id="stagger-meta-force-rebuild", anchor=_A_META_REBUILD, mode="replace", text=_N_META_REBUILD,
                  guard=r"bc_pair_capture != nullptr", rationale="Pair capture must rebuild B even when A/B graph UIDs match."),
            _edit(id="stagger-meta-set-type", anchor=_A_META_GRAPH_FN, mode="replace", text=_N_META_GRAPH_FN,
                  guard=r"struct bc_1361_graph_set \{", rationale="Immediately before Meta graph compute."),
            _edit(id="stagger-meta-capture", anchor=_A_META_AUX, mode="replace", text=_N_META_AUX,
                  guard=r"pair capture stops before aux/fallback execution", rationale="After subgraph build and before aux/fallback execution."),
            _edit(id="stagger-meta-pair", anchor=_A_META_IFACE, mode="replace", text=_N_META_IFACE,
                  guard=r"ggml_backend_meta_graph_compute_pair", rationale="Immediately before the Meta backend interface."),
            _edit(id="stagger-arena-fields", anchor=_A_ARENA_FIELDS, mode="replace", text=_N_ARENA_FIELDS,
                  guard=r"arena_galloc_pair", rationale="1340 per-device arena ownership fields."),
            _edit(id="stagger-arena-dtor", anchor=_A_ARENA_DTOR, mode="replace", text=_N_ARENA_DTOR,
                  guard=r"arena_plans_pair.clear", rationale="1340 Meta context destruction before simple backends."),
            _edit(id="stagger-meta-view-headroom", anchor=_A_META_HEADROOM, mode="replace", text=_N_META_HEADROOM,
                  guard=r"compute_headroom = 32;", rationale="Static Meta buffer view-container sizing."),
            _edit(id="stagger-arena-pair-helpers", anchor=_A_ARENA_ROTATE, mode="replace", text=_N_ARENA_ROTATE,
                  guard=r"ggml_backend_meta_reserve_graph_pair", rationale="After 1340 reserve/bind helpers and before container rotation."),
        ),
    ),
    FilePatch(
        path="ggml/src/ggml-backend.cpp",
        description="1361: second arena selector and narrow two-scheduler Meta pair dispatch",
        language="none",
        edits=(
            _edit(id="stagger-sched-type", anchor=_A_SCHED_TYPE, mode="replace", text=_N_SCHED_TYPE,
                  guard=r"struct bc_1361_input_stages;", rationale="Forward declaration immediately before scheduler private state."),
            _edit(id="stagger-sched-includes", anchor=_A_SCHED_INCLUDES, mode="replace", text=_N_SCHED_INCLUDES,
                  guard=r"#include <set>", rationale="The decline diagnostics keep a std::set<std::string> of reasons already logged."),
            _edit(id="stagger-sched-field", anchor=_A_SCHED_STRUCT, mode="replace", text=_N_SCHED_STRUCT,
                  guard=r"bc_stagger_pair_arena", rationale="Scheduler private state beside op_offload."),
            _edit(id="stagger-sched-decls", anchor=_A_BACKEND_DECLS, mode="replace", text=_N_BACKEND_DECLS,
                  guard=r"ggml_backend_meta_graph_compute_pair", rationale="1340 Meta backend-local declarations."),
            _edit(id="stagger-sched-copy-input", anchor=_A_SCHED_COPY_INPUT, mode="replace", text=_N_SCHED_COPY_INPUT,
                  guard=r"struct bc_1361_input_stage_slot", rationale="Extend the existing 1326 scheduler input path with pair-owned pinned staging."),
            _edit(id="stagger-1326-pair-gate", anchor=_A_1326_GATE, mode="replace", text=_N_1326_GATE,
                  guard=r"sched->bc_stagger_input_slot > 0 \|\| bc_async_inputs",
                  rationale="Production 1326 gate: pair-owned pinned staging bypasses only its 4 MiB policy cap; ordinary 1326 is unchanged."),
            _edit(id="stagger-1326-pinned-stage", anchor=_A_1326_STAGE, mode="replace", text=_N_1326_STAGE,
                  guard=r"if \(sched->bc_stagger_input_slot > 0\) \{", rationale="long-anchor: replace the complete 1326 staging block so pair mode can retain its existing eligibility/synchronization gate."),
            _edit(id="stagger-stage-free", anchor=_A_SCHED_FREE, mode="replace", text=_N_SCHED_FREE,
                  guard=r"delete sched->bc_stagger_input_stages", rationale="Scheduler owns and frees pair input staging/events."),
            _edit(id="stagger-sched-alloc-arena", anchor=_A_SCHED_ALLOC, mode="replace", text=_N_SCHED_ALLOC,
                  guard=r"failed to allocate per-device Meta arena%s", rationale="1340 compute-time Meta arena bind."),
            _edit(id="stagger-sched-reserve-arena", anchor=_A_SCHED_RESERVE, mode="replace", text=_N_SCHED_RESERVE,
                  guard=r"failed to reserve per-device Meta arena%s", rationale="1340 reserve-time Meta arena planning."),
            _edit(id="stagger-sched-prefix-start", anchor=_A_SCHED_PREFIX_START, mode="replace", text=_N_SCHED_PREFIX_START,
                  guard=r"ggml_backend_sched_compute_splits_range", rationale="Use original scheduler loop."),
            _edit(id="stagger-sched-prefix-loop", anchor=_A_SCHED_PREFIX_LOOP, mode="replace", text=_N_SCHED_PREFIX_LOOP,
                  guard=r"split_id < end_split; split_id\+\+", rationale="The original scheduler loop, bounded by the range."),
            _edit(id="stagger-sched-prefix-end", anchor=_A_SCHED_PREFIX_END, mode="replace", text=_N_SCHED_PREFIX_END,
                  guard=r"ggml_backend_sched_compute_splits_prefix", rationale="Expose prefix."),
            _edit(id="stagger-sched-pair-api", anchor=_A_SCHED_API_SITE, mode="replace", text=_N_SCHED_API_SITE,
                  guard=r"ggml_backend_sched_graph_compute_async_pair", rationale="Scheduler API implementation beside eval callback setter."),
        ),
    ),
    FilePatch(
        path="ggml/include/ggml-backend.h",
        description="1361: declare scheduler pair and B-arena selector",
        language="none",
        edits=(
            _edit(id="stagger-sched-pair-decl", anchor=_A_HEADER_SCHED, mode="replace", text=_N_HEADER_SCHED,
                  guard=r"ggml_backend_sched_graph_compute_async_pair", rationale="Scheduler compute declarations."),
        ),
    ),
    FilePatch(
        path="src/llama-context.h",
        description="1361: context-owned B scheduler/result and pair-copy state",
        language="none",
        edits=(
            _edit(id="stagger-context-fields", anchor=_A_CTX_FIELDS, mode="replace", text=_N_CTX_FIELDS,
                  guard=r"bc_1361_sched_b", rationale="Primary scheduler state in llama_context."),
        ),
    ),
    FilePatch(
        path="src/llama-context.cpp",
        description="1361: reserve scheduler B, prepare exact half-pairs, dispatch stagger, preserve existing extraction",
        language="none",
        edits=(
            _edit(id="stagger-llama-includes", anchor=_A_LLAMA_INCLUDES, mode="replace", text=_N_LLAMA_INCLUDES,
                  guard=r"#include <algorithm>", rationale="1361 uses none_of, strtoul, and once_flag."),
            _edit(id="stagger-reserve-b", anchor=_A_RESERVE_END, mode="replace", text=_N_RESERVE_END,
                  guard=r"BIGCHERRY_PREFILL_STAGGER reserve_b half", rationale="End of primary scheduler reserve after memory/full graph is available."),
            _edit(id="stagger-decode-pair-prep", anchor=_A_DECODE_LOOP, mode="replace", text=_N_DECODE_LOOP,
                  guard=r"bc_pair_candidate", rationale="Immediately before the existing per-ubatch decode loop."),
            _edit(id="stagger-decode-use-cache", anchor=_A_PROCESS, mode="replace", text=_N_PROCESS,
                  guard=r"bc_using_b", rationale="Existing process_ubatch call; cached pair results bypass only compute/build."),
            _edit(id="stagger-nextn-pair-flags", anchor=_A_NEXTN_BLOCK, mode="replace", text=_N_NEXTN_BLOCK,
                  guard=r"bc_1361_nextn_pair_last", rationale="Immediately before existing nextn extraction/1359 secondary copy."),
            _edit(id="stagger-nextn-pair-clear", anchor=_A_NEXTN_AFTER, mode="replace", text=_N_NEXTN_AFTER,
                  guard=r"bc_1361_nextn_pair = false", rationale="Immediately after nextn extraction."),
            _edit(id="stagger-decode-loop-tail", anchor=_A_LOOP_TAIL, mode="replace", text=_N_LOOP_TAIL,
                  guard=r"bc_loop_done = !mctx->next\(\);", rationale="Existing per-ubatch output counters and loop advance."),
            _edit(id="stagger-1359-gate", anchor=_A_1359_GATE, mode="replace", text=_N_1359_GATE,
                  guard=r"bc_1361_nextn_pair_slot", rationale="1359 stagger-off gate stays identical; pair B reuses A slot."),
            _edit(id="stagger-1359-size", anchor=_A_1359_BYTES, mode="replace", text=_N_1359_BYTES,
                  guard=r"bc_need_bytes", rationale="1359 pinned-buffer sizing for pair offsets."),
            _edit(id="stagger-1359-cap", anchor=_A_1359_CAP, mode="replace", text=_N_1359_CAP,
                  guard=r"std::max\(bc_need_bytes", rationale="1359 pinned-buffer allocation capacity."),
            _edit(id="stagger-1359-copy", anchor=_A_1359_COPY, mode="replace", text=_N_1359_COPY,
                  guard=r"For a stagger pair, B's event fences both A and B copies", rationale="1359 secondary NextN copy/event sequence."),
        ),
    ),
]

ENV_DOCS = (
    EnvDoc(
        "BIGCHERRY_PREFILL_STAGGER",
        "0|<half tokens>",
        "0 (off)",
        "for an exact two-ubatch causal prompt, reserve a second scheduler and stagger equal half-size Meta subgraphs; "
        "start the server with -b 2*half -ub half",
    ),
    EnvDoc(
        "BIGCHERRY_PREFILL_STAGGER_TIMING",
        "0|1",
        "0",
        "diagnostic: emit per-pair host microseconds for prefix, pinned staging, Meta capture/rebuild, view re-home, "
        "interleaved submit, total, plus staged bytes and sum count",
    ),
)
