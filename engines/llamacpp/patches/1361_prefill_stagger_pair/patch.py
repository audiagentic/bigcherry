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
        max_span_lines=anchor.count("\n") + 2,
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
        ggml_backend_t backend, struct ggml_cgraph * cgraph_a, struct ggml_cgraph * cgraph_b, ggml_status * status_out) {
    GGML_ASSERT(ggml_backend_is_meta(backend));
    auto * backend_ctx = static_cast<ggml_backend_meta_context *>(backend->context);
    const size_t n_backends = backend_ctx->backend_configs.size();
    if (status_out) {
        *status_out = GGML_STATUS_SUCCESS;
    }
    if (n_backends < 2 || backend_ctx->comm_ctx == nullptr) {
        return false;
    }

    bc_1361_graph_set sets[2];
    backend_ctx->bc_pair_capture = &sets[0];
    ggml_status status = ggml_backend_meta_graph_compute(backend, cgraph_a);
    backend_ctx->bc_pair_capture = nullptr;
    if (status != GGML_STATUS_SUCCESS) {
        if (status_out) *status_out = status;
        return false;
    }
    backend_ctx->bc_pair_capture = &sets[1];
    status = ggml_backend_meta_graph_compute(backend, cgraph_b);
    backend_ctx->bc_pair_capture = nullptr;
    if (status != GGML_STATUS_SUCCESS) {
        if (status_out) *status_out = status;
        return false;
    }
    if (sets[0].n_subgraphs != sets[1].n_subgraphs || sets[0].n_subgraphs == 0) {
        return false;
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
        return false;
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
                return false;
            }
        }
    }

    void * pending[2] = {nullptr, nullptr};
    auto finish = [&](int h) {
        if (pending[h] != nullptr) {
            end(pending[h]);
            state_free(pending[h]);
            pending[h] = nullptr;
        }
    };
    auto cleanup = [&]() { finish(0); finish(1); };

    for (size_t i = 0; i < sets[0].n_subgraphs; ++i) {
        for (int h = 0; h < 2; ++h) {
            finish(h);
            for (size_t j = 0; j < n_backends; ++j) {
                status = ggml_backend_graph_compute_async(backend_ctx->backend_configs[j].backend, sets[h].cgraphs[j][i]);
                if (status != GGML_STATUS_SUCCESS) {
                    cleanup();
                    if (status_out) *status_out = status;
                    return true;
                }
            }
            if (i + 1 < sets[h].n_subgraphs) {
                auto nodes = nodes_for(sets[h], i);
                pending[h] = state_new();
                if (pending[h] == nullptr || !begin(backend_ctx->comm_ctx, nodes.data(), pending[h])) {
                    if (pending[h] != nullptr) {
                        state_free(pending[h]);
                        pending[h] = nullptr;
                    }
                    cleanup();
                    if (status_out) *status_out = GGML_STATUS_FAILED;
                    return true;
                }
            }
        }
    }
    cleanup();
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
_A_SCHED_STRUCT = r'''    bool op_offload;

    int debug;
'''
_N_SCHED_STRUCT = r'''    bool op_offload;
    bool bc_stagger_pair_arena = false; // BigCherry 1361: scheduler B selects 1340's second physical owner.

    int debug;
'''

_A_BACKEND_DECLS = r'''bool ggml_backend_meta_alloc_graph(ggml_backend_t meta_backend, struct ggml_cgraph * cgraph);
void ggml_backend_meta_rotate_graph_containers(struct ggml_cgraph * cgraph);
'''
_N_BACKEND_DECLS = r'''bool ggml_backend_meta_alloc_graph(ggml_backend_t meta_backend, struct ggml_cgraph * cgraph);
bool ggml_backend_meta_reserve_graph_pair(ggml_backend_t meta_backend, struct ggml_cgraph * cgraph);
bool ggml_backend_meta_alloc_graph_pair(ggml_backend_t meta_backend, struct ggml_cgraph * cgraph);
bool ggml_backend_meta_graph_compute_pair(
    ggml_backend_t backend, struct ggml_cgraph * cgraph_a, struct ggml_cgraph * cgraph_b, ggml_status * status_out);
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
