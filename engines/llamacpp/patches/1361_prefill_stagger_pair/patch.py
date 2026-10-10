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
