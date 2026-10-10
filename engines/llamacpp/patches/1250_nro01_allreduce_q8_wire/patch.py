"""1250: P2P Q8 wire + fused residual overlay on shared 1272 codec/finish.

Requires 1252 (P2P provider) and 1272 (wire codec/host paths). This package
owns no Q8 codec, wire parser, or Q8 finish kernels: it extends the 1252 P2P
provider to call 1272's shared finish helper, routes 1272 Q8 copy-engine work
over P2P when available, and retains PNRO02's meta-backend residual fusion.
"""

import re as _re

from bigcherry.patcher import Edit, FilePatch

PROVENANCE = {
    "source-id": "nasone-rdna-optimizations",
    "plan-item": "NRO01/NRO02",
    "fork-commit": "e06dcf6300718227cb8cfda9e61fb12ccb693418",
    "port-mode": "migrated overlay: 1252 transport + 1272 shared codec/fused finish",
}

_BACKEND_TYPEDEF_ANCHOR = """    typedef bool   (*ggml_backend_comm_allreduce_tensor_t)(void * comm_ctx, struct ggml_tensor ** tensors);
"""
_BACKEND_TYPEDEF_INSERT = """    typedef bool   (*ggml_backend_comm_allreduce_tensor_fused_add_t)(
        void * comm_ctx, struct ggml_tensor ** tensors, struct ggml_tensor ** residuals, struct ggml_tensor ** outputs);
"""

PATCH_BACKEND_H = FilePatch(
    path="ggml/include/ggml-backend.h",
    description="1250: expose fused AllReduce+ADD communication hook",
    language="none",
    edits=(
        Edit(
            id="nro01-shared-backend-fused-typedef",
            anchor=_re.escape(_BACKEND_TYPEDEF_ANCHOR),
            mode="insert_after",
            text=_BACKEND_TYPEDEF_INSERT,
            guard=r"ggml_backend_comm_allreduce_tensor_fused_add_t",
            rationale="Retain PNRO02's optional fused residual communication entry point while wire ownership moves to 1272.",
            expect_matches=1,
            max_span_lines=2,
        ),
    ),
)

_META_CTX_ANCHOR = """    void *                               comm_ctx       = nullptr;
    ggml_backend_comm_allreduce_tensor_t comm_allreduce = nullptr;

    ggml_backend_meta_context(ggml_backend_dev_t meta_dev, const char * params) {
"""
_META_CTX_TEXT = """    void *                               comm_ctx       = nullptr;
    ggml_backend_comm_allreduce_tensor_t comm_allreduce = nullptr;
    ggml_backend_comm_allreduce_tensor_fused_add_t comm_allreduce_fused_add = nullptr;

    ggml_backend_meta_context(ggml_backend_dev_t meta_dev, const char * params) {
"""
_META_PROC_ANCHOR = """                    ggml_backend_get_device(simple_backends[0])), "ggml_backend_comm_allreduce_tensor");
            GGML_ASSERT(comm_allreduce != nullptr);
        }
    }
"""
_META_PROC_TEXT = """                    ggml_backend_get_device(simple_backends[0])), "ggml_backend_comm_allreduce_tensor");
            GGML_ASSERT(comm_allreduce != nullptr);
            comm_allreduce_fused_add = (ggml_backend_comm_allreduce_tensor_fused_add_t)
                ggml_backend_reg_get_proc_address(ggml_backend_dev_backend_reg(
                    ggml_backend_get_device(simple_backends[0])), "ggml_backend_comm_allreduce_tensor_fused_add");
        }
    }
"""
_META_LOOP_ANCHOR = """    for (size_t i = 0; i < backend_ctx->n_subgraphs; i++) {
"""
_META_COMPUTE_ANCHOR = """            const ggml_status status = ggml_backend_graph_compute_async(bcj.backend, bcj.cgraphs[i].cgraph_main);
"""
_META_COMPUTE_TEXT = """            ggml_cgraph * cgraph_compute = bcj.cgraphs[i].cgraph_main;
            uint32_t flags = 0;
            if (skip_node >= 0) {
                GGML_ASSERT(skip_node < cgraph_compute->n_nodes);
                flags = cgraph_compute->nodes[skip_node]->flags;
                cgraph_compute->nodes[skip_node]->flags &= ~GGML_TENSOR_FLAG_COMPUTE;
            }
            const ggml_status status = ggml_backend_graph_compute_async(bcj.backend, cgraph_compute);
            if (skip_node >= 0) {
                cgraph_compute->nodes[skip_node]->flags = flags;
            }
"""
_META_NEXT_ANCHOR = """        if (n_backends > 1 && i < backend_ctx->n_subgraphs - 1) {
"""
_META_AR_ANCHOR = """                backend_allreduce_success = backend_ctx->comm_allreduce(backend_ctx->comm_ctx, nodes.data());
"""
_META_AR_TEXT = """
                bool try_fused_add = backend_ctx->comm_allreduce_fused_add != nullptr &&
                    getenv("GGML_CUDA_AR_FUSED_RESIDUAL") != nullptr;
                std::vector<ggml_tensor *> residuals;
                std::vector<ggml_tensor *> outputs;
                const int i_next = backend_ctx->backend_configs[0].cgraphs[i + 1].offset;
                int i_add = i_next;
                if (try_fused_add) {
                    ggml_tensor * node = cgraph->nodes[i_next - 1];
                    const int i_next_end = i_next + backend_ctx->backend_configs[0].cgraphs[i + 1].cgraph_main->n_nodes;
                    while (i_add < i_next_end && cgraph->nodes[i_add]->op == GGML_OP_RESHAPE &&
                            cgraph->nodes[i_add]->src[0] == node && ggml_node_get_use_count(cgraph, i_add - 1) == 1) {
                        node = cgraph->nodes[i_add++];
                    }
                    try_fused_add = i_add < i_next_end;
                    ggml_tensor * add = try_fused_add ? cgraph->nodes[i_add] : nullptr;
                    try_fused_add = try_fused_add && add->op == GGML_OP_ADD && ggml_node_get_use_count(cgraph, i_add - 1) == 1 &&
                        ggml_are_same_shape(node, add) && node->type == GGML_TYPE_F32 && add->type == GGML_TYPE_F32 &&
                        (add->src[0] == node || add->src[1] == node);
                    if (try_fused_add) {
                        ggml_tensor * residual = add->src[0] == node ? add->src[1] : add->src[0];
                        try_fused_add = residual != nullptr && residual->type == GGML_TYPE_F32 &&
                            ggml_are_same_shape(node, residual) &&
                            ggml_backend_meta_get_split_state(residual, false).axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED &&
                            ggml_backend_meta_get_split_state(add, false).axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED;
                    }
                }
                if (try_fused_add) {
                    residuals.reserve(n_backends);
                    outputs.reserve(n_backends);
                    for (size_t j = 0; j < n_backends; j++) {
                        auto & bcj = backend_ctx->backend_configs[j];
                        ggml_cgraph * next = bcj.cgraphs[i + 1].cgraph_main;
                        const int i_add_next = i_add - i_next;
                        ggml_tensor * add = next->nodes[i_add_next];
                        ggml_tensor * reduced = i_add_next == 0 ? nodes[j] : next->nodes[i_add_next - 1];
                        ggml_tensor * residual = add->src[0] == reduced ? add->src[1] : add->src[0];
                        if (add->op != GGML_OP_ADD || residual == nullptr ||
                            !(add->src[0] == reduced || add->src[1] == reduced)) {
                            try_fused_add = false;
                            break;
                        }
                        residuals.push_back(residual);
                        outputs.push_back(add);
                    }
                }
                if (try_fused_add) {
                    backend_allreduce_success = backend_ctx->comm_allreduce_fused_add(
                        backend_ctx->comm_ctx, nodes.data(), residuals.data(), outputs.data());
                    skip_node = backend_allreduce_success ? i_add - i_next : -1;
                }
                if (!backend_allreduce_success) {
                    backend_allreduce_success = backend_ctx->comm_allreduce(backend_ctx->comm_ctx, nodes.data());
                }
"""

PATCH_META = FilePatch(
    path="ggml/src/ggml-backend-meta.cpp",
    description="1250: retain PNRO02 meta-backend ADD fusion around shared 1272 wire implementation",
    language="none",
    edits=(
        Edit(id="nro02-meta-context", anchor=_re.escape(_META_CTX_ANCHOR), mode="replace", text=_META_CTX_TEXT,
             guard=r"comm_allreduce_fused_add = nullptr;", rationale="Store optional fused communication hook.", expect_matches=1, max_span_lines=6),
        Edit(id="nro02-meta-proc", anchor=_re.escape(_META_PROC_ANCHOR), mode="replace", text=_META_PROC_TEXT,
             guard=r"comm_allreduce_fused_add = \(ggml_backend_comm_allreduce_tensor_fused_add_t\)", rationale="Resolve optional fused hook beside normal AllReduce.", expect_matches=1, max_span_lines=6),
        Edit(id="nro02-meta-skip-state", anchor=_re.escape(_META_LOOP_ANCHOR), mode="insert_before", text="    int skip_node = -1;\n",
             guard=r"int skip_node = -1;", rationale="Track the next-subgraph ADD node suppressed after successful fusion.", expect_matches=1, max_span_lines=2),
        Edit(id="nro02-meta-compute-skip", anchor=_re.escape(_META_COMPUTE_ANCHOR), mode="replace", text=_META_COMPUTE_TEXT,
             guard=r"ggml_cgraph \* cgraph_compute = bcj\.cgraphs\[i\]\.cgraph_main;", rationale="Temporarily clear compute on the fused ADD node.", expect_matches=1, max_span_lines=3),
        Edit(id="nro02-meta-reset-skip", anchor=_re.escape(_META_NEXT_ANCHOR), mode="insert_before", text="        skip_node = -1;\n\n",
             guard=r"skip_node = -1;\n\n        if \(n_backends > 1", rationale="Reset fusion skip state once each subgraph has executed.", expect_matches=1, max_span_lines=2),
        Edit(id="nro02-meta-fuse", anchor=_re.escape(_META_AR_ANCHOR), mode="replace", text=_META_AR_TEXT,
             guard=r"bool try_fused_add = backend_ctx->comm_allreduce_fused_add != nullptr", rationale="Fuse a single-use mirrored F32 ADD into the communication finish and fall back atomically if unavailable.", expect_matches=1, max_span_lines=2),
    ),
)

_P2P_SIG_ANCHOR = """template <typename T_src, typename T_dst>
static bool ggml_cuda_ar_allreduce_p2p_impl(
        ggml_cuda_ar_pipeline * p,
        ggml_backend_t        * backends,
        T_src * const           src_buf[GGML_CUDA_MAX_DEVICES],
        T_dst * const           dst_buf[GGML_CUDA_MAX_DEVICES],
        const bool              compute[GGML_CUDA_MAX_DEVICES],
        int64_t                 ne,
        size_t                  nbytes) {
"""
_P2P_SIG_TEXT = """template <typename T_src, typename T_dst>
static bool ggml_cuda_ar_allreduce_p2p_impl(
        ggml_cuda_ar_pipeline * p,
        ggml_backend_t        * backends,
        T_src * const           src_buf[GGML_CUDA_MAX_DEVICES],
        T_dst * const           dst_buf[GGML_CUDA_MAX_DEVICES],
        const bool              compute[GGML_CUDA_MAX_DEVICES],
        int64_t                 ne,
        size_t                  nbytes,
        T_dst * const           residual_buf[GGML_CUDA_MAX_DEVICES] = nullptr) {
"""
_P2P_FINISH_ANCHOR = """        const int block_size = 256;
        int n_blocks = (int) ((ne + block_size - 1) / block_size);
        if (n_blocks > 1024) {
            n_blocks = 1024;
        }
        ggml_cuda_ar_add_kernel<T_dst, T_src><<<n_blocks, block_size, 0, cuda_ctx[i]->stream()>>>(
            dst_buf[i], reinterpret_cast<const T_src *>(p->dev_tmp[i]), (int) ne);
        CUDA_CHECK(cudaGetLastError());
"""
_P2P_FINISH_TEXT = """        ggml_cuda_ar_launch_finish(
            src_buf[i], dst_buf[i], reinterpret_cast<const T_src *>(p->dev_tmp[i]),
            residual_buf ? residual_buf[i] : nullptr, i, ne, cuda_ctx[i]->stream());
        CUDA_CHECK(cudaGetLastError());
"""
_Q8_WIRE_ANCHOR = """template <typename T_dst>
static bool ggml_cuda_ar_allreduce_wire_q8(
"""
_Q8_P2P_HELPER = r'''template <typename T_dst>
static bool ggml_cuda_ar_allreduce_p2p_q8_outer(
        ggml_cuda_ar_pipeline * p,
        ggml_backend_t        * backends,
        block_q8_0 * const      src_buf[GGML_CUDA_MAX_DEVICES],
        T_dst * const            dst_buf[GGML_CUDA_MAX_DEVICES],
        T_dst * const            residual_buf[GGML_CUDA_MAX_DEVICES],
        int64_t                  ne) {
    GGML_ASSERT(p->p2p_enabled);
    const int64_t outer_max_blocks = (int64_t) (p->copy_bytes / sizeof(block_q8_0));
    GGML_ASSERT(outer_max_blocks > 0);
    const int64_t outer_max_elems = outer_max_blocks * QK8_0;
    bool compute[GGML_CUDA_MAX_DEVICES] = { true, true };

    if (getenv("BIGCHERRY_PATCH_TRACE") != nullptr) {
        static std::once_flag logged;
        std::call_once(logged, [] {
            GGML_LOG_WARN("BIGCHERRY_PATCH_HIT patch=1250_nro01 path=allreduce_q8_0_p2p_shared\n");
        });
    }

    bool ok = true;
    for (int64_t outer_start = 0; outer_start < ne && ok; outer_start += outer_max_elems) {
        const int64_t outer_ne = std::min(outer_max_elems, ne - outer_start);
        const int64_t outer_blocks = (outer_ne + QK8_0 - 1) / QK8_0;
        const size_t outer_nbytes = (size_t) outer_blocks * sizeof(block_q8_0);
        block_q8_0 * src[GGML_CUDA_MAX_DEVICES] = {};
        T_dst * dst[GGML_CUDA_MAX_DEVICES] = {};
        T_dst * residual[GGML_CUDA_MAX_DEVICES] = {};
        for (int i = 0; i < p->n_devices; ++i) {
            src[i] = src_buf[i] + outer_start / QK8_0;
            dst[i] = dst_buf[i] + outer_start;
            residual[i] = residual_buf ? residual_buf[i] + outer_start : nullptr;
        }
        ok = ggml_cuda_ar_allreduce_p2p_impl<block_q8_0, T_dst>(
            p, backends, src, dst, compute, outer_ne, outer_nbytes,
            residual_buf ? residual : nullptr);
    }
    return ok;
}

'''
_Q8_HOST_RETURN = """        return ggml_cuda_ar_allreduce_copy_q8_outer<T_dst>(p, backends, src, dst, nullptr, ne);
"""
_Q8_PROVIDER_RETURN = """        if (p->p2p_enabled) {
            return ggml_cuda_ar_allreduce_p2p_q8_outer<T_dst>(p, backends, src, dst, nullptr, ne);
        }
        return ggml_cuda_ar_allreduce_copy_q8_outer<T_dst>(p, backends, src, dst, nullptr, ne);
"""
_MUSA_ANCHOR = """#else // defined(GGML_USE_MUSA)
"""
_FUSED_API = r'''bool ggml_cuda_ar_allreduce_fused_add(
        ggml_cuda_ar_pipeline * p,
        ggml_backend_t        * backends,
        ggml_tensor           ** tensors,
        ggml_tensor           ** residuals,
        ggml_tensor           ** outputs) {
    if (p == nullptr || p->n_devices != 2 ||
            p->wire_override != ggml_cuda_ar_wire_override::q8_0) {
        return false;
    }

    const int64_t ne = ggml_nelements(tensors[0]);
    if (ne <= 0 || ne > std::numeric_limits<int>::max()) {
        return false;
    }
    for (int i = 0; i < p->n_devices; ++i) {
        if (tensors[i] == nullptr || residuals[i] == nullptr || outputs[i] == nullptr ||
                tensors[i]->type != GGML_TYPE_F32 || residuals[i]->type != GGML_TYPE_F32 || outputs[i]->type != GGML_TYPE_F32 ||
                ggml_nelements(tensors[i]) != ne || ggml_nelements(residuals[i]) != ne || ggml_nelements(outputs[i]) != ne) {
            return false;
        }
    }

    const int64_t q8_blocks = (ne + QK8_0 - 1) / QK8_0;
    const size_t q8_nbytes = (size_t) q8_blocks * sizeof(block_q8_0);
    if (p->copy_threshold == 0 || q8_nbytes < p->copy_threshold) {
        return false;
    }

    ggml_cuda_pool_alloc<block_q8_0> q8_tmp[GGML_CUDA_MAX_DEVICES];
    block_q8_0 * src[GGML_CUDA_MAX_DEVICES] = {};
    float * dst[GGML_CUDA_MAX_DEVICES] = {};
    float * residual[GGML_CUDA_MAX_DEVICES] = {};
    const int block_size = 256;
    int n_blocks = (int) ((q8_blocks * QK8_0 + block_size - 1) / block_size);
    n_blocks = std::min(n_blocks, 1024);

    for (int i = 0; i < p->n_devices; ++i) {
        auto * cuda_ctx = static_cast<ggml_backend_cuda_context *>(backends[i]->context);
        GGML_ASSERT(cuda_ctx->device == p->devices[i]);
        ggml_cuda_set_device(p->devices[i]);
        q8_tmp[i].pool = &cuda_ctx->pool();
        q8_tmp[i].alloc(q8_blocks);
        src[i] = q8_tmp[i].get();
        dst[i] = static_cast<float *>(outputs[i]->data);
        residual[i] = static_cast<float *>(residuals[i]->data);
        if ((tensors[i]->flags & GGML_TENSOR_FLAG_COMPUTE) != 0) {
            ggml_cuda_ar_quantize_q8_0_kernel<float><<<n_blocks, block_size, 0, cuda_ctx->stream()>>>(
                static_cast<const float *>(tensors[i]->data), src[i], ne, q8_blocks);
            CUDA_CHECK(cudaGetLastError());
        } else {
            CUDA_CHECK(cudaMemsetAsync(src[i], 0, q8_nbytes, cuda_ctx->stream()));
        }
    }

    if (p->p2p_enabled) {
        return ggml_cuda_ar_allreduce_p2p_q8_outer<float>(p, backends, src, dst, residual, ne);
    }
    return ggml_cuda_ar_allreduce_copy_q8_outer<float>(p, backends, src, dst, residual, ne);
}

'''
_MUSA_STUB_ANCHOR = """bool ggml_cuda_ar_allreduce(ggml_cuda_ar_pipeline *, ggml_backend_t *, ggml_tensor **) {
    return false;
}
"""
_MUSA_STUB_INSERT = """bool ggml_cuda_ar_allreduce_fused_add(
        ggml_cuda_ar_pipeline *, ggml_backend_t *, ggml_tensor **, ggml_tensor **, ggml_tensor **) {
    return false;
}
"""

PATCH_ALLREDUCE = FilePatch(
    path="ggml/src/ggml-cuda/allreduce.cu",
    description="1250: route 1272 Q8 wire through 1252 P2P and reuse 1272 shared finish for optional residual fusion",
    language="none",
    edits=(
        Edit(id="nro01-p2p-shared-finish-signature", anchor=_re.escape(_P2P_SIG_ANCHOR), mode="replace", text=_P2P_SIG_TEXT,
             guard=r"residual_buf\[GGML_CUDA_MAX_DEVICES\] = nullptr", rationale="Extend the provider finish with an optional residual pointer without changing existing callers.", expect_matches=1, max_span_lines=12),
        Edit(id="nro01-p2p-shared-finish-launch", anchor=_re.escape(_P2P_FINISH_ANCHOR), mode="replace", text=_P2P_FINISH_TEXT,
             guard=r"ggml_cuda_ar_launch_finish\(\n\s+src_buf\[i\], dst_buf\[i\], reinterpret_cast<const T_src", rationale="Reuse 1272's shared native/Q8 fused finish instead of owning a duplicate add/dequant kernel.", expect_matches=1, max_span_lines=10),
        Edit(id="nro01-p2p-q8-outer", anchor=_re.escape(_Q8_WIRE_ANCHOR), mode="insert_before", text=_Q8_P2P_HELPER,
             guard=r"static bool ggml_cuda_ar_allreduce_p2p_q8_outer\(", rationale="Slice block-Q8 wire buffers and feed them through the source-current 1252 P2P provider.", expect_matches=1, max_span_lines=3),
        Edit(id="nro01-q8-provider-route", anchor=_re.escape(_Q8_HOST_RETURN), mode="replace", text=_Q8_PROVIDER_RETURN,
             guard=r"return ggml_cuda_ar_allreduce_p2p_q8_outer<T_dst>", rationale="Select P2P for 1272 Q8 copy-engine traffic when the provider probe succeeded; retain host staging fallback.", expect_matches=1, max_span_lines=2),
        Edit(id="nro02-fused-api", anchor=_re.escape(_MUSA_ANCHOR), mode="insert_before", text=_FUSED_API,
             guard=r"bool ggml_cuda_ar_allreduce_fused_add\(", rationale="Quantize once with 1272's codec and finish Q8 transport directly into output+residual, over P2P or host staging.", expect_matches=1, max_span_lines=2),
        Edit(id="nro02-musa-stub", anchor=_re.escape(_MUSA_STUB_ANCHOR), mode="insert_after", text=_MUSA_STUB_INSERT,
             guard=r"ggml_cuda_ar_pipeline \*, ggml_backend_t \*, ggml_tensor \*\*, ggml_tensor \*\*, ggml_tensor \*\*", rationale="Keep the fused API link-complete on MUSA while returning unsupported.", expect_matches=1, max_span_lines=4),
    ),
)

_CUH_ANCHOR = """bool ggml_cuda_ar_allreduce(
    ggml_cuda_ar_pipeline * pipeline,
    ggml_backend_t        * backends,
    ggml_tensor           ** tensors);
"""
_CUH_INSERT = """
bool ggml_cuda_ar_allreduce_fused_add(
    ggml_cuda_ar_pipeline * pipeline,
    ggml_backend_t        * backends,
    ggml_tensor           ** tensors,
    ggml_tensor           ** residuals,
    ggml_tensor           ** outputs);
"""
PATCH_CUH = FilePatch(
    path="ggml/src/ggml-cuda/allreduce.cuh",
    description="1250: declare fused AllReduce+ADD entry point",
    language="none",
    edits=(
        Edit(id="nro02-cuh-fused", anchor=_re.escape(_CUH_ANCHOR), mode="insert_after", text=_CUH_INSERT,
             guard=r"bool ggml_cuda_ar_allreduce_fused_add\(", rationale="Expose PNRO02 fused finish to the CUDA communication dispatcher.", expect_matches=1, max_span_lines=5),
    ),
)

_CUDA_DISPATCH_ANCHOR = """    auto * comm_ctx = static_cast<ggml_backend_cuda_comm_context *>(comm_ctx_v);
    return comm_ctx->try_allreduce(comm_ctx, tensors);
}

"""
_CUDA_DISPATCH_INSERT = r'''static bool ggml_backend_cuda_comm_allreduce_tensor_fused_add(
        void * comm_ctx_v, struct ggml_tensor ** tensors, struct ggml_tensor ** residuals, struct ggml_tensor ** outputs) {
    if (comm_ctx_v == nullptr || getenv("GGML_CUDA_AR_FUSED_RESIDUAL") == nullptr) {
        return false;
    }
    auto * comm_ctx = static_cast<ggml_backend_cuda_comm_context *>(comm_ctx_v);
    if (comm_ctx->ar_pipeline == nullptr) {
        return false;
    }

    const size_t n_backends = comm_ctx->backends.size();
    for (size_t i = 0; i < n_backends; ++i) {
        if (tensors[i] == nullptr || residuals[i] == nullptr || outputs[i] == nullptr ||
            tensors[i]->type != GGML_TYPE_F32 || residuals[i]->type != GGML_TYPE_F32 || outputs[i]->type != GGML_TYPE_F32 ||
            !ggml_are_same_shape(tensors[i], residuals[i]) || !ggml_are_same_shape(tensors[i], outputs[i]) ||
            !ggml_is_contiguously_allocated(tensors[i]) || !ggml_is_contiguously_allocated(residuals[i]) ||
            !ggml_is_contiguously_allocated(outputs[i])) {
            return false;
        }
    }
    const bool fused = ggml_cuda_ar_allreduce_fused_add(
        comm_ctx->ar_pipeline, comm_ctx->backends.data(), tensors, residuals, outputs);
    if (fused && getenv("BIGCHERRY_PATCH_TRACE") != nullptr) {
        static std::once_flag bigcherry_nro02_logged;
        std::call_once(bigcherry_nro02_logged, [] {
            GGML_LOG_WARN("BIGCHERRY_PATCH_HIT patch=1250_nro02 path=allreduce_fused_residual\n");
        });
    }
    return fused;
}

'''
_CUDA_PROC_ANCHOR = """        return (void *)ggml_backend_cuda_comm_allreduce_tensor;
    }
    if (strcmp(name, "ggml_backend_register_host_buffer") == 0) {
"""
_CUDA_PROC_TEXT = """        return (void *)ggml_backend_cuda_comm_allreduce_tensor;
    }
    if (strcmp(name, "ggml_backend_comm_allreduce_tensor_fused_add") == 0) {
        return (void *)ggml_backend_cuda_comm_allreduce_tensor_fused_add;
    }
    if (strcmp(name, "ggml_backend_register_host_buffer") == 0) {
"""
PATCH_CUDA = FilePatch(
    path="ggml/src/ggml-cuda/ggml-cuda.cu",
    description="1250: expose fused residual AllReduce through CUDA comm proc registry",
    language="none",
    edits=(
        Edit(id="nro02-cuda-fused-dispatch", anchor=_re.escape(_CUDA_DISPATCH_ANCHOR), mode="insert_after", text=_CUDA_DISPATCH_INSERT,
             guard=r"static bool ggml_backend_cuda_comm_allreduce_tensor_fused_add\(", rationale="Validate F32 contiguous inputs and execute the shared fused finish only when explicitly enabled.", expect_matches=1, max_span_lines=5),
        Edit(id="nro02-cuda-fused-proc", anchor=_re.escape(_CUDA_PROC_ANCHOR), mode="replace", text=_CUDA_PROC_TEXT,
             guard=r'if \(strcmp\(name, "ggml_backend_comm_allreduce_tensor_fused_add"\) == 0\)', rationale="Publish the optional fused communication symbol to the meta backend.", expect_matches=1, max_span_lines=5),
    ),
)

PATCHES = [PATCH_BACKEND_H, PATCH_META, PATCH_ALLREDUCE, PATCH_CUH, PATCH_CUDA]
