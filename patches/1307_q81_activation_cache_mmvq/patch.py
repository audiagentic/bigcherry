"""1307 (PRBE05 / RD09 stage 2): wire 1235's Q8_1 activation cache into the MMVQ quantize step.

Flash-Next decode is launch-gap bound (QFP13): ~183 quantize_q8_1 launches per generated token per GPU, one per
MMVQ consumer, ~154 of them immediately followed by the matvec they feed. Several consumers read the same
activation (the GDN layer's qkv/gate/beta/alpha projections, shared-expert gate/up, routed experts), so the same
vector is re-quantized per consumer. With GGML_HIP_Q8_1_CACHE_MODE=on, ggml_cuda_mul_mat_vec_q looks the quantized
src1 up in 1235's generation-scoped cache (key: view root tensor, exact data address, stream, shape, strides) and
skips the quantize launch on a hit; a miss quantizes into a stable cache slab and publishes it. Any reserve failure
falls back to the original per-call pool allocation and quantizer. A generation begins at every
ggml_backend_cuda_graph_compute, and slab growth is blocked while a HIP graph is being captured (1235's contract),
so captured graphs only ever reference never-relocated slab memory. Requires 1235 (cache) and 1241 (anchor shape).
"""

from __future__ import annotations

import re

from bigcherry.patcher import Edit, FilePatch

GROUP = "rdna-boosts"
STATE = "untested"

_INCLUDE_ANCHOR = '#include "mmvq.cuh"\n'
_INCLUDE_TEXT = ('#if defined(GGML_USE_HIP)\n'
                 '#include "hip-q81-cache.h"  // bigcherry 1307: Q8_1 activation cache (1235)\n'
                 '#endif\n')

_QUANT_OLD = """\
    const int64_t ne10_padded = GGML_PAD(ne10, MATRIX_ROW_PADDING);
    ggml_cuda_pool_alloc<char> src1_q8_1(ctx.pool(), ne13*ne12 * ne11*ne10_padded * sizeof(block_q8_1)/QK8_1);
    {
        const int64_t s11 = src1->nb[1] / ts_src1;
        const int64_t s12 = src1->nb[2] / ts_src1;
        const int64_t s13 = src1->nb[3] / ts_src1;
        quantize_row_q8_1_cuda(src1_d, nullptr, src1_q8_1.get(), src0->type, ne10, s11, s12, s13, ne10_padded, ne11, ne12, ne13, stream);
    }
"""

_QUANT_NEW = """\
    const int64_t ne10_padded = GGML_PAD(ne10, MATRIX_ROW_PADDING);
    const size_t src1_q8_1_bytes = ne13*ne12 * ne11*ne10_padded * sizeof(block_q8_1)/QK8_1;
    ggml_cuda_pool_alloc<char> src1_q8_1(ctx.pool());  // only allocated when the cache does not supply the buffer
    const char * src1_q8_1_ptr = nullptr;
    {
        const int64_t s11 = src1->nb[1] / ts_src1;
        const int64_t s12 = src1->nb[2] / ts_src1;
        const int64_t s13 = src1->nb[3] / ts_src1;
#if defined(GGML_USE_HIP)
        // bigcherry 1307: reuse one Q8_1 quantization of src1 across every MMVQ consumer in this graph generation.
        if (ggml_hip_q81_cache_mode_get() != GGML_HIP_Q81_CACHE_OFF) {
            ggml_hip_q81_cache & q81 = ggml_hip_q81_cache_for_context(ctx);
            // Identity is the consumed node itself, not its view root: an in-place op's result is a view of the
            // same root at the same address and shape, so a root-based key would serve a stale quantization.
            const ggml_hip_q81_cache_key key = ggml_hip_q81_cache_make_key(
                q81, src1, src1->data, ctx.curr_stream_no,
                ne10, ne10_padded, ne11, ne12, ne13, s11, s12, s13);
            static const bool bigcherry_q81_trace = getenv("BIGCHERRY_Q81_TRACE") != nullptr;
            void * hit = ggml_hip_q81_cache_find(q81, key);
            // A contiguous RESHAPE of a published node (e.g. Qwen4Exp hc_norm = reshape of the RMSNorm output)
            // has identical Q8_1 bytes when neither layout needs row padding: the flatten is row-major and the
            // 32-element groups line up. Look it up under the source node's own key (a reshape never changes data).
            if (hit == nullptr && src1->op == GGML_OP_RESHAPE && src1->view_src != nullptr &&
                    src1->data == src1->view_src->data && ggml_is_contiguous(src1) &&
                    ggml_is_contiguous(src1->view_src) && ne10 == ne10_padded &&
                    src1->view_src->ne[0] % MATRIX_ROW_PADDING == 0 &&
                    ggml_nelements(src1) == ggml_nelements(src1->view_src)) {
                const ggml_tensor * vs = src1->view_src;
                const ggml_hip_q81_cache_key vkey = ggml_hip_q81_cache_make_key(
                    q81, vs, vs->data, ctx.curr_stream_no, vs->ne[0], vs->ne[0], vs->ne[1], vs->ne[2], vs->ne[3],
                    vs->ne[0], vs->ne[0]*vs->ne[1], vs->ne[0]*vs->ne[1]*vs->ne[2]);
                hit = ggml_hip_q81_cache_find(q81, vkey);
            }
            if (hit != nullptr) {
                src1_q8_1_ptr = (const char *) hit;
            } else {
                if (bigcherry_q81_trace) {  // diagnose producer/consumer key mismatches (1309/1310)
                    GGML_LOG_WARN("BIGCHERRY_Q81 miss gen=%llu src1=%p(%s op=%s view_of=%s) data=%p ne=%lld,%lld,%lld,%lld\\n",
                        (unsigned long long) ggml_hip_q81_cache_current_generation(q81), (const void *) src1, src1->name,
                        ggml_op_name(src1->op), src1->view_src ? src1->view_src->name : "-", src1->data,
                        (long long) ne10, (long long) ne11, (long long) ne12, (long long) ne13);
                }
                const ggml_hip_q81_cache_reservation r = ggml_hip_q81_cache_reserve(q81, src1_q8_1_bytes);
                if (r.ok) {
                    quantize_row_q8_1_cuda(src1_d, nullptr, r.ptr, src0->type, ne10, s11, s12, s13, ne10_padded, ne11, ne12, ne13, stream);
                    ggml_hip_q81_cache_publish(q81, key, r);
                    src1_q8_1_ptr = (const char *) r.ptr;
                }
            }
        }
#endif
        if (src1_q8_1_ptr == nullptr) {
            src1_q8_1.alloc(src1_q8_1_bytes);
            quantize_row_q8_1_cuda(src1_d, nullptr, src1_q8_1.get(), src0->type, ne10, s11, s12, s13, ne10_padded, ne11, ne12, ne13, stream);
            src1_q8_1_ptr = src1_q8_1.get();
        }
    }
"""

_SWITCH_OLD = "        src0->data, src0->type, src1_q8_1.get(), ids_d, fusion_local, dst_d, ne00,\n"
_SWITCH_NEW = "        src0->data, src0->type, src1_q8_1_ptr, ids_d, fusion_local, dst_d, ne00,\n"

_GEN_OLD = """\
static enum ggml_status ggml_backend_cuda_graph_compute(ggml_backend_t backend, ggml_cgraph * cgraph) {
    ggml_backend_cuda_context * cuda_ctx = (ggml_backend_cuda_context *) backend->context;

    ggml_cuda_set_device(cuda_ctx->device);
"""

_GEN_NEW = """\
static enum ggml_status ggml_backend_cuda_graph_compute(ggml_backend_t backend, ggml_cgraph * cgraph) {
    ggml_backend_cuda_context * cuda_ctx = (ggml_backend_cuda_context *) backend->context;

    ggml_cuda_set_device(cuda_ctx->device);
#if defined(GGML_USE_HIP)
    // bigcherry 1307: each graph evaluation is one Q8_1 activation-cache generation.
    const bool bigcherry_q81_on = ggml_hip_q81_cache_mode_get() != GGML_HIP_Q81_CACHE_OFF;
    if (bigcherry_q81_on) {
        ggml_hip_q81_cache_begin_generation(ggml_hip_q81_cache_for_context(*cuda_ctx));
    }
#endif
"""

_CAPTURE_OLD = "        CUDA_CHECK(cudaStreamBeginCapture(cuda_ctx->stream(), cudaStreamCaptureModeRelaxed));\n"
_CAPTURE_NEW = """\
#if defined(GGML_USE_HIP)
        if (bigcherry_q81_on) {  // bigcherry 1307: no slab growth while a graph is being captured
            ggml_hip_q81_cache_set_capture_active(ggml_hip_q81_cache_for_context(*cuda_ctx), true);
        }
#endif
"""

_EVAL_OLD = ("    ggml_cuda_graph_evaluate_and_capture(cuda_ctx, cgraph, use_cuda_graph, cuda_graph_update_required, graph_key);\n"
             "\n    return GGML_STATUS_SUCCESS;\n")
_EVAL_NEW = ("    ggml_cuda_graph_evaluate_and_capture(cuda_ctx, cgraph, use_cuda_graph, cuda_graph_update_required, graph_key);\n"
             "#if defined(GGML_USE_HIP)\n"
             "    if (bigcherry_q81_on) {  // bigcherry 1307: capture (if any) has ended\n"
             "        ggml_hip_q81_cache_set_capture_active(ggml_hip_q81_cache_for_context(*cuda_ctx), false);\n"
             "    }\n"
             "#endif\n"
             "\n    return GGML_STATUS_SUCCESS;\n")

PATCHES = [
    FilePatch(
        path="ggml/src/ggml-cuda/mmvq.cu",
        description="1307: MMVQ reuses cached Q8_1 activations (1235 cache) instead of re-quantizing per consumer",
        language="none",
        edits=(
            Edit(
                id="q81-mmvq-include",
                anchor=re.escape(_INCLUDE_ANCHOR),
                mode="insert_after",
                text=_INCLUDE_TEXT,
                guard=r"bigcherry 1307: Q8_1 activation cache",
                rationale="mmvq.cu's own header include; the cache API is HIP-only (1235 builds it for ggml-hip).",
                expect_matches=1,
                max_span_lines=2,
            ),
            Edit(
                id="q81-mmvq-quantize",
                anchor=re.escape(_QUANT_OLD),
                mode="replace",
                text=_QUANT_NEW,
                guard=r"bigcherry 1307: reuse one Q8_1 quantization of src1",
                rationale="The single per-call src1 quantize in ggml_cuda_mul_mat_vec_q (after 1241's F32 early return).",
                expect_matches=1,
                max_span_lines=9,
            ),
            Edit(
                id="q81-mmvq-consumer",
                anchor=re.escape(_SWITCH_OLD),
                mode="replace",
                text=_SWITCH_NEW,
                guard=r"src0->data, src0->type, src1_q8_1_ptr, ids_d",
                rationale="The one consumer of the quantized buffer in ggml_cuda_mul_mat_vec_q.",
                expect_matches=1,
                max_span_lines=2,
            ),
        ),
    ),
    FilePatch(
        path="ggml/src/ggml-cuda/ggml-cuda.cu",
        description="1307: per-graph-evaluation cache generation and capture-time growth block",
        language="none",
        edits=(
            Edit(
                id="q81-generation",
                anchor=re.escape(_GEN_OLD),
                mode="replace",
                text=_GEN_NEW,
                guard=r"bigcherry 1307: each graph evaluation is one Q8_1 activation-cache generation",
                rationale="Head of ggml_backend_cuda_graph_compute: one generation per graph evaluation.",
                expect_matches=1,
                max_span_lines=5,
            ),
            Edit(
                id="q81-capture-begin",
                anchor=re.escape(_CAPTURE_OLD),
                mode="insert_after",
                text=_CAPTURE_NEW,
                guard=r"bigcherry 1307: no slab growth while a graph is being captured",
                rationale="Immediately after cudaStreamBeginCapture in graph_compute (works with or without 1231's marker block).",
                expect_matches=1,
                max_span_lines=2,
            ),
            Edit(
                id="q81-capture-end",
                anchor=re.escape(_EVAL_OLD),
                mode="replace",
                text=_EVAL_NEW,
                guard=r"bigcherry 1307: capture \(if any\) has ended",
                rationale="After evaluate_and_capture, which ends any capture it began.",
                expect_matches=1,
                max_span_lines=4,
            ),
        ),
    ),
]
