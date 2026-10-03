"""PNRO03: opt-in two-GPU P2P path for the internal HIP AllReduce.

Port of nasone32/llama.cpp-RDNA3-7900xtx-opt 7c5bb5cb, generated with
bigcherry.patch.port_diff (verified byte-exact and idempotent), ADAPTED to
BigCherry's gfx1100 evidence that destination-current peer copies can return
success with wrong data:
  * no fixed issuer (the fork's GGML_CUDA_AR_P2P_ISSUER is not ported): each
    direction owns a stream + events on its SOURCE device and every
    cudaMemcpyPeerAsync is issued with ggml_cuda_set_device(source) current;
  * init runs ggml_cuda_ar_p2p_probe: asymmetric nonzero patterns, four sizes,
    both directions, twice, byte-compared; any mismatch keeps host staging;
  * GGML_CUDA_AR_P2P=1 opts in (default off); BIGCHERRY_PATCH_TRACE logs
    BIGCHERRY_PATCH_HIT patch=1252_nro03 when the P2P path executes.
The fork's HIP-enable hunks are already native at b11126 (upstream spelling
kept: s_sleep(4), MUSA-only exclusion).
"""

from bigcherry.patcher import Edit, FilePatch

PROVENANCE = {
    "source-id": "nasone-rdna-optimizations",
    "plan-item": "NRO03",
    "fork-commit": "7c5bb5cb991670676b89cddc5077c9456c5cf70e",
    "port-mode": "port_diff-generated; source-current push + content probe adaptation",
}

PATCH_01 = FilePatch(
    path='ggml/src/ggml-cuda/allreduce.cu',
    description='nro03-p2p: ggml/src/ggml-cuda/allreduce.cu (nasone 7c5bb5cb)',
    edits=(
        Edit(
            id='nro03-p2p-01-01',
            anchor='\\#include\\ <cstring>\\\n\\#include\\ <limits>\\\n\\\n\\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\\n',
            text='#include <cstring>\n#include <limits>\n#include <mutex>\n#include <vector>\n\n// ---------------------------------------------------------------------------\n',
            mode='replace',
            guard='\\#include\\ <mutex>',
            rationale='nro03-p2p-01 hunk 1: upstream lines 12-11 -> result lines 12-13',
            max_span_lines=6,
        ),
        Edit(
            id='nro03-p2p-01-02',
            anchor='\\ \\ \\ \\ size_t\\ \\ \\ copy_chunk_bytes;\\\n\\ \\ \\ \\ size_t\\ \\ \\ bf16_threshold;\\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\\n\\ \\ \\ \\ uint64_t\\ call_count;\\\n\\\n',
            text='    size_t   copy_chunk_bytes;\n    size_t   bf16_threshold; // tensors >= this size (bytes) are reduced via FP32->BF16 round-trip; 0 disables\n    bool     p2p_enabled;\n    uint64_t call_count;\n\n',
            mode='replace',
            guard='bool\\ \\ \\ \\ \\ p2p_enabled;',
            rationale='nro03-p2p-01 hunk 2: upstream lines 311-310 -> result lines 313-313',
            max_span_lines=6,
        ),
        Edit(
            id='nro03-p2p-01-03',
            anchor='\\ \\ \\ \\ char\\ \\*\\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ dev_tmp\\[GGML_CUDA_MAX_DEVICES\\];\\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\\n\\ \\ \\ \\ cudaStream_t\\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ streams\\[GGML_CUDA_MAX_DEVICES\\];\\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\\n\\ \\ \\ \\ ggml_cuda_ar_event_slot\\ \\ ev_pool\\[GGML_CUDA_MAX_DEVICES\\]\\[GGML_CUDA_AR_POOL_SIZE\\];\\\n\\\n',
            text='    char *                    dev_tmp[GGML_CUDA_MAX_DEVICES];    // device scratch for copy-engine path\n    cudaStream_t             streams[GGML_CUDA_MAX_DEVICES];   // non-blocking\n    cudaStream_t             p2p_stream[2];                     // per direction, on the SOURCE device\n    cudaEvent_t              p2p_done[2][GGML_CUDA_AR_POOL_SIZE];\n    ggml_cuda_ar_event_slot  ev_pool[GGML_CUDA_MAX_DEVICES][GGML_CUDA_AR_POOL_SIZE];\n\n',
            mode='replace',
            guard='cudaStream_t\\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ p2p_stream\\[2\\];\\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ //\\ per\\ direction,\\ on\\ the\\ SOURCE\\ device',
            rationale='nro03-p2p-01 hunk 3: upstream lines 318-317 -> result lines 321-322',
            max_span_lines=6,
        ),
        Edit(
            id='nro03-p2p-01-04',
            anchor='\\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\\n\\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\\n\\\nggml_cuda_ar_pipeline\\ \\*\\ ggml_cuda_ar_pipeline_init\\(const\\ int\\ \\*\\ devices,\\ size_t\\ n_devices\\)\\ \\{\\\n',
            text='// Init / free\n// ---------------------------------------------------------------------------\n\n// bigcherry NRO03: content-checked bidirectional P2P probe. API success is not\n// correctness evidence (gfx1100 destination-current peer copies returned\n// success with wrong data), so every direction is verified byte for byte with\n// asymmetric nonzero patterns before P2P is enabled.\nstatic bool ggml_cuda_ar_p2p_probe(ggml_cuda_ar_pipeline * p) {\n    static const size_t sizes[] = { 4096, 65536 + 256, (size_t) 1 << 20, (size_t) 4 << 20 };\n    const size_t max_bytes = (size_t) 4 << 20;\n    char * dev_src[2] = {};\n    char * dev_dst[2] = {};\n    std::vector<uint8_t> pattern(max_bytes);\n    std::vector<uint8_t> readback(max_bytes);\n    bool ok = true;\n    for (int i = 0; i < 2 && ok; ++i) {\n        ggml_cuda_set_device(p->devices[i]);\n        ok = cudaMalloc((void **) &dev_src[i], max_bytes) == cudaSuccess &&\n             cudaMalloc((void **) &dev_dst[i], max_bytes) == cudaSuccess;\n    }\n    for (int iter = 0; iter < 2 && ok; ++iter) {\n        for (size_t bytes : sizes) {\n            for (int src = 0; src < 2 && ok; ++src) {\n                const int dst = 1 - src;\n                for (size_t k = 0; k < bytes; ++k) {\n                    pattern[k] = (uint8_t) (((k * 131u + src * 71u + iter * 29u + bytes) & 0xfe) | 1);\n                }\n                ggml_cuda_set_device(p->devices[dst]);\n                ok = cudaMemset(dev_dst[dst], 0, bytes) == cudaSuccess && cudaDeviceSynchronize() == cudaSuccess;\n                ggml_cuda_set_device(p->devices[src]);\n                ok = ok && cudaMemcpy(dev_src[src], pattern.data(), bytes, cudaMemcpyHostToDevice) == cudaSuccess &&\n                     cudaMemcpyPeerAsync(dev_dst[dst], p->devices[dst], dev_src[src], p->devices[src], bytes,\n                                         p->p2p_stream[src]) == cudaSuccess &&\n                     cudaStreamSynchronize(p->p2p_stream[src]) == cudaSuccess;\n                ggml_cuda_set_device(p->devices[dst]);\n                ok = ok && cudaMemcpy(readback.data(), dev_dst[dst], bytes, cudaMemcpyDeviceToHost) == cudaSuccess &&\n                     memcmp(readback.data(), pattern.data(), bytes) == 0;\n                if (!ok) {\n                    GGML_LOG_WARN("%s: P2P probe %d -> %d failed at %zu bytes; using host staging\\n",\n                                  __func__, p->devices[src], p->devices[dst], bytes);\n                }\n            }\n        }\n    }\n    for (int i = 0; i < 2; ++i) {\n        ggml_cuda_set_device(p->devices[i]);\n        if (dev_src[i]) { cudaFree(dev_src[i]); }\n        if (dev_dst[i]) { cudaFree(dev_dst[i]); }\n    }\n    (void) cudaGetLastError();\n    return ok;\n}\n\nggml_cuda_ar_pipeline * ggml_cuda_ar_pipeline_init(const int * devices, size_t n_devices) {\n',
            mode='replace',
            guard='//\\ bigcherry\\ NRO03:\\ content\\-checked\\ bidirectional\\ P2P\\ probe\\.\\ API\\ success\\ is\\ not',
            rationale='nro03-p2p-01 hunk 4: upstream lines 399-398 -> result lines 404-453',
            max_span_lines=6,
        ),
        Edit(
            id='nro03-p2p-01-05',
            anchor='\\ \\ \\ \\ \\}\\\n\\\n\\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\\n\\ \\ \\ \\ const\\ size_t\\ arrival_bytes\\ =\\\n',
            text='    }\n\n#if defined(GGML_USE_HIP)\n    if (ggml_cuda_ar_env_u64("GGML_CUDA_AR_P2P", 0) != 0) {\n        int can_access[2] = {};\n        CUDA_CHECK(cudaDeviceCanAccessPeer(&can_access[0], p->devices[0], p->devices[1]));\n        CUDA_CHECK(cudaDeviceCanAccessPeer(&can_access[1], p->devices[1], p->devices[0]));\n\n        if (can_access[0] && can_access[1]) {\n            bool peer_enabled = true;\n            for (int i = 0; i < 2; ++i) {\n                ggml_cuda_set_device(p->devices[i]);\n                const cudaError_t rc = cudaDeviceEnablePeerAccess(p->devices[1 - i], 0);\n                if (rc == cudaErrorPeerAccessAlreadyEnabled) {\n                    // HIP leaves this expected error pending for the next launch.\n                    (void) cudaGetLastError();\n                } else if (rc != cudaSuccess) {\n                    GGML_LOG_WARN("%s: peer access %d -> %d failed: %s\\n", __func__, p->devices[i], p->devices[1 - i], cudaGetErrorString(rc));\n                    peer_enabled = false;\n                }\n            }\n\n            if (peer_enabled) {\n                // One stream + event set per direction, owned by the SOURCE device:\n                // every peer copy is issued with its source device current.\n                for (int direction = 0; direction < 2; ++direction) {\n                    ggml_cuda_set_device(p->devices[direction]);\n                    if (cudaStreamCreateWithFlags(&p->p2p_stream[direction], cudaStreamNonBlocking) != cudaSuccess) {\n                        GGML_LOG_ERROR("%s: P2P stream creation failed\\n", __func__);\n                        ggml_cuda_ar_pipeline_free(p);\n                        return nullptr;\n                    }\n                    for (int slot = 0; slot < GGML_CUDA_AR_POOL_SIZE; ++slot) {\n                        if (cudaEventCreateWithFlags(&p->p2p_done[direction][slot], cudaEventDisableTiming) != cudaSuccess) {\n                            GGML_LOG_ERROR("%s: P2P event creation failed\\n", __func__);\n                            ggml_cuda_ar_pipeline_free(p);\n                            return nullptr;\n                        }\n                    }\n                }\n                p->p2p_enabled = ggml_cuda_ar_p2p_probe(p);\n            }\n        } else {\n            GGML_LOG_WARN("%s: bidirectional P2P is unavailable; using host staging\\n", __func__);\n        }\n    }\n#endif\n\n    // Arrival ring: cache-line padded so each GPU\'s int is on its own line.\n    const size_t arrival_bytes =\n',
            mode='replace',
            guard='\\#if\\ defined\\(GGML_USE_HIP\\)',
            rationale='nro03-p2p-01 hunk 5: upstream lines 483-482 -> result lines 538-583',
            max_span_lines=6,
        ),
        Edit(
            id='nro03-p2p-01-06',
            anchor='\\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ ,\\\n\\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ __func__,\\ n_devices,\\ p\\->buf_bytes\\ >>\\ 10,\\ p\\->copy_bytes\\ >>\\ 20\\);\\\n',
            text='                  "%zu KB chunked kernel staging + %zu MB copy-engine staging per GPU, P2P %s\\n",\n                  __func__, n_devices, p->buf_bytes >> 10, p->copy_bytes >> 20, p->p2p_enabled ? "on" : "off");\n',
            mode='replace',
            guard='"%zu\\ KB\\ chunked\\ kernel\\ staging\\ \\+\\ %zu\\ MB\\ copy\\-engine\\ staging\\ per\\ GPU,\\ P2P\\ %s\\\\n",',
            rationale='nro03-p2p-01 hunk 6: upstream lines 536-537 -> result lines 637-638',
            max_span_lines=4,
        ),
        Edit(
            id='nro03-p2p-01-07',
            anchor='\\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ ggml_cuda_set_device\\(p\\->devices\\[i\\]\\);\\\n\\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ CUDA_CHECK\\(cudaStreamSynchronize\\(p\\->streams\\[i\\]\\)\\);\\\n\\ \\ \\ \\ \\ \\ \\ \\ \\}\\\n\\ \\ \\ \\ \\}\\\n',
            text='            ggml_cuda_set_device(p->devices[i]);\n            CUDA_CHECK(cudaStreamSynchronize(p->streams[i]));\n        }\n    }\n    for (int direction = 0; direction < 2; ++direction) {\n        if (p->p2p_stream[direction]) {\n            ggml_cuda_set_device(p->devices[direction]);\n            cudaStreamSynchronize(p->p2p_stream[direction]);\n        }\n    }\n',
            mode='replace',
            guard='if\\ \\(p\\->p2p_stream\\[direction\\]\\)\\ \\{',
            rationale='nro03-p2p-01 hunk 7: upstream lines 552-551 -> result lines 653-658',
            max_span_lines=6,
        ),
        Edit(
            id='nro03-p2p-01-08',
            anchor='\\ \\ \\ \\ \\ \\ \\ \\ \\}\\\n\\ \\ \\ \\ \\}\\\n\\ \\ \\ \\ p\\->arrival\\.free\\(\\);\\\n\\ \\ \\ \\ delete\\ p;\\\n',
            text='        }\n    }\n    for (int direction = 0; direction < 2; ++direction) {\n        if (p->p2p_stream[direction]) {\n            ggml_cuda_set_device(p->devices[direction]);\n            for (int slot = 0; slot < GGML_CUDA_AR_POOL_SIZE; ++slot) {\n                if (p->p2p_done[direction][slot]) {\n                    cudaEventDestroy(p->p2p_done[direction][slot]);\n                }\n            }\n            cudaStreamDestroy(p->p2p_stream[direction]);\n        }\n    }\n    p->arrival.free();\n    delete p;\n',
            mode='replace',
            guard='if\\ \\(p\\->p2p_done\\[direction\\]\\[slot\\]\\)\\ \\{',
            rationale='nro03-p2p-01 hunk 8: upstream lines 584-583 -> result lines 691-701',
            max_span_lines=6,
        ),
        Edit(
            id='nro03-p2p-01-09',
            anchor='\\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\\n\\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\\ntemplate\\ <typename\\ T_src,\\ typename\\ T_dst>\\\nstatic\\ bool\\ ggml_cuda_ar_allreduce_copy_impl\\(\\\n',
            text='// for bit-equivalence between GPUs and we skip the otherwise-needed\n// post-conversion entirely.\ntemplate <typename T_src, typename T_dst>\nstatic bool ggml_cuda_ar_allreduce_p2p_impl(\n        ggml_cuda_ar_pipeline * p,\n        ggml_backend_t        * backends,\n        T_src * const           src_buf[GGML_CUDA_MAX_DEVICES],\n        T_dst * const           dst_buf[GGML_CUDA_MAX_DEVICES],\n        const bool              compute[GGML_CUDA_MAX_DEVICES],\n        int64_t                 ne,\n        size_t                  nbytes) {\n    GGML_ASSERT(p->n_devices == 2);\n    GGML_ASSERT(p->p2p_enabled);\n    GGML_ASSERT(nbytes <= p->copy_bytes);\n    GGML_ASSERT(ne <= std::numeric_limits<int>::max());\n\n    const int slot = ggml_cuda_ar_acquire_slot(p).slot;\n    ggml_backend_cuda_context * cuda_ctx[2] = {};\n\n    for (int i = 0; i < 2; ++i) {\n        ggml_cuda_set_device(p->devices[i]);\n        cuda_ctx[i] = static_cast<ggml_backend_cuda_context *>(backends[i]->context);\n        GGML_ASSERT(cuda_ctx[i]->device == p->devices[i]);\n        if (!compute[i]) {\n            CUDA_CHECK(cudaMemsetAsync(src_buf[i], 0, nbytes, cuda_ctx[i]->stream()));\n        }\n        CUDA_CHECK(cudaEventRecord(p->ev_pool[i][slot].app, cuda_ctx[i]->stream()));\n    }\n\n    if (getenv("BIGCHERRY_PATCH_TRACE") != nullptr) {\n        static std::once_flag bigcherry_nro03_logged;\n        std::call_once(bigcherry_nro03_logged, [] {\n            GGML_LOG_WARN("BIGCHERRY_PATCH_HIT patch=1252_nro03 path=allreduce_p2p_source_push\\n");\n        });\n    }\n\n    cudaStream_t * copy_stream = p->p2p_stream;\n    for (int direction = 0; direction < 2; ++direction) {\n        const int destination = 1 - direction;\n        ggml_cuda_set_device(p->devices[direction]);  // source-current push\n        CUDA_CHECK(cudaStreamWaitEvent(copy_stream[direction], p->ev_pool[direction][slot].app));\n        if (p->dev_tmp_kernel_done_valid) {\n            CUDA_CHECK(cudaStreamWaitEvent(copy_stream[direction], p->dev_tmp_kernel_done[destination]));\n        }\n        CUDA_CHECK(cudaMemcpyPeerAsync(\n            p->dev_tmp[destination], p->devices[destination], src_buf[direction], p->devices[direction], nbytes,\n            copy_stream[direction]));\n        CUDA_CHECK(cudaEventRecord(p->p2p_done[direction][slot], copy_stream[direction]));\n    }\n\n    for (int i = 0; i < 2; ++i) {\n        ggml_cuda_set_device(p->devices[i]);\n        CUDA_CHECK(cudaStreamWaitEvent(cuda_ctx[i]->stream(), p->p2p_done[0][slot]));\n        CUDA_CHECK(cudaStreamWaitEvent(cuda_ctx[i]->stream(), p->p2p_done[1][slot]));\n\n        const int block_size = 256;\n        int n_blocks = (int) ((ne + block_size - 1) / block_size);\n        if (n_blocks > 1024) {\n            n_blocks = 1024;\n        }\n        ggml_cuda_ar_add_kernel<T_dst, T_src><<<n_blocks, block_size, 0, cuda_ctx[i]->stream()>>>(\n            dst_buf[i], reinterpret_cast<const T_src *>(p->dev_tmp[i]), (int) ne);\n        CUDA_CHECK(cudaGetLastError());\n        CUDA_CHECK(cudaEventRecord(p->dev_tmp_kernel_done[i], cuda_ctx[i]->stream()));\n        CUDA_CHECK(cudaEventRecord(p->ev_pool[i][slot].ker, cuda_ctx[i]->stream()));\n    }\n    p->dev_tmp_kernel_done_valid = true;\n\n    return true;\n}\n\ntemplate <typename T_src, typename T_dst>\nstatic bool ggml_cuda_ar_allreduce_copy_impl(\n',
            mode='replace',
            guard='static\\ bool\\ ggml_cuda_ar_allreduce_p2p_impl\\(',
            rationale='nro03-p2p-01 hunk 9: upstream lines 598-597 -> result lines 716-784',
            max_span_lines=6,
        ),
        Edit(
            id='nro03-p2p-01-10',
            anchor='\\ \\ \\ \\ \\ \\ \\ \\ ok\\ =\\ ggml_cuda_ar_allreduce_copy_impl<T_src,\\ T_dst>\\(\\\n\\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ p,\\ backends,\\ src,\\ dst,\\ compute,\\ outer_ne,\\ outer_nbytes\\);\\\n',
            text='        if (p->p2p_enabled) {\n            ok = ggml_cuda_ar_allreduce_p2p_impl<T_src, T_dst>(\n                p, backends, src, dst, compute, outer_ne, outer_nbytes);\n        } else {\n            ok = ggml_cuda_ar_allreduce_copy_impl<T_src, T_dst>(\n                p, backends, src, dst, compute, outer_ne, outer_nbytes);\n        }\n',
            mode='replace',
            guard='if\\ \\(p\\->p2p_enabled\\)\\ \\{',
            rationale='nro03-p2p-01 hunk 10: upstream lines 742-743 -> result lines 929-935',
            max_span_lines=4,
        ),
    ),
)

PATCHES = [PATCH_01]
