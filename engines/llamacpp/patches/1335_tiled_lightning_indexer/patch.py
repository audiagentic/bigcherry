"""1335 (QFP17): backport of upstream #29901 - the lightning indexer tiled over keys and tokens for 4 heads.

Upstream commit 1b43d3116 (merged after the b11402 pin). Qwen4Exp's QSA indexer has 4 heads of 128 dims, too few for
the WMMA tile kernel, so at the pin every (key, token) pair is scored by the vector kernel, which re-reads the key from
global memory for each token. The tile kernel scores 64 keys against 8 tokens per block: keys staged once in half
precision in shared memory, queries of all heads in float, products and sums in float. Batches smaller than 8 tokens
(decode) keep the vector kernel. Upstream's final revision of the kernel is the one that does not spill registers on
ROCm (36x faster on an R9700 than its first form).

Verbatim upstream code except: BIGCHERRY_INDEXER_TILE=0 keeps the vector kernel (same-binary A/B and off switch), and
an activation marker under BIGCHERRY_PATCH_TRACE. Not bit-identical to the vector kernel (non-F16 keys pass through
F16, different summation order). Superseded when the pin reaches a release containing #29901.
"""
import re as _re

from bigcherry.patcher import Edit, EnvDoc, FilePatch

GROUP = "core"
STATE = "superseded"

_A_INC = (
    '#include "convert.cuh"\n'
)

_N_INC = (
    '#include "convert.cuh"\n'
    '\n'
    '#include <cstdlib>  // BigCherry 1335: getenv\n'
)

_A_TODO = (
    '// TODO there is one ugly assumption used in this kernel - that WARP_SIZE is equal to 32\n'
)

_N_TODO = (
    '// tokens scored per block by the tile kernel\n'
    '#define LIGHTNING_INDEXER_TILE_TOKENS 8\n'
    '\n'
    '// BigCherry 1335: backport of upstream #29901; BIGCHERRY_INDEXER_TILE=0 keeps the vector kernel\n'
    'static bool bc_indexer_tile_enabled() {\n'
    '    static const bool on = getenv("BIGCHERRY_INDEXER_TILE") == nullptr || atoi(getenv("BIGCHERRY_INDEXER_TILE")) != 0;\n'
    '    return on;\n'
    '}\n'
    '\n'
    '// TODO there is one ugly assumption used in this kernel - that WARP_SIZE is equal to 32\n'
)

_A_CASE = (
    '#define LIGHTNING_INDEXER_CASE(lightning_indexer_kernel, n_embd, n_head, K, type_K)'
)

_N_CASE = (
    '// one block scores a tile of K_VECS_PER_BLOCK keys against TOKENS_PER_BLOCK tokens: the keys are\n'
    '// staged in half precision and the queries of every head in float, each thread owns KEYS_PER_THREAD\n'
    '// keys for one token, a warp shares its token so the query reads are broadcasts, and every key\n'
    '// element is widened once for all heads, so no dot product needs a cross thread reduction\n'
    'template <int WARPS_PER_BLOCK, int K_VECS_PER_BLOCK, int64_t N_EMBD, int64_t N_HEAD, ggml_type TYPE_K>\n'
    'static __global__ void lightning_indexer_kernel_tile(\n'
    '        const float * Q, const char * K, const float * W, const half * M, float * dst,\n'
    '        int64_t n_stream, int64_t n_batch, int64_t n_kv,\n'
    '        size_t nb1, size_t nb2, size_t nb3,\n'
    '        size_t nbq1, size_t nbq2, size_t nbq3,\n'
    '        size_t nbk1, size_t nbk2, size_t nbk3,\n'
    '        size_t nbw1, size_t nbw2, size_t nbw3,\n'
    '        size_t nbm1, size_t nbm2, size_t nbm3,\n'
    '        int64_t nem3\n'
    '    ) {\n'
    '\n'
    '    constexpr int THREADS_PER_BLOCK = WARPS_PER_BLOCK * WARP_SIZE;\n'
    '    constexpr int TOKENS_PER_BLOCK  = LIGHTNING_INDEXER_TILE_TOKENS;\n'
    '    constexpr int KEY_LANES         = THREADS_PER_BLOCK / TOKENS_PER_BLOCK;\n'
    '    constexpr int KEYS_PER_THREAD   = K_VECS_PER_BLOCK / KEY_LANES;\n'
    '    constexpr int N_EMBD_H2         = N_EMBD / 2;\n'
    '\n'
    '    static_assert(THREADS_PER_BLOCK % TOKENS_PER_BLOCK == 0, "threads must cover the token tile");\n'
    '    static_assert(K_VECS_PER_BLOCK % KEY_LANES == 0, "key lanes must cover the key tile");\n'
    '\n'
    '    const int tid         = threadIdx.y * WARP_SIZE + threadIdx.x;\n'
    '    const int start_kv    = blockIdx.x * K_VECS_PER_BLOCK;\n'
    '    const int start_batch = blockIdx.y * TOKENS_PER_BLOCK;\n'
    '    const int i_stream    = blockIdx.z;\n'
    '\n'
    '    // the row padding keeps the keys of consecutive threads in distinct banks\n'
    '    __shared__ half2 k_shared[K_VECS_PER_BLOCK][N_EMBD_H2 + 1];\n'
    '    __shared__ float2 q_shared[N_HEAD][TOKENS_PER_BLOCK][N_EMBD_H2];\n'
    '    __shared__ float w_shared[N_HEAD][TOKENS_PER_BLOCK];\n'
    '\n'
    '    // phase 1 - stage the key tile four elements at a time, rows past n_kv are zero\n'
    '\n'
    '#pragma unroll\n'
    '    for (int i = tid; i < K_VECS_PER_BLOCK * (N_EMBD / 4); i += THREADS_PER_BLOCK) {\n'
    '        const int r  = i / (N_EMBD / 4);\n'
    '        const int c4 = i % (N_EMBD / 4);\n'
    '\n'
    '        half2 lo = make_half2(0.0f, 0.0f);\n'
    '        half2 hi = lo;\n'
    '        if (start_kv + r < n_kv) {\n'
    '            const char * k_row = K + (start_kv + r)*nbk2 + i_stream*nbk3;\n'
    '            if constexpr (TYPE_K == GGML_TYPE_F16) {\n'
    '                lo = ((const half2 *) k_row)[2*c4 + 0];\n'
    '                hi = ((const half2 *) k_row)[2*c4 + 1];\n'
    '            } else {\n'
    '                float4 v;\n'
    '                if constexpr (TYPE_K == GGML_TYPE_F32) {\n'
    '                    v = ((const float4 *) k_row)[c4];\n'
    '                } else {\n'
    '                    constexpr dequantize_V_t dequantize_k = get_dequantize_V<TYPE_K, float, 4>();\n'
    '                    dequantize_k(k_row, &v, c4 * 4);\n'
    '                }\n'
    '                lo = make_half2(v.x, v.y);\n'
    '                hi = make_half2(v.z, v.w);\n'
    '            }\n'
    '        }\n'
    '\n'
    '        k_shared[r][2*c4 + 0] = lo;\n'
    '        k_shared[r][2*c4 + 1] = hi;\n'
    '    }\n'
    '\n'
    '    // phase 2 - stage the queries and weights of every head, tokens past n_batch are zero\n'
    '\n'
    '#pragma unroll\n'
    '    for (int i = tid; i < N_HEAD * TOKENS_PER_BLOCK * (N_EMBD / 4); i += THREADS_PER_BLOCK) {\n'
    '        const int h  = i / (TOKENS_PER_BLOCK * (N_EMBD / 4));\n'
    '        const int r  = i / (N_EMBD / 4) % TOKENS_PER_BLOCK;\n'
    '        const int c4 = i % (N_EMBD / 4);\n'
    '\n'
    '        float4 v = make_float4(0.0f, 0.0f, 0.0f, 0.0f);\n'
    '        if (start_batch + r < n_batch) {\n'
    '            v = *(const float4 *) ((const char *) Q + h*nbq1 + (start_batch + r)*nbq2 + i_stream*nbq3 + c4*sizeof(float4));\n'
    '        }\n'
    '\n'
    '        q_shared[h][r][2*c4 + 0] = make_float2(v.x, v.y);\n'
    '        q_shared[h][r][2*c4 + 1] = make_float2(v.z, v.w);\n'
    '    }\n'
    '\n'
    '    if (tid < N_HEAD * TOKENS_PER_BLOCK) {\n'
    '        const int h = tid / TOKENS_PER_BLOCK;\n'
    '        const int r = tid % TOKENS_PER_BLOCK;\n'
    '        w_shared[h][r] = start_batch + r < n_batch ?\n'
    '            ((const float *) ((const char *) W + (start_batch + r)*nbw1 + i_stream*nbw3))[h] : 0.0f;\n'
    '    }\n'
    '\n'
    '    __syncthreads();\n'
    '\n'
    '    // phase 3 - float products of the widened keys for every head, ReLU, weight\n'
    '\n'
    '    const int kl = tid % KEY_LANES;\n'
    '    const int tl = tid / KEY_LANES;\n'
    '\n'
    '    float qk[N_HEAD][KEYS_PER_THREAD] = { { 0.0f } };\n'
    '\n'
    '#pragma unroll 8\n'
    '    for (int c = 0; c < N_EMBD_H2; ++c) {\n'
    '        float2 k_val[KEYS_PER_THREAD];\n'
    '#pragma unroll\n'
    '        for (int j = 0; j < KEYS_PER_THREAD; ++j) {\n'
    '            k_val[j] = __half22float2(k_shared[kl + j*KEY_LANES][c]);\n'
    '        }\n'
    '#pragma unroll\n'
    '        for (int h = 0; h < N_HEAD; ++h) {\n'
    '            const float2 q_val = q_shared[h][tl][c];\n'
    '#pragma unroll\n'
    '            for (int j = 0; j < KEYS_PER_THREAD; ++j) {\n'
    '                qk[h][j] = fmaf(k_val[j].x, q_val.x, qk[h][j]);\n'
    '                qk[h][j] = fmaf(k_val[j].y, q_val.y, qk[h][j]);\n'
    '            }\n'
    '        }\n'
    '    }\n'
    '\n'
    '    float score[KEYS_PER_THREAD] = { 0.0f };\n'
    '\n'
    '#pragma unroll\n'
    '    for (int h = 0; h < N_HEAD; ++h) {\n'
    '#pragma unroll\n'
    '        for (int j = 0; j < KEYS_PER_THREAD; ++j) {\n'
    '            score[j] += fmaxf(qk[h][j], 0.0f) * w_shared[h][tl];\n'
    '        }\n'
    '    }\n'
    '\n'
    '    // phase 4 - add the mask and write, consecutive threads write consecutive keys\n'
    '\n'
    '    const int i_batch = start_batch + tl;\n'
    '    if (i_batch >= n_batch) {\n'
    '        return;\n'
    '    }\n'
    '\n'
    '    const half * m_base = (const half *) ((const char *) M + i_batch*nbm1 + (i_stream%nem3)*nbm3);\n'
    '    float * dst_base = (float *) ((char *) dst + i_batch*nb1 + i_stream*nb3);\n'
    '\n'
    '#pragma unroll\n'
    '    for (int j = 0; j < KEYS_PER_THREAD; ++j) {\n'
    '        const int i_kv = start_kv + kl + j*KEY_LANES;\n'
    '        if (i_kv < n_kv) {\n'
    '            dst_base[i_kv] = score[j] + __half2float(m_base[i_kv]);\n'
    '        }\n'
    '    }\n'
    '}\n'
    '\n'
    '#define LIGHTNING_INDEXER_CASE(lightning_indexer_kernel, n_embd, n_head, K, type_K)'
)

_A_BRANCH = (
    '    } else if (n_embd == 128 && n_head == 4) {\n'
    '        // too few heads for a wmma tile, use vector kernel\n'
)

_N_BRANCH = (
    '    } else if (n_embd == 128 && n_head == 4 && n_batch >= LIGHTNING_INDEXER_TILE_TOKENS && bc_indexer_tile_enabled()) {\n'
    '        // too few heads for a wmma tile, the tile kernel shares the keys across the tokens\n'
    '        constexpr int WARPS_PER_BLOCK = 8;\n'
    '        constexpr int K_VECS_PER_BLOCK = 64;\n'
    '\n'
    '        static bool bc_hit = false;\n'
    '        if (!bc_hit && getenv("BIGCHERRY_PATCH_TRACE") != nullptr) {\n'
    '            bc_hit = true;\n'
    '            GGML_LOG_WARN("BIGCHERRY_PATCH_HIT patch=1335_tiled_lightning_indexer n_kv=%lld n_batch=%lld\\n", (long long) n_kv, (long long) n_batch);\n'
    '        }\n'
    '\n'
    '        dim3 block(32, WARPS_PER_BLOCK);\n'
    '        int num_kv_blocks = (n_kv + (K_VECS_PER_BLOCK) - 1) / (K_VECS_PER_BLOCK);\n'
    '        int num_batch_blocks = (n_batch + LIGHTNING_INDEXER_TILE_TOKENS - 1) / LIGHTNING_INDEXER_TILE_TOKENS;\n'
    '        dim3 grid(num_kv_blocks, num_batch_blocks, n_stream);\n'
    '\n'
    '        LIGHTNING_INDEXER_CASE(lightning_indexer_kernel_tile, 128, 4, k, GGML_TYPE_F16)\n'
    '        LIGHTNING_INDEXER_CASE(lightning_indexer_kernel_tile, 128, 4, k, GGML_TYPE_Q4_0)\n'
    '        LIGHTNING_INDEXER_CASE(lightning_indexer_kernel_tile, 128, 4, k, GGML_TYPE_Q4_1)\n'
    '        LIGHTNING_INDEXER_CASE(lightning_indexer_kernel_tile, 128, 4, k, GGML_TYPE_Q5_0)\n'
    '        LIGHTNING_INDEXER_CASE(lightning_indexer_kernel_tile, 128, 4, k, GGML_TYPE_Q5_1)\n'
    '        LIGHTNING_INDEXER_CASE(lightning_indexer_kernel_tile, 128, 4, k, GGML_TYPE_Q8_0)\n'
    '        LIGHTNING_INDEXER_CASE(lightning_indexer_kernel_tile, 128, 4, k, GGML_TYPE_BF16)\n'
    '        LIGHTNING_INDEXER_CASE(lightning_indexer_kernel_tile, 128, 4, k, GGML_TYPE_F32)\n'
    '        GGML_ABORT("fatal error");\n'
    '    } else if (n_embd == 128 && n_head == 4) {\n'
    '        // a batch smaller than a token tile (or BIGCHERRY_INDEXER_TILE=0), use vector kernel\n'
)

PATCHES = [
    FilePatch(
        path="ggml/src/ggml-cuda/lightning-indexer.cu",
        description="1335: upstream #29901 tiled lightning indexer for 4 heads (off switch BIGCHERRY_INDEXER_TILE=0)",
        language="none",
        edits=(
            Edit(id="indexer-tile-include", anchor=_re.escape(_A_INC), mode="replace", text=_N_INC,
                 guard=r"#include <cstdlib>  // BigCherry 1335: getenv", rationale="Last project include of the file.",
                 expect_matches=1, max_span_lines=2),
            Edit(id="indexer-tile-tokens", anchor=_re.escape(_A_TODO), mode="replace", text=_N_TODO,
                 guard=r"#define LIGHTNING_INDEXER_TILE_TOKENS 8",
                 rationale="Before the vector kernel's comment, where upstream defines the tile width.",
                 expect_matches=1, max_span_lines=2),
            Edit(id="indexer-tile-kernel", anchor=_re.escape(_A_CASE), mode="replace", text=_N_CASE,
                 guard=r"static __global__ void lightning_indexer_kernel_tile\(",
                 rationale="Before the dispatch macro, after the vector kernel.", expect_matches=1, max_span_lines=1),
            Edit(id="indexer-tile-dispatch", anchor=_re.escape(_A_BRANCH), mode="replace", text=_N_BRANCH,
                 guard=r"n_batch >= LIGHTNING_INDEXER_TILE_TOKENS && bc_indexer_tile_enabled\(\)",
                 rationale="The 4-head branch of ggml_cuda_lightning_indexer.", expect_matches=1, max_span_lines=3),
        ),
    ),
]

ENV_DOCS = (
    EnvDoc("BIGCHERRY_INDEXER_TILE", "0|1", "1",
           "upstream #29901 tiled lightning indexer for 4-head QSA indexers (Qwen4Exp); 0 keeps the vector kernel"),
)
