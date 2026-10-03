"""1294: deterministic tie-break in the HIP parallel radix TOP_K (lowest column index wins).

The radix select finds the cut key, then top_k_radix_gather writes every column above the cut with an
atomic counter and fills the remaining `rank` slots with columns EQUAL to the cut in atomic arrival order.
Which tied columns are taken therefore varies run to run. Qwen4Exp's QSA indexer scores pooled KV blocks
with sums of ReLU, so many scores are exactly 0.0; once the context holds more pools than top_k/kpool the
cut falls inside that tie and the attended KV blocks change between runs (greedy text diverged across
server starts at 32K and 80K context with every AllReduce provider; tools/lab/flash-next/determinism.sh).

The gather now skips the equal branch, and one extra block per row walks the columns in ascending order
in tiles of 256, ranks each tied column by a ballot/popcount prefix count, and writes the first `rank` of
them: the lowest-index tied columns, independent of scheduling. Output order of the k indices is still
unspecified (the consumers treat them as a set). BIGCHERRY_TOPK_DETERMINISTIC=0 restores the old path.
BIGCHERRY_PATCH_HIT patch=1294_topk_deterministic_ties logs once under BIGCHERRY_PATCH_TRACE.
"""

import re as _re

from bigcherry.patcher import Edit, FilePatch

GROUP = "core"
STATE = "evaluated"

_TIE_KERNEL = """// BigCherry 1294: deterministic tie-break - the rank lowest-index columns equal to the cut, one block per row.
template<int BLOCK_SIZE>
static __global__ void top_k_radix_gather_ties(
        const float * __restrict__ src,
        int * __restrict__ dst,
        const top_k_radix_state * __restrict__ states,
        int ncols,
        int k) {
    const int row = blockIdx.x;
    const top_k_radix_state st = states[row];
    if (st.rank <= 0) {
        return;
    }
    const float * row_src = src + (size_t) row * ncols;
    int * row_dst = dst + (size_t) row * k;

    __shared__ int warp_cnt[BLOCK_SIZE / 32];
    __shared__ int base_s;
    const int lane   = threadIdx.x % warpSize;
    const int wid    = threadIdx.x / warpSize;
    const int nwarps = BLOCK_SIZE / warpSize;
    if (threadIdx.x == 0) {
        base_s = 0;
    }
    __syncthreads();

    for (int c0 = 0; c0 < ncols; c0 += BLOCK_SIZE) {
        const int col  = c0 + threadIdx.x;
        const bool tie = col < ncols && top_k_float_to_ordered(row_src[col]) == st.prefix;
        const unsigned long long m = __ballot(tie);
        const int below = __popcll(m & ((1ull << lane) - 1ull));
        if (lane == 0) {
            warp_cnt[wid] = __popcll(m);
        }
        __syncthreads();
        int off = base_s;
        int total = 0;
        for (int w = 0; w < nwarps; ++w) {
            off   += w < wid ? warp_cnt[w] : 0;
            total += warp_cnt[w];
        }
        const int r = off + below;
        if (tie && r < st.rank) {
            row_dst[k - st.rank + r] = col;
        }
        __syncthreads();
        if (threadIdx.x == 0) {
            base_s += total;
        }
        __syncthreads();
        if (base_s >= st.rank) {
            break;
        }
    }
}

static bool top_k_bc_deterministic_ties() {
    static const bool det = [] {
        const char * e = getenv("BIGCHERRY_TOPK_DETERMINISTIC");
        const bool on = e == nullptr || strcmp(e, "0") != 0;
        if (on && getenv("BIGCHERRY_PATCH_TRACE") != nullptr) {
            GGML_LOG_WARN("BIGCHERRY_PATCH_HIT patch=1294_topk_deterministic_ties\\n");
        }
        return on;
    }();
    return det;
}

"""

TOPK = FilePatch(
    path="ggml/src/ggml-cuda/top-k.cu",
    language="none",
    description="Radix TOP_K: lowest-index deterministic tie-break instead of atomic arrival order.",
    edits=(
        Edit(
            id="topk-gather-det-param",
            anchor=_re.escape(
                "static __global__ void top_k_radix_gather(\n"
                "        const float * __restrict__ src,\n"
                "        int * __restrict__ dst,\n"
                "        top_k_radix_state * __restrict__ states,\n"
                "        int ncols,\n"
                "        int k,\n"
                "        int blocks_per_row) {"
            ),
            text=(
                "static __global__ void top_k_radix_gather(\n"
                "        const float * __restrict__ src,\n"
                "        int * __restrict__ dst,\n"
                "        top_k_radix_state * __restrict__ states,\n"
                "        int ncols,\n"
                "        int k,\n"
                "        int blocks_per_row,\n"
                "        int det_ties) {"
            ),
            mode="replace",
            guard=r"int blocks_per_row,\n        int det_ties\) \{",
            expect_matches=1,
            rationale="The gather kernel's exact signature; it is the only writer of tied columns.",
        ),
        Edit(
            id="topk-gather-skip-equal",
            anchor=_re.escape("        } else if (key == state->prefix) {"),
            text="        } else if (!det_ties && key == state->prefix) {",
            mode="replace",
            guard=_re.escape("} else if (!det_ties && key == state->prefix) {"),
            expect_matches=1,
            rationale="The atomic-order equal branch in top_k_radix_gather.",
        ),
        Edit(
            id="topk-tie-kernel",
            anchor=r"(?m)^static void top_k_radix_cuda\(",
            text=_TIE_KERNEL,
            mode="insert_before",
            guard=r"top_k_radix_gather_ties",
            expect_matches=1,
            rationale="Defined before the launcher that uses it.",
        ),
        Edit(
            id="topk-launch-ties",
            anchor=_re.escape("            src, dst, states, ncols, k, blocks_per_row);\n}"),
            text=(
                "            src, dst, states, ncols, k, blocks_per_row, det_ties ? 1 : 0);\n"
                "    if (det_ties) {\n"
                "        top_k_radix_gather_ties<BLOCK_SIZE><<<nrows, BLOCK_SIZE, 0, stream>>>(src, dst, states, ncols, k);\n"
                "    }\n"
                "}"
            ),
            mode="replace",
            guard=r"top_k_radix_gather_ties<BLOCK_SIZE><<<nrows",
            expect_matches=1,
            rationale="The single gather launch at the end of top_k_radix_cuda.",
        ),
        Edit(
            id="topk-det-flag",
            anchor=_re.escape("    top_k_radix_reset_counters\n"),
            text="    const bool det_ties = top_k_bc_deterministic_ties();\n    top_k_radix_reset_counters\n",
            mode="replace",
            guard=r"const bool det_ties = top_k_bc_deterministic_ties\(\);",
            expect_matches=1,
            rationale="Read the switch once in the launcher before the gather launches.",
        ),
    ),
)

PATCHES = [TOPK]
