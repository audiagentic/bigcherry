"""RD30 (AMD-MOE-001): compact the MoE MMQ launch grid.

Upstream's non-stream-K MMQ launch builds one full worst-case tile-column
per MoE expert regardless of that expert's real token occupancy:
``launch_mul_mat_q``'s ``block_nums_xy_tiling = dim3(nty, ntx, ntzw)`` sizes
``ntx`` from ``ncols_max`` (the WORST-case expert width) and ``ntzw`` from
``nchannels_y`` (== n_expert for MoE) -- so every expert gets ``ntx`` tile
launches even when its own real token count is far below the worst case.
On the real production model (Qwen3.6-35B-A3B, n_expert=256, confirmed via
real rocprofv3 profiling: Grid_Size_Z=256 on live MMQ MoE dispatches) this
launches far more blocks than the real routed-token distribution needs.

Concept ported from AMD-Ecosystem/llama.cpp PR #63 (fork-only, not
ancestral to our b10502 pin) -- a compact block map that flattens
(expert, expert-local-tile) pairs into one linear grid dimension, sized to
the REAL total tile count rather than n_expert * worst-case-tile-count.

Why this is a smaller change than it looks: the kernel body's non-stream-K
branch ALREADY early-returns for out-of-range tiles once ``jt*J >=
col_diff`` (real source, mmq.cuh, unchanged by this patch) -- so "wasted"
blocks under the current rectangular grid already do minimal work (two
global reads of ``expert_bounds`` plus a branch), not full compute. This
patch's real benefit is therefore reduced launch/dispatch overhead, not
eliminated compute cycles -- consistent with AMD's own modest reported
gain (+1.9..5.4% prefill) rather than a dramatic win, and it bounds the
correctness blast radius: a compaction bug shows up as missing/misrouted
work, not as a large, hard-to-attribute timing regression.

Design (dev-gpt-agent, verified against real AMD PR #63 source and our
actual pinned b10502 tree -- see docs/planning/active/rdna-boost-
experiments/RD30.md for the full transcript):

  - New prep kernel ``mmq_build_moe_block_map``, NOT an extension of
    ``mm_ids_helper`` (mmid.cu) -- compaction depends on J, the resolved
    MMQ tile width, and ``mm_ids_helper`` runs before J is chosen.
    ``mm_ids_helper`` (unchanged by this patch) already computes
    ``expert_bounds[]`` -- the exclusive prefix-sum of real per-expert
    token counts -- on every real MoE dispatch, before this prep kernel
    needs it.
  - AMD-faithful ``block_start[n_experts+1]`` + ``block_expert[max_m_
    blocks]`` scheme (no ``-1``-sentinel tail: out-of-range tiles are
    rejected by comparing against ``block_start[n_experts]``, the real
    total, not by scanning for a sentinel).
  - ``grid.z`` stays 1 for the compact path: b10502's MUL_MAT_ID MMQ path
    already asserts ``nsamples_y == 1`` for MoE, so there is no real
    ``wt`` dimension to preserve.
  - Gated to real gfx1100 only (``cc == GGML_CUDA_CC_RDNA3`` -- exact,
    since ``ggml_cuda_parse_id`` parses "gfx1100" to precisely this cc
    value; RDNA3.5 parses to a different, higher cc and is intentionally
    excluded pending its own hardware validation).
  - Fails closed to the EXACT legacy rectangular grid whenever
    ``max_m_blocks`` would exceed the grid.y limit or the prep kernel's
    shared-memory prefix-sum would exceed the device's per-block shared
    memory budget -- both checked before the compact path is taken, never
    discovered mid-launch.
  - Stream-K is untouched: it owns a different, pre-existing flattened
    tile enumeration (``block_nums_stream_k``, ``mul_mat_q_stream_k_
    fixup``) that this patch does not touch at all; the stream-K launch
    site gains only null compact-map parameters so the kernel's shared
    parameter list stays valid.

Validation consequence: this changes the real per-launch cost of every
affected (type, J, fallback) MoE MMQ candidate, so existing MMQ MoE
candidate timing/promotions are stale under this patch even though
candidate identity (J) is unchanged -- re-measurement is required before
any promotion, not just a build-clean patch.

Hardware status (2026-08-25, real dual-gfx1100 Brutus rig, VALIDATED):
  - Build: clean on both gfx1100-only and full gfx1100+gfx1201+gfx1030
    multi-arch builds.
  - Correctness: test-backend-ops MUL_MAT_ID passes under both native
    (869/869) and tune (q4_K 73/73, q8_0 75/75, 100% MMQ dispatch
    coverage) dispatch modes; a dedicated real-hardware test
    (tools/tests/rd30_hostile_test.cu / test_rd30_hostile_routing.py)
    checks mmq_build_moe_block_map directly against single-hot,
    concentrated-8-of-256, Zipf-skew, uniform, and degenerate all-zero
    routing distributions at real production scale (n_experts=256)
    against a host-side reference implementation -- 0/5 failed.
  - Timing: real production model (Qwen3.6-35B-A3B Q4_K_M, real
    n_expert=256), three INTERLEAVED baseline/RD30 rounds (not a single
    before/after) to rule out drift/noise as the explanation -- RD30 won
    every round by +0.73%..+0.90% pp512, with the two clusters never
    overlapping. See docs/planning/active/rdna-boost-experiments/RD30.md
    for the full evidence trail.

See docs/planning/active/rdna-boost-experiments/RD30.md for the complete
validation record.
"""

GROUP = "rdna-boosts"
STATE = "validated"

import re

from bigcherry.patcher import Edit, FilePatch

PROVENANCE = {
    "source-id": "amd-ecosystem-llama-cpp",
    "plan-item": "RD30",
    "fork-commit": "f8864197a4e2d07321bfb85466e3ba071e020aa9",
    "fork-commit-title": "mmq: compacted MoE tiling for RDNA3.5 (default on)",
    "adaptations": [
        "block_start[]/block_expert[] scheme kept faithful to PR #63's own "
        "design (rejected via block_start[n_experts], not a -1 sentinel).",
        "grid.z forced to 1 for the compact path: b10502's MUL_MAT_ID MMQ "
        "asserts nsamples_y == 1, so PR #63's more general wt handling is "
        "not needed here.",
        "Gated to cc == GGML_CUDA_CC_RDNA3 (gfx1100 exactly) rather than "
        "PR #63's RDNA3.5 target -- ported hardware differs from PR #63's "
        "validated hardware, so promotion needs its own gfx1100 evidence.",
        "ggml_hip_mmq_workspace (BigCherry-only file, no upstream "
        "counterpart) extended with the same compact-map sizing logic so "
        "the tuner's workspace accounting stays truthful under this patch.",
    ],
}

# ---------------------------------------------------------------------------
# ggml/src/ggml-cuda/mmq.cuh
# ---------------------------------------------------------------------------

_HELPERS = """// bigcherry (RD30/AMD-MOE-001): compact MoE MMQ launch grid.
//
// grid.y limit (CUDA/HIP): a dim3's y component is a 16-bit field.
static constexpr int RD30_MMQ_MAX_GRIDDIM_Y = 65535;

// Upper bound on the number of (expert, expert-local-tile) pairs the real
// routed-token distribution could produce: worst case is every expert's
// real tile count rounding up independently, which is bounded by
// ceil((ncols_dst + n_experts*(J-1)) / J) -- AMD PR #63's own bound.
static inline int64_t ggml_cuda_mmq_moe_compact_max_blocks(
        const int64_t ncols_dst, const int n_experts, const int J) {
    return (ncols_dst + int64_t(n_experts) * (J - 1) + J - 1) / J;
}

// RD30 experimental scope: gfx1100 only. GGML_CUDA_CC_RDNA3 is exactly
// gfx1100 (ggml_cuda_parse_id parses "gfx1100" to this precise cc value) --
// deliberately not GGML_CUDA_CC_IS_RDNA3(), which also admits RDNA3.5
// (different cc range, no hardware evidence for this path yet).
static inline bool ggml_cuda_mmq_moe_compact_enabled(const int cc) {
    return cc == GGML_CUDA_CC_RDNA3;
}

// Single-block prep kernel: turns expert_bounds[] (real per-expert token
// counts, from mm_ids_helper, unchanged by this patch) into a flat list of
// (expert, expert-local J-tile) pairs, so the compact launch grid's y
// dimension indexes real work only instead of n_experts * worst-case tiles.
//
// block_start[e] = number of compact blocks strictly before expert e's own
// tiles (an exclusive prefix sum over per-expert tile counts);
// block_start[n_experts] is therefore the REAL total block count, used by
// the launch to reject any padded tail of the host-computed upper bound.
// block_expert[m] = which expert owns compact block m.
//
// n_experts is 256 on the real production model (Qwen3.6-35B-A3B, real
// rocprofv3 profiling) -- one thread scanning <= a few hundred integers for
// the prefix sum is negligible next to the parallel scatter below.
static __global__ void mmq_build_moe_block_map(
        const int32_t * __restrict__ expert_bounds,
        const int n_experts,
        const int J,
        int32_t * __restrict__ block_start,
        int32_t * __restrict__ block_expert) {

    extern __shared__ int32_t rd30_s_start[];
    const int tid = threadIdx.x;

    for (int e = tid; e < n_experts; e += blockDim.x) {
        const int count = expert_bounds[e + 1] - expert_bounds[e];
        rd30_s_start[e] = (count + J - 1) / J;
    }
    __syncthreads();

    if (tid == 0) {
        int total = 0;
        for (int e = 0; e < n_experts; ++e) {
            const int count = rd30_s_start[e];
            rd30_s_start[e] = total;
            total += count;
        }
        rd30_s_start[n_experts] = total;
    }
    __syncthreads();

    for (int e = tid; e <= n_experts; e += blockDim.x) {
        block_start[e] = rd30_s_start[e];
    }

    const int total = rd30_s_start[n_experts];

    for (int m = tid; m < total; m += blockDim.x) {
        int lo = 0;
        int hi = n_experts;
        while (lo < hi) {
            const int mid = (lo + hi) >> 1;
            if (rd30_s_start[mid] <= m) {
                lo = mid + 1;
            } else {
                hi = mid;
            }
        }
        block_expert[m] = lo - 1;
    }
}

"""

_KERNEL_SIG_OLD = (
    "        const uint3 sample_ratio, const uint3 nsamples_y, "
    "const int stride_sample_x, const int stride_sample_y, "
    "const int stride_sample_dst,\n"
    "        const uint3 ntx) {"
)

_KERNEL_SIG_NEW = (
    "        const uint3 sample_ratio, const uint3 nsamples_y, "
    "const int stride_sample_x, const int stride_sample_y, "
    "const int stride_sample_dst,\n"
    "        const uint3 ntx,\n"
    "        const int32_t * __restrict__ rd30_block_expert,\n"
    "        const int32_t * __restrict__ rd30_block_start,\n"
    "        const int rd30_n_experts) {\n"
    "\n"
    "    // bigcherry (RD30): stream-K specializations never receive a "
    "compact map.\n"
    "    GGML_UNUSED_VARS(rd30_block_expert, rd30_block_start, "
    "rd30_n_experts);"
)

_BODY_DERIVE_OLD = (
    "    if constexpr (!ggml_cuda_mmq_get_stream_k(type, J, fallback)) {\n"
    "        const uint2 tmp2 = fast_div_modulo(blockIdx.z, nchannels_y);\n"
    "        const int wt = tmp2.x;\n"
    "        const int zt = tmp2.y;\n"
    "        const int jt = blockIdx.y;\n"
    "        const int it = blockIdx.x;\n"
)

_BODY_DERIVE_NEW = (
    "    if constexpr (!ggml_cuda_mmq_get_stream_k(type, J, fallback)) {\n"
    "        int wt;\n"
    "        int zt;\n"
    "        int jt;\n"
    "        const int it = blockIdx.x;\n"
    "\n"
    "        // bigcherry (RD30): compact MoE enumeration. blockIdx.y "
    "indexes the\n"
    "        // packed (expert, expert-local-tile) list built by "
    "mmq_build_moe_block_map;\n"
    "        // max_m_blocks (launch-side) is only a host-known UPPER "
    "bound, so the\n"
    "        // real count -- block_start[rd30_n_experts] -- is what "
    "actually gates this.\n"
    "        if (rd30_block_expert != nullptr) {\n"
    "            const int m_block = blockIdx.y;\n"
    "            if (m_block >= rd30_block_start[rd30_n_experts]) {\n"
    "                return;\n"
    "            }\n"
    "            zt = rd30_block_expert[m_block];\n"
    "            jt = m_block - rd30_block_start[zt];\n"
    "            // b10502's MUL_MAT_ID MMQ path asserts nsamples_y == 1.\n"
    "            wt = 0;\n"
    "        } else {\n"
    "            const uint2 tmp2 = fast_div_modulo(blockIdx.z, "
    "nchannels_y);\n"
    "            wt = tmp2.x;\n"
    "            zt = tmp2.y;\n"
    "            jt = blockIdx.y;\n"
    "        }\n"
)

_LAUNCH_NONSTREAMK_OLD = (
    "    if (!ggml_cuda_mmq_get_stream_k(type, J, fallback, cc)) {\n"
    "        mul_mat_q<type, J, fallback>"
    "<<<block_nums_xy_tiling, block_dims, nbytes_shared, stream>>>\n"
    "            (args.x, args.y, args.ids_dst, args.expert_bounds, "
    "args.dst, nullptr, args.y_scale,\n"
    "             blocks_per_ne00_fd, args.nrows_x, args.ncols_dst, "
    "args.stride_row_x, args.ncols_y, args.nrows_dst,\n"
    "             channel_ratio_fd, nchannels_y_fd, args.stride_channel_x, "
    "args.stride_channel_y, args.stride_channel_dst,\n"
    "             sample_ratio_fd, nsamples_y_fd, args.stride_sample_x, "
    "args.stride_sample_y, args.stride_sample_dst,\n"
    "             ntx_fd);\n"
    "        return;\n"
    "    }\n"
)

_LAUNCH_NONSTREAMK_NEW = (
    "    if (!ggml_cuda_mmq_get_stream_k(type, J, fallback, cc)) {\n"
    "        dim3 block_nums = block_nums_xy_tiling;\n"
    "\n"
    "        const int32_t * rd30_block_expert_ptr = nullptr;\n"
    "        const int32_t * rd30_block_start_ptr  = nullptr;\n"
    "        int rd30_n_experts = 0;\n"
    "\n"
    "        ggml_cuda_pool_alloc<int32_t> rd30_block_start(ctx.pool(id));\n"
    "        ggml_cuda_pool_alloc<int32_t> rd30_block_expert(ctx.pool(id));"
    "\n"
    "\n"
    "        const bool rd30_compact_candidate =\n"
    "            args.ids_dst != nullptr &&\n"
    "            args.expert_bounds != nullptr &&\n"
    "            args.nsamples_y == 1 &&\n"
    "            ggml_cuda_mmq_moe_compact_enabled(cc);\n"
    "\n"
    "        if (rd30_compact_candidate) {\n"
    "            rd30_n_experts = (int) args.nchannels_y;\n"
    "\n"
    "            const int64_t rd30_max_m_blocks = "
    "ggml_cuda_mmq_moe_compact_max_blocks(\n"
    "                args.ncols_dst, rd30_n_experts, J);\n"
    "\n"
    "            const size_t rd30_map_smem =\n"
    "                size_t(rd30_n_experts + 1) * sizeof(int32_t);\n"
    "\n"
    "            // Fail closed to the exact legacy grid whenever grid.y "
    "or the prep\n"
    "            // kernel's shared-memory prefix sum would exceed device "
    "limits.\n"
    "            if (rd30_max_m_blocks > 0 &&\n"
    "                    rd30_max_m_blocks < RD30_MMQ_MAX_GRIDDIM_Y &&\n"
    "                    rd30_map_smem <= "
    "ggml_cuda_info().devices[id].smpbo) {\n"
    "\n"
    "                rd30_block_start.alloc(rd30_n_experts + 1);\n"
    "                rd30_block_expert.alloc((size_t) rd30_max_m_blocks);\n"
    "\n"
    "                constexpr int rd30_build_nthreads = 256;\n"
    "                mmq_build_moe_block_map"
    "<<<1, rd30_build_nthreads, rd30_map_smem, stream>>>(\n"
    "                    args.expert_bounds,\n"
    "                    rd30_n_experts,\n"
    "                    J,\n"
    "                    rd30_block_start.ptr,\n"
    "                    rd30_block_expert.ptr);\n"
    "\n"
    "                rd30_block_start_ptr  = rd30_block_start.ptr;\n"
    "                rd30_block_expert_ptr = rd30_block_expert.ptr;\n"
    "\n"
    "                block_nums = dim3(nty, (unsigned int) "
    "rd30_max_m_blocks, 1);\n"
    "            }\n"
    "        }\n"
    "\n"
    "        mul_mat_q<type, J, fallback>"
    "<<<block_nums, block_dims, nbytes_shared, stream>>>\n"
    "            (args.x, args.y, args.ids_dst, args.expert_bounds, "
    "args.dst, nullptr, args.y_scale,\n"
    "             blocks_per_ne00_fd, args.nrows_x, args.ncols_dst, "
    "args.stride_row_x, args.ncols_y, args.nrows_dst,\n"
    "             channel_ratio_fd, nchannels_y_fd, args.stride_channel_x, "
    "args.stride_channel_y, args.stride_channel_dst,\n"
    "             sample_ratio_fd, nsamples_y_fd, args.stride_sample_x, "
    "args.stride_sample_y, args.stride_sample_dst,\n"
    "             ntx_fd,\n"
    "             rd30_block_expert_ptr, rd30_block_start_ptr, "
    "rd30_n_experts);\n"
    "        return;\n"
    "    }\n"
)

_LAUNCH_STREAMK_OLD = (
    "    mul_mat_q<type, J, fallback>"
    "<<<block_nums_stream_k, block_dims, nbytes_shared, stream>>>\n"
    "        (args.x, args.y, args.ids_dst, args.expert_bounds, args.dst, "
    "tmp_fixup.ptr, args.y_scale,\n"
    "         blocks_per_ne00_fd, args.nrows_x, args.ncols_dst, "
    "args.stride_row_x, args.ncols_y, args.nrows_dst,\n"
    "         channel_ratio_fd, nchannels_y_fd, args.stride_channel_x, "
    "args.stride_channel_y, args.stride_channel_dst,\n"
    "         sample_ratio_fd, nsamples_y_fd, args.stride_sample_x, "
    "args.stride_sample_y, args.stride_sample_dst,\n"
    "         ntx_fd);\n"
)

_LAUNCH_STREAMK_NEW = (
    "    mul_mat_q<type, J, fallback>"
    "<<<block_nums_stream_k, block_dims, nbytes_shared, stream>>>\n"
    "        (args.x, args.y, args.ids_dst, args.expert_bounds, args.dst, "
    "tmp_fixup.ptr, args.y_scale,\n"
    "         blocks_per_ne00_fd, args.nrows_x, args.ncols_dst, "
    "args.stride_row_x, args.ncols_y, args.nrows_dst,\n"
    "         channel_ratio_fd, nchannels_y_fd, args.stride_channel_x, "
    "args.stride_channel_y, args.stride_channel_dst,\n"
    "         sample_ratio_fd, nsamples_y_fd, args.stride_sample_x, "
    "args.stride_sample_y, args.stride_sample_dst,\n"
    "         ntx_fd,\n"
    "         nullptr, nullptr, 0);\n"
)

MMQ_CUH_PATCH = FilePatch(
    path="ggml/src/ggml-cuda/mmq.cuh",
    description="RD30: compact MoE MMQ launch grid (prep kernel, kernel "
                "signature/body, launch-side grid construction)",
    edits=(
        Edit(
            id="rd30-helpers",
            # Anchored on the template line + signature together, not just
            # the signature: launch_mul_mat_q's own
            # `template <ggml_type type, int J, bool fallback>` line
            # (identical text appears 7 other times in this file) must stay
            # directly adjacent to the function it templates. insert_before
            # on the signature alone would splice HELPERS between that
            # template line and the function, orphaning it -- type/J/
            # fallback would no longer be in scope inside the function body
            # (caught by a real gfx1100 compile: "undeclared identifier
            # 'type'/'J'/'fallback'" throughout launch_mul_mat_q).
            anchor=r"^template <ggml_type type, int J, bool fallback>\n"
                   r"static void launch_mul_mat_q\(ggml_backend_cuda_context & ctx, "
                   r"const mmq_args & args, cudaStream_t stream\) \{$",
            rationale="insert the compact-map helpers and prep kernel "
                      "directly before launch_mul_mat_q (template line "
                      "included so the template stays attached to the "
                      "function), which is the only caller",
            mode="insert_before",
            text=_HELPERS,
            guard=r"mmq_build_moe_block_map",
        ),
        Edit(
            id="rd30-kernel-signature",
            anchor=re.escape(_KERNEL_SIG_OLD),
            rationale="extend the mul_mat_q kernel signature with the "
                      "compact-map parameters",
            mode="replace",
            text=_KERNEL_SIG_NEW,
            guard=r"rd30_block_expert, rd30_block_start, rd30_n_experts\);",
        ),
        Edit(
            id="rd30-kernel-body-derive",
            anchor=re.escape(_BODY_DERIVE_OLD),
            rationale="derive wt/zt/jt from the compact block map when "
                      "present, exactly as before otherwise; everything "
                      "downstream (col_low/col_high/offset_y/offset_dst/"
                      "offset_x) is unchanged because zt and jt keep their "
                      "original meaning",
            mode="replace",
            text=_BODY_DERIVE_NEW,
            guard=r"zt = rd30_block_expert\[m_block\];",
        ),
        Edit(
            id="rd30-launch-nonstreamk",
            anchor=re.escape(_LAUNCH_NONSTREAMK_OLD),
            rationale="build the compact block map and launch a compacted "
                      "grid.y when eligible; fails closed to the exact "
                      "legacy block_nums_xy_tiling grid otherwise",
            mode="replace",
            text=_LAUNCH_NONSTREAMK_NEW,
            guard=r"rd30_block_expert_ptr, rd30_block_start_ptr, rd30_n_experts\);",
        ),
        Edit(
            id="rd30-launch-streamk-params",
            anchor=re.escape(_LAUNCH_STREAMK_OLD),
            rationale="stream-K's own launch gains only null compact-map "
                      "parameters -- stream-K's tile enumeration is "
                      "untouched",
            mode="replace",
            text=_LAUNCH_STREAMK_NEW,
            guard=r"nullptr, nullptr, 0\);",
        ),
    ),
)

# ---------------------------------------------------------------------------
# ggml/src/ggml-cuda/hip-autotune-dispatch.cu
# ---------------------------------------------------------------------------

_WORKSPACE_MOE_RETURN_OLD = (
    "        const size_t ne_get_rows = (size_t) (sig.ne1[2] * sig.n_expert_used);\n"
    "        const size_t id_bytes =\n"
    "            2 * ne_get_rows * sizeof(int32_t)\n"
    "            + (size_t) (sig.n_expert + 1) * sizeof(int32_t);\n"
    "        return activation_bytes + id_bytes;\n"
    "    }\n"
)

_WORKSPACE_MOE_RETURN_NEW = (
    "        const size_t ne_get_rows = (size_t) (sig.ne1[2] * sig.n_expert_used);\n"
    "        const size_t id_bytes =\n"
    "            2 * ne_get_rows * sizeof(int32_t)\n"
    "            + (size_t) (sig.n_expert + 1) * sizeof(int32_t);\n"
    "\n"
    "        // bigcherry (RD30): the compact MoE MMQ launch (mmq.cuh) "
    "pool-allocates\n"
    "        // block_start[n_experts+1] + block_expert[max_m_blocks] "
    "when eligible.\n"
    "        // effective_J mirrors the candidate's own J when this "
    "descriptor names\n"
    "        // one (self->variant.primary != 0), falling back to "
    "native's own scan\n"
    "        // for the native/unforced descriptor -- the same fallback "
    "mmq.cuh's own\n"
    "        // launch uses when nothing forces J.\n"
    "        size_t rd30_compact_map_bytes = 0;\n"
    "        if (ggml_cuda_mmq_moe_compact_enabled(cc)) {\n"
    "            const int rd30_n_experts = (int) sig.n_expert;\n"
    "\n"
    "            int rd30_effective_J = self->variant.primary;\n"
    "            if (rd30_effective_J == 0) {\n"
    "                rd30_effective_J = ggml_cuda_mmq_native_j_best("
    "type, fallback, sig.ne1[2]);\n"
    "            }\n"
    "\n"
    "            if (rd30_effective_J > 0 &&\n"
    "                    !ggml_cuda_mmq_get_stream_k(type, "
    "rd30_effective_J, fallback, cc)) {\n"
    "                const int64_t rd30_max_m_blocks = "
    "ggml_cuda_mmq_moe_compact_max_blocks(\n"
    "                    (int64_t) ne_get_rows, rd30_n_experts, "
    "rd30_effective_J);\n"
    "\n"
    "                const size_t rd30_map_smem = size_t(rd30_n_experts "
    "+ 1) * sizeof(int32_t);\n"
    "\n"
    "                if (rd30_max_m_blocks > 0 &&\n"
    "                        rd30_max_m_blocks < "
    "RD30_MMQ_MAX_GRIDDIM_Y &&\n"
    "                        rd30_map_smem <= "
    "ggml_cuda_info().devices[ggml_cuda_get_device()].smpbo) {\n"
    "                    rd30_compact_map_bytes =\n"
    "                        (size_t(rd30_n_experts + 1) + "
    "size_t(rd30_max_m_blocks)) * sizeof(int32_t);\n"
    "                }\n"
    "            }\n"
    "        }\n"
    "\n"
    "        return activation_bytes + id_bytes + rd30_compact_map_bytes;\n"
    "    }\n"
)

WORKSPACE_PATCH = FilePatch(
    path="ggml/src/ggml-cuda/hip-autotune-dispatch.cu",
    description="RD30: account for the compact MoE MMQ launch's pool "
                "allocations in the tuner's workspace upper bound",
    edits=(
        Edit(
            id="rd30-workspace-moe",
            anchor=re.escape(_WORKSPACE_MOE_RETURN_OLD),
            rationale="add the compact block-map's pool bytes to the "
                      "MUL_MAT_ID workspace upper bound",
            mode="replace",
            text=_WORKSPACE_MOE_RETURN_NEW,
            guard=r"return activation_bytes \+ id_bytes \+ rd30_compact_map_bytes;",
        ),
    ),
)

PATCHES = [MMQ_CUH_PATCH, WORKSPACE_PATCH]
