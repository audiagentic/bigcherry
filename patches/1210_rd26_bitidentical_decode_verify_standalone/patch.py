"""RD26a: decode vs speculative-verify bit-identity -- base-standalone hunks.

Logical cluster (snapshot v2, five commits) that makes decode (n_q = 1) and
speculative-verify (n_q = n_draft+1) batches produce bit-identical logits,
which is a soundness precondition for speculative acceptance checks:

  - 93510434f  cuda : flash-attn decode/verify  (fattn-tile.cuh + fattn.cu)
  - b2655d381  cuda : non-flash attention decode/verify  (ggml-cuda.cu)
  - d152888fc  cpu  : decode/verify  (llamafile/sgemm.cpp)
  - 10b83d6b2  cuda : RDNA4 mmvq + fused SSM decode/verify   (RD26b, Wave 2)
  - 6cdf5aff9  cuda : RDNA3_0 mmvq decode/verify             (RD26b, Wave 2)

This module ports all three GPU/CPU halves at b11126 (PRBE20, 2026-09-25):

  1. b2655d381 in ggml-cuda.cu        -- MMVF decision for ne11 <= 8
  2. d152888fc in llamafile/sgemm.cpp -- reject n <= 8, not n < 2
  3. wave 1, flash attention          -- n_q <= 8 stays off WMMA and uses the
                                         decode tile config (fattn.cu,
                                         fattn-tile.cuh)
  4. wave 2, RDNA3/RDNA4 MMVQ         -- calc_nwarps whitelist applies to
                                         ncols_dst 1..8, not only 1

The fork was squash-rebased: 93510434f/10b83d6b2/6cdf5aff9 no longer exist
there; waves 1 and 2 are taken from "block 08" (5efcd85f, fused-core
prefill kernels and GPU bit-identical), restricted to the determinism
hunks. Its unrelated changes (Q6_K nwarps 2->8 tuning, fused prefill
kernels, fused SSM decode) are deliberately NOT ported. Wave 1 no longer
depends on 1202/1203: b11126's own tile launcher and WMMA gate are the
pre-images. Known gap: with a quantized KV cache, decode (n_q 1..2) takes the
vector kernel while n_q 3..8 take the tile kernel, so identity is only
claimed for F16/BF16 KV.
"""

import re

from bigcherry.patcher import Edit, FilePatch

PATCHES = [
    FilePatch(
        path="ggml/src/ggml-cuda/ggml-cuda.cu",
        description="MMVF kernel decision for all decode-scale batches (RD26a, rdna-boosts b2655d381)",
        edits=(
            Edit(
                id="rd26a-mmvf-decode-verify",
                anchor=r"    if \(ggml_cuda_should_use_mmvf\(src0->type, cc, src0->ne, src0->nb, ne11\)\) \{",
                rationale="rdna-boosts b2655d381 (RD26a): use the decode (ne11 = 1) MMVF decision for all batches with ne11 <= 8",
                mode="replace",
                text=(
                    "    // Speculative verify batches (ne11 = n_q <= 8) must run the same kernel as\n"
                    "    // decode (ne11 = 1): decode uses the MMVF kernel, while a larger batch can\n"
                    "    // fall through to MMF, which accumulates differently and produces different\n"
                    "    // logits. Use the decode (ne11 = 1) config for all small batches.\n"
                    "    const int64_t ne11_mmvf = ne11 <= MMVF_MAX_BATCH_SIZE ? 1 : ne11;\n"
                    "    if (ggml_cuda_should_use_mmvf(src0->type, cc, src0->ne, src0->nb, ne11_mmvf)) {"
                ),
                guard=r"const\ int64_t\ ne11_mmvf\ =\ ne11\ <=\ MMVF_MAX_BATCH_SIZE\ \?\ 1\ :\ ne11;",
                max_span_lines=1,
            ),
        ),
    ),
    FilePatch(
        path="ggml/src/ggml-cpu/llamafile/sgemm.cpp",
        description="llamafile sgemm batch gate n <= 8 (RD26a, rdna-boosts d152888fc)",
        edits=(
            Edit(
                id="rd26a-sgemm-batch-gate",
                anchor=(
                    r"[ ]{44,46}"
                    r"\n#if\ !defined\(__MMA__\)"
                    r"\n    if \(n\ <\ 2\)"
                    r"\n        return false;"
                    r"\n#endif"
                ),
                rationale="rdna-boosts d152888fc (RD26a): route every decode-scale batch (n_q 1..8) through the plain vec_dot path; llamafile only for prefill",
                mode="replace",
                text=(
                    "    // only enable sgemm for prompt processing\n"
                    "    // decode (n_q = 1) and speculative verify batches (n_q = n_draft+1) must run the\n"
                    "    // same kernels: llamafile rejects n < 2, so verify batches (n_q >= 2) ran tinyBLAS\n"
                    "    // while decode ran vec_dot, accumulating differently and breaking the\n"
                    "    // batch-vs-seq bit-identity that speculative acceptance checks rely on.\n"
                    "    // Route every decode-scale batch through the plain vec_dot path and use\n"
                    "    // llamafile only for prefill (n_q > 8, i.e. beyond the speculative batch range).\n"
                    "#if !defined(__MMA__)\n"
                    "    if (n <= 8)\n"
                    "        return false;\n"
                    "#endif"
                ),
                guard=r"    if \(n\ <=\ 8\)\n        return false;",
                max_span_lines=5,
            ),
        ),
    ),
    # PRBE20 Wave 1 (fork block 08, 5efcd85f; was 93510434f before the fork's
    # squash rebase): speculative verify batches (n_q <= 8) must run the same
    # flash-attention kernel and tile config as decode (n_q = 1).
    FilePatch(
        path="ggml/src/ggml-cuda/fattn.cu",
        description="Keep n_q <= 8 attention off WMMA (decode never uses it) (RD26 wave 1)",
        edits=(
            Edit(
                id="rd26-fattn-no-wmma-for-verify",
                anchor=re.escape(
                    "            Q->ne[1] * gqa_ratio_eff > (Q->ne[0] <= 128 ? 8 : 16)) {\n"
                    "        return BEST_FATTN_KERNEL_MMA_F16;"
                ),
                rationale="ggml_cuda_get_best_fattn_kernel: AMD WMMA selection; decode (n_q=1) "
                          "never qualifies, so verify batches (n_q<=8) must not either",
                mode="replace",
                text=(
                    "            Q->ne[1] * gqa_ratio_eff > (Q->ne[0] <= 128 ? 8 : 16) && Q->ne[1] > 8) {\n"
                    "        return BEST_FATTN_KERNEL_MMA_F16;"
                ),
                guard=r"Q->ne\[1\] \* gqa_ratio_eff > \(Q->ne\[0\] <= 128 \? 8 : 16\) && Q->ne\[1\] > 8\)",
            ),
        ),
    ),
    FilePatch(
        path="ggml/src/ggml-cuda/fattn-tile.cuh",
        description="Use the decode tile config for every n_q <= 8 (RD26 wave 1)",
        edits=(
            Edit(
                id="rd26-fattn-tile-decode-config-for-verify",
                anchor=re.escape(
                    "    constexpr size_t nbytes_shared = 0;\n"
                    "\n"
                    "#ifdef GGML_USE_HIP\n"
                    "    if constexpr (DKQ <= 128) {\n"
                    "        if (Q->ne[1] > 32/ncols2) {"
                ),
                rationale="launch_fattn_tile_switch_ncols1: cols_per_block picks nwarps/nbatch_fa; "
                          "n_q=1 selects max(ncols2,2), so n_q<=8 must select it too",
                mode="replace",
                text=(
                    "    constexpr size_t nbytes_shared = 0;\n"
                    "\n"
                    "    // bigcherry RD26: speculative verify batches (n_q = n_draft+1 <= 8) must\n"
                    "    // produce the same logits as decode (n_q = 1). nwarps and nbatch_fa come\n"
                    "    // from cols_per_block, so use decode's cols_per_block for every n_q <= 8;\n"
                    "    // prefill keeps the tuned configs below. Same instantiation decode uses.\n"
                    "    if (Q->ne[1] <= 8) {\n"
                    "        constexpr int cols_per_block = ncols2 > 2 ? ncols2 : 2;\n"
                    "        const int nwarps    = ggml_cuda_fattn_tile_get_nthreads (DKQ, DV, cols_per_block, cc) / warp_size;\n"
                    "        const int nbatch_fa = ggml_cuda_fattn_tile_get_nbatch_fa(DKQ, DV, cols_per_block, cc);\n"
                    "        fattn_kernel_t fattn_kernel = flash_attn_tile<DKQ, DV, cols_per_block/ncols2, ncols2, use_logit_softcap>;\n"
                    "        launch_fattn<DV, cols_per_block/ncols2, ncols2>\n"
                    "            (ctx, dst, fattn_kernel, nwarps, nbytes_shared, nbatch_fa, true, true, false, false, warp_size);\n"
                    "        return;\n"
                    "    }\n"
                    "\n"
                    "#ifdef GGML_USE_HIP\n"
                    "    if constexpr (DKQ <= 128) {\n"
                    "        if (Q->ne[1] > 32/ncols2) {"
                ),
                guard=r"constexpr int cols_per_block = ncols2 > 2 \? ncols2 : 2;",
            ),
        ),
    ),
    # PRBE20 Wave 2 (fork block 08): the RDNA3/RDNA4 nwarps whitelist applied
    # only to ncols_dst == 1, so a verify batch (ncols_dst 2..8) reduced each
    # row with a different warp split than decode.
    FilePatch(
        path="ggml/src/ggml-cuda/mmvq.cu",
        description="RDNA3/RDNA4 MMVQ nwarps identical for ncols_dst 1..8 (RD26 wave 2)",
        edits=(
            Edit(
                id="rd26-mmvq-nwarps-rdna4",
                anchor=re.escape(
                    "        if (ncols_dst == 1) {\n"
                    "            switch (type) {\n"
                    "                case GGML_TYPE_Q4_0:\n"
                    "                case GGML_TYPE_Q4_1:\n"
                    "                case GGML_TYPE_Q5_0:\n"
                    "                case GGML_TYPE_Q5_1:\n"
                    "                case GGML_TYPE_Q8_0:\n"
                    "                case GGML_TYPE_Q2_K:"
                ),
                rationale="calc_nwarps MMVQ_PARAMETERS_RDNA4 whitelist (the only branch listing Q2_K)",
                mode="replace",
                text=(
                    "        if (ncols_dst <= MMVQ_MAX_BATCH_SIZE) {\n"
                    "            switch (type) {\n"
                    "                case GGML_TYPE_Q4_0:\n"
                    "                case GGML_TYPE_Q4_1:\n"
                    "                case GGML_TYPE_Q5_0:\n"
                    "                case GGML_TYPE_Q5_1:\n"
                    "                case GGML_TYPE_Q8_0:\n"
                    "                case GGML_TYPE_Q2_K:"
                ),
                guard=r"if \(ncols_dst <= MMVQ_MAX_BATCH_SIZE\) \{\n\s*switch \(type\) \{\n(?:\s*case GGML_TYPE_\w+:\n){5}\s*case GGML_TYPE_Q2_K:",
            ),
            Edit(
                id="rd26-mmvq-nwarps-rdna3",
                anchor=re.escape(
                    "        if (ncols_dst == 1) {\n"
                    "            switch (type) {\n"
                    "                case GGML_TYPE_Q4_0:\n"
                    "                case GGML_TYPE_Q4_1:\n"
                    "                case GGML_TYPE_Q5_0:\n"
                    "                case GGML_TYPE_Q5_1:\n"
                    "                case GGML_TYPE_Q8_0:\n"
                    "                    return 8;"
                ),
                rationale="calc_nwarps MMVQ_PARAMETERS_RDNA3_0 whitelist (Q8_0 returns 8 directly)",
                mode="replace",
                text=(
                    "        if (ncols_dst <= MMVQ_MAX_BATCH_SIZE) {\n"
                    "            switch (type) {\n"
                    "                case GGML_TYPE_Q4_0:\n"
                    "                case GGML_TYPE_Q4_1:\n"
                    "                case GGML_TYPE_Q5_0:\n"
                    "                case GGML_TYPE_Q5_1:\n"
                    "                case GGML_TYPE_Q8_0:\n"
                    "                    return 8;"
                ),
                guard=r"if \(ncols_dst <= MMVQ_MAX_BATCH_SIZE\) \{\n\s*switch \(type\) \{\n(?:\s*case GGML_TYPE_\w+:\n){5}\s*return 8;",
            ),
        ),
    ),
]

GROUP = "rdna-boosts"
STATE = "untested"

PROVENANCE = {
    "source-id": "stew675-rdna-boosts",
    "plan-item": "RD26",
    # representative commit of the two base-standalone hunks ported here
    "fork-commit": "b2655d381b9575d644ebef794869165ede14b3a3",
    "fork-commits": [
        "b2655d381b9575d644ebef794869165ede14b3a3",  # ggml-cuda.cu hunk (ported here)
        "d152888fc34419c69fb946581f29f927a475b5fa",  # sgemm.cpp hunk (ported here)
    ],
    # waves 1+2 ported from the rebased fork's block 08 (the original
    # 93510434f/10b83d6b2/6cdf5aff9 identities were squashed away)
    "fork-commits-rebased": [
        "5efcd85fb4cd8845c6c7dd47c50e2666264aa4eb",  # block 08: fattn + mmvq nwarps hunks
    ],
    "snapshot-head": "9e46e1fdc7a880f9ae9a2f9a693ae3e14c142a22",
    "snapshot-base": "4df29be4f4c3673f428170fda944a5b19f743bb8",
    "port-mode": "hand-anchored-standalone-hunks",
}
