"""RD26a: decode vs speculative-verify bit-identity -- base-standalone hunks.

Logical cluster (snapshot v2, five commits) that makes decode (n_q = 1) and
speculative-verify (n_q = n_draft+1) batches produce bit-identical logits,
which is a soundness precondition for speculative acceptance checks:

  - 93510434f  cuda : flash-attn decode/verify  (fattn-tile.cuh + fattn.cu)
  - b2655d381  cuda : non-flash attention decode/verify  (ggml-cuda.cu)
  - d152888fc  cpu  : decode/verify  (llamafile/sgemm.cpp)
  - 10b83d6b2  cuda : RDNA4 mmvq + fused SSM decode/verify   (RD26b, Wave 2)
  - 6cdf5aff9  cuda : RDNA3_0 mmvq decode/verify             (RD26b, Wave 2)

This module ports the two hunks whose pre-images anchor on the framework
base (pin 4801e3c + overlay + 15 framework modules) without any other RD
patch:

  1. b2655d381 in ggml-cuda.cu    -- MMVF decision for ne11 <= 8
  2. d152888fc in llamafile/sgemm.cpp -- reject n <= 8, not n < 2

DEFERRED (composition-gated, per the no-big-bang isolated-test policy):
the two hunks of 93510434f do NOT anchor on the framework base. Their
pre-images already contain code introduced by 1202 (RD04: type_KV template
parameter on flash_attn_tile, need_f16_K/need_f16_V launch arguments) and
1203 (RD05/06: GGML_CUDA_FA_WMMA_256 / wmma_max_head gating). They will be
added to this module (or a follow-up) once 1202 and 1203 have been benched
and retained, and applied strictly after them. The determinism property
only holds when the full five-commit cluster is in place; bench the cluster
together at that point.

The RD26b commits (10b83d6b2, 6cdf5aff9) reference RD21/RD24 fixed regions
and are ported from branch-tip state in Wave 2 (see RD26 notes).
"""

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
    # deferred, composition-gated on 1202 (RD04) and 1203 (RD05/06):
    "fork-commits-deferred": [
        "93510434f34ef48194e8d2e6fcc4a22289c3b8d8",  # fattn-tile.cuh + fattn.cu hunks
    ],
    "snapshot-head": "9e46e1fdc7a880f9ae9a2f9a693ae3e14c142a22",
    "snapshot-base": "4df29be4f4c3673f428170fda944a5b19f743bb8",
    "port-mode": "hand-anchored-standalone-hunks",
}
