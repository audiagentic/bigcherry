"""Family-entry instrumentation and collection (coverage counters + HI13).

Not in the original plan. It exists because there is one question nothing else
answers: **what fraction of real matmul work actually reaches measured
dispatch?**

The dense selector is not the only route into the matmul families. The graph
optimiser calls `ggml_cuda_mul_mat_vec_q` and `ggml_cuda_mul_mat_vec_f`
directly for fused patterns, bypassing `ggml_cuda_mul_mat` entirely -- roughly
eight call sites. HI04 hooks two of the five collection points the plan lists
(section 9.1), so some unknown share of work is invisible to dispatch today.

Unknown is the problem. A tuning run over a fraction of the real work reports
its winners with exactly the same confidence as one over all of it. Until this
number exists, "we tuned the model" is an assumption.

`test-backend-ops` cannot produce it -- it drives ops directly rather than
through the graph optimiser, so every launch arrives via the dense selector and
coverage looks like 100%. It needs a real model.
"""

GROUP = "core"
STATE = "validated"

from bigcherry.patcher import Edit, FilePatch

# Both headers, because the family entries need the coverage counters (this
# patch) and the family collection hook (patch 0800). Including the dispatch
# header here also means these files see the declarations of the *_variant
# entry points they define, so a signature mismatch is a compile error rather
# than a link error.
_INCLUDE = """
#ifdef GGML_HIP_DISPATCH
#include "hip-autotune-coverage.h"
#include "hip-autotune-dispatch.cuh"
#endif
"""


def _count(family: str, fusion: str | None = None) -> str:
    """Coverage counter, then -- for the matmul families -- the HI13 hook.

    Emitted as one block rather than two patches, and the order inside it is
    load-bearing. The counter must run first: the hook returns early when
    dispatch handles the operation, so counting after it would miss exactly the
    launches the measurement exists to find.

    They are one patch because they attach to the same point and the second
    depends on the first. Two patches could not express that -- `apply_all`
    validates every patch against the on-disk file before writing anything, so
    a patch anchored on another patch's output can never pass the trial pass.
    """
    # Only the outermost entry counts. A dispatched launch re-enters its own
    # family entry point, so counting unconditionally inflates `executed` by
    # exactly the number of dispatched launches -- which made full coverage
    # read as 75% instead of 100% the first time this was measured.
    block = ("\n#ifdef GGML_HIP_DISPATCH\n"
             "    if (!ggml_hip_dispatch_is_reentrant()) {\n"
             f"        ggml_hip_coverage_count_executed({family});\n"
             "    }\n")
    if fusion is not None:
        block += (
            "    // bigcherry (HI13): family collection point. Catches launches\n"
            "    // that reached this family without passing through the dense\n"
            "    // selector -- fused graph paths, chiefly. Returns false for a\n"
            "    // re-entrant call already dispatched, so this fires only for\n"
            "    // traffic the dense hook cannot see.\n"
            f"    if (ggml_hip_dispatch_family(ctx, src0, src1, ids, dst, {fusion},\n"
            f"                                 {family})) {{\n"
            "        return;\n"
            "    }\n")
    return block + "#endif\n"


MMQ_PATCH = FilePatch(
    path="ggml/src/ggml-cuda/mmq.cu",
    description="count MMQ launches at the family entry point",
    edits=(
        Edit(
            id="mmq-coverage-include",
            anchor=r'^#include "mmid\.cuh"$',
            rationale="the mmq.cu include block",
            text=_INCLUDE,
            guard=r'#include "hip-autotune-coverage\.h"',
        ),
        Edit(
            id="mmq-coverage-count",
            anchor=r"^        int forced_J\) \{$",
            rationale="the top of ggml_cuda_mul_mat_q, after the HI06 patch",
            text=_count("GGML_HIP_FAMILY_MMQ", "/*fusion =*/ nullptr"),
            guard=r"ggml_hip_coverage_count_executed\(GGML_HIP_FAMILY_MMQ\)",
        ),
    ),
)

MMVQ_PATCH = FilePatch(
    path="ggml/src/ggml-cuda/mmvq.cu",
    description="count MMVQ launches at the family entry point",
    edits=(
        Edit(
            id="mmvq-coverage-include",
            anchor=r'^#include "vecdotq\.cuh"$',
            rationale="the mmvq.cu include block",
            text=_INCLUDE,
            guard=r'#include "hip-autotune-coverage\.h"',
        ),
        Edit(
            id="mmvq-coverage-count",
            # This is the entry the fused graph paths call directly, which is
            # exactly the traffic the dense-selector hook cannot see.
            #
            # The parameter list is matched as "however many continuation lines,
            # ending in `) {`" rather than as a fixed three lines: HI09 adds a
            # defaulted forced-geometry parameter on its own line, and upstream
            # is free to reflow the signature at any release.
            anchor=r"^void ggml_cuda_mul_mat_vec_q\(\n"
                   r"        ggml_backend_cuda_context & ctx,[^\n]*\n"
                   r"(?:[^\n]*\n)*?"
                   r"[^\n]*\) \{$",
            rationale="the top of ggml_cuda_mul_mat_vec_q",
            text=_count("GGML_HIP_FAMILY_MMVQ", "fusion"),
            guard=r"ggml_hip_coverage_count_executed\(GGML_HIP_FAMILY_MMVQ\)",
        ),
    ),
)

MMVF_PATCH = FilePatch(
    path="ggml/src/ggml-cuda/mmvf.cu",
    description="count MMVF launches at the family entry point",
    edits=(
        Edit(
            id="mmvf-coverage-include",
            anchor=r'^#include "mmvf\.cuh"$',
            rationale="the mmvf.cu include block",
            text=_INCLUDE,
            guard=r'#include "hip-autotune-coverage\.h"',
        ),
        Edit(
            id="mmvf-coverage-count",
            anchor=r"^    int forced_block_size, int forced_acc_f16\) \{$",
            rationale="the top of ggml_cuda_mul_mat_vec_f, after the HI07 patch",
            text=_count("GGML_HIP_FAMILY_MMVF", "fusion"),
            guard=r"ggml_hip_coverage_count_executed\(GGML_HIP_FAMILY_MMVF\)",
        ),
    ),
)

MMF_PATCH = FilePatch(
    path="ggml/src/ggml-cuda/mmf.cu",
    description="count MMF launches at the family entry point",
    edits=(
        Edit(
            id="mmf-coverage-include",
            anchor=r'^#include "mmf\.cuh"$',
            rationale="the mmf.cu include block",
            text=_INCLUDE,
            guard=r'#include "hip-autotune-coverage\.h"',
        ),
        Edit(
            id="mmf-coverage-count",
            anchor=r"^    int forced_nwarps\) \{$",
            rationale="the top of ggml_cuda_mul_mat_f, after the HI08 patch",
            text=_count("GGML_HIP_FAMILY_MMF", "/*fusion =*/ nullptr"),
            guard=r"ggml_hip_coverage_count_executed\(GGML_HIP_FAMILY_MMF\)",
        ),
    ),
)

BLAS_PATCH = FilePatch(
    path="ggml/src/ggml-cuda/ggml-cuda.cu",
    description="count BLAS launches at the family entry point",
    edits=(
        Edit(
            id="blas-coverage-include",
            anchor=r'^#include "ggml-cuda/mmvq\.cuh"$',
            rationale="the ggml-cuda.cu include block",
            text="\n#ifdef GGML_HIP_DISPATCH\n"
                 '#include "ggml-cuda/hip-autotune-coverage.h"\n'
                 "#endif\n",
            guard=r'#include "ggml-cuda/hip-autotune-coverage\.h"',
        ),
        Edit(
            id="blas-coverage-count",
            # Inside the real definition, not the HI04 forwarder, so launches
            # reaching cuBLAS by any route are counted. Counting only in the
            # forwarder would report zero executed against non-zero dispatched
            # -- which is what the first coverage run actually did, and it made
            # the total silently undercount.
            # Patch 0200 threads the optional runtime execution-options
            # pointer through this entry point before coverage is applied.
            # Accept both the upstream three-argument shape and that already
            # transformed four-argument shape; the exact one-match contract
            # still rejects any other signature.
            anchor=r"^static void ggml_cuda_mul_mat_cublas\(ggml_backend_cuda_context & ctx, "
                   r"const ggml_tensor \* src0, const ggml_tensor \* src1, ggml_tensor \* dst"
                   r"(?:, const void \* execution_options(?: = nullptr)?)?\) \{$",
            rationale="the definition of the cuBLAS dense entry point, before or after the dispatch ABI edit",
            text=_count("GGML_HIP_FAMILY_BLAS"),
            guard=r"ggml_hip_coverage_count_executed\(GGML_HIP_FAMILY_BLAS\)",
        ),
    ),
)

PATCHES = [MMQ_PATCH, MMVQ_PATCH, MMVF_PATCH, MMF_PATCH, BLAS_PATCH]
