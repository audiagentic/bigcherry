"""Upstream backport: quantized cpy kernels launched with 1 thread per block.

Cherry-picked from upstream commit 69bf6437914596fbbc4caf09a7ac16f2acdd1a94
(merged to ggml-org/llama.cpp master 2026-08-08, PR #26731 -- already merged
and upstream-CI-tested, not an open/unreviewed proposal).

Eleven quantized `f32<->{q8_0,q4_0,q4_1,q5_0,q5_1}`/`f32->iq4_nl` copy-kernel
launches in ``cpy.cu`` computed ``num_blocks`` as one block per quantized
block (or, worse, one block per *element*) and launched with a hardcoded
``1`` thread per block -- serializing an entire block's dequant/quant
conversion onto a single thread. Meanwhile other cpy paths in the same file
(the two generic ``ggml_cpy_cuda``/``ggml_compute_cpy_cuda`` entry points,
~30 lines above the first affected function) already correctly compute
``num_blocks`` from ``CUDA_CPY_BLOCK_SIZE`` (64, defined in ``cpy.cuh``) and
launch with that many threads per block. This looks like a plain oversight
in a handful of call sites that never got the treatment the rest of the
file already has, not new mechanism -- which is why it's trusted here
without independent re-derivation of the fix itself.

Not gated to CUDA-only (no ``GGML_USE_HIP`` guard), applies to HIP builds
as-is. ``cpy.cu`` is untouched by any other patch in this project (core or
upstream-fixes), so this is a clean, zero-file-conflict backport.

REJECTED (PIN_REBASE_REVIEW_B10502.md action A3): source commit 69bf6437
is an ancestor of this project's own pinned base -- it was already mainline
before this patch was ever ported. Verified: every edit is "already-applied"
against a pristine checkout at both the b10362 and b10502 pins. This patch
has been a silent no-op since the day it was ported, not a functional
change. Left in the tree (not deleted) for provenance, but STATE=rejected
so it is never selected by a default recipe -- a no-op patch whose guard
matches upstream's own text becomes a loud rebase failure the moment
upstream evolves that text for an unrelated reason.
"""

from bigcherry.patcher import Edit, FilePatch

GROUP = "upstream-fixes"
STATE = "rejected"

# (id_suffix, num_blocks_expr, qk_const, template_class, direction)
# direction: "to_q" has an extra `ne % QK == 0` assert above (not part of the
# anchor -- the 3-line span below is already unique via the template class
# name), "from_q" divides by nothing (`num_blocks = ne`) and needs the /QK
# term added since the fixed formula counts *quantized* blocks, not elements.
_ENTRIES = [
    ("q8_0_to_q",   "QK8_0",  "cpy_blck_f32_q8_0",                       "to_q"),
    ("q8_0_from_q", "QK8_0",  "cpy_blck_q8_0_f32",                       "from_q"),
    ("q4_0_to_q",   "QK4_0",  "cpy_blck_f32_q4_0",                       "to_q"),
    ("q4_0_from_q", "QK4_0",  "cpy_blck_q_f32<dequantize_q4_0, QK4_0>",  "from_q"),
    ("q4_1_to_q",   "QK4_1",  "cpy_blck_f32_q4_1",                       "to_q"),
    ("q4_1_from_q", "QK4_1",  "cpy_blck_q_f32<dequantize_q4_1, QK4_1>",  "from_q"),
    ("q5_0_to_q",   "QK5_0",  "cpy_blck_f32_q5_0",                       "to_q"),
    ("q5_0_from_q", "QK5_0",  "cpy_blck_q_f32<dequantize_q5_0, QK5_0>",  "from_q"),
    ("q5_1_to_q",   "QK5_1",  "cpy_blck_f32_q5_1",                       "to_q"),
    ("q5_1_from_q", "QK5_1",  "cpy_blck_q_f32<dequantize_q5_1, QK5_1>",  "from_q"),
    ("iq4_nl_to_q", "QK4_NL", "cpy_blck_f32_iq4_nl",                     "to_q"),
]


def _edits():
    edits = []
    for suffix, qk, template, direction in _ENTRIES:
        if direction == "to_q":
            old_formula = f"const int64_t num_blocks = ne / {qk};"
            launch_fn = "cpy_f32_q"
        else:
            old_formula = "const int64_t num_blocks = ne;"
            launch_fn = "cpy_q_f32"

        new_formula = f"const int64_t num_blocks = (ne/{qk} + CUDA_CPY_BLOCK_SIZE - 1) / CUDA_CPY_BLOCK_SIZE;"

        # Anchor spans formula + the unchanged assert + the launch line. The
        # launch line's template class name is what makes an otherwise
        # duplicated `num_blocks = ne;` (five identical occurrences across
        # the from_q functions) unique per edit.
        anchor = (
            re_escape(old_formula) + r"\n"
            r"    GGML_ASSERT\(num_blocks <= INT_MAX\);\n"
            r"    " + re_escape(launch_fn) + r"<" + re_escape(template) + r", " + qk
            + r"><<<num_blocks, 1, 0, stream>>>"
        )
        text = (
            f"{new_formula}\n"
            "    GGML_ASSERT(num_blocks <= INT_MAX);\n"
            f"    {launch_fn}<{template}, {qk}><<<num_blocks, CUDA_CPY_BLOCK_SIZE, 0, stream>>>"
        )
        # The formula alone is not a safe guard: to_q and from_q for the same
        # type divide by the identical qk constant, so their fixed formulas
        # are byte-identical text. A guard on the formula alone would make
        # whichever edit runs first satisfy the other's guard too -- the same
        # kind of silent skip this project has been bitten by before (see
        # HI56's notes on extending a patch's text without changing its
        # guard). Anchoring the guard on the formula *plus* the launch line's
        # template class name makes each of the eleven guards unique again.
        guard = re_escape(new_formula) + r"\n    GGML_ASSERT\(num_blocks <= INT_MAX\);\n    " + re_escape(launch_fn) + r"<" + re_escape(template)

        edits.append(Edit(
            id=f"cpy-{suffix}",
            anchor=anchor,
            rationale=f"the {suffix} quantized cpy launch (upstream PR #26731)",
            mode="replace",
            text=text,
            guard=guard,
        ))
    return tuple(edits)


def re_escape(s: str) -> str:
    import re
    return re.escape(s)


PATCH = FilePatch(
    path="ggml/src/ggml-cuda/cpy.cu",
    description="Fix quantized cpy kernels launching with 1 thread per block "
                "instead of CUDA_CPY_BLOCK_SIZE (upstream PR #26731)",
    edits=_edits(),
)

PATCHES = [PATCH]
