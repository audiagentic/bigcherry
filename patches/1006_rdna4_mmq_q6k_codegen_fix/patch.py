"""Upstream backport: RDNA4 MMQ Q6_K codegen fix (split from 1000).

Extracted 2026-09-16 (PA35) from the rejected combined patch
``1000_rdna4_mmq_q2k_q6k_fix`` after real hardware evidence found the Q6_K
half genuinely improves (1.365x, CI95 1.362x-1.367x, statistically
significant) while the Q2_K half regresses (0.959x, CI95 0.954x-0.963x) --
see ``patches/1000_rdna4_mmq_q2k_q6k_fix/SUMMARY.md``'s DEMOTION section for
the full disposition record and GPT design-review reference
(dev-gpt-agent, req_24ceb64100cb41dd). The two edits were already
structurally independent in 1000's ``patch.py``, so this extraction is a
plain split, not a reimplementation.

Cherry-picked from an unmerged upstream PR
(https://github.com/ggml-org/llama.cpp/pull/25940, still ``OPEN`` as of this
writing). ``ggml_cuda_mmq_vec_dot_q6_K_q8_1_mma`` -- the MFMA/WMMA tile-based
dot product used by RDNA3+/CDNA hardware -- multiplies an ``int`` tile value
against an ``int8_t`` scale without an explicit float promotion first; an
explicit ``(float)`` cast on ``C.x[l]`` is enough to change the code ROCm
generates.

The PR's own ``test-backend-ops perf -o MUL_MAT`` numbers (RDNA4, ROCm
7.15/TheRock 20260717): Q6_K n=512 36.62 -> 69.66 tok/s (1.90x) -- retained
here as the historical upstream claim. This project's own combined-package
real hardware measurement (2026-09-16, gfx1201, exact-shape backend-ops
paired A/B) found 1.365x [CI95 1.362x-1.367x], a real but smaller gain than
claimed. That measurement was made against the combined 1000 composition
(Q2_K change also present) and is NOT reused as sufficient promotion
evidence here -- this patch's own composition identity differs (Q6_K only),
so ``state`` starts at ``"untested"`` pending a fresh Q6-only hardware A/B
before any promotion to ``"validated"``.

Deliberately **not** backported: the PR's Q2_K unroll-guard change (real,
statistically significant regression on this project's hardware -- see
1000's rejection) and its second change (a hand-written RDNA4 native-select
heuristic narrowing ``ggml_cuda_should_use_mmq``'s heuristic), for the same
reason 1000 originally excluded it: this project's tuner already measures
candidates head-to-head per exact shape and hardware.
"""

from bigcherry.patcher import Edit, FilePatch

GROUP = "rdna-boosts"
STATE = "untested"

PATCH = FilePatch(
    path="ggml/src/ggml-cuda/mmq-vec-dot.cuh",
    description="RDNA4: float promotion for Q6_K MMA dot product "
                "(upstream PR #25940, split from 1000)",
    edits=(
        Edit(
            id="q6k-mma-float-promotion",
            # Unique in the file: the only occurrence of this exact
            # multiplicand chain (other sum[...] += C.x[l] * ... lines in this
            # file multiply against different operands entirely).
            anchor=r"sum\[\(j0/tile_C::J \+ n\)\*tile_C::ne \+ l\] \+= "
                   r"C\.x\[l\] \* sc\[k01/4\] \* x_df\[i\*sram_stride\] \* dB;",
            rationale="the Q6_K MMA accumulation inside "
                      "ggml_cuda_mmq_vec_dot_q6_K_q8_1_mma",
            mode="replace",
            text="sum[(j0/tile_C::J + n)*tile_C::ne + l] += "
                 "((float) C.x[l]) * sc[k01/4] * x_df[i*sram_stride] * dB;",
            guard=r"\(\(float\) C\.x\[l\]\) \* sc\[k01/4\] \* x_df\[i\*sram_stride\] \* dB;",
        ),
    ),
)

PATCHES = [PATCH]
