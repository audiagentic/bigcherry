"""HI70: deterministic direct-op correctness corpus for hard-to-reach candidates.

RE15 stage F's replay-full generate requires status='ok' correctness evidence
for every enumerated non-native candidate. 24 of gfx1201's 100 candidates
never got evidence from any real-model workload this project tried (several
llama-bench/server sweeps, three different real models) -- not because the
candidates are broken, but because reaching them requires a tensor shape or
batch width that well-formed production models structurally never produce:

- MMQ "fb1" (fallback) candidates only exist for a MUL_MAT whose src0 row
  count is NOT a multiple of 128 (mmq.cu's `fallback = ne01 % 128 != 0`) --
  K-quant block alignment means most production models never trigger this.
  12 of 16 fb1 candidates (q6_k/q8_0/q4_k) were eventually confirmed reachable
  by hunting for real models with an irregular dense dimension (gpt-oss-20B,
  Qwen3.5-9B, Qwen3.6-35B-A3B). The last 4 (q5_k) had no local model with an
  irregular Q5_K tensor.
- MMF f16 nwarps candidates require a MUL_MAT batch width (src1_ncols) in
  [1, 16] (mmf.cu's `should_use_mmf()` policy gate) -- a width only speculative
  decoding's multi-token draft-batch verification produces; no single-stream
  prompt-processing or decode call ever lands there.

Hunting for a model with exactly the right shape is not a reliable evidence
strategy -- it is luck-dependent and, for q5_k, ran out of local models
entirely (see HI70's notes for the full search). This patch instead adds a
small, deterministic, CPU-referenceable test_mul_mat corpus to
test-backend-ops that constructs the missing shapes directly, so `test
-o MUL_MAT` in tune-mode measures every candidate BigCherry's own eligibility
logic says should be reachable, without needing a lucky model or (for MMF)
speculative decoding at all.

Shapes, per HI70's gpt-auto-agent deep-dive recommendation:
- q5_k fb1 (4 candidates: j16/j32/j64/j128): M=127 (any M % 128 != 0), K=256,
  N=128 -- deterministically forces fallback=true regardless of quant type.
- mmf f16 nwarps (8 candidates: nw1..nw8): M=32 (MMF_ROWS_PER_BLOCK), K=256
  (RDNA4/wave32 F16 alignment), N swept across [2, 4, 8, 16] to cover
  should_use_mmf()'s whole [1, 16] eligibility window in one corpus rather
  than betting on a single N landing on every nwarps config's own screening
  path.

This is a correctness-evidence corpus, not a performance benchmark -- run in
`test` mode (CPU-referenced), not `perf`.
"""

GROUP = "core"
STATE = "validated"

from bigcherry.patcher import Edit, FilePatch

_CASES = """
    // bigcherry (HI70): deterministic direct-op corpus for MMQ fb1 / MMF
    // nwarps candidates a real-model workload structurally never reaches.
    // See patches/1100_hi70_direct_op_evidence.py for the full rationale.
    test_cases.emplace_back(new test_mul_mat(GGML_TYPE_Q5_K, GGML_TYPE_F32, 127, 128, 256, {1, 1}, {1, 1}));
    test_cases.emplace_back(new test_mul_mat(GGML_TYPE_F16, GGML_TYPE_F32, 32, 2, 256, {1, 1}, {1, 1}));
    test_cases.emplace_back(new test_mul_mat(GGML_TYPE_F16, GGML_TYPE_F32, 32, 4, 256, {1, 1}, {1, 1}));
    test_cases.emplace_back(new test_mul_mat(GGML_TYPE_F16, GGML_TYPE_F32, 32, 8, 256, {1, 1}, {1, 1}));
    test_cases.emplace_back(new test_mul_mat(GGML_TYPE_F16, GGML_TYPE_F32, 32, 16, 256, {1, 1}, {1, 1}));
"""

PATCH = FilePatch(
    path="tests/test-backend-ops.cpp",
    description="add a deterministic direct-op corpus for HI70's hard-to-reach MMQ fb1 / MMF nwarps candidates",
    edits=(
        Edit(
            id="hi70-direct-op-corpus",
            anchor=r"^    test_cases\.emplace_back\(new test_mul_mat_hadamard\(GGML_TYPE_F32, GGML_TYPE_F32, 1024, 1, 1024\)\);\s*$",
            rationale="right after the FWHT test block, still in the always-compiled active test-case list, before the disabled (#if 0) oversized-matrix block",
            text=_CASES,
            guard=r"bigcherry \(HI70\): deterministic direct-op corpus",
        ),
    ),
)

PATCHES = [PATCH]
