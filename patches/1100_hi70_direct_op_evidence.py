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
  Qwen3.5-9B, Qwen3.6-35B-A3B), and the last 4 (q5_k) via an earlier,
  narrower version of this corpus. All 4 types are covered directly here now
  (2026-08-19): a real acceptance run's own tune measurements are scoped to
  ONE specific compiled binary/manifest, so reusing evidence gathered against
  a DIFFERENT binary (a different session's model-hunting runs) is fragile --
  read_correctness_evidence() ties evidence to the exact tune artifact
  supplied, and a fresh acceptance run has no fb1 evidence for any type until
  something exercises it. Covering all 4 types directly, deterministically,
  means every future acceptance run's OWN tune pass can regenerate complete
  fb1 evidence in one corpus run against its own binary, with no dependency
  on which real models happen to be available locally.
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
- fb1 (16 candidates: j16/j32/j64/j128 x q4_k/q5_k/q6_k/q8_0): M=127 (any
  M % 128 != 0), K=256, N=128 -- deterministically forces fallback=true
  regardless of quant type, so the same shape is repeated once per type.
- mmf f16 nwarps: M=32 (MMF_ROWS_PER_BLOCK), K=256 (RDNA4/wave32 F16
  alignment), N swept exhaustively across [1, 16] -- one case per width.
  BigCherry's MMF candidate identity is mmf:<type>:w<width>:nw<nwarps>, with
  width an INDEPENDENT axis from nwarps (confirmed via gpt-auto-agent
  review), so a partial N spread only proves evidence for whichever widths
  it happens to hit. should_use_mmf()'s policy gate is exactly
  `src1_ncols > 16 -> ineligible`, so [1, 16] is the complete eligible
  range -- sweeping all of it, rather than guessing which single width a
  given restricted inventory needs, is the only way this corpus reliably
  covers every mmf:f16:wN:nw1..nw8 candidate a real replay-full run might
  enumerate.

This is a correctness-evidence corpus, not a performance benchmark -- run in
`test` mode (CPU-referenced), not `perf`.
"""

GROUP = "core"
STATE = "validated"

from bigcherry.patcher import Edit, FilePatch

_FB1_TYPE_CASES = "\n".join(
    "    test_cases.emplace_back(new test_mul_mat(GGML_TYPE_%s, GGML_TYPE_F32, 127, 128, 256, {1, 1}, {1, 1}));" % t
    for t in ("Q4_K", "Q5_K", "Q6_K", "Q8_0")
)

_MMF_WIDTH_CASES = "\n".join(
    "    test_cases.emplace_back(new test_mul_mat(GGML_TYPE_F16, GGML_TYPE_F32, 32, %d, 256, {1, 1}, {1, 1}));" % n
    for n in range(1, 17)
)

_CASES = """
    // bigcherry (HI70): deterministic direct-op corpus for MMQ fb1 / MMF
    // nwarps candidates a real-model workload structurally never reaches.
    // See patches/1100_hi70_direct_op_evidence.py for the full rationale.
    //
    // fb1: M=127 (%%128 != 0) forces mmq.cu's fallback=true regardless of
    // quant type -- one case per type so any acceptance run's own tune pass
    // can regenerate complete fb1 evidence without depending on which real
    // models happen to be locally available.
%s
    //
    // BigCherry's MMF candidate identity is mmf:<type>:w<width>:nw<nwarps> --
    // width (N here) is an independent axis from nwarps, not implied by it
    // (confirmed via gpt-auto-agent review, 2026-08-19). should_use_mmf()'s
    // policy gate is `src1_ncols > 16 -> ineligible`, so N=1..16 is the
    // complete eligible range; sweep it exhaustively here rather than
    // guessing which single width RE15's own restricted inventory needed --
    // this covers every mmf:f16:wN:nw1..nw8 candidate regardless of which
    // exact width(s) turn out to be the ones a real replay-full catalog run
    // actually enumerates.
%s
""" % (_FB1_TYPE_CASES, _MMF_WIDTH_CASES)

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
