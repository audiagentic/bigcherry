# 1250_nro01_allreduce_q8_wire

Plan: `NRO01`. State: `untested`. Backend: HIP. Required patch: `1001_hip_internal_allreduce`.

## Scope

This package is the first atomic extraction from nasone commit `e06dcf6300718227cb8cfda9e61fb12ccb693418`. It owns only Q8_0 encoding/decoding primitives and the runtime threshold state. Residual fusion is NRO02; P2P transport is NRO03; existing BigCherry split-reduce telemetry remains the observability authority.

The initial implementation is deliberately **not validation-ready**. It adds compile-time kernel primitives and an opt-in threshold field but does not dispatch production reductions through Q8. This prevents a lossy path from becoming reachable before a reference fixture and numerical policy exist.

## Draft behavior

`GGML_CUDA_AR_Q8_THRESHOLD` is parsed into the internal AllReduce pipeline. Default `0` means disabled. The patch adds a block-Q8_0 quantizer and a Q8_0 two-rank dequantized add kernel with explicit padded-tail handling. A later implementation step will connect these to the copy-engine path and add activation/wire-byte counters.

## Control / subject

Future causal comparison:

- control: `1001_hip_internal_allreduce`, exact FP32 wire, same provider/config;
- subject: control + this patch with Q8 enabled at a pre-registered threshold.

BF16, RCCL, residual fusion and P2P are characterization/composition arms, not substitutes for the focal control.

## Real hardware evidence (2026-09-12, Brutus, gfx1100)

This draft had never been built on real hardware before this session. The
first real attempt found a genuine compile failure: two `Edit` anchors in
`patch.py` ended at the `=` sign of a single-line C++ statement --
`insert_after` splices immediately after the matched text, not after the
enclosing statement, so both insertions landed mid-expression, corrupting
`GGML_CUDA_AR_COPY_THRESHOLD_DEFAULT`'s declaration and `p->bf16_threshold`'s
assignment into unparseable C++ (real compiler errors: "expected
expression", "use of undeclared identifier"). Fixed by extending both
anchors to match the complete single-line statement (one needed this
project's own LITERAL-placeholder technique to cross a noise-stripped
string literal -- see `patch.py`'s inline comments for the exact fix).
Re-verified on real hardware: clean build, generated source inspected and
confirmed well-formed.

This closes the first item of this draft's own acceptance criteria
("applies cleanly on b10705, builds HIP") for the first time -- it was
never actually true before this session, despite the patch being packaged
and passing all 13 existing offline tests (which check structural
properties, not actual compilation -- exactly the class of bug real
hardware compilation catches that offline anchor-matching cannot).

## Limitations

No live dispatch is wired yet (unaffected by the build fix above -- the
Q8 path remains deliberately unreachable). No Experiment Contract or
`validation.toml` is intentionally created until numerical fixtures define
a defensible tolerance. Do not add this patch to production recipes.
Remaining acceptance criteria (numerical correctness matrix, activation/
wire-byte evidence, model quality gates, 4-arm performance sweep) are all
still genuinely not started.

See `TESTING.md` and `docs/planning/active/nasone-rdna-optimizations/NRO01.md`.
