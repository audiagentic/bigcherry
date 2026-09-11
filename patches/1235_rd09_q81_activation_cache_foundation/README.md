# 1235 (RD09): per-graph Q8_1 activation cache -- foundation only

## Scope

**This patch is stage 1 of a 7-stage plan (RD09) and is deliberately, entirely
inert by construction.** It adds only:

- `src/ggml/src/ggml-cuda/hip-q81-cache.{h,cpp}`: generation-scoped
  find/reserve/publish cache API over stable, never-relocated slabs.
  `GGML_HIP_Q8_1_CACHE_MODE=off|on|verify` (default `off`),
  `GGML_HIP_Q8_1_CACHE_STATS=1` for counters.
- Three anchored edits: a CMake source-list addition, an opaque
  `void * q81_cache` member on `ggml_backend_cuda_context`, and destructor
  teardown wiring.

**No caller exists yet in `mmvq.cu`.** The cache foundation compiles and
links but is never invoked -- stage 2 (MMVQ integration) has not been
written.

## Why (of the full RD09 design, not just this stage)

`ggml_cuda_mul_mat_vec_q()` re-quantizes the F32 activation to `block_q8_1`
on every call, even when the same activation (same view-root, shape,
stride, stream, generation) was already quantized earlier in the same
graph. RD09 designs a bounded, generation-invalidated cache to skip the
redundant quantize launch on a hit.

## Upstream / provenance

Concept traced to fork commit `ff6fde5046ffb86672e05da640d2bfb20d4bfdfc`
("CUDA: cache quantized Q8_1 matmul inputs per graph"), rebased identity
`299f6eaf73b5eeb888bd94eaa66122d003136e6a` (BigCherry's stew675 snapshot is
patch-id-identical v2). **Deliberately NOT ported verbatim** -- design
review (`dev-gpt-agent`, session `ses_76b0fef0c94c434a`) found and rejected
two real weaknesses in the fork's own implementation before any code was
written:

1. The fork's cache key omits the exact view data address/offset (only
   view-root pointer + shape/stride/stream) -- two views of the same root
   with identical shape/strides but different offsets could collide.
   BigCherry's key adds the exact view byte-start field the fork lacks.
2. The fork's cache arena is relocatable (grows by allocate+copy+free),
   which conflicts with HIP/CUDA graph-capture lifetime (captured buffer
   addresses must stay stable once recorded). BigCherry uses stable
   retained slabs instead, with a hard cap and native-path fallback on
   exhaustion.

## Real hardware evidence (stage 1 only)

1. **GPT code review, round 1** (`req_9bd7db08f19d4e53`): found 2 real P1
   bugs before stage 2 could safely build on this foundation -- (a)
   `reserve()` grew immediately on a slab-boundary miss instead of walking
   existing later slabs first (could return an overlapping pointer), (b)
   the cache was a device-global singleton rather than context-scoped as
   designed (upstream doesn't guarantee one `ggml_backend_cuda_context` per
   device). Both fixed.
2. **GPT code review, round 2** (`req_c56ae6af774f4a2c`) on the fix commit:
   **PASS** -- "the two P1 findings are fixed, and the foundation is now
   suitable for stage 2... I would close RD09 stage 1 at the
   design/code-review level," conditioned on a real isolated HIP compile
   (no runtime benchmark required for stage 1 closure).
3. **Real isolated HIP compile** (Brutus, gfx1100,
   `bigcherry build --lane bigcherry-native:control:linux-multi --experiment rd09-only`):
   first attempt caught 2 real `[[nodiscard]]`-flagged ignored `hipError_t`
   returns from `hipFree` (destructor + reset_for_test), fixed across 2
   commits (a `replace_all` first missed the second call site due to
   differing indentation -- caught by CI, not silently accepted). Final
   rebuild: zero warnings, zero errors.
4. **30 source-contract tests** (`tools/tests/test_rd09_q81_cache_foundation.py`,
   hardware-free) cover the find/reserve/publish API, key strengthening,
   slab-walk-before-grow, and context-scoped ownership.

Stage 1 is closed per GPT's PASS verdict and the real compile confirmation
above. Nothing beyond this has been done.

## Known limitations -- most of RD09's real design is not yet built

- **Stage 2 (MMVQ integration)**: not started. Must wire BOTH (a) the cache
  lookup into `ggml_cuda_mul_mat_vec_q()` and (b)
  `ggml_hip_q81_cache_begin_generation()` at graph-evaluation entry in
  `ggml_backend_cuda_graph_compute()` -- these are not separable, since (a)
  without (b) risks a real stale-activation correctness bug.
- **Stage 3 (adversarial correctness)**: not started -- the full key-matrix
  test (same-root-different-offset must miss, capacity exhaustion falls
  back correctly, generation boundaries, dual-GPU cache isolation, etc.)
  has not been run.
- **Stage 4 (capture/topology)**: not started -- no graph-capture
  warm-up/replay/stable-address validation.
- **Stage 5 (causal bench)**: not started -- no measurement of actual
  quantize-launch reduction or TG/PP/memory effect.
- **No `validation.toml` adapter or Experiment Contract binding.**
- `state` correctly stays `"untested"` -- this is a sound, reviewed,
  compiling foundation with zero functional or performance evidence, since
  it is never called from anywhere yet.
