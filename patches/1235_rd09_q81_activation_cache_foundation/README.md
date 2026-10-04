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

<!-- QFP18 promotion record (generated by tools/lab/flash-next/promote-readme.py) -->

Promotion record (QFP18 lightweight evidence-reuse tier, pin 0504396). Mechanism, design and the incremental
measurements are in SUMMARY.md; this file records why the patch is promoted.

## Evidence

- Entered production profile **v3**, measured against **v2** (each patch is measured as promoted base vs
  promoted base + patch; the measurement is valid for that base).
- Incremental result: foundation required by 1307-1313 (Q8_1 cache reservation + GGML_HIP_Q8_1_CACHE_MODE gate); measured with them in the v3 profile ABBA. Greedy output identical between arms in every adoption run.
- Activation: GGML_HIP_Q8_1_CACHE_MODE=on; BIGCHERRY_Q81_TRACE hits.
- Mechanics: offline patch tests + patch-lint pass on pin 0504396; whole stack re-confirmed on 0504396 by the native
  comparison below and the v6 adoption screens.

## Native llama.cpp comparison (profile level, pin 0504396)

`tools/lab/flash-next/queue-promote.sh` (2026-10-05): native llama.cpp (source `llama-native`, no patches, default
all-reduce) vs production profile v6 (`BIGCHERRY_FEATURES=flashnext-v6`), Qwen3.8 Flash-Next UD-IQ4_XS, 2x 7900 XTX +
R9700, MTP draft on the 6900 XT, 64K context, f16 KV, ub512, 256 greedy tokens, ABA (v6 / native / v6):

| Depth | v6 decode (t/s) | native decode (t/s) | v6 ms/step | native ms/step | v6 vs native |
|---|---|---|---|---|---|
| ~8K  | 81.9 / 84.1 | 64.3 | 37.7 / 36.7 | 49.2 | +29% t/s |
| ~48K | 69.1 / 69.5 | 47.6 | 44.1 / 43.9 | 66.4 | +45% t/s |

Draft acceptance is comparable (172-175 accepted of 239-249 drafted), so the arms did the same work. v6 arms are
greedy-identical to each other; native differs in text (Q8_1 activation paths and deterministic top-k ties change
low-order bits). Native cannot load the 240K f16 deployment at all (its largest load was 192K with q8_0 KV), which
1302/1303 enable. Runs: `/mnt/data/bigcherry-work/runs/flashnext-native-d8k`, `flashnext-native-d48k`.

## Cross-model no-regression (one release build)

The Flash-Next set ships in every release build (validated-enhancements), with every runtime flag at its default.
Qwen3.8-27B Q8_0 dual-XTX production config (-sm tensor, built-in MTP4, default all-reduce), ABBA per depth,
promoted base (A) vs base + Flash-Next set (B), `tools/lab/flash-next/queue-p27b-3.sh` (2026-10-05):

| Depth | A decode (t/s) | B decode (t/s) | A prefill | B prefill | acceptance A / B |
|---|---|---|---|---|---|
| 10K | 70.9 / 72.9 | 72.8 / 73.5 | 1292.8 / 1292.1 | 1292.2 / 1293.4 | 177/310 / 177/310 |
| 32K | 72.2 / 72.3 | 72.2 / 72.1 | 1251.7 / 1253.7 | 1253.2 / 1253.7 | 181/293 / 181/293 |

Greedy text identical between A and B at both depths. 1303 fails closed when BIGCHERRY_ATTN_TS is set for a
non-qwen4exp model (observed on the first 27B attempt), so the flag cannot silently misplace another model's KV.
