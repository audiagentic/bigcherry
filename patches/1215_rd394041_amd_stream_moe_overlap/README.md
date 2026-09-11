# 1215 (RD39/RD40/RD41/RD42): honor active HIP stream, per-(device,stream) cuBLAS handles, dedicated concurrent scratch, and MoE shared-expert overlap

## Scope

Fixes `ggml_cuda_op_mul_mat` to honor the assigned stream instead of
hardcoding stream 0, and gives every (device, stream) pair its own cuBLAS
handle (RD39/RD40); replaces in-place branch-node interleaving with a
dedicated, reused scratch buffer for concurrent graph regions (RD41); and
uses that infrastructure to detect the MoE shared-expert join and run the
shared expert on an auxiliary stream during small-batch decode (RD42).

## Why

The single per-device cuBLAS handle's workspace corrupted concurrent GEMMs
sharing it; the pre-existing branch-node interleaving for concurrency was
fragile against allocator reuse. Ported as one net patch because RD40/RD41
are declared prerequisites of RD39, and RD42 of all three, in the original
plan items -- splitting them would leave the graph optimizer in an
uncompilable intermediate state. The upstream fork measured +7.4% tg128 on
Qwen3.6-35B-A3B Q4_K_M (not yet independently reproduced here -- see
"Known limitations").

## Upstream / provenance

Ported from AMD-Ecosystem/llama.cpp PR #36 (merge commit `367c4d04f`,
https://github.com/AMD-Ecosystem/llama.cpp). Fork-only work, not ancestral
to mainline or this project's pin. Reconciled 2026-08-30 against upstream
PR #26574, which independently widened `cublas_handles` to `[device][stream]`
-- the now-redundant cuBLAS-handle-widening hunk was removed and the
destructor hunk rewritten to layer only the still-needed
`concurrent_scratch` free onto upstream's real current destructor body.

## Real hardware evidence (indirect, via downstream dependents' campaigns)

1215 has never been run through its own dedicated correctness/performance
campaign. However, it IS two required patches' hard prerequisite (RD43/1216
and RD44/1217 both `requires = ["1215_rd394041_amd_stream_moe_overlap"]`),
and both of those patches' real PPL-equality correctness producers resolve
1215 as part of BOTH their control and subject compositions (RD43's own
campaign function docstring: "control = 1215 alone"). This means 1215's
real-hardware evidence to date is:

- **Builds cleanly** on real gfx1100/HIP hardware as part of both RD43's
  and RD44's control and subject builds (multiple real campaign runs this
  session and in prior sessions).
- **Does not corrupt inference output**: RD43's control arm (1215 applied
  alone, no RD43/RD44) produced PPL = 561.6933 on gpt-oss-20B/Q6_K against
  the real wikitext2 corpus -- **identical** to RD13's own control-arm PPL
  on the same model/corpus, which does NOT include 1215 at all. Two
  independently-resolved compositions (one with 1215, one without) landing
  on the exact same PPL is incidental but real evidence that 1215's
  stream/handle/scheduling changes do not alter output correctness for
  this model, at least under ordinary (non-`GGML_CUDA_GRAPH_OPT=1`)
  execution.
- **No graph-capture abort under `GGML_CUDA_GRAPH_OPT=1`**: RD43's control
  arm (1215 alone, that env var set) completed without the
  "capturing stream has unjoined work" abort RD43 exists to prevent for
  the *combined* 1215+1216 case -- a real signal 1215's own concurrent-
  region infrastructure is not, by itself, unstable under this condition.

None of this is a substitute for 1215's own dedicated evidence -- it is
real, but incidental (produced as a side effect of validating its
dependents, not by design).

## Known limitations

- **The core performance claim (+7.4% tg128 on Qwen3.6-35B-A3B Q4_K_M) has
  never been independently reproduced on this project's hardware.** All
  real evidence above is correctness-only (builds, doesn't corrupt PPL,
  doesn't abort under graph-opt); no A/B timing run isolating 1215's own
  effect exists yet.
- No dedicated correctness producer exists for 1215 itself (e.g. a direct
  test of the MoE shared-expert auxiliary-stream overlap in isolation,
  without RD43/RD44 layered on top).
- No `validation.toml` adapter or Experiment Contract binding exists.
- `state` stays `"untested"` -- the evidence above supports "does not
  appear to be broken" but not "validated" (no independent performance
  reproduction, no dedicated correctness producer).
