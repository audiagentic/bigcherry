# 1231 (HI14): real graph capture-lifecycle activation evidence

## Scope

Instruments the four real HIP graph API call sites in `ggml-cuda.cu`'s
`ggml_backend_cuda_graph_compute()` / `ggml_cuda_graph_evaluate_and_capture()`
(`cudaStreamBeginCapture`, `cudaStreamEndCapture`, `cudaGraphInstantiate`,
`cudaGraphLaunch`), each immediately after its `CUDA_CHECK()` succeeds, with
a once-per-process, opt-in `BIGCHERRY_GRAPH_LIFECYCLE stage=<name>` marker
gated behind `BIGCHERRY_GRAPH_LIFECYCLE_TRACE` (zero-cost by default, one
`getenv()` per call site). Adds `tools/bigcherry/graph_lifecycle_evidence.py`
to parse these markers from a captured log into the exact `capture_lifecycle`
dict shape the offline multi-GPU validator requires.

## Why

The offline validator (`multi_gpu_validate.py`) has always required explicit
`capture_begin`/`capture_end`/`instantiate`/`replay` evidence for an
"enabled graph" claim, but nothing in the runtime ever produced it -- a
plausible-looking run (server log shows graph reuse, dispatch records show
observations) is not the same as observing the actual HIP graph API
lifecycle fire. Uses `GGML_LOG_WARN` rather than `INFO` because `INFO` is
filtered out of `llama-server`'s default log verbosity below `-lv 4`
(confirmed on real hardware, HI90 -- see below).

## Upstream / provenance

Local design, part of this project's own correctness-evidence work (HI14),
following the same opt-in-marker convention established by HI85 (patch
1225).

## Real hardware evidence gathered so far

- **2026-08-22** (ledger `chg_20260822_053124`): real isolated-worktree
  materialization verified (clean apply, correct C++ output, all 4 markers
  present with proper structure); 9 new parser tests including a direct
  integration check against the real validator; full offline suite (1546
  tests) passed.
- **2026-08-23, HI90 fix** (ledger `chg_20260823_032603`): root-caused and
  fixed on real hardware during HI14's closure -- confirmed `llama-server`
  maps raw `ggml` `INFO` logs to a TRACE verbosity level filtered below
  `-lv 4`, so the original markers were silently invisible at any normal
  server verbosity even though graphs were genuinely capturing and
  replaying. Switched `GGML_LOG_INFO` to `GGML_LOG_WARN`, which is not
  filtered by that mechanism. Full offline suite green (1727 passed)
  afterward.

## Real live-server confirmation (2026-09-12, Brutus, dual gfx1100)

Closed the previously-open gap: built `llama-server` with this patch alone
(isolated `bigcherry-native` composition), launched it at **normal
verbosity** (no `-lv` flag), real Qwen3.8-27B-Q8_0 model, `-sm tensor`,
`BIGCHERRY_GRAPH_LIFECYCLE_TRACE=1`. Sent a real completion request. All
four expected markers appeared in the live server log exactly once, in the
correct order:

```
BIGCHERRY_GRAPH_LIFECYCLE stage=capture_begin
BIGCHERRY_GRAPH_LIFECYCLE stage=capture_end
BIGCHERRY_GRAPH_LIFECYCLE stage=instantiate
BIGCHERRY_GRAPH_LIFECYCLE stage=replay
```

This directly confirms the HI90 fix (INFO->WARN) works as intended on real
hardware at normal server verbosity -- the previously-untested assumption
is now hardware-confirmed, not just reasoned about.

## Lifecycle note

The HI90 fix session had explicitly attempted this re-verification on
Brutus but backed off after a dry-run apply showed the overlay pipeline
failing on an unrelated patch from another concurrent session's
in-progress work -- deliberately avoided risking a collision rather than
forcing the run. That gap is now closed by the confirmation above.

## Known limitations

- No `validation.toml` adapter exists; `kind = "diagnostic"`, not eligible
  for the local-framework adapter path.
- `state` stays `"untested"` -- this is a diagnostic/instrumentation
  patch, not a production dispatch patch, so "validated" in the
  performance-patch sense does not apply; the live-server confirmation
  above is this patch's own complete acceptance evidence.
