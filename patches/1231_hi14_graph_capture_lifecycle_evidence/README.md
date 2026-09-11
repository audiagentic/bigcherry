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

## Lifecycle note -- deliberately still `untested`

The HI90 fix session explicitly attempted real-hardware re-verification on
Brutus (to confirm the `WARN`-level markers are actually visible in a real
server log at normal verbosity) but **backed off** after a dry-run apply
showed the overlay pipeline failing on an unrelated patch from another
concurrent session's in-progress work -- deliberately avoided risking a
collision rather than forcing the run. That re-verification has not been
completed since. `STATE` stays `"untested"` because the fix itself
(WARN-level visibility) has real hardware-grounded reasoning but no direct
"markers observed in a live server log" confirmation yet -- this README
documents that gap honestly rather than closing it prematurely.

## Known limitations

- No `validation.toml` adapter exists; `kind = "diagnostic"`, not eligible
  for the local-framework adapter path.
- Real confirmation that the WARN-level markers appear in a live
  `llama-server` log at normal verbosity is the concrete remaining step
  before this patch can be promoted to `validated`.
