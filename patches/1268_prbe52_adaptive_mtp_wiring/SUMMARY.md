# 1268_prbe52_adaptive_mtp_wiring

**Status:** evaluated  
**Plan item:** PRBE52

## Current design

1268 wires 1255 behind `--spec-draft-n-min-adaptive` / `LLAMA_ARG_SPEC_DRAFT_N_MIN_ADAPTIVE`; zero preserves fixed depth. The manifest requires 1255 + 1321, not 1210.

The server resolves an effective per-sequence speculative budget. 1322 therefore uses the same current depth for its forced front and ahead tail. The adaptive limiter is skipped only during 1321 live-tail continuation. Acceptance feedback uses the actual target verify row count rather than mutable draft-call bookkeeping, covering fresh and promoted fronts.

## Determinism

Controller reset is per request. The controller has no time source, allocation-layout input, or process-global mutable policy state. Source audit found a separate upstream MTP request-state leak: prompt checkpoints restore target/draft KV but MTP did not serialize `pending_h`, so a warm request could leave a stale hidden boundary for the next checkpoint replay. 1268 now serializes/restores that row through `get_state/set_state`. If repeats still diverge after this fix, depth-dependent backend numerics remain the next hypothesis.

## Failure mode and trace

Speculative initialization failure is fatal to server startup in this composition. `BIGCHERRY_PATCH_TRACE=1` reports activation, depth changes, and shutdown depth distribution.

## 1210

1210 is retained only in `adaptive-mtp` for direct comparison. `adaptive-mtp-no1210` tests the production kernel paths with 1255+1268.
