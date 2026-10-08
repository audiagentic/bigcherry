# 1268_prbe52_adaptive_mtp_wiring

**Status:** rejected
**Plan item:** PRBE52

## Current design

1268 wires 1255 behind `--spec-draft-n-min-adaptive` / `LLAMA_ARG_SPEC_DRAFT_N_MIN_ADAPTIVE`; zero preserves fixed depth. The manifest requires 1255 + 1321, not 1210.

The server resolves an effective per-sequence speculative budget. 1322 therefore uses the same current depth for its forced front and ahead tail. The adaptive limiter is skipped only during 1321 live-tail continuation. Acceptance feedback uses the actual target verify row count rather than mutable draft-call bookkeeping, covering fresh and promoted fronts.

## Determinism

Controller reset is per request. The controller has no time source, allocation-layout input, or process-global mutable policy state. Source audit found a separate upstream MTP request-state leak: prompt checkpoints restore target/draft KV but MTP did not serialize `pending_h`, so a warm request could leave a stale hidden boundary for the next checkpoint replay. 1268 now serializes/restores that row through `get_state/set_state`. If repeats still diverge after this fix, depth-dependent backend numerics remain the next hypothesis.

## Failure mode and trace

Speculative initialization failure is fatal to server startup in this composition. `BIGCHERRY_PATCH_TRACE=1` reports activation, depth changes, and shutdown depth distribution.

## 1210

The `adaptive-mtp` and `adaptive-mtp-no1210` experiment sets were removed with the rejection (2026-10-09).

## Result at b11474: rejected (2026-10-09)

Measured on Brutus, pin b11474 (b9acf138), one binary (production + 1255 + 1268, build `b-metamem-ad5`), A = adaptive
off (fixed depth), B = the setting shown, ABBA per depth. The activation marker
(`BIGCHERRY_PATCH_HIT patch=1268_prbe52_adaptive_mtp_wiring`, including `event=depth_change` lines) is present in the
adaptive arms and absent from the fixed arms. With adaptive off the build equals production (same greedy text, decode
86-87 / 72-74 / 70-71 t/s at 8K / 24K / 98K), so the set itself costs nothing when off.

Qwen3.8-Flash-Next, production profile (fixed depth 3, look-ahead on), decode t/s, B against A:

| setting | 8K | 24K | 98K |
| --- | --- | --- | --- |
| cap 3, floor 2 (two ABBAs + one with markers) | 76.3-80.9 vs 86.0-87.1 | 73.0-75.6 vs 71.9-74.1 | 62.3-67.5 vs 70.0-71.4 |
| cap 3, floor 3 | identical text and speed (the controller cannot move) | same | same |
| fixed depth 5, no adaptation | 79.4-82.3 vs 85.9-86.5 | 65.4-65.8 vs 72.0-72.5 | 59.6-60.2 vs 69.6-70.5 |
| cap 5, floor 3 | 76.8-78.0 vs 85.6-87.0 | 67.2-67.6 vs 71.5 | 63.5-64.4 vs 70.1-71.0 |
| cap 5, floor 2 | 66.2-76.3 vs 86.0-86.3 | both adaptive runs produced no decode result | 58.9-61.3 vs 69.8-70.6 |

Qwen3.8-27B Q8_0, dual RX 7900 XTX production flags (fixed depth 4, no look-ahead), cap 4 floor 2, two ABBAs:
64.8-65.4 vs 71.4-74.6 t/s at 10K and 66.4-66.8 vs 72.9-73.3 at 32K.

Every adaptive configuration is slower than the fixed depth, with complete separation at 8K and 98K on Flash-Next and
at both depths on the 27B. Drafting deeper than the fixed setting loses (acceptance per drafted token falls), and
drafting shallower loses. The greedy text of the adaptive arms also differs between repeats on Flash-Next. The
24K result for cap 3 floor 2 overlaps the baseline and is the only non-loss.
