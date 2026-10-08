# 1268 PRBE52 adaptive MTP wiring

Evaluated runtime wiring for 1255. Default remains fixed-depth: `n_min_adaptive=0` disables the controller.

## Composition

Requires `1255_nro06_adaptive_mtp_depth` and production primitive `1321_mtp_ahead_primitives`. It does **not** require 1210. `adaptive-mtp` retains 1210 as a comparison experiment; `adaptive-mtp-no1210` is the production set plus 1255+1268.

## Look-ahead

Adaptive MTP is compatible with 1322 look-ahead. The controller exposes its current per-sequence budget to the server; fresh fronts and ahead forced-front/tail work use that same budget. Live-tail drafting bypasses the fresh-front cap. Controller feedback derives from `verify_h_rows - 1`, the front actually verified, so promoted fronts do not reuse stale draft accounting. If the controller changes depth after a verify, 1322's existing full-tail-size check rejects promotion for that transition round and the next round drafts fresh.

## Safety and evidence

A speculative-context initialization exception is fatal for server startup when 1268 is composed; it no longer silently serves without a drafter. With `BIGCHERRY_PATCH_TRACE=1`, shutdown emits `event=depth_hist depth=<d> rounds=<n>` markers. Controller state resets in MTP `begin()` for every request.
