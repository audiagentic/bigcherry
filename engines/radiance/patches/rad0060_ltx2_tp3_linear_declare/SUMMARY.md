# rad0060 — LTX2 TP3 native C01 declaration staging

**Diagnostic only.** Adds an opt-in C01Slice to the real Radiance 2.2
LTX2 `Lin::declare` path. For IQ8R, row-sharded QKV/KV selects the same
per-bank source rows for codes, scales and BF16 bias through the existing
LinearFP8/span_parts and RadBuilder APIs. Column-sharded output projections
require an aligned 256-K slice and currently **refuse any output bias**:
once-after-reduce ownership is not yet implemented.

Existing TP1 calls pass the default null slice and stay unchanged.
The LTX `world_size > 1` refusal is preserved; no runtime TP3 graph,
Q/K all-head RMS or mixed-card liveness is claimed.

**Host validation:** patch applies and is idempotent against the observed
frozen candidate's actual `arch/ltx2/ltx2.cpp`; C++23 compilation of
the patched translation unit and independent C01Slice tests pass.
This is compile/applicability evidence only, not a run of an actual
RadBuilder with model weights, a GPU projection or full model inference.

**Required before activation:** canonical RUM source/ABI44 composition,
an actual builder weight-view receipt test, controlled native one-rank
projection parity, rank-local buffers, global RMS/all-reduce, bias-once
and request/owner graph acceptance. No promotion or merge without the
project's Planning/Ledger and standard CI gates.
