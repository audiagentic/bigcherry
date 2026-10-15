# 1297_draft_vocab_trim

**Status:** validated
**Plan item:** QFN01

## What it does

Opt-in (`BIGCHERRY_DRAFT_VOCAB_N=N`): the Qwen4Exp MTP draft computes logits over output rows [0, N) plus all
control / user-defined / end-of-generation tokens, scattered into -inf full-vocabulary logits.

## Why

The MTP draft's own output.weight (Q8_0, 2560 x 248320, ~675 MB) is read for every draft token: ~31% of the
6900's draft kernel time per MTP step at 80K. The target verifies every draft token, so output cannot change;
only acceptance can drop.

Validation: decode ms/step and acceptance A/B for N = 16384 / 32768 / 65536 vs unset at 10K and 80K; greedy
parity is guaranteed by verification but checked without MTP-irrelevant changes (draft only).

## Result (2026-10-03)

Quick screens (~30K cached, ABA): N=32768 -10% ms/step but acceptance fell (163/273 vs 170-176), ~neutral
throughput; N=65536 -6% ms/step with acceptance in the base range, +8% t/s. Full ABBA (flashnext-trim-ab-4,
MTP3, f16 draft KV, 192K deployment config): 10K 50.0 / 51.0 -> 46.1 / 46.6 ms/step (-8%); 80K 58.6 / 57.6 ->
54.0 / 53.9 (-7%); complete separation, acceptance comparable. Deployment: BIGCHERRY_DRAFT_VOCAB_N=65536.
Run 2 crashed on MTP prompt-replay batches with no output rows (fixed: those keep the plain head).

## 2026-10-14 QFP05 follow-up: runtime trim is not a GGUF/VRAM trim

The validated 1297 runtime route retains the original full `model.output` buffer, copies its complete contents through host memory at context construction, then allocates/uploads an additional compact head. At 248,320×2,560 Q8_0, payload accounting is 644.14 MiB original + at least 170 MiB for a nominal 65,536-row head (plus special rows and alignment); transient host `full + rows` payload is similarly at least 814.14 MiB. These are source-derived byte calculations, not measured VRAM or startup latency. The default-unset path and 1297's validated performance remain unchanged.

QFP05 now gates a separate offline GGUF frequency-ranked `output.weight` + I64 `d2t` sidecar control before any dynamic vocabulary work. Existing EAGLE3 `d2t` and external Qwen4Exp fork `drluoto/llama.cpp@fb367b8` provide implementation references, not a merged upstream replacement. Require strict map/required-token checks, no simultaneous `BIGCHERRY_DRAFT_VOCAB_N`, exact MTP graph scatter, measured residency and paired gfx1030 draft qualification. **No patch implementation or promotion was performed by this documentation audit.**
