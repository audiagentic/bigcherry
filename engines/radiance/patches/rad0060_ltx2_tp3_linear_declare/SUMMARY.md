# rad0060 — LTX2 TP3 diagnostic IQ8R declarations

**Status:** diagnostic
**Plan item:** LT3

No production source recipe selects this patch.

## Summary

Optional `C01Slice` in the real 2.2 / ABI44 LTX `Lin::declare`.
Fused QKV/KV row shards record per-bank code, two-level IQ8R scales
and BF16 bias spans. Output columns use native aligned `RAD_SHARD_COL`
views. Biasless columns declare no bias; biased columns require
`C01Slice::columns_deferred`, record `Lin::deferred_bias`, and declare
**no partial-rank bias weight or table-add**. C04 reuses Radiance 2.2
exact all_reduce, then adds BF16 bias once per reduced output replica.
Actual GPU execution and full architecture integration remain gates.

The existing TP1 calls and LTX `world_size > 1` rejection are
unchanged. No activated GPU TP3 inference or numerical parity is claimed.

## Outcome

**Host-only validation:** 22 idempotent source edits compile on Radiance 2.2 / ABI44; the actual LTX declaration tests pass with explicitly deferred output biases, not GPU execution.

- 22 anchored edits to the pinned Radiance 2.2 LTX source;
  applicability and idempotence pass.
- Patched full LTX translation unit compiles with the native ABI44 SDK.
- Source-backed C01/C02 real-model declarations: 3,024 fused/biasless
  and 2,640 deferred-biased projections pass; 384 rank K cuts refuse.
- Original C02 negatives plus C04 duplicate-finalizer, non-TP3 world
  and invalid-K refusal.
- C04 verifies 2,016 source-backed native collective + bias graphs.
- C05 verifies 1,728 source-backed, 3,456 Q/K all-head native
  RMS graph declarations using existing fill/cast/all_reduce/rmsnorm.
  Q and K reuse caller-owned scratch; this is a host-only, full-width
  network diagnostic, not a qualified GPU performance path.
- C06 declares 576 source-backed video/audio self-attention graphs
  through the same native Attn/attn_core, with activation quantizers,
  exact Q/K and output reductions, RoPE head0 and gate.
  No production caller or GPU execution is enabled.
- Evidence: `docs/evidence/2026-11-22-ltx25-c04-native-reduce/README.md`.

## Detail

Actual model payload loading, end-to-end per-rank DiT architecture
and activation buffers, real GPU global Q/K RMS reduction, attention/FFN
actual three-rank collective execution, native multi-rank
serving and resource admission, device numerical quality/liveness,
and a recorded video-generation performance result. Planning/Ledger,
CI and review remain independent project gates.
