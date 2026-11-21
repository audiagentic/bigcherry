# rad0060 — LTX2 TP3 diagnostic IQ8R declarations

**Status:** diagnostic, not in any production source recipe
**Plan:** LT3 engineering backlog (not a Planning MCP identifier)

## Summary

Optional `C01Slice` in the real 2.2 / ABI44 LTX `Lin::declare`.
Fused QKV/KV row shards record per-bank code, two-level IQ8R scales
and BF16 bias spans. Output columns use native aligned `RAD_SHARD_COL`
views. Biasless columns declare no bias; biased columns require
`C01Slice::columns_deferred`, record `Lin::deferred_bias`, and declare
**no partial-rank bias weight or table-add**. An output collective and
unique post-reduction bias owner remain separate, unimplemented gates.

The existing TP1 calls and LTX `world_size > 1` rejection are
unchanged. No activated GPU TP3 inference or numerical parity is claimed.

## Outcome

**Host-only validation:** seven idempotent source edits compile on Radiance 2.2 / ABI44; the actual LTX declaration tests pass with explicitly deferred output biases, not GPU execution.

- Seven anchored edits to the pinned Radiance 2.2 LTX source;
  applicability and idempotence pass.
- Patched full LTX translation unit compiles with the native ABI44 SDK.
- Source-backed C01/C02 real-model declarations: 3,024 fused/biasless
  and 2,640 deferred-biased projections pass; 384 rank K cuts refuse.
- Four explicit negative cases; native CTests pass.
- Evidence: `docs/evidence/2026-11-22-ltx25-c03-deferred-bias/README.md`.

## Detail

Actual model payload loading, end-to-end per-rank DiT architecture
and activation buffers, global Q/K RMS reduction, attention/FFN
collectives, once-only output bias application, native multi-rank
serving and resource admission, device numerical quality/liveness,
and a recorded video-generation performance result. Planning/Ledger,
CI and review remain independent project gates.
