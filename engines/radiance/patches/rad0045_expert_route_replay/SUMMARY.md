# rad0045_expert_route_replay

**Status:** diagnostic
**Plan item:** VTP44

## Summary

`--expert-route-replay FILE` makes a run take, for every token and routed layer, the experts a reference run recorded with `--expert-route-record` (rad0044) instead of the router's own choice. KLD against a reference cannot rank KV caches on a routed model because any change in the arithmetic moves about 1.1% of the expert choices (42% of token-layer expert sets differ between chunk 64 and 128); replaying the reference's route removes that noise, so only the cache differs. Manifest field `engine.tuning.expert_route_replay` (contract amendment 22).

## Outcome

Qualified as a diagnostic tool (2026-11-20, record in `evidence/promotion.json`). Run on Flash-Next (hcbf16, TP2, the 4,163-token run): in every replayed run the per-layer expert counts equal the record's histogram in all 48 layers, replaying the reference's own record gives KLD exactly 0, and replay takes the chunk-128 KLD from 0.0372 to 0.0076, removing the routing floor so that bf16_i8 separates from fp8 (docs/evidence/2026-11-16-vtp44-generic-kv-layouts/FLASHNEXT-REPLAY.md; the 11-document check, which records an occasional route divergence, is docs/evidence/2026-11-16-vtp44-generic-kv-layouts/FLASHNEXT-REPLAY-11DOCS.md). One document pair, one run per arm.

## Detail

- libr4d (the sources the libr3 plugin is built from): a new op `router_topk_replay` (logits, forced ids -> expert ids, expert weights) is router_topk's row function with the selection loop skipped: lane r takes the recorded expert r of the row, and the weights come from this run's own logits over that set by the same expression. It is a separate op so that a plugin without it never resolves it (the block checks `rad_op_declared` and `rad_op_resolved`), and a replay with no such kernel fails the step instead of selecting on its own.
- core: `rad_route_replay_load` (abi, wrapper, test fake) copies a pass's recorded rows into the block's persistent step-tall buffer before the router. The engine reads and checks the record at start (complete, same model, layer and expert counts, routed layers, document starts, ids in range, exact size); every rank's context keeps it as a device table.
- Review (GPT, 8 findings) led to: per-pass coverage (every routed layer must load exactly the pass's rows, so an architecture without the hook fails instead of silently selecting its own experts), the pass-admission error failing the step, a table read per engine and file under a lock, a strictly parsed bounded header (1 GiB, size checked before allocating), a shape bound that matches the kernel (512 experts), and per-document token digests (rad0044 now records them) so a replay of different prompts of the same lengths is refused.
- A pass is looked up through the document starts: positions must continue, or restart at 0 for the next recorded document, so the chunk size cannot matter. Only one-sequence prefill passes without drafting are replayed; more documents than recorded, a longer document, a gap, another model's routed layers, a different top_k, or a run that does not consume the whole record fail the run.
- Recorded passes (the step tape) are off while replaying. The fused router ops are skipped on replayed passes (the separate router, the replay op and the scatter run).
- Refused with `--expert-route-record` and with `--max-num-seqs` other than 1.
- Replays only the ids: the weights are the measured run's own. If KLD is not near 0 on a self-replay the recorded weights would have to be replayed too.
