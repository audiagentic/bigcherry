# rad0045_expert_route_replay

## Summary

`--expert-route-replay FILE` makes a run take, for every token and routed layer, the experts a reference run recorded with `--expert-route-record` (rad0044) instead of the router's own choice. KLD against a reference cannot rank KV caches on a routed model because any change in the arithmetic moves about 1.1% of the expert choices (42% of token-layer expert sets differ between chunk 64 and 128); replaying the reference's route removes that noise, so only the cache differs. Manifest field `engine.tuning.expert_route_replay` (contract amendment 22).

## Outcome

Diagnostic on the promotion evidence below.

## Detail

### Promotion record

Tier: diagnostic

Mechanics: card-free patch-package tests (apply, anchors, idempotence) and patch-lint.
Activation: --expert-route-replay on Flash-Next: in every replayed run the per-layer expert counts equal the record's histogram in all 48 layers, and replaying the reference's own record gives KLD exactly 0; replay lowers the chunk-128 KLD from 0.0372 to 0.0076. One document pair, one run per arm; 11-document check in docs/evidence/2026-11-16-vtp44-generic-kv-layouts/FLASHNEXT-REPLAY-11DOCS.md (an occasional route divergence there is recorded, not hidden). See docs/evidence/2026-11-16-vtp44-generic-kv-layouts/FLASHNEXT-REPLAY.md.
