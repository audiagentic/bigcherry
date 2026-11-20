# rad0044_expert_route_record

## Summary

`--expert-route-record FILE` writes the experts each token's router chose, for every routed trunk layer, when the run ends. A KLD run on a routed model cannot rank KV caches because any change in the arithmetic moves about 1.1% of the choices (FLASHNEXT-ROUTING.md); to score a cache alone the measured run has to take a reference run's choices. This is the record half; replay is a later patch. Manifest field `engine.tuning.expert_route_record` (contract amendment 21).

## Outcome

Diagnostic on the promotion evidence below.

## Detail

### Promotion record

Tier: diagnostic

Mechanics: card-free patch-package tests (apply, anchors, idempotence) and patch-lint.
Activation: --expert-route-record writes the experts each token chose on Flash-Next (4,163 tokens x 48 layers x top-10); the record's per-layer histogram equals the --expert-stats counts of the same run in all 48 layers. See docs/evidence/2026-11-16-vtp44-generic-kv-layouts/FLASHNEXT-REPLAY.md and docs/evidence/2026-11-16-vtp44-generic-kv-layouts/flashnext-replay/.
