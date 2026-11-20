# rad0043_expert_stats_protected

## Summary

`--expert-stats` counts the routed experts a layer keeps in bf16 beside its quantised table (its protected experts). Before, their histogram indices had no entry in the stats map and were tallied as `unmapped`, so a quantised layer's counts came up short by exactly its protected experts' selections.

## Outcome

Diagnostic on the promotion evidence below.

## Detail

### Promotion record

Tier: diagnostic

Mechanics: card-free patch-package tests (apply, anchors, idempotence) and patch-lint.
Activation: on Flash-Next (qwen3.8-next-flash-hcbf16, two 7900 XTX TP2, deterministic) the --expert-stats tally counts the protected bf16 experts too: the run counts 1,998,240 routed selections over 48 layers and 4,163 tokens (top-10), and two repeat runs give identical per-layer counts. See docs/evidence/2026-11-16-vtp44-generic-kv-layouts/FLASHNEXT-ROUTING.md.
