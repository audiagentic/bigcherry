# rad0010_routing_evidence

## Summary

Opt-in counters at the engine's existing per-layer routing-histogram producer and reader that say how many histograms were issued, completed, credited, overwritten or left pending, per rank and phase (prefill, decode, mixed, draft). Built only into the separate `radiance-routing-evidence` source; off unless `RAD_ROUTING_EVIDENCE=1`.

## Outcome

Diagnostic: opt-in evidence only; not production-qualified.

## Detail

### Promotion record

Tier: diagnostic

Mechanics: Radiance patch-package fixtures, rad0010 patch and independent synthetic oracle tests, and repository patch-lint, card-free; these are mechanics only.

Activation: RHP01 2026-11-04 v1.16.0 run with RAD_ROUTING_EVIDENCE=1: rank-0 teardown RHP01 phase counters included prefill issued=3696/credited=3696, decode issued=26736/credited=26640/pending=96; overwritten=0. Sources: docs/planning/active/patching-radiance-heat-placement/RHP01.md ## Notes, package SUMMARY.md ## Outcome. Recorded on v1.16.0; NOT a v1.17 hardware reproduction.

### Scope and provenance

Observed on v1.16.0 at chunk 2048 and rank 0 only; v1.17 pin, larger chunks, MTP and broader hardware scenarios remain unqualified. These counters prove the opt-in diagnostic ran in the cited test, not performance or production-readiness.
