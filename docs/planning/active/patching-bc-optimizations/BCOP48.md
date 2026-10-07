---
id: BCOP48
order: 48
plan: patching-bc-optimizations
state: pending
created-at: '2026-10-07T15:04:00+11:00'
created-by: agent
priority: P2
---

# PRBE68 Vulkan submission batching disposition

## Audit result

PRBE68's proposed architecture-aware max-nodes table targets the wrong policy layer. Upstream already has a fixed 100-node safety ceiling plus an architecture/CU-sensitive FLOP submission budget from PR #25240. Issue #26679 shows gfx1201 performance is highly sensitive to FLOP-threshold submission semantics, but provides no evidence for a particular node ceiling.

## Authoritative owner

PRBE68 owns only qualification of the existing node/FLOP submission policy. Existing upstream submission predicates remain the implementation owner. PRBE62 separately owns queue-family selection.

## Unresolved action

Run a same-binary GGML_VK_MAX_NODES_PER_SUBMIT sweep on gfx1201/gfx1100 while recording which existing predicate closes each submission. Add only disposable closing-reason counters if current diagnostics are insufficient.

## Terminal disposition

- <1% E2E/firing-distribution sensitivity: close PRBE68.
- >=3% CI95-low-positive winner with <=1% control regression: prefer the existing env override.
- Code-default change is allowed only after first-party evidence proves the node bound itself causal for a stable architecture+driver class.
- FLOP-bound causality transfers to the existing Vulkan batching seam; do not create another scheduler/table.

## Dependencies / blockers

No hardware run, build, or prototype was performed by this audit. Active 1340/MSM02, QFP31-41, 1341/MSM03, QFP30/1281, MET08/10/11, FKE01 and RNX02 work was not modified.
