---
id: PGC06
order: 0
plan: patching-gpu-collectives
state: pending
created-at: '2026-09-29T09:34:05.557756+00:00'
breadth: ''
skill: intermediate
created-by: agent
priority: P2
work: S
---

# --allreduce-fuse residual arg (1250 NRO02 fused residual)

## Description
Expose 1250 fused residual as `--allreduce-fuse {none,residual}`; fail closed on unsupported providers. Depends on PGC04.

## Steps
1. Keep this CLI work separate from dense-decode candidate qualification.
2. Before using MTP server throughput as evidence, require full-vocabulary correctness where declared and drafted/accepted work equivalence.
3. Isolate output-changing candidates before composition: `combo-ab1` (1241+1206+1245) is VOID for throughput attribution because greedy output changed and acceptance was 0.95580 vs 0.90101.
4. For 1241 specifically, run a fixed-work `llama-bench` arm plus correctness/logprob-divergence check before any performance claim.

## Validation
Marker `patch=1250_nro02 path=allreduce_fused_residual` per arm; token/logprob parity as required; acceptance parity; fixed-work benchmark; then order-balanced server A/B using `tools/lab/native-vs-patched/server-ab-*.json`.

## Acceptance Criteria
No throughput claim from non-equivalent speculative work; no patch is described as validated/signed off solely from these plans.

## Change Log
- 2026-09-29: added work-equivalence and combo-ab1 void handling.
- 2026-09-29T09:34:05.557756+00:00 (created-by): Created by agent

## Ledger-events

- chg_20260929_135722_allreduce-methods-are-now-sele_7144
- 2026-09-29T13:57:32.092038+00:00 (updated-by): Updated: section:ledger-events
