---
id: BCOP36
order: 36
plan: patching-bc-optimizations
state: pending
created-at: '2026-10-05T21:50:00+00:00'
breadth: ''
skill: advanced
created-by: agent
priority: P2
work: M
---

# Optimize speculative decoding as a whole-system pipeline

## Description

Treat MTP/draft speculation as an end-to-end pipeline rather than optimizing draft depth or kernels independently. Model proposal generation, target verification, acceptance, KV/cache effects, device placement, synchronization and memory pressure together. RPL01 owns cross-device/system placement; this item owns speculation-specific measurements and candidate controls.

## Steps

1. Standardize speculation telemetry: proposed tokens, accepted tokens, acceptance distribution by depth, draft/verify ms, rollback/recompute work, KV bytes, synchronization and effective accepted tokens/s.
2. Sweep MTP/draft depth and batch/ubatch at short and long context. Measure marginal cost and marginal accepted work per additional draft token.
3. Compare same-device draft+target against candidate split-device execution where supported. Include transfer/sync costs; do not assume an auxiliary GPU helps.
4. Evaluate interaction with Flash Attention, sparse FA, quantized KV, expert residency/cache and scheduler copies.
5. Build an offline selector that predicts the best qualified speculation configuration from measured workload/context features. It may recommend existing controls but must not silently mutate runtime state.
6. Feed candidate configurations and measured costs to RPL01 so speculation is compared with competing VRAM/device uses.
7. Runtime adaptation is a later gate requiring stable workload-shift detection, bounded transition cost and multi-request correctness.

## Validation

- MTP off control plus depths 1..supported maximum.
- TG128/512, short/long context, code/prose/retrieval workloads.
- Cold/warm and multi-request same-process integrity.
- Report raw TG and effective accepted TG; promotion uses effective end-to-end throughput.

## Acceptance Criteria

- Speculation decisions optimize accepted end-to-end tokens/s rather than draft throughput alone.
- Memory/placement cost is visible to RPL01.
- Any adaptive selector predicts held-out winners before runtime control is enabled.
- Promoted configurations improve effective TG >=5% with <=2% regression in unaffected workloads and pass multi-request correctness.
