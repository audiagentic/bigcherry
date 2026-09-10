---
id: NRO05
order: 5
plan: nasone-rdna-optimizations
state: pending
created-at: '2026-09-08T09:50:40+10:00'
breadth: ''
skill: advanced
created-by: agent
priority: P0
work: L
---

# Chunk GDN MTP prefill prefix while preserving snapshot tail

## Description

Extract the MTP-specific prefix/tail composition from nasone commit `4169fbbf50d24beb6d269a2350e7f780b85369e6` into a separate experiment on top of NRO04. When GDN runs with `K>1` snapshot semantics, a fully chunked path cannot simply replace the recurrence because the final K token states must populate the exact snapshot slots used by speculative decode. The source solution chunks only the long prefix `n_tokens-K`, retains the resulting state, and runs the existing sequential recurrence for the final K tokens.

This item tests that orchestration independently of the gfx1100 WMMA kernel's raw speed. Control and subject use the same chunked prefix implementation; the treatment is the prefix+tail composition versus the existing fully sequential K>1 path.

## Steps

1. Require `1253_nro04_gfx1100_bf16_chunked_gdn` (and transitively RD50).
2. Enable only non-KDA, `K>1`, `n_seqs==1`, sufficiently long sequences, and supported head dimensions. Keep short or multi-sequence batches fully sequential.
3. Compute `n_prefix = n_tokens-K`; allocate temporary prefix state with existing pool ownership.
4. Run chunked GDN only over `n_prefix`. If launch rejects/fails synchronously, abandon the optimization and execute the original full sequential path.
5. Feed prefix state into the existing sequential kernel for exactly the final K tokens, preserving state-slot stride and snapshot order.
6. Add activation diagnostics: prefix length, K, selected chunked representation, fallback reason.
7. Construct reference tests that compare every snapshot slot, final recurrent state, output tokens, and continuation logits against fully sequential execution.
8. Sweep K=2/3/5/8 (where supported), prefix lengths around the activation threshold, and context/ubatch shapes.
9. Test graph capture/replay and repeated decode transitions; temporary state lifetime must not outlive its graph execution.
10. Measure whether saved prefix time exceeds allocation/transition overhead.

## Detailed Solution & Technical Design

Snapshot correctness is the invariant. MTP does not merely need the final GDN state; it needs the sequence of states corresponding to speculative positions. A prefix-only chunked transform can safely accelerate history before those snapshots, because all snapshot-producing steps remain in the proven sequential recurrence and see the same prefix state within the accepted numerical policy.

The experiment must separate two error sources: NRO04's BF16 chunked prefix error and NRO05's orchestration/snapshot mapping. Include an FP32 chunked-prefix arm where feasible so a snapshot-index bug cannot be hidden behind BF16 tolerance.

No attempt should be made to chunk across the final K snapshot positions until a new algorithm explicitly produces all intermediate states. That would be a distinct experiment.

## Code Samples & Guidance

Core control flow:

```text
if K>1 && n_seqs==1 && n_tokens>K+margin:
    prefix = n_tokens-K
    prefix_state = chunked(prefix)
    if prefix succeeded:
        sequential(last K, state_in=prefix_state, snapshots=K)
        return
fallback: existing full sequential path
```

## Files

- `docs/planning/active/nasone-rdna-optimizations/NRO05.md`
- `patches/1254_nro05_gdn_mtp_prefix_tail/{patch.toml,patch.py,SUMMARY.md,README.md,TESTING.md}`
- shared NRO package tests; future snapshot-integrity fixtures.

## Validation

Require per-slot state integrity, final-state integrity, output/logit parity within the parent kernel's numerical policy, and repeated MTP draft/verify cycles. Cover threshold-1/threshold/threshold+1, K extremes, single- versus multi-sequence, synchronous chunk-launch rejection, and pool reuse.

Performance evidence must include MTP acceptance statistics so a faster step that changes acceptance is not misreported as pure execution gain.

## Effort & Risk

High. State-slot off-by-one or stride errors can corrupt speculative verification while leaving ordinary generation apparently healthy. Temporary-state ownership and graph capture add lifetime risk.

## Standards

State restoration/snapshot correctness is affirmative evidence, not absence of crashes. Preserve fully sequential fallback and pre-register all numerical tolerances inherited from NRO04.

## Acceptance Criteria

- Every snapshot slot agrees with the reference within the registered tolerance.
- Final state and continuation behavior agree.
- All nonqualifying shapes take the original path.
- MTP acceptance does not regress beyond the registered quality budget.
- End-to-end MTP/prefill effect is positive and statistically supported.

## Notes

Same nasone source commit as NRO04; NRO04 owns external source identity and this item is an atomic BigCherry child.

## Change Log

- 2026-09-08T09:50:40+10:00 (created-by): Created from nasone GDN MTP-prefix design; P0.

## Ledger-events

- Pending: ag-ledger MCP unavailable in authoring session.
