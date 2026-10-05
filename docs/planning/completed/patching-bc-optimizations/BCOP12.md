---
id: BCOP12
order: 12
plan: patching-bc-optimizations
state: superseded
created-at: '2026-10-05T04:41:00+00:00'
breadth: ''
skill: advanced
created-by: agent
priority: P1
work: L
---

# Gate direct sparse QSA attention after upstream indexer tiling

## Description

Follow-through for QFP04/1295/1296. The preferred end-state is one canonical selected-index representation feeding backend-native sparse Flash Attention directly, avoiding repeated gather/materialization. This work proceeds only after upstream tiled-indexer qualification and measured proof that selected-cell gather/attention remains material.

## Steps

1. Qualify llama.cpp #29901 on gfx1100/gfx1201 and account for its PP/indexer gain before local sparse-attention attribution.
2. Profile selected-cell gather + attention at 80K/160K/220K after #29901/#29825-equivalent baseline.
3. Proceed to direct selected-index FA only if this path remains >=5% decode GPU time or >=0.5 ms/token at >=160K.
4. Implement one `bc_qsa_idx` representation; direct FA stages selected K/V tiles and dequantizes on load where quantized KV is used.
5. Compare direct FA against typed gather with f16/q8 KV, selection-count extremes, tails/sentinels and reference logits.
6. Keep typed gather only if it wins or is required as correctness fallback.

## Detailed Solution & Technical Design



## Code Samples & Guidance



## Files



## Validation



## Effort & Risk



## Standards



## Acceptance Criteria

- #29901 AMD qualification precedes local sparse-FA work.
- Direct FA is implemented only after the materiality gate passes.
- Promote for >=3% decode-wall or >=0.5 ms/token gain at >=160K with <=2% short-context regression and correctness intact.
- No second dense selection buffer or duplicate indexer implementation.

## Notes

2026-10-05 review: superseded by owner item QFP04, which already carries the #29901-first ordering, bc_qsa_idx and the 160K / 0.5 ms/token materiality gate. Not implemented; work stays open under QFP04.

## Related

QFP04, QFP17, QFP22, patches 1295/1296; llama.cpp #29901; SYCL sparse-FA mechanism evidence.

## Change Log

- 2026-10-05T04:37:19.506465+00:00 (updated-by): Updated: section:notes
- 2026-10-05T04:38:05.005082+00:00 (state-transition): State: pending → superseded
