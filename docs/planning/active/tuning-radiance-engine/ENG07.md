---
id: ENG07
order: 0
plan: tuning-radiance-engine
state: pending
created-at: '2026-11-19T15:40:13.822474+00:00'
breadth: ''
skill: advanced
created-by: agent
work: M
priority: P3
---

# Quantise the QSA block-key group of the bf16_i8 KV layout (design first, retrieval and KLD gates)

## Description

Split from ENG06 step 4. Flash-Next holds a QSA block-key group (3.00 KiB a page over 12 layers; 3.25 KiB with its scales) that no packed KV layout touches. With bf16_i8 the attention group is 36.75 KiB a 4-token page and the whole cache 43.06 KiB, so the block keys are about 7% of the cache. Storing them as int8 with a scale per block would take the whole-cache ratio against bf16 from 1.283x to about 1.31x (about 2% more tokens). The keys choose which blocks attention reads, so the risk is wrong block selection, not value noise: the gates are retrieval and KLD, not KLD alone.

## Steps

- Design (no GPU): read where the block keys are written and read (the QSA selection op and its page layout in the rad0038/rad0040/rad0041 patches and libr4d), decide the quantiser (int8 with an f16 scale per block or per 32 dims, as V does) and where rounding can flip a top-k boundary; write the layout row in the rad0040 table form.
- Block-selection agreement check: the fraction of (token, head, layer) top-k selections that change against bf16 keys on the long corpus at 48K, 100K and 248K, before any serving run.
- Gates as ENG06 step 2: KLD against bf16 on the long corpus (bar: no worse than bf16_i8 0.00269), needles at 48K and 100K 18 of 18 (tools/lab/radiance/needle_probe.py), decode and TTFT parity on flashnext-4gpu (flashnext-parity.sh, two launches), capacity from the engine log.
- Ship as an opt-in layout row, or reject with the agreement numbers.

## Detailed Solution & Technical Design



## Code Samples & Guidance



## Files



## Validation

Step 2 agreement report, KLD and needle numbers against the ENG06 evidence (docs/evidence/2026-11-19-eng06-bf16i8/), parity compare against the current-main baseline, engine capacity line.

## Effort & Risk



## Standards



## Acceptance Criteria

Either the block-key quantisation ships as an opt-in layout row with the whole-cache capacity ratio measured and KLD, needle and parity results at or better than bf16_i8's, or it is rejected with the selection-agreement and KLD numbers recorded.

## Notes

Design done 2026-11-20 (docs/evidence/2026-11-20-eng07-qsa-design/README.md). Recommended format: signed int8 QSA block keys with one f16 scale per 32 dims (alternatives: int8 + scale per 128-dim block, unscaled E4M3). Saving at one 262,144-token sequence over 12 QSA layers: bf16 192 MiB -> ~102 MiB (108 MiB with 16-byte page alignment); whole-cache bf16_i8 ratio ~1.283x -> ~1.33x bf16 (to be measured). It does not close Flash-Next bf16_i8's 5.47 GB eight-session shortfall. Risks: QSA group is not an ordinary KV layout (page pitch, residency, restore), top-k block-selection reversals (primary gate: selected-block overlap vs bf16 on recorded query tensors; KLD alone is insufficient), fused dequant register pressure on gfx1100, speculative rollback/MTP/prefix-reuse state. Priority lowered to P3 (owner direction 2026-11-20: finish items; gain is a few percent of capacity at retrieval-correctness risk); revisit if long-context capacity on Flash-Next becomes the binding constraint.

## Change Log

- 2026-11-19T15:40:13.822474+00:00 (created-by): Created by agent
- 2026-11-19T15:40:32.803620+00:00 (updated-by): Updated: section:description, section:steps, section:validation, section:acceptance_criteria
- 2026-11-19T19:42:34.688293+00:00 (updated-by): Updated: priority='P3', section:notes
