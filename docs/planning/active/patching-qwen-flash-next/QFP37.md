---
id: QFP37
order: 37
plan: patching-qwen-flash-next
state: pending
created-at: '2026-10-07T00:39:44.284926+00:00'
breadth: ''
skill: advanced
created-by: agent
priority: P2
work: M
---

# MMQ work distribution: stream-k for few-tile matmuls and expert tile shape for our quant

## Description

External report: (a) stream-k for few-tile Q8_0 matmuls (10240->320 projections spread over all CUs instead of 12 workgroups), 284 -> 115 us; (b) 32-row tiles for 160-row Q4_K expert slices, 360 -> 337 us. Our Flash-Next is UD-IQ4_XS and the 27B is Q8_0: (a) applies to the 27B and to Q8 tensors in Flash-Next, (b) needs redoing for the IQ4_XS expert slice shape.

## Steps

1. Census of MMQ calls with few tiles and their CU occupancy. 2. Stream-k variant behind a flag. 3. Tile-shape sweep for expert slices on RDNA3 and RDNA4. 4. ABBA on Flash-Next and 27B.

## Detailed Solution & Technical Design



## Code Samples & Guidance



## Files



## Validation

Bit-identical integer dot products where order is preserved, else equivalence; per-kernel timing; ABBA on both models.

## Effort & Risk



## Standards



## Acceptance Criteria



## Notes

## Change Log

- 2026-10-07T00:39:44.284926+00:00 (created-by): Created by agent
