---
id: BCOP09
order: 9
plan: bc-optimizations
state: superseded
created-at: '2026-10-05T04:38:00+00:00'
breadth: ''
skill: advanced
created-by: agent
priority: P1
work: M
---

# Gate runtime expert slicing on transient host-weight pipeline results

## Description

Follow-through for MET06. Before building a new expert-granular GGUF loader/materializer, qualify the cheaper upstream #29963-style transient host-weight scheduler path and pinned/prefetched variants.

## Steps

1. Qualify CPU-host execution, transient upload, pinned transient upload, one-layer-ahead prefetch and persistent MET01 residency at equal model placement/VRAM.
2. Measure ROCm host-registration cost, locked-memory high-water, H2D bandwidth, copy-engine utilization, overlap, PP/TG and TTFT.
3. Reuse #29963 scheduler lifetime/copy machinery; do not build a second staging scheduler.
4. Proceed to MET06 expert-granular materialization only if coarse whole-layer placement leaves >=5% PP/TG opportunity or >=1 GiB avoidable resident expert memory at equal throughput.
5. If expert slicing proceeds, preserve canonical host weights and coalesce contiguous expert slabs; avoid duplicate host/GPU backing with cache paths.

## Detailed Solution & Technical Design



## Code Samples & Guidance



## Files



## Validation



## Effort & Risk



## Standards



## Acceptance Criteria

- Transient/pinned/prefetched host-weight paths have AMD measurements.
- The expert-slicing implementation gate is explicitly passed or failed.
- No duplicate scheduler/staging/cache implementation is created.

## Notes

2026-10-05 review: superseded by owner item MET06, which is the same gate (#29963 transient/pinned/prefetch qualification, 5% / 1 GiB thresholds). Not implemented; work stays open under MET06.

## Related

MET01, MET05, MET06, QFP27; llama.cpp #29963.

## Change Log

- 2026-10-05T04:37:09.563871+00:00 (updated-by): Updated: section:notes
- 2026-10-05T04:37:55.079395+00:00 (state-transition): State: pending → superseded
