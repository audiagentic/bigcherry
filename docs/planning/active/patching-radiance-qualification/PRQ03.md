---
id: PRQ03
order: 0
plan: patching-radiance-qualification
state: pending
created-at: '2026-11-15T22:57:03.917108+00:00'
breadth: ''
skill: intermediate
created-by: agent
priority: P1
work: L
---

# Qualify or rule out the radiance patches left untested

## Description

Patches not promotable from existing evidence, each needs a documented decision (promote, keep opt-in experimental, or reject) after a small test. rad0003 (MTP full head; measured no acceptance gain 0.63-0.65 vs 0.62-0.64: candidate reject/supersede). rad0005 (FFmpeg 6 MP4 writer; never run on a host with FFmpeg 6; needs a mechanics build test or host-compat declaration). rad0007 (FP8 PF2 down; 999.6 us vs bf16 530.7 us, ~1.9x slower after tuning: candidate reject unless a new kernel shape). rad0030 (int8 pipelined prefill; TTFT -17..-21%, 4/6/8 streams +7..8%, but not bit-identical at every prompt length: needs a KLD/top-1 check against the serial step to qualify on quality). rad0036 (prefill governor; on-card -long result without a recorded ABBA with activation and identity). rad0001/rad0034/rad0035 are superseded (state only). For each patch the item records: SUMMARY outcome updated, the test run, the decision, and the owner.

## Steps

1. rad0030 KLD/top-1 vs serial step on the TP3 int8 layout (short lease).
2. rad0036 ABBA with activation marker and identity on the -long config.
3. rad0005 FFmpeg 6 build test or host-compat declaration.
4. rad0003 and rad0007: confirm the evidence, then reject/supersede with a reviewed transition record.
5. Update each SUMMARY Outcome and this item with the decision.

## Detailed Solution & Technical Design



## Code Samples & Guidance



## Files



## Validation

Each patch is validated, explicitly experimental with a stated reason, or rejected/superseded with a transition record; none stays untested without an entry here.

## Effort & Risk



## Standards



## Acceptance Criteria



## Notes



## Acceptance

patch-status shows no untested radiance patch without a decision recorded in this item.

## Change Log

- 2026-11-15T22:57:03.917108+00:00 (created-by): Created by agent
