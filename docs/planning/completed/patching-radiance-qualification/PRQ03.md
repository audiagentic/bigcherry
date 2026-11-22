---
id: PRQ03
order: 0
plan: patching-radiance-qualification
state: completed
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

Every historical named candidate is either validated, diagnostic with activation proof, rejected or superseded; no historical untested-in-production exception remains. Do not infer new model quality.

## Notes

2026-11-20: rad0005_mp4_ffmpeg6_writer is archived (superseded by Radiance 2.2, which includes the FFmpeg 6 write callback support upstream; RUM02). The rad0005 mentions above are historical; no FFmpeg 6 build test or host-compat declaration is needed for it any more. See engines/radiance/patches/archive/ARCHIVE.md.

2026-11-22 reviewed closure RV4328: The old untested-decision matrix now has terminal archived rad0003, rad0005, rad0006 and rad0007 decisions, validated rad0030 and rad0036 in production, and diagnostic rad0004 and rad0010 excluded from production. Evidence: docs/evidence/2026-11-22-prq01-prq05-reconciliation/README.md. Focused negative/positive source-class regression 11 passed, 5 subtests. No new GPU, throughput or code changes; preserve 2.2 RUM/LTX owners.

## Acceptance

patch-status shows no untested radiance patch without a decision recorded in this item.

## Change Log

- 2026-11-15T22:57:03.917108+00:00 (created-by): Created by agent

## Status 2026-11-16: patches still untested and the decision for each

- rad0030_int8_pipelined_prefill: IN PRODUCTION (qwen27b-tp3-int8 turns it on), so it cannot be dropped without breaking that layout. Not bit-identical to serial at 3 of 7 prompt lengths (VTP37). Needs the owner's reordering quality gate (VTP37.md: noise floor vs a serial run at 1024-token chunks, and no loss vs BF16 on the 42,761-position KLD corpus), "not yet run, held 2026-11-13". Test: KLD of int8 TP3 with and without rad0030. Then promote, or demote with the layout updated.
- rad0036_prefill_governor: IN PRODUCTION (-long configs use `kind=budget`). Hardware A/B exists (docs/evidence/2026-11-16-vtp42-prefill-governor/README.md: 100k mean 299 -> 227 s, mid-size first-token times held, no preemptions) but no hardware greedy-text identity of budget against cap arms, and decode beside the mix and fp8 -long are not measured. Test: one-stream text sha under both specs, a decode-beside-mix arm, fp8 -long. Then promote.
- rad0005_mp4_ffmpeg6_writer: host-compat (build enabler), in production, never run on an FFmpeg 6 host. Test: compile and mp4 write on an FFmpeg 6 host or declare it host-compat by the bump's compile/ABI/cpu-smoke gate (set comment says so).
- rad0004, rad0006, rad0010: diagnostics in diagnostic-only sources (moe-remote-diag, routing-evidence); remain untested by design, tracked here. rad0006's result is already recorded (mover ruled out, RAD14); rad0010's RHP01 measurements are in its SUMMARY.
- rad0003_mtp_full_head_switch (experiment only): no acceptance gain (0.63-0.65 vs 0.62-0.64). Decision: reject; needs a reviewed transition record.
- rad0007_hc_pf2_fp8_down: still about 1.9x slower than the bf16 path after tuning (999.6 vs 530.7 us at M=2048). Decision: reject unless a new kernel shape; needs a reviewed transition record.
- rad0001, rad0034, rad0035: superseded; state only.

## Outcome 2026-11-16

- rad0030_int8_pipelined_prefill: validated (#992) on a relative KLD gate against the VTP36 BF16 reference, per owner approval; the VTP14 BF16 reference was not on disk.
- rad0036_prefill_governor: validated (#992); the optimization tag was dropped (#991) because no llama.cpp native arm applies.
- rad0005_mp4_ffmpeg6_writer: compiled and exercised against FFmpeg 6; validated (#993).
- rad0003_mtp_full_head_switch and rad0007_hc_pf2_fp8_down: remain in progress, no rejection recorded (owner decision).
- rad0004, rad0010 (diagnostic) and rad0006 (superseded): handled by PRQ05.
- The production exception list (untested-in-production.toml) is empty; PRQ05 replaces it with source classes.
- 2026-11-20T12:47:11.749269+00:00 (updated-by): Updated: section:notes

## Reviews

- RV4328
- 2026-11-22T04:59:06.960387+00:00 (updated-by): Updated: section:acceptance_criteria, section:notes
- 2026-11-22T04:59:17.057996+00:00 (state-transition): State: pending → completed
