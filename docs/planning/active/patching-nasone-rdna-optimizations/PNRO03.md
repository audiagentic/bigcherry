---
id: PNRO03
order: 0
plan: patching-nasone-rdna-optimizations
state: pending
created-at: '2026-09-09T10:52:12.743676+00:00'
breadth: ''
skill: advanced
created-by: capability-rebaseline-v3
work: L
priority: P0
---

# Correctness-first HIP P2P transport provider for internal AllReduce

## Description

TODO, NOT-READY (rescoped). Verified via patch.py: 1252 only adds an env-var flag (`p->nro03_p2p_requested`) and a `cudaMemcpyPeerAsync` helper function -- there is no peer-capability/enable probe, bidirectional correctness probe, scratch ownership, transport selector, fallback wiring, or activation marker anywhere in the package. Its validation section's `--requires 1001_hip_internal_allreduce` is stale (b11126 already has HIP internal AllReduce natively; that composition flag should be removed).

## Steps

1. In `ggml_cuda_ar_pipeline_init()` (or the equivalent real init function -- verify exact name at implementation time), probe and enable both peer directions explicitly (not just read the env flag) and validate asymmetric nonzero copies in both directions before allowing P2P selection.
2. Add source-owned streams/events and destination-owned scratch buffers to the `ggml_cuda_ar_pipeline` struct (verify exact struct name), using the existing `cudaMemcpyPeerAsync` helper as the actual transfer primitive once probes pass.
3. Select P2P in the real allreduce dispatch function only after both directed probes succeed; otherwise fall back to the existing host-staging path unchanged.
4. Add a hit/fallback activation marker (BIGCHERRY_PATCH_TRACE-gated) so P2P selection is observable, since none exists today.
5. Remove the stale `--requires 1001_hip_internal_allreduce` from this item's validation command (b11126 already contains HIP internal AllReduce; 1252 does not need to require it as a separate composed patch unless its own patch.toml says otherwise -- verify at implementation time).
6. Require exactly two devices, bidirectional peer capability, peer enable, and completed correctness probes as the acceptance gate.
7. Sweep sizes, validate every element, compare host staging/P2P, then measure real internal-AllReduce decode/prefill; reject if no stable winning envelope.

## Detailed Solution & Technical Design

Capability owner: patching

Split assessment: One independent boundary; Build/Run support is a dependency.

Overlap assessment: No duplicate boundary found; related items are prerequisites or adjacent evidence.

## Code Samples & Guidance



## Files

patches/1252_nro03_allreduce_p2p_provider; provider selector/probes; static tests; dual-gfx1100 evidence

## Validation

Patch mechanics: `PYTHONPATH=tools python -m bigcherry patch-lint patches/1252_nro03_allreduce_p2p_provider`; `PYTHONPATH=tools python -m bigcherry patch-rebase-check --focal-overlay 1252_nro03_allreduce_p2p_provider --source bigcherry-tuning`; package pytest offline. Bidirectional synthetic validation across sizes/edges, nonzero/asymmetric values, peer-enable handling, forced fallback, GPU-count!=2 control. Hardware (Brutus only, dual gfx1100 required): `python -m bigcherry.patch.validation_campaign --overlay 1252_nro03_allreduce_p2p_provider --requires 1001_hip_internal_allreduce --arch gfx1100 --devices 2` -- correctness before bandwidth; reject if no stable winning envelope.

## Effort & Risk



## Standards

PGC corrected Brutus evidence; capability bits/API success are not correctness; fail closed to validated host staging.

## Acceptance Criteria

Both directed probes pass repeatedly with source-current push evidence; failures disable P2P without correctness loss; a repeatable collective-level winning envelope exists or the provider is rejected; no global default without independent topology coverage.

## Notes

Supersedes: NRO03
Migration: capability-rebaseline-v3-2026-09
Successor key: patching-nasone-rdna-optimizations-nro03

Supersedes: NRO03
Inherited semantic scope: preserve source-current push direction, probe/fallback, no direct peer reads, and correctness-first acceptance.
Migration: capability-rebaseline-v3-2026-09

REAL FINDING 2026-09-12: found the identical compile-breaking anchor bug fixed in patches/1250 (NRO01) this same session, by inspection (same author/batch, same pattern: anchor ending at '=' mid-statement). Fixed proactively and re-verified with a real isolated gfx1100 build (1001+1252 composition): clean build, BUILD_OK. Closes this draft's 'applies cleanly, builds HIP' bar for the first time. All other acceptance criteria (bidirectional probes, scratch ownership, fallback, content-checked performance campaign) remain entirely unstarted.

2026-09-24 relevance at b11126: IMPLEMENTED-AS-PATCH. patches/1252_nro03_allreduce_p2p_provider exists, state=untested. Prior session (2026-09-12) fixed the same anchor/compile bug class as PNRO01 and confirmed clean gfx1100 build (1001+1252 composition). No upstream equivalent. Disposition: validate/qualify existing patch; no GPT design needed.

2026-09-24 GPT review req_215c89d0b13a4bb7 applied: verified 1252 only has an env flag and a bare cudaMemcpyPeerAsync helper -- no probe/scratch/selector/fallback/marker exists. Added the required peer-probe/enable step in ggml_cuda_ar_pipeline_init(), source/destination stream+scratch ownership, the actual P2P-selection gate, a required activation marker (none existed), and removed the stale 1001 requires composition.

2026-09-25: IMPLEMENTED (commits bd2fc067/45219a0d). patches/1252 is now a real port of 7c5bb5cb adapted to this item's invariants: per-direction streams/events on the SOURCE device, set_device(source) before every cudaMemcpyPeerAsync, no issuer, ggml_cuda_ar_p2p_probe (4 sizes x 2 directions x 2 passes, byte compare) gates GGML_CUDA_AR_P2P. Marker patch=1252_nro03. Rebase CLEAN at b11126. Next: build + hardware arms per TESTING.md (P2P on vs off, same binary, 2x gfx1100).

## Change Log

- 2026-09-09T10:52:12.743676+00:00 (created-by): Created by capability-rebaseline-v3
- 2026-09-09T11:08:41.098537+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:files, section:validation, section:standards, section:acceptance_criteria, section:notes

## Ledger-events

- chg_20260909_115759_created-and-populated-the-192_2958
- 2026-09-09T11:58:01.061978+00:00 (updated-by): Updated: section:ledger-events
- chg_20260910_001436_completed-the-planning-rebasel_5794
- 2026-09-10T00:14:42.701391+00:00 (updated-by): Updated: section:ledger-events
- 2026-09-10T02:26:09.609126+00:00 (updated-by): Updated: section:description, section:steps, section:files, section:validation, section:standards, section:acceptance_criteria, section:notes
- chg_20260910_022800_five-nasone-successor-plans-no_4030
- 2026-09-10T02:28:00.291162+00:00 (updated-by): Updated: section:ledger-events
- 2026-09-12T03:47:57.075297+00:00 (updated-by): Updated: section:notes
- chg_20260912_034828_confirmed-on-real-hardware-tha_4451
- 2026-09-12T03:48:28.669796+00:00 (updated-by): Updated: section:ledger-events
- 2026-09-24T02:26:08.096956+00:00 (updated-by): Updated: section:validation, section:notes
- 2026-09-24T04:48:10.768762+00:00 (updated-by): Updated: section:description, section:steps, section:notes
- chg_20260925_111345_real-ports-of-the-nasone-allre_4524
- 2026-09-25T11:13:54.348499+00:00 (updated-by): Updated: section:ledger-events
- 2026-09-25T11:14:00.720422+00:00 (updated-by): Updated: section:notes
