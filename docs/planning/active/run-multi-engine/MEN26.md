---
id: MEN26
order: 0
plan: run-multi-engine
state: pending
created-at: '2026-11-03T00:02:59.199386+00:00'
breadth: ''
skill: intermediate
created-by: agent
priority: P1
work: M
---

# Radiance production release: promote what Flash-Next actually serves, and serve from release builds

## Description

Every radiance patch we serve is state=untested. All Flash-Next serving and A/B work runs the lab lane radiance-lab:stock:linux-radiance+moe-remote-handoff, which builds rad0002 (remote-MoE hand-off) and the moe-remote component as an experiment, while source.radiance (production) is serving-core [rad0008] + host-compat [rad0005] + serving-manifest [rad0009]. rad0002 sits in patch-set moe-remote-diag next to diagnostic-only rad0004/rad0006. No bc-radiance release has been cut since 1.8.0, and the pin is 1.14.0. The promotion machinery (bigcherry patch-promote with evidence, release lines in config/release.toml) exists but is unused for radiance. An evidence audit on 2026-11-02 (GPT, existing evidence only; /work/development/tmp/bigcherry/radiance-promotion/AUDIT.md) found: rad0002 has activation, stability, performance and mechanics evidence but no current-pin helper-on vs engine-local identity (the 1 Nov KLD compares against a 23 Oct reference made under radiance 1.8); rad0009 has no hardware --manifest launch (it gets one at the A4 cutover); rad0005's FFmpeg path is not compiled on this host (no libavformat); rad0008 is marked validated but its 1.9.0-scoped evidence fails patch-verify-evidence for the current pin; components have only per-setting statuses (4 of 56 moe-remote settings validated).

## Steps

1. One current-pin KLD of helper offload against a fresh engine-local reference (the only missing rad0002 measurement; nothing else re-run).
2. Fix patch-promote's _update_recipes() so a promoted patch leaves every other patch set (today it would stay in moe-remote-diag).
3. Promote rad0002 --profile-only (Flash-Next is the only model using helper offload) with the existing evidence plus step 1; move it to serving-core; retire experiment moe-remote-handoff and its lab-build argument; keep rad0004/rad0006 diagnostic-only.
4. Re-bind or re-record rad0008's evidence for the 1.14.0 pin from the existing kernel identity/perf record, or re-measure only if the record cannot bind.
5. rad0009: promote from the A4 cutover's hardware --manifest identity run; rad0005: record as host-compat on hosts with FFmpeg 6, untested here.
6. Cut a bc-radiance release through the release path; point the production launcher at the release build, with lab lanes for experiments only.

## Detailed Solution & Technical Design



## Code Samples & Guidance



## Files



## Validation

patch-verify-evidence passes for every promoted patch at the current pin; patch-lint and bigcherry check pass; the release build's composition proof equals the lab build used for Flash-Next minus experiments; a production-launcher start from the release build serves identical text to the lab build (1 stream, ABBA).

## Effort & Risk



## Standards



## Acceptance Criteria

Flash-Next serves from a bc-radiance release build whose patches are validated with current-pin evidence; no served patch is untested; rad0002 is no longer in a diagnostic set; the lab source is production plus experiments (enforced by test_radiance_sources_share_production from #538).

## Notes

Owner direction 2026-11-02: use the evidence we already have; do not re-run for its own sake. Only measurements no existing run performed are added (step 1, and rad0009 via A4).



PROGRESS 2026-11-03, step 1 done (/work/bigcherry/runs/men26-kld; config kld.json, kld-isolate.json from tools/lab/radiance/helper-kld.json).
- Build: origin/main 2f5fc39df (slice C merged); rad0002 patch_validation_subject_digest 663b0856fbc4da97703a059711349be1c5c3cd0373211e701f6e8864f215c947.
- Reference: freshly recorded engine-local reference (TP2 exact over the two XTX, corpus rad-kld-corpus-4.jsonl, 8240 positions).
- Determinism: three captures are bit-identical.
- Helper offload on the two XTX helpers PASSES every limit vs the R9700-alone control: KLD mean +0.003, median +0.0006, p99 -0.034, p99.9 +0.41 (limit 0.5), top-1 -0.36 pts, ppl +1.2%. All helpers served.
- This is the rad0002 identity evidence (profile-scoped: Flash-Next is the only helper-offload model).
- The production tail layout (adds the 6900 XT) fails p99.9 (+1.58); tracked as RAD46. That is a gfx1030 helper-kernel issue, not rad0002.
- Next: steps 2-3 once #538 (lab source includes production) and #543 (patch-promote leaves other sets) merge.

## Change Log

- 2026-11-03T00:02:59.199386+00:00 (created-by): Created by agent
- 2026-11-03T02:29:45.375114+00:00 (updated-by): Updated: section:notes

## Ledger-events

- chg_20261103_022951_measured-that-offloading-exper_7122
- 2026-11-03T02:29:55.053712+00:00 (updated-by): Updated: section:ledger-events
