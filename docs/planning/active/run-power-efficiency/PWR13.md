---
id: PWR13
order: 0
plan: run-power-efficiency
state: in_progress
created-at: '2026-11-26T01:45:56.843677+00:00'
breadth: ''
skill: advanced
created-by: agent
work: L
priority: P1
---

# Expand independent GPU qualification workloads and fail-closed proof

## Description

Extend existing platform gpu-ai-screen beyond repeated BLAS and sequential memory. Independent coverage only; no engine dependency or duplicate controller/profile/store. PWR12 retains physical research; RCS26–28 own native replay; PWR07/08/RSC02 own admission/recovery; PWR10/PWR02 own production.

## Steps

1. Review owner admission/restore before unattended mutation. 2. Typed cases, diverse CPU oracles and verifier fault checks. 3. Capability-proven instruction, memory, transfer, mixed and transition cases. 4. Managed evidence; observe-only hardware cohort then approved finalist tests.

## Detailed Solution & Technical Design

docs/design/gpu-qualification-code-guide.md sections 1–12.

## Code Samples & Guidance



## Files

tools/bigcherry/hardware/ai_probe.cpp, ai_probe.py, cli/ai_probe_cmds.py, tools/tests/hardware/test_ai_probe.py; existing power owners by coordination.

## Validation

CPU vectors, fault injection, malformed coverage, target compile/ISA proof, leased observe-only cases on supported cards, exact restore and matched finalist receipts.

## Effort & Risk



## Standards



## Acceptance Criteria

Required cases complete with positive verified work; missing/unsupported/thermal/recovery nonpass. No unattended writes until recovery gates pass. No auto promotion; native and serving gates separate.

## Notes

2026-11-26 re-review and first implemented slice: rotating dense/sparse/signed-diagonal exact corpus, guarded output and per-corpus host/device verification, input readback, comparator faults at three positions, native protocol 2 / observation schema 3; old reference and stale binary refused before candidate transaction. ROCm 7.2.4 builds gfx1030/gfx1100/gfx1201, CPU self-test, 197 hardware tests pass (one skipped), 774373 clean GEMMs over four physical cards and 28 full 128 MiB memory passes. Six observe-only managed captures harvested: 20261126T015614Z-gpu-ai-screen-512f35e587ee, 20261126T015751Z-gpu-ai-screen-82c5b9fae077, 20261126T015805Z-gpu-ai-screen-5cb973591371, 20261126T015818Z-gpu-ai-screen-835094a5e951, 20261126T015836Z-gpu-ai-screen-6c7afecb7598, 20261126T015921Z-gpu-ai-screen-4880e38ee7de. Detailed counts in AI_POWER_SCREEN.md. PWR13 stays in progress: varied shapes, direct instruction/wave/LDS, mixed traffic and hot/transient scopes outstanding; P0 owner admission and exact restoration fixes required before unattended mutation. No control or native owner edits.

2026-11-26 direct instruction slice: bc-platform 1.78.0 base, native protocol 3/screen schema 4; independent alu/wave/lds cases, exact CPU readback every launch, changing epochs, output guards/input integrity and strict geometry/count/headroom validation. New cases refuse profile application until owner recovery hardening. Actual code objects on gfx1030/gfx1100/gfx1201 show mul_hi/lo, bit extraction/rotate, ds_bpermute, explicit LDS read/write and barriers; initial volatile LDS emitted generic flat accesses and was corrected. 202 hardware tests passed, one skipped; CPU golden self-test passed. All 24 observe-only sustained/burst cases across four cards passed with 39635648512 verified 32-bit words and unchanged controls; four receipts harvested, IDs in AI_POWER_SCREEN.md. Host-readback-limited coverage, not hot or saturation proof. Remaining varied matrix geometry, native matrix/FP instructions, mixed residency/traffic and phase/transient/thermal work; P0 owner admission and native serving remain separate. No writer or production changes.

2026-11-26 rectangular geometry slice: explicit M,N,K,LDA,LDB,LDC NN layout, three exact corpora and full logical/padding comparison, strict geometry identity, protocol 4/schema 5. Initial missing native geometry serialization was correctly rejected (four harvested inconclusive receipts); serializer fixed and native validation failures now retain per-case diagnostic/output. Final four observe-only captures passed 15 requested cases: 2158560 GEMMs, 2082375360 GPU and 195222690 CPU padding comparisons; eight captures harvested, IDs in AI_POWER_SCREEN.md. CPU golden self-test and 210 hardware tests passed (one skipped), all three offload targets compiled. No power changes; explicit geometry refuses apply pending P0. Remaining mixed residency/traffic, direct matrix/vector instruction and phase/transient coverage; native and production gates separate.

2026-11-26 mixed-residency slice: protocol 5/schema 6, explicit resident budget and full known allocation headroom, independent nonblocking memory stream, 4 MiB changed-chunk exact host checks, initial/final full verification and complete compute-time sweep required. Three offload targets and CPU self-test passed; 213 hardware tests passed, one skipped. Five observe-only cases passed (129 MiB partial-chunk smoke plus 20992/20992/29184/12800 MiB full cohort); 671744 verified GEMMs and 66035122176 resident words across full-size runs, unchanged controls, all captures harvested (IDs in AI_POWER_SCREEN.md). Job-monitor restart interrupted two queued launches, resubmitted only after lease/process check. No actual overlap, hot floor or serving proof. New review confirms distinct gfx1100/gfx1201 WMMA layouts, no native WMMA on gfx1030, historical gfx1030 fdot2 builtin issue. Remaining native matrix/vector/packed-dot, peer traffic, phase/transition/health and P0 owner recovery/matched finalist gates. No control or native owner edits.

2026-11-26 tensor ISA slice: protocol 6/schema 7, vector FP, signed dot4/dot2, nonnegative dot2, FP16/BF16 WMMA signed/nonnegative, mixed IU8 WMMA. Three target code objects and CPU selftest passed; 222 hardware tests passed, one skipped. ISA proof as documented. Seventeen harvested observe-only captures: 70 passing cases, 7 strict mismatch cases, 28578021376 host-verified words in passing cases, all controls unchanged. gfx1100 both cards reproduce signed FP16/BF16 WMMA residue (0x3f7ffffa vs exact 1.0) and small signed dot2 discrepancies; nonnegative and mixed IU8 pass. gfx1201 passes signed and nonnegative; gfx1030 no native WMMA. Strict signed nonpasses retained; no tolerance or factory/production claim. Remaining recipe/transfer/phase/health/owner gates and separately reviewed precision contract.

2026-11-26: Implemented managed architecture/recipe-driven gpu-ai-qualify plan/run/report, digest-bound probe/profile/recipe/native scope, observed physical snapshots and VRAM-derived budgets, one leased idle-screen group, fail-closed harvested result aggregation. 235 hardware tests passed/1 skipped. Physical plan 20261126T034531Z-gpu-ai-qualification-plan-84c0394b3695 captured; GPU smoke queued behind active Gate E release acceptance, not yet proven. RCS27 model/release route execution remains its owner; candidate execution deliberately refused pending cross-owner recovery and reference gates.

2026-11-26: Managed screen end-to-end on helper0: report 20261126T035431Z-gpu-ai-qualification-report-2b0b6674b9be, six harvested groups: matrix/direct/wmma-positive pass; dot2-signed/wmma-fp16-signed/wmma-bf16-signed mismatch. Implemented additive telemetry schema 1 (timestamped power, raw clock/voltage and busy interfaces, memory/junction temperatures, gap-bounded interpolated energy across native load and requested warmup boundary); private read-only kernel log cursor, explicit inaccessible/overrun status, AMDGPU fault class counts stop work. 239 offline hardware tests passed/1 skipped. Hardware telemetry smoke not yet completed; Gate E holds lease. Estimates are diagnostic, not measured energy per token, and no control writes.

2026-11-26: Extended existing GP11/GP13 peer fixture with explicit physical BDF/architecture bindings, per-direction HIP capability reporting, per-endpoint allocation headroom and 48 exact copy/source verification cases (push, pull, explicit peer sync and async; six boundary sizes, both directions), leased managed wrapper with both thermal sensors, driver-health observation and unchanged snapshots. Compiled against ROCm 7.2.4 all three targets; 240 offline hardware tests passed/1 skipped. Six physical pairs probed, all 12 directional capabilities false; no copies launched, all unavailable not pass, controls unchanged, all harvested. IDs 20261126T040021Z-gpu-ai-peer-4f9358f01f4e, 20261126T040022Z-gpu-ai-peer-97262d198490, 20261126T040023Z-gpu-ai-peer-149d4d73955d, 20261126T040023Z-gpu-ai-peer-f55953947d8a, 20261126T040024Z-gpu-ai-peer-303657000cdd, 20261126T040025Z-gpu-ai-peer-b6056248773e. Host kernel log unavailable; health incomplete. Peer data-path hardware coverage remains blocked on peer-capable allocation; API success would not prove physical route. Production remains blocked.

2026-11-26: Real telemetry validation on engine R9700: transition smoke 20261126T040058Z-gpu-ai-screen-8dafa7337ac5 passed all five 0/1/2/10/100 ms load gaps, complete native-load energy estimates for each; observed hotspot 34-39 C, unchanged controls. 30+60 s sustained hot trial 20261126T040128Z-gpu-ai-screen-ee71f5be59e7 inconclusive (37-50 C, required >=60 C final window not reached); energy estimate 7202.76 J across native interval; unchanged controls, health unavailable. Both harvested. PWR10 owner holds GPU lease for first native min-cap apply; no duplicate control mutation, no profile promotion. PWR13 remains open for peer-capable data path, full hot/transition cohort, verified health and integration with native family/production owners.

2026-11-26: Scoped RDNA3 exceptions implemented (protocol 7/screen schema 8/recipe schema 2). Only gfx1100 signed dot2 and FP16/BF16 WMMA reference variants, zero offset and unchanged observed controls, exact CPU lattice and independent conservative arithmetic bound, eight SHA-256 epoch signatures, two earlier same-card verified captures and later exact match. Original receipts remain precision_observed/nonpass; all other checks strict. Both XTX signatures match across all three cases and sustained/burst patterns; 18 captures/36 cases, 10,244,849,664 compared words, max absolute dot2 1.1920928955078125e-7, WMMA 2.384185791015625e-6. Both cards passed all 12 exact workaround cases (FP32 FMA and bounded operand shift with correction, no serving change). Helper0 complete independent pass 20261126T043305Z-gpu-ai-qualification-report-8d680c97f580; helper1 reference cohort complete, remaining strict groups in progress. 249 offline tests pass, 1 skipped. Production/native qualification, hot/transition retest, peer-capable allocation and readable driver-health evidence remain gates. gfx1201 hot recipe selects FP32 based on previous effective load; thresholds unchanged. No power writes or blanket tolerances. PWR13 remains in_progress.

2026-11-26: Helper1 resumed and completed remaining strict matrix/direct/nonnegative groups. Both complete independent reports now pass with exactly the three precision allowances: 20261126T043305Z-gpu-ai-qualification-report-8d680c97f580 and 20261126T043753Z-gpu-ai-qualification-report-001a1473983b. Reports without references refuse all three signed groups. All other completion gates above remain outstanding.

2026-11-26 follow-up: documented measured-versus-unmeasured performance status and adoption gates in AI_POWER_SCREEN.md and gpu-qualification-code-guide.md section 21. PR1665 merged (751183e9b); no serving code/dispatch changed, no comparative kernel/serving benchmark, no speed or throughput claim beyond unchanged implementation. Static gfx1100 ISA confirms packed dot versus vector FMA/packing tradeoff, and single WMMA plus correction in biased variants. Reuse RCS27/28 native family selection, exact equivalent-work resource/replay and model quality/energy/no-regression gates before any serving replacement. Operational cohort queued in named existing GPU lease queue: engine FP32 hot first, then burst and remaining cards hot/burst (recipe schema 2, protocol 7; 60 C hot floor/80 C ceilings unchanged); one bounded lease per group, plan after lease admission, harvest/report existing managed captures, stop on unexpected nonpass or altered controls. At recording time RSC14 holds lease; no new hardware outcome yet. Peer capability and readable driver-health evidence remain blocked, production/native owner gates separate. No mutation.

2026-11-26 operational cohort completed: eight harvested observe-only captures, 20 native numerical cases pass, 3,283,552 GEMMs, all eight physical identities and control snapshots match before/after; no transaction requested. All 16 burst cases pass. Final 60 s >=60 C hot window passed engine (FP32, min 65 C), helper0 (BF16, min 61 C), helper1 (BF16, min 62 C); helper2 gfx1030 BF16 hot inconclusive (41–63 C over 90 s, no qualifying window, 263584 correct GEMMs). Four managed reports 20261126T050031Z-gpu-ai-qualification-report-e43773f7f083, 20261126T050606Z-gpu-ai-qualification-report-0378929b2047, 20261126T051140Z-gpu-ai-qualification-report-c9eb513d60d6, 20261126T051715Z-gpu-ai-qualification-report-87d0333aac4f (last nonpass). Driver health incomplete in all cases; no peer/native/production proof in these reports. PR1670 merged native scope/evidence binding separately, not execution proof. Keep PWR13 in_progress; investigate architecture-supported gfx1030 load selection using existing recipe and unchanged thermal limits, health access and peer capability under existing owners; no unattended mutation or profile promotion.

2026-11-26 follow-up gfx1030 FP32 hot diagnostic: scratch recipe selected supported rocBLAS FP32 (same N=768, batch 32, 30 s warmup, 60 s measured window, 60 C floor and 80 C ceilings). Managed plan 20261126T053453Z-gpu-ai-qualification-plan-e4d878519b21, harvested capture 20261126T053453Z-gpu-ai-screen-096213d56357 and report 20261126T053624Z-gpu-ai-qualification-report-f6ae62d0827e. Native arithmetic passed 295168 GEMMs; hotspot 41–58 C, no qualifying hot window, overall nonpass/inconclusive. Controls and identity exactly unchanged, no transaction, driver health incomplete. FP32 is not a replacement for BF16 on this evidence; preserve both nonpasses and investigate more sustained effective load or larger geometry under same limits. Two setup attempts failed before compute (missing YAML dependency in alternate Python; lease identity mismatch); corrected interpreter and lease owner before the successful diagnostic. No profile promotion.

2026-11-26: resolved remaining gfx1030 hot coverage by bounded observe-only geometry investigation. 1536 BF16 90+60 s requested stopped at unchanged 80 C ceiling after ~83 s wall, no numerical completion claim; harvested 20261126T055347Z-gpu-ai-screen-22528262f5c7, report 20261126T055510Z-gpu-ai-qualification-report-b374e720798c. Intermediate 1024 BF16 with 60 s warmup + 60 s measured passed 200576 GEMMs, final 60 s 66–74 C, overall 42–74 C; capture 20261126T055532Z-gpu-ai-screen-2118fdba6724, report 20261126T055733Z-gpu-ai-qualification-report-53177bcb5620. Both controls/identity unchanged, driver health incomplete. Canonical target hot_n/hot_warmup_seconds resolve exactly the passed options; no change to other groups, architectures, measured window or safety ceilings. All four cards now have successful individual hot and burst evidence across managed reports; not one aggregate cohort and not factory/candidate/production qualification. Peer capability and driver health remain blocked, native/owner gates separate.

2026-11-26 owner-directed policy correction: ordinary operational/extended sustained group (historical ID hot) no longer requests min_hotspot_c, with unchanged geometry, warmup, measured duration and upper safety ceilings; optional elevated-temperature recipe retains the prior 60 C coverage floor. Preserve historical reports without reclassification. CLI accepts warmup without a temperature floor. 293 hardware/CLI tests pass, one skipped; fresh floor-free candidate receipts use newly matched references. No change to observe-only restrictions or native/production gates.

## Change Log

- 2026-11-26T01:45:56.843677+00:00 (created-by): Created by agent
- 2026-11-26T01:52:35.520482+00:00 (state-transition): State: pending → in_progress
- 2026-11-26T02:00:20.973947+00:00 (updated-by): Updated: section:notes
- 2026-11-26T03:02:56.388933+00:00 (updated-by): Updated: section:notes
- 2026-11-26T03:10:45.923234+00:00 (updated-by): Updated: section:notes
- 2026-11-26T03:21:55.716861+00:00 (updated-by): Updated: section:notes
- 2026-11-26T03:42:09.263099+00:00 (updated-by): Updated: section:notes
- 2026-11-26T03:50:43.706946+00:00 (updated-by): Updated: section:notes
- 2026-11-26T03:56:02.771051+00:00 (updated-by): Updated: section:notes
- 2026-11-26T04:01:14.438957+00:00 (updated-by): Updated: section:notes
- 2026-11-26T04:03:55.901123+00:00 (updated-by): Updated: section:notes
- 2026-11-26T04:37:34.147358+00:00 (updated-by): Updated: section:notes
- 2026-11-26T04:38:17.313462+00:00 (updated-by): Updated: section:notes
- 2026-11-26T04:45:49.998369+00:00 (updated-by): Updated: section:notes
- 2026-11-26T05:32:58.821400+00:00 (updated-by): Updated: section:notes
- 2026-11-26T05:36:53.590760+00:00 (updated-by): Updated: section:notes
- 2026-11-26T05:59:28.544787+00:00 (updated-by): Updated: section:notes
- 2026-11-26T06:54:41.407843+00:00 (updated-by): Updated: section:notes
