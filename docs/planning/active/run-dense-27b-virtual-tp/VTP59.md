---
id: VTP59
order: 0
plan: run-dense-27b-virtual-tp
state: in_progress
created-at: '2026-11-27T09:44:54.101057+00:00'
breadth: ''
skill: ''
created-by: agent
priority: P1
work: M
---

# Fix medium-prompt concurrent prefill and decode batching before 2.3 bump

## Description

BC 2020.23.2 IQ8R TP2, 4096-token target and 128 generated tokens: four concurrent streams yield 42.25 aggregate tok/s and 11.08 s max TTFT, versus two streams 49.79 tok/s and 4 s. Decode tokens/step collapse from 6.05 to 3.10 without KV preemptions. Owner requires fix before bump completion.

## Steps

1. Audit resolved defaults and provenance on all shipped layouts: historical, 2.2, candidate. 2. Sparse boundary screen, then repeated fresh-start ABBA on anomalies. 3. Isolate interleave, mixed target, chunk budget, busy chunk, pipeline, governor and depth using measured critical path, not Cartesian sweep. 4. Reuse VTP42 simulator and scheduler tests; no duplicate controller. 5. Qualify per model/layout class, preserve long policy. 6. MRC08 matched cohort records and MEN17 exact runtime gates before bump.

## Detailed Solution & Technical Design



## Code Samples & Guidance



## Files



## Validation

Tier A: 64/128/256/512/1024/2048/4096/6144 and max context minus output reserve; output 32/128/512/1024; concurrency 1/2/4/6/8 and admission limit+1. Sparse screen first; exact realized B-1/B/B+1 at tile, chunk, checkpoint, page and speculative-row boundaries; repeat anomalies ABBA. Tier B: staggered arrivals, short beside long, sustained decoding beside prefill, independent/shared prefix cold/warm, cancellation and KV pressure. Long profiles 8k/16k/32k/64k/100k within capacity. IQ8R TP2/TP3, IQ6R V4 TP3, long fp8/bf16/bf16i8, FlashNext IQ4R/HCBF16 helper/native, single-card control. Match actual token IDs, model/drafter, power, topology, seed and effective config. Measure prefill and mixed duration, pure decode, active rows/step, prompt progress, TTFT, event and token gap tails, output throughput, errors/preemption, fairness, KV and independent quality. Predeclare noise/latency budgets.

## Effort & Risk



## Standards



## Acceptance Criteria

- [ ] Audit all shipped effective defaults and preserve intentional policies with provenance and tests.
- [ ] Boundary/arrival screening recorded; classify all regressions with matched independent repeated cohorts; no silent omissions.
- [ ] Per-class settings improve medium concurrency without material short/long/mixed or quality regression; explicit noise and latency budgets; separate profiles if needed.
- [ ] Same model/layout historical comparison identifies earliest regression or documents unavailable history.
- [ ] MRC08 and MEN17 bind exact old/current/candidate runtime and workload; candidate qualification before promotion.
- [ ] Merge validated implementation and evidence before bump promotion.

## Notes

Regression history: pre-2.2 default interleave=0; 2.2 defaults to auto:1 when omitted (documented upstream review 2026-11-18 and RUM08). FlashNext was explicitly protected but standard Qwen profiles were not. Earlier Qwen model was MXFP4, so direct historical speed parity not established. 2.2 IQ8R TP2 four-stream 4K/128 screen: interleave 0 54.879 output/410.158 decode tok/s, max TTFT 8068 ms, p95 gap 974 ms; fixed 2 54.019/334.455, 8164/947 ms; fixed 2 + mixed target 100 ms 41.836/193.816, 10964/139 ms, 512 prompt throughput falls 233.136 to 173.321. auto:0.25 50.014/219.278, 9122/947 ms. None qualified across all metrics. Existing mixed target controls aggregate prefill allowance during active decode; no new competing controller before controlled sweep. Preserve effective scheduler default drift audit as explicit bump gate. See docs/evidence/VTP59-scheduler-20261127/summary.json.

Owner 2026-11-27: extend concurrency to eight streams. Released TP2 defaults to six sessions; compare unchanged production cap separately from experimental eight-session cap, verify memory admission and active sequence counts, and use eight-session TP3 as topology control. Sweep 512/4096 prompts and 128/512 output, then validate mixed arrivals and quality. 250 ms mixed target + interleave 0 at four streams: 48.394 aggregate, 406.638 engine decode tok/s, 9.307 s max TTFT, 247 ms p95 event gap; promising compromise but not qualified.

Eight-session TP2 admitted 65536 tokens, 2.24 GiB KV/card and no preemptions. Same eight-session setting: 4096/128 auto:1 48.618 output/187.655 decode tok/s, 20.052 s max TTFT, 29.941 ms p95 gap; interleave 0 + mixed target 250 ms 50.435/640.001, 18.624 s, 265.936 ms, 23.27 tokens/step vs 4.90. 512/128 auto 262.466/554.707 vs tuned 252.609/654.547, so short end-to-end throughput drops 3.8%. These are screening measurements, not a production qualification. Larger batch budget 4096 with same mixed target isolates prefill throughput; retain 2048 and eight-session controls. Require repeat starts and quality, long/short arrivals, six-session production cap control, TP3 and candidate before final selection.

Completed eight-stream screens (3 reps each): 4096/512 auto 126.244 aggregate, 243.853 decode tok/s, 27.014 s max TTFT vs 0+250 ms 156.078/565.703, 18.951 s, 205 ms p95 gap. Raising batch budget to 4096 with 0+250 ms for 4096/128 gives 52.133/642.389, 18.025 s, 266 ms p95 gap. All owner exits clean; no preemptions in eight-stream screening. 19 benchmark report summaries retained; raw local evidence archived. Prefill remains limiting; next isolate pipelined prefill on supported TP2 and TP3 (already explicitly enabled on TP3), measure pure prefill and mixed-step timing separately, tune admission and latency using measured costs, verify quality and repeated starts before promotion. No production configuration changed.

TP3 IQ8R same release, 1/8 streams, 128 outputs, one repetition per cell, targets 128/1024/2048/4096/6144. Auto:1 eight-stream decode 721/400/245/220/137 tok/s vs interleave 0 + 250 ms mixed target 733/612/648/694/671. Aggregate 527/183/93/58/41 vs 502/182/110/61/45. Max TTFT 0.510/4.456/10.096/16.527/24.164 s vs 0.642/3.939/7.753/15.173/21.275 s. Tuned 128 prompt wall throughput -4.7%; event gap at 4K 250ms vs 24ms. Both clean stops, screening not qualification; exact boundary cases, fresh-start repeats, historical match, quality and other models pending. 29 reports in summary.

Fast host scheduler screen implemented in tools/lab/radiance/scheduler_burst.{cpp,py}, reuses real upstream sched_test.cpp and existing HIP-off rad_core archive. 62 scenarios, 18,157 checks, 0.06 s CPU (0.08 wall), 5.4 s compile/link; 23 launcher and existing kernel-harness pytest checks passed. Synthetic cost injection demonstrates 8x4096/128 auto 201 decode-containing steps peak 5 vs tuned 79 peak 8; all 1024 outputs and exact input rows completed. Not a GPU prediction; existing 2.2 host build scheduler/test source hashes match inspected 2.2 snapshot, not full candidate provenance. Promote to CTest in canonical build lane, generate scenarios from MRC08 workload keys, replay measured per-class cost curves, add negative mutation and arrival/KV pressure cases, native effective default drift tests; use unit_harness/kernel-group replay for changed kernels, narrow full serving gate for real latency/quality/communication.

VTP60 owns reusable fast qualification integration; see docs/design/radiance-fast-qualification.md. VTP59 retains pre-bump production scheduler qualification.

Use VTP60 native-first tuning path: rad-tune for kernel variants only, rad-kbench for correctness/replay, rad-info for scope, native --diagnose for measured serving and phase costs, sched_test sim_burst for deterministic policy screening after minimal mixed-target and clock extensions. Compare fixed and adaptive settings by workload progress, throughput, latency and fairness with frozen cost/acceptance bands. Retain full production quality and matched old/candidate serving qualification; do not treat simulation as hardware measurement.

2.3 exact candidate 69cccadc518f5282d312b0d84a08bc01: 6 managed IQ8R arms, 18 reports, all successful clean stops, no preemptions, matching prompt hashes/counts per cohort. TP2 0/250ms/4096 candidate 512/128,4096/128,4096/512 aggregate 261.646/52.583/157.212 vs auto 270.558/48.221/123.241; 4K512 p95 gap47.074ms vs auto34.82 and tuned 2048 budget203.904. TP3 0/200ms/2048 gives 278.018/61.314/189.469 vs auto296.786/62.496/148.594; 4K512 p95 gap41.186 vs34.287. TP3 0/100ms/4096 288.157/58.467/182.9, 4K128 decode715.476 but wall throughput lower. Recommend TP2 4096 budget and TP3 200ms 2048 as long-response finalists; retain default short-response controls. Single repetition only; require ABBA, mixed arrivals, latency bounds and independent quality; no production promotion. Evidence docs/evidence/VTP60-deterministic-screen/candidate-gpu-summary.json.

Combined 2.3 TP2 (pipeline on/min256 + 0/250ms/4096) and default: 3 reps each 512/192,4K/128,4K/512. Aggregate 355.173/60.880/172.127 vs 318.721/48.497/128.521; max TTFT 2.017/15.135/16.448s vs 2.638/20.020/26.102s. P95 long gaps 233/225ms vs29/37; no preemptions, matching actual prompts. Historical 2020.8.5 same target 512/192 TP2 335.7 aggregate/569 engine decode vs 355.173/673.717; old prompt identities not verified. TP3 default 347.917/570.182 vs tuned 350.280/644.140; old 361.2/574. TP3 tuned original interrupted after short workload; retry separate. Do not promote without ABBA, mixed arrival/quality and latency budgets. Evidence docs/evidence/VTP60-combined-screen/summary.json.

TP3 retry completed 3 reps each: 4K/128 aggregate 61.224 vs 56.919, engine decode 663.669 vs 221.807, TTFT 15.156 vs17.083s, p95 gap215 vs25ms; 4K/512 aggregate178.884 vs148.314, engine decode574.025 vs287.700, TTFT15.724 vs22.357s, p95 gap164 vs35ms. Retry separate launch after interrupted tuned short arm; all completed reports matched prompts and zero preemptions; not fresh-start ABBA. Prior single-run 189.469 not repeat median. TP2+TP3 totals 36 completed rounds. Do not select universal default; investigate latency-oriented profile and per-step efficiency.

TP3 IQ8R cold FP8 KV, eight streams, pipeline on/min256, batch2048, interleave off: fresh-launch ABBA mixed target 200/50/50/200 ms. 4K/512 aggregate 188.209/179.742/179.111/180.963 tok/s, max TTFT 14599.65/15818.91/15877.36/15615.70 ms, p95 event gaps 41.129/163.733/161.008/173.433 ms. 512/128 aggregate 270.325/280.315/278.785/278.959, max TTFT 2134.39/1973.44/2006.00/1989.36 ms, p95 gaps 194.062/40.337/40.248/40.697 ms. Eight completed reports, same prompt hashes and token counts, zero preemptions; all owner and stop exits zero. 50ms repeat stable, 200ms varies with schedule; text hashes differ, no independent quality gate. No universal default or production change. Evidence docs/evidence/VTP59-latency-abba/summary.json; raw evidence via existing evidence_dir. Next screen intermediate target and explicit interleave under mixed arrivals, trace phase variance and qualify quality independently.

RSC19 integration on node/serving-config-enhancement a3969b3c6: native /diagnostics before/after checks via existing lab-record adapter against exact manifest and explicit scheduler expectation. 100/150ms single-repeat TP3 IQ8R FP8 cold arms: 4K/512 aggregate 179.990/186.543, p95 gap 161.933/41.349ms; 512/128 aggregate 281.492/269.749, p95 gap 166.390/174.782ms. Same prompt hashes and token counts as ABBA, zero preemptions; no new default. Evidence owned once by RSC19 at docs/evidence/RSC19-VTP59-latency/summary.json after integration. Continue 75ms, interleave, mixed arrival, phase and independent quality gates using the shared receipt path; use matched inspector v3 once integrated.

RSC19 matched owner build 927a24bf85c2ac29d3dbe0b779063323, evidence commit c96b60638 on node/serving-config-enhancement: 75/50/75ms TP3 IQ8R FP8 cold, six reports all automatic pre/post receipt verified, identical prompt hashes and token counts vs earlier arms, zero preemptions. Short aggregate 279.875/270.253/279.928 t/s, p95 gaps 40.065/165.666/40.445ms; long aggregate 188.774/180.289/180.855 t/s, gaps 41.509/161.219/162.438ms. 75ms stable short results but two long scheduling regimes; no production promotion. Continue mixed arrivals, explicit auto interleave and phase attribution, independent quality and topology qualification; evidence docs/evidence/RSC19-VTP59-matched-owner/summary.json after integration.

VTP60 live trace qualification (node/live-prefill-trace e802b5b34, docs/evidence/VTP60-live-phase/summary.json). Same trace-experiment build control/trace 75ms TP3 cold, 4K512 180.396/151.656 t/s, 512128 278.761/228.477; tracing perturbs timing ~16-18%, not a performance selection. 969 valid pass records per rank; all three ranks exactly reconcile 253 long and 57 short main steps incl phase counts. Long trace mixed model pass median ~222ms at 452 rows, all_reduce and gemm_nt_q largest device interval categories (overlap, not additive wall). Historical matched owner 75ms fast vs slow: pure prefill 9/4, mixed 39/63, waiting max 3/7, earliest TTFT 6.49/2.69s, worst 14.57/15.76s; points to admission regime, not simply kernels. First trace run had runner stop CLI argument error, stopped with signal, retained separately and excluded from passing pair. No provider attribution or quality proof. Automatic-interleave runner preparation tool call blocked; no new interleave experiment yet. Continue arrival/admission controls and mixed workloads on matched enhanced owner.

RV4754: required baseline gate is resolving documented concurrent medium-prompt regression with matched old/new evidence and no unacceptable short/long/quality regression. Preserve FlashNext four-card layout; TP3 27B focus. Existing screens guide selection but single runs and tracing are not qualified. New shared KV mode and exhaustive policy/arrival tuning are follow-ups, not reasons to delay baseline unless a regression is demonstrated.

Matched TP2 regression screen: released 2.2 bc-radiance-2020.24.1 vs candidate 927a24bf85c2ac29d3dbe0b779063323 default vs tuned pipe on/min256 batch4096 interleave off mixed250; three repeats at 2/4 streams, 4096/128 and 512/128; 36 rounds, all exact prompt hashes and per-stream token counts matched, paired native receipts and six portable exports, zero preemptions. Long aggregate 2/4: old49.662/42.315, default50.790/42.763, tuned49.998/56.798; four-stream worst TTFT 11.0095/10.9684/7.7688s, p95 gap28.606/27.465/273.160ms. Short aggregate 148.286/229.950,153.690/232.900,159.728/218.523. Existing inversion persists at default; tuned restores scaling but materially worsens latency, not promote. Need narrower latency-safe scheduling, repeat ABBA and mixed/quality qualification. Evidence baseline-matched-tp2.json. Failed lab setup and record attempts preserved; no GPU lease left. Native record validity still unverified, no formal acceptance claim.

Historical raw recovery: 18 ab-tp2 VTP36 records indexed. Seed1 2020.8.5 W (TP2 wht6) and T (TP3 exact) both have 6 successful eight-stream rounds across two launches. Content-addressed records verified, all eight prompt SHA256 and actual prompt counts match existing 2.3 512/192 screen exactly; all generated counts 192. TP2 historical seed1 median327.1015 t/s/2496.67ms max TTFT/36.5655ms p95 gap vs tuned355.173/2017.25/37.732. TP3 360.8315/2053.02/38.5305 vs tuned350.280/1987.82/40.298. Same cohort inputs now established, but historical engine/model provenance remains legacy and launch spread differs; not independent formal acceptance. Eight older 4K/256 RUM06/RUM09 records recovered with same seed1 prompt family; different output length than 4K/128 or 512, distinct tuning. See historical-cohort-reconciliation.json. No need to repeat all historical short benchmarks.

Focused TP3 governor screen on exact candidate 927a24bf85c2ac29d3dbe0b779063323: 8 streams, 4096/512 then 512/128, 2 reps each, FP8 KV exact wire, pipeline min256, batch2048, interleave off, target75. Three verified fresh-start policies arrival; budget factor1.0/remaining/512 exemption/120000ms; cap2/remaining/512/120000ms. All 12 rounds complete, prompt hashes and actual counts match, paired tuning receipts, six portable exports, zero preemptions, clean stops. Long throughput / max TTFT / p95 event gap: arrival180.3305/15.596s/154.171ms; budget125.088/27.277s/162.6465ms; cap114.489/30.842s/158.798ms. Short throughput281.9865/278.9115/275.8155. End mixed-target misses/steps142/144,347/351,440/444, learned cap64 across all (one geometry quantum); source busy_cap raises effective allowance for participant fairness and backfill, so target is soft and reducing numeric target alone is unlikely to fix latency. Restricted admission worsens 8K-context burst performance; reject these arms, preserve long-profile governor and existing defaults. Evidence tp3-governor-screen.json. No new engine patch or qualified production setting. Next isolate per-mixed-step row/participant floor vs decode cost using native metrics/host scheduler simulation; only change scheduler if a latency-safe fair policy can be proven. Baseline release gate remains open; avoid repeating already covered screens.

## Change Log

- 2026-11-27T09:44:54.101057+00:00 (created-by): Created by agent
- 2026-11-27T09:45:22.314725+00:00 (state-transition): State: pending → in_progress
- 2026-11-27T09:51:15.313461+00:00 (updated-by): Updated: section:notes
- 2026-11-27T09:53:03.850531+00:00 (updated-by): Updated: section:notes
- 2026-11-27T09:59:08.618360+00:00 (updated-by): Updated: section:notes
- 2026-11-27T10:00:47.780286+00:00 (updated-by): Updated: section:notes
- 2026-11-27T10:08:15.236236+00:00 (updated-by): Updated: section:steps
- 2026-11-27T10:08:22.443495+00:00 (updated-by): Updated: section:validation
- 2026-11-27T10:08:29.661326+00:00 (updated-by): Updated: section:acceptance_criteria
- 2026-11-27T10:11:26.253821+00:00 (updated-by): Updated: section:notes
- 2026-11-27T10:17:55.733251+00:00 (updated-by): Updated: section:notes
- 2026-11-27T10:26:28.686503+00:00 (updated-by): Updated: section:notes
- 2026-11-27T10:53:41.556574+00:00 (updated-by): Updated: section:notes
- 2026-11-27T11:08:19.072053+00:00 (updated-by): Updated: section:notes
- 2026-11-27T11:47:13.447881+00:00 (updated-by): Updated: section:notes
- 2026-11-27T11:49:17.864160+00:00 (updated-by): Updated: section:notes
- 2026-11-27T13:31:22.876011+00:00 (updated-by): Updated: section:notes
- 2026-11-27T13:40:02.220162+00:00 (updated-by): Updated: section:notes
- 2026-11-27T14:10:04.816913+00:00 (updated-by): Updated: section:notes
- 2026-11-27T14:18:57.650130+00:00 (updated-by): Updated: section:notes
- 2026-11-27T15:13:21.107632+00:00 (updated-by): Updated: section:notes
- 2026-11-27T20:17:48.556882+00:00 (updated-by): Updated: section:notes
- 2026-11-27T20:33:02.555776+00:00 (updated-by): Updated: section:notes
- 2026-11-27T20:37:47.239009+00:00 (updated-by): Updated: section:notes
