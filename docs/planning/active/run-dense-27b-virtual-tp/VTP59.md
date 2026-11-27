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
