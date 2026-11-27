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

Reuse existing interleave, busy-prefill chunk, mixed-step target and governor; isolate each on the released runtime, then test finalists using matched repeats and mixed arrivals. Inspect scheduler plans before code changes. Keep VTP42 long policy untouched unless independently qualified. Carry verified changes to candidate.

## Detailed Solution & Technical Design



## Code Samples & Guidance



## Files



## Validation

Identical model/build/topology/power/input hashes; 1/2/4 streams at 512 and 4096 prompt targets, 128 and 512 output; per-request TTFT, token count, gap tails, decode overlap, actual prefill and decode time, no preemption, deterministic outputs and independent quality gate. Repeated starts, ABBA for finalist. Check short arrival beside long prefill.

## Effort & Risk



## Standards



## Acceptance Criteria

Demonstrate improved four-stream long-prompt throughput and decode overlap versus baseline with no material short-stream or mixed-arrival latency/quality regression; explicit evidence and qualified configuration or scheduler fix merged before bump promotion. No blanket disabling interleave.

## Notes

Regression history: pre-2.2 default interleave=0; 2.2 defaults to auto:1 when omitted (documented upstream review 2026-11-18 and RUM08). FlashNext was explicitly protected but standard Qwen profiles were not. Earlier Qwen model was MXFP4, so direct historical speed parity not established. 2.2 IQ8R TP2 four-stream 4K/128 screen: interleave 0 54.879 output/410.158 decode tok/s, max TTFT 8068 ms, p95 gap 974 ms; fixed 2 54.019/334.455, 8164/947 ms; fixed 2 + mixed target 100 ms 41.836/193.816, 10964/139 ms, 512 prompt throughput falls 233.136 to 173.321. auto:0.25 50.014/219.278, 9122/947 ms. None qualified across all metrics. Existing mixed target controls aggregate prefill allowance during active decode; no new competing controller before controlled sweep. Preserve effective scheduler default drift audit as explicit bump gate. See docs/evidence/VTP59-scheduler-20261127/summary.json.

Owner 2026-11-27: extend concurrency to eight streams. Released TP2 defaults to six sessions; compare unchanged production cap separately from experimental eight-session cap, verify memory admission and active sequence counts, and use eight-session TP3 as topology control. Sweep 512/4096 prompts and 128/512 output, then validate mixed arrivals and quality. 250 ms mixed target + interleave 0 at four streams: 48.394 aggregate, 406.638 engine decode tok/s, 9.307 s max TTFT, 247 ms p95 event gap; promising compromise but not qualified.

Eight-session TP2 admitted 65536 tokens, 2.24 GiB KV/card and no preemptions. Same eight-session setting: 4096/128 auto:1 48.618 output/187.655 decode tok/s, 20.052 s max TTFT, 29.941 ms p95 gap; interleave 0 + mixed target 250 ms 50.435/640.001, 18.624 s, 265.936 ms, 23.27 tokens/step vs 4.90. 512/128 auto 262.466/554.707 vs tuned 252.609/654.547, so short end-to-end throughput drops 3.8%. These are screening measurements, not a production qualification. Larger batch budget 4096 with same mixed target isolates prefill throughput; retain 2048 and eight-session controls. Require repeat starts and quality, long/short arrivals, six-session production cap control, TP3 and candidate before final selection.

Completed eight-stream screens (3 reps each): 4096/512 auto 126.244 aggregate, 243.853 decode tok/s, 27.014 s max TTFT vs 0+250 ms 156.078/565.703, 18.951 s, 205 ms p95 gap. Raising batch budget to 4096 with 0+250 ms for 4096/128 gives 52.133/642.389, 18.025 s, 266 ms p95 gap. All owner exits clean; no preemptions in eight-stream screening. 19 benchmark report summaries retained; raw local evidence archived. Prefill remains limiting; next isolate pipelined prefill on supported TP2 and TP3 (already explicitly enabled on TP3), measure pure prefill and mixed-step timing separately, tune admission and latency using measured costs, verify quality and repeated starts before promotion. No production configuration changed.

## Change Log

- 2026-11-27T09:44:54.101057+00:00 (created-by): Created by agent
- 2026-11-27T09:45:22.314725+00:00 (state-transition): State: pending → in_progress
- 2026-11-27T09:51:15.313461+00:00 (updated-by): Updated: section:notes
- 2026-11-27T09:53:03.850531+00:00 (updated-by): Updated: section:notes
- 2026-11-27T09:59:08.618360+00:00 (updated-by): Updated: section:notes
- 2026-11-27T10:00:47.780286+00:00 (updated-by): Updated: section:notes
