---
id: ENG05
order: 0
plan: tuning-radiance-engine
state: pending
created-at: '2026-11-19T07:30:54.932992+00:00'
breadth: ''
skill: advanced
created-by: agent
priority: P1
work: M
---

# Flash-Next 4-GPU decode: recover the ~3-4% step-time loss since v1.17 and raise decode tok/s

## Description

Flash-Next on flashnext-4gpu (R9700 engine, 2x XTX head helpers, RX 6900 XT tail) decodes at v1.17.0 tok/s only because draft acceptance happens to be higher: engine step time is ~3-4% slower (1 stream 25.9 vs 24.99 ms/step, 6 streams 48.5 vs 47.04 ms/step; 2026-11-19, policy on, docs/evidence/2026-11-19-flashnext-4gpu-regression). Same-session, policy off: release bc-radiance-1170.14.0 (Nov 9) 25.3 ms/step vs main 25.9, so ~2% arrived after Nov 9. Decode tok/s = (1 + depth x acceptance) / step time, so step time and MTP depth/acceptance are the two levers. Owner direction 2026-11-19: plan it; prefill and TTFT are at or above v1.17 with the live policy on and must not regress.

## Steps

1. Attribute the post-Nov 9 step-time loss (no model first): tools/lab/radiance/kbench_ab.py on the decode rows (MoE gate/up/down GEMV rows at decode row counts, attention decode, combine) between the installed release bc-radiance-1170.14.0's build (its install.json build_dir) and main; then a per-stage decode-step profile on each build (attention, routing, hand-off wait, combine, MTP verify). Suspects: v26 topology cutover #1014 (mailbox/combine per routed layer), GEMV launch-sized LDS dispatch #946. Fix or re-tune the stage that grew; one change per PR, ABBA under the gpu lease.
2. MTP depth sweep: num_speculative_tokens 2/3/4 at 1, 6 and 8 streams and at 256K context, several --seed prompts per arm (acceptance follows the text; report ms/step and tok/s separately). If the best depth differs by load, assess a load-dependent depth (engine support first; no hand-picked thresholds: derive from measured acceptance and step cost).
3. Decode-targeted placement: test whether keeping the hottest decode experts on the R9700 (spare VRAM; today the helpers own all 25,078) cuts the per-layer hand-off wait; heat recorded from a decode workload, budgets derived (RAD49), compare with the live policy on.
4. Power profile for decode: stock vs the host's tuned XTX profile (needs its PWR rule file); decode is latency/memory bound.
5. Host kernel (owner window only): stock vs 7.0.13-gfxoff A/B for decode step time (GFX12 GFXOFF wake latency on the R9700 between steps); BHK01 step 7.

## Detailed Solution & Technical Design



## Code Samples & Guidance



## Files



## Validation

Every arm: tools/lab/radiance/flashnext-parity.sh measure with the policy on (FLASHNEXT_PARITY_WARMUP_S as used for the comparison), at least 2 launches, compare against docs/reference/engines/radiance/baselines/flashnext-4gpu-parity.json and the current-main baseline; decode1/decode6/sess8/long8 ms_per_step and tok/s reported separately from prefill/TTFT; kbench rows per build for step 1.

## Effort & Risk



## Standards



## Acceptance Criteria

Decode ms/step at 1 and 6 streams within the v1.17.0 baseline allowance (or better), or each remaining gap attributed to a named change with its trade-off; any depth or placement change shows tok/s gain over >= 2 launches with no prefill/TTFT regression against the baseline.

## Notes



## Change Log

- 2026-11-19T07:30:54.932992+00:00 (created-by): Created by agent
- 2026-11-19T07:31:25.516109+00:00 (updated-by): Updated: section:description, section:steps, section:validation, section:acceptance_criteria
- 2026-11-19T07:34:30.174287+00:00 (updated-by): Updated: section:steps
