---
id: RHP17
order: 0
plan: patching-radiance-heat-placement
state: pending
created-at: '2026-11-10T00:10:41.824087+00:00'
breadth: ''
skill: advanced
created-by: agent
priority: P2
work: M
---

# Device-speed and phase-aware expert placement: minimise each layer's slowest-device finish time (offline simulation first)

## Description

The live planner (worker_heat.h propose) ranks experts by heat and minimises the tail's heat share; devices have only byte budgets, no speed or PCIe model, so it reported no_improvement_or_margin 30/30 on cold prompts while the 6900 XT tail was ~73% busy. Both GPT reviews: optimise the per-layer critical path C(expert, device, phase) = dispatch + transfer + compute + combine, consider a cold-latency and a sustained-throughput profile, and replicate a few hot tail experts only when it pays. A smaller tail is not the answer within today's RAM (7 GiB tail: cold TTFT -26% but 8-session fill -21%).

## Steps

1. Fit per-device stage costs by rows per expert (captures + helper stage logs; tail vs XTX ratios differ per stage).
2. Offline simulator over captured requests and heat: current placement vs per-layer-max objective within the same byte budgets.
3. Only if the simulation shows recoverable time: extend the planner objective (managed moe-remote change), live A/B on cold, doc and long-fill workloads.

GPT seed (gateway reviewer, 2026-11-10; full text runs/rad49-gpt-review/seed/):

Reproduce RHP16 policy-on and policy-off results with known calibration state and identical warmup.Capture full per-layer stage timelines for cold, document, eight-session long prefill and one/six-stream decode.Record rows-per-expert and unique activation/result rows for every relevant device.Fit per-device stage-cost curves using disjoint training and held-out captures.Validate the simulated baseline against measured critical-path and request timing.Run the existing heat-based planner in replay mode using the same snapshots.Add a separate speed-aware offline solver with identical device budgets and eligibility constraints.Evaluate predicted latency improvement and moved bytes, including migration amortisation.Repeat with cold-only, sustained-prefill and mixed-workload traces; report the trade-offs.If no credible improvement remains after migration cost and model uncertainty, close the item with the simulator and measurements.Otherwise, implement the new objective behind an opt-in planner setting, reusing RHP16's transaction and effect-gate machinery.Prove safe mid-scoring ownership transitions before a live rollout.Perform full A/B with quality and sustained-load gates.

## Detailed Solution & Technical Design

GPT seed (gateway reviewer, 2026-11-10; full text runs/rad49-gpt-review/seed/):

Implement an offline stage-cost model and event-based placement simulator before modifying `worker_heat::propose()`.Stage 1: Fit measured cost functions.For every device, fit separate gate/up, down, activation preparation, reduction/write and transfer costs. Inputs should include:\\\[ \\begin{aligned} C\_d = f\_d(&\\text{phase},\\ \\text{routed rows}, \\text{rows per expert},\\ &\\text{unique tokens},\\ \\text{bytes}, \\text{concurrent load}) \\end{aligned} \\\]Use the measured kernels and link behaviour, not a constant FLOP/s ratio between XTX and 6900 XT. Cold and prefill requests have substantially different per-expert occupancy and weight reuse.The model should also represent engine-local experts and the host expert pool, although the first solver should hold engine-local ownership and host residency fixed.Stage 2: Build a per-layer critical-path DAG.A simple approximation is:\\\[ T\_l \\approx \\max\\left( T\_{\\rm engine}, T\_{\\rm XTX0}, T\_{\\rm XTX1}, T\_{\\rm tail}, T\_{\\rm combine} \\right)+T\_{\\rm final\\ add}. \\\]But this is only an initial model. The real implementation should include RAD42's overlapping 256-row chunks, since combine and engine-add work may overlap.Conceptual simulator:def evaluate(owners, trace, curves): latency = 0.0 for step in trace: for layer in step.layers: dag = build\_layer\_events( layer=layer, owners=owners, stage\_costs=curves, chunk\_overlap=True, link\_contention=True, ) latency += critical\_path(dag) return latency`build_layer_events` is a proposed abstraction, not an existing API.Stage 3: Constrained placement optimisation.For each expert eee, device ddd, define ownership variable xe,dx\_{e,d}xe,d​. Require:\\\[ \\sum\_d x\_{e,d}=1 \\\]for every currently helper-owned expert, and:\\\[ \\sum\_e x\_{e,d},\\operatorname{bytes}(e) \\leq B\_d. \\\]Preserve the current eligibility matrix, source-encoding compatibility, placement generation and all transaction constraints.Rather than inventing weights that combine cold latency, throughput and decode, initially optimise cold latency subject to explicit no-regression constraints for document prefill, long-fill and decode. Also report the Pareto frontier, so a cold-latency-only alternative can be considered separately without becoming a hidden production policy.Stage 4: Account for migration economics.RHP16 measured multi-gigabyte migrations taking seconds, with moves constrained by drift, residence time, eligibility and calibration. Therefore, a speed-aware move must satisfy:\\\[ \\text{expected benefit over useful residency}\\text{migration cost + uncertainty margin}. \\\]Derive the useful residency horizon from RHP16's observed heat drift. Use its existing calibrated cadence, margin, sample coverage and effect gate; do not introduce another set of hand-picked hysteresis values.Stage 5: Treat replication separately.DeepSeek EPLB demonstrates expert replication, but BigCherry currently has an exact-once ownership and routing contract. Duplicating an expert requires additional VRAM and a separate rule determining which replica handles a route.That is a distinct protocol change. It should not be included in the initial RHP17 implementation.

## Code Samples & Guidance



## Files



## Validation

Simulation predicts and a live A/B confirms a cold-prompt TTFT gain with no loss on doc prefill, 8-session long fill or decode; KL unchanged (ownership changes alter reduction grouping, so gate it).

GPT seed (gateway reviewer, 2026-11-10; full text runs/rad49-gpt-review/seed/):

Offline model gate: On held-out captures, target median per-layer completion-time prediction error no greater than 10% and p95 error no greater than 20%, while reporting errors separately for tail-critical and XTX-critical layers. These are proposed model-acceptance criteria, not already achieved measurements.Require the simulator's predicted benefit to exceed both its uncertainty and amortised migration cost. A minimum 2% predicted end-to-end cold-latency gain with positive confidence margin is a reasonable threshold for proceeding to live investigation.Live performance gate: Compare with the production-derived RHP16 policy, not an artificially disadvantaged static placement. Cold TTFT should improve by at least 2% in repeated matched runs; document prefill, eight-session long-fill throughput and post-fill performance must not regress beyond 2%, and one/six-stream decode step latency must remain within that bound. p99 latency regression must be below 5%. Record migrations, moved bytes, seconds spent moving, live stalls, aborts and reversions.Correctness and KL: Placement changes can alter where expert arithmetic is performed and how per-token results are grouped, so bit-identical model output cannot be assumed. Preserve exact-once routing, slot ownership and transaction generations. Evaluate all six helper-quality metrics using RAD39's reference-noise treatment, while separately reporting any new deterministic KL outliers. Mid-scoring movement correctness remains a necessary gate after the RHP16 forced-churn evidence, which did not conclusively exercise mid-scoring transitions.

## Effort & Risk

GPT seed (gateway reviewer, 2026-11-10; full text runs/rad49-gpt-review/seed/):

M for offline modelling; L for live policy integration; higher again for replication. The main risks are bad cost-model generalisation, overfitting to synthetic cold prompts, excessive migrations, host-pool regressions, quality changes from new ownership and conflict with the existing RHP16 controller. Fail closed to the current RHP16 policy whenever calibration is incomplete, the prediction is uncertain or the proposed map violates any existing safety constraint.

## Standards



## Acceptance Criteria



## Notes

Evidence: runs/rad49-4gpu-coldtail, rad49-4gpu-tailsweep, rad49-4gpu-longfill; findings runs/rad49-findings-20261109.md. Refs: HybriMoE sec. IV-B, DeepSeek EPLB balanced_packing/replicate_experts, MoE-Infinity phase policies.

GPT seed (gateway reviewer, 2026-11-10; full text runs/rad49-gpt-review/seed/):

- Assessment: The modelling premise is sound, but its available upside is substantially constrained.The current `worker_heat.h` planner evaluates expert heat and produces a target placement using the static head/tail placement strategy. Its `Device` structure has a byte budget but no measured GPU compute-speed, transfer-time or combine-cost curve. This is verified in `worker_heat.h:46–75, 125–149`.However, the current planner is not completely speed-agnostic: `worker_share.h` already encodes different head shares and coldest-first tail allocation. What is missing is an explicit latency objective and per-layer device completion model.There are three important pieces of contradictory or limiting evidence:The RAD49 cold experiment reported `no_improvement_or_margin` in 30/30 evaluations. That does not establish optimality, because it evaluates the existing heat-share objective.RHP16's derived live policy previously moved experts from the tail to XTXs and improved filler TTFT by approximately 33%, with document prefill about 0.9% slower. Therefore, placement already delivers meaningful benefits when the workload heat shifts.A smaller tail funded by moving experts into host RAM improved cold TTFT 26% but reduced eight-session throughput 21%. This establishes that reducing tail responsibility at the cost of frequent host expert reads is not an acceptable default.RHP17 should therefore test whether a speed-aware objective improves upon the already-calibrated RHP16 policy, within unchanged memory budgets. It should not compare only against a static or uncalibrated baseline.
- References: [HybriMoE, Section IV-B, hybrid scheduling and makespan minimisation](https://arxiv.org/abs/2504.05897); [DeepSeek EPLB `eplb.py`, `balanced_packing` and `replicate_experts`](https://github.com/deepseek-ai/EPLB/blob/main/eplb.py); [MoE-Infinity phase-specific expert policies](https://github.com/EfficientMoE/MoE-Infinity/blob/main/docs/configuration.md); [MoE-Infinity original paper, request-level expert activation tracing](https://arxiv.org/abs/2401.14361). These support the scheduling approach, not a specific speedup prediction for this four-card system.  Local: `worker_heat.h:46–75, 125–285`; `worker_share.h:60–115`; `worker_policy_driver.h:190–285, 620–710`; `worker_policy_effect.h`; `worker_policy_replay.h`; RHP16; RAD49 findings.-
- Dependencies / ordering: Begin offline modelling after the shared critical-path trace and preferably after TK01+TK02, so kernel costs are calibrated to the new production form. Reuse RHP16's replay and live-policy machinery. Split the offline solver from live integration, and defer replication to its own future plan with explicit memory and exact-once protocol gates.

## Change Log

- 2026-11-10T00:10:41.824087+00:00 (created-by): Created by agent
