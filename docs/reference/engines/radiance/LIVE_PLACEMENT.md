# Live expert placement: what is built, how it is switched on, how to check it, how to restore it

Operator guide. The roadmap and per-item state are in
[the RHP plan group](../../../planning/active/patching-radiance-heat-placement/README.md); this page is the current state of the
running system and the checks to make before you trust a measurement.

## What is built (on main)

- **Heat recording** in pinned host memory and a **drift watch** (RHP foundation, #500-#502); start-up placement ordered by heat.
- **Live migration of helper experts** while serving: placement-epoch protocol, per-step admission and safe drain, live owner-map
  generations on the engine GPU, live helper slot migration, combine epoch checks, forced-move harness (RHP11 slices C-G:
  #542, #557, #558, #572).
- **Live placement policy** (RHP slices H-J, #597): the target map is the static placer's plan for the live heat.
- **Policy settings derived from calibration** (RHP16, #666 #693 #708): margin, minimum samples, cadence, move cap, residency and the effect
  gate come from measured quantities, never from literals. The policy is **off until the layout is calibrated**.
- **Measured effect** (RHP16.md; flashnext-4gpu, policy on versus off after the derived warm-up). Latest, on a freshly calibrated record
  (2026-11-10, ABBA, arm state checked from each manifest): cold 27k TTFT 7.4 s against 10.9 s (-32%); document prefill flat; 6-stream decode
  flat to +13%; 1-stream decode flat on average over six prompts (single prompts move -20% to +24% through speculative-decoding acceptance:
  moved experts change the greedy text). Per cold layer call the tail carries 1,258 routes instead of 2,874 and is no longer the straggler
  (5.5 ms against the engine's 6.8 ms). After a documents-only period the first cold prompt costs the static time and the policy is at
  steady state by the 3rd-4th prompt (~30-40 s). Quality (forced-churn KL): mean +0.008, p99 +0.23 (over the 0.2 limit: the known tail
  accuracy effect), top-1 -0.6 points.

Not built yet: the device-speed and phase-aware solver (RHP17, RHP04 stage 3), the single allocator (RHP13), request/time reevaluation (RHP07).

## How it is switched on

The policy is a field of the serving manifest: `spec.handoff.placement.live_policy`. bc-serve fills it when
`derive_manifest_live_policy` finds a record in the calibration store (`${BC_WORK_ROOT}/live-policy-calibration.toml`) for this
configuration's **calibration key** and the record is complete. Otherwise the field is absent and the policy is off. The launcher
reports ON (with measured cadence) or OFF (with the missing-record/unsupported-layout reason derived from manifest provenance).
A helper-free start is NOT_APPLICABLE. RHP18 tracks the remaining per-epoch runtime/degraded status and audit identity.

The calibration key is `sha256` of the plan key inputs without `engine_build` and `cache_version`
(`calibration_key` in `engines/radiance/components/serving/plan_cache.cpp`). The inputs are `card`, `engine`, `engine_args`,
`helper_budgets_mib`, `layout` and `model`. **Anything that changes one of them orphans the record**: a different layout, a changed
helper budget (`--budget`), different engine arguments. #790 added `helper_budgets_mib` on 2026-11-09, which orphaned the
flashnext-4gpu record seeded on 2026-11-07: from that point no production start of the layout carried a live policy until it was
re-calibrated.

**Instrumented runs.** Any start with `--experiment` (lane trace, `profile_ops`, `kv_resident`, ...) gets the *lab* service profile, and the
policy is derived only for the *production* profile: such a run never carries a policy unless one is passed explicitly with
`--live-policy FILE` (the derived `live_policy` object of a production manifest is a valid file). The experiment file's hash is also part of
the calibration key.

## RHP20 card and layout policy eligibility (2026-11-10 card-free review)

A startup ON state reports only startup derivation, **not** that every owner-generation transition is valid. The helper independently
refuses a manually supplied explicit policy on head-only, single-helper or multi-rank TP layouts, even if a lab experiment would
otherwise bypass derived policy selection. Only source/target slot eligibility plus the receiving helper's authoritative checks
can permit a transaction. gfx1201 and gfx1100 support w4nl64 and fp8_e4m3_block, but gfx1030 supports w4nl64 only.

| Layout | Helper roles | Live policy | Calibrated key / run | Hardware migration and quality |
| --- | --- | --- | --- | --- |
| flashnext-4gpu | gfx1100 head/head, gfx1030 tail; gfx1201 engine | Topology eligible, ON only with current-key calibration | Historical RHP16 key; current key recheck needed | 131 paced forced swaps, zero aborts (RHP06); current 1,000-epoch/format/KLD and per-card receipts pending |
| flashnext-2xtx-rdna2-tail | gfx1100 head/head, gfx1030 tail; gfx1201 engine | Topology eligible, not yet calibrated | None verified under current key | Cross-arch w4nl64 / 2 GiB tail KLD pending |
| flashnext-2xtx-rdna2-tail-combine | Same, combine enabled | Topology eligible, not yet calibrated | None verified | Combine/migration/KLD pending |
| flashnext-2xtx-rdna2-head | gfx1100 head/head, gfx1030 head | OFF; no tail, #805 | Not applicable | Static placement and forced-only safety pending |
| flashnext-2xtx / flashnext-2xtx-combine | gfx1100 head/head | OFF; no tail, #805 | Not applicable | False last-XTX-tail must never occur; baseline pending |
| flashnext-tp2xtx-r9700 | gfx1201 single helper, 2x gfx1100 engine TP | OFF; one helper plus TP | Not applicable | TP rank generation and baseline pending |
| flashnext-tp2xtx | gfx1201 head, gfx1030 tail; TP engine | OFF; unverified TP | Not applicable until TP validated | Both rank ACKs, format/KLD and epoch trials pending |
| flashnext-r9700 | gfx1201 engine, no helpers | NOT_APPLICABLE | Not applicable | Engine-only residency; known separate VRAM margin problem |
| qwen27b-2xtx | 2x gfx1100 dense engine, no helpers | NOT_APPLICABLE | Not applicable | No remote MoE helper migration |

The lease-holder acceptance commands and precise per-cell gates are in
docs/planning/active/patching-radiance-heat-placement/RHP20.md. Do not mark a layout verified without a recorded manifest,
key, helper transaction log, owner generation, RHP12 timing and numerical/KL verdict. When kernel architectures differ,
bit identity is not the criterion: use the documented numeric quality gates.

## How to check it is on (do this before any placement-sensitive measurement)

```
python3 - "$(ls -t ${BC_WORK_ROOT}/flashnext/manifest-*.json | head -1)" <<'PY'
import json,sys
p=json.load(open(sys.argv[1]))['spec']['handoff']['placement']
print(json.dumps(p.get('live_policy'), indent=1) if 'live_policy' in p else 'live_policy ABSENT: the policy is off')
PY
```

An A/B that moves the store aside (to measure "policy off") must say so in its result. Runs that did this by habit measured the static
start-up layout, not production.

## Measuring policy on versus off

Both arms get the same warm-up traffic (derived: `(min_samples + 1) * cadence + min_residency`). The off arm moves the calibration store
aside; each arm's manifest is read and the arm is marked INVALID if its state differs from its label (runs/policy-onoff-ab/run.sh is the
template). Report engine decode tok/s, client aggregate and ms/step separately; judge 1-stream decode over several prompts, not one.
An instrumented run (`--experiment`) needs `--live-policy FILE`. The launcher prints `live placement: ON (...)` or `OFF -- reason`
at start; a missing record, a layout the planner is not built for (heads followed by exactly one tail), and the lab profile are all
named there. The default heat period is 1 s (a 60 s period starved the policy). Servers the launcher starts inside a lease are still listed
by `agent-lease status` as outside any lease (they are detached).

## How to restore it after a key change

Re-calibrate the layout (lab only, takes the `gpu` lease): `launch-flashnext-best.sh start --build DIR --port N --calibrate
--record-period 1 --calibration-prompts-a FILE --calibration-prompts-b FILE --live-policy policy.json`. The launcher ingests the
record atomically into the store and a plain start then carries the derived `live_policy`. Recipe and prompt files:
`runs/rhp16-calib9` (filler 27k prompts plus 44k document prompts, both workload orders). Do not edit the record's key by hand.

## When measuring

- Warm the policy first: the derived warm-up is `(min_samples + 1) * cadence` of mixed filler and document traffic (about 451 s with
  the 2026-11-07 settings, 18 s with the 2026-11-10 ones); both arms of an on/off A/B get the same warm-up (RHP16.md).
- A transaction takes 1.5-2.0 s (p50/p95) and overlaps a decode step; two on-arm rounds in the long-context run were slowed by one.
- Cold-prompt tail load measured with the policy off (HL04: tail 11.36 ms and 2,874 routes against XTX 3.5-3.7 ms per layer call) is the
  static layout; re-measure policy-on before using it as the tail's cost.
