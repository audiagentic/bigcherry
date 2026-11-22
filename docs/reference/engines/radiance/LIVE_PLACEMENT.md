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

## RHP22 numerical quality disposition (2026-11-22)

**FAIL: four-GPU forced-churn KL p99 is +0.2341 against the +0.2000
limit.** The static/identical-repeat control is bit-identical; this is
not measurement noise and has **not been waived**. The historical forced
test committed swaps *before* its 15–17 s scoring pass, so its result
characterizes one changed fixed placement. It does **not** numerically
qualify the warmed ordinary live-policy placement or mid-scoring
ownership changes. The different two-XTX+RDNA2-tail arm (+0.112 p99)
passes its own observed numerical gate but is not a substitute.

A calibrated ON manifest means only that a layout-specific policy was
selected, **not** that its numerical quality passed. Under the existing
single ownership plan, RHP16 must freeze a **measured** warmed-policy
final map, compare it with a same-build static KL control using the
existing scorer, and receive explicit mid-scoring move proofs from
RHP06/RHP20 before declaring this layout's quality gate passed.
Do not treat the historical production enabled state as a fresh
approval; this documentation decision does not change running
configuration. The exact data and gate are recorded in
[the RHP22 disposition](../../../evidence/2026-11-22-rhp22-quality-disposition/README.md).

Not built yet: the device-speed and phase-aware solver (RHP17, RHP04 stage 3), the single allocator (RHP13), request/time reevaluation (RHP07).

## How it is switched on

The policy is a field of the serving manifest: `spec.handoff.placement.live_policy`. bc-serve fills it when
`derive_manifest_live_policy` finds a record in the calibration store (`${BC_WORK_ROOT}/live-policy-calibration.toml`) for this
configuration's **calibration key** and the record is complete. Otherwise the field is absent and the policy is off. The native manifest and `bc-serve explain CONFIG` expose the selection and reason; a startup ON is not proof of effective migration.
A helper-free start is NOT_APPLICABLE. RHP18 tracks the remaining per-epoch runtime/degraded status and audit identity.

The **policy-calibration key v2** is separate from the memory-plan key. Its SHA-256 input scope
includes the model-container identity, layout, objective, engine and helper GPU identities/architectures/VRAM,
helper roles/formats/kernels, heat-histogram period, named power profile, helper execution settings
(including combine/lanes), and remote service geometry. It **excludes** the engine/plugin build hash,
host pool, helper VRAM budgets, memory-plan identity, engine flags and local path names.
The move-byte cap is re-derived from the current helper budgets and expert stride at each start;
drift, lane service times and noise remain measured quantities, never guessed constants.

MRC04 v1 records retain `calibration_inputs` (the exact scope),
`calibration_key_fields` (the platform's declared narrow inputs), `source_plan_key`
(the full strict measured-plan identity), `source_engine_build`,
`source_revision`, `run_id`, `host_facts_digest` and `seeded` as provenance.
A source revision is recorded as `source_slice:<hash>` when immutable build
metadata is available, otherwise `unavailable` is recorded explicitly.
Records written after #1169 with `calibration_inputs`, `source_plan_key` and
`source_engine_build` remain **current and activatable** when the declared scope
hash and measured quantities validate, even if `record_schema_version` is absent.
The native reader maps these rows to v1 activation in memory only; it does not
invent the later revision/run/host-facts provenance or rewrite the TOML store.
Only genuinely hash-only rows lacking `calibration_inputs` are legacy and
receive `legacy_record_schema` as their non-activation reason.
The platform's `bigcherry calibration list|show|validate` reads the native
TOML store; the production reader/writer remain native C++. The reader verifies that
the scope hashes to the record key; production matches the current model/cards/layout and
service inputs before enabling. Changing model, physical cards/architectures, layout, histogram
period, power profile or helper execution behavior still requires compatible evidence.

**Calibration producer status (2026-11-23).** A normal
`bc-serve serve CONFIG` start reads a complete matching record from the
calibration store through `derive_manifest_live_policy`, or leaves the
policy off. `bc-serve explain CONFIG` reports selection without a start.
The standalone `bc-serve calibration-ingest` command can validate a
historically captured `RHP16_CALIBRATION_JSON` helper log against its
original lab-profile calibration manifest and append a record. It does
not drive workloads or obtain a fresh record.

The old foreground `serve --calibration-run` start and lab launcher were
removed. The standalone manifest writer still parses `--calibration-run`,
but that flag is **not** a supported session start: no current main-branch
workflow produces the required lab manifest and settled helper log.
Existing complete records remain usable; do not describe an uncalibrated
layout as self-calibrating. PRs #1375 and #1384 are implementing native
calibration slices, but are not yet an integrated operator procedure.
Recheck the actual CLI and host/GPU gates before documenting any replacement.

**Startup guards (a production start with the policy on).** Right after the start canary `bc-serve` checks the engine's
histogram stream and the worker's placement control. Two states of a fresh start are healthy and accepted, nothing else is
loosened: the engine's first snapshot may still hold no route counts (the check waits up to the declared histogram period or
the policy's allowed heat age, never for any other failure), and the worker's boot placement has transaction 0 with an
empty journal, because a transaction is numbered only when the worker begins one (it must be active, fault-free, with both
journal slots empty). Every refusal names the failing check.

A second production start of an unchanged configuration replays the first start's stored manifest (RSC02). The policy a
production manifest carries is derived by the start, not configured, so it is not part of the replay's execution
projection (the replay key binds the calibration store's identity); a lab manifest's explicit `--live-policy` is, and any
other configuration difference still refuses the replay.

**Legacy migration caveat:** historical records in `live-policy-calibration.toml` have only
the old hash, layout, objective and measurements; they lack model and helper-card provenance.
They remain readable for audit, but are not automatically promoted across budget or card changes.
To reuse the observations without another GPU calibration, an operator must establish an
authenticated matching historic strict calibration manifest and card inventory and migrate
the record with full provenance. Without those inputs, fail closed rather than trust a
layout-only match or hand-rename a hash. A missing record is an OFF startup decision with
a conspicuous warning, never a serving-start gate.

**Instrumented runs.** Any start with `--experiment` (lane trace, `profile_ops`, `kv_resident`, ...) gets the *lab* service profile, and the
policy is derived only for the *production* profile: such a run never carries a policy unless one is passed explicitly with
`--live-policy FILE` (the derived `live_policy` object of a production manifest is a valid file).
Lab experiment files are outside the production policy-calibration scope; explicit policy evidence is still
required for every instrumented lab run.

## RHP20 card and layout policy eligibility (2026-11-10 card-free review)

A startup ON state reports only startup derivation, **not** that every owner-generation transition is valid. The helper independently
refuses a manually supplied explicit policy on head-only, single-helper or multi-rank TP layouts, even if a lab experiment would
otherwise bypass derived policy selection. Only source/target slot eligibility plus the receiving helper's authoritative checks
can permit a transaction. gfx1201 and gfx1100 support w4nl64 and fp8_e4m3_block, but gfx1030 supports w4nl64 only.

| Layout | Helper roles | Live policy | Calibrated key / run | Hardware migration and quality |
| --- | --- | --- | --- | --- |
| flashnext-4gpu | gfx1100 head/head, gfx1030 tail; gfx1201 engine | Topology eligible, ON only with current-key calibration | Historical RHP16 key; current key recheck needed | **Forced-churn quality FAIL** (KL p99 +0.2341 vs +0.2); warmed-policy frozen-map and mid-scoring receipts **not yet accepted**; 131 paced swaps/zero aborts do not supersede the failed KL gate |
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
Instrumented lab runs need explicit policy evidence; do not rely on retired launcher flags. Inspect `spec.handoff.placement.live_policy` in the native-owned immutable manifest, and use the declared typed config. The default heat period was 1 s in historical runs (60 s starved the policy). The lease must cover the native process-owner lifetime, not just benchmark traffic.

## How to restore it after a key change

**Native calibration heat source (2026-11-19):** The serving engine writes the manifest-declared
spec.handoff.telemetry.histogram_file every period_s seconds. The helper coordinator currently uses
HistogramFileSource on <prefix>.rank0, not the per-step credited-heat ring. Those file deltas carry
complete_observations=false: the .bcproof sidecar proves publication generation and file identity,
**not** coherent per-step observation coverage. Consequently an explicit lab policy with
accept_unwatermarked_heat=false is incompatible with this source and the native owner now refuses it
before launching a manifest writer. For file-based calibration, the operator must explicitly set
accept_unwatermarked_heat=true in the lab --live-policy JSON, preserving the policy's provenance
and other safety checks; the owner never silently overrides false. This is not a claim that a
calibration will converge or that the file is an equivalent replacement for a credited stream.

The native owner also refuses absent --calibration-store paths by name; create an empty ordinary
file at that path before starting a fresh calibration. The heat-input watchdog is distinct from
the full four-phase convergence deadline: without two accepted deltas its bound is
(min_samples + 1) * histogram_period + max(cadence, max_heat_age) (terms in seconds), so
the recorded policy (1000, 1 s, 60 s, 120 s) fails without heat in 1,121 s instead of
borrowing a multi-hour four-phase policy horizon. The error reports BC_CFG_CALIBRATION_INVALID,
heat_watermarked, the accepted delta count and the derived limit. The archived launcher passed
--record-heat PREFIX explicitly; native serving instead derives the engine's histogram
path from the immutable manifest and binds publication to the launch nonce. The helper's file
consumer does not interpret .bcproof as an observation watermark. Radiance 2.x migration should
retain this typed source distinction rather than hard-code upstream flags.

Re-calibrate only through the supported native `bc-serve` typed calibration workflow under the GPU lease. The prior CLI recipe belongs to archived pre-S16 evidence (`runs/rhp16-calib9`); do not replay its shell flags as a new native mode or edit the calibration key by hand. See the current RSC12 configuration and S10 producer/consumer contract.

## When measuring

- Warm the policy first: the derived warm-up is `(min_samples + 1) * cadence` of mixed filler and document traffic (about 451 s with
  the 2026-11-07 settings, 18 s with the 2026-11-10 ones); both arms of an on/off A/B get the same warm-up (RHP16.md).
- A transaction takes 1.5-2.0 s (p50/p95) and overlaps a decode step; two on-arm rounds in the long-context run were slowed by one.
- Cold-prompt tail load measured with the policy off (HL04: tail 11.36 ms and 2,874 routes against XTX 3.5-3.7 ms per layer call) is the
  static layout; re-measure policy-on before using it as the tail's cost.

## RHP20 forced-migration stress, 2026-11-11 (flashnext-2xtx-rdna2-tail, -tail-combine)

Build `daf793f1.../a29e62ad...`, lab experiment with 1,002 scheduled AUTO_SWAP proposals, concurrent 6-stream decode (MTP 3) and `prefill-doc-44k@v1`. Runs: `runs/rhp20-tail-forced-1000`, `runs/rhp20-tail-combine-forced-1000`.

| Cell | Helper 0 attempted / committed / aborted / inflight | Decode, 6 streams | 44k doc prefill |
|---|---|---|---|
| flashnext-2xtx-rdna2-tail | 1002 / 1002 / 0 / 0 | 245.4 tok/s aggregate | 2,806 tok/s |
| flashnext-2xtx-rdna2-tail-combine | 1002 / 1002 / 0 / 0 (combine steps 14,476) | 251.3 tok/s aggregate | not recorded here |

The swap-stress gate (>=1,000 committed, zero aborts, zero inflight) passed on both. The runner verdict is FAIL on both because the required per-expert same-ISA identity record cannot be produced: no per-expert output probe exists (RHP20 README). Not yet run: that probe, the cross-ISA KLD arms (`kld-config`), and zero-owner destinations. A forced-churn manifest has no live policy, so the launcher reports OFF "lab experiment" (#851); these runs are not live-policy or calibration evidence.

### RHP20 cross-ISA KLD, flashnext-2xtx-rdna2-tail (2026-11-11)

`runs/rhp20-tail-kld` (one pair, 8,240 positions, 4-document corpus, reference `rad116-kld/reference`, same build as the stress cells). Static control vs forced-churn arm; 132 swaps committed while scoring (0 aborted, 0 in flight). Diagnostic evidence only: source provenance is not verified.

| Metric | static | forced | delta | limit |
|---|---|---|---|---|
| KL mean | 0.0620 | 0.0688 | +0.0069 | +0.02 |
| KL median | 0.0162 | 0.0155 | -0.0007 | +0.01 |
| KL p99 | 0.672 | 0.785 | +0.112 | +0.2 |
| KL p99.9 | 2.566 | 2.861 | +0.295 | +0.5 |
| top-1 agreement | 0.8843 | 0.8851 | +0.07 pt | -1 pt |
| ppl ratio | 1.0021 | 0.9944 | | +0.05 |

All thresholds pass. The tail-combine KLD (`runs/rhp20-tail-combine-kld`) reproduced these numbers exactly because scoring batches are 64 rows and the combine engages only on 256+ row lanes (helper log: combine steps 0), so it is **not** combine evidence. Retries with `engine.max_num_batched_tokens` 512 and 256 stopped at engine start out of memory (engine card has no room for the larger activation arena). Combine numerics are therefore untested; the combine path itself was exercised only by the swap-stress cell (14,476 combine steps). Closing this needs a combine KLD config that frees engine VRAM (for example a smaller context) and shows a nonzero combine-step count.
