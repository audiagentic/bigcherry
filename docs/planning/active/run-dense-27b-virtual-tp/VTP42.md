---
id: VTP42
order: 0
plan: run-dense-27b-virtual-tp
state: in_progress
created-at: '2026-11-15T13:01:55.739115+00:00'
breadth: ''
skill: advanced
created-by: agent
work: L
priority: P1
---

# Swappable prefill governor: token budget with reservation, backfill within a delay bound, card-free simulator

## Description

Owner proposal (2026-11-15/16). Replace fixed prefill admission numbers (VTP39 count cap, VTP41 threshold) with policies behind one interface. (a) Token budget: prefill tokens in flight (remaining, not whole prompts) limited to a multiple of the largest request (owner: 1.5x max request), capped by what free KV can hold, so a max-size request always fits and fp8/bf16 caches differ automatically. (b) Reservation: a request waiting past its max wait stops further admissions ahead of it until it fits. (c) Backfill: smaller later requests may go ahead of a waiting large one while the delay they add to it (tokens / measured prefill rate; steps are shared, so it is never zero) stays within a bound. Policies selected by a manifest field with parameters; engine stays generic.

## Steps

1. Governor interface in the scheduler (state in: running prefills with remaining tokens, waiting queue with sizes/ages/priorities, free budget and KV; out: admission decisions and walk order). Move arrival and VTP41 remaining logic behind it; randomised test that step plans are unchanged.
2. Card-free simulator: drive the scheduler with recorded or synthetic request mixes and a measured cost model (TTFT(N) = aN + bN^2, per-step times); calibrate against the VTP41 mixed A/B.
3. `budget` governor (multiple of max request, KV cap, reservation on age); simulate.
4. `backfill` governor (delay bound); simulate; tune.
5. Hardware A/B of the finalists against the VTP41 -long defaults on the mixed workload; owner picks defaults.

## Detailed Solution & Technical Design



## Code Samples & Guidance



## Files



## Validation

sched_test (identity of plans for existing policies, governor unit cases), simulator calibration error against hardware, mixed-workload A/B per request size, preemptions, decode beside the mix.

## Effort & Risk

L. Interface refactor of admission/walk order is the risky part (identity test guards it); simulator fidelity limits how far tuning offline can go.

## Standards



## Acceptance Criteria

- [ ] Existing policies reproduce their step plans exactly behind the interface.
- [ ] Simulator predicts per-size TTFT of the VTP41 A/B within 15%.
- [ ] A budget or backfill governor beats the VTP41 defaults on mid-size waits with no worse 100k mean, or the owner keeps VTP41 defaults with the comparison recorded.
- [ ] No request waits past its reservation bound (test).

## Notes

**2026-11-16 step 1 + 3 (rad0036).** Owner: no legacy mechanisms in the main code path, so rad0034 (cap) and rad0035 (order/threshold/age) are superseded by one governor: `--prefill-governor SPEC` / serving `prefill_governor` (kind=arrival|cap|budget; order, age_ms, max_seqs, long_tokens, factor), policy in core/sched/prefill_governor.h, four scheduler hooks (queue order, step shares, admission gate, overdue-recompute guard). Four serving fields become one (97 -> 94 options). Card-free: rad0034/35 tests ported to specs; equivalent-spec plan identity over 200 random schedules; budget limit and reservation tests; running prefills do not age (fix found on the card, see VTP41). Budget governor implemented (factor x max request, KV fit, reservation on age); hardware A/B pending. Steps 2 (simulator) and 4 (backfill) remain.

**2026-11-16 hardware** (`docs/evidence/2026-11-16-vtp42-prefill-governor/README.md`, -long-bf16 mixed workload, two starts an arm, no preemptions): -long cap spec = VTP41 options (100k mean 299 s, 4k-16k 3-27 s). Budget 1.5x without a short-prompt exemption held mid/short requests 39-49 s (220 s with the 120 s reservation), so `long_tokens` now exempts short prompts in every kind and from reservations. Budget 1.5x + long_tokens 16384 + remaining + age 120 s: mid-size unchanged, 100k mean 299 -> 228 s (-24%), last 100k 477 -> 390 s. Recommended as the -long default; owner to decide (cap spec kept until then). Remaining: simulator (step 2), backfill (step 4), decode beside the mix.

**2026-11-16 owner decision:** -long default switched to `kind=budget;factor=1.5;long_tokens=16384;order=remaining;age_ms=120000` (acceptance criterion 3: budget beats the VTP41 defaults on the 100k mean with mid-size waits unchanged).

RV4754: preserve deployed governor and existing long-profile admission in baseline. Test budget/cap/arrival and prefix-aware starvation/progress; only fix actual regressions. New policy-factor search and simulator/backfill extension after baseline unless required by VTP59.

Focused TP3 governor screen on exact candidate 927a24bf85c2ac29d3dbe0b779063323: 8 streams, 4096/512 then 512/128, 2 reps each, FP8 KV exact wire, pipeline min256, batch2048, interleave off, target75. Three verified fresh-start policies arrival; budget factor1.0/remaining/512 exemption/120000ms; cap2/remaining/512/120000ms. All 12 rounds complete, prompt hashes and actual counts match, paired tuning receipts, six portable exports, zero preemptions, clean stops. Long throughput / max TTFT / p95 event gap: arrival180.3305/15.596s/154.171ms; budget125.088/27.277s/162.6465ms; cap114.489/30.842s/158.798ms. Short throughput281.9865/278.9115/275.8155. End mixed-target misses/steps142/144,347/351,440/444, learned cap64 across all (one geometry quantum); source busy_cap raises effective allowance for participant fairness and backfill, so target is soft and reducing numeric target alone is unlikely to fix latency. Restricted admission worsens 8K-context burst performance; reject these arms, preserve long-profile governor and existing defaults. Evidence tp3-governor-screen.json. No new engine patch or qualified production setting. Next isolate per-mixed-step row/participant floor vs decode cost using native metrics/host scheduler simulation; only change scheduler if a latency-safe fair policy can be proven. Baseline release gate remains open; avoid repeating already covered screens.

## Change Log

- 2026-11-15T13:01:55.739115+00:00 (created-by): Created by agent
- 2026-11-15T13:50:52.673132+00:00 (updated-by): Updated: section:notes
- 2026-11-15T15:16:17.767179+00:00 (updated-by): Updated: section:notes
- 2026-11-15T15:16:25.831934+00:00 (state-transition): State: pending → in_progress
- 2026-11-15T21:14:43.227715+00:00 (updated-by): Updated: section:notes
- 2026-11-27T15:13:42.568082+00:00 (updated-by): Updated: section:notes
- 2026-11-27T20:37:55.186640+00:00 (updated-by): Updated: section:notes
