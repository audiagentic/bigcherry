---
id: VTP41
order: 0
plan: run-dense-27b-virtual-tp
state: pending
created-at: '2026-11-15T10:38:52.398726+00:00'
breadth: ''
skill: advanced
created-by: agent
work: M
priority: P1
---

# Prefill order by remaining work, and an adjustable long-prefill threshold (VTP39 follow-up)

## Description

VTP39's cap counts a prefill as long when more than one shared step (2048 tokens) is left; the threshold cannot be set without changing the step size. Running prefills are walked in priority-then-arrival order and every prefill behind the first reserves a share (budget/8), so a mid-size prompt admitted beside a long one gets ~256 tokens a step (an 8k prompt ~32 steps at depth) and eight running prefills all advance at ~256. Owner asked (2026-11-15) for both adjustments: a threshold for what counts as long, and ordering prefill work by what is left.

## Steps

1. `--long-prefill-tokens N` (manifest `tuning.long_prefill_tokens`): 0 keeps one shared step; otherwise the cap counts and exempts by N.
2. `--prefill-order arrival|remaining` (`tuning.prefill_order`, default arrival): under `remaining`, running prefills are walked and waiting prompts admitted by least remaining work (priority first), and prefills behind the first reserve at most one share between them.
3. `--prefill-age-ms T` (`tuning.prefill_age_ms`, 0 off): a prompt waiting longer than T since arrival sorts ahead of un-aged ones, in arrival order.
4. Card-free scheduler tests (defaults leave step plans identical; ordering; threshold; aging) and config tests; contract fields; regenerate.
5. Mixed-workload A/B on -long-bf16: eight 100k prompts plus 450 / 4k / 8k / 16k requests; arms: shipped (cap 2), + order remaining, + threshold 8k/16k, both.

## Detailed Solution & Technical Design



## Code Samples & Guidance



## Files



## Validation

sched_test and config_test (card-free); serving contract regeneration and serving tests; full-cache mixed client per-request TTFT by size, preemptions; KLD not affected (scheduling only) but stress run once.

## Effort & Risk

M. Scheduler walk order and admission order; risk is interaction with interleave holds, span, preemption order (preemption still uses priority/arrival from the back).

## Standards



## Acceptance Criteria

- [ ] Defaults produce identical step plans to today (randomised test).
- [ ] Mid-size (4k-16k) requests beside long prefills get first token within a few seconds with order remaining, without raising eight-long-prompt mean TTFT above the cap-2 result (268 s) by more than 5%.
- [ ] No long prompt waits past the age limit behind shorter work (test).
- [ ] Owner decides the -long defaults.

## Notes

**2026-11-15 implementation and hardware** (rad0035, PR #977; `docs/evidence/2026-11-15-vtp41-prefill-order/README.md`). -long-bf16 with the shipped cap 2 + interleave 2, eight 100k prompts plus 450/4k/8k/16k requests at 30 s and 150 s, two starts an arm: shipped 4k-16k requests wait 317-433 s (they count as long and queue behind all eight); `remaining` 53-73 s; threshold 16k 17-76 s; both 4-27 s with the 100k mean 270 -> 298 s (+10%). No preemptions. Correction recorded: with one long prefill running, arrival order already alternates the step via checkpoint-aligned chunks (a mid prompt gets about half), so ordering matters with several prefills running. Next: owner's choice of -long defaults; age limit measured on the card; real-time follow-ups proposed to the owner: cap from free KV, interleave from a target decode gap.

**2026-11-16 defaults and age limit on the card.** Owner chose the -long defaults: remaining order, 16k threshold, 120 s age. Confirm A/B (VUUV): the age as built in rad0035 also aged running prefills, so the 150 s wave's 4k/8k/16k waited 43-69/128-151/139-160 s (vs 4-7/11-13/20-23 s) with no 100k gain. Fixed in rad0036 (VTP42): only prompts waiting to start age (card-free test 5 vs 13 steps). The +10% 100k mean exceeds the 5% criterion; the owner accepted it with the defaults. The defaults ship as the rad0036 governor spec `kind=cap;max_seqs=2;long_tokens=16384;order=remaining;age_ms=120000`; rad0035 is superseded by rad0036.

## Change Log

- 2026-11-15T10:38:52.398726+00:00 (created-by): Created by agent
- 2026-11-15T12:25:00.334441+00:00 (updated-by): Updated: section:notes
- 2026-11-15T13:50:48.595963+00:00 (updated-by): Updated: section:notes
