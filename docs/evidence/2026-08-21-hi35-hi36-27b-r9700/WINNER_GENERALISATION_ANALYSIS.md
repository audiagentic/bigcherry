# Winner generalisation — offline regret analysis (HI36 steps 1–3)

Date: 2026-08-21. Driver scripts: `tmp/h36-regret-analysis.py`,
`tmp/h36-key-derivation.py`, plus inline same-family representative variants.
All numbers are computable offline from data already on disk; no GPU was used.

## Data

| Set | File | variant_set | flush_l2 | results | tuned (winner≠native) |
| --- | --- | --- | --- | --- | --- |
| **nm (primary)** | `artifacts/h65-local/nm.jsonl.measurements.jsonl` | full-max | **0 (production semantics)** | 1156 | 202 |
| r256 (robust) | `artifacts/h65-local/r256.jsonl.measurements.jsonl` | full-max | 1 | 1156 | 210 |
| e512 (robust) | `artifacts/h65-local/e512.jsonl.measurements.jsonl` | full-max | 1 | 1156 | 156 |

Families in the sweep: mmvq 362, blas 346, mmvf 281, mmq 167. Every result
carries a full `canonical` signature (all 21 ABI fields), so any grouping is
priceable from the medians already measured — including candidates that were
not the winner.

**Limitations, stated up front:**

- **No workload call weights exist for the 1156-signature inventory sweep.**
  It is an inventory sweep, not a workload record. The only workload record on
  disk (4B server, 61 signatures / 1429 calls) has **zero** signature overlap
  with the sweep. Call weights in this analysis are therefore uniform 1, and
  the group representative is the highest-*native*-time member (documented
  proxy, not a workload claim). Call-weighted numbers are the unweighted
  percentiles; the true call-weighted verdict requires a holdout record.
- **The Gemma E4B miss log is not on disk**, so the item's literal step-3 gate
  ("replay the 100 misses; if fewer than half convert, stop") cannot be run.
  The go/no-go below is the per-family regret verdict that precedes it.
- r256/e512 use a different preconditioning policy (flush=1) than nm (flush=0).
  Per the B2 execution contract these are different measurement semantics, so
  cross-run agreement is an upper bound on instability, not a noise floor.

## Method

For a proposed group key, one representative winner per group (highest
native-time member, or the same-family variant noted per finding). Regret of a
member = `100 × (rep_time − exact_winner_time) / exact_winner_time`, from
medians of candidates both measured for that signature. `ineligible` = the
representative is not in the member's measured candidate set (at runtime that
is a safe native fallback, but it converts nothing). Pre-declared promotion
thresholds (`generalise.REQUIRED_THRESHOLDS`): median regret ≤ 0.5%, bootstrap
upper ≤ 1.0%, no correctness failures. RV21 established a ~14% error on a
3-sample median; these full-max runs used more samples, so residual worst-case
regrets of a few percent are inside the measurement-noise band.

## Step 1 — which dimensions carry the winner

Coarsest key `(family, src0_type, src1_type)` — i.e. generalise on *everything*
else:

| family | groups | p95 regret | worst regret | ineligible |
| --- | --- | --- | --- | --- |
| mmvq | 23 | 8–13% | **279.6%** (nm) | 0–21 |
| mmq | 22 | 2.2–9.2% | 44–48% | 0 |
| mmvf | 3 | 5.9–12.2% | 64–70% | **114–185 / 281** |
| blas | 5 | 0% | 179–229% (1–2 sigs) | 0 |

Verdict: winner is *not* a function of types alone in any family. The big
worst-case numbers trace to two distinct mechanisms (findings 1–2 below).

Step 2 — progressively finer keys (same-family representative; worst across
nm/r256/e512 in brackets):

| family | key | groups | median | p95 | worst [worst over runs] | ineligible |
| --- | --- | --- | --- | --- | --- | --- |
| mmq | types+K+M | 59 (2.8× fewer than 167) | 0.0 | 0.0 | 0.0 [3.55] | 0 |
| mmq | types+ne0(full) | 86 | 0.0 | 0.0 | 0.0 [3.55] | 0 |
| mmvq | types+ne0(full) | 75 (4.8× fewer than 362) | 0.0 | 0.0 | 2.90 [6.02] | 0 |
| mmvq | types+ne0+ned(full) | 326 | 0.0 | 0.0 | 2.90 [2.90] | 0 |
| mmvf | types+K+M+ncols | 45 | 0.0 | 4.0–10.0 | 16.7 [70.7] | 0 |
| mmvf | types+ne0(full) | 127 | 0.0 | 1.5–3.6 | 15.4 [16.7] | 0 |
| mmvf | types+ne0+ned(full) | 267 (~no gain) | 0.0 | 0.0 | 0.0 [1.76] | 0 |
| blas | types+K+M | 33 | 0.0 | 0.0 | 179.4 [228.6] (1–2 sigs) | 0 |

## Step 3 — go/no-go per family

**mmq: GO (conditional).** Key `(family, src0_type, src1_type, K, M)`: 2.8×
fewer entries, zero median and p95 regret in all three runs, worst case ≤3.55%
(inside the noise band). Drops ncols_dst, batch extents, strides, flags — the
winner demonstrably does not depend on them at measured precision.

**mmvq: CONDITIONAL GO.** Key `(family, src0_type, src1_type, ne0)`: 4.8× fewer
entries, p95 = 0 in all runs, worst ≤6.02% (one signature, one run — above the
1.0% upper threshold but median/p95 clean). Requires finding 2 (same-family
representative) or the 6–19% cross-family regret returns.

**mmvf: NO-GO.** No safe coarse key exists. The winning variant (acc type
f16/f32/bf16, block size 32/64/96) changes with the *full* shape; even
`types+ne0+ned` leaves 1.76% worst and provides almost no entry reduction
(267 groups for 281 signatures). Generalisation cannot beat the exact digest
here without paying 5–70% regret.

**blas: NO-GO (not worth it).** 341/346 signatures keep the native winner;
there is no tuned value to generalise. The 179–229% single-signature cases are
cross-family winners (mmvf candidates winning blas rows) that a same-family
generalised entry cannot serve.

**Headline for the item's motivating question** (Gemma E4B: 21 hits / 100
misses on a similar workload): the safe generalised keys still key on the full
extent arrays for mmvq and on K+M for mmq — exactly the dimensions a *slightly
different* workload changes (context width → M, batch, ncols_dst). For 3 of
4 families the data says generalisation at safe regret does **not** convert a
meaningful fraction of new-workload misses; for mmq there is a bounded, real
opportunity (2.8× entry compression at zero measured regret). The step-3
"converts half the misses" gate therefore fails on available evidence for
mmvf/blas/mmvq and is **inconclusive-pending-miss-log** for mmq (the Gemma
miss log and any 27B-workload holdout record are not on disk — see gaps).

## Findings that constrain any step-4 design

1. **Representative eligibility is a first-class constraint.** With coarse
   keys, 114–185 of 281 mmvf members have no eligible representative (the
   candidate set varies per signature — applicability of w1/w4/w8, bs32/64/96
   is shape-dependent). Runtime `can_execute` → native fallback keeps it safe,
   but the exporter must count ineligible as *not converted*, and the key must
   be fine enough that the representative is universally eligible (the full
   extent key achieves 0 ineligible in every family).
2. **Same-family representative (or in-group native fallback) is mandatory.**
   Cross-family representatives produced the 6–19% mmvq regrets (an
   `mmq:native` representative served to mmvq shapes) and the 179–279%
   coarse-key worst cases. The current `group_representatives` picks by call
   weight with no family constraint; the exporter and any future
   `prove()` consumer must add the constraint.
3. **Inter-run stability is confounded by preconditioning policy.** The three
   runs differ in flush policy, so the 1–30% cross-run representative churn is
   an upper bound on instability, not the noise floor. A same-policy stability
   pass is required before any promotion claim (consistent with the B2
   contract: identical preconditioning, no cross-policy comparison).
4. **The full extent key is the practical floor.** Winners are a function of
   full extents at measured precision; the remaining residual (≤3.55%) is
   inside measurement noise and is indistinguishable from real sensitivity.
   Any claim of generalising *past* the extent arrays would be an
   over-claim this data cannot support.

## Gaps that must be filled before step 4 (C++ two-level lookup)

1. A **miss log** with `canonical` + `fallback` + candidate sets (Gemma E4B's
   100 misses, or a fresh MTP-workload replay) for the `what_if()` conversion
   gate — the item's literal go/no-go.
2. A **holdout record** (workload calls per signature) for the 27B MTP
   workload, so `prove()` can run its call-weighted gates (`min_holdout_calls`,
   `min_added_coverage_pct`) instead of uniform-weight proxies.
3. A **same-policy stability run** (two full-max tunes, flush=0, same
   driver/hardware) to bound inter-run representative churn at production
   semantics.
4. Small tooling extension: `generalise`'s named-field paths only express
   scalar extent positions (K/M/ncols_dst); full-array keys used here need
   `ne0`/`ned` full-array support (lists are unhashable in the current
   `group_key`) or new named paths (ne0_2/ne0_3/ned_2/ned_3). Add when step 4
   starts, with tests.

## Next step (recommendation)

Do **not** start step-4 C++ work yet. The offline evidence supports at most an
mmq-scoped generalisation (possibly +mmvq with the same-family constraint),
and the item's own gate requires the miss-log replay first. When the next MTP
workload tune runs, capture the miss log and holdout record (gaps 1–2) and
re-run `what_if()` — if mmq's key converts ≥ half of the mmq misses, scope
step 4 to mmq only; otherwise HI36 stops at this analysis and the remaining
value moves to reducing the number of distinct tuned shapes (tuning-budget
work, a different item).
