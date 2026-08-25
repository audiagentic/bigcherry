"""Offline sequential/bandit candidate-selection simulator (HI44).

Replays real round-aligned GPU samples already captured by a normal
screen-then-final tuning run under alternative allocation policies
(fixed-schedule baseline, successive halving, UCB1, Thompson sampling),
without launching anything on a GPU. No candidate order is invented: each
policy consumes a candidate's own recorded round sequence strictly in the
order it was actually measured, choosing only *how many* rounds to spend on
each candidate and *when to stop* -- which is what an adaptive allocator
actually controls. This is why the simulator needs raw per-round samples,
not the finished-measurement view `tools/bigcherry/ranking_policy.py`'s
`PrototypePolicy` interface deliberately provides (see that module's
`CandidateMetrics` docstring) -- that interface is for re-ranking already-
complete measurements, not for deciding sampling order mid-run. This module
is intentionally a separate interface, not an extension of that one.

Known data-shape limitation, confirmed against the only archived files with
real round data (`artifacts/tuning-runs/hi24-hi34-closure-20260809/`): round
samples are the raw GPU timing per round (`samples_us`); the host-adjusted
`effective_us` ranking metric the live tuner uses is not captured per round
(only as a final aggregate). Policies here therefore rank on raw GPU round
timing directly -- a reasonable, honestly-scoped proxy, not a claim of
exact parity with `effective_us`-based ranking.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable


class BanditSimulatorError(RuntimeError):
    pass


@dataclass(frozen=True)
class CandidateRounds:
    """One candidate's own recorded round sequence, order preserved."""

    name: str
    is_native: bool
    is_twin: bool
    rounds: tuple[float, ...]


@dataclass(frozen=True)
class DispatchRoundEvidence:
    """One dispatch's full round-level evidence: every finalist's own
    recorded sequence, plus the ground-truth decision to score against.
    """

    dispatch: str
    native_name: str
    target_winner: str
    target_is_degraded: bool  # True when target_winner falls back to `winner` (no provisional_winner -- pre-HI50 archive)
    candidates: tuple[CandidateRounds, ...]

    def non_twin(self) -> tuple[CandidateRounds, ...]:
        return tuple(c for c in self.candidates if not c.is_twin)

    def is_canary_only(self) -> bool:
        """True when the only sampled candidates are native and its own
        twin -- the same kernel measured twice. Any policy that declares a
        winner other than native here is a false promotion by construction
        (HI24's own noise-canary premise: native and a same-kernel twin run
        identical code, so any gap is measurement error).
        """
        names = {c.name.replace("#twin", "") for c in self.candidates if c.rounds}
        return names == {self.native_name}


def _finite(values: list[Any]) -> tuple[float, ...]:
    return tuple(v for v in values if isinstance(v, (int, float)) and math.isfinite(v) and v > 0.0)


def load_round_evidence(path: Path) -> list[DispatchRoundEvidence]:
    """Parse one `*.measurements.jsonl` file into round-level evidence.

    Rows without `schedule.candidates` (older artifact format -- confirmed
    present in `artifacts/tuning-logs/*.jsonl`, which carries no round data
    at all) or with fewer than one sampled finalist are silently skipped;
    the caller can compare `len(rows_in_file)` against
    `len(load_round_evidence(...))` to see how much of a file was usable.
    """
    evidence: list[DispatchRoundEvidence] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        if not raw.strip():
            continue
        row = json.loads(raw)
        if row.get("kind") != "result":
            continue
        schedule = row.get("schedule") or {}
        finalist_names = set(schedule.get("candidates") or [])
        if not finalist_names:
            continue
        native_name = row.get("native", "")
        if not native_name:
            continue
        candidates: list[CandidateRounds] = []
        for entry in row.get("candidates", []):
            name = entry.get("name", "")
            if name not in finalist_names:
                continue
            rounds = _finite(entry.get("samples_us") or [])
            if not rounds:
                continue
            is_twin = name.endswith("#twin")
            base_name = name[:-5] if is_twin else name
            candidates.append(CandidateRounds(
                name=name, is_native=(base_name == native_name and not is_twin),
                is_twin=is_twin, rounds=rounds,
            ))
        if not candidates:
            continue
        provisional = row.get("provisional_winner")
        target = provisional if provisional else row.get("winner", "")
        if not target:
            continue
        evidence.append(DispatchRoundEvidence(
            dispatch=row.get("dispatch", ""), native_name=native_name,
            target_winner=target, target_is_degraded=(provisional is None),
            candidates=tuple(candidates),
        ))
    return evidence


def _architecture_codes(paths: list[Path]) -> set[int]:
    """Distinct GPU architecture codes across a corpus's file headers."""
    codes: set[int] = set()
    for path in paths:
        for raw in path.read_text(encoding="utf-8").splitlines():
            if not raw.strip():
                continue
            row = json.loads(raw)
            if row.get("kind") != "header":
                continue
            key = row.get("hardware_key") or {}
            code = key.get("architecture_code")
            if code is not None:
                codes.add(code)
            break
    return codes


def load_corpus(paths: list[Path]) -> list[DispatchRoundEvidence]:
    out: list[DispatchRoundEvidence] = []
    for path in paths:
        out.extend(load_round_evidence(path))
    return out


# --- allocation policies -------------------------------------------------
#
# Each policy takes a dispatch's non-twin finalists (the real candidates a
# live allocator would choose among) and a `min_rounds` floor (mirrors the
# live tuner's own `min_paired_rounds` -- a decision below this floor is
# declared inconclusive, not forced), and returns (declared_winner_name,
# total_rounds_consumed). It must never consume more rounds than a
# candidate actually has recorded.

@dataclass
class PolicyResult:
    winner: str                 # raw fastest candidate by mean -- NOT yet a promotion decision
    rounds_consumed: int
    conclusive: bool
    means: dict[str, float] = field(default_factory=dict)  # every candidate's mean over its consumed rounds
    consumed_rounds: dict[str, tuple[float, ...]] = field(default_factory=dict)  # the actual truncated round sequence used per candidate


def binomial_tail_ge(wins: int, n: int) -> float:
    """One-sided p-value: P(X >= wins) under X ~ Binomial(n, 0.5). Mirrors
    the live tuner's own `binomial_tail_ge` (hip-autotune-tuner.cu) so this
    offline significance test uses the same statistic the real algorithm
    does, not an approximation.
    """
    if n <= 0:
        return 1.0
    return sum(math.comb(n, k) for k in range(wins, n + 1)) / (2 ** n)


def paired_sign_test(native_rounds: tuple[float, ...], challenger_rounds: tuple[float, ...],
                     min_rounds: int) -> tuple[float, int, int]:
    """One-sided paired sign test: is the challenger reliably faster than
    native round-for-round? Paired by original recording index -- valid
    because both sequences are truncations of the same real interleaved
    measurement schedule (round r for native and round r for challenger
    were genuinely measured as the same round in the original tuning run),
    even when an adaptive policy consumed different total counts of each.
    Mirrors the live tuner's own `paired_sign_test`. Returns
    (p_value, wins, rounds_compared); p_value defaults to 1.0 (not
    significant) below min_rounds, matching the live algorithm's own
    "declared inconclusive, not forced" rule.
    """
    rounds = min(len(native_rounds), len(challenger_rounds))
    wins = 0
    n = 0
    for r in range(rounds):
        a, b = native_rounds[r], challenger_rounds[r]
        if a == b:
            continue
        if b < a:
            wins += 1
        n += 1
    if n < min_rounds:
        return 1.0, wins, n
    return binomial_tail_ge(wins, n), wins, n


def declared_promotion(result: "PolicyResult", native_name: str,
                       threshold_pct: float = 1.0, alpha: float = 0.025,
                       min_paired_rounds: int = 8) -> str:
    """Apply the same dual gate the live tuner applies (standards 7.3,
    HI12 E1 -- "effect size and consistency are different questions and
    both must answer yes"): a raw fastest-mean candidate only becomes a
    genuine promotion if it beats native's own mean by at least
    `threshold_pct` AND that difference is statistically consistent
    round-for-round (one-sided paired sign test, p <= alpha). Effect size
    alone is not enough -- confirmed empirically: with only the effect-size
    gate, a native-vs-its-own-twin canary pair still "declared a winner"
    ~30% of the time on real 100-round data, because ordinary GPU timing
    noise regularly produces a >=1% mean gap between two identical kernels
    by chance. Adding the paired significance test is what the real
    algorithm actually does to guard against exactly this. Returns
    native_name when either condition fails (native retained), matching
    the live algorithm's own default.
    """
    if not result.conclusive or not result.winner or native_name not in result.means:
        return native_name
    if result.winner == native_name:
        return native_name
    native_mean = result.means[native_name]
    winner_mean = result.means.get(result.winner)
    if winner_mean is None or native_mean <= 0.0:
        return native_name
    improvement_pct = 100.0 * (native_mean - winner_mean) / native_mean
    if improvement_pct < threshold_pct:
        return native_name
    native_rounds = result.consumed_rounds.get(native_name, ())
    winner_rounds = result.consumed_rounds.get(result.winner, ())
    p_value, _, _ = paired_sign_test(native_rounds, winner_rounds, min_paired_rounds)
    return result.winner if p_value <= alpha else native_name


def _mean(values: tuple[float, ...]) -> float:
    return sum(values) / len(values) if values else math.inf


def fixed_schedule(candidates: tuple[CandidateRounds, ...], min_rounds: int) -> PolicyResult:
    """Baseline / control: consume every recorded round for every
    candidate, exactly reproducing what the original fixed final-stage
    schedule did. Not an allocator -- the reference point every adaptive
    policy is compared against for measurement-count savings.
    """
    consumed = sum(len(c.rounds) for c in candidates)
    eligible = {c.name: c.rounds for c in candidates if len(c.rounds) >= min_rounds}
    means = {n: _mean(r) for n, r in eligible.items()}
    if not means:
        return PolicyResult(winner="", rounds_consumed=consumed, conclusive=False, means=means, consumed_rounds=eligible)
    winner = min(means, key=means.get)
    return PolicyResult(winner=winner, rounds_consumed=consumed, conclusive=True, means=means, consumed_rounds=eligible)


def successive_halving(candidates: tuple[CandidateRounds, ...], min_rounds: int,
                       chunk: int = 8) -> PolicyResult:
    """Classic sequential-elimination bandit: measure every surviving
    candidate `chunk` more rounds, drop the worse half by running mean,
    repeat until one survivor remains or nobody has rounds left. Native is
    never eliminated (standards 11.4: "native is always kept") -- both
    because the live tuner's own screening obeys this, and because
    `declared_promotion`'s materiality check needs native's own mean to
    compare a challenger against; eliminating it would silently make every
    decision after that point unable to be judged as a real promotion.
    """
    native_names = {c.name for c in candidates if c.is_native}
    survivors = {c.name: 0 for c in candidates}
    by_name = {c.name: c for c in candidates}
    total_consumed = 0
    while len(survivors) > 1:
        progressed = False
        for name in list(survivors):
            c = by_name[name]
            take = min(chunk, len(c.rounds) - survivors[name])
            if take > 0:
                survivors[name] += take
                total_consumed += take
                progressed = True
        if not progressed:
            break
        means = {n: _mean(by_name[n].rounds[:k]) for n, k in survivors.items() if k > 0}
        if len(means) <= 1:
            break
        ranked = sorted(means, key=means.get)
        keep = max(1, math.ceil(len(ranked) / 2))
        kept = set(ranked[:keep]) | (native_names & set(ranked))
        survivors = {n: survivors[n] for n in ranked if n in kept}
    conclusive = bool(survivors) and all(survivors[n] >= min_rounds for n in survivors)
    consumed_rounds = {n: by_name[n].rounds[:k] for n, k in survivors.items() if k > 0}
    means = {n: _mean(r) for n, r in consumed_rounds.items()}
    if not survivors:
        return PolicyResult(winner="", rounds_consumed=total_consumed, conclusive=False, means=means, consumed_rounds=consumed_rounds)
    winner = min(survivors, key=lambda n: _mean(by_name[n].rounds[:survivors[n]]))
    return PolicyResult(winner=winner, rounds_consumed=total_consumed, conclusive=conclusive, means=means, consumed_rounds=consumed_rounds)


def ucb1(candidates: tuple[CandidateRounds, ...], min_rounds: int,
        exploration_c: float = 1.0, separation_sigma: float = 2.0) -> PolicyResult:
    """Bandit allocation: at each step, sample the candidate with the
    lowest lower-confidence-bound score (mean minus an exploration bonus --
    the minimization analogue of UCB1) from its own next unused recorded
    round. Stops early once the leading candidate's confidence interval no
    longer overlaps the runner-up's (a real early-stopping saving), or when
    every candidate's recorded rounds are exhausted.
    """
    consumed = {c.name: 0 for c in candidates}
    by_name = {c.name: c for c in candidates}
    total_budget = sum(len(c.rounds) for c in candidates)
    total_consumed = 0
    for t in range(1, total_budget + 1):
        available = [n for n in consumed if consumed[n] < len(by_name[n].rounds)]
        if not available:
            break
        def score(n: str) -> float:
            k = consumed[n]
            if k == 0:
                return -math.inf  # untried candidates are sampled first
            mean = _mean(by_name[n].rounds[:k])
            bonus = exploration_c * math.sqrt(2.0 * math.log(t) / k)
            return mean - bonus
        pick = min(available, key=score)
        consumed[pick] += 1
        total_consumed += 1
        if all(consumed[n] >= min_rounds for n in consumed):
            means_and_std = {
                n: (_mean(by_name[n].rounds[:k]),
                    (statistics.pstdev(by_name[n].rounds[:k]) if k > 1 else 0.0) / math.sqrt(k))
                for n, k in consumed.items() if k > 0
            }
            ranked = sorted(means_and_std, key=lambda n: means_and_std[n][0])
            if len(ranked) >= 2:
                leader_mean, leader_se = means_and_std[ranked[0]]
                runner_mean, runner_se = means_and_std[ranked[1]]
                if leader_mean + separation_sigma * leader_se < runner_mean - separation_sigma * runner_se:
                    break
            else:
                break
    consumed_rounds = {n: by_name[n].rounds[:k] for n, k in consumed.items() if k >= min_rounds}
    means = {n: _mean(r) for n, r in consumed_rounds.items()}
    if not means:
        return PolicyResult(winner="", rounds_consumed=total_consumed, conclusive=False, means=means, consumed_rounds=consumed_rounds)
    winner = min(means, key=means.get)
    return PolicyResult(winner=winner, rounds_consumed=total_consumed, conclusive=True, means=means, consumed_rounds=consumed_rounds)


def thompson_sampling(candidates: tuple[CandidateRounds, ...], min_rounds: int,
                      seed: int = 0, separation_sigma: float = 2.0) -> PolicyResult:
    """Bayesian allocation: maintain a Gaussian posterior over each
    candidate's mean (improper flat prior, updated from its own observed
    rounds so far); at each step, draw one sample per candidate from its
    posterior and measure whichever candidate's draw is lowest. Same
    early-stopping rule as ucb1 once every candidate has min_rounds.
    """
    rng = random.Random(seed)
    consumed = {c.name: 0 for c in candidates}
    by_name = {c.name: c for c in candidates}
    total_budget = sum(len(c.rounds) for c in candidates)
    total_consumed = 0
    for t in range(1, total_budget + 1):
        available = [n for n in consumed if consumed[n] < len(by_name[n].rounds)]
        if not available:
            break
        def draw(n: str) -> float:
            k = consumed[n]
            if k == 0:
                return -math.inf
            rounds = by_name[n].rounds[:k]
            mean = _mean(rounds)
            se = (statistics.pstdev(rounds) if k > 1 else mean * 0.1) / math.sqrt(k)
            return rng.gauss(mean, max(se, 1e-9))
        pick = min(available, key=draw)
        consumed[pick] += 1
        total_consumed += 1
        if all(consumed[n] >= min_rounds for n in consumed):
            means_and_std = {
                n: (_mean(by_name[n].rounds[:k]),
                    (statistics.pstdev(by_name[n].rounds[:k]) if k > 1 else 0.0) / math.sqrt(k))
                for n, k in consumed.items() if k > 0
            }
            ranked = sorted(means_and_std, key=lambda n: means_and_std[n][0])
            if len(ranked) >= 2:
                leader_mean, leader_se = means_and_std[ranked[0]]
                runner_mean, runner_se = means_and_std[ranked[1]]
                if leader_mean + separation_sigma * leader_se < runner_mean - separation_sigma * runner_se:
                    break
            else:
                break
    consumed_rounds = {n: by_name[n].rounds[:k] for n, k in consumed.items() if k >= min_rounds}
    means = {n: _mean(r) for n, r in consumed_rounds.items()}
    if not means:
        return PolicyResult(winner="", rounds_consumed=total_consumed, conclusive=False, means=means, consumed_rounds=consumed_rounds)
    winner = min(means, key=means.get)
    return PolicyResult(winner=winner, rounds_consumed=total_consumed, conclusive=True, means=means, consumed_rounds=consumed_rounds)


ALLOCATION_POLICIES: dict[str, Callable[..., PolicyResult]] = {
    "fixed-schedule": fixed_schedule,
    "successive-halving": successive_halving,
    "ucb1": ucb1,
    "thompson-sampling": thompson_sampling,
}


# --- evaluation ------------------------------------------------------------

def _resolve_target_name(evidence: DispatchRoundEvidence) -> str:
    target = evidence.target_winner
    if target.endswith("#twin"):
        target = target[:-5]
    return target


def evaluate_agreement(corpus: list[DispatchRoundEvidence], policy_name: str,
                       min_rounds: int = 8, threshold_pct: float = 1.0) -> dict[str, Any]:
    """Call-weighted-by-row-count agreement between one policy's *declared
    promotion* (raw fastest mean, gated through `declared_promotion`'s
    materiality check against native -- not the raw fastest mean alone) and
    the row's own recorded ground-truth decision. Restricted to rows with
    more than one distinct non-twin candidate -- a twin-only row has no
    real decision to agree or disagree with (see evaluate_false_promotion).
    """
    fn = ALLOCATION_POLICIES[policy_name]
    total = 0
    agree = 0
    inconclusive = 0
    rounds_used = 0
    rounds_baseline = 0
    disagreements: list[dict[str, Any]] = []
    for evidence in corpus:
        contenders = evidence.non_twin()
        if len({c.name for c in contenders}) < 2:
            continue
        result = fn(contenders, min_rounds)
        baseline = fixed_schedule(contenders, min_rounds)
        total += 1
        rounds_used += result.rounds_consumed
        rounds_baseline += baseline.rounds_consumed
        if not result.conclusive:
            inconclusive += 1
            continue
        declared = declared_promotion(result, evidence.native_name, threshold_pct)
        target = _resolve_target_name(evidence)
        if declared == target:
            agree += 1
        else:
            disagreements.append({
                "dispatch": evidence.dispatch, "declared": declared,
                "target": target, "degraded_target": evidence.target_is_degraded,
            })
    return {
        "policy": policy_name, "rows_evaluated": total, "agree": agree,
        "inconclusive": inconclusive,
        "agreement_rate": (agree / total) if total else None,
        "rounds_used": rounds_used, "rounds_baseline": rounds_baseline,
        "measurement_reduction_pct": (
            100.0 * (rounds_baseline - rounds_used) / rounds_baseline
            if rounds_baseline else None
        ),
        "disagreements": disagreements,
    }


BASELINE_POLICY_NAME = "fixed-schedule"


def _exact_mcnemar_p(k: int, n: int) -> float:
    """Two-sided exact McNemar p-value from `n` discordant pairs, `k` the
    smaller side's count. Standard doubled-tail binomial(n, 0.5) form --
    implemented against the stdlib (`math.comb`) rather than pulled in from
    scipy, matching this module's own dependency-free convention.
    """
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, i) for i in range(0, k + 1)) * (0.5 ** n)
    return min(1.0, 2.0 * tail)


def evaluate_paired(corpus: list[DispatchRoundEvidence], policy_a: str, policy_b: str,
                    min_rounds: int = 8, threshold_pct: float = 1.0) -> dict[str, Any]:
    """Paired comparison of two policies on the SAME dispatches (HI44).

    Comparing two policies via their independent Wilson intervals (as
    `hi_campaign_gates.py` does for a quick read) needs an enormous corpus to
    separate two rates that are close together -- ~7,100 dispatches to
    resolve successive-halving from the baseline at +/-0.5pp on the 389-
    dispatch corpus this was first measured on. That estimate is inflated by
    variance the two policies don't actually have independently: they are
    evaluated on the *same* dispatch, under the *same* hardware conditions,
    subject to the *same* measurement noise, so whatever made one policy's
    call harder on a given dispatch made the other's harder too. A paired
    test only spends its power on the dispatches where the two policies
    actually DISAGREE with each other (the "discordant pairs"); dispatches
    where both get it right or both get it wrong are uninformative for the
    comparison and correctly contribute nothing, rather than being folded
    into two independent rates whose difference has to fight through both
    policies' full variance.

    This is McNemar's test: among the discordant dispatches, under the null
    hypothesis that the two policies are equally likely to be the one that's
    right, the discordant pairs should split roughly 50/50. A lopsided split
    is evidence of a real difference; the exact binomial p-value quantifies
    "how lopsided is too lopsided" for this specific n.
    """
    fn_a = ALLOCATION_POLICIES[policy_a]
    fn_b = ALLOCATION_POLICIES[policy_b]
    both_agree = both_disagree = a_only = b_only = 0
    skipped_inconclusive = 0
    examples_a_only: list[str] = []
    examples_b_only: list[str] = []
    for evidence in corpus:
        contenders = evidence.non_twin()
        if len({c.name for c in contenders}) < 2:
            continue
        result_a = fn_a(contenders, min_rounds)
        result_b = fn_b(contenders, min_rounds)
        if not result_a.conclusive or not result_b.conclusive:
            skipped_inconclusive += 1
            continue
        target = _resolve_target_name(evidence)
        agree_a = declared_promotion(result_a, evidence.native_name, threshold_pct) == target
        agree_b = declared_promotion(result_b, evidence.native_name, threshold_pct) == target
        if agree_a and agree_b:
            both_agree += 1
        elif not agree_a and not agree_b:
            both_disagree += 1
        elif agree_a and not agree_b:
            a_only += 1
            if len(examples_a_only) < 5:
                examples_a_only.append(evidence.dispatch)
        else:
            b_only += 1
            if len(examples_b_only) < 5:
                examples_b_only.append(evidence.dispatch)

    n_discordant = a_only + b_only
    k = min(a_only, b_only)
    p_value = _exact_mcnemar_p(k, n_discordant)
    if n_discordant == 0:
        verdict = "IDENTICAL (zero discordant pairs -- both policies agree with ground truth on exactly the same dispatches)"
    elif p_value < 0.05:
        winner = policy_a if a_only > b_only else policy_b
        verdict = f"RESOLVED: {winner} is significantly better (p={p_value:.4f})"
    else:
        verdict = f"UNRESOLVED (p={p_value:.4f}; need more discordant pairs, not more dispatches)"

    return {
        "policy_a": policy_a, "policy_b": policy_b,
        "paired_dispatches": both_agree + both_disagree + a_only + b_only,
        "skipped_inconclusive": skipped_inconclusive,
        "both_agree": both_agree, "both_disagree": both_disagree,
        f"{policy_a}_only_correct": a_only, f"{policy_b}_only_correct": b_only,
        "mcnemar_p_value": p_value,
        "verdict": verdict,
        f"{policy_a}_only_correct_examples": examples_a_only,
        f"{policy_b}_only_correct_examples": examples_b_only,
    }


def evaluate_false_promotion(corpus: list[DispatchRoundEvidence], policy_name: str,
                             min_rounds: int = 8, threshold_pct: float = 1.0) -> dict[str, Any]:
    """False-promotion rate under a REAL global null: native measured
    against its own same-kernel twin (HI24's noise-canary pairs). A
    declared *promotion* (materiality-gated via `declared_promotion`, the
    same check the live tuner applies) away from native here is a false
    positive by construction -- there is no real difference to detect.
    This is a data-grounded null test, not a synthetic-p-value simulation:
    it tests whether this specific allocation policy's own adaptive
    stopping rule can be fooled by ordinary measurement noise, which a
    synthetic p-value check (like `tune_promotion.simulate_null_fdr`) does
    not exercise. Without the materiality gate this measured ~50% "false
    promotions" for every policy including the non-adaptive fixed-schedule
    baseline -- that was an artifact of comparing raw means with no effect-
    size threshold, not a real finding; the gate fixes that.
    """
    fn = ALLOCATION_POLICIES[policy_name]
    total = 0
    false_promotions = 0
    inconclusive = 0
    for evidence in corpus:
        if not evidence.is_canary_only():
            continue
        pair = tuple(c for c in evidence.candidates if c.rounds)
        if len(pair) != 2:
            continue
        total += 1
        result = fn(pair, min_rounds)
        if not result.conclusive:
            inconclusive += 1
            continue
        declared = declared_promotion(result, evidence.native_name, threshold_pct)
        if declared != evidence.native_name:
            false_promotions += 1
    return {
        "policy": policy_name, "canary_pairs_evaluated": total,
        "false_promotions": false_promotions, "inconclusive": inconclusive,
        "false_promotion_rate": (false_promotions / total) if total else None,
    }


def evaluate_reproducibility(corpus: list[DispatchRoundEvidence], policy_name: str,
                             min_rounds: int = 8, resamples: int = 200,
                             seed: int = 20260809, threshold_pct: float = 1.0) -> dict[str, Any]:
    """Bootstrap-resample each competitive candidate's own round sequence
    (with replacement, same length) and rerun the policy; report the
    fraction of resamples whose *declared promotion* (materiality-gated,
    same as evaluate_agreement/evaluate_false_promotion) agrees with the
    policy's own point-estimate decision on the original order. Measures
    stability of the promotion decision, not the raw fastest-mean ranking
    -- a policy can flip its raw argmin near a genuine tie while still
    stably declaring "native retained" both times, which is what actually
    matters operationally.
    """
    fn = ALLOCATION_POLICIES[policy_name]
    rng = random.Random(seed)
    per_dispatch: list[float] = []
    for evidence in corpus:
        contenders = evidence.non_twin()
        if len({c.name for c in contenders}) < 2:
            continue
        point = fn(contenders, min_rounds)
        if not point.conclusive or not point.winner:
            continue
        point_decision = declared_promotion(point, evidence.native_name, threshold_pct)
        stable = 0
        counted = 0
        for _ in range(resamples):
            resampled = tuple(
                CandidateRounds(
                    name=c.name, is_native=c.is_native, is_twin=c.is_twin,
                    rounds=tuple(rng.choice(c.rounds) for _ in range(len(c.rounds))),
                )
                for c in contenders
            )
            trial = fn(resampled, min_rounds)
            if not trial.conclusive:
                continue
            counted += 1
            trial_decision = declared_promotion(trial, evidence.native_name, threshold_pct)
            if trial_decision == point_decision:
                stable += 1
        if counted:
            per_dispatch.append(stable / counted)
    return {
        "policy": policy_name, "dispatches_evaluated": len(per_dispatch),
        "mean_stability": (sum(per_dispatch) / len(per_dispatch)) if per_dispatch else None,
        "min_stability": min(per_dispatch) if per_dispatch else None,
    }


def simulate(paths: list[Path], *, min_rounds: int = 8, resamples: int = 200,
            seed: int = 20260809) -> dict[str, Any]:
    corpus = load_corpus(paths)
    n_rows_raw = 0
    for path in paths:
        n_rows_raw += sum(
            1 for raw in path.read_text(encoding="utf-8").splitlines()
            if raw.strip() and json.loads(raw).get("kind") == "result"
        )
    arch_codes = _architecture_codes(paths)
    if len(arch_codes) >= 2:
        hardware_note = (
            f"multi-architecture: {len(arch_codes)} distinct GPU architecture codes "
            f"present in corpus ({sorted(arch_codes)})"
        )
    elif len(arch_codes) == 1:
        hardware_note = (
            "single architecture only (this corpus's files share one hardware digest); "
            "does not satisfy a multi-architecture requirement"
        )
    else:
        hardware_note = "no header rows found; architecture coverage unknown"
    report: dict[str, Any] = {
        "corpus_files": [str(p) for p in paths],
        "result_rows_in_files": n_rows_raw,
        "rows_with_usable_round_data": len(corpus),
        "hardware_coverage_note": hardware_note,
        "policies": {},
    }
    for name in ALLOCATION_POLICIES:
        report["policies"][name] = {
            "agreement": evaluate_agreement(corpus, name, min_rounds),
            "false_promotion": evaluate_false_promotion(corpus, name, min_rounds),
            "reproducibility": evaluate_reproducibility(corpus, name, min_rounds, resamples, seed),
        }

    # HI44: independent Wilson intervals need ~7,100 dispatches to separate
    # successive-halving from the baseline; a paired test on the SAME
    # dispatches only spends power on the ones where the two actually
    # disagree, and can resolve the question on a far smaller corpus. Run
    # automatically for every non-baseline policy against the baseline so a
    # report never has to be re-generated just to add this comparison.
    if BASELINE_POLICY_NAME in ALLOCATION_POLICIES:
        report["paired_vs_baseline"] = {
            name: evaluate_paired(corpus, BASELINE_POLICY_NAME, name, min_rounds)
            for name in ALLOCATION_POLICIES if name != BASELINE_POLICY_NAME
        }
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="bigcherry bandit-simulate")
    parser.add_argument("measurements", nargs="+", help="*.measurements.jsonl files to replay")
    parser.add_argument("--min-rounds", type=int, default=8)
    parser.add_argument("--resamples", type=int, default=200)
    parser.add_argument("--seed", type=int, default=20260809)
    parser.add_argument("--output", help="write the JSON report here too")
    args = parser.parse_args(argv)
    try:
        paths = [Path(p) for p in args.measurements]
        for p in paths:
            if not p.is_file():
                raise BanditSimulatorError(f"{p}: no such file")
        report = simulate(paths, min_rounds=args.min_rounds, resamples=args.resamples, seed=args.seed)
    except (OSError, BanditSimulatorError) as exc:
        print(f"invalid: {exc}", file=sys.stderr)
        return 1
    rendered = json.dumps(report, indent=2, sort_keys=True)
    if args.output:
        Path(args.output).write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
