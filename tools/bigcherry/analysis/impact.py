"""Offline, call-weighted tuning impact calculations (HI35).

This module deliberately consumes existing record and measurements JSONL.  It
does not alter either artifact or the replay/database wire formats.
"""

from __future__ import annotations

import argparse
import collections
import json
import math
import random
import statistics
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


class ImpactError(ValueError):
    pass


@dataclass
class Coverage:
    matched: int = 0
    record_only: list[str] = field(default_factory=list)
    measurement_only: list[str] = field(default_factory=list)
    calls_total: int = 0
    calls_covered: int = 0


def _finite(value: Any, name: str, *, positive: bool = True) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ImpactError(f"{name} must be numeric")
    value = float(value)
    if (
        not math.isfinite(value)
        or (positive and value <= 0)
        or (not positive and value < 0)
    ):
        raise ImpactError(
            f"{name} must be finite and {'positive' if positive else 'non-negative'}"
        )
    return value


def _candidate(result: dict[str, Any], name: str) -> dict[str, Any] | None:
    for candidate in result.get("candidates", []):
        if candidate.get("name") == name and candidate.get("status") == "ok":
            if "median_us" not in candidate:
                raise ImpactError(f"candidate {name!r} has no median_us")
            return candidate
    return None


def predicted_saving(
    observations: list[dict[str, Any]], results: list[dict[str, Any]]
) -> dict[str, Any]:
    """Calculate explicit coverage and call-weighted native-vs-winner saving."""
    calls: dict[str, int] = {}
    native_names: dict[str, str] = {}
    for row in observations:
        signature = row.get("signature")
        count = row.get("calls")
        if not isinstance(signature, str) or not signature:
            raise ImpactError("observation signature is required")
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            raise ImpactError(f"calls for {signature} must be a non-negative integer")
        if signature in calls:
            raise ImpactError(f"duplicate observation signature {signature}")
        calls[signature] = count
        native_names[signature] = row.get("native", "")

    coverage = Coverage(calls_total=sum(calls.values()))
    coverage.record_only = sorted(calls)
    measurement_signatures: set[str] = set()
    rows: list[dict[str, Any]] = []
    by_family: dict[str, dict[str, float | int]] = collections.defaultdict(
        lambda: {"calls": 0, "native_us": 0.0, "tuned_us": 0.0}
    )
    native_total = tuned_total = 0.0
    for result in results:
        signature = result.get("signature")
        if not isinstance(signature, str) or not signature:
            raise ImpactError("measurement signature is required")
        if signature in measurement_signatures:
            raise ImpactError(f"duplicate measurement signature {signature}")
        measurement_signatures.add(signature)
        if signature not in calls:
            continue
        coverage.record_only.remove(signature)
        native_name = result.get("native") or native_names[signature]
        winner_name = result.get("winner")
        if not isinstance(native_name, str) or not isinstance(winner_name, str):
            raise ImpactError(f"native and winner are required for {signature}")
        native = _candidate(result, native_name)
        winner = _candidate(result, winner_name)
        if native is None or winner is None:
            continue
        count = calls[signature]
        native_us_each = _finite(native["median_us"], "native median_us")
        winner_us_each = _finite(winner["median_us"], "winner median_us")
        native_us = count * native_us_each
        tuned_us = count * winner_us_each
        native_total += native_us
        tuned_total += tuned_us
        coverage.matched += 1
        coverage.calls_covered += count
        family = winner_name.split(":", 1)[0]
        bucket = by_family[family]
        bucket["calls"] += count
        bucket["native_us"] += native_us
        bucket["tuned_us"] += tuned_us
        rows.append(
            {
                "signature": signature,
                "calls": count,
                "saved_us_each": native_us_each - winner_us_each,
                "saved_us": native_us - tuned_us,
                "winner": winner_name,
            }
        )
    coverage.measurement_only = sorted(measurement_signatures - set(calls))
    rows.sort(key=lambda row: (-row["saved_us"], row["signature"]))
    return {
        "coverage": coverage,
        "native_total_us": native_total,
        "tuned_total_us": tuned_total,
        "saving_pct": (
            100.0 * (native_total - tuned_total) / native_total if native_total else 0.0
        ),
        "by_family": dict(by_family),
        "rows": rows,
        "slower": [row for row in rows if row["saved_us"] < 0],
    }


def expected_decode_effect(saving_pct: Any, decode_matmul_fraction: Any) -> float:
    """Return the predicted decode-wall effect, with explicit unit checks."""
    saving = _finite(saving_pct, "predicted saving", positive=False)
    fraction = _finite(decode_matmul_fraction, "decode matmul fraction", positive=False)
    if fraction > 1.0:
        raise ImpactError("decode matmul fraction must be between 0 and 1")
    if saving > 100.0:
        raise ImpactError("predicted saving cannot exceed 100 percent")
    return (saving / 100.0) * fraction * 100.0


# ------------------------------------------------------------------ loaders


def load_observations(path: Path) -> list[dict[str, Any]]:
    """Read a record file (JSONL with a header + observation rows).

    A missing header is an error, not an empty result: an empty
    observations list would read as 'zero calls' rather than as 'a file
    this loader cannot read'.

    Uses analysis/jsonl_io.py's shared strict reader rather than
    tuning/inventory.py's read_jsonl -- the two had DIFFERENT tolerance
    bugs (inventory.read_jsonl silently drops everything after the first
    malformed line, not just that line), and this module's job is
    validating input to a statistical calculation, not the wider
    inventory tool's own established (and separately-owned) tolerance
    behavior. Not delegating to inventory.read_jsonl here also avoids
    pulling in its larger dependency tree for this narrower need
    (gpt-dev-agent review round 2, 2026-08-31).
    """
    from . import jsonl_io

    try:
        rows = jsonl_io.read_rows(Path(path))
    except jsonl_io.JsonlReadError as exc:
        raise ImpactError(str(exc)) from exc
    if not any(row.get("kind") == "header" for row in rows):
        raise ImpactError(
            f"{Path(path)}: no header line. Either the file is not a "
            f"bigcherry record, or the run died before its first flush."
        )
    return [row for row in rows if row.get("kind") == "observation"]


def load_results(path: Path) -> list[dict[str, Any]]:
    """Read a measurements JSONL and return its result rows only.

    Header rows and any non-``result`` rows are dropped. Delegates to
    analysis/jsonl_io.py's shared strict reader: only a genuinely
    TRUNCATED final line (no trailing newline) is tolerated -- interior
    corruption, or a fully-written but corrupt final record, raises
    instead of silently dropping evidence this module feeds directly into
    a bootstrap confidence interval (gpt-dev-agent review round 2,
    2026-08-31: this loader's ORIGINAL bug -- `except: break` on ANY
    malformed line, discarding everything after it too, not just that one
    line -- was still present after report.py's independent round-1 fix;
    now both share one implementation).
    """
    from . import jsonl_io

    try:
        return jsonl_io.read_result_records(path)
    except jsonl_io.JsonlReadError as exc:
        raise ImpactError(str(exc)) from exc


# ------------------------------------------------------------------ interval


# Matches tuning/tune_promotion.py's own MIN_PAIRED_ROUNDS -- a production
# promotion decision requires at least this many paired rounds before
# trusting a bootstrap CI; this offline tool should not report false
# confidence from fewer rounds than production itself would ever act on.
# Kept as a local constant (not imported) so `bigcherry power`/`impact`
# stay cheap to invoke -- tune_promotion.py pulls in sqlite3/argparse and a
# larger dependency tree for something this module only needs one int from.
MIN_PAIRED_ROUNDS = 8


def _round_aligned_pairs(native_samples: Any, winner_samples: Any, name: str) -> list[tuple[float, float]]:
    """Zip native/winner ``samples_us`` INDEX-WISE and keep only rounds where
    both are present and finite -- the actual paired observation the tuner
    recorded, round r vs round r.

    gpt-dev-agent review, 2026-08-31: the previous version independently
    stripped nulls from each series and independently bootstrap-resampled
    them, which throws away the pairing entirely. With native=[100,200,300,
    400,500] and winner=[90,180,270,360,450] -- exactly 10% faster every
    single round -- the correct paired bootstrap always returns exactly
    10%; independently resampling two series that happen to be
    proportional does NOT preserve that exact ratio per draw, producing a
    spurious, overstated interval. Independently-stripped nulls could also
    silently misalign different-length series (native round 3 failed,
    winner round 3 succeeded) so index i in the old "native_samples" and
    index i in the old "winner_samples" were not even the same round.
    """
    if not isinstance(native_samples, (list, tuple)) or not isinstance(winner_samples, (list, tuple)):
        raise ImpactError(f"{name} samples_us must be a list")
    if len(native_samples) != len(winner_samples):
        raise ImpactError(
            f"{name} native/winner samples_us have different lengths -- not round-aligned"
        )
    pairs: list[tuple[float, float]] = []
    for native_value, winner_value in zip(native_samples, winner_samples):
        if native_value is None or winner_value is None:
            continue
        pairs.append(
            (
                _finite(native_value, f"{name} native sample", positive=True),
                _finite(winner_value, f"{name} winner sample", positive=True),
            )
        )
    return pairs


def saving_interval(
    observations: list[dict[str, Any]],
    results: list[dict[str, Any]],
    draws: int = 2000,
    seed: int = 0,
) -> tuple[float, float] | None:
    """Bootstrap the predicted total over the raw per-round PAIRED samples.

    A median times an exact call count is not an exact product: the median has
    a sampling distribution and many of them are being summed.  Reporting
    '10.2%' without a band asserts more than a handful of samples per signature
    supports.

    Each draw resamples ROUND INDICES (not native and winner independently)
    for every usable signature, so the native/winner relationship measured
    in each real round is preserved in every bootstrap draw -- see
    ``_round_aligned_pairs``. A signature with fewer than
    ``MIN_PAIRED_ROUNDS`` real paired rounds is excluded from the interval
    entirely (not silently included with a single round, which would report
    a zero-width "95% CI" from N=1 -- maximal apparent certainty from
    essentially no evidence).

    Returns ``(low, high)`` percentiles, or ``None`` when no signature has
    enough paired ``samples_us`` to bootstrap at all -- every artifact
    predating the E2 sample capture, including the ones the original hand
    calculation was computed from. An interval invented from MAD instead
    would be narrower than the truth and would look like the real thing,
    which is worse than not having one.
    """
    if draws < 2:
        raise ImpactError("draws must be at least 2")
    calls = {o["signature"]: int(o.get("calls", 0)) for o in observations}
    pairs = _usable_pairs(observations, results, calls)
    # A zero-call signature contributes nothing to either total and would
    # otherwise leave an empty `totals` list for that draw (IndexError on
    # the percentile lookup below) if it were the only usable signature.
    pairs = [(count, rounds) for count, rounds in pairs if count > 0]
    if not pairs:
        return None

    # The call weight is bound per signature (it is exact, never resampled);
    # only the ROUND INDEX is resampled, jointly for native and winner.
    rng = random.Random(seed)
    totals: list[float] = []
    for _ in range(draws):
        native_total = tuned_total = 0.0
        for count, rounds in pairs:
            sample = rng.choices(rounds, k=len(rounds))
            native_total += count * statistics.median(a for a, _ in sample)
            tuned_total += count * statistics.median(b for _, b in sample)
        if native_total > 0:
            totals.append(100.0 * (native_total - tuned_total) / native_total)
    if not totals:
        return None
    totals.sort()
    return (totals[int(0.025 * len(totals))], totals[int(0.975 * len(totals))])


def _usable_pairs(observations, results, calls):
    """Collect (count, [(native_round, winner_round), ...]) for every
    signature with at least MIN_PAIRED_ROUNDS real paired rounds.

    Uses the SAME native-name fallback as predicted_saving() (a result
    row's own "native" field, falling back to the observation's declared
    native) -- an earlier version omitted that fallback, so a result row
    that relied on it could silently be excluded here even though
    predicted_saving() counted it (gpt-dev-agent review round 2,
    2026-08-31). Keeping these two functions' inclusion criteria in sync
    is what lets sample_backed_coverage() report a real subset relationship.
    """
    native_names = {
        row["signature"]: row.get("native", "")
        for row in observations
        if isinstance(row.get("signature"), str)
    }
    pairs: list[tuple[int, list[tuple[float, float]]]] = []
    for result in results:
        signature = result.get("signature")
        if not isinstance(signature, str) or signature not in calls:
            continue
        native_name = result.get("native") or native_names.get(signature, "")
        native = _candidate(result, native_name)
        winner = _candidate(result, result.get("winner", ""))
        if native is None or winner is None:
            continue
        if "samples_us" not in native or "samples_us" not in winner:
            continue
        rounds = _round_aligned_pairs(native["samples_us"], winner["samples_us"], signature)
        if len(rounds) < MIN_PAIRED_ROUNDS:
            continue
        pairs.append((int(calls[signature]), rounds))
    return pairs


def sample_backed_coverage(observations: list[dict[str, Any]], results: list[dict[str, Any]]) -> dict[str, Any]:
    """How much of ``predicted_saving``'s point estimate is actually backed
    by the bootstrap CI's evidence, vs signatures counted in the point
    estimate purely from ``median_us`` with no ``samples_us`` at all (or
    too few paired rounds to bootstrap).

    The denominator is ``predicted_saving``'s OWN ``coverage.calls_covered``
    -- the calls actually included in the displayed point estimate -- not
    every recorded call. gpt-dev-agent review round 2, 2026-08-31: an
    earlier version divided by ALL recorded calls (including record-only/
    measurement-only signatures never in the point estimate at all), so a
    signature with 1M record-only calls and a fully sample-backed 10-call
    point estimate would report coverage of ~0.001% instead of 100% --
    exactly backwards from what this function exists to state. Since
    ``_usable_pairs`` now shares the same native-name fallback as
    ``predicted_saving``, sample-backed calls are a real subset of
    point-estimate-covered calls, so this ratio is always in [0, 1].
    """
    calls = {o["signature"]: int(o.get("calls", 0)) for o in observations}
    sample_backed = _usable_pairs(observations, results, calls)
    sample_backed_calls = sum(count for count, _ in sample_backed)
    point_estimate_calls = predicted_saving(observations, results)["coverage"].calls_covered
    return {
        "point_estimate_calls": point_estimate_calls,
        "sample_backed_calls": sample_backed_calls,
        "sample_backed_fraction": (
            sample_backed_calls / point_estimate_calls if point_estimate_calls else 0.0
        ),
        "sample_backed_signature_count": len(sample_backed),
    }


# ------------------------------------------------------------------ power


def repetitions_needed(
    effect_pct: Any,
    spread_pct: Any,
    power: float = 0.8,
    alpha: float = 0.05,
    paired_r: float = 0.0,
) -> int:
    """Repetitions per arm to resolve ``effect_pct`` against ``spread_pct``.

    Run before an A/B, not after one.  ``paired_r`` is the correlation between
    arms when rounds are interleaved within one session, which is the one lever
    that helps: pairing removes the shared machine-state variance, and the
    required n falls by ``(1 - r)``.

    ``spread_pct`` MUST be a per-arm standard deviation (the formula's model
    is Gaussian, roughly-equal-variance data) -- MAD, range, CV, or max-min
    spread are all different statistics and would silently produce a
    precise-looking but meaningless sample count if passed here instead
    (gpt-dev-agent review, 2026-08-31: this was previously undocumented).
    """
    effect = _finite(effect_pct, "effect_pct", positive=False)
    spread = _finite(spread_pct, "spread_pct")
    if effect <= 0:
        raise ImpactError("effect_pct must be positive")
    if not 0.5 <= power < 1.0:
        # Below 0.5, z_b = norm_ppf(power) goes negative and the formula's
        # (z_a + z_b) term is no longer monotonic in the requested power --
        # a caller could get a SMALLER n for a higher-sounding but
        # nonsensical request. No real study design targets < 50% power
        # anyway, so this is a floor, not a restriction anyone should hit
        # (gpt-dev-agent review, 2026-08-31).
        raise ImpactError("power must be in [0.5, 1)")
    if not 0.0 < alpha < 1.0:
        raise ImpactError("alpha must be in (0, 1)")
    if not 0.0 <= paired_r < 1.0:
        raise ImpactError("paired_r must be in [0, 1)")
    # Normal quantiles for the two-sided test, via the exact inverse normal
    # CDF (dependency-free).  ``effect`` and ``spread`` are the validated local
    # values so a rejected input cannot sneak into the formula.
    z_a = _norm_ppf(1.0 - alpha / 2.0)
    z_b = _norm_ppf(power)
    n = 2.0 * ((z_a + z_b) * spread / effect) ** 2
    return math.ceil(max(n * (1.0 - paired_r), 1.0))


def _norm_ppf(p: float) -> float:
    """Inverse normal CDF (Beasley-Springer-Moro via statistics.NormalDist)."""

    # statistics.NormalDist().inv_cdf is exact and dependency-free.
    return statistics.NormalDist().inv_cdf(p)


# ------------------------------------------------------------------ report


def _fmt_ms(us: float) -> str:
    return f"{us / 1000.0:.2f} ms"


def render_report(
    result: dict[str, Any], title: str, interval: tuple[float, float] | None = None,
    interval_coverage: dict[str, Any] | None = None,
) -> str:
    """Render the human-readable impact report (the standing IMPACT.md body)."""
    coverage: Coverage = result["coverage"]
    total_covered_pct = (
        100.0 * coverage.calls_covered / coverage.calls_total
        if coverage.calls_total
        else 0.0
    )
    lines = []
    lines.append(f"Predicted matmul time, {title}")
    lines.append("=" * (len(lines[0]) + 3))
    lines.append(
        f"coverage    {coverage.matched} signatures in both, "
        f"{len(coverage.record_only)} record-only, "
        f"{len(coverage.measurement_only)} measurement-only"
    )
    lines.append(
        f"            {coverage.calls_covered:,} of {coverage.calls_total:,} "
        f"recorded calls covered ({total_covered_pct:.1f}%)"
    )
    lines.append("")
    saving = result["saving_pct"]
    tail = ""
    if interval is not None:
        tail = f"  [{interval[0]:.1f} - {interval[1]:.1f}]"
    lines.append("            native      tuned       saved")
    lines.append(
        f"total       {_fmt_ms(result['native_total_us'])}\n"
        f"            "
        f"{_fmt_ms(result['tuned_total_us'])}\n"
        f"            {saving:.1f}%{tail}"
    )
    if interval is not None and interval_coverage is not None:
        # gpt-dev-agent review, 2026-08-31: the point estimate and the CI
        # can describe DIFFERENT populations -- a high-call signature with
        # only median_us (no raw samples_us) can dominate the point
        # estimate while contributing nothing to the interval. State the
        # gap explicitly rather than let a reader assume the interval
        # covers the whole displayed point.
        lines.append(
            f"            interval covers {interval_coverage['sample_backed_calls']:,} "
            f"of {interval_coverage['point_estimate_calls']:,} point-estimate calls "
            f"({100.0 * interval_coverage['sample_backed_fraction']:.1f}%) across "
            f"{interval_coverage['sample_backed_signature_count']} signature(s) "
            "with raw samples"
        )
    lines.append("")
    lines.append("by family   calls       native      tuned    saved   share of saving")
    total_saved = result["native_total_us"] - result["tuned_total_us"]
    for family in sorted(
        result["by_family"],
        key=lambda f: (
            -(result["by_family"][f]["native_us"] - result["by_family"][f]["tuned_us"])
        ),
    ):
        bucket = result["by_family"][family]
        fam_saved = bucket["native_us"] - bucket["tuned_us"]
        fam_pct = (
            100.0 * fam_saved / bucket["native_us"] if bucket["native_us"] else 0.0
        )
        share = 100.0 * fam_saved / total_saved if total_saved > 0 else 0.0
        lines.append(
            f"{family:<11} {int(bucket['calls']):>9,}"
            f"   {_fmt_ms(bucket['native_us']):>11}"
            f"   {_fmt_ms(bucket['tuned_us']):>10}"
            f"   {fam_pct:5.1f}%   {share:5.1f}%"
        )
    lines.append("")
    lines.append("top contributors (by absolute time saved, not by margin)")
    for row in result["rows"][:10]:
        if row["saved_us"] <= 0:
            continue
        share = 100.0 * row["saved_us"] / total_saved if total_saved > 0 else 0.0
        lines.append(
            f"  {row['signature'][:16]}  {row['calls']:>7,} x "
            f"{row['saved_us_each']:7.2f} us saved = "
            f"{_fmt_ms(row['saved_us']):>9}   {share:4.1f}% of total"
        )
    lines.append("")
    lines.append(
        f"signatures where the tuned winner is predicted SLOWER: {len(result['slower'])}"
    )
    return "\n".join(lines) + "\n"


# ------------------------------------------------------------------ CLI


def _cmd_impact(args: argparse.Namespace) -> int:
    try:
        observations = load_observations(args.observations)
        results = load_results(args.measurements)
        report = predicted_saving(observations, results)
        interval = saving_interval(
            observations, results, draws=args.draws, seed=args.seed
        )
        interval_coverage = (
            sample_backed_coverage(observations, results) if interval is not None else None
        )
    except (ImpactError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    text = render_report(report, args.title, interval, interval_coverage)
    if args.report:
        Path(args.report).write_text(text, encoding="utf-8")
    print(text, end="")
    if report["slower"] and args.fail_on_slower:
        print(
            f"error: {len(report['slower'])} signature(s) predict a slower "
            "tuned total; refusing to report a netted positive",
            file=sys.stderr,
        )
        return 1
    if report["slower"] and not args.fail_on_slower:
        print(
            f"warning: {len(report['slower'])} signature(s) predict a slower "
            "tuned total (see SLOWER count above)",
            file=sys.stderr,
        )
    return 0


def _cmd_power(args: argparse.Namespace) -> int:
    try:
        n = repetitions_needed(
            args.effect,
            args.spread,
            power=args.power,
            alpha=args.alpha,
            paired_r=args.paired_r,
        )
    except ImpactError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(f"repetitions per arm: {n}")
    print(
        f"  effect={args.effect:g}% spread={args.spread:g}% "
        f"alpha={args.alpha:g} power={args.power:g} paired_r={args.paired_r:g}"
    )
    return 0


def build_parser(subparsers) -> None:
    """Register the ``impact`` and ``power`` subcommands (HI35)."""
    impact_cmd = subparsers.add_parser(
        "impact", help="call-weighted predicted matmul saving (record x measurements)"
    )
    impact_cmd.add_argument(
        "--observations",
        required=True,
        help="record JSONL (header + observation rows with calls)",
    )
    impact_cmd.add_argument(
        "--measurements", required=True, help="measurements JSONL (tuning results)"
    )
    impact_cmd.add_argument(
        "--title", default="tuning run", help="workload title for the report header"
    )
    impact_cmd.add_argument(
        "--report", default=None, help="write the rendered report to this path"
    )
    impact_cmd.add_argument(
        "--draws",
        type=int,
        default=2000,
        help="bootstrap draws for the saving interval",
    )
    impact_cmd.add_argument("--seed", type=int, default=0, help="bootstrap RNG seed")
    impact_cmd.add_argument(
        "--fail-on-slower",
        action="store_true",
        help="exit non-zero if any signature predicts a slower tuned total",
    )
    impact_cmd.set_defaults(func=_cmd_impact)

    power_cmd = subparsers.add_parser(
        "power", help="repetitions per A/B arm to resolve an effect against spread"
    )
    power_cmd.add_argument(
        "--effect",
        type=float,
        required=True,
        help="expected end-to-end effect, percent",
    )
    power_cmd.add_argument(
        "--spread", type=float, required=True,
        help=(
            "observed benchmark STANDARD DEVIATION per arm, percent -- the "
            "formula assumes this, not MAD/range/CV/max-min spread; passing "
            "one of those instead yields a precise-looking but meaningless "
            "sample count"
        ),
    )
    power_cmd.add_argument("--alpha", type=float, default=0.05)
    power_cmd.add_argument("--power", type=float, default=0.8)
    power_cmd.add_argument(
        "--paired-r",
        type=float,
        default=0.0,
        help="between-arm correlation from interleaved pairing, [0,1)",
    )
    power_cmd.set_defaults(func=_cmd_power)
