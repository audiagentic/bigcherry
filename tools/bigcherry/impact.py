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

        Reuses the record parser so the impact tool and the inventory tool cannot
    drift apart on the wire format.  A missing header is an error, not an empty
    result: an empty observations list would read as 'zero calls' rather than as
    'a file this loader cannot read'.
    """
    from . import inventory  # local: inventory pulls in a large dependency tree

    record = inventory.read_jsonl(Path(path))
    return record.observations


def load_results(path: Path) -> list[dict[str, Any]]:
    """Read a measurements JSONL and return its result rows only.

    Header rows and any non-``result`` rows are dropped.  A truncated final
    line is tolerated (a run killed mid-flush leaves one) rather than fatal.
    """
    results: list[dict[str, Any]] = []
    with Path(path).open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                print(
                    f"warning: {Path(path).name} line {number} is truncated or "
                    "malformed; ignoring it and everything after",
                    file=sys.stderr,
                )
                break
            if row.get("kind") == "result":
                results.append(row)
    return results


# ------------------------------------------------------------------ interval


def _finite_samples(samples: Any, name: str) -> list[float]:
    """Strip the nulls out of a ``samples_us`` list.

    ``samples_us`` is round-aligned and NaN-padded on the C++ side, serialised
    as JSON ``null`` for a round where the candidate failed to launch.
    The padding exists so the paired sign test compares round r against round
    r; a bootstrap over medians does not need the alignment and must not
    resample the nulls.
    """
    if not isinstance(samples, (list, tuple)):
        raise ImpactError(f"{name} samples_us must be a list")
    cleaned = [s for s in samples if s is not None]
    if not cleaned:
        raise ImpactError(f"{name} has no usable samples_us (all null)")
    return [_finite(s, f"{name} sample", positive=True) for s in cleaned]


def saving_interval(
    observations: list[dict[str, Any]],
    results: list[dict[str, Any]],
    draws: int = 2000,
    seed: int = 0,
) -> tuple[float, float] | None:
    """Bootstrap the predicted total over the raw per-round samples.

    A median times an exact call count is not an exact product: the median has
    a sampling distribution and many of them are being summed.  Reporting
    '10.2%' without a band asserts more than a handful of samples per signature
    supports.

    Returns ``(low, high)`` percentiles, or ``None`` when the file carries no
    ``samples_us`` -- every artifact predating the E2 sample capture, including
    the ones the original hand calculation was computed from.  An interval
    invented from MAD instead would be narrower than the truth and would look
    like the real thing, which is worse than not having one.
    """
    if draws < 2:
        raise ImpactError("draws must be at least 2")
    calls = {o["signature"]: int(o.get("calls", 0)) for o in observations}
    pairs = _usable_pairs(observations, results, calls)
    if not pairs:
        return None

    # Each draw resamples the native and winner round samples for every usable
    # signature, multiplies each resampled median by that signature's exact
    # call count, and forms the call-weighted saving.  The call weight is bound
    # per signature (it is exact), never resampled.
    rng = random.Random(seed)
    totals: list[float] = []
    for _ in range(draws):
        native_total = tuned_total = 0.0
        for count, native_samples, winner_samples in pairs:
            native_total += count * statistics.median(
                rng.choices(native_samples, k=len(native_samples))
            )
            tuned_total += count * statistics.median(
                rng.choices(winner_samples, k=len(winner_samples))
            )
        if native_total > 0:
            totals.append(100.0 * (native_total - tuned_total) / native_total)
    totals.sort()
    return (totals[int(0.025 * len(totals))], totals[int(0.975 * len(totals))])


def _usable_pairs(observations, results, calls):
    """Collect (count, native_samples, winner_samples) for every usable signature."""
    pairs: list[tuple[int, list[float], list[float]]] = []
    for result in results:
        signature = result.get("signature")
        if not isinstance(signature, str) or signature not in calls:
            continue
        native = _candidate(result, result.get("native") or "")
        winner = _candidate(result, result.get("winner", ""))
        if native is None or winner is None:
            continue
        if "samples_us" not in native or "samples_us" not in winner:
            continue
        pairs.append(
            (
                int(calls[signature]),
                _finite_samples(native["samples_us"], "native"),
                _finite_samples(winner["samples_us"], "winner"),
            )
        )
    return pairs


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
    """
    effect = _finite(effect_pct, "effect_pct", positive=False)
    spread = _finite(spread_pct, "spread_pct")
    if effect <= 0:
        raise ImpactError("effect_pct must be positive")
    if not 0.0 < power < 1.0:
        raise ImpactError("power must be in (0, 1)")
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
    result: dict[str, Any], title: str, interval: tuple[float, float] | None = None
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
    except (ImpactError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    text = render_report(report, args.title, interval)
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
        "--spread", type=float, required=True, help="observed benchmark spread, percent"
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
