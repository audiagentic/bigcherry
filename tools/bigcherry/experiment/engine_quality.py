"""Engine-neutral quality and activation evidence for an A/B arm (RAD01 PR 7).

Two pieces every engine's campaign needs and each engine reports in its own words:

* activation: did the switched mechanism run? A component setting may declare an ``activation`` regular expression
  (component.toml); the server log of the arm that sets it must match, and the control's must not.
* quality: a full-distribution comparison against a recorded reference. radiance's KL mode
  (``--kld-record`` / ``--kld-ref`` / ``--kld-out``) writes a JSON document whose ``all`` section becomes a
  :class:`KldResult`, and :func:`quality_gate` decides pass/fail from explicit thresholds.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from ..core.components import Component
from ..patch.activation import ActivationEvidence

class QualityParseError(ValueError):
    """The KL output is not a complete KL result."""


@dataclass(frozen=True)
class KldResult:
    kld_mean: float
    kld_median: float
    kld_p99: float
    kld_p999: float
    top1_percent: float
    ppl: float
    ppl_reference: float

    @property
    def ppl_ratio(self) -> float:
        return self.ppl / self.ppl_reference


#: the --kld-out JSON keys of the 'all' section that make a KldResult (radiance 1.3.x: mean/median/p90/p95/p99/p99_9/
#: p99_99/max, top1_agreement, ppl); the reference perplexity is read from the first of these that is present
_ALL_REQUIRED = ("mean", "median", "p99", "p99_9", "top1_agreement", "ppl")
_PPL_REFERENCE_KEYS = ("ppl_ref", "ppl_reference", "ref_ppl")


def parse_radiance_kld_json(document: dict[str, Any]) -> KldResult:
    """The overall KL result of one radiance ``--kld-ref ... --kld-out`` run (the document's ``all`` section).

    ``top1_agreement`` is a percentage, as in the summary log line. The per-source section (``by_source``) is not
    gated; it stays in the document the caller records.
    """
    section = document.get("all") if isinstance(document, dict) else None
    if not isinstance(section, dict):
        raise QualityParseError("KL output has no 'all' section")
    missing = [key for key in _ALL_REQUIRED if not isinstance(section.get(key), (int, float))]
    reference_key = next((key for key in _PPL_REFERENCE_KEYS if isinstance(section.get(key), (int, float))), None)
    if missing or reference_key is None:
        wanted = [*missing, *([] if reference_key else ["/".join(_PPL_REFERENCE_KEYS)])]
        raise QualityParseError(f"KL output 'all' section lacks {', '.join(wanted)}; has {sorted(section)}")
    return KldResult(
        kld_mean=float(section["mean"]),
        kld_median=float(section["median"]),
        kld_p99=float(section["p99"]),
        kld_p999=float(section["p99_9"]),
        top1_percent=float(section["top1_agreement"]),
        ppl=float(section["ppl"]),
        ppl_reference=float(section[reference_key]),
    )


@dataclass(frozen=True)
class TokenKld:
    """Distribution of per-token KL against a reference, over all tokens or a subset of them."""

    count: int
    mean: float
    median: float
    p99: float
    p999: float
    top1_percent: float


def _percentile(sorted_values: list[float], q: float) -> float:
    """Linear-interpolated percentile (q in 0..100) of an ascending list."""
    rank = (len(sorted_values) - 1) * q / 100.0
    low = int(rank)
    high = min(low + 1, len(sorted_values) - 1)
    return sorted_values[low] + (sorted_values[high] - sorted_values[low]) * (rank - low)


def token_kld(kl: list[float], top1_agrees: list[bool], subset: list[bool] | None = None) -> TokenKld:
    """Mean, median, P99, P99.9 and top-1 agreement from per-token data.

    ``kl[i]`` is the KL divergence at token i, ``top1_agrees[i]`` whether the arm's most likely token equals the
    reference's, and ``subset[i]`` (when given) whether token i counts, for example text and assistant turns only.
    radiance 1.3.0 prints only a summary today, so nothing feeds this yet; see the RAD08 report.
    """
    if len(kl) != len(top1_agrees) or (subset is not None and len(subset) != len(kl)):
        raise ValueError("kl, top1_agrees and subset must be the same length")
    keep = range(len(kl)) if subset is None else [i for i, on in enumerate(subset) if on]
    values = sorted(kl[i] for i in keep)
    if not values:
        raise ValueError("no tokens selected")
    hits = sum(1 for i in keep if top1_agrees[i])
    return TokenKld(
        count=len(values),
        mean=sum(values) / len(values),
        median=_percentile(values, 50),
        p99=_percentile(values, 99),
        p999=_percentile(values, 99.9),
        top1_percent=100.0 * hits / len(values),
    )


@dataclass(frozen=True)
class QualityThresholds:
    """What an arm may lose against its control's distance to the same reference."""

    max_kld_mean_increase: float
    max_kld_median_increase: float
    max_kld_p99_increase: float
    max_kld_p999_increase: float
    max_top1_drop_points: float
    max_ppl_ratio_increase: float


def quality_gate(control: KldResult, subject: KldResult, limits: QualityThresholds) -> tuple[bool, list[str]]:
    """Pass when the subject is no further from the reference than the control by more than the limits."""
    problems = []
    if subject.kld_mean - control.kld_mean > limits.max_kld_mean_increase:
        problems.append(f"KLD mean {control.kld_mean:.6f} -> {subject.kld_mean:.6f}")
    for label, name, limit in (
        ("median", "kld_median", limits.max_kld_median_increase),
        ("P99", "kld_p99", limits.max_kld_p99_increase),
        ("P99.9", "kld_p999", limits.max_kld_p999_increase),
    ):
        before, after = getattr(control, name), getattr(subject, name)
        if after - before > limit:
            problems.append(f"KLD {label} {before:.6f} -> {after:.6f}")
    if control.top1_percent - subject.top1_percent > limits.max_top1_drop_points:
        problems.append(f"top-1 {control.top1_percent:.3f}% -> {subject.top1_percent:.3f}%")
    if subject.ppl_ratio - control.ppl_ratio > limits.max_ppl_ratio_increase:
        problems.append(f"PPL ratio {control.ppl_ratio:.4f} -> {subject.ppl_ratio:.4f}")
    return (not problems, problems)


def setting_activation(component: Component, setting: str, *, control_log: str, subject_log: str) -> ActivationEvidence:
    """Activation evidence for one component setting from the two arms' server logs."""
    declared = component.settings.get(setting)
    if declared is None:
        raise KeyError(f"{setting} is not a setting {component.name} declares")
    if not declared.activation:
        return ActivationEvidence("unobservable", f"{component.name}:{setting}", "no activation marker declared")
    marker = re.compile(declared.activation)
    in_control = bool(marker.search(control_log))
    in_subject = bool(marker.search(subject_log))
    detail = f"marker {declared.activation!r}: control {in_control}, subject {in_subject}"
    if in_subject and not in_control:
        return ActivationEvidence("executed", f"{component.name}:{setting}", detail)
    return ActivationEvidence("not_executed", f"{component.name}:{setting}", detail)
