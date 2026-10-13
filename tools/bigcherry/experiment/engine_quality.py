"""Engine-neutral quality and activation evidence for an A/B arm (RAD01 PR 7).

Two pieces every engine's campaign needs and each engine reports in its own words:

* activation: did the switched mechanism run? A component setting may declare an ``activation`` regular expression
  (component.toml); the server log of the arm that sets it must match, and the control's must not.
* quality: a full-distribution comparison against a recorded reference. radiance's KL mode
  (``--kld-record`` / ``--kld-ref``) prints ``I kld: KLD mean ...`` and ``I kld: top-1 agreement ...``; those lines
  become a :class:`KldResult`, and :func:`quality_gate` decides pass/fail from explicit thresholds.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from ..core.components import Component
from ..patch.activation import ActivationEvidence

_KLD = re.compile(
    r"KLD mean (?P<mean>[0-9.]+)\s+median (?P<median>[0-9.]+)\s+p99 (?P<p99>[0-9.]+)"
)
_TOP1 = re.compile(
    r"top-1 agreement (?P<top1>[0-9.]+)%\s+PPL (?P<ppl>[0-9.]+) vs reference (?P<ppl_ref>[0-9.]+)"
)


class QualityParseError(ValueError):
    """The log does not carry exactly one complete KL result."""


@dataclass(frozen=True)
class KldResult:
    kld_mean: float
    kld_median: float
    kld_p99: float
    top1_percent: float
    ppl: float
    ppl_reference: float

    @property
    def ppl_ratio(self) -> float:
        return self.ppl / self.ppl_reference


def parse_radiance_kld(log: str) -> KldResult:
    """The KL result of one radiance ``--kld-ref`` run. Exactly one KLD line and one top-1 line, or an error."""
    klds = list(_KLD.finditer(log))
    tops = list(_TOP1.finditer(log))
    if len(klds) != 1 or len(tops) != 1:
        raise QualityParseError(f"expected one KLD and one top-1 line, found {len(klds)} and {len(tops)}")
    k, t = klds[0], tops[0]
    return KldResult(
        kld_mean=float(k["mean"]),
        kld_median=float(k["median"]),
        kld_p99=float(k["p99"]),
        top1_percent=float(t["top1"]),
        ppl=float(t["ppl"]),
        ppl_reference=float(t["ppl_ref"]),
    )


@dataclass(frozen=True)
class QualityThresholds:
    """What an arm may lose against its control's distance to the same reference."""

    max_kld_mean_increase: float
    max_top1_drop_points: float
    max_ppl_ratio_increase: float


def quality_gate(control: KldResult, subject: KldResult, limits: QualityThresholds) -> tuple[bool, list[str]]:
    """Pass when the subject is no further from the reference than the control by more than the limits."""
    problems = []
    if subject.kld_mean - control.kld_mean > limits.max_kld_mean_increase:
        problems.append(f"KLD mean {control.kld_mean:.6f} -> {subject.kld_mean:.6f}")
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
