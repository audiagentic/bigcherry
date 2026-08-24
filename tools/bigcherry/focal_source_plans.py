"""Explicit focal-patch comparison construction: BASELINE / CONTROL /
SUBJECT / STOCK source compositions (patch-system PA02 / RS06).

Runbook sections 35-37 (docs/planning/active/patch-system/
PATCH_REFACTOR_RUNBOOK.md). The implicit "baseline = every patch whose
STATE == validated" definition is retired as the scientific attribution
baseline: the caller passes an explicitly resolved baseline composition
(the named BigCherry source composition from the canonical
campaign/source configuration), and this module derives:

    CONTROL = baseline + focal's prerequisites (without the focal)
    SUBJECT = baseline + the same prerequisites + the focal
    STOCK   = ()  (pristine pinned llama.cpp -- contextual only)

Authoritative attribution is SUBJECT vs CONTROL (section 36). If the
focal is not independently isolatable -- it is already in the baseline,
another baseline patch depends on it, it conflicts with the baseline or
its prerequisites, or a prerequisite is rejected -- the result is
BLOCKED, never a silent substitution or silent drop of a dependent
baseline patch (section 37).

This module is a pure function over the registry + patchset resolution
APIs: it never materializes anything.
"""

from __future__ import annotations

from dataclasses import dataclass

from . import patch_registry, patchset


class FocalComparisonError(ValueError):
    """Raised for structurally impossible inputs (unknown focal, unknown
    baseline members, internal invariant violations)."""


@dataclass(frozen=True)
class FocalComparison:
    """One explicit focal comparison. When ``blocked_reason`` is set,
    ``control``/``subject`` are empty: a blocked comparison must not hand
    a caller a composition that silently drops or substitutes patches."""

    focal: str
    baseline: tuple[str, ...]
    control: tuple[str, ...]
    subject: tuple[str, ...]
    prerequisites: tuple[str, ...]
    stock: tuple[str, ...]
    blocked_reason: str | None = None

    @property
    def is_blocked(self) -> bool:
        return self.blocked_reason is not None


def _ordered(registry: patch_registry.PatchRegistry, ids: set[str]) -> tuple[str, ...]:
    known = registry.by_id
    return tuple(sorted(ids, key=lambda patch_id: (known[patch_id].order, patch_id)))


def build_focal_comparison(
    focal: str,
    baseline: tuple[str, ...] | list[str],
    *,
    registry: patch_registry.PatchRegistry | None = None,
    root: "object" = None,
) -> FocalComparison:
    """Derive CONTROL/SUBJECT for focal patch ``focal`` over the explicit
    ``baseline`` composition.

    ``root`` defaults to ``paths.PATCHES`` when ``registry`` is None.
    Dependency closure uses the ordinary ``REQUIRES`` semantics
    (``patchset.expand_composition``), fail-closed.
    """
    if registry is None:
        from . import paths

        registry = patch_registry.load_registry(paths.PATCHES if root is None else root)
    try:
        focal_descriptor = registry.get(focal)
    except patch_registry.PatchRegistryError as exc:
        raise FocalComparisonError(f"unknown focal patch: {focal!r}") from exc

    baseline_ids = set(baseline)
    unknown_baseline = sorted(baseline_ids - set(registry.by_id))
    if unknown_baseline:
        raise FocalComparisonError(
            f"baseline contains unknown patch(es): {', '.join(unknown_baseline)}"
        )

    def blocked(reason: str) -> FocalComparison:
        return FocalComparison(
            focal=focal,
            baseline=tuple(baseline),
            control=(),
            subject=(),
            prerequisites=(),
            stock=(),
            blocked_reason=reason,
        )

    if focal in baseline_ids:
        return blocked("focal patch is already in the baseline composition")

    # Ordinary dependency closure of the focal (fail-closed on unknown
    # dependencies or REQUIRES cycles -- tree errors, raised, not
    # swallowed into a BLOCKED result).
    closure = patchset.expand_composition((focal,), directory=registry.root).expanded
    prerequisites = set(closure) - {focal}

    rejected = sorted(
        patch_id for patch_id in prerequisites
        if registry.by_id[patch_id].state == "rejected"
    )
    if rejected:
        return blocked(
            "prerequisite(s) in rejected state cannot form a controlled "
            f"composition: {', '.join(rejected)}"
        )

    conflicts = sorted(set(focal_descriptor.conflicts) & (baseline_ids | prerequisites))
    if conflicts:
        return blocked(
            f"focal conflicts with composition member(s): {', '.join(conflicts)}"
        )

    # If any baseline patch transitively requires the focal, the baseline
    # is not a valid control foundation (removing the focal would
    # invalidate those patches). BLOCKED -- never silently drop them.
    for patch_id in sorted(baseline_ids):
        member_closure = patchset.expand_composition(
            (patch_id,), directory=registry.root
        ).expanded
        if focal in member_closure:
            return blocked(
                f"baseline patch {patch_id!r} depends on the focal patch -- "
                "focal patch not independently isolatable"
            )

    # Normalize both compositions after dependency closure (runbook RS06
    # assertion wording): a control that carried a patch with an unmet
    # dependency would be a different tree than the subject, breaking the
    # diff.
    base_set = baseline_ids | prerequisites
    control_closed = patchset.expand_composition(
        tuple(sorted(base_set)), directory=registry.root
    ).expanded
    subject_closed = patchset.expand_composition(
        tuple(base_set | {focal}), directory=registry.root
    ).expanded
    control = _ordered(registry, set(control_closed))
    subject = _ordered(registry, set(subject_closed))

    if set(subject) - set(control) != {focal}:
        raise FocalComparisonError(
            "internal invariant violated: subject - control != {focal}"
        )

    return FocalComparison(
        focal=focal,
        baseline=tuple(baseline),
        control=control,
        subject=subject,
        prerequisites=tuple(_ordered(registry, prerequisites)),
        stock=(),
    )
