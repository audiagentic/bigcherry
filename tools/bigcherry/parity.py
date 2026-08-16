"""RE14 cutover parity gate: legacy mutable-checkout output vs new-path output.

Follows the same explicit-allowed-differences, fail-on-undeclared shape as
``comparisons.plan_pair`` (the existing benchmark-arm identity comparator),
but compares build-time identity artifacts (manifest, descriptor, candidate
set, binary) rather than runtime benchmark arms. The two comparators are
deliberately not merged: one gates "did cutover preserve what the build
produces", the other gates "is a benchmark pairing legitimate to report" --
different questions, at different pipeline stages, on different objects.

Candidate coverage is checked by exact stable-name SET equality, not count
equality: RE09's investigation into replay-full found a real case where two
candidate sets of equal size could differ in membership (a missing candidate
alongside an unexpected one), which a bare count comparison would not catch.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


class ParityError(ValueError):
    pass


@dataclass(frozen=True)
class CampaignArm:
    """One build's identity artifacts, from either the legacy or new path."""

    name: str
    manifest: dict[str, Any]
    descriptor: dict[str, Any]
    candidate_names: frozenset[str]
    binary_hash: str


@dataclass(frozen=True)
class ParityReport:
    left: str
    right: str
    label: str
    allowed_differences: frozenset[str]
    missing_candidates: frozenset[str]
    """Candidates ``left`` has that ``right`` lacks."""
    extra_candidates: frozenset[str]
    """Candidates ``right`` has that ``left`` lacks."""


_MANIFEST_FIELDS = ("source_revision", "variant_set", "manifest_hash")
_DESCRIPTOR_FIELDS = ("descriptor_hash",)


def check_parity(
    left: CampaignArm,
    right: CampaignArm,
    *,
    label: str,
    allowed_differences: frozenset[str] = frozenset(),
) -> ParityReport:
    """Compare two campaign arms' build-identity artifacts.

    Raises :class:`ParityError` naming every undeclared difference found.
    A caller proving RE14 cutover parity should pass ``allowed_differences
    = frozenset()`` -- an empty set is the actual acceptance bar; widening
    it is for known-and-accepted divergence (e.g. comparing across variant
    sets on purpose), not a default to reach for.
    """
    differences: set[str] = set()

    for field in _MANIFEST_FIELDS:
        if left.manifest.get(field) != right.manifest.get(field):
            differences.add(f"manifest.{field}")
    for field in _DESCRIPTOR_FIELDS:
        if left.descriptor.get(field) != right.descriptor.get(field):
            differences.add(f"descriptor.{field}")
    if left.binary_hash != right.binary_hash:
        differences.add("binary_hash")

    missing = left.candidate_names - right.candidate_names
    extra = right.candidate_names - left.candidate_names
    if missing or extra:
        differences.add("candidate_names")

    undeclared = differences - set(allowed_differences)
    if undeclared:
        detail = [f"  - {name}" for name in sorted(undeclared)]
        if "candidate_names" in undeclared:
            if missing:
                detail.append(f"    missing from {right.name!r}: {sorted(missing)}")
            if extra:
                detail.append(f"    unexpected in {right.name!r}: {sorted(extra)}")
        raise ParityError(
            f"parity {label!r} has undeclared difference(s) between "
            f"{left.name!r} and {right.name!r}:\n" + "\n".join(detail)
        )

    return ParityReport(
        left=left.name,
        right=right.name,
        label=label,
        allowed_differences=allowed_differences,
        missing_candidates=frozenset(missing),
        extra_candidates=frozenset(extra),
    )
