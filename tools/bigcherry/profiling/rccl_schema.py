"""GP07: shared RCCL compatibility-identity model.

``rccl_qualify.py`` (single-case harness) and ``rccl_qualify_campaign.py``
(matrix driver) both import ``RcclCompatibilityRevision`` from here so a
qualification result's admissibility can be scoped to the EXACT RCCL build
that produced it, per the runbook's P2.1 requirement
(docs/reference/testing/RCCL_HETEROGENEOUS_RUNBOOK.md).

This exists because GP06 (docs/planning/active/gpu-collectives/GP06.md)
found real hardware evidence that a topology's RCCL viability is NOT
portable across RCCL builds that report similar version strings: RCCL
2.30.4 from two independently-built ROCm installs (a TheRock dev build and
a stable ROCm 10.0.0 install) both regress the {0,2}/{1,2} topology class
that RCCL 1.0.70204 qualifies cleanly. A qualification result recorded
without this identity cannot be trusted to still hold for "the same RCCL
version" observed on a different install.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RcclCompatibilityRevision:
    """Binds a qualification (or benchmark) result to the exact RCCL
    implementation actually exercised.

    Must be recorded at the time the RCCL binary is actually run -- never
    inferred or assumed from a nearby ROCm package version alone (GP06's
    finding is precisely that two installs reporting "RCCL 2.30.4" are not
    interchangeable evidence).
    """

    # Reported by the library itself (e.g. via NCCL_DEBUG=INFO's own
    # version banner, or ncclGetVersion) -- always required, since it is
    # the one fact obtainable from every install without extra tooling.
    rccl_version: str

    # Exact git commit/tag of the RCCL source tree, when known -- this is
    # the REAL identity per the runbook's P2.1 (a version string alone is
    # not sufficient, per GP06). Prefer this over rccl_version whenever
    # both are available.
    rccl_source_revision: str | None = None

    # Local disambiguator only (e.g. "vendor/rocm/7.2.4") -- NEVER treated
    # as portable identity by itself; two hosts could use the same label
    # for genuinely different installs. Useful for humans re-finding the
    # exact install a result came from on the box it was captured on.
    rocm_install_label: str | None = None

    # Build configuration relevant to device kernels, when known (e.g.
    # "COLLTRACE=OFF,Release", target archs covered).
    build_config: str | None = None

    @property
    def revision_id(self) -> str:
        """Stable string key for grouping/lookup in qualification and
        benchmark artifacts. Prefers the real source revision; falls back
        to the packaged version string only when source revision isn't
        known (never silently drop identity -- an empty/unknown
        compatibility revision must not collide with a real one)."""
        if self.rccl_source_revision:
            return self.rccl_source_revision
        return self.rccl_version

    def to_json(self) -> dict:
        return {
            "rccl_version": self.rccl_version,
            "rccl_source_revision": self.rccl_source_revision,
            "rocm_install_label": self.rocm_install_label,
            "build_config": self.build_config,
        }
