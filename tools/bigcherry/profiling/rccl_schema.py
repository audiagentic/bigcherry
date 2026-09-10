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

gpt-dev-agent review (2026-09-02, request req_acf7c8da985f4f17) on the
first version of this module: falling back to a bare ``rccl_version``
string for ``revision_id`` "recreates the exact GP06 failure mode" --
GP06's whole finding is that two installs reporting the same version
string are NOT interchangeable. Fixed below: ``revision_id`` now requires
a genuinely durable identity component (``rccl_source_revision`` or
``library_build_id``) and raises rather than silently falling back to the
bare version string.
"""

from __future__ import annotations

from dataclasses import dataclass, field


class InsufficientCompatibilityIdentity(ValueError):
    """Raised when a RcclCompatibilityRevision has no durable identity
    component -- a bare version string is not enough (GP06's own finding:
    two installs reporting the same version string regressed differently)."""


@dataclass(frozen=True)
class RcclCompatibilityRevision:
    """Binds a qualification (or benchmark) result to the exact RCCL
    implementation actually exercised.

    Must be recorded at the time the RCCL binary is actually run -- never
    inferred or assumed from a nearby ROCm package version alone.
    """

    # Reported by the library itself (e.g. via NCCL_DEBUG=INFO's own
    # version banner, or ncclGetVersion). Always required as human-
    # readable metadata, but NOT by itself sufficient identity -- see
    # revision_id below.
    rccl_version: str

    # Exact git commit/tag of the RCCL source tree, when known.
    rccl_source_revision: str | None = None

    # SHA256 (or equivalent build-id) of the actual librccl.so linked at
    # run time -- the strongest available identity when a source revision
    # isn't recorded (e.g. a packaged install with no attached git
    # history). Prefer this over rccl_source_revision when both are
    # available and they might legitimately disagree (a rebuild from the
    # same source revision with different build config still produces a
    # different code object).
    library_build_id: str | None = None

    # ABI/API compatibility revision, when the RCCL build exposes one
    # distinct from its marketing version string.
    abi_revision: str | None = None

    # Real architecture/code-object coverage actually compiled into this
    # build (e.g. ("gfx1100", "gfx1201", "gfx1030")) -- verified via
    # clang-offload-bundler --list or equivalent, not assumed from the
    # AMDGPU_TARGETS build flag alone (a build can be requested for an
    # arch and silently not cover it -- see HI138's own P1.2 caution).
    code_object_arches: tuple[str, ...] = ()

    # Local disambiguator only (e.g. "vendor/rocm/7.2.4") -- NEVER treated
    # as portable identity by itself; two hosts could use the same label
    # for genuinely different installs. Useful for humans re-finding the
    # exact install a result came from on the box it was captured on.
    rocm_install_label: str | None = None

    # Build configuration relevant to device kernels, when known (e.g.
    # "COLLTRACE=OFF,Release").
    build_config: str | None = None

    @property
    def revision_id(self) -> str:
        """Stable string key for grouping/lookup in qualification and
        benchmark artifacts.

        Requires a genuinely durable identity component -- prefers
        ``library_build_id`` (the strongest: identifies the exact code
        object, immune to "same source, different build config" drift),
        falls back to ``rccl_source_revision``. Raises
        ``InsufficientCompatibilityIdentity`` rather than silently
        falling back to the bare version string: GP06's own finding is
        that two installs reporting the identical version string
        ("2.30.4") regressed a topology differently, so treating the
        version string as sufficient identity would silently recreate
        that exact failure mode in a future qualification/benchmark
        lookup.
        """
        if self.library_build_id:
            return self.library_build_id
        if self.rccl_source_revision:
            return self.rccl_source_revision
        raise InsufficientCompatibilityIdentity(
            f"RcclCompatibilityRevision(rccl_version={self.rccl_version!r}) has "
            "no durable identity (library_build_id or rccl_source_revision) -- "
            "a version string alone is not sufficient identity (GP06 found two "
            "installs reporting the same version regress differently). Record "
            "the exact librccl.so build-id or source commit before using this "
            "as a qualification/benchmark key."
        )

    def to_json(self) -> dict:
        return {
            "rccl_version": self.rccl_version,
            "rccl_source_revision": self.rccl_source_revision,
            "library_build_id": self.library_build_id,
            "abi_revision": self.abi_revision,
            "code_object_arches": list(self.code_object_arches),
            "rocm_install_label": self.rocm_install_label,
            "build_config": self.build_config,
        }


def qualification_key(compatibility: RcclCompatibilityRevision, case_id: str) -> str:
    """The composite identity a qualification (or benchmark) result is
    actually keyed by: exact RCCL revision + exact case. Never conflate
    the two into one string field on a result dataclass -- keep them
    separable (compatibility.revision_id, case_id) for filtering/grouping,
    but use this composite whenever a single lookup/comparison key is
    needed (e.g. GP08's qualification_id gate)."""
    return f"{compatibility.revision_id}::{case_id}"
