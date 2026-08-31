"""Per-patch SUMMARY.md rendering and release-doc merging.

Each patch package directory carries a short ``SUMMARY.md`` (see
``patches/_template/SUMMARY.md`` for the required shape: What it does / Why
/ Upstream, plus a Status/Group/Plan item header) -- the human-readable
counterpart to ``patch.toml``, which is the sole machine-metadata authority
for a packaged patch's state/group/plan-item (registry.py's
``_packaged_descriptor``; ``patch.py``'s own STATE/GROUP constants are only
read for the legacy flat-module shape, not for packaged patches -- see
``check_summary_consistency``'s docstring for why this distinction matters).
This module merges the SUMMARY.md of every patch in a given selection into
one release doc, alongside the llama.cpp pin it was built against -- so a
release has one document that says exactly what patches it carries and why,
not just a revision number.

Deliberately does not require every patch to have a SUMMARY.md when
rendering the placeholder path (``read_patch_summary``): a missing file
renders a visible placeholder rather than failing the merge, since a
release doc that silently omits an undocumented patch is worse than one
that flags it. An UNRESOLVABLE patch id (not present in the registry at
all) is a different, harder failure -- ``resolve_patch_descriptors`` raises
rather than silently dropping it, so a caller's N-patch selection can never
quietly render as an M<N-patch doc (gpt-dev-agent review, 2026-08-31).
"""

from __future__ import annotations

import re
from pathlib import Path

from ..core import paths
from . import registry as patch_registry

SUMMARY_FILENAME = "SUMMARY.md"

# Each field is anchored to its own line (^...$ with MULTILINE, not a bare
# \s* that could cross into the next line) so a blank line between fields,
# or a second/duplicate header block later in the file, cannot be silently
# absorbed into a match (gpt-dev-agent review, 2026-08-31).
_HEADER_PATTERN = re.compile(
    r"^\*\*Status:\*\*[ \t]*(?P<status>\S+)[ \t]*$\n"
    r"^\*\*Group:\*\*[ \t]*(?P<group>\S+)[ \t]*$\n"
    r"^\*\*Plan item:\*\*[ \t]*(?P<plan_item>\S.*?)[ \t]*$",
    re.MULTILINE,
)


class PatchDocError(Exception):
    """A requested patch id could not be resolved against the registry."""


def _summary_dir(descriptor: "patch_registry.PatchDescriptor", patches_root: Path) -> Path:
    # PatchDescriptor paths are stored RELATIVE to the registry root they
    # were loaded from (registry.py's _packaged_descriptor: package_root =
    # package_dir.relative_to(root)) -- always join against the caller's
    # own resolved root, never treat them as already-absolute.
    relative = descriptor.package_root if descriptor.package_root is not None \
        else descriptor.implementation_path.parent
    return patches_root / relative


def patch_summary_path(descriptor: "patch_registry.PatchDescriptor", patches_root: Path) -> Path:
    return _summary_dir(descriptor, patches_root) / SUMMARY_FILENAME


def _expected_plan_item(descriptor: "patch_registry.PatchDescriptor") -> str:
    """The canonical Plan item text a SUMMARY.md should carry.

    Prefers the structured, multi-plan-capable ``plan_ids`` list over the
    older singular ``plan_item`` string -- a patch spanning several plan
    items (e.g. 1215 covers RD39/RD40/RD41/RD42) should declare
    ``plan-ids = [...]`` in patch.toml, not cram them into one string
    field. Falls back to ``plan_item`` for patches that still only set
    that, and to the literal ``"none"`` when neither is set -- this
    function always returns a concrete expectation, never "skip the
    check" (gpt-dev-agent review, 2026-08-31: an absent plan-item used to
    make ANY SUMMARY.md value report clean).
    """
    if descriptor.plan_ids:
        return "/".join(descriptor.plan_ids)
    if descriptor.plan_item:
        return descriptor.plan_item
    return "none"


def read_patch_summary(descriptor: "patch_registry.PatchDescriptor", patches_root: Path) -> str:
    """The patch's SUMMARY.md content, or a visible placeholder if absent."""
    summary_path = patch_summary_path(descriptor, patches_root)
    if not summary_path.is_file():
        return (
            f"# {descriptor.patch_id}\n\n"
            f"**Status:** {descriptor.state}\n"
            f"**Group:** {descriptor.group}\n\n"
            "_No SUMMARY.md found for this patch -- add one under "
            f"`patches/{descriptor.patch_id}/SUMMARY.md` (see "
            "`patches/_template/SUMMARY.md`)._\n"
        )
    return summary_path.read_text(encoding="utf-8")


def parse_summary_header(text: str) -> dict[str, str] | None:
    """Extract the Status/Group/Plan item header fields, or None if the
    required shape (see patches/_template/SUMMARY.md) isn't present."""
    match = _HEADER_PATTERN.search(text)
    if not match:
        return None
    return {
        "status": match.group("status"),
        "group": match.group("group"),
        "plan_item": match.group("plan_item").strip(),
    }


def check_summary_consistency(patches_dir: Path | None = None) -> list[str]:
    """Fully mechanical drift check, no judgment involved: every patch's
    SUMMARY.md Status/Group header must equal patch.toml's own state/group
    (the sole metadata authority for a packaged patch -- packaged
    patch.py's STATE/GROUP constants, if present at all, are not read by
    the registry and are not what this compares against; a legacy flat
    module's patch.py constants ARE its state/group, since it has no
    patch.toml). Plan item must equal the canonical value from
    ``_expected_plan_item`` (plan_ids preferred over plan_item, "none" as
    the explicit floor -- never "skip the check"). Returns a list of
    problem descriptions (empty = clean); never raises.

    Exists because this drifted for real once already: 1201's patch.toml
    state changed (rejected -> superseded) without anyone -- human or
    agent -- re-checking whether the prose summary still agreed with it.
    A stale SUMMARY.md is worse than a missing one (it looks authoritative
    and is wrong), so this belongs in patch-lint's non-mutating gate, not
    left to be caught by a reviewer reading prose.

    Loads the registry exactly once (gpt-dev-agent review, 2026-08-31: an
    earlier version loaded it here AND again via patchset.catalog(), two
    reads of the same on-disk state for one check).
    """
    problems: list[str] = []
    registry = patch_registry.load_registry(patches_dir or paths.PATCHES)

    for descriptor in registry.descriptors:
        summary_path = patch_summary_path(descriptor, registry.root)
        if not summary_path.is_file():
            problems.append(f"{descriptor.patch_id}: missing SUMMARY.md")
            continue
        header = parse_summary_header(summary_path.read_text(encoding="utf-8"))
        if header is None:
            problems.append(
                f"{descriptor.patch_id}: SUMMARY.md is missing the required "
                "Status/Group/Plan item header (see patches/_template/SUMMARY.md)"
            )
            continue
        if header["status"] != descriptor.state:
            problems.append(
                f"{descriptor.patch_id}: SUMMARY.md Status={header['status']!r} "
                f"does not match patch.toml state={descriptor.state!r}"
            )
        if header["group"] != descriptor.group:
            problems.append(
                f"{descriptor.patch_id}: SUMMARY.md Group={header['group']!r} "
                f"does not match patch.toml group={descriptor.group!r}"
            )
        expected_plan_item = _expected_plan_item(descriptor)
        if header["plan_item"] != expected_plan_item:
            problems.append(
                f"{descriptor.patch_id}: SUMMARY.md Plan item={header['plan_item']!r} "
                f"does not match the canonical value {expected_plan_item!r} "
                "(from patch.toml plan-ids/plan-item)"
            )

    return problems


def resolve_patch_descriptors(
    patch_ids: "tuple[str, ...] | list[str]", *, patches_dir: Path | None = None,
) -> "tuple[list[patch_registry.PatchDescriptor], Path]":
    """Exact-resolve every id in ``patch_ids`` against the registry, in the
    SAME order given -- never silently drops an unresolvable id. Returns
    ``(descriptors, patches_root)``: the resolved root is required by any
    caller that goes on to read a descriptor's SUMMARY.md, since descriptor
    paths are stored relative to it.

    An unknown patch id used to be filtered out of the rendered doc
    entirely (gpt-dev-agent review, 2026-08-31: a real build's N-patch
    selection could silently render as "0 patch(es) included" with no
    error). A caller that legitimately cannot resolve every id should
    catch PatchDocError itself and decide whether that's fatal for it,
    not have this module make that call by omission.
    """
    registry = patch_registry.load_registry(patches_dir or paths.PATCHES)
    missing = [pid for pid in patch_ids if pid not in registry.by_id]
    if missing:
        raise PatchDocError(
            f"unresolvable patch id(s), not in the registry: {', '.join(missing)}"
        )
    return [registry.by_id[pid] for pid in patch_ids], registry.root


def render_release_doc(
    *,
    descriptors: list["patch_registry.PatchDescriptor"],
    patches_root: Path,
    pin_info: dict[str, str],
    selection_label: str,
) -> str:
    """Merge SUMMARY.md files for ``descriptors`` into one release doc, in
    the order given (the caller's resolved order is meaningful -- e.g. a
    dependency-respecting build order -- and is preserved rather than
    re-sorted by numeric id here).

    ``pin_info`` is caller-supplied header fields (e.g. llama.cpp revision,
    bigcherry revision, recipe/target name) rendered verbatim as a
    key/value block -- this module has no opinion on which fields matter,
    it only formats what it's given.
    """
    lines: list[str] = ["# Release patch set", "", f"Selection: {selection_label}", ""]
    for key in sorted(pin_info):
        lines.append(f"- **{key}:** {pin_info[key]}")
    lines.append("")
    lines.append(f"{len(descriptors)} patch(es) included.")
    lines.append("")
    lines.append("---")
    lines.append("")

    for descriptor in descriptors:
        lines.append(read_patch_summary(descriptor, patches_root).rstrip())
        lines.append("")
        lines.append("---")
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def render_patch_selection_doc(
    *,
    patch_ids: "tuple[str, ...] | list[str]",
    pin_info: dict[str, str],
    selection_label: str,
    patches_dir: Path | None = None,
) -> str:
    """Shared entry point for every caller (patch-doc CLI, pin-bump,
    campaign build): exact-resolve ``patch_ids`` then render the doc.
    Raises PatchDocError if any id doesn't resolve -- callers decide
    whether that's fatal for them (a best-effort caller catches it; the
    CLI lets it propagate as a real error).

    Only the pin_info dict is caller-specific (llama.cpp revision vs.
    source_slice_id vs. recipe/target name mean different things in
    different provenance contexts) -- deliberately NOT folded into this
    module, since forcing every caller's identity fields into one shape
    would make the abstraction worse, not better (gpt-dev-agent review,
    2026-08-31).
    """
    descriptors, patches_root = resolve_patch_descriptors(patch_ids, patches_dir=patches_dir)
    return render_release_doc(
        descriptors=descriptors, patches_root=patches_root,
        pin_info=pin_info, selection_label=selection_label,
    )
