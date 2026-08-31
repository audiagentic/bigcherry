"""Per-patch SUMMARY.md rendering and release-doc merging.

Each patch package directory carries a short ``SUMMARY.md`` (see
``patches/_template/SUMMARY.md`` for the required shape: What it does / Why
/ Upstream, plus a Status and Group header) -- the human-readable
counterpart to the machine-readable ``PROVENANCE``/``STATE`` in
``patch.py``/``patch.toml``. This module merges the SUMMARY.md of every
patch in a given selection into one release doc, alongside the llama.cpp
pin it was built against -- so a release has one document that says
exactly what patches it carries and why, not just a revision number.

Deliberately does not require every patch to have a SUMMARY.md: a missing
file renders a visible placeholder rather than failing the merge, since a
release doc that silently omits an undocumented patch is worse than one
that flags it.
"""

from __future__ import annotations

import re
from pathlib import Path

from ..core import paths
from . import patchset
from . import registry as patch_registry

SUMMARY_FILENAME = "SUMMARY.md"

_HEADER_PATTERN = re.compile(
    r"^\*\*Status:\*\*\s*(?P<status>\S+)\s*$\n"
    r"^\*\*Group:\*\*\s*(?P<group>\S+)\s*$\n"
    r"^\*\*Plan item:\*\*\s*(?P<plan_item>.+?)\s*$",
    re.MULTILINE,
)


def patch_summary_path(module: "patchset.PatchModule") -> Path:
    return module.path.parent / SUMMARY_FILENAME


def read_patch_summary(module: "patchset.PatchModule") -> str:
    """The patch's SUMMARY.md content, or a visible placeholder if absent."""
    summary_path = patch_summary_path(module)
    if not summary_path.is_file():
        return (
            f"# {module.patch_id}\n\n"
            f"**Status:** {module.state}\n"
            f"**Group:** {module.group}\n\n"
            "_No SUMMARY.md found for this patch -- add one under "
            f"`patches/{module.patch_id}/SUMMARY.md` (see "
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
    SUMMARY.md Status/Group header must equal its own module STATE/GROUP
    (patch.toml is authoritative where one exists -- see patchset.catalog),
    and Plan item must equal patch.toml's own plan-item field where a
    packaged patch declares one. Returns a list of problem descriptions
    (empty = clean); never raises.

    Exists because this drifted for real once already: 1201's patch.toml
    state changed (rejected -> superseded) without anyone -- human or
    agent -- re-checking whether the prose summary still agreed with it.
    A stale SUMMARY.md is worse than a missing one (it looks authoritative
    and is wrong), so this belongs in patch-lint's non-mutating gate, not
    left to be caught by a reviewer reading prose.
    """
    problems: list[str] = []
    registry = patch_registry.load_registry(patches_dir or paths.PATCHES)
    plan_items_by_id = {d.patch_id: d.plan_item for d in registry.descriptors}

    for module in patchset.catalog(patches_dir):
        summary_path = patch_summary_path(module)
        if not summary_path.is_file():
            problems.append(f"{module.patch_id}: missing SUMMARY.md")
            continue
        header = parse_summary_header(summary_path.read_text(encoding="utf-8"))
        if header is None:
            problems.append(
                f"{module.patch_id}: SUMMARY.md is missing the required "
                "Status/Group/Plan item header (see patches/_template/SUMMARY.md)"
            )
            continue
        if header["status"] != module.state:
            problems.append(
                f"{module.patch_id}: SUMMARY.md Status={header['status']!r} "
                f"does not match module STATE={module.state!r}"
            )
        if header["group"] != module.group:
            problems.append(
                f"{module.patch_id}: SUMMARY.md Group={header['group']!r} "
                f"does not match module GROUP={module.group!r}"
            )
        declared_plan_item = plan_items_by_id.get(module.patch_id)
        if declared_plan_item and header["plan_item"] != declared_plan_item:
            problems.append(
                f"{module.patch_id}: SUMMARY.md Plan item={header['plan_item']!r} "
                f"does not match patch.toml plan-item={declared_plan_item!r}"
            )

    return problems


def render_release_doc(
    *,
    modules: list["patchset.PatchModule"],
    pin_info: dict[str, str],
    selection_label: str,
) -> str:
    """Merge SUMMARY.md files for ``modules`` into one release doc.

    ``pin_info`` is caller-supplied header fields (e.g. llama.cpp revision,
    bigcherry revision, recipe/target name) rendered verbatim as a
    key/value block -- this module has no opinion on which fields matter,
    it only formats what it's given.
    """
    lines: list[str] = ["# Release patch set", "", f"Selection: {selection_label}", ""]
    for key in sorted(pin_info):
        lines.append(f"- **{key}:** {pin_info[key]}")
    lines.append("")
    lines.append(f"{len(modules)} patch(es) included.")
    lines.append("")
    lines.append("---")
    lines.append("")

    for module in sorted(modules, key=lambda m: m.order):
        lines.append(read_patch_summary(module).rstrip())
        lines.append("")
        lines.append("---")
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"
