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

from pathlib import Path

from . import patchset

SUMMARY_FILENAME = "SUMMARY.md"


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
