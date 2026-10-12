#!/usr/bin/env python3
"""Compose the production set plus one patch's requirements into a directory and try the patch on it, edit by edit.

For fixing anchors of a patch written without the composed source at hand: the files the patch touches are left in
<out-dir> as the patch sees them (the patch itself not applied unless every edit matches), and every edit's status
is printed.

Usage: compose-apply.py <patch-id> <out-dir>      (run from the repository root, PYTHONPATH=tools)
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

from bigcherry.core import paths
from bigcherry.patch import rebase as patch_rebase
from bigcherry.patch.pinned_source import copy_pinned, pinned_checkout
from bigcherry.patcher import apply_all


def nearest(path: Path, anchor: str) -> None:
    """Where a failed anchor most nearly matches, and how the source differs from it there.

    Anchors are regular expressions, nearly always re.escape() of literal source text; the escapes are removed to get
    that text back. The place is the longest run of the anchor's lines found in the file, widened to the anchor's
    length; a diff of the anchor against it shows what upstream changed."""
    import difflib
    import re

    literal = re.sub(r"\\(.)", r"\1", anchor, flags=re.S)
    want = [line for line in literal.split("\n")]
    while want and want[-1] == "":
        want.pop()
    have = path.read_text(encoding="utf-8", errors="replace").split("\n")
    strip = lambda rows: [r.strip() for r in rows]  # noqa: E731
    match = difflib.SequenceMatcher(None, strip(have), strip(want), autojunk=False).find_longest_match()
    if match.size == 0:
        best = difflib.get_close_matches(want[0].strip(), strip(have), n=1, cutoff=0.5)
        if not best:
            print("    nearest: no line of the anchor is in the file any more")
            return
        start = strip(have).index(best[0])
    else:
        start = max(0, match.a - match.b)
    window = have[start:start + len(want)]
    print(f"    nearest: {path.name} line {start + 1} ({match.size} of {len(want)} anchor line(s) found in a row)")
    for row in difflib.unified_diff(want, window, "anchor", "source", lineterm="", n=1):
        if not row.startswith(("---", "+++")):
            print(f"      {row[:170]}")


def main() -> int:
    patch_id, out = sys.argv[1], Path(sys.argv[2])
    repo = Path.cwd()
    spec = importlib.util.spec_from_file_location(
        "patch_" + patch_id, repo / "engines" / "llamacpp" / "patches" / patch_id / "patch.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    vendor = pinned_checkout("llamacpp") if paths.llama_root().is_dir() else paths.llama_root()
    selected = patch_rebase.resolve_selection("llamacpp", source_name="bigcherry", all_patches=False,
                                              focal_overlay_patch_id=patch_id)
    texts = patch_rebase._overlay_texts("llamacpp")
    overlay_paths = frozenset(texts)
    for mod in selected.modules:
        if mod.patch_id == patch_id:
            # the patch's own place in the composition: what comes later is applied on top of it, not under it
            break
        probe = patch_rebase.probe_patch("llamacpp", mod, vendor, texts, context_lines=3, previous_revision=None,
                                         revision="compose-apply", overlay_paths=overlay_paths)
        if probe.status not in (patch_rebase.STATUS_CLEAN, patch_rebase.STATUS_CLEAN_NOOP):
            # after a pin bump several base patches fail at once: say so and go on, so each can be looked at
            print(f"BASE {mod.patch_id}: {probe.status} (not applied; the files below lack its edits)")
    for fp in module.PATCHES:
        target = out / fp.path
        target.parent.mkdir(parents=True, exist_ok=True)
        if fp.path in texts:
            target.write_text(texts[fp.path], encoding="utf-8")
        else:
            copy_pinned("llamacpp", vendor / fp.path, target)
    results = apply_all(module.PATCHES, out)
    bad = 0
    for fp, res in zip(module.PATCHES, results):
        for edit in res.results:
            print(f"{edit.status:16} {fp.path} :: {edit.edit_id}")
            if edit.status == "failed":
                bad += 1
                print(f"    {edit.detail[:400]}")
                spec_edit = next((e for e in fp.edits if e.id == edit.edit_id), None)
                if spec_edit is not None:
                    nearest(out / fp.path, spec_edit.anchor)
            elif edit.status == "already-applied":
                # on a fresh base nothing of this patch is applied yet: the guard matches the base or an earlier
                # edit's output, and the engine has skipped this edit without saying so
                bad += 1
                print("    guard already matches before this edit ran: the edit was SKIPPED; make its guard unique")
    print(f"{bad} failed or skipped edit(s); files in {out}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
