"""Turn a whole-file source change into anchored patch edits.

Porting a fork's change to one file (e.g. a rewritten kernel) by hand as
anchored ``Edit``s is slow and error-prone once the diff runs to hundreds of
lines. This module derives the edits mechanically from the pristine upstream
file and the desired result:

* hunks come from a line diff, widened with context until each anchor is
  unique in the file as it stands when that edit is applied;
* each anchor is the ``re.escape``d NOISE-STRIPPED slice of the current text
  (comments and literals blanked, offsets preserved), because that is what the
  engine matches against;
* each guard is a line only the result contains, so re-application is a no-op;
* the edit list is verified by applying it with the real engine: the output
  must equal the desired file byte for byte, and a second application must
  change nothing.

Usage (prints a patch.py-ready FilePatch)::

    python -m bigcherry.patch.port_diff <old-file> <new-file> \\
        --path ggml/src/ggml-cuda/top-k.cu --prefix nro07-topk \\
        --description "hybrid TOP_K kernels (nasone 7f3e1e4d)"
"""

from __future__ import annotations

import argparse
import difflib
import re
import sys
import tempfile
from pathlib import Path

from bigcherry.core import csource
from bigcherry.patch.apply import Edit, FilePatch, apply_patch

_CONTEXT_STEP = 2
_MAX_CONTEXT = 60


class PortDiffError(RuntimeError):
    pass


def _hunks(old: list[str], new: list[str]) -> list[tuple[int, int, int, int]]:
    """(old_lo, old_hi, new_lo, new_hi) line ranges of each change, merged
    when separated by fewer than three unchanged lines."""
    ops = [op for op in difflib.SequenceMatcher(a=old, b=new, autojunk=False).get_opcodes() if op[0] != "equal"]
    merged: list[list[int]] = []
    for _, i1, i2, j1, j2 in ops:
        if merged and i1 - merged[-1][1] < 3:
            merged[-1][1] = i2
            merged[-1][3] = j2
        else:
            merged.append([i1, i2, j1, j2])
    return [tuple(h) for h in merged]


def _offsets(lines: list[str]) -> list[int]:
    out = [0]
    for line in lines:
        out.append(out[-1] + len(line))
    return out


def generate_edits(old_text: str, new_text: str, *, path: str, prefix: str) -> list[Edit]:
    if old_text == "":
        # A file the patch creates (FilePatch(create=True)): one insertion at
        # the start of the empty text, guarded by its first distinctive line.
        guard = next((re.escape(line.strip()) for line in new_text.splitlines() if len(line.strip()) >= 12), None)
        if guard is None:
            raise PortDiffError("created file has no distinctive line to guard on")
        return [Edit(id=f"{prefix}-01", anchor=r"\A", text=new_text, mode="insert_after", guard=guard,
                     rationale=f"{prefix}: create {path}", max_span_lines=1)]
    language = csource.language_for(path)
    old_lines = old_text.splitlines(keepends=True)
    new_lines = new_text.splitlines(keepends=True)
    hunks = _hunks(old_lines, new_lines)
    if not hunks:
        raise PortDiffError("files are identical; nothing to port")

    edits: list[Edit] = []
    current = old_text
    # Line shift between `current` (old with earlier hunks applied) and old.
    shift = 0
    for index, (o_lo, o_hi, n_lo, n_hi) in enumerate(hunks, start=1):
        cur_lines = current.splitlines(keepends=True)
        cur_off = _offsets(cur_lines)
        stripped = csource.strip_noise(current, language)
        c_lo, c_hi = o_lo + shift, o_hi + shift
        for ctx in range(0, _MAX_CONTEXT + 1, _CONTEXT_STEP):
            a_lo, a_hi = max(0, c_lo - ctx), min(len(cur_lines), c_hi + ctx)
            if a_lo == a_hi:
                continue
            anchor_text = stripped[cur_off[a_lo]:cur_off[a_hi]]
            if not anchor_text.strip():
                continue
            anchor = re.escape(anchor_text)
            if len(re.findall(anchor, stripped, re.MULTILINE)) != 1:
                continue
            before = c_lo - a_lo
            after = a_hi - c_hi
            n_a_lo, n_a_hi = n_lo - before, n_hi + after
            if n_a_lo < 0 or n_a_hi > len(new_lines):
                continue
            replacement = "".join(new_lines[n_a_lo:n_a_hi])
            guard = _guard(replacement, current, new_lines[n_lo:n_hi])
            if guard is not None:
                break
        else:
            raise PortDiffError(f"hunk {index}: no unique anchor/guard within {_MAX_CONTEXT} lines of context")
        edits.append(Edit(
            id=f"{prefix}-{index:02d}",
            anchor=anchor,
            text=replacement,
            mode="replace",
            guard=guard,
            rationale=f"{prefix} hunk {index}: upstream lines {o_lo + 1}-{o_hi} -> result lines {n_lo + 1}-{n_hi}",
            max_span_lines=anchor_text.count("\n") + 2,
        ))
        current = "".join(cur_lines[:a_lo]) + replacement + "".join(cur_lines[a_hi:])
        shift += (n_hi - n_lo) - (o_hi - o_lo)
    if current != new_text:
        raise PortDiffError("internal: sequential reconstruction does not reproduce the target")
    return edits


def _guard(replacement: str, current: str, changed: list[str]) -> str | None:
    """A regex only the patched text contains: the first changed line that
    occurs nowhere in the pre-edit text, else the whole replacement (a pure
    deletion's surrounding context); None when neither is distinctive yet,
    so the caller widens the context."""
    for line in changed:
        candidate = line.strip()
        if len(candidate) >= 12 and candidate not in current:
            return re.escape(candidate)
    if replacement.strip() and replacement not in current:
        return re.escape(replacement)
    return None


def verify(old_text: str, new_text: str, patch: FilePatch) -> None:
    """Apply with the real engine: exact result, then idempotent."""
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        target = root / patch.path
        target.parent.mkdir(parents=True, exist_ok=True)
        if not patch.create:
            target.write_text(old_text, encoding="utf-8", newline="")
        first = apply_patch(patch, root)
        if not first.ok:
            raise PortDiffError(f"generated edits fail to apply: {[r.detail for r in first.failed]}")
        if target.read_text(encoding="utf-8") != new_text:
            raise PortDiffError("generated edits do not reproduce the target byte for byte")
        second = apply_patch(patch, root)
        if second.changed or not second.ok:
            raise PortDiffError("generated edits are not idempotent")


def render(patch: FilePatch, variable: str) -> str:
    lines = [f"{variable} = FilePatch(", f"    path={patch.path!r},", f"    description={patch.description!r},"]
    if patch.create:
        lines.append("    create=True,")
    lines.append("    edits=(")
    for edit in patch.edits:
        lines += [
            "        Edit(",
            f"            id={edit.id!r},",
            f"            anchor={edit.anchor!r},",
            f"            text={edit.text!r},",
            f"            mode={edit.mode!r},",
            f"            guard={edit.guard!r},",
            f"            rationale={edit.rationale!r},",
            f"            max_span_lines={edit.max_span_lines},",
            "        ),",
        ]
    lines += ["    ),", ")"]
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="bigcherry.patch.port_diff", description=__doc__.splitlines()[0])
    parser.add_argument("old", type=Path)
    parser.add_argument("new", type=Path)
    parser.add_argument("--path", required=True, help="target path relative to the llama.cpp root")
    parser.add_argument("--prefix", required=True, help="edit id prefix")
    parser.add_argument("--description", default="")
    parser.add_argument("--variable", default="PATCH")
    args = parser.parse_args(argv)
    old_text = args.old.read_text(encoding="utf-8")
    new_text = args.new.read_text(encoding="utf-8")
    edits = generate_edits(old_text, new_text, path=args.path, prefix=args.prefix)
    patch = FilePatch(path=args.path, edits=tuple(edits), description=args.description)
    verify(old_text, new_text, patch)
    sys.stdout.write(render(patch, args.variable))
    print(f"# {len(edits)} edits, verified exact and idempotent", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
