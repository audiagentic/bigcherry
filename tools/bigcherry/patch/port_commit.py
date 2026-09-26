"""Port one fork commit onto the pinned upstream as anchored patch edits.

For every file the commit modifies, the fork's change is replayed onto the
upstream file with a 3-way merge (``git merge-file``: upstream as ours, the
commit's parent as base, the commit as theirs). A clean merge is then turned
into verified anchored edits by :mod:`bigcherry.patch.port_diff`. Files the
commit adds or deletes, and files whose merge conflicts, are reported rather
than guessed at -- they need a human decision (a new file belongs in the
overlay or folded into an existing translation unit).

    python -m bigcherry.patch.port_commit --repo <fork.git> --commit <sha> \\
        --upstream-repo work/upstream/llama.cpp.git --upstream-ref b11126 \\
        --prefix nro15 [--survey] [--output patch_edits.py]
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

from bigcherry.patch import port_diff
from bigcherry.patch.apply import FilePatch


@dataclass
class FilePort:
    path: str
    status: str  # "clean" | "identical-base" | "conflict" | "added" | "deleted" | "missing-upstream"
    patch: FilePatch | None = None
    detail: str = ""


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), *args], check=True, capture_output=True, text=True, encoding="utf-8"
    ).stdout


def _show(repo: Path, ref: str, path: str) -> str | None:
    result = subprocess.run(
        ["git", "-C", str(repo), "show", f"{ref}:{path}"], capture_output=True, text=True, encoding="utf-8"
    )
    return result.stdout if result.returncode == 0 else None


def _merge(ours: str, base: str, theirs: str) -> tuple[str, bool]:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        files = {}
        for name, text in (("ours", ours), ("base", base), ("theirs", theirs)):
            files[name] = root / name
            files[name].write_text(text, encoding="utf-8", newline="")
        result = subprocess.run(
            ["git", "merge-file", "-p", str(files["ours"]), str(files["base"]), str(files["theirs"])],
            capture_output=True, text=True, encoding="utf-8",
        )
        return result.stdout, result.returncode == 0


def port(repo: Path, commit: str, upstream_repo: Path, upstream_ref: str, prefix: str) -> list[FilePort]:
    changes = _git(repo, "diff-tree", "--no-commit-id", "-r", "--name-status", f"{commit}^", commit)
    ports: list[FilePort] = []
    for index, line in enumerate(changes.splitlines(), start=1):
        status, path = line.split("\t", 1)
        if status == "A":
            if _show(upstream_repo, upstream_ref, path) is not None:
                ports.append(FilePort(path, "conflict", detail="added by the commit but present upstream"))
                continue
            content = _show(repo, commit, path) or ""
            edits = port_diff.generate_edits("", content, path=path, prefix=f"{prefix}-{index:02d}")
            patch = FilePatch(path=path, edits=tuple(edits), description=f"{prefix}: create {path}", create=True)
            port_diff.verify("", content, patch)
            ports.append(FilePort(path, "added", patch))
            continue
        if status == "D":
            ports.append(FilePort(path, "deleted"))
            continue
        base = _show(repo, f"{commit}^", path)
        theirs = _show(repo, commit, path)
        ours = _show(upstream_repo, upstream_ref, path)
        if ours is None or base is None or theirs is None:
            ports.append(FilePort(path, "missing-upstream"))
            continue
        if base == ours:
            target, clean, status_name = theirs, True, "identical-base"
        else:
            target, clean = _merge(ours, base, theirs)
            status_name = "clean"
        if not clean:
            ports.append(FilePort(path, "conflict", detail="3-way merge conflicts"))
            continue
        try:
            edits = port_diff.generate_edits(ours, target, path=path, prefix=f"{prefix}-{index:02d}")
            patch = FilePatch(path=path, edits=tuple(edits), description=f"{prefix}: {path}")
            port_diff.verify(ours, target, patch)
        except port_diff.PortDiffError as exc:
            ports.append(FilePort(path, "conflict", detail=str(exc)))
            continue
        ports.append(FilePort(path, status_name, patch))
    return ports


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="bigcherry.patch.port_commit")
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--commit", required=True)
    parser.add_argument("--upstream-repo", type=Path, required=True)
    parser.add_argument("--upstream-ref", required=True)
    parser.add_argument("--prefix", required=True)
    parser.add_argument("--survey", action="store_true", help="report per-file status only")
    parser.add_argument("--output", type=Path, help="write rendered FilePatch definitions here")
    args = parser.parse_args(argv)
    ports = port(args.repo, args.commit, args.upstream_repo, args.upstream_ref, args.prefix)
    for item in ports:
        edits = len(item.patch.edits) if item.patch else 0
        print(f"{item.status:17} {edits:3} edits  {item.path}  {item.detail}", file=sys.stderr)
    if not args.survey and args.output:
        rendered = [
            port_diff.render(item.patch, f"PATCH_{number:02d}")
            for number, item in enumerate((p for p in ports if p.patch), start=1)
        ]
        args.output.write_text("\n".join(rendered), encoding="utf-8")
    return 0 if all(p.patch for p in ports) else 1


if __name__ == "__main__":
    raise SystemExit(main())
