"""WSL07: lab scripts may create a worktree only under the work root and must remove it on exit."""

from __future__ import annotations

import re
import shlex
from pathlib import Path

_ADD = re.compile(r"git\b.*\bworktree\s+add\b")
_TRAP_REMOVE = re.compile(r"^\s*trap\b.*git\b.*worktree\s+remove\b")
_WORK_ROOT = re.compile(r"\$\{?BC_WORK_ROOT\}?/worktrees/")
_OPTIONS_WITH_VALUE = frozenset({"-b", "-B"})


def _target(command: str) -> str | None:
    """The path argument of ``git worktree add``, or None when it cannot be read."""
    try:
        words = shlex.split(command, comments=True)
    except ValueError:
        return None
    for index, word in enumerate(words):
        if word == "add" and words[index - 1:index] == ["worktree"]:
            skip = False
            for item in words[index + 1:]:
                if skip:
                    skip = False
                elif item in _OPTIONS_WITH_VALUE:
                    skip = True
                elif not item.startswith("-"):
                    return item
    return None


def _assignments(lines: list[str]) -> dict[str, str]:
    found: dict[str, str] = {}
    for line in lines:
        match = re.match(r"^\s*(?:local\s+)?([A-Za-z_][A-Za-z0-9_]*)=(.*)$", line)
        if match:
            found[match.group(1)] = match.group(2)
    return found


def lint_script(text: str, name: str) -> list[str]:
    """Findings (``name:line: message``) for one shell script."""
    lines = text.splitlines()
    assigned = _assignments(lines)
    removes = any(_TRAP_REMOVE.match(line) for line in lines)
    findings: list[str] = []
    for number, line in enumerate(lines, 1):
        found = _ADD.search(line)
        if line.lstrip().startswith("#") or not found:
            continue
        resolved = _target(line[found.start():]) or ""
        variable = re.fullmatch(r"\$\{?([A-Za-z_][A-Za-z0-9_]*)\}?", resolved)
        if variable:
            resolved = assigned.get(variable.group(1), resolved)
        if not _WORK_ROOT.search(resolved):
            findings.append(f"{name}:{number}: git worktree add must target $BC_WORK_ROOT/worktrees/")
        elif not removes:
            findings.append(f"{name}:{number}: git worktree add needs a `trap ... git worktree remove` on exit")
    return findings


def lint_tree(root: Path) -> list[str]:
    findings: list[str] = []
    for path in sorted((root / "tools" / "lab").rglob("*.sh")):
        text = path.read_text(encoding="utf-8", errors="replace")
        findings.extend(lint_script(text, path.relative_to(root).as_posix()))
    return findings
