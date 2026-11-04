"""Model-constant execution guard (RCS12 step 1, static half).

BigCherry must not hardcode a model's shape into the core: a new model is added by data (container, manifest,
layout, arch). This scans the radiance components, patch payloads and lab scripts for model-shaped literals and
for the symbols the audit found them behind, and fails on any hit that is not listed in
engines/radiance/model_constant_guard.toml with a category and a reason.

Categories: A a bounded capacity that validates the real shape and refuses; B diagnostic or test-only; C a
model-specific value on an execution, selection or planning branch (a defect for any other shape); D configuration
that belongs in the manifest, layout or arch data; E naming, a message or a comment only. C and D entries must name
an OPEN plan item as owner: when that item is completed (or does not exist) the entry is invalid and the guard
fails, so an execution-path debt cannot outlive the change that owns it, and no calendar date is invented.
"""
from __future__ import annotations

import re
import tomllib
from dataclasses import dataclass
from pathlib import Path

CONFIG = Path("engines/radiance/model_constant_guard.toml")
PLANNING = Path("docs/planning/active")
CATEGORIES = {"A", "B", "C", "D", "E"}
OWNED = {"C", "D"}


@dataclass(frozen=True)
class Hit:
    path: str
    line: int
    text: str


def load(root: Path) -> dict:
    return tomllib.loads((root / CONFIG).read_text(encoding="utf-8"))


def _is_test(path: str, patterns: list[str]) -> bool:
    return any(re.search(p, path) for p in patterns)


def _code(line: str) -> str:
    """The line without comments; whole-line comments and preprocessor lines other than includes yield nothing."""
    line = re.sub(r"/\*.*?\*/", "", line)
    line = re.sub(r"//.*$", "", line)
    stripped = line.strip()
    if stripped.startswith(("#", "*", "/*")) and not stripped.startswith("#include"):
        return ""
    return line


def scan(root: Path, config: dict) -> list[Hit]:
    scope = config["scope"]
    exts = tuple(scope["extensions"])
    literal = re.compile(
        r"(?<![\w.])(" + "|".join(str(n) for n in scope["literals"]) + r")(?![\w.])"
        r"|\.rad\b|\b(" + "|".join(scope["symbols"]) + r")\b")
    hits: list[Hit] = []
    for base in scope["roots"]:
        for path in sorted((root / base).rglob("*")):
            rel = path.relative_to(root).as_posix()
            if (not path.is_file() or not rel.endswith(exts) or "/build" in rel
                    or _is_test(rel, scope["test_patterns"])):
                continue
            for number, line in enumerate(path.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
                code = _code(line)
                if code.strip() and literal.search(code):
                    hits.append(Hit(rel, number, line.strip()))
    return hits


def open_plan_items(root: Path) -> set[str]:
    ids: set[str] = set()
    for item in (root / PLANNING).rglob("*.md"):
        match = re.match(r"^[A-Z]+\d+", item.name)
        if match:
            ids.add(match.group(0))
    return ids


def check(root: Path) -> list[str]:
    """Every problem with the guard against the tree at `root`; empty means the tree is clean."""
    config = load(root)
    problems: list[str] = []
    allow = config.get("allow", [])
    open_items = open_plan_items(root)
    for index, entry in enumerate(allow):
        label = f"allow[{index}] {entry.get('path')}"
        for key in ("path", "anchor", "category", "reason"):
            if not entry.get(key):
                problems.append(f"{label}: missing {key}")
        if entry.get("category") not in CATEGORIES:
            problems.append(f"{label}: category must be one of {sorted(CATEGORIES)}")
        if entry.get("category") in OWNED:
            owner = entry.get("owner", "")
            if owner not in open_items:
                problems.append(f"{label}: category {entry.get('category')} needs an open plan item as owner, got {owner!r}")
    used = [0] * len(allow)
    for hit in scan(root, config):
        matched = False
        for index, entry in enumerate(allow):
            if entry.get("path") == hit.path and entry.get("anchor") and entry["anchor"] in hit.text:
                used[index] += 1
                matched = True
        if not matched:
            problems.append(f"{hit.path}:{hit.line}: model-shaped literal not in the allowlist: {hit.text[:110]}")
    for index, entry in enumerate(allow):
        if not used[index]:
            problems.append(f"allow[{index}] {entry.get('path')}: stale, no line contains {entry.get('anchor')!r}")
    return problems
