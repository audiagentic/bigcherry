"""Keep machine names and host-specific values out of tracked files.

Terms that identify a machine come from the untracked ``[forbidden] terms`` list in
``config/environment.local.toml`` or from ``BIGCHERRY_FORBIDDEN_TERMS``. They are never
written into a tracked file, so the list cannot leak the names it guards. The generic
patterns below cover paths, addresses and identities that need no per-machine data.
``config/forbidden-terms-allow.txt`` lists technical phrases that contain a term by accident
(for example ``integrated_gpu_host_buffer``); those phrases are removed before matching.

Run ``python3 -m bigcherry.forbidden_terms --staged`` to scan the index (used by the
pre-commit hook), ``--all`` for every tracked file, or ``--install-hook`` once per clone.
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import tomllib
from pathlib import Path

ALLOW_FILE = "config/forbidden-terms-allow.txt"
LOCAL_FILE = "config/environment.local.toml"
HOOKS_DIR = ".githooks"

GITHUB_NOREPLY_COAUTHOR = re.compile(
    r"(?i)^Co-authored-by:\s+[^\x3c\x3e\r\n]+\x3c([A-Za-z0-9._+%-]+@users\.noreply\.github\.com)\x3e\s*$"
)

GENERIC = (
    # Reject credential-shaped values without storing a real secret in the code.
    ("credential token", re.compile(
        r"\b(?:sk-(?:proj-)?[A-Za-z0-9_-]{30,}"
        r"|github_pat_[A-Za-z0-9_]{40,}"
        r"|gh[pousr]_[A-Za-z0-9_]{30,}"
        r"|AKIA[A-Z0-9]{16})\b")),
    ("credential assignment", re.compile(
        r"""(?i)\b(?:API_KEY|ACCESS_TOKEN|AUTH_TOKEN|CLIENT_SECRET)\b
            ["']?\s*[:=]\s*["'][A-Za-z0-9+/=._-]{28,}["']""", re.X)),
    ("private-key header", re.compile(
        r"-----BEGIN (?:RSA |OPENSSH |EC |DSA )?PRIVATE KEY-----")),
    ("local build or run path", re.compile(
        r"/work/bigcherry/(?:builds|runs)/[A-Za-z0-9]")),
    # The agent container's own roots (AGENTS.md): work root, model store, development tree, account and team folders.
    ("container work path", re.compile(
        r"/work/(?:bigcherry|models|development)/[A-Za-z0-9]|/(?:accounts|instances)/[a-z][a-z0-9_-]*/")),
    ("absolute srv path", re.compile(r"(?<![A-Za-z0-9_])/srv/(?!example)[A-Za-z0-9]")),
    ("absolute mount path", re.compile(r"/mnt/[A-Za-z0-9]")),
    ("absolute home path", re.compile(r"/home/[a-z][a-z0-9_-]*/")),
    ("drive-letter path", re.compile(r"\b[A-Za-z]:\\\\?[A-Za-z]{2,}")),
    ("private IPv4 address", re.compile(
        r"\b10(?:\.\d{1,3}){3}\b"
        r"|\b192\.168(?:\.\d{1,3}){2}\b"
        r"|\b172\.(?:1[6-9]|2\d|3[01])(?:\.\d{1,3}){2}\b")),
    # Reserved example and test domains are placeholders, not real identities.
    # Host locations are named BC_<KEY> (BC_MODEL_ROOT, BC_WORK_ROOT, ...); the older names are retired (and the cache
    # root, now a fixed part of the work root). They are
    # assembled from parts so this file does not itself contain them.
    ("retired variable name", re.compile(
        r"(?<![A-Za-z0-9_])(?:"
        + "|".join(stem + "_ROOT" for stem in ("MODELS", "DEV", "VAULT", "DATA", "BIGCHERRY", "LLAMACPP", "WIN_ROCM",
                                               "BIGCHERRY_WORK", "BIGCHERRY_MODEL", "BC_CACHE"))
        + "|" + "|".join(stem + "_IP" for stem in ("DEV", "GPU_HOST"))
        + "|BIGCHERRY_HOST" + "_[A-Z_]+"
        + r")(?![A-Za-z0-9_])")),
    ("email address", re.compile(
        r"(?<![A-Za-z0-9._%+-])[A-Za-z0-9._%+-]+@(?!example\.(?:com|org|net)\b|[A-Za-z0-9.-]*\.?(?:invalid|test)\b)"
        r"[A-Za-z0-9-]+\.[A-Za-z]{2,}")),
)


def _local_lists(root: Path) -> list[Path]:
    """The untracked local config of this checkout and, in a linked worktree, of the primary checkout (an untracked
    file is not copied into a worktree, so without this the list would apply only where it was written)."""
    found = [root / LOCAL_FILE]
    common = subprocess.run(["git", "-C", str(root), "rev-parse", "--path-format=absolute", "--git-common-dir"],
                            capture_output=True, text=True, check=False).stdout.strip()
    if common:
        primary = Path(common).parent
        if primary != root:
            found.append(primary / LOCAL_FILE)
    return [p for p in found if p.is_file()]


def configured_terms(root: Path) -> list[str]:
    """Terms from the untracked local config(s) and the environment; never from tracked files.

    A term may end in ``*``, which stands for any digits after it (``name*`` is name, name2, name17).
    """
    terms: list[str] = []
    for local in _local_lists(root):
        data = tomllib.loads(local.read_text(encoding="utf-8"))
        terms += [str(t) for t in data.get("forbidden", {}).get("terms", [])]
    terms += [t.strip() for t in os.environ.get("BIGCHERRY_FORBIDDEN_TERMS", "").split(",") if t.strip()]
    return sorted({t for t in terms if t})


def _allowed_phrases(root: Path) -> list[str]:
    path = root / ALLOW_FILE
    if not path.is_file():
        return []
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#")]


def _term_pattern(term: str) -> re.Pattern[str]:
    stem, numbered = (term[:-1], True) if term.endswith("*") else (term, False)
    return re.compile(rf"(?<![A-Za-z]){re.escape(stem)}{'[0-9]*' if numbered else ''}(?![A-Za-z])", re.IGNORECASE)


def scan_line(line: str, term_patterns: list[re.Pattern[str]], allowed: list[str]) -> list[str]:
    """Return the labels of every forbidden match on one line (never the matched text)."""
    cleaned = line
    for phrase in allowed:
        cleaned = re.sub(re.escape(phrase), " ", cleaned, flags=re.IGNORECASE)
    labels = [label for label, pattern in GENERIC if pattern.search(cleaned)]
    if any(p.search(cleaned) for p in term_patterns):
        labels.append("machine name")
    return labels


def _candidate_files(root: Path, staged: bool) -> list[str]:
    if staged:
        out = subprocess.run(["git", "-C", str(root), "diff", "--cached", "--name-only",
                              "--diff-filter=ACMR"], capture_output=True, text=True, check=True).stdout
    else:
        out = subprocess.run(["git", "-C", str(root), "ls-files"], capture_output=True,
                             text=True, check=True).stdout
    return [p for p in out.splitlines() if p]


def _content(root: Path, rel: str, staged: bool) -> str | None:
    if staged:
        raw = subprocess.run(["git", "-C", str(root), "show", f":{rel}"], capture_output=True, check=False).stdout
    else:
        try:
            raw = (root / rel).read_bytes()
        except OSError:
            return None
    if b"\x00" in raw:
        return None
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        return None


def _changed_files(root: Path, base: str) -> list[str]:
    out = subprocess.run(["git", "-C", str(root), "diff", "--name-only", "--diff-filter=ACMR",
                          f"{base}...HEAD"], capture_output=True, text=True, check=True).stdout
    return [p for p in out.splitlines() if p]


def findings(root: Path, *, staged: bool = False, base: str | None = None) -> list[tuple[str, int, str]]:
    """(path, line number, label) for every violation in tracked, staged or changed text files.

    With ``base`` only the files changed since that revision are scanned (CI on a pull request).
    """
    terms = [_term_pattern(t) for t in configured_terms(root)]
    allowed = _allowed_phrases(root)
    out: list[tuple[str, int, str]] = []
    candidates = _changed_files(root, base) if base else _candidate_files(root, staged)
    for rel in candidates:
        text = _content(root, rel, staged)
        if text is None:
            continue
        for number, line in enumerate(text.splitlines(), start=1):
            for label in scan_line(line, terms, allowed):
                out.append((rel, number, label))
    return out


def text_findings(root: Path, label: str, text: str) -> list[tuple[str, int, str]]:
    """(label, line number, finding) for free text: a commit message, a pull request title and description."""
    terms = [_term_pattern(t) for t in configured_terms(root)]
    allowed = _allowed_phrases(root)
    return [(label, number, hit) for number, line in enumerate(text.splitlines(), start=1)
            for hit in scan_line(line, terms, allowed)]


def message_findings(root: Path, base: str) -> list[tuple[str, int, str]]:
    """Every commit message between ``base`` and HEAD: a squash merge carries them (and the PR text) onto main."""
    revs = subprocess.run(["git", "-C", str(root), "rev-list", f"{base}..HEAD"], capture_output=True, text=True,
                          check=True).stdout.split()
    out: list[tuple[str, int, str]] = []
    for rev in revs:
        msg = subprocess.run(["git", "-C", str(root), "show", "-s", "--format=%B", rev], capture_output=True,
                             text=True, check=True).stdout
        lines = []
        for line in msg.splitlines():
            match = GITHUB_NOREPLY_COAUTHOR.fullmatch(line)
            if match:
                # Only GitHub-generated co-author identity is public metadata.
                # Keep the display name subject to ordinary machine-name checks.
                line = line[:match.start(1)] + "[GitHub noreply]" + line[match.end(1):]
            lines.append(line)
        out += text_findings(root, f"commit {rev[:10]} message", "\n".join(lines))
    return out


def ref_findings(root: Path, names: list[str]) -> list[tuple[str, int, str]]:
    """Branch and tag names: they are published on GitHub (a pull request keeps its head branch name), so the same
    terms apply. Only the label is reported, never the name."""
    terms = [_term_pattern(t) for t in configured_terms(root)]
    allowed = _allowed_phrases(root)
    return [("branch or tag name", number, hit) for number, name in enumerate(names, start=1)
            for hit in scan_line(name, terms, allowed)]


def pre_push_refs(stdin_text: str) -> list[str]:
    """The remote ref names a push creates or updates (git's pre-push input: local ref, local sha, remote ref, remote
    sha per line); a deletion publishes nothing."""
    zero = "0" * 40
    names = []
    for line in stdin_text.splitlines():
        parts = line.split()
        if len(parts) == 4 and set(parts[1]) != {"0"} and not parts[1].startswith(zero):
            names.append(parts[2].removeprefix("refs/heads/").removeprefix("refs/tags/"))
    return names


def install_hook(root: Path) -> None:
    subprocess.run(["git", "-C", str(root), "config", "core.hooksPath", HOOKS_DIR], check=True)
    print(f"core.hooksPath set to {HOOKS_DIR}; pre-commit now runs the forbidden-term scan")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="bigcherry.forbidden_terms")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--staged", action="store_true", help="scan the staged index (pre-commit)")
    mode.add_argument("--all", action="store_true", help="scan every tracked file")
    mode.add_argument("--base", metavar="REV", help="scan files changed and commit messages between REV and HEAD (CI)")
    mode.add_argument("--text", metavar="FILE", help="scan a text file: a commit message (commit-msg hook) or a PR's title and description (CI)")
    mode.add_argument("--ref", action="append", metavar="NAME", help="scan a branch or tag name (CI: a pull request's head branch); repeatable")
    mode.add_argument("--pre-push", action="store_true", help="scan the ref names a push publishes (git's pre-push input on stdin)")
    mode.add_argument("--install-hook", action="store_true", help="point core.hooksPath at .githooks")
    args = parser.parse_args(argv)
    root = Path(subprocess.run(["git", "rev-parse", "--show-toplevel"], capture_output=True,
                               text=True, check=True).stdout.strip())
    if args.install_hook:
        install_hook(root)
        return 0
    if args.ref or args.pre_push:
        hits = ref_findings(root, args.ref if args.ref else pre_push_refs(sys.stdin.read()))
        for _, number, label in hits:
            print(f"ref name #{number}: {label}")
        if hits:
            print("forbidden term in a branch or tag name: it is published on GitHub. Rename it "
                  "(git branch -m) and push the new name.", file=sys.stderr)
            return 1
        return 0
    if args.text:
        hits = text_findings(root, Path(args.text).name, Path(args.text).read_text(encoding="utf-8", errors="replace"))
    else:
        hits = findings(root, staged=args.staged, base=args.base)
        if args.base:
            hits += message_findings(root, args.base)
    for rel, number, label in hits:
        print(f"{rel}:{number}: {label}")
    if hits:
        print(f"{len(hits)} forbidden-term finding(s). Use a project-relative path, a ${{VAR}} "
              "reference, config/environment.local.toml, or a BIGCHERRY_* variable.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
