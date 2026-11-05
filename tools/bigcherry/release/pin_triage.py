"""Read-only upstream pin triage: changed anchors, exact source drift and patch exposure.

The shared upstream *bare mirror* is the only persistent state this command may
update (to fetch a missing ref). The vendor checkout and controller are never
checked out, patched or written.
"""

from __future__ import annotations

import difflib
import re
import subprocess
import tempfile
import tomllib
from collections import defaultdict
from pathlib import Path
from typing import Any

from ..core import components, paths
from ..core.context import ProjectContext
from ..engine_adapters import adapter, radiance_bump
from ..patch import registry as patch_registry
from ..source import upstream
from .pin import recipes_path


class PinTriageError(RuntimeError):
    """Usage or environment error (exit 2); an anchor drift is not an error."""


def _git(mirror: Path, *args: str) -> str:
    try:
        return upstream._git(mirror, *args)
    except upstream.UpstreamError as exc:
        raise PinTriageError(str(exc)) from exc


def _revision(mirror: Path, ref: str) -> str:
    # Fetch only when needed. In particular, do not fetch or touch the vendor
    # checkout if both refs are already in the shared mirror.
    try:
        return _git(mirror, "rev-parse", "--verify", f"{ref}^{{commit}}").strip()
    except PinTriageError:
        warning = upstream.sync_mirror_ref(mirror, ref)
        if warning:
            raise PinTriageError(warning)
        return _git(mirror, "rev-parse", "--verify", f"{ref}^{{commit}}").strip()


def _native_blocks(path: Path) -> list[tuple[str, str]]:
    """The block find/replace pairs, in the same order as the native rewriter."""
    raw = path.read_text(encoding="utf-8").splitlines()
    blocks: list[tuple[str, str]] = []
    find: list[str] = []
    replace: list[str] = []
    state = ""
    for line in raw:
        if line.startswith("@@ find "):
            find, replace, state = [], [], "find"
        elif line == "@@ replace" and state == "find":
            state = "replace"
        elif line == "@@ end" and state == "replace":
            blocks.append(("\n".join(find) + "\n", "\n".join(replace) + "\n"))
            state = ""
        elif state == "find":
            find.append(line)
        elif state == "replace":
            replace.append(line)
    return blocks


def _rewriters(engine: str) -> dict[str, components.Component]:
    return {
        name: component
        for name, component in components.load_all(paths.REPO_ROOT, engine).items()
        if (component.root / radiance_bump.REWRITE_SCRIPT).is_file()
    }


def _rewrite_anchors(engine: str, changed: list[str], mirror: Path, before: str, after: str) -> dict[str, list[str]]:
    """Intersect changed upstream sources with native block and textual rules.

    Textual rules apply to every rewritten source, not a fixed named file.
    Check their find-side text in either revision of each changed file.
    """
    declared = _rewriters(engine)
    anchored: dict[str, set[str]] = defaultdict(set)
    for name, component in sorted(declared.items()):
        if not (component.root / radiance_bump.REWRITE_SCRIPT).is_file():
            continue
        source = radiance_bump.REWRITE_SOURCE + "/"
        native = component.root / radiance_bump.REWRITE_NATIVE
        rules_file = component.root / radiance_bump.REWRITE_RULES
        native_files = {
            source + p.name[:-3]: f"{name}/native/{p.name}"
            for p in native.glob("*.rw")
        } if native.is_dir() else {}
        rules = []
        if rules_file.is_file():
            rules = [
                line.split(" ==> ", 1)[0]
                for line in rules_file.read_text(encoding="utf-8").splitlines()
                if line.strip() and not line.lstrip().startswith("#") and " ==> " in line
            ]
        for file in changed:
            if not file.startswith(source):
                continue
            if file in native_files:
                anchored[file].add(native_files[file])
            if rules:
                texts = []
                for revision in (before, after):
                    try:
                        texts.append(_git(mirror, "show", f"{revision}:{file}"))
                    except PinTriageError:
                        pass  # added/deleted file
                if any(rule in text for rule in rules for text in texts):
                    anchored[file].add(f"{name}/{radiance_bump.REWRITE_RULES}")
    return {file: sorted(anchors) for file, anchors in sorted(anchored.items())}


def _patch_anchors(engine: str, changed: list[str], context: ProjectContext) -> dict[str, list[str]]:
    registry = patch_registry.load_registry(engine, context.patches_root)
    changed_set = set(changed)
    touched: dict[str, set[str]] = defaultdict(set)
    for descriptor in registry.descriptors:
        for patch in patch_registry.load_implementation(engine, descriptor, root=context.patches_root):
            path = Path(patch.path).as_posix()
            if path in changed_set:
                touched[path].add(descriptor.patch_id)
    return {file: sorted(patch_ids) for file, patch_ids in sorted(touched.items())}


def _closest(path: Path, needles: list[str]) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    normalized = [line.strip() for line in lines]
    folded = [line.casefold() for line in normalized]
    candidates: dict[int, tuple[float, str]] = {}
    # The failed part of a multi-line block is the useful part to suggest:
    # unchanged surrounding lines would otherwise win a perfect match.
    missing = [needle.strip() for needle in needles if needle.strip() and needle.strip() not in normalized]
    for needle in missing or [needle.strip() for needle in needles if needle.strip()]:
        for match in difflib.get_close_matches(needle.casefold(), folded, n=2, cutoff=0.3):
            score = difflib.SequenceMatcher(None, needle.casefold(), match).ratio()
            index = folded.index(match)
            if index not in candidates or candidates[index][0] < score:
                candidates[index] = (score, lines[index].strip())
    return [
        {"line": index + 1, "text": value}
        for index, (_, value) in sorted(candidates.items(), key=lambda item: (-item[1][0], item[0]))[:2]
    ]


def _suggestions(engine: str, root: Path, file: str, problem: str) -> tuple[str | None, list[dict[str, Any]]]:
    if "/native/" in file and file.endswith(".rw"):
        component_name, _, native_name = file.partition("/native/")
        component = _rewriters(engine).get(component_name)
        if component is None:
            return None, []
        source_path = radiance_bump.REWRITE_SOURCE + "/" + native_name[:-3]
        block = re.search(r"\bblock (\d+)\b", problem)
        native = component.root / "native" / native_name
        if not block or not native.is_file():
            return source_path, []
        blocks = _native_blocks(native)
        index = int(block.group(1)) - 1
        if index >= len(blocks):
            return source_path, []
        find, _ = blocks[index]
        return source_path, _closest(root / source_path, find.splitlines())

    if file.endswith("/rewrites.txt"):
        # A textual rule may occur in any source. Search the extracted source
        # only; never consult the mutable vendor tree for a suggestion.
        match = re.search(r"rule matched nothing: (.*)$", problem)
        if not match:
            return None, []
        try:
            import ast
            needle = ast.literal_eval(match.group(1))
        except (ValueError, SyntaxError):
            return None, []
        best: tuple[float, str, list[dict[str, Any]]] | None = None
        for path in sorted((root / radiance_bump.REWRITE_SOURCE).rglob("*")):
            if not path.is_file():
                continue
            found = _closest(path, [needle])
            if found:
                score = difflib.SequenceMatcher(None, needle, found[0]["text"]).ratio()
                rel = path.relative_to(root).as_posix()
                if best is None or score > best[0]:
                    best = (score, rel, found)
        if best:
            return best[1], best[2]
    return None, []


def _drift(engine: str, context: ProjectContext, mirror: Path, target_sha: str) -> tuple[dict[str, list[str]], list[dict[str, Any]]]:
    # Radiance's only rewritten source is libr4d. Other engines that declare
    # no checks must not need an archive at all.
    if not _rewriters(engine):
        return {}, []
    with tempfile.TemporaryDirectory(prefix="pin-triage-") as work:
        root = Path(work)
        archive = subprocess.Popen(
            ["git", "-C", str(mirror), "archive", target_sha, radiance_bump.REWRITE_SOURCE],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        )
        assert archive.stdout is not None
        unpack = subprocess.run(["tar", "-x", "-C", str(root)], stdin=archive.stdout, check=False, capture_output=True, text=True)
        archive.stdout.close()
        archive_error = archive.communicate()[1].decode("utf-8", errors="replace")
        if archive.returncode or unpack.returncode:
            raise PinTriageError(f"cannot extract target source: {archive_error.strip()} {unpack.stderr.strip()}")
        try:
            checks = adapter(engine).source_drift_checks(context, root)
            failures: dict[str, list[str]] = {}
            for check in checks:
                for file, problems in check.run().items():
                    failures.setdefault(file, []).extend(problems)
        except Exception as exc:
            raise PinTriageError(f"source drift check could not run: {exc}") from exc
        findings = []
        for file, problems in sorted(failures.items()):
            failing_blocks = [
                int(m.group(1)) for problem in problems
                if (m := re.search(r"\bblock (\d+)\b", problem))
            ]
            chained = False
            if len(failing_blocks) > 1 and "/native/" in file:
                name, _, native_file = file.partition("/native/")
                component = _rewriters(engine).get(name)
                if component:
                    blocks = _native_blocks(component.root / "native" / native_file)
                    chained = any(
                        blocks[later - 1][0] in blocks[earlier - 1][1]
                        for earlier in failing_blocks for later in failing_blocks
                        if earlier < later and earlier <= len(blocks) and later <= len(blocks)
                    )
                # Later blocks can also chain through intervening successful
                # substitutions; name the risk whenever several in one file fail.
                chained = chained or len(failing_blocks) > 1
            for problem in problems:
                source_path, closest = _suggestions(engine, root, file, problem)
                findings.append({
                    "anchor": file, "failure": problem, "source": source_path,
                    "closest": closest, "chained": chained,
                })
        return failures, findings


def run(*, engine: str, target_ref: str, context: ProjectContext | None = None) -> dict[str, Any]:
    context = context or ProjectContext.resolve(engine=engine)
    mirror = context.upstream_repo
    if not (mirror / "HEAD").is_file():
        raise PinTriageError(f"shared upstream bare mirror does not exist: {mirror}")
    try:
        current = tomllib.loads(recipes_path(engine).read_text(encoding="utf-8"))["pinned"]
        if not isinstance(current, str) or not current:
            raise ValueError("pinned must be a nonempty string")
    except (OSError, tomllib.TOMLDecodeError, KeyError, ValueError) as exc:
        raise PinTriageError(f"cannot read current pin for {engine}: {exc}") from exc

    from_sha, to_sha = _revision(mirror, current), _revision(mirror, target_ref)
    commits = _git(mirror, "rev-list", "--count", f"{from_sha}..{to_sha}").strip()
    stat = _git(mirror, "diff", "--stat", from_sha, to_sha).strip()
    changed = sorted(filter(None, _git(mirror, "diff", "--name-only", from_sha, to_sha).splitlines()))
    rewrites = _rewrite_anchors(engine, changed, mirror, from_sha, to_sha)
    patches = _patch_anchors(engine, changed, context)
    subjects = _git(
        mirror, "log", "--format=%h %s", f"{from_sha}..{to_sha}", "--",
        "core/mem", "core/runtime", "core/engine.cpp",
    ).splitlines()
    failures, findings = _drift(engine, context, mirror, to_sha)
    drift_count = sum(map(len, failures.values()))
    verdict = "patches-affected" if patches else "drift-only" if drift_count else "clean"
    return {
        "engine": engine, "current_pin": current, "target": target_ref,
        "current_sha": from_sha, "target_sha": to_sha,
        "commit_count": int(commits), "diff_stat": stat, "changed_files": changed,
        "rewrite_anchors": rewrites, "patch_anchors": patches,
        "core_commit_subjects": subjects, "drift_failures": failures,
        "drift_details": findings,
        "verdict": verdict,
        "counts": {
            "changed_files": len(changed), "rewrite_anchor_files": len(rewrites),
            "patch_target_files": len(patches), "drift_failures": drift_count,
        },
    }


def format_report(report: dict[str, Any]) -> str:
    lines = [
        (f"pin-triage: {report['engine']} {report['current_pin']} -> {report['target']} "
        f"({report['commit_count']} commits)"),
        f"  diff: {report['diff_stat'].splitlines()[-1] if report['diff_stat'] else 'no changes'}",
        "  rewritten-source anchors:",
    ]
    lines.extend(
        f"    {file}: {', '.join(anchors)}" for file, anchors in report["rewrite_anchors"].items()
    )
    if not report["rewrite_anchors"]:
        lines.append("    none")
    lines.append("  managed patch targets:")
    lines.extend(
        f"    {file}: {', '.join(patches)}" for file, patches in report["patch_anchors"].items()
    )
    if not report["patch_anchors"]:
        lines.append("    none")
    lines.append("  core memory/runtime/engine commits:")
    lines.extend(f"    {subject}" for subject in report["core_commit_subjects"])
    if not report["core_commit_subjects"]:
        lines.append("    none")
    lines.append("  exact rewrite drift:")
    for item in report["drift_details"]:
        lines.append(f"    {item['anchor']}: {item['failure']}")
        if item["chained"]:
            lines.append("      chained native blocks: a later find may depend on an earlier replace")
        for suggestion in item["closest"]:
            lines.append(f"      closest {item['source']}:{suggestion['line']}: {suggestion['text']}")
    if not report["drift_details"]:
        lines.append("    none")
    counts = report["counts"]
    lines.append(
        f"  verdict: {report['verdict']} (rewrite anchors {counts['rewrite_anchor_files']}, "
        f"patch targets {counts['patch_target_files']}, drift failures {counts['drift_failures']})"
    )
    return "\n".join(lines)
