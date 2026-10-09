"""PA45: promote qualified patches into the production enhancement set."""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from ..core import paths
from ..patch import evidence as patch_evidence
from ..patch import profile_evidence
from . import notes as release_notes
from . import pin_release


class PatchPromoteError(RuntimeError):
    pass


@dataclass(frozen=True)
class PatchInfo:
    patch_id: str
    root: Path
    state: str
    kind: str
    plan_ids: tuple[str, ...]


@dataclass(frozen=True)
class PromotionResult:
    branch: str
    title: str
    pr_url: str
    changed_files: tuple[str, ...]
    release_pending: bool = False


def _run(root: Path, *cmd: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    tools = str(root / "tools")
    env["PYTHONPATH"] = tools + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    done = subprocess.run(
        list(cmd),
        cwd=root,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=env,
    )
    if check and done.returncode != 0:
        detail = (done.stderr or done.stdout).strip()
        raise PatchPromoteError(
            f"{' '.join(cmd)} failed ({done.returncode}): {detail[-1200:]}"
        )
    return done


def _git(root: Path, *args: str, check: bool = True) -> str:
    return _run(root, "git", *args, check=check).stdout.strip()


def _load_patch(root: Path, patch_id: str) -> PatchInfo:
    patch_root = paths.LLAMACPP.patches_root(root) / patch_id
    path = patch_root / "patch.toml"
    if not path.is_file():
        raise PatchPromoteError(f"unknown patch {patch_id!r}: {path} is missing")
    raw = tomllib.loads(path.read_text(encoding="utf-8"))
    if raw.get("id") != patch_id:
        raise PatchPromoteError(f"{path}: id does not match directory")
    state = str(raw.get("state", ""))
    if state not in {"untested", "validated"}:
        raise PatchPromoteError(
            f"{patch_id}: state {state!r} is not promotable"
        )
    kind = str(raw.get("kind", ""))
    plan_ids = tuple(str(x) for x in raw.get("plan-ids", []))
    if not plan_ids and raw.get("plan-item"):
        plan_ids = (str(raw["plan-item"]),)
    return PatchInfo(patch_id, patch_root, state, kind, plan_ids)


def _read_evidence(spec: str) -> tuple[Path, str]:
    if not spec.startswith("@") or len(spec) == 1:
        raise PatchPromoteError("--evidence must be @<file>")
    path = Path(spec[1:]).expanduser().resolve()
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise PatchPromoteError(f"cannot read evidence file {path}: {exc}") from exc
    if not text.strip():
        raise PatchPromoteError("evidence file is empty")
    return path, text.rstrip()


def _named_models(evidence: str) -> set[str]:
    models: set[str] = set()
    for line in evidence.splitlines():
        match = re.match(r"\s*(?:model|model-id)\s*:\s*(\S.*)\s*$", line, re.I)
        if match:
            models.add(match.group(1).strip().casefold())
    return models


def _replace_once(text: str, pattern: str, replacement: str, *, label: str, flags: int = 0) -> str:
    out, count = re.subn(pattern, replacement, text, count=1, flags=flags)
    if count != 1:
        raise PatchPromoteError(f"{label}: expected one match, found {count}")
    return out


def _set_state(info: PatchInfo) -> list[Path]:
    changed: list[Path] = []
    toml_path = info.root / "patch.toml"
    py_path = info.root / "patch.py"
    summary_path = info.root / "SUMMARY.md"
    for path in (toml_path, summary_path):
        if not path.is_file():
            raise PatchPromoteError(f"{info.patch_id}: required file missing: {path.name}")

    text = toml_path.read_text(encoding="utf-8")
    text = _replace_once(
        text,
        r'(?m)^state[ \t]*=[ \t]*"[^"]+"[ \t]*$',
        'state = "validated"',
        label=f"{info.patch_id}/patch.toml state",
    )
    toml_path.write_text(text, encoding="utf-8", newline="\n")
    changed.append(toml_path)

    if py_path.is_file():
        text = py_path.read_text(encoding="utf-8")
        # [ \t], not \s: \s would also take the line break and any blank lines after the field, which changes the
        # file beyond the state and with it the digest the evidence is bound to
        if re.search(r'(?m)^STATE[ \t]*=[ \t]*"[^"]+"[ \t]*$', text):
            text = _replace_once(
                text,
                r'(?m)^STATE[ \t]*=[ \t]*"[^"]+"[ \t]*$',
                'STATE = "validated"',
                label=f"{info.patch_id}/patch.py STATE",
            )
            py_path.write_text(text, encoding="utf-8", newline="\n")
            changed.append(py_path)

    text = summary_path.read_text(encoding="utf-8")
    text = _replace_once(
        text,
        r"(?m)^\*\*Status:\*\*[ \t]+\S+[ \t]*$",
        "**Status:** validated",
        label=f"{info.patch_id}/SUMMARY.md status",
    )
    summary_path.write_text(text, encoding="utf-8", newline="\n")
    changed.append(summary_path)
    return changed


def _promote_default_on(info: PatchInfo, evidence: str) -> list[Path]:
    models = _named_models(evidence)
    if len(models) < 2:
        raise PatchPromoteError(
            f"{info.patch_id}: --default-on requires evidence with at least two "
            "distinct 'Model:' or 'Model-ID:' lines"
        )
    py_path = info.root / "patch.py"
    text = py_path.read_text(encoding="utf-8")
    env_matches = re.findall(
        r'EnvDoc\(\s*"([^"]+)"[\s\S]{0,240}?"0 \(off\)"',
        text,
    )
    if len(env_matches) != 1:
        raise PatchPromoteError(
            f"{info.patch_id}: --default-on requires exactly one EnvDoc defaulted "
            f"to '0 (off)', found {len(env_matches)}"
        )
    flag = env_matches[0]
    if text.count(f'getenv("{flag}")') != 1:
        raise PatchPromoteError(
            f"{info.patch_id}: cannot identify one runtime gate for {flag}"
        )
    return_patterns = (
        r"return s != nullptr && atoi\(s\) != 0;",
        r"return s && atoi\(s\) != 0;",
    )
    matches = [p for p in return_patterns if re.search(p, text)]
    if len(matches) != 1:
        raise PatchPromoteError(
            f"{info.patch_id}: unsupported default-off gate shape for {flag}"
        )
    text = _replace_once(
        text,
        matches[0],
        "return s == nullptr || atoi(s) != 0;",
        label=f"{info.patch_id} default gate",
    )
    text = _replace_once(
        text,
        r'"0 \(off\)"',
        '"1 (on)"',
        label=f"{info.patch_id} EnvDoc default",
    )
    py_path.write_text(text, encoding="utf-8", newline="\n")

    note = (
        f"\n\nDefault-on promotion: `{flag}` now defaults on; set "
        f"`{flag}=0` to restore the prior path. Qualified on "
        f"{len(models)} named models.\n"
    )
    summary = info.root / "SUMMARY.md"
    summary.write_text(
        summary.read_text(encoding="utf-8").rstrip() + note,
        encoding="utf-8",
        newline="\n",
    )
    return [py_path, summary]


def _write_release_evidence(root: Path, info: PatchInfo, evidence: str) -> Path:
    path = root / "releases" / "evidence" / f"{info.patch_id}-promotion.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        f"# {info.patch_id} promotion evidence\n\n{evidence.strip()}\n",
        encoding="utf-8",
        newline="\n",
    )
    return path


def _write_promotion_record(info: PatchInfo, evidence: str) -> Path:
    readme = info.root / "README.md"
    existing = readme.read_text(encoding="utf-8") if readme.is_file() else f"# {info.patch_id}\n"
    block = f"\n\n## Promotion record\n\n{evidence.strip()}\n"
    if "## Promotion record" in existing:
        block = f"\n\n### Promotion evidence\n\n{evidence.strip()}\n"
    readme.write_text(existing.rstrip() + block, encoding="utf-8", newline="\n")
    return readme


def _replace_section_patches(text: str, section: str, patches: Iterable[str]) -> str:
    section_re = re.compile(
        rf"(?ms)(^\[{re.escape(section)}\]\s*\n)(.*?)(?=^\[|\Z)"
    )
    match = section_re.search(text)
    if not match:
        raise PatchPromoteError(f"config/recipes.toml: missing [{section}]")
    body = match.group(2)
    rendered_items = list(patches)
    if len(rendered_items) <= 1:
        rendered = "patches = [" + ", ".join(json.dumps(x) for x in rendered_items) + "]"
    else:
        rendered = "patches = [\n" + "".join(
            f"  {json.dumps(x)},\n" for x in rendered_items
        ) + "]"
    new_body, count = re.subn(
        r"(?ms)^patches\s*=\s*\[.*?\]",
        rendered,
        body,
        count=1,
    )
    if count != 1:
        raise PatchPromoteError(f"config/recipes.toml: [{section}] has no patches array")
    return text[: match.start(2)] + new_body + text[match.end(2) :]


def _append_section_patches(text: str, section: str, current: Iterable[str], added: Iterable[str]) -> str:
    """Append ids to a section's patches array in place: every existing line, comment and indent is kept."""
    added = list(added)
    if not added:
        return text
    section_re = re.compile(rf"(?ms)(^\[{re.escape(section)}\]\s*\n)(.*?)(?=^\[|\Z)")
    match = section_re.search(text)
    if not match:
        raise PatchPromoteError(f"config/recipes.toml: missing [{section}]")
    array = re.search(r"(?ms)^patches\s*=\s*\[[ \t]*\n(.*?)^\]", match.group(2))
    if not array:
        # a one-line or empty array has no per-entry comments to keep
        return _replace_section_patches(text, section, [*current, *added])
    body = array.group(1)
    indents = re.findall(r'(?m)^([ \t]*)"[^"\n]+"', body)
    indent = indents[-1] if indents else "    "
    lines = body.splitlines(keepends=True)
    for i in range(len(lines) - 1, -1, -1):  # the last entry needs a trailing comma before more follow
        entry = re.match(r'^([ \t]*"[^"\n]+")([ \t]*)(#.*)?(\r?\n?)$', lines[i])
        if entry:
            lines[i] = f"{entry.group(1)},{entry.group(2)}{entry.group(3) or ''}{entry.group(4)}"
            break
        if re.match(r'^[ \t]*"[^"\n]+",', lines[i]):
            break
    body = "".join(lines)
    if body and not body.endswith("\n"):
        body += "\n"
    body += "".join(f"{indent}{json.dumps(x)},\n" for x in added)
    start = match.start(2) + array.start(1)
    end = match.start(2) + array.end(1)
    return text[:start] + body + text[end:]


def _update_recipes(root: Path, patch_ids: tuple[str, ...]) -> Path:
    path = root / "config" / "recipes.toml"
    text = path.read_text(encoding="utf-8")
    raw = tomllib.loads(text)
    sets = raw.get("patch-set", {})
    production = list(sets.get("validated-enhancements", {}).get("patches", []))
    text = _append_section_patches(
        text, "patch-set.validated-enhancements", production, [p for p in patch_ids if p not in production]
    )

    experiments = raw.get("experiment", {})
    for name, config in experiments.items():
        selected = list(config.get("patches", []))
        reduced = [p for p in selected if p not in patch_ids]
        if reduced != selected:
            text = _replace_section_patches(text, f"experiment.{name}", reduced)
    path.write_text(text, encoding="utf-8", newline="\n")
    # Parse the emitted TOML so a formatting bug cannot escape to the checks.
    tomllib.loads(text)
    return path


def _check_command(root: Path, args: list[str], label: str) -> None:
    done = _run(root, *args, check=False)
    if done.returncode != 0:
        detail = (done.stderr or done.stdout).strip()
        raise PatchPromoteError(f"{label} failed:\n{detail[-1600:]}")


def _run_checks(root: Path, infos: tuple[PatchInfo, ...]) -> None:
    for info in infos:
        package_test = root / "tools" / "tests" / "patch" / f"test_{info.patch_id}.py"
        if package_test.is_file():
            _check_command(root, [sys.executable, str(package_test)], f"{info.patch_id} tests")
        _check_command(
            root,
            [
                sys.executable,
                "-m",
                "bigcherry",
                "patch-verify-evidence",
                info.patch_id,
                "--no-legacy-grandfather",
            ],
            f"{info.patch_id} evidence verification",
        )
    for test in (
        "tools/tests/patch/test_patch_catalog.py",
        "tools/tests/patch/test_patch_governance.py",
        "tools/tests/patch/test_recipes.py",
    ):
        _check_command(root, [sys.executable, test], test)
    _check_command(root, [sys.executable, "-m", "bigcherry", "patch-lint"], "patch-lint")
    _check_command(
        root,
        [sys.executable, "-m", "bigcherry", "patch-rebase-check", "--source", "bigcherry"],
        "production composition/rebase check",
    )


_CHECK_LABELS = {
    "mechanics": "mechanics",
    "activation": "activation",
    "identity": "identity",
    "a/b": "ab",
    "no-regression": "no-regression",
}


def _evidence_checks(evidence: str) -> dict[str, str]:
    """The 'Mechanics:', 'Activation:', 'Identity:', 'A/B:' and 'No-regression:' lines of an evidence file."""
    checks: dict[str, str] = {}
    for line in evidence.splitlines():
        match = re.match(r"\s*(?:[-*]\s*)?([A-Za-z/-]+)\s*:\s*(\S.*?)\s*$", line)
        if match and match.group(1).casefold() in _CHECK_LABELS:
            checks.setdefault(_CHECK_LABELS[match.group(1).casefold()], match.group(2))
    return checks


def _model_names(evidence: str) -> tuple[str, ...]:
    names: list[str] = []
    for line in evidence.splitlines():
        match = re.match(r"\s*(?:model|model-id)\s*:\s*(\S.*?)\s*$", line, re.I)
        if match and match.group(1).casefold() not in {n.casefold() for n in names}:
            names.append(match.group(1))
    return tuple(names)


def _is_default_on(info: PatchInfo) -> bool:
    """True when the patch declares a runtime flag that is on unless switched off."""
    py_path = info.root / "patch.py"
    if not py_path.is_file():
        return False
    return re.search(r'EnvDoc\(\s*"[^"]+"[\s\S]{0,240}?"1 \(on\)"', py_path.read_text(encoding="utf-8")) is not None


def _profile_evidence(root: Path, info: PatchInfo, evidence: str) -> profile_evidence.ProfileEvidence:
    """The lightweight promotion record for a patch, bound to the current pin and to its implementation.

    Built after any default flip, so the digest is the one of the implementation that is promoted.
    """
    pinned = str(tomllib.loads((root / "config" / "recipes.toml").read_text(encoding="utf-8"))["pinned"])
    record = profile_evidence.ProfileEvidence(
        patch_id=info.patch_id,
        pinned=pinned,
        subject_digest=patch_evidence.patch_validation_subject_digest(info.root / "patch.py"),
        default_on=_is_default_on(info),
        models=_model_names(evidence),
        checks=_evidence_checks(evidence),
    )
    lacking = profile_evidence.problems(record)
    if lacking:
        raise PatchPromoteError(
            f"{info.patch_id}: the evidence file does not meet the profile-evidence tier: " + "; ".join(lacking)
            + ". It needs 'Model:', 'Mechanics:', 'Activation:', 'Identity:' and 'A/B:' lines, and for a "
            "default-on patch a 'No-regression:' line and a second 'Model:' line."
        )
    return record


def _preflight(root: Path, infos: tuple[PatchInfo, ...]) -> None:
    """Promotion gates for each patch, against a rebase report made now for exactly that patch over production."""
    for info in infos:
        report = root / "artifacts" / "patch-promote" / f"{info.patch_id}-rebase.json"
        report.parent.mkdir(parents=True, exist_ok=True)
        _check_command(
            root,
            [
                sys.executable, "-m", "bigcherry", "patch-rebase-check",
                "--source", "bigcherry", "--focal-overlay", info.patch_id, "--json", str(report),
            ],
            f"{info.patch_id} rebase check over production",
        )
        _check_command(
            root,
            [
                sys.executable,
                "-m",
                "bigcherry",
                "patch-gates",
                info.patch_id,
                "--intent",
                "promote",
                "--source",
                "bigcherry",
                "--focal-overlay",
                "--no-legacy-grandfather",
                "--rebase-report",
                str(report),
            ],
            f"{info.patch_id} promotion gates",
        )


def _snapshot(paths: Iterable[Path]) -> dict[Path, bytes | None]:
    return {path: path.read_bytes() if path.exists() else None for path in paths}


def _restore(snapshot: dict[Path, bytes | None]) -> None:
    for path, content in snapshot.items():
        if content is None:
            path.unlink(missing_ok=True)
        else:
            path.write_bytes(content)


def _discard_promotion_slice(root: Path, worktree: Path, branch: str) -> None:
    # This worktree/branch was created by patch-promote and has not been
    # pushed yet. Restore its HEAD, remove it without --force, then drop the
    # ephemeral local ref. The primary checkout is never switched.
    _git(worktree, "reset", "--hard", "HEAD", check=False)
    _git(root, "worktree", "remove", str(worktree), check=False)
    _git(root, "update-ref", "-d", f"refs/heads/{branch}", check=False)
    _git(root, "worktree", "prune", check=False)


def _branch_name(infos: tuple[PatchInfo, ...]) -> str:
    numeric = "-".join(info.patch_id.split("_", 1)[0] for info in infos)
    item = infos[0].plan_ids[0].lower() if len(infos) == 1 and infos[0].plan_ids else "pa45"
    return f"feat/{item}-promote-{numeric}"


def _title(infos: tuple[PatchInfo, ...]) -> tuple[str, str]:
    enhancement = any(info.kind == "enhancement" for info in infos)
    prefix = "feat" if enhancement else "fix"
    ids = ", ".join(info.patch_id for info in infos)
    return prefix, f"{prefix}(patch): promote {ids}"


def _ledger_payload(infos: tuple[PatchInfo, ...], changed: Iterable[str], title: str) -> dict:
    plan_ids = sorted({p for info in infos for p in info.plan_ids})
    return {
        "change_class": "feature" if title.startswith("feat") else "code-fix",
        "files": sorted(set(changed)),
        "summary": title,
        "plan_item_ids": plan_ids,
    }


def _release_version(root: Path, bump: str) -> tuple[str, str, int]:
    recipes = tomllib.loads((root / "config" / "recipes.toml").read_text(encoding="utf-8"))
    llama_tag = str(recipes["pinned"])
    match = re.fullmatch(r"b(\d+)", llama_tag)
    if not match:
        raise PatchPromoteError(f"invalid pinned llama.cpp tag {llama_tag!r}")
    build = int(match.group(1))
    config = release_notes.load_config(root)
    remote = _git(root, "ls-remote", "--tags", "--refs", "origin", f"refs/tags/{config.tag_prefix}{build}.*", check=False)
    local = _git(root, "tag", "--list", f"{config.tag_prefix}{build}.*", check=False)
    names = {line.split("refs/tags/")[-1] for line in remote.splitlines() if "refs/tags/" in line}
    names.update(local.split())
    existing = [name[len(config.tag_prefix):] for name in names if name.startswith(config.tag_prefix)]
    version = pin_release.next_version(existing, build, bump)
    return llama_tag, version, build


def _finish_release_if_ready(root: Path, bump: str) -> bool:
    llama_tag, version, build = _release_version(root, bump)
    config = release_notes.load_config(root)
    branch = _git(root, "rev-parse", "--abbrev-ref", "HEAD")
    plan = pin_release.Plan(
        repo_root=root,
        llama_tag=llama_tag,
        build=build,
        release_tag=f"{config.tag_prefix}{version}",
        version=version,
        branch=branch,
    )
    pr = pin_release._release_pr(plan)
    if pr is None:
        pin_release._open_release_pr(plan)
        pr = pin_release._release_pr(plan)
    if pr is None:
        print(
            f"release pending: promotion must merge first; release-please will propose {version}"
        )
        return False
    pin_release.phase_release(plan)
    print(f"released {plan.release_tag}")
    return True


def promote(
    root: Path,
    patch_ids: tuple[str, ...],
    evidence_spec: str,
    *,
    default_on: bool = False,
    profile_only: bool = False,
    release: bool = False,
) -> PromotionResult:
    root = root.resolve()
    if not patch_ids:
        raise PatchPromoteError("at least one patch id is required")
    _, evidence = _read_evidence(evidence_spec)
    if not profile_only and len(_named_models(evidence)) < 2:
        raise PatchPromoteError(
            "promotion requires evidence with at least two distinct "
            "'Model:' or 'Model-ID:' lines; use --profile-only only for "
            "an explicitly profile-scoped promotion"
        )
    infos = tuple(_load_patch(root, patch_id) for patch_id in patch_ids)
    if all(info.state == "validated" for info in infos):
        if not release:
            raise PatchPromoteError("all requested patches are already validated")
        bump = "minor" if any(i.kind == "enhancement" for i in infos) else "patch"
        _finish_release_if_ready(root, bump)
        return PromotionResult("", _title(infos)[1], "", (), release_pending=True)
    if any(info.state == "validated" for info in infos):
        raise PatchPromoteError("cannot batch already-validated and unvalidated patches")

    if _git(root, "status", "--porcelain"):
        raise PatchPromoteError("working tree must be clean")
    _git(root, "fetch", "-q", "origin", "main")
    current = _git(root, "rev-parse", "--abbrev-ref", "HEAD")
    if current != "main":
        raise PatchPromoteError(f"run patch-promote from main, not {current!r}")
    if _git(root, "rev-parse", "HEAD") != _git(root, "rev-parse", "origin/main"):
        raise PatchPromoteError("local main must exactly match origin/main")

    for info in infos:  # fail on an incomplete evidence file before anything is created
        _profile_evidence(root, info, evidence)
    branch = _branch_name(infos)
    from ..cli.slice import start_slice

    try:
        worktree = start_slice(branch, primary_root=root)
    except RuntimeError as exc:
        raise PatchPromoteError(str(exc)) from exc

    work_infos = tuple(_load_patch(worktree, patch_id) for patch_id in patch_ids)
    try:
        changed: list[Path] = []
        # The record is written and the gates run while the patch is still unpromoted: the gates evaluate it as a
        # candidate over production, with the evidence that authorises the state change already in the package.
        for info in work_infos:
            if default_on:
                changed.extend(_promote_default_on(info, evidence))
            changed.append(profile_evidence.write(info.root, _profile_evidence(worktree, info, evidence)))
        _preflight(worktree, work_infos)
        for info in work_infos:
            changed.extend(_set_state(info))
            changed.append(_write_promotion_record(info, evidence))
            changed.append(_write_release_evidence(worktree, info, evidence))
        changed.append(_update_recipes(worktree, patch_ids))
        _run_checks(worktree, work_infos)
    except Exception:
        _discard_promotion_slice(root, worktree, branch)
        raise

    prefix, title = _title(work_infos)
    relative = tuple(
        sorted({str(path.resolve().relative_to(worktree)).replace("\\", "/") for path in changed})
    )
    _git(worktree, "add", "--", *relative)
    _git(worktree, "commit", "-m", title)
    _git(worktree, "push", "-u", "origin", branch)
    pr_body = (
        "## What this slice does\n\n"
        f"Promotes {', '.join(patch_ids)} into `[patch-set.validated-enhancements]` "
        "using the supplied qualification evidence, recorded in both the patch README "
        "and releases/evidence/.\n\n"
        "## Verified\n\n"
        "- promotion gates with current evidence\n"
        "- patch-local tests where present\n"
        "- catalog/governance/recipe tests\n"
        "- patch-lint and production patch-rebase-check\n\n"
        "## Could not verify\n\n"
        "- hardware measurements were not rerun; the supplied evidence is recorded verbatim in the patch README.\n"
    )
    made = _run(
        worktree,
        "gh",
        "pr",
        "create",
        "--base",
        "main",
        "--head",
        branch,
        "--title",
        title,
        "--body",
        pr_body,
        check=False,
    )
    if made.returncode != 0:
        raise PatchPromoteError(
            "promotion committed and pushed, but PR creation failed: "
            + (made.stderr or made.stdout).strip()[-800:]
        )
    pr_url = made.stdout.strip()
    payload = _ledger_payload(work_infos, relative, title)
    print("record_change_event " + json.dumps(payload, sort_keys=True))

    pending = False
    if release:
        bump = "minor" if prefix == "feat" else "patch"
        pending = not _finish_release_if_ready(root, bump)

    return PromotionResult(branch, title, pr_url, relative, pending)


def cmd_patch_promote(args) -> int:
    from ..core import paths as core_paths

    try:
        result = promote(
            core_paths.REPO_ROOT,
            tuple(args.patch_ids),
            args.evidence,
            default_on=args.default_on,
            profile_only=args.profile_only,
            release=args.release,
        )
    except PatchPromoteError as exc:
        print(f"patch-promote: {exc}", file=sys.stderr)
        return 1
    if result.pr_url:
        print(f"patch-promote: {result.pr_url}")
    return 0
