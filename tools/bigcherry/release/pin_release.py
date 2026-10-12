"""Release a completed pin bump through pull requests and release-please.

Phases are resumable and idempotent:

  gate     `pin-status --complete --all-remotes` must pass
  record   store gate output/hardware evidence and clear the transition marker
  notes    generate notes and set the llama.cpp package's release-please `release-as`
  main     publish/reuse the work-branch PR, wait for checks, merge it
  release  validate and merge release-please's component PR, wait for the tag
  unpin    remove `release-as` through a dedicated PR after the tag exists
  sync     merge current main back into the work branch

No phase pushes directly to main. The release version is carried only by the
package-scoped `release-as` key; commit-message Release-As footers are not used.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

from . import notes as _notes

PHASES = ("gate", "record", "notes", "main", "release", "unpin", "sync")


# The release-please package (a path in release-please-config.json) that carries the llama.cpp engine's release line.
RELEASE_PACKAGE = "engines/llamacpp"
_RELEASE_CONFIG = "release-please-config.json"
_MANIFEST = ".release-please-manifest.json"
_RULESET = ".github/rulesets/main.json"

_CHECK_SUCCESS = {"SUCCESS", "NEUTRAL", "SKIPPED"}
_CHECK_FAILURE = {
    "FAILURE", "ERROR", "CANCELLED", "TIMED_OUT", "ACTION_REQUIRED",
    "STALE", "STARTUP_FAILURE",
}


class PinReleaseError(RuntimeError):
    """A phase could not complete; the message says what to do before running again."""

    def __init__(self, phase: str, message: str):
        super().__init__(message)
        self.phase = phase


@dataclass(frozen=True)
class Plan:
    repo_root: Path
    llama_tag: str       # b11474
    build: int           # 11474
    release_tag: str     # bc-11474.0.0 - release-please's tag, the tag of record
    version: str         # 11474.0.0
    branch: str          # the work branch (current)
    main: str = "main"
    remote: str = "origin"


def _run(root: Path, *cmd: str, check: bool = True, timeout: int = 600) -> subprocess.CompletedProcess:
    res = subprocess.run(list(cmd), cwd=root, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=timeout)
    if check and res.returncode != 0:
        raise RuntimeError(f"{' '.join(cmd)} failed ({res.returncode}): {(res.stderr or res.stdout).strip()[-600:]}")
    return res


def _git(root: Path, *args: str, check: bool = True) -> str:
    return _run(root, "git", *args, check=check).stdout.strip()


def next_version(existing: list[str], build: int, bump: str | None) -> str:
    """The version to release: <build>.0.0 when the pin has no release yet, else the newest one raised by ``bump``.
    ``existing`` are the versions (without the tag prefix) already tagged for this build."""
    released = sorted(tuple(int(x) for x in v.split(".")) for v in existing
                      if re.fullmatch(rf"{build}\.\d+\.\d+", v))
    if not released:
        if bump is not None:
            raise PinReleaseError("gate", f"llama.cpp b{build} has no release yet: its first release is {build}.0.0, without --bump")
        return f"{build}.0.0"
    if bump is None:
        newest = ".".join(str(x) for x in released[-1])
        raise PinReleaseError("gate", f"b{build} is already released as {newest}: pass --bump minor (features) or --bump patch (fixes) "
                                      "for a further release on this pin, or --version")
    _, minor, patch = released[-1]
    return f"{build}.{minor + 1}.0" if bump == "minor" else f"{build}.{minor}.{patch + 1}"


def make_plan(repo_root: Path, llama_tag: str, main: str = "main", remote: str = "origin",
              version: str | None = None, bump: str | None = None, allow_existing: bool = False) -> Plan:
    m = re.fullmatch(r"b(\d+)", llama_tag)
    if not m:
        raise PinReleaseError("gate", f"{llama_tag!r} is not a llama.cpp release tag (b<number>)")
    build = int(m.group(1))
    config = _notes.load_config(repo_root)
    branch = _git(repo_root, "rev-parse", "--abbrev-ref", "HEAD")
    if branch in (main, "HEAD"):
        raise PinReleaseError("gate", f"run this from the work branch, not from {branch!r}")
    if version is None:
        refs = _git(repo_root, "ls-remote", "--tags", "--refs", remote, f"refs/tags/{config.tag_prefix}{build}.*", check=False)
        local = _git(repo_root, "tag", "--list", f"{config.tag_prefix}{build}.*", check=False)
        names = {line.split("refs/tags/")[-1] for line in refs.splitlines() if "refs/tags/" in line} | set(local.split())
        existing = [name[len(config.tag_prefix):] for name in names]
        # A notes commit identifies the release this work branch was publishing. Keep using it after the tag
        # appears so post-tag unpin/sync can resume with the same command. A requested --bump deliberately starts
        # a new version instead of reusing a completed notes commit.
        notes_log = _git(repo_root, "log", "--format=%s", "--fixed-strings",
                         f"--grep=chore: release notes {config.tag_prefix}{build}.", "-20", check=False)
        note_versions = [line.rsplit(config.tag_prefix, 1)[-1] for line in notes_log.splitlines()
                         if config.tag_prefix in line]
        pending = [candidate for candidate in note_versions if candidate not in existing]
        if pending:
            version = pending[0]
        elif bump is None and note_versions:
            version = note_versions[0]
        elif allow_existing and bump is None:
            released = sorted(tuple(int(x) for x in v.split(".")) for v in existing
                              if re.fullmatch(rf"{build}\.\d+\.\d+", v))
            version = ".".join(str(x) for x in released[-1]) if released else next_version(existing, build, bump)
        else:
            version = next_version(existing, build, bump)
    try:
        version = _notes.release_version(llama_tag, version)
    except _notes.ReleaseNotesError as exc:
        raise PinReleaseError("gate", str(exc)) from exc
    return Plan(repo_root, llama_tag, build, f"{config.tag_prefix}{version}", version, branch, main, remote)


def _require_clean(plan: Plan, phase: str) -> None:
    dirty = _git(plan.repo_root, "status", "--porcelain")
    if dirty:
        raise PinReleaseError(phase, "the checkout has uncommitted changes:\n" + dirty[:600])


def _tag_exists(plan: Plan) -> bool:
    return bool(_git(plan.repo_root, "ls-remote", "--tags", plan.remote, f"refs/tags/{plan.release_tag}", check=False))


def _release_package(root: Path, phase: str = "notes") -> dict:
    try:
        document = json.loads((root / _RELEASE_CONFIG).read_text(encoding="utf-8"))
        package = document["packages"][RELEASE_PACKAGE]
    except (OSError, json.JSONDecodeError, KeyError, TypeError) as exc:
        raise PinReleaseError(phase, f"{_RELEASE_CONFIG} has no valid {RELEASE_PACKAGE!r} package") from exc
    if not isinstance(package, dict):
        raise PinReleaseError(phase, f"{_RELEASE_CONFIG} package {RELEASE_PACKAGE!r} is not an object")
    return package


def _package_object_bounds(text: str) -> tuple[int, int]:
    marker = re.search(rf'(?m)^\s*{re.escape(json.dumps(RELEASE_PACKAGE))}\s*:\s*\{{', text)
    if marker is None:
        raise ValueError(f"package {RELEASE_PACKAGE!r} not found")
    opening = text.find("{", marker.start(), marker.end())
    depth = 0
    in_string = False
    escaped = False
    for i in range(opening, len(text)):
        ch = text[i]
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return opening, i
    raise ValueError(f"package {RELEASE_PACKAGE!r} object is not closed")


def _release_as_value(text: str, phase: str) -> str | None:
    try:
        value = json.loads(text)["packages"][RELEASE_PACKAGE].get("release-as")
    except (json.JSONDecodeError, KeyError, TypeError) as exc:
        raise PinReleaseError(phase, f"{_RELEASE_CONFIG} has no valid {RELEASE_PACKAGE!r} package") from exc
    if value is not None and not isinstance(value, str):
        raise PinReleaseError(phase, f"{RELEASE_PACKAGE}.release-as must be a string, got {value!r}")
    return value


def _set_release_as_text(text: str, version: str | None, phase: str, expected: str | None = None) -> tuple[str, bool]:
    current = _release_as_value(text, phase)
    if version is not None:
        if current is not None and current != version:
            raise PinReleaseError(phase, f"{RELEASE_PACKAGE}.release-as is already {current!r}, refusing to replace it with {version!r}")
        if current == version:
            return text, False
        try:
            opening, _ = _package_object_bounds(text)
        except ValueError as exc:
            raise PinReleaseError(phase, str(exc)) from exc
        line = re.match(r"(\r?\n)([ \t]+)", text[opening + 1:])
        if line is None:
            raise PinReleaseError(phase, f"cannot preserve formatting while editing {RELEASE_PACKAGE} in {_RELEASE_CONFIG}")
        newline, indent = line.group(1), line.group(2)
        updated = text[:opening + 1] + f'{newline}{indent}"release-as": {json.dumps(version)},' + text[opening + 1:]
        if _release_as_value(updated, phase) != version:
            raise PinReleaseError(phase, f"failed to set {RELEASE_PACKAGE}.release-as")
        return updated, True

    if current is None:
        return text, False
    if expected is not None and current != expected:
        raise PinReleaseError(phase, f"{RELEASE_PACKAGE}.release-as is {current!r}, expected {expected!r}; refusing to clear it")
    try:
        opening, closing = _package_object_bounds(text)
    except ValueError as exc:
        raise PinReleaseError(phase, str(exc)) from exc
    package_text = text[opening + 1:closing]
    match = re.search(r'(?m)^[ \t]*"release-as"[ \t]*:[^\r\n]*(?:\r?\n|$)', package_text)
    if match is None:
        raise PinReleaseError(phase, f"could not locate {RELEASE_PACKAGE}.release-as without reformatting {_RELEASE_CONFIG}")
    start = opening + 1 + match.start()
    end = opening + 1 + match.end()
    updated = text[:start] + text[end:]
    try:
        json.loads(updated)
    except json.JSONDecodeError:
        comma = text.rfind(",", opening + 1, start)
        if comma < 0:
            raise PinReleaseError(phase, f"could not remove {RELEASE_PACKAGE}.release-as without reformatting {_RELEASE_CONFIG}")
        updated = text[:comma] + text[comma + 1:start] + text[end:]
        try:
            json.loads(updated)
        except json.JSONDecodeError as exc:
            raise PinReleaseError(phase, f"could not remove {RELEASE_PACKAGE}.release-as without reformatting {_RELEASE_CONFIG}") from exc
    if _release_as_value(updated, phase) is not None:
        raise PinReleaseError(phase, f"failed to clear {RELEASE_PACKAGE}.release-as")
    return updated, True


def _edit_release_as(root: Path, version: str | None, phase: str, expected: str | None = None) -> bool:
    path = root / _RELEASE_CONFIG
    original = path.read_text(encoding="utf-8")
    updated, changed = _set_release_as_text(original, version, phase, expected)
    if changed:
        path.write_text(updated, encoding="utf-8", newline="")
    return changed


def phase_gate(plan: Plan) -> str:
    """Every required tree converged at the pin. Returns the gate's output, which the record keeps."""
    _require_clean(plan, "gate")
    res = _run(plan.repo_root, sys.executable, "-m", "bigcherry", "pin-status", "--complete", "--all-remotes", check=False, timeout=900)
    out = (res.stdout + res.stderr).strip()
    if res.returncode != 0:
        raise PinReleaseError("gate", "pin-status --complete --all-remotes did not pass:\n" + out[-1500:]
                              + "\nbring every required tree to the pin (pull it, run pin-bump there), then run again")
    return out


def phase_record(plan: Plan, gate_output: str, evidence: str) -> bool:
    """Completion into the release record, transition marker removed. Returns False when already done."""
    root = plan.repo_root
    record_path = root / "releases" / f"{plan.llama_tag}.json"
    marker = root / "releases" / "pin-transition.json"
    if not record_path.is_file():
        raise PinReleaseError("record", f"{record_path} does not exist: was `pin-bump {plan.llama_tag}` run in this tree?")
    record = json.loads(record_path.read_text(encoding="utf-8"))
    stamp = f"release {plan.version}"
    if stamp in (record.get("notes") or "") and not marker.exists():
        return False
    if not evidence.strip():
        raise PinReleaseError("record", "--evidence is required: state the hardware build, smoke and A/B result that makes this bump "
                                        "releasable (text, or @file); it is recorded verbatim in the release record")
    note = (f"{stamp} {time.strftime('%Y-%m-%d')}\n\ncompletion gate (pin-status --complete --all-remotes):\n{gate_output.strip()}\n\n"
            f"hardware evidence:\n{evidence.strip()}\n")
    record["notes"] = ((record.get("notes") or "").rstrip() + "\n\n" + note).lstrip()
    record_path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    _git(root, "add", str(record_path.relative_to(root)).replace("\\", "/"))
    if marker.exists():
        _git(root, "rm", "-q", "releases/pin-transition.json")
    what = "bump complete -- " if plan.version.endswith(".0.0") else ""
    _git(root, "commit", "-q", "-m", f"release: {plan.release_tag} {what}completion gate and hardware evidence recorded"
                                     + (", transition marker cleared" if what else ""))
    return True


def phase_notes(plan: Plan) -> bool:
    """Generate notes and pin the llama.cpp package version in one commit."""
    root = plan.repo_root
    subject = f"chore: release notes {plan.release_tag}"
    config_text = (root / _RELEASE_CONFIG).read_text(encoding="utf-8")
    committed = subject in _git(root, "log", "--format=%s", "--fixed-strings", f"--grep={subject}", "-5").splitlines()
    current_release_as = _release_as_value(config_text, "notes")
    if current_release_as is not None and current_release_as != plan.version:
        raise PinReleaseError("notes", f"{RELEASE_PACKAGE}.release-as is already {current_release_as!r}, "
                                       f"refusing to replace it with {plan.version!r}")
    if committed and (current_release_as == plan.version or (current_release_as is None and _tag_exists(plan))):
        return False
    _require_clean(plan, "notes")
    try:
        out = _notes.write_notes(root, _notes.load_config(root), plan.llama_tag, "HEAD", plan.version)
    except _notes.ReleaseNotesError as exc:
        raise PinReleaseError("notes", str(exc)) from exc
    _edit_release_as(root, plan.version, "notes")
    note_path = str(Path(out).resolve().relative_to(root.resolve())).replace("\\", "/")
    _git(root, "add", note_path, _RELEASE_CONFIG)
    body = ("release-please reads the package-scoped release-as value from release-please-config.json; "
            f"this commit carries BigCherry on llama.cpp {plan.llama_tag}.")
    _git(root, "commit", "-q", "-m", subject, "-m", body)
    return True


def _pr_for_head(plan: Plan, head: str, phase: str) -> dict | None:
    res = _run(plan.repo_root, "gh", "pr", "list", "--base", plan.main, "--head", head, "--state", "open",
               "--json", "number,title,headRefName,url", check=False)
    if res.returncode != 0:
        raise PinReleaseError(phase, "gh pr list failed: " + (res.stderr or res.stdout).strip()[-400:])
    rows = json.loads(res.stdout or "[]")
    return rows[0] if rows else None


def _ensure_pr(plan: Plan, head: str, title: str, phase: str, body: str) -> dict:
    current = _pr_for_head(plan, head, phase)
    if current is not None:
        return current
    made = _run(plan.repo_root, "gh", "pr", "create", "--base", plan.main, "--head", head,
                "--title", title, "--body", body, check=False)
    if made.returncode != 0:
        raise PinReleaseError(phase, f"could not open pull request {head} -> {plan.main}: "
                                      + (made.stderr or made.stdout).strip()[-400:])
    current = _pr_for_head(plan, head, phase)
    if current is None:
        raise PinReleaseError(phase, f"pull request for {head} was created but could not be found")
    return current


def _required_checks(plan: Plan, phase: str) -> tuple[str, ...]:
    """Required status-check contexts for main, read from the checked-in ruleset."""
    try:
        document = json.loads((plan.repo_root / _RULESET).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PinReleaseError(phase, f"cannot read {_RULESET}: {exc}") from exc
    names: list[str] = []
    for rule in document.get("rules", []):
        if not isinstance(rule, dict) or rule.get("type") != "required_status_checks":
            continue
        rows = rule.get("parameters", {}).get("required_status_checks", [])
        for row in rows:
            name = row.get("context") if isinstance(row, dict) else None
            if isinstance(name, str) and name and name not in names:
                names.append(name)
    if not names:
        raise PinReleaseError(phase, f"{_RULESET} does not declare any required status checks")
    return tuple(names)


def _pr_state(plan: Plan, pr: dict, phase: str) -> dict:
    viewed = _run(
        plan.repo_root, "gh", "pr", "view", str(pr["number"]),
        "--json", "mergeStateStatus,mergeable,state,url,headRefName", check=False,
    )
    if viewed.returncode != 0:
        raise PinReleaseError(
            phase, f"could not inspect pull request {pr['number']}: "
            + (viewed.stderr or viewed.stdout).strip()[-400:],
        )
    try:
        state = json.loads(viewed.stdout or "{}")
    except json.JSONDecodeError as exc:
        raise PinReleaseError(phase, f"could not parse pull request {pr['number']} state") from exc
    if not isinstance(state, dict):
        raise PinReleaseError(phase, f"pull request {pr['number']} state is not an object")
    return state


def _merge_manifest_text(release_text: str, main_text: str, phase: str = "release") -> str:
    """Resolve the shared manifest without reformatting it: this package from the release side, all others from main."""
    try:
        release = json.loads(release_text)
        main = json.loads(main_text)
    except json.JSONDecodeError as exc:
        raise PinReleaseError(phase, f"cannot resolve {_MANIFEST}: invalid JSON side") from exc
    if not isinstance(release, dict) or not isinstance(main, dict):
        raise PinReleaseError(phase, f"cannot resolve {_MANIFEST}: both sides must be objects")
    if RELEASE_PACKAGE not in release:
        raise PinReleaseError(phase, f"cannot resolve {_MANIFEST}: release side has no {RELEASE_PACKAGE!r} key")
    if set(release) != set(main):
        raise PinReleaseError(
            phase, f"cannot resolve {_MANIFEST}: package keys differ between release and main; refusing to invent formatting",
        )

    desired = dict(main)
    desired[RELEASE_PACKAGE] = release[RELEASE_PACKAGE]
    updated = release_text
    for key, value in desired.items():
        if not isinstance(value, str):
            raise PinReleaseError(phase, f"cannot resolve {_MANIFEST}: {key!r} has non-string version {value!r}")
        pattern = re.compile(
            rf'(?m)^([ \t]*){re.escape(json.dumps(key))}([ \t]*:[ \t]*)'
            r'(?P<value>"(?:\\.|[^"\\])*")(?P<tail>[ \t]*,?[ \t]*(?:\r?\n|$))'
        )
        match = pattern.search(updated)
        if match is None:
            raise PinReleaseError(phase, f"cannot preserve formatting while resolving {_MANIFEST} key {key!r}")
        encoded = json.dumps(value)
        updated = updated[:match.start("value")] + encoded + updated[match.end("value"):]

    try:
        resolved = json.loads(updated)
    except json.JSONDecodeError as exc:
        raise PinReleaseError(phase, f"resolved {_MANIFEST} is invalid JSON") from exc
    if resolved != desired:
        raise PinReleaseError(phase, f"resolved {_MANIFEST} does not match the release/main ownership rule")
    return updated


def _merge_pr_branch_locally(plan: Plan, pr: dict, phase: str) -> None:
    """Merge main into a conflicting PR branch in a temporary worktree; only the shared manifest may conflict."""
    root, remote = plan.repo_root, plan.remote
    state = _pr_state(plan, pr, phase)
    head = state.get("headRefName") or pr.get("headRefName")
    if not isinstance(head, str) or not head:
        raise PinReleaseError(phase, f"pull request {pr['number']} has no head branch")
    _git(
        root, "fetch", "-q", remote,
        f"{plan.main}:refs/remotes/{remote}/{plan.main}",
        f"{head}:refs/remotes/{remote}/{head}",
    )
    parent = Path(tempfile.mkdtemp(prefix="pin-release-pr-update-", dir=root.parent))
    worktree = parent / "tree"
    try:
        _git(root, "worktree", "add", "-q", "--detach", str(worktree), f"{remote}/{head}")
        merge = _run(
            worktree, "git", "merge", "--no-edit", "-m",
            f"merge {plan.main} into {head} before pull request merge",
            f"{remote}/{plan.main}", check=False,
        )
        if merge.returncode != 0:
            conflicts = tuple(
                line for line in _git(worktree, "diff", "--name-only", "--diff-filter=U", check=False).splitlines()
                if line
            )
            unexpected = tuple(path for path in conflicts if path != _MANIFEST)
            if unexpected or conflicts != (_MANIFEST,):
                _run(worktree, "git", "merge", "--abort", check=False)
                names = sorted(unexpected or conflicts)
                raise PinReleaseError(
                    phase, f"pull request {pr['number']} conflicts with {plan.main} in {', '.join(names) or 'unknown files'}; "
                    f"only {_MANIFEST} may be resolved automatically",
                )
            release_text = _git(worktree, "show", f":2:{_MANIFEST}")
            main_text = _git(worktree, "show", f":3:{_MANIFEST}")
            (worktree / _MANIFEST).write_text(
                _merge_manifest_text(release_text, main_text, phase),
                encoding="utf-8", newline="",
            )
            _git(worktree, "add", _MANIFEST)
            remaining = _git(worktree, "diff", "--name-only", "--diff-filter=U", check=False)
            if remaining:
                _run(worktree, "git", "merge", "--abort", check=False)
                raise PinReleaseError(phase, f"pull request {pr['number']} still has unresolved conflicts: {remaining}")
            _git(worktree, "commit", "-q", "--no-edit")
        pushed = _run(worktree, "git", "push", "-q", remote, f"HEAD:refs/heads/{head}", check=False)
        if pushed.returncode != 0:
            raise PinReleaseError(
                phase, f"could not push updated pull request branch {head}: "
                + (pushed.stderr or pushed.stdout).strip()[-400:],
            )
    finally:
        _run(root, "git", "worktree", "remove", "--force", str(worktree), check=False)
        shutil.rmtree(parent, ignore_errors=True)


def _bring_pr_up_to_main(plan: Plan, pr: dict, phase: str, log=print) -> bool:
    """Bring a PR branch to current main without touching the user's checkout."""
    state = _pr_state(plan, pr, phase)
    merge_state = str(state.get("mergeStateStatus") or "").upper()
    mergeable = str(state.get("mergeable") or "").upper()
    behind = merge_state in {"BEHIND", "DIRTY"} or mergeable == "CONFLICTING"
    if not behind:
        return False

    if mergeable == "CONFLICTING" or merge_state == "DIRTY":
        log(f"  {phase:<8} PR #{pr['number']} is behind with conflicts; merging {plan.main} in a temporary worktree")
        _merge_pr_branch_locally(plan, pr, phase)
        return True

    updated = _run(plan.repo_root, "gh", "pr", "update-branch", str(pr["number"]), check=False)
    if updated.returncode == 0:
        log(f"  {phase:<8} PR #{pr['number']} was behind; updated it to {plan.main} with gh pr update-branch")
        return True

    refreshed = _pr_state(plan, pr, phase)
    if (
        str(refreshed.get("mergeable") or "").upper() == "CONFLICTING"
        or str(refreshed.get("mergeStateStatus") or "").upper() == "DIRTY"
    ):
        log(f"  {phase:<8} PR #{pr['number']} became conflicting; merging {plan.main} in a temporary worktree")
        _merge_pr_branch_locally(plan, pr, phase)
        return True
    raise PinReleaseError(
        phase, f"could not update pull request {pr['number']} to {plan.main}: "
        + (updated.stderr or updated.stdout).strip()[-400:],
    )


def _trigger_pr_checks(plan: Plan, pr: dict, phase: str, log=print) -> None:
    state = _pr_state(plan, pr, phase)
    if (
        str(state.get("mergeStateStatus") or "").upper() in {"BEHIND", "DIRTY"}
        or str(state.get("mergeable") or "").upper() == "CONFLICTING"
    ):
        _bring_pr_up_to_main(plan, pr, phase, log)
        log(f"  {phase:<8} triggered required checks by updating PR #{pr['number']} to {plan.main}")
        return

    closed = _run(plan.repo_root, "gh", "pr", "close", str(pr["number"]), check=False)
    if closed.returncode != 0:
        raise PinReleaseError(
            phase, f"required checks never started and PR #{pr['number']} could not be closed to trigger them: "
            + (closed.stderr or closed.stdout).strip()[-400:],
        )
    reopened = _run(plan.repo_root, "gh", "pr", "reopen", str(pr["number"]), check=False)
    if reopened.returncode != 0:
        raise PinReleaseError(
            phase, f"PR #{pr['number']} was closed to trigger required checks but could not be reopened: "
            + (reopened.stderr or reopened.stdout).strip()[-400:],
        )
    log(f"  {phase:<8} required checks had not started; closed and reopened PR #{pr['number']} once to trigger them")


def _wait_required_checks(
    plan: Plan,
    pr: dict,
    phase: str,
    wait_s: int = 900,
    poll_s: int = 5,
    grace_s: int = 20,
    log=print,
) -> None:
    required = _required_checks(plan, phase)
    start = time.time()
    deadline = start + wait_s
    grace_deadline = min(deadline, start + grace_s)
    triggered = False

    while True:
        result = _run(
            plan.repo_root, "gh", "pr", "checks", str(pr["number"]),
            "--json", "name,state,link", check=False,
        )
        raw = result.stdout.strip()
        if raw:
            try:
                rows = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise PinReleaseError(phase, f"could not parse checks for pull request {pr['number']}") from exc
            if not isinstance(rows, list):
                raise PinReleaseError(phase, f"checks for pull request {pr['number']} are not a list")
        else:
            detail = (result.stderr or "").strip()
            if result.returncode != 0 and "no checks reported" not in detail.lower():
                raise PinReleaseError(
                    phase, f"could not read checks for pull request {pr['number']}: {detail[-400:]}",
                )
            rows = []

        by_name = {
            row.get("name"): row for row in rows
            if isinstance(row, dict) and row.get("name") in required
        }
        for name in required:
            row = by_name.get(name)
            if row is None:
                continue
            state = str(row.get("state") or "").upper()
            if state in _CHECK_FAILURE:
                url = row.get("link") or pr.get("url") or f"PR #{pr['number']}"
                raise PinReleaseError(phase, f"required check {name} failed: {url}")

        if all(
            name in by_name and str(by_name[name].get("state") or "").upper() in _CHECK_SUCCESS
            for name in required
        ):
            return

        now = time.time()
        if not by_name and not triggered and now >= grace_deadline:
            _trigger_pr_checks(plan, pr, phase, log)
            triggered = True
        if now >= deadline:
            pending = [
                name for name in required
                if name not in by_name or str(by_name[name].get("state") or "").upper() not in _CHECK_SUCCESS
            ]
            raise PinReleaseError(
                phase, f"required checks for pull request {pr['number']} did not pass within {wait_s} s: "
                + ", ".join(pending),
            )
        time.sleep(poll_s)


def _wait_and_merge_pr(plan: Plan, pr: dict, phase: str, wait_s: int = 900, verify=None, log=print) -> None:
    _bring_pr_up_to_main(plan, pr, phase, log)
    _wait_required_checks(plan, pr, phase, wait_s=wait_s, log=log)
    if verify is not None:
        verify()
    state = _pr_state(plan, pr, phase)
    if state.get("state") != "OPEN" or state.get("mergeable") != "MERGEABLE":
        raise PinReleaseError(
            phase, f"pull request {pr['number']} is not mergeable: "
            f"state={state.get('state')} mergeable={state.get('mergeable')}",
        )
    merged = _run(plan.repo_root, "gh", "pr", "merge", str(pr["number"]), "--merge", check=False)
    if merged.returncode != 0:
        raise PinReleaseError(
            phase, f"could not merge {pr.get('url', '#'+str(pr['number']))}: "
            + (merged.stderr or merged.stdout).strip()[-400:],
        )


def _main_pr_title(plan: Plan) -> str:
    return f"build: llama.cpp pin {plan.llama_tag} release record and notes"


def phase_main(plan: Plan, wait_s: int = 900, log=print) -> bool:
    """Publish the work branch and merge it to main through the shared checked-PR path."""
    if _tag_exists(plan):
        return False
    root, remote = plan.repo_root, plan.remote
    _require_clean(plan, "main")
    _git(root, "fetch", "-q", remote)
    head = _git(root, "rev-parse", "HEAD")
    main_ref = f"{remote}/{plan.main}"
    if _run(root, "git", "merge-base", "--is-ancestor", head, main_ref, check=False).returncode == 0:
        return False
    _git(root, "push", "-q", remote, f"HEAD:refs/heads/{plan.branch}")
    pr = _ensure_pr(plan, plan.branch, _main_pr_title(plan), "main",
                    f"Release record and notes for llama.cpp {plan.llama_tag} ({plan.version}).")
    _wait_and_merge_pr(plan, pr, "main", wait_s, log=log)
    return True


def _release_pr(plan: Plan) -> dict | None:
    res = _run(plan.repo_root, "gh", "pr", "list", "--base", plan.main, "--state", "open", "--label", "autorelease: pending",
               "--json", "number,title,headRefName,url", check=False)
    if res.returncode != 0:
        raise PinReleaseError("release", "gh pr list failed (is the GitHub CLI installed and logged in?):\n" + (res.stderr or res.stdout).strip()[-400:])
    # the repository releases several components; another component's release PR can carry the same version number
    # (2026-10-12: bc-platform's was merged in place of bc-llamacpp's), so the PR is matched by its release branch
    package = _release_package(plan.repo_root, "release")
    branch = f"release-please--branches--{plan.main}--components--{package['component']}"
    prs = [p for p in json.loads(res.stdout or "[]")
           if p.get("headRefName") == branch and plan.version in p.get("title", "")]
    return prs[0] if prs else None


_RP_HEADER = ":robot: I have created a release *beep* *boop*"
_RP_FOOTER = ("This PR was generated with [Release Please](https://github.com/googleapis/release-please). "
              "See [documentation](https://github.com/googleapis/release-please#release-please).")


def release_pr_body(changelog: str, version: str) -> str:
    """The body release-please writes for its release PR, rebuilt from the changelog section it committed on the
    release branch. After the merge release-please parses this body to find the release; any other text gives
    'Pull request body did not match' and no release or tag is made."""
    start = changelog.find(f"## [{version}]")
    if start < 0:
        raise PinReleaseError("release", f"the release branch's changelog has no section for {version}")
    end = changelog.find("\n## [", start + 1)
    section = changelog[start:end if end >= 0 else len(changelog)].rstrip()
    return f"{_RP_HEADER}\n---\n\n\n{section}\n\n---\n{_RP_FOOTER}"


def _open_release_pr(plan: Plan) -> bool:
    """release-please pushes its release branch even when the repository does not let Actions open pull requests
    (the workflow run then fails). Open the PR from that branch, labelled as release-please expects. Returns whether a
    PR was opened."""
    root = plan.repo_root
    package = _release_package(root, "release")
    component = package["component"]
    branch = f"release-please--branches--{plan.main}--components--{component}"
    if not _git(root, "ls-remote", "--heads", plan.remote, f"refs/heads/{branch}", check=False):
        return False
    _git(root, "fetch", "-q", plan.remote, branch)
    subject = _git(root, "log", "-1", "--format=%s", "FETCH_HEAD")
    if plan.version not in subject:
        return False   # the branch is still the previous release's
    title = f"release: {component} {plan.version}"
    body = release_pr_body(_git(root, "show", f"FETCH_HEAD:{RELEASE_PACKAGE}/{package['changelog-path']}"),
                           plan.version)
    made = _run(root, "gh", "pr", "create", "--base", plan.main, "--head", branch, "--title", title, "--body", body,
                "--label", "autorelease: pending", check=False)
    if made.returncode != 0:
        raise PinReleaseError("release", f"could not open the release PR from {branch}: {(made.stderr or made.stdout).strip()[-400:]}")
    return True


def _verify_release_pr_files(plan: Plan, pr: dict) -> None:
    package = _release_package(plan.repo_root, "release")
    expected = {_MANIFEST, f"{RELEASE_PACKAGE}/{package['changelog-path']}"}
    viewed = _run(plan.repo_root, "gh", "pr", "view", str(pr["number"]), "--json", "files", check=False)
    if viewed.returncode != 0:
        raise PinReleaseError("release", f"could not inspect changed files for {pr.get('url', pr['number'])}: "
                                         + (viewed.stderr or viewed.stdout).strip()[-400:])
    actual = {row.get("path") for row in json.loads(viewed.stdout or "{}").get("files", [])}
    if actual != expected:
        raise PinReleaseError("release", f"release PR {pr['number']} changes unexpected files: "
                                         f"expected {sorted(expected)}, got {sorted(x for x in actual if x)}")


def phase_release(plan: Plan, wait_s: int = 900, poll_s: int = 20, log=print) -> bool:
    """release-please's PR merged and the release tag present. Returns False when the tag already exists."""
    if _tag_exists(plan):
        return False
    deadline = time.time() + wait_s
    pr = None
    while pr is None:
        pr = _release_pr(plan)
        if pr is None and _open_release_pr(plan):
            continue
        if pr is None:
            if time.time() > deadline:
                raise PinReleaseError("release", f"no release-please PR for {plan.version} on {plan.main} after {wait_s} s. Check the "
                                                 "release-please workflow run on main: neither a release PR nor a release branch for "
                                                 "this version exists")
            time.sleep(poll_s)
    _wait_and_merge_pr(
        plan, pr, "release", wait_s,
        verify=lambda: _verify_release_pr_files(plan, pr),
        log=log,
    )
    deadline = time.time() + wait_s
    while not _tag_exists(plan):
        if time.time() > deadline:
            raise PinReleaseError("release", f"{pr['url']} is merged but tag {plan.release_tag} has not appeared after {wait_s} s: "
                                             "check the release-please workflow run on main, then run again")
        time.sleep(poll_s)
    return True


def _unpin_branch(plan: Plan) -> str:
    return f"chore/release-as-clear-{plan.version}"


def _unpin_title(plan: Plan) -> str:
    return f"chore: clear llama.cpp release-as {plan.version}"


def _prepare_unpin_branch(plan: Plan, branch: str) -> None:
    root, remote = plan.repo_root, plan.remote
    remote_ref = f"refs/heads/{branch}"
    if _git(root, "ls-remote", "--heads", remote, remote_ref, check=False):
        _git(root, "fetch", "-q", remote, f"{remote_ref}:refs/remotes/{remote}/{branch}")
        text = _git(root, "show", f"{remote}/{branch}:{_RELEASE_CONFIG}")
        if _release_as_value(text, "unpin") is not None:
            raise PinReleaseError("unpin", f"existing {branch} does not clear {RELEASE_PACKAGE}.release-as")
        return

    parent = Path(tempfile.mkdtemp(prefix="pin-release-unpin-", dir=root.parent))
    worktree = parent / "tree"
    try:
        _git(root, "worktree", "add", "-q", "--detach", str(worktree), f"{remote}/{plan.main}")
        if not _edit_release_as(worktree, None, "unpin", expected=plan.version):
            return
        _git(worktree, "add", _RELEASE_CONFIG)
        _git(worktree, "commit", "-q", "-m", _unpin_title(plan))
        _git(worktree, "push", "-q", remote, f"HEAD:{remote_ref}")
    finally:
        _run(root, "git", "worktree", "remove", "--force", str(worktree), check=False)
        shutil.rmtree(parent, ignore_errors=True)


def phase_unpin(plan: Plan, wait_s: int = 900, log=print) -> bool:
    """Clear package release-as after the release tag, through a dedicated PR."""
    if not _tag_exists(plan):
        raise PinReleaseError("unpin", f"{plan.release_tag} does not exist yet")
    root = plan.repo_root
    _git(root, "fetch", "-q", plan.remote)
    main_text = _git(root, "show", f"{plan.remote}/{plan.main}:{_RELEASE_CONFIG}")
    current = _release_as_value(main_text, "unpin")
    if current is None:
        return False
    if current != plan.version:
        raise PinReleaseError("unpin", f"{RELEASE_PACKAGE}.release-as is {current!r}, expected {plan.version!r}")
    branch = _unpin_branch(plan)
    _prepare_unpin_branch(plan, branch)
    pr = _ensure_pr(plan, branch, _unpin_title(plan), "unpin",
                    f"Remove the temporary release-as pin after {plan.release_tag} was created.")
    _wait_and_merge_pr(plan, pr, "unpin", wait_s, log=log)
    return True


def phase_sync(plan: Plan) -> bool:
    """The work branch contains the release and unpin commits. Returns False when already current."""
    root, remote = plan.repo_root, plan.remote
    _git(root, "fetch", "-q", "--tags", remote)
    main_ref = f"{remote}/{plan.main}"
    if _run(root, "git", "merge-base", "--is-ancestor", main_ref, "HEAD", check=False).returncode == 0:
        return False
    _require_clean(plan, "sync")
    merge = _run(root, "git", "merge", "--no-edit", main_ref, check=False)
    if merge.returncode != 0:
        _run(root, "git", "merge", "--abort", check=False)
        raise PinReleaseError("sync", f"{main_ref} does not merge cleanly back into {plan.branch}; merge it by hand")
    _git(root, "push", "-q", remote, f"HEAD:refs/heads/{plan.branch}")
    return True


def _manifest_versions(plan: Plan, *, release_pending: bool = True) -> dict[str, str]:
    try:
        manifest = json.loads((plan.repo_root / _MANIFEST).read_text(encoding="utf-8"))
        config = json.loads((plan.repo_root / _RELEASE_CONFIG).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PinReleaseError("release", f"cannot read release-please manifest/config: {exc}") from exc
    result: dict[str, str] = {}
    for package, current in manifest.items():
        configured = config.get("packages", {}).get(package, {})
        pinned = configured.get("release-as") if isinstance(configured, dict) else None
        if release_pending and package == RELEASE_PACKAGE:
            result[package] = plan.version
        elif release_pending and isinstance(pinned, str):
            result[package] = pinned
        else:
            result[package] = str(current)
    return result


def _dry_run(plan: Plan, stop: int, log=print) -> None:
    tagged = _tag_exists(plan)
    record_path = plan.repo_root / "releases" / f"{plan.llama_tag}.json"
    recorded = record_path.is_file() and f"release {plan.version}" in (
        json.loads(record_path.read_text(encoding="utf-8")).get("notes") or "")
    config_value = _release_as_value((plan.repo_root / _RELEASE_CONFIG).read_text(encoding="utf-8"), "notes")
    notes_subject = f"chore: release notes {plan.release_tag}"
    notes_done = notes_subject in _git(plan.repo_root, "log", "--format=%s", "--fixed-strings",
                                       f"--grep={notes_subject}", "-5", check=False).splitlines()

    for name in PHASES[:stop + 1]:
        if name == "gate":
            log("  gate     already passed (recorded)" if recorded else
                "  gate     would run: python -m bigcherry pin-status --complete --all-remotes")
        elif name == "record":
            log("  record   already done" if recorded else
                f"  record   would update releases/{plan.llama_tag}.json and clear releases/pin-transition.json")
        elif name == "notes":
            if notes_done and (config_value == plan.version or (config_value is None and tagged)):
                log("  notes    already done")
            else:
                log(f"  notes    would write notes and set {RELEASE_PACKAGE}.release-as={plan.version}")
        elif name == "main":
            if tagged:
                log(f"  main     already done ({plan.release_tag} exists)")
            else:
                pr = _pr_for_head(plan, plan.branch, "main")
                if pr is None:
                    log(f"  main     would open PR base={plan.main} head={plan.branch} title={_main_pr_title(plan)!r}")
                else:
                    log(f"  main     would reuse PR #{pr['number']} title={pr.get('title')!r} head={pr.get('headRefName')}")
                log(f"           will merge {plan.main} if behind; conflicting updates use a temporary worktree")
                log(f"           will wait for required checks: {', '.join(_required_checks(plan, 'main'))}; trigger once if none starts")
        elif name == "release":
            if tagged:
                log(f"  release  already done: {plan.release_tag} exists")
            else:
                pr = _release_pr(plan)
                if pr is None:
                    log("  release  release PR: none exists yet")
                else:
                    log(f"  release  would merge PR #{pr['number']} title={pr.get('title')!r} branch={pr.get('headRefName')}")
                log(f"           will merge {plan.main} if behind; only {_MANIFEST} conflicts are auto-resolved")
                log(f"           will wait for required checks: {', '.join(_required_checks(plan, 'release'))}; trigger once if none starts")
            for package, version in _manifest_versions(plan, release_pending=not tagged).items():
                log(f"           manifest {package}: {version}")
        elif name == "unpin":
            if tagged and config_value is None:
                log("  unpin    already done")
            else:
                head = _unpin_branch(plan)
                pr = _pr_for_head(plan, head, "unpin")
                if pr is None:
                    log(f"  unpin    would open PR base={plan.main} head={head} title={_unpin_title(plan)!r}")
                else:
                    log(f"  unpin    would reuse PR #{pr['number']} title={pr.get('title')!r} head={pr.get('headRefName')}")
                log(f"           will merge {plan.main} if behind; conflicting updates use a temporary worktree")
                log(f"           will wait for required checks: {', '.join(_required_checks(plan, 'unpin'))}; trigger once if none starts")
        elif name == "sync":
            log(f"  sync     would merge {plan.remote}/{plan.main} into {plan.branch} and push only {plan.branch}")


def run(repo_root: Path, llama_tag: str, evidence: str, through: str = "sync", dry_run: bool = False, log=print,
        version: str | None = None, bump: str | None = None) -> int:
    plan = make_plan(repo_root, llama_tag, version=version, bump=bump, allow_existing=dry_run)
    stop = PHASES.index(through)
    log(f"pin-release: {plan.llama_tag} -> {plan.release_tag} (version {plan.version}), branch {plan.branch} -> {plan.main}")
    if dry_run:
        _dry_run(plan, stop, log)
        return 0
    record_path = repo_root / "releases" / f"{plan.llama_tag}.json"
    recorded = record_path.is_file() and f"release {plan.version}" in (
        json.loads(record_path.read_text(encoding="utf-8")).get("notes") or "")
    if recorded:
        gate_output = ""
        log("  gate     already passed (recorded)")
    else:
        gate_output = phase_gate(plan)
        log("  gate     PASS")
    steps = (
        ("record", lambda: phase_record(plan, gate_output, evidence)),
        ("notes", lambda: phase_notes(plan)),
        ("main", lambda: phase_main(plan, log=log)),
        ("release", lambda: phase_release(plan, log=log)),
        ("unpin", lambda: phase_unpin(plan, log=log)),
        ("sync", lambda: phase_sync(plan)),
    )
    for name, step in steps:
        if PHASES.index(name) > stop:
            break
        log(f"  {name:<8} {'done' if step() else 'already done'}")
    if stop >= PHASES.index("release"):
        log(f"pin-release: {plan.release_tag} released")
    return 0
