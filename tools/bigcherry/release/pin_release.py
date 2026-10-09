"""Release a completed pin bump: completion gate -> record -> notes -> main -> release-please PR -> tag.

`bigcherry pin-bump` ends when one tree is reconciled at the new pin. This is the rest of the procedure, as one
resumable command, so a bump is released the same way every time:

  gate     `pin-status --complete --all-remotes` must pass (every required tree converged at the pin)
  record   the gate output and the hardware evidence go into the release record, the transition marker is removed
  notes    the release notes are generated and committed with the `Release-As: <llama build>.0.0` footer
  main     the work branch takes origin/main in (if main moved) and main is fast-forwarded to it
  release  release-please's PR is merged; the workflow tags bc-llamacpp-<version> and publishes the notes
  sync     the work branch takes the release commit back

Each phase first checks whether it is already done, so the command can be run again after any stop. Nothing here
decides whether the bump is GOOD: the caller states the hardware evidence (--evidence), which is recorded verbatim.

Versions and tags. release-please owns the tags: bc-llamacpp-<llama build>.<minor>.<patch> (config/release.toml tag-prefix).
The first release of a pin is <build>.0.0; the workflow also adds the readable tag bc-llamacpp-b<build>-r0 to it. While the pin stays, further
BigCherry releases raise minor (features) or patch (fixes): `pin-release <tag> --bump minor|patch` picks the next
version after the newest bc-<build>.* tag, `--version` states it. Every release goes through the same phases.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from . import notes as _notes

PHASES = ("gate", "record", "notes", "main", "release", "sync")


# The release-please package (a path in release-please-config.json) that carries the llama.cpp engine's release line.
RELEASE_PACKAGE = "engines/llamacpp"


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
              version: str | None = None, bump: str | None = None) -> Plan:
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
        # a release whose notes commit exists but is not tagged yet is the one in flight: resume it
        in_flight = _git(repo_root, "log", "--format=%s", "--fixed-strings", f"--grep=chore: release notes {config.tag_prefix}{build}.", "-20", check=False)
        pending = [s.rsplit(config.tag_prefix, 1)[-1] for s in in_flight.splitlines() if s.rsplit(config.tag_prefix, 1)[-1] not in existing]
        version = pending[0] if pending else next_version(existing, build, bump)
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
    """Release notes committed with the Release-As footer. Returns False when already done."""
    root = plan.repo_root
    subject = f"chore: release notes {plan.release_tag}"
    if subject in _git(root, "log", "--format=%s", "--fixed-strings", f"--grep={subject}", "-5").splitlines():
        return False
    _require_clean(plan, "notes")
    try:
        out = _notes.write_notes(root, _notes.load_config(root), plan.llama_tag, "HEAD", plan.version)
    except _notes.ReleaseNotesError as exc:
        raise PinReleaseError("notes", str(exc)) from exc
    _git(root, "add", str(Path(out).resolve().relative_to(root.resolve())).replace("\\", "/"))
    body = ("release-please reads only conventional-commit subjects; this commit carries the release version "
            f"(BigCherry on llama.cpp {plan.llama_tag}).\n\nRelease-As: {plan.version}")
    _git(root, "commit", "-q", "-m", subject, "-m", body)
    return True


def phase_main(plan: Plan) -> bool:
    """main fast-forwarded to the work branch (after taking main in, if it moved). Returns False when already done."""
    root, remote = plan.repo_root, plan.remote
    _require_clean(plan, "main")
    _git(root, "fetch", "-q", remote)
    head = _git(root, "rev-parse", "HEAD")
    main_ref = f"{remote}/{plan.main}"
    if _run(root, "git", "merge-base", "--is-ancestor", head, main_ref, check=False).returncode == 0:
        return False
    if _run(root, "git", "merge-base", "--is-ancestor", main_ref, head, check=False).returncode != 0:
        merge = _run(root, "git", "merge", "--no-edit", "-m", f"merge {plan.main} into {plan.branch} for the {plan.release_tag} release", main_ref, check=False)
        if merge.returncode != 0:
            _run(root, "git", "merge", "--abort", check=False)
            raise PinReleaseError("main", f"{main_ref} does not merge cleanly into {plan.branch}:\n{(merge.stdout + merge.stderr).strip()[-800:]}\n"
                                          "merge it by hand, commit, and run again")
    _git(root, "push", "-q", remote, f"HEAD:refs/heads/{plan.branch}")
    _git(root, "push", "-q", remote, f"HEAD:refs/heads/{plan.main}")   # a fast-forward by construction; git refuses anything else
    return True


def _release_pr(plan: Plan) -> dict | None:
    res = _run(plan.repo_root, "gh", "pr", "list", "--base", plan.main, "--state", "open", "--label", "autorelease: pending",
               "--json", "number,title,headRefName,url", check=False)
    if res.returncode != 0:
        raise PinReleaseError("release", "gh pr list failed (is the GitHub CLI installed and logged in?):\n" + (res.stderr or res.stdout).strip()[-400:])
    prs = [p for p in json.loads(res.stdout or "[]") if plan.version in p.get("title", "")]
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
    package = json.loads((root / "release-please-config.json").read_text(encoding="utf-8"))["packages"][RELEASE_PACKAGE]
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


def phase_release(plan: Plan, wait_s: int = 900, poll_s: int = 20) -> bool:
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
    merged = _run(plan.repo_root, "gh", "pr", "merge", str(pr["number"]), "--merge", check=False)
    if merged.returncode != 0:
        raise PinReleaseError("release", f"could not merge {pr['url']}:\n{(merged.stderr or merged.stdout).strip()[-400:]}")
    deadline = time.time() + wait_s
    while not _tag_exists(plan):
        if time.time() > deadline:
            raise PinReleaseError("release", f"{pr['url']} is merged but tag {plan.release_tag} has not appeared after {wait_s} s: "
                                             "check the release-please workflow run on main, then run again")
        time.sleep(poll_s)
    return True


def phase_sync(plan: Plan) -> bool:
    """The work branch contains the release commit. Returns False when it already does."""
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


def run(repo_root: Path, llama_tag: str, evidence: str, through: str = "sync", dry_run: bool = False, log=print,
        version: str | None = None, bump: str | None = None) -> int:
    plan = make_plan(repo_root, llama_tag, version=version, bump=bump)
    stop = PHASES.index(through)
    log(f"pin-release: {plan.llama_tag} -> {plan.release_tag} (version {plan.version}), branch {plan.branch} -> {plan.main}")
    if dry_run:
        for name in PHASES[:stop + 1]:
            log(f"  would run: {name}")
        return 0
    record_path = repo_root / "releases" / f"{plan.llama_tag}.json"
    recorded = record_path.is_file() and f"release {plan.version}" in (json.loads(record_path.read_text(encoding="utf-8")).get("notes") or "")
    if recorded:
        # the gate passed and its output is in the record; the commits made since (record, notes, merges) move this
        # tree's revision ahead of the others by design, so the gate is not asked again for the same release
        gate_output = ""
        log("  gate     already passed (recorded)")
    else:
        gate_output = phase_gate(plan)
        log("  gate     PASS")
    steps = (
        ("record", lambda: phase_record(plan, gate_output, evidence)),
        ("notes", lambda: phase_notes(plan)),
        ("main", lambda: phase_main(plan)),
        ("release", lambda: phase_release(plan)),
        ("sync", lambda: phase_sync(plan)),
    )
    for name, step in steps:
        if PHASES.index(name) > stop:
            break
        log(f"  {name:<8} {'done' if step() else 'already done'}")
    if stop >= PHASES.index("release"):
        log(f"pin-release: {plan.release_tag} released")
    return 0
