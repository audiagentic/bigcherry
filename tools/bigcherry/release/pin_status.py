"""Pin-consistency guard (RE48, revised per RV78).

Reads every pin surface of ONE tree and names its state; with remote trees
configured, also produces the cross-tree aggregate. This module READS ONLY.

Verdicts (local tree):
  consistent           vendor base revision == pinned revision
                       (always content-unverified: patch verification is
                       audit's job, dirty counts are reported, never judged)
  mid-rebase           a COMMITTED transition marker declares from->to,
                       pin == to, vendor == from, from's record not broken
  uncommitted-transition  the marker exists but is not committed: the
                       bump is declared in the working tree only
  drift                vendor != pin with no valid committed marker
                       (sub-reasons: stale-marker, marker-mismatch,
                       broken-base, no-transition)
  unavailable          no vendor checkout
  unresolvable-pin     the pinned ref does not resolve locally

Remote trees report only: consistent / mismatch / unreachable /
unresolvable-pin. The controller never infers a remote mid-rebase from its
own release records (RV78: unsound when the remote tree is behind).

Policies (exit codes are decided in __main__, not here):
  plain        diagnostic, never fails
  --strict     pipeline preflight for the gated tree
  --complete   bump completion: every required tree reachable, converged
               and consistent
"""

from __future__ import annotations

import json
import re
import shlex
import subprocess
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from .. import pin_transition
from ..pin_transition import MarkerError, PinTransition

VERDICT_CONSISTENT = "consistent"
VERDICT_MID_REBASE = "mid-rebase"
VERDICT_UNCOMMITTED = "uncommitted-transition"
VERDICT_DRIFT = "drift"
VERDICT_UNAVAILABLE = "unavailable"
VERDICT_UNRESOLVABLE_PIN = "unresolvable-pin"

REMOTE_CONSISTENT = "consistent"
REMOTE_MISMATCH = "mismatch"
REMOTE_UNREACHABLE = "unreachable"

_PIN_LINE = re.compile(r'^pinned\s*=\s*"([^"]+)"', re.MULTILINE)


class PinStatusError(ValueError):
    pass


def _git(root: Path, *args: str, timeout: int = 30) -> str:
    result = subprocess.run(
        ["git", "-C", str(root), *args],
        check=False,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    if result.returncode != 0:
        raise PinStatusError(f"git {' '.join(args)} in {root}: {result.stderr.strip()}")
    return result.stdout.strip()


def _git_ok(root: Path, *args: str, timeout: int = 30) -> str | None:
    try:
        return _git(root, *args, timeout=timeout)
    except PinStatusError:
        return None


@dataclass(frozen=True)
class RepoPaths:
    """Injectable locations so tests run against fake trees."""

    repo_root: Path
    llama_root: Path
    releases_dir: Path
    artifacts_dir: Path


@dataclass
class LocalStatus:
    pinned_ref: str | None
    pinned_sha: str | None
    vendor_head: str | None
    vendor_tags: tuple[str, ...]
    vendor_modified: int
    vendor_untracked: int
    marker: PinTransition | None
    marker_state: str
    records: dict[str, str]
    descriptors: tuple[str, ...]
    bigcherry_head: str
    verdict: str
    reasons: list[str] = field(default_factory=list)


@dataclass
class RemoteStatus:
    name: str
    reachable: bool
    vendor_head: str | None
    pinned_ref: str | None
    pinned_sha: str | None
    bigcherry_head: str | None
    verdict: str
    reasons: list[str] = field(default_factory=list)


@dataclass
class PinStatusReport:
    local: LocalStatus
    remotes: list[RemoteStatus]
    converged: bool | None
    aggregate_reasons: list[str] = field(default_factory=list)


def _read_pinned_ref(repo_root: Path) -> str | None:
    try:
        text = (repo_root / "config" / "recipes.toml").read_text(encoding="utf-8")
    except (OSError, FileNotFoundError):
        return None
    match = _PIN_LINE.search(text)
    return match.group(1) if match else None


def _resolve_sha(llama_root: Path, ref: str) -> str | None:
    """Resolve a ref to a commit SHA using LOCAL refs only (no network)."""
    return _git_ok(llama_root, "rev-parse", "--verify", "--quiet", f"{ref}^{{commit}}")


def _vendor_info(llama_root: Path) -> tuple[str, tuple[str, ...], int, int] | None:
    """HEAD, tags-at-HEAD, modified count, untracked count. None = absent."""
    if not (llama_root / ".git").exists():
        return None
    head = _git_ok(llama_root, "rev-parse", "HEAD")
    if head is None:
        return None
    tags = _git_ok(llama_root, "tag", "--points-at", "HEAD") or ""
    tag_list = tuple(sorted(t for t in tags.splitlines() if t.strip()))
    modified = untracked = 0
    status = _git_ok(llama_root, "status", "--porcelain") or ""
    for line in status.splitlines():
        if len(line) < 2:
            continue
        if line[:2] == "??":
            untracked += 1
        elif line[1] in ("M", "D", "A", "R", "T", "C"):
            modified += 1
    return head, tag_list, modified, untracked


def _release_records(releases_dir: Path) -> dict[str, str]:
    """revision -> stage, read straight from the per-release JSON files
    (deliberately not via releases.all_records(): this module must run
    against fake trees without importing the production release store)."""
    records: dict[str, str] = {}
    if not releases_dir.is_dir():
        return records
    for path in sorted(releases_dir.glob("*.json")):
        if path.name == "index.json":
            continue
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(raw, dict):
            continue
        revision = raw.get("revision")
        stage = raw.get("stage")
        if isinstance(revision, str) and isinstance(stage, str):
            records[revision] = stage
    return records


def _descriptors(artifacts_dir: Path) -> tuple[str, ...]:
    if not artifacts_dir.is_dir():
        return ()
    revs = [
        p.name
        for p in artifacts_dir.iterdir()
        if p.is_dir() and re.fullmatch(r"[0-9a-f]{8,40}", p.name)
    ]
    return tuple(sorted(revs))


def _is_ancestor(repo_root: Path, commit: str, head: str) -> bool:
    try:
        subprocess.run(
            ["git", "-C", str(repo_root), "merge-base", "--is-ancestor", commit, head],
            check=True,
            capture_output=True,
            timeout=30,
        )
        return True
    except (subprocess.CalledProcessError, subprocess.SubprocessError):
        return False


def local_status(paths: RepoPaths) -> LocalStatus:
    """Pure reader: one LocalStatus for the tree at `paths`."""
    pinned_ref = _read_pinned_ref(paths.repo_root)
    reasons: list[str] = []

    vendor = _vendor_info(paths.llama_root)
    if vendor is None:
        return LocalStatus(
            pinned_ref=pinned_ref,
            pinned_sha=None,
            vendor_head=None,
            vendor_tags=(),
            vendor_modified=0,
            vendor_untracked=0,
            marker=None,
            marker_state="absent",
            records={},
            descriptors=(),
            bigcherry_head=_git_ok(paths.repo_root, "rev-parse", "HEAD") or "",
            verdict=VERDICT_UNAVAILABLE,
            reasons=["vendor checkout missing or not a git repo"],
        )
    head, tags, modified, untracked = vendor

    pinned_sha: str | None = None
    if pinned_ref is not None:
        pinned_sha = _resolve_sha(paths.llama_root, pinned_ref)
    if pinned_sha is None:
        reason = (
            (f"pinned ref {pinned_ref!r} does not resolve in the local upstream clone")
            if pinned_ref is not None
            else "no pinned ref in config/recipes.toml"
        )
        return LocalStatus(
            pinned_ref=pinned_ref,
            pinned_sha=None,
            vendor_head=head,
            vendor_tags=tags,
            vendor_modified=modified,
            vendor_untracked=untracked,
            marker=None,
            marker_state="absent",
            records={},
            descriptors=(),
            bigcherry_head=_git_ok(paths.repo_root, "rev-parse", "HEAD") or "",
            verdict=VERDICT_UNRESOLVABLE_PIN,
            reasons=[reason],
        )

    records = _release_records(paths.releases_dir)
    bigcherry_head = _git_ok(paths.repo_root, "rev-parse", "HEAD") or ""

    try:
        marker = pin_transition.load(paths.releases_dir / "pin-transition.json")
    except MarkerError as exc:
        marker = None
        reasons.append(f"corrupt transition marker: {exc}")

    marker_state = "absent"
    if (paths.releases_dir / "pin-transition.json").is_file():
        marker_state = pin_transition.committed_state(
            paths.releases_dir / "pin-transition.json"
        )

    base = {
        "pinned_ref": pinned_ref,
        "pinned_sha": pinned_sha,
        "vendor_head": head,
        "vendor_tags": tags,
        "vendor_modified": modified,
        "vendor_untracked": untracked,
        "marker": marker,
        "marker_state": marker_state,
        "records": records,
        "descriptors": _descriptors(paths.artifacts_dir),
        "bigcherry_head": bigcherry_head,
    }

    if head == pinned_sha:
        if modified:
            reasons.append(
                f"content-unverified: {modified} tracked file(s) "
                "modified (expected after patch application; "
                "verified by audit, not here)"
            )
        return LocalStatus(verdict=VERDICT_CONSISTENT, reasons=reasons, **base)  # type: ignore[arg-type]

    # vendor != pin: the marker is the ONLY thing that can name this
    # mid-rebase. No marker / bad marker => drift, with the sub-reason.
    if marker is None:
        record_note = records.get(head)
        if record_note is not None:
            reasons.append(
                "a release record exists for the vendor revision "
                f"({record_note}), but records are evidence, not transitions: "
                "without a committed marker this is drift (the S1-incident "
                "class)"
            )
        else:
            reasons.append("vendor is at a revision the pin never declared")
        return LocalStatus(verdict=VERDICT_DRIFT, reasons=reasons, **base)  # type: ignore[arg-type]

    if marker.to_sha != pinned_sha:
        reasons.append(
            f"stale marker: marker declares -> {marker.tag} "
            f"({marker.to_sha[:12]}) but the pin is {pinned_ref} "
            f"({pinned_sha[:12]})"
        )
        return LocalStatus(verdict=VERDICT_DRIFT, reasons=reasons, **base)  # type: ignore[arg-type]

    if marker.from_sha != head:
        reasons.append(
            f"marker mismatch: declared from {marker.from_sha[:12]} but "
            f"vendor is at {head[:12]}"
        )
        return LocalStatus(verdict=VERDICT_DRIFT, reasons=reasons, **base)  # type: ignore[arg-type]

    if marker_state != "committed-clean":
        reasons.append(
            "transition marker is not committed: the bump is declared in "
            "the working tree only -- commit recipes.toml + the marker "
            "together (PIN_BUMP.md step 1)"
        )
        return LocalStatus(verdict=VERDICT_UNCOMMITTED, reasons=reasons, **base)  # type: ignore[arg-type]

    if bigcherry_head and not _is_ancestor(
        paths.repo_root, marker.declaring_commit, bigcherry_head
    ):
        reasons.append(
            "marker's declaring commit is not an ancestor of HEAD "
            "(history was rewritten or reset after the repin)"
        )
        return LocalStatus(verdict=VERDICT_DRIFT, reasons=reasons, **base)  # type: ignore[arg-type]

    from_stage = records.get(marker.from_sha)
    if from_stage == "broken":
        reasons.append(
            f"declared base {marker.from_sha[:12]} is a broken release "
            "record -- resuming from a broken base is drift"
        )
        return LocalStatus(verdict=VERDICT_DRIFT, reasons=reasons, **base)  # type: ignore[arg-type]

    reasons.append(
        f"declared bump {marker.from_sha[:12]} -> {marker.to_sha[:12]} "
        f"({marker.tag}) in flight; base record: "
        f"{from_stage or 'no record'}"
    )
    return LocalStatus(verdict=VERDICT_MID_REBASE, reasons=reasons, **base)  # type: ignore[arg-type]


# --- remotes -----------------------------------------------------------------

#: probe(alias, path) -> (vendor_head, pinned_ref, bigcherry_head)
#: raise/None parts mean that value is missing; raise PinStatusError means
#: unreachable.
Probe = Callable[[str, str], tuple[str | None, str | None, str | None]]


def _default_probe(alias: str, path: str) -> tuple[str | None, str | None, str | None]:
    remote = (
        f"cd {shlex.quote(path)} && "
        f"printf 'VENDOR %s\\n' \"$(git -C vendor/llama.cpp rev-parse HEAD "
        f'2>/dev/null || printf none)" && '
        f"printf 'PIN %s\\n' \"$(sed -n "
        f"'s/^pinned[[:space:]]*=[[:space:]]*\\\"\\(.*\\)\\\"$/\\1/p' "
        f'config/recipes.toml 2>/dev/null | sed -n 1p)" && '
        f"printf 'HEAD %s\\n' \"$(git rev-parse HEAD 2>/dev/null "
        f'|| printf none)"'
    )
    cmd = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", alias, remote]
    result = subprocess.run(
        cmd, check=False, capture_output=True, text=True, timeout=20
    )
    if result.returncode != 0:
        raise PinStatusError(f"ssh {alias}: {result.stderr.strip() or 'unreachable'}")
    values: dict[str, str | None] = {}
    for line in result.stdout.splitlines():
        parts = line.split(" ", 1)
        if len(parts) == 2 and parts[0] in ("VENDOR", "PIN", "HEAD"):
            values[parts[0]] = None if parts[1] == "none" else parts[1]
    return (values.get("VENDOR"), values.get("PIN"), values.get("HEAD"))


def remote_status(
    name: str,
    alias: str,
    remote_path: str,
    local: LocalStatus,
    llama_root: Path,
    probe: Probe = _default_probe,
) -> RemoteStatus:
    """Probe one remote tree. Verdicts: consistent / mismatch /
    unreachable / unresolvable-pin -- and nothing else (RV78)."""
    try:
        vendor_head, pinned_ref, bigcherry_head = probe(alias, remote_path)
    except (PinStatusError, subprocess.SubprocessError) as exc:
        return RemoteStatus(
            name, False, None, None, None, None, REMOTE_UNREACHABLE, [str(exc)]
        )

    reasons: list[str] = []
    pinned_sha = None
    verdict = REMOTE_MISMATCH
    if vendor_head is None:
        reasons.append("remote vendor checkout missing or not a git repo")
    elif pinned_ref is None:
        reasons.append("remote config/recipes.toml has no pinned ref")
    else:
        pinned_sha = _resolve_sha(llama_root, pinned_ref)
        if pinned_sha is None:
            return RemoteStatus(
                name,
                True,
                vendor_head,
                pinned_ref,
                None,
                bigcherry_head,
                VERDICT_UNRESOLVABLE_PIN,
                [
                    f"remote pin {pinned_ref!r} does not resolve in the local "
                    "upstream clone"
                ],
            )
        if vendor_head == pinned_sha:
            verdict = REMOTE_CONSISTENT
            if local.pinned_sha and pinned_sha != local.pinned_sha:
                reasons.append(
                    f"remote pin {pinned_ref} != local pin "
                    f"{local.pinned_ref} (aggregate will flag divergence)"
                )
        else:
            reasons.append(
                f"remote vendor {vendor_head[:12]} != remote pin "
                f"{pinned_ref} ({pinned_sha[:12]})"
            )

    return RemoteStatus(
        name,
        True,
        vendor_head,
        pinned_ref,
        pinned_sha,
        bigcherry_head,
        verdict,
        reasons,
    )


def build_report(
    paths: RepoPaths,
    trees: tuple,
    llama_root: Path | None = None,
    probe: Probe = _default_probe,
    local_tooling_expected: dict[str, str] | None = None,
) -> PinStatusReport:
    """Full report: local status + remotes + cross-tree aggregate.

    `trees` are config.Tree entries. `local_tooling_expected` maps tree
    name -> expected bigcherry HEAD for role=campaign trees (empty string
    = local HEAD).
    """
    local = local_status(paths)
    llama = llama_root if llama_root is not None else paths.llama_root
    expected = local_tooling_expected or {}

    remotes: list[RemoteStatus] = []
    for tree in trees:
        if tree.name == "local":
            continue
        status = remote_status(
            tree.name, tree.alias, tree.path, local, llama, probe=probe
        )
        if tree.role == "campaign" and status.reachable:
            want = expected.get(tree.name, tree.expected_tooling_revision)
            want = want or local.bigcherry_head
            remote_head = status.bigcherry_head
            if want and (remote_head is None or remote_head != want):
                status.verdict = REMOTE_MISMATCH
                status.reasons.append(
                    f"campaign tree tooling "
                    f"{(remote_head or 'unknown')[:12]} != expected "
                    f"{want[:12]}"
                )
        remotes.append(status)

    converged = True
    aggregate_reasons: list[str] = []
    for status in remotes:
        if not status.reachable:
            continue
        if (
            status.pinned_sha is not None
            and local.pinned_sha is not None
            and status.pinned_sha != local.pinned_sha
        ):
            converged = False
            aggregate_reasons.append(
                f"tree {status.name} pinned at {status.pinned_ref}, local "
                f"at {local.pinned_ref}"
            )

    return PinStatusReport(
        local=local,
        remotes=remotes,
        converged=converged,
        aggregate_reasons=aggregate_reasons,
    )


# --- policy tiers (used by __main__ for exit codes) ---------------------------


def campaign_preflight(
    clone_source: Path, repo_paths: RepoPaths
) -> tuple[list[str], bool]:
    """(failures, mid_rebase_warning) for the campaign build path.

    The campaign engine clones from `clone_source` per revision, so the
    guard attaches to the checkout that lives there (the legacy
    vendor/llama.cpp working tree in this project -- the S1 surface). A
    campaign-only tree with no git checkout at the clone source has
    nothing to guard: the lanes bind their own source identity.

      no git repo at clone_source      -> (no failures, no warning)
      consistent                       -> (no failures, no warning)
      mid-rebase (committed marker)    -> (no failures, WARNING: the
                                           revision-bound identity tolerates
                                           a declared bump in flight)
      drift / uncommitted /            -> (failures) fail closed
      unresolvable-pin                 ->
    """
    if not (clone_source / ".git").exists():
        return [], False
    status = local_status(
        RepoPaths(
            repo_root=repo_paths.repo_root,
            llama_root=clone_source,
            releases_dir=repo_paths.releases_dir,
            artifacts_dir=repo_paths.artifacts_dir,
        )
    )
    if status.verdict == VERDICT_UNAVAILABLE:
        return [], False
    if status.verdict == VERDICT_MID_REBASE:
        return [], True
    return strict_failure(
        PinStatusReport(local=status, remotes=[], converged=None)
    ), False


def strict_failure(report: PinStatusReport) -> list[str]:
    """--strict: pipeline preflight for the gated tree."""
    failures: list[str] = []
    verdict = report.local.verdict
    if verdict in (
        VERDICT_DRIFT,
        VERDICT_UNAVAILABLE,
        VERDICT_UNRESOLVABLE_PIN,
        VERDICT_UNCOMMITTED,
    ):
        failures.append(verdict + ": " + "; ".join(report.local.reasons))
    return failures


def complete_failures(report: PinStatusReport, trees: tuple) -> list[str]:
    """--complete: every required tree reachable, converged, consistent."""
    failures: list[str] = []
    verdict = report.local.verdict
    if verdict != VERDICT_CONSISTENT:
        failures.append(f"local tree is {verdict}: " + "; ".join(report.local.reasons))
    for tree in trees:
        if not tree.required or tree.name == "local":
            continue
        status = next((r for r in report.remotes if r.name == tree.name), None)
        if status is None:
            failures.append(f"required tree {tree.name} was not probed")
            continue
        if status.verdict == REMOTE_UNREACHABLE:
            failures.append(f"required tree {tree.name} unreachable")
        elif status.verdict != REMOTE_CONSISTENT:
            failures.append(
                f"required tree {tree.name} is {status.verdict}: "
                + "; ".join(status.reasons)
            )
    if report.converged == False:  # noqa: E712 -- tri-state: None = no remotes
        failures.append("trees diverged: " + "; ".join(report.aggregate_reasons))
    return failures
