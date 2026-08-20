"""External-source registry: where backported patches came from, and how to
keep watching those sources.

The registry lives at ``<repo>/external-sources.toml``. Every patch that
backports code from an external repository, fork branch, or upstream PR
declares a machine-readable ``PROVENANCE`` dict (see the patches in group
``rdna-boosts``); the cross-check below fails if the patch and the registry
disagree, so a future agent can always answer *where did this come from and
is it still up to date* without re-deriving anything.

Commands:
    python -m bigcherry sources status    offline: print the registry and
                                          validate patch<->registry links
    python -m bigcherry sources check     online: read-only fetch of each
                                          source into a temp dir (never the
                                          vendor tree) and report, per
                                          source: branch moved / rebased,
                                          tracked changes that merged into
                                          mainline (patch-id), and content
                                          drift of tracked changes.
"""

from __future__ import annotations

import argparse
import ast
import re
import subprocess
import sys
import tempfile
import tomllib
from pathlib import Path

from . import paths

REGISTRY_NAME = "external-sources.toml"

# Valid statuses for a tracked logical change.
TRACKED_STATUSES = (
    "ported-untested",
    # Benched in isolation on one arch (control/test pair) but not yet
    # cross-arch repeated or reviewed for promotion. Between untested and
    # validated; added 2026-08-20 for the first RD isolated bench (RD04).
    "ported-benched",
    "ported-validated",
    "planned",
    "superseded",
    "deferred-hardware",
    "excluded",
    "evidence-only",
)

_SHA_RE = re.compile(r"^[0-9a-f]{40}$")


def registry_path(override: str | Path | None = None) -> Path:
    return Path(override) if override is not None else paths.EXTERNAL_SOURCES


def load_registry(path: str | Path | None = None) -> dict:
    """Parse and structurally validate the registry."""
    reg_path = registry_path(path)
    if not reg_path.is_file():
        raise FileNotFoundError(f"external source registry not found: {reg_path}")
    raw = tomllib.loads(reg_path.read_text(encoding="utf-8"))
    if raw.get("version") != 1:
        raise ValueError(f"{REGISTRY_NAME}: unsupported version {raw.get('version')!r}")
    sources = raw.get("sources") or []
    if not sources:
        raise ValueError(f"{REGISTRY_NAME}: no [[sources]] entries")
    for source in sources:
        for field in ("id", "repo", "locator"):
            if not source.get(field):
                raise ValueError(f"[[sources]] missing {field!r}")
        snapshots = source.get("snapshots") or []
        if not snapshots:
            raise ValueError(f"source {source['id']}: no [[sources.snapshots]]")
        active = [s for s in snapshots if s.get("active")]
        if len(active) != 1:
            raise ValueError(
                f"source {source['id']}: exactly one snapshot must have "
                f"active=true (found {len(active)})"
            )
        for snap in snapshots:
            for field in ("head", "base"):
                if not _SHA_RE.match(str(snap.get(field, ""))):
                    raise ValueError(f"source {source['id']}: snapshot {snap.get('label')} {field} is not a 40-hex SHA")
        seen = set()
        for entry in source.get("tracked") or []:
            sha = str(entry.get("commit", ""))
            if not _SHA_RE.match(sha):
                raise ValueError(f"source {source['id']}: tracked commit {sha!r} is not 40-hex")
            if sha in seen:
                raise ValueError(f"source {source['id']}: duplicate tracked commit {sha}")
            seen.add(sha)
            status = entry.get("status", "")
            if status not in TRACKED_STATUSES:
                raise ValueError(f"source {source['id']}: {sha[:9]} has invalid status {status!r}")
            if entry.get("original") and not _SHA_RE.match(str(entry["original"])):
                raise ValueError(f"source {source['id']}: {sha[:9]} original is not a 40-hex SHA")
    return raw


# ---------------------------------------------------------------- cross-check


def _patch_provenance(patch_path: Path) -> dict | None:
    """Read a patch module's PROVENANCE literal without executing it."""
    try:
        tree = ast.parse(patch_path.read_text(encoding="utf-8"), filename=str(patch_path))
    except (OSError, SyntaxError):
        return None
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == "PROVENANCE" for t in node.targets
        ):
            try:
                value = ast.literal_eval(node.value)
            except (ValueError, TypeError):
                return None
            return value if isinstance(value, dict) else None
    return None


def cross_check_patches(registry: dict | None = None,
                        patches_dir: str | Path | None = None) -> list[str]:
    """Validate the two-way link between patch PROVENANCE dicts and the
    registry. Returns a list of problems (empty = consistent)."""
    registry = registry or load_registry()
    patches_dir = Path(patches_dir) if patches_dir is not None else paths.PATCHES

    by_source: dict[str, dict] = {s["id"]: s for s in registry.get("sources", [])}
    tracked: dict[tuple[str, str], dict] = {}
    for source in registry.get("sources", []):
        for entry in source.get("tracked", []):
            tracked[(source["id"], entry["commit"])] = entry

    problems: list[str] = []
    patch_files = sorted(p for p in patches_dir.glob("*.py") if not p.name.startswith("_"))
    for pfile in patch_files:
        prov = _patch_provenance(pfile)
        if prov is None:
            continue  # not an external backport (no PROVENANCE)
        sid = prov.get("source-id", "")
        if sid not in by_source:
            problems.append(f"{pfile.name}: PROVENANCE source-id {sid!r} not in registry")
            continue
        entry = tracked.get((sid, prov.get("fork-commit", "")))
        if entry is None:
            problems.append(
                f"{pfile.name}: fork-commit {prov.get('fork-commit', '')[:9]} "
                f"not tracked under source {sid}"
            )
            continue
        if entry.get("patch") != pfile.stem:
            problems.append(
                f"{pfile.name}: registry entry for {prov.get('fork-commit', '')[:9]} "
                f"points at patch {entry.get('patch')!r}, not this file"
            )
        if entry.get("plan-item") and prov.get("plan-item") != entry.get("plan-item"):
            problems.append(
                f"{pfile.name}: PROVENANCE plan-item {prov.get('plan-item')!r} "
                f"disagrees with registry {entry.get('plan-item')!r}"
            )

    # Registry side: every tracked entry that names a patch must match the file.
    for (sid, sha), entry in tracked.items():
        patch_id = entry.get("patch")
        if patch_id is None:
            continue
        pfile = patches_dir / f"{patch_id}.py"
        if not pfile.is_file():
            problems.append(f"registry {sid}/{sha[:9]}: patch {patch_id} does not exist")
            continue
        prov = _patch_provenance(pfile)
        if prov is None:
            problems.append(f"registry {sid}/{sha[:9]}: {patch_id} has no PROVENANCE dict")
        elif prov.get("fork-commit") != sha:
            problems.append(
                f"registry {sid}/{sha[:9]}: {patch_id} PROVENANCE fork-commit "
                f"disagrees ({prov.get('fork-commit', '')[:9]})"
            )
    return problems


# ---------------------------------------------------------------- status view


def cmd_sources(args: argparse.Namespace) -> int:
    if not getattr(args, "src_subcommand", None):
        print("usage: bigcherry sources {status,check}", file=sys.stderr)
        return 2
    if args.src_subcommand == "status":
        return _status()
    return _check(args)


def _status() -> int:
    try:
        registry = load_registry()
    except (FileNotFoundError, ValueError, tomllib.TOMLDecodeError) as exc:
        print(f"registry error: {exc}", file=sys.stderr)
        return 1

    problems = cross_check_patches(registry)
    for source in registry["sources"]:
        print(f"source: {source['id']}")
        print(f"  repo:     {source['repo']}")
        print(f"  locator:  {source['locator']} (locator only, not identity)")
        for snap in source["snapshots"]:
            marker = "ACTIVE" if snap.get("active") else "      "
            print(f"  snapshot {snap.get('label', '?'):>4}  {marker}  "
                  f"head {snap['head'][:9]}  base {snap['base'][:9]}  "
                  f"reviewed {snap.get('reviewed-at', '?')}")
        print("  tracked changes:")
        for entry in source.get("tracked", []):
            patch = f"  -> {entry['patch']}" if entry.get("patch") else ""
            orig = f" (orig {entry['original'][:9]})" if entry.get("original") else ""
            print(f"    {entry['commit'][:9]}  {entry['status']:<18} "
                  f"{entry.get('plan-item', '-'):>5}  {entry['title'][:60]}{orig}{patch}")
        if source.get("notes"):
            first = source["notes"].strip().splitlines()[0]
            print(f"  note: {first}")
        print()
    if problems:
        print("patch<->registry cross-check PROBLEMS:", file=sys.stderr)
        for p in problems:
            print(f"  {p}", file=sys.stderr)
        return 1
    print("patch<->registry cross-check: OK")
    return 0


# ---------------------------------------------------------------- online check


def _git(cwd: str, *argv: str, timeout: int) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", "-C", cwd, *argv],
        capture_output=True, text=True, timeout=timeout, check=False,
    )


def _check(args: argparse.Namespace) -> int:
    try:
        registry = load_registry()
    except (FileNotFoundError, ValueError, tomllib.TOMLDecodeError) as exc:
        print(f"registry error: {exc}", file=sys.stderr)
        return 1

    timeout = args.timeout
    findings = 0
    for source in registry["sources"]:
        sid = source["id"]
        print(f"== {sid} ==  {source['repo']}  locator={source['locator']}")
        active = next(s for s in source["snapshots"] if s.get("active"))
        with tempfile.TemporaryDirectory(prefix="bc-src-") as tmp:
            clone = subprocess.run(
                ["git", "clone", "--filter=blob:none", "--no-checkout",
                 "--single-branch", "--branch", source["locator"],
                 source["repo"], tmp],
                capture_output=True, text=True, timeout=timeout, check=False,
            )
            if clone.returncode != 0:
                print(f"  CLONE-FAILED: {clone.stderr.strip().splitlines()[-1] if clone.stderr else 'unknown'}")
                findings += 1
                continue
            tip_res = _git(tmp, "rev-parse", "HEAD", timeout=60)
            tip = tip_res.stdout.strip()
            print(f"  active snapshot: {active['label']} head {active['head'][:9]}")
            print(f"  current tip:     {tip[:9]}")

            if tip == active["head"]:
                print("  branch state: UNCHANGED since last review")
            else:
                findings += 1
                ahead = _git(tmp, "rev-list", "--count", f"{active['head']}..{tip}", timeout=120)
                is_anc = _git(tmp, "merge-base", "--is-ancestor", active["head"], tip, timeout=60)
                if is_anc.returncode == 0:
                    print(f"  FINDING moved: {ahead.stdout.strip()} new commit(s) since review:")
                    log = _git(tmp, "log", "--oneline", f"{active['head']}..{tip}", timeout=120)
                    for line in log.stdout.splitlines():
                        print(f"    {line}")
                elif is_anc.returncode == 1:
                    print("  FINDING rebased: active head is NOT an ancestor of the "
                          "current tip -- a NEW snapshot + re-audit is required "
                          "(compare patch-ids of tracked commits before re-planning)")
                else:
                    print("  FINDING unknown: could not relate tip to active head")

            # Mainline merge gate: patch-id equality with the upstream repo.
            up = _git(tmp, "remote", "add", "mainline", source["upstream"], timeout=60)
            if up.returncode != 0:
                findings += 1
                print("  FINDING mainline-unavailable: could not add upstream remote")
            else:
                fetch = _git(tmp, "fetch", "--filter=blob:none", "--depth=800",
                             "mainline", "master", timeout=timeout)
                if fetch.returncode != 0:
                    findings += 1
                    print("  FINDING mainline-unavailable: fetch failed")
                else:
                    mainline_tip = _git(tmp, "rev-parse", "mainline/master", timeout=60).stdout.strip()
                    cherry = _git(tmp, "cherry", "-v", "mainline/master", tip, timeout=300)
                    merged = {line.split()[1] for line in cherry.stdout.splitlines()
                              if line.startswith("- ")}
                    print(f"  mainline tip: {mainline_tip[:9]}")
                    for entry in source.get("tracked", []):
                        if entry["commit"] in merged:
                            findings += 1
                            print(f"  FINDING merged-upstream: {entry['commit'][:9]} "
                                  f"({entry.get('plan-item')}) has an equivalent "
                                  f"patch in mainline -- redundant after the next "
                                  f"pin bump; re-plan that item")

            # Content drift for tracked commits, only if the branch moved.
            if tip != active["head"]:
                moved_shas = set(
                    _git(tmp, "rev-list", f"{active['base']}..{tip}", timeout=120)
                    .stdout.split()
                )
                title_map: dict[str, str] = {}
                if moved_shas:
                    log = _git(tmp, "log", "--format=%H %s", f"{active['base']}..{tip}", timeout=120)
                    for line in log.stdout.splitlines():
                        sha, _, title = line.partition(" ")
                        title_map[title] = sha

                def pid_of(sha: str) -> str | None:
                    show = _git(tmp, "show", sha, timeout=300)
                    if show.returncode != 0:
                        return None
                    pid = subprocess.run(
                        ["git", "patch-id"], input=show.stdout,
                        capture_output=True, text=True, timeout=60, check=False,
                    )
                    return pid.stdout.split()[0] if pid.stdout.strip() else None

                for entry in source.get("tracked", []):
                    sha = entry["commit"]
                    if sha in moved_shas:
                        continue  # unchanged in the new history
                    equiv = title_map.get(entry.get("title", ""))
                    if equiv is None:
                        findings += 1
                        print(f"  FINDING drifted: {sha[:9]} ({entry.get('plan-item')}) "
                              f"not found by title in the new history -- review manually")
                        continue
                    if pid_of(sha) == pid_of(equiv):
                        print(f"  drift: {sha[:9]} re-committed as {equiv[:9]} "
                              f"(content identical)")
                    else:
                        findings += 1
                        print(f"  FINDING drifted: {sha[:9]} ({entry.get('plan-item')}) "
                              f"re-committed as {equiv[:9]} with CHANGED content -- "
                              f"re-audit before porting")
        print()

    if findings:
        print(f"sources check: {findings} finding(s) -- review before planning or pin bumps")
        return 2
    print("sources check: no movement, no mainline merges, no drift")
    return 0


def register(sub: argparse._SubParsersAction) -> None:
    """Attach the `sources` subcommand (called from __main__)."""
    sources_cmd = sub.add_parser(
        "sources",
        help="external-source registry: provenance + upstream tracking",
    )
    src_sub = sources_cmd.add_subparsers(dest="src_subcommand")
    src_sub.add_parser("status", help="offline registry view + cross-check")
    check_cmd = src_sub.add_parser(
        "check", help="online: branch moved / rebased / merged / drifted"
    )
    check_cmd.add_argument(
        "--timeout", type=int, default=600,
        help="per git operation timeout in seconds (default 600)",
    )
    sources_cmd.set_defaults(func=cmd_sources)
