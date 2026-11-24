"""WSL09: conservative, opt-in garbage collection of content-addressed build trees.

Only builds/<source-slice>/<build-plan-id> directories are eligible. Inventory
is read-only; apply takes the campaign's exact build-plan ResourceLock and
rechecks the candidate and process/manifest references while holding it.
Sources are never deleted by this module.
"""
from __future__ import annotations

import json
import math
import os
import re
import stat
import time
import uuid
from collections.abc import Callable
from pathlib import Path

from . import tree_activity
from .resources import ResourceError, ResourceLock


class BuildGCError(RuntimeError):
    def __init__(self, message: str, *, quarantine_path: Path | None = None):
        super().__init__(message)
        self.quarantine_path = quarantine_path


_QUARANTINE = ".gc-quarantine"


def _cache_id(name: str) -> bool:
    # The build cache only owns content-addressed source slices and plans.
    return re.fullmatch(r"[0-9a-f]{32}", name) is not None


def _reject_quarantine_mounts(directory: Path, mounts: set[Path]) -> None:
    # Quarantine is immediately under builds/, unlike a four-component build.
    builds = directory.parent
    for mount in sorted(mounts):
        if mount == builds or mount.is_relative_to(directory):
            raise BuildGCError(f"mount point on build GC path: {mount}")


def _quarantine_problem(directory: Path, mounts: set[Path]) -> str | None:
    """The quarantine must be private, same-device, unsymlinked and unmounted."""
    try:
        _reject_quarantine_mounts(directory, mounts)
        try:
            observed = directory.lstat()
        except FileNotFoundError:
            return None
        if (not stat.S_ISDIR(observed.st_mode)
                or observed.st_uid != os.geteuid()
                or stat.S_IMODE(observed.st_mode) != 0o700
                or observed.st_dev != directory.parent.lstat().st_dev):
            return f"untrusted build GC quarantine directory: {directory}"
    except (BuildGCError, OSError) as exc:
        return f"untrusted build GC quarantine: {exc}"
    return None


def _inside(path: str, directory: Path) -> bool:
    """A component-boundary path match, not a substring/prefix match."""
    if not path.startswith("/"):
        return False
    path = path.removesuffix(" (deleted)")
    normalized = os.path.normpath(path)
    base = str(directory.resolve())
    return normalized == base or normalized.startswith(base + os.sep)


def _outside_cgroup_namespace(proc: Path) -> bool:
    """True when the process lives outside this cgroup namespace (cgroup path
    escapes ours with "/..", cgroup v1 or v2). Unreadable means unknown, so
    in scope."""
    try:
        lines = (proc / "cgroup").read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return False
    paths = [line.split(":", 2)[2] for line in lines if line.count(":") >= 2]
    return bool(paths) and all(path == "/.." or path.startswith("/../") for path in paths)


def _proc_references(proc_root: Path, owner_uid: int) -> tuple[list[tuple[str, str]], list[str]]:
    """One /proc scan for all builds, including open descriptors and mappings.

    Builds are made and used only by the work root's owner inside this
    container, so only that user's processes in this cgroup namespace are in
    scope. Other users' processes (kernel threads, daemons) and processes
    outside the namespace (the host, other containers: their cgroup path
    escapes ours with a leading "/..") cannot hold one and are not inspected.
    Disappearing PIDs are normal; an unreadable *surviving* in-scope process
    (or one whose cgroup cannot be read) is not proof of inactivity and
    fails closed.
    """
    refs: list[tuple[str, str]] = []
    errors: list[str] = []
    try:
        processes = [p for p in proc_root.iterdir() if p.name.isdecimal() and p.is_dir()]
    except OSError as exc:
        return refs, [f"cannot enumerate {proc_root}: {exc}"]
    for proc in processes:
        pid = proc.name
        try:
            if proc.stat().st_uid != owner_uid:
                continue
        except OSError:
            continue  # exited while scanning
        if _outside_cgroup_namespace(proc):
            continue
        for name in ("exe", "cwd"):
            try:
                target = os.readlink(proc / name)
                if target.startswith("/"):
                    refs.append((target, f"live process {pid} {name}"))
            except FileNotFoundError:
                continue
            except OSError as exc:
                if proc.exists():
                    errors.append(f"cannot inspect process {pid} {name}: {exc}")
        try:
            with os.scandir(proc / "fd") as directory:
                for fd in directory:
                    try:
                        target = os.readlink(fd.path)
                        if target.startswith("/"):
                            refs.append((target, f"live process {pid} open file"))
                    except FileNotFoundError:
                        continue
                    except OSError as exc:
                        if proc.exists():
                            errors.append(f"cannot inspect process {pid} fd: {exc}")
        except FileNotFoundError:
            pass
        except OSError as exc:
            if proc.exists():
                errors.append(f"cannot inspect process {pid} open files: {exc}")
        try:
            with (proc / "maps").open(encoding="utf-8", errors="replace") as maps:
                for line in maps:
                    fields = line.split(maxsplit=5)
                    if len(fields) == 6:
                        target = fields[5].strip()
                        if target.startswith("/"):
                            refs.append((target, f"live process {pid} map"))
        except FileNotFoundError:
            pass
        except OSError as exc:
            if proc.exists():
                errors.append(f"cannot inspect process {pid} maps: {exc}")
    return refs, errors


def _manifest_refs(root: Path) -> tuple[dict[str, list[str]], list[str]]:
    """Resolve typed build edges and absolute build paths from run manifests.

    A corrupt/unreadable run record might hide a live reference, so none of
    the builds may be removed until it is repaired.
    """
    from . import legacy_runs, retention

    runs = root / "runs"
    if runs.is_symlink() or not runs.is_dir():
        return {}, [f"runs directory missing or untrusted: {runs}"]
    try:
        inv = retention.inspect_workspace(root, measure=False)
    except (OSError, ValueError) as exc:
        return {}, [f"cannot fully inspect runs directory: {exc}"]
    refs: dict[str, list[str]] = {}
    errors = [f"unreadable run manifest: {e.path}: {e.problem}" for e in inv.runs if e.manifest is None]
    for unknown in inv.unknown:
        if unknown.is_relative_to(runs):
            if unknown.parent == runs and unknown.is_file() and not unknown.is_symlink():
                # Loose historical documents have no run directory for a
                # manifest. Fully scan them for pins instead of ignoring refs.
                try:
                    edges, _ = legacy_runs.scan_loose_file(root, unknown.name)
                    for edge in edges:
                        refs.setdefault(edge["id"], []).append(f"loose run file {unknown.name}")
                except (OSError, legacy_runs.LegacyRunError) as exc:
                    errors.append(f"unclassified run file: {unknown}: {exc}")
            else:
                errors.append(f"unclassified run path: {unknown}")
    for e in inv.runs:
        if not e.manifest:
            continue
        m = e.manifest
        hold = m["retention"].get("hold", False)
        label = f"held run {e.run_id}" if hold else f"run manifest {e.run_id}"
        for ref in m["references"]:
            if ref["type"] == "build":
                refs.setdefault(str(ref["id"]), []).append(label)
        for item in m["inputs"]:
            if isinstance(item, dict) and isinstance(item.get("uri"), str):
                uri = item["uri"]
                uri = uri.removeprefix("file://")
                if uri.startswith("/"):
                    refs.setdefault(uri, []).append(label + " input")
    return refs, errors


def _release_refs(releases: Path) -> tuple[dict[str, list[str]], list[str]]:
    """Tracked release records explicitly pin their published build-plan IDs."""

    refs: dict[str, list[str]] = {}
    errors: list[str] = []
    if not releases.exists():
        return refs, errors
    if releases.is_symlink() or not releases.is_dir():
        return refs, ["release records directory is untrusted"]
    try:
        records = sorted(releases.glob("*.json"))
        for record in records:
            try:
                if record.is_symlink():
                    raise ValueError("symlinked release record")
                doc = json.loads(record.read_text(encoding="utf-8"))
                if not isinstance(doc, dict) or not isinstance(doc.get("notes", ""), str):
                    raise TypeError("invalid release record")
                for build_id in re.findall(r"\bbuild_plan_id=([0-9a-f]{32})\b", doc.get("notes", "")):
                    refs.setdefault(build_id, []).append(f"release record {record.name}")
            except (OSError, ValueError, TypeError) as exc:
                errors.append(f"unreadable release record {record.name}: {exc}")
    except OSError as exc:
        errors.append(f"cannot enumerate release records: {exc}")
    return refs, errors


def _installed_release_refs(root: Path) -> tuple[dict[str, list[str]], list[str]]:
    """Installed work-root release receipts pin their exact installed build."""
    releases = root / "releases"
    refs: dict[str, list[str]] = {}
    errors: list[str] = []
    if releases.is_symlink() or (releases.exists() and not releases.is_dir()):
        return refs, ["installed releases directory is untrusted"]
    if not releases.exists():
        return refs, errors
    try:
        for release in sorted(releases.iterdir()):
            try:
                if release.is_symlink() or not release.is_dir():
                    raise ValueError("untrusted release entry")
                receipt = release / "install.json"
                if receipt.is_symlink() or not receipt.is_file():
                    raise ValueError("missing or untrusted install.json")
                doc = json.loads(receipt.read_text(encoding="utf-8"))
                if not isinstance(doc, dict) or doc.get("tag") != release.name:
                    raise ValueError("invalid install receipt identity")
                slice_id, plan_id = doc.get("source_slice_id"), doc.get("plan_id")
                if (not isinstance(slice_id, str)
                        or re.fullmatch(r"[A-Za-z0-9_-]{1,100}", slice_id) is None
                        or not isinstance(plan_id, str)
                        or re.fullmatch(r"[0-9a-f]{32}", plan_id) is None):
                    raise ValueError("invalid install receipt build identity")
                expected = root / "builds" / slice_id / plan_id
                if doc.get("build_dir") != str(expected) or doc.get("source") not in ("ci", "record"):
                    raise ValueError("install receipt build path or source mismatch")
                refs.setdefault(str(expected), []).append(f"installed release {release.name}")
            except (OSError, ValueError, TypeError) as exc:
                errors.append(f"invalid installed release {release.name}: {exc}")
    except OSError as exc:
        errors.append(f"cannot enumerate installed releases: {exc}")
    return refs, errors


def _build_references(refs: dict[str, list[str]], build: Path) -> list[str]:
    keys = (build.name, f"{build.parent.name}/{build.name}",
            f"builds/{build.parent.name}/{build.name}", str(build))
    reasons = [reason for key in keys for reason in refs.get(key, ())]
    # MRC01 identity.build_id is the build metadata ID, not necessarily the
    # directory's build_plan_id. _age already refuses unsafe metadata.
    for metadata in build.glob("bigcherry-build-metadata-*.json"):
        try:
            if metadata.is_symlink():
                continue
            doc = json.loads(metadata.read_text(encoding="utf-8"))
            build_id = doc.get("build_id") if isinstance(doc, dict) else None
            if isinstance(build_id, str) and re.fullmatch(r"[0-9a-f]{32}", build_id):
                reasons.extend(refs.get(build_id, ()))
        except (OSError, ValueError, TypeError):
            pass  # _age protects malformed/unreadable sidecars
    for key, names in refs.items():
        if key.startswith("/") and _inside(key, build):
            reasons.extend(names)
    return list(dict.fromkeys(reasons))


def _age(build: Path, now: float) -> tuple[float, str, list[str]]:
    try:
        metadata = list(build.glob("bigcherry-build-metadata-*.json"))
        if metadata:
            mtimes = [build.stat().st_mtime]
            for path in metadata:
                if path.is_symlink() or not path.is_file():
                    return 0.0, "invalid", ["build metadata is not a regular file"]
                try:
                    data = json.loads(path.read_text(encoding="utf-8"))
                    if not isinstance(data, dict):
                        return 0.0, "invalid", ["build metadata is not an object"]
                    if (data.get("source_slice_id", build.parent.name) != build.parent.name
                            or data.get("build_plan_id", build.name) != build.name):
                        return 0.0, "invalid", ["build metadata identity disagrees with directory"]
                    mtimes.append(path.stat().st_mtime)
                except (OSError, ValueError) as exc:
                    return 0.0, "invalid", [f"unreadable build metadata: {exc}"]
            return (now - max(mtimes)) / 86400, "present-unverified", []
        return (now - build.stat().st_mtime) / 86400, "missing", []
    except OSError as exc:
        return 0.0, "untrusted", [f"cannot determine build age: {exc}"]


def _validate_policy(policy) -> None:
    value = policy.min_age_days
    if (not isinstance(value, (float, int)) or not math.isfinite(value)
            or value < 0 or type(policy.newest_k) is not int or policy.newest_k < 0):
        raise ValueError("build GC retention policy must be finite and non-negative")



def inspect(root: Path, *, policy, now: float | None = None,
            measure: bool = True, blocked: str | None = None,
            proc_root: Path = Path("/proc"), owned_lock: str | None = None,
            release_root: Path | None = None,
            directory_ids: dict[str, tuple[tuple[int, int], ...]] | None = None) -> list[dict]:
    """Candidate/protected inventory. Non-build cache paths are read-only."""
    from . import retention
    from .retention import dir_size

    root = Path(root).resolve()
    now = time.time() if now is None else now
    _validate_policy(policy)
    records: list[dict] = []
    reasons_by_path: dict[Path, list[str]] = {}
    builds_by_recipe: dict[Path, list[tuple[Path, float]]] = {}

    def record(path: Path, kind: str, metadata: str, reasons: list[str],
               age_days: float | None = None) -> None:
        records.append({
            "kind": kind, "path": str(path), "metadata": metadata,
            "size_bytes": (dir_size(path) if measure and path.is_dir()
                           and not path.is_symlink()
                           and (kind != "quarantine" or not quarantine_problem) else None),
            "age_days": round(age_days, 3) if age_days is not None else None,
            "action": "protected" if reasons else "candidate",
            "reasons": reasons or ["aged, unreferenced build; eligible for build GC"],
        })

    activity = tree_activity.cleanup_activity_blocker(root)
    blockers = [f"cleanup blocked: {why}" for why in (blocked, activity) if why]
    # Only a live install pins a build: an installed release root or a release bundle in the repo. Run manifests, run
    # records and stored serving manifests are history (their evidence lives in the docs), so they do not pin, and an
    # unreadable one never blocks cleanup.
    references: dict[str, list[str]] = {}
    manifest_errors: list[str] = []
    from . import paths
    release_refs, _ = _release_refs(release_root if release_root is not None else paths.REPO_ROOT / "releases")
    for key, names in release_refs.items():
        references.setdefault(key, []).extend(names)
    installed_refs, _ = _installed_release_refs(root)
    for key, names in installed_refs.items():
        references.setdefault(key, []).extend(names)
    extra_refs, _ = retention.native_installed_build_refs(root)
    for key, names in extra_refs.items():
        references.setdefault(key, []).extend(names)
    try:
        mounts = _mount_points()
        mount_error = None
    except BuildGCError as exc:
        mounts = set()
        mount_error = str(exc)
    proc_refs, proc_errors = _proc_references(proc_root, root.stat().st_uid)
    proc_errors = []   # a build a visible process uses is kept below; unreadable foreign processes do not block cleanup

    base = root / "builds"
    quarantine = base / _QUARANTINE
    quarantine_problem = (mount_error if mount_error else
                          _quarantine_problem(quarantine, mounts)) if base.is_dir() else None
    if quarantine_problem:
        blockers.append(quarantine_problem)
    proc_by_build: dict[Path, list[str]] = {}
    prefix = str(base) + os.sep
    for target, reason in proc_refs:
        normalized = os.path.realpath(target.removesuffix(" (deleted)"))
        if normalized.startswith(prefix):
            parts = normalized[len(prefix):].split(os.sep)
            if len(parts) >= 2:
                proc_by_build.setdefault(base / parts[0] / parts[1], []).append(reason)
    if base.is_symlink() or (base.exists() and not base.is_dir()):
        record(base, "build", "untrusted", ["build cache root is not a trusted directory"])
    elif base.is_dir():
        try:
            for recipe in sorted(base.iterdir()):
                if recipe.name == _QUARANTINE:
                    continue
                if recipe.is_symlink() or not recipe.is_dir():
                    record(recipe, "unclassified", "untrusted", ["unclassified build recipe or symlink"])
                    continue
                managed_recipe = _cache_id(recipe.name)
                if managed_recipe:
                    builds_by_recipe[recipe] = []
                for build in sorted(recipe.iterdir()):
                    if not managed_recipe:
                        record(build, "unclassified", "untrusted",
                               [f"non-content-addressed recipe: {recipe.name}"])
                        continue
                    if build.is_symlink() or not build.is_dir() or not _cache_id(build.name):
                        record(build, "unclassified", "untrusted",
                               ["non-build entry, symlink or invalid build-plan ID"])
                        continue
                    age, metadata, errors = _age(build, now)
                    builds_by_recipe[recipe].append((build, age))
                    reasons = list(errors)
                    if age < policy.min_age_days:
                        reasons.append(f"younger than {policy.min_age_days:g} days ({age:.1f})")
                    reasons.extend(_build_references(references, build))
                    reasons.extend(proc_by_build.get(build, ()))
                    reasons.extend(blockers)
                    reasons.extend(manifest_errors)
                    reasons.extend(proc_errors)
                    lock = ResourceLock(root / "resource-locks", f"build-{build.name}")
                    if build.name != owned_lock and (lock.path.exists() or lock.path.is_symlink()):
                        reasons.append(f"build-plan resource lock exists: {lock.path.name}")
                    reasons_by_path[build] = list(dict.fromkeys(reasons))
                    record(build, "build", metadata, [], age)
            # Interrupted removals are never eligible for automated recovery:
            # their original build ID, pins and lock cannot be inferred safely
            # from the generated quarantine name.
            def record_leftover(leftover: Path, other_reasons: list[str]) -> None:
                reasons = ["interrupted removal; verify and remove by hand", *other_reasons]
                try:
                    age = (now - leftover.lstat().st_mtime) / 86400
                except OSError as exc:
                    age = None
                    reasons.append(f"cannot stat quarantine entry: {exc}")
                record(leftover, "quarantine", "interrupted",
                       list(dict.fromkeys(reasons)), age)

            if quarantine_problem:
                if quarantine.is_symlink() or quarantine.exists():
                    record(quarantine, "quarantine", "untrusted", [quarantine_problem])
                # Report individual names only when the parent directory is
                # safe to enumerate, even if inspection of its children is not.
                if (quarantine not in mounts and quarantine.is_dir()
                        and not quarantine.is_symlink()
                        and _quarantine_problem(quarantine, set()) is None):
                    for leftover in sorted(quarantine.iterdir()):
                        record_leftover(leftover, [quarantine_problem])
            elif quarantine.is_dir():
                for leftover in sorted(quarantine.iterdir()):
                    record_leftover(leftover, blockers + manifest_errors + proc_errors)
            # Newest K per recipe, including any partial builds. They cannot
            # become candidates simply because another build was deleted.
            newest = {
                build for group in builds_by_recipe.values()
                for build, _ in sorted(group, key=lambda item: (item[1], item[0].name))[:policy.newest_k]
            } if policy.newest_k else set()
            for item in records:
                path = Path(item["path"])
                if path not in reasons_by_path:
                    continue
                reasons = reasons_by_path[path]
                if path in newest:
                    reasons.append(f"one of the newest {policy.newest_k} builds for recipe {path.parent.name}")
                if not reasons:
                    if mount_error:
                        reasons.append(mount_error)
                    else:
                        try:
                            _single_device_tree(path, mounts=mounts)
                            if directory_ids is not None:
                                directory_ids[str(path)] = _candidate_directory_ids(root, path)
                        except BuildGCError as exc:
                            reasons.append(str(exc))
                if reasons:
                    item["action"] = "protected"
                    item["reasons"] = reasons
        except OSError as exc:
            # Inventory cannot be complete; all enumerated candidates must
            # be revoked, including those inspected before the error.
            for item in records:
                if item["kind"] in ("build", "quarantine") and item["action"] == "candidate":
                    item["action"] = "protected"
                    item["reasons"] = [f"build listing incomplete: {exc}"]
            record(base, "build", "untrusted", [f"build listing incomplete: {exc}"])

    source = root / "sources"
    if source.is_symlink() or (source.exists() and not source.is_dir()):
        record(source, "source", "untrusted", ["source cache is not a trusted directory"])
    elif source.is_dir():
        try:
            for entry in sorted(source.iterdir()):
                if (entry.is_file() and not entry.is_symlink()
                        and entry.name.endswith(".metadata.json")
                        and (source / entry.name[:-len(".metadata.json")]).is_dir()):
                    continue
                if entry.is_symlink() or not entry.is_dir():
                    record(entry, "source", "untrusted", ["unclassified source entry or symlink"])
                    continue
                sidecar = source / (entry.name + ".metadata.json")
                if sidecar.is_file() and not sidecar.is_symlink():
                    try:
                        metadata = "present-unverified" if isinstance(json.loads(sidecar.read_text()), dict) else "invalid"
                    except (OSError, ValueError):
                        metadata = "invalid"
                else:
                    metadata = "missing"
                record(entry, "source", metadata, ["source cache has read-only inventory; source GC is not authorised"])
        except OSError as exc:
            record(source, "source", "untrusted", [f"source listing incomplete: {exc}"])
    return records


def _mount_points(mountinfo: Path = Path("/proc/self/mountinfo")) -> set[Path]:
    """Decode Linux mountinfo paths, including same-device bind mount targets."""
    try:
        lines = mountinfo.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise BuildGCError(f"cannot inspect mount table: {exc}") from exc
    points: set[Path] = set()
    for line in lines:
        fields = line.split(" - ", 1)[0].split()
        if len(fields) < 5 or not fields[4].startswith("/"):
            raise BuildGCError("invalid mount table entry")
        decoded = re.sub(r"\\([0-7]{3})", lambda m: chr(int(m.group(1), 8)), fields[4])
        points.add(Path(decoded))
    return points


def _reject_build_mounts(build: Path, mounts: set[Path]) -> None:
    """Protect mounts from builds/ down to the candidate and its descendants.

    A mount at/above the work root is an explicit policy choice; a mount at
    builds/, the recipe, the candidate or any descendant is never eligible.
    """
    builds = build.parent.parent
    for mount in sorted(mounts):
        if mount.is_relative_to(builds) and (
                build.is_relative_to(mount) or mount.is_relative_to(build)):
            raise BuildGCError(f"mount point on build GC path: {mount}")


def _single_device_tree(path: Path, *, mounts: set[Path] | None = None) -> None:
    """Refuse a symlinked build and any mount point in or above it. Symlinks, special files and files of other
    filesystems inside the tree are fine: removal unlinks them and never follows a link."""
    if mounts is None:
        mounts = _mount_points()
    try:
        if path.is_symlink():
            raise BuildGCError(f"symlinked build: {path}")
        _reject_build_mounts(path, mounts)
    except OSError as exc:
        raise BuildGCError(f"cannot verify build tree: {exc}") from exc


def _candidate_directory_ids(root: Path, path: Path) -> tuple[tuple[int, int], ...]:
    """Snapshot the inspected work-root/builds/recipe/build inode chain.

    lstat rejects final-component symlinks; removal independently opens
    every component relative to its verified parent and compares the IDs.
    """
    if path.parent.parent != root / "builds":
        raise BuildGCError(f"unexpected build GC path: {path}")
    identities = []
    for component in (root, root / "builds", path.parent, path):
        try:
            observed = component.lstat()
        except OSError as exc:
            raise BuildGCError(f"directory component unavailable: {component}: {exc}") from exc
        if not stat.S_ISDIR(observed.st_mode):
            raise BuildGCError(f"directory component changed or symlinked: {component}")
        identities.append((observed.st_dev, observed.st_ino))
    return tuple(identities)


def _remove_build_tree(path: Path, expected: tuple[tuple[int, int], ...]) -> str | None:
    """Delete the build directory. Symlinks and special files inside are unlinked, never followed."""
    import shutil

    if path.parent.parent.name != "builds" or path.is_symlink() or not path.is_dir():
        raise BuildGCError(f"not a build directory: {path}")
    _reject_build_mounts(path, _mount_points())
    shutil.rmtree(path)
    return None


def apply(root: Path, *, policy, blocked_check: Callable[[], str | None],
          now: float | None = None) -> dict:
    """Delete only freshly revalidated candidates, with an append-only audit."""
    _validate_policy(policy)
    root = Path(root).resolve()
    if not root.is_dir() or (root / "builds").is_symlink():
        raise BuildGCError("work root or build cache root is not a trusted directory")
    initial_ids: dict[str, tuple[tuple[int, int], ...]] = {}
    initial = inspect(root, policy=policy, now=now, blocked=blocked_check(),
                      directory_ids=initial_ids)
    candidates = sorted((e for e in initial
                         if e["kind"] == "build" and e["action"] == "candidate"),
                        key=lambda entry: (-(entry["age_days"] or 0), entry["path"]))
    removed: list[dict] = []
    skipped: list[dict] = []
    journal: Path | None = None

    def audit(record: dict) -> None:
        nonlocal journal
        if journal is None:
            directory = root / "cleanup-records"
            if directory.is_symlink():
                raise BuildGCError("cleanup-records is a symlink")
            directory.mkdir(exist_ok=True)
            journal = directory / f"build-gc-{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}-{uuid.uuid4().hex}.jsonl"
        with journal.open("a", encoding="utf-8") as out:
            out.write(json.dumps(record, sort_keys=True) + "\n")
            out.flush()
            os.fsync(out.fileno())

    for entry in candidates:
        path = Path(entry["path"])
        try:
            lock = ResourceLock(root / "resource-locks", f"build-{path.name}")
            lock.acquire(timeout_seconds=0)
            try:
                if path.is_symlink() or not path.is_dir():
                    skipped.append({"path": str(path), "reason": "no longer a build directory"})
                    continue
                record = {
                    "path": str(path), "size_bytes": entry["size_bytes"],
                    "age_days": entry["age_days"], "reason": entry["reasons"][0],
                    "recorded_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                }
                audit({**record, "state": "pending"})
                warning = _remove_build_tree(path, ())
                if warning:
                    record["warning"] = warning
                try:
                    audit({**record, "state": "removed"})
                except OSError as exc:
                    record["warning"] = f"completion journal entry not written: {exc}"
                removed.append(record)
            finally:
                lock.release()
        except (OSError, ResourceError, BuildGCError) as exc:
            skipped.append({"path": str(path), "reason": str(exc)})
    return {"removed": removed, "skipped": skipped, "record": str(journal) if journal else None}
