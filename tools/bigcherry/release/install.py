"""Install and verify a published Radiance runtime in its release-tag worktree.

The release tag, not the current development checkout, supplies both the record
and the immutable engine/config trees used by BU01 runtime_assets.restore.
"""

from __future__ import annotations

import json
import os
import re
import shlex
import subprocess
import sys
import tempfile
import uuid
from datetime import UTC, datetime
from pathlib import Path

from . import runtime_assets

_TAG = re.compile(r"bc-radiance-(\d+)\.(\d+)\.(\d+)\Z")
_RECORD = re.compile(r"releases/[A-Za-z0-9._-]+\.json\Z")
_LAUNCHER = "tools/lab/radiance/launch-flashnext-best.sh"


class ReleaseInstallError(RuntimeError):
    """A release is missing, unverifiable, or unsafe to serve."""


def _git(
    root: Path, *args: str, check: bool = True
) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        ["git", "-C", str(root), *args], text=True, capture_output=True, check=False
    )
    if check and result.returncode:
        raise ReleaseInstallError(
            f"git {' '.join(args[:2])} failed: {(result.stderr or result.stdout).strip()[-400:]}"
        )
    return result


def _record_at_tag(root: Path, tag: str) -> dict:
    """Find the pin's record *at this release tag*, never from the live checkout."""
    ref = f"refs/tags/{tag}"
    if _git(root, "rev-parse", "--verify", f"{ref}^{{commit}}", check=False).returncode:
        raise ReleaseInstallError(
            f"release tag not found locally: {tag} (fetch tags first)"
        )
    grep = _git(root, "grep", "-l", "-F", "-e", tag, ref, "--", "releases", check=False)
    if grep.returncode not in (0, 1):
        raise ReleaseInstallError(
            f"cannot inspect release records at {tag}: {grep.stderr.strip()}"
        )
    matches = []
    for line in grep.stdout.splitlines():
        path = line.partition(":")[2]
        if not _RECORD.fullmatch(path) or path == "releases/index.json":
            continue
        content = _git(root, "show", f"{ref}:{path}").stdout
        try:
            record = json.loads(content)
        except json.JSONDecodeError as exc:
            raise ReleaseInstallError(
                f"invalid release record at {tag}:{path}"
            ) from exc
        if (
            isinstance(record, dict)
            and isinstance(record.get("runtime_assets"), dict)
            and tag in record["runtime_assets"]
        ):
            if record.get("release_tag") != Path(path).stem:
                raise ReleaseInstallError(
                    f"release asset is not in its pin's record: {path}"
                )
            matches.append(record)
    if len(matches) != 1:
        raise ReleaseInstallError(
            f"{tag}: expected one pin release record with runtime_assets entry, found {len(matches)}"
        )
    return matches[0]


def resolve_tag(root: Path, requested: str) -> tuple[str, dict]:
    if requested != "latest":
        if not _TAG.fullmatch(requested):
            raise ReleaseInstallError(f"invalid Radiance release tag: {requested!r}")
        return requested, _record_at_tag(root, requested)
    tags = _git(root, "tag", "--list", "bc-radiance-*").stdout.splitlines()
    eligible = sorted(
        ((_TAG.fullmatch(tag), tag) for tag in tags),
        key=lambda pair: tuple(map(int, pair[0].groups())) if pair[0] else (-1,),
        reverse=True,
    )
    for match, tag in eligible:
        if match:
            try:
                return tag, _record_at_tag(root, tag)
            except ReleaseInstallError as exc:
                # An old tag with no runtime bundle is not an install candidate.
                if (
                    "expected one pin release record with runtime_assets entry, found 0"
                    not in str(exc)
                ):
                    raise
    raise ReleaseInstallError(
        "no bc-radiance release tag has a recorded runtime_assets bundle"
    )


def _clean_release_worktree(root: Path, src: Path, tag: str) -> None:
    if src.is_symlink() or not (src / ".git").is_file():
        raise ReleaseInstallError(
            f"release source is not a detached git worktree: {src}"
        )
    if (
        Path(_git(src, "rev-parse", "--show-toplevel").stdout.strip()).resolve()
        != src.resolve()
    ):
        raise ReleaseInstallError(f"release worktree points elsewhere: {src}")
    if _git(src, "symbolic-ref", "-q", "HEAD", check=False).returncode == 0:
        raise ReleaseInstallError(f"release worktree must be detached: {src}")
    expected = _git(root, "rev-parse", f"refs/tags/{tag}^{{commit}}").stdout.strip()
    if _git(src, "rev-parse", "HEAD").stdout.strip() != expected:
        raise ReleaseInstallError(f"release worktree is not at {tag}: {src}")
    if _git(src, "status", "--porcelain", "--untracked-files=all").stdout.strip():
        raise ReleaseInstallError(f"release worktree is dirty; refusing install: {src}")


def _create_worktree(root: Path, src: Path, tag: str) -> bool:
    if src.exists() or src.is_symlink():
        _clean_release_worktree(root, src, tag)
        return False
    src.parent.mkdir(parents=True, exist_ok=True)
    pending = src.parent / f".src-pending-{uuid.uuid4().hex}"
    try:
        _git(root, "worktree", "add", "--detach", str(pending), f"refs/tags/{tag}")
        _git(root, "worktree", "move", str(pending), str(src))
        _clean_release_worktree(root, src, tag)
        return True
    except Exception:
        if src.exists():
            _git(root, "worktree", "remove", "--force", str(src), check=False)
        raise
    finally:
        if pending.exists():
            _git(root, "worktree", "remove", "--force", str(pending), check=False)


def _receipt(tag: str, src: Path, build: Path, asset: dict) -> dict:
    return {
        "tag": tag,
        "worktree": str(src),
        "build_dir": str(build),
        "plan_id": asset["plan_id"],
        "source_slice_id": asset["source_slice_id"],
        "binary_sha256": asset["binary_sha256"],
        "archive_sha256": asset["sha256"],
        "manifest_sha256": asset["manifest_sha256"],
        "engine_tree": asset["engine_tree"],
        "config_tree": asset["config_tree"],
        "rocm_version": asset["rocm_version"],
        "installed_at": datetime.now(UTC).isoformat(),
    }


def _save_receipt(path: Path, receipt: dict) -> None:
    # Publication is the last step: no partial install.json becomes visible.
    tmp = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            dir=path.parent,
            prefix=".install-",
            suffix=".pending",
            delete=False,
            encoding="utf-8",
        ) as f:
            tmp = Path(f.name)
            json.dump(receipt, f, indent=2, sort_keys=True)
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())
        tmp.replace(path)
    finally:
        if tmp is not None and tmp.exists():
            tmp.unlink()


def install(requested: str, *, root: Path, work_root: Path, sdk: Path) -> dict:
    """Install once or verify an existing runtime, returning the installation receipt."""
    root, work_root = root.resolve(), work_root.resolve()
    tag, record = resolve_tag(root, requested)
    asset = record["runtime_assets"][tag]
    if not isinstance(asset, dict) or asset.get("schema") != runtime_assets.SCHEMA:
        raise ReleaseInstallError(
            f"{tag}: unsupported or missing runtime asset receipt"
        )
    # Fail *before* creating source/build directories when the selected SDK differs.
    actual_sdk = runtime_assets.sdk_version(sdk)
    if asset.get("rocm_version") != actual_sdk:
        raise ReleaseInstallError(
            f"{tag}: ROCm SDK mismatch: release {asset.get('rocm_version')}, installed {actual_sdk}"
        )
    source_id, plan_id = asset.get("source_slice_id"), asset.get("plan_id")
    if (
        not isinstance(source_id, str)
        or not re.fullmatch(r"[A-Za-z0-9_-]{1,100}", source_id)
        or not isinstance(plan_id, str)
        or not re.fullmatch(r"[0-9a-f]{32}", plan_id)
    ):
        raise ReleaseInstallError(
            f"{tag}: unsafe source slice or plan ID in release record"
        )
    target = work_root / "builds" / source_id / plan_id
    release_dir = work_root / "releases" / tag
    src = release_dir / "src"
    receipt_path = release_dir / "install.json"
    created = False
    try:
        created = _create_worktree(root, src, tag)
        _clean_release_worktree(root, src, tag)
        identity = runtime_assets.stock_identity(src)
        if any(identity[k] != asset.get(k) for k in ("engine_tree", "config_tree")):
            raise ReleaseInstallError(
                f"{tag}: release worktree source trees differ from runtime bundle"
            )
        if receipt_path.exists():
            try:
                saved = json.loads(receipt_path.read_text(encoding="utf-8"))
            except (OSError, ValueError) as exc:
                raise ReleaseInstallError(
                    f"{tag}: invalid existing install.json"
                ) from exc
            expected = _receipt(tag, src, target, asset)
            if not isinstance(saved, dict) or any(
                saved.get(k) != v for k, v in expected.items() if k != "installed_at"
            ):
                raise ReleaseInstallError(
                    f"{tag}: install.json differs from the signed release record or expected paths"
                )
        if target.exists():
            binary = target / "bin/radiance"
            if not binary.is_file() or runtime_assets.digest(binary) != asset.get(
                "binary_sha256"
            ):
                raise ReleaseInstallError(
                    f"{tag}: existing build binary SHA-256 mismatch; refusing to reuse {target}"
                )
            if not (target / "radiance_home/helpers/r3_moe_worker").is_file():
                raise ReleaseInstallError(
                    f"{tag}: existing build is missing the helper binary"
                )
        else:
            try:
                restored = runtime_assets.fetch(record, src, work_root, sdk, tag)
            except (runtime_assets.BundleError, OSError, KeyError, ValueError) as exc:
                raise ReleaseInstallError(
                    f"{tag}: verified runtime fetch/restore failed: {exc}"
                ) from exc
            if restored != target:
                raise ReleaseInstallError(
                    f"{tag}: restored unexpected build directory {restored}"
                )
        if receipt_path.is_file():
            return saved
        receipt = _receipt(tag, src, target, asset)
        _save_receipt(receipt_path, receipt)
        return receipt
    except Exception:
        if created:
            _git(root, "worktree", "remove", "--force", str(src), check=False)
            if release_dir.exists() and not any(release_dir.iterdir()):
                release_dir.rmdir()
        raise


def launch_command(receipt: dict) -> str:
    return (
        f"bash {shlex.quote(str(Path(receipt['worktree']) / _LAUNCHER))} "
        f"start --build {shlex.quote(receipt['build_dir'])}"
    )


def cmd_release_install(args) -> int:
    from ..campaign.build import CampaignBuildError, lane_sdk_root
    from ..core import engines, paths
    from ..core.context import ProjectContext

    if args.engine != engines.RADIANCE:
        print("release-install: only --engine radiance is supported", file=sys.stderr)
        return 2
    root = paths.REPO_ROOT
    try:
        work = ProjectContext.resolve(engine=args.engine, project_root=root).work_root
        sdk = lane_sdk_root(
            engines.load(root, args.engine).recipes_path(root), "linux-radiance"
        )
        receipt = install(args.tag, root=root, work_root=work, sdk=sdk)
    except (
        ReleaseInstallError,
        runtime_assets.BundleError,
        CampaignBuildError,
        OSError,
        ValueError,
        KeyError,
    ) as exc:
        print(f"release-install: {exc}", file=sys.stderr)
        return 1
    if args.json:
        print(json.dumps(receipt, sort_keys=True))
    else:
        print(launch_command(receipt))
    return 0
