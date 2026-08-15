"""Host-local upstream repository and immutable source materialisation."""

from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

from . import patcher, patchset
from .context import ProjectContext
from .source_identity import describe


class WorkspaceError(RuntimeError):
    pass


def _git(root: Path, *args: str) -> str:
    try:
        result = subprocess.run(
            ["git", "-C", str(root), *args],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        detail = getattr(exc, "stderr", "") or str(exc)
        raise WorkspaceError(detail.strip()) from exc
    return result.stdout.strip()


class UpstreamRepository:
    def __init__(self, path: Path):
        self.path = path.resolve()

    def resolve_ref(self, ref: str) -> str:
        return _git(self.path, "rev-parse", f"{ref}^{{commit}}")

    def ensure_commit(self, revision: str) -> None:
        _git(self.path, "cat-file", "-e", f"{revision}^{{commit}}")

    def add_detached_worktree(self, revision: str, path: Path) -> None:
        self.ensure_commit(revision)
        path = path.resolve()
        if path.exists() and any(path.iterdir()):
            raise WorkspaceError(f"source worktree target is not empty: {path}")
        path.parent.mkdir(parents=True, exist_ok=True)
        _git(self.path, "worktree", "add", "--detach", str(path), revision)
        actual = _git(path, "rev-parse", "HEAD")
        if actual != revision:
            raise WorkspaceError(f"worktree HEAD {actual} does not equal {revision}")

    def remove_worktree(self, path: Path) -> None:
        _git(self.path, "worktree", "remove", "--force", str(path.resolve()))


@dataclass(frozen=True)
class SourcePlan:
    upstream_revision: str
    overlay_enabled: bool
    patch_ids: tuple[str, ...]
    required_state: str | None = None


def materialize(
    context: ProjectContext,
    plan: SourcePlan,
    destination: Path,
    *,
    allow_dirty_bigcherry: bool = False,
) -> dict[str, object]:
    """Materialise one source plan and return external identity metadata."""
    if not allow_dirty_bigcherry:
        status = subprocess.run(
            ["git", "-C", str(context.project_root), "status", "--porcelain"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        if status:
            raise WorkspaceError("BigCherry repository is dirty; use explicit development override")
    repository = UpstreamRepository(context.upstream_repo)
    repository.add_detached_worktree(plan.upstream_revision, destination)
    allowed_untracked: set[str] = set()
    if plan.overlay_enabled:
            for source in sorted(context.overlay_root.rglob("*")):
                if not source.is_file():
                    continue
            relative = source.relative_to(context.overlay_root)
            target = destination / relative
            was_present = target.exists()
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
            if not was_present:
                allowed_untracked.add(relative.as_posix())
    selection = patchset.resolve_exact(
        plan.patch_ids,
        directory=context.patches_root,
        required_state=plan.required_state,
    )
    results = patcher.apply_all(patchset.load_resolved(selection), destination)
    if not all(result.ok for result in results):
        raise WorkspaceError("source patch application failed")
    metadata = describe(
        root=destination,
        upstream_revision=plan.upstream_revision,
        allowed_untracked=allowed_untracked,
    )
    metadata["plan"] = {
        "upstream_revision": plan.upstream_revision,
        "overlay_enabled": plan.overlay_enabled,
        "patches": [
            {"patch_id": item.patch_id, "content_hash": item.content_hash}
            for item in selection.modules
        ],
    }
    return metadata
