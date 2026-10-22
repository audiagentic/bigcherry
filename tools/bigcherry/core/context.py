"""Explicit project and host-local campaign paths.

The campaign migration must not hide its worktree and artifact roots in
module-location globals.  This small immutable context keeps those roots
explicit while ``paths.py`` remains available to legacy commands.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from . import engines, paths


def _absolute(value: str | os.PathLike[str]) -> Path:
    return Path(value).expanduser().resolve()


def _paths_overlap(first: Path, second: Path) -> bool:
    """Return whether two resolved roots alias or contain one another.

    ``os.path.commonpath`` handles path components correctly (unlike string
    prefixes such as ``work``/``work-old``) and raises on different Windows
    drives, which are necessarily disjoint.
    """
    try:
        common = Path(os.path.commonpath((str(first), str(second))))
    except ValueError:
        return False
    return common == first or common == second


# Engine declarations are read from the repository this code belongs to (fixed at import); their relative layout is
# then resolved against whichever project root a context is for.
_DECLARATIONS = paths.REPO_ROOT


@dataclass(frozen=True)
class ProjectContext:
    engine: str
    project_root: Path
    config_path: Path
    artifacts_root: Path
    work_root: Path
    upstream_repo: Path
    overlay_root: Path
    patches_root: Path

    @property
    def layout(self) -> engines.EngineLayout:
        """The engine's declaration (engines/<name>/engine.toml)."""
        return engines.load(_DECLARATIONS, self.engine)

    @property
    def vendor_root(self) -> Path:
        """The engine's vendor checkout; it belongs to the primary checkout, never to a slice worktree."""
        return self.layout.vendor_root(paths.primary_root())

    @property
    def upstream_url(self) -> str:
        """Where the engine's source is published (the remote ``pull`` clones and release tags are read from)."""
        return self.layout.upstream

    @classmethod
    def resolve(
        cls,
        *,
        engine: str,
        project_root: str | os.PathLike[str] | None = None,
        config_path: str | os.PathLike[str] | None = None,
        artifacts_root: str | os.PathLike[str] | None = None,
        work_root: str | os.PathLike[str] | None = None,
        upstream_repo: str | os.PathLike[str] | None = None,
    ) -> "ProjectContext":
        """Resolve slice-local source paths and primary-checkout shared state."""
        layout = engines.load(_DECLARATIONS, engine)
        project = _absolute(
            project_root
            or os.environ.get("BIGCHERRY_PROJECT_ROOT")
            or paths.REPO_ROOT
        )
        config = _absolute(
            config_path
            or os.environ.get("BIGCHERRY_CONFIG_PATH")
            or layout.recipes_path(project)
        )
        artifacts = _absolute(
            artifacts_root
            or os.environ.get("BIGCHERRY_ARTIFACT_ROOT")
            or project / "artifacts"
        )
        # Source/config/patches remain slice-local. Mutable build/cache state
        # defaults to the primary checkout so linked worktrees share it.
        # Failure to resolve the primary checkout is fatal; never fall back to
        # a worktree-local cache and silently create a second source/build tree.
        # The host file's [env] is the documented place for a machine's work root; the process
        # environment still wins, as for every other ${VAR} it carries.
        from .config import _local_env_table

        work = _absolute(
            work_root
            or os.environ.get("BIGCHERRY_WORK_ROOT")
            or _local_env_table().get("BIGCHERRY_WORK_ROOT")
            or paths.primary_root() / "work"
        )
        upstream_was_explicit = upstream_repo is not None
        upstream = _absolute(
            upstream_repo or work / "upstream" / f"{Path(layout.vendor).name}.git"
        )
        # The default deliberately nests its bare upstream cache below the
        # work root.  The guard protects explicit topology configuration,
        # where aliasing a caller-owned checkout would let campaign writes
        # mutate it.
        if upstream_was_explicit and _paths_overlap(work, upstream):
            raise ValueError(
                "work_root and upstream_repo must be disjoint: "
                f"{work} vs {upstream}"
            )
        return cls(
            engine=engine,
            project_root=project,
            config_path=config,
            artifacts_root=artifacts,
            work_root=work,
            upstream_repo=upstream,
            overlay_root=layout.overlay_root(project),
            patches_root=layout.patches_root(project),
        )
