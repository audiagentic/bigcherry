"""Explicit project and host-local campaign paths.

The campaign migration must not hide its worktree and artifact roots in
module-location globals.  This small immutable context keeps those roots
explicit while ``paths.py`` remains available to legacy commands.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from . import paths


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


@dataclass(frozen=True)
class ProjectContext:
    project_root: Path
    config_path: Path
    artifacts_root: Path
    work_root: Path
    upstream_repo: Path
    overlay_root: Path
    patches_root: Path

    @classmethod
    def resolve(
        cls,
        *,
        project_root: str | os.PathLike[str] | None = None,
        config_path: str | os.PathLike[str] | None = None,
        artifacts_root: str | os.PathLike[str] | None = None,
        work_root: str | os.PathLike[str] | None = None,
        upstream_repo: str | os.PathLike[str] | None = None,
    ) -> "ProjectContext":
        """Resolve explicit arguments, then environment, then project defaults."""
        project = _absolute(
            project_root
            or os.environ.get("BIGCHERRY_PROJECT_ROOT")
            or paths.REPO_ROOT
        )
        config = _absolute(
            config_path
            or os.environ.get("BIGCHERRY_CONFIG_PATH")
            or project / "config" / "recipes.toml"
        )
        artifacts = _absolute(
            artifacts_root
            or os.environ.get("BIGCHERRY_ARTIFACT_ROOT")
            or project / "artifacts"
        )
        # Tool output never defaults into the user profile (LOCALAPPDATA,
        # ~/.cache, ~): the gitignored project-local work/ is the default so
        # every run's output stays inside the project folder. An explicit
        # argument or BIGCHERRY_WORK_ROOT may still point elsewhere (e.g. a
        # large scratch volume on a campaign host).
        work = _absolute(
            work_root
            or os.environ.get("BIGCHERRY_WORK_ROOT")
            or project / "work"
        )
        upstream_was_explicit = upstream_repo is not None
        upstream = _absolute(
            upstream_repo or work / "upstream" / "llama.cpp.git"
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
            project_root=project,
            config_path=config,
            artifacts_root=artifacts,
            work_root=work,
            upstream_repo=upstream,
            overlay_root=project / "src",
            patches_root=project / "patches",
        )
