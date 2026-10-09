"""Engine pillars: which engines the platform carries and where each one's files live.

Every engine has one declaration, ``engines/<name>/engine.toml``: its upstream and its layout (patch packages,
overlay files added to the upstream tree, and the vendor checkout). Code that needs one of those locations asks
here, naming the engine, instead of spelling a directory. Moving an engine's files is then an edit to its
declaration plus the move itself.

This module has no dependency on the rest of the package so that ``core.paths`` can build on it.
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

LLAMACPP = "llamacpp"
RADIANCE = "radiance"

ENGINES_DIR = "engines"
DECLARATION = "engine.toml"
SCHEMA = 1


class EngineError(ValueError):
    """An engine declaration is missing or malformed."""


@dataclass(frozen=True)
class EngineLayout:
    """One engine's declaration. Paths are repository-relative, in POSIX form."""

    name: str
    upstream: str
    patches: str
    overlay: str
    vendor: str

    def patches_root(self, project_root: Path) -> Path:
        return project_root / self.patches

    def overlay_root(self, project_root: Path) -> Path:
        return project_root / self.overlay

    def vendor_root(self, primary_root: Path) -> Path:
        """The vendor checkout; it belongs to the primary checkout, never to a slice worktree."""
        return primary_root / self.vendor

    def patch_path_prefix(self) -> str:
        """Prefix of a repository-relative patch path, e.g. for filtering ``git diff --name-only`` output."""
        return self.patches + "/"


def _relative(value: object, field: str, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise EngineError(f"engine {name!r}: layout.{field} must be a non-empty string")
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or "\\" in value:
        raise EngineError(f"engine {name!r}: layout.{field} must be a repository-relative POSIX path, got {value!r}")
    return str(path)


def declaration_path(repo_root: Path, name: str) -> Path:
    return repo_root / ENGINES_DIR / name / DECLARATION


def load(repo_root: Path, name: str) -> EngineLayout:
    """Load ``engines/<name>/engine.toml`` from the repository at ``repo_root``. Fails closed."""
    path = declaration_path(repo_root, name)
    try:
        data = tomllib.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise EngineError(f"engine {name!r} is not declared: {path} does not exist") from exc
    except tomllib.TOMLDecodeError as exc:
        raise EngineError(f"engine {name!r}: {path} is not valid TOML: {exc}") from exc
    if data.get("schema") != SCHEMA:
        raise EngineError(f"engine {name!r}: unsupported schema {data.get('schema')!r} (expected {SCHEMA})")
    if data.get("name") != name:
        raise EngineError(f"engine {name!r}: declaration names {data.get('name')!r}; it must match its directory")
    upstream = data.get("upstream")
    if not isinstance(upstream, str) or not upstream:
        raise EngineError(f"engine {name!r}: upstream must be a non-empty string")
    layout = data.get("layout")
    if not isinstance(layout, dict):
        raise EngineError(f"engine {name!r}: missing [layout] table")
    unknown = sorted(set(layout) - {"patches", "overlay", "vendor"})
    if unknown:
        raise EngineError(f"engine {name!r}: unknown layout key(s): {', '.join(unknown)}")
    return EngineLayout(
        name=name,
        upstream=upstream,
        patches=_relative(layout.get("patches"), "patches", name),
        overlay=_relative(layout.get("overlay"), "overlay", name),
        vendor=_relative(layout.get("vendor"), "vendor", name),
    )


def names(repo_root: Path) -> tuple[str, ...]:
    """Every declared engine, sorted."""
    base = repo_root / ENGINES_DIR
    if not base.is_dir():
        return ()
    return tuple(sorted(entry.name for entry in base.iterdir() if (entry / DECLARATION).is_file()))


def load_all(repo_root: Path) -> dict[str, EngineLayout]:
    """Every declared engine. No two engines may share a patches, overlay or vendor location."""
    layouts = {name: load(repo_root, name) for name in names(repo_root)}
    for field in ("patches", "overlay", "vendor"):
        seen: dict[str, str] = {}
        for name, layout in layouts.items():
            location = getattr(layout, field)
            if location in seen:
                raise EngineError(f"engines {seen[location]!r} and {name!r} both declare layout.{field} = {location!r}")
            seen[location] = name
    return layouts
