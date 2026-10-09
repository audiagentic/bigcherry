"""Engine pillars: which engines the platform carries and where each one's files live.

Every engine has one declaration, ``engines/<name>/engine.toml``: its upstream and its layout (patch packages,
overlay files added to the upstream tree, and the vendor checkout). Code that needs one of those locations asks
here, naming the engine, instead of spelling a directory. Moving an engine's files is then an edit to its
declaration plus the move itself.

This module has no dependency on the rest of the package so that ``core.paths`` can build on it.
"""

from __future__ import annotations

import re
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
class DraftStats:
    """Where an engine reports how many tokens its drafter proposed and how many were accepted.

    ``source`` is "log" (a regular expression over the server log with named groups ``drafted`` and ``accepted``; the
    last match is the running total) or "metrics" (two counter names in the Prometheus text served at ``path``).
    """

    source: str
    pattern: str = ""
    path: str = ""
    drafted: str = ""
    accepted: str = ""


@dataclass(frozen=True)
class ServeSpec:
    """How an engine's server is launched, health-checked and stopped."""

    binary: str                      # relative to a build tree, e.g. bin/llama-server
    model_flag: str
    host_flag: str
    port_flag: str
    health: str                      # GET path that answers 200 when the server is ready
    shutdown: str                    # "sigint", or "http:<POST path>"
    env: tuple[tuple[str, str], ...] = ()               # fixed environment for the server process
    env_from_binary: tuple[tuple[str, str], ...] = ()   # variables set to a path relative to the binary's directory
    draft_stats: DraftStats | None = None

    @property
    def shutdown_method(self) -> str:
        return "sigint" if self.shutdown == "sigint" else "http"

    @property
    def shutdown_path(self) -> str:
        return self.shutdown.partition(":")[2] if self.shutdown.startswith("http:") else ""


@dataclass(frozen=True)
class EngineLayout:
    """One engine's declaration. Paths are repository-relative, in POSIX form."""

    name: str
    upstream: str
    patches: str
    overlay: str
    vendor: str
    serve: ServeSpec

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


def _text(table: dict, key: str, where: str, name: str) -> str:
    value = table.get(key)
    if not isinstance(value, str) or not value:
        raise EngineError(f"engine {name!r}: {where}.{key} must be a non-empty string")
    return value


def _pairs(table: dict, key: str, where: str, name: str) -> tuple[tuple[str, str], ...]:
    value = table.get(key, {})
    if not isinstance(value, dict) or not all(isinstance(k, str) and isinstance(v, str) for k, v in value.items()):
        raise EngineError(f"engine {name!r}: {where}.{key} must be a table of strings")
    return tuple(sorted(value.items()))


def _draft_stats(table: object, name: str) -> DraftStats | None:
    if table is None:
        return None
    where = "serve.draft-stats"
    if not isinstance(table, dict):
        raise EngineError(f"engine {name!r}: [{where}] must be a table")
    source = _text(table, "source", where, name)
    if source == "log":
        unknown = sorted(set(table) - {"source", "pattern"})
        pattern = _text(table, "pattern", where, name)
        try:
            groups = re.compile(pattern).groupindex
        except re.error as exc:
            raise EngineError(f"engine {name!r}: {where}.pattern is not a valid regular expression: {exc}") from exc
        if not {"drafted", "accepted"} <= set(groups):
            raise EngineError(f"engine {name!r}: {where}.pattern needs named groups 'drafted' and 'accepted'")
        stats = DraftStats(source=source, pattern=pattern)
    elif source == "metrics":
        unknown = sorted(set(table) - {"source", "path", "drafted", "accepted"})
        stats = DraftStats(source=source, path=_text(table, "path", where, name),
                           drafted=_text(table, "drafted", where, name), accepted=_text(table, "accepted", where, name))
    else:
        raise EngineError(f"engine {name!r}: {where}.source must be 'log' or 'metrics', got {source!r}")
    if unknown:
        raise EngineError(f"engine {name!r}: unknown {where} key(s): {', '.join(unknown)}")
    return stats


def _serve(table: object, name: str) -> ServeSpec:
    if not isinstance(table, dict):
        raise EngineError(f"engine {name!r}: missing [serve] table")
    known = {"binary", "model-flag", "host-flag", "port-flag", "health", "shutdown", "env", "env-from-binary", "draft-stats"}
    unknown = sorted(set(table) - known)
    if unknown:
        raise EngineError(f"engine {name!r}: unknown serve key(s): {', '.join(unknown)}")
    shutdown = _text(table, "shutdown", "serve", name)
    if shutdown != "sigint" and not (shutdown.startswith("http:/") and len(shutdown) > len("http:/")):
        raise EngineError(f"engine {name!r}: serve.shutdown must be 'sigint' or 'http:/<path>', got {shutdown!r}")
    health = _text(table, "health", "serve", name)
    if not health.startswith("/"):
        raise EngineError(f"engine {name!r}: serve.health must be a path starting with '/', got {health!r}")
    return ServeSpec(
        binary=_relative(table.get("binary"), "binary", name),
        model_flag=_text(table, "model-flag", "serve", name),
        host_flag=_text(table, "host-flag", "serve", name),
        port_flag=_text(table, "port-flag", "serve", name),
        health=health,
        shutdown=shutdown,
        env=_pairs(table, "env", "serve", name),
        env_from_binary=_pairs(table, "env-from-binary", "serve", name),
        draft_stats=_draft_stats(table.get("draft-stats"), name),
    )


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
        serve=_serve(data.get("serve"), name),
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
