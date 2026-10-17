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
from dataclasses import dataclass, field
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
class CpuSmokeDecl:
    """The tiny model and extra server arguments of an engine's RAM-only smoke test ([smoke.cpu], MEN15).

    ``model`` is relative to the model root (``--model-root``, ``BIGCHERRY_MODEL_ROOT`` or the host's model-root);
    ``sha256`` is the digest of that exact file. Until the owner pins a real model the digest is a PLACEHOLDER value,
    which the smoke runner refuses.
    """

    model: str
    sha256: str
    args: tuple[str, ...] = ()


@dataclass(frozen=True)
class AcceptanceEntry:
    """One post-bump hardware acceptance run of an engine ([[acceptance]], MEN17).

    ``config`` is the repository path of a server-comparison configuration (``ab-benchmark --server-config``);
    ``baselines`` are repository paths of baseline files (docs/reference/engines/<engine>/baselines/) holding the
    stored numbers its arms must hold. ``env`` is set for the run unless the caller already set the variable; the
    text ``{entry-dir}`` in a value is the entry's own output directory.
    """

    name: str
    config: str
    baselines: tuple[str, ...]
    rounds: int = 2
    settle_seconds: float = 20.0
    env: tuple[tuple[str, str], ...] = ()


@dataclass(frozen=True)
class EngineLayout:
    """One engine's declaration. Paths are repository-relative, in POSIX form."""

    name: str
    upstream: str
    patches: str
    #: what every patch ID of this engine starts with before its order digits ("" for llama.cpp, "rad" for radiance),
    #: so IDs never collide across engines
    patch_id_prefix: str
    overlay: str
    vendor: str
    recipes: str
    serve: ServeSpec
    #: the CMake cache variable a HIP build names its GPU targets in (llama.cpp AMDGPU_TARGETS, radiance RAD_GPU_TARGETS)
    targets_option: str
    #: the binary a lane builds and publishes unless the request names another
    build_binary: str
    #: CMake targets a served build also needs (plugins, runtime home); built after the binary
    runtime_targets: tuple[str, ...] = ()
    #: build-tree directory components install their outputs into; empty when the engine has none
    runtime_home: str = ""
    #: (name, repository-relative directory) of each component the engine builds, from [components]
    components: tuple[tuple[str, str], ...] = ()
    #: repository-relative registry of the external sources (forks, upstream PRs) this engine's patches are ported
    #: from; "" when the engine has none
    external_sources: str = ""
    #: where the major of a release version comes from: "build-tag" (the pin is a numbered upstream build, b<N>, and N
    #: is the major) or "pinned-version" (the recipes' pinned-version, 1.3.0, with its dots removed: 130)
    version_source: str = field(default="", kw_only=True)
    #: [smoke.cpu]: the RAM-only smoke test's model; None when the engine declares none
    smoke_cpu: CpuSmokeDecl | None = field(default=None, kw_only=True)
    #: [[acceptance]]: the post-bump hardware acceptance entries (MEN17); empty when the engine declares none
    acceptance: tuple[AcceptanceEntry, ...] = field(default=(), kw_only=True)

    @property
    def upstream_name(self) -> str:
        """What the upstream project is called: the last segment of its URL (llama.cpp, radiance)."""
        return self.upstream.rstrip("/").rsplit("/", 1)[-1]

    def external_sources_path(self, project_root: Path) -> Path | None:
        return project_root / self.external_sources if self.external_sources else None

    def recipes_path(self, project_root: Path) -> Path:
        return project_root / self.recipes

    def release_package(self) -> str:
        """The release-please package (a path in release-please-config.json) that carries this engine's release line."""
        return f"{ENGINES_DIR}/{self.name}"

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


def _smoke_cpu(table: object, name: str) -> CpuSmokeDecl | None:
    if table is None:
        return None
    if not isinstance(table, dict) or set(table) - {"cpu"}:
        raise EngineError(f"engine {name!r}: [smoke] may only hold a [smoke.cpu] table")
    cpu = table["cpu"]
    where = "smoke.cpu"
    if not isinstance(cpu, dict):
        raise EngineError(f"engine {name!r}: [{where}] must be a table")
    unknown = sorted(set(cpu) - {"model", "sha256", "args"})
    if unknown:
        raise EngineError(f"engine {name!r}: unknown {where} key(s): {', '.join(unknown)}")
    model = _relative(_text(cpu, "model", where, name), f"{where}.model", name)
    args = cpu.get("args", [])
    if not isinstance(args, list) or not all(isinstance(a, str) for a in args):
        raise EngineError(f"engine {name!r}: {where}.args must be a list of strings")
    return CpuSmokeDecl(model=model, sha256=_text(cpu, "sha256", where, name), args=tuple(args))


def _acceptance(value: object, name: str) -> tuple[AcceptanceEntry, ...]:
    if value is None:
        return ()
    if not isinstance(value, list) or not all(isinstance(item, dict) for item in value):
        raise EngineError(f"engine {name!r}: [[acceptance]] must be an array of tables")
    entries: list[AcceptanceEntry] = []
    for item in value:
        where = "acceptance"
        unknown = sorted(set(item) - {"name", "config", "baselines", "rounds", "settle-seconds", "env"})
        if unknown:
            raise EngineError(f"engine {name!r}: unknown {where} key(s): {', '.join(unknown)}")
        entry_name = _text(item, "name", where, name)
        if not re.fullmatch(r"[A-Za-z0-9_-]+", entry_name):
            raise EngineError(f"engine {name!r}: acceptance name {entry_name!r} must be a simple identifier")
        baselines = item.get("baselines")
        if not isinstance(baselines, list) or not baselines or not all(isinstance(b, str) for b in baselines):
            raise EngineError(f"engine {name!r}: acceptance {entry_name!r}: baselines must be a non-empty list of paths")
        rounds = item.get("rounds", 2)
        if not isinstance(rounds, int) or isinstance(rounds, bool) or rounds < 1:
            raise EngineError(f"engine {name!r}: acceptance {entry_name!r}: rounds must be a positive integer")
        settle = item.get("settle-seconds", 20.0)
        if not isinstance(settle, (int, float)) or isinstance(settle, bool) or settle < 0:
            raise EngineError(f"engine {name!r}: acceptance {entry_name!r}: settle-seconds must be non-negative")
        entries.append(AcceptanceEntry(
            name=entry_name,
            config=_relative(item.get("config"), f"acceptance.{entry_name}.config", name),
            baselines=tuple(_relative(b, f"acceptance.{entry_name}.baselines", name) for b in baselines),
            rounds=rounds,
            settle_seconds=float(settle),
            env=_pairs(item, "env", f"acceptance.{entry_name}", name),
        ))
    if len({e.name for e in entries}) != len(entries):
        raise EngineError(f"engine {name!r}: acceptance entry names must be unique")
    return tuple(entries)


def _build_table(table: object, name: str) -> dict[str, object]:
    if not isinstance(table, dict):
        raise EngineError(f"engine {name!r}: missing [build] table")
    unknown = sorted(set(table) - {"targets-option", "binary", "runtime-targets", "runtime-home"})
    if unknown:
        raise EngineError(f"engine {name!r}: unknown build key(s): {', '.join(unknown)}")
    return {
        "targets_option": _text(table, "targets-option", "build", name),
        "build_binary": _relative(table.get("binary"), "build.binary", name),
        "runtime_targets": tuple(table.get("runtime-targets", ())),
        "runtime_home": _relative(table["runtime-home"], "build.runtime-home", name) if "runtime-home" in table else "",
    }


def _components(table: object, name: str) -> tuple[tuple[str, str], ...]:
    if not isinstance(table, dict):
        raise EngineError(f"engine {name!r}: [components] must be a table of name = directory")
    return tuple(sorted((key, _relative(value, f"components.{key}", name)) for key, value in table.items()))


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
    unknown = sorted(set(layout) - {"patches", "patch-id-prefix", "overlay", "vendor", "recipes", "external-sources"})
    if unknown:
        raise EngineError(f"engine {name!r}: unknown layout key(s): {', '.join(unknown)}")
    return EngineLayout(
        name=name,
        upstream=upstream,
        patches=_relative(layout.get("patches"), "patches", name),
        patch_id_prefix=_patch_id_prefix(layout.get("patch-id-prefix"), name),
        overlay=_relative(layout.get("overlay"), "overlay", name),
        vendor=_relative(layout.get("vendor"), "vendor", name),
        recipes=_relative(layout.get("recipes"), "recipes", name),
        serve=_serve(data.get("serve"), name),
        **_build_table(data.get("build"), name),
        components=_components(data.get("components", {}), name),
        external_sources=(
            _relative(layout["external-sources"], "external-sources", name) if "external-sources" in layout else ""
        ),
        version_source=_version_source(data.get("release"), name),
        smoke_cpu=_smoke_cpu(data.get("smoke"), name),
        acceptance=_acceptance(data.get("acceptance"), name),
    )


VERSION_SOURCES = ("build-tag", "pinned-version")


def _version_source(value: object, name: str) -> str:
    source = value.get("version-source") if isinstance(value, dict) else None
    if source not in VERSION_SOURCES:
        raise EngineError(f"engine {name!r}: [release] version-source must be one of {', '.join(VERSION_SOURCES)}")
    return source


def _patch_id_prefix(value: object, name: str) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"[a-z]*", value):
        raise EngineError(f"engine {name!r}: layout.patch-id-prefix must be a lowercase string (\"\" for none)")
    return value


def names(repo_root: Path) -> tuple[str, ...]:
    """Every declared engine, sorted."""
    base = repo_root / ENGINES_DIR
    if not base.is_dir():
        return ()
    return tuple(sorted(entry.name for entry in base.iterdir() if (entry / DECLARATION).is_file()))


def load_all(repo_root: Path) -> dict[str, EngineLayout]:
    """Every declared engine. No two engines may share a patches, overlay or vendor location."""
    layouts = {name: load(repo_root, name) for name in names(repo_root)}
    for field in ("patches", "overlay", "vendor", "recipes"):
        seen: dict[str, str] = {}
        for name, layout in layouts.items():
            location = getattr(layout, field)
            if location in seen:
                raise EngineError(f"engines {seen[location]!r} and {name!r} both declare layout.{field} = {location!r}")
            seen[location] = name
    return layouts
