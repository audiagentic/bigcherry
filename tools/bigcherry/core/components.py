"""Engine components: BigCherry-owned source built against an engine's pinned source and installed into its runtime.

A component is declared by ``component.toml`` in a directory the engine lists under ``[components]`` in its
``engine.toml`` (RAD01 PR 5; docs/reference/engines/RADIANCE_PLATFORM_DESIGN.md decision 2). It is not a patch: it
never modifies the vendor tree. Its runtime settings (the environment variables it reads) are declared with a
default and a status, so profiles and experiments can be checked against them instead of free-form strings.
"""

from __future__ import annotations

import hashlib
import tomllib
from dataclasses import dataclass
from pathlib import Path

from . import engines

SCHEMA = 1
DECLARATION = "component.toml"
STATUSES = ("production", "experimental", "evaluated", "diagnostic", "tuning")


class ComponentError(ValueError):
    """A component declaration is missing or malformed."""


@dataclass(frozen=True)
class Setting:
    name: str
    default: str
    status: str
    summary: str
    #: a regular expression the server log matches when the setting's path ran; empty when there is none
    activation: str = ""


@dataclass(frozen=True)
class Component:
    name: str
    engine: str
    root: Path
    targets: tuple[str, ...]
    built_against: str
    outputs: tuple[str, ...]
    settings: dict[str, Setting]
    #: CMake cache values for the component's configure; {engine_source} and {targets} are substituted
    configure: tuple[tuple[str, str], ...] = ()

    def content_digest(self) -> str:
        """sha256 over every file of the component (path and bytes), the identity a build binds."""
        digest = hashlib.sha256()
        for path in sorted(p for p in self.root.rglob("*") if p.is_file() and "__pycache__" not in p.parts):
            digest.update(path.relative_to(self.root).as_posix().encode("utf-8") + b"\0")
            digest.update(path.read_bytes() + b"\0")
        return digest.hexdigest()


def _strings(value: object, where: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not value or not all(isinstance(v, str) and v for v in value):
        raise ComponentError(f"{where} must be a non-empty list of strings")
    return tuple(value)


def load(root: Path, engine: str) -> Component:
    """Load and validate ``<root>/component.toml`` for ``engine``. Fails closed."""
    path = root / DECLARATION
    try:
        data = tomllib.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ComponentError(f"no component declaration at {path}") from exc
    except tomllib.TOMLDecodeError as exc:
        raise ComponentError(f"{path} is not valid TOML: {exc}") from exc
    if data.get("schema") != SCHEMA:
        raise ComponentError(f"{path}: unsupported schema {data.get('schema')!r} (expected {SCHEMA})")
    name = data.get("name")
    if name != root.name:
        raise ComponentError(f"{path}: name {name!r} must match its directory {root.name!r}")
    if data.get("engine") != engine:
        raise ComponentError(f"{path}: engine {data.get('engine')!r} is not {engine!r}, the engine listing it")
    unknown = sorted(set(data) - {"schema", "name", "engine", "targets", "built-against", "outputs", "setting", "build"})
    if unknown:
        raise ComponentError(f"{path}: unknown key(s): {', '.join(unknown)}")
    built_against = data.get("built-against")
    if not isinstance(built_against, str) or not built_against:
        raise ComponentError(f"{path}: built-against must name the engine revision the component is written against")
    settings: dict[str, Setting] = {}
    raw = data.get("setting", {})
    if not isinstance(raw, dict):
        raise ComponentError(f"{path}: [setting.*] must be tables")
    for key, body in raw.items():
        where = f"{path}: setting.{key}"
        if not isinstance(body, dict) or not {"default", "status", "summary"} <= set(body) or set(body) - {
            "default", "status", "summary", "activation"}:
            raise ComponentError(f"{where} must have default, status and summary, and may have activation")
        if not all(isinstance(body[k], str) for k in body):
            raise ComponentError(f"{where}: every field must be a string")
        if body["status"] not in STATUSES:
            raise ComponentError(f"{where}: status {body['status']!r} is not one of {', '.join(STATUSES)}")
        settings[key] = Setting(key, body["default"], body["status"], body["summary"], body.get("activation", ""))
    return Component(
        name=name,
        engine=engine,
        root=root,
        targets=_strings(data.get("targets"), f"{path}: targets"),
        built_against=built_against,
        outputs=_strings(data.get("outputs"), f"{path}: outputs"),
        settings=settings,
        configure=tuple(sorted((str(k), str(v)) for k, v in (data.get("build", {}).get("configure", {}) or {}).items())),
    )


def load_all(repo_root: Path, engine: str) -> dict[str, Component]:
    """Every component ``engine`` declares, by name."""
    layout = engines.load(repo_root, engine)
    return {name: load(repo_root / relative, engine) for name, relative in layout.components}


def check_environment(component: Component, environment: dict[str, str], *, production: bool) -> list[str]:
    """Problems with ``environment`` (variables a profile or experiment sets) against the component's settings:
    an undeclared variable, a value equal to the default (a profile restates nothing), and - for a production
    profile - a setting whose status is not production."""
    problems = []
    for key, value in sorted(environment.items()):
        setting = component.settings.get(key)
        if setting is None:
            if key.startswith("R3_"):
                problems.append(f"{key} is not a setting {component.name} declares")
            continue
        if value == setting.default:
            problems.append(f"{key}={value} restates {component.name}'s default")
        if production and setting.status != "production":
            problems.append(f"{key} is {setting.status} in {component.name}; a production profile may not set it")
    return problems
