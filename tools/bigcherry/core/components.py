"""Engine components: BigCherry-owned source built against an engine's pinned source and installed into its runtime.

A component is declared by ``component.toml`` in a directory the engine lists under ``[components]`` in its
``engine.toml`` (RAD01 PR 5; docs/reference/engines/radiance/PLATFORM_DESIGN.md decision 2). It is not a patch: it
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
STATUSES = ("production", "validated", "experimental", "evaluated", "diagnostic", "tuning")
#: statuses that claim the setting is fit to be on by default; a non-off default in one of them needs evidence
PROMOTED = ("production", "validated")
#: default values that mean the setting is off
OFF_VALUES = ("", "0", "off", "false")


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
    #: repo-relative run.json captures (ab-benchmark --server-config) that prove the setting; see promotion_problems
    evidence: tuple[str, ...] = ()


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
            "default", "status", "summary", "activation", "evidence"}:
            raise ComponentError(f"{where} must have default, status and summary, and may have activation and evidence")
        evidence = body.get("evidence", [])
        if not isinstance(evidence, list) or not all(isinstance(e, str) and e for e in evidence):
            raise ComponentError(f"{where}: evidence must be a list of run.json paths")
        if not all(isinstance(v, str) for k, v in body.items() if k != "evidence"):
            raise ComponentError(f"{where}: every field but evidence must be a string")
        if body["status"] not in STATUSES:
            raise ComponentError(f"{where}: status {body['status']!r} is not one of {', '.join(STATUSES)}")
        settings[key] = Setting(key, body["default"], body["status"], body["summary"], body.get("activation", ""), tuple(evidence))
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
        if production and setting.status not in PROMOTED:
            problems.append(f"{key} is {setting.status} in {component.name}; a production profile may not set it")
    return problems


def _effective(arm: dict, name: str, default: str) -> str:
    return arm.get("settings", {}).get(name, default)


def _evidence_problem(setting: Setting, run: dict) -> str | None:
    """None when this run.json varies ``setting`` alone between two arms, the arm at the setting's default carries an
    ``executed`` activation record against the other, and the capture is a clean, completed run."""
    arms = run.get("arm_variables")
    if not isinstance(arms, dict) or len(arms) < 2:
        return "no arm_variables with two or more arms"
    activation = run.get("setting_activation", {})
    for subject, variables in arms.items():
        if _effective(variables, setting.name, setting.default) != setting.default:
            continue
        for control, other in arms.items():
            if control == subject or _effective(other, setting.name, setting.default) == setting.default:
                continue
            others = {k for k in set(variables.get("settings", {})) | set(other.get("settings", {}))
                      if _effective(variables, k, "") != _effective(other, k, "")}
            if others != {setting.name} or variables.get("environment") != other.get("environment"):
                continue
            record = activation.get(subject, {}).get(setting.name)
            if not isinstance(record, dict) or record.get("status") != "executed":
                return f"arm {subject} (setting at its default) has no 'executed' activation record for {setting.name}"
            if any(r.get("returncode") != 0 for r in run.get("runs", []) if r.get("mode") in (subject, control)) or (
                    run.get("quality") is not None and not run["quality"].get("passed", False)):
                return "the capture has a failed cell or a failed quality gate"
            return None
    return f"no two arms differ in {setting.name} alone with one at the default {setting.default!r}"


def promotion_problems(component: Component, repo_root: Path) -> tuple[list[str], list[str]]:
    """(problems, warnings) for the settings' promotion lifecycle (MEN11), the same lightweight tier as a patch.

    A setting whose status is ``validated`` must name evidence; so must one that is ``production`` with a default other
    than off/0 (a ``production`` setting without evidence is a warning: it predates the lifecycle and nothing
    demotes it silently). Each evidence path is a ``run.json`` of an ab-benchmark capture that exists, has the
    setting as its only varied variable between two arms, shows an ``executed`` activation record for it, and has no
    failed cell. Balanced separation and greedy identity/quality are read from that capture, not re-checked here.
    ``experimental``, ``evaluated``, ``diagnostic`` and ``tuning`` settings are not judged. Evidence on a setting
    that is not promoted is still checked when present.
    """
    import json

    problems: list[str] = []
    warnings: list[str] = []
    for setting in component.settings.values():
        where = f"{component.name}:{setting.name}"
        on_by_default = setting.default.lower() not in OFF_VALUES
        needs = setting.status == "validated" or (setting.status == "production" and on_by_default)
        if not setting.evidence:
            if needs:
                (problems if setting.status == "validated" else warnings).append(
                    f"{where} is {setting.status} with default {setting.default!r} and names no evidence")
            continue
        for relative in setting.evidence:
            path = repo_root / relative
            try:
                run = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError) as exc:
                problems.append(f"{where}: evidence {relative} is not a readable run.json ({exc})")
                continue
            reason = _evidence_problem(setting, run)
            if reason:
                problems.append(f"{where}: evidence {relative}: {reason}")
    return problems, warnings
