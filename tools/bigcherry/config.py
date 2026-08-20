"""Strict version-2 campaign configuration models.

This parser is intentionally separate from ``recipes.py``.  Existing v1
recipes remain the compatibility input until their conversion is reviewed;
they must never be silently interpreted as v2.
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path


class ConfigError(ValueError):
    pass


def _table(raw: object, where: str) -> dict[str, object]:
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise ConfigError(f"{where} must be a table")
    return raw


def _strings(raw: object, where: str) -> tuple[str, ...]:
    if raw is None:
        return ()
    if not isinstance(raw, list) or not all(isinstance(v, str) and v for v in raw):
        raise ConfigError(f"{where} must be a list of non-empty strings")
    if len(set(raw)) != len(raw):
        raise ConfigError(f"{where} contains duplicates")
    return tuple(raw)


def _options(raw: object, where: str) -> tuple[tuple[str, str], ...]:
    table = _table(raw, where)
    result: list[tuple[str, str]] = []
    for key, value in table.items():
        if not isinstance(key, str) or not key:
            raise ConfigError(f"{where} contains an invalid option name")
        if isinstance(value, bool):
            raise ConfigError(
                f"{where}.{key} must use \"ON\"/\"OFF\", not a TOML boolean"
            )
        if not isinstance(value, (str, int, float)):
            raise ConfigError(f"{where}.{key} must be a scalar")
        result.append((key, str(value)))
    return tuple(sorted(result))


@dataclass(frozen=True)
class PatchSet:
    name: str
    patches: tuple[str, ...]
    required_state: str


#: RE30: which cmake_configure_args backend adapter a source's lanes use.
#: "hip" is the default so every source predating RE30 phase 1 is unchanged.
BACKENDS: tuple[str, ...] = ("hip", "vulkan")


@dataclass(frozen=True)
class Source:
    name: str
    ref: str
    overlay: bool
    patch_sets: tuple[str, ...]
    backend: str = "hip"


@dataclass(frozen=True)
class Build:
    name: str
    options: tuple[tuple[str, str], ...]
    variant_set: str | None
    needs: frozenset[str]


@dataclass(frozen=True)
class Platform:
    name: str
    targets: tuple[str, ...]
    options: tuple[tuple[str, str], ...]
    c_compiler: str | None = None
    cxx_compiler: str | None = None


@dataclass(frozen=True)
class Experiment:
    name: str
    patches: tuple[str, ...]
    cmake_options: tuple[tuple[str, str], ...]
    runtime_env: tuple[tuple[str, str], ...]
    requires: tuple[str, ...]
    conflicts: tuple[str, ...]


@dataclass(frozen=True)
class CampaignLaneSelector:
    """One (source, build, platform) selection within a named campaign
    profile -- RE19's replacement for legacy's default=true recipe
    aggregation. Deliberately just an identity triple, not a full
    CampaignLane (that's RE18's planning object, produced by expanding
    this selector against real config).
    """

    source: str
    build: str
    platform: str


@dataclass(frozen=True)
class CampaignProfile:
    name: str
    lanes: tuple[CampaignLaneSelector, ...]


@dataclass(frozen=True)
class Config:
    pinned: str
    patch_sets: dict[str, PatchSet]
    sources: dict[str, Source]
    builds: dict[str, Build]
    platforms: dict[str, Platform]
    experiments: dict[str, Experiment]
    campaigns: dict[str, CampaignProfile]
    path: Path


def load(path: str | Path) -> Config:
    """Load only an explicit ``version = 2`` document."""
    path = Path(path)
    try:
        raw = tomllib.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise ConfigError(f"no config file at {path}") from None
    except tomllib.TOMLDecodeError as exc:
        raise ConfigError(f"{path}: {exc}") from None

    if raw.get("version") != 2:
        raise ConfigError(
            f"{path}: expected explicit version = 2; v1 requires conversion"
        )
    unknown_top_level = sorted(
        set(raw) - {"version", "pinned", "patch-set", "source", "build", "platform",
                    "experiment", "compat", "campaign"}
    )
    if unknown_top_level:
        raise ConfigError(
            f"{path}: unknown top-level field(s): {', '.join(unknown_top_level)}"
        )
    pinned = raw.get("pinned")
    if not isinstance(pinned, str) or not pinned:
        raise ConfigError(f"{path}: pinned must be a non-empty string")

    patch_sets: dict[str, PatchSet] = {}
    for name, body in _table(raw.get("patch-set"), "patch-set").items():
        data = _table(body, f"patch-set.{name}")
        state = data.get("required-state", "validated")
        if not isinstance(state, str) or not state:
            raise ConfigError(f"patch-set.{name}.required-state must be a string")
        patch_sets[name] = PatchSet(
            name=name,
            patches=_strings(data.get("patches"), f"patch-set.{name}.patches"),
            required_state=state,
        )

    sources: dict[str, Source] = {}
    for name, body in _table(raw.get("source"), "source").items():
        data = _table(body, f"source.{name}")
        ref = data.get("ref", "pinned")
        if not isinstance(ref, str) or not ref:
            raise ConfigError(f"source.{name}.ref must be a non-empty string")
        overlay = data.get("overlay")
        if not isinstance(overlay, bool):
            raise ConfigError(f"source.{name}.overlay must be true or false")
        patch_refs = _strings(data.get("patch-sets"), f"source.{name}.patch-sets")
        unknown = sorted(set(patch_refs) - set(patch_sets))
        if unknown:
            raise ConfigError(
                f"source.{name} references unknown patch set(s): {', '.join(unknown)}"
            )
        if "groups" in data or "states" in data:
            raise ConfigError(
                f"source.{name} must use exact patch-sets, not groups/states selectors"
            )
        backend = data.get("backend", "hip")
        if backend not in BACKENDS:
            raise ConfigError(
                f"source.{name}.backend={backend!r} must be one of "
                f"{', '.join(BACKENDS)}"
            )
        sources[name] = Source(name, ref, overlay, patch_refs, backend)

    builds: dict[str, Build] = {}
    for name, body in _table(raw.get("build"), "build").items():
        data = _table(body, f"build.{name}")
        needs = _strings(data.get("needs"), f"build.{name}.needs")
        builds[name] = Build(
            name=name,
            options=_options(data.get("options"), f"build.{name}.options"),
            variant_set=data.get("variant-set"),
            needs=frozenset(needs),
        )

    platforms: dict[str, Platform] = {}
    for name, body in _table(raw.get("platform"), "platform").items():
        data = _table(body, f"platform.{name}")
        c_compiler = data.get("c-compiler")
        if c_compiler is not None and not isinstance(c_compiler, str):
            raise ConfigError(f"platform.{name}.c-compiler must be a string")
        cxx_compiler = data.get("cxx-compiler")
        if cxx_compiler is not None and not isinstance(cxx_compiler, str):
            raise ConfigError(f"platform.{name}.cxx-compiler must be a string")
        platforms[name] = Platform(
            name=name,
            targets=_strings(data.get("targets"), f"platform.{name}.targets"),
            options=_options(data.get("options"), f"platform.{name}.options"),
            c_compiler=c_compiler,
            cxx_compiler=cxx_compiler,
        )

    experiments: dict[str, Experiment] = {}
    for name, body in _table(raw.get("experiment"), "experiment").items():
        data = _table(body, f"experiment.{name}")
        experiments[name] = Experiment(
            name=name,
            patches=_strings(data.get("patches"), f"experiment.{name}.patches"),
            cmake_options=_options(data.get("cmake-options"), f"experiment.{name}.cmake-options"),
            runtime_env=_options(data.get("runtime-env"), f"experiment.{name}.runtime-env"),
            requires=_strings(data.get("requires"), f"experiment.{name}.requires"),
            conflicts=_strings(data.get("conflicts"), f"experiment.{name}.conflicts"),
        )
    campaigns: dict[str, CampaignProfile] = {}
    for name, body in _table(raw.get("campaign"), "campaign").items():
        data = _table(body, f"campaign.{name}")
        raw_lanes = data.get("lanes")
        if not isinstance(raw_lanes, list) or not raw_lanes:
            raise ConfigError(f"campaign.{name}.lanes must be a non-empty list")
        lanes: list[CampaignLaneSelector] = []
        for index, raw_lane in enumerate(raw_lanes):
            lane_data = _table(raw_lane, f"campaign.{name}.lanes[{index}]")
            lane_source = lane_data.get("source")
            lane_build = lane_data.get("build")
            lane_platform = lane_data.get("platform")
            if not all(isinstance(v, str) and v for v in (lane_source, lane_build, lane_platform)):
                raise ConfigError(
                    f"campaign.{name}.lanes[{index}] must set non-empty "
                    f"source/build/platform strings"
                )
            if lane_source not in sources:
                raise ConfigError(
                    f"campaign.{name}.lanes[{index}] references unknown source {lane_source!r}"
                )
            if lane_build not in builds:
                raise ConfigError(
                    f"campaign.{name}.lanes[{index}] references unknown build {lane_build!r}"
                )
            if lane_platform not in platforms:
                raise ConfigError(
                    f"campaign.{name}.lanes[{index}] references unknown platform {lane_platform!r}"
                )
            lanes.append(CampaignLaneSelector(
                source=lane_source, build=lane_build, platform=lane_platform))
        campaigns[name] = CampaignProfile(name=name, lanes=tuple(lanes))

    return Config(pinned, patch_sets, sources, builds, platforms, experiments, campaigns, path)
