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


def _required_string(data: dict[str, object], key: str, where: str) -> str:
    value = data.get(key)
    if not isinstance(value, str) or not value:
        raise ConfigError(f"{where}.{key} must be a non-empty string")
    return value


def _strings(raw: object, where: str) -> tuple[str, ...]:
    if raw is None:
        return ()
    if not isinstance(raw, list) or not all(isinstance(v, str) and v for v in raw):
        raise ConfigError(f"{where} must be a list of non-empty strings")
    if len(set(raw)) != len(raw):
        raise ConfigError(f"{where} contains duplicates")
    return tuple(raw)


def _argv(raw: object, where: str) -> tuple[str, ...]:
    """An ordered list of strings where duplicates are expected and
    meaningful (e.g. a CLI argv: repeated flag VALUES like "q8_0" appearing
    twice for two different flags are normal), unlike ``_strings()``'s set
    semantics (patch/architecture lists, where a duplicate is a mistake)."""
    if raw is None:
        return ()
    if not isinstance(raw, list) or not all(isinstance(v, str) and v for v in raw):
        raise ConfigError(f"{where} must be a list of non-empty strings")
    return tuple(raw)


def _options(raw: object, where: str) -> tuple[tuple[str, str], ...]:
    table = _table(raw, where)
    result: list[tuple[str, str]] = []
    for key, value in table.items():
        if not isinstance(key, str) or not key:
            raise ConfigError(f"{where} contains an invalid option name")
        if isinstance(value, bool):
            raise ConfigError(f'{where}.{key} must use "ON"/"OFF", not a TOML boolean')
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
class BackendStack:
    """Requested software stack profile, distinct from machine platform."""

    name: str
    backend: str
    sdk_root: str | None
    c_compiler: str | None
    cxx_compiler: str | None
    runtime_library_dirs: tuple[str, ...]
    environment: tuple[tuple[str, str], ...]
    required_providers: tuple[str, ...]


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
class RuntimeProfile:
    """HI130: server args + tuner context bundled together and named, so a
    tune-campaign run is reproducible from one --runtime-profile flag
    instead of an operator hand-assembling matching launch flags each time.

    ``tune_context`` is deliberately separate from ``production_context``:
    the tuner's per-candidate timing workspace needs VRAM headroom beyond
    just weights+KV-cache, so tune-mode must never inherit a model's own
    (possibly huge) max context by default -- that OOM'd a real single-GPU
    R9700 27B tune run this session. ``min_free_vram_bytes_per_device`` is a
    preflight threshold (see core/gpu.py), not a soft hint -- a request that
    fails it is rejected before launch, never silently downgraded.
    """

    name: str
    server_args: tuple[str, ...]
    tune_context: int
    production_context: int
    min_free_vram_bytes_per_device: int


@dataclass(frozen=True)
class Tree:
    """RE48: a known working tree of this repo, probed by pin-status.

    ``required`` trees are part of the ``--complete`` obligation (a
    required unreachable tree fails completion); ``role = "campaign"``
    trees additionally carry the expected-tooling-revision gate, because a
    campaign launched on stale tooling is a silent desync (the J: tree ran
    the 27B campaign four commits behind on 2026-08-21)."""

    name: str
    alias: str
    path: str
    required: bool
    role: str
    expected_tooling_revision: str


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
    # RE48: default () keeps every pre-RE48 `Config(...)` constructor
    # (which never knew `trees`) source-compatible.
    trees: tuple[Tree, ...] = ()
    # HI130: default {} keeps every pre-HI130 `Config(...)` constructor
    # (which never knew `runtime_profiles`) source-compatible.
    runtime_profiles: dict[str, RuntimeProfile] = None  # type: ignore[assignment]
    stacks: dict[str, BackendStack] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.runtime_profiles is None:
            object.__setattr__(self, "runtime_profiles", {})
        if self.stacks is None:
            object.__setattr__(self, "stacks", {})


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
        set(raw)
        - {
            "version",
            "pinned",
            "patch-set",
            "source",
            "build",
            "platform",
            "experiment",
            "compat",
            "campaign",
            "trees",
            "runtime-profile",
            "stack",
        }
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
        if not isinstance(backend, str) or backend not in BACKENDS:
            raise ConfigError(
                f"source.{name}.backend={backend!r} must be one of "
                f"{', '.join(BACKENDS)}"
            )
        sources[name] = Source(name, ref, overlay, patch_refs, backend)

    builds: dict[str, Build] = {}
    for name, body in _table(raw.get("build"), "build").items():
        data = _table(body, f"build.{name}")
        needs = _strings(data.get("needs"), f"build.{name}.needs")
        variant_set = data.get("variant-set")
        if variant_set is not None and not isinstance(variant_set, str):
            raise ConfigError(f"build.{name}.variant-set must be a string")
        builds[name] = Build(
            name=name,
            options=_options(data.get("options"), f"build.{name}.options"),
            variant_set=variant_set,
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

    stacks: dict[str, BackendStack] = {}
    for name, body in _table(raw.get("stack"), "stack").items():
        data = _table(body, f"stack.{name}")
        backend = data.get("backend")
        if not isinstance(backend, str) or backend not in BACKENDS:
            raise ConfigError(
                f"stack.{name}.backend={backend!r} must be one of "
                f"{', '.join(BACKENDS)}"
            )
        sdk_root = data.get("sdk-root")
        if sdk_root is not None and (not isinstance(sdk_root, str) or not sdk_root):
            raise ConfigError(f"stack.{name}.sdk-root must be a non-empty string")
        c_compiler = data.get("c-compiler")
        if c_compiler is not None and (not isinstance(c_compiler, str) or not c_compiler):
            raise ConfigError(f"stack.{name}.c-compiler must be a non-empty string")
        cxx_compiler = data.get("cxx-compiler")
        if cxx_compiler is not None and (not isinstance(cxx_compiler, str) or not cxx_compiler):
            raise ConfigError(f"stack.{name}.cxx-compiler must be a non-empty string")
        stacks[name] = BackendStack(
            name=name,
            backend=backend,
            sdk_root=sdk_root,
            c_compiler=c_compiler,
            cxx_compiler=cxx_compiler,
            runtime_library_dirs=_strings(
                data.get("runtime-library-dirs"), f"stack.{name}.runtime-library-dirs"
            ),
            environment=_options(data.get("environment"), f"stack.{name}.environment"),
            required_providers=_strings(
                data.get("required-providers"), f"stack.{name}.required-providers"
            ),
        )

    experiments: dict[str, Experiment] = {}
    for name, body in _table(raw.get("experiment"), "experiment").items():
        data = _table(body, f"experiment.{name}")
        experiments[name] = Experiment(
            name=name,
            patches=_strings(data.get("patches"), f"experiment.{name}.patches"),
            cmake_options=_options(
                data.get("cmake-options"), f"experiment.{name}.cmake-options"
            ),
            runtime_env=_options(
                data.get("runtime-env"), f"experiment.{name}.runtime-env"
            ),
            requires=_strings(data.get("requires"), f"experiment.{name}.requires"),
            conflicts=_strings(data.get("conflicts"), f"experiment.{name}.conflicts"),
        )
    trees: list[Tree] = []
    raw_trees = raw.get("trees")
    if raw_trees is not None and not isinstance(raw_trees, list):
        raise ConfigError("trees must be a list of tables")
    _tree_keys = {
        "name",
        "alias",
        "path",
        "required",
        "role",
        "expected-tooling-revision",
    }
    for i, body in enumerate(raw_trees or []):
        data = _table(body, f"trees[{i}]")
        unknown = sorted(set(data) - _tree_keys)
        if unknown:
            raise ConfigError(f"trees[{i}] has unknown field(s): {', '.join(unknown)}")
        name = _required_string(data, "name", f"trees[{i}]")
        # alias/path may be given as the empty string to mean "unset" --
        # alias defaults to the tree name, path to "." (the local tree).
        alias = data.get("alias") or name
        tpath = data.get("path") or "."
        if not isinstance(alias, str) or not alias:
            raise ConfigError(f"trees[{i}].alias must be a non-empty string")
        if not isinstance(tpath, str) or not tpath:
            raise ConfigError(f"trees[{i}].path must be a non-empty string")
        role = data.get("role", "local")
        if not isinstance(role, str) or role not in ("local", "campaign"):
            raise ConfigError(f"trees[{i}].role={role!r} must be 'local' or 'campaign'")
        expected = data.get("expected-tooling-revision", "")
        if not isinstance(expected, str):
            raise ConfigError(f"trees[{i}].expected-tooling-revision must be a string")
        required = data.get("required", False)
        if not isinstance(required, bool):
            raise ConfigError(f"trees[{i}].required must be a boolean")
        trees.append(
            Tree(
                name=name,
                alias=alias,
                path=tpath,
                required=required,
                role=role,
                expected_tooling_revision=expected,
            )
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
            if (
                not isinstance(lane_source, str)
                or not lane_source
                or not isinstance(lane_build, str)
                or not lane_build
                or not isinstance(lane_platform, str)
                or not lane_platform
            ):
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
            lanes.append(
                CampaignLaneSelector(
                    source=lane_source, build=lane_build, platform=lane_platform
                )
            )
        campaigns[name] = CampaignProfile(name=name, lanes=tuple(lanes))

    runtime_profiles: dict[str, RuntimeProfile] = {}
    for name, body in _table(raw.get("runtime-profile"), "runtime-profile").items():
        data = _table(body, f"runtime-profile.{name}")
        tune_context = data.get("tune-context")
        if not isinstance(tune_context, int) or isinstance(tune_context, bool) or tune_context <= 0:
            raise ConfigError(f"runtime-profile.{name}.tune-context must be a positive integer")
        production_context = data.get("production-context")
        if (
            not isinstance(production_context, int)
            or isinstance(production_context, bool)
            or production_context <= 0
        ):
            raise ConfigError(
                f"runtime-profile.{name}.production-context must be a positive integer"
            )
        min_free_vram = data.get("min-free-vram-bytes-per-device")
        if (
            not isinstance(min_free_vram, int)
            or isinstance(min_free_vram, bool)
            or min_free_vram < 0
        ):
            raise ConfigError(
                f"runtime-profile.{name}.min-free-vram-bytes-per-device must be a "
                f"non-negative integer"
            )
        runtime_profiles[name] = RuntimeProfile(
            name=name,
            server_args=_argv(data.get("server-args"), f"runtime-profile.{name}.server-args"),
            tune_context=tune_context,
            production_context=production_context,
            min_free_vram_bytes_per_device=min_free_vram,
        )

    return Config(
        pinned,
        patch_sets,
        sources,
        builds,
        platforms,
        experiments,
        campaigns,
        path,
        tuple(trees),
        runtime_profiles,
        stacks,
    )
