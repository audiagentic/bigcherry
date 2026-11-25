"""How the cards of a host are used by radiance's remote-MoE offload (RAD23): card types and layouts as data.

``engines/radiance/archs.toml`` describes a card type (its kernel plugin, the weight formats it runs, whether it takes the few-rows
kernel); the serving configs (native YAML files, ``BIGCHERRY_CONFIGS``, default ``engines/radiance/configs/``) name which cards play which part. This module reads both and produces the environment each process needs. It never selects a card: the cards come from bc-serve's detection (``radiance_inventory``). No architecture, device index or budget is named in code.
"""

from __future__ import annotations

import os
import json
import re
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

from ..core import paths

SCHEMA = 1
WEIGHT_FORMATS = ("w4nl64", "fp8_e4m3_block")   # the names moe_backend.h knows (parse_formats)
ROLES = ("engine", "helper")
PLACEMENTS = ("head", "tail")
_SCHEMA = Path("engines/radiance/components/moe-remote/topology-v26.schema.json")


class LayoutError(ValueError):
    pass


@dataclass(frozen=True)
class Arch:
    name: str
    description: str
    plugin: str
    formats: tuple[str, ...]
    rows_kernel: bool
    reserve_mib: int
    roles: tuple[str, ...]


@dataclass(frozen=True)
class HelperCard:
    index: int                    # HIP/ROCR ordinal of the card
    arch: Arch
    vram_mib: int
    #: the expert budget; when ``budget_derived`` it is the card limit (VRAM less archs.toml reserve-mib), an upper bound: bc-serve
    #: derives the expert budget the owner planner and the helper use as this less the helper's own lanes, pointer banks, slot map
    #: and incoming slot (moe-remote worker_footprint.h, RAD49), which needs the model's shape
    budget_mib: int
    placement: str
    share: int
    budget_derived: bool = False


@dataclass(frozen=True)
class Resolved:
    name: str
    engine_indices: tuple[int, ...]  # inventory indices, in TP-rank order
    engine_arch: str              # the first rank's architecture ("+"-joined in rank order for a mixed world)
    engine_plugin: str            # engine kernel plugin(s) (archs.toml), comma-joined in rank order for a mixed world
    model: str                    # portable path, relative to BC_MODEL_ROOT
    tp_wire: str | None
    server_args: tuple[str, ...]
    helpers: tuple[HelperCard, ...]
    #: environment for every process of the offload (engine and helpers)
    common_env: dict[str, str] = field(default_factory=dict)

    @property
    def engine_index(self) -> int:
        return self.engine_indices[0]

    def helper_env(self, i: int) -> dict[str, str]:
        """What helper ``i`` needs on top of the common environment: its card, its plugin, the formats its backend runs."""
        h = self.helpers[i]
        return {"ROCR_VISIBLE_DEVICES": str(h.index), "HIP_VISIBLE_DEVICES": "0", "R3_MOE_WORKER_INDEX": str(i), **card_env(h.arch)}


def card_env(arch: Arch) -> dict[str, str]:
    """What a card type adds to a helper process's environment: the plugin it loads, the formats its backend runs, the few-rows kernel."""
    env = {"R3_MOE_WORKER_PLUGIN_FILE": arch.plugin, "R3_MOE_WORKER_FORMATS": ",".join(arch.formats)}
    if arch.rows_kernel:
        env["R3_MOE_WORKER_ROWS"] = "1"
    return env


def _root(repo_root: Path | None) -> Path:
    return Path(repo_root) if repo_root else paths.REPO_ROOT





def _wire_schema(repo_root: Path | None) -> dict[str, int]:
    """Versioned native v26 representation schema, parity-checked by r3_moe_shm_test."""
    path = _root(repo_root) / _SCHEMA
    try:
        schema = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise LayoutError(f"{path}: {exc}") from exc
    required = {"wire_version", "owner_id_bits", "combine_mask_bits",
                "participant_mask_bits", "count_bits", "lane_classes", "ring_slots"}
    if not isinstance(schema, dict) or set(schema) != required or schema.get("wire_version") != 26 or any(
        type(v) is not int or v <= 0 or v > 64 for v in schema.values()
    ):
        raise LayoutError(f"{path}: invalid v26 topology representation schema")
    return schema


def topology_refusal(helpers: int, ranks: int, repo_root: Path | None = None) -> str | None:
    """Mirror the native v26::counts_for *typed representation predicates*, not a slot ceiling."""
    sc = _wire_schema(repo_root)
    if not helpers or not ranks:
        return "ZeroTopology"
    u32 = (1 << sc["count_bits"]) - 1
    slots = helpers * ranks
    if slots > u32:
        return "SlotOverflow"
    lanes = slots * sc["lane_classes"]
    if lanes > u32:
        return "LaneOverflow"
    if lanes * sc["ring_slots"] > u32:
        return "RingOverflow"
    if helpers + ranks > u32:
        return "AckOverflow"
    if slots + ranks > u32:
        return "ResultOverflow"
    if helpers > (1 << sc["owner_id_bits"]) - 1:
        return "OwnerId"
    if helpers > sc["combine_mask_bits"]:
        return "CombineMask"
    if helpers + ranks > sc["participant_mask_bits"]:
        return "ParticipantMask"
    return None


def max_helpers(repo_root: Path | None = None) -> int:
    """Number of contributors representable in the native uint8 combine mask."""
    return _wire_schema(repo_root)["combine_mask_bits"]


def _load(path: Path) -> dict:
    try:
        raw = tomllib.loads(path.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError) as exc:
        raise LayoutError(f"{path}: {exc}") from exc
    if raw.get("schema") != SCHEMA:
        raise LayoutError(f"{path}: schema must be {SCHEMA}")
    return raw


def _allow(table: dict, keys: set[str], where: str) -> None:
    extra = set(table) - keys
    if extra:
        raise LayoutError(f"{where}: unknown key(s) {sorted(extra)}")


def load_archs(repo_root: Path | None = None) -> dict[str, Arch]:
    path = _root(repo_root) / "engines" / "radiance" / "archs.toml"
    raw = _load(path)
    _allow(raw, {"schema", "arch"}, str(path))
    out: dict[str, Arch] = {}
    for name, a in (raw.get("arch") or {}).items():
        where = f"{path}: arch.{name}"
        _allow(a, {"description", "plugin", "formats", "rows-kernel", "reserve-mib", "roles"}, where)
        for key in ("plugin", "formats", "rows-kernel", "reserve-mib", "roles"):
            if key not in a:
                raise LayoutError(f"{where}: {key} is required")
        formats, roles = tuple(a["formats"]), tuple(a["roles"])
        if not re.fullmatch(r"gfx[0-9a-f]+", name):
            raise LayoutError(f"{where}: not a gfx architecture name")
        if not isinstance(a["plugin"], str) or not re.fullmatch(r"[A-Za-z0-9_.-]+\.so", a["plugin"]):
            raise LayoutError(f"{where}: plugin must be a plain .so file name")
        if not formats or any(f not in WEIGHT_FORMATS for f in formats):
            raise LayoutError(f"{where}: formats must be a non-empty list of {WEIGHT_FORMATS}")
        if not roles or any(r not in ROLES for r in roles):
            raise LayoutError(f"{where}: roles must be a non-empty list of {ROLES}")
        if not isinstance(a["rows-kernel"], bool) or not isinstance(a["reserve-mib"], int) or isinstance(a["reserve-mib"], bool) or a["reserve-mib"] < 0:
            raise LayoutError(f"{where}: rows-kernel is a boolean, reserve-mib a non-negative integer")
        out[name] = Arch(name, str(a.get("description", "")), a["plugin"], formats, a["rows-kernel"], a["reserve-mib"], roles)
    if not out:
        raise LayoutError(f"{path}: no arch tables")
    return out


def layout_sources(repo_root: Path | None = None) -> list[Path]:
    """The serving configs the host tools read: the native YAML files (NAME.yaml, with a closed x-bigcherry section) of
    BIGCHERRY_CONFIGS (os.pathsep-separated directories or files), else the repository's engines/radiance/configs/. A config is
    its YAML: there is no separate layout file; bc-serve reads the same files."""
    env = "" if repo_root is not None else os.environ.get("BIGCHERRY_CONFIGS", "")
    given = [Path(x) for x in env.split(os.pathsep) if x] or [_root(repo_root) / "engines" / "radiance" / "configs"]
    files: list[Path] = []
    for g in given:
        if g.is_dir():
            found = sorted(f for f in g.glob("*.yaml") if f.is_file())
            if not found:
                raise LayoutError(f"{g}: holds no *.yaml config file")
            files += found
        elif g.is_file():
            files.append(g)
        else:
            raise LayoutError(f"{g}: neither a config file nor a directory")
    return files


def bc_serve_env(repo_root: Path | None = None) -> dict[str, str]:
    """BIGCHERRY_CONFIGS for a bc-serve child: the directories this module reads."""
    return {"BIGCHERRY_CONFIGS": os.pathsep.join(dict.fromkeys(str(f.parent) for f in layout_sources(repo_root)))}


def _layout_of(path: Path) -> dict:
    """The topology a config's YAML states, as the table this module resolves against an inventory (the shape the layout
    files had): its model, rank groups, helper groups, tp wire and helper.combine. Everything else in the file (engine
    options, tuning, memory, handoff) is the engine's and bc-serve's, not the host tools'."""
    import yaml   # tooling dependency, imported here so a command that reads no config needs no PyYAML

    doc = yaml.safe_load(path.read_text(encoding="utf-8"))
    try:
        section = doc["x-bigcherry"]
        model_id, model = next(iter(doc["models"].items()))
    except (TypeError, KeyError, StopIteration) as exc:
        raise LayoutError(f"{path}: not a serving config (needs models and an x-bigcherry section)") from exc
    layout: dict = {"description": section["description"], "model": model["model"],
                    "engine": [{"arch": r["arch"], "count": r["count"]} for r in section["ranks"]]}
    if "tp-wire" in model:
        layout["tp-wire"] = model["tp-wire"]
    if section.get("helpers"):
        layout["helper"] = [{k: v for k, v in h.items()} for h in section["helpers"]]
    if "combine" in (section.get("helper") or {}):
        layout["settings"] = {"helper": {"combine": section["helper"]["combine"]}}
    del model_id
    return layout


def load_layouts(repo_root: Path | None = None) -> dict[str, dict]:
    out: dict[str, dict] = {}
    seen: dict[str, Path] = {}
    for path in layout_sources(repo_root):
        name = path.stem
        if name in seen:
            raise LayoutError(f"config {name} is defined in both {seen[name]} and {path}")
        seen[name] = path
        out[name] = _layout_of(path)
    return out


def names(repo_root: Path | None = None) -> list[str]:
    return sorted(load_layouts(repo_root))
