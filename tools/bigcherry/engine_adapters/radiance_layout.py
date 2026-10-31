"""How the cards of a host are used by radiance's remote-MoE offload (RAD23): card types and layouts as data.

``engines/radiance/archs.toml`` describes a card type (its kernel plugin, the weight formats it runs, whether it takes the few-rows
kernel); ``engines/radiance/layouts.toml`` names which cards play which part. This module reads both, resolves a layout against a host's
device inventory (``config/environment.local.toml``) and produces the environment each process needs. It checks everything that can be
checked without a GPU, so an inconsistent layout is refused before a card is touched. No architecture, device index or budget is named in
code: the helper, the launcher and the harness adapter get them from here.
"""

from __future__ import annotations

import re
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

from ..core import environment, paths

SCHEMA = 1
WEIGHT_FORMATS = ("w4nl64", "fp8_e4m3_block")   # the names moe_backend.h knows (parse_formats)
ROLES = ("engine", "helper")
PLACEMENTS = ("head", "tail")
_HEADER = Path("engines/radiance/components/moe-remote/moe_remote_shm.h")


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
    budget_mib: int
    placement: str
    share: int


@dataclass(frozen=True)
class Resolved:
    name: str
    engine_index: int
    engine_arch: str
    engine_plugin: str            # the engine card's kernel plugin file (archs.toml)
    helpers: tuple[HelperCard, ...]
    #: environment for every process of the offload (engine and helpers)
    common_env: dict[str, str] = field(default_factory=dict)

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


def max_helpers(repo_root: Path | None = None) -> int:
    """The protocol's helper bound, read from the header that defines it (kMaxHelpers), so the two cannot drift."""
    text = (_root(repo_root) / _HEADER).read_text(encoding="utf-8")
    m = re.search(r"constexpr\s+uint32_t\s+kMaxHelpers\s*=\s*(\d+)\s*;", text)
    if not m:
        raise LayoutError(f"{_HEADER}: kMaxHelpers not found")
    return int(m.group(1))


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


def load_layouts(repo_root: Path | None = None) -> dict[str, dict]:
    path = _root(repo_root) / "engines" / "radiance" / "layouts.toml"
    raw = _load(path)
    _allow(raw, {"schema", "layout"}, str(path))
    return dict(raw.get("layout") or {})


def names(repo_root: Path | None = None) -> list[str]:
    return sorted(load_layouts(repo_root))


def _pick(inventory, spec: dict, role: str, archs: dict[str, Arch], where: str, taken: set[int]):
    """The inventory devices a group selects: by explicit index, or by architecture (all of them, or the first `count`)."""
    by_index = {d.index: d for d in inventory}
    if "devices" in spec:
        if "arch" in spec:
            raise LayoutError(f"{where}: give arch or devices, not both")
        picked = []
        for i in spec["devices"]:
            if i not in by_index:
                raise LayoutError(f"{where}: device {i} is not in the host inventory (has {sorted(by_index)})")
            picked.append(by_index[i])
    else:
        arch = spec.get("arch")
        if not arch:
            raise LayoutError(f"{where}: arch or devices is required")
        pool = [d for d in inventory if d.arch == arch]
        if not pool:
            raise LayoutError(f"{where}: the host inventory has no {arch} card")
        count = spec.get("count", "all" if role == "helper" else 1)
        if count == "all":
            picked = pool
        elif isinstance(count, int) and not isinstance(count, bool) and count > 0:
            picked = pool[:count]
            if len(pool) < count:
                raise LayoutError(f"{where}: wants {count} {arch} card(s), the host inventory has {len(pool)}")
        else:
            raise LayoutError(f"{where}: count is 'all' or a positive integer")
    for d in picked:
        if d.arch not in archs:
            raise LayoutError(f"{where}: device {d.index} is {d.arch}, which engines/radiance/archs.toml does not describe")
        if role not in archs[d.arch].roles:
            raise LayoutError(f"{where}: {d.arch} may not be used as {role} (archs.toml roles {list(archs[d.arch].roles)})")
        if d.index in taken:
            raise LayoutError(f"{where}: device {d.index} is already used by another part of the layout")
        taken.add(d.index)
    return picked


def resolve(name: str, host: environment.Host, *, budgets: dict[str, int] | None = None, repo_root: Path | None = None) -> Resolved:
    """A named layout against a host's inventory. `budgets` overrides budget-mib per architecture (arch -> MiB)."""
    archs = load_archs(repo_root)
    layouts = load_layouts(repo_root)
    if name not in layouts:
        raise LayoutError(f"unknown layout {name!r}; known: {sorted(layouts)}")
    lay = layouts[name]
    where = f"layout.{name}"
    _allow(lay, {"description", "engine", "helper"}, where)
    budgets = dict(budgets or {})
    taken: set[int] = set()
    engine_spec = lay.get("engine")
    if not isinstance(engine_spec, dict):
        raise LayoutError(f"{where}: engine is required")
    _allow(engine_spec, {"arch", "devices"}, f"{where}.engine")
    engine = _pick(host.devices, engine_spec, "engine", archs, f"{where}.engine", taken)
    if len(engine) != 1:
        raise LayoutError(f"{where}.engine: exactly one card runs the model, got {len(engine)}")
    groups = lay.get("helper") or []
    if not groups:
        raise LayoutError(f"{where}: at least one helper group is required")
    helpers: list[HelperCard] = []
    used_overrides: set[str] = set()
    for gi, g in enumerate(groups):
        gwhere = f"{where}.helper[{gi}]"
        _allow(g, {"arch", "devices", "count", "budget-mib", "placement", "share"}, gwhere)
        placement = g.get("placement", "head")
        if placement not in PLACEMENTS:
            raise LayoutError(f"{gwhere}: placement is one of {PLACEMENTS}")
        if placement == "tail" and (gi != len(groups) - 1 or gi == 0):
            raise LayoutError(f"{gwhere}: a tail group must be the last group and follow a head group")
        cards = _pick(host.devices, g, "helper", archs, gwhere, taken)
        if placement == "tail" and len(cards) != 1:
            raise LayoutError(f"{gwhere}: a tail group is one card (tail placement gives the last helper the leftovers)")
        share = g.get("share", 1)
        if not isinstance(share, int) or isinstance(share, bool) or share < 0:
            raise LayoutError(f"{gwhere}: share is a non-negative integer")
        for d in cards:
            budget = g.get("budget-mib")
            if d.arch in budgets:
                budget = budgets[d.arch]
                used_overrides.add(d.arch)
            if not isinstance(budget, int) or isinstance(budget, bool) or budget <= 0:
                raise LayoutError(f"{gwhere}: budget-mib is a positive integer")
            limit = d.vram_mib - archs[d.arch].reserve_mib
            if budget > limit:
                raise LayoutError(f"{gwhere}: budget {budget} MiB on device {d.index} ({d.arch}, {d.vram_mib} MiB, reserve {archs[d.arch].reserve_mib}) exceeds {limit} MiB")
            helpers.append(HelperCard(d.index, archs[d.arch], d.vram_mib, budget, placement, share))
    unused = set(budgets) - used_overrides
    if unused:
        raise LayoutError(f"{where}: a budget override names {sorted(unused)}, which no helper group uses")
    if len(helpers) > max_helpers(repo_root):
        raise LayoutError(f"{where}: {len(helpers)} helpers exceed the protocol's {max_helpers(repo_root)}")
    if not any(h.placement == "head" and h.share > 0 for h in helpers):
        raise LayoutError(f"{where}: no head helper with a share")
    common = {
        "R3_MOE_WORKER_COUNT": str(len(helpers)),
        "R3_MOE_WORKER_ENGINE_RANKS": "1",
        "R3_MOE_WORKER_MIBS": ",".join(str(h.budget_mib) for h in helpers),
        # one weight per helper (the tail helper's is not read: tail placement gives it what the heads left); only when a head differs from 1
        "R3_MOE_WORKER_SHARES": ",".join(str(h.share if h.placement == "head" else 0) for h in helpers)
                                if any(h.placement == "head" and h.share != 1 for h in helpers) else "",
    }
    if any(h.placement == "tail" for h in helpers):
        common["R3_MOE_WORKER_PLACEMENT"] = "tail"
    return Resolved(name, engine[0].index, engine[0].arch, archs[engine[0].arch].plugin, tuple(helpers), {k: v for k, v in common.items() if v != ""})
