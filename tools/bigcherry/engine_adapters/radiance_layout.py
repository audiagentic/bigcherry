"""How the cards of a host are used by radiance's remote-MoE offload (RAD23): card types and layouts as data.

``engines/radiance/archs.toml`` describes a card type (its kernel plugin, the weight formats it runs, whether it takes the few-rows
kernel); the layout definitions (``BIGCHERRY_LAYOUTS``, default ``engines/radiance/layouts/``) name which cards play which part. This module reads both, resolves a layout against a host's
device inventory (``config/environment.local.toml``) and produces the environment each process needs. It checks everything that can be
checked without a GPU, so an inconsistent layout is refused before a card is touched. No architecture, device index or budget is named in
code: the helper, the launcher and the harness adapter get them from here.
"""

from __future__ import annotations

import os
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




def _protocol_bound(name: str, repo_root: Path | None) -> int:
    text = (_root(repo_root) / _HEADER).read_text(encoding="utf-8")
    m = re.search(r"constexpr\s+uint32_t\s+" + name + r"\s*=\s*(\d+)\s*;", text)
    if not m:
        raise LayoutError(f"{_HEADER}: {name} not found")
    return int(m.group(1))


def max_helpers(repo_root: Path | None = None) -> int:
    """The protocol's helper bound, read from the header that defines it (kMaxHelpers), so the two cannot drift."""
    return _protocol_bound("kMaxHelpers", repo_root)


def max_slots(repo_root: Path | None = None) -> int:
    """The protocol's request-slot bound (kMaxSlots): one slot a (helper, engine rank) pair (RAD49)."""
    return _protocol_bound("kMaxSlots", repo_root)


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
    """The layout definition files: BIGCHERRY_LAYOUTS (os.pathsep-separated files or directories of *.toml), else the
    repository's example directory engines/radiance/layouts/. Layouts are run-time data; bc-serve reads the same list."""
    # an explicit repository root (a test's, a checkout's) names its own example layouts; otherwise BIGCHERRY_LAYOUTS
    env = "" if repo_root is not None else os.environ.get("BIGCHERRY_LAYOUTS", "")
    given = [Path(x) for x in env.split(os.pathsep) if x] or [_root(repo_root) / "engines" / "radiance" / "layouts"]
    files: list[Path] = []
    for g in given:
        if g.is_dir():
            found = sorted(f for f in g.glob("*.toml") if f.is_file())
            if not found:
                raise LayoutError(f"{g}: holds no *.toml layout file")
            files += found
        elif g.is_file():
            files.append(g)
        else:
            raise LayoutError(f"{g}: neither a layout file nor a directory")
    return files


def bc_serve_env(repo_root: Path | None = None) -> dict[str, str]:
    """BIGCHERRY_LAYOUTS for a bc-serve child: the same sources this module reads."""
    return {"BIGCHERRY_LAYOUTS": ":".join(str(f) for f in layout_sources(repo_root))}


def load_layouts(repo_root: Path | None = None) -> dict[str, dict]:
    out: dict[str, dict] = {}
    seen: dict[str, Path] = {}
    for path in layout_sources(repo_root):
        raw = _load(path)
        _allow(raw, {"schema", "layout"}, str(path))
        for name, table in (raw.get("layout") or {}).items():
            if name in seen:
                raise LayoutError(f"layout {name} is defined in both {seen[name]} and {path}")
            seen[name] = path
            out[name] = table
    return out


def names(repo_root: Path | None = None) -> list[str]:
    return sorted(load_layouts(repo_root))


def _pick(inventory, spec: dict, role: str, archs: dict[str, Arch], where: str, taken: set[int]):
    """The inventory devices a group selects: by explicit index, or by architecture (all of them, or the first `count`)."""
    by_index = {d.index: d for d in inventory}
    if "devices" in spec:
        if "arch" in spec or "count" in spec:
            raise LayoutError(f"{where}: give devices alone, or arch (with count), not both")
        if len(set(spec["devices"])) != len(spec["devices"]):
            raise LayoutError(f"{where}: devices lists a card twice: {spec['devices']}")
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


def _engine_archs(engine) -> str:
    return "+".join(dict.fromkeys(d.arch for d in engine))


def _engine_plugins(engine, archs: dict[str, Arch]) -> str:
    return ",".join(dict.fromkeys(archs[d.arch].plugin for d in engine))


def resolve(name: str, host: environment.Host, *, budgets: dict[str, int] | None = None, repo_root: Path | None = None) -> Resolved:
    """A named layout against a host's inventory. `budgets` overrides budget-mib per architecture (arch -> MiB)."""
    archs = load_archs(repo_root)
    layouts = load_layouts(repo_root)
    if name not in layouts:
        raise LayoutError(f"unknown layout {name!r}; known: {sorted(layouts)}")
    lay = layouts[name]
    where = f"layout.{name}"
    _allow(lay, {"description", "model", "engine", "tp-wire", "server-args", "helper", "env", "tuning"}, where)
    # [layout.NAME.tuning]: engine tuning fields (tuning.*; bc-serve checks each key and type against the contract)
    if not isinstance(lay.get("tuning", {}), dict):
        raise LayoutError(f"{where}.tuning must be a table of engine tuning fields")
    model = lay.get("model")
    if not isinstance(model, str) or not model or model.startswith("/") or not model.endswith(".rad") or any(
        part in ("", ".", "..") for part in model.split("/")
    ):
        raise LayoutError(f"{where}.model: expected a .rad path relative to BC_MODEL_ROOT, without traversal")
    budgets = dict(budgets or {})
    taken: set[int] = set()
    engine_spec = lay.get("engine")
    # one group { arch, count } (or devices), or a list of { arch, count } groups in rank order: a world of more than one
    # card type, each rank running its own architecture's plugin (radiance --rank-devices)
    specs = engine_spec if isinstance(engine_spec, list) else [engine_spec]
    if not specs or not all(isinstance(s, dict) for s in specs):
        raise LayoutError(f"{where}: engine is required: {{ arch, count }} or a list of them")
    if isinstance(engine_spec, list) and len({s.get("arch") for s in specs}) != len(specs):
        raise LayoutError(f"{where}.engine: an architecture is named in two groups; give one count")
    engine = []
    for si, spec in enumerate(specs):
        swhere = f"{where}.engine" + (f"[{si}]" if isinstance(engine_spec, list) else "")
        _allow(spec, {"arch", "count"} if isinstance(engine_spec, list) else {"arch", "devices", "count"}, swhere)
        engine += _pick(host.devices, spec, "engine", archs, swhere, taken)
    if not engine:
        raise LayoutError(f"{where}.engine: at least one engine card is required")
    # One group: every rank shares an architecture and its plugin. Several: a mixed world, one plugin per architecture.
    if not isinstance(engine_spec, list) and len({d.arch for d in engine}) != 1:
        raise LayoutError(f"{where}.engine: all tensor-parallel ranks must share an architecture")
    groups = lay.get("helper") or []
    tp_wire = lay.get("tp-wire")
    if len(engine) > 1:
        if tp_wire not in ("exact", "wht6"):
            raise LayoutError(f"{where}: a multi-rank engine requires tp-wire exact or wht6")
    elif tp_wire is not None:
        raise LayoutError(f"{where}: tp-wire is only for multiple engine ranks")
    server_args = lay.get("server-args", [])
    # Reject data that could override model, rank/card, host transport,
    # kernel selection or the plan safety flags.
    allowed = {"--max-model-len", "--max-num-seqs", "--max-num-batched-tokens",
               "--kv-cache-dtype", "--num-speculative-tokens", "--checkpoint-interval",
               "--host-pool-mib", "--host-kv-mib", "--ngram-placement"}
    if (not isinstance(server_args, list) or len(server_args) % 2
            or any(not isinstance(x, str) for x in server_args)
            or any(server_args[i] not in allowed or not server_args[i + 1]
                   or server_args[i + 1].startswith("-") for i in range(0, len(server_args), 2))
            or len(set(server_args[::2])) != len(server_args[::2])):
        raise LayoutError(f"{where}.server-args: expected unique, allowed option/value pairs")
    if not groups:
        if budgets:
            raise LayoutError(f"{where}: budget overrides require remote-MoE helpers")
        if lay.get("env"):
            raise LayoutError(f"{where}: R3_MOE_WORKER_* settings require remote-MoE helpers")
        return Resolved(name, tuple(d.index for d in engine), _engine_archs(engine),
                        _engine_plugins(engine, archs), model, tp_wire, tuple(server_args), (), {})
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
            limit = d.vram_mib - archs[d.arch].reserve_mib
            # absent: derived by bc-serve from the card (this limit) less the helper's own footprint; not a literal in the layout
            budget = g.get("budget-mib", limit)
            derived = "budget-mib" not in g
            if d.arch in budgets:
                budget = budgets[d.arch]
                derived = False
                used_overrides.add(d.arch)
            if not isinstance(budget, int) or isinstance(budget, bool) or budget <= 0:
                raise LayoutError(f"{gwhere}: budget-mib is a positive integer")
            if budget > limit:
                raise LayoutError(f"{gwhere}: budget {budget} MiB on device {d.index} ({d.arch}, {d.vram_mib} MiB, reserve {archs[d.arch].reserve_mib}) exceeds {limit} MiB")
            helpers.append(HelperCard(d.index, archs[d.arch], d.vram_mib, budget, placement, share, derived))
    unused = set(budgets) - used_overrides
    if unused:
        raise LayoutError(f"{where}: a budget override names {sorted(unused)}, which no helper group uses")
    if len(helpers) > max_helpers(repo_root):
        raise LayoutError(f"{where}: {len(helpers)} helpers exceed the protocol's {max_helpers(repo_root)}")
    if not any(h.placement == "head" and h.share > 0 for h in helpers):
        raise LayoutError(f"{where}: no head helper with a share")
    if helpers and len(helpers) * len(engine) > max_slots(repo_root):
        # moe_remote_shm.h slots_for: one request slot a (helper, engine rank); the hook and the helpers refuse more at start
        raise LayoutError(f"{where}: {len(helpers)} helpers behind a {len(engine)}-rank engine need {len(helpers) * len(engine)} "
                          f"request slots; the mailbox protocol has {max_slots(repo_root)}")
    common = {
        "R3_MOE_WORKER_COUNT": str(len(helpers)),
        "R3_MOE_WORKER_ENGINE_RANKS": str(len(engine)),
        # the expert budgets, when the layout (or an override) gives them all; a derived one is bc-serve's (worker_footprint.h)
        "R3_MOE_WORKER_MIBS": "" if any(h.budget_derived for h in helpers) else ",".join(str(h.budget_mib) for h in helpers),
        # one weight per helper (the tail helper's is not read: tail placement gives it what the heads left); only when a head differs from 1
        "R3_MOE_WORKER_SHARES": ",".join(str(h.share if h.placement == "head" else 0) for h in helpers)
                                if any(h.placement == "head" and h.share != 1 for h in helpers) else "",
    }
    if any(h.placement == "tail" for h in helpers):
        common["R3_MOE_WORKER_PLACEMENT"] = "tail"
    # the layout's own settings for the engine and every helper (the helper-row combine, for one); never a name the cards above decide
    env = lay.get("env", {})
    if not isinstance(env, dict):
        raise LayoutError(f"{where}.env must be a table of R3_MOE_WORKER_* settings")
    for key, value in env.items():
        if not re.fullmatch(r"R3_MOE_WORKER_[A-Z0-9_]+", key) or not isinstance(value, str):
            raise LayoutError(f"{where}.env: {key!r} must be an R3_MOE_WORKER_* setting with a string value")
        if key in common or key in {"R3_MOE_WORKER_LAYOUT", "R3_MOE_WORKER_DEVICES"}:
            raise LayoutError(f"{where}.env: {key} is decided by the cards of the layout")
        common[key] = value
    if common.get("R3_MOE_WORKER_COMBINE") == "1" and len(helpers) < 2:
        # the combine adds the rows of two to kCombineHelpers (= kMaxHelpers, the bound checked above) helpers; one has nothing to add
        raise LayoutError(f"{where}: the helper-row combine needs at least two helpers; {len(helpers)} would run with it silently off")
    return Resolved(name, tuple(d.index for d in engine), _engine_archs(engine), _engine_plugins(engine, archs),
                    model, tp_wire, tuple(server_args), tuple(helpers), {k: v for k, v in common.items() if v != ""})
