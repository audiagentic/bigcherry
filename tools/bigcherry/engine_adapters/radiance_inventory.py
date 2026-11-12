"""One serving GPU inventory: the bc-serve KFD/PCI detector and resolver.

No device identity, GPU ordinal or role is read from environment.local.toml.
The local environment can supply scalar paths and explicit BDF-scoped power
overrides, never the set of available cards or the placement order.
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..core import environment
from . import radiance_layout

_BDF = re.compile(r"^[0-9a-f]{4}:[0-9a-f]{2}:[0-9a-f]{2}\.[0-7]$")


class InventoryError(ValueError):
    """Refusal to use an unavailable, malformed or inconsistent detector result."""


@dataclass(frozen=True)
class Gpu:
    bdf: str
    ordinal: int
    arch: str
    vram_mib: int


@dataclass(frozen=True)
class Inventory:
    gpus: tuple[Gpu, ...]

    @classmethod
    def from_facts(cls, facts: object) -> Inventory:
        if not isinstance(facts, dict) or not isinstance(facts.get("gpus"), list):
            raise InventoryError("bc-serve output lacks facts.gpus (a JSON array)")
        devices: list[Gpu] = []
        bdfs: set[str] = set()
        ordinals: set[int] = set()
        for i, raw in enumerate(facts["gpus"]):
            if not isinstance(raw, dict):
                raise InventoryError(f"facts.gpus[{i}] is not an object")
            bdf, ordinal, arch, vram = (
                raw.get("bdf"), raw.get("hip_ordinal"), raw.get("arch"), raw.get("vram_total_mib")
            )
            if not isinstance(bdf, str) or not _BDF.fullmatch(bdf):
                raise InventoryError(f"facts.gpus[{i}] has invalid PCI BDF {bdf!r}")
            if bdf in bdfs:
                raise InventoryError(f"duplicate detector BDF {bdf}")
            if type(ordinal) is not int or ordinal < 0 or ordinal in ordinals:
                raise InventoryError(f"duplicate/invalid detector HIP ordinal {ordinal!r}")
            if not isinstance(arch, str) or not re.fullmatch(r"gfx[0-9a-f]+", arch):
                raise InventoryError(f"facts.gpus[{i}] has invalid architecture {arch!r}")
            if type(vram) is not int or vram <= 0:
                raise InventoryError(f"facts.gpus[{i}] has invalid VRAM {vram!r}")
            bdfs.add(bdf)
            ordinals.add(ordinal)
            devices.append(Gpu(bdf, ordinal, arch, vram))
        return cls(tuple(devices))

    def by_bdf(self, bdf: str) -> Gpu:
        for gpu in self.gpus:
            if gpu.bdf == bdf:
                return gpu
        raise InventoryError(f"stale or unavailable PCI BDF {bdf}; detector has {sorted(g.bdf for g in self.gpus)}")

    def by_ordinal(self, ordinal: int) -> Gpu:
        for gpu in self.gpus:
            if gpu.ordinal == ordinal:
                return gpu
        raise InventoryError(f"device {ordinal} is not in the bc-serve detector inventory")

    def check_power_overrides(self, host: environment.Host) -> None:
        """Local overrides are keyed by physical BDF, not by ordinal or card model."""
        used: set[str] = set()
        for device in host.devices:
            if device.power is None:
                continue
            if not device.locator:
                raise InventoryError("a configured device power override requires its PCI BDF locator")
            if device.locator in used:
                raise InventoryError(f"duplicate power override BDF {device.locator}")
            used.add(device.locator)
            self.by_bdf(device.locator)


@dataclass(frozen=True)
class Layout:
    resolved: radiance_layout.Resolved
    inventory: Inventory
    engine_bdfs: tuple[str, ...]
    helper_bdfs: tuple[str, ...]

    def derive_resources(self, model, targets, host, *, worker_footprints_bytes):
        """Prepare the inactive v2 budget from this exact bc-serve BDF resolution.

        The measurements are deliberately mandatory. V1 uses its own C++
        resolver until the single RSC11 S13 cutover.
        """
        from .radiance_resources import derive
        return derive(self, model, targets, host,
                      worker_footprints_bytes=worker_footprints_bytes)


def binary_path(explicit: Path | None = None) -> Path:
    name = str(explicit or os.environ.get("BC_SERVE_BINARY") or
               (Path(os.environ["RADIANCE_HOME"]) / "bin/bc-serve" if os.environ.get("RADIANCE_HOME") else "") or
               (shutil.which("bc-serve") or ""))
    if not name or not Path(name).is_file():
        raise InventoryError(
            "bc-serve executable not found: set BC_SERVE_BINARY or RADIANCE_HOME, "
            "or supply --bc-serve (no static inventory fallback)"
        )
    return Path(name)


def _run(binary: Path | None, args: list[str]) -> dict[str, Any]:
    cmd = [str(binary_path(binary)), *args]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=False, timeout=30)
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise InventoryError(f"bc-serve invocation failed: {exc}") from exc
    if result.returncode:
        raise InventoryError(f"bc-serve {' '.join(args[:2])} failed ({result.returncode}): {result.stderr.strip()}")
    try:
        import json
        parsed = json.loads(result.stdout)
    except ValueError as exc:
        raise InventoryError("bc-serve did not emit a JSON object") from exc
    if not isinstance(parsed, dict):
        raise InventoryError("bc-serve JSON output must be an object")
    return parsed


def detect(*, binary: Path | None = None, sys_root: Path | None = None) -> Inventory:
    args = ["--detect", "--json"]
    if sys_root is not None:
        args.extend(["--sys-root", str(sys_root)])
    return Inventory.from_facts(_run(binary, args))


def resolve(
    name: str,
    *,
    binary: Path | None = None,
    sys_root: Path | None = None,
    site: Path | None = None,
    profile: Path | None = None,
    host: environment.Host | None = None,
) -> Layout:
    """Use bc-serve's actual role selection, not a Python copy of its selector."""
    recipes = radiance_layout.load_layouts()
    if name not in recipes:
        raise InventoryError(f"unknown layout {name!r}; known: {sorted(recipes)}")
    recipe = recipes[name]
    args = ["--explain", "--json", "--set", f"layout.name={name}"]
    if sys_root is not None:
        args += ["--sys-root", str(sys_root)]
    if site is not None:
        args += ["--site", str(site)]
    if profile is not None:
        args += ["--profile", str(profile)]
    # The explanatory resolver expects a complete plan. These non-device
    # settings are sourced from the named recipe and host's scalar roots.
    if site is None:
        root = os.environ.get("BC_MODEL_ROOT") or (host.model_root if host else "")
        if root:
            args += ["--set", f"engine.model={Path(root) / recipe['model']}"]
        home = os.environ.get("RADIANCE_HOME", "")
        if home:
            args += ["--set", f"engine.radiance_home={home}"]
        args += ["--set", "handoff.heat=off", "--set", "handoff.histogram=off"]
    doc = _run(binary, args)
    try:
        values = doc["values"]
        facts = doc["facts"]
        layout = values["layout"]
        engine_bdfs = tuple(layout["engine_cards"])
        helper_bdfs = tuple(layout["helper_cards"])
        process_helpers = doc["process_helpers"]
        if layout["name"] != name:
            raise InventoryError(f"bc-serve resolved {layout['name']!r}, not {name!r}")
        if not isinstance(process_helpers, list) or len(process_helpers) != len(helper_bdfs):
            raise InventoryError("bc-serve process_helpers does not match layout.helper_cards")
        if not engine_bdfs:
            raise InventoryError("bc-serve selected no engine cards")
        if len(engine_bdfs + helper_bdfs) != len(set(engine_bdfs + helper_bdfs)):
            raise InventoryError("duplicate engine/helper PCI BDF in bc-serve resolution")
        inventory = Inventory.from_facts(facts)
        if host is not None:
            inventory.check_power_overrides(host)
        engine = [inventory.by_bdf(b) for b in engine_bdfs]
        helper = [inventory.by_bdf(b) for b in helper_bdfs]
        # a world of more than one card type is a layout's engine groups (bc-serve hands the engine its ranks' cards as
        # layout.rank_devices); every rank's architecture must be a described card type
        archs = radiance_layout.load_archs()
        for g in engine:
            if g.arch not in archs:
                raise InventoryError(f"unknown engine architecture {g.arch}")
        rank_devices = values["layout"].get("rank_devices")
        if len({g.arch for g in engine}) > 1 and list(rank_devices or []) != list(engine_bdfs):
            raise InventoryError("bc-serve selected a mixed-architecture engine without its rank devices in rank order")
        helpers = []
        for i, (gpu, rec) in enumerate(zip(helper, process_helpers)):
            if not isinstance(rec, dict) or rec.get("bdf") != gpu.bdf or rec.get("hip_ordinal") != gpu.ordinal or rec.get("arch") != gpu.arch:
                raise InventoryError(f"bc-serve helper{i} role/BDF/ordinal disagrees with detected KFD/PCI inventory")
            if gpu.arch not in archs:
                raise InventoryError(f"unknown helper architecture {gpu.arch}")
            if rec.get("plugin_file") != archs[gpu.arch].plugin or rec.get("formats") != list(archs[gpu.arch].formats) or rec.get("rows_kernel") != archs[gpu.arch].rows_kernel:
                raise InventoryError(f"bc-serve helper{i} architecture metadata disagrees with archs.toml")
            budget = rec.get("budget_mib")
            share = rec.get("share")
            placement = rec.get("placement")
            if type(budget) is not int or budget <= 0 or type(share) is not int or share < 0 or placement not in ("head", "tail"):
                raise InventoryError(f"bc-serve helper{i} has invalid budget/share/placement")
            helpers.append(radiance_layout.HelperCard(gpu.ordinal, archs[gpu.arch], gpu.vram_mib, budget, placement, share))
        common = {}
        if helpers:
            common = {
                "R3_MOE_WORKER_COUNT": str(len(helpers)),
                "R3_MOE_WORKER_ENGINE_RANKS": str(len(engine)),
                "R3_MOE_WORKER_MIBS": ",".join(str(h.budget_mib) for h in helpers),
            }
            if any(h.share != 1 and h.placement == "head" for h in helpers):
                common["R3_MOE_WORKER_SHARES"] = ",".join(str(h.share if h.placement == "head" else 0) for h in helpers)
            if any(h.placement == "tail" for h in helpers):
                common["R3_MOE_WORKER_PLACEMENT"] = "tail"
            common.update(recipe.get("env", {}))
        # rank order; a mixed world (several engine groups) names each architecture and plugin once
        engine_arch = "+".join(dict.fromkeys(g.arch for g in engine))
        engine_plugin = ",".join(dict.fromkeys(archs[g.arch].plugin for g in engine))
        selected = radiance_layout.Resolved(
            name, tuple(g.ordinal for g in engine), engine_arch, engine_plugin,
            recipe["model"], values["engine"].get("tp_wire"), tuple(recipe.get("server-args", [])),
            tuple(helpers), common,
        )
        return Layout(selected, inventory, engine_bdfs, helper_bdfs)
    except (KeyError, TypeError, IndexError) as exc:
        raise InventoryError(f"bc-serve --explain --json lacks required structured serving field: {exc}") from exc
