"""S7 v1 launch bridge: collect card-free measurements for the pure RSC03 ledger.

bc-serve remains the authority for BDF selection, helper footprints and
owner maps. The RAD directory provides the remaining expert/embedding bytes.
This is an input adapter; the budget arithmetic lives in radiance_resources.
"""
from __future__ import annotations

import argparse
import json
import math
import struct
import subprocess
import sys
import tempfile
import tomllib
from pathlib import Path

from . import radiance_inventory, radiance_layout
from . import radiance_resources as resources

_MIB = resources.MIB
_RAD_MAGIC = 0x31444152
_RAD_VERSION = 2
_RAD_ENTRY = 136  # RadFileEntry, abi/rad_format.h, RAD_MAX_RANK=6


class LaunchResourceError(ValueError):
    pass


def _rad_sizes(model: Path) -> tuple[dict[tuple[int, int], int], int, int]:
    """Read only RAD v2 directory metadata (no tensor payload, no GPU)."""
    with model.open("rb") as f:
        header = f.read(104)
        if len(header) != 104 or struct.unpack_from("<II", header) != (_RAD_MAGIC, _RAD_VERSION):
            raise LaunchResourceError("model is not a RAD v2 container")
        file_size = struct.unpack_from("<Q", header, 8)[0]
        string_offset, string_size = struct.unpack_from("<QQ", header, 56)
        directory_offset, count = struct.unpack_from("<QQ", header, 88)
        if (file_size != model.stat().st_size or not count
                or string_offset + string_size > file_size
                or directory_offset + count * _RAD_ENTRY > file_size):
            raise LaunchResourceError("model has an inconsistent RAD directory")
        f.seek(string_offset)
        strings = f.read(string_size)
        f.seek(directory_offset)
        directory = f.read(count * _RAD_ENTRY)
    expert: dict[tuple[int, int], int] = {}
    embedding = ngram = 0
    for pos in range(0, len(directory), _RAD_ENTRY):
        raw = directory[pos:pos + _RAD_ENTRY]
        name_off = struct.unpack_from("<Q", raw, 0)[0]
        layer, index, slot = struct.unpack_from("<iii", raw, 80)
        byte_count = struct.unpack_from("<Q", raw, 112)[0]
        if name_off >= len(strings) or byte_count > file_size:
            raise LaunchResourceError("model has invalid weight directory entries")
        name = strings[name_off:].split(b"\x00", 1)[0].decode("utf-8", errors="replace").lower()
        # A weight may be remapped at load. Round up to the worker's plane
        # alignment, never undercount the stored source unit.
        size = ((byte_count + 255) // 256) * 256
        if layer >= 0 and index >= 0 and slot >= 0:
            key = (layer, index)
            expert[key] = expert.get(key, 0) + size
        elif "ngram" in name:
            ngram += size
        elif "embed" in name:
            embedding += size
    if not expert:
        raise LaunchResourceError("model has no routed expert directory entries")
    return expert, embedding, ngram


def _calibration_target(layout: str, key: str, path: Path) -> int:
    raw = tomllib.loads(path.read_text(encoding="utf-8"))
    candidates = [x for x in raw["entry"] if x["key"] == key
                  and x.get("match", {}).get("layout", layout) == layout]
    if not candidates:
        return 0
    best = max(candidates, key=lambda e: len(e.get("match", {})))
    value = best["value"]
    if type(value) is not int or value < 0:
        raise LaunchResourceError(f"bad calibrated {key}")
    return value


def _collect(args: argparse.Namespace) -> resources.ResourcePlan:
    if args.sessions <= 0 or args.context <= 0:
        raise LaunchResourceError("sessions and context must be positive")
    archs = radiance_layout.load_archs()
    with tempfile.TemporaryDirectory(prefix="bc-rsc03-") as td:
        temporary = Path(td)
        site = temporary / "site.toml"
        # Paths only describe the current start. This never reselects a card:
        # the one bc-serve resolver reads the exact same site and layout.
        site.write_text(
            "[engine]\nmodel = " + json.dumps(str(args.model)) +
            "\nradiance_home = " + json.dumps(str(args.radiance_home)) +
            "\n[handoff]\nheat = " + json.dumps(str(args.heat)) +
            "\nhistogram = " + json.dumps(str(args.histogram)) + "\n",
            encoding="utf-8",
        )
        layout = radiance_inventory.resolve(args.layout, binary=args.bc_serve, site=site)
        raw = radiance_inventory._run(args.bc_serve, [
            "--explain", "--json", "--site", str(site),
            "--set", f"layout.name={args.layout}",
        ])
        # Unmeasured manifests still contain the real model- and heat-dependent
        # owner map. The measuring launch later creates its own manifest.
        unmeasured = temporary / "owner-map.json"
        cmd = [str(args.bc_serve), "--write-manifest", str(unmeasured),
               "--unmeasured", "--site", str(site), "--power-profile", "stock",
               "--set", f"layout.name={args.layout}",
               "--set", f"service.max_sessions={args.sessions}",
               "--set", f"service.max_context={args.context}"]
        run = subprocess.run(cmd, text=True, capture_output=True, check=False, timeout=90)
        if run.returncode:
            raise LaunchResourceError(
                f"bc-serve cannot publish owner map ({run.returncode}): {run.stderr.strip()}"
            )
        try:
            owners = json.loads(unmeasured.read_text(encoding="utf-8"))["spec"]["plan"]["owner_map"]
        except (OSError, ValueError, KeyError, TypeError) as exc:
            raise LaunchResourceError("bc-serve omitted the real owner map") from exc

    weight_bytes, embedding, ngram = _rad_sizes(args.model)
    selected: set[tuple[int, int]] = set()
    for helper in owners["helpers"]:
        for pair in helper["experts"]:
            key = tuple(pair)
            if key not in weight_bytes or key in selected:
                raise LaunchResourceError("owner map does not match model expert directory")
            selected.add(key)
    host_experts = sum(v for key, v in weight_bytes.items() if key not in selected)
    if args.ngram != "ram":
        ngram = 0

    # Each helper budget is already model- and lane-derived by C++ bc-serve.
    # Recover its non-expert footprint to make the pure Python ledger verify
    # the same BDF/card/arch arithmetic, never invent another budget.
    worker_bytes: dict[str, int] = {}
    for helper in layout.resolved.helpers:
        gpu = layout.inventory.by_ordinal(helper.index)
        exact_footprint = (gpu.vram_mib - archs[gpu.arch].reserve_mib -
                           helper.budget_mib) * _MIB
        if exact_footprint <= 0:
            raise LaunchResourceError(f"{gpu.bdf}: no model-derived worker footprint")
        worker_bytes[gpu.bdf] = exact_footprint

    facts = raw["facts"]["host"]
    cg = [v["room_mib"] for v in facts.get("memory_cgroups", [])]
    pin = facts["amdgpu_pin"]
    pin_mib = (pin["cap_mib"] if pin.get("loaded") and not pin.get("lifted")
               else facts["mem_total_mib"])
    available = facts["mem_available_mib"]
    reserve = raw["values"]["memory"]["host_reserve_mib"]
    # Calibration is a measured *target*, not an output budget. The pure
    # derivation must still admit it against current RAM and model placement.
    calibration = Path(__file__).resolve().parents[3] / "engines/radiance/components/serving/calibration.toml"
    kv_target = _calibration_target(args.layout, "calibration.host_kv_tier.target_mib", calibration)
    margin = _calibration_target(args.layout, "calibration.host_ledger_margin.mib", calibration)
    total_tokens = args.context * args.sessions
    kv_bytes_per_token = max(1, math.ceil(kv_target * _MIB / total_tokens))
    retained = min(total_tokens, (kv_target * _MIB) // kv_bytes_per_token)
    model = resources.ModelMemory(
        owners["owner_map_sha256"], embedding, host_experts,
        max(weight_bytes.values()), ngram, 0, 0, kv_bytes_per_token,
    )
    targets = resources.Targets(
        args.context, args.sessions, retained, margin * _MIB, reserve * _MIB,
    )
    host = resources.HostLimits(
        available * _MIB, min(cg, default=available) * _MIB, pin_mib * _MIB,
    )
    return layout.derive_resources(
        model, targets, host, worker_footprints_bytes=worker_bytes
    )


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="bigcherry resources derive-launch")
    p.add_argument("derive-launch", choices=("derive-launch",))
    p.add_argument("--layout", required=True)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--bc-serve", type=Path, required=True)
    p.add_argument("--radiance-home", type=Path, required=True)
    p.add_argument("--heat", type=Path, required=True)
    p.add_argument("--histogram", required=True)
    p.add_argument("--sessions", type=int, required=True)
    p.add_argument("--context", type=int, required=True)
    p.add_argument("--ngram", choices=("ram", "disk", "vram"), required=True)
    args = p.parse_args(argv)
    try:
        plan = _collect(args)
    except (ValueError, OSError, KeyError, subprocess.TimeoutExpired,
            radiance_inventory.InventoryError, resources.ResourceError) as exc:
        print(f"resources: {exc}", file=sys.stderr)
        return 2
    # The output is shell assignments only; the ledger goes to stderr.
    print(f"HOST_POOL_MIB={plan.host_pool_mib}")
    print(f"HOST_KV_MIB={plan.host_kv_mib}")
    print("RSC03 derived resource ledger: " +
          json.dumps(plan.explain(), sort_keys=True), file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
