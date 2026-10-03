#!/usr/bin/env python3
"""Render/install Brutus Slurm host config from accepted BigCherry inventory.

Safe by default: without --apply/--dest-root it only prints the plan. Tests use
--dest-root so rendering is executable without root or Slurm installed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT / "tools") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "tools"))

from bigcherry.hardware.inventory import InventoryCatalog  # noqa: E402
from bigcherry.hardware.slurm import render_gres_conf, render_node_gres  # noqa: E402


def render_files(
    *,
    project_root: Path,
    hardware_root: Path,
    executor_id: str,
    node_name: str,
    cpus: int,
    real_memory_mib: int,
) -> dict[str, str]:
    if cpus < 1 or real_memory_mib < 1:
        raise ValueError("cpus and real_memory_mib must be positive")
    inventory = InventoryCatalog(hardware_root).load(executor_id)
    if inventory.host_id != node_name:
        raise ValueError(
            f"accepted inventory host_id {inventory.host_id!r} does not match node {node_name!r}"
        )
    template_root = project_root.resolve() / "config" / "slurm"
    slurm_template = (template_root / "slurm.conf.example").read_text(encoding="utf-8")
    cgroup = (template_root / "cgroup.conf.example").read_text(encoding="utf-8")
    replacements = {
        "@NODE_NAME@": node_name,
        "@CPUS@": str(cpus),
        "@REAL_MEMORY_MIB@": str(real_memory_mib),
        "@NODE_GRES@": render_node_gres(inventory),
    }
    slurm = slurm_template
    for key, value in replacements.items():
        slurm = slurm.replace(key, value)
    unresolved = sorted(token for token in replacements if token in slurm)
    if unresolved or "@" in "".join(
        line for line in slurm.splitlines() if not line.lstrip().startswith("#")
    ):
        raise ValueError(f"unresolved Slurm template placeholder(s): {unresolved}")
    header = (
        f"# accepted_inventory_hash={inventory.material_hash}\n"
        f"# executor_id={executor_id}\n"
    )
    return {
        "slurm.conf": header + slurm,
        "gres.conf": header + render_gres_conf(inventory),
        "cgroup.conf": header + cgroup,
    }


def _digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def install_rendered(files: dict[str, str], dest: Path) -> tuple[str, ...]:
    dest.mkdir(parents=True, exist_ok=True)
    written: list[str] = []
    for name, text in sorted(files.items()):
        target = dest / name
        temp = target.with_suffix(target.suffix + ".tmp")
        temp.write_text(text, encoding="utf-8")
        temp.replace(target)
        written.append(str(target))
    return tuple(written)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="render_bigcherry_slurm")
    parser.add_argument("--project-root", type=Path, default=_REPO_ROOT)
    parser.add_argument("--hardware-root", type=Path, required=True)
    parser.add_argument("--executor-id", default="brutus")
    parser.add_argument("--node-name", default="brutus")
    parser.add_argument("--cpus", type=int, required=True)
    parser.add_argument("--real-memory-mib", type=int, required=True)
    parser.add_argument("--dest-root", type=Path, default=None)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args(argv)
    try:
        files = render_files(
            project_root=args.project_root,
            hardware_root=args.hardware_root,
            executor_id=args.executor_id,
            node_name=args.node_name,
            cpus=args.cpus,
            real_memory_mib=args.real_memory_mib,
        )
        destination = args.dest_root
        if destination is None and args.apply:
            destination = Path("/etc/slurm")
        written: tuple[str, ...] = ()
        if destination is not None:
            written = install_rendered(files, destination)
        result = {
            "ok": True,
            "dry_run": destination is None,
            "destination": None if destination is None else str(destination),
            "written": list(written),
            "files": {
                name: {"bytes": len(text.encode("utf-8")), "sha256": _digest(text)}
                for name, text in sorted(files.items())
            },
        }
        print(json.dumps(result, sort_keys=True))
        return 0
    except Exception as exc:
        print(json.dumps({"ok": False, "error": f"{type(exc).__name__}: {exc}"}, sort_keys=True))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
