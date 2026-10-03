"""Deterministic Slurm GRES rendering from an accepted hardware inventory."""
from __future__ import annotations

from collections import Counter

from .model import HardwareInventory


class GresRenderError(RuntimeError):
    pass


def render_gres_conf(inventory: HardwareInventory) -> str:
    """Render one explicit File= line per accepted GPU, sorted by render node.

    Stable IDs never appear in Slurm ``Type``. A device without a render node
    cannot safely be published as an explicit GPU resource and fails closed.
    """
    devices = sorted(
        inventory.devices,
        key=lambda d: (d.architecture, d.render_node or "", d.device_id),
    )
    lines: list[str] = []
    seen_files: set[str] = set()
    for device in devices:
        if not device.render_node:
            raise GresRenderError(
                f"device {device.device_id} has no render_node; cannot render gres.conf"
            )
        if device.render_node in seen_files:
            raise GresRenderError(f"duplicate render node in inventory: {device.render_node}")
        seen_files.add(device.render_node)
        lines.append(
            f"Name=gpu Type={device.architecture} File={device.render_node} Flags=amd_gpu_env"
        )
    return "\n".join(lines) + ("\n" if lines else "")


def render_node_gres(inventory: HardwareInventory) -> str:
    counts = Counter(device.architecture for device in inventory.devices)
    return ",".join(f"gpu:{arch}:{counts[arch]}" for arch in sorted(counts))
