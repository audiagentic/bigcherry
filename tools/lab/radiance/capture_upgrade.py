#!/usr/bin/env python3
"""Upgrade r3_moe_worker request captures from version 2 (Flash-Next constants) to version 3 (the model's shape in the header).

A version 2 file is: magic, version, layer, rows, route_count, hidden (six u32), then the layer's slot map (512 i32), the routes,
the quantised activation rows. Version 3 adds experts, top_k and act_group (three u32) to the header; the payload is unchanged.
Files are rewritten in place; Flash-Next is hidden 2560, 512 experts, top-k 10, group 128.

usage: capture_upgrade.py DIR_OR_FILE...
"""
import struct
import sys
from pathlib import Path

MAGIC = 0x52335243
V2 = struct.Struct("<6I")
V3 = struct.Struct("<9I")
FLASH_NEXT = (512, 10, 128)


def upgrade(path: Path) -> str:
    data = path.read_bytes()
    if len(data) < V2.size:
        return "too short"
    magic, version, layer, rows, routes, hidden = V2.unpack_from(data, 0)
    if magic != MAGIC:
        return "not a capture"
    if version == 3:
        return "already version 3"
    if version != 2 or hidden != 2560:
        return f"version {version}, hidden {hidden}: not a Flash-Next v2 capture"
    path.write_bytes(V3.pack(magic, 3, layer, rows, routes, hidden, *FLASH_NEXT) + data[V2.size:])
    return "upgraded"


def main(argv):
    if not argv:
        print(__doc__)
        return 2
    for arg in argv:
        p = Path(arg)
        for f in sorted(p.glob("*.bin")) if p.is_dir() else [p]:
            print(f"{f}: {upgrade(f)}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
