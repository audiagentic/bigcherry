#!/usr/bin/env python3
"""Render/install the systemd unit that applies each card's configured power settings at boot (PWR04).

The unit runs `bigcherry gpu-power apply` as root once the GPU driver is up and `gpu-power restore` when it stops, from
a checkout and environment file given here (never written into the unit's code). Safe by default: without --apply it
only prints the unit, or writes it under --dest-root for tests; --apply installs it into /etc/systemd/system and enables
it (root required).
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path, PurePosixPath

UNIT = "bigcherry-gpu-power.service"


def unit_text(*, project_root: PurePosixPath, environment_file: PurePosixPath, python: PurePosixPath) -> str:
    env = f"BIGCHERRY_ENVIRONMENT={environment_file} PYTHONPATH={project_root / 'tools'} PYTHONDONTWRITEBYTECODE=1"
    return f"""[Unit]
Description=BigCherry per-card GPU power settings (bigcherry gpu-power)
After=systemd-modules-load.service systemd-udev-settle.service
Wants=systemd-udev-settle.service

[Service]
Type=oneshot
RemainAfterExit=yes
WorkingDirectory={project_root}
Environment={env}
ExecStart={python} -m bigcherry gpu-power apply
ExecStop={python} -m bigcherry gpu-power restore

[Install]
WantedBy=multi-user.target
"""


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--project-root", required=True, help="checkout the unit runs bigcherry from (on the GPU host)")
    p.add_argument("--environment-file", required=True, help="the host's environment.local.toml with the device power settings")
    p.add_argument("--python", default="/usr/bin/python3")
    p.add_argument("--dest-root", help="write the unit under this directory instead of installing it (tests)")
    p.add_argument("--apply", action="store_true", help="install into /etc/systemd/system and enable it (root)")
    a = p.parse_args(argv)
    text = unit_text(project_root=PurePosixPath(a.project_root), environment_file=PurePosixPath(a.environment_file),
                     python=PurePosixPath(a.python))
    if a.dest_root:
        out = Path(a.dest_root) / "etc/systemd/system" / UNIT
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text, encoding="utf-8")
        print(out)
        return 0
    if not a.apply:
        print(text, end="")
        return 0
    if os.geteuid() != 0:
        print("install_bigcherry_gpu_power: --apply needs root", file=sys.stderr)
        return 2
    Path("/etc/systemd/system", UNIT).write_text(text, encoding="utf-8")
    subprocess.run(["systemctl", "daemon-reload"], check=True)
    subprocess.run(["systemctl", "enable", "--now", UNIT], check=True)
    print(f"installed and started {UNIT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
