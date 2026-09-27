#!/usr/bin/env python3
"""Render/install the BigCherry job-service systemd integration.

Safe by default: --apply is required for writes outside --dest-root.  The same
renderer is used by tests with a temporary dest-root so install behavior is
validated without root/systemd.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path


def unit_texts(*, project_root: Path, work_root: Path, python: Path, user: str) -> dict[str, str]:
    pending = work_root / "jobs" / "inbox" / "pending"
    service = f"""[Unit]
Description=BigCherry durable job inbox ingestion
After=slurmctld.service slurmd.service

[Service]
Type=oneshot
User={user}
WorkingDirectory={project_root}
EnvironmentFile=-/etc/bigcherry/jobs.env
ExecStart={python} -m bigcherry jobs ingest --once
"""
    path = f"""[Unit]
Description=Watch BigCherry durable job inbox

[Path]
PathChanged={pending}
Unit=bigcherry-jobs-ingest.service

[Install]
WantedBy=multi-user.target
"""
    timer = """[Unit]
Description=BigCherry job inbox fallback poll

[Timer]
OnBootSec=30s
OnUnitActiveSec=30s
Unit=bigcherry-jobs-ingest.service
Persistent=true

[Install]
WantedBy=timers.target
"""
    return {
        "bigcherry-jobs-ingest.service": service,
        "bigcherry-jobs-ingest.path": path,
        "bigcherry-jobs-ingest.timer": timer,
    }


def env_text(*, project_root: Path, work_root: Path) -> str:
    return "\n".join((
        f"BIGCHERRY_PROJECT_ROOT={project_root}",
        f"BIGCHERRY_WORK_ROOT={work_root}",
        f"BIGCHERRY_JOBS_ROOT={work_root / 'jobs'}",
        f"BIGCHERRY_HARDWARE_ROOT={work_root / 'hardware'}",
        f"BIGCHERRY_EXECUTORS_CONFIG={project_root / 'config' / 'jobs' / 'executors.toml'}",
        f"PYTHONPATH={project_root / 'tools'}",
        "",
    ))


def doctor(project_root: Path) -> dict[str, object]:
    commands = ("munge", "slurmctld", "slurmd", "sbatch", "squeue", "scontrol")
    found = {name: shutil.which(name) for name in commands}
    files = {
        "executors": project_root / "config" / "jobs" / "executors.toml",
        "slurm": project_root / "config" / "slurm" / "slurm.conf.example",
        "cgroup": project_root / "config" / "slurm" / "cgroup.conf.example",
        "gres": project_root / "config" / "slurm" / "gres.conf.example",
    }
    return {
        "commands": found,
        "files": {key: str(path) for key, path in files.items()},
        "files_present": {key: path.is_file() for key, path in files.items()},
        "ok": all(found.values()) and all(path.is_file() for path in files.values()),
    }


def install(args: argparse.Namespace) -> dict[str, object]:
    project = args.project_root.resolve()
    work = args.work_root.resolve()
    python = args.python.resolve()
    units = unit_texts(project_root=project, work_root=work, python=python, user=args.user)
    if args.dest_root is not None:
        root = args.dest_root.resolve()
        systemd = root / "etc" / "systemd" / "system"
        envdir = root / "etc" / "bigcherry"
    else:
        if not args.apply:
            return {"apply": False, "units": units, "environment": env_text(project_root=project, work_root=work), "doctor": doctor(project)}
        systemd = Path("/etc/systemd/system")
        envdir = Path("/etc/bigcherry")
    systemd.mkdir(parents=True, exist_ok=True)
    envdir.mkdir(parents=True, exist_ok=True)
    for name, text in units.items():
        (systemd / name).write_text(text, encoding="utf-8")
    (envdir / "jobs.env").write_text(env_text(project_root=project, work_root=work), encoding="utf-8")
    for path in (work / "jobs" / "inbox" / "pending", work / "hardware"):
        path.mkdir(parents=True, exist_ok=True)
    if args.dest_root is None and args.apply:
        subprocess.run(("systemctl", "daemon-reload"), check=True)
        if args.enable:
            subprocess.run(("systemctl", "enable", "--now", "bigcherry-jobs-ingest.path", "bigcherry-jobs-ingest.timer"), check=True)
    return {"apply": True, "systemd_root": str(systemd), "environment_file": str(envdir / "jobs.env"), "units": sorted(units), "doctor": doctor(project)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--work-root", type=Path, required=True)
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--user", default=os.environ.get("SUDO_USER") or os.environ.get("USER") or "bigcherry")
    parser.add_argument("--dest-root", type=Path, default=None, help="sandbox root for tests/offline rendering")
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--enable", action="store_true")
    parser.add_argument("--doctor-only", action="store_true")
    args = parser.parse_args(argv)
    result = doctor(args.project_root.resolve()) if args.doctor_only else install(args)
    print(json.dumps(result, sort_keys=True, indent=2))
    return 0 if result.get("ok", result.get("apply") is not None) else 1


if __name__ == "__main__":
    raise SystemExit(main())
