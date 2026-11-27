#!/usr/bin/env python3
"""Run a host-only burst screen against an existing Radiance CMake host build.

Uses the native sched_test translation unit, compile options and link dependencies,
not a Python scheduler model. No engine launch, model weights or GPU lease.
Unix Makefiles + CMAKE_EXPORT_COMPILE_COMMANDS=ON, RAD_WITH_HIP=OFF only.
The caller owns source/build provenance; this is a lab screen, not a release gate.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

FIXTURE = Path(__file__).with_name("scheduler_burst.cpp").resolve()


def digest(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def replace_one(args: list[str], old: str, new: str) -> None:
    if args.count(old) != 1:
        raise ValueError(f"expected exactly one occurrence of {old!r}")
    args[args.index(old)] = new


def commands(build: Path, scratch: Path):
    build = build.resolve()
    cache = (build / "CMakeCache.txt").read_text().splitlines()
    if "RAD_WITH_HIP:STRING=OFF" not in cache or "CMAKE_GENERATOR:INTERNAL=Unix Makefiles" not in cache:
        raise ValueError("requires a configured HIP-off Unix Makefiles build")
    rows = [row for row in json.loads((build / "compile_commands.json").read_text())
            if Path(row["file"]).name == "sched_test.cpp"]
    if len(rows) != 1:
        raise ValueError("expected one native scheduler compile command")
    row = rows[0]
    source = Path(row["file"]).resolve()
    if not source.is_file() or not (source.parent.parent / "core/sched/scheduler.cpp").is_file():
        raise ValueError("native scheduler sources are absent")
    cwd = Path(row["directory"]).resolve()
    if cwd != build / "tests":
        raise ValueError("unexpected scheduler build directory")
    comp = list(row["arguments"]) if "arguments" in row else shlex.split(row["command"])
    original_object = comp[comp.index("-o") + 1]
    obj = scratch / "burst.o"
    binary = scratch / "burst"
    replace_one(comp, row["file"], str(FIXTURE))
    replace_one(comp, original_object, str(obj))
    comp += ["-I" + str(source.parent), "-O0"]
    link = shlex.split((cwd / "CMakeFiles/sched_test.dir/link.txt").read_text())
    replace_one(link, original_object, str(obj))
    replace_one(link, "sched_test", str(binary))
    link = [a for a in link if not a.startswith("-Wl,--dependency-file=")]
    return cwd, source, comp, link, binary


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--build", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    root = Path(os.environ["BC_WORK_ROOT"])
    if not root.is_dir():
        raise ValueError("BC_WORK_ROOT must be an existing work root")
    env = dict(os.environ, ROCR_VISIBLE_DEVICES="99", HIP_VISIBLE_DEVICES="",
               OMP_WAIT_POLICY="PASSIVE", OMP_NUM_THREADS="4", RAD_TEST_STRICT="1")
    with tempfile.TemporaryDirectory(prefix="scheduler-burst-", dir=root) as temp:
        work = Path(temp)
        env["RAD_TEST_TMPDIR"] = str(work)
        cwd, source, comp, link, binary = commands(args.build.resolve(), work)
        start = time.monotonic()
        for command in (comp, link):
            subprocess.run(command, cwd=cwd, env=env, check=True, timeout=180)
        build_s = time.monotonic() - start
        start = time.monotonic()
        run = subprocess.run([str(binary), "bc_burst_"], cwd=work, env=env,
                             capture_output=True, text=True, timeout=60)
        result = {"kind": "host-scheduler-screen", "fixture_sha256": digest(FIXTURE),
                  "upstream_test_sha256": digest(source),
                  "scheduler_source_sha256": digest(source.parent.parent / "core/sched/scheduler.cpp"),
                  "core_archive_sha256": digest(args.build / "core/librad_core.a"),
                  "test_binary_sha256": digest(binary), "build_s": build_s,
                  "test_s": time.monotonic() - start, "exit_code": run.returncode,
                  "stdout": run.stdout, "stderr": run.stderr}
        from bigcherry.hardware.power_receipt import write_json_atomic
        write_json_atomic(args.output, result)
        print(run.stdout, end="")
        if run.stderr:
            print(run.stderr, end="")
        raise SystemExit(run.returncode)


if __name__ == "__main__":
    main()
