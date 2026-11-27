#!/usr/bin/env python3
"""VTP60: native host scheduler gate on a private, attested source snapshot.

The existing scheduler fixtures and CMake test target own compilation. This
adapter stages only test additions, never edits the shared source/build or
starts an engine. A fresh build is mandatory; ccache may reuse objects.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import xml.etree.ElementTree as ET

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from bigcherry.core import components
from bigcherry.source.identity import SourceAttestation, verify_source_attestation
from bigcherry.hardware.power_receipt import write_json_atomic

FIXTURE = Path(__file__).with_name("scheduler_burst.cpp").resolve()
CASES = ("the_device_fakes_are_the_ones_linked",
         "bc_burst_prompt_boundary_contract",
         "bc_burst_detects_decode_overlap_loss")
DEFAULT_FIXTURE = FIXTURE.with_name("scheduler_defaults.cpp")
CONFIG_CASES = ("bc_interleave_effective_default", "bc_interleave_effective_exceptions_and_precedence",
                "prefill_interleave_takes_a_step_count_up_to_64",
                "mixed_step_target_is_an_opt_in_soft_millisecond_target")
TESTS = tuple("bc_host_" + case for case in CASES + CONFIG_CASES)


def digest(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def attestation(source: Path, expected_slice: str) -> SourceAttestation:
    meta = json.loads(source.with_name(source.name + ".metadata.json").read_text())
    if meta["source_slice_id"] != expected_slice:
        raise ValueError("source slice does not match requested candidate")
    a = SourceAttestation(meta["upstream_revision"], meta["source_tree_oid"],
                          meta["git_object_format"], meta["source_slice_id"],
                          frozenset(meta["allowed_untracked"]))
    verify_source_attestation(source, a)
    return a


def tree_files(root: Path) -> dict[str, str]:
    return {p.relative_to(root).as_posix(): ("link:" + os.readlink(p) if p.is_symlink() else digest(p))
            for p in root.rglob("*") if (p.is_file() or p.is_symlink())
            and ".git" not in p.relative_to(root).parts and "__pycache__" not in p.parts}


def copy_snapshot(source: Path, dest: Path) -> None:
    for p in source.rglob("*"):
        if p.is_symlink() and (Path(os.readlink(p)).is_absolute()
                               or not p.resolve().is_relative_to(source)):
            raise ValueError("external source symlink")
    shutil.copytree(source, dest, symlinks=True, ignore=shutil.ignore_patterns(".git", "__pycache__"))
    if tree_files(source) != tree_files(dest):
        raise ValueError("source snapshot differs")


def extract_interleave_block(text: str) -> str:
    """Extract the real bring-up policy; refuse drift in the integration seam."""
    start = "    if (!cfg_.prefill_interleave_set) {"
    stop = "    /* Control vectors are resolved before any rank declares"
    if text.count(start) != 1 or text.count(stop) != 1:
        raise ValueError("interleave bring-up seam changed; review source")
    lo, hi = text.index(start), text.index(stop)
    block = text[lo:hi].rstrip()
    if hi <= lo or not block.endswith("\n    }"):
        raise ValueError("invalid interleave bring-up block")
    return block + "\n"


def stage_defaults(tree: Path) -> None:
    tests = tree / "tests"
    block = extract_interleave_block((tree / "core/engine_bringup.cpp").read_text())
    (tests / "bc_interleave_default.inc").write_text(block)
    shutil.copyfile(DEFAULT_FIXTURE, tests / "bc_scheduler_default_cases.h")
    with (tests / "config_test.cpp").open("a") as f:
        f.write('\n#include "bc_scheduler_default_cases.h"\n')


def stage_fixture(tree: Path) -> None:
    tests = tree / "tests"
    native = tests / "sched_test.cpp"
    cmake = tests / "CMakeLists.txt"
    # The fixture is included in the existing native target, so no second
    # scheduler/test executable or ad hoc compiler/link command is needed.
    text = FIXTURE.read_text()
    include = '#include "sched_test.cpp"'
    if text.count(include) != 1 or "bc_burst_prompt_boundary_contract" in native.read_text():
        raise ValueError("ambiguous fixture insertion")
    header = tests / "bc_scheduler_burst_cases.h"
    if header.exists():
        raise ValueError("test fixture already staged")
    header.write_text(text.replace(include, "", 1))
    with native.open("a") as f:
        f.write('\n#include "bc_scheduler_burst_cases.h"\n')
    stage_defaults(tree)
    register_cases(cmake, "sched_test", CASES)
    register_cases(cmake, "config_test", CONFIG_CASES)


def register_cases(cmake: Path, target: str, cases: tuple[str, ...]) -> None:
    with cmake.open("a") as f:
        for case in cases:
            name = "bc_host_" + case
            f.write(f'\nadd_test(NAME {name} COMMAND {target} {case})\n')
            f.write(f'set_tests_properties({name} PROPERTIES LABELS "unit;fast;bc_host" '
                    'TIMEOUT 60 ENVIRONMENT "${RAD_TEST_ENV}")\n')


def validate_results(path: Path) -> None:
    cases = list(ET.parse(path).iter("testcase"))
    if sorted(c.attrib.get("name") for c in cases) != sorted(TESTS):
        raise ValueError("missing, duplicate or unexpected native test")
    for case in cases:
        name = case.attrib["name"]
        output = case.findtext("system-out", "")
        expected = name.removeprefix("bc_host_")
        if (case.find("failure") is not None or case.find("error") is not None
                or case.find("skipped") is not None or case.attrib.get("status") == "notrun"
                or "FAIL " in output or "SKIP " in output
                or not any(line.strip() == "ok   " + expected for line in output.splitlines())):
            raise ValueError(f"native test did not prove pass: {name}")


def run_logged(command, log: Path, env: dict) -> float:
    print("+", " ".join(map(str, command)), flush=True)
    start = time.monotonic()
    with log.open("w") as stream:
        subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT,
                       env=env, check=True)
    return time.monotonic() - start


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", type=Path, required=True)
    ap.add_argument("--expected-slice", required=True)
    ap.add_argument("--component-repo", type=Path, required=True)
    ap.add_argument("--expected-component-sha256", help="independently expected serving component digest")
    ap.add_argument("--output", type=Path, required=True, help="new local run directory")
    ap.add_argument("--native-sim", action="store_true", help="stage native simulator extension")
    ap.add_argument("--cc", default="gcc-14")
    ap.add_argument("--cxx", default="g++-14")
    ap.add_argument("--jobs", type=int, default=2)
    ap.add_argument("--prefix-path", default="")
    args = ap.parse_args()
    os.nice(19)
    if args.jobs < 1:
        ap.error("jobs must be positive")
    source = args.source.resolve(strict=True)
    root = args.output.resolve()
    if root.is_relative_to(source) or root.is_relative_to(args.component_repo.resolve()):
        ap.error("output must be outside input trees")
    a = attestation(source, args.expected_slice)
    component = args.component_repo.resolve(strict=True) / "engines/radiance/components/serving"
    before = components.load(component, "radiance").content_digest()
    if args.expected_component_sha256 is not None and before != args.expected_component_sha256:
        raise ValueError("serving component does not match expected digest")
    component_revision = subprocess.check_output(["git", "-C", str(args.component_repo), "rev-parse", "HEAD"], text=True).strip()
    root.mkdir(parents=True, exist_ok=False)
    tree, build = root / "src", root / "build"
    serving = root / "component/engines/radiance/components/serving"
    copy_snapshot(source, tree)
    copy_snapshot(component, serving)
    if components.load(serving, "radiance").content_digest() != before:
        raise ValueError("component snapshot differs")
    # Recheck the live source once copied; do not build against a moving cache.
    verify_source_attestation(source, a)
    stage_fixture(tree)
    if args.native_sim:
        from scheduler_sim import stage_sim
        stage_sim(tree)
    env = dict(os.environ, ROCR_VISIBLE_DEVICES="99", HIP_VISIBLE_DEVICES="",
               OMP_WAIT_POLICY="PASSIVE", OMP_NUM_THREADS="4", RAD_TEST_STRICT="1")
    for key in ("RAD_TEST_FILTER", "RAD_SCHED_SIM", "RAD_TEST_LEVEL"):
        env.pop(key, None)
    configure = ["cmake", "-S", str(tree), "-B", str(build), "-G", "Ninja",
                 "-DCMAKE_BUILD_TYPE=Release", "-DRAD_WITH_HIP=OFF",
                 "-DRAD_BUILD_TESTS=ON", "-DRAD_WITH_FFMPEG=ON",
                 f"-DCMAKE_C_COMPILER={args.cc}", f"-DCMAKE_CXX_COMPILER={args.cxx}",
                 "-DCMAKE_C_COMPILER_LAUNCHER=ccache", "-DCMAKE_CXX_COMPILER_LAUNCHER=ccache",
                 f"-DBC_SERVING_V2_CMAKE={serving / 'bc_serving_v2.cmake'}",
                 f"-DCMAKE_PREFIX_PATH={args.prefix_path}"]
    receipt = {"kind": "native-host-scheduler-screen", "source_slice_id": a.source_slice_id,
               "source_tree_oid": a.tree_oid, "source_revision": a.upstream_revision,
               "fixture_sha256": digest(FIXTURE), "default_fixture_sha256": digest(DEFAULT_FIXTURE),
               "interleave_block_sha256": digest(tree / "tests/bc_interleave_default.inc"), "serving_component_sha256": before,
               "component_repo_revision": component_revision,
               "tests": list(TESTS), "status": "failed",
               "native_sim": args.native_sim,
               "staged_scheduler_test_sha256": digest(tree / "tests/sched_test.cpp")}
    try:
        receipt["configure_s"] = run_logged(configure, root / "configure.log", env)
        receipt["build_s"] = run_logged(["cmake", "--build", str(build), "--target",
                                         "sched_test", "config_test", "--parallel", str(args.jobs)],
                                        root / "build.log", env)
        receipt["test_s"] = run_logged(["ctest", "--test-dir", str(build), "-L", "^bc_host$",
                                        "--no-tests=error", "--output-on-failure",
                                        "--test-output-size-passed", "1048576",
                                        "--output-junit", str(root / "tests.xml")],
                                       root / "test.log", env)
        validate_results(root / "tests.xml")
        receipt["test_binary_sha256"] = digest(build / "tests/sched_test")
        receipt["config_binary_sha256"] = digest(build / "tests/config_test")
        receipt["core_archive_sha256"] = digest(build / "core/librad_core.a")
        receipt["junit_sha256"] = digest(root / "tests.xml")
        receipt["status"] = "passed"
    finally:
        write_json_atomic(root / "result.json", receipt)
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
