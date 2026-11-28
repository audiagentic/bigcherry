"""The runtime a served build needs beside its binary (RAD01 PR 8b).

Some engines load plugins at run time (radiance: kernel libraries from ``RADIANCE_HOME``). After the lane builds the
engine binary, this builds the engine's declared runtime targets in the same build tree, then each component the
engine declares (core/components.py): configured against the engine installed from this build and the lane's own
materialised source, built, and its outputs copied into the build's runtime home. An engine without runtime targets
or components (llama.cpp) gets nothing here.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import Callable

from ..core import components, engines, paths
from ..core import config as campaign_config
from ..core.process import run_checked
from .build import resolve_launcher, rocm_configure_options

Runner = Callable[[list[str], Path], None]
#: (component build tree, cwd) -> every CTest test registered there, by name, with its labels
TestLister = Callable[[Path, Path], dict[str, list[str]]]
#: the CTest label a test that needs a GPU carries; such a test is registered but never declared card-free
GPU_LABEL = "gpu"


def _cache_value(tree: Path, name: str) -> str | None:
    """Read the effective value from this CMake configuration, if present."""
    try:
        for line in (tree / "CMakeCache.txt").read_text(encoding="utf-8", errors="replace").splitlines():
            if line.startswith(f"{name}:"):
                return line.split("=", 1)[1].strip()
    except OSError:
        pass
    return None


def component_build_dir(build_dir: Path, name: str, source: Path | None = None) -> Path:
    """Where ``build_runtime`` configures and builds component ``name`` from ``source`` (its CMakeLists directory).

    A build dir is content-addressed, so every checkout with the same sources shares it, but a component's CMake cache
    pins the source path it was first configured from and CMake refuses another. The plain ``components/<name>`` tree
    serves the first path (and any later run from it); a run from a different path gets its own
    ``components/<name>@<hash of the path>`` tree beside it. The existing tree is never reconfigured or removed: another
    run may be building in it. Without ``source`` the plain directory is returned."""
    plain = build_dir / "components" / name
    if source is None:
        return plain
    pinned = _cache_value(plain, "CMAKE_HOME_DIRECTORY")
    if pinned is None or Path(pinned).resolve() == Path(source).resolve():
        return plain
    return build_dir / "components" / f"{name}@{hashlib.sha256(str(Path(source).resolve()).encode()).hexdigest()[:10]}"


class RuntimeBundleError(RuntimeError):
    """A runtime target or component did not build, or did not produce a declared output."""


def stage_marker(work_root: Path, stage: str) -> Path:
    """The file a staged lane build writes when its stage completed. The campaign pipeline records a stage's exception
    as text, so the CLI recognises a finished stage by this file, not by the error."""
    return work_root / "lane-stage" / (re.sub(r"[^A-Za-z0-9_.-]", "_", stage) + ".done")


class LaneStageComplete(Exception):
    """A staged lane build (BC_LANE_STAGE) finished its stage. It publishes nothing: no build metadata, no provenance, no
    artifact; it only fills the compiler cache. The full, unstaged lane build is the one that records a build."""

    def __init__(self, stage: str):
        super().__init__(f"lane stage {stage} complete")
        self.stage = stage


#: the CMake install steps a component needs from the engine: its headers and its package files. They read only source and
#: configure-time files, so a staged component build can lay them out before the engine compiles (the components use
#: radiance::abi, an interface target, and RadAddPlugin; none links an engine library).
_DEV_DESTINATIONS = ("include/", "lib/cmake/")


def install_development_files(build_dir: Path, prefix: Path, runner: Runner = run_checked) -> None:
    """Run only the configured engine's header and package-file install steps (from CMake's own cmake_install.cmake
    scripts) into ``prefix``."""
    steps = []
    for script in sorted(build_dir.rglob("cmake_install.cmake")):
        if "components" in script.relative_to(build_dir).parts:
            continue
        text = script.read_text(encoding="utf-8", errors="replace")
        for match in re.finditer(r'file\(INSTALL DESTINATION "\$\{CMAKE_INSTALL_PREFIX\}/([^"]*)"(?:[^()"]|"[^"]*")*\)', text):
            if match.group(1).startswith(_DEV_DESTINATIONS) or match.group(1) in ("include", "lib/cmake"):
                steps.append(match.group(0))
    if not steps:
        raise RuntimeBundleError(f"{build_dir} has no header or package install steps; was it configured?")
    script = build_dir / "bc-dev-install.cmake"
    script.write_text("\n".join(steps) + "\n", encoding="utf-8")
    runner(["cmake", f"-DCMAKE_INSTALL_PREFIX={prefix}", "-P", str(script)], build_dir)


def build_runtime(
    engine: str,
    *,
    source_root: Path,
    build_dir: Path,
    platform: campaign_config.Platform,
    runner: Runner = run_checked,
    only: tuple[str, ...] | None = None,
    engine_built: bool = True,
) -> dict[str, str]:
    """Build ``engine``'s runtime targets and components into ``build_dir``. Returns {component: content digest}.

    ``launcher_options`` (RAD18): each component is configured through its OWN separate ``cmake -S -B`` invocation,
    which previously forwarded only ``CMAKE_C_COMPILER``/``CMAKE_CXX_COMPILER`` -- never the platform's
    ``CMAKE_*_COMPILER_LAUNCHER`` options (e.g. ``ccache``), so every component's compile ran uncached on every
    build regardless of whether its source actually changed. Found by measuring a real CI build log: radiance's
    ``libr3`` component (92-99 HIP "units") took the same ~11-14 minutes on two consecutive runs of near-identical
    source, while the main engine build (which DOES get the launcher via the top-level configure) showed real cache
    hits. A component's own declared ``configure`` entries still win if they already set the same key."""
    layout = engines.load(paths.REPO_ROOT, engine)
    c_compiler, cxx_compiler = platform.c_compiler, platform.cxx_compiler
    launcher_options = tuple((k, resolve_launcher(v)) for k, v in platform.options if k.endswith("_COMPILER_LAUNCHER"))
    built: dict[str, str] = {}
    if layout.runtime_targets and engine_built:
        runner(["cmake", "--build", str(build_dir), "-j", "--target", *layout.runtime_targets], source_root)
    declared = components.load_all(paths.REPO_ROOT, engine)
    if only is not None:
        unknown = set(only) - set(declared)
        if unknown:
            raise RuntimeBundleError(f"no such component(s): {', '.join(sorted(unknown))}")
        declared = {name: component for name, component in declared.items() if name in only}
    if not declared:
        return built
    if not layout.runtime_home:
        raise RuntimeBundleError(f"engine {engine!r} declares components but no build.runtime-home")
    home = build_dir / layout.runtime_home
    destinations: dict[str, str] = {}
    for name, component in sorted(declared.items()):
        for output in component.outputs:
            if output in destinations:
                raise RuntimeBundleError(f"component {name} and {destinations[output]} both declare {output}")
            destinations[output] = name
    # Components run the same ROCm lookups as the engine, in their own configure: pin them to the same SDK.
    rocm_options = rocm_configure_options(platform, layout.rocm_paths)
    prefix = build_dir / "runtime-prefix"
    if prefix.exists():
        shutil.rmtree(prefix)
    if engine_built:
        runner(["cmake", "--install", str(build_dir), "--prefix", str(prefix)], source_root)
    else:
        install_development_files(build_dir, prefix, runner)
    serving_edition = _cache_value(build_dir, "BC_SERVING_CONTRACT_VERSION") if engine == engines.RADIANCE else None
    for name, component in sorted(declared.items()):
        component_build = component_build_dir(build_dir, name, component.root)
        values = {"engine_source": str(source_root), "targets": ";".join(component.targets)}
        options = {key: value.format(**values) for key, value in component.configure}
        for key, value in launcher_options:
            options.setdefault(key, value)
        options.update(rocm_options)
        if serving_edition is not None and name in {"serving", "moe-remote"}:
            options["BC_SERVING_CONTRACT_VERSION"] = serving_edition
        options["CMAKE_PREFIX_PATH"] = str(prefix)
        if c_compiler:
            options["CMAKE_C_COMPILER"] = c_compiler
        if cxx_compiler:
            options["CMAKE_CXX_COMPILER"] = cxx_compiler
        runner(["cmake", "-S", str(component.root), "-B", str(component_build), "-G", "Ninja",
                *(f"-D{key}={value}" for key, value in sorted(options.items()))], source_root)
        runner(["cmake", "--build", str(component_build), "-j"], source_root)
        for output in component.outputs:
            # rad_add_plugin emits into the component's runtime home; the
            # serving and worker executables emit into bin/ or helpers/.
            # Ignore unrelated/stale basenames elsewhere in the build tree.
            produced = [path for path in (component_build / layout.runtime_home / output,
                                          component_build / output) if path.is_file()]
            if not produced:
                raise RuntimeBundleError(f"component {name} did not produce {output}")
            if len(produced) != 1:
                raise RuntimeBundleError(f"component {name} produced ambiguous outputs for {output}: "
                                         + ", ".join(str(path) for path in produced))
            target = home / output
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(produced[0], target)
        built[name] = component.content_digest()
    return built


def component_test_log(build_dir: Path, name: str, source: Path | None = None) -> Path:
    """Where ``run_component_tests`` writes component ``name``'s ctest log."""
    return component_build_dir(build_dir, name, source) / "ctest-component-tests.log"


def ctest_tests(tree: Path, cwd: Path) -> dict[str, list[str]]:
    """Every CTest test registered in ``tree``, with its labels (ctest --show-only=json-v1)."""
    done = subprocess.run(["ctest", "--test-dir", str(tree), "--show-only=json-v1"],
                          cwd=cwd, check=True, capture_output=True, text=True)
    tests = {}
    for test in json.loads(done.stdout).get("tests", []):
        labels = next((p["value"] for p in test.get("properties", []) if p.get("name") == "LABELS"), [])
        tests[test["name"]] = list(labels)
    return tests


def run_component_tests(engine: str, *, source_root: Path, build_dir: Path, runner: Runner = run_checked,
                        lister: TestLister = ctest_tests) -> list[str]:
    """Build every component's declared test targets (component.toml ``[test] targets``) in its build tree and run
    exactly those tests with ctest, one log per component. Returns one failure line per component that did not pass.
    The pin-bump gate and PR CI both run this; nothing else runs component tests.

    Before anything runs, the tree's registered tests must match the declaration exactly: every declared target has a
    registered CTest test of the same name, and every registered test is either declared or labelled ``gpu`` (it needs a
    GPU, so it is deliberately not run card-free). Only the declared tests run (``-R`` on their escaped names)."""
    failures = []
    for name, component in sorted(components.load_all(paths.REPO_ROOT, engine).items()):
        if not component.test_targets:
            continue
        declared = list(component.test_targets)
        repeated = sorted({t for t in declared if declared.count(t) > 1})
        if repeated:
            failures.append(f"{name}: test target(s) {', '.join(repeated)} declared more than once in component.toml")
            continue
        tree = component_build_dir(build_dir, name, component.root)
        log = component_test_log(build_dir, name, component.root)
        build = ["cmake", "--build", str(tree), "-j", "--target", *declared]
        try:
            runner(build, source_root)
        except subprocess.CalledProcessError as exc:
            failures.append(f"{name}: building the test targets failed (`{' '.join(build)}` exited {exc.returncode}); "
                            "the build output above says why")
            continue
        try:
            registered = lister(tree, source_root)
        except subprocess.CalledProcessError as exc:
            failures.append(f"{name}: listing the CTest tests in {tree} failed (exit {exc.returncode})")
            continue
        missing = [t for t in declared if t not in registered]
        undeclared = sorted(t for t, labels in registered.items() if t not in declared and GPU_LABEL not in labels)
        if missing or undeclared:
            if missing:
                failures.append(f"{name}: declared test target(s) {', '.join(missing)} have no CTest test of that "
                                f"name in {tree} (add_test NAME must equal the target)")
            if undeclared:
                failures.append(f"{name}: CTest test(s) {', '.join(undeclared)} are registered but neither declared in "
                                f"component.toml [test] targets nor labelled {GPU_LABEL!r}")
            continue
        pattern = "^(" + "|".join(re.escape(t) for t in declared) + ")$"
        try:
            runner(["ctest", "--test-dir", str(tree), "--output-on-failure", "--no-tests=error", "-R", pattern,
                    "--parallel", str(len(os.sched_getaffinity(0))), "--output-log", str(log)], source_root)
        except subprocess.CalledProcessError as exc:
            failures.append(f"{name}: the tests failed (ctest exited {exc.returncode}); log: {log}")
    return failures
