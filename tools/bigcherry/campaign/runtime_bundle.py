"""The runtime a served build needs beside its binary (RAD01 PR 8b).

Some engines load plugins at run time (radiance: kernel libraries from ``RADIANCE_HOME``). After the lane builds the
engine binary, this builds the engine's declared runtime targets in the same build tree, then each component the
engine declares (core/components.py): configured against the engine installed from this build and the lane's own
materialised source, built, and its outputs copied into the build's runtime home. An engine without runtime targets
or components (llama.cpp) gets nothing here.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path
from typing import Callable

from ..core import components, engines, paths
from ..core import config as campaign_config
from ..core.process import run_checked
from .build import rocm_configure_options

Runner = Callable[[list[str], Path], None]
#: (component build tree, cwd) -> every CTest test registered there, by name, with its labels
TestLister = Callable[[Path, Path], dict[str, list[str]]]
#: the CTest label a test that needs a GPU carries; such a test is registered but never declared card-free
GPU_LABEL = "gpu"


def component_build_dir(build_dir: Path, name: str) -> Path:
    """Where ``build_runtime`` configures and builds component ``name``."""
    return build_dir / "components" / name


class RuntimeBundleError(RuntimeError):
    """A runtime target or component did not build, or did not produce a declared output."""


def build_runtime(
    engine: str,
    *,
    source_root: Path,
    build_dir: Path,
    platform: campaign_config.Platform,
    runner: Runner = run_checked,
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
    launcher_options = tuple((k, v) for k, v in platform.options if k.endswith("_COMPILER_LAUNCHER"))
    built: dict[str, str] = {}
    if layout.runtime_targets:
        runner(["cmake", "--build", str(build_dir), "-j", "--target", *layout.runtime_targets], source_root)
    declared = components.load_all(paths.REPO_ROOT, engine)
    if not declared:
        return built
    if not layout.runtime_home:
        raise RuntimeBundleError(f"engine {engine!r} declares components but no build.runtime-home")
    home = build_dir / layout.runtime_home
    # Components run the same ROCm lookups as the engine, in their own configure: pin them to the same SDK.
    rocm_options = rocm_configure_options(platform, layout.rocm_paths)
    prefix = build_dir / "runtime-prefix"
    if prefix.exists():
        shutil.rmtree(prefix)
    runner(["cmake", "--install", str(build_dir), "--prefix", str(prefix)], source_root)
    for name, component in sorted(declared.items()):
        component_build = component_build_dir(build_dir, name)
        values = {"engine_source": str(source_root), "targets": ";".join(component.targets)}
        options = {key: value.format(**values) for key, value in component.configure}
        for key, value in launcher_options:
            options.setdefault(key, value)
        options.update(rocm_options)
        options["CMAKE_PREFIX_PATH"] = str(prefix)
        if c_compiler:
            options["CMAKE_C_COMPILER"] = c_compiler
        if cxx_compiler:
            options["CMAKE_CXX_COMPILER"] = cxx_compiler
        runner(["cmake", "-S", str(component.root), "-B", str(component_build), "-G", "Ninja",
                *(f"-D{key}={value}" for key, value in sorted(options.items()))], source_root)
        runner(["cmake", "--build", str(component_build), "-j"], source_root)
        for output in component.outputs:
            produced = sorted(component_build.rglob(Path(output).name))
            if not produced:
                raise RuntimeBundleError(f"component {name} did not produce {output}")
            target = home / output
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(produced[0], target)
        built[name] = component.content_digest()
    return built


def component_test_log(build_dir: Path, name: str) -> Path:
    """Where ``run_component_tests`` writes component ``name``'s ctest log."""
    return component_build_dir(build_dir, name) / "ctest-component-tests.log"


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
        tree = component_build_dir(build_dir, name)
        log = component_test_log(build_dir, name)
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
                    "--output-log", str(log)], source_root)
        except subprocess.CalledProcessError as exc:
            failures.append(f"{name}: the tests failed (ctest exited {exc.returncode}); log: {log}")
    return failures
