"""The runtime a served build needs beside its binary (RAD01 PR 8b).

Some engines load plugins at run time (radiance: kernel libraries from ``RADIANCE_HOME``). After the lane builds the
engine binary, this builds the engine's declared runtime targets in the same build tree, then each component the
engine declares (core/components.py): configured against the engine installed from this build and the lane's own
materialised source, built, and its outputs copied into the build's runtime home. An engine without runtime targets
or components (llama.cpp) gets nothing here.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import Callable

from ..core import components, engines, paths

Runner = Callable[[list[str], Path], None]


def _run(argv: list[str], cwd: Path) -> None:
    subprocess.run(argv, cwd=cwd, check=True)


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
    c_compiler: str | None,
    cxx_compiler: str | None,
    launcher_options: tuple[tuple[str, str], ...] = (),
    runner: Runner = _run,
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
    built: dict[str, str] = {}
    if layout.runtime_targets:
        runner(["cmake", "--build", str(build_dir), "-j", "--target", *layout.runtime_targets], source_root)
    declared = components.load_all(paths.REPO_ROOT, engine)
    if not declared:
        return built
    if not layout.runtime_home:
        raise RuntimeBundleError(f"engine {engine!r} declares components but no build.runtime-home")
    home = build_dir / layout.runtime_home
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
