"""radiance's bump gates (MEN10 step f): the component gate for pin-bump's compile phase and the engine's source audit.

radiance has a surface llama.cpp lacks: BigCherry plugin components (``engines/radiance/components/*``, listed under
``[components]`` in engine.toml) that are built against the engine's plugin ABI and internal headers, so a bump can break
them with no patch anchor failing. The compile gate therefore does what ``bigcherry build`` does for the radiance lane --
configures and builds the engine, then ``campaign.runtime_bundle.build_runtime`` builds every declared component into the
build's runtime home -- and then checks what a build cannot prove by itself: that every built plugin reports the ABI
version the bumped source declares, and that each component's offline tests (``[test] targets`` in component.toml) pass.

Everything that needs a real checkout, a toolchain or a Linux host fails cleanly with a message when it is missing; the
subprocess boundaries (``Runner``, ``AbiReader``) are injectable so the offline tests drive the checks with fakes.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Callable

from dataclasses import replace

from ..campaign import build as campaign_build
from ..campaign import runtime_bundle
from ..core import components, engines, paths
from ..core import config as campaign_config
from .base import BumpCheck, BumpCheckError, DriftCheck

if TYPE_CHECKING:
    from ..core.context import ProjectContext
    from ..source.audit import AuditContext

Runner = Callable[[list[str], Path], None]
#: reads ``rad_plugin_abi_version()`` of a built plugin
AbiReader = Callable[[Path], int]

#: the engine's plugin ABI header, relative to the engine source (the components include ``${RADIANCE_SRC}/abi``)
ABI_HEADER = Path("abi") / "rad_abi.h"
ABI_MACRO = "RAD_ABI_VERSION"
#: the engine loads every library in this directory of its runtime home as a plugin and refuses one without the symbol
PLUGIN_DIR = "kernels"
_ABI_DECLARATION = re.compile(rf"^[ \t]*(?:#[ \t]*define[ \t]+{ABI_MACRO}[ \t]+|.*\b{ABI_MACRO}\b[ \t]*=[ \t]*)\(?[ \t]*(\d+)", re.MULTILINE)


def _run(argv: list[str], cwd: Path) -> None:
    subprocess.run(argv, cwd=cwd, check=True)


def expected_abi_version(source_dir: Path) -> int:
    """``RAD_ABI_VERSION`` as the bumped engine source declares it."""
    header = source_dir / ABI_HEADER
    try:
        text = header.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        raise BumpCheckError(f"cannot read the engine's plugin ABI header {header}: {exc}") from exc
    match = _ABI_DECLARATION.search(text)
    if match is None:
        raise BumpCheckError(f"{header} declares no numeric {ABI_MACRO}; the ABI check needs updating for this revision")
    return int(match.group(1))


_READ_ABI = (
    "import ctypes, sys\n"
    "fn = ctypes.CDLL(sys.argv[1]).rad_plugin_abi_version\n"
    "fn.restype = ctypes.c_uint32\n"
    "print(fn())\n"
)


def read_plugin_abi(so: Path) -> int:
    """dlopen the plugin and call its exported ``rad_plugin_abi_version`` -- the engine's own handshake (the helper's
    ``load_plugin`` does the same). ``nm`` could only show the symbol exists, not its value, and the value is what must
    match. Done in a child process so a plugin that crashes on load cannot take pin-bump down with it."""
    try:
        result = subprocess.run([sys.executable, "-c", _READ_ABI, str(so)], capture_output=True, text=True, timeout=120)
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise BumpCheckError(f"cannot load {so} to read its ABI version: {exc}") from exc
    if result.returncode != 0:
        detail = result.stderr.strip().splitlines()[-1] if result.stderr.strip() else f"exit {result.returncode}"
        raise BumpCheckError(f"cannot read rad_plugin_abi_version from {so}: {detail}")
    try:
        return int(result.stdout.strip())
    except ValueError as exc:
        raise BumpCheckError(f"{so}: unexpected rad_plugin_abi_version output {result.stdout.strip()!r}") from exc


#: the gate adds nothing to the platform's options: it builds the engine as the platform alone describes it
_GATE_BUILD = campaign_config.Build(name="bump-compile-gate", options=(), variant_set=None, needs=frozenset())


def _build_platform(engine: str, platform_name: str | None) -> campaign_config.Platform:
    """The recipe ``[platform.*]`` ``bigcherry build`` would configure with, its ``${VAR}`` compilers resolved from the
    environment. ValueError (a clean pin-bump STOP) when none is named, it is unknown, or a compiler variable is unset."""
    recipes = engines.load(paths.REPO_ROOT, engine).recipes_path(paths.REPO_ROOT)
    platforms = campaign_config.load(recipes).platforms
    if platform_name is None:
        raise ValueError(
            f"the {engine} compile gate builds the engine and needs the build platform: pass --platform "
            f"(one of {', '.join(sorted(platforms)) or 'none declared'} in {recipes.name})")
    if platform_name not in platforms:
        raise ValueError(f"unknown --platform {platform_name!r} for {engine}; {recipes.name} declares: {', '.join(sorted(platforms))}")
    platform = platforms[platform_name]
    campaign_config.require_resolved(platform.c_compiler, f"platform.{platform_name}.c-compiler")
    campaign_config.require_resolved(platform.cxx_compiler, f"platform.{platform_name}.cxx-compiler")
    return platform


class ComponentGate:
    """The shared state of the three checks: one build of the bumped engine and its components."""

    def __init__(
        self, context: ProjectContext, source_dir: Path, workdir: Path, runner: Runner, abi_reader: AbiReader,
        platform_name: str | None,
    ):
        self.engine = context.engine
        self.layout = context.layout
        self.source_dir = source_dir
        self.build_dir = workdir / f"{context.engine}-build"
        self.runner = runner
        self.abi_reader = abi_reader
        self.declared = components.load_all(paths.REPO_ROOT, context.engine)
        self.platform = _build_platform(context.engine, platform_name) if self.declared else None
        self.built = False

    @property
    def home(self) -> Path:
        return self.build_dir / self.layout.runtime_home

    def build(self) -> list[str]:
        """Configure and build the engine for the components' GPU targets, then build the components (runtime_bundle)."""
        if not self.declared:
            return []
        platform = self.platform
        targets = sorted({*platform.targets, *(t for component in self.declared.values() for t in component.targets)})
        # The configure argv is the one `bigcherry build` makes for this platform (its compilers, options and targets);
        # the components' own targets are added so every declared component can build.
        argv = campaign_build.cmake_configure_args(
            _GATE_BUILD, replace(platform, targets=tuple(targets)), self.source_dir, self.build_dir,
            c_compiler=platform.c_compiler, cxx_compiler=platform.cxx_compiler, targets_option=self.layout.targets_option)
        try:
            self.runner(argv, self.source_dir)
            self.runner(["cmake", "--build", str(self.build_dir), "-j"], self.source_dir)
            runtime_bundle.build_runtime(self.engine, source_root=self.source_dir, build_dir=self.build_dir,
                                         c_compiler=platform.c_compiler, cxx_compiler=platform.cxx_compiler, runner=self.runner)
        except subprocess.CalledProcessError as exc:
            return [f"`{' '.join(map(str, exc.cmd))}` exited {exc.returncode} (build tree {self.build_dir})"]
        except (runtime_bundle.RuntimeBundleError, OSError) as exc:
            return [str(exc)]
        self.built = True
        return []

    def _require_built(self) -> None:
        if not self.built:
            raise BumpCheckError("the engine and components were not built (the components-build check did not pass)")

    def plugin_abi(self) -> list[str]:
        """Every plugin in the runtime home (the engine's own and the components') must report the declared ABI."""
        if not self.declared:
            return []
        self._require_built()
        expected = expected_abi_version(self.source_dir)
        plugins = sorted((self.home / PLUGIN_DIR).glob("*.so"))
        if not plugins:
            raise BumpCheckError(f"no plugin libraries in {self.home / PLUGIN_DIR}")
        for output in (o for component in self.declared.values() for o in component.outputs):
            if output.startswith(f"{PLUGIN_DIR}/") and not (self.home / output).is_file():
                raise BumpCheckError(f"declared plugin {output} is missing from {self.home}")
        failures = []
        for so in plugins:
            actual = self.abi_reader(so)
            if actual != expected:
                failures.append(f"{so.name}: rad_plugin_abi_version() is {actual}, the engine source declares {ABI_MACRO} {expected}")
        return failures

    def component_tests(self) -> list[str]:
        """Build each component's declared test targets and run them with ctest in the component's build tree."""
        self._require_built()
        failures = []
        for name, component in sorted(self.declared.items()):
            if not component.test_targets:
                continue
            tree = runtime_bundle.component_build_dir(self.build_dir, name)
            try:
                self.runner(["cmake", "--build", str(tree), "-j", "--target", *component.test_targets], self.source_dir)
                self.runner(["ctest", "--test-dir", str(tree), "--output-on-failure"], self.source_dir)
            except subprocess.CalledProcessError as exc:
                failures.append(f"{name}: `{' '.join(map(str, exc.cmd))}` exited {exc.returncode}")
        return failures


def compile_checks(
    context: ProjectContext, source_dir: Path, workdir: Path, *, platform_name: str | None, runner: Runner = _run,
    abi_reader: AbiReader = read_plugin_abi,
) -> tuple[BumpCheck, ...]:
    gate = ComponentGate(context, source_dir, workdir, runner, abi_reader, platform_name)
    cmake_hint = "install cmake and ninja (Ubuntu/Debian: sudo apt-get install cmake ninja-build) on a host with the ROCm toolchain"
    return (
        BumpCheck("components-build", "COMPONENT_BUILD_FAILED", ("cmake", "ninja"), cmake_hint, gate.build),
        BumpCheck("component-plugin-abi", "PLUGIN_ABI_MISMATCH", (), "", gate.plugin_abi),
        BumpCheck("component-tests", "COMPONENT_TEST_FAILED", ("ctest",), cmake_hint, gate.component_tests),
    )


# ---------------------------------------------------------------------------------------------------------- drift

#: a component that turns the engine's own sources into another form carries this script (libr3: libr4d -> gfx11 forms)
REWRITE_SCRIPT = "r3_rewrite.py"
REWRITE_RULES = "rewrites.txt"
REWRITE_NATIVE = "native"
REWRITE_SOURCE = "libr4d"
NATIVE_REWRITE_DRIFT = "NATIVE_REWRITE_DRIFT"
RewriteRunner = Callable[[list[str]], "subprocess.CompletedProcess[str]"]


def _run_rewrite(argv: list[str]) -> "subprocess.CompletedProcess[str]":
    return subprocess.run(argv, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=600)


def rewrite_drift(component: components.Component, source_dir: Path, runner: RewriteRunner = _run_rewrite) -> dict[str, list[str]]:
    """The component's rewrite script in its no-write mode against ``source_dir``: failing rules and blocks by file name,
    prefixed with the component. BumpCheckError when the script cannot run or does not answer with the JSON document."""
    script = component.root / REWRITE_SCRIPT
    argv = [sys.executable, str(script), "--check", "--source", str(source_dir / REWRITE_SOURCE),
            "--rules", str(component.root / REWRITE_RULES)]
    if (component.root / REWRITE_NATIVE).is_dir():
        argv += ["--native", str(component.root / REWRITE_NATIVE)]
    try:
        result = runner(argv)
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise BumpCheckError(f"{component.name}: cannot run {script}: {exc}") from exc
    try:
        document = json.loads(result.stdout)
        failures = document["failures"]
    except (json.JSONDecodeError, KeyError, TypeError) as exc:
        detail = (result.stderr or result.stdout).strip()[-400:]
        raise BumpCheckError(f"{component.name}: {REWRITE_SCRIPT} --check exited {result.returncode} without a failure report: {detail}") from exc
    if (result.returncode == 0) != (not failures):
        raise BumpCheckError(f"{component.name}: {REWRITE_SCRIPT} --check exited {result.returncode} but reported {len(failures)} failing file(s)")
    return {f"{component.name}/{name}": list(problems) for name, problems in sorted(failures.items())}


def source_drift_checks(context: ProjectContext, source_dir: Path, runner: RewriteRunner = _run_rewrite) -> tuple[DriftCheck, ...]:
    """One drift check per declared component that rewrites the engine's sources (it carries r3_rewrite.py)."""
    declared = components.load_all(paths.REPO_ROOT, context.engine)
    checks = []
    for name, component in sorted(declared.items()):
        if not (component.root / REWRITE_SCRIPT).is_file():
            continue

        def run(component: components.Component = component) -> dict[str, list[str]]:
            return rewrite_drift(component, source_dir, runner)

        checks.append(DriftCheck(
            f"{name}-native-rewrite", NATIVE_REWRITE_DRIFT,
            f"re-port the failing blocks in engines/{context.engine}/components/{name} (rewrites.txt, native/*.rw) to the bumped "
            f"{REWRITE_SOURCE} sources, or run `python {component.root / REWRITE_SCRIPT} --check ...` to list them again", run))
    return tuple(checks)


# ------------------------------------------------------------------------------------------------------------ audit

def check_overlay_sync(ctx: AuditContext) -> None:
    """The engine's tracked overlay against the checkout, as for any engine; an engine with no overlay has nothing to sync."""
    from ..source import audit

    if not ctx.overlay_root.is_dir():
        ctx.ok("overlay.vendor_sync", f"no overlay directory {ctx.overlay_root}: nothing to mirror onto the checkout")
        return
    audit.check_overlay_sync(ctx)


def check_patch_state(ctx: AuditContext) -> None:
    """No registered patch is half-applied: for each patch file, either every edit's guard matches the checkout or none."""
    from ..patch import registry as patch_registry

    root = paths.patches_root(ctx.engine)
    registry = patch_registry.load_registry(ctx.engine, root)
    applied, unapplied, partial, missing = [], [], [], []
    for descriptor in registry.descriptors:
        for file_patch in patch_registry.load_implementation(ctx.engine, descriptor, root=root):
            target = ctx.root / file_patch.path
            if not target.is_file():
                (unapplied if file_patch.create else missing).append(f"{descriptor.patch_id}:{file_patch.path}")
                continue
            text = target.read_text(encoding="utf-8", errors="replace")
            hits = [bool(re.search(edit.guard_pattern(), text, re.MULTILINE)) for edit in file_patch.edits]
            if all(hits):
                applied.append(descriptor.patch_id)
            elif any(hits):
                partial.append(f"{descriptor.patch_id}:{file_patch.path}")
            else:
                unapplied.append(descriptor.patch_id)
    if partial or missing:
        ctx.fail(
            "patches.application_state",
            "patch edits are partly applied or target files the tree does not have: " + ", ".join(sorted(set(partial + missing))),
            expected="every patch fully applied or not applied at all, on files that exist", actual=sorted(set(partial + missing)))
    else:
        ctx.ok("patches.application_state",
               f"{len(set(applied))} patch(es) fully applied, {len(set(unapplied))} not applied, none partial")


def check_components(ctx: AuditContext) -> None:
    """Each declared component's source is present and loadable, and the engine's plugin ABI header can be read."""
    declared = components.load_all(paths.REPO_ROOT, ctx.engine)
    missing = []
    for name, component in sorted(declared.items()):
        if not (component.root / "CMakeLists.txt").is_file():
            missing.append(f"{name}: no CMakeLists.txt in {component.root}")
    if missing:
        ctx.fail("components.source_present", "; ".join(missing), expected="a buildable source directory per component", actual=missing)
    else:
        ctx.ok("components.source_present", f"{len(declared)} component(s) with source: {', '.join(sorted(declared)) or 'none'}")
    if not declared:
        return
    try:
        version = expected_abi_version(ctx.root)
    except BumpCheckError as exc:
        ctx.fail("components.abi_header", str(exc))
    else:
        ctx.ok("components.abi_header", f"{ABI_HEADER.as_posix()} declares {ABI_MACRO} {version}")


def audit_checks(context: ProjectContext):
    return (check_overlay_sync, check_patch_state, check_components)
