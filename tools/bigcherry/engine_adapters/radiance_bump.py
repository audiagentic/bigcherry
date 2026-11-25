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
from ..core import component_coupling, components, engines, paths
from ..core import config as campaign_config
from ..core.process import run_checked
from .base import BumpCheck, BumpCheckError, DriftCheck

if TYPE_CHECKING:
    from ..core.context import ProjectContext
    from ..source.audit import AuditContext

Runner = Callable[[list[str], Path], None]
#: reads ``rad_plugin_abi_version()`` and ``rad_plugin_abi_layout()`` of a built plugin
AbiReader = Callable[[Path], "tuple[int, int | None]"]

#: the engine's plugin ABI header, relative to the engine source (the components include ``${RADIANCE_SRC}/abi``)
ABI_HEADER = Path("abi") / "rad_abi.h"
ABI_MACRO = "RAD_ABI_VERSION"
#: the engine loads every library in this directory of its runtime home as a plugin and refuses one without the symbol
PLUGIN_DIRS = ("kernels", "architectures", "quantizers")
#: the engine's own kernel plugin, built from the engine's abi/ headers: its layout fingerprint is the reference
REFERENCE_PLUGIN = "libr4d.so"
_ABI_DECLARATION = re.compile(rf"^[ \t]*(?:#[ \t]*define[ \t]+{ABI_MACRO}[ \t]+|.*\b{ABI_MACRO}\b[ \t]*=[ \t]*)\(?[ \t]*(\d+)", re.MULTILINE)


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


# Lazy binding: a plugin may call functions the radiance executable exports (libr3moe uses rad_event_*), which are
# absent outside the engine. ctypes.CDLL always binds now and refuses such a plugin; dlopen with RTLD_LAZY resolves a
# function only when it is first called, and rad_plugin_abi_version calls none.
_READ_ABI = (
    "import ctypes, os, sys\n"
    "libc = ctypes.CDLL(None)\n"
    "libc.dlopen.restype = ctypes.c_void_p; libc.dlopen.argtypes = [ctypes.c_char_p, ctypes.c_int]\n"
    "libc.dlsym.restype = ctypes.c_void_p; libc.dlsym.argtypes = [ctypes.c_void_p, ctypes.c_char_p]\n"
    "libc.dlerror.restype = ctypes.c_char_p\n"
    "h = libc.dlopen(sys.argv[1].encode(), os.RTLD_LAZY | os.RTLD_LOCAL)\n"
    "if not h: sys.exit((libc.dlerror() or b'dlopen failed').decode())\n"
    "p = libc.dlsym(h, b'rad_plugin_abi_version')\n"
    "if not p: sys.exit('no rad_plugin_abi_version export')\n"
    "q = libc.dlsym(h, b'rad_plugin_abi_layout')\n"
    "print(ctypes.CFUNCTYPE(ctypes.c_uint32)(p)(), ctypes.CFUNCTYPE(ctypes.c_uint64)(q)() if q else '-')\n"
)


def read_plugin_abi(so: Path) -> tuple[int, int | None]:
    """dlopen the plugin and call its exported ``rad_plugin_abi_version`` -- the engine's own handshake (the helper's
    ``load_plugin`` does the same). ``nm`` could only show the symbol exists, not its value, and the value is what must
    match. Done in a child process so a plugin that crashes on load cannot take pin-bump down with it. Loaded lazily
    (see _READ_ABI) so a plugin that uses functions only the engine executable exports can still be read. Returns the
    ABI version and the ``rad_plugin_abi_layout`` fingerprint (None when the plugin does not export one)."""
    try:
        result = subprocess.run([sys.executable, "-c", _READ_ABI, str(so)], capture_output=True, text=True, timeout=120)
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise BumpCheckError(f"cannot load {so} to read its ABI version: {exc}") from exc
    if result.returncode != 0:
        detail = result.stderr.strip().splitlines()[-1] if result.stderr.strip() else f"exit {result.returncode}"
        raise BumpCheckError(f"cannot read rad_plugin_abi_version from {so}: {detail}")
    try:
        version, layout = result.stdout.split()
        return int(version), None if layout == "-" else int(layout)
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
        platform_name: str | None, lister: runtime_bundle.TestLister,
    ):
        self.engine = context.engine
        self.layout = context.layout
        self.source_dir = source_dir
        self.build_dir = workdir / f"{context.engine}-build"
        self.runner = runner
        self.abi_reader = abi_reader
        self.lister = lister
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
        try:
            argv = campaign_build.cmake_configure_args(
                _GATE_BUILD, replace(platform, targets=tuple(targets)), self.source_dir, self.build_dir,
                c_compiler=platform.c_compiler, cxx_compiler=platform.cxx_compiler, targets_option=self.layout.targets_option,
                rocm_paths=self.layout.rocm_paths, engine_options=self.layout.repo_path_options(paths.REPO_ROOT))
            self.runner(argv, self.source_dir)
            self.runner(["cmake", "--build", str(self.build_dir), "-j"], self.source_dir)
            # RAD18: forward the platform's compiler-launcher options (e.g. ccache) into each component's own
            # separate cmake configure -- without this, a declared component (libr3's 92-99 HIP "units" among
            # them) never got ccache at all and recompiled from scratch on every single build.
            runtime_bundle.build_runtime(self.engine, source_root=self.source_dir, build_dir=self.build_dir,
                                         platform=platform, runner=self.runner)
        except subprocess.CalledProcessError as exc:
            return [f"`{' '.join(map(str, exc.cmd))}` exited {exc.returncode} (build tree {self.build_dir})"]
        except (campaign_build.CampaignBuildError, runtime_bundle.RuntimeBundleError, OSError) as exc:
            return [str(exc)]
        self.built = True
        return []

    def _require_built(self) -> None:
        if not self.built:
            raise BumpCheckError("the engine and components were not built (the components-build check did not pass)")

    def plugin_abi(self) -> list[str]:
        """Every plugin in the runtime home (the engine's own and the components') must report the declared ABI version
        and export the struct-layout fingerprint of the engine's own kernel plugin (libr4d, built from the same abi/
        headers); the engine's loader refuses a plugin with no fingerprint or a different one."""
        if not self.declared:
            return []
        self._require_built()
        expected = expected_abi_version(self.source_dir)
        plugins = sorted(
            so for directory in PLUGIN_DIRS for so in (self.home / directory).glob("*.so")
        )
        if not plugins:
            searched = ", ".join(str(self.home / directory) for directory in PLUGIN_DIRS)
            raise BumpCheckError(f"no plugin libraries in {searched}")
        for output in (o for component in self.declared.values() for o in component.outputs):
            if output.split("/", 1)[0] in PLUGIN_DIRS and not (self.home / output).is_file():
                raise BumpCheckError(f"declared plugin {output} is missing from {self.home}")
        reference = self.home / "kernels" / REFERENCE_PLUGIN
        if not reference.is_file():
            raise BumpCheckError(f"the engine's own plugin {REFERENCE_PLUGIN} is missing from {self.home / 'kernels'}")
        _, expected_layout = self.abi_reader(reference)
        if expected_layout is None:
            raise BumpCheckError(f"{REFERENCE_PLUGIN} exports no rad_plugin_abi_layout; the ABI check needs updating for this revision")
        failures = []
        for so in plugins:
            label = so.relative_to(self.home).as_posix()
            version, layout = self.abi_reader(so)
            if version != expected:
                failures.append(f"{label}: rad_plugin_abi_version() is {version}, the engine source declares {ABI_MACRO} {expected}")
            if layout is None:
                failures.append(f"{label}: exports no rad_plugin_abi_layout (add RAD_PLUGIN_ABI_LAYOUT_EXPORT, rad_abi_layout.h)")
            elif layout != expected_layout:
                failures.append(f"{label}: rad_plugin_abi_layout() is {layout:#018x}, {REFERENCE_PLUGIN} reports {expected_layout:#018x}")
        return failures

    def component_tests(self) -> list[str]:
        """Each component's declared test targets, built and run by runtime_bundle.run_component_tests (the same run
        PR CI makes)."""
        self._require_built()
        return runtime_bundle.run_component_tests(self.engine, source_root=self.source_dir, build_dir=self.build_dir,
                                                  runner=self.runner, lister=self.lister)


COMPOSITION_COMPILE_FAILED = "COMPOSITION_COMPILE_FAILED"
COMPOSITION_ARCHES = ("gfx1100", "gfx1201")


class CompositionGates:
    """RCS19 gates 1 and 2a on the patched tree, before the lane build: the components' rewrite check against the patched
    source, then a syntax-only host and device compile of the units the production patches touch, both architectures
    (patch/compose_check.py: the same code `patch-compose-check` runs)."""

    def __init__(self, context: ProjectContext, source_dir: Path) -> None:
        self.context, self.source_dir = context, source_dir

    def anchors(self) -> list[str]:
        from ..patch import compose_check

        return [f"{name}: {problem}" for name, problems in sorted(compose_check.gate1(self.context.engine, self.source_dir).items())
                for problem in problems]

    def syntax(self) -> list[str]:
        from ..patch import compose_check

        engine = self.context.engine
        if compose_check.compiler() is None:
            raise BumpCheckError("no ROCm clang++ (set ROCM_PATH): the composition compile gate cannot run")
        source = engines.load(paths.REPO_ROOT, engine).promotion_source
        ids = compose_check.production_ids(engine, source) if source else []
        report = compose_check.Report()
        compose_check.compile_gate(engine, self.source_dir, compose_check.touched_files(engine, list(ids)),
                                   list(COMPOSITION_ARCHES), "syntax", report)
        return list(report.compile_failures)


def compile_checks(
    context: ProjectContext, source_dir: Path, workdir: Path, *, platform_name: str | None, runner: Runner = run_checked,
    abi_reader: AbiReader = read_plugin_abi, lister: runtime_bundle.TestLister = runtime_bundle.ctest_tests,
) -> tuple[BumpCheck, ...]:
    gate = ComponentGate(context, source_dir, workdir, runner, abi_reader, platform_name, lister)
    composition = CompositionGates(context, source_dir)
    cmake_hint = "install cmake and ninja (Ubuntu/Debian: sudo apt-get install cmake ninja-build) on a host with the ROCm toolchain"
    return (
        BumpCheck("composition-anchors", NATIVE_REWRITE_DRIFT, (), "", composition.anchors),
        BumpCheck("composition-syntax", COMPOSITION_COMPILE_FAILED, (), "", composition.syntax),
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
ORIGIN_BLOCKED = "OWNED_ORIGIN_BLOCKED"
ORIGIN_CHANGED = "OWNED_ORIGIN_CHANGED"
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

    def coupling_run() -> dict[str, list[str]]:
        if not source_dir.is_dir():
            raise BumpCheckError(f"component coupling: source tree is unavailable: {source_dir}")
        upstream_paths = {
            path.relative_to(source_dir).as_posix()
            for path in source_dir.rglob("*") if path.is_file()
        }
        abi = expected_abi_version(source_dir)
        return {
            f"{name}/component.toml": faults
            for name, component in sorted(declared.items())
            if (faults := component_coupling.problems(
                component, upstream_paths=upstream_paths, upstream_abi=abi,
                patch_root=context.patches_root
            ))
        }

    def origins(state: str) -> Callable[[], dict[str, list[str]]]:
        def run() -> dict[str, list[str]]:
            from ..release import pin_triage

            try:
                target = subprocess.run(["git", "-C", str(source_dir), "rev-parse", "HEAD"], capture_output=True, text=True,
                                        check=True).stdout.strip()
                found = pin_triage.origin_drift(context.engine, context.upstream_repo, target)
            except (OSError, subprocess.CalledProcessError, pin_triage.PinTriageError) as exc:
                raise BumpCheckError(f"owned-unit origins: cannot read the upstream history: {exc}") from exc
            return {f"{item['component']}/{item['unit']}": [item["reason"], *item["commits"]]
                    for item in found if item["state"] == state}
        return run

    checks = [DriftCheck(
        "component-coupling", "COMPONENT_COUPLING_DRIFT",
        "update owned [coupling] declarations and the consumer for this upstream pin, then rerun pin-bump",
        coupling_run,
    )]
    for name, component in sorted(declared.items()):
        if not (component.root / REWRITE_SCRIPT).is_file():
            continue

        def run(component: components.Component = component) -> dict[str, list[str]]:
            return rewrite_drift(component, source_dir, runner)

        checks.append(DriftCheck(
            f"{name}-native-rewrite", NATIVE_REWRITE_DRIFT,
            f"re-port the failing blocks in engines/{context.engine}/components/{name} (rewrites.txt, native/*.rw) to the bumped "
            f"{REWRITE_SOURCE} sources, or run `python {component.root / REWRITE_SCRIPT} --check ...` to list them again", run))
    checks.extend((DriftCheck(
        "owned-origins-blocked", ORIGIN_BLOCKED,
        "an owned unit's origin cannot be checked or has gone (missing commit, wrong path, rename or removal, target not a descendant): "
        "repair its entry in the component's origins.toml or port the unit to upstream's new name, then rerun the bump",
        origins("blocked"),
    ), DriftCheck(
        "owned-origins-changed", ORIGIN_CHANGED,
        "upstream changed an owned unit (or the extracted header): review the listed commits against the owned source (a three-way merge "
        "for a whole unit, a structural review for an extracted family), re-prove it with the unit harness, then set last-integrated in "
        "origins.toml to the target commit by hand; the bump never advances it",
        origins("candidate"),
    )))
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
