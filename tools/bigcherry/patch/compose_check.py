"""Compose check (RCS19 gates 1 and 2): does a patch composition still build on every architecture?

A patch that edits a shared source can break what other components rewrite or compile from the same text. Matching an
anchor once proves little: two edits can both match and still define a type twice, repeat a template default or clash in
device code only. This materialises ONE composition (the production patch set of an engine plus optional extra patch
ids, at the pinned revision) and runs, on that tree:

  gate 1   the components' rewrite check (r3_rewrite.py --check): every rule and native block still finds its text;
  gate 2a  a syntax-only compile (clang -fsyntax-only, host pass and device pass) of every unit the patches touch,
           for each architecture: libr4d's own source for gfx12, libr3's rewritten copy for gfx11;
  gate 2b  optionally, the real compile of those units (code generation, no link).

2a needs seconds per unit; 2b minutes. Without the ROCm compiler the compile gates report SKIPPED, never a pass.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

from ..core import components, engines, paths

SOURCE_DIR = "libr4d"
UNIT_LINE = re.compile(r'^\s*"(r4d_[A-Za-z0-9_]+)\|([^"]*)"')
INCLUDE = re.compile(r'^\s*#\s*include\s+"([^"]+)"', re.M)


class ComposeError(RuntimeError):
    pass


@dataclass(frozen=True)
class Arch:
    name: str
    #: gfx11 units are compiled from libr3's rewritten copy of libr4d; gfx12 units from libr4d itself
    rewritten: bool


ARCHES = {"gfx1100": Arch("gfx1100", True), "gfx1201": Arch("gfx1201", False)}


@dataclass
class Report:
    gate1: dict[str, list[str]] = field(default_factory=dict)
    compile_failures: list[str] = field(default_factory=list)
    checked: list[str] = field(default_factory=list)
    skipped: str | None = None

    @property
    def ok(self) -> bool:
        return not self.gate1 and not self.compile_failures


# ------------------------------------------------------------------------------------------------------ composition
def materialize_tree(engine: str, patch_ids: list[str], out: Path, *,
                     vendor_root: Path | None = None) -> dict[str, list[str]]:
    """Write the pristine pinned source plus the patches, in dependency order, into ``out``. Returns, per patch id, the
    files it edits or creates (relative to the source root)."""
    from . import patchset, pinned_source
    from . import registry as patch_registry
    from . import apply as patcher

    pristine = pinned_source.pinned_checkout(engine, vendor_root=vendor_root)
    shutil.copytree(pristine, out, symlinks=True, dirs_exist_ok=True)
    from ..core.context import ProjectContext
    patches_root = ProjectContext.resolve(engine=engine).patches_root
    selection = patchset.resolve_exact(engine, patch_ids, directory=patches_root, required_state=None)
    registry = patch_registry.load_registry(engine, patches_root)
    loaded, touched = [], {}
    for module in selection.modules:
        descriptor = registry.get(module.patch_id)
        for patch in patch_registry.load_implementation(engine, descriptor, root=patches_root):
            loaded.append(patch)
            touched.setdefault(module.patch_id, []).append(patch.path)
    results = patcher.apply_all(loaded, out)
    bad = [r for r in results if not r.ok]
    if bad:
        raise ComposeError("composition does not apply:\n" + patcher.format_results(results))
    return {patch_id: sorted(set(files)) for patch_id, files in touched.items()}


def production_ids(engine: str, source_name: str, experiment: str | None = None) -> list[str]:
    """The patch ids of an engine's named source (and optionally one experiment) in the campaign's own resolution."""
    from ..campaign import source as campaign_source
    from ..core import config as campaign_config
    from ..core.context import ProjectContext
    from . import patchset

    context = ProjectContext.resolve(engine=engine)
    plan = campaign_source.source_plan_for(
        engine, campaign_config.load(context.config_path), source_name,
        catalog=patchset.catalog(engine, directory=context.patches_root),
        catalog_directory=context.patches_root, experiment=experiment)
    return list(plan.patch_ids)


def touched_files(engine: str, patch_ids: list[str]) -> list[str]:
    """Files the named patches edit or create (relative to the source root), without applying anything."""
    from . import patchset
    from . import registry as patch_registry
    from ..core.context import ProjectContext

    patches_root = ProjectContext.resolve(engine=engine).patches_root
    selection = patchset.resolve_exact(engine, patch_ids, directory=patches_root, required_state=None)
    registry = patch_registry.load_registry(engine, patches_root)
    return sorted({patch.path for module in selection.modules
                   for patch in patch_registry.load_implementation(engine, registry.get(module.patch_id), root=patches_root)})


# ------------------------------------------------------------------------------------------------------ units
def units(tree: Path) -> dict[str, str]:
    """unit name -> its extra compile flags, as libr4d's build file lists them."""
    found = {}
    for line in (tree / SOURCE_DIR / "CMakeLists.txt").read_text(encoding="utf-8").splitlines():
        match = UNIT_LINE.match(line)
        if match:
            found[match.group(1)] = match.group(2).strip()
    return found


def _includes(path: Path, root: Path, seen: set[Path]) -> set[Path]:
    if path in seen or not path.is_file():
        return seen
    seen.add(path)
    for name in INCLUDE.findall(path.read_text(encoding="utf-8", errors="replace")):
        candidate = (path.parent / name)
        if not candidate.is_file():
            candidate = root / SOURCE_DIR / name
        if candidate.is_file():
            _includes(candidate.resolve(), root, seen)
    return seen


def affected_units(tree: Path, touched: list[str]) -> list[str]:
    """The units whose source, directly or through a local include, is one of the files the patches edit."""
    touched_paths = {(tree / rel).resolve() for rel in touched if rel.startswith(SOURCE_DIR + "/")}
    result = []
    for unit in sorted(units(tree)):
        source = tree / SOURCE_DIR / (unit + ".hip")
        if not source.is_file():
            source = tree / SOURCE_DIR / (unit + ".cpp")
        if source.is_file() and (_includes(source.resolve(), tree, set()) & touched_paths):
            result.append(unit)
    return result


# ------------------------------------------------------------------------------------------------------ gates
def compiler(engine: str) -> Path | None:
    """Find the same named ROCm SDK as the recipe, never a generic PATH clang.

    An explicit ROCM_PATH selects the SDK for an already-resolved build environment.
    For a standalone check, a unique SDK root from the engine's platforms suffices.
    Missing or ambiguous SDK configuration is a skipped compile, not a fallback.
    """
    base = os.environ.get("ROCM_PATH")
    if not base:
        from ..core import config
        from ..core.context import ProjectContext
        try:
            recipes = config.load(ProjectContext.resolve(engine=engine).config_path)
        except (OSError, ValueError):
            return None
        roots = {platform.sdk_root for platform in recipes.platforms.values()
                 if platform.sdk_root}
        if len(roots) != 1:
            return None
        base = roots.pop()
    candidate = Path(base) / "lib" / "llvm" / "bin" / "clang++"
    return candidate if candidate.is_file() else None


def gate1(engine: str, tree: Path) -> dict[str, list[str]]:
    from ..engine_adapters import radiance_bump

    failures: dict[str, list[str]] = {}
    for component in components.load_all(paths.REPO_ROOT, engine).values():
        if (component.root / radiance_bump.REWRITE_SCRIPT).is_file():
            failures.update(radiance_bump.rewrite_drift(component, tree))
    return failures


def _rewritten_copy(engine: str, tree: Path, into: Path) -> Path:
    component = components.load_all(paths.REPO_ROOT, engine)["libr3"]
    argv = [sys.executable, str(component.root / "r3_rewrite.py"), "--source", str(tree / SOURCE_DIR),
            "--rules", str(component.root / "rewrites.txt"), "--out", str(into)]
    if (component.root / "native").is_dir():
        argv += ["--native", str(component.root / "native")]
    done = subprocess.run(argv, capture_output=True, text=True)
    if done.returncode != 0:
        raise ComposeError("libr3's rewrite of the composition failed:\n" + (done.stderr or done.stdout)[-1500:])
    return into


def _command(clang: Path, arch: Arch, engine: str, tree: Path, rewritten: Path | None, unit: str, flags: str,
             *, mode: str, side: str, scratch: Path) -> list[str]:
    source_dir = (rewritten if arch.rewritten else tree / SOURCE_DIR)
    file = source_dir / (unit + ".hip")
    if not file.is_file():
        file = source_dir / (unit + ".cpp")
    if arch.rewritten:
        # libr3 compiles a wrapper that includes the compatibility header first, then the rewritten unit
        wrapper = source_dir.parent / "units" / (unit + ".hip")
        wrapper.parent.mkdir(parents=True, exist_ok=True)
        wrapper.write_text(f'#include "r3_compat.h"\n#include "../{source_dir.name}/{file.name}"\n', encoding="utf-8")
        file = wrapper
    argv = [str(clang), "-x", "hip", f"--offload-arch={arch.name}", "-std=gnu++17", "-ffp-contract=off",
            "-D__HIP_PLATFORM_AMD__=1", "-D__HIP_ROCclr__=1", "-I", str(source_dir), "-I", str(tree / "abi")]
    if arch.rewritten:
        component = components.load_all(paths.REPO_ROOT, engine)["libr3"]
        argv += ["-DR3_TARGETS=\"gfx1100\"", "-DRAD_HAVE_HIP=1", "-I", str(component.root),
                 "-I", str(component.root.parent / "serving")]
    argv += flags.split()
    argv += {"host": ["--cuda-host-only"], "device": ["--cuda-device-only"]}[side]
    if mode == "syntax":
        argv += ["-fsyntax-only"]
    else:
        argv += ["-O3", "-c", "-o", str(scratch / f"{arch.name}-{unit}-{side}.o")]
    return argv + [str(file)]


def skipped_by_libr3(engine: str) -> set[str]:
    """Units libr3 leaves to the upstream plugin (exclude.txt): they are not compiled for gfx11."""
    component = components.load_all(paths.REPO_ROOT, engine)["libr3"]
    exclude = component.root / "exclude.txt"
    if not exclude.is_file():
        return set()
    return {line.split()[0] for line in exclude.read_text(encoding="utf-8").splitlines()
            if line.startswith("r4d_")}


def compile_gate(engine: str, tree: Path, touched: list[str], arches: list[str], mode: str, report: Report) -> None:
    clang = compiler(engine)
    if clang is None:
        report.skipped = "no ROCm clang++ found (set ROCM_PATH): compile gates skipped"
        return
    selected = affected_units(tree, touched)
    if not selected:
        report.skipped = "the patches touch no libr4d unit: compile gates have nothing to check"
        return
    all_units = units(tree)
    with tempfile.TemporaryDirectory(prefix="compose-check-") as tmp:
        rewritten = None
        if any(ARCHES[a].rewritten for a in arches):
            rewritten = _rewritten_copy(engine, tree, Path(tmp) / "r4d-src")
        for arch_name in arches:
            arch = ARCHES[arch_name]
            for unit in selected:
                if arch.rewritten and unit in skipped_by_libr3(engine):
                    continue
                for side in ("host", "device"):
                    argv = _command(clang, arch, engine, tree, rewritten, unit, all_units[unit], mode=mode, side=side,
                                    scratch=Path(tmp))
                    done = subprocess.run(argv, capture_output=True, text=True)
                    label = f"{arch.name} {unit} ({side} pass, {mode})"
                    report.checked.append(label)
                    if done.returncode != 0:
                        errors = [l for l in done.stderr.splitlines() if "error" in l][:3]
                        report.compile_failures.append(f"{label}: " + " | ".join(errors or [done.stderr[-300:]]))


def run(engine: str, *, tree: Path, touched: list[str], arches: list[str], mode: str) -> Report:
    report = Report()
    report.gate1 = gate1(engine, tree)
    if report.gate1:
        return report                      # the copy the compile gates need cannot be made from a drifted tree
    if mode != "none":
        compile_gate(engine, tree, touched, arches, mode, report)
    return report


def render(report: Report) -> str:
    lines = []
    for name, problems in sorted(report.gate1.items()):
        for problem in problems:
            lines.append(f"NATIVE_REWRITE_DRIFT: {name}: {problem}")
    for failure in report.compile_failures:
        lines.append(f"COMPOSITION_COMPILE_FAILED: {failure}")
    if report.skipped:
        lines.append(f"SKIPPED: {report.skipped}")
    if report.checked and report.ok:
        lines.append(f"compose-check: {len(report.checked)} compile pass(es) clean")
    lines.append("compose-check: " + ("PASS" if report.ok and not (report.skipped and not report.checked) else
                                      "FAIL" if not report.ok else "PASS (gate 1 only; compile gates skipped)"))
    return "\n".join(lines)


def main(args) -> int:
    engine = args.engine
    out = Path(args.tree) if args.tree else Path(tempfile.mkdtemp(prefix="compose-check-tree-"))
    cleanup = not args.tree and not args.keep
    try:
        if args.tree:
            touched = [t for t in (args.touched or [])]
        else:
            source = args.source if args.source is not None else engines.load(paths.REPO_ROOT, engine).promotion_source
            ids = list(production_ids(engine, source, args.experiment)) if source else []
            ids += [p for p in args.patch or [] if p not in ids]
            by_patch = materialize_tree(engine, ids, out)
            wanted = args.only_patch or list(by_patch)
            touched = sorted({f for patch_id in wanted for f in by_patch.get(patch_id, [])})
        report = run(engine, tree=out, touched=touched, arches=args.arch.split(","), mode=args.compile)
        print(render(report), file=sys.stderr if not report.ok else sys.stdout)
        if args.json:
            print(json.dumps({"gate1": report.gate1, "compile_failures": report.compile_failures,
                              "checked": report.checked, "skipped": report.skipped, "ok": report.ok}))
        return 0 if report.ok else 1
    finally:
        if cleanup:
            shutil.rmtree(out, ignore_errors=True)
