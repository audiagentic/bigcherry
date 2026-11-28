#!/usr/bin/env python3
"""Unit harness (RCS25): build one libr4d unit or family into a small test plugin and check it against the shipped library.

  unit_harness.py run --family iqr --arch gfx1201 [--patch ID ...] [--release bc-radiance-2020.11.4]

builds only the named units (a gfx1100 plugin through the libr3 component, a gfx1201 plugin through libr4d's own build
with the other units left out), in minutes instead of a lane build, then checks four things against the shipped release:

  resources  the compiled kernels' VGPR / SGPR / spill / private / LDS / workgroup records, kernel by kernel: added,
             removed, changed (no card);
  hash       the output bytes of the family's selftest case over its own matrix and a list of trunk shapes and row counts
             (tails and ragged N), the same selftest binary driving both plugins (card);
  replay     a cold replay (256 MiB flush) of the configured rows, the two plugins in ABBA order in
             separate processes, in windows until two medians agree (card);
  selftest   the failures of the case's own oracle checks on the new plugin (card).

The card stages hold the `gpu` lease: `run` takes it for them (`agent-lease acquire gpu --wait`), so the build and the
resource check never hold it. The selftest binary does not depend on the units, so it is built once per selftest source
and architecture and kept in the work root.

  unit_harness.py plan ...      resolve build groups into configured routes without a card
  unit_harness.py build ...     only build and the resource check
  unit_harness.py measure OUT   the card stages, on a build made by `build` (what `run` calls under the lease)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import struct
import subprocess
import sys
import tempfile
import time
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "tools"))

FAMILIES_FILE = Path(__file__).with_name("unit_families.toml")
ENGINE = "radiance"
PLUGIN = {"gfx1100": "libr3.so", "gfx1201": "libr4d.so"}
SELFTEST_TARGET = {"gfx1100": "r3_selftest", "gfx1201": "r4d_selftest"}
RESOURCE_KEYS = ("vgpr_count", "sgpr_count", "vgpr_spill_count", "sgpr_spill_count",
                 "private_segment_fixed_size", "group_segment_fixed_size", "max_flat_workgroup_size")


class HarnessError(RuntimeError):
    pass


# ---------------------------------------------------------------------------------------------------- configuration

def families() -> dict:
    return tomllib.loads(FAMILIES_FILE.read_text(encoding="utf-8"))["family"]


def overlays() -> dict:
    return tomllib.loads(FAMILIES_FILE.read_text(encoding="utf-8")).get("overlay", {})


def work_root() -> Path:
    root = os.environ.get("BC_WORK_ROOT")
    if not root:
        raise HarnessError("BC_WORK_ROOT is not set")
    return Path(root)


def rocm_tool(name: str) -> str:
    from bigcherry.patch import compose_check
    clang = compose_check.compiler("radiance")
    if clang is None:
        raise HarnessError("no ROCm compiler (set ROCM_PATH)")
    tool = clang.parent / name
    if not tool.exists():
        raise HarnessError(f"{tool} is missing")
    return str(tool)


def device_index(arch: str, bdf: str) -> str:
    """Use HIP's actual BDF resolver, not rocminfo ordering."""
    import ctypes
    from bigcherry.hardware import lease
    if not lease.held_by_this_process():
        raise HarnessError("GPU lease not verified")
    if not re.fullmatch(r"[0-9a-f]{4}:[0-9a-f]{2}:[0-9a-f]{2}\.[0-7]",bdf):
        raise HarnessError("explicit canonical physical BDF required")
    if any(k in os.environ for k in ("HIP_VISIBLE_DEVICES","ROCR_VISIBLE_DEVICES","CUDA_VISIBLE_DEVICES","GPU_DEVICE_ORDINAL")):
        raise HarnessError("inherited GPU visibility mask prevents unambiguous physical selection")
    try:
        sdk=os.environ.get("ROCM_PATH")
        library=Path(sdk)/"lib/libamdhip64.so" if sdk else None
        hip=ctypes.CDLL(str(library) if library and library.is_file() else "libamdhip64.so")
    except OSError as exc:
        raise HarnessError("HIP runtime unavailable for physical BDF resolution") from exc
    index=ctypes.c_int(-1)
    if hip.hipDeviceGetByPCIBusId(ctypes.byref(index),bdf.encode()) or index.value<0:
        raise HarnessError("physical BDF not visible to HIP")
    actual=ctypes.create_string_buffer(32)
    if hip.hipDeviceGetPCIBusId(actual,len(actual),index) or actual.value.decode().lower()!=bdf:
        raise HarnessError("HIP physical BDF readback differs")
    return str(index.value)


def release_dir(tag: str | None) -> Path:
    base = work_root() / "installed-radiance" / "releases"
    if tag:
        path = base / tag
    else:
        found = sorted(base.glob("bc-radiance-*"), key=lambda p: [int(x) for x in re.findall(r"\d+", p.name)])
        if not found:
            raise HarnessError(f"no installed release under {base}")
        path = found[-1]
    if not path.is_dir():
        raise HarnessError(f"release {path.name} is not installed")
    return path


def release_plugin(tag: str | None, arch: str) -> Path:
    path = release_dir(tag) / "radiance_home" / "kernels" / PLUGIN[arch]
    if not path.is_file():
        raise HarnessError(f"{path} is missing")
    return path


# ------------------------------------------------------------------------------------------------------- group planning

def read_group_manifest(path: Path) -> dict[str, str]:
    """Read the existing CMake TSV as unit -> group, refusing ambiguous inventory."""
    rows = path.read_text(encoding="utf-8").splitlines()
    if not rows or rows[0] != "group\tunit":
        raise HarnessError(f"invalid group manifest header: {path}")
    result = {}
    for line in rows[1:]:
        fields = line.split("\t")
        if len(fields) != 2:
            raise HarnessError(f"invalid group manifest row: {line!r}")
        group, unit = fields
        if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", group) or not re.fullmatch(r"r4d_[A-Za-z0-9_]+", unit):
            raise HarnessError(f"invalid group/unit: {line!r}")
        if unit in result:
            raise HarnessError(f"duplicate group assignment: {unit}")
        result[unit] = group
    if not result:
        raise HarnessError("empty group inventory")
    return result


def canonical_groups(tree: Path, component: Path) -> dict[str, str]:
    """Ask the build's existing classifier, including units excluded from libr3."""
    with tempfile.TemporaryDirectory(prefix="r3-group-plan-", dir=work_root()) as tmp:
        manifest = Path(tmp) / "r3_groups.tsv"
        done = subprocess.run(["cmake", f"-DR3_GROUP_SOURCE={tree}", f"-DR3_GROUP_OUTPUT={manifest}",
                               "-P", str(component / "r3_groups.cmake")],
                              capture_output=True, text=True, check=False)
        if done.returncode != 0:
            raise HarnessError(f"group inventory failed: {done.stderr}")
        return read_group_manifest(manifest)


def excluded_units(component: Path, arch: str, inventory: dict[str, str]) -> set[str]:
    meta = tomllib.loads((component / "component.toml").read_text(encoding="utf-8"))
    if arch not in meta["targets"]:
        return set()
    entries = [line for line in (component / "exclude.txt").read_text(encoding="utf-8").splitlines()
               if line.startswith("r4d_")]
    if len(entries) != len(set(entries)) or not set(entries) <= inventory.keys():
        raise HarnessError("invalid or stale component exclusions")
    return set(entries)


def verify_group_manifest(actual: dict[str, str], inventory: dict[str, str], excluded: set[str]) -> None:
    expected = {unit: group for unit, group in inventory.items() if unit not in excluded}
    missing = sorted(expected.keys() - actual.keys())
    extra = sorted(actual.keys() - expected.keys())
    wrong = sorted(unit for unit in actual.keys() & expected.keys() if actual[unit] != expected[unit])
    if missing or extra or wrong:
        raise HarnessError(f"not a full matching group inventory: missing={missing}; extra={extra}; wrong={wrong}")


def overlay_origins(tree: Path, components: Path, arch: str, inventory: dict[str, str], specs: dict) -> dict:
    """Map qualification cases to origins, not to another hand-maintained unit roster."""
    from bigcherry.patch import compose_check

    result = {}
    for name, spec in sorted(specs.items()):
        component = components / spec["component"]
        meta = tomllib.loads((component / "component.toml").read_text(encoding="utf-8"))
        if arch not in meta["targets"]:
            continue
        if f"kernels/{spec['plugin']}.so" not in meta["outputs"]:
            raise HarnessError(f"overlay {name}: plugin not declared by component")
        units = tomllib.loads((component / "origins.toml").read_text(encoding="utf-8"))["unit"]
        origins = [u for u in units if spec["case"] in u.get("base-rows", ()) and arch in u.get("architectures", ())]
        if len(origins) != 1:
            raise HarnessError(f"overlay {name}: expected one target origin for case {spec['case']}")
        origin = origins[0]
        base_rows, overlay_rows = origin["base-rows"], origin["overlay-rows"]
        if len(base_rows) != len(overlay_rows) or len(set(base_rows)) != len(base_rows) or len(set(overlay_rows)) != len(overlay_rows):
            raise HarnessError(f"overlay {name}: ambiguous origin row mapping")
        if not (component / origin["owned-source"]).is_file():
            raise HarnessError(f"overlay {name}: owned source absent")
        # The same include-closure helper used by compose_check includes wrappers
        # (e.g. part units) without guessing their names. A header origin alone
        # cannot stand for runtime row ownership.
        related = set(compose_check.affected_units(tree, [origin["upstream-path"]]))
        if origin["name"] not in related or not related <= inventory.keys():
            raise HarnessError(f"overlay {name}: origin not in composed unit inventory")
        result[name] = {"units": sorted(related), "base_rows": base_rows, "overlay_rows": overlay_rows,
                        "selftest_case": spec["case"]}
    return result


def resolve_group_plan(inventory: dict[str, str], excluded: set[str], family_specs: dict, origin_specs: dict,
                       groups: list[str], units: list[str]) -> dict:
    """Resolve the selection. Coverage means a configured test route, never a pass."""
    if not groups and not units:
        raise HarnessError("select at least one group or unit")
    unknown_groups = set(groups) - set(inventory.values())
    unknown_units = set(units) - inventory.keys()
    if unknown_groups or unknown_units:
        raise HarnessError(f"unknown groups={sorted(unknown_groups)}; units={sorted(unknown_units)}")
    selected = set(units) | {u for u, g in inventory.items() if g in groups}
    covered = set()
    runs = []
    for name, spec in sorted(family_specs.items()):
        primary = set(spec["units"])
        support = set(spec.get("support_units", ()))
        if not primary or primary & support:
            raise HarnessError(f"family {name}: invalid primary/support units")
        if not primary & selected:
            continue
        if (primary | support) - inventory.keys():
            raise HarnessError(f"family {name}: stale configured unit")
        # Do not compile an excluded base in place of the production overlay.
        if (primary | support) & excluded:
            continue
        runs.append({"kind": "family", "name": name, "units": sorted(primary),
                     "support_units": sorted(support), "selftest_case": spec["case"]})
        covered.update(primary)
    for name, origin in sorted(origin_specs.items()):
        primary = set(origin["units"])
        if not primary & selected:
            continue
        runs.append({"kind": "overlay", "name": name, **origin,
                     "other_rows_need_evidence": sorted(set(origin["base_rows"]) - {origin["selftest_case"]})})
        covered.update(primary & excluded)
    return {"selected_units": sorted(selected), "excluded_from_base": sorted(selected & excluded),
            "runs": runs, "uncovered_units": sorted(selected - covered)}


def ninja_unit_deps(build: Path, inventory: dict[str, str]) -> dict[str, list[str]]:
    """Per inventory unit, the files its object depends on, from a libr3 build's own Ninja dependency log
    (`ninja -t deps`: what the compiler reported reading, not an include scan)."""
    done = subprocess.run(["ninja", "-C", str(build), "-t", "deps"], capture_output=True, text=True)
    if done.returncode:
        raise HarnessError(f"ninja -t deps failed in {build}: {done.stderr.strip()[-300:]}")
    deps: dict[str, list[str]] = {}
    current = None
    for line in done.stdout.splitlines():
        head = re.match(r"^(\S.*?): #deps \d+, deps mtime \d+ \((\w+)\)$", line)
        if head:
            unit = re.search(r"(?:^|/)units/([A-Za-z0-9_]+)\.(?:hip|cpp)\.o$", head.group(1))
            if unit and head.group(2) != "VALID":
                raise HarnessError(f"stale dependency record for {head.group(1)}; rebuild before planning")
            current = unit.group(1) if unit and unit.group(1) in inventory else None
            if current is not None:
                deps[current] = []
        elif line.startswith("    ") and current is not None:
            deps[current].append(line.strip())
    if not deps:
        raise HarnessError(f"{build} has no libr3 unit dependency records")
    return deps


def depends_on(dep: str, touched: str) -> bool:
    """Whether a recorded dependency is the composed upstream path ``touched``: libr3 compiles libr4d sources as copies
    under r4d-src/, other upstream files by their path under the source tree."""
    dep = dep.replace("\\", "/")
    if touched.startswith("libr4d/"):
        return dep.endswith("r4d-src/" + touched[len("libr4d/"):])
    return dep.endswith("/" + touched)


def compiler_selection(build: Path, inventory: dict[str, str], touched: list[str]) -> dict:
    """Affected units from compiler dependency records. A changed path no unit records (a build file, the registry, a
    removed or new file) cannot be scoped this way and widens to the full inventory."""
    if not touched or any(not isinstance(f, str) or not f or Path(f).is_absolute() or ".." in Path(f).parts for f in touched):
        raise HarnessError("invalid/empty affected path inventory")
    deps = ninja_unit_deps(build, inventory)
    units, unrecorded = set(), []
    for path in sorted(set(touched)):
        hit = {unit for unit, files in deps.items() if any(depends_on(d, path) for d in files)}
        units |= hit
        if not hit:
            unrecorded.append(path)
    widen = bool(unrecorded) or not units
    return {"touched": sorted(set(touched)), "compiler_units": sorted(units), "unrecorded": unrecorded,
            "units": sorted(inventory if widen else units),
            "scope": "full_inventory_conservative" if widen else "compiler_dependencies"}


def affected_selection(tree: Path, inventory: dict[str,str], touched: list[str]) -> dict:
    """Conservative pin/compose scope. Local includes are a lower bound, not a compiler depfile."""
    from bigcherry.patch import compose_check
    if not touched or any(not isinstance(f,str) or not f or Path(f).is_absolute()
                          or ".." in Path(f).parts for f in touched):
        raise HarnessError("invalid/empty affected path inventory")
    known = set(compose_check.affected_units(tree,touched))
    if not known <= inventory.keys():
        raise HarnessError("affected units absent from canonical inventory")
    # Without compiler dependency evidence, headers, registry, build flags,
    # removed files and even unrecognized paths can affect every unit.
    direct={f"libr4d/{u}{suffix}" for u in inventory for suffix in (".hip",".cpp")
            if (tree/"libr4d"/(u+suffix)).is_file()}
    widen=any(f not in direct for f in touched)
    if not known:
        widen=True
    return {"touched":sorted(set(touched)),"local_include_units":sorted(known),
            "units":sorted(inventory if widen else known),
            "scope":"full_inventory_conservative" if widen else "local_include_closure"}


def resolve_rows(rows, family_specs, origin_specs):
    """Select exact configured row names, never guess from model/quant names.

    A requested runtime row is an input claim, not proof of actual dispatch.
    """
    if len(rows)!=len(set(rows)) or any(not re.fullmatch(r"[A-Za-z0-9_]+",r) for r in rows):
        raise HarnessError("duplicate or invalid requested row")
    result=[]
    units=set()
    for row in rows:
        matches=[]
        for name,spec in sorted(family_specs.items()):
            if row==spec["case"] or row in spec.get("rows",[]) or row in spec.get("replay",{}).get("rows",[]):
                matches.append({"kind":"family","name":name,"units":spec["units"],
                                "oracle_case":spec["case"]})
        for name,origin in sorted(origin_specs.items()):
            if row in origin["overlay_rows"]:
                index=origin["overlay_rows"].index(row)
                base=origin["base_rows"][index]
                matches.append({"kind":"overlay","name":name,"units":origin["units"],
                                "oracle_case":origin["selftest_case"],"base_row":base})
        for m in matches:
            units.update(m["units"])
        # Other rows in the same translation unit do not inherit its oracle.
        covered=len(matches)==1 and (matches[0].get("base_row",row)==matches[0]["oracle_case"])
        result.append({"row":row,"routes":matches,"status":"configured_route" if covered else "uncovered"})
    return sorted(units),result


def model_graph_scope(path: Path) -> dict:
    """Read the existing Radiance /graph export, not a runtime execution trace."""
    raw=path.read_bytes()
    graph=json.loads(raw)
    model=graph["model"]
    rank,world=model["rank"],model["world"]
    if type(rank) is not int or type(world) is not int or not 0<=rank<world:
        raise HarnessError("invalid graph rank/world")
    ops,order=graph["ops"],graph["order"]
    if not isinstance(ops,list) or not ops or not isinstance(order,list) or any(
            type(i) is not int or not 0<=i<len(ops) for i in order):
        raise HarnessError("invalid graph operation inventory")
    if len(order)!=graph["counts"]["ops"]:
        raise HarnessError("incomplete graph declaration order")
    from collections import Counter
    counts=Counter(order)
    plugins={p["name"]:p for p in graph["plugins"]}
    if len(plugins)!=len(graph["plugins"]):
        raise HarnessError("ambiguous graph plugin identity")
    declared=[]
    for i,op in enumerate(ops):
        if counts[i]<=0 or counts[i]!=op["count"] or not isinstance(op["bands"],list) or not op["bands"]:
            raise HarnessError("incomplete graph band inventory")
        for band in op["bands"]:
            for domain in ("device","host"):
                entry=band.get(domain)
                if entry is None:
                    continue
                row,plugin=entry["kernel"],entry["plugin"]
                if plugin not in plugins or not isinstance(row,str) or not re.fullmatch(r"[A-Za-z0-9_]+",row):
                    raise HarnessError("invalid graph row or plugin")
                declared.append({"row":row,"plugin":plugin,"domain":domain,
                                 "op":op["op"],"geometry":entry.get("geometry",op["geometry"]),
                                 "span":band["span"],"count":op["count"]})
    if not declared:
        raise HarnessError("graph has no resolved rows")
    misses=graph["misses"]
    if not isinstance(misses,list):
        raise HarnessError("invalid graph misses")
    return {"sha256":hashlib.sha256(raw).hexdigest(),"model":model,
            "plugins":graph["plugins"],"declared":declared,"misses":misses,
            "gaps":["declared bands are not physical execution evidence"]
                   +(["other ranks require their own graph"] if world>1 else [])
                   +(["unresolved graph bands"] if misses else [])}


def stage_plan(args) -> int:
    tree = Path(args.tree).resolve(strict=True)
    components = ROOT / "engines" / ENGINE / "components"
    component = components / "libr3"
    inventory = canonical_groups(tree, component)
    excluded = excluded_units(component, args.arch, inventory)
    if args.manifest:
        targets = tomllib.loads((component / "component.toml").read_text(encoding="utf-8"))["targets"]
        if args.arch not in targets:
            raise HarnessError("r3_groups.tsv is only an actual build inventory for the libr3 component's targets")
        verify_group_manifest(read_group_manifest(Path(args.manifest)), inventory, excluded)
    # The reference must exist for the requested architecture. A plan is not
    # proof of matched compiler flags or source/build identities.
    release_plugin(args.release, args.arch)
    origins = overlay_origins(tree, components, args.arch, inventory, overlays())
    specs=families()
    touched=list(args.touched or [])
    triage=None
    if args.pin_triage:
        source=Path(args.pin_triage).read_bytes()
        triage=json.loads(source)
        if triage.get("engine")!=ENGINE or not isinstance(triage.get("changed_files"),list):
            raise HarnessError("invalid existing pin-triage report")
        touched.extend(triage["changed_files"])
    if touched and args.deps_build:
        affected=compiler_selection(Path(args.deps_build),inventory,touched)
    else:
        affected=affected_selection(tree,inventory,touched) if touched else None
    graph_scope=model_graph_scope(Path(args.model_graph)) if args.model_graph else None
    requested=set(args.row or [])
    if len(requested)!=len(args.row or []):
        raise HarnessError("duplicate requested row")
    if graph_scope:
        requested.update(r["row"] for r in graph_scope["declared"])
    row_units,row_scope=resolve_rows(sorted(requested),specs,origins)
    selected=sorted(set(args.unit or [])|set(row_units)|set(affected["units"] if affected else []))
    # An entirely unknown row selection is still useful as explicit nonpass.
    result = (resolve_group_plan(inventory,excluded,specs,origins,args.group or [],selected)
              if selected or args.group else
              {"selected_units":[],"excluded_from_base":[],"runs":[],"uncovered_units":[]})
    if not selected and not args.group and not row_scope:
        raise HarnessError("select at least one group, unit, affected path or row")
    result["affected_scope"]=affected
    planned={(r["kind"],r["name"]) for r in result["runs"]}
    for row in row_scope:
        if row["status"]=="configured_route" and not all((m["kind"],m["name"]) in planned for m in row["routes"]):
            row["status"]="uncovered"
    if graph_scope:
        for row in row_scope:
            providers={r["plugin"] for r in graph_scope["declared"] if r["row"]==row["row"]}
            if providers:
                expected={Path(PLUGIN[args.arch]).stem if r["kind"]=="family"
                          else overlays()[r["name"]]["plugin"] for r in row["routes"]}
                if not providers<=expected or any(r["domain"]!="device" for r in graph_scope["declared"]
                                                  if r["row"]==row["row"]):
                    row["status"]="uncovered"
        result["model_graph"]=graph_scope
    result["requested_rows"]=row_scope
    if triage is not None:
        result["pin_triage"]={"sha256":hashlib.sha256(source).hexdigest(),
                             "current_sha":triage.get("current_sha"),"target_sha":triage.get("target_sha")}
    # This records the reference bytes, not a claim that the composed tree
    # or actual loaded model route matches the installed build.
    result["reference_sha256"]=file_sha256(release_plugin(args.release,args.arch))
    result["family_policy_sha256"]=file_sha256(FAMILIES_FILE)
    result.update(arch=args.arch, release=args.release, tree=str(tree), inventory_units=len(inventory),
                  inventory_kind="matching_libr3_membership" if args.manifest else "canonical_source_only",
                  outcome="diagnostic", qualification="not_run")
    for item in result["runs"]:
        common = ["--arch", args.arch, "--tree", str(tree), "--release", args.release]
        if item["kind"] == "family":
            item["build_args"] = ["build", "--family", item["name"], *common]
            item["measure_args"] = ["measure", "<build-output>","--bdf","<physical-bdf>"]
        else:
            # The installed base can intentionally omit the overlay's rows.
            # Require an explicit compatible base reference for these cases.
            if set(item["units"]) & excluded:
                item["reference_required"] = True
                if args.overlay_reference:
                    item["reference_required"] = False
                    common += ["--reference", str(Path(args.overlay_reference).resolve(strict=True))]
                else:
                    common += ["--reference", "<compatible-base-plugin>"]
            item["build_args"] = ["overlay-build", "--overlay", item["name"], *common]
            item["measure_args"] = ["overlay-measure", "<build-output>","--bdf","<physical-bdf>"]
    print(json.dumps(result, indent=2, sort_keys=True))
    return 1 if graph_scope or result["uncovered_units"] or any(r["status"]!="configured_route" for r in row_scope) or any(r.get("other_rows_need_evidence") or r.get("reference_required") for r in result["runs"]) else 0


# ------------------------------------------------------------------------------------------------------- composition

def compose(patches: list[str], tree: Path | None, scratch: Path, without: list[str] = ()) -> Path:
    """Build from a private composition; never edit a shared source/cache tree."""
    if tree is not None:
        if patches or without:
            raise HarnessError("explicit composed source cannot also apply/remove patches")
        source=tree.resolve(strict=True)
        out=(scratch/"src").resolve()
        if source==out or source in out.parents or out.exists():
            raise HarnessError("build scratch must be new and outside the source tree")
        # Do not follow arbitrary links out of the pinned/composed source.
        for path in source.rglob("*"):
            if path.is_symlink() and (Path(os.readlink(path)).is_absolute() or not path.resolve().is_relative_to(source)):
                raise HarnessError("external source symlink cannot be copied into private build")
        shutil.copytree(source,out,ignore=shutil.ignore_patterns(".git"),symlinks=True)
        return out
    from bigcherry.core import engines, paths
    from bigcherry.patch import compose_check
    source = engines.load(paths.REPO_ROOT, ENGINE).promotion_source
    ids = list(compose_check.production_ids(ENGINE, source)) if source else []
    ids = [p for p in ids if p not in without]
    ids += [p for p in patches if p not in ids]
    out = scratch / "src"
    compose_check.materialize_tree(ENGINE, ids, out)
    return out


# ---------------------------------------------------------------------------------------------------------- building

PREFIX_CONFIG = r'''# a find_package(radiance) for a source tree: the ABI target and the plugin helper of the tree itself
set(RADIANCE_ABI_VERSION 0)
add_library(radiance::abi INTERFACE IMPORTED)
target_include_directories(radiance::abi INTERFACE "@TREE@/abi")
target_include_directories(radiance::abi SYSTEM INTERFACE "@ROCM@/include")
target_compile_definitions(radiance::abi INTERFACE RAD_DEFAULT_HOME="/usr/local/share/radiance" RAD_HAVE_HIP=1 __HIP_PLATFORM_AMD__=1)
set(RAD_HIP_FOUND ON)
set(RAD_GPU_TARGETS "@ARCH@")
find_library(RAD_HIP_RUNTIME amdhip64 HINTS "@ROCM@/lib" REQUIRED)
if(NOT DEFINED RAD_PLUGIN_OUT)
  set(RAD_PLUGIN_OUT ${CMAKE_BINARY_DIR}/radiance_home)
endif()
include("@TREE@/cmake/RadAddPlugin.cmake")
rad_hip_rule_fixes()
'''


def write_prefix(tree: Path, arch: str, prefix: Path) -> None:
    config = prefix / "lib" / "cmake" / "radiance"
    config.mkdir(parents=True, exist_ok=True)
    rocm = os.environ.get("ROCM_PATH")
    if not rocm:
        raise HarnessError("ROCM_PATH is not set (the SDK of the host's rocm-sdks)")
    text = PREFIX_CONFIG.replace("@TREE@", str(tree)).replace("@ARCH@", arch).replace("@ROCM@", rocm)
    (config / "radiance-config.cmake").write_text(text, encoding="utf-8")


# The selftest is driven by environment variables the harness sets; the two edits below add what the shipped selftest
# does not have: the hash of every IQR case's output, and extra IQR cases "N:K:M" from UH_HASH_CASES.
HASH_ANCHOR = ("    st_memcpy(yr.data(), d_yr, yr.size() * 2, hipMemcpyDeviceToHost);\n"
               "    /* M <= 16 is the GEMV or the 16-token tile, which sum per wave before summing across waves:\n")
HASH_PRINT = ('    {\n'
              '        uint64_t uh = 1469598103934665603ull;\n'
              '        for (uint16_t v : y) { uh ^= v; uh *= 1099511628211ull; }\n'
              '        std::printf("UNITHASH %s %016llx\\n", nm, (unsigned long long)uh);\n'
              '        {   /* a plain bf16 table is not an IQR weight: the row must refuse it as the base does */\n'
              '            Planes pp; pp.enc = rad_enc_plain(RAD_BF16); pp.size(256, 256);\n'
              '            std::vector<uint8_t> po; RadLayout pl2{};\n'
              '            std::printf("UNITLAYOUT plain_bf16 rc=%d\\n", stored(r, lp, 2, pp, { "codes" }, &po, &pl2));\n'
              '        }\n'
              '    }\n')
CASES_ANCHOR = "                iqr_gemm_case(pl, f, 300, 512, 1024);\n"
CASES_BLOCK = ('                if (const char* uh_cases = std::getenv("UH_HASH_CASES"); uh_cases && f.kind == 6 && (f.B == 6 || f.B == 7))\n'
               '                    for (const char* s = uh_cases; *s;) {\n'
               '                        long long un = 0, uk = 0, um = 0; int used = 0;\n'
               '                        if (std::sscanf(s, "%lld:%lld:%lld%n", &un, &uk, &um, &used) < 3) break;\n'
               '                        iqr_gemm_case(pl, f, um, un, uk);\n'
               '                        s += used;\n'
               '                        if (*s == \',\') ++s;\n'
               '                    }\n')


# The IQ8R case (iq8r_numeric) gets the same two edits: extra "N:K:M" shapes from UH_HASH_CASES (the 16-token row up to 64
# rows, the tiled row above) and a hash of every GEMM output. An extra shape is checked by its hash against the shipped
# library alone: the case's double-precision host fold is O(M*N*K) and would take hours at the trunk shapes.
IQ8R_SHAPES_ANCHOR = ("    for (const Shape& sh : shapes) {\n"
                      "        const int64_t M = sh.M, N = sh.N, K = sh.K, nkb = K / G;\n")
IQ8R_SHAPES_NEW = ("    std::vector<Shape> uh_all(std::begin(shapes), std::end(shapes));\n"
                   "    std::vector<bool> uh_extra(uh_all.size(), false);\n"
                   "    if (const char* uh_cases = std::getenv(\"UH_HASH_CASES\"))\n"
                   "        for (const char* s = uh_cases; *s;) {\n"
                   "            long long un = 0, uk = 0, um = 0; int used = 0;\n"
                   "            if (std::sscanf(s, \"%lld:%lld:%lld%n\", &un, &uk, &um, &used) < 3) break;\n"
                   "            uh_all.push_back({ um <= 64 ? gn : gt, um, un, uk });\n"
                   "            uh_extra.push_back(true);\n"
                   "            s += used;\n"
                   "            if (*s == ',') ++s;\n"
                   "        }\n"
                   "    for (size_t uh_i = 0; uh_i < uh_all.size(); ++uh_i) {\n"
                   "        const Shape& sh = uh_all[uh_i];\n"
                   "        const int64_t M = sh.M, N = sh.N, K = sh.K, nkb = K / G;\n")
IQ8R_HASH_ANCHOR = ("        std::vector<uint16_t> got((size_t)(M * N));\n"
                    "        st_memcpy(got.data(), d_y, got.size() * 2, hipMemcpyDeviceToHost);\n"
                    "        /* The exact fold of the device's codes, and the f32 product the planes stand for. */\n")
IQ8R_HASH_PRINT = ('        {\n'
                   '            uint64_t uh = 1469598103934665603ull;\n'
                   '            for (uint16_t hv : got) { uh ^= hv; uh *= 1099511628211ull; }\n'
                   '            std::printf("UNITHASH %s %016llx\\n", nm, (unsigned long long)uh);\n'
                   '        }\n'
                   '        if (uh_extra[uh_i]) continue;\n')


# The int8 vocabulary head (i8_logits_numeric): extra "N:K:M" heads from UH_HASH_CASES (the case's own host oracle runs on
# them too; it is a parallel O(N*K*M) loop, minutes at a 27B shard) and a hash of every head output.
HEAD_CASES_ANCHOR = ("        for (const int64_t M : { (int64_t)1, (int64_t)8 })\n"
                     "            i8_logits_numeric(pl, 2560, 2560, M);\n")
HEAD_CASES_BLOCK = ('        if (const char* uh_cases = std::getenv("UH_HASH_CASES"))\n'
                    '            for (const char* s = uh_cases; *s;) {\n'
                    '                long long un = 0, uk = 0, um = 0; int used = 0;\n'
                    '                if (std::sscanf(s, "%lld:%lld:%lld%n", &un, &uk, &um, &used) < 3) break;\n'
                    '                i8_logits_numeric(pl, un, uk, um);\n'
                    '                s += used;\n'
                    "                if (*s == ',') ++s;\n"
                    '            }\n')
HEAD_HASH_ANCHOR = ("    std::vector<float> got((size_t)M * N);\n"
                    "    st_memcpy(got.data(), d_y, got.size() * 4, hipMemcpyDeviceToHost);\n"
                    "    double num = 0, den = 0;\n")
HEAD_HASH_PRINT = ('    {\n'
                   '        uint64_t uh = 1469598103934665603ull;\n'
                   '        for (float hv : got) { uint32_t hb; std::memcpy(&hb, &hv, 4); uh ^= hb; uh *= 1099511628211ull; }\n'
                   '        std::printf("UNITHASH logits_gemm_i8 N=%lld K=%lld M=%lld %016llx\\n", (long long)N, (long long)K,\n'
                   '                    (long long)M, (unsigned long long)uh);\n'
                   '    }\n')


# The int8 trunk case (i8a8_numeric, group 128): extra "N:K:M" shapes from UH_HASH_CASES (the 16-token row up to 64 rows,
# the tiled row above; the case's own oracle samples about 96 columns a row, so it runs on them too) and a hash of every
# GEMM output.
I8A8_CASES_ANCHOR = '        i8a8_numeric(pl, "gemm_i8a8_tiled", 2048, 512, 2560);\n'
I8A8_CASES_BLOCK = ('        if (const char* uh_cases = std::getenv("UH_HASH_CASES"))\n'
                    '            for (const char* s = uh_cases; *s;) {\n'
                    '                long long un = 0, uk = 0, um = 0; int used = 0;\n'
                    '                if (std::sscanf(s, "%lld:%lld:%lld%n", &un, &uk, &um, &used) < 3) break;\n'
                    '                i8a8_numeric(pl, um <= 64 ? "gemm_i8a8_nt_m16" : "gemm_i8a8_tiled", um, un, uk);\n'
                    '                s += used;\n'
                    "                if (*s == ',') ++s;\n"
                    '            }\n')
I8A8_HASH_ANCHOR = ("    std::vector<uint16_t> hy((size_t)(M * N));\n"
                    "    st_memcpy(hy.data(), d_y, hy.size() * 2, hipMemcpyDeviceToHost);\n")
I8A8_HASH_PRINT = ('    {\n'
                   '        uint64_t uh = 1469598103934665603ull;\n'
                   '        for (uint16_t hv : hy) { uh ^= hv; uh *= 1099511628211ull; }\n'
                   '        std::printf("UNITHASH %s rb=%lld %016llx\\n", nm, (long long)rb, (unsigned long long)uh);\n'
                   '    }\n')


# The grouped int8-activation MoE GEMM (moe_w4_gemm): extra expert shapes from UH_HASH_CASES as
# "N0:K0:N1:K1:parts:experts:sorted:tokens" (the family's own `cases`), each run at the 64-group codebook forms the
# selftest runs (sub 2 and 4) at top 10, and a hash of every form's output. The output anchor also occurs in
# moe_q4_gemm, so the edit is made inside moe_w4_gemm's body only.
MOE_FUNCTION = "static void moe_w4_gemm("
MOE_CASES_BLOCK = ('        if (const char* uh_cases = std::getenv("UH_HASH_CASES"); uh_cases && w4)\n'
                   '            for (const char* s = uh_cases; *s;) {\n'
                   '                long long n0 = 0, k0 = 0, n1 = 0, k1 = 0, parts = 0, ne = 0, sorted = 0, tok = 0; int used = 0;\n'
                   '                if (std::sscanf(s, "%lld:%lld:%lld:%lld:%lld:%lld:%lld:%lld%n", &n0, &k0, &n1, &k1, &parts, &ne,\n'
                   '                                &sorted, &tok, &used) < 8) break;\n'
                   '                static std::vector<std::string> uh_names;   /* MoeW4Shape keeps the name pointer */\n'
                   '                uh_names.push_back(std::to_string(n0) + "x" + std::to_string(k0) + ":" + std::to_string(n1) + "x" +\n'
                   '                                   std::to_string(k1) + " parts=" + std::to_string(parts) + " experts=" + std::to_string(ne));\n'
                   '                const MoeW4Shape uh_sh{ uh_names.back().c_str(), n0, k0, n1, k1, parts, ne, sorted != 0 };\n'
                   '                for (const int uh_sub : { 2, 4 })\n'
                   '                    moe_w4_gemm(pl, tok * 10, uh_sh, uh_sub, false, false, 10, 1, 128);   /* rows: tokens x top 10 */\n'
                   '                s += used;\n'
                   "                if (*s == ',') ++s;\n"
                   '            }\n')
MOE_LARGE_PASSES = "            moe_down_mt_case(pl, 4096);\n            moe_down_mt_case(pl, 8192);\n"
MOE_HASH_ANCHOR = ("        std::vector<uint16_t> got((size_t)(Tr * N));\n"
                   "        st_memcpy(got.data(), d_y, got.size() * 2, hipMemcpyDeviceToHost);\n"
                   "        double worst = 0.0;\n")
MOE_HASH_PRINT = ('        {\n'
                  '            uint64_t uh = 1469598103934665603ull;\n'
                  '            for (uint16_t hv : got) { uh ^= hv; uh *= 1099511628211ull; }\n'
                  '            std::printf("UNITHASH %s sub=%d top_k=%lld %016llx\\n", label.c_str(), sub, (long long)top_k,\n'
                  '                        (unsigned long long)uh);\n'
                  '        }\n')


# The IQR case's trellis forms run 256-row runs only; Qwen 27B IQR-V4 stores its ffn_gate_up with 512-row runs (the
# computed trellis at 6 and 7 bits under one f32 scale a row, rotated). The case gains a run length (256 by default, so
# every existing form and its hash name is unchanged), its table and host decode window follow it, and the form list
# gains the two 512-row forms. A trellis weight is whole runs (the layout and the rows refuse anything else with
# RAD_E_SHAPE), so a 512-row form skips a shape whose N is not a multiple of 512; its hash name carries "run=512".
RUN_STRUCT_ANCHOR = "struct IqrCase { int kind, B, V, L, rb; bool two, rot; bool rowf = false; };\n"
RUN_STRUCT_NEW = "struct IqrCase { int kind, B, V, L, rb; bool two, rot; bool rowf = false; int run = 256; };\n"
RUN_FUNCTION = "static void iqr_gemm_case("
RUN_NAME_ANCHOR = ("    const RadKernelInfo* r = find(pl, row);\n"
                   "    const RadKernelInfo* ri = find(pl, ref);\n")
RUN_NAME = ("    if (f.run != 256) {\n"
            "        const size_t uh_l = std::strlen(nm);\n"
            "        std::snprintf(nm + uh_l, sizeof nm - uh_l, \" run=%d\", f.run);\n"
            "        if (N % f.run) return;   /* a trellis weight is whole runs: the layout and the rows refuse a part */\n"
            "    }\n")
RUN_TABLE_ANCHOR = "        p.enc.plane[t].block[0] = 256;\n"
RUN_TABLE_NEW = "        p.enc.plane[t].block[0] = f.run;\n"
RUN_WINDOW_ANCHOR = "                const int64_t rr = n / 256 * 256 + (n % 256 - q + 256) % 256;\n"
RUN_WINDOW_NEW = "                const int64_t rr = n / f.run * f.run + (n % f.run - q + f.run) % f.run;\n"
RUN_FORMS_ANCHOR = "                { 6, 2, 0, 15, 1, false, true, true },\n"
RUN_FORMS_NEW = (RUN_FORMS_ANCHOR +
                 "                { 6, 6, 0, 16, 1, false, true, true, 512 }, { 6, 7, 0, 16, 1, false, true, true, 512 },\n")

# The gated row (gemm_iqr_gated: the gate_up GEMM with its gated activation folded in) is checked byte for byte against
# its chain on planes from iqr_v4_planes, which also builds the --perfiqr timing planes; both had 256-row runs only.
# iqr_v4_planes takes a run (0: UH_IQR_RUN, else 256, so --perfiqr can time 512-row runs), the gated case passes its own,
# and the case gains the 27B gate_up per XTX rank (n 6656, so N 13312 = 26 runs of 512, K 5120) at 512-row runs, B 6
# and 7, decode, draft and verify row counts.
V4_PLANES_FUNCTION = "static Planes iqr_v4_planes("
V4_PLANES_SIG = "static Planes iqr_v4_planes(int64_t N, int64_t K, int B, uint64_t z, bool bf16s = false) {\n"
V4_PLANES_SIG_NEW = ("static Planes iqr_v4_planes(int64_t N, int64_t K, int B, uint64_t z, bool bf16s = false, int run = 0) {\n"
                     "    if (run <= 0) run = std::getenv(\"UH_IQR_RUN\") ? std::atoi(std::getenv(\"UH_IQR_RUN\")) : 256;\n")
V4_PLANES_RUN_ANCHOR = "    p.enc.plane[t].block[0] = 256;\n"
V4_PLANES_RUN_NEW = "    p.enc.plane[t].block[0] = run;\n"
GATED_SIG = "static void iqr_gated_case(const Plugin& pl, int B, int64_t M, int64_t n, int64_t K) {\n"
GATED_SIG_NEW = "static void iqr_gated_case(const Plugin& pl, int B, int64_t M, int64_t n, int64_t K, int run = 256) {\n"
GATED_NAME_ANCHOR = ('    std::snprintf(nm, sizeof nm, "gemm_iqr_gated trellis-mad B=%d row-f32 M=%lld n=%lld K=%lld", B,\n'
                     '                  (long long)M, (long long)n, (long long)K);\n')
GATED_NAME_RUN = ("    if (run != 256) {\n"
                  "        const size_t uh_l = std::strlen(nm);\n"
                  "        std::snprintf(nm + uh_l, sizeof nm - uh_l, \" run=%d\", run);\n"
                  "        if (N % run) return;   /* a trellis weight is whole runs */\n"
                  "    }\n")
GATED_PLANES_ANCHOR = "    Planes p = iqr_v4_planes(N, K, B, 0x6A7ED00Du + (uint64_t)(B * 97 + M));\n"
GATED_PLANES_NEW = "    Planes p = iqr_v4_planes(N, K, B, 0x6A7ED00Du + (uint64_t)(B * 97 + M), false, run);\n"
RUN_GATED_CASES = ("            for (const int uh_b : { 6, 7 })\n"
                   "                for (const int64_t uh_m : { (int64_t)1, (int64_t)4, (int64_t)16 })\n"
                   "                    iqr_gated_case(pl, uh_b, uh_m, 6656, 5120, 512);\n")


def _edit_function(text: str, function: str, anchor: str, replacement: str) -> str:
    """Replace ``anchor`` once inside the body of ``function`` (up to the next top-level static function)."""
    start = text.find(function)
    if start < 0 or text.count(function) != 1:
        raise HarnessError(f"the selftest changed: {function.strip()!r} is not found exactly once")
    end = text.find("\nstatic ", start + len(function))
    body = text[start:end if end >= 0 else len(text)]
    if body.count(anchor) != 1:
        raise HarnessError(f"the selftest changed: the harness's edit anchor in {function.strip()!r} is not found exactly once "
                           f"({anchor.splitlines()[0].strip()!r})")
    return text[:start] + body.replace(anchor, replacement, 1) + text[start + len(body):]


# A plugin that carries only some units has no entry point for the rest: the loader notes the absence and goes on
# (the rows without a kernel then return "unsupported", like libr3's own stubbed rows).
SYM_ANCHOR = '    if (e || !p) { std::printf("  FAIL  dlsym(%s): %s\\n", n, e ? e : "null"); ++g_fail; return false; }\n'
SYM_TOLERANT = ('    if ((e || !p) && std::getenv("UH_TOLERATE_ABSENT")) { *out = nullptr; return true; }\n')


# Overlay checks: a row of an overlay plugin is found by its prefixed name (UH_ROW_PREFIX), the embed case prints a hash of
# its output and, with UH_TIME_ITERS, a cold median launch time; UH_EMBED_CASES adds "rows:dim" tables to the case.
FIND_ANCHOR = ("static const RadKernelInfo* find(const Plugin& pl, const char* name) {\n"
               "    for (int k = 0; k < pl.kernel_count(); ++k)\n"
               "        if (!std::strcmp(pl.kernel_at(k)->name, name)) return pl.kernel_at(k);\n"
               "    return nullptr;\n"
               "}\n")
FIND_NEW = ("/* an overlay run: the base plugin the overlay's absent rows (the int8 reference rows) come from */\n"
            "static const Plugin* uh_base() {\n"
            "    static Plugin uh_pl; static bool uh_tried = false, uh_ok = false;\n"
            "    if (!uh_tried) { uh_tried = true; if (const char* b = std::getenv(\"UH_BASE_PLUGIN\")) uh_ok = load(&uh_pl, b); }\n"
            "    return uh_ok ? &uh_pl : nullptr;\n"
            "}\n"
            "static const RadKernelInfo* find(const Plugin& pl, const char* name) {\n"
            "    if (const char* uh_pre = std::getenv(\"UH_ROW_PREFIX\")) {\n"
            "        const std::string uh_name = std::string(uh_pre) + name;\n"
            "        for (int k = 0; k < pl.kernel_count(); ++k)\n"
            "            if (uh_name == pl.kernel_at(k)->name) return pl.kernel_at(k);\n"
            "    }\n"
            "    for (int k = 0; k < pl.kernel_count(); ++k)\n"
            "        if (!std::strcmp(pl.kernel_at(k)->name, name)) return pl.kernel_at(k);\n"
            "    if (const Plugin* uh_pl = uh_base())\n"
            "        for (int k = 0; k < uh_pl->kernel_count(); ++k)\n"
            "            if (!std::strcmp(uh_pl->kernel_at(k)->name, name)) return uh_pl->kernel_at(k);\n"
            "    return nullptr;\n"
            "}\n")
EMBED_ANCHOR = ("    std::vector<uint16_t> got((size_t)(M * dim)), want((size_t)dim);\n"
                "    st_memcpy(got.data(), d_x, got.size() * 2, hipMemcpyDeviceToHost);\n")
EMBED_PRINT = ('    {\n'
               '        uint64_t uh = 1469598103934665603ull;\n'
               '        for (uint16_t v : got) { uh ^= v; uh *= 1099511628211ull; }\n'
               '        std::printf("UNITHASH %s rows=%lld dim=%lld %016llx\\n", nm, (long long)rows, (long long)dim, (unsigned long long)uh);\n'
               '        {   /* a plain bf16 table is not an IQR weight: the row must refuse it as the base does */\n'
               '            Planes pp; pp.enc = rad_enc_plain(RAD_BF16); pp.size(64, 128);\n'
               '            std::vector<uint8_t> po; RadLayout pl2{};\n'
               '            const int prc = stored(r, lp, 1, pp, { "codes" }, &po, &pl2);\n'
               '            std::printf("UNITLAYOUT plain_bf16 rc=%d\\n", prc);\n'
               '        }\n'
               '        if (const char* it = std::getenv("UH_TIME_ITERS")) {\n'
               '            const int n = std::atoi(it);\n'
               '            const size_t fb = 256ull << 20;\n'
               '            void* flush = dalloc(fb);\n'
               '            hipEvent_t e0, e1;\n'
               '            (void)hipEventCreate(&e0); (void)hipEventCreate(&e1);\n'
               '            std::vector<float> us;\n'
               '            for (int i = 0; i < n + 1 && flush; ++i) {\n'
               '                (void)hipMemset(flush, i, fb); (void)hipDeviceSynchronize();\n'
               '                (void)hipEventRecord(e0);\n'
               '                const int rc2 = launch_row(r, { T(d_t, RAD_I32, { M }), T(d_w, lay.dtype, { rows, dim }),\n'
               '                                                T(d_x, RAD_BF16, { M, dim }) }, ep);\n'
               '                (void)hipEventRecord(e1); (void)hipEventSynchronize(e1);\n'
               '                float ms = 0; (void)hipEventElapsedTime(&ms, e0, e1);\n'
               '                if (rc2 == RAD_OK && i) us.push_back(ms * 1000.0f);\n'
               '            }\n'
               '            if (!us.empty()) { std::sort(us.begin(), us.end()); std::printf("UNITTIME %s rows=%lld dim=%lld %.2f\\n", nm, (long long)rows, (long long)dim, us[us.size() / 2]); }\n'
               '        }\n'
               '    }\n')
EMBED_CASES_ANCHOR = "            iqr_embed_case(pl, f, 512, 640, 3.25f);\n        }\n"
EMBED_CASES_BLOCK = ('        if (const char* uh_e = std::getenv("UH_EMBED_CASES"))\n'
                     '            for (const IqrEmbedCase& f : es)\n'
                     '                for (const char* q = uh_e; *q;) {\n'
                     '                    long long ur = 0, ud = 0; int used = 0;\n'
                     '                    if (std::sscanf(q, "%lld:%lld%n", &ur, &ud, &used) < 2) break;\n'
                     '                    iqr_embed_case(pl, f, ur, ud, 1.0f);\n'
                     '                    q += used;\n'
                     '                    if (*q == \',\') ++q;\n'
                     '                }\n')


VOCAB_ANCHOR = ("    std::vector<float> y((size_t)(M * V));\n"
                "    st_memcpy(y.data(), d_y, y.size() * 4, hipMemcpyDeviceToHost);\n")
VOCAB_PRINT = ('    {\n'
               '        uint64_t uh = 1469598103934665603ull;\n'
               '        for (float v : y) { uint32_t u; std::memcpy(&u, &v, 4); uh ^= u; uh *= 1099511628211ull; }\n'
               '        std::printf("UNITHASH %s %016llx\\n", nm, (unsigned long long)uh);\n'
               '        {   /* a plain bf16 table is not an IQR weight: the row must refuse it as the base does */\n'
               '            Planes pp; pp.enc = rad_enc_plain(RAD_BF16); pp.size(256, 256);\n'
               '            std::vector<uint8_t> po; RadLayout pl2{};\n'
               '            const int prc = stored(r, gp, 1, pp, { "codes" }, &po, &pl2);\n'
               '            std::printf("UNITLAYOUT plain_bf16 rc=%d\\n", prc);\n'
               '        }\n'
               '        if (const char* it = std::getenv("UH_TIME_ITERS")) {\n'
               '            const int n = std::atoi(it);\n'
               '            const size_t fb = 256ull << 20;\n'
               '            void* flush = dalloc(fb);\n'
               '            hipEvent_t e0, e1;\n'
               '            (void)hipEventCreate(&e0); (void)hipEventCreate(&e1);\n'
               '            std::vector<float> us;\n'
               '            for (int i = 0; i < n + 1 && flush; ++i) {\n'
               '                (void)hipMemset(flush, i, fb); (void)hipDeviceSynchronize();\n'
               '                (void)hipEventRecord(e0);\n'
               '                const int rc2 = r->launch(&a, (RadStream) nullptr);\n'
               '                (void)hipEventRecord(e1); (void)hipEventSynchronize(e1);\n'
               '                float ms = 0; (void)hipEventElapsedTime(&ms, e0, e1);\n'
               '                if (rc2 == RAD_OK && i) us.push_back(ms * 1000.0f);\n'
               '            }\n'
               '            if (!us.empty()) { std::sort(us.begin(), us.end()); std::printf("UNITTIME %s %.2f\\n", nm, us[us.size() / 2]); }\n'
               '        }\n'
               '    }\n')
VOCAB_CASES_ANCHOR = "        vocab_head_case(pl, { { 5, 1024 } }, 8, 5120, 0);\n"
VOCAB_CASES_BLOCK = ('        if (const char* uh_v = std::getenv("UH_VOCAB_CASES"))\n'
                     '            for (const char* q = uh_v; *q;) {\n'
                     '                long long um = 0, uk = 0, ur = 0; int used = 0;\n'
                     '                if (std::sscanf(q, "%lld:%lld:%lld%n", &um, &uk, &ur, &used) < 3) break;\n'
                     '                vocab_head_case(pl, { { 4, ur } }, um, uk, 0);\n'
                     '                q += used;\n'
                     '                if (*q == \',\') ++q;\n'
                     '            }\n')


DEVICE_ANCHOR = "int main(int argc, char** argv) {\n"
DEVICE_CHECK = r"""#ifdef RAD_HAVE_HIP
    if (const char* expected = std::getenv("UH_EXPECT_BDF")) {
        int count=0; char actual[32]={}; hipDeviceProp_t props{};
        const char* arch=std::getenv("UH_EXPECT_ARCH");
        if (!arch || hipGetDeviceCount(&count)!=hipSuccess || count!=1 ||
            hipSetDevice(0)!=hipSuccess || hipDeviceGetPCIBusId(actual,sizeof(actual),0)!=hipSuccess ||
            hipGetDeviceProperties(&props,0)!=hipSuccess || std::strcmp(actual,expected) ||
            std::string(props.gcnArchName).substr(0,std::string(props.gcnArchName).find(':'))!=arch) {
            std::printf("FAIL physical GPU identity\n"); return 2;
        }
        std::printf("UNITDEVICE %s %s\n",actual,arch);
    }
#endif
"""


# The owned IQR forms (bc-iqr serves the V4 trunk: the computed trellis, 6 and 7-bit codes). With UH_IQR_OWNED set the
# selftest's `gemm_iqr` case runs only those forms through its GEMM rows and its gated row, so an overlay that owns them is
# compared on what it serves rather than reported missing on every other form; the gated row hashes its three outputs and
# takes extra "M:n:K" cases from UH_GATED_CASES (each at both owned widths).
OWNED_GEMM_ANCHOR = ('    if (!sel(row) && !sel("gemm_iqr")) return;\n'
                     '    static const char* kKind[] = { "", "signed", "zero", "table", "grid", "trellis", "trellis-mad" };\n')
OWNED_GEMM_SKIP = '    if (std::getenv("UH_IQR_OWNED") && !(f.kind == 6 && (f.B == 6 || f.B == 7))) return;\n'
OWNED_GATED_ANCHOR = ('    if (!sel("gemm_iqr_gated") && !sel("gemm_iqr")) return;\n'
                      '    const int64_t N = 2 * n, nkb = K / 128;\n')
OWNED_GATED_SKIP = '    if (std::getenv("UH_IQR_OWNED") && B != 6 && B != 7) return;\n'
GATED_HASH_ANCHOR = ('    std::snprintf(d, sizeof d, "%zu codes, %zu scales, %zu products differ", dq, ds, dy);\n'
                     '    ok(nm, dq == 0 && ds == 0 && dy == 0, d);\n')
GATED_HASH_PRINT = ('    {\n'
                    '        uint64_t uh = 1469598103934665603ull;\n'
                    '        for (const std::vector<uint8_t>* v : { &q, &sc, &yb })\n'
                    '            for (uint8_t c : *v) { uh ^= c; uh *= 1099511628211ull; }\n'
                    '        std::printf("UNITHASH %s %016llx\\n", nm, (unsigned long long)uh);\n'
                    '        {   /* a plain bf16 table is not an IQR weight: the gated row must refuse it as the base does */\n'
                    '            Planes pp; pp.enc = rad_enc_plain(RAD_BF16); pp.size(256, 256);\n'
                    '            std::vector<uint8_t> po; RadLayout pl2{};\n'
                    '            std::printf("UNITLAYOUT plain_bf16 rc=%d\\n", stored(rg, lp, 2, pp, { "codes" }, &po, &pl2));\n'
                    '        }\n'
                    '    }\n')
GATED_CASES_ANCHOR = "            iqr_gated_case(pl, 4, 9, 17408, 5120);\n"
GATED_CASES_BLOCK = ('            if (const char* uh_g = std::getenv("UH_GATED_CASES"))\n'
                     '                for (const char* s = uh_g; *s;) {\n'
                     '                    long long um = 0, un = 0, uk = 0; int used = 0;\n'
                     '                    if (std::sscanf(s, "%lld:%lld:%lld%n", &um, &un, &uk, &used) < 3) break;\n'
                     '                    for (const int b : { 6, 7 }) iqr_gated_case(pl, b, um, un, uk);\n'
                     '                    s += used;\n'
                     '                    if (*s == \',\') ++s;\n'
                     '                }\n')


# The IQR case pins the int8 reference row's K split to the IQR row's, so the two sum their partials in one order. In an
# overlay run the int8 row comes from the base plugin, so the pin must reach the base too (else the reference sums in its
# own order and differs from the IQR row by rounding at M 17 to 64, while the IQR output itself is unchanged).
PIN_ANCHOR = ("    if (M <= 64 && pl.pin_split_waves) pl.pin_split_waves(512);\n"
              "    const int rr = launch(ri, ti, gp);\n"
              "    if (M <= 64 && pl.pin_split_waves) pl.pin_split_waves(0);\n")
PIN_NEW = ("    const Plugin* uh_pin = uh_base() ? uh_base() : &pl;\n"
           "    if (M <= 64 && uh_pin->pin_split_waves) uh_pin->pin_split_waves(512);\n"
           "    const int rr = launch(ri, ti, gp);\n"
           "    if (M <= 64 && uh_pin->pin_split_waves) uh_pin->pin_split_waves(0);\n")


def inject_selftest(text: str) -> str:
    if text.count(DEVICE_ANCHOR)!=1:
        raise HarnessError("selftest main anchor missing or ambiguous")
    text=text.replace(DEVICE_ANCHOR,DEVICE_ANCHOR+DEVICE_CHECK,1)
    if text.count(SYM_ANCHOR) != 1:
        raise HarnessError("the selftest changed: the dlsym check is not found exactly once")
    text = text.replace(SYM_ANCHOR, SYM_TOLERANT + SYM_ANCHOR, 1)
    for anchor in (HASH_ANCHOR, CASES_ANCHOR):
        if text.count(anchor) != 1:
            raise HarnessError("the selftest changed: the harness's edit anchor is not found exactly once "
                               f"({anchor.splitlines()[0].strip()!r})")
    text = text.replace(HASH_ANCHOR, HASH_ANCHOR.splitlines(keepends=True)[0] + HASH_PRINT + HASH_ANCHOR.splitlines(keepends=True)[1], 1)
    text = text.replace(CASES_ANCHOR, CASES_ANCHOR + CASES_BLOCK, 1)
    for anchor in (OWNED_GEMM_ANCHOR, OWNED_GATED_ANCHOR, GATED_HASH_ANCHOR, GATED_CASES_ANCHOR):
        if text.count(anchor) != 1:
            raise HarnessError("the selftest changed: the harness's owned-IQR edit anchor is not found exactly once "
                               f"({anchor.splitlines()[0].strip()!r})")
    first = OWNED_GEMM_ANCHOR.splitlines(keepends=True)
    text = text.replace(OWNED_GEMM_ANCHOR, first[0] + OWNED_GEMM_SKIP + first[1], 1)
    first = OWNED_GATED_ANCHOR.splitlines(keepends=True)
    text = text.replace(OWNED_GATED_ANCHOR, first[0] + OWNED_GATED_SKIP + first[1], 1)
    last = GATED_HASH_ANCHOR.splitlines(keepends=True)
    text = text.replace(GATED_HASH_ANCHOR, last[0] + GATED_HASH_PRINT + last[1], 1)
    text = text.replace(GATED_CASES_ANCHOR, GATED_CASES_ANCHOR + GATED_CASES_BLOCK, 1)
    for anchor in (IQ8R_SHAPES_ANCHOR, IQ8R_HASH_ANCHOR):
        if text.count(anchor) != 1:
            raise HarnessError("the selftest changed: the harness's IQ8R edit anchor is not found exactly once "
                               f"({anchor.splitlines()[0].strip()!r})")
    text = text.replace(IQ8R_SHAPES_ANCHOR, IQ8R_SHAPES_NEW, 1)
    # the family's expert shapes run after the case's own large down passes, inside its w4 block
    if text.count(MOE_LARGE_PASSES) != 1:
        raise HarnessError("the selftest changed: the MoE large down passes are not found exactly once")
    text = text.replace(MOE_LARGE_PASSES, MOE_LARGE_PASSES + MOE_CASES_BLOCK, 1)
    moe = MOE_HASH_ANCHOR.splitlines(keepends=True)
    text = _edit_function(text, MOE_FUNCTION, MOE_HASH_ANCHOR, moe[0] + moe[1] + MOE_HASH_PRINT + moe[2])
    for anchor in (I8A8_CASES_ANCHOR, I8A8_HASH_ANCHOR):
        if text.count(anchor) != 1:
            raise HarnessError("the selftest changed: the harness's int8 trunk edit anchor is not found exactly once "
                               f"({anchor.splitlines()[0].strip()!r})")
    text = text.replace(I8A8_CASES_ANCHOR, I8A8_CASES_ANCHOR + I8A8_CASES_BLOCK, 1)
    text = text.replace(I8A8_HASH_ANCHOR, I8A8_HASH_ANCHOR + I8A8_HASH_PRINT, 1)
    for anchor in (HEAD_CASES_ANCHOR, HEAD_HASH_ANCHOR):
        if text.count(anchor) != 1:
            raise HarnessError("the selftest changed: the harness's int8 head edit anchor is not found exactly once "
                               f"({anchor.splitlines()[0].strip()!r})")
    text = text.replace(HEAD_CASES_ANCHOR, HEAD_CASES_ANCHOR + HEAD_CASES_BLOCK, 1)
    head = HEAD_HASH_ANCHOR.splitlines(keepends=True)
    text = text.replace(HEAD_HASH_ANCHOR, head[0] + head[1] + HEAD_HASH_PRINT + head[2], 1)
    lines = IQ8R_HASH_ANCHOR.splitlines(keepends=True)
    text = text.replace(IQ8R_HASH_ANCHOR, lines[0] + lines[1] + IQ8R_HASH_PRINT + lines[2], 1)
    for anchor in (RUN_STRUCT_ANCHOR, RUN_FORMS_ANCHOR):
        if text.count(anchor) != 1:
            raise HarnessError("the selftest changed: the harness's IQR run-length edit anchor is not found exactly once "
                               f"({anchor.splitlines()[0].strip()!r})")
    text = text.replace(RUN_STRUCT_ANCHOR, RUN_STRUCT_NEW, 1)
    text = text.replace(RUN_FORMS_ANCHOR, RUN_FORMS_NEW, 1)
    text = _edit_function(text, RUN_FUNCTION, RUN_NAME_ANCHOR, RUN_NAME + RUN_NAME_ANCHOR)
    text = _edit_function(text, RUN_FUNCTION, RUN_TABLE_ANCHOR, RUN_TABLE_NEW)
    text = _edit_function(text, RUN_FUNCTION, RUN_WINDOW_ANCHOR, RUN_WINDOW_NEW)
    for anchor in (V4_PLANES_SIG, GATED_SIG, GATED_NAME_ANCHOR, GATED_PLANES_ANCHOR):
        if text.count(anchor) != 1:
            raise HarnessError("the selftest changed: the harness's gated run-length edit anchor is not found exactly once "
                               f"({anchor.splitlines()[0].strip()!r})")
    text = text.replace(V4_PLANES_SIG, V4_PLANES_SIG_NEW, 1)
    text = _edit_function(text, V4_PLANES_FUNCTION, V4_PLANES_RUN_ANCHOR, V4_PLANES_RUN_NEW)
    text = text.replace(GATED_SIG, GATED_SIG_NEW, 1)
    text = text.replace(GATED_NAME_ANCHOR, GATED_NAME_ANCHOR + GATED_NAME_RUN, 1)
    text = text.replace(GATED_PLANES_ANCHOR, GATED_PLANES_NEW, 1)
    if text.count(GATED_CASES_ANCHOR + GATED_CASES_BLOCK) != 1:
        raise HarnessError("the gated cases block is not in place for the run-length cases")
    text = text.replace(GATED_CASES_ANCHOR + GATED_CASES_BLOCK, GATED_CASES_ANCHOR + GATED_CASES_BLOCK + RUN_GATED_CASES, 1)
    for anchor in (FIND_ANCHOR, EMBED_ANCHOR, EMBED_CASES_ANCHOR, VOCAB_ANCHOR, VOCAB_CASES_ANCHOR):
        if text.count(anchor) != 1:
            raise HarnessError(f"the selftest changed: the harness's edit anchor is not found exactly once ({anchor.splitlines()[0].strip()!r})")
    text = text.replace(FIND_ANCHOR, FIND_NEW, 1)
    if text.count(PIN_ANCHOR) != 1:
        raise HarnessError("the selftest changed: the int8 reference split pin is not found exactly once")
    text = text.replace(PIN_ANCHOR, PIN_NEW, 1)
    text = text.replace(EMBED_ANCHOR, EMBED_ANCHOR + EMBED_PRINT, 1)
    text = text.replace(EMBED_CASES_ANCHOR, EMBED_CASES_ANCHOR + EMBED_CASES_BLOCK, 1)
    text = text.replace(VOCAB_ANCHOR, VOCAB_ANCHOR + VOCAB_PRINT, 1)
    return text.replace(VOCAB_CASES_ANCHOR, VOCAB_CASES_ANCHOR + VOCAB_CASES_BLOCK, 1)


def run(argv: list[str], log: Path, *, env: dict | None = None, cwd: Path | None = None) -> None:
    with log.open("ab") as handle:
        handle.write(("$ " + " ".join(argv) + "\n").encode())
        handle.flush()
        done = subprocess.run(argv, stdout=handle, stderr=subprocess.STDOUT, env=env, cwd=cwd, check=False)
    if done.returncode:
        tail = log.read_text(errors="replace").splitlines()[-25:]
        raise HarnessError(f"{argv[0]} failed ({done.returncode}); {log}:\n" + "\n".join(tail))


UNIT_BLOCK = re.compile(r'set\(R4D_UNITS\n(.*?)\n\)\n', re.S)
STUBS = r'''
# harness: entry points of the units left out become stubs, so the plugin loads with only the units under test
find_program(UH_NM nm REQUIRED)
find_package(Python3 REQUIRED COMPONENTS Interpreter)
add_custom_command(TARGET libr4d PRE_LINK
  COMMAND ${Python3_EXECUTABLE} "@STUBGEN@" ${UH_NM} ${CMAKE_BINARY_DIR}/uh_stubs.S "@REFERENCE@" $<TARGET_OBJECTS:libr4d> ${R4D_HSIMD_OBJS}
  COMMAND ${CMAKE_CXX_COMPILER} -c ${CMAKE_BINARY_DIR}/uh_stubs.S -o ${CMAKE_BINARY_DIR}/uh_stubs.o
  COMMAND_EXPAND_LISTS VERBATIM)
target_link_options(libr4d PRIVATE ${CMAKE_BINARY_DIR}/uh_stubs.o)
'''


def cmake_build(source: Path, build: Path, options: list[str], targets: list[str], log: Path) -> None:
    env = dict(os.environ)
    run(["cmake", "-S", str(source), "-B", str(build), "-G", "Ninja", *options], log, env=env)
    run(["cmake", "--build", str(build), "--target", *targets, "-j", str(max(1, (os.cpu_count() or 2) // 2))], log, env=env)


def build_plugin(arch: str, tree: Path, units: list[str], out: Path, reference: Path, libr3: Path | None = None) -> Path:
    """Build ``units`` for ``arch`` into a plugin; returns the plugin file."""
    log = out / "build.log"
    prefix = out / "prefix"
    write_prefix(tree, arch, prefix)
    build = out / "build"
    if arch == "gfx1100":
        options = ["-DCMAKE_BUILD_TYPE=Release", f"-DCMAKE_PREFIX_PATH={prefix}", f"-DRADIANCE_SRC={tree}", "-DR3_TARGETS=gfx1100", "-DR3_NATIVE=ON",
                   "-DR3_ONLY=" + ";".join(units)]
        cmake_build((libr3 or ROOT / "engines/radiance/components/libr3"), build, options, ["libr3"], log)
    elif arch == "gfx1201":
        cmakefile = tree / "libr4d" / "CMakeLists.txt"
        text = cmakefile.read_text(encoding="utf-8")
        match = UNIT_BLOCK.search(text)
        if not match:
            raise HarnessError("libr4d/CMakeLists.txt: the R4D_UNITS block was not found")
        lines = {m.group(1): m.group(0) for m in re.finditer(r'^\s*"(r4d_[A-Za-z0-9_]+)\|[^"]*"', match.group(1), re.M)}
        missing = [u for u in units if u not in lines]
        if missing:
            raise HarnessError(f"not libr4d units: {missing}")
        text = text[:match.start()] + "set(R4D_UNITS\n" + "\n".join(lines[u] for u in units) + "\n)\n" + text[match.end():]
        text += STUBS.replace("@STUBGEN@", str(Path(__file__).with_name("unit_stubs.py"))).replace("@REFERENCE@", str(reference))
        cmakefile.write_text(text, encoding="utf-8")
        cmake_build(tree / "libr4d", build, ["-DCMAKE_BUILD_TYPE=Release", f"-DCMAKE_PREFIX_PATH={prefix}", "-DRAD_GPU_TARGETS=gfx1201"], ["libr4d"], log)
    else:
        raise HarnessError(f"architecture {arch}: gfx1100 or gfx1201")
    found = sorted(build.rglob(PLUGIN[arch]))
    if not found:
        raise HarnessError(f"{PLUGIN[arch]} was not built; see {log}")
    return found[0]


def selftest_binary(arch: str, tree: Path, out: Path, libr3: Path | None = None) -> Path:
    """The selftest with the harness's edits, built once per (architecture, edited source) and kept."""
    if arch == "gfx1100":
        # the rewritten copy libr3 runs on gfx11; configure makes it, one cheap unit is enough
        scratch = out / "selftest-build"
        prefix = out / "prefix"
        probe = scratch / "r4d-src" / "r4d_selftest.cpp"
        run(["cmake", "-S", str(libr3 or ROOT / "engines/radiance/components/libr3"), "-B", str(scratch), "-G", "Ninja",
             "-DCMAKE_BUILD_TYPE=Release", f"-DCMAKE_PREFIX_PATH={prefix}", f"-DRADIANCE_SRC={tree}",
             "-DR3_TARGETS=gfx1100", "-DR3_NATIVE=ON", "-DR3_ONLY=r4d_prefetch"], out / "selftest.log")
        source = probe
        target_build = scratch
    else:
        source = tree / "libr4d" / "r4d_selftest.cpp"
        target_build = out / "build"
    edited = inject_selftest(source.read_text(encoding="utf-8"))
    key = hashlib.sha256(edited.encode()).hexdigest()[:16]
    cached = work_root() / "unit-harness" / f"selftest-{arch}-{key}"
    if cached.is_file():
        return cached
    source.write_text(edited, encoding="utf-8")
    log = out / "selftest.log"
    run(["cmake", "--build", str(target_build), "--target", SELFTEST_TARGET[arch], "-j", str(max(1, (os.cpu_count() or 2) // 2))], log)
    binary = next(iter(sorted(target_build.rglob(SELFTEST_TARGET[arch]))), None)
    if binary is None or not binary.is_file():
        raise HarnessError(f"{SELFTEST_TARGET[arch]} was not built; see {log}")
    cached.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(binary, cached)
    return cached


# --------------------------------------------------------------------------------------------------------- resources

BUNDLE_MAGIC = b"__CLANG_OFFLOAD_BUNDLE__"


def embedded_objects(plugin: Path, arch: str) -> list[bytes]:
    """The device code objects for ``arch`` carried in a plugin's offload bundles."""
    raw = plugin.read_bytes()
    found: list[bytes] = []
    for match in re.finditer(BUNDLE_MAGIC, raw):
        start = match.start()
        try:
            count = struct.unpack_from("<Q", raw, start + len(BUNDLE_MAGIC))[0]
            if not 1 <= count <= 60:
                continue
            pointer = start + len(BUNDLE_MAGIC) + 8
            for _ in range(count):
                offset, size, name_length = struct.unpack_from("<QQQ", raw, pointer)
                pointer += 24
                if name_length > 200 or size > 2 ** 28 or start + offset + size > len(raw):
                    raise ValueError
                target = raw[pointer:pointer + name_length].decode("ascii")
                pointer += name_length
                if target.endswith(arch):
                    found.append(raw[start + offset:start + offset + size])
        except (UnicodeDecodeError, ValueError, struct.error):
            continue
    return found


def parse_notes(notes: str) -> dict[str, dict[str, int]]:
    kernels: dict[str, dict[str, int]] = {}
    for part in notes.split("  - .args:"):
        name = re.search(r"\.name:\s+(\S+)", part)
        if not name or not re.search(r"\.vgpr_count:", part):
            continue
        values = {}
        for key in RESOURCE_KEYS:
            hit = re.search(r"\." + key + r":\s+(\d+)", part)
            if not hit:
                raise HarnessError(f"kernel {name.group(1)}: no {key} in the code object notes")
            values[key] = int(hit.group(1))
        kernels[name.group(1)] = values
    return kernels


def kernel_code(plugin: Path, arch: str) -> dict[str, list[str]]:
    """Per kernel name, a digest of the instructions of every code object that carries it, addresses and comments taken
    out. Equal digests mean the same code: a row whose kernels are all unchanged needs no replay, however its timing
    against another library build moves (the 2.3.0 IQ8R tiled row measured -14% to +65% against byte-identical code)."""
    code: dict[str, list[str]] = {}
    objdump = rocm_tool("llvm-objdump")
    with tempfile.TemporaryDirectory(prefix="uh-code-", dir=str(work_root() / "unit-harness")) as scratch:
        for index, blob in enumerate(embedded_objects(plugin, arch)):
            path = Path(scratch) / f"{index}.co"
            path.write_bytes(blob)
            text = subprocess.run([objdump, "-d", "--no-show-raw-insn", "--no-leading-addr", str(path)],
                                  capture_output=True, text=True, check=True).stdout
            for match in re.finditer(r"\n<?([^\s<>]+)>?:\n(.*?)(?=\n<?[^\s<>]+>?:\n|\Z)", text, re.S):
                body = re.sub(r"0x[0-9a-f]+|<[^>]*>", "", re.sub(r"//.*", "", match.group(2)))
                code.setdefault(match.group(1), []).append(hashlib.sha256(body.encode()).hexdigest())
    return {name: sorted(digests) for name, digests in code.items()}


def code_changed(ref: dict[str, list[str]], cand: dict[str, list[str]], pattern: re.Pattern | None) -> list[str]:
    """The candidate's kernels (of the family, when it names a pattern) whose code differs from the reference or is new."""
    return sorted(name for name, digests in cand.items()
                  if (pattern is None or pattern.search(name)) and ref.get(name) != digests)


def kernel_records(plugin: Path, arch: str) -> dict[str, list[dict[str, int]]]:
    """Per kernel name, the records of every code object that carries it (units repeat names: an anonymous-namespace
    template instantiated in several units is one name with several records)."""
    records: dict[str, list[dict[str, int]]] = {}
    readobj = rocm_tool("llvm-readobj")
    with tempfile.TemporaryDirectory(prefix="uh-co-", dir=str(work_root() / "unit-harness")) as scratch:
        for index, blob in enumerate(embedded_objects(plugin, arch)):
            path = Path(scratch) / f"{index}.co"
            path.write_bytes(blob)
            notes = subprocess.run([readobj, "--notes", str(path)], capture_output=True, text=True, check=True).stdout
            for name, values in parse_notes(notes).items():
                records.setdefault(name, []).append(values)
    for name in records:
        records[name].sort(key=lambda v: tuple(v[k] for k in RESOURCE_KEYS))
    return records


def diff_records(ref: dict, cand: dict, family: re.Pattern | None) -> dict:
    """Kernels the candidate adds or changes against the reference, and the family's kernels it no longer has. A kernel
    is changed when its records differ as a multiset (a name carried by several units is compared as a whole)."""
    added = sorted(k for k in cand if k not in ref)
    changed = {}
    for k in sorted(cand):
        if k in ref and ref[k] != cand[k]:
            changed[k] = {"ref": ref[k], "cand": cand[k]} if len(ref[k]) != len(cand[k]) else {
                key: [[r[key] for r in ref[k]], [c[key] for c in cand[k]]]
                for key in RESOURCE_KEYS if [r[key] for r in ref[k]] != [c[key] for c in cand[k]]}
    removed = sorted(k for k in ref if k not in cand and family is not None and family.search(k))
    return {"added": added, "changed": changed, "removed": removed,
            "unchanged": sum(1 for k in cand if k in ref and ref[k] == cand[k])}


def resource_summary(diff: dict, cand: dict) -> str:
    lines = [f"resources: {len(diff['added'])} added, {len(diff['changed'])} changed, {len(diff['removed'])} removed, "
             f"{diff['unchanged']} unchanged"]
    if diff["added"]:
        recs = [r for k in diff["added"] for r in cand[k]]
        vg = [r["vgpr_count"] for r in recs]
        spills = sum(r["vgpr_spill_count"] + r["sgpr_spill_count"] for r in recs)
        private = sum(r["private_segment_fixed_size"] for r in recs)
        lds = sorted({r["group_segment_fixed_size"] for r in recs})
        lines.append(f"  added kernels: VGPR {min(vg)}..{max(vg)}, spills {spills}, private bytes {private}, LDS {lds}")
    for kernel, delta in list(diff["changed"].items())[:20]:
        lines.append(f"  CHANGED {kernel[:90]}: " + ", ".join(f"{k} {a}->{b}" for k, (a, b) in delta.items()
                                                              if isinstance(b, list)))
    for kernel in diff["removed"][:20]:
        lines.append(f"  REMOVED {kernel[:100]}")
    return "\n".join(lines)


# ------------------------------------------------------------------------------------------------------------ hash

def hash_cases(spec: dict) -> str:
    if "cases" in spec:                                   # a family whose case takes its own case format
        return ",".join(spec["cases"])
    cases = [f"{n}:{k}:{m}" for n, k in spec["shapes"] for m in spec["rows"]]
    n, k = spec["tail_shape"]
    cases += [f"{n}:{k}:{m}" for m in spec["tail_rows"]]
    return ",".join(cases)


def selftest_env(arch: str, device: tuple[str,str], extra: dict | None = None) -> dict:
    env = {k:v for k,v in os.environ.items() if not k.startswith(("UH_","R4D_PERF_ONLY"))}
    env.update(HIP_VISIBLE_DEVICES=device[0], UH_EXPECT_BDF=device[1], UH_EXPECT_ARCH=arch, OPENBLAS_NUM_THREADS="8", UH_TOLERATE_ABSENT="1")
    if extra and set(extra)&{"HIP_VISIBLE_DEVICES","UH_EXPECT_BDF","UH_EXPECT_ARCH"}:
        raise HarnessError("selftest extra environment cannot override physical identity")
    env.update(extra or {})
    return env


def run_selftest(binary: Path, plugin: Path, args: list[str], env: dict, timeout: int) -> str:
    from bigcherry.hardware import lease
    if not lease.held_by_this_process():
        raise HarnessError("GPU lease not verified before selftest")
    # Poll while the child runs; terminate only this child on timeout/lease loss.
    with tempfile.TemporaryFile() as log:
        proc=subprocess.Popen([str(binary),str(plugin),*args],env=env,stdout=log,stderr=subprocess.STDOUT)
        deadline=time.monotonic()+timeout
        failure=None
        try:
            while proc.poll() is None:
                if not lease.held_by_this_process() or time.monotonic()>=deadline:
                    failure="lease lost or selftest timeout"
                    break
                time.sleep(0.25)
        finally:
            if proc.poll() is None:
                proc.kill()
            proc.wait()
        log.seek(0)
        text=log.read().decode(errors="replace")
    marker="UNITDEVICE "+env["UH_EXPECT_BDF"]+" "+env["UH_EXPECT_ARCH"]
    if text.splitlines().count(marker)!=1:
        text+="\nFAIL missing or mismatched physical GPU identity\n"
    if proc.returncode or failure or not lease.held_by_this_process():
        text+=f"\nFAIL selftest process exit {proc.returncode}: {failure or 'process/lease status'}\n"
    return text


def file_sha256(path: Path) -> str:
    with path.open("rb") as source:
        return hashlib.file_digest(source,"sha256").hexdigest()


def parse_hashes(text: str) -> tuple[dict[str, str], list[str]]:
    hashes = {}
    failures = [line for line in text.splitlines() if re.match(r"^\s*(FAIL|not ok|skip)\b", line,re.I) or re.fullmatch(r"\s*(?:[1-9]\d* failure\(s\), \d+|\d+ failure\(s\), [1-9]\d*) skipped\s*",line)]
    for line in text.splitlines():
        if not line.startswith("UNITHASH"):
            continue
        match=re.fullmatch(r"UNITHASH (.+) ([0-9a-f]{16})",line)
        if not match:
            failures.append("malformed hash record")
            continue
        name,value=match.groups()
        if name in hashes:
            failures.append(f"duplicate hash case: {name}")
        else:
            hashes[name]=value
    return hashes, failures


def compare_hashes(ref: dict[str, str], cand: dict[str, str]) -> dict:
    return {"compared": sum(1 for k in ref if k in cand),
            "differ": sorted(k for k in ref if k in cand and ref[k] != cand[k]),
            "missing_in_candidate": sorted(k for k in ref if k not in cand),
            "only_in_candidate": sorted(k for k in cand if k not in ref)}


# ---------------------------------------------------------------------------------------------------------- replay

def converged(windows: list[float], tol: float) -> bool:
    """Two medians agree: the two lowest window medians are within ``tol`` of each other. The lowest, not the latest:
    a card's slow mode (the R9700 runs 1.7x slower for stretches) makes a window slow, never fast, and two slow windows
    can agree."""
    low = sorted(windows)[:2]
    return len(windows) >= 4 and len(low) == 2 and low[1] - low[0] <= tol * low[0]


def estimate(windows: list[float], drop_first: bool = False) -> float:
    """The arm's time: its fastest window's median (a slow-mode window is slow, never fast)."""
    return min(windows)


REPLAY_MODES = ("cold","warm")


def replay_command(spec,point,iters,mode):
    args=[a.format(**point,iters=iters) for a in spec["command"]]
    if args.count("--cold")!=1 or mode not in REPLAY_MODES:
        raise HarnessError("replay needs one configured cold flush and an explicit mode")
    i=args.index("--cold")
    if i+1>=len(args) or not args[i+1].isdigit() or int(args[i+1])<=0:
        raise HarnessError("invalid configured cold flush")
    if mode=="warm":
        del args[i:i+2]
    return args


def point_time(binary: Path, plugin: Path, arch: str, device: tuple[str,str], spec: dict, point: dict, row: str, iters: int,
               extra_env: dict | None = None) -> float:
    args = replay_command(spec,point,iters,point["mode"])
    label = spec["label"].format(**point)
    text = run_selftest(binary, plugin, args, selftest_env(arch, device, {"R4D_PERF_ONLY": row, **(extra_env or {})}), timeout=900)
    values = [float(m.group(1)) for line in text.splitlines() if label in line for m in re.finditer(r"(\d+(?:\.\d+)?)\s+us", line)]
    if parse_hashes(text)[1] or len(values)!=1 or not math.isfinite(values[0]) or values[0]<=0:
        raise HarnessError(f"no valid time for {label} {point}: {text[-300:]}")
    return values[0]


def replay_point(binary: Path, ref: Path, cand: Path, arch: str, device: tuple[str,str], spec: dict, point: dict, row: str,
                 *, iters: int, tol: float, max_windows: int, arm_env: dict | None = None) -> dict:
    """ABBA windows, each plugin in its own process, until each arm's last two windows agree (R9700: the first window
    is the power ramp and is dropped)."""
    windows = {"ref": [], "cand": []}
    order = ("ref", "cand", "cand", "ref")
    plugins = {"ref": ref, "cand": cand}
    drop_first = False
    done = False
    rounds = 0
    while not done and rounds < max_windows:
        for arm in order[:2] if rounds % 2 == 0 else order[2:]:
            windows[arm].append(point_time(binary, plugins[arm], arch, device, spec, point, row, iters, (arm_env or {}).get(arm)))
        rounds += 1
        done = all(converged(w, tol) for w in windows.values())
    ref_us, cand_us = estimate(windows["ref"], drop_first), estimate(windows["cand"], drop_first)
    return {**point, "ref_us": round(ref_us, 2), "cand_us": round(cand_us, 2), "ratio": round(cand_us / ref_us, 4),
            "windows": rounds, "converged": done}


def replay_points(spec: dict, bits: list[int] | None = None, rows: list[int] | None = None) -> list[dict]:
    return [{"N":n,"K":k,"M":m,"B":b,"mode":mode} for n,k in spec["shapes"] for m in (rows or spec["rows_m"]) for b in (bits or spec["bits"]) for mode in REPLAY_MODES]


def classify(ratio: float, tol: float) -> str:
    return "win" if ratio < 1 - tol else "regress" if ratio > 1 + tol else "same"


# ------------------------------------------------------------------------------------------------------------ stages

def artifact_identity(state):
    paths={key:Path(state[key]) for key in ("plugin","reference_plugin","selftest","base_stage") if key in state}
    return {key:file_sha256(path) for key,path in paths.items()} | {"family_policy":file_sha256(FAMILIES_FILE)}


def require_artifacts(state):
    if state.get("artifact_sha256")!=artifact_identity(state):
        raise HarnessError("build artifact or family policy changed/missing identity; rebuild")


def stage_build(args) -> Path:
    fam = families().get(args.family) if args.family else None
    units = list(fam["units"]) if fam else [u for chunk in (args.unit or []) for u in chunk.split(",")]
    if not units:
        raise HarnessError("name a --family or --unit")
    out = Path(args.out) if args.out else work_root() / "unit-harness" / f"{args.family or 'units'}-{args.arch}-{time.strftime('%Y%m%d-%H%M%S', time.gmtime())}"
    if any((out / f).exists() for f in ("build.json","measure.json","measure-replay.json","src")):
        raise HarnessError("output already contains build/evidence; select a new --out")
    out.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    tree = compose(args.patch or [], Path(args.tree) if args.tree else None, out, args.without or [])
    tree_key = out / "tree"
    tree_key.write_text(str(tree), encoding="utf-8")
    ref_plugin = Path(args.reference).resolve(strict=True) if args.reference else release_plugin(args.release, args.arch)
    plugin = build_plugin(args.arch, tree, units + [u for u in (fam or {}).get("support_units", []) if u not in units], out, ref_plugin,
                         Path(args.libr3) if args.libr3 else None)
    built = time.monotonic() - started
    binary = selftest_binary(args.arch, tree, out, Path(args.libr3) if args.libr3 else None)
    ref = kernel_records(ref_plugin, args.arch)
    cand = kernel_records(plugin, args.arch)
    pattern = re.compile(fam["kernels"]) if fam else None
    diff = diff_records(ref, cand, pattern)
    changed_code = code_changed(kernel_code(ref_plugin, args.arch), kernel_code(plugin, args.arch), pattern)
    state = {"family": args.family, "units": units, "arch": args.arch, "plugin": str(plugin), "selftest": str(binary),
             "release": ref_plugin.parts[-4] if not args.reference else "reference plugin " + str(ref_plugin), "reference_plugin": str(ref_plugin), "resources": diff,
             "code_changed": changed_code, "build_seconds": round(built, 1)}
    state["artifact_sha256"] = artifact_identity(state)
    (out / "build.json").write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")
    print(f"built {', '.join(units)} for {args.arch} in {built:.0f} s (selftest {'cached' if built < 5 else 'ready'}): {plugin}")
    print(resource_summary(diff, cand))
    print(f"code: {len(changed_code)} family kernel(s) differ from the reference" + ("" if changed_code else "; replay not needed"))
    return out


def stage_measure(out: Path, *, bdf: str, replay_iters: int = 15, tol: float = 0.015, max_windows: int = 14, verdict_tol: float = 0.04, skip_replay: bool = False,
                  skip_hash: bool = False, bits: list[int] | None = None, rows: list[int] | None = None) -> dict:
    if (out / "measure.json").exists():
        raise HarnessError("measurement already exists; use a new build output")
    state = json.loads((out / "build.json").read_text(encoding="utf-8"))
    require_artifacts(state)
    fam = families().get(state["family"]) if state["family"] else None
    if not fam:
        raise HarnessError("measure needs a family (hash and replay matrices come from unit_families.toml)")
    arch = state["arch"]
    device = (device_index(arch,bdf),bdf)
    from bigcherry.hardware.ai_probe import snapshot
    before,_=snapshot(bdf)
    binary, plugin, ref = Path(state["selftest"]), Path(state["plugin"]), Path(state["reference_plugin"])
    result: dict = {"arch": arch, "device":device[0],"bdf":bdf,"artifact_sha256":state["artifact_sha256"]}
    env = {"UH_HASH_CASES": hash_cases(fam["hash"])}
    texts = {}
    for arm, path in ((("ref", ref), ("cand", plugin)) if not skip_hash else ()):
        started = time.monotonic()
        texts[arm] = run_selftest(binary, path, ["--case", fam["case"]], selftest_env(arch, device, env), timeout=3 * 3600)
        (out / f"selftest-{arm}.log").write_text(texts[arm], encoding="utf-8")
        print(f"selftest {arm}: {time.monotonic() - started:.0f} s")
    ref_hashes, ref_fail = parse_hashes(texts.get("ref", ""))
    cand_hashes, cand_fail = parse_hashes(texts.get("cand", ""))
    # a skip the family names (the case's own refusal of a form outside the kernel's contract) is expected, not a failure
    expected = [re.compile(p) for p in fam.get("expected_skips", [])]
    def unexpected(lines):
        left = [l for l in lines if not (re.match(r"^\s*skip\b", l) and any(p.search(l) for p in expected))]
        skipped = any(re.match(r"^\s*skip\b", l) for l in lines)
        # the selftest exits 2 when its only findings are skips; with every skip expected that exit is clean
        if skipped and not any(re.match(r"^\s*skip\b", l) for l in left):
            left = [l for l in left if not l.startswith("FAIL selftest process exit 2:")]
        return left
    ref_fail, cand_fail = unexpected(ref_fail), unexpected(cand_fail)
    result["hash"] = compare_hashes(ref_hashes, cand_hashes)
    result["selftest_failures"] = {"ref": len(ref_fail), "cand": len(cand_fail), "cand_lines": cand_fail[:10]}
    print(f"hash: {result['hash']['compared']} compared, {len(result['hash']['differ'])} differ, "
          f"{len(result['hash']['missing_in_candidate'])} missing in the candidate; selftest failures: ref {len(ref_fail)}, candidate {len(cand_fail)}")
    # Replay times changed code: when no kernel of the family differs from the reference, there is nothing to time.
    result["code_changed"] = state.get("code_changed")
    # A family without a timing command can still pass on identical code; changed code then cannot be certified.
    result["replay_configured"] = "replay" in fam
    changed_rows = list(fam["replay"]["rows"]) if "replay" in fam and state.get("code_changed") != [] else []
    result["required_replay"] = [{"row": row, **point} for row in changed_rows for point in replay_points(fam["replay"])]
    result["replay"] = []
    if changed_rows and not skip_replay:
        spec = fam["replay"]
        for row in changed_rows:
            for point in replay_points(spec, bits, rows):
                got = replay_point(binary, ref, plugin, arch, device, spec, point, row, iters=replay_iters, tol=tol,
                                   max_windows=max_windows)
                got["row"], got["verdict"] = row, classify(got["ratio"], verdict_tol)
                result["replay"].append(got)
                print(f"replay {row} {point['mode']} N={point['N']} K={point['K']} M={point['M']} B={point['B']}: "
                      f"{got['ref_us']} -> {got['cand_us']} us ({got['ratio']}) {got['verdict']}"
                      f"{'' if got['converged'] else ' UNCONVERGED'}", flush=True)
    elif state.get("code_changed") == []:
        print("replay: not needed, the family's kernel code is identical to the reference")
    elif not changed_rows:
        print("replay: no rows configured")
    result["observed_start"]=before
    result["observed_stop"]=snapshot(bdf)[0]
    require_artifacts(state)
    (out / "measure.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def configured_measurement_outcome(measure: dict) -> tuple[str, list[str]]:
    """Verdict on the discrete harness checks, not on any production/power profile."""
    h = measure["hash"]
    failures = measure["selftest_failures"]
    rejected, incomplete = [], []
    if failures["ref"]:
        incomplete.append("reference selftest failed")
    if failures["cand"]:
        rejected.append("candidate selftest failed")
    if h["differ"] or h["missing_in_candidate"]:
        rejected.append("candidate output differs or is missing")
    if h["compared"] <= 0 or h["only_in_candidate"]:
        incomplete.append("no positive matching hash coverage")
    def key(point):
        return tuple(point[k] for k in ("row", "N", "K", "M", "B", "mode"))
    required = measure.get("required_replay", [])
    actual = measure["replay"]
    code_identical = measure.get("code_changed") == [] and not required and not actual
    if measure.get("replay_configured") is False and measure.get("code_changed") != []:
        incomplete.append("kernel code changed and the family has no replay configured")
    elif (not required and not code_identical) or len({key(p) for p in required}) != len(required) or len({key(p) for p in actual}) != len(actual) or {key(p) for p in required} != {key(p) for p in actual}:
        incomplete.append("configured replay coverage incomplete")
    if any(p.get("mode") not in REPLAY_MODES or type(p.get("converged")) is not bool or p.get("verdict") not in ("win","same","regress") for p in actual):
        incomplete.append("invalid replay verdict")
    if any(not p["converged"] for p in actual):
        incomplete.append("replay did not converge")
    if any(p["converged"] and p["verdict"] == "regress" for p in actual):
        rejected.append("converged replay regression")
    # A failed reference cannot establish a comparison, even if the candidate
    # also failed. Keep both reasons, but do not certify a regression.
    if failures["ref"]:
        return "inconclusive", incomplete + rejected
    if rejected:
        return "rejected", rejected + incomplete
    if incomplete:
        return "inconclusive", incomplete
    return "checks_passed", []


def summary(out: Path) -> tuple[str, bool]:
    build = json.loads((out / "build.json").read_text(encoding="utf-8"))
    measure_path = out / "measure.json"
    outcome, reasons = "diagnostic", ["measurement not run"]
    lines = [f"unit harness: {build['family'] or ','.join(build['units'])} on {build['arch']} against {build['release']}"]
    res = build["resources"]
    lines.append(f"resources: {len(res['added'])} added, {len(res['changed'])} changed, {len(res['removed'])} removed, {res['unchanged']} unchanged")
    if res["removed"] or res["changed"]:
        lines.append("  REGRESSION CHECK: changed or removed kernels need a reason (see build.json)")
    if measure_path.exists():
        measure = json.loads(measure_path.read_text(encoding="utf-8"))
        h = measure["hash"]
        outcome, reasons = configured_measurement_outcome(measure)
        try:
            identity_ok=bool(build.get("artifact_sha256")) and measure.get("artifact_sha256")==build["artifact_sha256"]==artifact_identity(build)
        except OSError:
            identity_ok=False
        if not measure.get("observed_start") or measure.get("observed_start")!=measure.get("observed_stop"):
            outcome="inconclusive"
            reasons.append("physical controls/identity missing or changed")
        if not identity_ok:
            outcome="inconclusive"
            reasons.append("measurement artifact identity missing or stale")
        if build["resources"]["removed"]:
            outcome = "inconclusive" if outcome == "checks_passed" else outcome
            reasons.append("removed resource records require review")
        lines.append(f"hash: {h['compared']} compared, {len(h['differ'])} differ"
                     + (f"; first {h['differ'][0]}" if h["differ"] else ""))
        lines.append(f"selftest failures: reference {measure['selftest_failures']['ref']}, candidate {measure['selftest_failures']['cand']}")
        rep = measure["replay"]
        if rep:
            regress = [r for r in rep if r["verdict"] == "regress"]
            wins = [r for r in rep if r["verdict"] == "win"]
            lines.append(f"replay: {len(rep)} points, {len(wins)} win, {len(regress)} regress, "
                         f"{sum(1 for r in rep if r['verdict'] == 'same')} same, {sum(1 for r in rep if not r['converged'])} unconverged")
            for r in regress:
                lines.append(f"  REGRESS {r['row']} N={r['N']} K={r['K']} M={r['M']} B={r['B']}: {r['ref_us']} -> {r['cand_us']} us ({r['ratio']})")
    lines.extend(f"  {reason}" for reason in reasons)
    lines.append(f"unit harness: {outcome} (configured checks only; not profile qualification)")
    return "\n".join(lines), outcome == "checks_passed"


# ------------------------------------------------------------------------------------------------------------ overlays

def build_overlay(tree: Path, component: Path, plugin: str, out: Path) -> Path:
    """The overlay plugin (both architectures in one file), built from the composed source."""
    prefix = out / "prefix"
    write_prefix(tree, "gfx1100 gfx1201", prefix)
    build = out / "overlay-build"
    cmake_build(component, build, [f"-DCMAKE_PREFIX_PATH={prefix}", f"-DRADIANCE_SRC={tree}"], [plugin], out / "overlay.log")
    found = sorted(build.rglob(plugin + ".so"))
    if not found:
        raise HarnessError(f"{plugin}.so was not built; see {out / 'overlay.log'}")
    return found[0]


def stage_overlay_build(args) -> Path:
    ov = overlays().get(args.overlay)
    if not ov:
        raise HarnessError(f"unknown overlay {args.overlay!r} in unit_families.toml")
    out = Path(args.out) if args.out else work_root() / "unit-harness" / f"overlay-{args.overlay}-{args.arch}-{time.strftime('%Y%m%d-%H%M%S', time.gmtime())}"
    if any((out / f).exists() for f in ("build.json","measure.json","measure-replay.json","src")):
        raise HarnessError("output already contains build/evidence; select a new --out")
    out.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    tree = compose(args.patch or [], Path(args.tree) if args.tree else None, out, args.without or [])
    ref_plugin = Path(args.reference).resolve(strict=True) if args.reference else release_plugin(args.release, args.arch)
    # The base the overlay runs over is a minimal build of the same source (one unit; every other row's entry point is a stub, the
    # embed launch included), so a launch that reaches the base's kernel fails and a hash that matches the real base's proves
    # that the overlay's own kernel ran.
    minimal = build_plugin(args.arch, tree, list(ov.get("base-units", ["r4d_prefetch"])), out, ref_plugin,
                           Path(args.libr3) if args.libr3 else None)
    binary = selftest_binary(args.arch, tree, out, Path(args.libr3) if args.libr3 else None)
    so = build_overlay(tree, ROOT / "engines" / ENGINE / "components" / ov["component"], ov["plugin"], out)
    stage = out / "stage"
    stage.mkdir(exist_ok=True)
    shutil.copy2(minimal, stage / PLUGIN[args.arch])
    staged = stage / so.name
    shutil.copy2(so, staged)
    pattern = re.compile(ov["kernels"])
    ref = {k: v for k, v in kernel_records(ref_plugin, args.arch).items() if pattern.search(k)}
    cand = {k: v for k, v in kernel_records(staged, args.arch).items() if pattern.search(k)}
    diff = diff_records(ref, cand, None)   # what the base has beyond the overlay is not "removed": the overlay owns a part
    state = {"overlay": args.overlay, "arch": args.arch, "plugin": str(staged), "base_stage": str(stage / PLUGIN[args.arch]), "selftest": str(binary), "reference_plugin": str(ref_plugin),
             "release": ref_plugin.parts[-4] if not args.reference else "reference plugin " + str(ref_plugin), "resources": diff, "kernels_compared": len(cand), "build_seconds": round(time.monotonic() - started, 1)}
    state["artifact_sha256"] = artifact_identity(state)
    (out / "build.json").write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")
    print(f"built overlay {ov['plugin']} for {args.arch}: {len(cand)} kernels matching {ov['kernels']!r}; against {state['release']}: "
          f"{len(diff['added'])} added, {len(diff['changed'])} changed, {len(diff['removed'])} removed, {diff['unchanged']} identical records")
    for kernel, delta in list(diff["changed"].items())[:10]:
        print(f"  CHANGED {kernel[:90]}: " + ", ".join(f"{k} {a}->{b}" for k, (a, b) in delta.items() if isinstance(b, list)))
    return out


def consistent_round_hashes(texts: list[str]) -> tuple[dict[str, str], list[str]]:
    """Every measured round must agree, not only the first."""
    first, failures = parse_hashes(texts[0] if texts else "")
    for idx, text in enumerate(texts[1:], 2):
        hashes, errors = parse_hashes(text)
        failures.extend(errors)
        if hashes != first:
            failures.append(f"round {idx}: hash coverage or values differ from first round")
    return first, failures


def overlay_timing(texts, *, expect_timing):
    import statistics
    per_arm={}
    gaps=[]
    for arm,rounds in texts.items():
        values=[]
        for text in rounds:
            row={}
            for line in text.splitlines():
                if not line.startswith("UNITTIME"):
                    continue
                m=re.fullmatch(r"UNITTIME (.+) ([0-9.]+)",line)
                if not m:
                    gaps.append(f"{arm}: malformed timing")
                    continue
                name,raw=m.groups()
                try:
                    value=float(raw)
                except ValueError:
                    value=0
                if name in row or not math.isfinite(value) or value<=0:
                    gaps.append(f"{arm}: duplicate or invalid timing {name}")
                else:
                    row[name]=value
            if expect_timing and set(row)!=set(parse_hashes(text)[0]):
                gaps.append(f"{arm}: timing cases differ from hash cases")
            values.append(row)
        per_arm[arm]=values
    all_rounds=per_arm["ref"]+per_arm["cand"]
    keys=set(all_rounds[0]) if all_rounds else set()
    if ((expect_timing and not keys) or not all_rounds or
            len(per_arm["ref"])!=len(per_arm["cand"]) or any(set(r)!=keys for r in all_rounds)):
        gaps.append("overlay timing coverage differs by round/arm")
    timing=[]
    if not gaps:
        for name in sorted(keys):
            ref=statistics.median(r[name] for r in per_arm["ref"])
            cand=statistics.median(r[name] for r in per_arm["cand"])
            timing.append({"case":name,"ref_us":ref,"cand_us":cand,"ratio":cand/ref})
    return timing,gaps


def overlay_layout(texts):
    values={arm:[sorted(re.findall(r"^UNITLAYOUT plain_bf16 rc=(-?\d+)$",t,re.M)) for t in rounds]
            for arm,rounds in texts.items()}
    all_rounds=values["ref"]+values["cand"]
    first=all_rounds[0] if all_rounds else []
    complete=all(len(v)==len(parse_hashes(text)[0]) and
                 sum(line.startswith("UNITLAYOUT") for line in text.splitlines())==len(v)
                 for arm,rounds in texts.items() for v,text in zip(values[arm],rounds))
    gaps=[] if complete and first and "0" not in first and all(v==first for v in all_rounds) else ["overlay layout refusal coverage incomplete"]
    return {arm:sorted(set(rounds[0])) if rounds else [] for arm,rounds in values.items()},gaps


def stage_overlay_measure(out: Path, rounds: int | None = None, replay_only: bool = False, rows: list[int] | None = None,
                          bits: list[int] | None = None, shape: int | None = None, *, bdf: str) -> dict:
    if (out / ("measure-replay.json" if replay_only else "measure.json")).exists():
        raise HarnessError("measurement already exists; use a new build output")
    state = json.loads((out / "build.json").read_text(encoding="utf-8"))
    require_artifacts(state)
    ov = overlays()[state["overlay"]]
    rounds = ov.get("rounds", 4) if rounds is None else rounds
    if rounds <= 0:
        raise HarnessError("rounds must be positive")
    arch = state["arch"]
    device = (device_index(arch,bdf),bdf)
    from bigcherry.hardware.ai_probe import snapshot
    before,_=snapshot(bdf)
    binary, cand_plugin, ref_plugin = Path(state["selftest"]), Path(state["plugin"]), Path(state["reference_plugin"])
    env_base: dict = {"UH_TIME_ITERS": "31"} if "cases" in ov else {}
    if "cases" in ov:
        env_base[ov.get("cases-env", "UH_EMBED_CASES")] = ov["cases"]
    if "hash-family" in ov:
        env_base["UH_HASH_CASES"] = hash_cases(families()[ov["hash-family"]]["hash"])
    if "hash" in ov:                                      # the overlay's own extra shapes
        env_base["UH_HASH_CASES"] = hash_cases(ov["hash"])
    env_base.update(ov.get("env", {}))                    # selections both arms run under (UH_IQR_OWNED, UH_GATED_CASES)
    cand_env = {"UH_ROW_PREFIX": ov["row-prefix"], "UH_BASE_PLUGIN": state["base_stage"]}
    arms = {"ref": (ref_plugin, {}), "cand": (cand_plugin, cand_env)}
    texts: dict[str, list[str]] = {"ref": [], "cand": []}
    for r in range(0 if replay_only else rounds):
        for arm in (("ref", "cand") if r % 2 == 0 else ("cand", "ref")):
            plugin, extra = arms[arm]
            texts[arm].append(run_selftest(binary, plugin, ["--case", ov["case"]], selftest_env(arch, device, {**env_base, **extra}), timeout=3 * 3600))
    (out / "overlay-ref.log").write_text("\n".join(texts["ref"]), encoding="utf-8")
    (out / "overlay-cand.log").write_text("\n".join(texts["cand"]), encoding="utf-8")
    ref_hashes, ref_fail = consistent_round_hashes(texts["ref"])
    cand_hashes, cand_fail = consistent_round_hashes(texts["cand"])
    timing, timing_gaps = overlay_timing(texts, expect_timing="cases" in ov)
    layout, layout_gaps = overlay_layout(texts)
    result = {"arch": arch, "device":device[0],"bdf":bdf,"artifact_sha256":state["artifact_sha256"],
              "plain_bf16_layout_rc": layout, "hash": compare_hashes(ref_hashes, cand_hashes),
              "selftest_failures": {"ref": len(ref_fail), "cand": len(cand_fail), "cand_lines": cand_fail[:10]},
              "timing": timing, "coverage_gaps":timing_gaps+layout_gaps,"replay": []}
    h = result["hash"]
    print(f"overlay hash: {h['compared']} compared, {len(h['differ'])} differ, {len(h['missing_in_candidate'])} missing; selftest failures: "
          f"ref {len(ref_fail)}, overlay {len(cand_fail)}")
    print(f"overlay plain bf16 table: layout rc base {layout['ref']}, overlay {layout['cand']} (must agree and refuse)")
    for t in timing:
        print(f"overlay timing {t['case']}: {t['ref_us']} -> {t['cand_us']} us ({t['ratio']})")
    if "replay-family" in ov:
        spec = dict(families()[ov["replay-family"]]["replay"])
        spec["shapes"] = ov.get("replay-shapes", spec["shapes"])
        required=[{"row":"gemm_iqr_nt_m64" if point["M"]<=64 else row,**point}
                  for row in spec["rows"] for point in replay_points(spec,ov.get("replay-bits"),ov.get("replay-rows"))]
        result["required_replay"]=required
        if shape is not None:
            spec["shapes"] = [spec["shapes"][shape]]
        for row in spec["rows"]:
            for point in replay_points(spec, bits or ov.get("replay-bits"), rows or ov.get("replay-rows")):
                # --perfiqr times the 16/64-token row up to 64 rows and the prefill row above (its label differs)
                small = point["M"] <= 64
                got = replay_point(binary, ref_plugin, cand_plugin, arch, device, {**spec, "label": "iqr k{B} m64" if small else spec["label"]},
                                   point, "gemm_iqr_nt_m64" if small else row, iters=15, tol=0.015, max_windows=10,
                                   arm_env={"cand": cand_env})
                got["row"], got["verdict"] = ("gemm_iqr_nt_m64" if small else row), classify(got["ratio"], 0.04)
                result["replay"].append(got)
                print(f"overlay replay {got['row']} {point['mode']} N={point['N']} K={point['K']} M={point['M']} B={point['B']}: {got['ref_us']} -> {got['cand_us']} us "
                      f"({got['ratio']}) {got['verdict']}{'' if got['converged'] else ' UNCONVERGED'}", flush=True)
    if "replay-family" in ov:
        key=lambda p: tuple(p[k] for k in ("row","N","K","M","B","mode"))
        actual=result["replay"]
        if len({key(p) for p in actual})!=len(actual) or {key(p) for p in actual}!={key(p) for p in required}:
            result["coverage_gaps"].append("overlay replay coverage incomplete")
    result["observed_start"]=before
    result["observed_stop"]=snapshot(bdf)[0]
    require_artifacts(state)
    (out / ("measure-replay.json" if replay_only else "measure.json")).write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def add_build_arguments(sp) -> None:
    sp.add_argument("--family", help="a family of unit_families.toml")
    sp.add_argument("--unit", action="append", help="libr4d unit name(s), comma separated (resource records only)")
    sp.add_argument("--arch", required=True, choices=sorted(PLUGIN))
    sp.add_argument("--patch", action="append", help="a patch id added to the production composition")
    sp.add_argument("--reference", help="a plugin file to compare against instead of an installed release (e.g. an earlier harness build)")
    sp.add_argument("--libr3", help="the libr3 component directory to build (gfx1100; default: this checkout's)")
    sp.add_argument("--without", action="append", help="a production patch id left out of the composition")
    sp.add_argument("--tree", help="an already composed source tree instead of the production composition")
    sp.add_argument("--release", help="the installed release to compare against (default: the newest)")
    sp.add_argument("--out", help="output folder (default: under the work root)")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    plan_p = commands.add_parser("plan", help="resolve existing build groups into configured qualification routes (no GPU)")
    plan_p.add_argument("--tree", required=True, help="the actual composed Radiance source tree")
    plan_p.add_argument("--manifest", help="verify a full configured libr3 r3_groups.tsv against the source and exclusions")
    plan_p.add_argument("--arch", required=True, choices=sorted(PLUGIN))
    plan_p.add_argument("--release", required=True, help="explicit installed reference release")
    plan_p.add_argument("--overlay-reference", help="explicit compatible base plugin for excluded overlay rows; not a qualification proof")
    plan_p.add_argument("--group", action="append", help="select a build group; repeat to select several")
    plan_p.add_argument("--unit", action="append", help="select an affected upstream unit; repeat as needed")
    plan_p.add_argument("--touched",action="append",help="composed upstream changed path; unknown/header scope widens")
    plan_p.add_argument("--pin-triage",help="existing pin-triage --json report; imports changed_files")
    plan_p.add_argument("--deps-build",help="a libr3 build of the same composed source: scope changed paths by its compiler "
                                           "dependency records instead of the local include scan")
    plan_p.add_argument("--model-graph",help="saved Radiance /graph JSON (declared bands, not execution proof)")
    plan_p.add_argument("--row",action="append",help="exact observed/model-selected runtime row name; repeat, no inferred coverage")
    build_p = commands.add_parser("build", help="build the units and check resource records (no card)")
    add_build_arguments(build_p)
    measure_p = commands.add_parser("measure", help="hash and replay on a build (holds the card: run under the gpu lease)")
    measure_p.add_argument("out")
    measure_p.add_argument("--bdf",required=True)
    measure_p.add_argument("--iters", type=int, default=15)
    measure_p.add_argument("--tolerance", type=float, default=0.015)
    measure_p.add_argument("--max-windows", type=int, default=12)
    measure_p.add_argument("--no-replay", action="store_true")
    measure_p.add_argument("--bits", type=int, nargs="+", help="code widths to replay (default: the family's)")
    measure_p.add_argument("--rows", type=int, nargs="+", help="row counts to replay (default: the family's)")
    measure_p.add_argument("--no-hash", action="store_true", help="skip the selftest hash stage")
    ob = commands.add_parser("overlay-build", help="build an overlay plugin and compare its kernels' resource records with the base release's")
    add_build_arguments(ob)
    ob.add_argument("--overlay", required=True, help="an [overlay.*] of unit_families.toml")
    om = commands.add_parser("overlay-measure", help="hash and cold timing of the overlay row against the base row (holds the card: run under the gpu lease)")
    om.add_argument("out")
    om.add_argument("--bdf",required=True)
    om.add_argument("--replay-only", action="store_true", help="skip the selftest rounds (rerun replay points)")
    om.add_argument("--rows", type=int, nargs="+", help="row counts to replay (default: the overlay's)")
    om.add_argument("--bits", type=int, nargs="+", help="code widths to replay (default: the overlay's)")
    om.add_argument("--shape", type=int, help="index of the one replay shape to use")
    om.add_argument("--rounds", type=int, default=None, help="selftest rounds per arm (default: the overlay's, else 4)")
    run_p = commands.add_parser("run", help="build, resource check, then hash and replay under the gpu lease")
    add_build_arguments(run_p)
    run_p.add_argument("--bdf",required=True)
    run_p.add_argument("--ttl", default="40m", help="gpu lease ttl, one unit (default 40m)")
    run_p.add_argument("--est", default="20m")
    run_p.add_argument("--no-replay", action="store_true")
    args = parser.parse_args(argv)
    try:
        if args.command == "plan":
            return stage_plan(args)
        if args.command == "build":
            stage_build(args)
            return 0
        if args.command == "overlay-build":
            stage_overlay_build(args)
            return 0
        if args.command == "overlay-measure":
            result = stage_overlay_measure(Path(args.out), args.rounds, args.replay_only, args.rows, args.bits, args.shape,bdf=args.bdf)
            h = result["hash"]
            if args.replay_only:
                print("unit harness overlay replay: diagnostic (replay-only is not qualification)")
                return 1
            lay = result["plain_bf16_layout_rc"]
            regress = [r for r in result["replay"] if r["verdict"] == "regress"]
            for r in regress:
                print(f"  REGRESS {r['row']} N={r['N']} K={r['K']} M={r['M']} B={r['B']}: {r['ref_us']} -> {r['cand_us']} us ({r['ratio']})")
            partial = args.rows is not None or args.bits is not None or args.shape is not None or args.rounds is not None
            ok = (not partial and result["observed_start"]==result["observed_stop"] and not result["coverage_gaps"] and not h["differ"] and not h["missing_in_candidate"] and not h["only_in_candidate"]
                  and not any(result["selftest_failures"][arm] for arm in ("ref", "cand")) and h["compared"] > 0
                  and lay["ref"] == lay["cand"] and lay["cand"] and "0" not in lay["cand"] and not regress
                  and all(r["converged"] for r in result["replay"])
                  and all(classify(r["ratio"],0.04)!="regress" for r in result["timing"]))
            print("unit harness overlay: " + ("checks_passed (not profile qualification)" if ok else "nonpass"))
            return 0 if ok else 1
        if args.command == "measure":
            stage_measure(Path(args.out), bdf=args.bdf,replay_iters=args.iters, tol=args.tolerance, max_windows=args.max_windows,
                          skip_replay=args.no_replay, skip_hash=args.no_hash, bits=args.bits, rows=args.rows)
            text, ok = summary(Path(args.out))
            print(text)
            return 0 if ok else 1
        out = stage_build(args)
        command = [sys.executable, str(Path(__file__).resolve()), "measure", str(out),"--bdf",args.bdf] + (["--no-replay"] if args.no_replay else [])
        lease = ["agent-lease", "acquire", "gpu", "--ttl", args.ttl, "--est", args.est, "--wait",
                 "--purpose", f"unit harness {args.family or 'units'} {args.arch}", "--", *command]
        code = subprocess.run(lease, check=False).returncode
        if code != 0:
            return 1
        text, ok = summary(out)
        print(text)
        return 0 if ok and code == 0 else 1
    except (HarnessError, OSError, ValueError, KeyError) as error:
        print(f"unit harness: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
