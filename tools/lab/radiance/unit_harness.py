#!/usr/bin/env python3
"""Unit harness (RCS25): build one libr4d unit or family into a small test plugin and check it against the shipped library.

  unit_harness.py run --family iqr --arch gfx1201 [--patch ID ...] [--release bc-radiance-2020.11.4]

builds only the named units (a gfx1100 plugin through the libr3 component, a gfx1201 plugin through libr4d's own build
with the other units left out), in minutes instead of a lane build, then checks four things against the shipped release:

  resources  the compiled kernels' VGPR / SGPR / spill / private / LDS / workgroup records, kernel by kernel: added,
             removed, changed (no card);
  hash       the output bytes of the family's selftest case over its own matrix and a list of trunk shapes and row counts
             (tails and ragged N), the same selftest binary driving both plugins (card);
  replay     a cold replay (256 MiB flush) of only the rows whose kernels changed, the two plugins in ABBA order in
             separate processes, in windows until two medians agree (card);
  selftest   the failures of the case's own oracle checks on the new plugin (card).

The card stages hold the `gpu` lease: `run` takes it for them (`agent-lease acquire gpu --wait`), so the build and the
resource check never hold it. The selftest binary does not depend on the units, so it is built once per selftest source
and architecture and kept in the work root.

  unit_harness.py build ...     only build and the resource check
  unit_harness.py measure OUT   the card stages, on a build made by `build` (what `run` calls under the lease)
"""
from __future__ import annotations

import argparse
import hashlib
import json
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


def work_root() -> Path:
    root = os.environ.get("BC_WORK_ROOT")
    if not root:
        raise HarnessError("BC_WORK_ROOT is not set")
    return Path(root)


def rocm_tool(name: str) -> str:
    from bigcherry.patch import compose_check
    clang = compose_check.compiler()
    if clang is None:
        raise HarnessError("no ROCm compiler (set ROCM_PATH)")
    tool = clang.parent / name
    if not tool.exists():
        raise HarnessError(f"{tool} is missing")
    return str(tool)


def device_index(arch: str) -> str:
    """HIP's index of the first card of this architecture (rocminfo lists the GPU agents in HIP's order)."""
    out = subprocess.run(["rocminfo"], capture_output=True, text=True, check=False).stdout
    names = [m.group(1) for m in re.finditer(r"^\s+Name:\s+(gfx\w+)", out, re.M)]
    if arch not in names:
        raise HarnessError(f"no {arch} card in rocminfo ({names})")
    return str(names.index(arch))


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


# ------------------------------------------------------------------------------------------------------- composition

def compose(patches: list[str], tree: Path | None, scratch: Path, without: list[str] = ()) -> Path:
    """The source the plugin is built from: ``tree`` as given, or the production composition plus ``patches``."""
    if tree is not None:
        return tree
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


# A plugin that carries only some units has no entry point for the rest: the loader notes the absence and goes on
# (the rows without a kernel then return "unsupported", like libr3's own stubbed rows).
SYM_ANCHOR = '    if (e || !p) { std::printf("  FAIL  dlsym(%s): %s\\n", n, e ? e : "null"); ++g_fail; return false; }\n'
SYM_TOLERANT = ('    if ((e || !p) && std::getenv("UH_TOLERATE_ABSENT")) { *out = nullptr; return true; }\n')


def inject_selftest(text: str) -> str:
    if text.count(SYM_ANCHOR) != 1:
        raise HarnessError("the selftest changed: the dlsym check is not found exactly once")
    text = text.replace(SYM_ANCHOR, SYM_TOLERANT + SYM_ANCHOR, 1)
    for anchor in (HASH_ANCHOR, CASES_ANCHOR):
        if text.count(anchor) != 1:
            raise HarnessError("the selftest changed: the harness's edit anchor is not found exactly once "
                               f"({anchor.splitlines()[0].strip()!r})")
    text = text.replace(HASH_ANCHOR, HASH_ANCHOR.splitlines(keepends=True)[0] + HASH_PRINT + HASH_ANCHOR.splitlines(keepends=True)[1], 1)
    return text.replace(CASES_ANCHOR, CASES_ANCHOR + CASES_BLOCK, 1)


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
        options = [f"-DCMAKE_PREFIX_PATH={prefix}", f"-DRADIANCE_SRC={tree}", "-DR3_TARGETS=gfx1100", "-DR3_NATIVE=ON",
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
        cmake_build(tree / "libr4d", build, [f"-DCMAKE_PREFIX_PATH={prefix}", "-DRAD_GPU_TARGETS=gfx1201"], ["libr4d"], log)
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
    cases = [f"{n}:{k}:{m}" for n, k in spec["shapes"] for m in spec["rows"]]
    n, k = spec["tail_shape"]
    cases += [f"{n}:{k}:{m}" for m in spec["tail_rows"]]
    return ",".join(cases)


def selftest_env(arch: str, device: str, extra: dict | None = None) -> dict:
    env = dict(os.environ)
    env.update(HIP_VISIBLE_DEVICES=device, OPENBLAS_NUM_THREADS="8", UH_TOLERATE_ABSENT="1")
    env.update(extra or {})
    return env


def run_selftest(binary: Path, plugin: Path, args: list[str], env: dict, timeout: int) -> str:
    done = subprocess.run([str(binary), str(plugin), *args], env=env, capture_output=True, text=True, timeout=timeout, check=False)
    return done.stdout + done.stderr


def parse_hashes(text: str) -> tuple[dict[str, str], list[str]]:
    hashes = {m.group(1): m.group(2) for m in re.finditer(r"^UNITHASH (.+) ([0-9a-f]{16})$", text, re.M)}
    failures = [line for line in text.splitlines() if re.match(r"^\s*(FAIL|not ok)\b", line)]
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


def point_time(binary: Path, plugin: Path, arch: str, device: str, spec: dict, point: dict, row: str, iters: int) -> float:
    args = [a.format(**point, iters=iters) for a in spec["command"]]
    label = spec["label"].format(**point)
    text = run_selftest(binary, plugin, args, selftest_env(arch, device, {"R4D_PERF_ONLY": row}), timeout=900)
    values = [float(m.group(1)) for line in text.splitlines() if label in line for m in re.finditer(r"(\d+(?:\.\d+)?)\s+us", line)]
    if not values:
        raise HarnessError(f"no time for {label} {point}: {text[-300:]}")
    return values[-1]


def replay_point(binary: Path, ref: Path, cand: Path, arch: str, device: str, spec: dict, point: dict, row: str,
                 *, iters: int, tol: float, max_windows: int) -> dict:
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
            windows[arm].append(point_time(binary, plugins[arm], arch, device, spec, point, row, iters))
        rounds += 1
        done = all(converged(w, tol) for w in windows.values())
    ref_us, cand_us = estimate(windows["ref"], drop_first), estimate(windows["cand"], drop_first)
    return {**point, "ref_us": round(ref_us, 2), "cand_us": round(cand_us, 2), "ratio": round(cand_us / ref_us, 4),
            "windows": rounds, "converged": done}


def replay_points(spec: dict) -> list[dict]:
    return [{"N": n, "K": k, "M": m, "B": b} for n, k in spec["shapes"] for m in spec["rows_m"] for b in spec["bits"]]


def classify(ratio: float, tol: float) -> str:
    return "win" if ratio < 1 - tol else "regress" if ratio > 1 + tol else "same"


# ------------------------------------------------------------------------------------------------------------ stages

def stage_build(args) -> Path:
    fam = families().get(args.family) if args.family else None
    units = list(fam["units"]) if fam else [u for chunk in (args.unit or []) for u in chunk.split(",")]
    if not units:
        raise HarnessError("name a --family or --unit")
    out = Path(args.out) if args.out else work_root() / "unit-harness" / f"{args.family or 'units'}-{args.arch}-{time.strftime('%Y%m%d-%H%M%S', time.gmtime())}"
    out.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    tree = compose(args.patch or [], Path(args.tree) if args.tree else None, out, args.without or [])
    tree_key = out / "tree"
    tree_key.write_text(str(tree), encoding="utf-8")
    ref_plugin = Path(args.reference) if args.reference else release_plugin(args.release, args.arch)
    plugin = build_plugin(args.arch, tree, units + [u for u in (fam or {}).get("support_units", []) if u not in units], out, ref_plugin,
                         Path(args.libr3) if args.libr3 else None)
    built = time.monotonic() - started
    binary = selftest_binary(args.arch, tree, out, Path(args.libr3) if args.libr3 else None)
    ref = kernel_records(ref_plugin, args.arch)
    cand = kernel_records(plugin, args.arch)
    pattern = re.compile(fam["kernels"]) if fam else None
    diff = diff_records(ref, cand, pattern)
    state = {"family": args.family, "units": units, "arch": args.arch, "plugin": str(plugin), "selftest": str(binary),
             "release": ref_plugin.parts[-4] if not args.reference else "reference plugin " + ref_plugin.parts[-5], "reference_plugin": str(ref_plugin), "resources": diff,
             "build_seconds": round(built, 1)}
    (out / "build.json").write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")
    print(f"built {', '.join(units)} for {args.arch} in {built:.0f} s (selftest {'cached' if built < 5 else 'ready'}): {plugin}")
    print(resource_summary(diff, cand))
    return out


def stage_measure(out: Path, *, replay_iters: int = 15, tol: float = 0.015, max_windows: int = 14, verdict_tol: float = 0.04, skip_replay: bool = False,
                  skip_hash: bool = False) -> dict:
    state = json.loads((out / "build.json").read_text(encoding="utf-8"))
    fam = families().get(state["family"]) if state["family"] else None
    if not fam:
        raise HarnessError("measure needs a family (hash and replay matrices come from unit_families.toml)")
    arch = state["arch"]
    device = device_index(arch)
    binary, plugin, ref = Path(state["selftest"]), Path(state["plugin"]), Path(state["reference_plugin"])
    result: dict = {"arch": arch, "device": device}
    env = {"UH_HASH_CASES": hash_cases(fam["hash"])}
    texts = {}
    for arm, path in ((("ref", ref), ("cand", plugin)) if not skip_hash else ()):
        started = time.monotonic()
        texts[arm] = run_selftest(binary, path, ["--case", fam["case"]], selftest_env(arch, device, env), timeout=3 * 3600)
        (out / f"selftest-{arm}.log").write_text(texts[arm], encoding="utf-8")
        print(f"selftest {arm}: {time.monotonic() - started:.0f} s")
    ref_hashes, ref_fail = parse_hashes(texts.get("ref", ""))
    cand_hashes, cand_fail = parse_hashes(texts.get("cand", ""))
    result["hash"] = compare_hashes(ref_hashes, cand_hashes)
    result["selftest_failures"] = {"ref": len(ref_fail), "cand": len(cand_fail), "cand_lines": cand_fail[:10]}
    print(f"hash: {result['hash']['compared']} compared, {len(result['hash']['differ'])} differ, "
          f"{len(result['hash']['missing_in_candidate'])} missing in the candidate; selftest failures: ref {len(ref_fail)}, candidate {len(cand_fail)}")
    changed_rows = [row for row in fam["replay"]["rows"]] if (state["resources"]["added"] or state["resources"]["changed"]
                                                              or state["resources"]["removed"]) else []
    result["replay"] = []
    if changed_rows and not skip_replay:
        spec = fam["replay"]
        for row in changed_rows:
            for point in replay_points(spec):
                got = replay_point(binary, ref, plugin, arch, device, spec, point, row, iters=replay_iters, tol=tol,
                                   max_windows=max_windows)
                got["row"], got["verdict"] = row, classify(got["ratio"], verdict_tol)
                result["replay"].append(got)
                print(f"replay {row} N={point['N']} K={point['K']} M={point['M']} B={point['B']}: "
                      f"{got['ref_us']} -> {got['cand_us']} us ({got['ratio']}) {got['verdict']}"
                      f"{'' if got['converged'] else ' UNCONVERGED'}", flush=True)
    elif not changed_rows:
        print("replay: no row changed, nothing replayed")
    (out / "measure.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def summary(out: Path) -> tuple[str, bool]:
    build = json.loads((out / "build.json").read_text(encoding="utf-8"))
    measure_path = out / "measure.json"
    ok = not build["resources"]["removed"]
    lines = [f"unit harness: {build['family'] or ','.join(build['units'])} on {build['arch']} against {build['release']}"]
    res = build["resources"]
    lines.append(f"resources: {len(res['added'])} added, {len(res['changed'])} changed, {len(res['removed'])} removed, {res['unchanged']} unchanged")
    if res["removed"] or res["changed"]:
        lines.append("  REGRESSION CHECK: changed or removed kernels need a reason (see build.json)")
    if measure_path.exists():
        measure = json.loads(measure_path.read_text(encoding="utf-8"))
        h = measure["hash"]
        bad_hash = bool(h["differ"] or h["missing_in_candidate"])
        ok = ok and not bad_hash and not measure["selftest_failures"]["cand"]
        lines.append(f"hash: {h['compared']} compared, {len(h['differ'])} differ"
                     + (f"; first {h['differ'][0]}" if h["differ"] else ""))
        lines.append(f"selftest failures (candidate): {measure['selftest_failures']['cand']}")
        rep = measure["replay"]
        if rep:
            regress = [r for r in rep if r["verdict"] == "regress"]
            wins = [r for r in rep if r["verdict"] == "win"]
            lines.append(f"replay: {len(rep)} points, {len(wins)} win, {len(regress)} regress, "
                         f"{sum(1 for r in rep if r['verdict'] == 'same')} same, {sum(1 for r in rep if not r['converged'])} unconverged")
            for r in regress:
                lines.append(f"  REGRESS {r['row']} N={r['N']} K={r['K']} M={r['M']} B={r['B']}: {r['ref_us']} -> {r['cand_us']} us ({r['ratio']})")
            ok = ok and not regress
    verdict = "PASS" if ok else "FAIL"
    lines.append(f"unit harness: {verdict}")
    return "\n".join(lines), ok


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
    build_p = commands.add_parser("build", help="build the units and check resource records (no card)")
    add_build_arguments(build_p)
    measure_p = commands.add_parser("measure", help="hash and replay on a build (holds the card: run under the gpu lease)")
    measure_p.add_argument("out")
    measure_p.add_argument("--iters", type=int, default=15)
    measure_p.add_argument("--tolerance", type=float, default=0.015)
    measure_p.add_argument("--max-windows", type=int, default=12)
    measure_p.add_argument("--no-replay", action="store_true")
    measure_p.add_argument("--no-hash", action="store_true", help="skip the selftest hash stage")
    run_p = commands.add_parser("run", help="build, resource check, then hash and replay under the gpu lease")
    add_build_arguments(run_p)
    run_p.add_argument("--ttl", default="40m", help="gpu lease ttl, one unit (default 40m)")
    run_p.add_argument("--est", default="20m")
    run_p.add_argument("--no-replay", action="store_true")
    args = parser.parse_args(argv)
    try:
        if args.command == "build":
            stage_build(args)
            return 0
        if args.command == "measure":
            stage_measure(Path(args.out), replay_iters=args.iters, tol=args.tolerance, max_windows=args.max_windows,
                          skip_replay=args.no_replay, skip_hash=args.no_hash)
            text, ok = summary(Path(args.out))
            print(text)
            return 0 if ok else 1
        out = stage_build(args)
        command = [sys.executable, str(Path(__file__).resolve()), "measure", str(out)] + (["--no-replay"] if args.no_replay else [])
        lease = ["agent-lease", "acquire", "gpu", "--ttl", args.ttl, "--est", args.est, "--wait",
                 "--purpose", f"unit harness {args.family or 'units'} {args.arch}", "--", *command]
        code = subprocess.run(lease, check=False).returncode
        text, ok = summary(out)
        print(text)
        return 0 if ok and code == 0 else 1
    except HarnessError as error:
        print(f"unit harness: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
