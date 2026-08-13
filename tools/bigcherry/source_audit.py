"""Strict-mode audit of an upstream llama.cpp checkout (HI01).

bigcherry's patches assume specific shapes in upstream code: that MMQ's tile
width J is chosen from a switch over ``range(8, 129, 8)``, that MMVF picks a
block size from ``range(32, 257, 32)``, that MMVQ's geometry comes from
``calc_nwarps``/``calc_rows_per_block``, and so on. Those assumptions are what
makes forced-variant dispatch possible at all.

When upstream changes one of them, the failure we want is a loud one *here*,
naming the invariant, rather than a subtly wrong candidate catalog discovered
three phases later. So every release runs this audit before anything is
patched, and the resulting JSON is archived alongside the build.

Invariants are those listed in HIP_AUTOTUNE_STANDARDS section 13.2.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import re
import subprocess
import sys
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Callable

from . import ARTIFACT_VERSION
from . import csource
from . import paths

# --------------------------------------------------------------------- expected

# Standards 13.2. These are written as explicit ranges rather than magic lists
# so a diff against upstream reads as a statement about upstream's design.
MMQ_J_VALUES = list(range(8, 129, 8))
MMVF_BLOCK_SIZES = list(range(32, 257, 32))
MMF_WIDTHS = list(range(1, 17))
MMF_NWARPS = list(range(1, 9))
MMVQ_MAX_BATCH_SIZE = 8

# Architecture-specific MMQ config headers (standards 13.2).
#
# All five are required, because bigcherry targets every AMD part and the
# candidate catalog is derived from these tables. A missing header is not a
# degraded build -- it is an architecture whose MMQ candidates cannot be
# enumerated at all.
#
# The keys mirror upstream's fallthrough in ggml_cuda_mmq_get_config: CDNA,
# then RDNA4, then RDNA3.5, then RDNA3, with everything older -- RDNA2, RDNA1,
# GCN and Vega -- landing on the RDNA2 table.
MMQ_CONFIG_REQUIRED = {
    "mmq-config-cdna.cuh":    "CDNA (gfx908/90a/942/950)",
    "mmq-config-rdna4.cuh":   "RDNA4 (gfx1200/1201)",
    "mmq-config-rdna3-5.cuh": "RDNA3.5 (gfx115x)",
    "mmq-config-rdna3.cuh":   "RDNA3 (gfx110x)",
    "mmq-config-rdna2.cuh":   "RDNA2 and older (gfx803..gfx103x)",
}
MMQ_CONFIG_OPTIONAL: dict[str, str] = {}

# The two structs below are separate identity namespaces.  Keep this check
# source-level and deliberately narrow: it protects the current ABI boundary
# without trying to infer future fields from their spelling or changing the
# ABI as part of the audit.
AUTOTUNE_TYPES_HEADER = "hip-autotune-types.h"
SIGNATURE_STRUCT = "ggml_hip_dispatch_signature_v1"
CANDIDATE_STRUCT = "ggml_hip_candidate_descriptor"

# Entry points the dispatch refactor (HI04) rewires. If one disappears or is
# renamed, the anchored patches will fail -- better to say so up front.
REQUIRED_ENTRY_POINTS = {
    "ggml-cuda.cu": ["ggml_cuda_mul_mat", "ggml_cuda_mul_mat_id"],
    "mmq.cu": ["ggml_cuda_mul_mat_q", "ggml_cuda_should_use_mmq"],
    "mmvf.cu": ["ggml_cuda_mul_mat_vec_f", "ggml_cuda_should_use_mmvf"],
    "mmf.cu": ["ggml_cuda_mul_mat_f", "ggml_cuda_should_use_mmf"],
    "mmvq.cu": ["ggml_cuda_mul_mat_vec_q", "ggml_cuda_should_use_mmvq"],
}

SEVERITY_ERROR = "error"
SEVERITY_WARNING = "warning"


@dataclass
class CheckResult:
    id: str
    ok: bool
    detail: str
    severity: str = SEVERITY_ERROR
    expected: Any = None
    actual: Any = None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class AuditContext:
    root: Path
    cuda: Path
    instances: Path
    results: list[CheckResult] = field(default_factory=list)

    def add(self, result: CheckResult) -> CheckResult:
        self.results.append(result)
        return result

    def ok(self, check_id: str, detail: str, **kw: Any) -> CheckResult:
        return self.add(CheckResult(check_id, True, detail, **kw))

    def fail(self, check_id: str, detail: str, **kw: Any) -> CheckResult:
        return self.add(CheckResult(check_id, False, detail, **kw))

    def compare(
        self,
        check_id: str,
        expected: list[Any],
        actual: list[Any],
        subject: str,
        *,
        ordered: bool = True,
        severity: str = SEVERITY_ERROR,
    ) -> CheckResult:
        matched = actual == expected if ordered else sorted(actual) == sorted(expected)
        if matched:
            return self.ok(check_id, f"{subject}: {len(actual)} values as expected",
                           expected=expected, actual=actual)
        missing = [v for v in expected if v not in actual]
        extra = [v for v in actual if v not in expected]
        parts = [f"{subject} does not match"]
        if missing:
            parts.append(f"missing={missing}")
        if extra:
            parts.append(f"unexpected={extra}")
        if not missing and not extra:
            parts.append("same values in a different order")
        return self.fail(check_id, "; ".join(parts), severity=severity,
                         expected=expected, actual=actual)

    def read(self, relative: str) -> str | None:
        path = self.cuda / relative
        if not path.is_file():
            self.fail(f"file.{relative}", f"missing required source file {relative}")
            return None
        return csource.read(path)


# ------------------------------------------------------------------------ MMQ

def check_mmq_types(ctx: AuditContext) -> None:
    """Generated MMQ types must equal the runtime switch cases, as a set.

    A type present in one but not the other means either a compiled kernel no
    runtime path can reach, or a runtime case with no kernel behind it.
    """
    generator = paths.generate_cu_files_py(ctx.root)
    generated = read_types_mmq(generator)
    if generated is None:
        ctx.fail("mmq.generator_types_readable",
                 f"could not read TYPES_MMQ from {generator}")
        return
    ctx.ok("mmq.generator_types_readable",
           f"TYPES_MMQ has {len(generated)} entries", actual=generated)

    source = ctx.read("mmq.cu")
    if source is None:
        return
    body = csource.function_body(source, "ggml_cuda_mul_mat_q_switch_type")
    if body is None:
        ctx.fail("mmq.runtime_switch_present",
                 "ggml_cuda_mul_mat_q_switch_type not found in mmq.cu")
        return
    runtime = re.findall(r"mul_mat_q_case<\s*(GGML_TYPE_\w+)\s*>", body)
    ctx.compare("mmq.generated_types_match_runtime", generated, runtime,
                "MMQ generated types vs runtime switch cases", ordered=False)

    # Every generated type also needs its translation unit on disk, or the
    # kernel is simply not compiled regardless of what the lists say.
    present = {
        p.name[len("mmq-instance-"):-len(".cu")]
        for p in ctx.instances.glob("mmq-instance-*.cu")
    }
    expected = {t.removeprefix("GGML_TYPE_").lower() for t in generated}
    ctx.compare("mmq.instance_files_match_generator",
                sorted(expected), sorted(present),
                "MMQ instance translation units", ordered=False)


def check_mmq_j(ctx: AuditContext) -> None:
    """The J switch and the J search loop must cover the same values.

    The audit is specified to run on a pristine checkout (standards 13.3), but
    people re-run it, and on an already-patched tree the switch has moved into
    ``mul_mat_q_launch_forced_J`` and the scan into ``mul_mat_q_compute_J_best``.
    Reporting a failure then would be a false alarm about our own patch. So each
    construct is looked for in its post-patch home first, and the check reports
    which state it found.
    """
    source = ctx.read("mmq.cuh")
    if source is None:
        return

    switch_body = csource.function_body(source, "mul_mat_q_launch_forced_J")
    patched = switch_body is not None
    if switch_body is None:
        switch_body = csource.function_body(source, "mul_mat_q_switch_J")
    if switch_body is None:
        ctx.fail("mmq.j_switch_present",
                 "neither mul_mat_q_switch_J nor mul_mat_q_launch_forced_J "
                 "found in mmq.cuh -- the forced-J patch (HI06) has nothing "
                 "to hook")
        return

    where = "patched" if patched else "pristine"
    values = csource.int_captures(
        switch_body, r"launch_mul_mat_q<\s*type\s*,\s*(\d+)\s*,\s*fallback\s*>")
    ctx.compare("mmq.j_switch_values", MMQ_J_VALUES, values,
                f"MMQ J switch cases ({where} tree)")

    # The switch is the dispatchable set; the search loop is what native
    # selection walks. They have to agree, or forced-J covers the wrong range.
    scan_body = csource.function_body(source, "mul_mat_q_compute_J_best") \
        or csource.function_body(source, "mul_mat_q_switch_J")
    if scan_body and "ggml_cuda_mmq_native_j_best" in scan_body:
        mmq_cu = ctx.read("mmq.cu") or ""
        scan_body = csource.function_body(mmq_cu, "ggml_cuda_mmq_native_j_best")
    if scan_body is None:
        ctx.fail("mmq.j_search_loop_bounds", "no J search loop found in mmq.cuh")
        return

    loop = re.search(
        r"for\s*\(\s*int\s+J\s*=\s*(\d+)\s*;\s*J\s*<=\s*(\d+)[^;]*;\s*J\s*\+=\s*(\d+)\s*\)",
        csource.strip_noise(scan_body))
    if not loop:
        # b10362 expresses the same bounded search with a runtime `ret`
        # cursor (`ret = min(ne11, 512); ret -= ret % 8; for (; ret > 0;
        # ret -= 8)`) rather than a literal J loop.  Accept that pinned
        # upstream form; requiring the older textual spelling made a clean
        # current checkout fail its own strict pre-flight audit.
        loop = re.search(
            r"ret\s*=\s*std::min\([^;]+,\s*int64_t\((\d+)\)\).*?"
            r"ret\s*-=\s*ret\s*%\s*(\d+).*?"
            r"for\s*\(\s*;\s*ret\s*>\s*0\s*;\s*ret\s*-=?\s*(\d+)\s*\)",
            csource.strip_noise(scan_body), re.S)
        if loop:
            # The runtime ceiling is an upstream search bound; the dispatch
            # switch remains the authoritative supported set. Compare the
            # common 8-step range up to the switch's maximum below.
            start, stop, step = 8, max(MMQ_J_VALUES), int(loop.group(2))
            found = list(range(start, stop + 1, step))
            ctx.compare("mmq.j_search_loop_bounds", MMQ_J_VALUES, found,
                        f"MMQ J search loop (runtime bound, step {step})")
            return
    if not loop:
        ctx.fail("mmq.j_search_loop_bounds",
                 "could not find the J search loop in mmq.cuh")
        return
    start, stop, step = (int(loop.group(i)) for i in (1, 2, 3))
    found = list(range(start, stop + 1, step))
    ctx.compare("mmq.j_search_loop_bounds", MMQ_J_VALUES, found,
                f"MMQ J search loop (J={start}..{stop} step {step})")


def check_mmq_configs(ctx: AuditContext) -> None:
    for filename, arch in MMQ_CONFIG_REQUIRED.items():
        if (ctx.cuda / filename).is_file():
            ctx.ok(f"mmq.config.{arch}", f"{filename} present for {arch}")
        else:
            ctx.fail(f"mmq.config.{arch}",
                     f"{filename} missing -- MMQ configs for {arch} cannot be read")
    for filename, arch in MMQ_CONFIG_OPTIONAL.items():
        present = (ctx.cuda / filename).is_file()
        ctx.add(CheckResult(
            f"mmq.config.{arch}", present,
            f"{filename} {'present' if present else 'missing'} for {arch}",
            severity=SEVERITY_WARNING))


# ----------------------------------------------------------------------- MMVF

def check_mmvf(ctx: AuditContext) -> None:
    source = ctx.read("mmvf.cu")
    if source is None:
        return

    body = csource.function_body(source, "launch_mul_mat_vec_f_cuda")
    if body is None:
        ctx.fail("mmvf.block_size_switch_present",
                 "launch_mul_mat_vec_f_cuda not found in mmvf.cu")
    else:
        values = csource.int_captures(
            body,
            r"mul_mat_vec_f_switch_fusion<\s*T\s*,\s*type_acc\s*,\s*ncols_dst\s*,\s*(\d+)\s*,")
        ctx.compare("mmvf.block_size_switch_values", MMVF_BLOCK_SIZES, values,
                    "MMVF block-size switch cases")

    ncols_body = csource.function_body(source, "mul_mat_vec_f_cuda_switch_ncols_dst")
    if ncols_body is None:
        ctx.fail("mmvf.ncols_dst_switch_present",
                 "mul_mat_vec_f_cuda_switch_ncols_dst not found in mmvf.cu")
    else:
        widths = csource.int_captures(
            ncols_body,
            r"launch_mul_mat_vec_f_cuda<\s*T\s*,\s*type_acc\s*,\s*(\d+)\s*>")
        ctx.compare("mmvf.ncols_dst_switch_values", list(range(1, 9)), widths,
                    "MMVF ncols_dst switch cases")

    # Standards 3.2: F16 source has two accumulator modes and they are genuine
    # performance variants; everything else is F32-only. If upstream collapses
    # this, the MMVF candidate set must shrink to match.
    dispatch = csource.function_body(source, "mul_mat_vec_f_cuda")
    if dispatch is None:
        ctx.fail("mmvf.accumulator_modes",
                 "mul_mat_vec_f_cuda not found in mmvf.cu")
        return
    has_half = re.search(
        r"mul_mat_vec_f_cuda_switch_ncols_dst<\s*T\s*,\s*half\s*>", dispatch)
    has_float = re.search(
        r"mul_mat_vec_f_cuda_switch_ncols_dst<\s*T\s*,\s*float\s*>", dispatch)
    if has_half and has_float:
        ctx.ok("mmvf.accumulator_modes",
               "both F16 and F32 accumulator dispatches present",
               actual=["half", "float"])
    else:
        found = [n for n, m in (("half", has_half), ("float", has_float)) if m]
        ctx.fail("mmvf.accumulator_modes",
                 "expected both half and F32 accumulator dispatches in "
                 "mul_mat_vec_f_cuda",
                 expected=["half", "float"], actual=found)


# ------------------------------------------------------------------------ MMF

def check_mmf(ctx: AuditContext) -> None:
    source = ctx.read("mmf.cuh")
    if source is None:
        return

    body = csource.function_body(source, "mul_mat_f_cuda")
    if body is None:
        ctx.fail("mmf.nwarps_switch_present", "mul_mat_f_cuda not found in mmf.cuh")
    else:
        values = csource.int_captures(
            body,
            r"mul_mat_f_switch_ids<\s*T\s*,\s*rows_per_block\s*,\s*cols_per_block\s*,\s*(\d+)\s*>")
        ctx.compare("mmf.nwarps_switch_values", MMF_NWARPS, values,
                    "MMF nwarps switch cases")

    cols_body = csource.function_body(source, "mul_mat_f_switch_cols_per_block")
    if cols_body is None:
        ctx.fail("mmf.cols_per_block_switch_present",
                 "mul_mat_f_switch_cols_per_block not found in mmf.cuh")
    else:
        widths = csource.int_captures(
            cols_body, r"mul_mat_f_cuda<\s*T\s*,\s*rows_per_block\s*,\s*(\d+)\s*>")
        ctx.compare("mmf.cols_per_block_switch_values", MMF_WIDTHS, widths,
                    "MMF cols_per_block switch cases")

    present = sorted(
        int(p.name[len("mmf-instance-ncols_"):-len(".cu")])
        for p in ctx.instances.glob("mmf-instance-ncols_*.cu")
    )
    ctx.compare("mmf.generated_widths", MMF_WIDTHS, present,
                "MMF generated width translation units", ordered=False)


# ----------------------------------------------------------------------- MMVQ

def check_mmvq(ctx: AuditContext) -> None:
    source = ctx.read("mmvq.cu")
    if source is None:
        return
    stripped = csource.strip_noise(source)

    # HI09 replaces these constexpr policies with explicit template parameters
    # and re-expresses the current policy as one explicit candidate. Both must
    # exist and both must be constexpr, or there is no policy to re-express.
    for name in ("calc_nwarps", "calc_rows_per_block"):
        match = re.search(
            r"static\s+constexpr\s+__host__\s+__device__\s+int\s+" + name + r"\s*\(",
            stripped)
        if match:
            ctx.ok(f"mmvq.{name}_constexpr",
                   f"{name} is a constexpr __host__ __device__ policy")
        else:
            loose = re.search(r"\b" + name + r"\s*\(", stripped)
            ctx.fail(
                f"mmvq.{name}_constexpr",
                f"{name} is not a constexpr __host__ __device__ int policy"
                + ("" if loose else f" -- no symbol named {name} found at all"))

    header = ctx.read("mmvq.cuh")
    if header is None:
        return
    match = re.search(r"#define\s+MMVQ_MAX_BATCH_SIZE\s+(\d+)",
                      csource.strip_noise(header))
    if not match:
        ctx.fail("mmvq.max_batch_size",
                 "MMVQ_MAX_BATCH_SIZE not defined in mmvq.cuh")
        return
    actual = int(match.group(1))
    if actual == MMVQ_MAX_BATCH_SIZE:
        ctx.ok("mmvq.max_batch_size", f"MMVQ_MAX_BATCH_SIZE == {actual}",
               expected=MMVQ_MAX_BATCH_SIZE, actual=actual)
    else:
        ctx.fail("mmvq.max_batch_size",
                 f"MMVQ_MAX_BATCH_SIZE is {actual}; generated width bounds and "
                 f"the static_asserts in the explicit kernel assume "
                 f"{MMVQ_MAX_BATCH_SIZE}",
                 expected=MMVQ_MAX_BATCH_SIZE, actual=actual)


# ---------------------------------------------------------- identity namespaces

def _struct_fields(source: str, name: str) -> list[str] | None:
    """Return direct field names from one C/C++ struct declaration.

    This is intentionally not a general C++ parser.  The declarations being
    audited are flat ABI records; callbacks, arrays, pointers, and typedefs
    all end in a field identifier, while nested records remain represented by
    their containing field (for example ``variant``).  Comments and literals
    are removed before locating the brace-balanced declaration so a stale
    example cannot satisfy the audit.
    """
    stripped = csource.strip_noise(source)
    declaration = re.search(
        r"\bstruct\s+" + re.escape(name) + r"\s*\{", stripped)
    if declaration is None:
        return None
    block = csource.find_braced_block(stripped, declaration.end() - 1)
    if block is None:
        return None
    start, end = block
    body = stripped[start + 1:end - 1]

    fields: list[str] = []
    statement_start = 0
    depth = 0
    for index, char in enumerate(body):
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
        elif char == ";" and depth == 0:
            statement = body[statement_start:index]
            statement_start = index + 1
            # A declaration's final identifier is its field name.  Array
            # extents are numeric and therefore do not become a match.
            names = re.findall(r"\b[A-Za-z_]\w*\b", statement)
            if names:
                fields.append(names[-1])
    return fields


def check_identity_namespace_separation(ctx: AuditContext) -> None:
    """Ensure signature and candidate ABI fields stay in separate records."""
    source = ctx.read(AUTOTUNE_TYPES_HEADER)
    if source is None:
        return
    signature = _struct_fields(source, SIGNATURE_STRUCT)
    candidate = _struct_fields(source, CANDIDATE_STRUCT)
    if signature is None:
        ctx.fail("identity.signature_struct_present",
                 f"{SIGNATURE_STRUCT} declaration not found in "
                 f"{AUTOTUNE_TYPES_HEADER}")
        return
    if candidate is None:
        ctx.fail("identity.candidate_struct_present",
                 f"{CANDIDATE_STRUCT} declaration not found in "
                 f"{AUTOTUNE_TYPES_HEADER}")
        return

    overlap = sorted(set(signature) & set(candidate))
    if overlap:
        ctx.fail(
            "identity.signature_candidate_separation",
            "signature and candidate declarations share field(s): "
            + ", ".join(overlap),
            expected={"signature_only": sorted(set(signature)),
                      "candidate_only": sorted(set(candidate))},
            actual={"overlap": overlap},
        )
    else:
        ctx.ok(
            "identity.signature_candidate_separation",
            "signature and candidate declarations have disjoint direct fields",
            expected={"signature_only": sorted(set(signature)),
                      "candidate_only": sorted(set(candidate))},
            actual={"overlap": []},
        )


# ----------------------------------------------------------------- build files

def check_build(ctx: AuditContext) -> None:
    cmake_path = ctx.root / "ggml" / "src" / "ggml-hip" / "CMakeLists.txt"
    if not cmake_path.is_file():
        ctx.fail("build.hip_cmake_present",
                 f"missing {cmake_path.relative_to(ctx.root)}")
        return
    text = csource.read(cmake_path)

    # The overlay drops new .cu files into ggml-cuda/ and template-instances/
    # and relies on these globs to compile them without touching the file list.
    required_globs = [
        "../ggml-cuda/*.cu",
        "../ggml-cuda/*.cuh",
        "../ggml-cuda/template-instances/mmq*.cu",
        "../ggml-cuda/template-instances/mmf*.cu",
    ]
    missing = [g for g in required_globs if g not in text]
    if missing:
        ctx.fail("build.hip_globs",
                 "ggml-hip/CMakeLists.txt no longer globs: " + ", ".join(missing),
                 expected=required_globs)
    else:
        ctx.ok("build.hip_globs", "HIP build globs ggml-cuda sources and "
               "mmq/mmf template instances", expected=required_globs)

    for package, symbol in (("hipblas", "roc::hipblas"), ("rocblas", "roc::rocblas")):
        found = f"find_package({package}" in text and symbol in text
        if found:
            ctx.ok(f"build.links_{package}", f"{package} found and linked")
        else:
            ctx.fail(f"build.links_{package}",
                     f"{package} is not both found and linked as {symbol} -- "
                     "the BLAS candidate family has no backing library")


def check_entry_points(ctx: AuditContext) -> None:
    for filename, names in REQUIRED_ENTRY_POINTS.items():
        source = ctx.read(filename)
        if source is None:
            continue
        for name in names:
            if csource.function_body(source, name) is not None:
                ctx.ok(f"entry.{name}", f"{name} defined in {filename}")
            else:
                ctx.fail(f"entry.{name}",
                         f"{name} not defined in {filename} -- anchored patches "
                         "targeting it will not apply")


# --------------------------------------------------------------------- helpers

def read_types_mmq(generator: Path) -> list[str] | None:
    """Read ``TYPES_MMQ`` out of generate_cu_files.py without importing it.

    Standards 13.2 requires AST reading rather than import: the generator
    deletes ``*.cu`` in its working directory at module scope, so importing it
    to inspect a list would wipe the template instances.
    """
    import ast

    if not generator.is_file():
        return None
    try:
        tree = ast.parse(generator.read_text(encoding="utf-8"), str(generator))
    except SyntaxError:
        return None
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        targets = [t.id for t in node.targets if isinstance(t, ast.Name)]
        if "TYPES_MMQ" not in targets:
            continue
        try:
            value = ast.literal_eval(node.value)
        except ValueError:
            return None
        if isinstance(value, list) and all(isinstance(v, str) for v in value):
            return value
    return None


def _git(root: Path, *args: str) -> str | None:
    try:
        out = subprocess.run(("git", "-C", str(root)) + args,
                             capture_output=True, text=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None
    return out.stdout.strip()


def git_revision(root: Path, *, check_dirty: bool = True) -> tuple[str, bool]:
    """Return (revision, dirty). Revision is ``unknown`` outside a git tree.

    The dirty check stats the whole tree, which on a network share costs
    minutes rather than milliseconds. It is worth paying during an audit --
    a dirty tree makes the build unreproducible and that must be recorded --
    but not on every incidental call, hence ``check_dirty``.
    """
    revision = _git(root, "rev-parse", "HEAD") or "unknown"
    if not check_dirty:
        return revision, False
    return revision, bool(_git(root, "status", "--porcelain"))


ALL_CHECKS: tuple[Callable[[AuditContext], None], ...] = (
    check_mmq_types,
    check_mmq_j,
    check_mmq_configs,
    check_mmvf,
    check_mmf,
    check_mmvq,
    check_identity_namespace_separation,
    check_build,
    check_entry_points,
)


def audit(root: Path) -> dict[str, Any]:
    ctx = AuditContext(
        root=root,
        cuda=paths.cuda_dir(root),
        instances=paths.template_instances_dir(root),
    )
    if not ctx.cuda.is_dir():
        ctx.fail("root.is_llama_checkout",
                 f"{root} does not look like a llama.cpp checkout "
                 f"(no ggml/src/ggml-cuda)")
    else:
        for check in ALL_CHECKS:
            check(ctx)

    revision, dirty = git_revision(root)
    errors = [r for r in ctx.results if not r.ok and r.severity == SEVERITY_ERROR]
    warnings = [r for r in ctx.results if not r.ok and r.severity == SEVERITY_WARNING]
    return {
        "artifact_version": ARTIFACT_VERSION,
        "generated_at": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "root": str(root),
        "source_revision": revision,
        "source_dirty": dirty,
        "summary": {
            "total": len(ctx.results),
            "passed": sum(1 for r in ctx.results if r.ok),
            "errors": len(errors),
            "warnings": len(warnings),
        },
        "checks": [r.as_dict() for r in ctx.results],
    }


def passed(report: dict[str, Any], *, strict: bool) -> bool:
    summary = report["summary"]
    if summary["errors"]:
        return False
    return not (strict and summary["warnings"])


def format_report(report: dict[str, Any], *, verbose: bool) -> str:
    lines: list[str] = []
    for check in report["checks"]:
        if check["ok"] and not verbose:
            continue
        mark = "ok  " if check["ok"] else ("WARN" if check["severity"] == SEVERITY_WARNING else "FAIL")
        lines.append(f"  [{mark}] {check['id']}: {check['detail']}")
        if not check["ok"] and check.get("expected") is not None:
            lines.append(f"           expected: {check['expected']}")
            lines.append(f"           actual:   {check['actual']}")
    summary = report["summary"]
    lines.append(
        f"  {summary['passed']}/{summary['total']} checks passed, "
        f"{summary['errors']} error(s), {summary['warnings']} warning(s)")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="bigcherry audit",
        description="Audit an upstream llama.cpp checkout for the invariants "
                    "bigcherry's patches depend on.")
    parser.add_argument("--llama-root", default=None,
                        help="llama.cpp checkout (default: vendor/llama.cpp)")
    parser.add_argument("--out", default=None,
                        help="write the audit JSON here "
                             "(default: artifacts/<rev>/source-audit.json)")
    parser.add_argument("--strict", action="store_true", default=True,
                        help="treat warnings as failures (default)")
    parser.add_argument("--no-strict", dest="strict", action="store_false")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="also print passing checks")
    args = parser.parse_args(argv)

    root = paths.llama_root(args.llama_root)
    report = audit(root)
    report["strict"] = args.strict

    out = Path(args.out) if args.out else \
        paths.artifact_dir(report["source_revision"]) / "source-audit.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    good = passed(report, strict=args.strict)
    print(f"source audit of {root}")
    print(f"  revision {report['source_revision'][:12]}"
          f"{' (dirty)' if report['source_dirty'] else ''}")
    print(format_report(report, verbose=args.verbose))
    print(f"  report: {out}")
    print("  RESULT: " + ("PASS" if good else "FAIL"))
    return 0 if good else 1


if __name__ == "__main__":
    sys.exit(main())
