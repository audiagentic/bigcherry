"""Per-candidate device code size, extracted from a built HIP library (HI52).

This is a build-time tool. Nothing here runs inside the tuner, and nothing here
touches a GPU -- it reads a built shared library and the generated candidate
manifest, and reports how many bytes of device `.text` each candidate's kernels
occupy, per architecture.

What the number IS
------------------
The summed ELF size of the `__global__` kernel symbols a candidate's variant
instantiates, for one GPU architecture, read out of the AMDGPU code objects
embedded in the fat binary.

What the number is NOT
----------------------
That candidate's contribution to overall binary size. Helpers inlined into more
than one instantiation are counted once per instantiation that inlined them,
`.rodata` and the `.kd` kernel-descriptor objects are excluded, and host-side
launcher code is not counted at all. A consumer that wants "how much would the
binary shrink if this candidate were dropped" is asking a different question and
must not use this number for it.

Mapping
-------
Function pointers in the generated registry are per *family*, not per candidate
(see `autotune_catalog.render_registry`): every mmq candidate shares one launch
function and differs only in its `variant`. So there is no per-candidate symbol
to look up directly. The variant fields do map onto template parameters exactly,
and that is the mapping this module implements:

    mmq.cuh:  template <ggml_type type, int J, bool fallback>
              __global__ void mul_mat_q(...)
              template <ggml_type type, int J, bool fallback>
              __global__ void mul_mat_q_stream_k_fixup(...)

    manifest: config = {"type": "q8_0", "j": 64, "fallback": false, ...}

Both kernels are counted: a candidate's device footprint is the sum of its
kernels, and omitting the stream-k fixup would under-report every mmq candidate.

mmq, mmvq, mmvf and mmf all have verified template-parameter rules (see the
"mmvq/mmvf/mmf maps" section below), each pinned against real demangled
symbols from a built gfx1100 library rather than guessed. Each family sums
over axes that are compiled separately but are not part of a candidate's
config (mmq's stream_k_fixup kernel; has_fusion for mmvq/mmvf;
rows_per_block/has_ids for mmf) into that candidate's total, the same way a
candidate's footprint has always been "the sum of its kernels".
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from .inventory import type_name

ARTIFACT_VERSION = 1

# Only these families have a verified symbol mapping. Everything else is
# reported as unmapped rather than guessed at.
MAPPED_FAMILIES = ("mmq", "mmvq", "mmvf", "mmf")


class BinarySizeError(RuntimeError):
    pass


# --------------------------------------------------------------------- tooling

_LLVM_SEARCH_DIRS = (
    "/opt/rocm/llvm/bin",
    "/opt/rocm/bin",
)


def find_tool(name: str, override: str | None = None) -> str:
    """Locate an LLVM binutil.

    ROCm ships these outside the default PATH on most installs, so PATH alone is
    not enough; an explicit override always wins so a caller can pin a specific
    toolchain rather than whichever one happens to be first.
    """
    if override:
        if not Path(override).is_file():
            raise BinarySizeError(f"{name}: {override} does not exist")
        return override
    env = os.environ.get(f"BIGCHERRY_{name.upper().replace('-', '_')}")
    if env:
        return env
    found = shutil.which(name)
    if found:
        return found
    for directory in _LLVM_SEARCH_DIRS:
        candidate = Path(directory) / name
        if candidate.is_file():
            return str(candidate)
    raise BinarySizeError(
        f"{name} not found on PATH or in {', '.join(_LLVM_SEARCH_DIRS)}; "
        f"pass an explicit path or set BIGCHERRY_{name.upper().replace('-', '_')}"
    )


def _run(argv: list[str]) -> str:
    try:
        completed = subprocess.run(
            argv, check=True, capture_output=True, text=True, errors="replace")
    except FileNotFoundError as exc:
        raise BinarySizeError(f"cannot execute {argv[0]}") from exc
    except subprocess.CalledProcessError as exc:
        raise BinarySizeError(
            f"{argv[0]} failed ({exc.returncode}): {exc.stderr.strip()[:400]}") from exc
    return completed.stdout


# ------------------------------------------------------------- code objects

# llvm-objdump --offloading names its output
# "<input>.<index>.<offload-kind>-<triple>-<arch>", e.g.
# "lib.so.0.hipv4-amdgcn-amd-amdhsa--gfx1100". The arch is the trailing
# component after the final '-'.
_BUNDLE_SUFFIX = re.compile(r"\.(?P<index>\d+)\.(?P<kind>[^.]+)-(?P<arch>gfx\w+)$")


def extract_code_objects(library: Path, workdir: Path,
                         objdump: str) -> dict[str, list[Path]]:
    """Split a HIP fat binary into per-architecture AMDGPU code objects.

    `llvm-objdump --offloading` writes its output *next to the input file*, so
    the library is copied into `workdir` first rather than littering the build
    tree. One code object per translation unit per target: a real 172 MB
    `libggml-hip.so` yields ~135 objects per architecture, not one.
    """
    workdir.mkdir(parents=True, exist_ok=True)
    staged = workdir / library.name
    if not staged.exists():
        shutil.copy2(library, staged)
    _run([objdump, "--offloading", str(staged)])

    by_arch: dict[str, list[Path]] = {}
    for path in sorted(workdir.iterdir()):
        if path == staged or not path.is_file():
            continue
        match = _BUNDLE_SUFFIX.search(path.name)
        if match is None:
            continue  # host bundle, or the staged library itself
        by_arch.setdefault(match.group("arch"), []).append(path)
    if not by_arch:
        raise BinarySizeError(
            f"{library}: no AMDGPU offload bundles found -- not a HIP fat binary, "
            "or built without device code")
    return by_arch


# ------------------------------------------------------------------- symbols

# llvm-readelf --syms prints EVERY symbol table in the object, and an AMDGPU
# code object carries both .dynsym and .symtab with overlapping contents -- so
# each kernel is listed twice by design. Deduplicating per object is required;
# treating the second listing as a duplicate definition would fire a spurious
# disagreement warning on literally every kernel.
_SYM_ROW = re.compile(
    r"^\s*\d+:\s+[0-9a-fA-F]+\s+(?P<size>\d+)\s+(?P<type>\w+)\s+\S+\s+\S+\s+\S+\s+"
    r"(?P<name>\S+)\s*$")


def symbol_sizes(obj: Path, readelf: str) -> dict[str, int]:
    """`{mangled name: byte size}` for the FUNC symbols in one code object.

    Zero-size symbols and non-FUNC entries (notably the 64-byte `.kd` kernel
    descriptors, which are OBJECT) are excluded: they are not device code.
    """
    sizes: dict[str, int] = {}
    for line in _run([readelf, "--syms", str(obj)]).splitlines():
        match = _SYM_ROW.match(line)
        if match is None or match.group("type") != "FUNC":
            continue
        size = int(match.group("size"))
        if size <= 0:
            continue
        name = match.group("name")
        # Same symbol from .dynsym and .symtab: identical by construction, so
        # max() is a no-op here and only guards against a malformed object.
        sizes[name] = max(sizes.get(name, 0), size)
    return sizes


def fold_objects(objects: Iterable[Path],
                 readelf: str) -> tuple[dict[str, int], list[str]]:
    """Merge per-object symbol tables into one `{symbol: size}` for an arch.

    A kernel instantiation normally lives in exactly one translation unit, but
    small shared helpers (`no_device_code`, etc.) are emitted into dozens of
    them at slightly different sizes depending on what got inlined around them.
    That is ordinary and not actionable, so warnings are:

    - **aggregated per symbol**, one line with the observed range and how many
      objects took part, never one line per pair of objects (which on a real
      library produced ~70 near-identical lines about a single helper); and
    - **restricted to mapped-family kernels**, the only symbols whose size this
      tool actually reports. A helper's size varying between TUs says nothing
      about any candidate's number.

    The larger size wins, so a candidate is never under-reported.
    """
    observed: dict[str, list[tuple[str, int]]] = {}
    for obj in objects:
        for name, size in symbol_sizes(obj, readelf).items():
            observed.setdefault(name, []).append((obj.name, size))

    merged: dict[str, int] = {}
    warnings: list[str] = []
    for name, sightings in observed.items():
        sizes = [size for _, size in sightings]
        merged[name] = max(sizes)
        if min(sizes) != max(sizes) and _parse_mapped_symbol(name) is not None:
            warnings.append(
                f"{name}: size varies across {len(sightings)} code objects "
                f"({min(sizes)}-{max(sizes)} B); taking the larger. Same kernel "
                "compiled differently per translation unit -- check build flags.")
    return merged, sorted(warnings)


# ------------------------------------------------------------------- mmq map

# Itanium mangling of the two mmq kernel templates. The length prefixes (9 and
# 24) are the identifier lengths of "mul_mat_q" and "mul_mat_q_stream_k_fixup";
# both forms were confirmed present in a real gfx1100 code object dump.
_MMQ_SYMBOL = re.compile(
    r"^_ZL(?:9(?P<main>mul_mat_q)|24(?P<fixup>mul_mat_q_stream_k_fixup))"
    r"IL9ggml_type(?P<type>\d+)ELi(?P<j>\d+)ELb(?P<fallback>[01])EE")


@dataclass(frozen=True)
class MmqKey:
    type_name: str
    j: int
    fallback: bool


def parse_mmq_symbol(symbol: str) -> tuple[MmqKey, str] | None:
    """`(variant key, kernel name)` for an mmq kernel symbol, else None.

    Returns None -- rather than raising -- for every symbol that is not one of
    the two mmq kernel templates, since the vast majority of symbols in the
    library legitimately are not.
    """
    match = _MMQ_SYMBOL.match(symbol)
    if match is None:
        return None
    name = type_name(int(match.group("type")))
    if name is None:
        return None
    kernel = match.group("main") or match.group("fixup")
    return MmqKey(name, int(match.group("j")), match.group("fallback") == "1"), kernel


def mmq_key_of_candidate(candidate: dict[str, Any]) -> MmqKey | None:
    config = candidate.get("config") or {}
    if "type" not in config or "j" not in config or "fallback" not in config:
        return None  # native wrapper, or a shape this rule does not cover
    return MmqKey(str(config["type"]), int(config["j"]), bool(config["fallback"]))


# ----------------------------------------------------------- mmvq/mmvf/mmf maps
#
# Verified 2026-08-11 against a real gfx1100 code object (bc-build-vault,
# ROCm 7.2.4 clang), by demangling representative symbols with c++filt rather
# than guessing at the mangling. Two things the earlier, unverified plan got
# wrong are corrected here because the real symbols disagree with it:
#
#   1. Trailing template arguments equal to their default are NOT omitted
#      from the mangling in this build -- nwarps_explicit=0/
#      rows_per_block_explicit=0 (the "derive as upstream" default) are
#      mangled explicitly as `Li0ELi0E`, not dropped. So no special
#      defaulting logic is needed: the regexes below capture every
#      instantiation, defaulted or not, the same way.
#   2. mmf.cuh's `T` is not the plain element type -- it is the *packed*
#      compute type (`__half2`, `__hip_bfloat162`, or `float`), confirmed by
#      demangling `mul_mat_f<__hip_bfloat162, 32, 10, 1, false>(...)`. Mapped
#      back to the same f16/f32/bf16 names candidates use via
#      _FLOAT_TYPE_TOKENS below.

# `{mangled length-prefixed identifier: ggml float type name}`. Covers both
# mmvf's plain element types and mmf's packed compute types, since both
# collapse onto the same three source types this catalog enumerates
# (FLOAT_TYPES in autotune_catalog.py).
_FLOAT_TYPE_TOKENS = {
    "__half": "f16", "__half2": "f16",
    "__hip_bfloat16": "bf16", "__hip_bfloat162": "bf16",
}

# A length-prefixed Itanium identifier: <decimal length><that many chars>.
_LENGTH_PREFIXED = re.compile(r"^(?P<len>\d+)")


def _decode_float_type_token(remainder: str) -> tuple[str | None, str]:
    """Consume one type token from the head of `remainder`.

    Returns (canonical type name or None if unrecognised, rest of string).
    Handles the three forms seen in real symbols: the single-character
    built-in-type code `f` (float), a length-prefixed vendor type
    (`6__half`, `15__hip_bfloat162`, ...), and an Itanium substitution
    back-reference (`S0_`, `S_`, ...) used when type_acc == T verbatim --
    observed only for mmvf's `<__half, __half, ...>` (f16 source, f16
    accumulator) instantiation.
    """
    if remainder.startswith("f"):
        return "f32", remainder[1:]
    if remainder.startswith("S"):
        match = re.match(r"^S\d*_", remainder)
        if match is None:
            return None, remainder
        return "__SUBSTITUTION__", remainder[match.end():]
    match = _LENGTH_PREFIXED.match(remainder)
    if match is None:
        return None, remainder
    length = int(match.group("len"))
    start = match.end()
    name = remainder[start:start + length]
    return _FLOAT_TYPE_TOKENS.get(name), remainder[start + length:]


# mmvf.cu:12 -- template <typename T, typename type_acc, int ncols_dst,
# int block_size, bool has_fusion = false, bool is_multi_token_id = false>.
# has_fusion/is_multi_token_id are compiled separately (both bool values are
# real, distinct symbols in the library) but are not part of a candidate's
# config, so they are runtime-selectable variants of *one* candidate and are
# deliberately excluded from the key -- their sizes are meant to sum together
# under the one "mul_mat_vec_f" kernel name, the same way MMQ's main and
# stream_k_fixup kernels sum into one candidate's text_bytes.
_MMVF_PREFIX = "_ZL13mul_mat_vec_fI"
_MMVF_TAIL = re.compile(r"^Li(?P<width>-?\d+)ELi(?P<block_size>-?\d+)ELb[01]ELb[01]EE")


@dataclass(frozen=True)
class MmvfKey:
    type_name: str
    width: int
    block_size: int
    accumulator: str


def parse_mmvf_symbol(symbol: str) -> tuple[MmvfKey, str] | None:
    if not symbol.startswith(_MMVF_PREFIX):
        return None
    rest = symbol[len(_MMVF_PREFIX):]
    source_type, rest = _decode_float_type_token(rest)
    if source_type is None or source_type == "__SUBSTITUTION__":
        return None  # T is never a substitution; only type_acc can be
    accumulator, rest = _decode_float_type_token(rest)
    if accumulator == "__SUBSTITUTION__":
        accumulator = source_type  # S0_ means type_acc == T verbatim
    if accumulator is None:
        return None
    match = _MMVF_TAIL.match(rest)
    if match is None:
        return None
    key = MmvfKey(source_type, int(match.group("width")), int(match.group("block_size")),
                  accumulator)
    return key, "mul_mat_vec_f"


def mmvf_key_of_candidate(candidate: dict[str, Any]) -> MmvfKey | None:
    config = candidate.get("config") or {}
    if not {"type", "width", "block_size", "accumulator"} <= config.keys():
        return None
    return MmvfKey(str(config["type"]), int(config["width"]), int(config["block_size"]),
                   str(config["accumulator"]))


# mmvq.cu:493 -- template <ggml_type type, int ncols_dst, bool has_fusion,
# bool small_k = false, int nwarps_explicit = 0, int rows_per_block_explicit
# = 0>. has_fusion is compiled separately (both values are real, distinct
# symbols) but is not in a candidate's config, so -- exactly as with mmvf --
# it is excluded from the key and left to sum under one kernel name.
# small_k *is* part of config and is kept.
#
# The length prefix (13) is what keeps this from ever matching
# mul_mat_vec_q_moe (prefix 17, a structurally different, two-parameter
# kernel): Itanium mangling encodes the identifier length before the name,
# so "13mul_mat_vec_qI" and "17mul_mat_vec_q_moeI" cannot collide even though
# "mul_mat_vec_q" is a literal prefix of "mul_mat_vec_q_moe".
_MMVQ_SYMBOL = re.compile(
    r"^_ZL13mul_mat_vec_qIL9ggml_type(?P<type>\d+)ELi(?P<width>-?\d+)"
    r"ELb[01]ELb(?P<small_k>[01])ELi(?P<nwarps>-?\d+)ELi(?P<rows>-?\d+)EE")

# mmvq.cu:738 -- template <ggml_type type, int c_rows_per_block>. Not mapped
# to any candidate: this catalog does not enumerate MoE-mode MMVQ candidates
# (no candidate config carries a MoE flag), so every mul_mat_vec_q_moe
# instantiation is upstream's own routing code and correctly reports as
# unmapped. Parsed here only so it can never be swallowed by _MMVQ_SYMBOL.
_MMVQ_MOE_SYMBOL = re.compile(
    r"^_ZL17mul_mat_vec_q_moeIL9ggml_type(?P<type>\d+)ELi(?P<rows>-?\d+)EE")


@dataclass(frozen=True)
class MmvqKey:
    type_name: str
    width: int
    nwarps: int
    rows_per_block: int
    small_k: bool


def parse_mmvq_symbol(symbol: str) -> tuple[MmvqKey, str] | None:
    if _MMVQ_MOE_SYMBOL.match(symbol) is not None:
        return None  # a different kernel; must not be mis-parsed as the main one
    match = _MMVQ_SYMBOL.match(symbol)
    if match is None:
        return None
    name = type_name(int(match.group("type")))
    if name is None:
        return None
    key = MmvqKey(name, int(match.group("width")), int(match.group("nwarps")),
                  int(match.group("rows")), match.group("small_k") == "1")
    return key, "mul_mat_vec_q"


def mmvq_key_of_candidate(candidate: dict[str, Any]) -> MmvqKey | None:
    config = candidate.get("config") or {}
    if not {"type", "width", "nwarps", "rows_per_block", "small_k"} <= config.keys():
        return None
    return MmvqKey(str(config["type"]), int(config["width"]), int(config["nwarps"]),
                   int(config["rows_per_block"]), bool(config["small_k"]))


# mmf.cuh:49 -- template <typename T, int rows_per_block, int cols_per_block,
# int nwarps, bool has_ids>. Neither rows_per_block nor has_ids is part of a
# candidate's config (only type/width/nwarps are, matching upstream's own
# `cols_per_block` and `nwarps` axes); both are compiled-separately,
# runtime-selected variants of one candidate and are excluded from the key,
# left to sum together the same way MMQ and the two families above do.
_MMF_PREFIX = "_ZL9mul_mat_fI"
_MMF_TAIL = re.compile(r"^Li-?\d+ELi(?P<cols>-?\d+)ELi(?P<nwarps>-?\d+)ELb[01]EE")


@dataclass(frozen=True)
class MmfKey:
    type_name: str
    width: int
    nwarps: int


def parse_mmf_symbol(symbol: str) -> tuple[MmfKey, str] | None:
    if not symbol.startswith(_MMF_PREFIX):
        return None
    rest = symbol[len(_MMF_PREFIX):]
    source_type, rest = _decode_float_type_token(rest)
    if source_type is None or source_type == "__SUBSTITUTION__":
        return None
    match = _MMF_TAIL.match(rest)
    if match is None:
        return None
    key = MmfKey(source_type, int(match.group("cols")), int(match.group("nwarps")))
    return key, "mul_mat_f"


def mmf_key_of_candidate(candidate: dict[str, Any]) -> MmfKey | None:
    config = candidate.get("config") or {}
    if not {"type", "width", "nwarps"} <= config.keys():
        return None
    return MmfKey(str(config["type"]), int(config["width"]), int(config["nwarps"]))


# One entry point per mapped family: (symbol parser, candidate-key-builder).
# analyse_architecture() and fold_objects() dispatch through this table so a
# new family is one entry, not a rewrite of either function.
_FAMILY_PARSERS: dict[str, tuple[Any, Any]] = {
    "mmq":  (parse_mmq_symbol, mmq_key_of_candidate),
    "mmvq": (parse_mmvq_symbol, mmvq_key_of_candidate),
    "mmvf": (parse_mmvf_symbol, mmvf_key_of_candidate),
    "mmf":  (parse_mmf_symbol, mmf_key_of_candidate),
}


def _parse_mapped_symbol(symbol: str) -> tuple[str, Any, str] | None:
    """(family, key, kernel name) for the first family whose parser matches, else None."""
    for family, (parser, _) in _FAMILY_PARSERS.items():
        parsed = parser(symbol)
        if parsed is not None:
            key, kernel = parsed
            return family, key, kernel
    return None


# --------------------------------------------------------------------- report

@dataclass
class ArchResult:
    architecture: str
    code_objects: int
    candidates: dict[str, dict[str, Any]] = field(default_factory=dict)
    unmapped_symbols: list[str] = field(default_factory=list)
    unresolved_candidates: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def symbol_map(self) -> dict[str, list[str]]:
        """`{mangled symbol: [stable_name]}`, the shape `resource_report` wants.

        Emitted per architecture on purpose: one `(type, J, fallback)`
        instantiation resolves to different stable names on different
        architectures, because the mmq config table gives it a different
        thread/occupancy config there, and a flattened cross-architecture map
        would therefore be ambiguous for exactly the symbols that matter most.
        """
        out: dict[str, list[str]] = {}
        for stable_name, entry in self.candidates.items():
            for symbol in entry["symbols"]:
                out.setdefault(symbol, []).append(stable_name)
        return {symbol: sorted(names) for symbol, names in sorted(out.items())}


def analyse_architecture(architecture: str, objects: list[Path],
                         candidates: list[dict[str, Any]],
                         readelf: str) -> ArchResult:
    sizes, warnings = fold_objects(objects, readelf)
    result = ArchResult(architecture=architecture, code_objects=len(objects),
                        warnings=warnings)

    # By family: each family's keys are its own dataclass type (MmqKey,
    # MmvqKey, ...), never comparable across families, so one shared dict
    # keyed on (family, key) cannot accidentally collide two families'
    # instantiations onto the same bucket.
    by_family_key: dict[tuple[str, Any], list[tuple[str, str, int]]] = {}
    for symbol, size in sizes.items():
        parsed = _parse_mapped_symbol(symbol)
        if parsed is None:
            continue
        family, key, kernel = parsed
        by_family_key.setdefault((family, key), []).append((symbol, kernel, size))

    claimed: set[str] = set()
    for candidate in candidates:
        family = candidate["family"]
        if family not in _FAMILY_PARSERS:
            continue
        if architecture not in candidate.get("architectures", []):
            continue
        _, key_of_candidate = _FAMILY_PARSERS[family]
        key = key_of_candidate(candidate)
        if key is None:
            continue  # native wrapper: no instantiation of its own
        found = by_family_key.get((family, key))
        if not found:
            result.unresolved_candidates.append(candidate["stable_name"])
            continue
        kernels: dict[str, int] = {}
        for symbol, kernel, size in found:
            kernels[kernel] = kernels.get(kernel, 0) + size
            claimed.add(symbol)
        result.candidates[candidate["stable_name"]] = {
            "text_bytes": sum(kernels.values()),
            "kernels": dict(sorted(kernels.items())),
            "symbols": sorted(symbol for symbol, _, _ in found),
        }

    result.unmapped_symbols = sorted(
        symbol for symbol in sizes
        if _parse_mapped_symbol(symbol) is not None and symbol not in claimed)
    return result


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_report(library: Path, manifest: dict[str, Any], workdir: Path,
                 *, objdump: str, readelf: str) -> dict[str, Any]:
    objects_by_arch = extract_code_objects(library, workdir, objdump)
    candidates = manifest.get("candidates", [])
    if not candidates:
        raise BinarySizeError("manifest contains no candidates")

    results = {
        arch: analyse_architecture(arch, objects, candidates, readelf)
        for arch, objects in sorted(objects_by_arch.items())
    }
    return {
        "artifact_version": ARTIFACT_VERSION,
        "library": str(library),
        "library_sha256": _sha256(library),
        "source_revision": manifest.get("source_revision"),
        "producer_manifest_hash": manifest.get("producer_manifest_hash"),
        "mapped_families": list(MAPPED_FAMILIES),
        "unmapped_symbols_note": (
            "mmq-template symbols present in the build that no manifest candidate "
            "claims -- overwhelmingly upstream llama.cpp's own instantiations for "
            "quant types and J values this catalog does not enumerate candidates "
            "for. Expected to be large; not an error."
        ),
        "metric": (
            "summed ELF size of the __global__ kernel symbols a candidate's variant "
            "instantiates, per architecture; excludes .rodata, .kd descriptors and "
            "host launcher code, and counts inlined helpers once per instantiation"
        ),
        "architectures": {
            arch: {
                "code_objects": r.code_objects,
                "candidates": r.candidates,
                "unmapped_symbols": r.unmapped_symbols,
                "unresolved_candidates": r.unresolved_candidates,
                "warnings": r.warnings,
                "symbol_map": r.symbol_map(),
            }
            for arch, r in results.items()
        },
    }


def unresolved_summary(report: dict[str, Any]) -> list[str]:
    """Architectures whose mapped-family candidates did not all resolve.

    A partially-populated table is worse than none: a Pareto profile ranking on
    this axis would treat every missing candidate as costing zero bytes.
    """
    problems = []
    for arch, entry in report["architectures"].items():
        if entry["unresolved_candidates"]:
            problems.append(
                f"{arch}: {len(entry['unresolved_candidates'])} candidate(s) resolved to "
                f"no symbol: {', '.join(entry['unresolved_candidates'][:5])}"
                + (" ..." if len(entry["unresolved_candidates"]) > 5 else ""))
    return problems


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="bigcherry candidate-binary-size",
        description="Per-candidate device .text size from a built HIP library")
    parser.add_argument("library", type=Path,
                        help="built libggml-hip.so (the real one, not a stub)")
    parser.add_argument("--manifest", type=Path, required=True,
                        help="artifacts/<rev>/hip-autotune-manifest.json")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workdir", type=Path, default=None,
                        help="where extracted code objects land (default: <output>.objects)")
    parser.add_argument("--symbol-map-dir", type=Path, default=None,
                        help="also write per-architecture {symbol: [stable_name]} maps "
                             "in the shape `bigcherry resource-report --symbol-map` reads")
    parser.add_argument("--objdump", default=None, help="path to llvm-objdump")
    parser.add_argument("--readelf", default=None, help="path to llvm-readelf")
    parser.add_argument("--allow-unresolved", action="store_true",
                        help="report unresolved candidates instead of failing "
                             "(diagnostic use only -- the resulting table is incomplete)")
    args = parser.parse_args(argv)

    if not args.library.is_file():
        raise BinarySizeError(f"{args.library}: not a file")
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    workdir = args.workdir or args.output.with_suffix(".objects")

    report = build_report(
        args.library, manifest, workdir,
        objdump=find_tool("llvm-objdump", args.objdump),
        readelf=find_tool("llvm-readelf", args.readelf))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    if args.symbol_map_dir:
        args.symbol_map_dir.mkdir(parents=True, exist_ok=True)
        for arch, entry in report["architectures"].items():
            (args.symbol_map_dir / f"symbol-map-{arch}.json").write_text(
                json.dumps(entry["symbol_map"], indent=2) + "\n", encoding="utf-8")

    for arch, entry in report["architectures"].items():
        print(f"{arch}: {len(entry['candidates'])} candidates, "
              f"{entry['code_objects']} code objects, "
              f"{len(entry['unmapped_symbols'])} unmapped symbols, "
              f"{len(entry['warnings'])} warnings")
        for warning in entry["warnings"]:
            print(f"  warning: {warning}", file=sys.stderr)

    problems = unresolved_summary(report)
    if problems and not args.allow_unresolved:
        for problem in problems:
            print(f"error: {problem}", file=sys.stderr)
        raise BinarySizeError(
            "candidate-to-symbol mapping is incomplete; refusing to emit a partial "
            "table (pass --allow-unresolved to inspect it anyway)")
    for problem in problems:
        print(f"warning: {problem}", file=sys.stderr)
    return 0


if __name__ == "__main__":  # pragma: no cover
    try:
        sys.exit(main())
    except BinarySizeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        sys.exit(2)
