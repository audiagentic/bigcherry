"""Candidate catalog generator — the single source of truth (HI03).

Standards 2.5: this module produces *every* downstream artifact.

    hip-autotune-manifest.json      the candidate set, for tools and tests
    hip-autotune-registry.inc       the compiled registry table
    mmvq-autotune-instance-*.cu     generated MMVQ template instances
    hip-autotune-build-hash.h       the manifest hash, embedded in the binary

Nothing here is hand-maintained anywhere else. A candidate in the manifest with
no registration, or a generated source with no catalog record, is a bug the
test suite is required to catch (HI16).

The important design decision is that the catalog is **read out of upstream**
rather than duplicated from it. MMQ is the clearest case: its configuration
table is sparse. Upstream's ``CASE`` macro defines only certain
(type, J, fallback) combinations, and ``ggml_cuda_mmq_get_config`` returns a
sentinel for the rest, which the native J search skips. Enumerating all sixteen
J values from ``range(8, 129, 8)`` would therefore invent candidates that abort
on launch. So the generator parses the architecture-specific config headers and
emits exactly the rows that exist — which also yields the fully resolved config
the stable name is required to carry (standards 2.2).
"""

from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from . import ARTIFACT_VERSION
from . import autotune_schema as schema
from . import csource
from . import paths
from .source_audit import read_types_mmq, git_revision
from .resource_report import ResourceError, load_blacklist

# Architecture -> the MMQ config header that defines its table, derived from the
# family map rather than restated, so adding an architecture in one place is
# enough. Mirrors upstream's fallthrough in ggml_cuda_mmq_get_config.
ARCH_CONFIG_HEADERS = {
    arch: f"mmq-config-{family}.cuh"
    for arch, family in schema.ARCHITECTURE_FAMILY.items()
}

# Every AMD target. The candidate set stays proportionate because MMQ candidates
# are keyed by resolved config, so the many architectures sharing a config table
# share one candidate carrying all their architecture bits.
DEFAULT_ARCHITECTURES = tuple(schema.ARCHITECTURE_GROUPS["all"])

# Float source types the MMVF and MMF families accept.
FLOAT_TYPES = ("f32", "f16", "bf16")

# Standards 14 / HI09: the bounded matrix explicit MMVQ geometry is generated
# over. Widths are capped by MMVQ_MAX_BATCH_SIZE, which the audit pins at 8.
MMVQ_WIDTHS = tuple(range(1, 9))
MMVQ_NWARPS = (1, 2, 4, 6, 8)
MMVQ_ROWS_PER_BLOCK = (1, 2)

# HI09 sequencing: prove the mechanism on one type before widening. `full-max`
# ignores this and generates everything, as a compile audit.
MMVQ_STAGED_TYPES = ("q8_0", "q6_k")

MMVF_BLOCK_SIZES = tuple(range(32, 257, 32))
MMVF_WIDTHS = tuple(range(1, 9))
MMF_WIDTHS = tuple(range(1, 17))
MMF_NWARPS = tuple(range(1, 9))

IMPLEMENTATION_VERSION = 1


class CatalogError(RuntimeError):
    pass


# ----------------------------------------------------------------- MMQ config

@dataclass(frozen=True)
class MMQConfigRow:
    """One row of an architecture's MMQ ``CASE`` table."""

    type: str          # short name, e.g. "q8_0"
    nthreads: int
    occupancy: int
    i: int
    j: int
    sram_layout: str   # short name, e.g. "q8_0"
    k_vram: int
    stream_k: bool
    fallback: bool


# CASE(type, nthreads, occupancy, I, J, sram_layout, K_vram, stream_k, fallback)
_CASE_RE = re.compile(
    r"CASE\(\s*GGML_TYPE_(\w+)\s*,"
    r"\s*(\d+)\s*,"
    r"\s*(\d+)\s*,"
    r"\s*(\d+)\s*,"
    r"\s*(\d+)\s*,"
    r"\s*GGML_CUDA_MMQ_SRAM_LAYOUT_(\w+)\s*,"
    r"\s*([A-Z_0-9]+)\s*,"
    r"\s*(true|false)\s*,"
    r"\s*(true|false)\s*\)"
)

# K_vram is written as a macro in the tables; resolve it to its numeric value so
# the stable name records what was actually compiled.
_K_VRAM_MACROS = {
    "MMQ_ITER_K": 256,
    "MMQ_ITER_K_FP4": 512,
}


def read_mmq_config_table(cuda_dir: Path, header: str) -> list[MMQConfigRow]:
    path = cuda_dir / header
    if not path.is_file():
        raise CatalogError(f"MMQ config header not found: {path}")
    text = csource.strip_noise(csource.read(path))

    rows: list[MMQConfigRow] = []
    for match in _CASE_RE.finditer(text):
        k_vram_token = match.group(7)
        if k_vram_token not in _K_VRAM_MACROS:
            raise CatalogError(
                f"{header}: unknown K_vram macro {k_vram_token!r}. Add it to "
                f"_K_VRAM_MACROS -- guessing would put a wrong value into a "
                f"stable name, which is a persistent database identity.")
        rows.append(MMQConfigRow(
            type=match.group(1).lower(),
            nthreads=int(match.group(2)),
            occupancy=int(match.group(3)),
            i=int(match.group(4)),
            j=int(match.group(5)),
            sram_layout=match.group(6).lower(),
            k_vram=_K_VRAM_MACROS[k_vram_token],
            stream_k=match.group(8) == "true",
            fallback=match.group(9) == "true",
        ))
    if not rows:
        raise CatalogError(
            f"{header}: no CASE rows parsed. The table format has changed and "
            f"the MMQ candidate set cannot be derived from this checkout.")
    return rows


def mmq_table_coverage(cuda_dir: Path) -> dict[str, Any]:
    """Report which (type, J, fallback) combinations each table defines.

    The tables are sparse and upstream extends them over time. Because the
    catalog is re-derived from the tables on every release, a newly added row
    becomes a new candidate automatically -- there is no list to remember to
    update. This report exists so that change is *visible*: it is written
    beside the manifest, and diffing two releases' copies shows exactly which
    solutions upstream added or withdrew.

    Note that a new row changes the manifest hash, which opens a new build
    namespace (standards 13.1). That is intended: the candidate set differs, so
    prior winners were chosen from a different set of options.
    """
    coverage: dict[str, Any] = {}
    for family in sorted(set(schema.ARCHITECTURE_FAMILY.values())):
        header = f"mmq-config-{family}.cuh"
        if not (cuda_dir / header).is_file():
            continue
        per_type: dict[str, dict[str, list[int]]] = {}
        for row in read_mmq_config_table(cuda_dir, header):
            entry = per_type.setdefault(row.type, {"fallback": [], "direct": []})
            entry["fallback" if row.fallback else "direct"].append(row.j)
        for entry in per_type.values():
            entry["fallback"] = sorted(entry["fallback"])
            entry["direct"] = sorted(entry["direct"])
        coverage[family] = {
            "header": header,
            "rows": sum(len(e["fallback"]) + len(e["direct"])
                        for e in per_type.values()),
            "types": dict(sorted(per_type.items())),
        }
    return coverage


# ------------------------------------------------------------------ candidates

@dataclass
class Candidate:
    stable_name: str
    family: str
    source_class: str
    architectures: list[str]
    config: dict[str, Any]
    graph_safe: bool = False
    deterministic: bool = True
    implementation_version: int = IMPLEMENTATION_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "stable_name": self.stable_name,
            "family": self.family,
            "source_class": self.source_class,
            "implementation_version": self.implementation_version,
            "architectures": sorted(self.architectures),
            "architecture_mask": schema.architecture_mask(self.architectures),
            "graph_safe": self.graph_safe,
            "deterministic": self.deterministic,
            "config": self.config,
        }


@dataclass
class Inventory:
    """Observed types and paths from a record-mode run (standards 6.2).

    ``workload-max`` is the practical maximal build: it emits variants only for
    what the workload actually executed. An empty inventory means "everything",
    which is only correct for ``full-max``.
    """

    mmq_types: set[str] = field(default_factory=set)
    mmvq_types: set[str] = field(default_factory=set)
    mmvf_types: set[str] = field(default_factory=set)
    mmf_types: set[str] = field(default_factory=set)
    widths: set[int] = field(default_factory=set)
    uses_blas: bool = True

    @property
    def quant_types(self) -> set[str]:
        """Every quantised type observed, in any family.

        Deliberately a union rather than per-family. The tuner compares
        families for unfused signatures (plan 11.3), so a q8_0 operation that
        native always sent to MMQ must still have MMVQ candidates available --
        otherwise "MMQ won" only means "MMQ was the only thing compiled", which
        is not a measurement.
        """
        return self.mmq_types | self.mmvq_types

    @property
    def float_types(self) -> set[str]:
        """Every float type observed, in any family.

        Same reasoning. On the 27B MTP profile `mmf_types` came back empty --
        native never chose MMF -- while `mmvf_types` had f32. Restricting MMF
        to its own observed set would generate no MMF candidates at all, and
        the tuner could never discover that MMF beats MMVF for a shape.
        """
        return self.mmvf_types | self.mmf_types

    @classmethod
    def from_json(cls, path: Path) -> "Inventory":
        data = json.loads(path.read_text(encoding="utf-8"))
        return cls(
            mmq_types=set(data.get("mmq_types", [])),
            mmvq_types=set(data.get("mmvq_types", [])),
            mmvf_types=set(data.get("mmvf_types", [])),
            mmf_types=set(data.get("mmf_types", [])),
            widths=set(int(w) for w in data.get("widths", [])),
            uses_blas=bool(data.get("uses_blas", True)),
        )


def _keep(observed: set[str], name: str, *, restrict: bool) -> bool:
    return not restrict or name in observed


def _keep_width(observed: set[int], width: int, *, restrict: bool) -> bool:
    return not restrict or width in observed


def apply_resource_blacklist(
        candidates: list[Candidate],
        blacklist: dict[tuple[str, str], tuple[str, ...]],
    ) -> list[Candidate]:
    """Remove compiler-proven generated candidates before manifest creation.

    Resource reports are an architecture-specific compile audit.  Keep a
    candidate on unaffected targets when a catalog row spans architectures;
    drop it only when every target is blacklisted.  Runtime/native wrappers
    are intentionally not affected by this HI09b input.
    """
    kept: list[Candidate] = []
    for candidate in candidates:
        if candidate.source_class != "new_generated_variant":
            kept.append(candidate)
            continue
        remaining = [arch for arch in candidate.architectures
                     if (candidate.stable_name, arch) not in blacklist]
        if remaining:
            candidate.architectures = remaining
            kept.append(candidate)
    return kept


# ---------------------------------------------------------------- enumeration

def enumerate_mmq(cuda_dir: Path, architectures: Iterable[str],
                  types: list[str], inventory: Inventory,
                  *, restrict: bool) -> list[Candidate]:
    """One candidate per distinct (CASE row), carrying every architecture whose
    table defines it.

    Grouping matters: a configuration shared by gfx1100 and gfx1201 becomes one
    candidate with both architecture bits rather than two candidates that would
    measure and store separately for no reason.
    """
    known = {t.removeprefix("GGML_TYPE_").lower() for t in types}
    by_name: dict[str, Candidate] = {}

    for arch in architectures:
        header = ARCH_CONFIG_HEADERS.get(arch)
        if header is None:
            raise CatalogError(
                f"no MMQ config header known for architecture {arch!r}")
        for row in read_mmq_config_table(cuda_dir, header):
            if row.type not in known:
                # A config row for a type the generator does not build is dead
                # weight upstream, not our problem -- but never a candidate.
                continue
            if not _keep(inventory.quant_types, row.type, restrict=restrict):
                continue

            # Standards 2.2: the name carries the whole resolved config, not
            # just J, because the config table can change independently of J.
            name = (f"mmq:{row.type}:j{row.j}:fb{int(row.fallback)}"
                    f":t{row.nthreads}:o{row.occupancy}:i{row.i}"
                    f":sram-{row.sram_layout}:k{row.k_vram}"
                    f":sk{int(row.stream_k)}:v{IMPLEMENTATION_VERSION}")

            existing = by_name.get(name)
            if existing is not None:
                if arch not in existing.architectures:
                    existing.architectures.append(arch)
                continue

            by_name[name] = Candidate(
                stable_name=name,
                family="mmq",
                source_class="existing_runtime",
                architectures=[arch],
                # MMQ is deterministic except under stream-k, whose fixup pass
                # reduces partial tiles in a launch-order-dependent way.
                deterministic=not row.stream_k,
                config={
                    "type": row.type,
                    "j": row.j,
                    "fallback": row.fallback,
                    "nthreads": row.nthreads,
                    "occupancy": row.occupancy,
                    "i": row.i,
                    "sram_layout": row.sram_layout,
                    "k_vram": row.k_vram,
                    "stream_k": row.stream_k,
                },
            )
    return list(by_name.values())


def enumerate_mmvf(architectures: list[str], inventory: Inventory,
                   *, restrict: bool) -> list[Candidate]:
    candidates: list[Candidate] = []
    for source_type in FLOAT_TYPES:
        if not _keep(inventory.float_types, source_type, restrict=restrict):
            continue
        # Standards 3.2: two accumulator modes exist for F16 and are genuine
        # performance variants. For F32 and BF16 only F32 accumulation exists,
        # and a lower-precision accumulator would be a different operation, not
        # a variant of this one.
        acc_modes = ("f32", "f16") if source_type == "f16" else ("f32",)
        for width in MMVF_WIDTHS:
            if not _keep_width(inventory.widths, width, restrict=restrict):
                continue
            for block_size in MMVF_BLOCK_SIZES:
                for acc in acc_modes:
                    candidates.append(Candidate(
                        stable_name=(f"mmvf:{source_type}:w{width}"
                                     f":bs{block_size}:acc{acc}"
                                     f":v{IMPLEMENTATION_VERSION}"),
                        family="mmvf",
                        source_class="existing_runtime",
                        architectures=list(architectures),
                        config={
                            "type": source_type,
                            "width": width,
                            "block_size": block_size,
                            "accumulator": acc,
                        },
                    ))
    return candidates


def enumerate_mmf(architectures: list[str], inventory: Inventory,
                  *, restrict: bool) -> list[Candidate]:
    candidates: list[Candidate] = []
    for source_type in FLOAT_TYPES:
        if not _keep(inventory.float_types, source_type, restrict=restrict):
            continue
        # MMF needs a matrix core for the source type, and which one depends on
        # the type: F32 is MFMA-only, so an `mmf:f32` candidate cannot run on
        # any RDNA part. Narrowing the architecture list here is what lets the
        # tuner reject it up front as GGML_HIP_REJECT_ARCHITECTURE instead of
        # carrying it through can_execute on every signature (review RV06).
        mmf_archs = schema.mmf_architectures(source_type, architectures)
        if not mmf_archs:
            continue
        for width in MMF_WIDTHS:
            if not _keep_width(inventory.widths, width, restrict=restrict):
                continue
            for nwarps in MMF_NWARPS:
                candidates.append(Candidate(
                    stable_name=(f"mmf:{source_type}:w{width}:nw{nwarps}"
                                 f":v{IMPLEMENTATION_VERSION}"),
                    family="mmf",
                    source_class="existing_runtime",
                    architectures=list(mmf_archs),
                    config={
                        "type": source_type,
                        "width": width,
                        "nwarps": nwarps,
                    },
                ))
    return candidates


def mmvq_geometry_is_valid(width: int, nwarps: int, rows_per_block: int) -> bool:
    """Standards 14: the bounds the explicit kernel static_asserts.

    Checked here so an invalid geometry never reaches a translation unit. A
    compile error in generated code is a far worse diagnostic than a candidate
    that was never generated.
    """
    return (1 <= width <= 8
            and 1 <= nwarps <= 8
            and rows_per_block >= 1)


def enumerate_mmvq(architectures: list[str], types: list[str],
                   inventory: Inventory, *, restrict: bool,
                   staged: bool) -> list[Candidate]:
    """Explicit MMVQ geometry variants (HI09).

    These are the only candidates needing new compilation units: upstream
    derives geometry from ``calc_nwarps``/``calc_rows_per_block`` at compile
    time, so an alternative geometry is a new template instance rather than a
    runtime argument.
    """
    known = sorted(t.removeprefix("GGML_TYPE_").lower() for t in types)
    candidates: list[Candidate] = []
    for quant in known:
        if staged and quant not in MMVQ_STAGED_TYPES:
            continue
        if not _keep(inventory.quant_types, quant, restrict=restrict):
            continue
        for width in MMVQ_WIDTHS:
            if not _keep_width(inventory.widths, width, restrict=restrict):
                continue
            for nwarps in MMVQ_NWARPS:
                for rows in MMVQ_ROWS_PER_BLOCK:
                    if not mmvq_geometry_is_valid(width, nwarps, rows):
                        continue
                    candidates.append(_mmvq_candidate(
                        architectures, quant, width, nwarps, rows,
                        small_k=False))

                # The small-K path is a distinct compiled geometry, not another
                # value of rows_per_block: upstream sets rows_per_block = nwarps
                # so each warp has more work when K is small enough that the
                # whole block covers every K block in one iteration. It exists
                # only for nwarps > 1 -- at nwarps == 1 it is the ordinary
                # geometry, and emitting it would duplicate a candidate.
                if nwarps > 1 and mmvq_geometry_is_valid(width, nwarps, nwarps):
                    candidates.append(_mmvq_candidate(
                        architectures, quant, width, nwarps, nwarps,
                        small_k=True))
    return candidates


def _mmvq_candidate(architectures: list[str], quant: str, width: int,
                    nwarps: int, rows: int, *, small_k: bool) -> Candidate:
    """One explicit MMVQ geometry.

    Deliberately **not** parameterised on fusion, though the prework prototype
    emitted `f0`/`f1` candidates. Fusion is selected at runtime inside
    `mul_mat_vec_q_switch_fusion` -- one compiled instance serves both the fused
    and unfused kernel -- so a pair of candidates differing only in fusion would
    name the same code and double the measurement work for nothing.

    Fusion is properly part of the *signature*: standards 11.1 makes a fused
    graph pattern a different semantic operation, tuned within its already
    selected family. So it belongs on the operation side of the key, not the
    candidate side, and `ggml_hip_dispatch_signature_v1.fusion` already carries
    it.

    Every dimension that *is* here appears in the stable name, because the name
    is the durable database identity (standards 2.1). Adding one later
    invalidates every stored winner, which is why small_k is present from the
    start rather than added when first measured.
    """
    return Candidate(
        stable_name=(f"mmvq:{quant}:w{width}:nw{nwarps}:rpb{rows}"
                     f":sk{int(small_k)}:v{IMPLEMENTATION_VERSION}"),
        family="mmvq",
        source_class="new_generated_variant",
        architectures=list(architectures),
        config={
            "type": quant,
            "width": width,
            "nwarps": nwarps,
            "rows_per_block": rows,
            "small_k": small_k,
        },
    )


def enumerate_natives(architectures: list[str],
                      families: Iterable[str]) -> list[Candidate]:
    """One native wrapper per family (standards 2.3, 7.3).

    The native candidate is upstream's own choice given a stable identity. It is
    always measured: it is both the correctness reference and the baseline a
    challenger has to beat by the replacement threshold.
    """
    return [
        Candidate(
            stable_name=f"{family}:native:v{IMPLEMENTATION_VERSION}",
            family=family,
            source_class="native_wrapper",
            architectures=list(architectures),
            # The native path is what upstream already runs under graph
            # capture, so it is graph-safe by construction. Every other
            # candidate must earn the flag in HI14 (standards 2.4).
            graph_safe=True,
            config={"policy": "upstream"},
        )
        for family in families
    ]


def enumerate_blas(architectures: list[str]) -> list[Candidate]:
    """Standards 16.1: hipBLAS is one opaque candidate to begin with.

    Enumerating internal hipBLAS solutions would have to namespace results by
    the exact ROCm/hipBLASLt version (16.2), which is deliberately out of scope.
    """
    return [Candidate(
        stable_name=f"blas:hipblas-auto:v{IMPLEMENTATION_VERSION}",
        family="blas",
        source_class="vendor_auto",
        architectures=list(architectures),
        graph_safe=False,
        # hipBLAS may pick a different internal kernel from run to run, so
        # bitwise determinism is not something we can assert without measuring.
        deterministic=False,
        config={"library": "hipblas", "algorithm": "auto"},
    )]


# ------------------------------------------------------------------- manifest

def read_winners(measurements: Path) -> set[str]:
    """Stable names that won at least one signature in a tuning run.

    Reads the same JSONL the replay-cache writer does, so a `replay-slim`
    build and the cache it will load are derived from one set of measurements
    rather than two readings of it.
    """
    winners: set[str] = set()
    with measurements.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue          # truncated tail; everything before it stands
            if record.get("kind") == "result" and record.get("winner"):
                winners.add(record["winner"])
    return winners


def enumerate_supported_candidates(cuda_dir: Path,
                                   architectures: list[str],
                                   types: list[str]) -> list[Candidate]:
    """Enumerate the architecture-valid candidate universe independently.

    Workload profiles intentionally restrict candidates to observed types and
    widths. Coverage must still answer whether an unobserved type is supported
    by this source checkout, so this mirrors ``full-max`` without consulting a
    record-mode inventory.
    """
    known = {t.removeprefix("GGML_TYPE_").lower() for t in types}
    unrestricted = Inventory(
        mmq_types=known,
        mmvq_types=known,
        mmvf_types=set(FLOAT_TYPES),
        mmf_types=set(FLOAT_TYPES),
        widths=set(range(1, max(MMF_WIDTHS) + 1)),
    )
    candidates = enumerate_natives(
        architectures, ["mmq", "mmvq", "mmvf", "mmf", "blas"])
    candidates += enumerate_mmq(cuda_dir, architectures, types, unrestricted,
                                restrict=False)
    candidates += enumerate_mmvf(architectures, unrestricted, restrict=False)
    candidates += enumerate_mmf(architectures, unrestricted, restrict=False)
    candidates += enumerate_mmvq(architectures, types, unrestricted,
                                 restrict=False, staged=False)
    candidates += enumerate_blas(architectures)
    return candidates


def build_manifest(root: Path, *, variant_set: str,
                   architectures: list[str],
                   inventory: Inventory | None,
                   source_revision: str,
                   winners: set[str] | None = None,
                   resource_blacklist: dict[tuple[str, str], tuple[str, ...]] | None = None) -> dict[str, Any]:
    if variant_set not in schema.VARIANT_SETS:
        raise CatalogError(
            f"unknown variant set {variant_set!r}; expected one of "
            f"{schema.VARIANT_SETS}")

    cuda = paths.cuda_dir(root)
    types = read_types_mmq(paths.generate_cu_files_py(root))
    if types is None:
        raise CatalogError(
            "could not read TYPES_MMQ from generate_cu_files.py -- the "
            "candidate set cannot be derived from this checkout")

    inventory = inventory or Inventory()
    supported_candidates = enumerate_supported_candidates(
        cuda, architectures, types)
    # `inventory` builds carry native candidates only: they exist to discover
    # which signatures a workload actually executes (standards 6.2).
    native_only = variant_set == "inventory"
    # `full-max` is the compile-audit profile and deliberately ignores the
    # observed workload; every other tuned profile restricts to it.
    restrict = variant_set not in ("full-max", "inventory")
    staged_mmvq = variant_set != "full-max"

    families = ["mmq", "mmvq", "mmvf", "mmf", "blas"]
    candidates: list[Candidate] = enumerate_natives(architectures, families)

    if not native_only:
        candidates += enumerate_mmq(cuda, architectures, types, inventory,
                                    restrict=restrict)
        candidates += enumerate_mmvf(architectures, inventory, restrict=restrict)
        candidates += enumerate_mmf(architectures, inventory, restrict=restrict)
        candidates += enumerate_mmvq(architectures, types, inventory,
                                     restrict=restrict, staged=staged_mmvq)
        # Always emitted, not only when native chose it. Plan 11.3 item 5
        # makes BLAS a candidate whenever it is semantically legal, and
        # `ggml_hip_blas_can_execute` is what decides that per signature.
        # Gating on "was it observed" would mean BLAS can only win where it
        # already won.
        candidates += enumerate_blas(architectures)

    if resource_blacklist:
        candidates = apply_resource_blacklist(candidates, resource_blacklist)
        supported_candidates = apply_resource_blacklist(
            supported_candidates, resource_blacklist)

    # `replay-slim` is the production profile: compile only the variants a
    # tuning run actually chose, so the binary carries the winners and nothing
    # else. `replay-full` keeps the whole tuned set, for a production build that
    # would rather spend the compile time than re-tune when the workload shifts.
    if variant_set == "replay-slim":
        if winners is None:
            raise CatalogError(
                "replay-slim needs the winners from a tuning run; pass "
                "--winners <db>.measurements.jsonl")
        kept = [c for c in candidates
                # Native wrappers always survive. They are not an optimisation:
                # a replay miss falls back to native (standards 9.2), so a build
                # without them has nothing to serve a signature the cache does
                # not cover, and the schema rejects a family that lacks one.
                if c.source_class == "native_wrapper"
                or c.stable_name in winners]
        unknown = winners - {c.stable_name for c in candidates}
        if unknown:
            raise CatalogError(
                f"{len(unknown)} winner(s) are not in the generated catalog, "
                f"e.g. {sorted(unknown)[0]}. The measurements came from a "
                f"different inventory or architecture set than this build.")
        candidates = kept

    manifest = {
        "artifact_version": ARTIFACT_VERSION,
        "generated_at": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "variant_set": variant_set,
        "source_revision": source_revision,
        "architectures": sorted(architectures),
        "signature_schema_version": 1,
        "hardware_schema_version": 1,
        "candidates": sorted((c.to_dict() for c in candidates),
                             key=lambda c: (c["family"], c["stable_name"])),
    }
    manifest["coverage"] = candidate_coverage(
        manifest["candidates"], inventory, variant_set)
    manifest["supported_coverage"] = supported_candidate_coverage(
        [c.to_dict() for c in supported_candidates], inventory)
    manifest["summary"] = schema.validate_and_summarise(manifest)
    manifest["manifest_hash"] = manifest_hash(manifest)
    manifest["build_descriptor"] = build_descriptor(manifest)
    return manifest


def candidate_coverage(candidates: list[dict[str, Any]],
                       inventory: Inventory,
                       variant_set: str) -> dict[str, Any]:
    """Describe observed workload coverage separately from catalog support.

    A workload-derived profile may intentionally contain no alternative for an
    observed type. That is useful evidence, not an implicit claim of complete
    support, so the report records native and alternative counts explicitly.
    """
    observed = sorted(inventory.quant_types | inventory.float_types)
    rows: dict[str, dict[str, Any]] = {}
    native_candidates = [c for c in candidates
                         if c.get("source_class") == "native_wrapper"]
    for type_name in observed:
        alternatives = [c for c in candidates
                        if c.get("source_class") != "native_wrapper"
                        and c.get("config", {}).get("type") == type_name]
        # Native wrappers are intentionally type-agnostic: their runtime
        # selector handles every supported type, so they have no config.type.
        native = native_candidates
        families = sorted({c["family"] for c in alternatives})
        row: dict[str, Any] = {
            "observed": True,
            "candidate_count": len(native) + len(alternatives),
            "native_count": len(native),
            "alternative_count": len(alternatives),
            "alternative_families": families,
        }
        if not alternatives:
            row["zero_alternative_reason"] = (
                "inventory profile contains native wrappers only"
                if variant_set == "inventory"
                else "no supported alternative was generated for this type")
        rows[type_name] = row
    return {
        "schema_version": 1,
        "variant_set": variant_set,
        "observed_types": observed,
        "by_type": rows,
    }


def supported_candidate_coverage(candidates: list[dict[str, Any]],
                                 inventory: Inventory) -> dict[str, Any]:
    """Report supported candidates separately from observed workload types."""
    observed = sorted(inventory.quant_types | inventory.float_types)
    native_candidates = [c for c in candidates
                         if c.get("source_class") == "native_wrapper"]
    alternatives = [c for c in candidates
                    if c.get("source_class") != "native_wrapper"
                    and c.get("config", {}).get("type")]
    supported_types = sorted({c["config"]["type"] for c in alternatives})
    report_types = sorted(set(observed) | set(supported_types))
    rows: dict[str, dict[str, Any]] = {}

    for type_name in report_types:
        type_alternatives = [c for c in alternatives
                             if c.get("config", {}).get("type") == type_name]
        type_candidates = native_candidates + type_alternatives
        architectures = sorted({arch for c in type_candidates
                                 for arch in c.get("architectures", [])})
        by_architecture = {
            arch: {
                "native_count": sum(
                    arch in c.get("architectures", [])
                    for c in native_candidates),
                "alternative_count": sum(
                    arch in c.get("architectures", [])
                    for c in type_alternatives),
            }
            for arch in architectures
        }
        is_supported = bool(type_alternatives)
        row: dict[str, Any] = {
            "observed": type_name in observed,
            "supported": is_supported,
            "candidate_count": len(type_candidates),
            "native_count": len(native_candidates),
            "alternative_count": len(type_alternatives),
            "alternative_families": sorted({c["family"] for c in type_alternatives}),
            "architectures": architectures,
            "by_architecture": by_architecture,
        }
        if not is_supported:
            row["zero_alternative_reason"] = (
                "type is not present in the supported candidate universe")
        rows[type_name] = row

    return {
        "schema_version": 1,
        "observed_types": observed,
        "supported_types": supported_types,
        "by_type": rows,
    }


def build_descriptor(manifest: dict[str, Any]) -> dict[str, Any]:
    """The configured/generated/compiled catalog identity (HI43/RV30).

    Deliberately timestamp-free and canonical, so the same object can be
    written beside the manifest, embedded in the generated build header, and
    later re-derived by a release probe scanning a compiled binary for it
    (release_validate.py) -- all three must agree byte for byte.
    """
    summary = manifest["summary"]
    descriptor: dict[str, Any] = {
        "schema_version": 1,
        "source_revision": manifest["source_revision"],
        "variant_set": manifest["variant_set"],
        "manifest_hash": manifest["manifest_hash"],
        "candidate_count": summary["total"],
        "by_family": summary["by_family"],
        "by_source_class": summary["by_source_class"],
        "target_architectures": manifest["architectures"],
    }
    canonical = json.dumps(descriptor, sort_keys=True, separators=(",", ":"))
    descriptor["descriptor_hash"] = hashlib.blake2b(
        canonical.encode("utf-8"), digest_size=16).hexdigest()
    validate_profile_descriptor(descriptor, manifest["variant_set"])
    return descriptor


def validate_profile_descriptor(descriptor: dict[str, Any], expected_profile: str) -> None:
    """Fail when a structurally present catalog cannot satisfy its profile."""
    if descriptor.get("schema_version") != 1:
        raise CatalogError("build descriptor schema is not version 1")
    if descriptor.get("variant_set") != expected_profile:
        raise CatalogError(
            f"generated profile {descriptor.get('variant_set')!r} does not "
            f"match requested profile {expected_profile!r}")
    count = descriptor.get("candidate_count", 0)
    by_family = descriptor.get("by_family", {})
    by_source = descriptor.get("by_source_class", {})
    native_count = by_source.get("native_wrapper", 0)
    if set(by_family) != set(schema.FAMILIES) or native_count != len(schema.FAMILIES):
        raise CatalogError("catalog must contain exactly one native wrapper per family")
    if expected_profile == "inventory":
        if count != len(schema.FAMILIES) or set(by_source) != {"native_wrapper"}:
            raise CatalogError("inventory profile must contain only five native wrappers")
    elif expected_profile in ("workload-max", "full-max", "replay-full"):
        if count <= len(schema.FAMILIES):
            raise CatalogError(
                f"{expected_profile} profile has only native wrappers; "
                "generation did not add any tunable candidates")
    elif expected_profile == "replay-slim" and count < len(schema.FAMILIES):
        raise CatalogError("replay-slim must retain every native family fallback")


def manifest_hash(manifest: dict[str, Any]) -> str:
    """Deterministic hash of the candidate set.

    Computed over the candidates alone, deliberately excluding the timestamp
    and the summary. Regenerating an unchanged catalog must produce an unchanged
    hash, or every rebuild would open a new build namespace and orphan the
    previous measurements (standards 13.1).
    """
    payload = {
        "artifact_version": manifest["artifact_version"],
        "variant_set": manifest["variant_set"],
        "architectures": manifest["architectures"],
        "signature_schema_version": manifest["signature_schema_version"],
        "hardware_schema_version": manifest["hardware_schema_version"],
        "candidates": manifest["candidates"],
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.blake2b(canonical.encode("utf-8"), digest_size=16).hexdigest()


# ------------------------------------------------------------------- emission

_FAMILY_ENUM = {
    "mmvq": "GGML_HIP_FAMILY_MMVQ",
    "mmq":  "GGML_HIP_FAMILY_MMQ",
    "mmvf": "GGML_HIP_FAMILY_MMVF",
    "mmf":  "GGML_HIP_FAMILY_MMF",
    "blas": "GGML_HIP_FAMILY_BLAS",
}

_SOURCE_ENUM = {
    "native_wrapper":        "GGML_HIP_SOURCE_NATIVE_WRAPPER",
    "existing_runtime":      "GGML_HIP_SOURCE_EXISTING_RUNTIME",
    "existing_alternative":  "GGML_HIP_SOURCE_EXISTING_ALTERNATIVE",
    "new_generated_variant": "GGML_HIP_SOURCE_NEW_GENERATED_VARIANT",
    "vendor_auto":           "GGML_HIP_SOURCE_VENDOR_AUTO",
    "vendor_explicit":       "GGML_HIP_SOURCE_VENDOR_EXPLICIT",
}

GENERATED_BANNER = (
    "// Generated by bigcherry autotune_catalog.py. Do not edit.\n"
    "// Regenerate with: python -m bigcherry generate\n")


def variant_fields(candidate: dict[str, Any]) -> dict[str, Any]:
    """The numeric contents of ggml_hip_variant_params for one candidate.

    Split out from `_variant_initialiser` so the replay-cache writer packs the
    same numbers the registry emits. Anything that reproduces this mapping
    independently is a copy that can silently disagree, and a disagreement here
    is a winner replayed as a different kernel -- see review RV09.

    `src0_type` is returned as the enumerator name, since that is what the C
    source wants; the cache writer resolves it to its numeric value.
    """
    text = _variant_initialiser(candidate)
    inner = text.strip().lstrip("{").rstrip("}").strip()
    parts = [p.strip() for p in inner.split(",")]
    return {
        "primary":   int(parts[0]),
        "secondary": int(parts[1]),
        "width":     int(parts[2]),
        "acc_f16":   int(parts[3]),
        "fallback":  int(parts[4]),
        "small_k":   int(parts[5]),
        "src0_type": parts[6],
    }


def _variant_initialiser(candidate: dict[str, Any]) -> str:
    """Pack a candidate's config into ggml_hip_variant_params.

    Each family reads only the fields it owns; the rest stay zero. Keeping the
    mapping in one place is what stops the registry and the launchers from
    disagreeing about which slot holds J.
    """
    config = candidate["config"]
    family = candidate["family"]
    primary = secondary = width = 0
    acc_f16 = fallback = small_k = 0
    src0_type = "0"

    # A native wrapper has no fixed variant: upstream's policy computes one at
    # launch time from the actual shape. Leaving the params zero is what marks
    # it as "ask the native selector", not "use J=0".
    if candidate["source_class"] == "native_wrapper":
        return "{ 0, 0, 0, 0, 0, 0, 0 }"

    # Each matmul family names itself for one src0 type and every family's
    # launch path dispatches on the runtime type, so eligibility needs this for
    # all four -- see ggml_hip_variant_params. Emitted as the enumerator rather
    # than its value so a renumbering upstream cannot silently repoint every
    # candidate at a different type.
    #
    # BLAS is the exception and stays 0: it has no per-type identity, it serves
    # whatever the capability predicate admits, and HI17 has yet to decompose
    # it. ggml_hip_blas_can_execute reads no variant field at all.
    type_name = config.get("type")
    src0_type = f"GGML_TYPE_{type_name.upper()}" if type_name else "0"

    if family == "mmq":
        primary = config["j"]
        fallback = int(config["fallback"])
    elif family == "mmvf":
        primary = config["block_size"]
        width = config["width"]
        acc_f16 = int(config["accumulator"] == "f16")
    elif family == "mmf":
        primary = config["nwarps"]
        width = config["width"]
    elif family == "mmvq":
        primary = config["nwarps"]
        secondary = config["rows_per_block"]
        width = config["width"]
        # small_k is a template argument of the compiled instance, so it has to
        # reach the launcher. It was in the stable name from the start and
        # dropped here, which meant the registry could name an instance it had
        # no way to ask for.
        small_k = int(config.get("small_k", False))

    return (f"{{ {primary}, {secondary}, {width}, "
            f"{acc_f16}, {fallback}, {small_k}, {src0_type} }}")


def render_registry(manifest: dict[str, Any]) -> str:
    """Emit the compiled registry table.

    Function pointers are per *family*, not per candidate: a candidate differs
    from its siblings only in ``variant``, which the family launcher reads. That
    keeps this file a flat data table -- thousands of rows compile quickly,
    where thousands of generated thunks would not.
    """
    lines = [
        GENERATED_BANNER,
        "//",
        "// Included once by hip-autotune-dispatch.cu, inside GGML_USE_HIP.",
        "// Every referenced *_can_execute / *_launch / *_workspace symbol is",
        "// declared in hip-autotune-dispatch.cuh.",
        "",
        f"#define GGML_HIP_AUTOTUNE_CANDIDATE_COUNT {len(manifest['candidates'])}",
        "",
        "static const ggml_hip_candidate_descriptor "
        "ggml_hip_candidate_registry[] = {",
    ]

    for index, candidate in enumerate(manifest["candidates"]):
        family = candidate["family"]
        prefix = f"ggml_hip_{family}"
        is_native = int(candidate["source_class"] == "native_wrapper")
        lines.append(
            f"    {{ {index}u, \"{candidate['stable_name']}\","
            f" {_FAMILY_ENUM[family]}, {_SOURCE_ENUM[candidate['source_class']]},"
            f" {candidate['implementation_version']},")
        lines.append(
            f"      UINT64_C({candidate['architecture_mask']}),"
            f" {int(candidate['graph_safe'])}, {int(candidate['deterministic'])},"
            f" {is_native}, 0,")
        lines.append(f"      {_variant_initialiser(candidate)},")
        lines.append(
            f"      {prefix}_can_execute, {prefix}_launch, {prefix}_workspace }},")

    lines += ["};", ""]
    return "\n".join(lines)


def render_arch_header() -> str:
    """Emit the C++ architecture enumeration and the cc -> code mapping.

    Generated rather than hand-written because the codes are bit positions in a
    persisted mask: if the Python list and a hand-maintained C++ enum ever
    disagreed, candidates would silently claim support for hardware they were
    never measured on. One source, two outputs (standards 2.5).
    """
    lines = [
        GENERATED_BANNER,
        "",
        "#pragma once",
        "",
        "// Standards 2.6: the enumerator value is the bit position in",
        "// architecture_mask. APPEND ONLY -- edit ARCHITECTURES in",
        "// tools/bigcherry/autotune_schema.py, never this file.",
        "",
        "enum ggml_hip_architecture_code {",
    ]
    for code, arch in enumerate(schema.ARCHITECTURES):
        name = "UNKNOWN" if arch == "unknown" else arch.upper()
        family = schema.ARCHITECTURE_FAMILY.get(arch)
        comment = f" // {family}" if family else " // matches nothing"
        lines.append(f"    GGML_HIP_ARCH_{name:<8} = {code},{comment}")
    lines += [
        f"    GGML_HIP_ARCH_COUNT      = {len(schema.ARCHITECTURES)},",
        "};",
        "",
        "// Maps a gfx identifier (the low bits of an AMD `cc`) to its code.",
        "// Anything unlisted returns UNKNOWN, which matches no candidate and so",
        "// falls back to native selection.",
        "static inline int ggml_hip_arch_code_from_gfx(int gfx) {",
        "    switch (gfx) {",
    ]
    for code, arch in enumerate(schema.ARCHITECTURES):
        if arch == "unknown":
            continue
        gfx = arch.removeprefix("gfx")
        lines.append(f"        case 0x{int(gfx, 16):04x}: return {code}; // {arch}")
    lines += [
        "        default:    return 0;",
        "    }",
        "}",
        "",
    ]
    return "\n".join(lines)


def render_build_hash(manifest: dict[str, Any]) -> str:
    descriptor = manifest["build_descriptor"]
    descriptor_json = json.dumps(descriptor, sort_keys=True, separators=(",", ":"))
    return "\n".join([
        GENERATED_BANNER,
        "",
        "#pragma once",
        "",
        "// Standards 13.1: the build namespace is (source revision, manifest",
        "// hash, ABI versions). A changed upstream selector or candidate set",
        "// must open a new namespace rather than silently reuse old winners.",
        "",
        f"#define GGML_HIP_AUTOTUNE_MANIFEST_HASH_STR   \"{manifest['manifest_hash']}\"",
        f"#define GGML_HIP_AUTOTUNE_SOURCE_REVISION_STR \"{manifest['source_revision']}\"",
        f"#define GGML_HIP_AUTOTUNE_VARIANT_SET_STR     \"{manifest['variant_set']}\"",
        f"#define GGML_HIP_AUTOTUNE_ARTIFACT_VERSION    {manifest['artifact_version']}",
        f"#define GGML_HIP_AUTOTUNE_CANDIDATE_COUNT     {descriptor['candidate_count']}",
        # HI46: release_validate.py's binary scanner looks for both of these
        # byte sequences in the compiled binary to prove the configured,
        # generated and compiled catalog identity all agree.
        f"#define GGML_HIP_AUTOTUNE_DESCRIPTOR_HASH_STR \"{descriptor['descriptor_hash']}\"",
        f"#define GGML_HIP_AUTOTUNE_DESCRIPTOR_JSON     {json.dumps(descriptor_json)}",
        "",
    ])


def render_mmvq_instances(manifest: dict[str, Any]) -> str:
    """Emit every MMVQ geometry instance into one include.

    The instances live in mmvq.cu's translation unit rather than one file each,
    because the templates they instantiate -- mul_mat_vec_q_switch_fusion and
    calc_launch_params -- are `static` inside mmvq.cu and therefore invisible to
    any other unit. MMQ and MMF put their equivalents in headers, so their
    instances can be separate files; MMVQ has no such split upstream, and
    creating one would mean relocating several hundred lines of upstream code
    on every release.

    The cost is that these compile serially in a single object file. That is
    acceptable for `workload-max`, which emits tens to low hundreds of
    instances, and merely slow for `full-max`, which is a compile audit that is
    expected to be slow. If it becomes the bottleneck, the alternative is to
    move the templates into a header -- but that is a large, churn-prone diff
    and should wait for evidence.
    """
    rows = [c for c in manifest["candidates"]
            if c["family"] == "mmvq"
            and c["source_class"] == "new_generated_variant"]

    lines = [
        GENERATED_BANNER,
        "//",
        "// Included once by mmvq.cu, inside GGML_HIP_DISPATCH, after",
        "// ggml_hip_mmvq_launch_instance and before the switch functions that",
        "// call the lookup below.",
        "",
    ]
    if not rows:
        lines += ["// No explicit MMVQ geometry candidates in this variant set.",
                  ""]
    for candidate in rows:
        config = candidate["config"]
        lines.append(f"// {candidate['stable_name']}")
        lines.append(
            f"DECL_MMVQ_AUTOTUNE_CASE_SK(GGML_TYPE_{config['type'].upper()}, "
            f"{config['width']}, {config['nwarps']}, "
            f"{config['rows_per_block']}, {int(config.get('small_k', False))})")
    lines.append("")

    # A lookup rather than a table: the key space is sparse and compile-time
    # constant, so a switch lets the compiler do the work and keeps the
    # generated file readable.
    #
    # Emitted even when there are no instances -- mmvq.cu calls it
    # unconditionally, and a profile with native candidates only (`inventory`)
    # would otherwise fail to compile. An empty body returning nullptr is the
    # correct answer for that build: nothing is compiled, so nothing resolves.
    lines += [
        "// Resolve a geometry to its compiled instance. Returns nullptr when",
        "// this build carries no instance for the request. The caller decides",
        "// what that means -- measured dispatch treats it as fatal, because a",
        "// silent fall back to native would time the native geometry under an",
        "// explicit name.",
        "static ggml_hip_mmvq_instance_fn ggml_hip_mmvq_find_instance(",
        "        ggml_type type, int width, int nwarps, int rows_per_block,",
        "        bool small_k) {",
    ]
    if not rows:
        lines += ["    GGML_UNUSED(type);",
                  "    GGML_UNUSED(width);",
                  "    GGML_UNUSED(nwarps);",
                  "    GGML_UNUSED(rows_per_block);",
                  "    GGML_UNUSED(small_k);"]
    for candidate in rows:
        config = candidate["config"]
        small_k = int(config.get("small_k", False))
        name = (f"ggml_hip_mmvq_instance_GGML_TYPE_{config['type'].upper()}"
                f"_w{config['width']}_nw{config['nwarps']}"
                f"_r{config['rows_per_block']}_sk{small_k}")
        lines.append(
            f"    if (type == GGML_TYPE_{config['type'].upper()} && "
            f"width == {config['width']} && nwarps == {config['nwarps']} &&")
        lines.append(
            f"            rows_per_block == {config['rows_per_block']} && "
            f"small_k == {'true' if small_k else 'false'}) {{")
        lines.append(f"        return {name};")
        lines.append("    }")
    lines += ["    return nullptr;", "}", ""]
    return "\n".join(lines)


def mmvq_instance_filename(candidate: dict[str, Any]) -> str:
    config = candidate["config"]
    return (f"mmvq-autotune-instance-{config['type']}"
            f"-w{config['width']}-nw{config['nwarps']}"
            f"-r{config['rows_per_block']}.cu")


def render_mmvq_instance(candidate: dict[str, Any]) -> str:
    config = candidate["config"]
    return "\n".join([
        GENERATED_BANNER,
        f"// Candidate: {candidate['stable_name']}",
        "",
        "#include \"../mmvq-autotune.cuh\"",
        "",
        f"DECL_MMVQ_AUTOTUNE_CASE(GGML_TYPE_{config['type'].upper()}, "
        f"{config['width']}, {config['nwarps']}, {config['rows_per_block']});",
        "",
    ])


@dataclass
class EmitResult:
    manifest_path: Path
    registry_path: Path
    build_hash_path: Path
    arch_header_path: Path
    descriptor_path: Path
    instance_paths: list[Path]
    removed_paths: list[Path]


def emit(manifest: dict[str, Any], root: Path, artifacts: Path) -> EmitResult:
    """Write every artifact.

    Generated files that are no longer in the manifest are deleted. A shrinking
    catalog must not leave orphan translation units behind: they would still
    compile, still register, and still be launchable -- candidates with no
    catalog record, which is exactly what standards 2.5 forbids.
    """
    cuda = paths.cuda_dir(root)
    instances = paths.template_instances_dir(root)
    artifacts.mkdir(parents=True, exist_ok=True)
    instances.mkdir(parents=True, exist_ok=True)

    manifest_text = json.dumps(manifest, indent=2) + "\n"

    manifest_path = artifacts / "hip-autotune-manifest.json"
    manifest_path.write_text(manifest_text, encoding="utf-8", newline="")

    # Beside the manifest, a plain record of what each MMQ config table
    # defines. Diffing this between releases shows which solutions upstream
    # added or withdrew, which is otherwise buried in a 3000-candidate manifest.
    (artifacts / "mmq-table-coverage.json").write_text(
        json.dumps(mmq_table_coverage(cuda), indent=2) + "\n",
        encoding="utf-8", newline="")

    registry_path = cuda / "hip-autotune-registry.inc"
    registry_path.write_text(render_registry(manifest), encoding="utf-8",
                             newline="")

    build_hash_path = cuda / "hip-autotune-build-hash.h"
    build_hash_path.write_text(render_build_hash(manifest), encoding="utf-8",
                               newline="")

    arch_header_path = cuda / "hip-autotune-arch.h"
    arch_header_path.write_text(render_arch_header(), encoding="utf-8",
                                newline="")

    # The manifest is also copied into the tree so a build carries the catalog
    # it was generated from, not merely its hash.
    (cuda / "hip-autotune-manifest.json").write_text(
        manifest_text, encoding="utf-8", newline="")

    # Same object embedded in the build header (render_build_hash) above --
    # written out separately, beside the manifest and into the tree, so a
    # release probe (release_validate.py) can compare a compiled binary's
    # embedded copy against the one this exact generation run produced.
    descriptor_text = json.dumps(
        manifest["build_descriptor"], sort_keys=True, indent=2) + "\n"
    descriptor_path = artifacts / "hip-autotune-build-descriptor.json"
    descriptor_path.write_text(descriptor_text, encoding="utf-8", newline="")
    (cuda / "hip-autotune-build-descriptor.json").write_text(
        descriptor_text, encoding="utf-8", newline="")

    # All MMVQ geometry instances go into one include, compiled inside
    # mmvq.cu's translation unit -- see render_mmvq_instances for why.
    instances_path = cuda / "hip-autotune-mmvq-instances.inc"
    instances_path.write_text(render_mmvq_instances(manifest), encoding="utf-8",
                              newline="")

    # Earlier revisions emitted one .cu per instance. Remove any left behind, or
    # they would still compile, still register, and still be launchable --
    # candidates with no catalog record, which standards 2.5 forbids.
    removed: list[Path] = []
    for stale in sorted(instances.glob("mmvq-autotune-instance-*.cu")):
        stale.unlink()
        removed.append(stale)

    written = [instances_path]

    return EmitResult(manifest_path, registry_path, build_hash_path,
                      arch_header_path, descriptor_path, written, removed)


# ----------------------------------------------------------------------- main

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="bigcherry generate",
        description="Generate the candidate catalog and all downstream artifacts.")
    parser.add_argument("--llama-root", default=None)
    parser.add_argument("--variant-set", default="inventory",
                        choices=schema.VARIANT_SETS)
    parser.add_argument("--arch", default="all",
                        help="comma-separated architectures or group names "
                             f"({', '.join(sorted(schema.ARCHITECTURE_GROUPS))})")
    parser.add_argument("--inventory", default=None,
                        help="inventory JSON from a record-mode run "
                             "(required for workload-max)")
    parser.add_argument("--winners", default=None,
                        help="measurements JSONL from a tuning run "
                             "(required for replay-slim)")
    parser.add_argument("--resource-report", action="append", default=[],
                        help="validated compiler resource report; may be repeated")
    parser.add_argument("--dry-run", action="store_true",
                        help="build and validate the manifest, write nothing")
    args = parser.parse_args(argv)

    root = paths.llama_root(args.llama_root)
    try:
        architectures = schema.resolve_architectures(args.arch)
    except schema.SchemaError as exc:
        print(f"bad --arch: {exc}", file=sys.stderr)
        return 2
    if not architectures:
        print("--arch resolved to no architectures", file=sys.stderr)
        return 2

    inventory = None
    if args.inventory:
        inventory = Inventory.from_json(Path(args.inventory))
    elif args.variant_set in ("workload-max", "replay-full", "replay-slim"):
        print(f"{args.variant_set} requires --inventory: without an observed "
              "type and path set there is nothing to restrict the catalog to.",
              file=sys.stderr)
        return 2

    winners = None
    if args.winners:
        winners = read_winners(Path(args.winners))
        if not winners:
            print(f"no winners found in {args.winners}", file=sys.stderr)
            return 2
    elif args.variant_set == "replay-slim":
        print("replay-slim requires --winners: a slim build is defined by the "
              "variants a tuning run chose, and there is no way to guess them.",
              file=sys.stderr)
        return 2

    revision, _ = git_revision(root, check_dirty=False)

    try:
        resource_blacklist: dict[tuple[str, str], tuple[str, ...]] = {}
        for report_path in args.resource_report:
            resource_blacklist.update(load_blacklist(Path(report_path)))
        manifest = build_manifest(
            root, variant_set=args.variant_set, architectures=architectures,
            inventory=inventory, source_revision=revision, winners=winners,
            resource_blacklist=resource_blacklist)
    except (CatalogError, ResourceError, schema.SchemaError) as exc:
        print(f"catalog generation failed: {exc}", file=sys.stderr)
        return 1

    summary = manifest["summary"]
    print(f"variant set: {manifest['variant_set']}")
    print(f"  architectures: {len(manifest['architectures'])} "
          f"({manifest['architectures'][0]}..{manifest['architectures'][-1]})")
    print(f"  candidates:    {summary['total']}")
    for family, count in summary["by_family"].items():
        print(f"    {family:<6} {count}")
    print("  by source class:")
    for source, count in summary["by_source_class"].items():
        print(f"    {source:<24} {count}")
    print(f"  manifest hash: {manifest['manifest_hash']}")

    if args.dry_run:
        print("  (dry run -- nothing written)")
        return 0

    result = emit(manifest, root, paths.artifact_dir(revision))
    print(f"  manifest:   {result.manifest_path}")
    print(f"  registry:   {result.registry_path}")
    print(f"  build hash: {result.build_hash_path}")
    print(f"  arch enum:  {result.arch_header_path}")
    print(f"  descriptor: {result.descriptor_path}")
    print(f"  MMVQ instances: {len(result.instance_paths)} written, "
          f"{len(result.removed_paths)} removed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
