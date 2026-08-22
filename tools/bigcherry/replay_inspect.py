"""Offline cache/registry inspection through the C++ hip-autotune-inspect tool.

HI15/HI16 shared gap: a tool that judges the replay cache and the compiled
registry with the REAL loader, not a reimplementation that could disagree
with it. The C++ tool links ggml-hip and reports what the production loader
does; this module owns the JSON, runs the binary, and reconciles its report
against the build's manifest, so catalog<->registry agreement is checked
the way the project checks every cross-language identity: both sides,
independently derived, compared.

Python never parses the binary cache here. Anything about the cache file is
the loader's judgement, passed through unchanged.
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Any

from .autotune_catalog import CatalogError, build_descriptor

# hip-autotune-inspect exit codes (src/ggml/src/ggml-cuda/hip-autotune-inspect.cpp).
EXIT_OK = 0
EXIT_REGISTRY_ANOMALY = 1
EXIT_USAGE = 2
EXIT_CACHE_REJECTED = 3
EXIT_NO_USABLE_ENTRIES = 4

# Exit codes whose JSON report is still complete and useful.
_REPORTABLE = (
    EXIT_OK,
    EXIT_REGISTRY_ANOMALY,
    EXIT_CACHE_REJECTED,
    EXIT_NO_USABLE_ENTRIES,
)

_ENV_TOOL = "BIGCHERRY_INSPECT_TOOL"


def find_tool(explicit: str | None = None) -> Path:
    """Locate the hip-autotune-inspect binary.

    Precedence: explicit path, then $BIGCHERRY_INSPECT_TOOL, then the
    conventional campaign build locations ($BIGCHERRY_BUILD_DIR/bin and its
    two ancestors). The binary is a build artifact, not a repo file; a
    missing one is an actionable error with the exact expected locations.
    """
    candidates: list[Path] = []
    if explicit:
        candidates.append(Path(explicit))
    env_tool = os.environ.get(_ENV_TOOL)
    if env_tool:
        candidates.append(Path(env_tool))
    build_dir = os.environ.get("BIGCHERRY_BUILD_DIR")
    if build_dir:
        base = Path(build_dir)
        candidates += [
            base / "bin" / "hip-autotune-inspect",
            base / "ggml" / "src" / "ggml-hip" / "hip-autotune-inspect",
        ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    checked = "\n  ".join(str(c) for c in candidates) or "(no candidates given)"
    raise SystemExit(
        "hip-autotune-inspect binary not found; checked:\n  "
        f"{checked}\n"
        f"build a GGML_HIP_DISPATCH_REPLAY (or GGML_HIP_AUTOTUNE) campaign "
        f"build, or set {_ENV_TOOL}"
    )


def run_tool(
    tool: Path, cache: Path | None = None, interpreter: list[str] | None = None
) -> dict[str, Any]:
    """Run the C++ tool and parse its --json output.

    Registry, cache outcome, per-entry state and the build's embedded
    catalog identity come back exactly as the tool (i.e. the real loader
    and registry) produced them. The raw exit code is preserved under
    ``_exit``; a usage error (code 2) raises, because there is no report
    to reason about.

    ``interpreter`` prefixes the command (``["python3", script]``-style
    invocation); the tests use it to exercise the same parse/reconcile
    path against a scripted stand-in, which keeps the contract tests
    runnable on a machine with no ROCm build.
    """
    command = [*(interpreter or []), str(tool), "--json"]
    if cache is not None:
        command.append(str(cache))
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    if completed.returncode == EXIT_USAGE:
        raise SystemExit(
            f"{tool.name} usage error (exit {EXIT_USAGE}):\n"
            f"{completed.stdout}{completed.stderr}"
        )
    try:
        report = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise SystemExit(
            f"{tool.name} emitted unparseable JSON (exit "
            f"{completed.returncode}):\n{completed.stdout}"
        ) from exc
    if completed.returncode not in _REPORTABLE:
        raise SystemExit(
            f"{tool.name} exited {completed.returncode}:\n"
            f"{completed.stdout}{completed.stderr}"
        )
    report["_exit"] = completed.returncode
    return report


def manifest_agreement(report: dict[str, Any], manifest_path: Path) -> dict[str, Any]:
    """Catalog<->registry agreement (HI16), the Python half.

    The registry is compiled from the catalog, so the descriptor hash
    embedded in the binary equals the manifest's descriptor hash if and
    only if the compiled table is structurally this catalog. Recomputing
    the descriptor from the manifest's fields (rather than trusting the
    manifest's embedded copy) also catches a manifest that disagrees with
    its own summary.
    """
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    try:
        descriptor = build_descriptor(manifest)
    except CatalogError as exc:
        raise SystemExit(
            f"manifest {manifest_path} failed descriptor validation: {exc}"
        ) from exc
    embedded = manifest.get("build_descriptor")
    diffs: list[dict[str, Any]] = []
    build = report.get("build", {})
    for field in (
        "descriptor_hash",
        "manifest_hash",
        "source_revision",
        "variant_set",
        "candidate_count",
    ):
        actual = build.get(field)
        if actual != descriptor.get(field):
            diffs.append(
                {
                    "field": field,
                    "manifest": descriptor.get(field),
                    "binary": actual,
                }
            )
    # The registry's own per-family tallies, judged by the loader's build,
    # against the catalog's summary.
    registry = report.get("registry", {})
    family_diffs = {
        "manifest": descriptor["by_family"],
        "registry": registry.get("by_family", {}),
    }
    agreement: dict[str, Any] = {
        "agrees": not diffs,
        "diffs": diffs,
        "by_family": family_diffs,
        "by_family_agrees": family_diffs["manifest"] == family_diffs["registry"],
        "registry_anomalies": registry.get("anomalies", []),
        "manifest_embedded_descriptor_agrees": isinstance(embedded, dict)
        and embedded.get("descriptor_hash") == descriptor["descriptor_hash"],
    }
    return agreement


def format_report(report: dict[str, Any]) -> str:
    """The human rendering, mirroring the C++ tool's text mode."""
    lines = ["hip-autotune-inspect"]
    build = report.get("build", {})
    lines.append(
        f"  build   manifest {build.get('manifest_hash', '?')}  "
        f"source {str(build.get('source_revision', '?'))[:12]}  "
        f"variant {build.get('variant_set', '?')}"
    )
    registry = report.get("registry", {})
    if registry.get("anomalies"):
        lines.append(
            f"  registry {registry.get('count', '?')} candidate(s), "
            f"{len(registry['anomalies'])} anomaly/anomalies:"
        )
        for anomaly in registry["anomalies"]:
            lines.append(f"    - {anomaly}")
    else:
        families = " ".join(
            f"{family} {count}"
            for family, count in sorted(registry.get("by_family", {}).items())
        )
        lines.append(
            f"  registry {registry.get('count', '?')} candidate(s)  {families}  ok"
        )
    cache = report.get("cache")
    if cache is not None:
        lines.append(
            f"  cache   {cache.get('path', '?')} -> {cache.get('outcome', '?')}"
        )
        if cache.get("outcome") == "loaded":
            lines.append(
                f"          {cache.get('winner_slots', 0)} winner slot(s), "
                f"{cache.get('usable', 0)} usable on this build, "
                f"stale={str(bool(cache.get('stale'))).lower()}"
            )
            for entry in cache.get("entries", []):
                lines.append(
                    f"          gen {entry['generation']}  {entry['winner']}  "
                    f"fresh={str(entry['fresh']).lower()} "
                    f"registered={str(entry['registered']).lower()} "
                    f"stale_impl={str(entry['stale_impl']).lower()} "
                    f"match_kind={entry['match_kind']} "
                    f"transform={entry['transform_id']}  {entry['dispatch'][:16]}..."
                )
    agreement = report.get("manifest")
    if isinstance(agreement, dict):
        if (
            agreement["agrees"]
            and agreement["by_family_agrees"]
            and not agreement["registry_anomalies"]
        ):
            lines.append("  manifest  agrees with the compiled registry")
        else:
            lines.append("  manifest  DISAGREES with the compiled registry:")
            for diff in agreement["diffs"]:
                lines.append(
                    f"    - {diff['field']}: manifest={diff['manifest']} "
                    f"binary={diff['binary']}"
                )
            if not agreement["by_family_agrees"]:
                lines.append(
                    f"    - by_family: manifest={agreement['by_family']['manifest']} "
                    f"registry={agreement['by_family']['registry']}"
                )
            for anomaly in agreement["registry_anomalies"]:
                lines.append(f"    - registry anomaly: {anomaly}")
        if not agreement.get("manifest_embedded_descriptor_agrees", True):
            lines.append(
                "  manifest  its embedded build_descriptor does not "
                "match its own recomputed descriptor"
            )
    return "\n".join(lines)
