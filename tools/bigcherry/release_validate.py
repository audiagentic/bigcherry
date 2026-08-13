"""Isolated upstream patch-compatibility probe (HI46).

Clones a candidate ref into its own throwaway checkout and runs it through
the current recipe/build pipeline (`bigcherry pull` + `bigcherry build`)
without touching the pinned checkout or the shared `artifacts/<revision>/`
tree -- a probe against `master` (ahead of the pin) must never be mistaken
for a real release's candidate set. Answers "do the patches still apply and
build cleanly against this ref", nothing more: full release validation (the
record -> tune -> promote -> replay -> coverage gate sequence) is separate,
larger work.

The old (pre-reset) version of this file drove builds itself via a hardcoded
PROFILES dict of raw cmake command strings. That duplicated what recipes.py
now owns -- which build options a platform needs -- and would drift from it
silently. This version has no build knowledge of its own: it shells out to
the same `bigcherry pull`/`bigcherry build` a human would run, so a probe and
a real build can never disagree about what "building this recipe" means.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

from . import paths
from .autotune_schema import VARIANT_SETS
from .multi_gpu_validate import validate_multi_gpu_claim
from .device_state_validate import validate_device_state_report


class ReleaseGateError(ValueError):
    """A validated-release claim is missing or contradicts its evidence."""


def _architecture_report(record: dict[str, Any]) -> tuple[dict[str, Any], set[str]]:
    """Return architecture evidence and the architectures required by a claim.

    New records use an explicit report with ``required``, ``observed``,
    ``validated`` and ``by_architecture`` fields.  The older flat mapping is
    still accepted for inventory diagnostics, where it is evidence about the
    observed machine rather than an optimized release-support claim.
    """
    raw = record.get("architecture_coverage")
    if not isinstance(raw, dict) or not raw:
        raise ReleaseGateError(
            "validated release claim lacks architecture_coverage evidence")

    if "by_architecture" in raw:
        by_architecture = raw.get("by_architecture")
        required = raw.get("required")
        observed = raw.get("observed")
        validated = raw.get("validated")
        for name, value in (("required", required), ("observed", observed),
                            ("validated", validated)):
            if not isinstance(value, list) or any(
                not isinstance(item, str) or not item.strip() for item in value
            ) or len(set(value)) != len(value):
                raise ReleaseGateError(
                    f"architecture_coverage.{name} must be a unique list")
        if not isinstance(by_architecture, dict) or not by_architecture:
            raise ReleaseGateError(
                "architecture_coverage.by_architecture must be a non-empty object")
        declared_required = record.get("required_architectures", required)
        if (not isinstance(declared_required, list)
                or set(required) != set(declared_required)):
            raise ReleaseGateError(
                "architecture_coverage required set disagrees with claim")
        observed_by_entry = {
            name for name, evidence in by_architecture.items()
            if isinstance(evidence, dict) and evidence.get("observed") is True
        }
        validated_by_entry = {
            name for name, evidence in by_architecture.items()
            if isinstance(evidence, dict) and evidence.get("validated") is True
        }
        if (set(observed) != observed_by_entry
                or set(validated) != validated_by_entry):
            raise ReleaseGateError(
                "architecture_coverage observed/validated set disagrees with evidence")
        return by_architecture, set(required)

    # Legacy flat map: retain this path for inventory-only diagnostic claims.
    expected = record.get("required_architectures", record.get("architectures"))
    if expected is None:
        expected = list(raw)
    if (not isinstance(expected, list) or not expected
            or any(not isinstance(item, str) or not item.strip() for item in expected)):
        raise ReleaseGateError(
            "architecture_coverage lacks required architecture reporting")
    return raw, set(expected)


def validate_release_claim(record: dict[str, Any]) -> None:
    """Fail closed when a record claims validation without coverage evidence.

    ``probe`` records compatibility only and therefore does not need hardware
    evidence.  A separate producer may add ``claim: validated`` once the full
    record/tune/promote/replay pipeline has run.  At that boundary both
    architecture coverage and candidate coverage are mandatory and must agree
    with the identities they describe.
    """
    validate_multi_gpu_claim(record)
    if any(key in record for key in ("device_state_pre", "device_state_post", "device_clock_drift")):
        validate_device_state_report(record)
    if record.get("claim") != "validated" and record.get("stage") != "validated":
        return

    candidates = record.get("candidate_coverage")
    if not isinstance(candidates, dict):
        raise ReleaseGateError(
            "validated release claim lacks candidate_coverage evidence")

    architecture, required_architectures = _architecture_report(record)
    variant_set = candidates.get("variant_set")
    if variant_set == "inventory" and "by_architecture" not in record["architecture_coverage"]:
        # Inventory is diagnostic evidence.  A legacy flat map may describe
        # what was observed without pretending that every target was tested.
        required_architectures = set()

    expected_architectures = record.get("architectures")
    if expected_architectures is not None:
        if (not isinstance(expected_architectures, list)
                or not expected_architectures
                or (variant_set != "inventory"
                    and set(expected_architectures) != required_architectures)):
            raise ReleaseGateError(
                "architecture_coverage required set does not match declared architectures")

    for arch, evidence in architecture.items():
        if not isinstance(arch, str) or not arch.strip():
            raise ReleaseGateError("architecture coverage contains an invalid key")
        if not isinstance(evidence, dict):
            raise ReleaseGateError(f"architecture coverage for {arch!r} is invalid")
        validated = evidence.get("validated", evidence.get("status") == "validated")
        observed = evidence.get(
            "observed", evidence.get("status") in {"observed", "validated"})
        if not isinstance(observed, bool) or not isinstance(validated, bool):
            raise ReleaseGateError(
                f"architecture coverage for {arch!r} lacks boolean observed/validated fields")
        if validated and not observed:
            raise ReleaseGateError(
                f"architecture coverage for {arch!r} is validated without observation")
        if arch in required_architectures and (not observed or not validated):
            raise ReleaseGateError(
                f"required architecture {arch!r} lacks observed and validated evidence")
        if evidence.get("candidate_coverage") is not True:
            raise ReleaseGateError(
                f"architecture coverage for {arch!r} lacks candidate coverage")

    missing = required_architectures - set(architecture)
    if missing:
        raise ReleaseGateError(
            f"required architecture coverage is missing: {sorted(missing)}")

    # Optimized claims must use the explicit report so a partial flat map
    # cannot accidentally be interpreted as complete architecture support.
    if variant_set != "inventory" and "by_architecture" not in record["architecture_coverage"]:
        raise ReleaseGateError(
            "optimized release claim requires explicit per-architecture coverage report")

    observed_types = candidates.get("observed_types")
    by_type = candidates.get("by_type")
    variant_set = candidates.get("variant_set")
    if variant_set not in VARIANT_SETS:
        raise ReleaseGateError(
            "candidate coverage has an invalid or missing variant_set")
    if (not isinstance(observed_types, list) or not observed_types
            or not isinstance(by_type, dict)
            or set(observed_types) != set(by_type)):
        raise ReleaseGateError(
            "candidate coverage observed_types and by_type are inconsistent")
    for type_name, evidence in by_type.items():
        if not isinstance(type_name, str) or not type_name.strip():
            raise ReleaseGateError("candidate coverage contains an invalid type")
        if not isinstance(evidence, dict) or evidence.get("observed") is not True:
            raise ReleaseGateError(
                f"candidate coverage for {type_name!r} is not observed")
        count = evidence.get("candidate_count")
        if not isinstance(count, int) or isinstance(count, bool) or count < 1:
            raise ReleaseGateError(
                f"candidate coverage for {type_name!r} has no candidates")
        alternatives = evidence.get("alternative_count")
        if not isinstance(alternatives, int) or isinstance(alternatives, bool) \
                or alternatives < 0:
            raise ReleaseGateError(
                f"candidate coverage for {type_name!r} has an invalid "
                "alternative count")
        # Inventory is deliberately native-only: it is a diagnostic profile
        # used to establish observed signatures and must not be treated as an
        # optimization claim. Every optimized release profile, however, must
        # have at least one non-native candidate for every observed type.
        if variant_set != "inventory" and alternatives < 1:
            raise ReleaseGateError(
                f"optimized candidate coverage for {type_name!r} has zero "
                "alternatives")


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", value).strip(".-") or "upstream"


def _write(run: Path, record: dict[str, Any]) -> Path:
    validate_release_claim(record)
    record["finished_at"] = dt.datetime.now(dt.timezone.utc).isoformat()
    path = run / "run.json"
    path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _run_logged(command: list[str], *, cwd: Path, log: Path) -> bool:
    completed = subprocess.run(command, cwd=cwd, text=True, capture_output=True)
    log.write_text(
        f"$ {' '.join(command)}\n\n{completed.stdout}{completed.stderr}",
        encoding="utf-8",
    )
    return completed.returncode == 0


def _command_text(command: list[str]) -> str:
    return subprocess.list2cmdline(command)


def _failure_class(stage: str) -> str:
    return "patch-drift" if stage == "build" else f"{stage}-failed"


def probe(
    run_id: str,
    staging_root: Path,
    ref: str,
    recipe: str,
    inventory: Path | None = None,
) -> tuple[int, Path]:
    """Prove `ref` still audits, patches, and builds clean under `recipe`.

    Never touches the pinned checkout: `pull`/`build` are both told to use
    this run's own throwaway `--llama-root`, so a probe against an unrelated
    ref (typically `master`, ahead of the pin) cannot corrupt or be confused
    with the shared checkout or its `artifacts/<revision>/` tree -- the
    revision embedded in anything `build` generates for this checkout is the
    probed ref's own, which is exactly what makes it distinguishable from a
    real release's candidates rather than requiring a separate override flag.
    """
    run = staging_root / safe_name(run_id)
    checkout = run / "llama.cpp"
    if run.exists():
        raise FileExistsError(f"run already exists: {run}")
    run.mkdir(parents=True)
    record: dict[str, Any] = {
        "schema_version": 2, "run_id": run_id, "ref": ref, "recipe": recipe,
        "checkout": str(checkout), "source_revision": ref,
        "bigcherry_revision": "unknown",
    }
    try:
        record["bigcherry_revision"] = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=paths.REPO_ROOT,
            check=True, capture_output=True, text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        pass

    pull_command = [sys.executable, "-m", "bigcherry", "--llama-root",
                    str(checkout), "pull", "--ref", ref]
    pull_ok = _run_logged(
        pull_command,
        cwd=paths.REPO_ROOT, log=run / "pull.log",
    )
    record["pull"] = {"ok": pull_ok, "log": "pull.log", "stage": "pull",
                       "command": _command_text(pull_command),
                       "exit_code": 0 if pull_ok else 1}
    if not pull_ok:
        record["outcome"] = "pull-failed"
        record["failure"] = {"stage": "pull",
                              "command": _command_text(pull_command),
                              "exit_code": 1,
                              "failure_class": _failure_class("pull")}
        return 1, _write(run, record)

    build_command = [
        sys.executable, "-m", "bigcherry", "--llama-root", str(checkout),
        "build", "--recipe", recipe,
    ]
    if inventory is not None:
        build_command += ["--inventory", str(inventory)]
    build_ok = _run_logged(
        build_command,
        cwd=paths.REPO_ROOT, log=run / "build.log",
    )
    record["build"] = {"ok": build_ok, "log": "build.log", "stage": "build",
                        "command": _command_text(build_command),
                        "exit_code": 0 if build_ok else 1}
    record["outcome"] = "compatible" if build_ok else "patch-drift-or-build-failed"
    if not build_ok:
        record["failure"] = {"stage": "build",
                              "command": _command_text(build_command),
                              "exit_code": 1,
                              "failure_class": _failure_class("build")}
    return (0 if build_ok else 1), _write(run, record)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="bigcherry probe-release", description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--staging-root", default=str(paths.ARTIFACTS / "release-runs"))
    parser.add_argument("--ref", default="master")
    parser.add_argument("--recipe", default="bigcherry")
    parser.add_argument(
        "--inventory", default=None,
        help="record-mode inventory JSON for recipes whose build includes tuning",
    )
    args = parser.parse_args(argv)
    code, path = probe(
        args.run_id, Path(args.staging_root), args.ref, args.recipe,
        Path(args.inventory) if args.inventory else None,
    )
    print(f"record: {path}")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
