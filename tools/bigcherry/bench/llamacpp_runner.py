"""MRC06 launcher for an external llama.cpp bench/run_bench.py.

The runner remains in its own checkout. Intercept its append_result boundary
*before* it imports the writer; no new results.json/family JSON or SQLite sync
is allowed. Only the returned measured row is converted through the shared
MRC01 API. The native benchmark itself is not modified or reimplemented.
"""

from __future__ import annotations

import argparse
import importlib
import json
import runpy
import sys
from pathlib import Path

from ..experiment import attestation
from . import llamacpp_records


def _observed_from_measured_log(row: dict) -> attestation.ExecutionAttestation | None:
    name = row.get("log_file")
    if not isinstance(name, str) or not name:
        return None
    path = Path(name)
    if not path.is_file() or path.is_symlink():
        return None
    # VA25 derives execution identity solely from the benchmarked process's
    # own backend-initialisation evidence, never config or a rig label.
    log = path.read_text(encoding="utf-8", errors="replace")
    if row.get("bench_type") == "server-bench":
        return attestation.parse_llama_server_attestation(log)
    return attestation.parse_rocm_attestation(log)


def run_with_records(
    runner_root: Path,
    argv: list[str],
    *,
    expected: attestation.ExecutionIdentity,
    work_root: Path | None = None,
    binary_path: Path | None = None,
) -> list[str]:
    """Run a real run_bench.py with its historical result writes redirected.

    An external run_bench invoked directly still owns its own legacy writer;
    use this wrapper as the deliberate MRC06 production entry point.
    """
    runner_root = Path(runner_root).resolve()
    script = runner_root / "bench/run_bench.py"
    bench = script.parent
    if not script.is_file():
        raise ValueError(f"missing llama.cpp run_bench.py: {script}")
    path_backup, argv_backup = sys.path[:], sys.argv[:]
    changed: list[tuple[object, str, object]] = []
    produced: list[str] = []
    # A single canonical append_result call per measured row; family routing
    # is a second invocation of the same row, not another measurement.
    try:
        sys.path.insert(0, str(bench))
        sys.argv = [str(script), *argv]
        lib_results = importlib.import_module("lib.results")
        config = importlib.import_module("lib.config")
        # Prevent a preloaded similarly named library from some other checkout
        # silently taking this run's write interception.
        for module in (lib_results, config):
            source = getattr(module, "__file__", None)
            if not source or not Path(source).resolve().is_relative_to(bench):
                raise ValueError(
                    "external llama harness module belongs to a different checkout"
                )
        canonical_file = Path(config.RESULTS_FILE).resolve()

        def replace_legacy_append(path, row):
            if Path(path).resolve() != canonical_file:
                return {"results": []}
            emitted = dict(row)
            if binary_path is not None:
                emitted["binary_path"] = str(Path(binary_path).resolve())
            digest = llamacpp_records.publish_row(
                emitted,
                work_root=work_root,
                expected=expected,
                observed=_observed_from_measured_log(row),
            )
            produced.append(digest)
            print(f"run-record v1: {digest}")
            return {"results": []}

        changed.append((lib_results, "append_result", lib_results.append_result))
        lib_results.append_result = replace_legacy_append
        # run_bench's final legacy SQLite sync only mirrors results.json;
        # it is not a separate measurement and must not execute.
        try:
            results_db = importlib.import_module("results_db")
        except ModuleNotFoundError:
            results_db = None
        if results_db is not None:
            source = getattr(results_db, "__file__", None)
            if not source or not Path(source).resolve().is_relative_to(bench):
                raise ValueError(
                    "external llama result database belongs to a different checkout"
                )
            changed.append((results_db, "sync", results_db.sync))
            results_db.sync = lambda: None
        runpy.run_path(str(script), run_name="__main__")
    finally:
        for module, name, original in reversed(changed):
            setattr(module, name, original)
        sys.path[:] = path_backup
        sys.argv[:] = argv_backup
    if not produced:
        raise ValueError("run_bench produced no measured result rows")
    return produced


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runner-root", type=Path, required=True)
    parser.add_argument("--work-root", type=Path)
    parser.add_argument(
        "--binary-path",
        type=Path,
        help="exact measured executable when run_bench row lacks binary_path",
    )
    parser.add_argument(
        "--expected-identity",
        type=Path,
        required=True,
        help="pre-admitted VA25 backend/architectures/locators JSON",
    )
    arguments = list(sys.argv[1:] if argv is None else argv)
    if "--" not in arguments:
        parser.error("pass actual run_bench.py arguments after --")
    boundary = arguments.index("--")
    options = parser.parse_args(arguments[:boundary])
    data = json.loads(options.expected_identity.read_text(encoding="utf-8"))
    expected = attestation.ExecutionIdentity(
        data["backend"],
        tuple(data["architectures"]),
        tuple(data["locators"]) if data.get("locators") is not None else None,
    )
    run_with_records(
        options.runner_root,
        arguments[boundary + 1 :],
        expected=expected,
        work_root=options.work_root,
        binary_path=options.binary_path,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
