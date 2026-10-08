"""Offline validation entrypoint for changed BigCherry patch packages.

The caller must provide a pinned llama.cpp checkout at vendor/llama.cpp (normally
via \`\`python -m bigcherry pull --source bigcherry\`\`). This command performs no
network access itself.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import re
import subprocess
import sys
import time
import tomllib
from pathlib import Path
from typing import Sequence

_REPO = Path(__file__).resolve().parents[3]
_PATCH_TEST_RE = re.compile(r"^tools/tests/patch/test_(\d{4}_[A-Za-z0-9_]+)\.py$")
_DISPOSITION_RE = re.compile(r"^dispositions/(\d{4}_[^/]+)\.json$")
_DEFAULT_CHECK_TIMEOUT_SECONDS = 600
_DEFAULT_EXPERIMENT_WORKERS = 6


def _run_git(*args: str) -> str:
    proc = subprocess.run(
        ["git", "-C", str(_REPO), *args],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if proc.returncode:
        raise RuntimeError(
            f"git {' '.join(args)} failed ({proc.returncode}): {proc.stderr.strip()}"
        )
    return proc.stdout.strip()


def _resolve(ref: str) -> str:
    return _run_git("rev-parse", "--verify", f"{ref}^{{commit}}")


def _changed_files(base: str, head: str) -> tuple[str, ...]:
    out = _run_git(
        "diff",
        "--name-only",
        "--diff-filter=ACMRTD",
        base,
        head,
        "--",
    )
    return tuple(line for line in out.splitlines() if line)


def _patch_ids(paths: Sequence[str]) -> tuple[str, ...]:
    found: set[str] = set()
    for path in paths:
        if path.startswith("patches/"):
            parts = path.split("/", 2)
            if len(parts) >= 3 and parts[1] and not parts[1].startswith("_"):
                found.add(parts[1])
        match = _PATCH_TEST_RE.match(path)
        if match:
            found.add(match.group(1))
        match = _DISPOSITION_RE.match(path)
        if match:
            found.add(match.group(1))
    return tuple(sorted(found))


def _blob_text_lf(ref: str, path: str) -> bytes | None:
    """File content at ref with line endings normalised to LF; None if absent."""
    proc = subprocess.run(
        ["git", "-C", str(_REPO), "show", f"{ref}:{path}"],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    if proc.returncode:
        return None
    return proc.stdout.replace(b"\r\n", b"\n").replace(b"\r", b"\n")


def _only_line_endings_changed(base: str, head: str, path: str) -> bool:
    before = _blob_text_lf(base, path)
    after = _blob_text_lf(head, path)
    return before is not None and before == after


def _implementation_patch_ids(
    paths: Sequence[str], base: str | None = None, head: str | None = None
) -> tuple[str, ...]:
    """Patch implementations changed in this range.

    Metadata-only changes are covered by catalog/governance and composition
    checks; they do not invent a requirement for a per-package mechanics test.
    A patch.py whose only difference is its line endings (repository
    renormalisation) is not an implementation change.
    """
    found: set[str] = set()
    for path in paths:
        if not path.startswith("patches/") or not path.endswith("/patch.py"):
            continue
        if base and head and _only_line_endings_changed(base, head, path):
            continue
        parts = path.split("/", 2)
        if len(parts) >= 3 and parts[1] and not parts[1].startswith("_"):
            found.add(parts[1])
    return tuple(sorted(found))


def _composition_patch_ids(
    paths: Sequence[str], base: str | None = None, head: str | None = None
) -> tuple[str, ...]:
    """Patches whose composition inputs changed: patch.py (not a line-ending-only change) or patch.toml.

    A change to a patch's tests, README or SUMMARY cannot change how any experiment composes, so it must not
    trigger the experiment audit: a test-only pull request touching patches used by many experiments spent the
    whole 30-minute job re-checking compositions that could not have moved.
    """
    found = set(_implementation_patch_ids(paths, base, head))
    for path in paths:
        if path.startswith("patches/") and path.endswith("/patch.toml"):
            parts = path.split("/", 2)
            if len(parts) >= 3 and parts[1] and not parts[1].startswith("_"):
                found.add(parts[1])
    return tuple(sorted(found))


def _env() -> dict[str, str]:
    env = os.environ.copy()
    tools = str(_REPO / "tools")
    current = env.get("PYTHONPATH")
    env["PYTHONPATH"] = tools if not current else tools + os.pathsep + current
    return env


def _run(
    name: str,
    command: Sequence[str],
    checks: list[dict[str, object]],
    *,
    timeout_seconds: int,
) -> int:
    print(f"::group::{name}", flush=True)
    print("+ " + " ".join(command), flush=True)
    started = time.monotonic()
    try:
        proc = subprocess.run(
            command,
            cwd=_REPO,
            env=_env(),
            check=False,
            timeout=timeout_seconds,
        )
        rc = proc.returncode
        detail = None
    except subprocess.TimeoutExpired:
        rc = 124
        detail = f"timed out after {timeout_seconds}s"
        print(f"{name}: {detail}", file=sys.stderr, flush=True)
    elapsed = time.monotonic() - started
    print("::endgroup::", flush=True)
    record: dict[str, object] = {
        "name": name,
        "command": list(command),
        "returncode": rc,
        "elapsed_seconds": round(elapsed, 3),
    }
    if detail:
        record["detail"] = detail
    checks.append(record)
    return rc


def _write_json(path: Path | None, payload: dict[str, object]) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _load_recipes() -> dict[str, object]:
    return tomllib.loads((_REPO / "config" / "recipes.toml").read_text(encoding="utf-8"))


def _source_patch_ids(raw: dict[str, object], source: str) -> set[str]:
    source_table = raw.get("source", {})
    patch_sets = raw.get("patch-set", {})
    if not isinstance(source_table, dict) or not isinstance(patch_sets, dict):
        return set()
    source_cfg = source_table.get(source, {})
    if not isinstance(source_cfg, dict):
        return set()
    ids: set[str] = set()
    for patch_set in source_cfg.get("patch-sets", []):
        cfg = patch_sets.get(patch_set, {})
        if isinstance(cfg, dict):
            ids.update(cfg.get("patches", []))
    return ids


def _experiment_usage(raw: dict[str, object]) -> dict[str, set[str]]:
    experiments = raw.get("experiment", {})
    sources = raw.get("source", {})
    if not isinstance(experiments, dict) or not isinstance(sources, dict):
        return {}
    known_sources = set(sources)
    usage = {name: set() for name in experiments}

    campaigns = raw.get("campaign", {})
    if isinstance(campaigns, dict):
        for campaign in campaigns.values():
            if not isinstance(campaign, dict):
                continue
            for lane in campaign.get("lanes", []):
                if not isinstance(lane, dict):
                    continue
                name = lane.get("experiment")
                source = lane.get("source")
                if name in usage and source in known_sources:
                    usage[name].add(source)

    cmd_patterns = (
        re.compile(r"--source\s+([A-Za-z0-9_-]+).*?--experiment\s+([A-Za-z0-9_-]+)"),
        re.compile(r"--experiment\s+([A-Za-z0-9_-]+).*?--source\s+([A-Za-z0-9_-]+)"),
    )
    lane_pattern = re.compile(r'source\s*=\s*"([^"]+)".*?experiment\s*=\s*"([^"]+)"')
    queue_pattern = re.compile(
        r"\b([A-Za-z0-9_-]+):[A-Za-z0-9_-]+:[A-Za-z0-9_-]+\s+([A-Za-z0-9_-]+)(?:\s|$)"
    )

    for base in ("tools", "docs/reference", ".github"):
        root = _REPO / base
        if not root.exists():
            continue
        for path in sorted(root.rglob("*")):
            if not path.is_file():
                continue
            try:
                lines = path.read_text(encoding="utf-8").splitlines()
            except (UnicodeDecodeError, OSError):
                continue
            for line in lines:
                first = cmd_patterns[0].search(line)
                if first:
                    source, name = first.groups()
                    if name in usage and source in known_sources:
                        usage[name].add(source)
                second = cmd_patterns[1].search(line)
                if second:
                    name, source = second.groups()
                    if name in usage and source in known_sources:
                        usage[name].add(source)
                for pattern in (lane_pattern, queue_pattern):
                    match = pattern.search(line)
                    if match:
                        source, name = match.groups()
                        if name in usage and source in known_sources:
                            usage[name].add(source)

    return usage


def _experiment_names_for_audit(
    raw: dict[str, object],
    changed_patch_ids: Sequence[str],
    *,
    full_audit: bool,
) -> tuple[str, ...]:
    experiments = raw.get("experiment", {})
    if not isinstance(experiments, dict):
        return ()
    if full_audit:
        return tuple(sorted(experiments))
    changed = set(changed_patch_ids)
    names = []
    for name, cfg in experiments.items():
        if not isinstance(cfg, dict):
            continue
        if changed.intersection(cfg.get("patches", [])):
            names.append(name)
    return tuple(sorted(names))


def _experiment_sources(
    raw: dict[str, object],
    usage: dict[str, set[str]],
    name: str,
) -> tuple[str, ...]:
    experiments = raw.get("experiment", {})
    sources = raw.get("source", {})
    if not isinstance(experiments, dict) or not isinstance(sources, dict):
        return ()
    known_sources = set(sources)

    # This experiment intentionally overlays 1333 onto upstream llama.cpp;
    # 1333 is already in BigCherry production, so a production source is wrong.
    if name == "native-plus-1333" and "llama-native" in known_sources:
        return ("llama-native",)

    referenced = sorted(source for source in usage.get(name, set()) if source in known_sources)
    if referenced:
        return tuple(referenced)

    cfg = experiments.get(name, {})
    if not isinstance(cfg, dict):
        return ()
    patches = set(cfg.get("patches", []))
    production_ids = _source_patch_ids(raw, "bigcherry")
    preferred = "bigcherry" if patches <= production_ids else "bigcherry-tuning"

    required: set[str] = set()
    for patch_id in patches:
        manifest = _REPO / "patches" / patch_id / "patch.toml"
        if not manifest.is_file():
            continue
        metadata = tomllib.loads(manifest.read_text(encoding="utf-8"))
        required.update(metadata.get("requires", []))
    required -= patches

    candidates = (
        preferred,
        "bigcherry-tuning",
        "llama-native",
        "bigcherry-qualification",
        "bigcherry-qualification-tuning",
        "bigcherry",
        "bigcherry-serving-base",
    )
    for source in dict.fromkeys(candidates):
        if source in known_sources and required <= _source_patch_ids(raw, source):
            return (source,)
    return (preferred,) if preferred in known_sources else ()


def _run_experiment_audit(
    raw: dict[str, object],
    names: Sequence[str],
    checks: list[dict[str, object]],
    *,
    workers: int,
    timeout_seconds: int,
) -> bool:
    usage = _experiment_usage(raw)
    jobs: list[tuple[str, str]] = []
    setup_failed = False
    for name in names:
        sources = _experiment_sources(raw, usage, name)
        if not sources:
            setup_failed = True
            checks.append(
                {
                    "name": f"experiment {name}",
                    "returncode": 2,
                    "detail": "no audit source could be selected",
                }
            )
            continue
        jobs.extend((name, source) for source in sources)

    if not jobs:
        print("experiment audit: no matching experiments", flush=True)
        return setup_failed

    def check_selection(selection: tuple[str, str]) -> dict[str, object]:
        name, source = selection
        started = time.monotonic()
        command = [
            sys.executable,
            "-m",
            "bigcherry",
            "patch-rebase-check",
            "--source",
            source,
            "--experiment",
            name,
        ]
        try:
            result = subprocess.run(
                command,
                cwd=_REPO,
                env=_env(),
                check=False,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                timeout=timeout_seconds,
            )
            rc = result.returncode
            output = result.stdout
            detail = None
        except subprocess.TimeoutExpired as exc:
            rc = 124
            output = exc.stdout or ""
            if isinstance(output, bytes):
                output = output.decode("utf-8", errors="replace")
            detail = f"timed out after {timeout_seconds}s"
        except OSError as exc:
            rc = 127
            output = ""
            detail = str(exc)
        return {
            "name": f"experiment {name} ({source})",
            "experiment": name,
            "source": source,
            "command": command,
            "returncode": rc,
            "elapsed_seconds": round(time.monotonic() - started, 3),
            "output": output,
            **({"detail": detail} if detail else {}),
        }

    pool_size = min(max(1, workers), len(jobs))
    print(f"experiment audit: {len(jobs)} selection(s), workers={pool_size}", flush=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=pool_size) as executor:
        results = list(executor.map(check_selection, jobs))

    failures = [result for result in results if int(result["returncode"]) != 0]
    for result in results:
        print(
            f"{result['name']}: {'PASS' if result['returncode'] == 0 else 'FAIL'} "
            f"({result['elapsed_seconds']:.1f}s)",
            flush=True,
        )
        checks.append({key: value for key, value in result.items() if key != "output"})

    print(
        f"Experiment selections checked: {len(results)}; failed: {len(failures)}",
        flush=True,
    )
    if failures:
        print("Failed experiment selections:", file=sys.stderr, flush=True)
        for result in failures:
            print(
                f"  - {result['experiment']} source={result['source']} "
                f"rc={result['returncode']}"
                + (f" ({result['detail']})" if result.get("detail") else ""),
                file=sys.stderr,
                flush=True,
            )
            output = str(result.get("output", ""))
            if output:
                print(output, file=sys.stderr, flush=True)
    return setup_failed or bool(failures)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run fail-closed offline checks for patch packages changed between two git refs."
    )
    parser.add_argument("--base", required=True, help="base git ref/commit")
    parser.add_argument("--head", default="HEAD", help="head git ref/commit (default: HEAD)")
    parser.add_argument(
        "--source",
        default="bigcherry",
        help="canonical source used for patch-rebase-check (default: bigcherry)",
    )
    parser.add_argument("--json", type=Path, default=None, help="write a compact JSON result")
    parser.add_argument(
        "--audit-experiments",
        action="store_true",
        help="audit experiments that name a changed patch",
    )
    parser.add_argument(
        "--audit-all-experiments",
        action="store_true",
        help="audit every configured experiment",
    )
    parser.add_argument(
        "--experiment-workers",
        type=int,
        default=_DEFAULT_EXPERIMENT_WORKERS,
        help=f"parallel experiment workers (default: {_DEFAULT_EXPERIMENT_WORKERS})",
    )
    parser.add_argument(
        "--check-timeout",
        type=int,
        default=_DEFAULT_CHECK_TIMEOUT_SECONDS,
        help=f"per subprocess timeout in seconds (default: {_DEFAULT_CHECK_TIMEOUT_SECONDS})",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    checks: list[dict[str, object]] = []

    try:
        base = _resolve(args.base)
        head = _resolve(args.head)
        changed = _changed_files(base, head)
    except RuntimeError as exc:
        print(f"patch-offline-check: {exc}", file=sys.stderr)
        _write_json(
            args.json,
            {
                "base": args.base,
                "head": args.head,
                "result": "FAIL",
                "error": str(exc),
                "checks": checks,
            },
        )
        return 2

    if args.check_timeout <= 0 or args.experiment_workers <= 0:
        print("patch-offline-check: worker count and timeout must be positive", file=sys.stderr)
        return 2

    patch_ids = _patch_ids(changed)
    implementation_patch_ids = set(_implementation_patch_ids(changed, base, head))
    print(f"range: {base[:12]}..{head[:12]}")
    print(f"changed patch ids: {', '.join(patch_ids) if patch_ids else '(none)'}")

    failed = False
    test_modules = [
        "tools.tests.patch.test_patch_catalog",
        "tools.tests.patch.test_patch_governance",
        "tools.tests.patch.test_offline_check",
    ]
    missing_tests: list[str] = []
    for patch_id in patch_ids:
        patch_toml = _REPO / "patches" / patch_id / "patch.toml"
        if not patch_toml.is_file():
            continue
        test_path = _REPO / "tools" / "tests" / "patch" / f"test_{patch_id}.py"
        if test_path.is_file():
            test_modules.append(f"tools.tests.patch.test_{patch_id}")
        elif patch_id in implementation_patch_ids:
            missing_tests.append(patch_id)

    if missing_tests:
        failed = True
        message = "changed patch.py missing mechanics test module(s): " + ", ".join(missing_tests)
        print(f"patch-offline-check: {message}", file=sys.stderr)
        checks.append({"name": "patch-test-presence", "returncode": 1, "detail": message})

    unit_rc = _run(
        "patch unit/catalog/governance tests",
        [sys.executable, "-m", "unittest", *test_modules],
        checks,
        timeout_seconds=args.check_timeout,
    )
    failed |= unit_rc != 0

    rc = _run(
        "rebase production source",
        [
            sys.executable,
            "-m",
            "bigcherry",
            "patch-rebase-check",
            "--source",
            args.source,
        ],
        checks,
        timeout_seconds=args.check_timeout,
    )
    failed |= rc != 0

    full_audit = args.audit_all_experiments or "config/recipes.toml" in changed
    if args.audit_experiments or full_audit:
        try:
            raw = _load_recipes()
            experiment_names = _experiment_names_for_audit(
                raw,
                _composition_patch_ids(changed, base, head),
                full_audit=full_audit,
            )
            print(
                "experiment audit scope: "
                + ("full" if full_audit else f"{len(experiment_names)} changed-patch experiment(s)"),
                flush=True,
            )
            failed |= _run_experiment_audit(
                raw,
                experiment_names,
                checks,
                workers=args.experiment_workers,
                timeout_seconds=args.check_timeout,
            )
        except (OSError, tomllib.TOMLDecodeError, ValueError) as exc:
            failed = True
            print(f"experiment audit setup failed: {exc}", file=sys.stderr)
            checks.append(
                {
                    "name": "experiment audit setup",
                    "returncode": 2,
                    "detail": str(exc),
                }
            )

    result = "FAIL" if failed else "PASS"
    payload: dict[str, object] = {
        "base": base,
        "head": head,
        "source": args.source,
        "changed_files": changed,
        "patch_ids": patch_ids,
        "checks": checks,
        "result": result,
    }
    _write_json(args.json, payload)
    print(f"patch-offline-check: {result}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
