"""Offline validation entrypoint for changed BigCherry patch packages.

The caller must provide a pinned llama.cpp checkout at vendor/llama.cpp (normally
via ``python -m bigcherry pull --source bigcherry``). This command performs no
network access itself.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Sequence

_REPO = Path(__file__).resolve().parents[3]
_PATCH_TEST_RE = re.compile(r"^tools/tests/patch/test_(\d{4}_[A-Za-z0-9_]+)\.py$")
_DISPOSITION_RE = re.compile(r"^dispositions/(\d{4}_[^/]+)\.json$")


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


def _env() -> dict[str, str]:
    env = os.environ.copy()
    tools = str(_REPO / "tools")
    current = env.get("PYTHONPATH")
    env["PYTHONPATH"] = tools if not current else tools + os.pathsep + current
    return env


def _run(name: str, command: Sequence[str], checks: list[dict[str, object]]) -> int:
    print(f"::group::{name}", flush=True)
    print("+ " + " ".join(command), flush=True)
    proc = subprocess.run(command, cwd=_REPO, env=_env(), check=False)
    print("::endgroup::", flush=True)
    checks.append(
        {
            "name": name,
            "command": list(command),
            "returncode": proc.returncode,
        }
    )
    return proc.returncode


def _write_json(path: Path | None, payload: dict[str, object]) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


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

    patch_ids = _patch_ids(changed)
    print(f"range: {base[:12]}..{head[:12]}")
    print(f"changed patch ids: {', '.join(patch_ids) if patch_ids else '(none)'}")

    failed = False
    test_modules = [
        "tools.tests.patch.test_patch_catalog",
        "tools.tests.patch.test_patch_governance",
    ]
    missing_tests: list[str] = []
    current_patch_ids: list[str] = []
    for patch_id in patch_ids:
        patch_toml = _REPO / "patches" / patch_id / "patch.toml"
        if not patch_toml.is_file():
            # A deliberately removed patch has no focal overlay to probe; the
            # catalog/governance tests below must validate the removal.
            continue
        current_patch_ids.append(patch_id)
        test_path = _REPO / "tools" / "tests" / "patch" / f"test_{patch_id}.py"
        if not test_path.is_file():
            missing_tests.append(patch_id)
        else:
            test_modules.append(f"tools.tests.patch.test_{patch_id}")

    if missing_tests:
        failed = True
        message = "missing patch test module(s): " + ", ".join(missing_tests)
        print(f"patch-offline-check: {message}", file=sys.stderr)
        checks.append({"name": "patch-test-presence", "returncode": 1, "detail": message})

    unit_rc = _run(
        "patch unit/catalog/governance tests",
        [sys.executable, "-m", "unittest", *test_modules],
        checks,
    )
    failed |= unit_rc != 0

    if current_patch_ids:
        for patch_id in current_patch_ids:
            rc = _run(
                f"rebase {patch_id}",
                [
                    sys.executable,
                    "-m",
                    "bigcherry",
                    "patch-rebase-check",
                    "--source",
                    args.source,
                    "--focal-overlay",
                    patch_id,
                ],
                checks,
            )
            failed |= rc != 0
    else:
        # Harness/config-only changes still prove the production source
        # composition against the currently pinned upstream checkout.
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
        )
        failed |= rc != 0

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
