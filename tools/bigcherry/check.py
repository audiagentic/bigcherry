"""Deterministic local CI gates for the BigCherry patch system.

This command is intentionally non-mutating: it audits source and metadata but
never compiles ROCm, launches a model, or changes a checkout.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Callable

from . import paths
from .patch import catalog as patch_catalog
from .source import audit as source_audit

TIERS = ("quick", "default", "full")


@dataclass(frozen=True)
class CheckSpec:
    id: str
    tier: str
    run: Callable[[Path], str | None]


@dataclass(frozen=True)
class CheckResult:
    id: str
    status: str
    detail: str
    duration_ms: int = 0
    skipped: bool = False

    @property
    def ok(self) -> bool:
        return self.status == "passed"


def _source_audit(root: Path) -> str | None:
    report = source_audit.audit(root / "vendor" / "llama.cpp")
    return None if source_audit.passed(report, strict=True) else json.dumps(report, sort_keys=True)


def _catalog(root: Path) -> str | None:
    problems = patch_catalog.cross_check()
    return None if not problems else "\n".join(problems)


def _python_compile(root: Path) -> str | None:
    result = subprocess.run(
        [sys.executable, "-m", "compileall", "-q", str(root / "tools" / "bigcherry")],
        cwd=root, text=True, capture_output=True, timeout=120,
    )
    return None if result.returncode == 0 else (result.stdout + result.stderr).strip()


def check_specs() -> tuple[CheckSpec, ...]:
    return (
        CheckSpec("patch-catalog", "quick", _catalog),
        CheckSpec("source-audit", "default", _source_audit),
        CheckSpec("python-compile", "full", _python_compile),
    )


def run_checks(*, root: Path, tier: str = "default", fail_fast: bool = False) -> dict[str, object]:
    if tier not in TIERS:
        raise ValueError(f"unknown check tier {tier!r}")
    rank = {name: index for index, name in enumerate(TIERS)}
    results: list[CheckResult] = []
    for spec in check_specs():
        if rank[spec.tier] > rank[tier]:
            continue
        started = time.monotonic()
        try:
            detail = spec.run(root)
            result = CheckResult(spec.id, "passed" if detail is None else "failed",
                                 "ok" if detail is None else detail,
                                 int((time.monotonic() - started) * 1000))
        except subprocess.TimeoutExpired as exc:
            result = CheckResult(spec.id, "failed", f"timeout: {exc}",
                                 int((time.monotonic() - started) * 1000))
        except Exception as exc:  # a broken check is a failed gate, never green
            result = CheckResult(spec.id, "failed", f"exception: {type(exc).__name__}: {exc}",
                                 int((time.monotonic() - started) * 1000))
        results.append(result)
        if fail_fast and not result.ok:
            break
    return {
        "tier": tier,
        "fail_fast": fail_fast,
        "passed": all(result.ok for result in results),
        "checks": [asdict(result) for result in results],
    }


def main(argv: list[str] | None = None, *, root: Path | None = None) -> int:
    parser = argparse.ArgumentParser(prog="bigcherry check")
    tier = parser.add_mutually_exclusive_group()
    tier.add_argument("--quick", action="store_const", const="quick", dest="tier")
    tier.add_argument("--default", action="store_const", const="default", dest="tier")
    tier.add_argument("--full", action="store_const", const="full", dest="tier")
    parser.set_defaults(tier="default")
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--json", metavar="PATH", default=None)
    args = parser.parse_args(argv)
    report = run_checks(root or paths.REPO_ROOT, tier=args.tier, fail_fast=args.fail_fast)
    encoded = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.json:
        Path(args.json).write_text(encoded, encoding="utf-8")
    else:
        for result in report["checks"]:
            print(f"[{result['status'].upper():6}] {result['id']}: {result['detail']}")
    return 0 if report["passed"] else 1
