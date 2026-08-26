"""Deterministic local CI gates for the BigCherry patch system.

This command is intentionally non-mutating: it audits source and metadata but
never compiles ROCm, launches a model, or changes a checkout.
"""
from __future__ import annotations

import argparse
import ast
import json
import re
import subprocess
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Callable

from .core import paths
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


@dataclass(frozen=True)
class HygieneDiagnostic:
    """One stable, actionable TR14 tooling-boundary finding."""

    code: str
    severity: str
    path: str
    message: str
    remediation: str

    def as_dict(self) -> dict[str, str]:
        return asdict(self)


_TOP_LEVEL_PYTHON_ALLOWLIST = frozenset({
    "residency_gates.py",
    "verify_slice_a.py",
})
_PROTECTED_DOMAINS = frozenset({
    "release",
    "source",
    "build",
    "experiment",
    "patch",
    "campaign",
    "tuning",
})
_PATH_AUTHORITY_FILES = frozenset({"tools/bigcherry/core/paths.py"})
_DISPOSITION_ROW = re.compile(
    r"^\|\s*`(?P<path>[^`]+)`\s*\|\s*\*\*(?P<disposition>[A-Z-]+)\*\*\s*\|"
)


def _relative(root: Path, path: Path) -> str:
    return path.relative_to(root).as_posix()


def _diagnostic(
    root: Path,
    code: str,
    severity: str,
    path: Path,
    message: str,
    remediation: str,
) -> HygieneDiagnostic:
    return HygieneDiagnostic(code, severity, _relative(root, path), message, remediation)


def _python_imports(path: Path) -> set[str]:
    """Return statically spelled imports; malformed files are handled by caller."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    imports: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imports.add(node.module)
    return imports


def _has_fixed_parent_depth(path: Path) -> list[int]:
    """Find actual ``thing.parents[N]`` AST nodes, not textual mentions."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    lines: list[int] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Subscript):
            continue
        if not isinstance(node.value, ast.Attribute) or node.value.attr != "parents":
            continue
        index = node.slice
        if isinstance(index, ast.Constant) and isinstance(index.value, int):
            lines.append(node.lineno)
    return sorted(set(lines))


def _facade_target(path: Path) -> str | None:
    """Return a canonical import target for the narrow facade pattern."""
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except SyntaxError:
        return None
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        function = node.func
        if not (
            isinstance(function, ast.Attribute)
            and function.attr == "import_module"
            and isinstance(function.value, ast.Name)
            and function.value.id == "importlib"
            and node.args
            and isinstance(node.args[0], ast.Constant)
            and isinstance(node.args[0].value, str)
        ):
            continue
        target = node.args[0].value
        if target.startswith("bigcherry."):
            return target
    return None


def _disposition_rows(root: Path) -> dict[str, str]:
    path = root / "docs" / "planning" / "active" / "rationalisation" / "TOOL_DISPOSITION.md"
    if not path.is_file():
        return {}
    rows: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        match = _DISPOSITION_ROW.match(line)
        if match:
            rows[match.group("path").replace("\\", "/")] = match.group("disposition")
    return rows


def tooling_hygiene(root: Path) -> tuple[HygieneDiagnostic, ...]:
    """Inspect only repository tooling boundaries; never mutate or inspect vendor/src."""
    root = root.resolve()
    tools_root = root / "tools"
    product_root = tools_root / "bigcherry"
    findings: list[HygieneDiagnostic] = []

    if tools_root.is_dir():
        for path in sorted(tools_root.iterdir(), key=lambda item: item.name):
            if path.is_file() and path.suffix == ".py" and path.name not in _TOP_LEVEL_PYTHON_ALLOWLIST:
                findings.append(_diagnostic(
                    root, "TR14.TOP_LEVEL_SCRIPT", "error", path,
                    "maintained Python tooling is at the tools root",
                    "move it to tools/bigcherry/<domain>/ or retain it only as an explicit compatibility wrapper",
                ))
            if path.is_file() and path.suffix.lower() in {".bat", ".cmd", ".ps1", ".sh"}:
                findings.append(_diagnostic(
                    root, "TR14.ENVIRONMENT_SCRIPT", "warning", path,
                    "environment or shell tooling is at the tools root",
                    "move environment bootstrap scripts to tools/env/ and update their callers",
                ))

    if product_root.is_dir():
        for path in sorted(product_root.rglob("*.py")):
            relative = _relative(root, path)
            try:
                imports = _python_imports(path)
                parent_lines = _has_fixed_parent_depth(path)
            except (OSError, SyntaxError) as exc:
                findings.append(_diagnostic(
                    root, "TR14.PYTHON_PARSE", "error", path,
                    f"tooling source cannot be parsed: {type(exc).__name__}",
                    "fix the syntax before relying on this tooling boundary",
                ))
                continue

            for imported in sorted(imports):
                if imported == "tools.lab" or imported.startswith("tools.lab."):
                    findings.append(_diagnostic(
                        root, "TR14.PRODUCTION_LAB_IMPORT", "error", path,
                        f"production tooling imports lab code ({imported})",
                        "keep lab experiments unimported; graduate durable code into a maintained domain",
                    ))
                if imported == "tools.tests" or imported.startswith("tools.tests."):
                    findings.append(_diagnostic(
                        root, "TR14.PRODUCTION_TEST_IMPORT", "error", path,
                        f"production tooling imports tests ({imported})",
                        "move shared behavior into a product domain and keep test helpers under tools/tests",
                    ))
                for domain in sorted(_PROTECTED_DOMAINS):
                    if relative.startswith(f"tools/bigcherry/{domain}/") and (
                        imported == "bigcherry.cli" or imported.startswith("bigcherry.cli.")
                    ):
                        findings.append(_diagnostic(
                            root, "TR14.DOMAIN_CLI_IMPORT", "error", path,
                            f"{domain} domain imports CLI assembly ({imported})",
                            "keep CLI wiring in bigcherry.cli and depend on domain APIs in one direction",
                        ))
                    if relative.startswith(f"tools/bigcherry/{domain}/") and (
                        imported == "bigcherry.analysis" or imported.startswith("bigcherry.analysis.")
                    ):
                        findings.append(_diagnostic(
                            root, "TR14.DOMAIN_ANALYSIS_IMPORT", "error", path,
                            f"{domain} domain imports analysis ({imported})",
                            "keep product workflows independent of offline analysis modules",
                        ))

            if relative not in _PATH_AUTHORITY_FILES:
                for line in parent_lines:
                    findings.append(_diagnostic(
                        root, "TR14.FIXED_PARENT_DEPTH", "error", path,
                        f"fixed Path.parents index at line {line}",
                        "resolve the repository or package boundary by searching for a structural marker",
                    ))

            target = _facade_target(path)
            if target is not None and path.name not in {"__init__.py", "__main__.py"}:
                findings.append(_diagnostic(
                    root, "TR14.ROOT_FACADE", "warning", path,
                    f"root compatibility facade duplicates canonical module {target}",
                    "migrate supported consumers, document the owner and retire the facade only with compatibility evidence",
                ))

    lab_root = tools_root / "lab"
    if lab_root.is_dir():
        for path in sorted(lab_root.rglob("__init__.py")):
            findings.append(_diagnostic(
                root, "TR14.LAB_PACKAGE", "error", path,
                "lab tooling has been made into a Python package",
                "remove the package marker; lab is intentionally non-importable",
            ))
        for topic in sorted(path for path in lab_root.iterdir() if path.is_dir() and path.name != "_template"):
            if not (topic / "README.md").is_file():
                findings.append(_diagnostic(
                    root, "TR14.LAB_METADATA", "error", topic,
                    "lab topic has no experiment metadata README",
                    "add a topic README covering question, inputs, outputs, runtime, safety and disposition",
                ))
        disposition_paths = set(_disposition_rows(root))
        for path in sorted(lab_root.rglob("*")):
            if path.is_file() and path.name != "README.md" and not any(
                part in {"__pycache__", ".ruff_cache", "_template"} or part.startswith(".")
                for part in path.relative_to(lab_root).parts
            ) and path.suffix != ".pyc":
                relative = _relative(root, path)
                if relative not in disposition_paths:
                    findings.append(_diagnostic(
                        root, "TR14.LAB_UNCLASSIFIED", "warning", path,
                        "lab tooling is absent from TOOL_DISPOSITION.md",
                        "record a disposition and plan owner before retaining or graduating the experiment",
                    ))

    for relative, disposition in sorted(_disposition_rows(root).items()):
        if disposition != "DELETE":
            continue
        path = root / Path(relative)
        if path.exists():
            findings.append(_diagnostic(
                root, "TR14.DISPOSITION_DELETE_PENDING", "error", path,
                "disposition map marks this path DELETE but it still exists",
                "complete caller/reference proof, then remove it in the owning migration slice",
            ))

    return tuple(sorted(findings, key=lambda item: (item.code, item.path, item.message)))


def _tooling_hygiene(root: Path) -> str | None:
    findings = tooling_hygiene(root)
    if not findings:
        return None
    return "\n".join(
        f"{item.code} [{item.severity}] {item.path}: {item.message}; remediation: {item.remediation}"
        for item in findings
    )


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
        CheckSpec("tooling-hygiene", "quick", _tooling_hygiene),
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
