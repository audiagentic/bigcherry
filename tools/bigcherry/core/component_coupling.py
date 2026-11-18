"""Card-free Radiance component/patch boundary coupling inventory gate.

This checks literal headers and process flags/environment against the owned
component's [coupling] declaration. When an upstream source inventory is
supplied, it also fails for deleted/moved upstream files and stale ABI.
RUM02-07 consume these declarations; they do not add a second config resolver.
"""
from __future__ import annotations

import fnmatch
import re
from pathlib import Path

_CPP = {".cpp", ".cc", ".cxx", ".h", ".hpp", ".hip"}
_ENV_PREFIXES = ("R3_", "RAD_", "RADIANCE_", "ROCR_", "HIP_", "BC_", "GPU_")
_NOT_ENV = ("BC_CFG_", "BC_RELEASE_", "RAD_E_")
_INCLUDE = re.compile(r'^\s*#\s*include\s*[<"]([^">]+)[">]', re.MULTILINE)
_FLAG = re.compile(r'"(--[a-z][a-z0-9-]+)"')
_ENV = re.compile(r'"([A-Z][A-Z0-9_]+)"')


def source_literals(root: Path) -> tuple[set[str], set[str], set[str]]:
    files = [p for p in root.rglob("*") if p.is_file() and p.suffix in _CPP
             and "test" not in p.stem and "generated" not in p.parts
             and "third_party" not in p.parts and "contract" not in p.parts]
    code = "\n".join(p.read_text(encoding="utf-8", errors="replace") for p in files)
    local = {p.name for p in root.rglob("*") if p.is_file()}
    headers = {inc.rsplit("/", 1)[-1] for inc in _INCLUDE.findall(code)
               if inc.rsplit("/", 1)[-1].startswith(("rad_", "r4d_"))
               and inc.rsplit("/", 1)[-1] not in local}
    flags = set(_FLAG.findall(code))
    env = {key for key in _ENV.findall(code) if key.startswith(_ENV_PREFIXES)
           and not key.startswith(_NOT_ENV) and not key.endswith("_")}
    return headers, flags, env


def problems(component, *, upstream_paths: set[str] | None = None,
             upstream_abi: int | None = None, patch_root: Path | None = None) -> list[str]:
    """Reject undeclared/new/stale coupling; upstream inputs are from a pinned source inventory.

    Upstream checks are optional for the offline unit suite, mandatory when
    checking a pin or patch-applied source. The current-pin declarations are
    still checked against every component source in the core CI suite.
    """
    declared = component.coupling
    if not declared:
        return [f"{component.name}: missing [coupling] declaration"]
    observed_headers, flags, env = source_literals(component.root)
    registry = declared["option-registry"]
    from_registry: set[str] = set()
    if registry:
        import tomllib
        path = component.root / registry
        try:
            fields = tomllib.loads(path.read_text(encoding="utf-8"))["field"]
        except (OSError, KeyError, ValueError) as exc:
            return [f"{component.name}: option registry cannot be read: {exc}"]
        from_registry = {f["radiance_option"] for f in fields if f.get("radiance_option")}
        if not from_registry:
            return [f"{component.name}: option registry is empty"]
    expected = {
        "upstream-headers": observed_headers,
        "runtime-env": env,
        "protocol-flags": flags & set(declared["protocol-flags"]),
        "owned-flags": flags - from_registry - set(declared["protocol-flags"]),
    }
    faults = []
    for name, actual in expected.items():
        recorded = set(declared[name])
        for missing in sorted(actual - recorded):
            faults.append(f"{component.name}: undeclared {name}: {missing}")
        for stale in sorted(recorded - actual):
            faults.append(f"{component.name}: stale {name}: {stale}")
    # Owned protocol and wire constants have an independent identity from the
    # upstream plugin ABI; a compatible plugin can still misinterpret a ring.
    for entry in declared.get("wire-constants", ()):
        file, literal = entry.split("::", 1)
        path = component.root / file
        try:
            body = path.read_text(encoding="utf-8")
        except OSError as exc:
            faults.append(f"{component.name}: wire constant source missing: {file}: {exc}")
            continue
        symbol, expected_value = literal.split(" = ", 1)
        match = re.search(
            rf"(?m)^\s*constexpr\s+[^;\n=]+\b{re.escape(symbol)}\s*=\s*(0x[0-9a-fA-F]+|[0-9]+)(?:u|ul|ull|U|UL|ULL)?\s*;",
            body,
        )
        if match is None or int(match.group(1), 0) != int(expected_value, 0):
            faults.append(f"{component.name}: wire constant changed: {entry}")
    if set(declared["protocol-flags"]) & set(declared["owned-flags"]):
        faults.append(f"{component.name}: protocol and owned flags overlap")
    source_files = set(declared["upstream-files"])
    for header in declared["upstream-headers"]:
        expected_path = ("abi/" if header.startswith("rad_") else "libr4d/") + header
        if expected_path not in source_files:
            faults.append(f"{component.name}: header {header} has no upstream-file declaration")
    if upstream_paths is not None:
        for file in sorted(source_files):
            if file not in upstream_paths:
                faults.append(f"{component.name}: upstream file no longer exists: {file}")
        for pattern in declared["upstream-source-globs"]:
            if not any(fnmatch.fnmatchcase(path, pattern) for path in upstream_paths):
                faults.append(f"{component.name}: upstream source pattern no longer exists: {pattern}")
    if upstream_paths is not None and upstream_abi is None:
        faults.append(f"{component.name}: target plugin ABI could not be determined")
    elif upstream_abi is not None and declared["abi-version"] != upstream_abi:
        faults.append(f"{component.name}: plugin ABI {declared['abi-version']} differs from upstream {upstream_abi}")
    if patch_root is not None:
        packages = [path.read_text(encoding="utf-8", errors="replace")
                    for path in (patch_root).glob("*/patch.py")]
        for file in declared["patch-provided-files"]:
            if not any(file in body for body in packages):
                faults.append(f"{component.name}: patch-provided file no longer exists: {file}")
    return faults
