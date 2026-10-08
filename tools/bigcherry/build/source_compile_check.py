"""PA44-A: compile/type probes for composed CUDA source without a GPU SDK.

The full HIP compiler check remains tools/bigcherry/build/compile_check.py. This
module catches source-level host/device and helper-signature mistakes on normal
CI by deriving tiny translation units from the composed production source and
asking clang to type-check them.
"""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


class SourceCompileCheckError(RuntimeError):
    pass


@dataclass(frozen=True)
class CallSite:
    path: Path
    line: int
    arity: int
    owner: str | None


def _run(cmd: list[str], *, cwd: Path | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        cmd,
        cwd=cwd,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )


def _matching_paren(text: str, open_pos: int) -> int:
    depth = 0
    quote: str | None = None
    escape = False
    for pos in range(open_pos, len(text)):
        ch = text[pos]
        if quote is not None:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == quote:
                quote = None
            continue
        if ch in {'"', "'"}:
            quote = ch
        elif ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
            if depth == 0:
                return pos
    raise SourceCompileCheckError("unbalanced parentheses in composed source")


def _arity(args: str) -> int:
    if not args.strip():
        return 0
    depth = 0
    quote: str | None = None
    escape = False
    count = 1
    for ch in args:
        if quote is not None:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == quote:
                quote = None
            continue
        if ch in {'"', "'"}:
            quote = ch
        elif ch in "([{<":
            depth += 1
        elif ch in ")]}>":
            depth = max(0, depth - 1)
        elif ch == "," and depth == 0:
            count += 1
    return count


def _line(text: str, pos: int) -> int:
    return text.count("\n", 0, pos) + 1


def _owner(text: str, pos: int) -> str | None:
    start = max(0, pos - 1200)
    matches = list(re.finditer(r"bigcherry\s+(\d{4})\b", text[start:pos], re.I))
    return matches[-1].group(1) if matches else None


def _source_files(root: Path) -> list[Path]:
    cuda = root / "ggml" / "src" / "ggml-cuda"
    if not cuda.is_dir():
        raise SourceCompileCheckError(f"missing composed CUDA source: {cuda}")
    return sorted(
        path
        for path in cuda.rglob("*")
        if path.suffix in {".cu", ".cuh", ".h", ".hpp"}
    )


def _find_signature_arity(files: list[Path], name: str) -> int:
    pattern = re.compile(rf"\b(?:static\s+)?bool\s+{re.escape(name)}\s*\(")
    found: set[int] = set()
    for path in files:
        text = path.read_text(encoding="utf-8", errors="replace")
        for match in pattern.finditer(text):
            open_pos = text.find("(", match.start())
            close = _matching_paren(text, open_pos)
            found.add(_arity(text[open_pos + 1 : close]))
    if not found:
        raise SourceCompileCheckError(f"cannot find declaration/definition for {name}")
    if len(found) != 1:
        raise SourceCompileCheckError(
            f"{name} has inconsistent declaration arities in composed source: {sorted(found)}"
        )
    return next(iter(found))


def _call_sites(files: list[Path], name: str) -> list[CallSite]:
    token = name + "("
    sites: list[CallSite] = []
    definition = re.compile(
        rf"\b(?:static\s+)?(?:bool|int|void|constexpr\s+int)\s+{re.escape(name)}\s*$"
    )
    for path in files:
        text = path.read_text(encoding="utf-8", errors="replace")
        pos = 0
        while True:
            pos = text.find(token, pos)
            if pos < 0:
                break
            open_pos = pos + len(name)
            close = _matching_paren(text, open_pos)
            prefix = text[max(0, pos - 120) : pos]
            suffix = text[close + 1 : close + 40]
            # A declaration/definition contributes the expected arity but is
            # not a call site. Exclude it from generated calls.
            if definition.search(prefix.rstrip()) and re.match(
                r"\s*(?:const\s*)?(?:;|\{)", suffix
            ):
                pos = close + 1
                continue
            sites.append(
                CallSite(path, _line(text, pos), _arity(text[open_pos + 1 : close]), _owner(text, pos))
            )
            pos = close + 1
    return sites


def _function_ranges(text: str) -> list[tuple[int, int, str]]:
    """Return coarse C/C++ function ranges as (start,end,header).

    This intentionally recognizes only definition-shaped headers and ignores
    control statements. It is sufficient for locating zero-arg helper calls
    inside host launch wrappers vs __global__/__device__ functions.
    """
    pattern = re.compile(
        r"(?ms)(?P<header>(?:template\s*<[^{};]+>\s*)?"
        r"(?!\s*(?:if|for|while|switch|catch)\b)"
        r"[^{};\n]*\b[A-Za-z_]\w*\s*\([^{};]*\)\s*)\{"
    )
    ranges: list[tuple[int, int, str]] = []
    for match in pattern.finditer(text):
        depth = 1
        pos = match.end()
        quote: str | None = None
        escape = False
        while pos < len(text) and depth:
            ch = text[pos]
            if quote is not None:
                if escape:
                    escape = False
                elif ch == "\\":
                    escape = True
                elif ch == quote:
                    quote = None
            elif ch in {'"', "'"}:
                quote = ch
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
            pos += 1
        if depth == 0:
            ranges.append((match.start(), pos, match.group("header")))
    return ranges


def _host_device_calls(files: list[Path], name: str) -> list[CallSite]:
    sites: list[CallSite] = []
    token = name + "()"
    for path in files:
        text = path.read_text(encoding="utf-8", errors="replace")
        ranges = _function_ranges(text)
        pos = 0
        while True:
            pos = text.find(token, pos)
            if pos < 0:
                break
            after = text[pos + len(token) : pos + len(token) + 30]
            before = text[max(0, pos - 120) : pos]
            # Exclude the helper's own declaration/definition.
            if re.search(rf"\b{re.escape(name)}\s*$", before) and re.match(
                r"\s*(?:;|\{)", after
            ):
                pos += len(token)
                continue
            enclosing = [entry for entry in ranges if entry[0] <= pos < entry[1]]
            if enclosing:
                _, _, header = min(enclosing, key=lambda item: item[1] - item[0])
                device = any(mark in header for mark in ("__global__", "__device__", "GGML_DEVICE"))
                if not device:
                    sites.append(CallSite(path, _line(text, pos), 0, _owner(text, pos)))
            pos += len(token)
    return sites


def _compile(path: Path, clangxx: str, *, cuda_host: bool = False) -> tuple[bool, str]:
    cmd = [clangxx, "-std=c++17", "-fsyntax-only"]
    if cuda_host:
        cmd += ["-x", "cuda", "--cuda-host-only", "-nocudainc", "-nocudalib"]
    cmd.append(str(path))
    done = _run(cmd)
    return done.returncode == 0, (done.stderr or done.stdout).strip()


def _write_arity_probe(path: Path, name: str, expected: int, sites: list[CallSite]) -> None:
    params = ", ".join("int" for _ in range(expected))
    lines = [f"bool {name}({params});"]
    for index, site in enumerate(sites):
        args = ", ".join("0" for _ in range(site.arity))
        lines.append(f"void probe_{index}() {{ (void) {name}({args}); }}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_device_probe(path: Path, has_host_call: bool) -> None:
    qualifier = "" if has_host_call else "__attribute__((device)) "
    path.write_text(
        "__attribute__((device)) constexpr int ggml_cuda_get_physical_warp_size() { return 64; }\n"
        f"{qualifier}int probe() {{ return ggml_cuda_get_physical_warp_size(); }}\n",
        encoding="utf-8",
    )


def check_cuda_contracts(source_root: Path, workdir: Path, clangxx: str) -> list[str]:
    files = _source_files(source_root)
    workdir.mkdir(parents=True, exist_ok=True)
    failures: list[str] = []

    helper = "ggml_cuda_should_use_mmvf"
    expected = _find_signature_arity(files, helper)
    calls = _call_sites(files, helper)
    if not calls:
        failures.append(f"{helper}: no call sites found in composed source")
    else:
        probe = workdir / "mmvf-arity.cpp"
        _write_arity_probe(probe, helper, expected, calls)
        ok, detail = _compile(probe, clangxx)
        if not ok:
            bad = [site for site in calls if site.arity != expected]
            where = ", ".join(
                f"{site.path.relative_to(source_root)}:{site.line}"
                + (f" owner=patch-{site.owner}" if site.owner else "")
                + f" args={site.arity}/{expected}"
                for site in bad
            )
            failures.append(f"{helper} signature mismatch: {where}\n{detail[-1200:]}")

    device_helper = "ggml_cuda_get_physical_warp_size"
    # Restrict this lightweight lexical scan to BigCherry-owned sites. The
    # pinned source has legitimate device-only uses inside template/macro
    # contexts that this parser cannot classify reliably; the PA44-A
    # regression was an injected 1345 host call and carries its owner marker.
    host_calls = [
        site for site in _host_device_calls(files, device_helper)
        if site.owner is not None
    ]

    # Prove both compiler directions explicitly. Device->device is a positive
    # probe and must compile. Host->device is deliberately invalid and must
    # fail; that expected negative-probe failure must not fail the workflow.
    device_probe = workdir / "physical-warp-device.cu"
    _write_device_probe(device_probe, False)
    ok_device, detail_device = _compile(device_probe, clangxx, cuda_host=True)
    if not ok_device:
        failures.append(
            "clang CUDA device-only positive probe is not usable on this runner:\n"
            + detail_device[-1200:]
        )

    host_probe = workdir / "physical-warp-host.cu"
    _write_device_probe(host_probe, True)
    ok_host, _ = _compile(host_probe, clangxx, cuda_host=True)
    if ok_host:
        failures.append(
            "clang CUDA negative probe unexpectedly accepted a host call to "
            f"{device_helper}"
        )

    if host_calls:
        where = ", ".join(
            f"{site.path.relative_to(source_root)}:{site.line}"
            + f" owner=patch-{site.owner}"
            for site in host_calls
        )
        failures.append(
            f"{device_helper} is device-only but is called from host code: {where}"
        )

    return failures


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="bigcherry-source-compile-check")
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--workdir", type=Path, required=True)
    parser.add_argument("--clangxx", default="clang++")
    args = parser.parse_args(argv)

    if shutil.which(args.clangxx) is None:
        print(f"source-compile-check: compiler not found: {args.clangxx}", file=sys.stderr)
        return 2
    try:
        failures = check_cuda_contracts(
            args.source_root.resolve(), args.workdir.resolve(), args.clangxx
        )
    except (OSError, SourceCompileCheckError) as exc:
        print(f"source-compile-check: {exc}", file=sys.stderr)
        return 2
    if failures:
        for failure in failures:
            print(f"source-compile-check: FAIL: {failure}", file=sys.stderr)
        return 1
    print("source-compile-check: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
