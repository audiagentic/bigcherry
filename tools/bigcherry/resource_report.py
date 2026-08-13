"""Strict ROCm/Clang AMDGPU resource-report parsing and policy (HI09b)."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal


Status = Literal["resolved", "ambiguous", "missing", "schema_unknown"]


class ResourceError(RuntimeError):
    pass


@dataclass(frozen=True)
class ResourceRecordV1:
    stable_name: str | None
    mangled_symbol: str
    architecture: str
    vgpr: int | None
    sgpr: int | None
    agpr: int | None
    lds_bytes: int | None
    scratch_bytes: int | None
    occupancy: float | None
    sgpr_spills: int | None
    vgpr_spills: int | None
    parse_status: Status
    source_span: tuple[int, int]


@dataclass(frozen=True)
class ResourcePolicyV1:
    policy_id: str
    architecture: str
    reject_scratch_gt: int = 0
    reject_lds_gt: int | None = None
    warn_occupancy_lt: float | None = None


@dataclass(frozen=True)
class ParseResult:
    records: tuple[ResourceRecordV1, ...]
    diagnostics: tuple[str, ...]
    recognized_schema: bool
    parser_id: str


_DIAGNOSTIC = re.compile(
    r"(?m)^(?P<location>.*?:\d+:\d+):\s+remark:\s+(?P<body>.*?)\s*"
    r"\[-Rpass-analysis=kernel-resource-usage\]\s*$"
)
_OLD = re.compile(
    r"(?P<sgpr>\d+) SGPRs (?P<vgpr>\d+) VGPRs (?P<agpr>\d+) AGPRs "
    r"(?P<scratch>\d+) scratch bytes/thread (?P<occupancy>[0-9.]+) occupancy waves/SIMD "
    r"(?P<sgpr_spills>\d+) SGPR spills (?P<vgpr_spills>\d+) VGPR spills "
    r"(?P<lds>\d+) LDS size bytes/block"
)
_LABELS = {
    "TotalSGPRs": "sgpr",
    "SGPRs": "sgpr",
    "VGPRs": "vgpr",
    "AGPRs": "agpr",
    "ScratchSize [bytes/lane]": "scratch",
    "Occupancy [waves/SIMD]": "occupancy",
    "SGPRs Spill": "sgpr_spills",
    "VGPRs Spill": "vgpr_spills",
    "LDS Size [bytes/block]": "lds",
}


def _resolve(symbol: str, symbol_map: dict[str, list[str]]) -> tuple[str | None, Status]:
    names = symbol_map.get(symbol, [])
    if len(names) == 1:
        return names[0], "resolved"
    if len(names) > 1:
        return None, "ambiguous"
    return None, "missing"


def parse_clang_text_v1(
    raw: str,
    *,
    compiler_major: int,
    architecture: str,
    symbol_map: dict[str, list[str]],
) -> ParseResult:
    """Parse exactly the supported LLVM 21 text remark contract."""
    parser_id = f"clang-{compiler_major}-kernel-resource-text-v1"
    if compiler_major != 21:
        return ParseResult((), ("unsupported compiler major",), False, parser_id)
    matches = list(_DIAGNOSTIC.finditer(raw))
    if not matches:
        return ParseResult((), ("no kernel-resource-usage remarks",), False, parser_id)

    records: list[ResourceRecordV1] = []
    diagnostics: list[str] = []
    current_symbol = ""
    current_start = 0
    fields: dict[str, int | float | None] = {}

    def finish(end: int) -> None:
        nonlocal current_symbol, current_start, fields
        if not current_symbol:
            return
        required = {"sgpr", "vgpr", "scratch", "occupancy", "sgpr_spills", "vgpr_spills"}
        missing = sorted(required - fields.keys())
        stable_name, status = _resolve(current_symbol, symbol_map)
        if missing:
            status = "schema_unknown"
            diagnostics.append(f"{current_symbol}: missing fields {', '.join(missing)}")
        records.append(ResourceRecordV1(
            stable_name, current_symbol, architecture,
            int(fields["vgpr"]) if fields.get("vgpr") is not None else None,
            int(fields["sgpr"]) if fields.get("sgpr") is not None else None,
            int(fields["agpr"]) if fields.get("agpr") is not None else None,
            int(fields["lds"]) if fields.get("lds") is not None else None,
            int(fields["scratch"]) if fields.get("scratch") is not None else None,
            float(fields["occupancy"]) if fields.get("occupancy") is not None else None,
            int(fields["sgpr_spills"]) if fields.get("sgpr_spills") is not None else None,
            int(fields["vgpr_spills"]) if fields.get("vgpr_spills") is not None else None,
            status, (current_start, end),
        ))
        current_symbol, fields = "", {}

    for match in matches:
        body = match.group("body").strip()
        old = _OLD.fullmatch(body)
        if old:
            finish(match.start())
            symbol_match = re.search(r"(?:Function Name|Kernel Name):\s*(\S+)",
                                     raw[max(0, match.start() - 512):match.start()])
            symbol = symbol_match.group(1) if symbol_match else match.group("location")
            stable_name, status = _resolve(symbol, symbol_map)
            values = old.groupdict()
            records.append(ResourceRecordV1(
                stable_name, symbol, architecture, int(values["vgpr"]),
                int(values["sgpr"]), int(values["agpr"]), int(values["lds"]),
                int(values["scratch"]), float(values["occupancy"]),
                int(values["sgpr_spills"]), int(values["vgpr_spills"]),
                status, (match.start(), match.end()),
            ))
            continue
        function = re.fullmatch(r"(?:Function Name|Kernel Name):\s*(\S+)", body)
        if function:
            finish(match.start())
            current_symbol = function.group(1)
            current_start = match.start()
            continue
        label = re.fullmatch(r"(.+?):\s*([0-9.]+)", body)
        if current_symbol and label and label.group(1) in _LABELS:
            field = _LABELS[label.group(1)]
            value = float(label.group(2)) if field == "occupancy" else int(float(label.group(2)))
            fields[field] = value
        elif "Dynamic Stack:" not in body:
            diagnostics.append(f"unrecognized remark: {body}")
    finish(matches[-1].end())
    recognized = bool(records) and not any(
        record.parse_status == "schema_unknown" for record in records)
    return ParseResult(tuple(records), tuple(diagnostics), recognized, parser_id)


def policy_hash(policy: ResourcePolicyV1) -> str:
    canonical = json.dumps(asdict(policy), sort_keys=True, separators=(",", ":"))
    return hashlib.blake2b(canonical.encode(), digest_size=16).hexdigest()


def apply_policy(
    records: tuple[ResourceRecordV1, ...], policy: ResourcePolicyV1,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    exclusions: list[dict[str, Any]] = []
    advisories: list[dict[str, Any]] = []
    for record in records:
        if record.architecture != policy.architecture or record.parse_status != "resolved":
            continue
        reasons: list[str] = []
        if record.scratch_bytes is not None and record.scratch_bytes > policy.reject_scratch_gt:
            reasons.append("scratch")
        if (record.sgpr_spills or 0) > 0 or (record.vgpr_spills or 0) > 0:
            reasons.append("register_spill")
        if (policy.reject_lds_gt is not None and record.lds_bytes is not None and
                record.lds_bytes > policy.reject_lds_gt):
            reasons.append("lds_limit")
        if reasons:
            exclusions.append({
                "stable_name": record.stable_name,
                "architecture": record.architecture,
                "reasons": reasons,
                "source_span": list(record.source_span),
            })
        elif (policy.warn_occupancy_lt is not None and record.occupancy is not None and
              record.occupancy < policy.warn_occupancy_lt):
            advisories.append({
                "stable_name": record.stable_name,
                "architecture": record.architecture,
                "reason": "low_occupancy",
                "occupancy": record.occupancy,
            })
    return exclusions, advisories


def load_blacklist(path: Path) -> dict[tuple[str, str], tuple[str, ...]]:
    """Load the validated, architecture-specific exclusions from a report.

    This is deliberately an offline boundary.  A report that was not produced
    by a recognized parser, or that has malformed exclusion rows, cannot alter
    a catalog.  The catalog uses the tuple key to remove an architecture from
    a multi-architecture candidate without blacklisting the same geometry on
    a target where it compiled cleanly.
    """
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ResourceError(f"invalid resource report {path}: {exc}") from exc
    if not isinstance(report, dict) or report.get("schema_version") != 1:
        raise ResourceError(f"{path}: unsupported resource report schema")
    if report.get("recognized_schema") is not True:
        raise ResourceError(f"{path}: resource report schema is not recognized")
    architecture = report.get("architecture")
    if not isinstance(architecture, str) or not architecture:
        raise ResourceError(f"{path}: missing report architecture")
    exclusions = report.get("exclusions")
    if not isinstance(exclusions, list):
        raise ResourceError(f"{path}: exclusions must be a list")

    result: dict[tuple[str, str], tuple[str, ...]] = {}
    for row in exclusions:
        if not isinstance(row, dict):
            raise ResourceError(f"{path}: malformed blacklist row")
        stable_name = row.get("stable_name")
        row_arch = row.get("architecture", architecture)
        reasons = row.get("reasons")
        if (not isinstance(stable_name, str) or not stable_name or
                row_arch != architecture or not isinstance(reasons, list) or
                not reasons or any(not isinstance(reason, str) or not reason
                                   for reason in reasons)):
            raise ResourceError(f"{path}: malformed blacklist row")
        key = (stable_name, architecture)
        if key in result:
            raise ResourceError(f"{path}: duplicate blacklist row for {stable_name}")
        result[key] = tuple(sorted(set(reasons)))
    return result


def build_report(
    raw: bytes,
    *,
    compiler_family: str,
    compiler_major: int,
    compiler_version: str,
    architecture: str,
    source_revision: str,
    manifest_hash: str,
    symbol_map: dict[str, list[str]],
    policy: ResourcePolicyV1,
) -> dict[str, Any]:
    raw_hash = hashlib.blake2b(raw, digest_size=16).hexdigest()
    if compiler_family != "clang":
        result = ParseResult((), ("unsupported compiler family",), False,
                             f"{compiler_family}-{compiler_major}-unknown")
    else:
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ResourceError("raw resource stream is not UTF-8") from exc
        result = parse_clang_text_v1(
            text, compiler_major=compiler_major, architecture=architecture,
            symbol_map=symbol_map)
    exclusions, advisories = apply_policy(result.records, policy) if result.recognized_schema else ([], [])
    counts = {status: sum(record.parse_status == status for record in result.records)
              for status in ("resolved", "ambiguous", "missing", "schema_unknown")}
    report = {
        "schema_version": 1,
        "compiler_family": compiler_family,
        "compiler_major": compiler_major,
        "compiler_version": compiler_version,
        "architecture": architecture,
        "source_revision": source_revision,
        "input_manifest_hash": manifest_hash,
        "raw_report_hash": raw_hash,
        "parser_id": result.parser_id,
        "recognized_schema": result.recognized_schema,
        "diagnostics": list(result.diagnostics),
        "records": [asdict(record) for record in result.records],
        "resolution_counts": counts,
        "policy": asdict(policy),
        "policy_hash": policy_hash(policy),
        "exclusions": exclusions,
        "advisories": advisories,
    }
    canonical = json.dumps(report, sort_keys=True, separators=(",", ":"))
    report["report_hash"] = hashlib.blake2b(canonical.encode(), digest_size=16).hexdigest()
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="bigcherry resource-report")
    parser.add_argument("raw", type=Path)
    parser.add_argument("--symbol-map", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--compiler-family", default="clang")
    parser.add_argument("--compiler-major", type=int, required=True)
    parser.add_argument("--compiler-version", required=True)
    parser.add_argument("--architecture", required=True)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--manifest-hash", required=True)
    parser.add_argument("--reject-lds-gt", type=int)
    parser.add_argument("--warn-occupancy-lt", type=float)
    args = parser.parse_args(argv)
    symbol_map = json.loads(args.symbol_map.read_text(encoding="utf-8"))
    policy = ResourcePolicyV1(
        f"{args.architecture}-spill-v1", args.architecture,
        reject_lds_gt=args.reject_lds_gt,
        warn_occupancy_lt=args.warn_occupancy_lt,
    )
    report = build_report(
        args.raw.read_bytes(), compiler_family=args.compiler_family,
        compiler_major=args.compiler_major, compiler_version=args.compiler_version,
        architecture=args.architecture, source_revision=args.source_revision,
        manifest_hash=args.manifest_hash, symbol_map=symbol_map, policy=policy)
    args.output.write_text(json.dumps(report, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    if not report["recognized_schema"]:
        print("schema_unknown: raw evidence retained; no blacklist emitted")
        return 1
    print(f"wrote {args.output}: {len(report['records'])} records, {len(report['exclusions'])} exclusions")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
