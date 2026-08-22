"""HI78 extension: human-readable report generator for the end-to-end smoke
campaign (tools/bigcherry/e2e_smoke_campaign.py).

Reads bench.json (Campaign.s6_bench()'s three-way stock/native/replay
tokens/sec comparison) and the tune stage's measurements.jsonl, and writes
report.md: a pp/tg comparison table plus a promotion-decision summary that
keeps a genuine statistical win visually distinct from an actually-deployed
one (e.g. "improvement_pct: +49.5%, promotion_status: rejected_no_correctness_evidence"
must never read like a shipped result).

Usable both as Campaign.run()'s S7_report stage and as its own CLI:
    python -m bigcherry.e2e_smoke_report <workdir>
"""

from __future__ import annotations

import argparse
import json
import math
import os
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

# v2 (HI82 item 8) adds build_role/build_identity to each config -- purely
# additive to the configs/metrics/rows shape this report reads, so bump the
# accepted version rather than widen the check to a range.
BENCH_SCHEMA_VERSION = 2
BENCH_CONFIGS = ("stock", "native", "replay")
BENCH_WORKLOADS = ("pp", "tg")

# HI82 item 8: the "native" bench arm is the TUNE build run in native
# dispatch mode, not a fourth build -- see e2e_smoke_campaign.py's s6_bench().
BENCH_CONFIG_BUILD_ROLES = {"stock": "stock", "native": "tune", "replay": "replay"}

# Mirrors builds.CompletedBuildEvidence.campaign_identity()'s exact keys
# (also enforced structurally in e2e_smoke_campaign.py's
# _require_completed_build_identity_shape()) -- kept in sync deliberately
# rather than imported, since this module must stay usable as a standalone
# report reader over a bench.json produced by any campaign run.
BENCH_BUILD_IDENTITY_REQUIRED_KEYS = (
    "effective_build_id", "compile_verification_id", "compile_commands_digest",
    "hip_compile_commands_digest", "runtime_bundle_hash", "runtime_artifacts",
)


def _load_json(path: Path) -> Any:
    path = Path(path)
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except FileNotFoundError:
        raise
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"invalid JSON in {path}: line {exc.lineno}, column {exc.colno}: {exc.msg}"
        ) from exc


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    path = Path(path)
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise RuntimeError(
                    f"invalid JSONL in {path} at line {line_number}: {exc.msg}"
                ) from exc
            if not isinstance(value, dict):
                raise RuntimeError(
                    f"{path}:{line_number}: expected JSON object, got {type(value).__name__}"
                )
            rows.append(value)
    return rows


def _require_mapping(value: Any, *, description: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise RuntimeError(f"{description} must be an object, got {type(value).__name__}")
    return value


def _require_float(value: Any, *, description: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"{description} must be numeric, got {value!r}") from exc
    if not math.isfinite(result):
        raise RuntimeError(f"{description} must be finite, got {value!r}")
    return result


def _validate_bench(value: Any, *, source: Path) -> Mapping[str, Any]:
    bench = _require_mapping(value, description=f"{source} root")
    schema_version = bench.get("schema_version")
    if schema_version != BENCH_SCHEMA_VERSION:
        raise RuntimeError(
            f"{source}: unsupported bench schema_version {schema_version!r}; "
            f"expected {BENCH_SCHEMA_VERSION}"
        )
    params = _require_mapping(bench.get("params"), description=f"{source} params")
    for key in ("n_prompt", "n_gen", "repetitions"):
        try:
            number = int(params[key])
        except (KeyError, TypeError, ValueError) as exc:
            raise RuntimeError(f"{source}: params.{key} must be an integer") from exc
        if number <= 0:
            raise RuntimeError(f"{source}: params.{key} must be > 0")
    configs = _require_mapping(bench.get("configs"), description=f"{source} configs")
    for config_name in BENCH_CONFIGS:
        config = _require_mapping(
            configs.get(config_name), description=f"{source} configs.{config_name}"
        )

        expected_role = BENCH_CONFIG_BUILD_ROLES[config_name]
        actual_role = config.get("build_role")
        if actual_role != expected_role:
            raise RuntimeError(
                f"{source}: configs.{config_name}.build_role must be {expected_role!r}, "
                f"got {actual_role!r}"
            )
        build_identity = _require_mapping(
            config.get("build_identity"), description=f"{source} configs.{config_name}.build_identity"
        )
        missing = [
            key for key in BENCH_BUILD_IDENTITY_REQUIRED_KEYS if key not in build_identity
        ]
        if missing:
            raise RuntimeError(
                f"{source}: configs.{config_name}.build_identity is missing required "
                f"field(s) {missing!r}"
            )

        metrics = _require_mapping(
            config.get("metrics"), description=f"{source} configs.{config_name}.metrics"
        )
        for workload in BENCH_WORKLOADS:
            metric = _require_mapping(
                metrics.get(workload),
                description=f"{source} configs.{config_name}.metrics.{workload}",
            )
            _require_float(
                metric.get("avg_ts"),
                description=f"{source} configs.{config_name}.metrics.{workload}.avg_ts",
            )
            _require_float(
                metric.get("stddev_ts"),
                description=f"{source} configs.{config_name}.metrics.{workload}.stddev_ts",
            )
    return bench


def _metric(bench: Mapping[str, Any], config_name: str, workload: str) -> tuple[float, float]:
    configs = _require_mapping(bench["configs"], description="bench configs")
    config = _require_mapping(configs[config_name], description=f"bench config {config_name}")
    metrics = _require_mapping(config["metrics"], description=f"bench {config_name} metrics")
    metric = _require_mapping(
        metrics[workload], description=f"bench {config_name} {workload} metric"
    )
    return (
        _require_float(metric["avg_ts"], description=f"{config_name}.{workload}.avg_ts"),
        _require_float(metric["stddev_ts"], description=f"{config_name}.{workload}.stddev_ts"),
    )


def _delta_pct(value: float, baseline: float) -> float | None:
    if baseline == 0.0:
        return None
    return ((value / baseline) - 1.0) * 100.0


def _format_rate(avg_ts: float, stddev_ts: float) -> str:
    return f"{avg_ts:.2f} ± {stddev_ts:.2f}"


def _format_delta(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:+.2f}%"


def _format_improvement(value: Any) -> str:
    if value is None:
        return "—"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return _md_cell(value)
    if not math.isfinite(number):
        return _md_cell(value)
    return f"{number:+.2f}%"


def _md_cell(value: Any) -> str:
    if value is None:
        return "—"
    if isinstance(value, (dict, list, tuple)):
        text = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    else:
        text = str(value)
    text = text.strip()
    if not text:
        return "—"
    text = text.replace("\\", "\\\\")
    text = text.replace("|", "\\|")
    text = text.replace("\r\n", "<br>")
    text = text.replace("\n", "<br>")
    text = text.replace("\r", "<br>")
    return text


def _merge_decision_fields(row: Mapping[str, Any]) -> dict[str, Any]:
    merged = dict(row)
    for key in ("result", "decision", "promotion", "promotion_decision"):
        nested = row.get(key)
        if isinstance(nested, Mapping):
            for nested_key, nested_value in nested.items():
                merged[nested_key] = nested_value
    return merged


def _first_present(row: Mapping[str, Any], keys: Iterable[str]) -> Any:
    for key in keys:
        if key not in row:
            continue
        value = row[key]
        if value is None:
            continue
        if isinstance(value, str) and not value.strip():
            continue
        return value
    return None


def _dispatch_digest(row: Mapping[str, Any]) -> Any:
    direct = _first_present(
        row, ("dispatch_digest", "dispatch_sha256", "dispatch_hash", "cache_digest", "dispatch")
    )
    if direct is not None and not isinstance(direct, Mapping):
        return direct
    dispatch = row.get("dispatch")
    if isinstance(dispatch, Mapping):
        nested = _first_present(dispatch, ("digest", "dispatch_digest", "sha256", "hash"))
        if nested is not None:
            return nested
        return dispatch
    return direct


def _winner(row: Mapping[str, Any]) -> Any:
    return _first_present(row, ("winner", "promoted_winner", "deployed_winner"))


def _provisional_winner(row: Mapping[str, Any]) -> Any:
    return _first_present(
        row, ("provisional_winner", "statistical_winner", "candidate_winner")
    )


def _promotion_status(row: Mapping[str, Any]) -> Any:
    return _first_present(row, ("promotion_status", "status"))


def _decision_rows(measurements: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    decisions: list[dict[str, Any]] = []
    for original in measurements:
        row = _merge_decision_fields(original)
        status = _promotion_status(row)
        if status is None:
            continue
        decisions.append({
            "improvement_pct": _first_present(row, ("improvement_pct", "speedup_pct")),
            "dispatch_digest": _dispatch_digest(row),
            "winner": _winner(row),
            "provisional_winner": _provisional_winner(row),
            "promotion_status": status,
            "reason": _first_present(row, ("reason", "promotion_reason", "rejection_reason")),
        })
    return decisions


def _render_benchmark_table(bench: Mapping[str, Any]) -> list[str]:
    params = _require_mapping(bench["params"], description="bench params")
    prompt = int(params["n_prompt"])
    gen = int(params["n_gen"])
    lines = [
        "## Benchmark comparison",
        "",
        "| Workload | Stock tok/s | Native tok/s | Native vs stock | Replay tok/s | Replay vs stock |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    workload_labels = {"pp": f"pp{prompt}", "tg": f"tg{gen}"}
    for workload in BENCH_WORKLOADS:
        stock_avg, stock_stddev = _metric(bench, "stock", workload)
        native_avg, native_stddev = _metric(bench, "native", workload)
        replay_avg, replay_stddev = _metric(bench, "replay", workload)
        native_delta = _delta_pct(native_avg, stock_avg)
        replay_delta = _delta_pct(replay_avg, stock_avg)
        lines.append(
            "| " + " | ".join((
                workload_labels[workload],
                _format_rate(stock_avg, stock_stddev),
                _format_rate(native_avg, native_stddev),
                _format_delta(native_delta),
                _format_rate(replay_avg, replay_stddev),
                _format_delta(replay_delta),
            )) + " |"
        )
    lines.append("")
    return lines


def _render_promotion_summary(decisions: Sequence[Mapping[str, Any]]) -> list[str]:
    lines = ["## Promotion decisions", ""]
    if not decisions:
        lines.extend([
            "No rows containing `promotion_status` were found in `measurements.jsonl`.",
            "",
        ])
        return lines
    counts = Counter(str(row["promotion_status"]) for row in decisions)
    lines.extend([
        "### Status counts", "", "| promotion_status | Count |", "| --- | ---: |",
    ])
    for status, count in sorted(counts.items(), key=lambda item: item[0]):
        lines.append(f"| {_md_cell(status)} | {count} |")
    lines.extend([
        "",
        "### Decision rows",
        "",
        "A `provisional_winner` is shown separately from the selected `winner`. "
        "The report never treats a provisional statistical winner as deployed; "
        "`promotion_status`, `winner`, and `reason` record whether the candidate "
        "actually passed the promotion gates.",
        "",
        "| Improvement | Dispatch digest | Winner | Provisional winner | Promotion status | Reason |",
        "| ---: | --- | --- | --- | --- | --- |",
    ])
    for row in decisions:
        lines.append(
            "| " + " | ".join((
                _format_improvement(row.get("improvement_pct")),
                _md_cell(row.get("dispatch_digest")),
                _md_cell(row.get("winner")),
                _md_cell(row.get("provisional_winner")),
                _md_cell(row.get("promotion_status")),
                _md_cell(row.get("reason")),
            )) + " |"
        )
    lines.append("")
    return lines


def _render_report(bench: Mapping[str, Any], measurements: Sequence[Mapping[str, Any]]) -> str:
    model = bench.get("model")
    params = _require_mapping(bench["params"], description="bench params")
    decisions = _decision_rows(measurements)
    lines = [
        "# BigCherry end-to-end smoke campaign report",
        "",
        f"**Model:** `{_md_cell(model)}`",
        "",
        (
            "**Benchmark:** "
            f"pp={int(params['n_prompt'])}, tg={int(params['n_gen'])}, "
            f"repetitions={int(params['repetitions'])}"
        ),
        "",
    ]
    lines.extend(_render_benchmark_table(bench))
    lines.extend(_render_promotion_summary(decisions))
    return "\n".join(lines).rstrip() + "\n"


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent, text=True,
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise


def generate_report(
    workdir: Path, *, bench_path: Path | None = None,
    measurements_path: Path | None = None, output_path: Path | None = None,
) -> Path:
    """Generate report.md for one e2e smoke campaign.

    bench_path/measurements_path/output_path default to
    <workdir>/{bench.json,measurements.jsonl,report.md} but the campaign's
    real measurements file is named tune.jsonl.measurements.jsonl, so
    Campaign.s7_report() always passes measurements_path explicitly.
    """
    workdir = Path(workdir)
    bench_path = Path(bench_path) if bench_path is not None else workdir / "bench.json"
    measurements_path = (
        Path(measurements_path) if measurements_path is not None
        else workdir / "measurements.jsonl"
    )
    output_path = Path(output_path) if output_path is not None else workdir / "report.md"

    bench = _validate_bench(_load_json(bench_path), source=bench_path)
    measurements = _load_jsonl(measurements_path)
    report = _render_report(bench, measurements)
    _atomic_write_text(output_path, report)
    return output_path


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate a human-readable report for one BigCherry end-to-end smoke campaign."
    )
    parser.add_argument("workdir", type=Path, help="Campaign work directory.")
    parser.add_argument("--bench", type=Path, default=None)
    parser.add_argument("--measurements", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)
    output = generate_report(
        args.workdir, bench_path=args.bench, measurements_path=args.measurements,
        output_path=args.output,
    )
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
