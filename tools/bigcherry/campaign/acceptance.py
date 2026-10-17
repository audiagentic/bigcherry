"""Post-bump hardware acceptance against stored baselines (MEN17): ``bigcherry --engine <name> accept``.

An engine declares acceptance entries in ``engine.toml`` ([[acceptance]]): a server-comparison configuration plus the
baseline files its arms must hold. Each entry is run through the maintained server-comparison path
(:func:`bigcherry.campaign.benchmark.run_server_comparison_capture`, speed cells and the quality block alike), and the
numbers in its ``run.json`` are compared with the baselines instead of being read by eye.

A baseline file (``docs/reference/engines/<engine>/baselines/<name>.json``)::

    {"schema": 1, "config": "<repo path of the comparison config>", "pin": "1.7.0", "run": "<run id>",
     "date": "2026-10-17", "baselines": [
       {"arm": "rows0", "metric": "pp4096_tps", "value": 2049.5, "direction": "higher-better",
        "tolerance": {"relative": 0.03}}, ...]}

``tolerance`` is ``{"relative": f}`` (a fraction of the value) or ``{"absolute": a}``, taken from the measured noise.
It is one-sided: a higher-better metric fails below ``value - tolerance``, a lower-better one above ``value +
tolerance``. Doing better than the baseline is never a failure.

A metric is a serving metric (``pp4096_tps``: the mean over the arm's clean cells, as the comparison summarises it) or
a quality metric (a KL result field: ``kld_mean``, ``kld_median``, ``kld_p99``, ``kld_p999``, ``top1_percent``,
``ppl``, ``ppl_reference``, or ``ppl_ratio``) read from ``run.json`` ``quality.arms.<arm>.result``.
"""

from __future__ import annotations

import contextlib
import dataclasses
import json
import os
import statistics
import time
from pathlib import Path
from typing import Any, Iterator

from ..core import engines

SCHEMA = 1
DIRECTIONS = ("higher-better", "lower-better")
#: quality metrics: the fields of a KL result, plus the ratio derived from two of them
QUALITY_METRICS = ("kld_mean", "kld_median", "kld_p99", "kld_p999", "top1_percent", "ppl", "ppl_reference", "ppl_ratio")
EVIDENCE_NAME = "acceptance.json"


class AcceptanceError(ValueError):
    """The acceptance declaration, a baseline file or a config is missing or malformed; nothing was measured."""


@dataclasses.dataclass(frozen=True)
class Baseline:
    arm: str
    metric: str
    value: float
    direction: str
    tolerance_kind: str   # "absolute" | "relative"
    tolerance: float

    @property
    def allowance(self) -> float:
        """The tolerance as an absolute amount."""
        return abs(self.value) * self.tolerance if self.tolerance_kind == "relative" else self.tolerance

    @property
    def limit(self) -> float:
        """The worst measured value that still passes."""
        return self.value - self.allowance if self.direction == "higher-better" else self.value + self.allowance

    def passes(self, measured: float) -> bool:
        # A hair of slack so that a value exactly on the limit is not failed by float rounding (0.0642 + 0.01).
        slack = 1e-9 * max(1.0, abs(self.limit))
        return measured >= self.limit - slack if self.direction == "higher-better" else measured <= self.limit + slack


@dataclasses.dataclass(frozen=True)
class BaselineSet:
    path: str
    config: str
    pin: str
    run: str
    date: str
    baselines: tuple[Baseline, ...]


def _number(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and value == value and abs(value) != float("inf")


def parse_baselines(document: Any, path: str) -> BaselineSet:
    """A baseline file's content as a :class:`BaselineSet`, or AcceptanceError."""
    if not isinstance(document, dict) or document.get("schema") != SCHEMA:
        raise AcceptanceError(f"{path}: baseline file needs schema {SCHEMA}")
    unknown = sorted(set(document) - {"schema", "config", "pin", "run", "date", "baselines", "note"})
    if unknown:
        raise AcceptanceError(f"{path}: unknown key(s): {', '.join(unknown)}")
    for key in ("config", "pin", "run", "date"):
        if not isinstance(document.get(key), str) or not document[key]:
            raise AcceptanceError(f"{path}: {key} must be a non-empty string")
    rows = document.get("baselines")
    if not isinstance(rows, list) or not rows:
        raise AcceptanceError(f"{path}: baselines must be a non-empty list")
    parsed: list[Baseline] = []
    for index, row in enumerate(rows):
        where = f"{path}: baselines[{index}]"
        if not isinstance(row, dict):
            raise AcceptanceError(f"{where} must be an object")
        extra = sorted(set(row) - {"arm", "metric", "value", "direction", "tolerance", "note"})
        if extra:
            raise AcceptanceError(f"{where}: unknown key(s): {', '.join(extra)}")
        for key in ("arm", "metric"):
            if not isinstance(row.get(key), str) or not row[key]:
                raise AcceptanceError(f"{where}: {key} must be a non-empty string")
        if not _number(row.get("value")):
            raise AcceptanceError(f"{where}: value must be a finite number")
        if row.get("direction") not in DIRECTIONS:
            raise AcceptanceError(f"{where}: direction must be one of {', '.join(DIRECTIONS)}")
        tolerance = row.get("tolerance")
        if not isinstance(tolerance, dict) or len(tolerance) != 1 or next(iter(tolerance)) not in ("absolute", "relative"):
            raise AcceptanceError(f"{where}: tolerance must be exactly one of absolute or relative")
        (kind, amount), = tolerance.items()
        if not _number(amount) or amount < 0:
            raise AcceptanceError(f"{where}: tolerance.{kind} must be a non-negative number")
        parsed.append(Baseline(row["arm"], row["metric"], float(row["value"]), row["direction"], kind, float(amount)))
    keys = [(b.arm, b.metric) for b in parsed]
    if len(set(keys)) != len(keys):
        raise AcceptanceError(f"{path}: an arm and metric pair is baselined twice")
    return BaselineSet(path, document["config"], document["pin"], document["run"], document["date"], tuple(parsed))


def load_baselines(repo_root: Path, relative: str) -> BaselineSet:
    path = repo_root / relative
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise AcceptanceError(f"baseline file {relative} does not exist") from exc
    except (OSError, ValueError) as exc:
        raise AcceptanceError(f"baseline file {relative} is unreadable: {exc}") from exc
    return parse_baselines(document, relative)


@dataclasses.dataclass
class Row:
    entry: str
    arm: str
    metric: str
    baseline: float
    measured: float | None
    limit: float
    direction: str
    passed: bool
    detail: str = ""

    @property
    def delta(self) -> float | None:
        return None if self.measured is None else self.measured - self.baseline

    def document(self) -> dict[str, Any]:
        return {**dataclasses.asdict(self), "delta": self.delta}


def measured_value(run: dict[str, Any], arm: str, metric: str) -> float | None:
    """The arm's value of the metric in a comparison's ``run.json``, or None when the run did not measure it."""
    if metric in QUALITY_METRICS:
        result = (((run.get("quality") or {}).get("arms") or {}).get(arm) or {}).get("result")
        if not isinstance(result, dict):
            return None
        if metric == "ppl_ratio":
            ppl, reference = result.get("ppl"), result.get("ppl_reference")
            return float(ppl) / float(reference) if _number(ppl) and _number(reference) and reference else None
        value = result.get(metric)
        return float(value) if _number(value) else None
    from .benchmark import _arm_metric

    values = [
        v for cell in run.get("runs", [])
        if cell.get("mode") == arm and cell.get("returncode") == 0 and (v := _arm_metric(cell, metric)) is not None
    ]
    return statistics.mean(values) if values else None


def compare(entry: str, baselines: BaselineSet, run: dict[str, Any]) -> list[Row]:
    rows: list[Row] = []
    for base in baselines.baselines:
        measured = measured_value(run, base.arm, base.metric)
        if measured is None:
            rows.append(Row(entry, base.arm, base.metric, base.value, None, base.limit, base.direction, False,
                            "not measured by this run"))
        else:
            rows.append(Row(entry, base.arm, base.metric, base.value, measured, base.limit, base.direction,
                            base.passes(measured)))
    return rows


@dataclasses.dataclass
class EntryResult:
    name: str
    config: str
    output: str
    returncode: int | None
    error: str
    rows: list[Row]
    baselines: list[dict[str, str]]

    @property
    def passed(self) -> bool:
        return self.returncode == 0 and not self.error and bool(self.rows) and all(r.passed for r in self.rows)

    def document(self) -> dict[str, Any]:
        return {"name": self.name, "config": self.config, "output": self.output, "returncode": self.returncode,
                "error": self.error, "passed": self.passed, "baselines": self.baselines,
                "rows": [r.document() for r in self.rows]}


def select_entries(layout: engines.EngineLayout, only: list[str] | None) -> list[engines.AcceptanceEntry]:
    if not layout.acceptance:
        raise AcceptanceError(f"engine {layout.name!r} declares no [[acceptance]] entries")
    if not only:
        return list(layout.acceptance)
    known = {e.name: e for e in layout.acceptance}
    unknown = sorted(set(only) - set(known))
    if unknown:
        raise AcceptanceError(f"engine {layout.name!r} has no acceptance entry {', '.join(unknown)} "
                              f"(declared: {', '.join(known)})")
    return [known[name] for name in dict.fromkeys(only)]


def prepare(repo_root: Path, layout: engines.EngineLayout, only: list[str] | None
            ) -> list[tuple[engines.AcceptanceEntry, Path, list[BaselineSet]]]:
    """Every selected entry with its config path and parsed baselines. Raises before anything runs."""
    prepared = []
    for entry in select_entries(layout, only):
        config = repo_root / entry.config
        if not config.is_file():
            raise AcceptanceError(f"acceptance {entry.name!r}: config {entry.config} does not exist")
        try:
            declared_engine = json.loads(config.read_text(encoding="utf-8")).get("engine")
        except (OSError, ValueError, AttributeError) as exc:
            raise AcceptanceError(f"acceptance {entry.name!r}: config {entry.config} is unreadable: {exc}") from exc
        if declared_engine != layout.name:
            raise AcceptanceError(
                f"acceptance {entry.name!r}: config {entry.config} runs engine {declared_engine!r}, not {layout.name!r}")
        sets = [load_baselines(repo_root, relative) for relative in entry.baselines]
        for found in sets:
            if found.config != entry.config:
                raise AcceptanceError(
                    f"baseline file {found.path} is for config {found.config}, not {entry.config} (entry {entry.name!r})")
        prepared.append((entry, config, sets))
    return prepared


@contextlib.contextmanager
def _entry_environment(entry: engines.AcceptanceEntry, entry_dir: Path) -> Iterator[None]:
    """The entry's declared variables, unless the caller already set them."""
    added: list[str] = []
    try:
        for key, value in entry.env:
            if key not in os.environ:
                os.environ[key] = value.replace("{entry-dir}", str(entry_dir))
                added.append(key)
        yield
    finally:
        for key in added:
            os.environ.pop(key, None)


def run_entry(entry: engines.AcceptanceEntry, config: Path, sets: list[BaselineSet], output: Path) -> EntryResult:
    """Run one entry through the server-comparison capture and compare it with its baselines."""
    from .benchmark import run_server_comparison_capture

    entry_dir = output / entry.name
    run_dir = entry_dir / "run"
    returncode: int | None = None
    error = ""
    entry_dir.mkdir(parents=True, exist_ok=True)
    with _entry_environment(entry, entry_dir):
        try:
            returncode = run_server_comparison_capture(
                config, run_dir, rounds=entry.rounds, seed=0, settle_seconds=entry.settle_seconds)
        except (ValueError, OSError, KeyError) as exc:
            error = f"{type(exc).__name__}: {exc}"
    run: dict[str, Any] = {}
    run_json = run_dir / "run.json"
    if run_json.is_file():
        try:
            run = json.loads(run_json.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            error = error or f"run.json is unreadable: {exc}"
    if returncode not in (0, None) and not error:
        error = f"the comparison run exited {returncode} (see {run_json})"
    rows = [row for found in sets for row in compare(entry.name, found, run)]
    return EntryResult(
        entry.name, entry.config, str(run_dir), returncode, error, rows,
        [{"file": s.path, "pin": s.pin, "run": s.run, "date": s.date} for s in sets])


def render(results: list[EntryResult]) -> str:
    header = ("entry", "arm", "metric", "baseline", "measured", "delta", "limit", "result")
    table = [header]
    for result in results:
        for row in result.rows:
            table.append((
                result.name, row.arm, row.metric, f"{row.baseline:g}",
                "-" if row.measured is None else f"{row.measured:.6g}",
                "-" if row.delta is None else f"{row.delta:+.4g}",
                ("min " if row.direction == "higher-better" else "max ") + f"{row.limit:.6g}",
                "PASS" if row.passed else ("FAIL" + (f" ({row.detail})" if row.detail else "")),
            ))
        if result.error or result.returncode not in (0, None):
            table.append((result.name, "-", "(run)", "-", "-", "-", "-", f"FAIL ({result.error or result.returncode})"))
    widths = [max(len(line[i]) for line in table) for i in range(len(header))]
    return "\n".join("  ".join(cell.ljust(widths[i]) for i, cell in enumerate(line)).rstrip() for line in table)


def accept(
    repo_root: Path, layout: engines.EngineLayout, only: list[str] | None, output: Path,
) -> tuple[bool, list[EntryResult], Path]:
    """Run the selected entries; (all passed, their results, the evidence file). Raises AcceptanceError first when a
    declaration, config or baseline file is bad, before any hardware is used."""
    prepared = prepare(repo_root, layout, only)
    output.mkdir(parents=True, exist_ok=True)
    started = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    results = [run_entry(entry, config, sets, output) for entry, config, sets in prepared]
    passed = all(r.passed for r in results)
    evidence = output / EVIDENCE_NAME
    evidence.write_text(json.dumps({
        "schema": SCHEMA, "engine": layout.name, "started": started, "passed": passed,
        "entries": [r.document() for r in results],
    }, indent=2) + "\n", encoding="utf-8")
    return passed, results, evidence
