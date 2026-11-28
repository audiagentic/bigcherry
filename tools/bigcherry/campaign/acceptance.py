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


@dataclasses.dataclass(frozen=True)
class CandidateRuntime:
    build_dir: Path
    build_plan_id: str
    binary_sha256: str
    upstream_revision: str

    def verify(self, layout: engines.EngineLayout) -> Path:
        from ..build.builds import binary_hash
        from .build import CampaignBuildError
        from .workers import _source_attestation, _verify_source

        binary = (self.build_dir / layout.serve.binary).resolve()
        try:
            meta = json.loads((self.build_dir / f"bigcherry-build-metadata-{binary.name}.json").read_text())
            if not self.upstream_revision:
                raise ValueError("candidate upstream revision missing")
            if not self.build_plan_id or meta["build_plan_id"] != self.build_plan_id:
                raise ValueError("candidate plan ID mismatch")
            if meta["binary_hash"] != self.binary_sha256 or binary_hash(binary) != self.binary_sha256:
                raise ValueError("candidate binary SHA-256 mismatch")
            cache = (self.build_dir / "CMakeCache.txt").read_text()
            source = next(Path(line.split("=", 1)[1]) for line in cache.splitlines()
                          if line.startswith("CMAKE_HOME_DIRECTORY:"))
            attestation = _source_attestation(source, upstream_revision=self.upstream_revision,
                                              source_slice_id=meta["source_slice_id"])
            if attestation is None:
                raise ValueError("candidate source attestation missing")
            _verify_source(source, attestation)
        except (OSError, KeyError, TypeError, ValueError, StopIteration, CampaignBuildError) as exc:
            raise AcceptanceError(f"candidate runtime verification failed: {exc}") from exc
        return binary


def validate_qualified_entries(layout: engines.EngineLayout, prepared: list, output: Path = Path(".")) -> None:
    """Radiance release acceptance must not use legacy diagnostic captures.

    Keep declaration parsing and the diagnostic benchmark runner available.
    A diagnostic/observe capture or a reference recorded by the candidate
    cannot qualify a release. Validate the whole selected cohort before
    running any entry.
    """
    if layout.name != engines.RADIANCE:
        return
    problems = []
    for entry, config, sets in prepared:
        doc = json.loads(config.read_text(encoding="utf-8"))
        quality = doc.get("quality")
        if any(b.metric in QUALITY_METRICS for group in sets for b in group.baselines) and not isinstance(quality, dict):
            problems.append(f"{entry.name}: quality baseline without quality block")
        if isinstance(quality, dict):
            if "record" in quality or "corpus" in quality:
                problems.append(f"{entry.name}: candidate-recorded quality reference")
            if not quality.get("reference_metadata"):
                problems.append(f"{entry.name}: missing independent reference metadata")
        arms = doc.get("arms")
        if not isinstance(arms, list) or not arms or any(
            not isinstance(arm, dict) or arm.get("binary") != "${RAD_BIN}" for arm in arms
        ):
            problems.append(f"{entry.name}: arms must use the bound candidate binary")
        else:
            bound_keys = {"RAD_BIN", "RAD_HOME", "RADIANCE_HOME"}
            controls = [dict(entry.env), doc.get("environment", {})]
            for arm in arms:
                controls.extend([arm.get("environment", {}), arm.get("settings", {})])
            if any(not isinstance(c, dict) or bound_keys.intersection(c) for c in controls):
                problems.append(f"{entry.name}: candidate runtime environment override")
        if doc.get("evidence_role") != "production" or doc.get("execution_evidence") != "required":
            problems.append(f"{entry.name}: diagnostic or unverified execution evidence")
    if problems:
        raise AcceptanceError("unqualified Radiance acceptance declarations: " + "; ".join(problems))
    from .quality_capture import check_reference_state, validate_quality_block
    for entry, config, _ in prepared:
        doc = json.loads(config.read_text(encoding="utf-8"))
        if "quality" in doc:
            try:
                with _entry_environment(entry, output / entry.name):
                    quality = validate_quality_block(doc["quality"], [arm["name"] for arm in doc["arms"]])
                    if Path(quality["reference_dir"]).resolve().is_relative_to(output.resolve()):
                        raise ValueError("reference must be independent of this output directory")
                    check_reference_state(quality)
            except (OSError, ValueError, KeyError) as exc:
                raise AcceptanceError(f"{entry.name}: independent quality reference not usable: {exc}") from exc


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


def run_entry(entry: engines.AcceptanceEntry, config: Path, sets: list[BaselineSet], output: Path,
              *, require_admitted: bool = False) -> EntryResult:
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
    if require_admitted and any(
            base.metric not in QUALITY_METRICS for found in sets for base in found.baselines):
        if run.get("performance_admitted") is not True:
            error = error or "performance capture is not admitted"
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
    *, candidate: CandidateRuntime | None = None,
) -> tuple[bool, list[EntryResult], Path]:
    """Run the selected entries; (all passed, their results, the evidence file). Raises AcceptanceError first when a
    declaration, config or baseline file is bad, before any hardware is used."""
    prepared = prepare(repo_root, layout, only)
    validate_qualified_entries(layout, prepared, output)
    if layout.name == engines.RADIANCE and candidate is None:
        raise AcceptanceError("Radiance acceptance requires a verified candidate lane receipt")
    binary = candidate.verify(layout) if candidate else None
    if candidate and layout.name == engines.RADIANCE and not (candidate.build_dir / layout.runtime_home).is_dir():
        raise AcceptanceError("candidate runtime home missing")
    output.mkdir(parents=True, exist_ok=True)
    started = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    bound = {} if binary is None or layout.name != engines.RADIANCE else {
        "RAD_BIN": str(binary), "RAD_HOME": str((candidate.build_dir / layout.runtime_home).resolve()),
        "RADIANCE_HOME": str((candidate.build_dir / layout.runtime_home).resolve()),
    }
    saved = {k: os.environ.get(k) for k in bound}
    try:
        os.environ.update(bound)
        results = [run_entry(entry, config, sets, output, require_admitted=layout.name == engines.RADIANCE)
                   for entry, config, sets in prepared]
    finally:
        for key, value in saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
    passed = all(r.passed for r in results)
    evidence = output / EVIDENCE_NAME
    evidence.write_text(json.dumps({
        "schema": SCHEMA, "engine": layout.name, "started": started, "passed": passed,
        "entries": [r.document() for r in results],
        **({"candidate": {"build_plan_id": candidate.build_plan_id, "binary_sha256": candidate.binary_sha256,
                          "upstream_revision": candidate.upstream_revision}} if candidate else {}),
    }, indent=2) + "\n", encoding="utf-8")
    return passed, results, evidence
