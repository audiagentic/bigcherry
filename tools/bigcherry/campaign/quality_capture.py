"""Quality measurement inside the server comparison (MEN11): each arm scored against a recorded reference.

The comparison config may carry a ``quality`` block. For every arm the engine adapter says how to run its KL
measurement (``EngineAdapter.quality_command``); the arm's sidecars (expert helpers) run during it, exactly as in a
serving cell. The result is recorded per arm in ``run.json`` under ``quality``.

Who judges the result is the block's ``gate``:

* ``"control"`` (default): the first arm (or ``quality.control``) is the control and every other arm is gated against
  it with the block's ``thresholds`` (the six :class:`QualityThresholds` fields, required).
* ``"baselines"`` (MEN17): no arm is compared with another; the run only measures, and the numbers are judged against
  stored baselines (``bigcherry accept``). ``thresholds`` is not allowed. A control that reproduces its own reference
  exactly (KL 0, top-1 100%) therefore cannot produce a false top-1 drop. ``control`` then only names the arm whose
  binary and environment record the reference.

Block keys: ``reference_dir`` (required), ``args`` (engine arguments of the KL run, required), ``gate``, ``thresholds``,
``control`` (arm name, default first arm), ``timeout_s``, and one of

* ``corpus`` + ``record``: record the reference first (``record.args`` and ``record.environment`` replace the arm's
  layout, no settings or sidecars; ``reference_dir`` must not hold a reference yet);
* ``reference_metadata``: a STORED reference. ``reference_dir`` already holds it, and this JSON file beside it says
  where it came from (required keys ``pin``, ``layout``, ``recorded_on``). Its content is copied into the ``quality``
  section of ``run.json`` so a result always names the pin and layout its reference was recorded on;
* neither: the reference must already exist (no provenance recorded).
"""

from __future__ import annotations

import dataclasses
import json
import os
import subprocess
from pathlib import Path
from typing import Any

from ..engine_adapters import EngineAdapter, QualityCommand, SidecarError, SidecarSet
from ..experiment.engine_quality import (
    KldResult, QualityParseError, QualityThresholds, parse_radiance_kld_json, quality_gate,
)

_KEYS = {"reference_dir", "args", "thresholds", "control", "timeout_s", "corpus", "record", "gate", "reference_metadata"}
GATES = ("control", "baselines")
#: what a stored reference's metadata file must say
REFERENCE_METADATA_KEYS = ("pin", "layout", "recorded_on")


def validate_quality_block(quality: Any, names: list[str]) -> dict[str, Any]:
    """The block with its paths expanded, or ValueError. Does not touch the filesystem."""
    if not isinstance(quality, dict):
        raise ValueError("quality must be an object")
    unknown = sorted(set(quality) - _KEYS)
    if unknown:
        raise ValueError(f"quality: unknown key(s): {', '.join(unknown)}")
    for key in ("reference_dir", "args"):
        if key not in quality:
            raise ValueError(f"quality requires {key}")
    if not isinstance(quality["args"], list) or not all(isinstance(a, str) for a in quality["args"]):
        raise ValueError("quality.args must be a list of strings")
    gate = quality.get("gate", "control")
    if gate not in GATES:
        raise ValueError(f"quality.gate must be one of {', '.join(GATES)}")
    if gate == "baselines":
        if "thresholds" in quality:
            raise ValueError("quality.thresholds is not used with gate 'baselines'; the baseline files carry the tolerances")
    else:
        if "thresholds" not in quality:
            raise ValueError("quality requires thresholds (or gate 'baselines')")
        fields = {f.name for f in dataclasses.fields(QualityThresholds)}
        thresholds = quality["thresholds"]
        if not isinstance(thresholds, dict) or set(thresholds) != fields or not all(
                isinstance(v, (int, float)) and not isinstance(v, bool) and v >= 0 for v in thresholds.values()):
            raise ValueError(f"quality.thresholds must give exactly {', '.join(sorted(fields))} as non-negative numbers")
    if "reference_metadata" in quality:
        if "record" in quality:
            raise ValueError("quality.reference_metadata names a stored reference; it cannot be combined with record")
        if not isinstance(quality["reference_metadata"], str) or not quality["reference_metadata"]:
            raise ValueError("quality.reference_metadata must be a path")
    if quality.get("control", names[0]) not in names:
        raise ValueError(f"quality.control must be one of {', '.join(names)}")
    if ("record" in quality) != ("corpus" in quality):
        raise ValueError("quality.record and quality.corpus go together (record the reference) or neither (reuse it)")
    expanded = dict(quality)
    expanded["gate"] = gate
    if "reference_metadata" in quality:
        expanded["reference_metadata"] = os.path.expandvars(quality["reference_metadata"])
    expanded["reference_dir"] = os.path.expandvars(quality["reference_dir"])
    expanded["args"] = [os.path.expandvars(a) for a in quality["args"]]
    expanded.setdefault("control", names[0])
    if "record" in quality:
        record = quality["record"]
        if not isinstance(record, dict) or set(record) - {"args", "environment"} or not isinstance(
                record.get("args"), list):
            raise ValueError("quality.record must be {args: [...], environment: {...}}")
        expanded["corpus"] = os.path.expandvars(quality["corpus"])
        expanded["record"] = {
            "args": [os.path.expandvars(a) for a in record["args"]],
            "environment": {k: os.path.expandvars(v) for k, v in record.get("environment", {}).items()},
        }
    return expanded


def load_reference_metadata(quality: dict[str, Any]) -> dict[str, Any] | None:
    """The stored reference's provenance (pin, layout, recorded_on, ...), or None when the block names none."""
    if "reference_metadata" not in quality:
        return None
    path = Path(quality["reference_metadata"])
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ValueError(f"quality.reference_metadata {path} is unreadable: {exc}") from exc
    missing = [key for key in REFERENCE_METADATA_KEYS
               if not isinstance(document, dict) or not isinstance(document.get(key), str) or not document[key]]
    if missing:
        raise ValueError(f"quality.reference_metadata {path} must give {', '.join(missing)} (the pin and layout the "
                         "reference was recorded on)")
    return document


def check_reference_state(quality: dict[str, Any]) -> None:
    """A recorded reference is never overwritten; a reused one must be there (a stored one says where it came from)."""
    load_reference_metadata(quality)
    reference = Path(quality["reference_dir"])
    present = reference.is_dir() and any(reference.iterdir())
    if "record" in quality and present:
        raise ValueError(f"quality.reference_dir {reference} already holds a reference; drop record or use a new directory")
    if "record" not in quality and not present:
        raise ValueError(f"quality.reference_dir {reference} is missing or empty; add record + corpus to record it")


def _run(command: QualityCommand) -> dict[str, Any]:
    command.log_path.parent.mkdir(parents=True, exist_ok=True)
    record: dict[str, Any] = {"argv": list(command.argv), "log": str(command.log_path)}
    with command.log_path.open("wb") as log:
        try:
            done = subprocess.run(
                list(command.argv), env=command.env, stdout=log, stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL, timeout=command.timeout_s,
            )
            record["returncode"] = done.returncode
        except subprocess.TimeoutExpired:
            record["returncode"] = None
            record["error"] = f"quality run exceeded {command.timeout_s:g}s"
        except OSError as exc:
            record["returncode"] = None
            record["error"] = f"cannot start {command.argv[0]}: {exc}"
    return record


def run_quality_capture(
    *, adapter: EngineAdapter, quality: dict[str, Any], prepared: dict[str, Any], names: list[str],
    output: Path, reference_env: dict[str, str],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """(the ``quality`` section of run.json, one pseudo-cell per arm for activation evidence). Never raises on a
    failed measurement: the failure is recorded and ``passed`` is false."""
    qdir = output / "quality"
    qdir.mkdir(parents=True, exist_ok=True)
    section: dict[str, Any] = {
        "reference_dir": quality["reference_dir"], "control": quality["control"], "gate": quality["gate"],
        "thresholds": quality.get("thresholds"), "reference_metadata": load_reference_metadata(quality),
        "recorded_reference": None, "arms": {}, "passed": False,
    }
    cells: list[dict[str, Any]] = []
    control = prepared[quality["control"]]
    if "record" in quality:
        record_quality = {**quality, "args": quality["record"]["args"]}
        # The engine writes its reference files into this directory; it does not create it.
        Path(quality["reference_dir"]).mkdir(parents=True, exist_ok=True)
        command = adapter.quality_command(
            "record", {"name": "reference", "model": control["model"], "settings": {}},
            {**reference_env, **quality["record"]["environment"]}, qdir, control["binary"], control["model"],
            record_quality)
        recorded = _run(command)
        section["recorded_reference"] = recorded
        if recorded.get("returncode") != 0:
            section["error"] = "reference recording failed"
            return section, cells

    results: dict[str, KldResult] = {}
    for name in names:
        arm = prepared[name]
        arm_dict = {"name": name, "model": arm["model"], "settings": dict(arm["settings"])}
        entry: dict[str, Any] = {"settings": dict(arm["settings"])}
        section["arms"][name] = entry
        command = adapter.quality_command("score", arm_dict, arm["env"], qdir, arm["binary"], arm["model"], quality)
        sidecars = SidecarSet(adapter.sidecars(arm_dict, arm["env"], qdir, arm["binary"]))
        try:
            sidecars.start()
            entry.update(_run(command))
        except SidecarError as exc:
            entry["error"] = str(exc)
        finally:
            entry["sidecars"] = sidecars.stop()
        bad = [r["name"] for r in entry["sidecars"] if r["killed"] or r["exited_before_stop"]]
        if bad and "error" not in entry:
            entry["error"] = f"sidecar(s) {', '.join(bad)} died or had to be killed"
        if entry.get("returncode") not in (None, 0):
            entry.setdefault("error", f"quality run exited {entry['returncode']}")
        if "error" not in entry:
            try:
                document = json.loads(command.result_path.read_text(encoding="utf-8"))
                results[name] = parse_radiance_kld_json(document)
                entry["result"] = dataclasses.asdict(results[name])
                entry["all"] = document["all"]
                entry["by_source"] = document.get("by_source")
            except (OSError, ValueError, QualityParseError) as exc:
                entry["error"] = f"no usable KL result: {exc}"
        cells.append({"mode": name, "returncode": 0 if "error" not in entry else 1,
                      "server_log": str(command.log_path), "sidecars": entry["sidecars"]})

    passed = all("error" not in entry for entry in section["arms"].values())
    if quality["gate"] == "control" and quality["control"] in results:
        limits = QualityThresholds(**quality["thresholds"])
        for name, entry in section["arms"].items():
            if name == quality["control"] or name not in results:
                continue
            ok, problems = quality_gate(results[quality["control"]], results[name], limits)
            entry["gate"] = {"passed": ok, "problems": problems}
            passed = passed and ok
    section["passed"] = passed
    return section, cells
