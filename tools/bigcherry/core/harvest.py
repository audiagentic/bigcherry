"""Harvest (WSL02): extract what we keep from a finished run into a small, hash-verified evidence bundle.

A run is finished when what we need from it is in a durable record. A RECIPE (versioned, declarative) names, per run kind,
the files that must exist, the files to keep, the derived summaries to compute from bulk BEFORE the bulk can ever go
(a kernel-time breakdown from a rocprof trace, helper-wait windows from helper logs), and the raw bulk. `harvest` copies the
kept files and runs the extractors into evidence-hold/<run-id>/, records every file's sha256 and size, an explicit inventory of
the bulk (so a later cleanup deletes exactly that list, not a glob), verifies the bundle by re-reading it, and only then marks the
run's harvest state `verified`. Nothing is deleted here. A run that failed, or that is marked suspect (for example measured while
the host was loaded), is harvested as a record of the failure and the bundle says `diagnostic_only`.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from . import run_manifest as rm

BUNDLE_NAME = "bundle.json"
BUNDLE_SCHEMA = 1


class HarvestError(RuntimeError):
    pass


@dataclass(frozen=True)
class Recipe:
    id: str
    required: tuple = ()         # globs (relative to the run dir) that must each match at least one file
    keep: tuple = ()             # globs copied into the bundle
    bulk: tuple = ()             # globs listed (with sizes) as raw bulk eligible for cleanup after harvest and retention
    extractors: tuple = ()       # extractor ids, run in order

    def sha256(self) -> str:
        return hashlib.sha256(json.dumps(self.__dict__, sort_keys=True, default=list).encode()).hexdigest()


_BENCH_KEEP = ("run.json", "advisories.json", "*.stream-bench.json", "stream-bench.json", "pair-*/cell.json",
               "pair-*/stream-bench.json", "pair-*/bench/*.json", "docs-note.md")
_BENCH_BULK = ("pair-*/server.log", "pair-*/helper*.log", "pair-*/bench/*.log")

RECIPES: dict[str, Recipe] = {r.id: r for r in (
    Recipe("ab/v1", required=("run.json",), keep=_BENCH_KEEP, bulk=_BENCH_BULK, extractors=("trace-windows/v1",)),
    Recipe("stream/v1", required=("pair-*/stream-bench.json",), keep=_BENCH_KEEP, bulk=_BENCH_BULK,
           extractors=("trace-windows/v1",)),
    Recipe("accept/v1", required=("acceptance.json",),
           keep=("acceptance.json", "*/run/run.json", "*/run/advisories.json", "*/run/pair-*/cell.json", "*/run/pair-*/stream-bench.json"),
           bulk=("*/run/pair-*/*.log", "*/kld-reference/*", "*/run/pair-*/bench/*")),
    Recipe("profile/v1", required=("**/*_kernel_stats.csv",),
           keep=("**/*_kernel_stats.csv", "**/*_agent_info.csv", "**/*.timings.json", "**/*.vram.txt", "**/*.greedy.txt"),
           bulk=("**/*_kernel_trace.csv", "**/*_memory_copy_trace.csv", "**/*.server.log"), extractors=("kernel-fraction/v1",)),
    Recipe("kl/v1", required=("**/*.json",), keep=("**/control.json", "**/kld.json", "**/*.rows", "**/tokens.i32", "**/top.bin"),
           bulk=("**/logp.f16", "**/*.log")),
    Recipe("replay/v1", keep=("*.json",), bulk=("*.log",)),
    Recipe("capture/v1", keep=("*.json",), bulk=()),
    Recipe("build/v1", keep=("*.json",), bulk=()),
    Recipe("bump/v1", keep=("*.json",), bulk=("*.log",)),
    Recipe("release/v1", keep=("*.json",), bulk=("*.log",)),
)}
DEFAULT_RECIPE = {"ab": "ab/v1", "stream": "stream/v1", "accept": "accept/v1", "profile": "profile/v1", "kl": "kl/v1",
                  "replay": "replay/v1", "capture": "capture/v1", "build": "build/v1", "bump": "bump/v1", "release": "release/v1"}


# ------------------------------------------------------------------------ extractors
# Each takes (run_dir, out_dir) and returns [(relative output path, producer string)]. Deterministic: same inputs, same bytes.

def _extract_trace_windows(run_dir: Path, out: Path) -> list:
    from ..bench import trace_windows

    made = []
    for rec in sorted(run_dir.glob("**/stream-bench.json")):
        summary = trace_windows.summarise_cell(rec.parent)
        if summary is None:
            continue
        rel = rec.parent.relative_to(run_dir)
        name = ("trace-windows." + "-".join(rel.parts) + ".json") if rel.parts else "trace-windows.json"
        (out / name).write_text(json.dumps(summary, indent=1, sort_keys=True) + "\n", encoding="utf-8")
        made.append((name, "extractor:trace-windows/v1"))
    return made


def _extract_kernel_fraction(run_dir: Path, out: Path) -> list:
    from ..analysis import kernel_fraction as kf

    made = []
    for csv in sorted(run_dir.glob("**/*_kernel_trace.csv")):
        trace = kf.parse_kernel_trace([csv])
        summary = kf.family_summary(trace)
        doc = {"schema": "kernel-fraction/v1", "source": csv.name, "source_bytes": csv.stat().st_size,
               "rows": len(trace["rows"]), "wall_ns": trace["wall_ns"], "columns": trace["columns"], "summary": summary}
        rel = csv.parent.relative_to(run_dir)
        name = "kernel-fraction." + "-".join((*rel.parts, csv.stem)) + ".json"
        (out / name).write_text(json.dumps(doc, indent=1, sort_keys=True, default=str) + "\n", encoding="utf-8")
        made.append((name, "extractor:kernel-fraction/v1"))
    return made


EXTRACTORS: dict[str, Callable] = {"trace-windows/v1": _extract_trace_windows, "kernel-fraction/v1": _extract_kernel_fraction}


# ------------------------------------------------------------------------ bundle
def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _contained(run_dir: Path, p: Path) -> bool:
    """True when no component between the run dir and the file (inclusive) is a symlink."""
    rel = p.relative_to(run_dir)
    cur = run_dir
    for part in rel.parts:
        cur = cur / part
        if cur.is_symlink():
            return False
    return True


def _matches(run_dir: Path, patterns) -> list:
    seen, out = set(), []
    for pat in patterns:
        for p in sorted(run_dir.glob(pat)):
            if p not in seen and p.is_file() and _contained(run_dir, p):
                seen.add(p)
                out.append(p)
    return out


def _bundle_id(run_id: str, recipe_sha: str, diagnostic: bool, files: list, bulk: list) -> str:
    payload = {"run_id": run_id, "recipe": recipe_sha, "diagnostic_only": diagnostic,
               "files": [(f["path"], f["bytes"], f["sha256"], f["producer"]) for f in files],
               "bulk": [(b["path"], b["bytes"]) for b in bulk]}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def bundle_dir(root: Path, run_id: str) -> Path:
    return Path(root) / "evidence-hold" / run_id


def harvest(root: Path, run_dir: Path, *, now: float | None = None) -> dict:
    """Harvest one run. Returns the bundle document. Raises HarvestError (and records `failed`) when it cannot be verified."""
    root, run_dir = Path(root), Path(run_dir)
    m = rm.read_manifest(run_dir)
    if m["state"] not in rm.TERMINAL_STATES:
        raise HarvestError(f"run {m['run_id']} is {m['state']}; only a finished run can be harvested")
    if m["harvest"]["state"] == "verified":
        raise HarvestError(f"run {m['run_id']} is already harvested (bundle {m['harvest']['bundle_id']})")
    recipe = RECIPES.get(m["harvest"].get("recipe") or DEFAULT_RECIPE[m["kind"]])
    if recipe is None:
        raise HarvestError(f"unknown harvest recipe {m['harvest'].get('recipe')!r}")
    gen = m["generation"]
    if m["harvest"]["state"] in ("pending", "failed"):
        m = rm.update_manifest(run_dir, expected_generation=gen, harvest_state="harvesting", patch={"harvest": {"recipe": recipe.id}}, now=now)
        gen = m["generation"]
    try:
        doc = _build(root, run_dir, m, recipe, now)
    except (HarvestError, OSError, ValueError, KeyError) as exc:
        rm.update_manifest(run_dir, expected_generation=gen, harvest_state="failed", patch={"harvest": {"error": str(exc)[:500]}}, now=now)
        raise HarvestError(str(exc)) from exc
    rm.update_manifest(run_dir, expected_generation=gen, harvest_state="verified",
                       patch={"harvest": {"bundle_id": doc["bundle_id"], "error": None}}, now=now)
    return doc


def _build(root: Path, run_dir: Path, m: dict, recipe: Recipe, now: float | None) -> dict:
    for pat in recipe.required:
        if not _matches(run_dir, (pat,)):
            raise HarvestError(f"required evidence {pat!r} is missing from {run_dir.name}")
    final = bundle_dir(root, m["run_id"])
    hold = final.parent
    if hold.is_symlink():
        raise HarvestError(f"{hold} is a symlink; refusing to write the bundle through it")
    hold.mkdir(parents=True, exist_ok=True)
    # build in a private staging directory and publish by rename; a bundle that already exists is moved aside, never deleted
    out = hold / f".staging-{m['run_id']}-{os.getpid()}-{time.monotonic_ns()}"
    out.mkdir()
    try:
        doc = _fill(out, run_dir, m, recipe, now)
        if final.exists() or final.is_symlink():
            final.rename(hold / f"{final.name}.prev-{time.time_ns()}")
        out.rename(final)
    except BaseException:
        shutil.rmtree(out, ignore_errors=True)   # only our own staging directory
        raise
    return doc


def _fill(out: Path, run_dir: Path, m: dict, recipe: Recipe, now: float | None) -> dict:
    (out / "files").mkdir()
    (out / "derived").mkdir()
    files = []
    for p in _matches(run_dir, recipe.keep):
        rel = p.relative_to(run_dir)
        dest = out / "files" / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, dest)
        files.append({"path": "files/" + rel.as_posix(), "bytes": dest.stat().st_size, "sha256": _sha256(dest), "producer": "copy"})
    extractors = []
    for eid in recipe.extractors:
        fn = EXTRACTORS.get(eid)
        if fn is None:
            raise HarvestError(f"unknown extractor {eid!r}")
        made = fn(run_dir, out / "derived")
        for name, producer in made:
            dest = out / "derived" / name
            files.append({"path": "derived/" + name, "bytes": dest.stat().st_size, "sha256": _sha256(dest), "producer": producer})
        extractors.append({"id": eid, "outputs": len(made)})
    if not files:
        raise HarvestError("the recipe kept no files; refusing an empty bundle")
    bulk = [{"path": p.relative_to(run_dir).as_posix(), "bytes": p.stat().st_size} for p in _matches(run_dir, recipe.bulk)]
    files.sort(key=lambda f: f["path"])
    bulk.sort(key=lambda b: b["path"])
    suspect = m.get("suspect")
    diagnostic = bool(suspect) or m["state"] != "completed" or m["execution"].get("exit_code") not in (None, 0)
    bundle_id = _bundle_id(m["run_id"], recipe.sha256(), diagnostic, files, bulk)
    doc = {"schema_version": BUNDLE_SCHEMA, "bundle_id": bundle_id, "run_id": m["run_id"], "kind": m["kind"],
           "recipe": {"id": recipe.id, "sha256": recipe.sha256()}, "harvested_at": rm.utc_stamp(now),
           "diagnostic_only": diagnostic, "suspect": suspect, "files": files, "extractors": extractors, "bulk": bulk,
           "bulk_bytes": sum(b["bytes"] for b in bulk)}
    rm.write_atomic(out / BUNDLE_NAME, json.dumps(doc, indent=1, sort_keys=True) + "\n")
    problems = verify_bundle(out)
    if problems:
        raise HarvestError("bundle failed verification: " + "; ".join(problems[:3]))
    return doc


def verify_bundle(out: Path) -> list:
    """Re-read a bundle and return the problems found (empty when every file matches its recorded hash and size)."""
    out = Path(out)
    try:
        doc = json.loads((out / BUNDLE_NAME).read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        return [f"bundle.json unreadable: {exc}"]
    problems = []
    if doc.get("schema_version") != BUNDLE_SCHEMA:
        problems.append(f"unknown bundle schema {doc.get('schema_version')!r}")
    for f in doc.get("files", []):
        p = out / f["path"]
        if not p.is_file():
            problems.append(f"{f['path']} is missing")
        elif p.stat().st_size != f["bytes"] or _sha256(p) != f["sha256"]:
            problems.append(f"{f['path']} does not match its recorded hash")
    seen = set()
    for b in doc.get("bulk", []):
        parts = Path(b["path"]).parts
        if Path(b["path"]).is_absolute() or ".." in parts or not parts:
            problems.append(f"bulk path {b['path']!r} is not a relative path inside the run")
        if b["path"] in seen:
            problems.append(f"bulk path {b['path']} is listed twice")
        seen.add(b["path"])
    kept = {Path(f["path"]).relative_to(Path(f["path"]).parts[0]).as_posix() for f in doc.get("files", []) if f["path"].startswith("files/")}
    if seen & kept:
        problems.append("a file is both kept and listed as bulk")
    try:
        expect = _bundle_id(doc["run_id"], doc["recipe"]["sha256"], doc["diagnostic_only"], doc["files"], doc["bulk"])
    except KeyError as exc:
        return problems + [f"bundle.json lacks {exc}"]
    if doc.get("bundle_id") != expect:
        problems.append("bundle_id does not match the bundle contents")
    return problems
