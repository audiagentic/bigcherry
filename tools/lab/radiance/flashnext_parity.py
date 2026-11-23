"""Flash-Next performance parity: the before/after measurement of a Radiance pin bump (bump-radiance section 4).

flashnext-parity.sh drives the launches and writes raw files under OUT/launch-<i>/; this module does everything that is
not process control, so it can be tested without cards:

  run-info OUT --build DIR --launches N --port P --set NAME=VALUE...   OUT/run.json: build, revision, pin, date, workload
  archive-session SESSION --owner-pid PID --out DIR   native session record and per-process logs
  warmup-plan MANIFEST                  {"policy": bool, "warmup_s": float, ...} from spec.handoff.placement.live_policy
  doc-calibrate --url U --tokens T --out FILE   chars per token of the docs corpus, measured with one probe request
  doc-prefill --url U --calibration FILE --slot measure:K|warmup:K --out FILE   one long document prefill, max_tokens 1
  policy-line LOG                       the helper's last `policy placement` line as JSON (committed, bytes_moved)
  record-launch OUT --launch N          commit one completed launch into a run-record v1 (the record is the result;
                                        baseline and compare are `bigcherry results baseline|compare`)

Tokens/s is never gated across builds (MTP draft acceptance differs per build): the gated metrics are cold prefill and
engine step time; acceptance and the live policy's transactions are reported only. The native engine never imports Python; this file is a lab-side wrapper.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import os
import re
import shutil
import statistics
import subprocess
import sys
import time
import tomllib
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
SCHEMA = 1
RECIPES = REPO / "engines/radiance/recipes.toml"

# name -> (direction, gated). direction: which way is better.
METRICS = {
    "cold_ttft_s": ("lower", True),
    "doc_prefill_tps": ("higher", True),
    "decode1_ms_per_step": ("lower", True),
    "decode6_ms_per_step": ("lower", True),
    "sess8_ttft_s": ("lower", True),
    "sess8_ms_per_step": ("lower", True),
    "sess8_warm_ms_per_step": ("lower", False),
    "long7_prefill_tps": ("higher", True),
    "long8_prefill_tps": ("higher", True),
    # Informational, not gated: bimodal between launches (v1.17.0 baseline: 42-44 ms/step in one launch, 28 in the other).
    # The greedy text differs between launches, and with it where the live policy moves experts; long-context decode is
    # gated on long8_ms_per_step.
    "long7_ms_per_step": ("lower", False),
    "long8_ms_per_step": ("lower", True),
    "decode1_acceptance": ("higher", False),
    "decode6_acceptance": ("higher", False),
    "long7_acceptance": ("higher", False),
    "long8_acceptance": ("higher", False),
    "policy_transactions": ("higher", False),
}
# The long-context seeds (flashnext-parity.sh LONG_SEEDS): each seed is its own prompt, so its own metrics.
LONG_SEEDS = (7, 8)
class ParityError(RuntimeError):
    pass


def _load(path: Path):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ParityError(f"cannot read {path}: {exc}") from exc


def _write(path: Path, obj) -> None:
    Path(path).write_text(json.dumps(obj, indent=1, sort_keys=True) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------------------------------------------- run info

def git_revision(repo: Path = REPO) -> str:
    try:
        rev = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"], capture_output=True, text=True, check=True)
        dirty = subprocess.run(["git", "-C", str(repo), "status", "--porcelain", "--untracked-files=no"],
                               capture_output=True, text=True, check=True)
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ParityError(f"cannot read the checkout's git revision: {exc}") from exc
    return rev.stdout.strip() + ("+dirty" if dirty.stdout.strip() else "")


def radiance_pin(recipes: Path = RECIPES) -> dict:
    """The pin the repo builds radiance at: engines/radiance/recipes.toml `pinned` (tag) and `pinned-version`."""
    try:
        data = tomllib.loads(Path(recipes).read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError) as exc:
        raise ParityError(f"cannot read the radiance pin from {recipes}: {exc}") from exc
    if "pinned" not in data:
        raise ParityError(f"{recipes} has no `pinned` key")
    return {"pinned": data["pinned"], "pinned_version": data.get("pinned-version")}


def measured_radiance_pin(build: Path, engine_sha256: str, recipes: Path = RECIPES) -> dict:
    """Identify the measured engine, or explicitly label the checkout fallback."""
    for name in ("bundle.json", "source-bundle.json"):
        path = build / name
        if not path.is_file():
            continue
        bundle = _load(path)
        source = bundle.get("built_from", {})
        pin = source.get("pinned")
        if bundle.get("binary_sha256") != engine_sha256 or not isinstance(pin, str) or not pin:
            raise ParityError(f"{path}: published source does not identify the measured engine")
        return {"pinned": pin, "pinned_version": source.get("pinned_version"), "source": name}

    source_path = build / "source-provenance.json"
    identity_path = build / "qualification-build-metadata.json"
    if source_path.is_file() and identity_path.is_file():
        source = _load(source_path)
        identity = _load(identity_path)
        native = source.get("source_identity", {})
        pin = source.get("upstream")
        if (identity.get("binary_hash") != engine_sha256 or not isinstance(pin, str) or not pin
                or native.get("upstream_revision") != pin or not native.get("source_slice_id")
                or native["source_slice_id"] != identity.get("source_slice_id")
                or not source.get("catalog_revision")
                or source["catalog_revision"] != identity.get("catalog_revision")):
            raise ParityError(f"{build}: qualification source does not identify the measured engine")
        catalog = build / "catalog/engines/radiance/recipes.toml"
        version = None
        if catalog.is_file():
            catalog_pin = radiance_pin(catalog)
            if catalog_pin["pinned"] == pin:
                version = catalog_pin["pinned_version"]
        return {"pinned": pin, "pinned_version": version, "source": source_path.name}

    return {**radiance_pin(recipes), "source": "invoking_catalog"}


def build_identity(build: Path) -> dict:
    """The build's own record (bigcherry-build-metadata-radiance.json) when it has one: which build was measured."""
    meta = Path(build) / "bigcherry-build-metadata-radiance.json"
    if not meta.is_file():
        return {}
    data = _load(meta)
    return {k: data.get(k) for k in ("build_id", "build_plan_id", "source_slice_id", "binary_hash", "runtime_bundle_hash")}


def capture_build(build: Path, out: Path) -> dict[str, str]:
    """Hash the runtime closure as measured, using the Radiance release member layout."""
    from bigcherry.release.runtime_assets import BUILD_METADATA, RUN_MANIFEST, digest

    members = {}
    for pattern, _ in RUN_MANIFEST:   # the config of a launch is its arm YAML (launch-N/native-input.yaml), not a member
        for path in sorted(build.glob(pattern)):
            if path.is_file():
                members[path.relative_to(build).as_posix()] = digest(path)
    if (build / BUILD_METADATA).is_file():
        shutil.copyfile(build / BUILD_METADATA, out / BUILD_METADATA)
    _write(out / "runtime-members.json", members)
    return members


def cmd_run_info(args) -> int:
    workload = {}
    for item in args.set or []:
        name, sep, value = item.partition("=")
        if not sep:
            raise ParityError(f"--set wants NAME=VALUE, got {item!r}")
        workload[name] = int(value) if re.fullmatch(r"-?\d+", value) else value
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    build = Path(args.build).resolve()
    members = capture_build(build, out)
    if "bin/radiance" not in members:
        raise ParityError(f"{build}: missing measured engine binary")
    pin = measured_radiance_pin(build, members["bin/radiance"])
    _write(out / "run.json", {
        "schema": SCHEMA, "build": str(build), "build_identity": build_identity(build),
        "revision": git_revision(), "radiance_pin": pin,
        "date": _dt.datetime.now(_dt.UTC).isoformat(timespec="seconds"),
        "launches": args.launches, "port": args.port, "workload": workload})
    return 0


def cmd_archive_session(args) -> int:
    path, out = Path(args.session), Path(args.out)
    if not path.is_file():
        return 0
    raw = path.read_bytes()
    session = json.loads(raw)
    if session["owner_pid"] != args.owner_pid:
        return 0
    (out / "session.json").write_bytes(raw)
    manifest = session.get("manifest")
    if manifest and not (out / "manifest.json").is_file() and Path(manifest).is_file():
        shutil.copyfile(manifest, out / "manifest.json")
    logs = out / "native-logs"
    logs.mkdir(exist_ok=True)
    for process in session["processes"]:
        source = Path(process["log"])
        if source.is_file():
            saved = logs / source.name
            shutil.copyfile(source, saved)
            if process["role"] in ("engine", "helper0"):
                shutil.copyfile(saved, out / f"{process['role']}.log")
    return 0


# ------------------------------------------------------------------------------------------------------------------ warmup

def warmup_plan(manifest: dict) -> dict:
    """With the live expert-placement policy on, wait until it has had (min_samples + 1) heat windows plus a second, so a
    measurement never starts before the policy could have moved anything. Without a policy there is nothing to wait for."""
    policy = (((manifest.get("spec") or {}).get("handoff") or {}).get("placement") or {}).get("live_policy")
    if not policy:
        return {"policy": False, "warmup_s": 0.0}
    cadence_ns, min_samples = int(policy["cadence_ns"]), int(policy["min_samples"])
    return {"policy": True, "cadence_ns": cadence_ns, "min_samples": min_samples,
            "warmup_s": (min_samples + 1) * cadence_ns / 1e9 + 1}


def cmd_warmup_plan(args) -> int:
    plan = warmup_plan(_load(Path(args.manifest)))
    plan["manifest"] = str(args.manifest)
    rendered = json.dumps(plan, sort_keys=True)
    if args.out:
        Path(args.out).write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0


# ------------------------------------------------------------------------------------------------------- document prefill

def docs_corpus(repo: Path = REPO) -> str:
    """The repo's docs/**/*.md in path order, one text: the same checkout gives the same corpus."""
    files = sorted(p for p in (repo / "docs").rglob("*.md") if p.is_file())
    if not files:
        raise ParityError(f"no docs/**/*.md under {repo}")
    return "\n\n".join(p.read_text(encoding="utf-8", errors="replace") for p in files)


def _completion(url: str, prompt: str, timeout: float) -> tuple[int, float]:
    """One non-streaming completion of max_tokens 1 at temperature 0: (server prompt_tokens, wall seconds)."""
    body = json.dumps({"model": "default", "prompt": prompt, "max_tokens": 1, "temperature": 0, "stream": False}).encode()
    req = urllib.request.Request(url.rstrip("/") + "/v1/completions", data=body,
                                 headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            payload = json.loads(resp.read())
    except (OSError, ValueError) as exc:
        raise ParityError(f"document prefill request failed: {exc}") from exc
    wall = time.perf_counter() - t0
    tokens = int((payload.get("usage") or {}).get("prompt_tokens") or 0)
    if tokens <= 0:
        raise ParityError("the server reported no usage.prompt_tokens for a document prefill")
    return tokens, wall


def probe_text(corpus: str, target_tokens: int) -> str:
    """The probe is the corpus' last `target_tokens` characters (at least one character a token, so it is never longer
    than a document prompt), a region no measured or warmup document prompt uses."""
    return corpus[-target_tokens:]


def doc_slices(corpus_len: int, chars: int, probe_chars: int, measured: int):
    """(measured offsets, warmup region start, warmup region length). Measured prompts are the first `measured` slots of
    the corpus; warmup prompts cycle through the rest before the probe region, so a measured prompt is never a prefix the
    server has seen."""
    usable = corpus_len - probe_chars
    warm_start = measured * chars
    warm_len = usable - warm_start
    if warm_len < 2 * chars:
        raise ParityError(f"the docs corpus ({corpus_len} chars) is too short for {measured} disjoint document prompts of "
                          f"{chars} chars, a warmup region and a probe")
    return [i * chars for i in range(measured)], warm_start, warm_len


def doc_offset(slot: str, chars: int, corpus_len: int, probe_chars: int, measured: int) -> int:
    kind, _, index = slot.partition(":")
    k = int(index)
    offsets, warm_start, warm_len = doc_slices(corpus_len, chars, probe_chars, measured)
    if kind == "measure":
        if not 0 <= k < measured:
            raise ParityError(f"measured slot {k} outside 0..{measured - 1}")
        return offsets[k]
    if kind == "warmup":
        span = warm_len - chars                      # the last start that still leaves a whole prompt in the region
        return warm_start + (k * chars) % (span + 1)
    raise ParityError(f"slot must be measure:K or warmup:K, got {slot!r}")


def cmd_doc_calibrate(args) -> int:
    corpus = docs_corpus()
    probe = probe_text(corpus, args.tokens)
    tokens, wall = _completion(args.url, probe, args.timeout)
    cpt = len(probe) / tokens
    chars = round(args.tokens * cpt)
    doc_slices(len(corpus), chars, len(probe), args.measured)        # refuse now, before any warmup, if it cannot fit
    _write(Path(args.out), {"target_tokens": args.tokens, "chars_per_token": cpt, "chars": chars,
                            "measured_slots": args.measured, "probe_chars": len(probe), "probe_prompt_tokens": tokens,
                            "probe_wall_s": wall, "corpus_chars": len(corpus),
                            "corpus_sha256": hashlib.sha256(corpus.encode("utf-8")).hexdigest()})
    return 0


def cmd_doc_prefill(args) -> int:
    cal = _load(Path(args.calibration))
    corpus = docs_corpus()
    if hashlib.sha256(corpus.encode("utf-8")).hexdigest() != cal["corpus_sha256"]:
        raise ParityError("the docs corpus changed since the calibration of this launch")
    offset = doc_offset(args.slot, cal["chars"], len(corpus), cal["probe_chars"], cal["measured_slots"])
    prompt = corpus[offset:offset + cal["chars"]]
    tokens, wall = _completion(args.url, prompt, args.timeout)
    _write(Path(args.out), {"slot": args.slot, "offset": offset, "chars": len(prompt), "target_tokens": cal["target_tokens"],
                            "prompt_tokens": tokens, "wall_s": wall, "prefill_tps": tokens / wall,
                            "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest()})
    return 0


# ------------------------------------------------------------------------------------------------------------ policy line

_POLICY = re.compile(r"policy placement helper=(\d+)\b.*?\bcommitted=(\d+)\b.*?\bbytes_moved=(\d+)\b")


def policy_line(text: str) -> dict:
    """The last `policy placement` line of a helper log: its counters are cumulative for the helper's life."""
    last = None
    for line in text.splitlines():
        m = _POLICY.search(line)
        if m:
            last = (line.strip(), m)
    if last is None:
        return {"found": False}
    line, m = last
    return {"found": True, "helper": int(m.group(1)), "committed": int(m.group(2)), "bytes_moved": int(m.group(3)),
            "line": line}


def cmd_policy_line(args) -> int:
    text = Path(args.log).read_text(encoding="utf-8", errors="replace") if Path(args.log).is_file() else ""
    rendered = json.dumps(policy_line(text), sort_keys=True)
    if args.out:
        Path(args.out).write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0


# --------------------------------------------------------------------------------------------------------------- summarise

def _rounds(path: Path) -> list:
    rec = _load(path)
    if not rec.get("ok"):
        raise ParityError(f"{path}: a stream failed (see its per_stream errors)")
    rounds = rec.get("rounds") or []
    if not rounds:
        raise ParityError(f"{path}: no rounds")
    return rounds


def _engine(rnd: dict, key: str, path: Path) -> float:
    eng = rnd.get("engine") or {}
    if key not in eng:
        raise ParityError(f"{path}: round {rnd.get('rep')} has no engine {key} (does the server expose /metrics?)")
    return float(eng[key])


def _files(launch: Path, pattern: str, expect: int | None = None) -> list:
    files = sorted(launch.glob(pattern))
    if not files or (expect is not None and len(files) != expect):
        raise ParityError(f"{launch}: expected {expect or 'some'} {pattern}, found {len(files)}")
    return files


def launch_metrics(launch: Path, workload: dict) -> tuple[dict, dict, dict]:
    """(metric -> per-launch value, metric -> repeat groups, detail) from one launch's raw files.

    A repeat group holds the values of the rounds of ONE prompt (the --reps rounds of one seed): only those measure run-to-run
    noise. Distinct prompts (the cold fills, the document slices, the long-context seeds) differ by content, so they are
    averaged into the per-launch value (or kept as separate metrics) and never form a group."""
    value, repeats, detail = {}, {}, {}

    fills = [r["per_stream"][0]["ttft_ms"] / 1000.0
             for f in _files(launch, "cold-fill-*.json", workload.get("COLD_FILLS")) for r in _rounds(f)]
    value["cold_ttft_s"] = statistics.fmean(fills)
    detail["cold_ttft_s"] = fills

    docs = [_load(f)["prefill_tps"] for f in _files(launch, "doc-prefill-*.json", workload.get("DOC_FILLS"))]
    value["doc_prefill_tps"] = statistics.fmean(docs)
    detail["doc_prefill_tps"] = docs

    for name, file in (("decode1", "decode1.json"), ("decode6", "decode6.json")):
        rounds = _rounds(launch / file)
        steps = [_engine(r, "decode_ms_per_step", launch / file) for r in rounds]
        acc = [_engine(r, "draft_acceptance", launch / file) for r in rounds]
        value[f"{name}_ms_per_step"], repeats[f"{name}_ms_per_step"] = statistics.fmean(steps), [steps]
        value[f"{name}_acceptance"], repeats[f"{name}_acceptance"] = statistics.fmean(acc), [acc]
        detail[f"{name}_tokens_per_step"] = [_engine(r, "decode_tokens_per_step", launch / file) for r in rounds]
        detail[f"{name}_ttft_s"] = [r["aggregate"]["ttft_ms_max"] / 1000.0 for r in rounds]

    rounds = _rounds(launch / "sess8.json")
    cold = [r for r in rounds if r.get("rep") == 0]
    if not cold:
        raise ParityError(f"{launch}/sess8.json: no first (cold) round")
    value["sess8_ttft_s"] = cold[0]["aggregate"]["ttft_ms_p50"] / 1000.0
    detail["sess8_ttft_p50_s"] = [r["aggregate"]["ttft_ms_p50"] / 1000.0 for r in rounds]
    # Gated: the cold round only. Later rounds hit the prefix cache to a varying degree, so a different number of the 8
    # sessions decode together in each step (all 8 on a full hit: longer steps, several times the throughput); their
    # ms/step is informational, never comparable across rounds or builds.
    value["sess8_ms_per_step"] = _engine(cold[0], "decode_ms_per_step", launch / "sess8.json")
    warm = [_engine(r, "decode_ms_per_step", launch / "sess8.json") for r in rounds if r.get("rep") != 0]
    if warm:
        value["sess8_warm_ms_per_step"] = statistics.fmean(warm)
    detail["sess8_ms_per_step_rounds"] = [_engine(r, "decode_ms_per_step", launch / "sess8.json") for r in rounds]

    seeds = str(workload.get("LONG_SEEDS", " ".join(map(str, LONG_SEEDS)))).split()
    if tuple(int(x) for x in seeds) != LONG_SEEDS:
        raise ParityError(f"the run's long-context seeds {seeds} are not the metrics' seeds {list(LONG_SEEDS)}")
    for seed in LONG_SEEDS:
        f = launch / f"long-{seed}.json"
        rounds = _rounds(f)
        cold = [r for r in rounds if r.get("rep") == 0]       # a repeat of the same prompt is a prefix-cache hit
        if not cold:
            raise ParityError(f"{f}: no cold round")
        s = cold[0]["per_stream"][0]
        value[f"long{seed}_prefill_tps"] = s["prompt_tokens"] / (s["ttft_ms"] / 1000.0)
        steps = [_engine(r, "decode_ms_per_step", f) for r in rounds]
        acc = [_engine(r, "draft_acceptance", f) for r in rounds]
        value[f"long{seed}_ms_per_step"], repeats[f"long{seed}_ms_per_step"] = statistics.fmean(steps), [steps]
        value[f"long{seed}_acceptance"], repeats[f"long{seed}_acceptance"] = statistics.fmean(acc), [acc]
        detail[f"long{seed}_prompt_tokens"] = [r["per_stream"][0]["prompt_tokens"] for r in rounds]

    pol = _load(launch / "policy.json") if (launch / "policy.json").is_file() else {"found": False}
    value["policy_transactions"] = pol["committed"] if pol.get("found") else None
    detail["policy_bytes_moved"] = pol.get("bytes_moved")
    return value, repeats, detail


# ------------------------------------------------------------------------------------------------------------- MRC06 v1

def record_metric_phase(name: str) -> str:
    """Keep parity metric names intact, but separate their measured phases."""
    if name in ("cold_ttft_s", "sess8_ttft_s"):
        return "ttft"
    if name == "doc_prefill_tps" or name.startswith(("long7_prefill_", "long8_prefill_")):
        return "prefill"
    if name in METRICS:
        return "decode"
    raise ParityError(f"unknown parity metric {name}")


def _platform():
    # This standalone harness imports the shared platform producer by path,
    # never opening a second measurement-result write path.
    tools = str(REPO / "tools")
    if tools not in sys.path:
        sys.path.insert(0, tools)
    from bigcherry.core import engines, feature_states, measurement_producers
    return engines, feature_states, measurement_producers


def capture_host_facts(launch: Path, run_dir: Path) -> None:
    """Copy native owner start receipt and bound session before measuring."""
    tools = str(REPO / "tools")
    if tools not in sys.path:
        sys.path.insert(0, tools)
    from bigcherry.core import host_facts
    from shutil import copyfile

    host_facts.read_owner_start(run_dir)
    copyfile(Path(run_dir) / host_facts.START_RECEIPT_FILE,
             Path(launch) / host_facts.START_RECEIPT_FILE)
    copyfile(Path(run_dir) / "session.json", Path(launch) / "native-session-start.json")
    host_facts.read_owner_start(launch, session_name="native-session-start.json")


def capture_launch_provenance(
    launch: Path, argv: list[str], *, heat: Path,
    calibration: Path | None, power_profile: str
) -> None:
    """Capture precisely the owner invocation, not a reconstructed CLI."""
    if not argv or not heat.is_file():
        raise ParityError("missing native owner argv or the selected heat file")
    # Intentional allowlist: do not persist ambient credentials or whole env.
    keys = ("ROCM_PATH", "HIP_PATH", "BC_WORK_ROOT", "BC_MODEL_ROOT",
            "FLASHNEXT_PARITY_POWER_PROFILE", "FLASHNEXT_PARITY_HEAT",
            "FLASHNEXT_PARITY_SETS", "FLASHNEXT_PARITY_WARMUP_S")
    _write(launch / "launch-provenance.json", {
        "argv": list(argv), "environment": {k: os.environ[k] for k in keys if k in os.environ},
        "heat_path": str(heat.resolve()),
        "calibration_store": str(calibration.resolve()) if calibration else None,
        "power_profile": power_profile,
        "recorded_at": _dt.datetime.now(_dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
    })


def selected_calibration(path: Path, key: str) -> dict:
    """Use the platform MRC04 strict current/invalid classifications.

    This reader verifies the full declared key and quantities; a row it does not know is refused.
    """
    from bigcherry.core.calibration_records import read_native_toml

    selected = [item for item in read_native_toml(path) if item["key"] == key]
    if len(selected) != 1:
        raise ParityError("selected native calibration key is missing or ambiguous")
    item = selected[0]
    if item["status"] != "current":
        raise ParityError(f"selected calibration is {item['status']}: {item['reason']}")
    return item["native"]


def selected_calibration_key(launch: Path, state: dict) -> str:
    """The calibration key the native owner selected for this launch.

    The owner reports it in its session start record (calibration_record); the manifest's policy state may leave
    its own key empty. When the manifest does carry one it must agree.
    """
    selection = _load(launch / "native-session-start.json").get("calibration_record")
    key = selection.get("key_sha256") if isinstance(selection, dict) else None
    if (not isinstance(selection, dict) or selection.get("status") != "selected"
            or selection.get("kind") != "radiance.live_policy"
            or not isinstance(key, str) or not key):
        raise ParityError("active policy lacks selected calibration store/key")
    manifest_key = state.get("calibration_key")
    if manifest_key not in (None, "") and manifest_key != key:
        raise ParityError("manifest calibration key differs from the owner's selected key")
    return key


def publish_launch_record(
    out: Path, launch: Path, *, work_root: Path | None = None
) -> str:
    """Publish exactly one complete parity launch through the platform producer.

    Missing native feature state, build identity, or selected calibration is a
    producer error. None of those can be manufactured from the measurements.
    """
    engines, feature_states, measurement_producers = _platform()
    out, launch = Path(out), Path(launch)
    run = _load(out / "run.json")
    manifest_file = launch / "manifest.json"
    manifest = _load(manifest_file)
    snapshot = _load(launch / "launch-provenance.json")
    warmup = _load(launch / "launch.json")
    from bigcherry.core import host_facts as host_capture
    try:
        facts = host_capture.read_owner_start(launch, session_name="native-session-start.json")
    except (ValueError, TypeError, KeyError) as exc:
        raise ParityError(f"native owner sealed start receipt unavailable: {exc}") from exc
    owner = _load(launch / "native-session-start.json")
    native_manifest = Path(owner.get("manifest", ""))
    if not native_manifest.is_file() or native_manifest.read_bytes() != manifest_file.read_bytes():
        raise ParityError("native owner admitted manifest differs from measured parity manifest")
    states = feature_states.read_feature_states(launch / feature_states.FILE_NAME)
    if any(item["engine"] != engines.RADIANCE for item in states):
        raise ParityError("native feature-state engine differs from Radiance")
    build = run.get("build_identity") or {}
    meta_file = Path(run["build"]) / "bigcherry-build-metadata-radiance.json"
    metadata = _load(meta_file)
    for name in ("build_id", "binary_hash", "source_slice_id", "build_plan_id"):
        if not isinstance(metadata.get(name), str) or not metadata[name]:
            raise ParityError(f"build metadata lacks required {name}")
    if any(build.get(k) != metadata.get(k) for k in
           ("build_id", "binary_hash", "source_slice_id", "build_plan_id")):
        raise ParityError("run.json build identity differs from build metadata")
    if not re.fullmatch(r"[0-9a-f]{64}", metadata["binary_hash"]):
        raise ParityError("build metadata has no SHA-256 binary identity")
    spec = manifest["spec"]
    placement = spec["handoff"]["placement"]
    state = placement["live_policy_state"]
    if type(state.get("configured")) is not bool or state["configured"] != bool(placement.get("live_policy")):
        raise ParityError("manifest live policy and policy state disagree")
    if bool(warmup["policy"]) != state["configured"]:
        raise ParityError("warmup and manifest disagree about live policy")
    if not isinstance(snapshot.get("argv"), list) or not snapshot["argv"] or not isinstance(snapshot.get("environment"), dict):
        raise ParityError("missing captured native argv/environment")
    values, repeats, _ = launch_metrics(launch, run["workload"])
    metrics = {phase: {} for phase in ("decode", "prefill", "ttft")}
    metric_specs = {phase: {} for phase in metrics}
    for name, (direction, gated) in METRICS.items():
        phase = record_metric_phase(name)
        value = values.get(name)
        if value is None:
            if gated:
                raise ParityError(f"required gated metric {name} is missing")
            continue
        metrics[phase][name] = value
        definition = {"direction": direction, "gated": gated}
        groups = [group for group in repeats.get(name, []) if len(group) >= 2]
        if groups:
            definition["repeat_groups"] = groups
        metric_specs[phase][name] = definition

    inputs = {
        "run_info": out / "run.json",
        "heat": Path(snapshot["heat_path"]),
        "policy_state": measurement_producers.canonical_blob(state),
        "warmup": launch / "launch.json",
        "native_feature_states": launch / feature_states.FILE_NAME,
        "native_host_start": launch / host_capture.START_RECEIPT_FILE,
        "native_session_start": launch / "native-session-start.json",
        "build_metadata": meta_file,
    }
    # Raw source bytes remain content-addressed, never recreated from metrics.
    for role, filename in (("doc_calibration", "doc-calibration.json"),
                           ("policy_placement", "policy.json")):
        path = launch / filename
        if path.is_file():
            inputs[role] = path
    if state["configured"]:
        # The calibration record is provenance, not a gate: a run whose live policy was given no calibration store
        # carries none, and the record says so by omitting it. When a store is given the selected record is verified.
        cal = snapshot.get("calibration_store")
        if cal:
            key = selected_calibration_key(launch, state)
            inputs["calibration_record"] = measurement_producers.canonical_blob(
                selected_calibration(Path(cal), key)
            )
    runtime_members = _load(out / "runtime-members.json")
    identity = {
        "engine": engines.RADIANCE, "build_id": metadata["build_id"],
        # Use the runtime member captured before launch; some release metadata
        # identifies an unstripped artifact rather than the executable run.
        "binary_sha256": runtime_members["bin/radiance"],
        "source_revision": run["revision"],
        "model": spec["plan"]["engine_memory"]["model_identity"],
        "layout": spec["intent"]["layout"],
    }
    return measurement_producers.publish(
        work_root=work_root,
        run_id=manifest["metadata"]["instance_id"],
        # The sealed owner start receipt marks the actual measurement start (the argv was selected earlier).
        recorded_at=facts["captured_at"],
        identity=identity,
        workload={"name": "flashnext-parity", "definition": run["workload"]},
        metrics=metrics, metric_specs=metric_specs,
        effective_config=manifest_file.read_bytes(),
        argv=snapshot["argv"], environment=snapshot["environment"],
        power_settings={
            "effective": spec["plan"].get("power", spec.get("power", {})),
            "requested_profile": snapshot["power_profile"],
        },
        inputs=inputs,
        host_facts=facts, feature_states=states,
    )


def cmd_capture_host_facts(args) -> int:
    capture_host_facts(Path(args.launch), Path(args.run_dir))
    return 0


def cmd_capture_provenance(args) -> int:
    if not args.argv:
        raise ParityError("capture-provenance requires the real native argv after --")
    capture_launch_provenance(
        Path(args.launch), args.argv, heat=Path(args.heat),
        calibration=Path(args.calibration) if args.calibration else None,
        power_profile=args.power_profile,
    )
    return 0


def cmd_record_launch(args) -> int:
    digest = publish_launch_record(
        Path(args.out), Path(args.out) / f"launch-{args.launch}",
        work_root=Path(args.work_root) if args.work_root else None,
    )
    print(digest)
    return 0


# -------------------------------------------------------------------------------------------------------------------- main

def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("run-info")
    p.add_argument("out")
    p.add_argument("--build", required=True)
    p.add_argument("--launches", type=int, required=True)
    p.add_argument("--port", type=int, required=True)
    p.add_argument("--set", action="append", help="a workload constant NAME=VALUE (repeat)")
    p.set_defaults(fn=cmd_run_info)
    p = sub.add_parser("archive-session")
    p.add_argument("session")
    p.add_argument("--owner-pid", type=int, required=True)
    p.add_argument("--out", required=True)
    p.set_defaults(fn=cmd_archive_session)
    p = sub.add_parser("warmup-plan")
    p.add_argument("manifest")
    p.add_argument("--out")
    p.set_defaults(fn=cmd_warmup_plan)
    p = sub.add_parser("doc-calibrate")
    p.add_argument("--url", required=True)
    p.add_argument("--tokens", type=int, required=True)
    p.add_argument("--measured", type=int, required=True, help="number of measured document prompts")
    p.add_argument("--timeout", type=float, required=True)
    p.add_argument("--out", required=True)
    p.set_defaults(fn=cmd_doc_calibrate)
    p = sub.add_parser("doc-prefill")
    p.add_argument("--url", required=True)
    p.add_argument("--calibration", required=True)
    p.add_argument("--slot", required=True, help="measure:K or warmup:K")
    p.add_argument("--timeout", type=float, required=True)
    p.add_argument("--out", required=True)
    p.set_defaults(fn=cmd_doc_prefill)
    p = sub.add_parser("policy-line")
    p.add_argument("log")
    p.add_argument("--out")
    p.set_defaults(fn=cmd_policy_line)
    p = sub.add_parser("capture-host-facts", help="pin SHA-sealed native owner start")
    p.add_argument("launch")
    p.add_argument("--run-dir", required=True)
    p.set_defaults(fn=cmd_capture_host_facts)
    p = sub.add_parser("capture-provenance", help="capture native argv and input selection")
    p.add_argument("launch")
    p.add_argument("--heat", required=True)
    p.add_argument("--calibration")
    p.add_argument("--power-profile", required=True)
    p.add_argument("argv", nargs=argparse.REMAINDER)
    p.set_defaults(fn=cmd_capture_provenance)
    p = sub.add_parser("record-launch", help="commit one completed launch into run-record v1")
    p.add_argument("out")
    p.add_argument("--launch", type=int, required=True)
    p.add_argument("--work-root")
    p.set_defaults(fn=cmd_record_launch)
    return ap


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.fn(args)
    except (ParityError, KeyError, ValueError, OSError, TypeError) as exc:
        print(f"flashnext-parity: ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
