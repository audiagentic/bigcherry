"""MRC06 Radiance laboratory producers: snapshot native evidence into MRC01.

Every published value has a verified source. The C++ serving owner writes its
own facts/receipts and never loads Python; this platform adapter runs after a
native lab run or a lab probe. No historical JSON result is imported.
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import json
import re
import urllib.parse
import urllib.request
from datetime import UTC, datetime
from pathlib import Path

from ..core import (
    engines,
    feature_states,
    host_facts,
    measurement_producers,
    run_records,
)

_SHA = re.compile(r"[0-9a-f]{64}\Z")


def _read_json(path: Path) -> dict:
    if path.is_symlink() or not path.is_file():
        raise run_records.RecordError(
            f"required native evidence is missing or unsafe: {path}"
        )
    doc = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(doc, dict):
        raise run_records.RecordError(f"expected an evidence object: {path}")
    return doc


def _file(path: Path) -> Path:
    if path.is_symlink() or not path.is_file():
        raise run_records.RecordError(f"missing or unsafe evidence file: {path}")
    return path


def _digest(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _session_context(run_dir: Path, *, require_stopped: bool = False) -> dict:
    run_dir = Path(run_dir).resolve(strict=True)
    session_file = _file(run_dir / "session.json")
    session = _read_json(session_file)
    allowed = (
        {"stopped"}
        if require_stopped
        else {"running", "healthy"}
    )
    if session.get("state") not in allowed or session.get("errors"):
        raise run_records.RecordError("native session is not in an admitted state")
    manifest_file = _file(Path(session["manifest"]))
    # Do not trust a symlink or archived/replaced manifest. A native owner
    # writes a distinct manifest per launch and keeps it in the run directory.
    if manifest_file.parent.resolve() != run_dir:
        raise run_records.RecordError(
            "active manifest is outside the admitted run directory"
        )
    raw = manifest_file.read_bytes()
    manifest = json.loads(raw)
    if not isinstance(manifest, dict) or not isinstance(manifest.get("spec"), dict):
        raise run_records.RecordError("native strict manifest is malformed")
    spec = manifest["spec"]
    intent = spec["intent"]
    runtime = spec["processes"]["engine"]["runtime"]
    build = Path(runtime["radiance_home"]).parent
    metadata_file = _file(build / "bigcherry-build-metadata-radiance.json")
    metadata = _read_json(metadata_file)
    for key in ("build_id", "binary_hash", "source_slice_id"):
        if not isinstance(metadata.get(key), str) or not metadata[key]:
            raise run_records.RecordError(f"native build metadata lacks {key}")
    if not _SHA.fullmatch(metadata["binary_hash"]):
        raise run_records.RecordError("native build binary hash is not SHA-256")
    from .workloads import _release_identity
    try:
        metadata, release_inputs = _release_identity(build, metadata, None)
    except (OSError, ValueError, KeyError) as exc:
        raise run_records.RecordError(f"invalid native release identity: {exc}") from exc
    binary = build / "bin/radiance"
    if binary.is_file() and _digest(binary.read_bytes()) != metadata["binary_hash"]:
        raise run_records.RecordError(
            "native engine binary changed since build metadata"
        )
    native_states = _file(run_dir / feature_states.FILE_NAME)
    states = feature_states.read_feature_states(native_states)
    start_file = _file(run_dir / host_facts.START_RECEIPT_FILE)
    try:
        start = host_facts.read_owner_start(run_dir)
    except ValueError as exc:
        raise run_records.RecordError(f"invalid native start receipt: {exc}") from exc
    if any(state["engine"] != engines.RADIANCE for state in states):
        raise run_records.RecordError("native feature-state engine differs")
    instance = manifest.get("metadata", {}).get("instance_id")
    if not isinstance(instance, str) or not instance:
        raise run_records.RecordError("strict manifest has no run instance ID")
    if not isinstance(intent.get("layout"), str) or not isinstance(
        intent.get("model"), str
    ):
        raise run_records.RecordError("strict manifest lacks model/layout")
    identity = {
        "engine": engines.RADIANCE,
        "build_id": metadata["build_id"],
        "binary_sha256": metadata["binary_hash"],
        "source_revision": metadata.get("source_commit")
        or ("source_slice:" + metadata["source_slice_id"]),
        "model": spec.get("plan", {}).get("engine_memory", {}).get("model_identity")
        or intent["model"],
        "layout": intent["layout"],
    }
    source_inputs: dict[str, bytes | Path] = {
        "native_host_facts_start": start_file,
        "native_session": session_file,
        "native_feature_states": native_states,
        "native_build_metadata": metadata_file,
        **release_inputs,
    }
    for p in session.get("processes", []):
        if not isinstance(p, dict) or not isinstance(p.get("log"), str):
            continue
        logfile = Path(p["log"])
        if (
            logfile.parent.resolve() == run_dir
            and logfile.is_file()
            and not logfile.is_symlink()
        ):
            source_inputs[f"native_log_{p['role']}"] = logfile
    return {
        "run_dir": run_dir,
        "session": session,
        "manifest": manifest,
        "manifest_file": manifest_file,
        "manifest_bytes": raw,
        "identity": identity,
        "run_id": instance,
        "feature_states": states,
        "host_start": start,
        "inputs": source_inputs,
    }


def _host(snapshot: dict | None) -> dict:
    if snapshot is None:
        raise run_records.RecordError(
            "missing measurement-start host facts; never sample at result ingest time"
        )
    try:
        return host_facts.validate_start_snapshot(snapshot)
    except ValueError as exc:
        raise run_records.RecordError(f"invalid measurement-start host facts: {exc}") from exc


def _now() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _native_argv(session: dict) -> list[str]:
    processes = session.get("processes", [])
    for p in processes:
        if p.get("role") == "engine" and isinstance(p.get("argv"), list) and p["argv"]:
            return p["argv"]
    raise run_records.RecordError(
        "native session does not record measured process argv"
    )


def _native_environment(session: dict) -> dict:
    # Native session may not expose child environment. Never reconstruct the
    # whole ambient environment or copy credentials from a process namespace.
    for proc in session.get("processes", []):
        if proc.get("role") == "engine":
            env = proc.get("env") or {}
            if isinstance(env, dict):
                return {
                    k: v
                    for k, v in env.items()
                    if k
                    in {
                        "ROCR_VISIBLE_DEVICES",
                        "HIP_VISIBLE_DEVICES",
                        "ROCM_PATH",
                        "RADIANCE_HOME",
                    }
                    and isinstance(v, str)
                }
    return {}


def verify_live_target(run_dir: Path, url: str) -> None:
    """Refuse an unrelated/missing native owner before sending any probe HTTP.

    Binding the endpoint prevents attributing a healthy response from one
    server to a different run's strict manifest.
    """
    _verify_live_context(_session_context(run_dir), url)


def _verify_live_context(context: dict, url: str) -> None:
    if context["session"]["state"] not in ("running", "healthy"):
        raise run_records.RecordError(
            "probe target is not a running native serving owner"
        )
    endpoint = urllib.parse.urlsplit(url)
    if (
        endpoint.scheme != "http"
        or endpoint.hostname not in ("127.0.0.1", "localhost", "::1")
        or endpoint.username is not None
        or endpoint.password is not None
    ):
        raise run_records.RecordError(
            "probe target must be the local native owner's HTTP endpoint"
        )
    try:
        port = endpoint.port
    except ValueError as exc:
        raise run_records.RecordError("probe URL has an invalid port") from exc
    native_port = context["session"].get("port")
    declared_port = (
        context["manifest"]["spec"]["processes"]["engine"].get("server", {}).get("port")
    )
    if (
        type(port) is not int
        or type(native_port) is not int
        or port != native_port
        or (declared_port is not None and port != declared_port)
    ):
        raise run_records.RecordError(
            "probe HTTP port differs from the admitted native owner"
        )


# Capture only performance configuration, never the diagnostic environment or host.
TUNING_DIAGNOSTIC_OPTIONS = frozenset({
    "--tp", "--tp-wire", "--tp-wire-min-kb", "--tp-ownership", "--tune-pin",
    "--pipelined-prefill", "--pipelined-prefill-min-rows", "--prefill-interleave",
    "--mixed-step-target-ms", "--max-num-batched-tokens", "--max-num-seqs",
    "--kv-cache-dtype", "--state-cache-dtype", "--attn-prefill-fp8",
    "--busy-prefill-chunk", "--prefill-governor", "--prefill-room-gate",
    "--long-prefill-token-threshold", "--preempt-wait-ms", "--no-prefix-cache",
})


def reconcile_tuning_diagnostics(document: dict, expected: dict[str, str]) -> dict:
    """Compare native diagnostic spellings, not a second interpretation of YAML."""
    if not expected or not set(expected) <= TUNING_DIAGNOSTIC_OPTIONS:
        raise run_records.RecordError("missing or unsupported tuning expectation")
    if not all(isinstance(v, str) and v for v in expected.values()):
        raise run_records.RecordError("invalid expected tuning value")
    if not isinstance(document, dict) or document.get("startup_failed") is not False or document.get("unsupported_configuration") is not False:
        raise run_records.RecordError("native diagnostics do not prove a supported startup")
    rows = document.get("configuration")
    if not isinstance(rows, list):
        raise run_records.RecordError("native diagnostics configuration is missing")
    observed = {}
    for row in rows:
        if not isinstance(row, dict) or not isinstance(row.get("option"), str):
            raise run_records.RecordError("malformed native diagnostics row")
        flag = row["option"]
        if flag not in TUNING_DIAGNOSTIC_OPTIONS:
            continue
        if flag in observed or not all(isinstance(row.get(k), str) and row[k] for k in ("value", "source")):
            raise run_records.RecordError(f"duplicate or malformed native tuning row: {flag}")
        observed[flag] = {k: row[k] for k in ("value", "source")}
    for flag, value in expected.items():
        if observed.get(flag, {}).get("value") != value:
            raise run_records.RecordError(f"effective tuning differs or is missing: {flag}")
    return observed

def _manifest_tuning_expectations(manifest: dict) -> dict[str, str]:
    """Direct fields with stable native diagnostic spellings (not dynamic defaults)."""
    spec = manifest["spec"]
    engine = spec["processes"]["engine"]
    expected = {}
    for section, keys in (
        ("scheduler", ("tp", "max_num_batched_tokens")),
        ("execution", ("kv_cache_dtype", "tp_wire")),
        ("tuning", ("tp_ownership", "tune_pin", "tp_wire_min_kb",
                    "pipelined_prefill", "pipelined_prefill_min_rows")),
    ):
        for key in keys:
            value = engine.get(section, {}).get(key)
            if value is not None and value != "auto" and value != "":
                if key == "pipelined_prefill_min_rows" and value == 0:
                    continue
                expected["--" + key.replace("_", "-")] = str(value)
    interleave = engine.get("tuning", {}).get("prefill_interleave")
    if interleave is not None:
        expected["--prefill-interleave"] = "off" if interleave == 0 else str(interleave)
    sessions = spec["intent"].get("service", {}).get("max_sessions")
    if sessions is not None:
        expected["--max-num-seqs"] = str(sessions)
    if engine.get("execution", {}).get("no_prefix_cache") == "on":
        expected["--no-prefix-cache"] = "prefix caching off"
    return expected


def _context_tuning_expectations(context: dict) -> dict[str, str]:
    """Use the matched owner's parsed scheduler, bound to the sealed native YAML."""
    expected = _manifest_tuning_expectations(context["manifest"])
    selection = context["session"].get("selection", {})
    if "native_scheduler" not in selection:
        return expected  # older owner; explicit arm expectations remain necessary
    receipt = selection["native_scheduler"]
    native = context["manifest"]["spec"].get("native_config", {})
    if (not isinstance(receipt, dict) or set(receipt) != {"snapshot_sha256", "parsed"}
            or not native.get("sha256") or receipt["snapshot_sha256"] != native["sha256"]):
        raise run_records.RecordError("parsed scheduler does not bind native YAML")
    parsed = receipt["parsed"]
    if not isinstance(parsed, dict):
        raise run_records.RecordError("malformed parsed scheduler")
    for key in ("interleave_authored", "mixed_step_target_authored", "room_gate",
                "room_gate_authored", "long_prefill_authored", "preempt_wait_authored"):
        if type(parsed.get(key)) is not bool:
            raise run_records.RecordError(f"malformed parsed scheduler: {key}")
    for key, lo, hi in (("interleave_steps", -1, 64), ("mixed_step_target_ms", 0, 60000),
                        ("long_prefill_token_threshold", 0, 2**63-1), ("preempt_wait_ms", 0, 2**63-1)):
        if type(parsed.get(key)) is not int or not lo <= parsed[key] <= hi:
            raise run_records.RecordError(f"malformed parsed scheduler: {key}")
    # Interleave omission is resolved by engine bring-up, not assumed off.
    steps = parsed["interleave_steps"]
    share = parsed.get("interleave_share")
    if steps == -1:
        try:
            if not isinstance(share, str) or not 0 < float(share) <= 4:
                raise ValueError()
        except ValueError as exc:
            raise run_records.RecordError("malformed parsed interleave share") from exc
    elif share is not None:
        raise run_records.RecordError("unexpected parsed interleave share")
    scheduler = {
        "--mixed-step-target-ms": str(parsed["mixed_step_target_ms"] or "off"),
        "--prefill-room-gate": "on" if parsed["room_gate"] else "off",
        "--long-prefill-token-threshold": str(parsed["long_prefill_token_threshold"] or "0 (off)"),
        "--preempt-wait-ms": str(parsed["preempt_wait_ms"]),
    }
    if parsed["interleave_authored"]:
        scheduler["--prefill-interleave"] = (
            f"auto:{float(share):.6f}" if steps == -1 else str(steps or "off")
        )
    elif steps != 0 or "--prefill-interleave" in expected:
        raise run_records.RecordError("omitted interleave contradicts parsed/manifest values")
    for flag, value in scheduler.items():
        if flag in expected and expected[flag] != value:
            raise run_records.RecordError(f"parsed scheduler contradicts manifest: {flag}")
    expected.update(scheduler)
    return expected


def capture_tuning_receipt(run_dir: Path, url: str, expected: dict[str, str]) -> dict:
    """Bind live native diagnostics to the admitted build/session and exact manifest."""
    context = _session_context(run_dir)
    _verify_live_context(context, url)
    manifest_expected = _context_tuning_expectations(context)
    if not expected and "native_scheduler" not in context["session"].get("selection", {}):
        raise run_records.RecordError("older native owner requires explicit tuning expectations")
    for flag, value in expected.items():
        if flag in manifest_expected and manifest_expected[flag] != value:
            raise run_records.RecordError(f"tuning expectation contradicts manifest: {flag}")
    expected = {**manifest_expected, **expected}
    endpoint = urllib.parse.urlsplit(url)
    if endpoint.path not in ("", "/") or endpoint.query or endpoint.fragment:
        raise run_records.RecordError("tuning receipt requires a bare owner endpoint")
    with urllib.request.urlopen(url.rstrip("/") + "/diagnostics", timeout=10) as response:
        data = json.load(response)
    observed = reconcile_tuning_diagnostics(data, expected)
    after = _session_context(run_dir)
    _verify_live_context(after, url)
    if (after["run_id"], after["manifest_bytes"], after["identity"]) != (
        context["run_id"], context["manifest_bytes"], context["identity"]
    ) or _context_tuning_expectations(after) != manifest_expected:
        raise run_records.RecordError("native owner changed during tuning receipt capture")
    return {
        "schema_version": 1, "run_id": context["run_id"],
        "manifest_sha256": _digest(context["manifest_bytes"]),
        "identity": context["identity"], "expected": expected,
        "observed": observed, "recorded_at": _now(),
    }


TUNING_RECEIPT_FILES = ("native-tuning-receipt.v1.json", "native-tuning-receipt-end.v1.json")


def _tuning_receipt_inputs(context: dict) -> dict:
    paths = [context["run_dir"] / name for name in TUNING_RECEIPT_FILES]
    if not any(p.exists() or p.is_symlink() for p in paths):
        return {}
    receipts = [_read_json(p) for p in paths]
    for receipt in receipts:
        if (set(receipt) != {"schema_version", "run_id", "manifest_sha256", "identity",
                            "expected", "observed", "recorded_at"}
                or receipt["schema_version"] != 1 or receipt["run_id"] != context["run_id"]
                or receipt["manifest_sha256"] != _digest(context["manifest_bytes"])
                or receipt["identity"] != context["identity"]):
            raise run_records.RecordError("native tuning receipt identity mismatch")
        expected, observed = receipt["expected"], receipt["observed"]
        if not isinstance(expected, dict) or not isinstance(observed, dict):
            raise run_records.RecordError("malformed tuning receipt")
        if not all(flag in expected and expected[flag] == value for flag, value in
                   _context_tuning_expectations(context).items()):
            raise run_records.RecordError("tuning receipt does not reconcile the manifest")
        if not set(observed) <= TUNING_DIAGNOSTIC_OPTIONS or not all(
                isinstance(row, dict) and set(row) == {"value", "source"} for row in observed.values()):
            raise run_records.RecordError("unexpected native tuning receipt fields")
        reconcile_tuning_diagnostics({
            "startup_failed": False, "unsupported_configuration": False,
            "configuration": [{"option": flag, **row} for flag, row in observed.items()],
        }, expected)
    if any(receipts[0][key] != receipts[1][key] for key in
           ("run_id", "manifest_sha256", "identity", "expected", "observed")):
        raise run_records.RecordError("native tuning configuration changed during measurement")
    return {"native_tuning_start": paths[0], "native_tuning_end": paths[1]}


def publish_probe(
    run_dir: Path,
    *,
    kind: str,
    result: dict,
    definition: dict,
    argv: list[str],
    extra_inputs: dict[str, bytes | Path],
    work_root: Path | None = None,
    host_snapshot: dict | None = None,
) -> str:
    """Publish exactly the recorded probe outcome, never retest/rejudge it."""
    if kind not in ("needle", "image"):
        raise run_records.RecordError("unknown Radiance lab probe")
    c = _session_context(run_dir)
    if host_snapshot is not None and host_snapshot != c["host_start"]:
        raise run_records.RecordError("probe host facts differ from native owner start")
    if kind == "needle":
        rows = result["rows"]
        if not rows or any(type(r.get("hit")) is not bool for r in rows):
            raise run_records.RecordError(
                "needle probe has missing/nonboolean observed hits"
            )
        rate = sum(r["hit"] for r in rows) / len(rows)
        metric_name, value = "needle_hit_rate", rate
        direction = "higher"
    else:
        if result.get("verdict") not in ("answered", "refused", "wrong"):
            raise run_records.RecordError("image probe has unknown verdict")
        if result.get("expected") not in ("answered", "refused"):
            raise run_records.RecordError("image probe lacks expected verdict")
        metric_name = "image_expected_match"
        value = float(result["verdict"] == result["expected"])
        direction = "higher"
    metrics = {"decode": {}, "prefill": {metric_name: value}, "ttft": {}}
    specs = {
        "decode": {},
        "prefill": {
            metric_name: {"direction": direction, "gated": False},
        },
        "ttft": {},
    }
    inputs = dict(c["inputs"])
    inputs["probe_raw_result"] = measurement_producers.canonical_blob(result)
    inputs.update(extra_inputs)
    return measurement_producers.publish(
        work_root=work_root,
        run_id=f"{c['run_id']}-{kind}-{_digest(inputs['probe_raw_result'])[:16]}",
        recorded_at=_now(),
        identity=c["identity"],
        workload={"name": f"{kind}-probe", "definition": definition},
        metrics=metrics,
        metric_specs=specs,
        effective_config=c["manifest_bytes"],
        argv=argv,
        environment=_native_environment(c["session"]),
        power_settings=c["manifest"]["spec"].get("power", {}),
        inputs=inputs,
        host_facts=_host(c["host_start"]),
        feature_states=c["feature_states"],
        validity={"state": "unverified", "reason_codes": ["VA25_NOT_ATTESTED"]},
        session_id=c["run_id"],
    )


def publish_rsc02_arm(
    result: dict,
    snapshot: dict,
    *,
    work_root: Path | None = None,
    recorded_at: str | None = None,
    host_snapshot: dict | None = None,
) -> str:
    """Capture one still-active RSC02 S15 leased acceptance arm after clean stop."""
    if result.get("state") != "stopped" or result.get("clean_stop") is not True:
        raise run_records.RecordError("RSC02 arm lacks clean verified stop")
    try:
        raw = base64.b64decode(snapshot["manifest_raw_base64"], validate=True)
    except (KeyError, ValueError, binascii.Error) as exc:
        raise run_records.RecordError(
            "RSC02 has no captured exact native manifest bytes"
        ) from exc
    if _digest(raw) != snapshot.get("manifest_file_sha256"):
        raise run_records.RecordError("RSC02 manifest hash mismatch")
    manifest = json.loads(raw)
    if manifest != snapshot["manifest"]:
        raise run_records.RecordError("RSC02 manifest content differs from snapshot")
    roles = snapshot["processes"]["roles"]
    engine = [r for r in roles if r["role"] == "engine"]
    if len(engine) != 1 or not engine[0].get("argv"):
        raise run_records.RecordError("RSC02 lacks a single observed engine argv")
    spec = manifest["spec"]
    runtime = spec["processes"]["engine"]["runtime"]
    build_meta = _file(
        Path(runtime["radiance_home"]).parent / "bigcherry-build-metadata-radiance.json"
    )
    build = _read_json(build_meta)
    meta = result["build_identity"]["engine"]["sha256"]
    if (
        not isinstance(meta, str)
        or not _SHA.fullmatch(meta)
        or meta != build.get("binary_hash")
    ):
        raise run_records.RecordError(
            "RSC02 binary hash differs from verified native build"
        )
    if result["build_identity"].get("build_metadata"):
        declared = result["build_identity"]["build_metadata"].get("sha256")
        if declared != _digest(build_meta.read_bytes()):
            raise run_records.RecordError("RSC02 build metadata digest mismatch")
    if not build.get("source_slice_id") or not build.get("build_id"):
        raise run_records.RecordError(
            "RSC02 native build metadata lacks source/build identity"
        )
    if "completion_sha256" not in result or not _SHA.fullmatch(
        result["completion_sha256"]
    ):
        raise run_records.RecordError("RSC02 lacks exact observed completion digest")
    inputs = {
        "rsc02_observation": measurement_producers.canonical_blob(result),
        "rsc02_live_snapshot": measurement_producers.canonical_blob(snapshot),
        "rsc02_build_metadata": build_meta,
    }
    states = []
    # The historical A arm remains lab evidence, never a native measured record.
    if result["arm"] != "B":
        raise run_records.RecordError("only the sealed native owner arm may publish")
    if result["arm"] == "B":
        if not isinstance(result.get("run_dir"), str):
            raise run_records.RecordError(
                "RSC02 native arm lacks a recorded owner run directory"
            )
        run_dir = Path(result["run_dir"])
        native_session = _read_json(_file(run_dir / "session.json"))
        if native_session.get("state") != "stopped" or native_session.get("errors"):
            raise run_records.RecordError(
                "RSC02 native session has no clean stop receipt"
            )
        replay_manifest = _file(Path(native_session["manifest"]))
        if _digest(replay_manifest.read_bytes()) != _digest(raw):
            raise run_records.RecordError(
                "RSC02 native session's manifest differs from observed arm"
            )
        states_file = _file(run_dir / feature_states.FILE_NAME)
        states = feature_states.read_feature_states(states_file)
        if any(x["engine"] != engines.RADIANCE for x in states):
            raise run_records.RecordError("RSC02 native feature-state engine differs")
        inputs["rsc02_native_feature_states"] = states_file
        inputs["rsc02_native_session"] = run_dir / "session.json"
        native_start_file = _file(run_dir / host_facts.START_RECEIPT_FILE)
        native_start = host_facts.read_owner_start(run_dir)
        if host_snapshot is not None and host_snapshot != native_start:
            raise run_records.RecordError("RSC02 host facts differ from native owner start")
        inputs["rsc02_native_host_start"] = native_start_file
    return measurement_producers.publish(
        work_root=work_root,
        run_id=f"rsc02-{manifest['metadata']['instance_id']}-{result['arm']}",
        recorded_at=recorded_at or _now(),
        identity={
            "engine": engines.RADIANCE,
            "build_id": build["build_id"],
            "binary_sha256": meta,
            "source_revision": build.get("source_commit")
            or ("source_slice:" + build["source_slice_id"]),
            "model": spec["intent"].get("model", "unknown"),
            "layout": spec["intent"]["layout"],
        },
        workload={
            "name": f"rsc02-s15-{result['cohort']}",
            "definition": {"arm": result["arm"], "canary": "Paris", "mode": "leased"},
        },
        metrics={
            "decode": {"canary_correct": float("Paris" in result.get("canary", ""))},
            "prefill": {},
            "ttft": {},
        },
        metric_specs={
            "decode": {"canary_correct": {"direction": "higher", "gated": False}},
            "prefill": {},
            "ttft": {},
        },
        effective_config=raw,
        argv=engine[0]["argv"],
        environment=engine[0].get("env", {}),
        power_settings=result.get("power_before_stop", {}),
        inputs=inputs,
        host_facts=_host(native_start),
        feature_states=states,
        validity={
            "state": "unverified",
            "reason_codes": ["VA25_NOT_ATTESTED"],
        },
    )


def publish_native_lab(run_dir: Path, sidecar: Path, *,
                       work_root: Path | None = None,
                       host_snapshot: dict | None = None) -> str:
    """Ingest explicit metrics after a clean native owner stop, never infer."""
    c = _session_context(run_dir, require_stopped=True)
    path = _file(Path(sidecar))
    if path.parent.resolve() != c["run_dir"]:
        raise run_records.RecordError("native lab evidence is outside the stopped run")
    doc = _read_json(path)
    if (doc.get("schema_version") != 1 or doc.get("run_id") != c["run_id"]
            or doc.get("manifest_sha256") != _digest(c["manifest_bytes"])):
        raise run_records.RecordError("native lab manifest, run ID or schema mismatch")
    metrics = doc.get("metrics")
    if (not isinstance(metrics, dict)
            or set(metrics) != {"decode", "prefill", "ttft"}
            or not any(metrics.values())):
        raise run_records.RecordError("native lab has no typed measured metrics")
    workload = doc.get("workload")
    if (not isinstance(workload, dict) or not isinstance(workload.get("name"), str)
            or not workload["name"] or not isinstance(workload.get("definition"), dict)):
        raise run_records.RecordError("native lab has no typed workload")
    specs = doc.get("metric_specs")
    if not isinstance(specs, dict) or set(specs) != {"decode", "prefill", "ttft"}:
        raise run_records.RecordError("native lab has no typed metric specs")
    return measurement_producers.publish(
        work_root=work_root, run_id=c["run_id"] + "-lab",
        recorded_at=doc["recorded_at"], identity=c["identity"],
        workload=workload, metrics=metrics, metric_specs=specs,
        effective_config=c["manifest_bytes"], argv=_native_argv(c["session"]),
        environment=_native_environment(c["session"]),
        power_settings=c["manifest"]["spec"].get("power", {}),
        inputs={**c["inputs"], "native_lab_measurement": path, **_tuning_receipt_inputs(c)},
        host_facts=_host(host_snapshot), feature_states=c["feature_states"],
        validity={"state": "unverified", "reason_codes": ["VA25_NOT_ATTESTED"]},
        session_id=c["run_id"],
    )
