"""MRC06 Radiance laboratory producers: snapshot native evidence into MRC01.

Every published value has a verified source. The C++ serving owner writes its
own facts/receipts and never loads Python; this platform adapter runs after the
native calibration, or after a lab probe. No historical JSON result is imported.
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import json
import re
import urllib.parse
from datetime import UTC, datetime
from pathlib import Path

from ..core import engines
from ..core import (
    calibration_records,
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
        else {"running", "calibrating", "healthy", "calibration_completed"}
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
    binary = build / "bin/radiance"
    if binary.is_file() and _digest(binary.read_bytes()) != metadata["binary_hash"]:
        raise run_records.RecordError(
            "native engine binary changed since build metadata"
        )
    native_states = _file(run_dir / feature_states.FILE_NAME)
    states = feature_states.read_feature_states(native_states)
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
        "native_session": session_file,
        "native_feature_states": native_states,
        "native_build_metadata": metadata_file,
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
        "inputs": source_inputs,
    }


def _host(snapshot: dict | None) -> dict:
    # The platform ingest may happen after the native owner stopped. Unlike
    # native feature states, these host readings are at ingest time only.
    return snapshot if snapshot is not None else host_facts.collect_host_facts()


def _now() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _native_argv(session: dict, *, fallback: list[str] | None = None) -> list[str]:
    processes = session.get("processes", [])
    for p in processes:
        if p.get("role") == "engine" and isinstance(p.get("argv"), list) and p["argv"]:
            return p["argv"]
    if fallback:
        return fallback
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


def _selected_calibration(path: Path, key: str) -> dict:
    rows = [
        row for row in calibration_records.read_native_toml(path) if row["key"] == key
    ]
    if len(rows) != 1 or rows[0]["status"] != "current":
        raise run_records.RecordError(
            "calibration key is missing, legacy or invalid in MRC04 reader"
        )
    if rows[0]["format"] != "native_v1":
        raise run_records.RecordError(
            "fresh native calibration-run must produce an MRC04 v1 record"
        )
    return rows[0]["record"]


def publish_calibration_run(
    run_dir: Path,
    *,
    work_root: Path | None = None,
    host_snapshot: dict | None = None,
) -> str:
    """Ingest only a durable, completed C++ calibration and its pinned TOML."""
    c = _session_context(run_dir, require_stopped=True)
    events = {e.get("kind") for e in c["session"].get("events", [])}
    if "calibration_completed" not in events:
        raise run_records.RecordError("native calibration completion event is absent")
    receipt_file = _file(c["run_dir"] / "calibration-evidence.v1.json")
    receipt = _read_json(receipt_file)
    key = receipt.get("calibration_store_key")
    if (
        receipt.get("schema_version") != 1
        or receipt.get("session_id") != c["run_id"]
        or not isinstance(key, str)
        or not _SHA.fullmatch(key)
        or c["session"].get("selection", {}).get("calibration_store_key") != key
        or receipt.get("manifest_sha256") != _digest(c["manifest_bytes"])
    ):
        raise run_records.RecordError(
            "native calibration receipt/manifest key or sha256 mismatch"
        )
    store_path = _file(c["run_dir"] / "calibration-store.v1.toml")
    if receipt.get("calibration_store_sha256") != _digest(store_path.read_bytes()):
        raise run_records.RecordError(
            "native calibration store snapshot sha256 mismatch"
        )
    helper_name = receipt.get("helper_log")
    if (
        not isinstance(helper_name, str)
        or Path(helper_name).name != helper_name
        or helper_name in (".", "..")
    ):
        raise run_records.RecordError("unsafe native calibration helper log name")
    helper = _file(c["run_dir"] / helper_name)
    if receipt.get("helper_log_sha256") != _digest(helper.read_bytes()):
        raise run_records.RecordError("native helper log sha256 mismatch")
    # Native-side snapshots are mandatory: no reconstructing missing heat or
    # prompts from mutable site/config paths after a calibration completes.
    pinned_inputs = {}
    for name, role, check in (
        ("calibration-prompts-a.v1", "calibration_prompts_a", "prompts_a_sha256"),
        ("calibration-prompts-b.v1", "calibration_prompts_b", "prompts_b_sha256"),
        ("calibration-heat.v1", "calibration_heat", "heat_sha256"),
    ):
        source = _file(c["run_dir"] / name)
        if receipt.get(check) != _digest(source.read_bytes()):
            raise run_records.RecordError(f"native {role} sha256 mismatch")
        pinned_inputs[role] = source
    expected_heat = (
        c["manifest"]["spec"].get("handoff", {}).get("placement", {}).get("heat_file")
    )
    if not isinstance(expected_heat, str) or not expected_heat:
        raise run_records.RecordError(
            "native calibration manifest lacks heat source identity"
        )
    selected = _selected_calibration(store_path, key)
    provenance = selected["provenance"]
    if (
        provenance["run_id"] != c["run_id"]
        or provenance["source_revision"] != c["identity"]["source_revision"]
    ):
        raise run_records.RecordError(
            "MRC04 calibration run/revision differs from native session"
        )
    if "facts" in c["manifest"]["spec"]:
        facts = measurement_producers.canonical_blob(
            c["manifest"]["spec"]["facts"]
        ).rstrip(b"\n")
        if provenance["host_facts_digest"] != _digest(facts):
            raise run_records.RecordError(
                "native calibration host facts digest differs from strict manifest"
            )
    else:
        raise run_records.RecordError(
            "native calibration manifest lacks declared host facts"
        )
    if selected["key"]["inputs"]["layout"] != c["identity"]["layout"]:
        raise run_records.RecordError(
            "calibration key scope layout differs from measured manifest"
        )
    engine_key = (
        c["manifest"]["spec"]
        .get("plan", {})
        .get("engine_memory", {})
        .get("engine_identity")
    )
    if engine_key and provenance["build_id"] != engine_key:
        raise run_records.RecordError(
            "calibrated engine identity differs from measured manifest"
        )

    readings = selected["quantities"]["measurements"]
    metrics = {"decode": {}, "prefill": {}, "ttft": {}}
    specs = {"decode": {}, "prefill": {}, "ttft": {}}
    # These are calibrated decode/handoff quantities, NOT TTFT or prefill
    # timing. Other quantities and explicit absences are retained in full
    # in selected_calibration_record.
    for name, value in readings.items():
        if type(value) is not int:
            continue  # effect_window_noise_ladder is a structured observation
        metrics["decode"][name] = value
        specs["decode"][name] = {
            "direction": ("higher" if name.endswith(("_per_s", "_rate")) else "lower"),
            "gated": False,
        }
    inputs = dict(c["inputs"])
    inputs.update(pinned_inputs)
    inputs.update(
        {
            "native_calibration_receipt": receipt_file,
            "native_calibration_store": store_path,
            "calibration_helper_log": helper,
            "selected_calibration_record": measurement_producers.canonical_blob(
                selected
            ),
        }
    )
    sha = measurement_producers.publish(
        work_root=work_root,
        run_id=c["run_id"] + "-calibration",
        recorded_at=receipt["observed_at"],
        identity=c["identity"],
        workload={
            "name": "live-policy-calibration",
            "definition": {
                "kind": "radiance.live_policy",
                "key": key,
                "objective": selected["key"]["inputs"]["objective"],
                "method": "native-finite-ABBA",
            },
        },
        metrics=metrics,
        metric_specs=specs,
        effective_config=c["manifest_bytes"],
        argv=_native_argv(c["session"]),
        environment=_native_environment(c["session"]),
        power_settings=c["manifest"]["spec"].get("power", {}),
        inputs=inputs,
        host_facts=_host(host_snapshot),
        feature_states=c["feature_states"],
        calibration_selection={
            "status": "selected", "record": selected,
            "quantity_blobs": {
                "native_calibration_quantities": measurement_producers.canonical_blob(
                    selected["quantities"]
                )
            },
        },
        validity={"state": "unverified", "reason_codes": ["VA25_NOT_ATTESTED"]},
        session_id=c["run_id"],
    )
    pointer = c["run_dir"] / "calibration-record.sha256"
    if pointer.exists() and pointer.read_text().strip() != sha:
        raise run_records.RecordError(
            "calibration run already points to another record"
        )
    pointer.write_text(sha + "\n")
    return sha


def verify_live_target(run_dir: Path, url: str) -> None:
    """Refuse an unrelated/missing native owner before sending any probe HTTP.

    Binding the endpoint prevents attributing a healthy response from one
    server to a different run's strict manifest.
    """
    context = _session_context(run_dir)
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
        host_facts=_host(host_snapshot),
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
    # The B arm is the native owner, which publishes MRC03. The A arm is the
    # legacy bash owner and must never inherit B's stale feature state file.
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
        host_facts=_host(host_snapshot),
        feature_states=states,
        validity={
            "state": "unverified",
            "reason_codes": ["LEGACY_ARM_NOT_VA25_ATTESTED"],
        },
    )
