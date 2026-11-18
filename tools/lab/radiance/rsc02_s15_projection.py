"""RSC02 S15 semantic-execution witness and card-free fake-owner rehearsal.

This is a comparison/admission preflight, not a new statistical benchmark.
The hardware lane uses bigcherry ab-benchmark's balanced schedule and recorded
workloads. No command in this file can acquire a GPU lease.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import subprocess
import sys
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
V2_EXAMPLE = REPO / "docs/design/serving-manifest-v2/example.flashnext-4gpu.json"


class ProjectionError(ValueError):
    pass


def _dict(value: object, where: str) -> dict:
    if not isinstance(value, dict):
        raise ProjectionError(f"{where}: expected an object")
    return value


def _get(record: dict, name: str, where: str) -> object:
    if name not in record:
        raise ProjectionError(f"{where}: missing {name}")
    return record[name]


def _stable_facts(facts: object) -> dict:
    src = _dict(facts, "spec.facts")
    host = _dict(_get(src, "host", "spec.facts"), "spec.facts.host")
    gpus = _get(src, "gpus", "spec.facts")
    if not isinstance(gpus, list) or not gpus:
        raise ProjectionError("spec.facts.gpus: expected nonempty ordered card list")
    devices = []
    for i, gpu in enumerate(gpus):
        g = _dict(gpu, f"spec.facts.gpus[{i}]")
        vram = _dict(_get(g, "vram", f"gpu[{i}]"), f"gpu[{i}].vram")
        devices.append({
            "bdf": _get(g, "bdf", f"gpu[{i}]"),
            "arch": _get(g, "arch", f"gpu[{i}]"),
            "hip_ordinal": _get(g, "hip_ordinal", f"gpu[{i}]"),
            "vram": {"total_mib": _get(vram, "total_mib", f"gpu[{i}].vram")},
        })
    if len({str(g["bdf"]) for g in devices}) != len(devices):
        raise ProjectionError("spec.facts.gpus: duplicate physical card")
    # Dynamic free RAM/VRAM and cgroup room are live-admission evidence, not
    # stable execution identity. The resolved resource ledger is NOT removed.
    return {"cpu": _get(host, "cpu", "spec.facts.host"), "gpus": devices}


def _processes(value: object, manifest_path: str, expected_roles: list[str]) -> dict:
    proc = _dict(value, "snapshot.processes")
    rows = _get(proc, "roles", "snapshot.processes")
    if not isinstance(rows, list) or not rows:
        raise ProjectionError("snapshot.processes.roles: nonempty observed role list required")
    results = []
    for i, row in enumerate(rows):
        p = _dict(row, f"snapshot.processes.roles[{i}]")
        role = _get(p, "role", f"process[{i}]")
        argv = _get(p, "argv", f"process[{i}]")
        env = _get(p, "env", f"process[{i}]")
        if not isinstance(role, str) or not role:
            raise ProjectionError("empty process role")
        if not isinstance(argv, list) or not argv or any(not isinstance(s, str) for s in argv):
            raise ProjectionError(f"{role}: argv must be a nonempty list of strings")
        if not isinstance(env, dict) or any(not isinstance(k, str) or not isinstance(v, str) for k, v in env.items()):
            raise ProjectionError(f"{role}: env must be observed key/value strings")
        if argv.count("--manifest") != 1:
            raise ProjectionError(f"{role}: exactly one --manifest FILE is required")
        idx = argv.index("--manifest")
        if idx + 1 >= len(argv) or argv[idx + 1] != manifest_path:
            raise ProjectionError(f"{role}: observed manifest argument not bound to captured bytes")
        if argv.count("--role") != 1:
            raise ProjectionError(f"{role}: exactly one --role NAME is required")
        role_idx = argv.index("--role")
        if role_idx + 1 >= len(argv) or argv[role_idx + 1] != role:
            raise ProjectionError(f"{role}: observed --role must match captured process role")
        normal = list(argv)
        normal[idx + 1] = "<manifest-bytes>"
        results.append({"role": role, "argv": normal, "env": dict(sorted(env.items()))})
    if len({p["role"] for p in results}) != len(results):
        raise ProjectionError("duplicate process role")
    observed = {p["role"] for p in results}
    expected = set(expected_roles)
    missing = sorted(expected - observed)
    extra = sorted(observed - expected)
    if missing:
        raise ProjectionError(f"snapshot.processes.roles: missing observed roles: {missing}")
    if extra:
        raise ProjectionError(f"snapshot.processes.roles: unexpected observed roles: {extra}")
    return {"roles": results}


def projection(snapshot: dict, *, verify_bytes: bool = True) -> dict:
    """Project all launch semantics, excluding only explicitly named bookkeeping.

    Real acceptance requires a byte-bound snapshot; the private dry-run path
    exercises normalization with generated fixtures. A strict-v2 reader and
    producer digest verification are independent required acceptance gates.
    """
    record = _dict(snapshot, "snapshot")
    if record.get("schema_version") != 1:
        raise ProjectionError("snapshot must use schema_version 1")
    path = _get(record, "manifest_path", "snapshot")
    if not isinstance(path, str) or not path:
        raise ProjectionError("snapshot.manifest_path must be nonempty")
    manifest = _dict(_get(record, "manifest", "snapshot"), "snapshot.manifest")
    if verify_bytes:
        expected = _get(record, "manifest_file_sha256", "snapshot")
        if not isinstance(expected, str) or len(expected) != 64:
            raise ProjectionError("snapshot.manifest_file_sha256 is required")
        contents = Path(path).read_bytes()
        if hashlib.sha256(contents).hexdigest() != expected or json.loads(contents) != manifest:
            raise ProjectionError("manifest path, observed bytes and SHA-256 disagree")
    metadata = _dict(_get(manifest, "metadata", "manifest"), "manifest.metadata")
    producer = _dict(_get(metadata, "producer", "metadata"), "metadata.producer")
    build = _get(producer, "build_sha256", "metadata.producer")
    if not isinstance(build, str) or len(build) != 64:
        raise ProjectionError("producer build SHA-256 is required")
    spec = copy.deepcopy(_dict(_get(manifest, "spec", "manifest"), "manifest.spec"))
    facts = spec.pop("facts", None)
    if facts is None:
        raise ProjectionError("manifest.spec.facts missing")
    spec["stable_facts"] = _stable_facts(facts)
    if "handoff" in spec:
        handoff = _dict(spec["handoff"], "spec.handoff")
        transport = _dict(_get(handoff, "transport", "spec.handoff"), "spec.handoff.transport")
        shm = _get(transport, "shm", "spec.handoff.transport")
        if not isinstance(shm, str) or not shm.startswith("/") or len(shm) < 2:
            raise ProjectionError("handoff shared-memory namespace invalid")
        # Distinct per-owner namespaces must not spuriously fail semantic A/B.
        # They are separately admitted for collision safety on every start.
        transport["shm"] = "<per-owner-shm>"
        telemetry = _dict(_get(handoff, "telemetry", "spec.handoff"), "spec.handoff.telemetry")
        if telemetry.get("histogram_file") is not None:
            if not isinstance(telemetry["histogram_file"], str) or not telemetry["histogram_file"]:
                raise ProjectionError("histogram output must be a valid path")
            telemetry["histogram_file"] = "<per-run-histogram-output>"
    proc = _dict(_get(spec, "processes", "spec"), "spec.processes")
    helpers = _get(proc, "helpers", "spec.processes")
    if not isinstance(helpers, list):
        raise ProjectionError("spec.processes.helpers: expected a list")
    expected_roles = ["engine"]
    for i, helper in enumerate(helpers):
        h = _dict(helper, f"spec.processes.helpers[{i}]")
        role = _get(h, "role", f"spec.processes.helpers[{i}]")
        if not isinstance(role, str) or not role or role in expected_roles:
            raise ProjectionError(f"spec.processes.helpers[{i}]: invalid or duplicate role")
        expected_roles.append(role)
    engine = _dict(_get(proc, "engine", "spec.processes"), "spec.processes.engine")
    server = _dict(_get(engine, "server", "spec.processes.engine"), "spec.processes.engine.server")
    port = _get(server, "port", "spec.processes.engine.server")
    if type(port) is not int or not 0 < port < 65536:
        raise ProjectionError("engine server port must be an actual TCP port")
    # A/B may use separate bounded endpoints; the test client is bound to
    # each observed port. Port is not an inference or GPU-execution parameter.
    server["port"] = "<session-endpoint>"
    return {
        "producer_build_sha256": build,
        "spec": spec,
        "processes": _processes(_get(record, "processes", "snapshot"), path, expected_roles),
    }


def difference_paths(a: object, b: object, at: str = "") -> list[str]:
    if type(a) is not type(b):
        return [at or "/"]
    if isinstance(a, dict):
        results = []
        for key in sorted(set(a) | set(b)):
            child = at + "/" + key.replace("~", "~0").replace("/", "~1")
            if key not in a or key not in b:
                results.append(child)
            else:
                results.extend(difference_paths(a[key], b[key], child))
        return results
    if isinstance(a, list):
        if len(a) != len(b):
            return [at + "/length"]
        return [x for i, (av, bv) in enumerate(zip(a, b))
                for x in difference_paths(av, bv, f"{at}/{i}")]
    return [] if a == b else [at or "/"]


def compare(legacy: dict, native: dict, *, verify_bytes: bool = True) -> dict:
    a = projection(legacy, verify_bytes=verify_bytes)
    b = projection(native, verify_bytes=verify_bytes)
    differences = difference_paths(a, b)
    return {
        "status": "PASS" if not differences else "FAIL",
        "equal": not differences,
        "different_paths": differences,
        "legacy_projection_sha256": hashlib.sha256(
            json.dumps(a, sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
        "native_projection_sha256": hashlib.sha256(
            json.dumps(b, sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
    }


def _fixture(manifest: dict, location: str) -> dict:
    helpers = manifest["spec"]["processes"]["helpers"]
    roles = [{"role": "engine", "argv": ["radiance", "--manifest", location, "--role", "engine"],
              "env": {"ROCR_VISIBLE_DEVICES": "0"}}]
    roles += [
        {"role": f"helper{i}",
         "argv": ["r3_moe_worker", "--manifest", location, "--role", f"helper{i}"],
         "env": {"ROCR_VISIBLE_DEVICES": str(i + 1)}}
        for i, _ in enumerate(helpers)
    ]
    return {"schema_version": 1, "manifest_path": location, "manifest": manifest,
            "processes": {"roles": roles}}


def dry_run(native_test: Path, *, legacy_launch_tests: bool = False) -> int:
    """Run two real fake-process lifecycles, plus projection failure injection.

    This is NOT proof that real Bash and native manifest projections match.
    """
    original = json.loads(V2_EXAMPLE.read_text(encoding="utf-8"))
    other = copy.deepcopy(original)
    other["metadata"]["instance_id"] = "dry-run-nonce-2"
    other["metadata"]["content_digest"] = "dry-run-nontrusted-placeholder"
    other["provenance"] = {"": {"source": "dry-run"}}
    other["spec"]["handoff"]["transport"]["shm"] = "/r3-dry-run-unique"
    other["spec"]["handoff"]["telemetry"]["histogram_file"] = "/run-two.hist"
    other["spec"]["processes"]["engine"]["server"]["port"] += 1
    other["spec"]["facts"]["host"]["memory"]["mem_available_mib"] = 1
    for card in other["spec"]["facts"]["gpus"]:
        card["vram"]["available_mib"] = 1
    a = _fixture(original, "/fake-run-a/manifest.json")
    b = _fixture(other, "/fake-run-b/manifest.json")
    if not compare(a, b, verify_bytes=False)["equal"]:
        raise ProjectionError("dry-run: non-semantic per-run identities changed projection")
    for category, field in [("engine", "num_speculative_tokens"), ("handoff", "period_s")]:
        modified = copy.deepcopy(b)
        if category == "engine":
            node = modified["manifest"]["spec"]["processes"]["engine"]["execution"]
        else:
            node = modified["manifest"]["spec"]["handoff"]["telemetry"]
        node[field] += 1
        if compare(a, modified, verify_bytes=False)["equal"]:
            raise ProjectionError(f"dry-run: {field} semantic mismatch was dropped")
    # The comparator rejects incomplete process captures before projecting
    # either arm. A deliberate missing-role injection must trigger that typed
    # refusal; it is not a failed rehearsal or evidence of A/B equivalence.
    incomplete = {**b, "processes": {"roles": b["processes"]["roles"][:-1]}}
    try:
        report = compare(a, incomplete, verify_bytes=False)
    except ProjectionError as error:
        if "missing observed roles" not in str(error):
            raise
    else:
        if report["equal"]:
            raise ProjectionError("dry-run: missing helper was ignored")
    print("RSC02 S15: projection unit gate and deliberate mismatches PASS", flush=True)

    result = subprocess.run([str(native_test)], cwd=native_test.parent,
                            timeout=240, check=False)
    if result.returncode:
        print("RSC02 S15: native stored-manifest/real owner fake engine FAILED", file=sys.stderr)
        return 1
    print("RSC02 S15: real native owner -> fake engine PASS", flush=True)
    if not legacy_launch_tests:
        print("RSC02 S15: legacy Bash fake-launcher runs require --legacy-launch-tests (not CI)")
        return 0
    os.environ.setdefault("BIGCHERRY_ENVIRONMENT", str(REPO / "config/environment.example.toml"))
    scratch = native_test.parent / "rsc02-s15-tmp"
    scratch.mkdir(parents=True, exist_ok=True)
    os.environ["TMPDIR"] = str(scratch)
    sys.path.insert(0, str(REPO / "tools"))
    sys.path.insert(0, str(REPO / "tools/tests/lab"))
    from test_launch_flashnext_cleanup import LaunchCleanupTests
    tests = unittest.TestSuite([
        LaunchCleanupTests("test_dense_layout_starts_from_its_manifest_like_every_other_layout"),
        LaunchCleanupTests("test_a_tensor_parallel_engine_with_two_helpers_starts_from_its_manifest"),
    ])
    result = unittest.TextTestRunner(verbosity=2).run(tests)
    print("RSC02 S15: Bash launcher -> fake engine dry-run", "PASS" if result.wasSuccessful() else "FAIL")
    return 0 if result.wasSuccessful() else 1


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="command", required=True)
    comp = sub.add_parser("compare", help="require equal byte-bound observed launch projections")
    comp.add_argument("--legacy", type=Path, required=True)
    comp.add_argument("--native", type=Path, required=True)
    comp.add_argument("--output", type=Path)
    rehearsal = sub.add_parser("dry-run", help="run real card-free fake engine call graphs")
    rehearsal.add_argument("--native-test", type=Path, required=True)
    rehearsal.add_argument("--legacy-launch-tests", action="store_true",
                           help="run the long legacy fake-process launcher cases separately from CI")
    args = ap.parse_args(argv)
    try:
        if args.command == "dry-run":
            return dry_run(args.native_test.resolve(), legacy_launch_tests=args.legacy_launch_tests)
        report = compare(json.loads(args.legacy.read_text()), json.loads(args.native.read_text()))
        if args.output:
            args.output.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report, indent=2))
        return 0 if report["equal"] else 1
    except (OSError, ValueError, KeyError, subprocess.TimeoutExpired) as exc:
        print(f"RSC02 S15: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
