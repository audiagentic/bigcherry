"""Current-schema managed experiment bundle runner and validator (HI47)."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import time
import uuid
from pathlib import Path
from typing import Any

from .tune_journal import atomic_write, canonical

SCHEMA_VERSION = 1
ALLOWED_ENV = {
    "GGML_HIP_DISPATCH_MODE", "GGML_HIP_DISPATCH_CACHE",
    "GGML_HIP_TUNE_WORKLOAD", "GGML_HIP_TUNE_HOT_SIGNATURES",
    "GGML_HIP_TUNE_WARMUP", "GGML_HIP_TUNE_SCREEN_SAMPLES",
    "GGML_HIP_TUNE_FINAL_SAMPLES", "GGML_HIP_TUNE_LAUNCHES_PER_SAMPLE",
    "HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES",
}
SECRET = re.compile(r"(?i)(secret|token|password|passwd|api[_-]?key|credential)")


class BundleError(RuntimeError):
    pass


def file_hash(path: Path) -> str:
    state = hashlib.blake2b(digest_size=16)
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            state.update(chunk)
    return state.hexdigest()


def relative_artifact(root: Path, path: Path) -> str:
    resolved_root = root.resolve()
    resolved = path.resolve()
    try:
        return resolved.relative_to(resolved_root).as_posix()
    except ValueError as exc:
        raise BundleError(f"artifact escapes experiment bundle: {path}") from exc


def safe_environment(environment: dict[str, str]) -> dict[str, str]:
    captured: dict[str, str] = {}
    for key in sorted(ALLOWED_ENV):
        value = environment.get(key)
        if value is None:
            continue
        if SECRET.search(key) or SECRET.search(value):
            raise BundleError(f"secret-pattern environment value refused: {key}")
        captured[key] = value
    return captured


def bundle_hash(document: dict[str, Any]) -> str:
    payload = dict(document)
    payload.pop("bundle_hash", None)
    return hashlib.blake2b(
        canonical(payload), digest_size=16, person=b"bc-experiment-v1"
    ).hexdigest()


def write_document(path: Path, document: dict[str, Any]) -> None:
    output = dict(document)
    output["bundle_hash"] = bundle_hash(output)
    atomic_write(path, json.dumps(output, sort_keys=True, indent=2).encode("ascii") + b"\n")


def run_managed(
    root: Path,
    command: list[str],
    *,
    source_revision: str,
    manifest_hash: str,
    build_descriptor_hash: str,
    model: Path,
    role: str,
    environment: dict[str, str] | None = None,
    parent_experiment_id: str | None = None,
) -> int:
    if root.exists():
        raise BundleError(f"bundle already exists: {root}")
    root.mkdir(parents=True)
    experiment_id = str(uuid.uuid4())
    env = os.environ.copy()
    env.update(environment or {})
    intent = {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": experiment_id,
        "parent_experiment_id": parent_experiment_id,
        "state": "intent",
        "role": role,
        "source_revision": source_revision,
        "manifest_hash": manifest_hash,
        "build_descriptor_hash": build_descriptor_hash,
        "model": {"path": model.name, "hash": file_hash(model), "bytes": model.stat().st_size},
        "command": command,
        "environment": safe_environment(env),
        "host_started_ns": time.time_ns(),
        "monotonic_started_ns": time.monotonic_ns(),
        "artifacts": [],
        "intended_artifacts": ["stdout.log", "stderr.log"],
        "capabilities": [],
    }
    document_path = root / "experiment.json"
    write_document(document_path, intent)  # intent is durable before child start
    stdout = root / "stdout.log"
    stderr = root / "stderr.log"
    started = time.monotonic_ns()
    stdout_bytes = b""
    stderr_bytes = b""
    try:
        completed = subprocess.run(command, text=False, capture_output=True, env=env)
        stdout_bytes = completed.stdout
        stderr_bytes = completed.stderr
        state = "completed" if completed.returncode == 0 else "failed"
        returncode = completed.returncode
    except KeyboardInterrupt:
        stderr_bytes = b"interrupted by host\n"
        state, returncode = "interrupted", 130
    except OSError as exc:
        stderr_bytes = f"process launch failed: {exc}\n".encode("utf-8", "backslashreplace")
        state, returncode = "failed", 127
    stdout.write_bytes(stdout_bytes)
    stderr.write_bytes(stderr_bytes)
    final = dict(intent)
    final.update(
        state=state, returncode=returncode,
        host_finished_ns=time.time_ns(),
        monotonic_duration_ns=time.monotonic_ns() - started,
        artifacts=[
            {"role": "stdout", "path": relative_artifact(root, stdout), "bytes": stdout.stat().st_size, "hash": file_hash(stdout)},
            {"role": "stderr", "path": relative_artifact(root, stderr), "bytes": stderr.stat().st_size, "hash": file_hash(stderr)},
        ],
        capabilities=["process_evidence"],
    )
    write_document(document_path, final)
    return returncode


def validate(root: Path, *, required_capabilities: set[str] | None = None) -> dict[str, Any]:
    path = root / "experiment.json"
    try:
        document = json.loads(path.read_text(encoding="ascii"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise BundleError(f"cannot read current experiment: {exc}") from exc
    if document.get("schema_version") != SCHEMA_VERSION:
        raise BundleError("rerun_or_external_conversion_required: experiment schema")
    if document.get("bundle_hash") != bundle_hash(document):
        raise BundleError("experiment document hash mismatch")
    if document.get("state") not in {"completed", "failed", "interrupted", "intent"}:
        raise BundleError("invalid lifecycle state")
    roles: set[str] = set()
    for artifact in document.get("artifacts", []):
        role = artifact.get("role")
        if role in roles:
            raise BundleError(f"duplicate artifact role: {role}")
        roles.add(role)
        raw_path = artifact.get("path", "")
        if Path(raw_path).is_absolute() or ".." in Path(raw_path).parts:
            raise BundleError("absolute/path-escaping artifact")
        target = root / raw_path
        if (not target.is_file() or target.stat().st_size != artifact.get("bytes") or
                file_hash(target) != artifact.get("hash")):
            raise BundleError(f"artifact missing or modified: {raw_path}")
    required = required_capabilities or set()
    missing = required - set(document.get("capabilities", []))
    if missing:
        raise BundleError("capability-incomplete: " + ", ".join(sorted(missing)))
    promotable = document["state"] == "completed" and document.get("returncode") == 0 and not missing
    return {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": document["experiment_id"],
        "bundle_hash": document["bundle_hash"],
        "state": document["state"],
        "valid": True,
        "promotable": promotable,
        "capabilities": document.get("capabilities", []),
    }


def validate_parent_graph(documents: list[dict[str, Any]]) -> None:
    parents = {doc["experiment_id"]: doc.get("parent_experiment_id") for doc in documents}
    for start in parents:
        seen: set[str] = set()
        current: str | None = start
        while current is not None and current in parents:
            if current in seen:
                raise BundleError("experiment parent graph contains a cycle")
            seen.add(current)
            current = parents[current]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="bigcherry experiment")
    sub = parser.add_subparsers(dest="command", required=True)
    validate_cmd = sub.add_parser("validate")
    validate_cmd.add_argument("bundle", type=Path)
    validate_cmd.add_argument("--require", action="append", default=[])
    run_cmd = sub.add_parser("run")
    run_cmd.add_argument("bundle", type=Path)
    run_cmd.add_argument("--source-revision", required=True)
    run_cmd.add_argument("--manifest-hash", required=True)
    run_cmd.add_argument("--build-descriptor-hash", required=True)
    run_cmd.add_argument("--model", type=Path, required=True)
    run_cmd.add_argument("--role", required=True)
    run_cmd.add_argument("--parent-experiment-id", default=None)
    run_cmd.add_argument("--env", action="append", default=[])
    child_command: list[str] = []
    parse_argv = list(argv) if argv is not None else sys.argv[1:]
    if parse_argv[:1] == ["run"] and "--" in parse_argv:
        separator = parse_argv.index("--")
        child_command = parse_argv[separator + 1:]
        parse_argv = parse_argv[:separator]
    args = parser.parse_args(parse_argv)
    try:
        if args.command == "validate":
            result = validate(args.bundle, required_capabilities=set(args.require))
            status = 0
        else:
            command = child_command
            if not command:
                raise BundleError("managed run command is required after --")
            environment: dict[str, str] = {}
            for item in args.env:
                key, separator, value = item.partition("=")
                if not separator or key not in ALLOWED_ENV:
                    raise BundleError(f"environment override is not allow-listed: {key}")
                environment[key] = value
            status = run_managed(
                args.bundle, command, source_revision=args.source_revision,
                manifest_hash=args.manifest_hash,
                build_descriptor_hash=args.build_descriptor_hash,
                model=args.model, role=args.role, environment=environment,
                parent_experiment_id=args.parent_experiment_id,
            )
            result = validate(args.bundle)
    except (OSError, BundleError) as exc:
        print(f"invalid: {exc}")
        return 1
    print(json.dumps(result, sort_keys=True))
    return status


if __name__ == "__main__":
    raise SystemExit(main())
