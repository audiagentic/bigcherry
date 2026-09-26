"""Offline/CI integration smoke for RCD job-service process semantics.

Unlike mock_pipeline.py this imports and executes current BigCherry production
modules and launches real child processes. It deliberately does not claim
Slurm/ROCm/GPU validation.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

from bigcherry.experiment.bundle import run_managed, validate
from bigcherry.tuning.journal import JournalWriter, read_current

REV = "a" * 40
MANIFEST = "b" * 32
BUILD = "c" * 32


def check(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def run_bundle_cases(root: Path) -> int:
    checks = 0
    model = root / "model.gguf"
    model.write_bytes(b"fake-model-for-process-smoke\n")

    # Direct production-module call, real child process, success lifecycle.
    success = root / "success"
    rc = run_managed(
        success,
        [sys.executable, "-c", "import sys; print('OUT'); print('ERR', file=sys.stderr)"],
        source_revision=REV,
        manifest_hash=MANIFEST,
        build_descriptor_hash=BUILD,
        model=model,
        role="rcd-process-smoke",
    )
    check(rc == 0, "run_managed success returncode")
    checks += 1
    doc = validate(success)
    check(doc["state"] == "completed" and doc["promotable"], "success validates/promotable")
    checks += 1
    check((success / "stdout.log").read_text().strip() == "OUT", "stdout persisted")
    checks += 1
    check((success / "stderr.log").read_text().strip() == "ERR", "stderr persisted")
    checks += 1

    # Scientific/harness-style nonzero remains durable failure evidence.
    failed = root / "failed-76"
    rc = run_managed(
        failed,
        [sys.executable, "-c", "raise SystemExit(76)"],
        source_revision=REV,
        manifest_hash=MANIFEST,
        build_descriptor_hash=BUILD,
        model=model,
        role="rcd-process-smoke",
    )
    check(rc == 76, "nonzero child code preserved")
    checks += 1
    failed_doc = validate(failed)
    check(failed_doc["state"] == "failed" and not failed_doc["promotable"], "failure validates but is not promotable")
    checks += 1

    # Launch failure is represented as failed process evidence, not success.
    launch = root / "launch-failure"
    rc = run_managed(
        launch,
        [str(root / "definitely-not-an-executable")],
        source_revision=REV,
        manifest_hash=MANIFEST,
        build_descriptor_hash=BUILD,
        model=model,
        role="rcd-process-smoke",
    )
    check(rc == 127, "launch failure returns 127")
    checks += 1
    launch_doc = validate(launch)
    check(launch_doc["state"] == "failed", "launch failure durable state")
    checks += 1

    # Exercise actual module CLI in a separate BigCherry Python process.
    cli = root / "cli-success"
    env = os.environ.copy()
    command = [
        sys.executable,
        "-m",
        "bigcherry.experiment.bundle",
        "run",
        str(cli),
        "--source-revision",
        REV,
        "--manifest-hash",
        MANIFEST,
        "--build-descriptor-hash",
        BUILD,
        "--model",
        str(model),
        "--role",
        "rcd-cli-smoke",
        "--",
        sys.executable,
        "-c",
        "print('CLI_CHILD_OK')",
    ]
    completed = subprocess.run(command, env=env, text=True, capture_output=True)
    check(completed.returncode == 0, f"module CLI success: {completed.stderr}")
    checks += 1
    check(validate(cli)["state"] == "completed", "CLI-generated bundle validates")
    checks += 1

    # Moderate large-output regression: validates artifact handling while also
    # keeping the known capture_output=True scalability gap visible in RCD plans.
    large = root / "large-output"
    bytes_expected = 8 * 1024 * 1024
    rc = run_managed(
        large,
        [sys.executable, "-c", f"import sys; sys.stdout.buffer.write(b'x'*{bytes_expected})"],
        source_revision=REV,
        manifest_hash=MANIFEST,
        build_descriptor_hash=BUILD,
        model=model,
        role="rcd-process-smoke",
    )
    check(rc == 0 and (large / "stdout.log").stat().st_size == bytes_expected, "8MiB child output persisted")
    checks += 1
    check(validate(large)["state"] == "completed", "large-output bundle validates")
    checks += 1
    return checks


def run_journal_cases(root: Path) -> int:
    checks = 0
    path = root / "journal.jsonl"
    writer = JournalWriter(
        path,
        "rcd-smoke",
        REV,
        MANIFEST,
        "d" * 32,
        durability_mode="durable_each",
    )
    writer.append("progress", {"step": 1})
    writer.complete({"ok": True})
    current = read_current(path)
    check(current["complete"] and current["last_durable_sequence"] == 3, "HI48 complete journal")
    checks += 1

    # Simulate a killed/torn final append after a durable prefix.
    with path.open("ab") as handle:
        handle.write(b'{"torn":')
        handle.flush()
        os.fsync(handle.fileno())
    recovered = read_current(path, recover_tail=True)
    check(recovered["corrupt_tail"], "HI48 detects torn tail")
    checks += 1
    check(recovered["last_durable_sequence"] == 3, "HI48 preserves complete prefix")
    checks += 1
    check(not recovered["complete"], "corrupt tail cannot be reported complete")
    checks += 1
    return checks


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="bigcherry-rcd-process-") as temp:
        root = Path(temp)
        checks = run_bundle_cases(root) + run_journal_cases(root)
    print(json.dumps({
        "checks": checks,
        "ok": True,
        "scope": "real-bigcherry-modules-and-child-processes",
        "known_gap": "experiment.bundle.run_managed currently buffers child output; RCD requires streaming before 1.5GB-log production use",
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
