"""Failure-injection checks against the real HI47 managed-process seam."""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

import bigcherry.experiment.bundle as bundle

REV = "a" * 40
MANIFEST = "b" * 32
BUILD = "c" * 32


def check(value: bool, message: str) -> int:
    if not value:
        raise AssertionError(message)
    return 1


def kwargs(model: Path) -> dict[str, object]:
    return {
        "source_revision": REV,
        "manifest_hash": MANIFEST,
        "build_descriptor_hash": BUILD,
        "model": model,
        "role": "rcd-failure-injection",
    }


def main() -> int:
    checks = 0
    with tempfile.TemporaryDirectory(prefix="bigcherry-bundle-failure-") as temp:
        root = Path(temp)
        model = root / "model.gguf"
        model.write_bytes(b"model\n")

        # The durable intent must exist before subprocess.run can be invoked.
        intent_root = root / "intent-before-spawn"
        observed: dict[str, object] = {}

        def inspect_before_spawn(command, **_):
            document = json.loads((intent_root / "experiment.json").read_text())
            observed.update(document)
            return subprocess.CompletedProcess(command, 0, b"child-ok\n", b"")

        with patch.object(bundle.subprocess, "run", side_effect=inspect_before_spawn):
            rc = bundle.run_managed(intent_root, ["fake-child"], **kwargs(model))
        checks += check(rc == 0, "intent-before-spawn case failed")
        checks += check(observed.get("state") == "intent", "intent not durable before spawn")
        checks += check(observed.get("artifacts") == [], "terminal artifacts existed before spawn")
        checks += check("returncode" not in observed, "terminal return code existed before spawn")
        checks += check(bundle.validate(intent_root)["state"] == "completed", "post-spawn bundle invalid")

        # Host interruption is a durable non-success state, never inferred success.
        interrupted = root / "interrupted"
        with patch.object(bundle.subprocess, "run", side_effect=KeyboardInterrupt):
            rc = bundle.run_managed(interrupted, ["fake-child"], **kwargs(model))
        checks += check(rc == 130, "KeyboardInterrupt return code is not 130")
        interrupted_doc = bundle.validate(interrupted)
        checks += check(interrupted_doc["state"] == "interrupted", "interruption not durable")
        checks += check(not interrupted_doc["promotable"], "interrupted bundle became promotable")
        checks += check("interrupted by host" in (interrupted / "stderr.log").read_text(), "interruption evidence missing")

        # An existing bundle is evidence and must never be truncated/reused.
        try:
            bundle.run_managed(intent_root, ["fake-child"], **kwargs(model))
        except bundle.BundleError as exc:
            checks += check("already exists" in str(exc), "wrong duplicate-bundle failure")
        else:
            raise AssertionError("existing bundle was overwritten")

        # Artifact mutation after completion is detected by validation.
        tamper = root / "tamper"
        rc = bundle.run_managed(
            tamper,
            [sys.executable, "-c", "print('immutable')"],
            **kwargs(model),
        )
        checks += check(rc == 0 and bundle.validate(tamper)["valid"], "tamper fixture did not start valid")
        with (tamper / "stdout.log").open("ab") as handle:
            handle.write(b"tampered\n")
        try:
            bundle.validate(tamper)
        except bundle.BundleError as exc:
            checks += check("modified" in str(exc), "tamper rejected for unexpected reason")
        else:
            raise AssertionError("modified stdout evidence validated")

    print(json.dumps({"checks": checks, "ok": True, "scope": "real-managed-process-failure-injection"}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
