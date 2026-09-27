from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from bigcherry.experiment.bundle import run_managed, validate


class ManagedStreamingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.model = self.root / "model.gguf"
        self.model.write_bytes(b"model-fixture")

    def tearDown(self) -> None:
        self.temp.cleanup()

    def _run(self, bundle: Path, command: list[str]) -> int:
        return run_managed(
            bundle,
            command,
            source_revision="a" * 40,
            manifest_hash="b" * 32,
            build_descriptor_hash="c" * 32,
            model=self.model,
            role="test",
        )

    def test_large_stdout_stderr_stream_directly_to_files(self):
        bundle = self.root / "bundle-large"
        child = (
            "import os; "
            "chunk=b'x'*(1024*1024); "
            "[(os.write(1,chunk)) for _ in range(16)]; "
            "[(os.write(2,chunk)) for _ in range(3)]"
        )
        # Successful run_managed must not use subprocess.run/capture_output;
        # patching it to explode proves the production success path is Popen
        # with file-backed stdout/stderr.
        with mock.patch(
            "bigcherry.experiment.bundle.subprocess.run",
            side_effect=AssertionError("managed success path must not buffer output"),
        ):
            rc = self._run(bundle, [sys.executable, "-c", child])
        self.assertEqual(rc, 0)
        self.assertEqual((bundle / "stdout.log").stat().st_size, 16 * 1024 * 1024)
        self.assertEqual((bundle / "stderr.log").stat().st_size, 3 * 1024 * 1024)
        result = validate(bundle, required_capabilities={"process_evidence"})
        self.assertTrue(result["promotable"])
        document = json.loads((bundle / "experiment.json").read_text(encoding="ascii"))
        by_role = {item["role"]: item for item in document["artifacts"]}
        self.assertEqual(by_role["stdout"]["bytes"], 16 * 1024 * 1024)
        self.assertEqual(by_role["stderr"]["bytes"], 3 * 1024 * 1024)

    def test_launch_failure_is_terminal_and_durable(self):
        bundle = self.root / "bundle-launch-fail"
        rc = self._run(bundle, [str(self.root / "definitely-does-not-exist")])
        self.assertEqual(rc, 127)
        self.assertTrue((bundle / "stdout.log").is_file())
        self.assertTrue((bundle / "stderr.log").is_file())
        self.assertIn("process launch failed", (bundle / "stderr.log").read_text())
        document = json.loads((bundle / "experiment.json").read_text(encoding="ascii"))
        self.assertEqual(document["state"], "failed")
        self.assertEqual(document["returncode"], 127)
        self.assertFalse(validate(bundle)["promotable"])


if __name__ == "__main__":
    unittest.main()
