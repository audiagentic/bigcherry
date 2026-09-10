import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from bigcherry.cli.main import main


class RuntimeMatrixCliTests(unittest.TestCase):
    def test_dry_run_loads_canonical_environment_and_registry(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = root / "matrix.json"
            output = root / "out"
            config.write_text(json.dumps({
                "host": "build-server",
                "cells": [{
                    "cell_id": "gpu0",
                    "model_id": "tierB-qwen9b-q6k",
                    "devices": [0],
                    "topology": "single",
                    "runtime_profile": "production-safe-single",
                    "arm": "native",
                    "build_id": "native-build",
                    "binary": "/bin/llama-server",
                    "workload": {"delegate_argv": ["never-run"]},
                }],
            }), encoding="utf-8")
            with patch.dict(os.environ, {"PYTHONPATH": "tools"}, clear=False):
                self.assertEqual(main(["runtime-matrix", "--config", str(config), "--output", str(output), "--dry-run"]), 0)
            resolved = json.loads((output / "resolved-matrix.json").read_text(encoding="utf-8"))
            self.assertEqual(resolved["host"], "build-server")
            self.assertEqual(resolved["cells"][0]["visibility"], {
                "HIP_VISIBLE_DEVICES": "0", "ROCR_VISIBLE_DEVICES": "0"
            })

    def test_invalid_model_fails_before_output_matrix(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = root / "matrix.json"
            output = root / "out"
            config.write_text(json.dumps({
                "cells": [{
                    "cell_id": "bad", "model_id": "not-registered", "devices": [0],
                    "topology": "single", "runtime_profile": "prod", "arm": "native",
                    "build_id": "b", "binary": "/bin/x", "workload": {"delegate_argv": ["x"]},
                }]
            }), encoding="utf-8")
            self.assertEqual(main(["runtime-matrix", "--config", str(config), "--output", str(output), "--dry-run"]), 2)
            self.assertFalse((output / "resolved-matrix.json").exists())

