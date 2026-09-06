"""Server cells retain failures and require observed clean teardown."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from bigcherry.campaign.benchmark import run_server_arm_capture
from bigcherry.tuning.server_runner import ShutdownResult


class ServerCaptureTests(unittest.TestCase):
    def test_clean_and_forced_shutdown(self):
        for forced in (False, True):
            with self.subTest(forced=forced), tempfile.TemporaryDirectory() as tmp:
                class Server:
                    def __init__(self, **kwargs):
                        self.host, self.port = "127.0.0.1", 4567
                        self.last_shutdown = None

                    def __enter__(self):
                        return self

                    def __exit__(self, *args):
                        self.last_shutdown = ShutdownResult("http", True, forced, -9 if forced else 0)

                with patch("bigcherry.tuning.server_runner.ServerRunner", Server), patch(
                    "bigcherry.campaign.bench_runner.run_bench_runner_server_bench",
                    return_value={"tg512_tps": 30.0},
                ) as bench:
                    result = run_server_arm_capture(
                        binary=Path("llama-server"), model=Path("model.gguf"), extra_args=(),
                        output=Path(tmp), pair=0, side="native", position=1, env={},
                        bench_configs="tg512", runner_root=Path("runner"),
                        required_metrics=("tg512_tps",),
                    )
                self.assertEqual(result["returncode"], int(forced))
                self.assertEqual("metrics" in result, not forced)
                self.assertFalse(result["performance_admitted"])
                self.assertEqual(result["position"], 1)
                self.assertEqual(bench.call_args.kwargs["server_url"], "http://127.0.0.1:4567")
                persisted = json.loads((Path(tmp) / "pair-001-native/cell.json").read_text())
                self.assertEqual(persisted["shutdown"]["forced"], forced)

    def test_rejects_missing_expected_metrics_before_launch(self):
        with patch("bigcherry.tuning.server_runner.ServerRunner") as server:
            with self.assertRaisesRegex(ValueError, "expected metrics"):
                run_server_arm_capture(
                    binary=Path("server"), model=Path("model"), extra_args=(),
                    output=Path("unused"), pair=0, side="native", position=0,
                    env={}, bench_configs="full", runner_root=Path("runner"), required_metrics=(),
                )
        server.assert_not_called()
