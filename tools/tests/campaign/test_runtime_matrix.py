import json
import tempfile
import unittest
from pathlib import Path

from bigcherry.campaign.runtime_matrix import (
    MatrixResolutionError,
    resolve_matrix,
    run_matrix,
)
from bigcherry.core.environment import Device, Host


def host() -> Host:
    return Host(
        name="test", description="", hostname="test", address="", home="",
        repo="", cache_root="", share="", model_root="", bench_harness="",
        rocm="", rocm_shim="", production_port=0, bench_port=0,
        devices=(Device(0, "gfx1100", "x", 1), Device(1, "gfx1100", "x", 1)),
    )


def cell(cell_id="a", *, arm="native", devices=(1,)):
    return {
        "cell_id": cell_id, "model_id": "model-9b", "devices": list(devices),
        "topology": "single", "runtime_profile": "prod", "arm": arm,
        "build_id": "build-1", "binary": "/bin/server",
        **({"cache_id": "cache-1"} if arm == "replay" else {}),
        "workload": {"pp": 512},
    }


class RuntimeMatrixResolutionTests(unittest.TestCase):
    def test_resolves_canonical_visibility_and_stable_identity(self):
        one = resolve_matrix([cell()], host=host(), known_models=frozenset({"model-9b"}))[0]
        two = resolve_matrix([cell()], host=host(), known_models=frozenset({"model-9b"}))[0]
        self.assertEqual(dict(one.visibility), {"ROCR_VISIBLE_DEVICES": "1", "HIP_VISIBLE_DEVICES": "0"})
        self.assertEqual(one.identity_digest, two.identity_digest)

    def test_preflight_rejects_unknown_model_duplicate_and_missing_replay_cache(self):
        with self.assertRaises(MatrixResolutionError):
            resolve_matrix([cell()], host=host(), known_models=frozenset())
        with self.assertRaises(MatrixResolutionError):
            resolve_matrix([cell(), cell()], host=host())
        replay = cell(arm="replay")
        replay.pop("cache_id")
        with self.assertRaises(MatrixResolutionError):
            resolve_matrix([replay], host=host())


class RuntimeMatrixRunTests(unittest.TestCase):
    def test_serial_progress_and_child_verdict_are_preserved(self):
        cells = resolve_matrix([cell("a"), cell("b", devices=(0,))], host=host())
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            seen = []
            result = run_matrix(cells, output=output, execute=lambda item: seen.append(item.cell_id) or {"performance_admitted": False})
            self.assertEqual(result["state"], "completed")
            self.assertEqual(seen, ["a", "b"])
            self.assertFalse(result["results"][0]["child_result"]["performance_admitted"])
            status = json.loads((output / "status.json").read_text())
            events = [json.loads(line) for line in (output / "events.jsonl").read_text().splitlines()]
            self.assertEqual(status["state"], "completed")
            self.assertEqual(events[-1]["state"], "completed")
            self.assertEqual(events[-1]["completed"], 2)

    def test_quiescence_failure_aborts_without_running_cell(self):
        cells = resolve_matrix([cell()], host=host())
        with tempfile.TemporaryDirectory() as temporary:
            called = []
            result = run_matrix(cells, output=temporary, execute=lambda item: called.append(item), quiescent=lambda: False)
            self.assertEqual(result["state"], "failed")
            self.assertEqual(called, [])
            self.assertEqual(json.loads((Path(temporary) / "status.json").read_text())["state"], "failed")

    def test_child_failure_is_terminal_and_not_upgraded(self):
        cells = resolve_matrix([cell("a"), cell("b", devices=(0,))], host=host())
        with tempfile.TemporaryDirectory() as temporary:
            def fail(item):
                if item.cell_id == "b":
                    raise RuntimeError("child rejected")
                return {"performance_admitted": False}
            result = run_matrix(cells, output=temporary, execute=fail)
            self.assertEqual(result["state"], "failed")
            self.assertEqual(result["completed"], 1)
