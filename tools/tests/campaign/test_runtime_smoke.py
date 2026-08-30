"""RE14 runtime-smoke: argv construction and llama-bench JSON result validation."""

from __future__ import annotations

import json
import unittest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from bigcherry.campaign.smoke import (RuntimeSmokeSpec, SmokeError,  # noqa: E402
                                     evaluate_smoke_result, smoke_argv)

# Trimmed but structurally real llama-bench `-o json` output, matching the
# shape actually observed from this session's Brutus runs (RCCL validation).
_REAL_SHAPE_OUTPUT = json.dumps([
    {"build_commit": "4801e3c", "n_prompt": 512, "n_gen": 0, "avg_ts": 18987.616529,
     "samples_ts": [17268.9, 19850.4, 19843.6]},
    {"build_commit": "4801e3c", "n_prompt": 0, "n_gen": 128, "avg_ts": 232.280224,
     "samples_ts": [229.7, 233.6, 233.5]},
])


class SmokeArgvTests(unittest.TestCase):
    def test_default_spec_builds_expected_argv(self):
        spec = RuntimeSmokeSpec(model_path=Path("/models/m.gguf"))
        args = smoke_argv(Path("/bin/llama-bench"), spec)
        self.assertIn("-p", args)
        self.assertIn("512", args)
        self.assertIn("-n", args)
        self.assertIn("128", args)
        self.assertIn("-sm", args)
        self.assertIn("none", args)
        self.assertNotIn("-ts", args)

    def test_tensor_split_included_when_set(self):
        spec = RuntimeSmokeSpec(
            model_path=Path("/models/m.gguf"), split_mode="tensor",
            tensor_split=(1.0, 1.0))
        args = smoke_argv(Path("/bin/llama-bench"), spec)
        self.assertIn("-ts", args)
        self.assertIn("1.0/1.0", args)


class EvaluateSmokeResultTests(unittest.TestCase):
    def test_real_shaped_output_passes(self):
        rows = evaluate_smoke_result(_REAL_SHAPE_OUTPUT)
        self.assertEqual(len(rows), 2)

    def test_invalid_json_rejected(self):
        with self.assertRaises(SmokeError):
            evaluate_smoke_result("not json")

    def test_hip_diagnostic_preamble_on_stdout_is_stripped(self):
        # Found live on the first real Windows local-GPU smoke test: HIP's
        # own runtime prints "HIP Library Path: ..." to stdout ahead of
        # llama-bench's -o json output on Windows, an upstream quirk this
        # project has no patch for.
        preamble = "HIP Library Path: C:\\WINDOWS\\SYSTEM32\\amdhip64_7.dll\n"
        rows = evaluate_smoke_result(preamble + _REAL_SHAPE_OUTPUT)
        self.assertEqual(len(rows), 2)

    def test_still_fails_closed_with_no_bracket_at_all(self):
        with self.assertRaises(SmokeError):
            evaluate_smoke_result("HIP Library Path: nothing else here")

    def test_empty_list_rejected(self):
        with self.assertRaises(SmokeError):
            evaluate_smoke_result("[]")

    def test_wrong_row_count_rejected(self):
        with self.assertRaises(SmokeError):
            evaluate_smoke_result(_REAL_SHAPE_OUTPUT, expected_rows=3)

    def test_zero_throughput_rejected(self):
        rows = json.loads(_REAL_SHAPE_OUTPUT)
        rows[0]["avg_ts"] = 0
        with self.assertRaises(SmokeError):
            evaluate_smoke_result(json.dumps(rows))

    def test_missing_throughput_rejected(self):
        rows = json.loads(_REAL_SHAPE_OUTPUT)
        del rows[1]["avg_ts"]
        with self.assertRaises(SmokeError):
            evaluate_smoke_result(json.dumps(rows))


if __name__ == "__main__":
    unittest.main()
