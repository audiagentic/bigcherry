"""Card-free tests: corpus/prompt identity, metadata parsing, append and comparisons."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from bigcherry.bench import workloads as w


class WorkloadsTest(unittest.TestCase):
    def test_catalog_and_canonical_ids(self):
        entries = w.catalog()
        self.assertEqual(set(entries), {
            "prefill-doc-44k", "cold-27k-ttft", "decode-1stream",
            "decode-6stream", "pwr-phases",
        })
        self.assertTrue(all(v["version"] == 1 for v in entries.values()))

    def test_doc_slices_retain_original_offset_and_prompt(self):
        blob = "some-very-long-doc-body-" * 20000
        rows = list(w.doc_slices(blob, 44000, 1, 3))
        self.assertEqual(len(rows), 3)
        for k, prompt in rows:
            offset = (k * 7919 * 44000) % max(1, len(blob) - int(44000 * 3.9))
            self.assertEqual(prompt, f"[doc 44000 {k}] " + blob[offset:offset + int(44000 * 3.9)])

    def test_doc_record_exact_usage_median_and_corpus_hash(self):
        with tempfile.TemporaryDirectory() as tmp:
            docs = Path(tmp) / "docs"
            docs.mkdir()
            (docs / "one.md").write_text("paragraph " * 10000)
            cfg = {"target_tokens": 44000, "reps": 3, "max_tokens": 1, "chars_per_token": 3.9}
            with patch.object(w, "request", side_effect=[
                (10, {"usage": {"prompt_tokens": 46000, "completion_tokens": 1}}),
                (20, {"usage": {"prompt_tokens": 44000, "completion_tokens": 1}}),
                (10, {"usage": {"prompt_tokens": 49000, "completion_tokens": 1}}),
            ]):
                metrics, detail = w._run_doc(cfg, "http://example", Path(tmp))
            self.assertEqual(metrics["prefill_tps"]["median"], 4600)
            self.assertEqual([r["slice"] for r in metrics["prefill_tps"]["per_rep"]], [1, 2, 3])
            self.assertEqual(len(detail["corpus_sha256"]), 64)

    def test_readonly_power_state_per_bdf(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            drm = root / "class/drm"
            drm.mkdir(parents=True)
            dev = root / "devices/pci0000:00/0000:03:00.0"
            (dev / "hwmon/hwmon3").mkdir(parents=True)
            (drm / "card0").mkdir()
            (drm / "card0/device").symlink_to(dev, target_is_directory=True)
            (dev / "hwmon/hwmon3/power1_cap").write_text("272000000\n")
            (dev / "power_dpm_force_performance_level").write_text("auto\n")
            (dev / "pp_od_clk_voltage").write_text(
                "OD_SCLK:\n0: 500MHz\n1: 3040MHz\nOD_VDDGFX_OFFSET:\n-125mV\n"
            )
            self.assertEqual(w.power_state(root)["0000:03:00.0"], {
                "power1_cap_uw": 272000000, "vddgfx_offset_mv": -125,
                "od_sclk_max_mhz": 3040, "perf_level": "auto",
            })

    def test_exact_manifest_options_and_unknown_without_proof(self):
        self.assertEqual(w.manifest_info(None)["policy"]["state"], "unknown")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "radiance_home").mkdir()
            (root / "bigcherry-build-metadata-radiance.json").write_text(
                json.dumps({"source_commit": "abc", "build_id": "b123"})
            )
            m = {
                "metadata": {"content_digest": "sha256:001"},
                "spec": {
                    "processes": {"engine": {
                        "runtime": {"radiance_home": str(root / "radiance_home")},
                        "execution": {"num_speculative_tokens": 3, "kv_cache_dtype": "fp8"},
                        "scheduler": {"max_num_batched_tokens": 2048},
                    }},
                    "resources": {"host": {"pool_mib": 9216, "kv_mib": 12288}},
                    "handoff": {"placement": {"live_policy": {}}},
                    "intent": {"layout": "flashnext-4gpu", "model": "model"},
                },
            }
            path = root / "manifest.json"
            path.write_text(json.dumps(m))
            row = w.manifest_info(path)
            self.assertEqual(row["build_commit"], "abc")
            self.assertEqual(row["policy"]["state"], "on")
            self.assertEqual(row["layout"], "flashnext-4gpu")
            self.assertEqual(row["options"]["spec_depth"], 3)
            self.assertEqual(row["options"]["host_pool_mib"], 9216)

    def test_index_append_one_line_and_comparison_flags(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "results" / "index.jsonl"
            cfg = {"version": 1}
            base = {"build_path": "/x", "layout": "a", "policy": {"state": "off"},
                        "options": {"spec_depth": 3}, "kernel": "k", "model": "m", "power_state": {"gpu0": "stock"}}
            a = w.make_row("prefill-doc-44k", cfg, {
                "prefill_tps": {"median": 3312, "per_rep": [3312], "unit": "tok/s"},
            }, {"completed_reps": 1}, base, run_folder="/tmp/1", kernel="k", power=base["power_state"])
            b = w.make_row("prefill-doc-44k", cfg, {
                "prefill_tps": {"median": 3708, "per_rep": [3708], "unit": "tok/s"},
            }, {"completed_reps": 1}, {**base, "layout": "b", "policy": {"state": "on"}},
                run_folder="/tmp/2", kernel="k", power=base["power_state"])
            other = w.make_row("decode-1stream", cfg, {}, {}, base,
                               run_folder="/tmp/3", kernel="k", power={})
            for row in (a, b, other):
                w.append_index(path, row)
            rows = w.read_index(path)
            self.assertEqual(len(rows), 3)
            lines = path.read_text().splitlines()
            self.assertEqual(len(lines), 3)
            self.assertTrue(all(json.loads(line) for line in lines))
            table = w.compare(rows, "prefill-doc-44k", 1)
            self.assertIn("3312", table)
            self.assertIn("3708", table)
            self.assertIn("WARNING", table)
            self.assertIn("layout", table)
            self.assertIn("policy", table)
            self.assertNotIn("decode-1stream", table)
            self.assertIn("1 rows", w.compare(rows, "prefill-doc-44k", 1, layout="a"))
            self.assertEqual(w.compare(rows, "prefill-doc-44k", 2).splitlines()[0],
                             "prefill-doc-44k@2: 0 rows")

    def test_fail_closed_on_corrupt_index(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "index.jsonl"
            path.write_text('{"schema":"bigcherry.workload-result.v1"}\npartial')
            with self.assertRaisesRegex(w.WorkloadError, "invalid JSONL"):
                w.read_index(path)

    def test_http_missing_usage_does_not_invent_tokens(self):
        class FakeResponse:
            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return None

            def read(self):
                return b'{"usage":{},"choices":[{"text":"x"}]}'

        with patch.object(w.urllib.request, "urlopen", return_value=FakeResponse()), self.assertRaisesRegex(w.WorkloadError, "exact usage"):
            w.request("http://host", "hello", 1)


if __name__ == "__main__":
    unittest.main()
