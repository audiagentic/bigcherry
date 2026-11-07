"""Card-free tests of the Flash-Next parity harness (flashnext-parity.sh, flashnext_parity.py) with synthetic raw files."""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
SCRIPT = ROOT / "tools" / "lab" / "radiance" / "flashnext-parity.sh"
sys.path.insert(0, str(ROOT / "tools" / "lab" / "radiance"))
import flashnext_parity as fp  # noqa: E402

WORKLOAD = {"COLD_FILLS": 8, "DOC_FILLS": 6, "LONG_SEEDS_COUNT": 2, "DECODE_STREAMS": "1 6"}


def stream_record(rounds):
    """A stream-bench record; each round is (ttft_ms list, prompt_tokens, engine dict or None)."""
    out = []
    for rep, (ttfts, prompt, engine) in enumerate(rounds):
        per = [{"stream": s, "ok": True, "ttft_ms": t, "prompt_tokens": prompt} for s, t in enumerate(ttfts)]
        out.append({"streams": len(ttfts), "ok": True, "rep": rep, "per_stream": per, "engine": engine,
                    "aggregate": {"ttft_ms_p50": fp.statistics.median(ttfts), "ttft_ms_max": max(ttfts)}})
    return {"schema": "bigcherry.stream-benchmark.v1", "ok": True, "rounds": out}


def eng(ms, acc=0.7, tps=2.5):
    return {"decode_ms_per_step": ms, "draft_acceptance": acc, "decode_tokens_per_step": tps}


def write_launch(out: Path, i: int, scale: float, policy=True, committed=5):
    """One launch whose times are the reference numbers times `scale` (rates divided by it)."""
    d = out / f"launch-{i}"
    d.mkdir(parents=True)
    w = lambda name, obj: (d / name).write_text(json.dumps(obj))  # noqa: E731
    w("launch.json", {"policy": policy, "warmup_s": 61.0 if policy else 0.0})
    for k, s in enumerate(range(501, 509)):
        w(f"cold-fill-{s}.json", stream_record([([(2000.0 + 100 * (k % 2)) * scale], 27000, eng(30.0))]))
    for k in range(6):
        w(f"doc-prefill-{k}.json", {"prefill_tps": (4000.0 + 40 * (k % 2)) / scale, "prompt_tokens": 44000})
    w("decode1.json", stream_record([([100.0], 900, eng(20.0 * scale, 0.70)), ([90.0], 900, eng(20.2 * scale, 0.72))]))
    w("decode6.json", stream_record([([100.0] * 6, 900, eng(30.0 * scale, 0.60)),
                                     ([100.0] * 6, 900, eng(30.0 * scale, 0.62))]))
    w("sess8.json", stream_record([([3000.0 * scale] * 8, 13600, eng(40.0 * scale)),
                                   ([500.0] * 8, 13600, eng(40.0 * scale))]))
    for seed in (7, 8):
        w(f"long-{seed}.json", stream_record([([100000.0 * scale], 256000, eng(25.0 * scale, 0.5)),
                                              ([1000.0], 256000, eng(25.0 * scale, 0.5))]))
    w("policy.json", {"found": True, "committed": committed, "bytes_moved": 1 << 20} if policy else {"found": False})


def make_run(out: Path, scales, policy=True):
    out.mkdir(parents=True, exist_ok=True)
    (out / "run.json").write_text(json.dumps({"schema": 1, "build": "/b", "revision": "abc123", "workload": WORKLOAD,
                                              "radiance_pin": {"pinned": "v1.17.0", "pinned_version": "1.17.0"},
                                              "date": "2026-11-07T00:00:00+00:00"}))
    for i, s in enumerate(scales, 1):
        write_launch(out, i, s, policy)
    parity = fp.summarise(out)
    (out / "parity.json").write_text(json.dumps(parity))
    return parity


def quiet(fn, *a):
    buf = StringIO()
    with redirect_stdout(buf):
        rc = fn(*a)
    return rc, buf.getvalue()


class SummariseTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.dir = Path(self.tmp.name)

    def test_medians_from_raw_files(self):
        p = make_run(self.dir / "run", [1.0, 1.1, 1.2])
        m = p["metrics"]
        self.assertAlmostEqual(m["cold_ttft_s"]["median"], 2.05 * 1.1)                     # mean of the 8 fills
        self.assertAlmostEqual(m["doc_prefill_tps"]["median"], 4020.0 / 1.1)
        self.assertAlmostEqual(m["decode1_ms_per_step"]["median"], 20.1 * 1.1)
        self.assertAlmostEqual(m["decode6_ms_per_step"]["median"], 30.0 * 1.1)
        self.assertAlmostEqual(m["sess8_ttft_s"]["median"], 3.0 * 1.1)                     # the cold round only
        self.assertAlmostEqual(m["sess8_ms_per_step"]["median"], 40.0 * 1.1)
        self.assertAlmostEqual(m["long_prefill_tps"]["median"], 256000 / (100.0 * 1.1))     # cold rounds only
        self.assertAlmostEqual(m["long_ms_per_step"]["median"], 25.0 * 1.1)
        self.assertAlmostEqual(m["decode1_acceptance"]["median"], 0.71)
        self.assertEqual(m["policy_transactions"]["median"], 5)
        self.assertEqual(len(m["cold_ttft_s"]["samples"][0]), 8)
        self.assertTrue(p["policy"])
        self.assertEqual(p["warmup_s"], [61.0, 61.0, 61.0])
        self.assertEqual(m["decode1_ms_per_step"]["direction"], "lower")
        self.assertEqual(m["long_prefill_tps"]["direction"], "higher")
        self.assertFalse(m["long_acceptance"]["gated"])
        self.assertEqual(p["revision"], "abc123")
        self.assertEqual(p["radiance_pin"]["pinned"], "v1.17.0")

    def test_missing_engine_block_fails(self):
        out = self.dir / "run"
        make_run(out, [1.0, 1.0])
        rec = json.loads((out / "launch-1" / "decode1.json").read_text())
        rec["rounds"][0]["engine"] = None
        (out / "launch-1" / "decode1.json").write_text(json.dumps(rec))
        with self.assertRaises(fp.ParityError):
            fp.summarise(out)

    def test_wrong_fill_count_fails(self):
        out = self.dir / "run"
        make_run(out, [1.0, 1.0])
        (out / "launch-2" / "cold-fill-508.json").unlink()
        with self.assertRaises(fp.ParityError):
            fp.summarise(out)


class BaselineTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.dir = Path(self.tmp.name)

    def test_tolerance_is_the_larger_spread(self):
        p = make_run(self.dir / "run", [1.0, 1.1])
        b = fp.make_baseline(p, "v1.17.0", "src")
        dec = b["metrics"]["decode6_ms_per_step"]                 # both rounds equal: the launches set the tolerance
        self.assertAlmostEqual(dec["tolerance"], (33.0 - 30.0) / 31.5)
        self.assertEqual(dec["spread_within_launch"], 0.0)
        cold = b["metrics"]["cold_ttft_s"]                        # fills spread 2000..2100 = 4.9%, launches 9.5%
        self.assertAlmostEqual(cold["tolerance"], max(cold["spread_across_launches"], cold["spread_within_launch"]))
        same = fp.make_baseline(make_run(self.dir / "same", [1.0, 1.0]), "x", "src")
        self.assertAlmostEqual(same["metrics"]["cold_ttft_s"]["tolerance"], 100.0 / 2050.0)   # within-launch spread
        self.assertEqual(same["metrics"]["cold_ttft_s"]["spread_across_launches"], 0.0)
        self.assertNotIn("tolerance", b["metrics"]["long_acceptance"])
        self.assertEqual((b["revision"], b["radiance_pin"]["pinned"], b["source_run"]), ("abc123", "v1.17.0", "src"))

    def test_one_launch_refused(self):
        p = make_run(self.dir / "run", [1.0])
        with self.assertRaises(fp.ParityError):
            fp.make_baseline(p, "x", "src")
        self.assertEqual(fp.main(["baseline", str(self.dir / "run"), "--name", "x", "--to", str(self.dir / "b.json")]), 2)
        self.assertFalse((self.dir / "b.json").exists())

    def test_cli_writes_file(self):
        make_run(self.dir / "run", [1.0, 1.05])
        rc, _ = quiet(fp.main, ["baseline", str(self.dir / "run"), "--name", "n", "--to", str(self.dir / "b.json")])
        self.assertEqual(rc, 0)
        self.assertEqual(json.loads((self.dir / "b.json").read_text())["name"], "n")


class CompareTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.dir = Path(self.tmp.name)
        make_run(self.dir / "base", [1.0, 1.04])
        self.baseline = self.dir / "baseline.json"
        quiet(fp.main, ["baseline", str(self.dir / "base"), "--name", "b", "--to", str(self.baseline)])

    def run_compare(self, scales, mutate=None):
        out = self.dir / f"now-{len(list(self.dir.iterdir()))}"
        make_run(out, scales)
        if mutate:
            p = json.loads((out / "parity.json").read_text())
            mutate(p)
            (out / "parity.json").write_text(json.dumps(p))
        return quiet(fp.main, ["compare", str(out), "--baseline", str(self.baseline)])

    def test_within_allowance_passes(self):
        rc, text = self.run_compare([1.01, 1.03])
        self.assertEqual(rc, 0, text)
        self.assertIn("PASS", text)

    def test_regression_fails(self):
        rc, text = self.run_compare([1.3, 1.3])
        self.assertEqual(rc, 1)
        self.assertIn("REGRESSION", text)

    def test_improvement_passes(self):
        rc, text = self.run_compare([0.6, 0.6])
        self.assertEqual(rc, 0, text)
        self.assertIn("better", text)
        self.assertNotIn("REGRESSION", text)

    def test_acceptance_never_gated(self):
        def drop(p):
            for name in ("decode1_acceptance", "decode6_acceptance", "long_acceptance", "policy_transactions"):
                p["metrics"][name]["median"] = (p["metrics"][name]["median"] or 1) * 0.1
        rc, text = self.run_compare([1.02, 1.02], drop)
        self.assertEqual(rc, 0, text)
        self.assertIn("info", text)

    def test_missing_gated_metric_fails(self):
        def drop(p):
            p["metrics"]["long_ms_per_step"]["median"] = None
        rc, text = self.run_compare([1.02, 1.02], drop)
        self.assertEqual(rc, 1)
        self.assertIn("MISSING", text)

    def test_workload_mismatch_refused(self):
        def change(p):
            p["workload"] = dict(p["workload"], COLD_FILLS=4)
        rc, _ = self.run_compare([1.02, 1.02], change)
        self.assertEqual(rc, 2)


class HelperTests(unittest.TestCase):
    def test_warmup_plan(self):
        m = {"spec": {"handoff": {"placement": {"live_policy": {"cadence_ns": 2_000_000_000, "min_samples": 3}}}}}
        self.assertEqual(fp.warmup_plan(m)["warmup_s"], (3 + 1) * 2.0 + 1)
        self.assertTrue(fp.warmup_plan(m)["policy"])
        self.assertEqual(fp.warmup_plan({"spec": {"handoff": {"placement": {}}}}), {"policy": False, "warmup_s": 0.0})

    def test_policy_line_takes_the_last(self):
        log = ("x\npolicy placement helper=0 evaluated=3 no_change=1 proposed=2 committed=1 aborted=0 bytes_moved=10 a=1\n"
               "policy placement helper=0 evaluated=9 no_change=1 proposed=4 committed=3 aborted=0 bytes_moved=99 a=1\n")
        got = fp.policy_line(log)
        self.assertEqual((got["committed"], got["bytes_moved"]), (3, 99))
        self.assertEqual(fp.policy_line("nothing"), {"found": False})

    def test_doc_slots_are_disjoint(self):
        chars, probe, n = 1000, 300, 6
        offsets, warm_start, warm_len = fp.doc_slices(20000, chars, probe, n)
        self.assertEqual(offsets, [0, 1000, 2000, 3000, 4000, 5000])
        for k in range(50):
            off = fp.doc_offset(f"warmup:{k}", chars, 20000, probe, n)
            self.assertGreaterEqual(off, n * chars)
            self.assertLessEqual(off + chars, 20000 - probe)
        with self.assertRaises(fp.ParityError):
            fp.doc_slices(7000, chars, probe, n)

    def test_radiance_pin_read_from_recipes(self):
        pin = fp.radiance_pin()
        self.assertTrue(pin["pinned"])


class ScriptTests(unittest.TestCase):
    def test_bash_syntax(self):
        subprocess.run(["bash", "-n", str(SCRIPT)], check=True)

    def test_measure_without_build_is_usage(self):
        r = subprocess.run(["bash", str(SCRIPT), "measure", "--out", "/nonexistent-parity-out"],
                           capture_output=True, text=True)
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("measure --build DIR --out DIR", r.stderr)
        self.assertFalse(Path("/nonexistent-parity-out").exists())

    def test_unknown_command_is_usage(self):
        r = subprocess.run(["bash", str(SCRIPT)], capture_output=True, text=True)
        self.assertEqual(r.returncode, 2)
        self.assertIn("Usage", r.stderr)


if __name__ == "__main__":
    unittest.main()
