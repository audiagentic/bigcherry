"""Card-free tests of the Flash-Next parity harness (flashnext-parity.sh, flashnext_parity.py) with synthetic raw files."""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from unittest.mock import patch
from bigcherry.core import engines, feature_registry

ROOT = Path(__file__).resolve().parents[3]
SCRIPT = ROOT / "tools" / "lab" / "radiance" / "flashnext-parity.sh"
sys.path.insert(0, str(ROOT / "tools" / "lab" / "radiance"))
import flashnext_parity as fp  # noqa: E402

WORKLOAD = {"COLD_FILLS": 8, "DOC_FILLS": 6, "LONG_SEEDS": "7 8", "LONG_SEEDS_COUNT": 2, "DECODE_STREAMS": "1 6"}
# seed 7 is a different workload from seed 8: slower decode and prefill, by content (identical in every launch)
LONG = {7: (43.0, 120000.0), 8: (27.0, 100000.0)}


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
    for seed, (ms, ttft) in LONG.items():
        w(f"long-{seed}.json", stream_record([([ttft * scale], 256000, eng(ms * scale, 0.5)),
                                              ([1000.0], 256000, eng(ms * 1.01 * scale, 0.5))]))
    w("policy.json", {"found": True, "committed": committed, "bytes_moved": 1 << 20} if policy else {"found": False})


def make_run(out: Path, scales, policy=True):
    out.mkdir(parents=True, exist_ok=True)
    (out / "run.json").write_text(json.dumps({"schema": 1, "build": str(out.parent / "build"), "revision": "abc123", "workload": WORKLOAD,
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
        self.assertAlmostEqual(m["long7_prefill_tps"]["median"], 256000 / (120.0 * 1.1))    # cold round only
        self.assertAlmostEqual(m["long8_prefill_tps"]["median"], 256000 / (100.0 * 1.1))
        self.assertAlmostEqual(m["long7_ms_per_step"]["median"], 43.0 * 1.005 * 1.1)        # both reps of seed 7
        self.assertAlmostEqual(m["long8_ms_per_step"]["median"], 27.0 * 1.005 * 1.1)
        self.assertNotIn("long_ms_per_step", m)
        self.assertIn("note", m["long7_ms_per_step"])
        self.assertEqual(len(m["long7_ms_per_step"]["repeats"][0]), 1)                      # one group: seed 7's reps
        self.assertAlmostEqual(m["decode1_acceptance"]["median"], 0.71)
        self.assertEqual(m["policy_transactions"]["median"], 5)
        self.assertEqual(len(p["launches"][0]["detail"]["cold_ttft_s"]), 8)
        self.assertEqual(m["cold_ttft_s"]["repeats"], [[], [], []])                          # distinct prompts: no repeats
        self.assertTrue(p["policy"])
        self.assertEqual(p["warmup_s"], [61.0, 61.0, 61.0])
        self.assertEqual(m["decode1_ms_per_step"]["direction"], "lower")
        self.assertEqual(m["long8_prefill_tps"]["direction"], "higher")
        self.assertFalse(m["long7_acceptance"]["gated"])
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
        self.assertEqual(dec["spread_repeats"], 0.0)
        self.assertNotIn("tolerance", b["metrics"]["long7_acceptance"])
        self.assertEqual((b["revision"], b["radiance_pin"]["pinned"], b["source_run"]), ("abc123", "v1.17.0", "src"))

    def test_true_repeats_widen_the_tolerance(self):
        b = fp.make_baseline(make_run(self.dir / "same", [1.0, 1.0]), "x", "src")
        dec = b["metrics"]["decode1_ms_per_step"]                 # reps of one prompt: 20.0 and 20.2 ms
        self.assertEqual(dec["spread_across_launches"], 0.0)
        self.assertAlmostEqual(dec["tolerance"], 0.2 / 20.1)
        long8 = b["metrics"]["long8_ms_per_step"]                 # the two reps of seed 8: 1% apart
        self.assertAlmostEqual(long8["tolerance"], 0.01 / 1.005)
        long7 = b["metrics"]["long7_ms_per_step"]                 # bimodal between launches: informational
        self.assertNotIn("tolerance", long7)
        self.assertIn("note", long7)

    def test_distinct_prompt_spread_does_not_widen_the_tolerance(self):
        # identical launches whose distinct prompts differ a lot: cold fills 2000/2100 ms, doc slices 4000/4040 tok/s,
        # seed 7 at 43 ms/step against seed 8 at 27: none of that is noise, so the tolerance is the launches' spread (0)
        b = fp.make_baseline(make_run(self.dir / "same", [1.0, 1.0]), "x", "src")
        for metric in ("cold_ttft_s", "doc_prefill_tps", "long7_prefill_tps", "long8_prefill_tps", "sess8_ttft_s"):
            self.assertEqual(b["metrics"][metric]["tolerance"], 0.0, metric)
            self.assertIsNone(b["metrics"][metric]["spread_repeats"], metric)
        self.assertLess(b["metrics"]["long8_ms_per_step"]["tolerance"], 0.011)   # not the 43-vs-27 seed difference
        p = make_run(self.dir / "apart", [1.0, 1.1])
        cold = fp.make_baseline(p, "x", "src")["metrics"]["cold_ttft_s"]
        self.assertAlmostEqual(cold["tolerance"], 0.1 / 1.05)                    # across launches alone

    def test_one_launch_refused(self):
        p = make_run(self.dir / "run", [1.0])
        with self.assertRaises(fp.ParityError):
            fp.make_baseline(p, "x", "src")
        self.assertEqual(fp.main(["baseline", str(self.dir / "run"), "--name", "x", "--to", str(self.dir / "b.json")]), 2)
        self.assertFalse((self.dir / "b.json").exists())

    def test_cli_writes_file_with_portable_paths(self):
        make_run(self.dir / "run", [1.0, 1.05])
        p = json.loads((self.dir / "run" / "parity.json").read_text())
        p["build"] = str(self.dir / "builds" / "x")
        (self.dir / "run" / "parity.json").write_text(json.dumps(p))
        args = ["baseline", str(self.dir / "run"), "--name", "n", "--to", str(self.dir / "b.json")]
        with patch.dict(os.environ, {"BC_WORK_ROOT": str(self.dir.resolve())}):
            rc, _ = quiet(fp.main, args)
        self.assertEqual(rc, 0)
        b = json.loads((self.dir / "b.json").read_text())
        self.assertEqual((b["name"], b["build"], b["source_run"]), ("n", "${BC_WORK_ROOT}/builds/x", "${BC_WORK_ROOT}/run"))
        with patch.dict(os.environ, {"BC_WORK_ROOT": "/elsewhere"}):
            self.assertEqual(fp.main(args), 2)                    # a local path outside the work root is refused


class CompareTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.dir = Path(self.tmp.name).resolve()
        env = patch.dict(os.environ, {"BC_WORK_ROOT": str(self.dir)})
        env.start()
        self.addCleanup(env.stop)
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
            for name in ("decode1_acceptance", "decode6_acceptance", "long7_acceptance", "long8_acceptance",
                         "policy_transactions"):
                p["metrics"][name]["median"] = (p["metrics"][name]["median"] or 1) * 0.1
        rc, text = self.run_compare([1.02, 1.02], drop)
        self.assertEqual(rc, 0, text)
        self.assertIn("info", text)

    def test_missing_gated_metric_fails(self):
        def drop(p):
            p["metrics"]["long8_ms_per_step"]["median"] = None
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



class RunRecordsTests(unittest.TestCase):
    """Two fixture harness launches, content-pinned production inputs, MRC05 verdict."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name).resolve()
        self.build = self.root / "build"
        self.build.mkdir()
        self.metadata = {
            "build_id": "b" * 32,
            "build_plan_id": "a" * 32,
            "binary_hash": "c" * 64,
            "source_slice_id": "d" * 32,
        }
        (self.build / "bigcherry-build-metadata-radiance.json").write_text(
            json.dumps(self.metadata)
        )
        self.heat = self.root / "heat.hot-first"
        self.heat.write_bytes(b"real heat bytes" + bytes([0, 255]))
        self.calibration = self.root / "calibration.toml"
        self.calibration.write_text(
            'schema_version = 1\n[[live_policy_record]]\nplan_cache_key = "' +
            "e" * 64 + '"\nlayout = "flashnext-4gpu"\nobjective = "balanced"\n'
        )

    def run_fixture(self, name, scales):
        out = self.root / name
        make_run(out, scales)
        info = json.loads((out / "run.json").read_text())
        info["build"] = str(self.build)
        info["build_identity"] = fp.build_identity(self.build)
        (out / "run.json").write_text(json.dumps(info))
        for i, _ in enumerate(scales, 1):
            launch = out / f"launch-{i}"
            manifest = {
                "metadata": {"instance_id": f"{name}-launch-{i}"},
                "spec": {
                    "plan": {"engine_memory": {"model_identity": "m" * 64}},
                    "intent": {"layout": "flashnext-4gpu"},
                    "handoff": {"placement": {"live_policy_state": {
                        "configured": False, "requested": "off",
                        "reason_code": "no_calibration", "calibration_key": ""
                    }}},
                    "power": {"profile": "stock"}
                }
            }
            (launch / "manifest.json").write_text(json.dumps(manifest))
            (launch / "feature-states.v1.json").write_text(json.dumps([{
                "feature_id": "live_policy", "state": "inactive",
                "reason_code": "no_calibration", "reason_detail": "not configured",
                "engine": engines.RADIANCE, "observed_at": "2026-11-19T00:00:00Z"
            }] + [{
                "feature_id": name, "state": "inactive",
                "reason_code": "fixture_not_configured", "reason_detail": "",
                "engine": engines.RADIANCE, "observed_at": "2026-11-19T00:00:00Z",
            } for name in feature_registry.feature_ids(engines.RADIANCE) if name != "live_policy"]))
            (launch / "host-facts.json").write_text(json.dumps({"kernel_release": {
                "value": "fixture", "readable": True
            }}))
            (launch / "launch-provenance.json").write_text(json.dumps({
                "argv": ["bc-serve", "serve", "flashnext-4gpu", "--set", "engine.port=18450"],
                "environment": {"ROCM_PATH": "fixture-rocm"},
                "heat_path": str(self.heat),
                "calibration_store": None,
                "power_profile": "stock",
                "recorded_at": "2026-11-19T00:00:00Z",
            }))
            # Real warmup duration, not only the planner's minimum.
            (launch / "launch.json").write_text(json.dumps({
                "policy": False, "warmup_s": 0.0, "actual_warmup_s": 0.0,
                "warmup_rounds": 0,
            }))
        return out

    def test_two_harness_records_match_results_compare_verdict(self):
        from bigcherry.core import results, run_records
        from bigcherry.__main__ import main as bigcherry_main

        base = self.run_fixture("base", [1.0, 1.04])
        now = self.run_fixture("now", [1.30, 1.30])
        with patch.dict(os.environ, {"BC_WORK_ROOT": str(self.root)}):
            base_ids = [
                fp.publish_launch_record(base, base / f"launch-{i}")
                for i in (1, 2)
            ]
            now_ids = [
                fp.publish_launch_record(now, now / f"launch-{i}")
                for i in (1, 2)
            ]
            store = run_records.RunRecordStore(self.root)
            self.assertEqual(len(set(base_ids + now_ids)), 4)
            for digest in base_ids + now_ids:
                rec = store.get_record(digest)
                self.assertEqual(rec["workload"]["definition"], WORKLOAD)
                self.assertEqual(rec["identity"]["engine"], "radiance")
                self.assertEqual(rec["identity"]["build_id"], self.metadata["build_id"])
                self.assertEqual(rec["identity"]["source_revision"], "abc123")
                self.assertEqual(rec["identity"]["model"], "m" * 64)
                self.assertIn("decode1_ms_per_step", rec["metrics"]["decode"])
                self.assertIn("long8_prefill_tps", rec["metrics"]["prefill"])
                self.assertIn("cold_ttft_s", rec["metrics"]["ttft"])
                self.assertFalse(rec["metric_specs"]["decode"]["long7_acceptance"]["gated"])
                self.assertEqual(rec["feature_states"][0]["feature_id"], "live_policy")
                inputs = {v["role"]: store.get_blob(v["blob"])
                          for v in rec["provenance"]["inputs"]}
                self.assertEqual(inputs["heat"], self.heat.read_bytes())
                self.assertEqual(json.loads(inputs["policy_state"])["configured"], False)
                self.assertIn("warmup", inputs)
                self.assertEqual(store.get_blob(rec["provenance"]["effective_config"]),
                                 (base if digest in base_ids else now).joinpath(
                                     "launch-1" if digest in (base_ids[0], now_ids[0]) else "launch-2",
                                     "manifest.json").read_bytes())
            reference = results.make_baseline(
                store, [{"sha256": x, "record": store.get_record(x)} for x in base_ids], "parity"
            )
            reference_file = self.root / "reference.json"
            reference_file.write_text(json.dumps(reference))
            stdout = StringIO()
            with redirect_stdout(stdout):
                rc = bigcherry_main([
                    "results", "compare", "--work-root", str(self.root),
                    "--baseline", str(reference_file),
                    "--json", "--record", now_ids[0], "--record", now_ids[1],
                ])
            self.assertEqual(rc, 1)
            report = json.loads(stdout.getvalue())
            old_base = fp.make_baseline(fp.summarise(base), "parity", "fixture")
            old_rows, old_failed = fp.compare(fp.summarise(now), old_base)
            self.assertEqual(report["failed"], old_failed)
            mapped = {f"{fp.record_metric_phase(name)}.{name}": verdict
                      for name, _, _, _, _, verdict in old_rows}
            for key, verdict in mapped.items():
                self.assertEqual(report["metrics"][key]["verdict"], verdict, key)
            self.assertTrue(report["failed"])

    def test_record_launch_cli_and_passive_host_snapshot(self):
        from bigcherry.core import run_records

        out = self.run_fixture("cli", [1.0])
        launch = out / "launch-1"
        fake_facts = {"kernel_build": {"value": "fixture-host", "readable": True}}
        with patch.dict(os.environ, {"BC_WORK_ROOT": str(self.root)}), patch(
            "bigcherry.core.host_facts.collect_host_facts", return_value=fake_facts
        ):
            self.assertEqual(fp.main(["capture-host-facts", str(launch)]), 0)
            self.assertEqual(json.loads((launch / "host-facts.json").read_text()), fake_facts)
            self.assertEqual(fp.main([
                "capture-provenance", "--heat", str(self.heat),
                "--power-profile", "stock", str(launch), "--",
                "bc-serve", "serve", "flashnext-4gpu", "--set", "engine.port=18450",
            ]), 0)
            captured = json.loads((launch / "launch-provenance.json").read_text())
            self.assertEqual(captured["argv"][-2:], ["--set", "engine.port=18450"])
            output = StringIO()
            with redirect_stdout(output):
                self.assertEqual(fp.main([
                    "record-launch", str(out), "--launch", "1",
                    "--work-root", str(self.root),
                ]), 0)
            digest = output.getvalue().strip()
            self.assertEqual(len(digest), 64)
            store = run_records.RunRecordStore(self.root)
            rec = store.get_record(digest)
            self.assertEqual(rec["host_facts"], fake_facts)
            self.assertEqual(
                json.loads(store.get_blob(rec["provenance"]["argv"]))[-2:],
                ["--set", "engine.port=18450"],
            )

    def test_missing_required_native_provenance_does_not_publish(self):
        from bigcherry.core import run_records

        out = self.run_fixture("missing", [1.0])
        (out / "launch-1" / "feature-states.v1.json").unlink()
        with patch.dict(os.environ, {"BC_WORK_ROOT": str(self.root)}):
            with self.assertRaises((fp.ParityError, ValueError, OSError)):
                fp.publish_launch_record(out, out / "launch-1")
            self.assertEqual(run_records.RunRecordStore(self.root).query(), [])


    def test_active_policy_snapshots_only_verified_selected_calibration(self):
        import hashlib

        from bigcherry.core import run_records

        out = self.run_fixture("active", [1.0])
        launch = out / "launch-1"
        from bigcherry.core import calibration_records

        # MRC04's production snapshot includes the complete narrow key and
        # all quantities. Synthetic partial keys are correctly invalid.
        fixture = ROOT / "tools/tests/fixtures/calibration/production-2026-11-19.toml"
        self.calibration.write_bytes(fixture.read_bytes())
        current = [item for item in calibration_records.read_native_toml(self.calibration)
                   if item["status"] == "current"]
        self.assertEqual(len(current), 1)
        key = current[0]["key"]
        manifest = json.loads((launch / "manifest.json").read_text())
        manifest["spec"]["handoff"]["placement"]["live_policy_state"] = {
            "configured": True, "requested": "on",
            "reason_code": "configured", "calibration_key": key,
        }
        manifest["spec"]["handoff"]["placement"]["live_policy"] = {
            "cadence_ns": 1000000000, "min_samples": 1
        }
        (launch / "manifest.json").write_text(json.dumps(manifest))
        provenance = json.loads((launch / "launch-provenance.json").read_text())
        provenance["calibration_store"] = str(self.calibration)
        (launch / "launch-provenance.json").write_text(json.dumps(provenance))
        (launch / "launch.json").write_text(json.dumps({"policy": True, "warmup_s": 3.0}))
        states = json.loads((launch / "feature-states.v1.json").read_text())
        states[0]["state"] = "active"
        states[0]["reason_code"] = "configured"
        (launch / "feature-states.v1.json").write_text(json.dumps(states))

        with patch.dict(os.environ, {"BC_WORK_ROOT": str(self.root)}):
            digest = fp.publish_launch_record(out, launch)
            store = run_records.RunRecordStore(self.root)
            rec = store.get_record(digest)
            self.assertEqual(rec["feature_states"][0]["state"], "active")
            inputs = {v["role"]: store.get_blob(v["blob"])
                      for v in rec["provenance"]["inputs"]}
            self.assertEqual(json.loads(inputs["calibration_record"])["plan_cache_key"], key)
            self.assertEqual(json.loads(inputs["policy_state"])["calibration_key"], key)
            self.assertEqual(json.loads(inputs["warmup"])["warmup_s"], 3.0)

            # Valid data under the wrong key must never become live evidence.
            # A changed scoped quantity is not silently accepted.
            original = self.calibration.read_text()
            self.calibration.write_text(original.replace(
                'plan_cache_key = "' + key + '"', 'plan_cache_key = "' + "e" * 64 + '"'
            ))
            with self.assertRaises((fp.ParityError, ValueError)):
                fp.publish_launch_record(out, launch)
            self.assertEqual(len(store.query()), 1)


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
