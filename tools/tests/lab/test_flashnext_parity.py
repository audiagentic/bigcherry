"""Card-free tests of the Flash-Next parity harness (flashnext-parity.sh, flashnext_parity.py) with synthetic raw files."""
from __future__ import annotations

import hashlib
import json
import os
import statistics
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
import flashnext_parity as fp
from bigcherry.__main__ import main as bigcherry_main

WORKLOAD = {"COLD_FILLS": 8, "DOC_FILLS": 6, "LONG_SEEDS": "7 8", "LONG_SEEDS_COUNT": 2, "DECODE_STREAMS": "1 6",
            "SESS8_COVARIATE": fp.SESS8_COVARIATE}
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


def eng(ms, acc=0.7, tps=2.5, mixed=0):
    """Engine deltas of a round of 100 decode steps plus `mixed` mixed steps at `tps` decode tokens per decoding step;
    like stream-bench, the pure-decode tokens per step is reported only without mixed steps."""
    out = {"decode_ms_per_step": ms, "draft_acceptance": acc, "decode_tokens_total": tps * (100 + mixed),
           "engine_steps_decode_total": 100.0, "engine_steps_mixed_total": float(mixed)}
    if not mixed:
        out["decode_tokens_per_step"] = tps
    return out


def write_launch(out: Path, i: int, scale: float, policy=True, committed=5, sess=None):
    """One launch whose times are the reference numbers times `scale` (rates divided by it)."""
    d = out / f"launch-{i}"
    d.mkdir(parents=True)
    w = lambda name, obj: (d / name).write_text(json.dumps(obj))
    w("launch.json", {"policy": policy, "warmup_s": 61.0 if policy else 0.0})
    for k, s in enumerate(range(501, 509)):
        w(f"cold-fill-{s}.json", stream_record([([(2000.0 + 100 * (k % 2)) * scale], 27000, eng(30.0))]))
    for k in range(6):
        w(f"doc-prefill-{k}.json", {"prefill_tps": (4000.0 + 40 * (k % 2)) / scale, "prompt_tokens": 44000})
    w("decode1.json", stream_record([([100.0], 900, eng(20.0 * scale, 0.70)), ([90.0], 900, eng(20.2 * scale, 0.72))]))
    w("decode6.json", stream_record([([100.0] * 6, 900, eng(30.0 * scale, 0.60)),
                                     ([100.0] * 6, 900, eng(30.0 * scale, 0.62))]))
    sess_ms, sess_tps = sess if sess else (40.0 * scale, 2.5)
    w("sess8.json", stream_record([([3000.0 * scale] * 8, 13600, eng(sess_ms, tps=sess_tps)),
                                   ([500.0] * 8, 13600, eng(sess_ms, tps=sess_tps))]))
    for seed, (ms, ttft) in LONG.items():
        w(f"long-{seed}.json", stream_record([([ttft * scale], 256000, eng(ms * scale, 0.5)),
                                              ([1000.0], 256000, eng(ms * 1.01 * scale, 0.5))]))
    w("policy.json", {"found": True, "committed": committed, "bytes_moved": 1 << 20} if policy else {"found": False})


def make_run(out: Path, scales, policy=True, sess=None):
    out.mkdir(parents=True, exist_ok=True)
    (out / "run.json").write_text(json.dumps({"schema": 1, "build": str(out.parent / "build"), "revision": "abc123", "workload": WORKLOAD,
                                              "radiance_pin": {"pinned": "v1.17.0", "pinned_version": "1.17.0"},
                                              "date": "2026-11-07T00:00:00+00:00"}))
    for i, s in enumerate(scales, 1):
        write_launch(out, i, s, policy, sess=sess[i - 1] if sess else None)
    return out


def quiet(fn, *a):
    buf = StringIO()
    with redirect_stdout(buf):
        rc = fn(*a)
    return rc, buf.getvalue()


class ProvenanceTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.dir = Path(self.tmp.name)
        self.out = self.dir / "out"
        self.out.mkdir()

    def write(self, name, content):
        path = self.dir / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        return path

    def test_captures_actual_runtime_bytes_and_build_metadata(self):
        files = {
            "bin/radiance": b"engine",
            "radiance_home/bin/bc-serve": b"owner",
            "radiance_home/helpers/r3_moe_worker": b"worker",
            "radiance_home/kernels/libr3.so": b"kernels",
            "bigcherry-build-metadata-radiance.json": b'{"build_id":"checkpoint","runtime_artifacts":{}}\n',
        }
        for name, content in files.items():
            self.write(f"build/{name}", content)
        fp.capture_build(self.dir / "build", self.out)
        members = json.loads((self.out / "runtime-members.json").read_bytes())
        self.assertEqual(members, {name: hashlib.sha256(content).hexdigest() for name, content in files.items()})
        self.assertEqual((self.out / "bigcherry-build-metadata-radiance.json").read_bytes(),
                         files["bigcherry-build-metadata-radiance.json"])

    def test_run_info_names_measured_release_source(self):
        engine = b"release engine"
        self.write("old/bin/radiance", engine)
        release = self.write("old/source-bundle.json", json.dumps({
            "built_from": {"pinned": "v1.17.0", "pinned_version": "1.17.0"},
            "binary_sha256": hashlib.sha256(engine).hexdigest(),
        }).encode())
        build = release.parent
        args = ["run-info", str(self.out), "--build", str(build), "--launches", "1", "--port", "18450"]
        self.assertEqual(fp.main(args), 0)
        record = json.loads((self.out / "run.json").read_text())
        self.assertEqual(record["radiance_pin"], {
            "pinned": "v1.17.0", "pinned_version": "1.17.0", "source": "source-bundle.json",
        })
        self.write("old/bin/radiance", b"unrelated engine")
        with self.assertRaisesRegex(fp.ParityError, "published source does not identify"):
            fp.measured_radiance_pin(build, hashlib.sha256(b"unrelated engine").hexdigest())

    def test_pin_requires_matching_candidate_source_and_engine(self):
        binary = hashlib.sha256(b"candidate").hexdigest()
        revision = "frozen-catalog"
        pin = "upstream-sha"
        build = self.dir / "candidate"
        self.write("candidate/source-provenance.json", json.dumps({
            "catalog_revision": revision, "upstream": pin,
            "source_identity": {"upstream_revision": pin, "source_slice_id": "source-slice"},
        }).encode())
        self.write("candidate/qualification-build-metadata.json", json.dumps({
            "catalog_revision": revision, "binary_hash": binary, "source_slice_id": "source-slice",
        }).encode())
        self.write("candidate/catalog/engines/radiance/recipes.toml",
                   b'pinned = "upstream-sha"\npinned-version = "2.2.0"\n')
        self.assertEqual(fp.measured_radiance_pin(build, binary), {
            "pinned": pin, "pinned_version": "2.2.0", "source": "source-provenance.json",
        })
        self.assertEqual(fp.measured_radiance_pin(self.dir, binary)["source"], "invoking_catalog")

    def test_archives_failed_session_logs_only_for_the_launch_owner(self):
        manifest = self.write("native/manifest.json", b"{}\n")
        processes = [
            {"role": role, "log": str(self.write(f"native/{role}-1.log", role.encode()))}
            for role in ("engine", "helper0", "helper1")
        ]
        record = json.dumps({"owner_pid": 91, "state": "failed", "processes": processes,
                             "manifest": str(manifest)}).encode()
        session = self.write("native/session.json", record)
        def archive(pid):
            return fp.main(["archive-session", str(session), "--owner-pid", str(pid), "--out", str(self.out)])

        self.assertEqual(archive(92), 0)
        self.assertEqual(list(self.out.iterdir()), [])
        self.assertEqual(archive(91), 0)
        self.assertEqual((self.out / "session.json").read_bytes(), record)
        self.assertEqual((self.out / "manifest.json").read_bytes(), b"{}\n")
        self.assertEqual((self.out / "native-logs/helper1-1.log").read_bytes(), b"helper1")
        self.assertEqual((self.out / "helper0.log").read_bytes(), b"helper0")


class LaunchMetricsTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.dir = Path(self.tmp.name)

    def medians(self, out, launches):
        rows = [fp.launch_metrics(out / f"launch-{i}", WORKLOAD) for i in range(1, launches + 1)]
        return {name: statistics.median(v[name] for v, _, _ in rows if v.get(name) is not None)
                for name in fp.METRICS if any(v.get(name) is not None for v, _, _ in rows)}, rows

    def test_metrics_from_raw_files(self):
        out = make_run(self.dir / "run", [1.0, 1.1, 1.2])
        m, rows = self.medians(out, 3)
        self.assertAlmostEqual(m["cold_ttft_s"], 2.05 * 1.1)                     # mean of the 8 fills
        self.assertAlmostEqual(m["doc_prefill_tps"], 4020.0 / 1.1)
        self.assertAlmostEqual(m["decode1_ms_per_step"], 20.1 * 1.1)
        self.assertAlmostEqual(m["decode6_ms_per_step"], 30.0 * 1.1)
        self.assertAlmostEqual(m["sess8_ttft_s"], 3.0 * 1.1)                     # the cold round only
        self.assertAlmostEqual(m["sess8_ms_per_step"], 40.0 * 1.1)
        self.assertAlmostEqual(m["sess8_tokens_per_step"], 2.5)
        self.assertAlmostEqual(m["long7_prefill_tps"], 256000 / (120.0 * 1.1))    # cold round only
        self.assertAlmostEqual(m["long8_prefill_tps"], 256000 / (100.0 * 1.1))
        self.assertAlmostEqual(m["long7_ms_per_step"], 43.0 * 1.005 * 1.1)        # both reps of seed 7
        self.assertAlmostEqual(m["long8_ms_per_step"], 27.0 * 1.005 * 1.1)
        self.assertNotIn("long_ms_per_step", m)
        self.assertEqual(len(rows[0][1]["long7_ms_per_step"]), 1)                  # one group: seed 7's reps
        self.assertAlmostEqual(m["decode1_acceptance"], 0.71)
        self.assertEqual(m["policy_transactions"], 5)
        self.assertEqual(len(rows[0][2]["cold_ttft_s"]), 8)
        self.assertNotIn("cold_ttft_s", rows[0][1])                                # distinct prompts: no repeats
        self.assertEqual(fp.METRICS["decode1_ms_per_step"], ("lower", True))
        self.assertEqual(fp.METRICS["long8_prefill_tps"], ("higher", True))
        self.assertFalse(fp.METRICS["long7_acceptance"][1])

    def test_missing_engine_block_fails(self):
        out = make_run(self.dir / "run", [1.0, 1.0])
        rec = json.loads((out / "launch-1" / "decode1.json").read_text())
        rec["rounds"][0]["engine"] = None
        (out / "launch-1" / "decode1.json").write_text(json.dumps(rec))
        with self.assertRaises(fp.ParityError):
            fp.launch_metrics(out / "launch-1", WORKLOAD)

    def test_wrong_fill_count_fails(self):
        out = make_run(self.dir / "run", [1.0, 1.0])
        (out / "launch-2" / "cold-fill-508.json").unlink()
        with self.assertRaises(fp.ParityError):
            fp.launch_metrics(out / "launch-2", WORKLOAD)

    def test_mixed_steps_use_decode_tokens_over_decoding_steps(self):
        # 2.3: the cold sess8 round and the 6-stream rounds have mixed steps, so stream-bench gives no pure-decode
        # tokens per step; the sess8 covariate counts decode tokens over decode plus mixed steps, and records the count.
        out = make_run(self.dir / "run", [1.0])
        d = out / "launch-1"
        w = lambda name, obj: (d / name).write_text(json.dumps(obj))
        w("sess8.json", stream_record([([3000.0] * 8, 13600, eng(40.0, tps=2.0, mixed=45)),
                                       ([500.0] * 8, 13600, eng(60.0, tps=2.0, mixed=1))]))
        w("decode6.json", stream_record([([100.0] * 6, 900, eng(30.0, 0.60, mixed=2)),
                                         ([100.0] * 6, 900, eng(30.0, 0.62))]))
        values, _, detail = fp.launch_metrics(d, WORKLOAD)
        self.assertAlmostEqual(values["sess8_tokens_per_step"], 2.0)
        self.assertEqual(values["sess8_mixed_steps"], 45)
        self.assertEqual(detail["decode6_tokens_per_step"], [None, 2.5])

    def test_another_covariate_definition_is_refused(self):
        out = make_run(self.dir / "run", [1.0])
        for workload in ({k: v for k, v in WORKLOAD.items() if k != "SESS8_COVARIATE"},
                         {**WORKLOAD, "SESS8_COVARIATE": "pure-decode"}):
            with self.assertRaisesRegex(fp.ParityError, "SESS8_COVARIATE"):
                fp.launch_metrics(out / "launch-1", workload)

    def test_script_records_the_covariate_definition(self):
        self.assertIn(f"SESS8_COVARIATE={fp.SESS8_COVARIATE}", SCRIPT.read_text())
        self.assertIn("--set SESS8_COVARIATE=$SESS8_COVARIATE", SCRIPT.read_text())


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
        offsets, _warm_start, _warm_len = fp.doc_slices(20000, chars, probe, n)
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

    def run_fixture(self, name, scales, sess=None):
        out = self.root / name
        make_run(out, scales, sess=sess)
        info = json.loads((out / "run.json").read_text())
        (out / "runtime-members.json").write_text(json.dumps({"bin/radiance": "e" * 64}))
        info["build"] = str(self.build)
        info["build_identity"] = fp.build_identity(self.build)
        (out / "run.json").write_text(json.dumps(info))
        for i, _ in enumerate(scales, 1):
            launch = out / f"launch-{i}"
            manifest = {
                "metadata": {"instance_id": f"{name}-launch-{i}"},
                "spec": {
                    "plan": {"engine_memory": {"model_identity": "m" * 64}, "power": {"profile": "stock"}},
                    "intent": {"layout": "flashnext-4gpu"},
                    "handoff": {"placement": {"live_policy_state": {
                        "configured": False, "requested": "off",
                        "reason_code": "no_calibration", "calibration_key": ""
                    }}},
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
            from bigcherry.core import host_facts
            import hashlib
            start = host_facts.snapshot_at_start(
                {"kernel_release": {"value": "fixture", "readable": True}},
                phase="owner_start", captured_at="2026-11-19T00:00:00Z",
            )
            start_path = launch / host_facts.START_RECEIPT_FILE
            start_path.write_text(json.dumps(start))
            (launch / "native-session-start.json").write_text(json.dumps({
                "schema_version": 1, "state": "running",
                "manifest": str(launch / "manifest.json"),
                "started_at": "2026-11-18T23:59:59Z",
                "host_facts_start_sha256": hashlib.sha256(start_path.read_bytes()).hexdigest(),
            }))
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

    def test_parity_launch_uses_shared_record_producer_once(self):
        from bigcherry.core import measurement_producers, run_records

        out = self.run_fixture("sole-writer", [1.0])
        with patch.dict(os.environ, {"BC_WORK_ROOT": str(self.root)}), patch(
            "bigcherry.core.measurement_producers.publish",
            wraps=measurement_producers.publish,
        ) as publish:
            digest = fp.publish_launch_record(out, out / "launch-1", work_root=self.root)
        self.assertEqual(publish.call_count, 1)
        self.assertEqual(publish.call_args.kwargs["identity"]["binary_sha256"], "e" * 64)
        self.assertEqual(publish.call_args.kwargs["power_settings"]["effective"], {"profile": "stock"})
        self.assertEqual(
            run_records.RunRecordStore(self.root).get_record(digest)["run_id"],
            "sole-writer-launch-1",
        )
        self.assertFalse((out / "parity.json").exists())

    def test_two_harness_records_match_results_compare_verdict(self):
        from bigcherry.core import results, run_records

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
            self.assertTrue(report["failed"])
            self.assertEqual(report["metrics"]["decode.decode6_ms_per_step"]["verdict"], "REGRESSION")
            self.assertEqual(report["metrics"]["ttft.cold_ttft_s"]["verdict"], "REGRESSION")
            self.assertEqual(report["metrics"]["decode.decode1_acceptance"]["verdict"], "info")

    def test_record_launch_cli_requires_original_native_start_seal(self):
        from bigcherry.core import host_facts, run_records

        out = self.run_fixture("cli", [1.0])
        launch = out / "launch-1"
        native = self.root / "native-owner"
        native.mkdir()
        (native / host_facts.START_RECEIPT_FILE).write_bytes(
            (launch / host_facts.START_RECEIPT_FILE).read_bytes()
        )
        (native / "session.json").write_bytes(
            (launch / "native-session-start.json").read_bytes()
        )
        with patch.dict(os.environ, {"BC_WORK_ROOT": str(self.root)}), patch(
            "bigcherry.core.host_facts.collect_host_facts",
            side_effect=AssertionError("no Python re-sampling permitted"),
        ):
            self.assertEqual(fp.main([
                "capture-host-facts", str(launch), "--run-dir", str(native),
            ]), 0)
            snapshot = host_facts.read_owner_start(
                launch, session_name="native-session-start.json",
            )
            self.assertEqual(snapshot["capture_phase"], "owner_start")
            self.assertEqual(snapshot["facts"]["kernel_release"]["value"], "fixture")
            # the profile this launch asked for is read from the launch's own arm YAML (absent key: stock)
            (launch / "native-input.yaml").write_text("cards: [0]\nx-bigcherry:\n  schema: 1\n", encoding="utf-8")
            self.assertEqual(fp.requested_power_profile(launch / "native-input.yaml"), "stock")
            (launch / "native-input.yaml").write_text(
                "x-bigcherry:\n  power: {profile: efficient}\n", encoding="utf-8")
            self.assertEqual(fp.requested_power_profile(launch / "native-input.yaml"), "efficient")
            (launch / "native-input.yaml").write_text("cards: [0]\n", encoding="utf-8")
            self.assertEqual(fp.main([
                "capture-provenance", "--heat", str(self.heat), str(launch), "--",
                "bc-serve", "serve", "flashnext-4gpu", "--set", "engine.port=18450",
            ]), 0)
            output = StringIO()
            with redirect_stdout(output):
                self.assertEqual(fp.main([
                    "record-launch", str(out), "--launch", "1",
                    "--work-root", str(self.root),
                ]), 0)
            digest = output.getvalue().strip()
            store = run_records.RunRecordStore(self.root)
            rec = store.get_record(digest)
            self.assertEqual(rec["host_facts"], snapshot)
            roles = {x["role"] for x in rec["provenance"]["inputs"]}
            self.assertIn("native_host_start", roles)
            self.assertIn("native_session_start", roles)
            (launch / host_facts.START_RECEIPT_FILE).write_text("unsealed bytes")
            with self.assertRaises((ValueError, fp.ParityError)):
                fp.publish_launch_record(out, launch)

    def test_missing_required_native_provenance_does_not_publish(self):
        from bigcherry.core import run_records

        out = self.run_fixture("missing", [1.0])
        (out / "launch-1" / "feature-states.v1.json").unlink()
        with patch.dict(os.environ, {"BC_WORK_ROOT": str(self.root)}):
            with self.assertRaises((fp.ParityError, ValueError, OSError)):
                fp.publish_launch_record(out, out / "launch-1")
            self.assertEqual(run_records.RunRecordStore(self.root).query(), [])


    def test_active_policy_snapshots_only_verified_selected_calibration(self):

        from bigcherry.core import run_records

        out = self.run_fixture("active", [1.0])
        launch = out / "launch-1"
        from bigcherry.core import calibration_records

        # MRC04's production snapshot includes the complete narrow key and
        # all quantities. Synthetic partial keys are correctly invalid.
        fixture = ROOT / "tools/tests/fixtures/calibration/production-2026-11-20.toml"
        self.calibration.write_bytes(fixture.read_bytes())
        current = [item for item in calibration_records.read_native_toml(self.calibration)
                   if item["status"] == "current"]
        self.assertEqual(len(current), 1)
        key = current[0]["key"]
        manifest = json.loads((launch / "manifest.json").read_text())
        manifest["spec"]["handoff"]["placement"]["live_policy_state"] = {
            "configured": True, "requested": "on",
            "reason_code": "configured", "calibration_key": "",
        }
        manifest["spec"]["handoff"]["placement"]["live_policy"] = {
            "cadence_ns": 1000000000, "min_samples": 1
        }
        (launch / "manifest.json").write_text(json.dumps(manifest))
        provenance = json.loads((launch / "launch-provenance.json").read_text())
        provenance["calibration_store"] = str(self.calibration)
        (launch / "launch-provenance.json").write_text(json.dumps(provenance))
        (launch / "launch.json").write_text(json.dumps({"policy": True, "warmup_s": 3.0}))
        # Native bc-serve leaves the manifest key empty and reports the selection in the owner session.
        session_path = launch / "native-session-start.json"
        session = json.loads(session_path.read_text())
        session["calibration_record"] = {
            "key_sha256": key, "kind": "radiance.live_policy", "status": "selected"}
        session_path.write_text(json.dumps(session))
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
            self.calibration.write_text(original)

            # A run whose owner reported no selected calibration, or a different one, is refused.
            for change in (lambda c: c.pop("calibration_record"),
                           lambda c: c.update(calibration_record={**c["calibration_record"], "status": "none"}),
                           lambda c: c["calibration_record"].update(key_sha256="e" * 64)):
                bad = json.loads(session_path.read_text())
                change(bad)
                session_path.write_text(json.dumps(bad))
                with self.assertRaises((fp.ParityError, ValueError)):
                    fp.publish_launch_record(out, launch)
                session_path.write_text(json.dumps(session))
            self.assertEqual(len(store.query()), 1)

    def test_active_policy_without_a_calibration_store_publishes_without_one(self):
        from bigcherry.core import run_records

        out = self.run_fixture("nostore", [1.0])
        launch = out / "launch-1"
        manifest = json.loads((launch / "manifest.json").read_text())
        manifest["spec"]["handoff"]["placement"]["live_policy_state"] = {
            "configured": True, "requested": "on",
            "reason_code": "configured", "calibration_key": "",
        }
        manifest["spec"]["handoff"]["placement"]["live_policy"] = {
            "cadence_ns": 1000000000, "min_samples": 1
        }
        (launch / "manifest.json").write_text(json.dumps(manifest))
        (launch / "launch.json").write_text(json.dumps({"policy": True, "warmup_s": 3.0}))
        states = json.loads((launch / "feature-states.v1.json").read_text())
        states[0]["state"] = "active"
        states[0]["reason_code"] = "configured"
        (launch / "feature-states.v1.json").write_text(json.dumps(states))
        provenance = json.loads((launch / "launch-provenance.json").read_text())
        self.assertFalse(provenance.get("calibration_store"))
        # Calibration is provenance, not a gate: with no store the record carries none and still publishes.
        with patch.dict(os.environ, {"BC_WORK_ROOT": str(self.root)}):
            digest = fp.publish_launch_record(out, launch)
            rec = run_records.RunRecordStore(self.root).get_record(digest)
            roles = {v["role"] for v in rec["provenance"]["inputs"]}
            self.assertNotIn("calibration_record", roles)
            self.assertIn("policy_state", roles)

    def test_results_baseline_compare_over_published_launch_records(self):
        from bigcherry.core import results, run_records

        base = self.run_fixture("base-new", [1.0, 1.04])
        now = self.run_fixture("now-new", [1.3, 1.3])
        with patch.dict(os.environ, {"BC_WORK_ROOT": str(self.root)}):
            ids = {}
            for out in (base, now):
                ids[out] = [fp.publish_launch_record(out, out / f"launch-{i}") for i in (1, 2)]
            path = self.root / "portable.json"
            scope = ["--work-root", str(self.root), "--engine", "radiance", "--layout", "flashnext-4gpu",
                     "--workload", "flashnext-parity"]
            stdout = StringIO()
            with redirect_stdout(stdout):
                rc = bigcherry_main(["results", "baseline", *scope, "--name", "mrc05", "--to", str(path),
                                     "--record", ids[base][0], "--record", ids[base][1]])
            self.assertEqual(rc, 0, stdout.getvalue())
            baseline = json.loads(path.read_text())
            results.validate_baseline(baseline)
            self.assertEqual(baseline["kind"], results.BASELINE_KIND)
            self.assertEqual(len(baseline["records"]), 2)
            self.assertGreater(len(baseline["blobs"]), 4)
            stdout = StringIO()
            with redirect_stdout(stdout):
                rc = bigcherry_main(["results", "compare", "--work-root", str(self.root), "--baseline", str(path),
                                     "--record", ids[now][0], "--record", ids[now][1]])
            self.assertEqual(rc, 1, stdout.getvalue())
            self.assertIn("REGRESSION", stdout.getvalue())
            self.assertLess(stdout.getvalue().index("provenance differences"), stdout.getvalue().index("cold_ttft_s"))
            self.assertEqual(len(run_records.RunRecordStore(self.root).query()), 4)

    def test_shipped_parity_baseline_is_a_valid_run_record_baseline(self):
        from bigcherry.core import results

        shipped = ROOT / "docs/reference/engines/radiance/baselines/flashnext-4gpu-parity.json"
        baseline = json.loads(shipped.read_text())
        results.validate_baseline(baseline)
        self.assertEqual(baseline["kind"], results.BASELINE_KIND)
        self.assertGreaterEqual(len({r["record"]["run_id"] for r in baseline["records"]}), 2)
        self.assertIn("decode.decode6_ms_per_step", baseline["metrics"])

    def test_sess8_step_time_is_gated_at_the_baselines_tokens_per_step(self):
        """The 8-session step lengthens with the tokens it accepts a step (more sessions decoding together); that is not
        a regression, a slower step at the same acceptance is."""
        from bigcherry.core import results

        def line(tps, bump=1.0):                       # the build's step time: 30 ms plus 4 ms a token a step
            return ((30.0 + 4.0 * tps) * bump + (0.03 if tps > 3.0 else -0.03), tps)

        base = self.run_fixture("b8", [1.0] * 4, sess=[line(2.5), line(2.7), line(3.0), line(3.2)])
        more = self.run_fixture("m8", [1.0] * 2, sess=[line(3.6), line(3.7)])             # +7% a step, same build
        slower = self.run_fixture("s8", [1.0] * 2, sess=[line(2.8, 1.08), line(2.9, 1.08)])  # same acceptance, +8%
        with patch.dict(os.environ, {"BC_WORK_ROOT": str(self.root)}):
            ids = {out: [fp.publish_launch_record(out, out / f"launch-{i}") for i in range(1, len(list(out.glob("launch-*"))) + 1)]
                   for out in (base, more, slower)}
            path = self.root / "sess8.json"
            scope = ["--work-root", str(self.root), "--engine", "radiance", "--layout", "flashnext-4gpu",
                     "--workload", "flashnext-parity"]
            records = []
            for r in ids[base]:
                records += ["--record", r]
            with redirect_stdout(StringIO()):
                self.assertEqual(bigcherry_main(["results", "baseline", *scope, "--name", "s8", "--to", str(path)] + records), 0)
            baseline = json.loads(path.read_text())
            results.validate_baseline(baseline)
            step = baseline["metrics"]["decode.sess8_ms_per_step"]
            self.assertTrue(step["gated"])
            self.assertEqual(step["covariate_fit"]["metric"], "decode.sess8_tokens_per_step")
            self.assertAlmostEqual(step["covariate_fit"]["slope"], 4.0, delta=0.1)
            outputs = {}
            for name, run in (("more", more), ("slower", slower)):
                buf = StringIO()
                args = ["results", "compare", "--work-root", str(self.root), "--baseline", str(path)]
                for r in ids[run]:
                    args += ["--record", r]
                with redirect_stdout(buf):
                    rc = bigcherry_main(args)
                outputs[name] = (rc, buf.getvalue())
        row = lambda name: next(l for l in outputs[name][1].splitlines() if "sess8_ms_per_step" in l)
        self.assertNotIn("REGRESSION", row("more"))
        self.assertIn("REGRESSION", row("slower"))
        self.assertEqual(outputs["slower"][0], 1)

    def test_results_refuse_a_path_only_baseline_and_a_single_launch(self):
        one = self.run_fixture("single-new", [1.0])
        with patch.dict(os.environ, {"BC_WORK_ROOT": str(self.root)}):
            digest = fp.publish_launch_record(one, one / "launch-1")
            scope = ["--work-root", str(self.root), "--engine", "radiance", "--layout", "flashnext-4gpu",
                     "--workload", "flashnext-parity"]
            with redirect_stdout(StringIO()), patch("sys.stderr", new=StringIO()):
                self.assertEqual(bigcherry_main(["results", "baseline", *scope, "--name", "bad", "--to",
                                                 str(self.root / "bad.json"), "--record", digest]), 2)
                self.assertFalse((self.root / "bad.json").exists())
                legacy = self.root / "legacy.json"
                legacy.write_text(json.dumps({"schema": 1, "kind": "flashnext-parity-baseline",
                                              "source_run": "deleted", "name": "historical"}))
                self.assertEqual(bigcherry_main(["results", "compare", "--work-root", str(self.root),
                                                 "--baseline", str(legacy), "--record", digest]), 2)


class ScriptTests(unittest.TestCase):
    def test_bash_syntax(self):
        subprocess.run(["bash", "-n", str(SCRIPT)], check=True)

    def test_measure_without_build_is_usage(self):
        r = subprocess.run(["bash", str(SCRIPT), "measure", "--out", "/nonexistent-parity-out"],
                           check=False, capture_output=True, text=True)
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("measure --build DIR --out DIR", r.stderr)
        self.assertFalse(Path("/nonexistent-parity-out").exists())

    def test_compare_needs_the_launch_records(self):
        with tempfile.TemporaryDirectory() as out:
            r = subprocess.run(["bash", str(SCRIPT), "compare", "--out", out, "--baseline", "b.json"],
                               check=False, capture_output=True, text=True)
            self.assertEqual(r.returncode, 2)
            self.assertIn("no launch records", r.stderr)

    def test_unknown_command_is_usage(self):
        r = subprocess.run(["bash", str(SCRIPT)], check=False, capture_output=True, text=True)
        self.assertEqual(r.returncode, 2)
        self.assertIn("Usage", r.stderr)


if __name__ == "__main__":
    unittest.main()
