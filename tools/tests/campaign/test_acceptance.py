"""MEN17: post-bump hardware acceptance against stored baselines, and stored KLD references. Offline, fake captures."""
import argparse
import contextlib
import io
import json
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path
from unittest import mock

from bigcherry.campaign import acceptance as acc
from bigcherry.campaign import quality_capture as qc
from bigcherry.cli.accept import cmd_accept
from bigcherry.core import engines, paths
from bigcherry.engine_adapters.base import QualityCommand
from bigcherry.engine_adapters.radiance import RadianceAdapter


def base(arm="a", metric="pp_tps", value=100.0, direction="higher-better", tolerance=None):
    return {"arm": arm, "metric": metric, "value": value, "direction": direction,
            "tolerance": tolerance or {"relative": 0.05}}


def baseline_file(rows, config="cfg/run.json"):
    return {"schema": 1, "config": config, "pin": "1.0.0", "run": "r1", "date": "2026-10-17", "baselines": rows}


def parse(rows):
    return acc.parse_baselines(baseline_file(rows), "b.json")


class ToleranceTests(unittest.TestCase):
    def test_relative_higher_better_limit_and_boundary(self):
        b = parse([base(value=2000, tolerance={"relative": 0.03})]).baselines[0]
        self.assertAlmostEqual(b.limit, 1940.0)
        self.assertTrue(b.passes(1940.0))
        self.assertTrue(b.passes(2500))            # doing better never fails
        self.assertFalse(b.passes(1939.9))

    def test_absolute_lower_better_limit_and_boundary(self):
        b = parse([base(metric="kld_mean", value=0.0642, direction="lower-better", tolerance={"absolute": 0.01})]).baselines[0]
        self.assertAlmostEqual(b.limit, 0.0742)
        self.assertTrue(b.passes(0.0742))
        self.assertTrue(b.passes(0.0))
        self.assertFalse(b.passes(0.0743))

    def test_absolute_higher_better_top1(self):
        b = parse([base(metric="top1_percent", value=88.7, tolerance={"absolute": 1.0})]).baselines[0]
        self.assertTrue(b.passes(87.7))
        self.assertFalse(b.passes(87.6))

    def test_rejects_bad_files(self):
        good = base()
        for bad in (
            {**baseline_file([good]), "schema": 2},
            {**baseline_file([good]), "extra": 1},
            {**baseline_file([good]), "pin": ""},
            baseline_file([]),
            baseline_file([{**good, "direction": "up"}]),
            baseline_file([{**good, "tolerance": {"relative": 0.1, "absolute": 1}}]),
            baseline_file([{**good, "tolerance": {"percent": 1}}]),
            baseline_file([{**good, "tolerance": {"absolute": -1}}]),
            baseline_file([{**good, "value": True}]),
            baseline_file([{**good, "value": "100"}]),
            baseline_file([good, good]),
        ):
            with self.subTest(bad=bad), self.assertRaises(acc.AcceptanceError):
                acc.parse_baselines(bad, "b.json")


def run_json(speed=None, quality=None, failed=()):
    """A run.json as the server comparison writes it: per-cell metrics (and repetitions) plus the quality section."""
    runs = []
    for pair, (arm, value) in enumerate(speed or []):
        runs.append({"pair": pair + 1, "mode": arm, "returncode": 0, "metrics": {"pp_tps": value}})
    for arm, value in failed:
        runs.append({"pair": 9, "mode": arm, "returncode": 1, "metrics": {"pp_tps": value}})
    document = {"runs": runs}
    if quality is not None:
        document["quality"] = {"arms": {arm: {"result": result} for arm, result in quality.items()}}
    return document


KLD = {"kld_mean": 0.064, "kld_median": 0.03, "kld_p99": 0.68, "kld_p999": 2.0, "top1_percent": 88.7,
       "ppl": 9.0, "ppl_reference": 9.1}


class MeasuredValueTests(unittest.TestCase):
    def test_speed_is_the_mean_of_the_arms_clean_cells(self):
        run = run_json(speed=[("a", 100.0), ("b", 50.0), ("a", 110.0)], failed=[("a", 1.0)])
        self.assertEqual(acc.measured_value(run, "a", "pp_tps"), 105.0)

    def test_speed_uses_repetitions_when_present(self):
        run = {"runs": [{"mode": "a", "returncode": 0, "metrics": {}, "repetitions": {"pp_tps": [10.0, 20.0]}}]}
        self.assertEqual(acc.measured_value(run, "a", "pp_tps"), 15.0)

    def test_quality_fields_and_ratio(self):
        run = run_json(quality={"on": KLD})
        self.assertEqual(acc.measured_value(run, "on", "kld_p99"), 0.68)
        self.assertAlmostEqual(acc.measured_value(run, "on", "ppl_ratio"), 9.0 / 9.1)

    def test_unmeasured_is_none(self):
        run = run_json(speed=[("a", 1.0)], quality={"on": KLD})
        self.assertIsNone(acc.measured_value(run, "zzz", "pp_tps"))
        self.assertIsNone(acc.measured_value(run, "a", "tg_tps"))
        self.assertIsNone(acc.measured_value(run, "zzz", "kld_mean"))
        self.assertIsNone(acc.measured_value({"runs": []}, "a", "pp_tps"))


class CompareTests(unittest.TestCase):
    def rows(self, baselines, run):
        return acc.compare("e", parse(baselines), run)

    def test_pass(self):
        rows = self.rows([base(value=100)], run_json(speed=[("a", 99.0), ("a", 101.0)]))
        self.assertTrue(rows[0].passed)
        self.assertEqual(rows[0].delta, 0.0)

    def test_higher_better_regression(self):
        rows = self.rows([base(value=100, tolerance={"relative": 0.03})], run_json(speed=[("a", 90.0), ("a", 90.0)]))
        self.assertFalse(rows[0].passed)
        self.assertAlmostEqual(rows[0].delta, -10.0)

    def test_lower_better_regression_and_improvement(self):
        baselines = [base(metric="kld_mean", value=0.0642, direction="lower-better", tolerance={"absolute": 0.01})]
        self.assertFalse(self.rows(baselines, run_json(quality={"a": {**KLD, "kld_mean": 0.09}}))[0].passed)
        self.assertTrue(self.rows(baselines, run_json(quality={"a": {**KLD, "kld_mean": 0.01}}))[0].passed)

    def test_missing_measurement_fails(self):
        rows = self.rows([base(arm="gone")], run_json(speed=[("a", 100.0), ("a", 100.0)]))
        self.assertFalse(rows[0].passed)
        self.assertIsNone(rows[0].measured)
        self.assertIn("not measured", rows[0].detail)


TOML = textwrap.dedent("""\
    schema = 1
    name = "fake"
    upstream = "https://example.org/fake"
    [layout]
    patch-id-prefix = "fk"
    patches = "engines/fake/patches"
    overlay = "engines/fake/overlay"
    vendor = "vendor/fake"
    recipes = "engines/fake/recipes.toml"
    [build]
    targets-option = "T"
    binary = "bin/fake"
    [release]
    version-source = "pinned-version"
    [serve]
    binary = "bin/fake"
    model-flag = "--model"
    host-flag = "--host"
    port-flag = "--port"
    health = "/health"
    shutdown = "sigint"
    """)
ENTRY = textwrap.dedent("""\
    [[acceptance]]
    name = "speed"
    config = "cfg/run.json"
    baselines = ["docs/b.json"]
    rounds = 2
    settle-seconds = 0
    env = { ACC_TEST_REF = "{entry-dir}/ref" }
    """)


class FakeRepo:
    """A temporary repository root holding a fake engine, its acceptance config and baselines."""

    def __init__(self, entry=ENTRY, rows=None, config_engine="fake", write_baselines=True):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        (self.root / "engines/fake").mkdir(parents=True)
        (self.root / "engines/fake/engine.toml").write_text(TOML + entry, encoding="utf-8")
        (self.root / "cfg").mkdir()
        (self.root / "cfg/run.json").write_text(json.dumps({"engine": config_engine}), encoding="utf-8")
        (self.root / "docs").mkdir()
        if write_baselines:
            document = baseline_file(rows or [base(value=100, tolerance={"relative": 0.05})])
            (self.root / "docs/b.json").write_text(json.dumps(document), encoding="utf-8")

    def layout(self):
        return engines.load(self.root, "fake")

    def close(self):
        self.tmp.cleanup()


def fake_capture(speed, returncode=0, seen=None):
    """A stand-in for run_server_comparison_capture that writes the run.json the real one would."""
    def capture(config, output, *, rounds, seed, settle_seconds):
        if seen is not None:
            seen.append({"config": config, "rounds": rounds, "settle": settle_seconds,
                         "ref": __import__("os").environ.get("ACC_TEST_REF")})
        output.mkdir(parents=True)
        (output / "run.json").write_text(json.dumps(run_json(speed=speed)), encoding="utf-8")
        return returncode
    return capture


class AcceptTests(unittest.TestCase):
    def setUp(self):
        self.repo = FakeRepo()
        self.addCleanup(self.repo.close)
        self.out = self.repo.root / "out"

    def accept(self, capture, only=None):
        with mock.patch("bigcherry.campaign.benchmark.run_server_comparison_capture", capture):
            return acc.accept(self.repo.root, self.repo.layout(), only, self.out)

    def test_pass_writes_evidence_and_sets_the_entry_environment(self):
        seen = []
        passed, results, evidence = self.accept(fake_capture([("a", 99.0), ("a", 101.0)], seen=seen))
        self.assertTrue(passed)
        document = json.loads(evidence.read_text(encoding="utf-8"))
        self.assertTrue(document["passed"])
        self.assertEqual(document["entries"][0]["rows"][0]["measured"], 100.0)
        self.assertEqual(document["entries"][0]["baselines"][0]["pin"], "1.0.0")
        self.assertEqual((seen[0]["rounds"], seen[0]["settle"]), (2, 0.0))
        self.assertEqual(seen[0]["ref"], f"{self.out / 'speed'}/ref")
        self.assertNotIn("ACC_TEST_REF", __import__("os").environ)

    def test_regression_fails(self):
        passed, results, _ = self.accept(fake_capture([("a", 80.0), ("a", 80.0)]))
        self.assertFalse(passed)
        self.assertIn("FAIL", acc.render(results))

    def test_nonzero_run_fails_even_when_numbers_hold(self):
        passed, results, _ = self.accept(fake_capture([("a", 100.0), ("a", 100.0)], returncode=1))
        self.assertFalse(passed)
        self.assertIn("exited 1", results[0].error)

    def test_capture_that_raises_is_a_failed_entry_not_a_crash(self):
        def boom(*args, **kwargs):
            raise ValueError("no GPUs")
        passed, results, evidence = self.accept(boom)
        self.assertFalse(passed)
        self.assertIn("no GPUs", results[0].error)
        self.assertTrue(evidence.is_file())

    def test_table_shows_baseline_measured_and_delta(self):
        _, results, _ = self.accept(fake_capture([("a", 99.0), ("a", 101.0)]))
        text = acc.render(results)
        for word in ("metric", "baseline", "measured", "delta", "PASS", "pp_tps"):
            self.assertIn(word, text)

    def test_only_selects_and_unknown_is_refused(self):
        passed, results, _ = self.accept(fake_capture([("a", 100.0), ("a", 100.0)]), only=["speed"])
        self.assertTrue(passed)
        with self.assertRaisesRegex(acc.AcceptanceError, "no acceptance entry nope"):
            acc.prepare(self.repo.root, self.repo.layout(), ["nope"])


class DeclarationTests(unittest.TestCase):
    def prepare_error(self, **kwargs):
        repo = FakeRepo(**kwargs)
        self.addCleanup(repo.close)
        with self.assertRaises(acc.AcceptanceError) as caught:
            acc.prepare(repo.root, repo.layout(), None)
        return str(caught.exception)

    def test_missing_baseline_file(self):
        self.assertIn("does not exist", self.prepare_error(write_baselines=False))

    def test_baseline_for_another_config(self):
        repo = FakeRepo()
        self.addCleanup(repo.close)
        (repo.root / "docs/b.json").write_text(json.dumps(baseline_file([base()], config="cfg/other.json")), encoding="utf-8")
        with self.assertRaisesRegex(acc.AcceptanceError, "is for config cfg/other.json"):
            acc.prepare(repo.root, repo.layout(), None)

    def test_config_of_another_engine(self):
        self.assertIn("not 'fake'", self.prepare_error(config_engine="other"))

    def test_engine_without_entries(self):
        self.assertIn("declares no [[acceptance]]", self.prepare_error(entry=""))

    def test_bad_toml_entries(self):
        for bad in (
            'name = "x"\nconfig = "c"\nbaselines = ["b"]',                                  # not an array of tables
            '[[acceptance]]\nname = "x"\nconfig = "c"',                                      # no baselines
            '[[acceptance]]\nname = "x"\nconfig = "/abs"\nbaselines = ["b"]',               # absolute path
            '[[acceptance]]\nname = "x y"\nconfig = "c"\nbaselines = ["b"]',                # bad name
            '[[acceptance]]\nname = "x"\nconfig = "c"\nbaselines = ["b"]\nrounds = 0',      # rounds
            '[[acceptance]]\nname = "x"\nconfig = "c"\nbaselines = ["b"]\nwhat = 1',        # unknown key
            '[[acceptance]]\nname = "x"\nconfig = "c"\nbaselines = ["b"]\n'
            '[[acceptance]]\nname = "x"\nconfig = "c"\nbaselines = ["b"]',                  # duplicate name
        ):
            with self.subTest(bad=bad):
                repo = FakeRepo(entry=bad)
                self.addCleanup(repo.close)
                with self.assertRaises(engines.EngineError):
                    repo.layout()

    def test_declared_engines_resolve_against_the_repository(self):
        """Every engine's [[acceptance]] entries point at real configs and baseline files that parse."""
        for name, layout in engines.load_all(paths.REPO_ROOT).items():
            for entry in layout.acceptance:
                with self.subTest(engine=name, entry=entry.name):
                    prepared = acc.prepare(paths.REPO_ROOT, layout, [entry.name])
                    self.assertTrue(prepared[0][2])
        radiance = engines.load(paths.REPO_ROOT, "radiance")
        self.assertEqual({e.name for e in radiance.acceptance}, {"helper-ab", "helper-kld"})


class CliTests(unittest.TestCase):
    def setUp(self):
        self.repo = FakeRepo()
        self.addCleanup(self.repo.close)
        self.out = self.repo.root / "out"

    def run_cli(self, capture, only=None, engine="fake"):
        args = argparse.Namespace(engine=engine, only=only, output=str(self.out))
        stdout, stderr = io.StringIO(), io.StringIO()
        with mock.patch.object(paths, "REPO_ROOT", self.repo.root), \
                mock.patch("bigcherry.campaign.benchmark.run_server_comparison_capture", capture), \
                contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            code = cmd_accept(args)
        return code, stdout.getvalue(), stderr.getvalue()

    def test_exit_codes(self):
        code, out, _ = self.run_cli(fake_capture([("a", 100.0), ("a", 100.0)]))
        self.assertEqual(code, 0)
        self.assertIn("accept: PASS", out)
        self.out = self.repo.root / "out2"
        code, out, _ = self.run_cli(fake_capture([("a", 50.0), ("a", 50.0)]))
        self.assertEqual(code, 1)
        self.assertIn("accept: FAIL", out)

    def test_bad_declaration_exits_two_before_running_anything(self):
        calls = []
        code, _, err = self.run_cli(fake_capture([], seen=calls), only=["nope"])
        self.assertEqual(code, 2)
        self.assertIn("no acceptance entry nope", err)
        self.assertEqual(calls, [])

    def test_undeclared_engine_exits_two(self):
        code, _, err = self.run_cli(fake_capture([]), engine="absent")
        self.assertEqual(code, 2)
        self.assertIn("not declared", err)

    def test_parser_registers_accept(self):
        from bigcherry.cli.main import build_parser
        args = build_parser().parse_args(["--engine", "radiance", "accept", "--only", "helper-ab", "--output", "o"])
        self.assertEqual((args.only, args.output), (["helper-ab"], "o"))
        args = build_parser().parse_args(["--engine", "radiance", "pin-bump", "x", "--accept"])
        self.assertTrue(args.accept)
        self.assertFalse(build_parser().parse_args(["--engine", "radiance", "pin-bump", "x"]).accept)


class PinBumpAcceptPhaseTests(unittest.TestCase):
    """pin-bump --accept: the phase STOPs without GPUs or on a regression and records a pass in the release record."""

    def phase(self, *, gpu=True, passed=True):
        from bigcherry.core import gpu as gpu_module
        from bigcherry.release import pin_bump
        row = acc.Row("speed", "a", "pp_tps", 100.0, 80.0, 95.0, "higher-better", False)
        result = acc.EntryResult("speed", "cfg/run.json", "out", 0, "", [row], [])
        record = mock.Mock(notes="")
        patches = [
            mock.patch.object(gpu_module, "free_vram_bytes",
                              **({"return_value": {0: 1}} if gpu else {"side_effect": gpu_module.GpuQueryError("none")})),
            mock.patch.object(acc, "accept", return_value=(passed, [result], Path("out/acceptance.json"))),
            mock.patch.object(pin_bump.releases, "record_for_checkout", return_value=record),
        ]
        with contextlib.ExitStack() as stack:
            for patch in patches:
                stack.enter_context(patch)
            try:
                pin_bump.run_phase_accept(engine="radiance", out_dir=Path("out"), target_ref="x", vendor_root=Path("v"))
                stop = None
            except pin_bump.PinBumpStop as exc:
                stop = exc
        return stop, record

    def test_no_gpu_stops(self):
        stop, _ = self.phase(gpu=False)
        self.assertEqual(stop.code, "ACCEPT_NO_GPU")

    def test_regression_stops_with_the_failing_rows(self):
        stop, record = self.phase(passed=False)
        self.assertEqual(stop.code, "ACCEPT_FAILED")
        self.assertIn("pp_tps", stop.evidence["failed"][0])
        record.save.assert_not_called()

    def test_pass_is_recorded_in_the_release_record(self):
        stop, record = self.phase()
        self.assertIsNone(stop)
        self.assertIn("hardware acceptance: PASSED", record.notes)
        record.save.assert_called_once()


# --- stored KLD references ---------------------------------------------------------------------------------------

_PROGRAM = (
    "import json,sys;out=sys.argv[1];mean=float(sys.argv[2]);top1=float(sys.argv[3]);"
    "json.dump({'all':{'kld':{'mean':mean,'median':mean/2,'p99':mean*8,'p99_9':mean*30},'top1_agreement':top1,"
    "'ppl':{'candidate':8.55,'reference':8.56,'ratio':0.9988}},'by_source':{}},open(out,'w'))"
)


class FakeAdapter(RadianceAdapter):
    """The control reproduces its own reference exactly (KL 0, top-1 100%): the case a control gate misreads."""

    def quality_command(self, kind, arm, env, run_dir, binary, model, quality):
        mean, top1 = {"control": ("0", "1.0"), "native": ("0.004", "0.966")}[arm["name"]]
        result = run_dir / f"{arm['name']}.kld.json"
        return QualityCommand((sys.executable, "-c", _PROGRAM, str(result), mean, top1), dict(env),
                              run_dir / f"{arm['name']}.score.log", result, 60.0)


class StoredReferenceTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.ref = self.root / "ref"
        self.ref.mkdir()
        (self.ref / "logits.bin").write_bytes(b"x")
        self.meta = self.root / "ref.meta.json"
        self.meta.write_text(json.dumps({"pin": "1.7.0", "layout": "tp2 exact wire", "recorded_on": "2026-10-17"}),
                             encoding="utf-8")

    def block(self, **updates):
        block = {"reference_dir": str(self.ref), "reference_metadata": str(self.meta), "gate": "baselines",
                 "args": ["--tp", "2"]}
        block.update(updates)
        return {k: v for k, v in block.items() if v is not None}

    def test_valid_stored_block_has_no_thresholds_and_loads_metadata(self):
        block = qc.validate_quality_block(self.block(), ["control", "native"])
        qc.check_reference_state(block)
        self.assertEqual(qc.load_reference_metadata(block)["pin"], "1.7.0")

    def test_gate_rules(self):
        names = ["control", "native"]
        thresholds = {f: 1 for f in ("max_kld_mean_increase", "max_kld_median_increase", "max_kld_p99_increase",
                                     "max_kld_p999_increase", "max_top1_drop_points", "max_ppl_ratio_increase")}
        for bad in (self.block(thresholds=thresholds), self.block(gate="other"),
                    self.block(gate="control"),                                    # control gate needs thresholds
                    self.block(corpus="c", record={"args": [], "environment": {}})):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                qc.validate_quality_block(bad, names)
        self.assertEqual(qc.validate_quality_block(self.block(gate="control", thresholds=thresholds), names)["gate"],
                         "control")

    def test_metadata_must_name_pin_layout_and_date(self):
        block = qc.validate_quality_block(self.block(), ["control", "native"])
        self.meta.write_text(json.dumps({"pin": "1.7.0"}), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "layout, recorded_on"):
            qc.check_reference_state(block)
        self.meta.unlink()
        with self.assertRaisesRegex(ValueError, "unreadable"):
            qc.check_reference_state(block)

    def test_a_missing_stored_reference_is_refused(self):
        (self.ref / "logits.bin").unlink()
        block = qc.validate_quality_block(self.block(), ["control", "native"])
        with self.assertRaisesRegex(ValueError, "missing or empty"):
            qc.check_reference_state(block)

    def test_control_reproducing_its_reference_is_not_a_false_regression(self):
        names = ["control", "native"]
        block = qc.validate_quality_block(self.block(), names)
        prepared = {n: {"binary": Path("b/bin/radiance"), "model": Path("m"), "env": {}, "settings": {}} for n in names}
        section, _ = qc.run_quality_capture(adapter=FakeAdapter(), quality=block, prepared=prepared, names=names,
                                            output=self.root / "out", reference_env={})
        self.assertTrue(section["passed"], section)
        self.assertEqual(section["gate"], "baselines")
        self.assertEqual(section["reference_metadata"]["layout"], "tp2 exact wire")
        self.assertNotIn("gate", section["arms"]["native"])
        self.assertEqual(section["arms"]["control"]["result"]["top1_percent"], 100.0)

    def test_accept_gates_the_arms_against_the_stored_baselines(self):
        names = ["control", "native"]
        block = qc.validate_quality_block(self.block(), names)
        prepared = {n: {"binary": Path("b/bin/radiance"), "model": Path("m"), "env": {}, "settings": {}} for n in names}
        section, _ = qc.run_quality_capture(adapter=FakeAdapter(), quality=block, prepared=prepared, names=names,
                                            output=self.root / "out", reference_env={})
        run = {"runs": [], "quality": section}
        rows = acc.compare("q", parse([
            base(arm="native", metric="kld_mean", value=0.0040, direction="lower-better", tolerance={"absolute": 0.002}),
            base(arm="native", metric="top1_percent", value=96.6, tolerance={"absolute": 1.0}),
        ]), run)
        self.assertTrue(all(r.passed for r in rows), rows)
        worse = parse([base(arm="native", metric="top1_percent", value=99.0, tolerance={"absolute": 1.0})])
        self.assertFalse(acc.compare("q", worse, run)[0].passed)


if __name__ == "__main__":
    unittest.main()
