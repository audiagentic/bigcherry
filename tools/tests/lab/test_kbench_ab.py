"""Card-free KAB01 tests with the rad-kbench 1.16 Speed/Correctness Markdown format."""
from __future__ import annotations

import argparse
from contextlib import redirect_stdout
from io import StringIO
import json
import math
import subprocess
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "tools" / "lab" / "radiance"))
import kbench_ab  # noqa: E402


def report(rows, speed_override=None):
    """Each row: (op, kernel, geometry, device_us, verdict, rel_l2, tolerance, operands)."""
    lines = ["# radiance kernel libraries", "",
             "- fixture: `fixture.rkb`, 2 case(s) recorded test from `model.rad`",
             "", "## Correctness", "",
             "| op | kernel | library | domain | geometry | tuned | n | verdict | rel_l2 | tol |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for op, kernel, geom, us, verdict, rel, tol, operands in rows:
        lines.append(f"| {chr(96)}{op}{chr(96)} | {chr(96)}{kernel}{chr(96)} | libr3 | device "
                     f"| {chr(96)}{geom}{chr(96)} | - | 1 | {verdict} | {rel} | {tol} |")
    lines += ["", "## Speed", "",
              "| op | kernel | library | geometry | device us | host us | min | p90 | n | ±% | MHz | operands | notes |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for op, kernel, geom, us, verdict, rel, tol, operands in rows:
        lines.append(f"| {chr(96)}{op}{chr(96)} | {chr(96)}{kernel}{chr(96)} | libr3 "
                     f"| {chr(96)}{geom}{chr(96)} | {us} | - | {us} | {us} | 6 | 1.0 | 1500 | {operands} | - |")
    return "\n".join(lines) + "\n" if speed_override is None else speed_override


def case(n=4, time=10.0, op="gemm_nt", verdict="ok", kernel="gemm_bf16", size=4,
         rel="1e-4", tol="1e-3", operands="16 KiB"):
    return (op, kernel, f"n={n} dtype=bf16 M={size}", str(time), verdict, rel, tol, operands)


class KbenchABTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.dir = Path(self.tmp.name)

    def parse(self, rows, name="a.md"):
        file = self.dir / name
        file.write_text(report(rows), encoding="utf-8")
        return kbench_ab.parse_report(file)

    def test_identical_sides_ratio_one(self):
        a = self.parse([case(), case(n=8, time=25)], "a.md")
        b = self.parse([case(), case(n=8, time=25)], "b.md")
        result = kbench_ab.compare(a, b)
        self.assertEqual([r["ratio_b_over_a"] for r in result["cases"]], [1., 1.])
        self.assertEqual(result["summaries"]["gemm_nt"]["ratio_b_over_a"], 1.)
        self.assertFalse(result["only_a"])
        self.assertFalse(result["only_b"])
        self.assertEqual(result["cases"][0]["a_margin"], 0.0009)

    def test_one_sided_rows_and_shape_step_never_mixed(self):
        a = self.parse([case(), case(n=8)], "a.md")
        b = self.parse([case(), case(n=4, size=8)], "b.md")
        result = kbench_ab.compare(a, b)
        self.assertEqual(len(result["cases"]), 1)
        self.assertEqual(len(result["only_a"]), 1)
        self.assertEqual(len(result["only_b"]), 1)
        self.assertEqual(result["only_b"][0]["step"], 8)
        self.assertEqual(result["only_a"][0]["geometry"]["n"], "8")

    def test_divergence_can_never_win_or_contribute_to_summary(self):
        a = self.parse([case()], "a.md")
        b = self.parse([case(time=1, verdict="**FAIL**", rel="0.1")], "b.md")
        result = kbench_ab.compare(a, b)
        self.assertEqual(result["cases"][0]["status"], "invalid")
        self.assertEqual(result["cases"][0]["b_margin"], -0.099)
        self.assertIsNone(result["summaries"]["gemm_nt"]["ratio_b_over_a"])
        self.assertEqual(result["summaries"]["gemm_nt"]["invalid"], 1)

    def test_noise_check_uses_measured_spread_not_constant(self):
        a = self.parse([case(time=10)], "a.md")
        b = self.parse([case(time=9.5)], "b.md")
        again = self.parse([case(time=11)], "again.md")
        result = kbench_ab.compare(a, b, again)
        self.assertEqual(result["cases"][0]["status"], "indistinguishable")
        self.assertAlmostEqual(result["cases"][0]["noise_log_spread"], math.log(1.1))
        self.assertEqual(kbench_ab.compare(a, b)["cases"][0]["status"], "B faster")
        missing_noise = self.parse([case(n=8)], "other.md")
        self.assertEqual(kbench_ab.compare(a, b, missing_noise)["cases"][0]["status"], "noise-unavailable")

    def test_timing_only_is_unverified_not_correct(self):
        a = self.parse([case(verdict="timed", rel="-")], "a.md")
        b = self.parse([case(time=5, verdict="timed", rel="-")], "b.md")
        result = kbench_ab.compare(a, b)
        self.assertEqual(result["cases"][0]["status"], "unverified")
        self.assertIsNone(result["cases"][0]["a_margin"])
        self.assertEqual(result["summaries"]["gemm_nt"]["unverified"], 1)

    def test_malformed_empty_or_ambiguous_reports_rejected(self):
        path = self.dir / "bad.md"
        for bad in ("", "## Speed\n", report([case()]).replace("| 10.0 | - |", "| garbage | - |"),
                    report([case(), case()])):
            with self.subTest(bad=bad[:35]):
                path.write_text(bad, encoding="utf-8")
                with self.assertRaises(ValueError):
                    kbench_ab.parse_report(path)
        with self.assertRaises(ValueError):
            self.parse([case(time=0)], "zero.md")

    def test_graph_time_weighted_arithmetic(self):
        a = self.parse([case(n=4,time=10), case(n=8,time=20)], "a.md")
        b = self.parse([case(n=4,time=5), case(n=8,time=30)], "b.md")
        graph = self.dir / "graph.txt"
        graph.write_text("gemm_nt n=4 dtype=bf16 M=4\n"
                         "  weights blk.0.up\n"
                         "gemm_nt n=4 dtype=bf16 M=4\n"
                         "  weights blk.1.up\n"
                         "gemm_nt n=4 dtype=bf16 M=4\n"
                         "  weights blk.2.up\n"
                         "gemm_nt n=8 dtype=bf16 M=4\n"
                         "  weights blk.3.up\nselection table\n", encoding="utf-8")
        calls, missing = kbench_ab.graph_calls(graph, a, 4, 0, 3)
        self.assertFalse(missing)
        weighted = kbench_ab.compare(a, b, calls=calls)
        self.assertAlmostEqual(weighted["summaries"]["gemm_nt"]["ratio_b_over_a"], 45/50)
        self.assertEqual(weighted["summaries"]["gemm_nt"]["calls"], 4)
        equal = kbench_ab.compare(a, b)
        self.assertAlmostEqual(equal["summaries"]["gemm_nt"]["ratio_b_over_a"], 35/30)

    def test_cli_compare_outputs_markdown_and_json(self):
        a = self.dir / "a.md"
        b = self.dir / "b.md"
        a.write_text(report([case()]), encoding="utf-8")
        b.write_text(report([case(time=9)]), encoding="utf-8")
        out = self.dir / "joined"
        self.assertEqual(kbench_ab.main(["compare", str(a), str(b), "--out", str(out)]), 0)
        self.assertIn("Op-family summary", out.with_suffix(".md").read_text())
        self.assertEqual(json.loads(out.with_suffix(".json").read_text())["cases"][0]["ratio_b_over_a"], 0.9)


    def test_cli_rejects_reports_from_different_fixtures(self):
        a, b = self.dir / "a.md", self.dir / "b.md"
        a.write_text(report([case()]), encoding="utf-8")
        b.write_text(report([case()]).replace("fixture.rkb", "other.rkb"),
                     encoding="utf-8")
        out = self.dir / "unsafe"
        self.assertEqual(kbench_ab.main(["compare", str(a), str(b),
                                         "--out", str(out)]), 2)
        self.assertFalse(out.with_suffix(".md").exists())

    def runner_args(self, ops=None, repeat_a=False):
        fixture = self.dir / "fixture.rkb"
        fixture.write_text("synthetic fixture; no hardware used", encoding="utf-8")
        sides = {}
        for label in ("a", "b"):
            home = self.dir / ("home-" + label)
            (home / "bin").mkdir(parents=True, exist_ok=True)
            (home / "bin" / "rad-kbench").write_text("synthetic binary", encoding="utf-8")
            sides[label] = dict(name=label, radiance_home=str(home),
                                kernels=["libr3", "libref"], env={"CUSTOM": label})
        sidefile = self.dir / "sides.json"
        sidefile.write_text(json.dumps(sides), encoding="utf-8")
        return argparse.Namespace(sides=sidefile, fixture=fixture, out=self.dir/"out",
                                  device="0", op=ops or ["gemm_nt", "add"],
                                  seed=23, max_cases=8, bench_warm_ms=100,
                                  bench_reps=7, bench_flush=64, repeat_a=repeat_a,
                                  timeout_s=7.25, resume=False)

    def report_success(self, cmd, **kwargs):
        self.assertEqual(kwargs["env"]["HIP_VISIBLE_DEVICES"], "0")
        self.assertEqual(kwargs["timeout"], 7.25)
        self.assertEqual(cmd.count("--op"), 1)
        self.assertIn("--timing-only", cmd)
        op = cmd[cmd.index("--op") + 1]
        Path(cmd[cmd.index("--report") + 1]).write_text(
            report([case(op=op)]), encoding="utf-8")
        return SimpleNamespace(returncode=0, stdout="ok", stderr="")

    def test_run_split_per_op_progress_manifest_and_repeat_a(self):
        args = self.runner_args(repeat_a=True)
        with patch.object(kbench_ab.subprocess, "run", side_effect=self.report_success) as runner:
            self.assertEqual(kbench_ab.run_sides(args), 0)
        self.assertEqual(runner.call_count, 6)
        manifest = json.loads((args.out/"run.json").read_text())
        records = manifest["executions"]
        self.assertEqual([(r["name"], r["op"]) for r in records],
                         [("a","gemm_nt"),("a","add"),("b","gemm_nt"),("b","add"),
                          ("a.again","gemm_nt"),("a.again","add")])
        self.assertTrue(all(r["exit_code"] == 0 and r["status"] == "ok"
                            and r["wall_time_s"] >= 0 for r in records))
        self.assertTrue(all(r["env"]["CUSTOM"] == r["name"].split(".")[0]
                            for r in records))
        self.assertTrue(all(r["command"][-2] == "--report" for r in records))
        self.assertEqual(manifest["configuration"]["seed"], 23)
        self.assertEqual(manifest["configuration"]["bench_flush_mib"], 64)
        self.assertTrue((args.out/"a"/"gemm_nt.report.md").is_file())
        self.assertTrue((args.out/"a.again"/"add.report.md").is_file())
        progress = (args.out/"progress.log").read_text().splitlines()
        self.assertEqual(len(progress), 6)
        self.assertEqual([(r.split("\t")[0], r.split("\t")[1], r.split("\t")[3])
                          for r in progress],
                         [(r["name"], r["op"], "0") for r in records])
        self.assertTrue(all(float(line.split("\t")[2]) >= 0 for line in progress))

    def test_timeout_and_failure_recorded_and_following_ops_run(self):
        args = self.runner_args()
        counter = 0
        def fake(cmd, **kwargs):
            nonlocal counter
            counter += 1
            if counter == 1:
                raise subprocess.TimeoutExpired(cmd, kwargs["timeout"],
                                                output=b"partial", stderr=b"deadline")
            if counter == 2:
                return SimpleNamespace(returncode=9, stdout="", stderr="failed")
            return self.report_success(cmd, **kwargs)
        with patch.object(kbench_ab.subprocess, "run", side_effect=fake) as runner:
            self.assertEqual(kbench_ab.run_sides(args), 1)
        self.assertEqual(runner.call_count, 4)
        records = json.loads((args.out/"run.json").read_text())["executions"]
        self.assertEqual([r["status"] for r in records],
                         ["timed_out", "failed", "ok", "ok"])
        self.assertIsNone(records[0]["exit_code"])
        self.assertTrue(records[0]["timed_out"])
        self.assertEqual(records[1]["exit_code"], 9)
        self.assertEqual(records[0]["timeout_s"], 7.25)
        self.assertIn("deadline", (args.out/"a"/"gemm_nt.stderr.log").read_text())
        self.assertEqual([line.split("\t")[-1] for line in
                          (args.out/"progress.log").read_text().splitlines()],
                         ["timeout", "9", "0", "0"])

    def test_resume_skips_nonempty_and_reruns_empty_report(self):
        args = self.runner_args()
        with patch.object(kbench_ab.subprocess, "run", side_effect=self.report_success):
            self.assertEqual(kbench_ab.run_sides(args), 0)
        with self.assertRaises(ValueError):
            kbench_ab.run_sides(args)
        (args.out/"b"/"add.report.md").write_text("", encoding="utf-8")
        args.resume = True
        args.timeout_s = 11.0  # Caller may resize timeout between attempts.
        def fake(cmd, **kwargs):
            self.assertEqual(kwargs["timeout"], 11.0)
            self.assertEqual(cmd[cmd.index("--op")+1], "add")
            Path(cmd[cmd.index("--report")+1]).write_text(
                report([case(op="add")]), encoding="utf-8")
            return SimpleNamespace(returncode=0, stdout="", stderr="")
        with patch.object(kbench_ab.subprocess, "run", side_effect=fake) as runner:
            self.assertEqual(kbench_ab.run_sides(args), 0)
        self.assertEqual(runner.call_count, 1)
        manifest = json.loads((args.out/"run.json").read_text())
        self.assertEqual(len(manifest["executions"]), 5)
        self.assertEqual(len(manifest["skipped"]), 3)
        self.assertEqual(len((args.out/"progress.log").read_text().splitlines()), 5)
        args.seed = 24
        with self.assertRaisesRegex(ValueError, "differs"):
            kbench_ab.run_sides(args)

    def test_compare_merges_per_op_directories_and_lists_missing_ops(self):
        a, b = self.dir/"a", self.dir/"b"
        a.mkdir()
        b.mkdir()
        (a/"gemm_nt.report.md").write_text(report([case()]), encoding="utf-8")
        (a/"add.report.md").write_text(report([case(op="add")]), encoding="utf-8")
        (b/"gemm_nt.report.md").write_text(report([case(time=9)]), encoding="utf-8")
        prefix = self.dir/"comparison"
        self.assertEqual(kbench_ab.main(["compare", str(a), str(b),
                                         "--out", str(prefix)]), 0)
        result = json.loads(prefix.with_suffix(".json").read_text())
        self.assertEqual(result["missing_ops_a"], [])
        self.assertEqual(result["missing_ops_b"], ["add"])
        self.assertEqual(result["only_a"][0]["op"], "add")
        self.assertIn("Missing ops on B (1)", prefix.with_suffix(".md").read_text())
        # Directory and a single report can be compared together, too.
        self.assertEqual(kbench_ab.main(["compare", str(a),
                                         str(b/"gemm_nt.report.md"),
                                         "--out", str(self.dir/"mixed")]), 0)

    def test_compare_lists_op_missing_from_both_sides_using_manifest_or_arg(self):
        a, b = self.dir/"a", self.dir/"b"
        a.mkdir()
        b.mkdir()
        for folder in (a, b):
            (folder/"gemm_nt.report.md").write_text(
                report([case()]), encoding="utf-8")
        manifest = {"configuration": {
            "ops": ["gemm_nt", "add"], "sides": {
                "a": {"name": "a"}, "b": {"name": "b"}}}}
        (self.dir/"run.json").write_text(json.dumps(manifest), encoding="utf-8")
        prefix = self.dir/"bothmissing"
        self.assertEqual(kbench_ab.main([
            "compare", str(a), str(b), "--expected-op", "softmax",
            "--out", str(prefix)]), 0)
        result = json.loads(prefix.with_suffix(".json").read_text())
        self.assertEqual(result["missing_ops_a"], ["add", "softmax"])
        self.assertEqual(result["missing_ops_b"], ["add", "softmax"])

    def test_directory_fixture_check_on_every_report_and_op_filename(self):
        a, b = self.dir/"a", self.dir/"b"
        a.mkdir()
        b.mkdir()
        (a/"gemm_nt.report.md").write_text(report([case()]), encoding="utf-8")
        (a/"add.report.md").write_text(
            report([case(op="add")]).replace("fixture.rkb", "other.rkb"),
            encoding="utf-8")
        (b/"gemm_nt.report.md").write_text(report([case()]), encoding="utf-8")
        out = self.dir/"badfixture"
        self.assertEqual(kbench_ab.main(["compare", str(a), str(b),
                                         "--out", str(out)]), 2)
        self.assertFalse(out.with_suffix(".json").exists())
        (a/"add.report.md").write_text(report([case()]), encoding="utf-8")
        self.assertEqual(kbench_ab.main(["compare", str(a), str(b),
                                         "--out", str(out)]), 2)

    def test_probe_executes_one_side_one_op_max_cases_one_and_prints_rows(self):
        args = self.runner_args()
        def fake(cmd, **kwargs):
            self.assertEqual(cmd.count("--op"), 1)
            self.assertEqual(cmd[cmd.index("--op") + 1], "gemm_nt")
            self.assertEqual(cmd[cmd.index("--max-cases") + 1], "1")
            self.assertEqual(kwargs["timeout"], 7.25)
            Path(cmd[cmd.index("--report") + 1]).write_text(
                report([case(), case(n=8)]), encoding="utf-8")
            return SimpleNamespace(returncode=0, stdout="", stderr="")
        cli = ["probe", "--sides", str(args.sides), "--fixture", str(args.fixture),
               "--out", str(args.out), "--device", "0", "--timeout-s", "7.25",
               "--side", "a", "--op", "gemm_nt"]
        printed = StringIO()
        with patch.object(kbench_ab.subprocess, "run", side_effect=fake) as runner:
            with redirect_stdout(printed):
                self.assertEqual(kbench_ab.main(cli), 0)
        self.assertEqual(runner.call_count, 1)
        self.assertIn("2 Speed rows", printed.getvalue())
        self.assertIn(" s,", printed.getvalue())
        self.assertEqual(len(json.loads((args.out/"run.json").read_text())["executions"]), 1)
        self.assertEqual(len((args.out/"progress.log").read_text().splitlines()), 1)

    def test_checked_failure_without_speed_is_not_silently_dropped(self):
        first = case()
        failing = case(n=8, verdict="**FAIL**", rel="0.2")
        text = report([first, failing])
        # A checked failure is not timed by upstream; keep Correctness row.
        line = next(line for line in text.splitlines() if line.startswith("| ") and
                    "n=8 dtype=bf16" in line and "| 10.0 | - |" in line)
        path = self.dir / "checked.md"
        path.write_text(text.replace(line + "\n", ""), encoding="utf-8")
        cases = kbench_ab.parse_report(path)
        self.assertEqual(len(cases), 2)
        missing = [r for r in cases.values() if r["us"] is None]
        self.assertEqual(len(missing), 1)
        self.assertEqual(missing[0]["verdict"], "FAIL")



if __name__ == "__main__":
    unittest.main()
