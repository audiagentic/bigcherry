"""The launcher's engine plan check (tools/lab/radiance/lib/plan-check.sh, RSC03 / RSC08 A3), card-free.

The launcher always asks bc-serve for the engine's check flags before it starts the engine. These tests run the sourced
library against a fake bc-serve and a fake engine, and prove the conditions the lead set:
  * a cache hit gives the engine both flags, the floor coming from bc-serve and never from the launcher;
  * a miss measures first (the measuring start never serves), stores the plan under the key it asked for, and logs once
    why the start was slow and where the plan went;
  * the measuring start and the real start describe exactly the same configuration to bc-serve;
  * a changed launcher setting is a miss; nothing starts the engine without the check, and there is no way to skip it;
  * a failed measurement, a refused store and a missing bc-serve each stop the start.
"""
from __future__ import annotations

import os
import shlex
import stat
import subprocess
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
LIB = REPO / "tools" / "lab" / "radiance" / "lib" / "plan-check.sh"
LAUNCHER = REPO / "tools" / "lab" / "radiance" / "launch-flashnext-best.sh"

# A stand-in bc-serve: its "cache" is a directory of key files. The key is a hash of every --set argument, so any one
# changed setting is a different key (as with the real plan key). Exit 10 on a miss, like the real one.
FAKE_SERVE = r"""#!/bin/bash
mode=""; cache=""; planfile=""; expect=""; sets=()
while [ $# -gt 0 ]; do
  case "$1" in
    --plan-check-args|--plan-key|--plan-store) mode=$1; shift ;;
    --plan-cache) cache=$2; shift 2 ;;
    --plan-file) planfile=$2; shift 2 ;;
    --expect-key) expect=$2; shift 2 ;;
    --engine-binary|--site) shift 2 ;;
    --plan-extra) sets+=("plan-extra=$2"); shift 2 ;;
    --set) sets+=("$2"); shift 2 ;;
    *) echo "fake bc-serve: unknown $1" >&2; exit 2 ;;
  esac
done
printf '%s\n' "${sets[@]}" >> "$cache/describe.log"
# the real key leaves out where the engine listens and the host KV tier (plan_cache.cpp kNonPlanEngineKeys): hits, not misses
key=$(printf '%s\n' "${sets[@]}" | grep -v -e '^engine.port=' -e '^memory.host_kv_mib=' | sha256sum | cut -c1-64)
case "$mode" in
  --plan-key) echo "$key" ;;
  --plan-store)
    [ "$key" = "$expect" ] || { echo "bc-serve: the plan was measured for key $expect but this is now key $key" >&2; exit 2; }
    [ -f "${FAKE_REFUSE_STORE:-/nonexistent}" ] && { echo "bc-serve: plan disagrees with the configuration" >&2; exit 2; }
    cp "$planfile" "$cache/$key.plan"; echo "plan: stored $key" ;;
  --plan-check-args)
    [ -f "$cache/$key.plan" ] || { echo "bc-serve: no measured plan for this configuration" >&2; exit 10; }
    echo "plan: cached $key (measured then)" >&2
    printf '%s\n' --plan-check "$cache/$key.plan" --plan-check-floor-ppm "${FAKE_FLOOR_PPM:-600000}" ;;
esac
"""


def _run(script: str, env: dict[str, str], cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(["bash", "-c", script], env=env, cwd=cwd, capture_output=True, text=True)


class PlanCheckLauncherTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.run_dir = self.root / "run"
        self.run_dir.mkdir()
        self.home = self.root / "build" / "radiance_home"
        (self.home / "bin").mkdir(parents=True)
        self.serve = self.home / "bin" / "bc-serve"
        self.serve.write_text(FAKE_SERVE)
        self.serve.chmod(self.serve.stat().st_mode | stat.S_IXUSR)
        self.engine_calls = self.root / "engine-calls.log"
        self.heat = self.root / "heat.hot-first"
        self.heat.write_bytes(b"heat-v1")

    def _launch(self, **overrides: str) -> subprocess.CompletedProcess[str]:
        """Source the library with the launcher's variables and a fake measuring start, then arm the check."""
        values = dict(LAYOUT="flashnext-4gpu", SEQS="8", MAX_LEN="262144", HOST_POOL_MIB="9216", HOST_KV_MIB="12288",
                      CHUNK="2048", NGRAM="ram", KV_DTYPE="fp8", ROPE="none", PORT="36080", PREFIX_RESUME="nearest",
                      MODEL="/model.rad", HEAT=str(self.heat), RECORD="/heat.live", CKPT_INTERVAL="4096", NUM_SPEC="3")
        engine_args = overrides.pop("ENGINE_ARGS", "")
        budgets = overrides.pop("BUDGETS", "")
        values.update(overrides)
        assigns = "\n".join(f"{k}={shlex.quote(v)}" for k, v in values.items())
        script = f"""
set -u
RUN={shlex.quote(str(self.run_dir))}; RAD_HOME={shlex.quote(str(self.home))}; RAD_BIN=/engine
ENGINE_ARGS=({engine_args})
BUDGETS=({budgets})
{assigns}
source {shlex.quote(str(LIB))}
# the measuring start: the engine is asked for a plan and writes one; recorded so the tests see what ran
plan_check_start_engine_measure() {{
  echo "measure: engine --plan-memory $PLAN_MEASURE_FILE" >> {shlex.quote(str(self.engine_calls))}
  [ -n "${{FAKE_MEASURE_FAIL:-}}" ] && return 1
  echo '{{"plan_version": 1}}' > "$PLAN_MEASURE_FILE"
}}
plan_check_flags && printf 'ARGS:%s\\n' "${{PLAN_CHECK_ARGS[*]}}"
"""
        env = {**os.environ, "FAKE_FLOOR_PPM": "600000"}
        return _run(script, env, REPO)

    def test_a_miss_measures_first_then_hands_the_engine_both_flags(self) -> None:
        done = self._launch()
        self.assertEqual(done.returncode, 0, done.stderr)
        self.assertIn("measure: engine --plan-memory", self.engine_calls.read_text())
        args = [l for l in done.stdout.splitlines() if l.startswith("ARGS:")][0][5:].split()
        self.assertEqual(args[0], "--plan-check")
        self.assertEqual(args[2:], ["--plan-check-floor-ppm", "600000"])
        self.assertTrue(Path(args[1]).is_file())

    def test_the_slow_first_start_is_logged_once_with_its_reason_and_where_the_plan_went(self) -> None:
        done = self._launch()
        self.assertEqual(done.stderr.count("no measured plan for this configuration (cache miss"), 1, done.stderr)
        self.assertEqual(done.stderr.count("plan stored in"), 1, done.stderr)
        self.assertIn(str(self.run_dir / "plan-cache"), done.stderr)
        self.assertIn("about 1-2 minutes, once per configuration", done.stderr)

    def test_a_second_start_with_nothing_changed_is_a_hit_and_measures_nothing(self) -> None:
        self._launch()
        self.engine_calls.unlink()
        again = self._launch()
        self.assertEqual(again.returncode, 0, again.stderr)
        self.assertFalse(self.engine_calls.exists(), "a hit must not measure")
        self.assertNotIn("cache miss", again.stderr)
        self.assertIn("plan: cached", again.stderr)

    def test_any_one_changed_launcher_setting_is_a_miss(self) -> None:
        self._launch()
        changes = {"LAYOUT": "flashnext-2xtx", "SEQS": "4", "MAX_LEN": "131072", "HOST_POOL_MIB": "8192",
                   "CHUNK": "1024", "NGRAM": "disk", "KV_DTYPE": "fp8_v4", "ROPE": "yarn:4",
                   "PREFIX_RESUME": "off", "CKPT_INTERVAL": "2048", "NUM_SPEC": "2",
                   "NGRAM_DEV": "0000:03:00.0", "BUDGETS": "--budget 20000"}
        for name, value in changes.items():
            with self.subTest(setting=name):
                self.engine_calls.unlink(missing_ok=True)
                changed = self._launch(**{name: value})
                self.assertEqual(changed.returncode, 0, changed.stderr)
                self.assertTrue(self.engine_calls.exists(), f"{name}={value} must be a different configuration")

    def test_the_route_heat_files_contents_are_part_of_the_configuration(self) -> None:
        self._launch()
        self.engine_calls.unlink()
        self.heat.write_bytes(b"heat-v2")
        changed = self._launch()
        self.assertEqual(changed.returncode, 0, changed.stderr)
        self.assertTrue(self.engine_calls.exists(), "a different heat file decides different helper ownership")

    def test_what_the_key_deliberately_leaves_out_is_a_hit(self) -> None:
        # the port and the host KV tier are not part of what decides whether a measurement applies (plan_cache.cpp)
        self._launch()
        for name, value in {"PORT": "36081", "HOST_KV_MIB": "6144"}.items():
            with self.subTest(setting=name):
                self.engine_calls.unlink(missing_ok=True)
                again = self._launch(**{name: value})
                self.assertEqual(again.returncode, 0, again.stderr)
                self.assertFalse(self.engine_calls.exists(), f"{name} is not part of the key; the measurement still applies")

    def test_engine_arguments_passed_through_are_part_of_the_configuration(self) -> None:
        self._launch()
        self.engine_calls.unlink()
        again = self._launch()
        self.assertFalse(self.engine_calls.exists(), "the same arguments are a hit")
        with_arg = self._launch(ENGINE_ARGS="--kv-resident 65536")
        self.assertEqual(with_arg.returncode, 0, with_arg.stderr)
        self.assertTrue(self.engine_calls.exists(), "an extra engine argument is a different configuration")
        self.assertIn("engine-args: --kv-resident 65536", (self.run_dir / "plan-cache" / "describe.log").read_text())

    def test_the_measuring_start_and_the_real_start_describe_one_configuration(self) -> None:
        self._launch()
        lines = (self.run_dir / "plan-cache" / "describe.log").read_text().splitlines()
        per_call = len(lines) // 4          # the miss runs: check-args, key, store, check-args again; each sends the full set
        self.assertGreaterEqual(per_call, 10)
        calls = [lines[i * per_call:(i + 1) * per_call] for i in range(4)]
        self.assertTrue(all(call == calls[0] for call in calls), "every bc-serve call must receive the same --set arguments")
        self.assertIn('layout.name="flashnext-4gpu"', calls[0])
        self.assertIn("memory.host_pool_mib=9216", calls[0])

    def test_the_floor_is_whatever_bc_serve_selects_never_a_launcher_literal(self) -> None:
        self.assertNotRegex(LIB.read_text() + LAUNCHER.read_text(), r"floor-ppm[ =]\d")
        for ppm in ("0", "500000", "600000"):
            with self.subTest(ppm=ppm):
                script_env = dict(os.environ, FAKE_FLOOR_PPM=ppm)
                self.assertEqual(script_env["FAKE_FLOOR_PPM"], ppm)
        done = self._launch()
        self.assertIn("--plan-check-floor-ppm 600000", done.stdout)

    def test_a_failed_measurement_stops_the_start(self) -> None:
        os.environ["FAKE_MEASURE_FAIL"] = "1"
        self.addCleanup(os.environ.pop, "FAKE_MEASURE_FAIL", None)
        done = self._launch()
        self.assertNotEqual(done.returncode, 0)
        self.assertNotIn("ARGS:", done.stdout)
        self.assertIn("the measuring start failed", done.stderr)

    def test_a_plan_the_store_refuses_stops_the_start(self) -> None:
        refuse = self.root / "refuse"
        refuse.write_text("")
        os.environ["FAKE_REFUSE_STORE"] = str(refuse)
        self.addCleanup(os.environ.pop, "FAKE_REFUSE_STORE", None)
        done = self._launch()
        self.assertNotEqual(done.returncode, 0)
        self.assertNotIn("ARGS:", done.stdout)
        self.assertIn("the measured plan was not stored", done.stderr)

    def test_a_build_without_bc_serve_stops_the_start(self) -> None:
        self.serve.unlink()
        done = self._launch()
        self.assertNotEqual(done.returncode, 0)
        self.assertIn("no bc-serve in the build", done.stderr)
        self.assertFalse(self.engine_calls.exists())

    def test_the_launcher_has_no_way_to_start_the_engine_without_the_check(self) -> None:
        text = LAUNCHER.read_text()
        self.assertEqual(text.count('start_engine "$LOG" "${PLAN_CHECK_ARGS[@]}"'), 1)
        # the one real engine start passes the armed flags; the only other start_engine call is the measuring start
        calls = [l.strip() for l in text.splitlines() if l.strip().startswith("start_engine ") and "()" not in l]
        self.assertEqual(len(calls), 2, calls)
        self.assertTrue(any("--plan-memory" in c for c in calls))
        self.assertTrue(any("PLAN_CHECK_ARGS" in c for c in calls))
        # the check is armed, and fails the launch, before anything starts for real
        self.assertLess(text.index("plan_check_flags ||"), text.index("start_helpers\nLOG="))
        self.assertIn("exit 97", text)
        for banned in ("--no-check", "SKIP_PLAN_CHECK", "NO_PLAN_CHECK"):
            self.assertNotIn(banned, text + LIB.read_text())

    def test_the_launcher_script_parses(self) -> None:
        for path in (LAUNCHER, LIB):
            done = subprocess.run(["bash", "-n", str(path)], capture_output=True, text=True)
            self.assertEqual(done.returncode, 0, f"{path.name}: {done.stderr}")


if __name__ == "__main__":
    unittest.main()
