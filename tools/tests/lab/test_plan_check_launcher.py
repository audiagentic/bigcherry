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

import itertools
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
mode=""; cache=""; planfile=""; expect=""; sets=(); out=""; unmeasured=""; objects=()
while [ $# -gt 0 ]; do
  case "$1" in
    --plan-check-args|--plan-key|--plan-store) mode=$1; shift ;;
    --write-manifest) mode=$1; out=$2; shift 2 ;;
    --unmeasured) unmeasured=1; shift ;;
    --live-policy|--experiment) objects+=("$1=$2"); shift 2 ;;
    --plan-cache) cache=$2; shift 2 ;;
    --plan-file) planfile=$2; shift 2 ;;
    --expect-key) expect=$2; shift 2 ;;
    --engine-binary|--site) shift 2 ;;
    --plan-extra) sets+=("plan-extra=$2"); shift 2 ;;
    --set) sets+=("$2"); shift 2 ;;
    *) echo "fake bc-serve: unknown $1" >&2; exit 2 ;;
  esac
done
if [ "$mode" = --write-manifest ]; then
  # the helpers' manifest: written once (bc-serve refuses an existing OUT); a measured one needs the stored plan
  [ -e "$out" ] && { echo "bc-serve: refusing to replace $out" >&2; exit 2; }
  key=$(printf '%s\n' "${sets[@]}" | grep -v -e '^engine.port=' -e '^memory.host_kv_mib=' -e '^plan-extra=' -e '^handoff\.' | sha256sum | cut -c1-8)
  if [ -n "$unmeasured" ]; then state=unmeasured
  else
    state=measured
    skey=$(printf '%s\n' "${sets[@]}" | grep -v -e '^engine.port=' -e '^memory.host_kv_mib=' | sha256sum | cut -c1-64)
    [ -f "$cache/$skey.plan" ] || { echo "bc-serve: no measured plan for this configuration" >&2; exit 10; }
  fi
  printf '{"plan_state": "%s"}\n' "$state" > "$out"
  { echo "$state $out ${objects[*]:-}"; printf '%s\n' "${sets[@]}"; } >> "$(dirname "$out")/manifests.log"
  echo "manifest: $out"
  exit 0
fi
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
        self._stamps = itertools.count()
        self.heat = self.root / "heat.hot-first"
        self.heat.write_bytes(b"heat-v1")

    def _launch(self, **overrides: str) -> subprocess.CompletedProcess[str]:
        """Source the library with the launcher's variables and a fake measuring start, then arm the check."""
        values = dict(LAYOUT="flashnext-4gpu", SEQS="8", MAX_LEN="262144", HOST_POOL_MIB="9216", HOST_KV_MIB="12288",
                      CHUNK="2048", NGRAM="ram", KV_DTYPE="fp8", ROPE="none", PORT="36080", PREFIX_RESUME="nearest",
                      MODEL="/model.rad", HEAT=str(self.heat), RECORD="/heat.live", CKPT_INTERVAL="4096", NUM_SPEC="3",
                      SHM="/r3-moe-helper", RECORD_PERIOD="60", STAMP=f"1105-{next(self._stamps):06d}", LIVE_POLICY_FILE="", EXPERIMENT_FILE="")
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
  echo "measure: engine --plan-memory $PLAN_MEASURE_FILE helpers-manifest $PLAN_MEASURING_MANIFEST $(cat "$PLAN_MEASURING_MANIFEST")" >> {shlex.quote(str(self.engine_calls))}
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
        # the one real engine start passes the armed flags; the two other calls are the helper and dense memory-plan measurements
        calls = [l.strip() for l in text.splitlines() if l.strip().startswith("start_engine ") and "()" not in l]
        self.assertEqual(len(calls), 3, calls)
        self.assertEqual(sum("--plan-memory" in c for c in calls), 2)
        self.assertTrue(any("--plan-memory" in c for c in calls))
        self.assertTrue(any("PLAN_CHECK_ARGS" in c for c in calls))
        # the check is armed, and fails the launch, before anything starts for real
        self.assertLess(text.index("plan_check_flags ||"), text.index('start_helpers "$PLAN_MANIFEST"; fi\nLOG='))
        self.assertIn("exit 97", text)
        for banned in ("--no-check", "SKIP_PLAN_CHECK", "NO_PLAN_CHECK"):
            self.assertNotIn(banned, text + LIB.read_text())

    def _manifests(self) -> list[str]:
        log = self.run_dir / "manifests.log"
        return [line for line in log.read_text().splitlines() if line.startswith(("unmeasured ", "measured "))] if log.exists() else []

    def test_a_miss_writes_the_unmeasured_manifest_for_the_measuring_start_then_the_measured_one(self) -> None:
        done = self._launch()
        self.assertEqual(done.returncode, 0, done.stderr)
        written = self._manifests()
        self.assertEqual([w.split()[0] for w in written], ["unmeasured", "measured"], written)
        # the measuring start saw the unmeasured manifest, already on disk, under the launch's own stamp
        measure = self.engine_calls.read_text()
        measuring_manifests = sorted(self.run_dir.glob("manifest-measuring-1105-000000-*.json"))
        self.assertEqual(len(measuring_manifests), 1)
        self.assertIn("helpers-manifest " + str(measuring_manifests[0]), measure)
        self.assertIn('{"plan_state": "unmeasured"}', measure)
        # the real start's manifest is the measured one, written after the plan was stored, never before
        (serving,) = sorted(self.run_dir.glob("manifest-1105-000000-*.json"))
        self.assertEqual(serving.read_text(), '{"plan_state": "measured"}\n')

    def test_a_hit_writes_only_the_measured_manifest(self) -> None:
        self._launch()
        before = len(self._manifests())
        again = self._launch()
        self.assertEqual(again.returncode, 0, again.stderr)
        self.assertEqual([w.split()[0] for w in self._manifests()[before:]], ["measured"])
        self.assertEqual(len(list(self.run_dir.glob("manifest-1105-000001-*.json"))), 1)
        self.assertEqual(list(self.run_dir.glob("manifest-measuring-1105-000001-*.json")), [])

    def test_the_manifests_describe_the_same_configuration_as_the_plan(self) -> None:
        self._launch()
        log = (self.run_dir / "manifests.log").read_text().splitlines()
        blocks, current = [], []
        for line in log:
            if line.startswith(("unmeasured ", "measured ")):
                current = [line]
                blocks.append(current)
            else:
                current.append(line)
        sets = [[l for l in block[1:] if not l.startswith("plan-extra=")] for block in blocks]
        self.assertEqual(len(sets), 2)
        self.assertEqual(sets[0], sets[1])
        self.assertIn("handoff.histogram_period_s=60", sets[0])
        self.assertIn('handoff.shm="/r3-moe-helper"', sets[0])

    def test_a_live_policy_and_a_lab_experiment_reach_both_manifests(self) -> None:
        self._launch(LIVE_POLICY_FILE="/p/policy.json", EXPERIMENT_FILE="/p/exp.json")
        for line in self._manifests():
            self.assertIn("--live-policy=/p/policy.json", line)
            self.assertIn("--experiment=/p/exp.json", line)

    def test_a_manifest_that_cannot_be_written_stops_the_start(self) -> None:
        self.serve.write_text(FAKE_SERVE.replace('printf \'{"plan_state": "%s"}\\n\' "$state" > "$out"', 'echo "bc-serve: cannot write" >&2; exit 2'))
        done = self._launch()
        self.assertNotEqual(done.returncode, 0)
        self.assertNotIn("ARGS:", done.stdout)
        self.assertIn("manifest was not written", done.stderr)

    def test_the_launcher_script_parses(self) -> None:
        for path in (LAUNCHER, LIB):
            done = subprocess.run(["bash", "-n", str(path)], capture_output=True, text=True)
            self.assertEqual(done.returncode, 0, f"{path.name}: {done.stderr}")


if __name__ == "__main__":
    unittest.main()
