"""The launcher's serving manifest for a helper layout (tools/lab/radiance/lib/plan-check.sh, RSC03 / RSC08 S3), card-free.

The engine and the helpers start from one measured manifest bc-serve writes for the exact configuration about to launch, and
the engine admits the plan that manifest carries. These tests run the sourced library against a fake bc-serve and a fake
engine, and prove the conditions the lead set:
  * a miss measures first (the measuring start never serves) and bc-serve writes the measured manifest from that plan, storing
    it; the start is logged once with its reason and where the plan went;
  * the measuring start and the real start describe exactly the same configuration to bc-serve;
  * a changed launcher setting is a miss; nothing starts the engine without the manifest, and there is no way to skip the check;
  * a failed measurement, a refused plan and a missing bc-serve each stop the start;
  * no --plan-check flag, no bc-serve plan-store/plan-key/plan-check-args mode and no raw engine argument survive.
"""
from __future__ import annotations

import itertools
import json
import os
import re
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
cache=""; sets=(); out=""; unmeasured=""; measured_plan=""; objects=(); helpers_mib=""; engine_mib=""
while [ $# -gt 0 ]; do
  case "$1" in
    --write-manifest) out=$2; shift 2 ;;
    --unmeasured) unmeasured=1; shift ;;
    --measured-plan) measured_plan=$2; shift 2 ;;
    --helpers-resident-mib) helpers_mib=$2; shift 2 ;;
    --engine-resident-mib) engine_mib=$2; shift 2 ;;
    --live-policy|--experiment) objects+=("$1=$2"); shift 2 ;;
    --plan-cache) cache=$2; shift 2 ;;
    --engine-binary|--site) shift 2 ;;
    --plan-extra) sets+=("plan-extra=$2"); shift 2 ;;
    --set) sets+=("$2"); shift 2 ;;
    *) echo "fake bc-serve: unknown $1" >&2; exit 2 ;;
  esac
done
[ -n "$out" ] || { echo "fake bc-serve: only --write-manifest is a mode" >&2; exit 2; }
# the manifest: written once (bc-serve refuses an existing OUT); a measured one needs a plan, from the cache or --measured-plan
[ -e "$out" ] && { echo "bc-serve: refusing to replace $out" >&2; exit 2; }
if [ -n "$unmeasured" ]; then state=unmeasured
else
  state=measured
  # the real key leaves out where the engine listens and the host KV tier (plan_cache.cpp kNonPlanEngineKeys): hits, not misses
  skey=$(printf '%s\n' "${sets[@]}" | grep -v -e '^engine.port=' -e '^memory.host_kv_mib=' | sha256sum | cut -c1-64)
  if [ -n "$measured_plan" ]; then
    [ -n "$helpers_mib" ] && [ -n "$engine_mib" ] || { echo "bc-serve: --measured-plan needs the resident host memory" >&2; exit 2; }
    echo "resident $helpers_mib $engine_mib" >> "$(dirname "$out")/resident.log"
    [ -f "${FAKE_REFUSE_STORE:-/nonexistent}" ] && { echo "bc-serve: plan disagrees with the configuration" >&2; exit 2; }
    cp "$measured_plan" "$cache/$skey.plan"; echo "plan: stored $skey" >&2
  fi
  [ -f "$cache/$skey.plan" ] || { echo "bc-serve: no measured plan for this configuration" >&2; exit 10; }
  echo "plan: cached $skey (measured then)" >&2
fi
printf '%s\n' "${sets[@]}" >> "$(dirname "$out")/plan-cache/describe.log"
port=$(printf '%s\n' "${sets[@]}" | sed -n 's/^engine\.port=//p')
# The fake bc-serve emits a structurally valid v1-shaped manifest with a
# provenance-backed policy absence. It never falsely claims calibration.
printf '{"plan_state": "%s", "port": %s}\n' "$state" "${port:-0}" > "$out"
# The fake now adds the closed v1 policy shape and its explicit OFF
# provenance, preserving the original manifest-write failure injection.
python3 - "$out" <<'MANIFEST_POLICY'
import json
import sys
from pathlib import Path
path = Path(sys.argv[1])
manifest = json.loads(path.read_text())
manifest["spec"] = {"handoff": {"placement": {}, "telemetry": {"period_s": 1}}}
manifest["provenance"] = {
    "/spec/handoff/placement": {
        "source": "live policy disabled: fake bc-serve has no calibration record"
    }
}
path.write_text(json.dumps(manifest) + "\n")
MANIFEST_POLICY
{ echo "$state $out ${objects[*]:-}"; printf '%s\n' "${sets[@]}"; } >> "$(dirname "$out")/manifests.log"
echo "manifest: $out"
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
        """Source the library with the launcher's variables and a fake measuring start, then make the manifest."""
        values = dict(LAYOUT="flashnext-4gpu", SEQS="8", MAX_LEN="262144", HOST_POOL_MIB="9216", HOST_KV_MIB="12288",
                      CHUNK="2048", NGRAM="ram", KV_DTYPE="fp8", ROPE="none", PORT="36080", PREFIX_RESUME="nearest",
                      MODEL="/model.rad", HEAT=str(self.heat), RECORD="/heat.live", CKPT_INTERVAL="4096", NUM_SPEC="3",
                      SHM="/r3-moe-helper", RECORD_PERIOD="1", STAMP=f"1105-{next(self._stamps):06d}", LIVE_POLICY_FILE="", EXPERIMENT_FILE="")
        budgets = overrides.pop("BUDGETS", "")
        # what `bigcherry layout env` resolves for flashnext-4gpu, the per-helper budgets with any --budget override applied
        layout_common = overrides.pop("LAYOUT_COMMON", "R3_MOE_WORKER_COUNT=3 R3_MOE_WORKER_MIBS=22528,22528,14336")
        values.update(overrides)
        assigns = "\n".join(f"{k}={shlex.quote(v)}" for k, v in values.items())
        script = f"""
set -u
RUN={shlex.quote(str(self.run_dir))}; RAD_HOME={shlex.quote(str(self.home))}; RAD_BIN=/engine
BUDGETS=({budgets})
LAYOUT_COMMON=({layout_common})
{assigns}
source {shlex.quote(str(LIB))}
# the measuring start: the engine is asked for a plan and writes one; recorded so the tests see what ran
plan_check_start_engine_measure() {{
  PLAN_HELPERS_RESIDENT_MIB=${{FAKE_HELPERS_MIB-2304}}
  echo "radiance: --plan-memory: wrote $PLAN_MEASURE_FILE; engine start-up took ${{FAKE_ENGINE_MIB-1536}} MiB of MemAvailable; nothing loaded" >> "$RUN/engine-measure.log"
  echo "measure: engine --manifest $PLAN_MEASURING_MANIFEST --plan-memory $PLAN_MEASURE_FILE $(cat "$PLAN_MEASURING_MANIFEST")" >> {shlex.quote(str(self.engine_calls))}
  [ -n "${{FAKE_MEASURE_FAIL:-}}" ] && return 1
  echo '{{"plan_version": 1}}' > "$PLAN_MEASURE_FILE"
}}
plan_manifest_prepare && printf 'MANIFEST:%s\\n' "$PLAN_MANIFEST"
"""
        return _run(script, dict(os.environ), REPO)

    def _manifest(self, done: subprocess.CompletedProcess[str]) -> Path:
        line = [l for l in done.stdout.splitlines() if l.startswith("MANIFEST:")][0]
        return Path(line[len("MANIFEST:"):])

    def test_a_miss_measures_first_then_the_measured_manifest_is_the_one_to_start_from(self) -> None:
        done = self._launch()
        self.assertEqual(done.returncode, 0, done.stderr)
        self.assertIn("measure: engine --manifest", self.engine_calls.read_text())
        self.assertIn("--plan-memory", self.engine_calls.read_text())
        self.assertEqual(json.loads(self._manifest(done).read_text())["plan_state"], "measured")

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

    def test_any_one_changed_launcher_setting_is_a_miss(self) -> None:
        self._launch()
        changes = {"LAYOUT": "flashnext-2xtx", "SEQS": "4", "MAX_LEN": "131072", "HOST_POOL_MIB": "8192",
                   "CHUNK": "1024", "NGRAM": "disk", "KV_DTYPE": "fp8_v4", "ROPE": "yarn:4",
                   "PREFIX_RESUME": "off", "CKPT_INTERVAL": "2048", "NUM_SPEC": "2", "BUDGETS": "--budget 20000"}
        for name, value in changes.items():
            with self.subTest(setting=name):
                self.engine_calls.unlink(missing_ok=True)
                changed = self._launch(**{name: value})
                self.assertEqual(changed.returncode, 0, changed.stderr)
                self.assertTrue(self.engine_calls.exists(), f"{name}={value} must be a different configuration")

    def test_a_budget_override_reaches_bc_serve_as_the_helper_budgets(self) -> None:
        # --budget gfx1030=7168: the launcher hands bc-serve the budgets `layout env` resolved with the override, in helper order,
        # so the helpers' manifest carries them (not only the plan key); without --budget bc-serve derives them from the layout.
        self._launch(BUDGETS="gfx1030=7168", LAYOUT_COMMON="R3_MOE_WORKER_COUNT=3 R3_MOE_WORKER_MIBS=22528,22528,7168")
        sets = [l for l in self._blocks()[0][1:] if not l.startswith("plan-extra=")]
        self.assertIn("memory.helper_budgets_mib=[22528,22528,7168]", sets)
        self.engine_calls.unlink(missing_ok=True)
        self._launch()
        sets = [l for l in self._blocks()[-1][1:] if not l.startswith("plan-extra=")]
        self.assertFalse([l for l in sets if l.startswith("memory.helper_budgets_mib=")])

    def test_a_budget_override_without_resolved_budgets_stops_the_start(self) -> None:
        done = self._launch(BUDGETS="gfx1030=7168", LAYOUT_COMMON="R3_MOE_WORKER_COUNT=3")
        self.assertNotEqual(done.returncode, 0)
        self.assertIn("--budget given but the layout resolved no helper budgets", done.stderr)

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

    def test_there_is_no_raw_engine_argument_and_the_n_gram_card_is_refused(self) -> None:
        # RSC08 amendment 11: the engine's flags are manifest fields (spec.experiment.engine, spec.processes.engine), so the launcher
        # has no way to pass one raw; the n-gram card is not a manifest field yet and every layout refuses it
        text = LAUNCHER.read_text()
        self.assertNotIn("--engine-arg)", text)
        self.assertNotIn("ENGINE_ARGS", text)
        self.assertIn("--ngram-device cannot be used", text)
        self.assertIn("exit 99", text[text.index("--ngram-device cannot be used"):][:400])
        self.assertNotIn("engine-args:", LIB.read_text())
        self.assertNotIn("ngram-device:", LIB.read_text())

    def test_the_launcher_exports_no_setting_to_the_engine_or_the_helpers(self) -> None:
        # RSC11/RSC12: the manifest is the one configuration; the process environment is only where the runtime lives
        text = LAUNCHER.read_text()
        code = "\n".join(line.split("#", 1)[0] for line in text.splitlines())
        # an assignment; the sed that READS the layout's plugin file is not one
        self.assertEqual(re.findall(r"(?<![\^\w])(?:R3_[A-Z0-9_]+|RAD_(?!BIN=|HOME=)[A-Z0-9_]+|BC_GFX12_[A-Z0-9_]+)=", code), [])
        self.assertIn('ENGINE_ENV=(RADIANCE_HOME="$RAD_HOME")', text)

    def test_the_measuring_start_and_the_real_start_describe_one_configuration(self) -> None:
        self._launch()
        blocks = self._blocks()
        self.assertEqual([b[0].split()[0] for b in blocks], ["unmeasured", "measured"])
        sets = [[l for l in b[1:] if not l.startswith("plan-extra=")] for b in blocks]
        self.assertEqual(sets[0], sets[1], "both bc-serve calls must receive the same --set arguments")
        self.assertGreaterEqual(len(sets[0]), 10)
        self.assertIn('layout.name="flashnext-4gpu"', sets[0])
        self.assertIn("memory.host_pool_mib=9216", sets[0])
        self.assertIn("handoff.histogram_period_s=1", sets[0])
        self.assertIn('handoff.shm="/r3-moe-helper"', sets[0])

    def test_the_floor_is_never_a_launcher_literal(self) -> None:
        # the floor reaches the engine inside the measured manifest, from bc-serve's selected calibration (engine_view); the
        # launcher neither knows it nor passes it
        self.assertNotRegex(LIB.read_text() + LAUNCHER.read_text(), r"floor-ppm|floor_ppm")

    def test_the_resident_host_memory_of_the_measuring_start_goes_to_bc_serve_with_the_plan(self) -> None:
        done = self._launch()
        self.assertEqual(done.returncode, 0, done.stderr)
        self.assertEqual((self.run_dir / "resident.log").read_text().split(), ["resident", "2304", "1536"])
        # a hit measures nothing and passes nothing: the figures live in the cache entry
        again = self._launch()
        self.assertEqual(again.returncode, 0, again.stderr)
        self.assertEqual((self.run_dir / "resident.log").read_text().count("resident"), 1)

    def test_a_measuring_start_that_does_not_report_the_resident_memory_stops_the_start(self) -> None:
        for name in ("FAKE_ENGINE_MIB", "FAKE_HELPERS_MIB"):
            with self.subTest(missing=name):
                os.environ[name] = ""
                try:
                    done = self._launch()
                finally:
                    os.environ.pop(name, None)
                self.assertNotEqual(done.returncode, 0)
                self.assertNotIn("MANIFEST:", done.stdout)
                self.assertIn("did not report the resident host memory", done.stderr)
                for f in self.run_dir.glob("manifest-*"):
                    f.unlink()
                for f in (self.run_dir / "plan-cache").glob("*"):
                    f.unlink()

    def test_a_failed_measurement_stops_the_start(self) -> None:
        os.environ["FAKE_MEASURE_FAIL"] = "1"
        self.addCleanup(os.environ.pop, "FAKE_MEASURE_FAIL", None)
        done = self._launch()
        self.assertNotEqual(done.returncode, 0)
        self.assertNotIn("MANIFEST:", done.stdout)
        self.assertIn("the measuring start failed", done.stderr)

    def test_a_plan_bc_serve_refuses_stops_the_start(self) -> None:
        refuse = self.root / "refuse"
        refuse.write_text("")
        os.environ["FAKE_REFUSE_STORE"] = str(refuse)
        self.addCleanup(os.environ.pop, "FAKE_REFUSE_STORE", None)
        done = self._launch()
        self.assertNotEqual(done.returncode, 0)
        self.assertNotIn("MANIFEST:", done.stdout)
        self.assertIn("the measured manifest was not written", done.stderr)

    def test_a_build_without_bc_serve_stops_the_start(self) -> None:
        self.serve.unlink()
        done = self._launch()
        self.assertNotEqual(done.returncode, 0)
        self.assertIn("no bc-serve in the build", done.stderr)
        self.assertFalse(self.engine_calls.exists())

    def test_the_launcher_has_no_way_to_start_the_engine_without_the_manifest_and_its_check(self) -> None:
        text = LAUNCHER.read_text()
        lib = LIB.read_text()
        # every layout's one real engine start is from the measured manifest; the measuring start is from the unmeasured one
        self.assertEqual(text.count('start_engine_manifest "$LOG" "$PLAN_MANIFEST"'), 1)
        self.assertEqual(text.count('start_engine_manifest "$RUN/engine-measure.log" "$PLAN_MEASURING_MANIFEST" --plan-memory'), 1)
        # there is no argv start left: not for a helper layout, and not for a dense one either
        self.assertNotIn("start_engine ", text.replace("start_engine_manifest", ""))
        # the manifest is made, and fails the launch, before anything starts for real
        self.assertLess(text.index("plan_manifest_prepare ||"), text.index('start_helpers "$PLAN_MANIFEST"; fi'))
        self.assertLess(text.index('start_helpers "$PLAN_MANIFEST"; fi'), text.index('start_engine_manifest "$LOG" "$PLAN_MANIFEST"'))
        self.assertIn("exit 97", text)
        for banned in ("--no-check", "SKIP_PLAN_CHECK", "NO_PLAN_CHECK", "--plan-check", "plan-check-floor-ppm", "PLAN_CHECK_ARGS",
                       "--plan-store", "--plan-key", "--plan-check-args", "plan_check_flags"):
            self.assertNotIn(banned, text + lib)

    def _blocks(self) -> list[list[str]]:
        log = self.run_dir / "manifests.log"
        blocks: list[list[str]] = []
        for line in log.read_text().splitlines() if log.exists() else []:
            if line.startswith(("unmeasured ", "measured ")):
                blocks.append([line])
            else:
                blocks[-1].append(line)
        return blocks

    def _manifests(self) -> list[str]:
        return [block[0] for block in self._blocks()]

    def test_a_miss_writes_the_unmeasured_manifest_for_the_measuring_start_then_the_measured_one(self) -> None:
        done = self._launch()
        self.assertEqual(done.returncode, 0, done.stderr)
        written = self._manifests()
        self.assertEqual([w.split()[0] for w in written], ["unmeasured", "measured"], written)
        # the measuring start saw the unmeasured manifest, already on disk, under the launch's own stamp
        measure = self.engine_calls.read_text()
        measuring_manifests = sorted(self.run_dir.glob("manifest-measuring-1105-000000-*.json"))
        self.assertEqual(len(measuring_manifests), 1)
        self.assertIn("--manifest " + str(measuring_manifests[0]), measure)
        self.assertIn('"plan_state": "unmeasured"', measure)
        # the real start's manifest is the measured one, written after the plan was measured, never before
        (serving,) = sorted(self.run_dir.glob("manifest-1105-000000-*.json"))
        self.assertEqual(json.loads(serving.read_text())["plan_state"], "measured")

    def test_a_hit_writes_only_the_measured_manifest(self) -> None:
        self._launch()
        before = len(self._manifests())
        again = self._launch()
        self.assertEqual(again.returncode, 0, again.stderr)
        self.assertEqual([w.split()[0] for w in self._manifests()[before:]], ["measured"])
        self.assertEqual(len(list(self.run_dir.glob("manifest-1105-000001-*.json"))), 1)
        self.assertEqual(list(self.run_dir.glob("manifest-measuring-1105-000001-*.json")), [])

    def test_a_live_policy_and_a_lab_experiment_reach_both_manifests(self) -> None:
        self._launch(LIVE_POLICY_FILE="/p/policy.json", EXPERIMENT_FILE="/p/exp.json")
        for line in self._manifests():
            self.assertIn("--live-policy=/p/policy.json", line)
            self.assertIn("--experiment=/p/exp.json", line)

    def test_a_manifest_that_cannot_be_written_stops_the_start(self) -> None:
        self.serve.write_text(FAKE_SERVE.replace('printf \'{"plan_state": "%s", "port": %s}\\n\' "$state" "${port:-0}" > "$out"', 'echo "bc-serve: cannot write" >&2; exit 2'))
        done = self._launch()
        self.assertNotEqual(done.returncode, 0)
        self.assertNotIn("MANIFEST:", done.stdout)
        self.assertIn("manifest was not written", done.stderr)

    def _engine_only(self, **overrides: str) -> subprocess.CompletedProcess[str]:
        """Describe a helper-free layout: LAYOUT_HELPER_COUNT=0 and its resolved server-args, no heat file, no hand-off settings."""
        values = dict(LAYOUT="qwen27b-2xtx", PORT="36080", MODEL="/model.rad", LAYOUT_HELPER_COUNT="0", LAYOUT_ENGINE_RANKS="2")
        values.update(overrides)
        server_args = values.pop("SERVER_ARGS", "")
        assigns = "\n".join(f"{k}={shlex.quote(v)}" for k, v in values.items())
        script = f"""
set -u
RUN={shlex.quote(str(self.run_dir))}; RAD_HOME={shlex.quote(str(self.home))}; RAD_BIN=/engine; BUDGETS=()
{assigns}
LAYOUT_SERVER_ARGS=({server_args})
source {shlex.quote(str(LIB))}
plan_check_describe /site.toml && printf '%s\\n' "${{PLAN_SET[@]}}" "${{PLAN_EXTRA[@]}}" "${{PLAN_SITE[@]}}"
"""
        return _run(script, dict(os.environ), REPO)

    def test_a_helper_free_layout_is_described_by_its_layout_and_its_options_alone(self) -> None:
        done = self._engine_only(SERVER_ARGS="--max-model-len 8192 --max-num-seqs 6 --max-num-batched-tokens 2048 "
                                             "--kv-cache-dtype fp8 --num-speculative-tokens 3 --checkpoint-interval 2048")
        self.assertEqual(done.returncode, 0, done.stderr)
        sets = done.stdout.splitlines()
        for want in ('layout.name="qwen27b-2xtx"', "engine.port=36080", "service.max_context=8192", "service.max_sessions=6",
                     "engine.max_num_batched_tokens=2048", 'engine.kv_cache_dtype="fp8"', "engine.num_speculative_tokens=3",
                     "engine.checkpoint_interval=2048", 'tuning.p2p="off"'):
            self.assertIn(want, sets)
        self.assertFalse([x for x in sets if x.startswith("handoff.")], sets)   # no hand-off
        for want in ("memory.host_pool_mib=0", "memory.host_kv_mib=0", 'engine.ngram_placement="disk"'):   # the engine's own defaults
            self.assertIn(want, sets)
        self.assertFalse([x for x in sets if "heat-sha256" in x], sets)

    def test_one_engine_card_needs_no_p2p_setting_and_an_unknown_option_is_refused(self) -> None:
        done = self._engine_only(LAYOUT_ENGINE_RANKS="1")
        self.assertEqual(done.returncode, 0, done.stderr)
        self.assertNotIn('tuning.p2p="off"', done.stdout.splitlines())
        bad = self._engine_only(SERVER_ARGS="--tp 4")
        self.assertNotEqual(bad.returncode, 0)
        self.assertIn("not a manifest setting", bad.stderr)

    def test_a_tensor_parallel_engine_with_helpers_exchanges_through_host_memory(self) -> None:
        """RAD49: a helper layout with two engine ranks asks bc-serve for the host-memory TP path, as a helper-free one does."""
        for ranks, p2p in (("2", True), ("1", False)):
            with self.subTest(ranks=ranks):
                heat = Path(self.temp.name) / "heat"
                heat.write_bytes(b"\0")
                script = f"""
set -u
RUN={shlex.quote(str(self.run_dir))}; RAD_HOME={shlex.quote(str(self.home))}; RAD_BIN=/engine; BUDGETS=(); WORK=""
LAYOUT=t; LAYOUT_HELPER_COUNT=1; LAYOUT_ENGINE_RANKS={ranks}; SEQS=6; MAX_LEN=65536; HOST_POOL_MIB=9216; HOST_KV_MIB=12288
CHUNK=2048; NGRAM=ram; KV_DTYPE=fp8; ROPE=none; PORT=36080; CKPT_INTERVAL=4096; NUM_SPEC=3; SHM=/s; RECORD_PERIOD=1
PREFIX_RESUME=nearest; HEAT={shlex.quote(str(heat))}; EXPERIMENT_FILE=""; LIVE_POLICY_FILE=""; CALIBRATE=0
source {shlex.quote(str(LIB))}
plan_check_describe /site.toml && printf '%s\\n' "${{PLAN_SET[@]}}"
"""
                done = _run(script, dict(os.environ), REPO)
                self.assertEqual(done.returncode, 0, done.stderr)
                sets = done.stdout.splitlines()
                self.assertIn('handoff.shm="/s"', sets)
                self.assertEqual('tuning.p2p="off"' in sets, p2p, sets)

    def test_the_launcher_script_parses(self) -> None:
        for path in (LAUNCHER, LIB):
            done = subprocess.run(["bash", "-n", str(path)], capture_output=True, text=True)
            self.assertEqual(done.returncode, 0, f"{path.name}: {done.stderr}")


if __name__ == "__main__":
    unittest.main()
