"""RSC11 B: the measured serving manifest of a campaign arm (radiance_plan), against fake bc-serve and engine programs."""
import json
import os
import stat
import subprocess
import tempfile
import textwrap
import unittest
from unittest.mock import patch
from pathlib import Path

from bigcherry.core import engines, paths
from bigcherry.engine_adapters import radiance_plan

SERVE = engines.load(paths.REPO_ROOT, "radiance").serve

# A fake bc-serve: records its arguments; exit 10 until the cache holds the plan (a file named by BC_FAKE_CACHE).
BC_SERVE = textwrap.dedent('''\
    #!/bin/bash
    echo "$@" >> "$FAKE_LOG/bc-serve.args"
    out=$2
    manifest='{"spec": {"processes": {"engine": {"server": {"port": 18431}}, "helpers": []}}}'
    case " $* " in
      *" --unmeasured "*) echo "$manifest" > "$out"; exit 0 ;;
      *" --measured-plan "*) echo "$manifest" > "$out"; touch "$FAKE_LOG/cached"; exit 0 ;;
    esac
    if [ -e "$FAKE_LOG/cached" ]; then echo "$manifest" > "$out"; exit 0; fi
    echo "no measured plan" >&2; exit 10
''')
ENGINE = textwrap.dedent('''\
    #!/bin/bash
    echo "$@" >> "$FAKE_LOG/engine.args"
    [ -z "${FAKE_ENGINE_FAIL:-}" ] || exit 3
    while [ $# -gt 0 ]; do [ "$1" = --plan-memory ] && plan=$2; shift; done
    echo '{"plan_version": 1}' > "$plan"
    echo "radiance: --plan-memory: wrote $plan; engine start-up took 1536 MiB of MemAvailable; nothing loaded" >&2
''')


class PlanTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.log = self.root / "log"
        self.log.mkdir()
        self.binary = self.root / "build" / "bin" / "radiance"
        for path, body in ((self.binary, ENGINE), (self.root / "build/radiance_home/bin/bc-serve", BC_SERVE)):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(body, encoding="utf-8")
            path.chmod(path.stat().st_mode | stat.S_IXUSR)
        try:
            subprocess.run([str(self.binary), "--plan-memory"], capture_output=True, env={"FAKE_LOG": str(self.log)})
        except PermissionError:
            self.skipTest("the temporary directory is mounted noexec; run with TMPDIR on an executable filesystem")
        (self.log / "engine.args").unlink(missing_ok=True)
        self.env = {"FAKE_LOG": str(self.log), "PATH": os.environ["PATH"]}
        patcher = patch.dict(os.environ, {"FAKE_LOG": str(self.log)})   # bc-serve inherits the ambient environment
        patcher.start()
        self.addCleanup(patcher.stop)

    def make(self, serving=None, env=None):
        return radiance_plan.write_manifest(
            name="arm", serving=serving or {"layout": "qwen27b-2xtx", "set": {"service.max_sessions": 6}},
            binary=self.binary, model=Path("/m/model.rad"), env=env or self.env, serve=SERVE,
            run_dir=self.root / "run", plan_cache=self.root / "cache", host="127.0.0.1", helper_specs=lambda manifest: [])

    def test_a_cache_miss_is_measured_once_then_served_from_the_cache(self):
        manifest = self.make()
        self.assertEqual(json.loads(manifest.read_text())["spec"]["processes"]["engine"]["server"]["port"], 18431)
        engine = (self.log / "engine.args").read_text().split()
        self.assertEqual(engine[:4], ["--manifest", str(self.root / "run/manifest-measuring.json"), "--role", "engine"])
        self.assertEqual(engine[4], "--plan-memory")
        calls = (self.log / "bc-serve.args").read_text().splitlines()
        self.assertEqual(len(calls), 3)   # cache miss, unmeasured, measured
        self.assertIn("--set layout.name=\"qwen27b-2xtx\"", calls[0])
        self.assertIn("--set service.max_sessions=6", calls[0])
        self.assertRegex(calls[0], r"--set engine\.port=\d+")
        self.assertIn("--unmeasured", calls[1])
        self.assertIn("--helpers-resident-mib 0 --engine-resident-mib 1536", calls[2])
        (self.root / "run" / "manifest.json").unlink()
        self.make()   # hit: no second measuring start
        self.assertEqual(len((self.log / "engine.args").read_text().splitlines()), 1)

    def test_the_engine_runs_in_the_arms_environment_with_the_runtime_home(self):
        self.env["FAKE_LOG"] = str(self.log)
        (self.binary.parent.parent / "radiance_home" / "x").parent.mkdir(exist_ok=True)
        self.make()
        self.assertTrue((self.log / "engine.args").exists())

    def test_a_failed_measuring_start_is_an_error_naming_the_log(self):
        with self.assertRaisesRegex(radiance_plan.PlanError, r"measuring start failed \(engine exit 3\)"):
            self.make(env={**self.env, "FAKE_ENGINE_FAIL": "1"})
        self.assertFalse((self.log / "cached").exists())

    def test_serving_takes_only_its_keys_and_a_layout(self):
        for serving in ({"set": {}}, {"layout": "x", "flags": []}):
            with self.subTest(serving=serving), self.assertRaisesRegex(radiance_plan.PlanError, "serving takes"):
                self.make(serving)
        with self.assertRaisesRegex(radiance_plan.PlanError, "strings, numbers or booleans"):
            self.make({"layout": "x", "set": {"a": [1]}})

    def test_helpers_resident_memory_counts_anonymous_memory_and_the_segment_once(self):
        status = self.root / "status"
        status.write_text("RssAnon:\t2048 kB\nRssShmem:\t1024 kB\n", encoding="utf-8")
        fake = self.root / "proc"
        for pid in (1, 2):
            (fake / str(pid)).mkdir(parents=True)
            (fake / str(pid) / "status").write_text(status.read_text())
        real = Path.read_text
        with patch.object(Path, "read_text", lambda self_, *a, **k: real(fake / self_.parts[2] / "status", *a, **k)
                          if str(self_).startswith("/proc/") else real(self_, *a, **k)):
            self.assertEqual(radiance_plan.helpers_resident_mib([1, 2]), 5)   # (2048+2048+1024) kB = 5 MiB
