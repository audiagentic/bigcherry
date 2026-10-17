"""MEN15: the RAM-only CPU smoke -- adapter specs, [smoke.cpu] declarations, the generic runner, the CLI, and pin-bump."""

from __future__ import annotations

import contextlib
import hashlib
import io
import json
import os
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from bigcherry.campaign import cpu_smoke
from bigcherry.cli import engine_scope
from bigcherry.cli.main import build_parser
from bigcherry.core import engines, paths
from bigcherry.core.context import ProjectContext
from bigcherry.engine_adapters import CompletionApi, CpuSmokeSpec, DeviceAssertion, adapter
from bigcherry.engine_adapters.base import GPU_HIDING_ENV
from bigcherry.release import pin_bump

FAKE_SERVER = Path(__file__).with_name("fake_cpu_smoke_server.py")
SERVE = engines.ServeSpec(
    binary="bin/fake", model_flag="--model", host_flag="--host", port_flag="--port", health="/health",
    shutdown="http:/shutdown",
)
LLAMA_STYLE = CompletionApi(
    path="/completion", prompt_field="prompt", token_field="n_predict",
    fixed=(("temperature", 0), ("seed", 1)), text_path=("content",),
)
OPENAI_STYLE = CompletionApi(
    path="/v1/completions", prompt_field="prompt", token_field="max_tokens",
    fixed=(("temperature", 0), ("seed", 1)), text_path=("choices", 0, "text"), models_path="/v1/models",
)
GPU_LINE = DeviceAssertion((r"ggml_cuda_init: found [1-9]\d* ROCm devices?",))


def _write_model(directory: Path) -> tuple[Path, str]:
    model = directory / "tiny.bin"
    model.write_bytes(b"tiny model bytes")
    return model, hashlib.sha256(model.read_bytes()).hexdigest()


def _fake_spec(directory: Path, *extra_args: str, completion: CompletionApi = LLAMA_STYLE, **overrides) -> CpuSmokeSpec:
    model, digest = _write_model(directory)
    spec = CpuSmokeSpec(
        engine="fake", binary=FAKE_SERVER, serve=SERVE, args=tuple(extra_args),
        env_overrides={name: "" for name in GPU_HIDING_ENV}, model=model, model_sha256=digest,
        completion=completion, device=GPU_LINE, command_prefix=(sys.executable,), health_timeout_s=60,
        request_timeout_s=30,
    )
    return replace(spec, **overrides)


class AdapterSpecTests(unittest.TestCase):
    def _spec(self, engine: str) -> CpuSmokeSpec:
        spec = adapter(engine).cpu_smoke(ProjectContext.resolve(engine=engine), Path("/b"), Path("/models"))
        self.assertIsNotNone(spec)
        return spec

    def test_every_gpu_selection_variable_is_empty_for_every_engine(self):
        for engine in engines.names(paths.REPO_ROOT):
            with self.subTest(engine=engine):
                spec = self._spec(engine)
                for name in ("HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES"):
                    self.assertEqual(spec.env_overrides[name], "")
                self.assertEqual([n for n in GPU_HIDING_ENV if spec.env_overrides[n] != ""], [])
                self.assertEqual(spec.engine, engine)

    def test_binary_and_model_come_from_the_declaration(self):
        for engine in engines.names(paths.REPO_ROOT):
            with self.subTest(engine=engine):
                layout = engines.load(paths.REPO_ROOT, engine)
                spec = self._spec(engine)
                self.assertEqual(spec.binary, Path("/b") / layout.serve.binary)
                self.assertEqual(spec.model, Path("/models") / layout.smoke_cpu.model)
                self.assertEqual(spec.model_sha256, layout.smoke_cpu.sha256)
                self.assertEqual(spec.args[-len(layout.smoke_cpu.args):], layout.smoke_cpu.args)

    def test_llamacpp_offloads_nothing(self):
        args = self._spec("llamacpp").args
        self.assertEqual(args[args.index("-ngl") + 1], "0")
        self.assertEqual(args[args.index("--device") + 1], "none")

    def test_radiance_uses_cpu_kernels_on_one_rank(self):
        args = self._spec("radiance").args
        self.assertEqual(args[args.index("--kernels") + 1], "libavx,libref")
        self.assertEqual(args[args.index("--tp") + 1], "1")
        self.assertEqual(self._spec("radiance").completion.models_path, "/v1/models")

    def test_repository_models_are_pinned_and_a_missing_file_is_refused(self):
        for engine in engines.names(paths.REPO_ROOT):
            with self.subTest(engine=engine), tempfile.TemporaryDirectory() as td:
                spec = replace(self._spec(engine), model=Path(td) / "x")
                self.assertRegex(spec.model_sha256, r"^[0-9a-f]{64}$")
                with self.assertRaisesRegex(cpu_smoke.CpuSmokeError, "not found"):
                    cpu_smoke.verify_model(spec)

    def test_engine_without_a_declaration_has_no_smoke(self):
        context = ProjectContext.resolve(engine="llamacpp")
        layout = replace(context.layout, smoke_cpu=None)
        with patch.object(ProjectContext, "layout", property(lambda self: layout)):
            self.assertIsNone(adapter("llamacpp").cpu_smoke(context, Path("/b"), Path("/m")))


class DeclarationTests(unittest.TestCase):
    _BASE = (
        'schema = 1\nname = "alpha"\nupstream = "https://example.invalid/alpha"\n\n[layout]\n'
        'patches = "engines/alpha/patches"\npatch-id-prefix = ""\noverlay = "engines/alpha/overlay"\n'
        'vendor = "vendor/alpha"\nrecipes = "vendor/alpha.recipes.toml"\n\n[release]\nversion-source = "build-tag"\n\n'
        '[build]\ntargets-option = "AMDGPU_TARGETS"\nbinary = "bin/server"\n\n[serve]\nbinary = "bin/server"\n'
        'model-flag = "--model"\nhost-flag = "--host"\nport-flag = "--port"\nhealth = "/health"\nshutdown = "sigint"\n\n'
    )

    def _load(self, smoke: str):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            path = engines.declaration_path(root, "alpha")
            path.parent.mkdir(parents=True)
            path.write_text(self._BASE + smoke, encoding="utf-8")
            return engines.load(root, "alpha")

    def test_declaration_is_loaded(self):
        layout = self._load('[smoke.cpu]\nmodel = "smoke/t.gguf"\nsha256 = "ab"\nargs = ["-c", "512"]\n')
        self.assertEqual(layout.smoke_cpu, engines.CpuSmokeDecl("smoke/t.gguf", "ab", ("-c", "512")))

    def test_no_smoke_table_is_allowed(self):
        self.assertIsNone(self._load("").smoke_cpu)

    def test_missing_sha256_is_refused(self):
        with self.assertRaisesRegex(engines.EngineError, "smoke.cpu.sha256"):
            self._load('[smoke.cpu]\nmodel = "smoke/t.gguf"\n')

    def test_missing_model_unknown_key_and_bad_args_are_refused(self):
        for body, message in (
            ('[smoke.cpu]\nsha256 = "ab"\n', "smoke.cpu.model"),
            ('[smoke.cpu]\nmodel = "m"\nsha256 = "ab"\nport = 1\n', "unknown smoke.cpu key"),
            ('[smoke.cpu]\nmodel = "m"\nsha256 = "ab"\nargs = [1]\n', "list of strings"),
            ('[smoke.cpu]\nmodel = "/abs/m"\nsha256 = "ab"\n', "repository-relative"),
            ('[smoke.gpu]\nmodel = "m"\n', "only hold a"),
        ):
            with self.subTest(body=body), self.assertRaisesRegex(engines.EngineError, message):
                self._load(body)


class RunnerTests(unittest.TestCase):
    def _run(self, spec: CpuSmokeSpec, out: Path) -> dict:
        return cpu_smoke.run(spec, out, None)

    def test_deterministic_server_passes_and_the_result_is_written(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            result = self._run(_fake_spec(root), root / "out")
            on_disk = json.loads((root / "out" / cpu_smoke.RESULT_NAME).read_text(encoding="utf-8"))
            self.assertTrue(result["ok"])
            self.assertEqual(on_disk["ok"], True)
            self.assertEqual(len(result["completions"]), 3)
            self.assertTrue(all(c["first"] and c["first"] == c["second"] for c in result["completions"]))
            self.assertTrue(result["shutdown"]["clean"])
            self.assertEqual(result["gpu_log_hits"], [])
            self.assertIn("wall_s", result)
            self.assertIn("startup_s", result)
            if sys.platform.startswith("linux"):
                self.assertGreater(result["peak_rss_bytes"], 0)
            else:
                self.assertIsNone(result["peak_rss_bytes"])
                self.assertIn("not measured", result["peak_rss_note"])

    def test_gpu_selection_variables_reach_the_server_empty(self):
        with tempfile.TemporaryDirectory() as td, patch.dict(os.environ, {"HIP_VISIBLE_DEVICES": "0,1"}):
            root = Path(td)
            self._run(_fake_spec(root), root / "out")
            log = (root / "out" / "server.log").read_text(encoding="utf-8")
            self.assertIn("env HIP_VISIBLE_DEVICES=''", log)
            self.assertIn("env ROCR_VISIBLE_DEVICES=''", log)

    def test_ambient_engine_variables_are_unset_by_prefix(self):
        with tempfile.TemporaryDirectory() as td, patch.dict(os.environ, {"GGML_FOO": "1"}):
            root = Path(td)
            self._run(_fake_spec(root, env_unset_prefixes=("GGML_",)), root / "out")
            self.assertIn("env GGML_FOO=None", (root / "out" / "server.log").read_text(encoding="utf-8"))

    def test_nondeterministic_server_fails(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            with self.assertRaises(cpu_smoke.CpuSmokeError) as raised:
                self._run(_fake_spec(root, "--nondeterministic"), root / "out")
            self.assertIn("differs between the two passes", str(raised.exception))
            on_disk = json.loads((root / "out" / cpu_smoke.RESULT_NAME).read_text(encoding="utf-8"))
            self.assertFalse(on_disk["ok"])
            self.assertTrue(on_disk["shutdown"]["clean"])  # the server is still stopped

    def test_empty_completion_fails(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            with self.assertRaisesRegex(cpu_smoke.CpuSmokeError, "empty completion"):
                self._run(_fake_spec(root, "--empty"), root / "out")

    def test_gpu_device_init_in_the_log_fails(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            with self.assertRaisesRegex(cpu_smoke.CpuSmokeError, "GPU being opened"):
                self._run(_fake_spec(root, "--gpu-log"), root / "out")

    def test_openai_style_server_uses_the_served_model_id(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            result = self._run(_fake_spec(root, completion=OPENAI_STYLE), root / "out")
            self.assertTrue(result["ok"])

    def test_sha256_mismatch_is_refused_before_any_server_starts(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            spec = replace(_fake_spec(root), model_sha256="0" * 64)
            with self.assertRaisesRegex(cpu_smoke.CpuSmokeError, "has sha256"):
                self._run(spec, root / "out")
            self.assertFalse((root / "out" / "server.log").exists())

    def test_missing_model_and_binary_are_refused(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            with self.assertRaisesRegex(cpu_smoke.CpuSmokeError, "model not found"):
                self._run(replace(_fake_spec(root), model=root / "absent"), root / "out")
            with self.assertRaisesRegex(cpu_smoke.CpuSmokeError, "binary not found"):
                self._run(replace(_fake_spec(root), binary=root / "absent"), root / "out")

    def test_a_visible_gpu_in_the_spec_environment_is_refused(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            spec = _fake_spec(root)
            spec = replace(spec, env_overrides={**spec.env_overrides, "HIP_VISIBLE_DEVICES": "0"})
            with self.assertRaisesRegex(cpu_smoke.CpuSmokeError, "HIP_VISIBLE_DEVICES must be set to the empty string"):
                self._run(spec, root / "out")


class CliTests(unittest.TestCase):
    def _args(self, argv):
        parser = build_parser()
        args = parser.parse_args(argv)
        engine_scope.resolve(args, parser)
        return args

    def test_scope_is_engine_and_the_flag_is_required(self):
        self.assertEqual(engine_scope.scope("smoke-cpu"), engine_scope.ENGINE)
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            self._args(["smoke-cpu", "--build", "b"])
        self.assertEqual(self._args(["--engine", "radiance", "smoke-cpu", "--build", "b"]).engine, "radiance")

    def test_build_is_required(self):
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            self._args(["--engine", "radiance", "smoke-cpu"])

    def test_failure_exits_non_zero_with_a_clear_message(self):
        err = io.StringIO()
        with tempfile.TemporaryDirectory() as td, contextlib.redirect_stderr(err):
            args = self._args(["--engine", "llamacpp", "smoke-cpu", "--build", "b", "--model-root", "m", "--out", td])
            code = args.func(args)
        self.assertEqual(code, 1)
        self.assertIn("smoke-cpu: FAIL (llamacpp)", err.getvalue())
        self.assertIn("smoke model not found", err.getvalue())

    def test_success_exits_zero(self):
        args = self._args(["--engine", "radiance", "smoke-cpu", "--build", "b", "--model-root", "m", "--out", "o"])
        passed = {"wall_s": 1.5, "startup_s": 1.0, "peak_rss_bytes": 5 * 1048576, "peak_rss_note": "VmHWM"}
        out = io.StringIO()
        with patch.object(cpu_smoke, "run_for_engine", return_value=passed) as run, contextlib.redirect_stdout(out):
            code = args.func(args)
        self.assertEqual(code, 0)
        run.assert_called_once_with("radiance", Path("b"), "m", Path("o"), None)
        self.assertIn("smoke-cpu: PASS (radiance)", out.getvalue())

    def test_pin_bump_takes_the_smoke_flags(self):
        args = self._args(["--engine", "llamacpp", "pin-bump", "b1", "--cpu-smoke-build", "bd", "--skip-cpu-smoke"])
        self.assertEqual((args.cpu_smoke_build, args.skip_cpu_smoke), ("bd", True))

    def test_pin_bump_takes_the_lane_build_profile(self):
        args = self._args(["--engine", "radiance", "pin-bump", "b1", "--build", "stock", "--platform", "p"])
        self.assertEqual((args.build_profile, args.platform, args.cpu_smoke_build), ("stock", "p", None))


class PinBumpPhaseTests(unittest.TestCase):
    def test_cpu_smoke_runs_after_compile_and_before_any_later_phase(self):
        phases = pin_bump.PHASES
        self.assertLess(phases.index("compile"), phases.index("lane-build"))
        self.assertLess(phases.index("lane-build"), phases.index("cpu-smoke"))
        self.assertLess(phases.index("cpu-smoke"), phases.index("reaudit"))
        self.assertLess(phases.index("cpu-smoke"), phases.index("complete"))

    def test_missing_build_stops_and_names_the_explicit_skip(self):
        with self.assertRaises(pin_bump.PinBumpStop) as stopped:
            pin_bump.run_phase_cpu_smoke(engine="llamacpp", build_dir=None, out_dir=Path("o"), target_ref="b1")
        exc = stopped.exception
        self.assertEqual((exc.phase, exc.code), ("cpu-smoke", "CPU_SMOKE_BUILD_MISSING"))
        self.assertTrue(any("--skip-cpu-smoke" in action for action in exc.recommended_actions))

    def test_failed_smoke_stops_with_its_failures_in_the_envelope(self):
        failure = cpu_smoke.CpuSmokeError(["greedy completion differs"])
        with patch.object(cpu_smoke, "run_for_engine", side_effect=failure):
            with self.assertRaises(pin_bump.PinBumpStop) as stopped:
                pin_bump.run_phase_cpu_smoke(engine="radiance", build_dir=Path("b"), out_dir=Path("o"), target_ref="b1")
        exc = stopped.exception
        self.assertEqual(exc.code, "CPU_SMOKE_FAILED")
        envelope = pin_bump.failure_envelope("r", {}, "c", {}, exc)
        self.assertEqual(envelope["phase"], "cpu-smoke")
        self.assertEqual(envelope["failure"]["evidence"]["failures"], ["greedy completion differs"])

    def test_passing_smoke_returns_quietly(self):
        with patch.object(cpu_smoke, "run_for_engine", return_value={"ok": True}) as run:
            pin_bump.run_phase_cpu_smoke(engine="radiance", build_dir=Path("b"), out_dir=Path("o"), target_ref="b1")
        run.assert_called_once_with("radiance", Path("b"), None, Path("o"), None)

    def test_skip_is_recorded_in_the_saved_state(self):
        state = pin_bump.PinBumpState(
            schema_version=3, run_id="r", from_ref="a", from_sha="1" * 40, to_ref="b", to_sha="2" * 40,
            transition_commit="3" * 40, tree_name="local", tree_path="/t", cpu_smoke_status="skipped",
        )
        with tempfile.TemporaryDirectory() as td:
            state.save(Path(td))
            document = json.loads((Path(td) / "state.json").read_text(encoding="utf-8"))
            self.assertEqual(document["cpu_smoke"], {"status": "skipped", "build_override": ""})
            self.assertEqual(pin_bump.PinBumpState.load(Path(td)).cpu_smoke_status, "skipped")


if __name__ == "__main__":
    unittest.main()
