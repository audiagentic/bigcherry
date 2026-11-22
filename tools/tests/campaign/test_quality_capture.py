"""MEN11: quality measurement inside the server comparison (KL per arm, sidecars up, gate against the control)."""
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from dataclasses import replace
from unittest.mock import patch

from bigcherry.campaign import quality_capture as qc
from bigcherry.engine_adapters import SidecarSpec, adapter
from bigcherry.engine_adapters.base import QualityCommand
from bigcherry.engine_adapters.radiance import RadianceAdapter

THRESHOLDS = {
    "max_kld_mean_increase": 0.002, "max_kld_median_increase": 0.001, "max_kld_p99_increase": 0.02,
    "max_kld_p999_increase": 0.1, "max_top1_drop_points": 0.5, "max_ppl_ratio_increase": 0.005,
}
_PROGRAM = (
    "import json,sys;out=sys.argv[1];mean=float(sys.argv[2]);"
    "json.dump({'all':{'kld':{'mean':mean,'median':mean/2,'p99':mean*8,'p99_9':mean*30},'top1_agreement':0.968,"
    "'ppl':{'candidate':8.55,'reference':8.56,'ratio':0.9988}},'by_source':{'docs':{'kld':{'mean':mean}}}},open(out,'w'))"
)


class FakeAdapter(RadianceAdapter):
    """radiance's sidecars (none here) with a quality command that runs a tiny stand-in for the KL run."""

    def quality_command(self, kind, arm, env, run_dir, binary, model, quality):
        mean = {"reference": "0", "control": "0.0035", "good": "0.0036", "bad": "0.5"}[arm["name"]]
        result = run_dir / f"{arm['name']}.kld.json"
        return QualityCommand((sys.executable, "-c", _PROGRAM, str(result), mean), dict(env),
                              run_dir / f"{arm['name']}.score.log", result, 60.0)


class QualityBlockTests(unittest.TestCase):
    def block(self, **updates):
        return {"reference_dir": "ref", "args": ["--tp", "1"], "thresholds": dict(THRESHOLDS), **updates}

    def test_valid_block_defaults_control_to_first_arm(self):
        self.assertEqual(qc.validate_quality_block(self.block(), ["a", "b"])["control"], "a")

    def test_rejects_bad_blocks(self):
        for bad in (self.block(extra=1), {**self.block(), "thresholds": {"max_kld_mean_increase": 1}},
                    self.block(control="zzz"), self.block(corpus="c.jsonl"), self.block(args="x"),
                    {k: v for k, v in self.block().items() if k != "args"}):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                qc.validate_quality_block(bad, ["a", "b"])

    def test_reference_state_never_overwrites_and_reuse_needs_one(self):
        with tempfile.TemporaryDirectory() as tmp:
            ref = Path(tmp) / "ref"
            reuse = qc.validate_quality_block(self.block(reference_dir=str(ref)), ["a", "b"])
            with self.assertRaisesRegex(ValueError, "missing or empty"):
                qc.check_reference_state(reuse)
            ref.mkdir()
            (ref / "logits.bin").write_bytes(b"x")
            qc.check_reference_state(reuse)
            yaml = Path(tmp) / "r.yaml"
            yaml.write_text("cards: [0]\n", encoding="utf-8")
            record = qc.validate_quality_block(self.block(
                reference_dir=str(ref), corpus="c",
                record={"serving": {"config": "x", "native_config": str(yaml)}}), ["a", "b"])
            with self.assertRaisesRegex(ValueError, "already holds"):
                qc.check_reference_state(record)


class RadianceQualityCommandTests(unittest.TestCase):
    quality = {"reference_dir": "/ref", "corpus": "/c.jsonl", "args": ["--tp", "1"]}

    def test_native_arm_is_scored_by_the_owner_from_its_own_yaml(self):
        from types import SimpleNamespace
        arm = {"name": "attn-native", "model": Path("/m.rad"), "settings": {},
               "serving": {"config": "qwen27b-2xtx", "native_config": "/arms/attn-native.yaml"}}
        quality = {"reference_dir": "/ref", "args": [], "timeout_s": 60}
        rad = adapter("radiance")
        with patch("bigcherry.engine_adapters.radiance._host", return_value=SimpleNamespace(model_root="/models")), \
                patch.dict(os.environ, {"BC_MODEL_ROOT": ""}):
            command = rad.quality_command("score", arm, {}, Path("/run"), Path("/b/bin/radiance"), Path("/m.rad"), quality)
            self.assertEqual(command.argv, (
                "/b/radiance_home/bin/bc-serve", "quality", "qwen27b-2xtx", "--native-config", "/arms/attn-native.yaml",
                "--model-root", "/models", "--engine-binary", "/b/bin/radiance", "--run-dir", "/run/attn-native.owner",
                "--kld-ref", "/ref", "--kld-out", "/run/attn-native.kld.json"))
            with self.assertRaisesRegex(ValueError, "quality.args must be empty"):
                rad.quality_command("score", arm, {}, Path("/run"), Path("/b/bin/radiance"), Path("/m.rad"),
                                    {**quality, "args": ["--tp", "2"]})

    def test_record_and_score_invocations_are_the_owners(self):
        from types import SimpleNamespace
        rad = adapter("radiance")
        arm = {"name": "reference", "model": Path("/m.rad"), "settings": {},
               "serving": {"config": "flashnext-tp2xtx", "native_config": "/arms/reference.yaml"}}
        with patch("bigcherry.engine_adapters.radiance._host", return_value=SimpleNamespace(model_root="/models")), \
                patch.dict(os.environ, {"BC_MODEL_ROOT": ""}):
            record = rad.quality_command("record", arm, {"A": "1"}, Path("/run"), Path("/b/bin/radiance"), Path("/m.rad"),
                                         {**self.quality, "args": []})
            self.assertEqual(record.argv, (
                "/b/radiance_home/bin/bc-serve", "quality", "flashnext-tp2xtx", "--native-config", "/arms/reference.yaml",
                "--model-root", "/models", "--engine-binary", "/b/bin/radiance", "--run-dir", "/run/reference.owner",
                "--kld-record", "/ref", "--kld-corpus", "/c.jsonl"))
            self.assertIsNone(record.result_path)
            self.assertEqual(record.env, {"A": "1"})
            with self.assertRaisesRegex(ValueError, "needs quality.corpus"):
                rad.quality_command("record", arm, {}, Path("/run"), Path("/b/bin/radiance"), Path("/m.rad"),
                                    {"reference_dir": "/ref", "args": []})
            with self.assertRaises(ValueError):
                rad.quality_command("other", arm, {}, Path("/run"), Path("/b"), Path("/m"), {**self.quality, "args": []})

    def test_a_radiance_arm_without_a_native_yaml_has_no_quality_run(self):
        with self.assertRaisesRegex(ValueError, "native arm"):
            adapter("radiance").quality_command("score", {"name": "raw", "model": Path("/m.rad"), "settings": {}}, {},
                                                Path("run"), Path("/b/bin/radiance"), Path("/m.rad"), {**self.quality, "args": []})

    def test_the_record_block_is_a_native_arm(self):
        base = {"reference_dir": "ref", "args": [], "thresholds": THRESHOLDS, "corpus": "c.jsonl"}
        with tempfile.TemporaryDirectory() as tmp:
            yaml = Path(tmp) / "reference.yaml"
            yaml.write_text("cards: [0]\n", encoding="utf-8")
            good = {**base, "record": {"serving": {"config": "flashnext-tp2xtx", "native_config": str(yaml)}}}
            self.assertEqual(qc.validate_quality_block(good, ["a", "b"])["record"]["serving"]["native_config"], str(yaml))
            for bad in ({**base, "record": {"args": [], "environment": {}}},
                        {**base, "record": {"serving": {"config": "x"}}},
                        {**base, "record": {"serving": {"config": "x", "native_config": str(Path(tmp) / "none.yaml")}}},
                        {**base, "record": {"serving": {"config": "x", "native_config": str(yaml), "set": {}}}}):
                with self.subTest(bad=bad), self.assertRaises(ValueError):
                    qc.validate_quality_block(bad, ["a", "b"])

    def test_llamacpp_has_no_quality_mode(self):
        self.assertIsNone(adapter("llamacpp").quality_command("score", {}, {}, Path("."), Path("b"), Path("m"), {}))


class RunQualityTests(unittest.TestCase):
    def run_arms(self, names):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            quality = qc.validate_quality_block({"reference_dir": "ref", "args": [], "thresholds": THRESHOLDS}, names)
            prepared = {n: {"binary": Path("b/bin/radiance"), "model": Path("m"), "env": {}, "settings": {}, "manifest": None} for n in names}
            section, cells = qc.run_quality_capture(
                adapter=FakeAdapter(), quality=quality, prepared=prepared, names=names, output=out, reference_env={})
            persisted = json.loads(json.dumps(section))
        return persisted, cells

    def test_recording_creates_the_reference_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            ref = Path(tmp) / "new" / "ref"
            names = ["control", "good"]
            (Path(tmp) / "r.yaml").write_text("cards: [0]\n", encoding="utf-8")
            quality = qc.validate_quality_block({"reference_dir": str(ref), "args": [], "thresholds": THRESHOLDS,
                                                 "corpus": "c.jsonl", "record": {"serving": {"config": "x", "native_config": str(Path(tmp) / "r.yaml")}}}, names)
            prepared = {n: {"binary": Path("b/bin/radiance"), "model": Path("m"), "env": {}, "settings": {}, "manifest": None} for n in names}
            qc.run_quality_capture(adapter=FakeAdapter(), quality=quality, prepared=prepared, names=names,
                                   output=Path(tmp) / "out", reference_env={})
            self.assertTrue(ref.is_dir())

    def test_neutral_arm_passes_and_is_recorded(self):
        section, cells = self.run_arms(["control", "good"])
        self.assertTrue(section["passed"], section)
        self.assertEqual(section["arms"]["good"]["gate"], {"passed": True, "problems": []})
        self.assertEqual(section["arms"]["good"]["by_source"], {"docs": {"kld": {"mean": 0.0036}}})
        self.assertEqual([c["mode"] for c in cells], ["control", "good"])

    def test_degraded_arm_fails_the_gate(self):
        section, _ = self.run_arms(["control", "bad"])
        self.assertFalse(section["passed"])
        self.assertFalse(section["arms"]["bad"]["gate"]["passed"])
        self.assertTrue(section["arms"]["bad"]["gate"]["problems"])


class SidecarExitQualityTests(unittest.TestCase):
    def capture(self, *, primary_fails=False, exited=True, killed=False, real_sidecar=False):
        class ScoringAdapter(FakeAdapter):
            def quality_command(self, kind, arm, env, run_dir, binary, model, quality):
                command = super().quality_command(kind, arm, env, run_dir, binary, model, quality)
                if primary_fails:
                    # Keep the real sidecar alive long enough to serve the
                    # primary, then let it exit before the primary fails.
                    script = ("import time,sys;time.sleep(0.35);sys.exit(7)"
                              if real_sidecar else "import sys;sys.exit(7)")
                    return replace(command, argv=(sys.executable, "-c", script))
                return command

            def sidecars(self, arm, env, run_dir, binary):
                if not real_sidecar:
                    return []
                program = "import time,sys;print('ready',flush=True);time.sleep(0.03);sys.exit(3)"
                return [SidecarSpec(
                    name="helper0", argv=(sys.executable, "-c", program), env={},
                    ready_pattern=r"^ready$", ready_timeout_s=4.0,
                    log_path=run_dir / "helper0.log", stop_timeout_s=1.0,
                )]

        class FakeSet:
            def __init__(self, specs):
                pass

            def start(self):
                pass

            def stop(self):
                return [{
                    "name": "helper0", "argv": ["fake"], "log": "helper0.log",
                    "ready_s": 0.0, "returncode": 3 if exited else 0,
                    "exited_before_stop": exited, "killed": killed,
                }]

        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)
            quality = {"reference_dir": "ref", "control": "control", "gate": "baselines", "args": []}
            prepared = {"control": {
                "binary": Path("b/bin/radiance"), "model": Path("m"), "env": {}, "settings": {}, "manifest": None,
            }}
            def run():
                return qc.run_quality_capture(
                    adapter=ScoringAdapter(), quality=quality, prepared=prepared,
                    names=["control"], output=output, reference_env={},
                )
            if real_sidecar:
                section, cells = run()
            else:
                with patch.object(qc, "SidecarSet", FakeSet):
                    section, cells = run()
            return section, cells

    def test_self_exit_after_zero_returncode_passes_and_retains_result(self):
        section, cells = self.capture()
        self.assertTrue(section["passed"], section)
        arm = section["arms"]["control"]
        self.assertEqual(arm["returncode"], 0)
        self.assertIn("result", arm)
        self.assertNotIn("error", arm)
        self.assertTrue(arm["sidecars"][0]["exited_before_stop"])
        self.assertEqual(cells[0]["returncode"], 0)

    def test_self_exit_with_nonzero_primary_returncode_fails(self):
        section, cells = self.capture(primary_fails=True)
        self.assertFalse(section["passed"])
        arm = section["arms"]["control"]
        self.assertEqual(arm["returncode"], 7)
        self.assertIn("sidecar(s) helper0", arm["error"])
        self.assertEqual(cells[0]["returncode"], 1)

    def test_sigkill_always_fails_even_when_primary_succeeds(self):
        for primary_fails in (False, True):
            with self.subTest(primary_fails=primary_fails):
                section, cells = self.capture(
                    primary_fails=primary_fails, exited=False, killed=True)
                self.assertFalse(section["passed"])
                self.assertIn("sidecar(s) helper0", section["arms"]["control"]["error"])
                self.assertEqual(cells[0]["returncode"], 1)

    def test_real_sidecar_exits_during_failed_primary(self):
        section, cells = self.capture(primary_fails=True, real_sidecar=True)
        self.assertFalse(section["passed"])
        arm = section["arms"]["control"]
        self.assertEqual(arm["returncode"], 7)
        self.assertTrue(arm["sidecars"][0]["exited_before_stop"])
        self.assertEqual(arm["sidecars"][0]["returncode"], 3)
        self.assertIn("sidecar(s) helper0", arm["error"])
        self.assertEqual(cells[0]["returncode"], 1)


if __name__ == "__main__":
    unittest.main()
