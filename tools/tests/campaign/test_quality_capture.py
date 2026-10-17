"""MEN11: quality measurement inside the server comparison (KL per arm, sidecars up, gate against the control)."""
import json
import sys
import tempfile
import unittest
from pathlib import Path

from bigcherry.campaign import quality_capture as qc
from bigcherry.engine_adapters import adapter
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
        mean = {"control": "0.0035", "good": "0.0036", "bad": "0.5"}[arm["name"]]
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
            record = qc.validate_quality_block(self.block(
                reference_dir=str(ref), corpus="c", record={"args": [], "environment": {}}), ["a", "b"])
            with self.assertRaisesRegex(ValueError, "already holds"):
                qc.check_reference_state(record)


class RadianceQualityCommandTests(unittest.TestCase):
    quality = {"reference_dir": "/ref", "corpus": "/c.jsonl", "args": ["--tp", "1"]}

    def test_record_and_score_invocations(self):
        rad = adapter("radiance")
        arm = {"name": "on", "model": Path("/m.rad"), "settings": {}}
        record = rad.quality_command("record", arm, {"A": "1"}, Path("run"), Path("/b/bin/radiance"), Path("/m.rad"), self.quality)
        self.assertEqual(record.argv, (str(Path("/b/bin/radiance")), "--model", str(Path("/m.rad")), "--tp", "1",
                                       "--kld-record", "/ref", "--kld-corpus", "/c.jsonl"))
        self.assertIsNone(record.result_path)
        score = rad.quality_command("score", arm, {"A": "1"}, Path("run"), Path("/b/bin/radiance"), Path("/m.rad"), self.quality)
        self.assertEqual(score.argv[-4:], ("--kld-ref", "/ref", "--kld-out", str(Path("run/on.kld.json"))))
        self.assertEqual(score.env, {"A": "1"})
        with self.assertRaises(ValueError):
            rad.quality_command("other", arm, {}, Path("run"), Path("/b"), Path("/m"), self.quality)

    def test_llamacpp_has_no_quality_mode(self):
        self.assertIsNone(adapter("llamacpp").quality_command("score", {}, {}, Path("."), Path("b"), Path("m"), {}))


class RunQualityTests(unittest.TestCase):
    def run_arms(self, names):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            quality = qc.validate_quality_block({"reference_dir": "ref", "args": [], "thresholds": THRESHOLDS}, names)
            prepared = {n: {"binary": Path("b/bin/radiance"), "model": Path("m"), "env": {}, "settings": {}} for n in names}
            section, cells = qc.run_quality_capture(
                adapter=FakeAdapter(), quality=quality, prepared=prepared, names=names, output=out, reference_env={})
            persisted = json.loads(json.dumps(section))
        return persisted, cells

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


if __name__ == "__main__":
    unittest.main()
