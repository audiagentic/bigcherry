"""PA39 defect #1 fix: ``--producer-corpus`` must actually reach
``ProducerContext.corpus`` through the real ``--validation-producer`` CLI
dispatch path (``_run_validation_producer()`` in
``bigcherry.patch.validation_campaign``).

Before this fix, ``_run_validation_producer()`` hardcoded
``ProducerContext(..., corpus=None, ...)`` -- there was no CLI flag at all,
so any producer requiring a real corpus (1203's RD05/RD07 backend_reference
checks) always failed closed on real hardware regardless of device
availability. PA37's existing hardware-free tests never caught this because
they construct ``ProducerContext`` directly and never exercise
``_run_validation_producer()``'s own CLI-arg-to-context wiring.

This test drives the REAL ``_run_validation_producer()`` function (not a
reimplementation of its logic) against the real, already-committed 1203
patch registration, with only ``execute_validation_producer()`` replaced by
a capturing stub -- so everything upstream of that call (argparse defaults,
registry/descriptor resolution, contract loading, ``ProducerContext``
construction) is the real production code path.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest import mock

TOOLS_DIR = Path(__file__).resolve().parents[2]
REPO_ROOT = TOOLS_DIR.parent
sys.path.insert(0, str(TOOLS_DIR))

from bigcherry.patch import validation as pv  # noqa: E402
from bigcherry.patch import validation_campaign as vc  # noqa: E402
from bigcherry.patch.campaign import producer as campaign_producer  # noqa: E402
from bigcherry.patch import validation_policy as patch_validation_policy  # noqa: E402
from bigcherry.patch import validation_producer as vp  # noqa: E402

_PATCH_ID = "1203_rd050607_rdna4_wmma_fa_q6k_mmq"
_PATCH_DIR = REPO_ROOT / "patches" / _PATCH_ID


class _CapturedExit(Exception):
    pass


class ProducerCorpusCliWiringTests(unittest.TestCase):
    """Everything here runs the REAL ``_run_validation_producer()`` /
    ``main()`` CLI dispatch code path against the real, already-committed
    1203 patch directory and registry entry. The only two things stubbed
    are (1) ``require_execution_package()`` -- 1203's currently-registered
    ``validation.toml`` does not yet satisfy every universal capability
    (``apply``/``build``) the generic policy layer independently requires,
    an unrelated pre-existing gap this fix does not touch -- and (2)
    ``execute_validation_producer()`` itself, purely to capture the real
    ``ProducerContext`` the CLI path constructs and to stop before any real
    build/subprocess work. The bug this test guards (``ProducerContext.corpus``
    hardcoded to ``None`` regardless of any CLI flag) lived entirely in the
    code between those two calls, so stubbing them does not hide it."""

    def _run_with_corpus(self, *, producer_corpus: Path | None) -> vp.ProducerContext:
        import argparse
        import tempfile

        captured: dict[str, vp.ProducerContext] = {}

        def _fake_execute_validation_producer(*, producer_context, **_kwargs):
            captured["producer_context"] = producer_context
            raise _CapturedExit()

        fake_plan = pv.ValidationPlan(patch_id=_PATCH_ID, checks=(), universal_capabilities=())

        with mock.patch.object(campaign_producer, "execute_validation_producer", _fake_execute_validation_producer), \
             mock.patch.object(
                 patch_validation_policy, "require_execution_package", return_value=fake_plan,
             ):
            with tempfile.TemporaryDirectory(prefix="pa39-corpus-cli-") as tmp:
                args = argparse.Namespace(
                    patch=_PATCH_ID,
                    hip_path=Path(r"C:\hip"),
                    workdir=Path(tmp) / "workdir",
                    amdgpu_targets="gfx1100;gfx1201;gfx1030",
                    device_map=(),
                    worktree_root=Path(tmp) / "worktrees",
                    model=None,
                    producer_corpus=producer_corpus,
                    correctness_evidence=None,
                    run_performance_benchmark=False,
                    bench_prompt=512,
                    bench_gen=128,
                )
                # 1203's rd050607 descriptor declares control_model as a
                # REQUIRED producer input; the dispatcher's fail-fast
                # validate_producer_inputs() (PA36 sub-slice 2, T5) gates
                # it before the (stubbed) executor, so the wiring test
                # supplies it. The gate is a pure membership check -- no
                # file is read.
                with self.assertRaises(_CapturedExit):
                    campaign_producer._run_validation_producer(
                        args, producer_id="rd050607",
                        provided_inputs={
                            "control_model": str(Path(tmp) / "control.gguf"),
                        },
                    )
        return captured["producer_context"]

    def test_producer_corpus_flag_reaches_producer_context_corpus(self) -> None:
        corpus_path = Path(r"C:\corpus\wiki.test.raw")
        ctx = self._run_with_corpus(producer_corpus=corpus_path)
        self.assertEqual(ctx.corpus, corpus_path)
        self.assertEqual(ctx.patch_dir, _PATCH_DIR)

    def test_absent_producer_corpus_flag_is_truthfully_none(self) -> None:
        ctx = self._run_with_corpus(producer_corpus=None)
        self.assertIsNone(ctx.corpus)

    def test_main_parses_and_threads_producer_corpus_end_to_end(self) -> None:
        """Drives the real ``main()`` CLI entry point (real argparse, real
        --validation-producer dispatch branch) with the same two seams
        stubbed, proving ``--producer-corpus`` survives real argument
        parsing all the way into ``ProducerContext.corpus``."""
        import tempfile

        captured: dict[str, vp.ProducerContext] = {}

        def _fake_execute_validation_producer(*, producer_context, **_kwargs):
            captured["producer_context"] = producer_context
            raise _CapturedExit()

        fake_plan = pv.ValidationPlan(patch_id=_PATCH_ID, checks=(), universal_capabilities=())

        with mock.patch.object(campaign_producer, "execute_validation_producer", _fake_execute_validation_producer), \
             mock.patch.object(
                 patch_validation_policy, "require_execution_package", return_value=fake_plan,
             ):
            with tempfile.TemporaryDirectory(prefix="pa39-corpus-cli-main-") as tmp:
                corpus_path = Path(tmp) / "wiki.test.raw"
                argv = [
                    "--patch", _PATCH_ID,
                    "--hip-path", str(Path(tmp) / "hip"),
                    "--workdir", str(Path(tmp) / "workdir"),
                    "--amdgpu-targets", "gfx1100;gfx1201;gfx1030",
                    "--validation-producer", f"{_PATCH_ID}/rd050607",
                    "--producer-corpus", str(corpus_path),
                    # rd050607 declares control_model required (PA39
                    # defect #3); the dispatcher's fail-fast input gate
                    # needs it before the stubbed executor runs.
                    "--producer-input", "control_model=" + str(Path(tmp) / "control.gguf"),
                ]
                with self.assertRaises(_CapturedExit):
                    vc.main(argv)

        ctx = captured["producer_context"]
        self.assertEqual(ctx.corpus, corpus_path)


if __name__ == "__main__":
    unittest.main()
