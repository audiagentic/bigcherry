"""PA36 migration #2 (dev-gpt-agent req_69d011a2acf44acc): the RD04 PPL
correctness + paired-benchmark measurement moved from
validation_campaign.py's deleted run_rd04_contract_correctness() and
run_rd04_benchmark_evidence() into the patch-local producer at
patches/1202_rd04_bf16_flash_attn_tile/validation/producer.py.

These tests drive the REAL producer module (loaded through the real
resolve_producer() loader, so producer.toml policy is exercised too)
against a fake ProducerRuntime and a faked subprocess.run (the PPL
runner seam inside the producer module) -- the same fake machinery the
deleted CLI-path tests used, adapted to the producer protocol. The
dispatcher/binder/exit-semantics side of the migration is covered by
test_patch_validation_campaign_rd04_contract_cli.py.
"""

from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import Mapping
from unittest import mock

TOOLS_ROOT = Path(__file__).resolve().parents[2]
if str(TOOLS_ROOT) not in sys.path:
    sys.path.insert(0, str(TOOLS_ROOT))

from bigcherry.experiment.attestation import ExecutionIdentity  # noqa: E402
from bigcherry.patch import source as psi  # noqa: E402
from bigcherry.patch import validation_producer as vp  # noqa: E402
from bigcherry.patch.validation import ArtifactRef  # noqa: E402

SUBJECT_PATCH = "1202_rd04_bf16_flash_attn_tile"
PATCH_DIR = TOOLS_ROOT.parent / "patches" / SUBJECT_PATCH
CONTRACT_ARCHITECTURES = ("gfx1100", "gfx1201", "gfx1030")
FAT_TARGETS = "gfx1100;gfx1201;gfx1030"
PPL_EXTRA_ARGS = ("-fa", "on", "-ctk", "bf16", "-ctv", "bf16")

# SCAFFOLD (llama-bench pair) build identities -- deliberately distinct
# from the producer-built PPL pair's identities so a test can tell the
# two domains apart.
_SCAFFOLD_IDENTITIES = {
    "control": {"effective_build_id": "scaffold-control-id", "configure": ["scaffold"]},
    "subject": {"effective_build_id": "scaffold-subject-id", "configure": ["scaffold"]},
}
_PPL_PAIR_IDENTITIES = {
    "control": {"build_id": "ppl-pair-control-build"},
    "subject": {"build_id": "ppl-pair-subject-build"},
}


class _FakeRuntime:
    """ProducerRuntime seam fake: one prebuilt PPL pair, one real
    ProducerDeviceContext, real artifact files under run_dir, and a
    recording paired-benchmark fake (the producer must call it with the
    SCAFFOLD llama-bench binaries from ctx.validation_binaries, never a
    second build pair)."""

    def __init__(
        self,
        *,
        run_dir: Path,
        pair: vp.ProducerBuildPair,
        device: vp.ProducerDeviceContext | None,
        benchmark_stats: Mapping[str, Mapping[str, object]] | None = None,
    ) -> None:
        self.run_dir = run_dir
        self.pair = pair
        self.device = device
        self.benchmark_stats = benchmark_stats
        self.build_pair_calls: list[dict[str, object]] = []
        self.benchmark_calls: list[dict[str, object]] = []

    def build_pair(
        self,
        *,
        targets: tuple[str, ...],
        primary_target: str,
        common_extra_patches: tuple[str, ...] = (),
        baseline_source: str = "bigcherry",
        control_extra_cmake_args: tuple[str, ...] = (),
        subject_extra_cmake_args: tuple[str, ...] = (),
        require_parity: bool = False,
    ) -> vp.ProducerBuildPair:
        self.build_pair_calls.append(
            {
                "targets": tuple(targets),
                "primary_target": primary_target,
                "common_extra_patches": tuple(common_extra_patches),
                "baseline_source": baseline_source,
                "require_parity": require_parity,
            }
        )
        return self.pair

    def device_contexts(
        self,
        *,
        device_map: Mapping[str, tuple[int, ...]],
    ) -> tuple[vp.ProducerDeviceContext, ...]:
        return (self.device,) if self.device is not None else ()

    def write_artifact(
        self,
        *,
        name: str,
        payload: Mapping[str, object],
    ) -> ArtifactRef:
        return self._write(name, json.dumps(payload, indent=2))

    def write_text_artifact(self, *, name: str, text: str) -> ArtifactRef:
        return self._write(name, text)

    def run_paired_llama_benchmark(
        self,
        *,
        control_binary: Path,
        subject_binary: Path,
        model: Path,
        workloads: tuple[str, ...] = ("decode", "prefill"),
        patch_args: tuple[str, ...] = (),
        runtime_args: tuple[str, ...] = (),
        pairs: int = 3,
        log_context: str,
        device: vp.ProducerDeviceContext | None = None,
    ) -> vp.ProducerPairedBenchmarkOutcome:
        self.benchmark_calls.append(
            {
                "control_binary": control_binary,
                "subject_binary": subject_binary,
                "model": model,
                "patch_args": tuple(patch_args),
                "pairs": pairs,
                "log_context": log_context,
                "device": device,
            }
        )
        stats = self.benchmark_stats or {
            "decode": {"geometric_effect_pct": 3.41, "p_value": 0.02},
            "prefill": {"geometric_effect_pct": 2.97, "p_value": 0.04},
        }
        metrics = {"decode": "tg128", "prefill": "pp512"}

        def _lane(key: str) -> SimpleNamespace:
            return SimpleNamespace(
                runs=[
                    {"metric": metrics[key], "role": "control", "value": 100.0},
                    {"metric": metrics[key], "role": "subject", "value": 103.4},
                ],
                stats=dict(stats[key]),
            )

        return vp.ProducerPairedBenchmarkOutcome(
            runs={"decode": _lane("decode"), "prefill": _lane("prefill")},
            commands={
                "decode": {
                    "control": ("control-bench", "-m", "m.gguf"),
                    "subject": ("subject-bench", "-m", "m.gguf"),
                },
                "prefill": {
                    "control": ("control-bench", "-m", "m.gguf"),
                    "subject": ("subject-bench", "-m", "m.gguf"),
                },
            },
            raw_logs=(
                {"path": "artifacts/rd04-benchmark-decode.log", "sha256": "0" * 64},
                {"path": "artifacts/rd04-benchmark-prefill.log", "sha256": "0" * 64},
            ),
        )

    def _write(self, name: str, text: str) -> ArtifactRef:
        path = self.run_dir / "artifacts" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(text.encode("utf-8"))
        return ArtifactRef(
            name=name,
            path=path.relative_to(self.run_dir).as_posix(),
            sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        )


def _make_pair(temp: Path) -> vp.ProducerBuildPair:
    return vp.ProducerBuildPair(
        base_revision="a" * 40,
        control_source=temp / "trees" / "control",
        subject_source=temp / "trees" / "subject",
        control_composition=(),
        subject_composition=((SUBJECT_PATCH, "c2"),),
        control_bin=temp / "pair" / "CONTROL-BIN" / "llama-perplexity",
        subject_bin=temp / "pair" / "SUBJECT-BIN" / "llama-perplexity",
        validation_build_identities=_PPL_PAIR_IDENTITIES,
    )


def _make_device(architecture: str) -> vp.ProducerDeviceContext:
    return vp.ProducerDeviceContext(
        architecture=architecture,
        device_index=0,
        execution_identity=ExecutionIdentity(
            backend="ROCm",
            architectures=(architecture,),
        ),
        env_overrides={"HIP_VISIBLE_DEVICES": "0"},
        env_unset=("ROCR_VISIBLE_DEVICES",),
    )


def _fake_ppl_subprocess_run(
    *,
    subject_ppl: str = "10.50",
    control_ppl: str = "10.52",
    subject_line: bool = True,
    control_line: bool = True,
):
    """Stands in for subprocess.run inside the producer module (the PPL
    runner seam): llama-perplexity argv = [binary, -m, model, -f, corpus,
    -c, 2048, -ngl, 99, -fa, on, -ctk, bf16, -ctv, bf16]."""
    calls: list[tuple[str, list[str], dict[str, str]]] = []

    def run(argv, **kwargs):
        executable = str(argv[0])
        if "SUBJECT-BIN" in executable:
            arm, ppl, has_line = "subject", subject_ppl, subject_line
        elif "CONTROL-BIN" in executable:
            arm, ppl, has_line = "control", control_ppl, control_line
        else:
            raise AssertionError(f"unexpected binary: {executable!r}")
        calls.append((arm, list(argv), dict(kwargs.get("env") or {})))
        return SimpleNamespace(
            returncode=0,
            stdout=(
                f"Final estimate: PPL = {ppl} +/- 0.05\n" if has_line else "no line\n"
            ),
            stderr="",
        )

    return run, calls


def _run_producer(
    *,
    architecture: str = "gfx1100",
    with_device: bool = True,
    validation_binaries: dict[str, dict[str, Path]] | None = None,
    model: Path | None = None,
    corpus: Path | None = None,
    subject_ppl: str = "10.50",
    control_ppl: str = "10.52",
    subject_line: bool = True,
    control_line: bool = True,
    benchmark_stats: Mapping[str, Mapping[str, object]] | None = None,
):
    temp = Path(tempfile.mkdtemp())
    run_dir = temp / "run"
    run_dir.mkdir(parents=True)
    pair = _make_pair(temp)
    device = _make_device(architecture) if with_device else None
    runtime = _FakeRuntime(
        run_dir=run_dir,
        pair=pair,
        device=device,
        benchmark_stats=benchmark_stats,
    )
    bench_control = temp / "scaffold" / "control" / "llama-bench"
    bench_subject = temp / "scaffold" / "subject" / "llama-bench"
    if validation_binaries is None:
        validation_binaries = {
            "control": {
                "llama-server": temp / "scaffold" / "control" / "llama-server",
                "llama-bench": bench_control,
            },
            "subject": {
                "llama-server": temp / "scaffold" / "subject" / "llama-server",
                "llama-bench": bench_subject,
            },
        }
    fake_run, ppl_calls = _fake_ppl_subprocess_run(
        subject_ppl=subject_ppl,
        control_ppl=control_ppl,
        subject_line=subject_line,
        control_line=control_line,
    )
    ctx = vp.ProducerContext(
        repo_root=TOOLS_ROOT.parent,
        patch_dir=PATCH_DIR,
        workdir=run_dir,
        campaign_id=f"{SUBJECT_PATCH}/rd04",
        base_revision="a" * 40,
        hip_path=Path("/opt/rocm"),
        fat_targets=vp.FatTargetPlan(targets=(architecture,)),
        model=Path("/models/m.gguf") if model is None else model,
        corpus=Path("/corpus/c.txt") if corpus is None else corpus,
        build_env={"HIP_PATH": "/opt/rocm"},
        inputs={},
        validation_build_identities=_SCAFFOLD_IDENTITIES,
        patch_id=SUBJECT_PATCH,
        device_map={architecture: (0,)},
        runtime=runtime,
        validation_binaries=validation_binaries,
    )

    with (
        mock.patch.object(psi, "git_worktree_tree", lambda p: f"tree:{p}"),
        mock.patch("subprocess.run", fake_run),
    ):
        selection = vp.resolve_producer(patch_dir=PATCH_DIR, producer_id="rd04")
        result = selection.producer(ctx)
    return (
        result,
        run_dir,
        runtime,
        pair,
        ppl_calls,
        selection,
        bench_control,
        bench_subject,
    )


def _correctness_doc(result: vp.ProducerResult) -> dict[str, object]:
    value = result.correctness
    if not isinstance(value, Mapping):
        raise AssertionError(f"producer returned no correctness dict: {value!r}")
    return dict(value)


def _performance_artifact(result: vp.ProducerResult) -> dict[str, object]:
    value = result.performance_evidence
    if not isinstance(value, Mapping):
        raise AssertionError(f"producer returned no performance evidence: {value!r}")
    artifact = value.get("artifact")
    if not isinstance(artifact, Mapping):
        raise AssertionError(f"no performance artifact ref: {value!r}")
    return dict(artifact)


class Rd04ProducerCorrectnessTests(unittest.TestCase):
    def test_contract_architectures_pass_with_forced_bf16_flash_attn_flags(
        self,
    ) -> None:
        for architecture in CONTRACT_ARCHITECTURES:
            with self.subTest(architecture=architecture):
                (
                    result,
                    run_dir,
                    runtime,
                    pair,
                    ppl_calls,
                    selection,
                    bench_control,
                    bench_subject,
                ) = _run_producer(architecture=architecture)

                # producer.toml policy is the real one, loaded for real.
                self.assertEqual(selection.spec.standard_campaign, "run")
                self.assertEqual(selection.spec.trace_probe, "skip")
                self.assertEqual(selection.spec.correctness_evidence_cli, "forbid")
                self.assertEqual(selection.spec.performance_benchmark_cli, "forbid")

                # One sanctioned PPL pair build: fat multi-arch,
                # bigcherry baseline, llama-perplexity primary target.
                self.assertEqual(len(runtime.build_pair_calls), 1)
                call = runtime.build_pair_calls[0]
                self.assertEqual(call["targets"], CONTRACT_ARCHITECTURES)
                self.assertEqual(call["primary_target"], "llama-perplexity")
                self.assertEqual(call["baseline_source"], "bigcherry")
                self.assertEqual(call["common_extra_patches"], ())
                # GPT review req_7a72896b609a48b5 BLOCKER #3: RD04's PPL
                # comparison is parity-dependent, so the producer demands
                # the concrete runtime run
                # assert_validation_subject_parity() on its own pair.
                self.assertTrue(call["require_parity"])

                # Exactly two real perplexity executions (subject, then
                # control), each forced onto the focal BF16 flash-attn
                # path with the sanctioned HIP-only selector env.
                self.assertEqual(
                    [arm for arm, _, _ in ppl_calls], ["subject", "control"]
                )
                for _arm, argv, env in ppl_calls:
                    for flag, value in (
                        ("-fa", "on"),
                        ("-ctk", "bf16"),
                        ("-ctv", "bf16"),
                    ):
                        self.assertIn(flag, argv)
                        self.assertEqual(argv[argv.index(flag) + 1], value)
                    self.assertEqual(env["HIP_VISIBLE_DEVICES"], "0")
                    self.assertNotIn("ROCR_VISIBLE_DEVICES", env)
                    # The pair's own binaries, model and corpus are real.
                    self.assertIn("-m", argv)
                    self.assertIn("-f", argv)

                # Semantic correctness only; NO activation or trace
                # evidence (1202 has no valid RD04 marker -- the declared
                # activation check stays honestly BLOCKED).
                correctness = _correctness_doc(result)
                self.assertEqual(
                    set(correctness), {"disposition", "mechanism", "detail"}
                )
                self.assertEqual(correctness["disposition"], "passed")
                self.assertEqual(
                    correctness["mechanism"],
                    "rd04-bf16-flash-attn-ppl-comparison",
                )
                self.assertIn("backend_reference=PASS", str(correctness["detail"]))
                self.assertIn("ppl_equality=PASS", str(correctness["detail"]))
                self.assertEqual(result.activation_evidence, None)
                self.assertEqual(result.trace_evidence, None)
                self.assertEqual(result.check_results, ())
                self.assertEqual(result.lane_effects, ())

                # GPT review req_7a72896b609a48b5 BLOCKER #2: the typed
                # named contract-correctness results the producer measured
                # (backend_reference + ppl_equality), both passing here.
                named = result.contract_correctness_results
                self.assertEqual(len(named), 2)
                self.assertEqual(
                    {r.check for r in named}, {"backend_reference", "ppl_equality"}
                )
                self.assertTrue(all(r.passed for r in named))

                # The record-level validation identities come from the
                # producer's OWN PPL pair (the scaffold identities stay in
                # the raw performance document only).
                self.assertEqual(
                    result.validation_build_identities, _PPL_PAIR_IDENTITIES
                )

                # Both artifacts emitted, allowlisted, on disk, sha-bound.
                expected_names = {
                    f"rd04-correctness-{architecture}.json",
                    f"rd04-performance-{architecture}.json",
                }
                self.assertEqual(result.emitted_artifacts, frozenset(expected_names))
                for name in sorted(expected_names):
                    target = run_dir / "artifacts" / name
                    self.assertTrue(target.is_file(), name)
                    self.assertIn(name, selection.spec.artifact_names)

                doc = json.loads(
                    (
                        run_dir / "artifacts" / f"rd04-correctness-{architecture}.json"
                    ).read_text(encoding="utf-8"),
                )
                self.assertEqual(doc["contract_id"], "RD04-BF16-FLASH-ATTN-TILE")
                self.assertEqual(doc["architecture"], architecture)
                self.assertEqual(doc["compiled_targets"], FAT_TARGETS)
                self.assertEqual(doc["subject_patch"], SUBJECT_PATCH)
                self.assertEqual(doc["perplexity_extra_args"], list(PPL_EXTRA_ARGS))
                self.assertEqual(
                    set(doc["results"]), {"backend_reference", "ppl_equality"}
                )
                self.assertTrue(doc["results"]["backend_reference"]["passed"])
                self.assertTrue(doc["results"]["ppl_equality"]["passed"])
                self.assertIsNotNone(doc["comparison"])
                self.assertEqual(
                    doc["control_build_identity"], _PPL_PAIR_IDENTITIES["control"]
                )
                self.assertEqual(
                    doc["subject_build_identity"], _PPL_PAIR_IDENTITIES["subject"]
                )
                self.assertEqual(
                    doc["control_source_tree"], f"tree:{pair.control_source}"
                )
                self.assertEqual(
                    doc["subject_source_tree"], f"tree:{pair.subject_source}"
                )

                # Raw performance document: SCAFFOLD identities, real
                # command/log provenance, finite stats on both lanes.
                perf_doc = json.loads(
                    (
                        run_dir / "artifacts" / f"rd04-performance-{architecture}.json"
                    ).read_text(encoding="utf-8"),
                )
                self.assertTrue(perf_doc["passed"])
                self.assertEqual(perf_doc["campaign_id"], f"{SUBJECT_PATCH}/rd04")
                self.assertEqual(perf_doc["architecture"], architecture)
                self.assertEqual(
                    perf_doc["validation_build_identities"], _SCAFFOLD_IDENTITIES
                )
                self.assertIn("decode", perf_doc["metrics"])
                self.assertIn("prefill", perf_doc["metrics"])
                self.assertEqual(perf_doc["metrics"]["decode"]["metric"], "tg128")
                self.assertEqual(perf_doc["metrics"]["prefill"]["metric"], "pp512")

                # The semantic performance ref points at the produced
                # artifact, bound with its real sha.
                artifact = _performance_artifact(result)
                perf_path = (
                    run_dir / "artifacts" / f"rd04-performance-{architecture}.json"
                )
                self.assertEqual(
                    artifact["path"], f"artifacts/rd04-performance-{architecture}.json"
                )
                self.assertEqual(
                    artifact["sha256"],
                    hashlib.sha256(perf_path.read_bytes()).hexdigest(),
                )

    def test_ppl_env_device_selector_wins_and_unset_last(self) -> None:
        # GPT review req_7a72896b609a48b5 MAJOR #4: the env merge order is
        # base (os.environ + supplied) -> device selector -> env_unset LAST.
        # An adversarial SUPPLIED env (from run_perplexity) AND a stale
        # ambient ROCR_VISIBLE_DEVICES must both lose to the sanctioned
        # HIP-only device selector.
        import os

        from bigcherry.experiment import perplexity

        temp = Path(tempfile.mkdtemp())
        run_dir = temp / "run"
        run_dir.mkdir(parents=True)
        pair = _make_pair(temp)
        runtime = _FakeRuntime(
            run_dir=run_dir,
            pair=pair,
            device=_make_device("gfx1100"),
        )
        fake_run, ppl_calls = _fake_ppl_subprocess_run()
        ctx = vp.ProducerContext(
            repo_root=TOOLS_ROOT.parent,
            patch_dir=PATCH_DIR,
            workdir=run_dir,
            campaign_id=f"{SUBJECT_PATCH}/rd04",
            base_revision="a" * 40,
            hip_path=Path("/opt/rocm"),
            fat_targets=vp.FatTargetPlan(targets=("gfx1100",)),
            model=Path("/models/m.gguf"),
            corpus=Path("/corpus/c.txt"),
            build_env={"HIP_PATH": "/opt/rocm"},
            inputs={},
            validation_build_identities=_SCAFFOLD_IDENTITIES,
            patch_id=SUBJECT_PATCH,
            device_map={"gfx1100": (0,)},
            runtime=runtime,
            validation_binaries={
                "control": {"llama-bench": temp / "scaffold" / "c" / "llama-bench"},
                "subject": {"llama-bench": temp / "scaffold" / "s" / "llama-bench"},
            },
        )
        hostile_supplied = {
            "HIP_VISIBLE_DEVICES": "7",
            "ROCR_VISIBLE_DEVICES": "1,2,3",
        }

        def fake_run_perplexity(
            binary,
            *,
            model,
            corpus,
            ctx_size=2048,
            runner,
            extra_args=(),
        ) -> SimpleNamespace:
            argv = [
                str(binary),
                "-m",
                str(model),
                "-f",
                str(corpus),
                "-c",
                str(ctx_size),
                "-ngl",
                "99",
                *extra_args,
            ]
            # Adversarial: the runner seam is handed an env carrying a
            # competing selector AND a re-introduced ROCR key.
            runner(argv, capture_output=True, text=True, env=dict(hostile_supplied))
            return SimpleNamespace(ppl=10.5, uncertainty=0.05)

        with (
            mock.patch.object(psi, "git_worktree_tree", lambda p: f"tree:{p}"),
            mock.patch("subprocess.run", fake_run),
            mock.patch.object(
                perplexity,
                "run_perplexity",
                fake_run_perplexity,
            ),
            mock.patch.dict(
                os.environ,
                {"ROCR_VISIBLE_DEVICES": "0,1", "HIP_VISIBLE_DEVICES": "3"},
            ),
        ):
            selection = vp.resolve_producer(patch_dir=PATCH_DIR, producer_id="rd04")
            selection.producer(ctx)

        # Both PPL runs reached subprocess.run with the selector winning.
        self.assertEqual(len(ppl_calls), 2)
        for _arm, _argv, env in ppl_calls:
            # Device selector beats the hostile supplied "7" AND ambient "3".
            self.assertEqual(env["HIP_VISIBLE_DEVICES"], "0")
            # env_unset popped the stale ROCR key LAST (supplied + ambient).
            self.assertNotIn("ROCR_VISIBLE_DEVICES", env)
        import shutil

        shutil.rmtree(temp, ignore_errors=True)

    def test_benchmark_reuses_scaffold_binaries_never_a_second_pair(self) -> None:
        # T11 regression: the paired benchmark must run the STANDARD
        # SCAFFOLD's llama-bench pair (ctx.validation_binaries) -- the
        # producer never builds a second pair for it.
        (
            result,
            run_dir,
            runtime,
            pair,
            ppl_calls,
            selection,
            bench_control,
            bench_subject,
        ) = _run_producer()
        self.assertEqual(len(runtime.benchmark_calls), 1)
        bench = runtime.benchmark_calls[0]
        self.assertEqual(bench["control_binary"], bench_control)
        self.assertEqual(bench["subject_binary"], bench_subject)
        # Never the producer-built PLL pair binaries:
        self.assertNotEqual(bench["control_binary"], pair.control_bin)
        self.assertNotEqual(bench["subject_binary"], pair.subject_bin)
        self.assertEqual(bench["patch_args"], PPL_EXTRA_ARGS)
        self.assertEqual(bench["pairs"], 3)
        self.assertEqual(bench["log_context"], "rd04")
        # The benchmark reuses the producer's real model (ctx.model), not a
        # second pair's binary/model.
        self.assertEqual(bench["model"], Path("/models/m.gguf"))
        self.assertIsNotNone(bench["device"])

    def test_ppl_divergence_fails_both_named_results(self) -> None:
        # control PPL far outside combined uncertainty: sigma > 3.
        result, run_dir, runtime, _, _, _, _, _ = _run_producer(
            subject_ppl="10.50",
            control_ppl="15.00",
        )
        correctness = _correctness_doc(result)
        self.assertEqual(correctness["disposition"], "failed")
        self.assertIn("backend_reference=FAIL", str(correctness["detail"]))
        self.assertIn("ppl_equality=FAIL", str(correctness["detail"]))
        doc = json.loads(
            (run_dir / "artifacts" / "rd04-correctness-gfx1100.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertFalse(doc["results"]["backend_reference"]["passed"])
        self.assertFalse(doc["results"]["ppl_equality"]["passed"])
        self.assertIn("sigma=", doc["results"]["backend_reference"]["detail"])

    def test_unparseable_ppl_output_fails_closed(self) -> None:
        # A zero-exit perplexity run with no "Final estimate" line must
        # never be treated as passing evidence.
        result, _, _, _, _, _, _, _ = _run_producer(control_line=False)
        correctness = _correctness_doc(result)
        self.assertEqual(correctness["disposition"], "failed")
        self.assertIn(
            "could not produce a real BF16 flash-attn perplexity comparison",
            str(correctness["detail"]),
        )

    def test_unsupported_or_multi_architecture_fails_before_build(self) -> None:
        for targets in (("gfx1151",), ("gfx1100", "gfx1201")):
            with self.subTest(targets=targets):
                temp = Path(tempfile.mkdtemp())
                run_dir = temp / "run"
                run_dir.mkdir()
                pair = _make_pair(temp)
                runtime = _FakeRuntime(run_dir=run_dir, pair=pair, device=None)
                ctx = vp.ProducerContext(
                    repo_root=TOOLS_ROOT.parent,
                    patch_dir=PATCH_DIR,
                    workdir=run_dir,
                    campaign_id=f"{SUBJECT_PATCH}/rd04",
                    base_revision="a" * 40,
                    hip_path=Path("/opt/rocm"),
                    fat_targets=vp.FatTargetPlan(targets=targets),
                    model=Path("/models/m.gguf"),
                    corpus=Path("/corpus/c.txt"),
                    build_env={},
                    inputs={},
                    validation_build_identities={},
                    patch_id=SUBJECT_PATCH,
                    device_map={},
                    runtime=runtime,
                )
                selection = vp.resolve_producer(
                    patch_dir=PATCH_DIR,
                    producer_id="rd04",
                )
                with self.assertRaisesRegex(
                    vp.ValidationProducerError, "exactly one contract architecture"
                ):
                    selection.producer(ctx)
                # Fail fast: no build attempt.
                self.assertEqual(runtime.build_pair_calls, [])

    def test_missing_model_or_corpus_fails_before_build(self) -> None:
        for model, corpus in (
            (None, Path("/corpus/c.txt")),
            (Path("/models/m.gguf"), None),
        ):
            with self.subTest(model=model, corpus=corpus):
                temp = Path(tempfile.mkdtemp())
                run_dir = temp / "run"
                run_dir.mkdir()
                pair = _make_pair(temp)
                runtime = _FakeRuntime(run_dir=run_dir, pair=pair, device=None)
                ctx = vp.ProducerContext(
                    repo_root=TOOLS_ROOT.parent,
                    patch_dir=PATCH_DIR,
                    workdir=run_dir,
                    campaign_id=f"{SUBJECT_PATCH}/rd04",
                    base_revision="a" * 40,
                    hip_path=Path("/opt/rocm"),
                    fat_targets=vp.FatTargetPlan(targets=("gfx1100",)),
                    model=model,
                    corpus=corpus,
                    build_env={},
                    inputs={},
                    validation_build_identities={},
                    patch_id=SUBJECT_PATCH,
                    device_map={"gfx1100": (0,)},
                    runtime=runtime,
                )
                selection = vp.resolve_producer(
                    patch_dir=PATCH_DIR,
                    producer_id="rd04",
                )
                with self.assertRaisesRegex(
                    vp.ValidationProducerError, "BOTH a real whole model"
                ):
                    selection.producer(ctx)
                self.assertEqual(runtime.build_pair_calls, [])

    def test_no_device_for_run_architecture_fails_closed(self) -> None:
        with self.assertRaisesRegex(vp.ValidationProducerError, "no device mapped"):
            _run_producer(with_device=False)

    def test_missing_scaffold_bench_binaries_fails_closed(self) -> None:
        # standard_campaign='skip' populates validation_binaries={}: the
        # producer must refuse rather than benchmark with nothing.
        with self.assertRaisesRegex(
            vp.ValidationProducerError,
            "scaffold's llama-bench",
        ):
            _run_producer(validation_binaries={})

    def test_nonfinite_benchmark_stats_fail_performance_not_correctness(self) -> None:
        # The benchmark's `passed` means "executed with finite statistics
        # on both lanes" -- it never touches the PPL correctness
        # disposition.
        result, run_dir, _, _, _, _, _, _ = _run_producer(
            benchmark_stats={
                "decode": {"geometric_effect_pct": 3.41, "p_value": 0.02},
                "prefill": {"geometric_effect_pct": None, "p_value": None},
            },
        )
        correctness = _correctness_doc(result)
        self.assertEqual(correctness["disposition"], "passed")
        perf_doc = json.loads(
            (run_dir / "artifacts" / "rd04-performance-gfx1100.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertFalse(perf_doc["passed"])


if __name__ == "__main__":
    unittest.main()
