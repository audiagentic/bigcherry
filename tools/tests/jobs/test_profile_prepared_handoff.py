from __future__ import annotations

import argparse
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from bigcherry.patch.campaign import profile


def _opts(root: Path) -> argparse.Namespace:
    return argparse.Namespace(
        patch="patch-a",
        arch="gfx1100",
        device="0",
        model=root / "model.gguf",
        workload="decode",
        hip_path=root / "rocm",
        worktree_root=root / "worktrees",
        build_root=root / "builds",
        out=root / "out",
        baseline_source="bigcherry-tuning",
        common_patches=("common-a",),
        env=[],
        allow_rejected=False,
        prepare_only=False,
        prepared_manifest=root / "prepared.json",
    )


def _bins(root: Path) -> dict[str, Path]:
    control = root / "builds" / "control-src" / "control" / "bin" / "llama-bench"
    subject = root / "builds" / "subject-src" / "validation-subject" / "bin" / "llama-bench"
    control.parent.mkdir(parents=True)
    subject.parent.mkdir(parents=True)
    control.write_bytes(b"control-v1")
    subject.write_bytes(b"subject-v1")
    return {"control": control, "subject": subject}


class PreparedProfileHandoffTests(unittest.TestCase):
    def test_manifest_roundtrip_binds_binary_bytes_and_inputs(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            opts = _opts(root)
            bins = _bins(root)
            profile._write_prepared_manifest(opts.prepared_manifest, opts, bins)
            loaded = profile._load_prepared_manifest(opts.prepared_manifest, opts)
            self.assertEqual({k: v.resolve() for k, v in bins.items()}, loaded)

            bins["subject"].write_bytes(b"subject-v2")
            with self.assertRaisesRegex(RuntimeError, "binary (size|bytes) drifted"):
                profile._load_prepared_manifest(opts.prepared_manifest, opts)

    def test_manifest_rejects_selector_drift(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            opts = _opts(root)
            profile._write_prepared_manifest(opts.prepared_manifest, opts, _bins(root))
            changed = argparse.Namespace(**vars(opts))
            changed.common_patches = ("different",)
            with self.assertRaisesRegex(RuntimeError, "identity mismatch for common_patches"):
                profile._load_prepared_manifest(opts.prepared_manifest, changed)

    def test_cli_prepare_then_profile_never_rebuilds_second_phase(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            opts = _opts(root)
            opts.model.write_bytes(b"model")
            bins = _bins(root)
            base = [
                "--patch", opts.patch,
                "--arch", opts.arch,
                "--device", opts.device,
                "--model", str(opts.model),
                "--workload", opts.workload,
                "--hip-path", str(opts.hip_path),
                "--worktree-root", str(opts.worktree_root),
                "--build-root", str(opts.build_root),
                "--out", str(opts.out),
                "--common-patches", "common-a",
                "--prepared-manifest", str(opts.prepared_manifest),
            ]
            with patch.object(profile, "build_pair", return_value=bins) as build, patch.object(
                profile, "profile_arm"
            ) as arm:
                self.assertEqual(profile.main([*base, "--prepare-only"]), 0)
                build.assert_called_once()
                arm.assert_not_called()

            with patch.object(
                profile, "build_pair", side_effect=AssertionError("GPU phase must not rebuild")
            ), patch.object(
                profile,
                "profile_arm",
                side_effect=lambda **kw: {
                    "returncode": 0,
                    "traces": [],
                    "report": str(kw["out"] / "kernel-fraction.json"),
                    "wrapper_crashed_after_measurement": False,
                },
            ) as arm:
                self.assertEqual(profile.main(base), 0)
                self.assertEqual(arm.call_count, 2)
            self.assertTrue((opts.out / "profile.json").is_file())


if __name__ == "__main__":
    unittest.main()
