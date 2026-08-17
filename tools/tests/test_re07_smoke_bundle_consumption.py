"""RE07 follow-up (GPT review, 2026-08-17): make_smoke_worker must consume
the verified runtime-bundle, not just take binary_ref on faith. binary_ref
alone only proves the launcher is unchanged; the .so files it actually
loads at runtime (where the real HIP dispatch logic lives, per RE09) were
previously never re-verified before execution.

No prior test exercised make_smoke_worker's real implementation at all --
test_campaign_lane.py always patches it out with a fake. This file is the
first real coverage of it.
"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bigcherry import campaign_workers  # noqa: E402
from bigcherry.artifacts import ArtifactStore  # noqa: E402
from bigcherry.campaign_build import CampaignBuildError  # noqa: E402
from bigcherry.pipeline import ArtifactRef  # noqa: E402
from bigcherry import provenance  # noqa: E402
from bigcherry.runtime_smoke import RuntimeSmokeSpec  # noqa: E402

_FAKE_SMOKE_STDOUT = json.dumps(
    [
        {
            "build_commit": "abc",
            "n_prompt": 512,
            "n_gen": 0,
            "avg_ts": 100.0,
            "samples_ts": [100.0],
        },
        {
            "build_commit": "abc",
            "n_prompt": 0,
            "n_gen": 128,
            "avg_ts": 50.0,
            "samples_ts": [50.0],
        },
    ]
)


class _Fixture:
    """A real ArtifactStore with a real binary + runtime-bundle published
    to it, exactly as make_build_worker would leave them -- so
    make_smoke_worker's own verification runs against real store state.
    """

    def __init__(self, directory: Path):
        self.store = ArtifactStore(directory / "store")
        self.prefix = "builds/s1/bp1"
        doc = provenance.make(
            project={},
            source={"source_slice_id": "s1"},
            build={"build_plan_id": "bp1"},
            workload={},
            campaign={"run_id": "run1"},
        )

        binary_relative = f"{self.prefix}/llama-bench"
        binary_digest = self.store.publish_bytes(binary_relative, b"launcher-bytes")
        so_relative = f"{self.prefix}/libggml-hip.so.0"
        so_digest = self.store.publish_bytes(so_relative, b"hip-dispatch-bytes")

        self.binary_ref = ArtifactRef(
            kind="binary",
            path=self.store.resolve(binary_relative),
            content_hash=binary_digest,
            provenance=doc,
        )

        manifest = {
            "entrypoint": "llama-bench",
            "members": {"llama-bench": binary_digest, "libggml-hip.so.0": so_digest},
            "runtime_bundle_hash": "irrelevant-for-this-test",
            "effective_build_id": "eb1",
            "generated_compile_inputs_hash": None,
            "toolchain": [],
        }
        bundle_relative = f"{self.prefix}/runtime-bundle-x.json"
        bundle_digest = self.store.publish_json(bundle_relative, manifest)
        self.bundle_ref = ArtifactRef(
            kind="runtime-bundle",
            path=self.store.resolve(bundle_relative),
            content_hash=bundle_digest,
            provenance=doc,
        )
        self.so_relative = so_relative

    def worker(self):
        spec = RuntimeSmokeSpec(model_path=Path(tempfile.mkstemp()[1]))
        return campaign_workers.make_smoke_worker(
            run_id="run1",
            store=self.store,
            source_slice_id="s1",
            build_plan_id="bp1",
            workload_id=None,
            spec=spec,
            # RE25.3: mechanism test with fixture (non-production)
            # provenance docs -- development class (see
            # test_re07_build_identity for the rationale).
            local_provenance_class="development",
        )


class SmokeConsumesVerifiedBundleTests(unittest.TestCase):
    def test_smoke_succeeds_when_bundle_and_binary_are_consistent_and_verified(self):
        with tempfile.TemporaryDirectory() as directory:
            fx = _Fixture(Path(directory))
            worker = fx.worker()
            with patch("bigcherry.campaign_workers.subprocess.run") as fake_run:
                fake_run.return_value.returncode = 0
                fake_run.return_value.stdout = _FAKE_SMOKE_STDOUT
                fake_run.return_value.stderr = ""
                refs = worker((fx.binary_ref, fx.bundle_ref))
            self.assertEqual(refs[0].kind, "smoke-result")

    def test_smoke_fails_closed_when_a_bundle_member_is_tampered(self):
        with tempfile.TemporaryDirectory() as directory:
            fx = _Fixture(Path(directory))
            worker = fx.worker()
            # Tamper with the published .so bytes directly, after publish --
            # exactly what a mutated runtime dependency looks like.
            (fx.store.root / fx.so_relative).write_bytes(b"TAMPERED")
            with patch("bigcherry.campaign_workers.subprocess.run") as fake_run:
                with self.assertRaises(CampaignBuildError):
                    worker((fx.binary_ref, fx.bundle_ref))
                fake_run.assert_not_called()

    def test_smoke_fails_closed_when_runtime_bundle_input_is_missing(self):
        with tempfile.TemporaryDirectory() as directory:
            fx = _Fixture(Path(directory))
            worker = fx.worker()
            with patch("bigcherry.campaign_workers.subprocess.run") as fake_run:
                with self.assertRaises(CampaignBuildError):
                    worker((fx.binary_ref,))  # no runtime-bundle at all
                fake_run.assert_not_called()

    def test_smoke_fails_closed_when_entrypoint_hash_disagrees_with_binary_ref(self):
        with tempfile.TemporaryDirectory() as directory:
            fx = _Fixture(Path(directory))
            worker = fx.worker()
            # A binary_ref that claims a DIFFERENT hash than what the
            # bundle manifest recorded for the entrypoint -- e.g. build
            # published two different-content binaries under confusable
            # identities. Must not silently run either as "the" bundle.
            wrong_binary_ref = ArtifactRef(
                kind="binary",
                path=fx.binary_ref.path,
                content_hash="0" * 64,
                provenance=fx.binary_ref.provenance,
            )
            with patch("bigcherry.campaign_workers.subprocess.run") as fake_run:
                with self.assertRaises(CampaignBuildError):
                    worker((wrong_binary_ref, fx.bundle_ref))
                fake_run.assert_not_called()


if __name__ == "__main__":
    unittest.main()
