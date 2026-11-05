"""BU01: release bundles must bind a fresh lane result, not an old pin receipt."""
import sys
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from bigcherry.release import pin_release
from bigcherry.release import pin_bump


class LaneBindingTests(unittest.TestCase):
    def setUp(self):
        self.plan = pin_release.Plan(
            repo_root=Path(__file__).resolve().parents[3], engine="radiance",
            package="engines/radiance", upstream_name="radiance",
            llama_tag="v1.16.0", build=1160, version="1160.0.1",
            release_tag="bc-radiance-1160.0.1", branch="feat/staged")
        self.old_id = "a" * 32
        self.new_id = "b" * 32
        self.new_sha = "c" * 64
        self.record = {"notes": f"pin-bump lane build: radiance:stock:linux-radiance "
                                f"build_plan_id={self.old_id} binary_sha256={'d'*64}"}

    def test_fresh_canonical_lane_result_wins_over_older_record(self):
        result = SimpleNamespace(build_plan_id=self.new_id, binary_sha256=self.new_sha)
        receipt = {"plan_id": self.new_id}
        with mock.patch.object(pin_bump, "execute_lane_build", return_value=result) as build, \
             mock.patch.object(pin_release, "_asset_environment",
                               return_value=((self.plan.repo_root / "work/bu01-test"), (self.plan.repo_root / "work/sdk-test"))), \
             mock.patch.object(pin_release._runtime_assets, "pack",
                               return_value=receipt) as pack:
            pin_release._prepare_runtime_asset(self.plan, self.record)
        build.assert_called_once()
        self.assertEqual(build.call_args.kwargs["lane"], "radiance:stock:linux-radiance")
        self.assertIn(f"build_plan_id={self.new_id} binary_sha256={self.new_sha}",
                      self.record["notes"].splitlines()[-1])
        self.assertEqual(self.record["runtime_assets"][self.plan.release_tag], receipt)
        pack.assert_called_once_with(self.record, self.plan.repo_root,
                                     (self.plan.repo_root / "work/bu01-test"), (self.plan.repo_root / "work/sdk-test"), self.plan.release_tag)

    def test_failed_current_lane_never_generates_an_asset(self):
        with mock.patch.object(pin_bump, "execute_lane_build",
                               side_effect=pin_bump.LaneBuildError("build", "missing patch")):
            with self.assertRaisesRegex(pin_release.PinReleaseError, "missing patch"):
                pin_release._prepare_runtime_asset(self.plan, self.record)
        self.assertNotIn("runtime_assets", self.record)


if __name__ == "__main__":
    unittest.main()
