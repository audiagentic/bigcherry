"""PRBE111: hardware-free coverage for resolve_selected_device_execution_identity()
-- the shared helper resolving a real, host-configured ExecutionIdentity
(architecture + verified PCI locator) from the ambient HIP_VISIBLE_DEVICES,
never guessing a locator. Design reviewed with GPT (req_41097ce8b904467a)
before implementation."""

from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from bigcherry.patch.campaign import build as campaign_build  # noqa: E402


def _device(index: int, arch: str, locator: str | None) -> SimpleNamespace:
    return SimpleNamespace(index=index, arch=arch, locator=locator)


_HOST_DEVICES = (
    _device(0, "gfx1100", "0000:03:00.0"),
    _device(1, "gfx1100", "0000:06:00.0"),
    _device(2, "gfx1201", "0000:09:00.0"),
    _device(3, "gfx1030", None),  # deliberately unverified, for a real test
)


class ResolveSelectedDeviceExecutionIdentityTests(unittest.TestCase):
    def setUp(self) -> None:
        self._old_hip = os.environ.get("HIP_VISIBLE_DEVICES")
        self._old_rocr = os.environ.get("ROCR_VISIBLE_DEVICES")

    def tearDown(self) -> None:
        if self._old_hip is None:
            os.environ.pop("HIP_VISIBLE_DEVICES", None)
        else:
            os.environ["HIP_VISIBLE_DEVICES"] = self._old_hip
        if self._old_rocr is None:
            os.environ.pop("ROCR_VISIBLE_DEVICES", None)
        else:
            os.environ["ROCR_VISIBLE_DEVICES"] = self._old_rocr

    def test_resolves_real_configured_device(self) -> None:
        os.environ["HIP_VISIBLE_DEVICES"] = "0"
        identity, selector_env = campaign_build.resolve_selected_device_execution_identity(
            expected_arch="gfx1100", host_devices=_HOST_DEVICES,
        )
        self.assertEqual(identity.backend, "ROCm")
        self.assertEqual(identity.architectures, ("gfx1100",))
        self.assertEqual(identity.locators, ("0000:03:00.0",))
        self.assertEqual(selector_env, {"HIP_VISIBLE_DEVICES": "0"})

    def test_resolves_second_device_of_same_architecture_distinctly(self) -> None:
        # Two real gfx1100 cards are not interchangeable once locator
        # identity is pinned -- index 1 must resolve to ITS OWN locator.
        os.environ["HIP_VISIBLE_DEVICES"] = "1"
        identity, _ = campaign_build.resolve_selected_device_execution_identity(
            expected_arch="gfx1100", host_devices=_HOST_DEVICES,
        )
        self.assertEqual(identity.locators, ("0000:06:00.0",))

    def test_missing_hip_visible_devices_fails_closed(self) -> None:
        os.environ.pop("HIP_VISIBLE_DEVICES", None)
        with self.assertRaises(campaign_build.PatchCampaignError):
            campaign_build.resolve_selected_device_execution_identity(
                expected_arch="gfx1100", host_devices=_HOST_DEVICES,
            )

    def test_multi_device_selector_fails_closed(self) -> None:
        os.environ["HIP_VISIBLE_DEVICES"] = "0,1"
        with self.assertRaises(campaign_build.PatchCampaignError):
            campaign_build.resolve_selected_device_execution_identity(
                expected_arch="gfx1100", host_devices=_HOST_DEVICES,
            )

    def test_non_numeric_selector_fails_closed(self) -> None:
        os.environ["HIP_VISIBLE_DEVICES"] = "gfx1100"
        with self.assertRaises(campaign_build.PatchCampaignError):
            campaign_build.resolve_selected_device_execution_identity(
                expected_arch="gfx1100", host_devices=_HOST_DEVICES,
            )

    def test_unconfigured_index_fails_closed(self) -> None:
        os.environ["HIP_VISIBLE_DEVICES"] = "9"
        with self.assertRaises(campaign_build.PatchCampaignError):
            campaign_build.resolve_selected_device_execution_identity(
                expected_arch="gfx1100", host_devices=_HOST_DEVICES,
            )

    def test_device_with_no_verified_locator_fails_closed(self) -> None:
        # Index 3 (gfx1030) deliberately has locator=None in this fixture --
        # must never silently proceed without a real, verified locator.
        os.environ["HIP_VISIBLE_DEVICES"] = "3"
        with self.assertRaises(campaign_build.PatchCampaignError):
            campaign_build.resolve_selected_device_execution_identity(
                expected_arch="gfx1030", host_devices=_HOST_DEVICES,
            )

    def test_wrong_expected_architecture_fails_closed(self) -> None:
        # Index 2 is really gfx1201 -- requesting gfx1100 against it must
        # fail rather than silently attest the wrong architecture.
        os.environ["HIP_VISIBLE_DEVICES"] = "2"
        with self.assertRaises(campaign_build.PatchCampaignError):
            campaign_build.resolve_selected_device_execution_identity(
                expected_arch="gfx1100", host_devices=_HOST_DEVICES,
            )

    def test_uses_real_host_inventory_by_default(self) -> None:
        # Without an explicit host_devices override, this must resolve
        # against the REAL config/environment.toml -- index 0 is really
        # gfx1100 with a real verified locator on Brutus.
        os.environ["HIP_VISIBLE_DEVICES"] = "0"
        identity, _ = campaign_build.resolve_selected_device_execution_identity(
            expected_arch="gfx1100",
        )
        self.assertEqual(identity.architectures, ("gfx1100",))
        self.assertIsNotNone(identity.locators)
        self.assertTrue(identity.locators[0])


if __name__ == "__main__":
    unittest.main()
