"""mtp_server_lane() derives its locator-aware ExecutionIdentity internally
from a device= argument (PVPS13), rather than each producer copy-pasting
``dataclasses.replace(device.execution_identity, locators=(device.locator,))``
by hand. Regression coverage for the argument contract only -- exercising the
real server/bench plumbing belongs to the hardware campaign, not this suite.
"""

from __future__ import annotations

import unittest
from types import SimpleNamespace

from bigcherry.experiment.attestation import ExecutionIdentity
from bigcherry.patch import producer_support as support
from bigcherry.patch import validation_producer as vp


def _device(*, locator: str | None) -> vp.ProducerDeviceContext:
    return vp.ProducerDeviceContext(
        architecture="gfx1100",
        device_index=0,
        execution_identity=ExecutionIdentity(backend="rocm", architectures=("gfx1100",)),
        env_overrides={},
        env_unset=(),
        locator=locator,
    )


class MtpServerLaneDeviceArgumentTests(unittest.TestCase):
    def test_neither_device_nor_expected_fails_closed(self) -> None:
        ctx = SimpleNamespace(model=None, corpus=None)
        with self.assertRaisesRegex(vp.ValidationProducerError, "exactly one of device= or expected="):
            support.mtp_server_lane(
                ctx, control_binary=None, subject_binary=None, env={}, label="test",
            )

    def test_both_device_and_expected_fails_closed(self) -> None:
        ctx = SimpleNamespace(model=None, corpus=None)
        with self.assertRaisesRegex(vp.ValidationProducerError, "exactly one of device= or expected="):
            support.mtp_server_lane(
                ctx, control_binary=None, subject_binary=None, env={}, label="test",
                device=_device(locator="0000:01:00.0"),
                expected=ExecutionIdentity(backend="rocm", architectures=("gfx1100",)),
            )

    def test_device_with_locator_reaches_past_identity_resolution(self) -> None:
        # No model/corpus configured -- proves the device= locator-merge
        # itself didn't raise; the next real check (model/corpus) did.
        ctx = SimpleNamespace(model=None, corpus=None)
        with self.assertRaisesRegex(vp.ValidationProducerError, "needs --model and --producer-corpus"):
            support.mtp_server_lane(
                ctx, control_binary=None, subject_binary=None, env={}, label="test",
                device=_device(locator="0000:01:00.0"),
            )

    def test_device_without_locator_reaches_past_identity_resolution(self) -> None:
        ctx = SimpleNamespace(model=None, corpus=None)
        with self.assertRaisesRegex(vp.ValidationProducerError, "needs --model and --producer-corpus"):
            support.mtp_server_lane(
                ctx, control_binary=None, subject_binary=None, env={}, label="test",
                device=_device(locator=None),
            )


if __name__ == "__main__":
    unittest.main()
