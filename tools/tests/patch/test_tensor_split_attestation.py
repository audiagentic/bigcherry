"""Tensor-split (-sm tensor) servers: timed processes bind to an RCCL preflight."""

from __future__ import annotations

import unittest

from bigcherry.experiment import server_execution as se
from bigcherry.experiment.attestation import ExecutionAttestation, ObservedDevice
from bigcherry.patch import producer_support as support

PREPARED = (
    "0.1 I llama_prepare_model_devices: - device 0: ROCm0 (AMD Radeon RX 7900 XTX)\n"
    "0.1 I llama_prepare_model_devices: - device 1: ROCm1 (AMD Radeon RX 7900 XTX)\n"
)
META = "0.2 D load_tensors: layer   0 assigned to device Meta(), is_swa = 0\n"
PREFLIGHT = ExecutionAttestation(
    backend="ROCm",
    devices=(ObservedDevice("gfx1100", "0000:03:00.0"), ObservedDevice("gfx1100", "0000:06:00.0")),
)


class MetaBindingTests(unittest.TestCase):
    def test_meta_process_over_the_preflight_devices_matches(self) -> None:
        self.assertTrue(se.meta_process_matches_preflight(PREPARED + META, PREFLIGHT))

    def test_device_count_mismatch_fails(self) -> None:
        one_device = PREPARED.splitlines(keepends=True)[0] + META
        self.assertFalse(se.meta_process_matches_preflight(one_device, PREFLIGHT))

    def test_no_meta_assignment_fails(self) -> None:
        self.assertFalse(se.meta_process_matches_preflight(PREPARED, PREFLIGHT))

    def test_failed_preflight_never_binds(self) -> None:
        failed = ExecutionAttestation(backend=None, devices=PREFLIGHT.devices, failure_signature="x")
        self.assertFalse(se.meta_process_matches_preflight(PREPARED + META, failed))


class TensorSplitDetectionTests(unittest.TestCase):
    def test_detects_both_spellings(self) -> None:
        self.assertTrue(support._is_tensor_split(("-ngl", "99", "-sm", "tensor")))
        self.assertTrue(support._is_tensor_split(("--split-mode", "tensor")))
        self.assertFalse(support._is_tensor_split(("-sm", "layer")))
        self.assertFalse(support._is_tensor_split(()))


if __name__ == "__main__":
    unittest.main()
