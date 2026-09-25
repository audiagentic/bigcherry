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


class LayerSplitPreflightTests(unittest.TestCase):
    def test_preflight_swaps_only_the_split_mode(self) -> None:
        from pathlib import Path
        from unittest import mock

        captured = {}

        def fake_preflight(*, binary, model, extra_args, output, env, expected_execution):
            captured[binary.name] = extra_args
            return ({"backend": "ROCm", "devices": [
                {"architecture": "gfx1100", "locator": "0000:03:00.0"},
                {"architecture": "gfx1100", "locator": "0000:06:00.0"}], "telemetry": {}}, {})

        class _Dev:
            def __init__(self, index, locator):
                self.index, self.arch, self.locator = index, "gfx1100", locator

        class _Host:
            devices = (_Dev(0, "0000:03:00.0"), _Dev(1, "0000:06:00.0"))

        env_obj = mock.Mock()
        env_obj.host.return_value = _Host()
        args = ("--parallel", "1", "-sm", "tensor", "--fit", "off")
        with mock.patch("bigcherry.campaign.benchmark._run_server_attestation_preflight", fake_preflight), \
             mock.patch("bigcherry.core.environment.load_default", return_value=env_obj):
            out = support.tensor_split_preflights(
                {"control": Path("c/llama-server")}, model=Path("m.gguf"), server_args=args,
                env={"HIP_VISIBLE_DEVICES": "0,1"}, workdir=Path("w"), label="t",
            )
        self.assertEqual(captured["llama-server"], ("--parallel", "1", "-sm", "layer", "--fit", "off"))
        self.assertEqual(out["control"].telemetry["attested_by"], "layer-split-preflight")
        self.assertEqual(len(out["control"].devices), 2)


class TensorSplitDetectionTests(unittest.TestCase):
    def test_detects_both_spellings(self) -> None:
        self.assertTrue(support._is_tensor_split(("-ngl", "99", "-sm", "tensor")))
        self.assertTrue(support._is_tensor_split(("--split-mode", "tensor")))
        self.assertFalse(support._is_tensor_split(("-sm", "layer")))
        self.assertFalse(support._is_tensor_split(()))


if __name__ == "__main__":
    unittest.main()
