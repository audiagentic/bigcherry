"""VA25: execution attestation -- the evidence-validity gate upstream of statistics.

Every case here is derived from a real failure this project has had, not from
imagined ones:

  * the CPU-fallback table (observed 2026-09-05, 72x wrong, tighter variance
    than the real GPU run, backend still labelled "ROCm");
  * gfx1030's HIP runtime failing to detect the device, which produced RD08's
    false subject_hit=false on 2026-09-01;
  * VA22's double-env-selector bug, which selects no device and recurred on
    2026-09-05 despite being documented.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from bigcherry.experiment import attestation as att  # noqa: E402


# The exact shape llama.cpp prints when ROCm comes up on this host.
_REAL_GFX1201 = """
ggml_cuda_init: found 1 ROCm devices (Total VRAM: 32624 MiB):
  Device 0: AMD Radeon Graphics, gfx1201 (0x1201), VMM: no, Wave Size: 32, VRAM: 32624 MiB
| model                | size | backend | ngl |  test |          t/s |
| qwen35 9B Q6_K       | 7.15 |    ROCm |  -1 | pp512 | 3160.50±31.80 |
"""

# The CPU-fallback run: a well-formed table, backend labelled ROCm, TIGHTER
# variance than the real GPU run, and one line of stderr as the only signal.
_CPU_FALLBACK = """
ggml_cuda_init: failed to initialize ROCm: no ROCm-capable device is detected
| model                | size | backend | ngl |  test |        t/s |
| qwen35 9B Q6_K       | 7.15 |    ROCm |  -1 | pp512 | 43.70±0.06 |
"""

_REAL_DUAL_GFX1100 = """
ggml_cuda_init: found 2 ROCm devices (Total VRAM: 49152 MiB):
  Device 0: AMD Radeon RX 7900 XTX, gfx1100 (0x1100), VMM: no, Wave Size: 32, VRAM: 24576 MiB
  Device 1: AMD Radeon RX 7900 XTX, gfx1100 (0x1100), VMM: no, Wave Size: 32, VRAM: 24576 MiB
"""


class ParseRocmAttestationTests(unittest.TestCase):
    def test_real_single_gpu_output_parses(self):
        obs = att.parse_rocm_attestation(_REAL_GFX1201)
        self.assertIsNotNone(obs)
        self.assertEqual(obs.backend, "ROCm")
        self.assertEqual(obs.device_count, 1)
        self.assertEqual(obs.devices[0].architecture, "gfx1201")
        self.assertEqual(obs.telemetry["first_device_vram_mib"], 32624)

    def test_real_dual_gpu_output_parses_both_devices(self):
        obs = att.parse_rocm_attestation(_REAL_DUAL_GFX1100)
        self.assertEqual(obs.device_count, 2)
        self.assertEqual(
            tuple(d.architecture for d in obs.devices), ("gfx1100", "gfx1100")
        )

    def test_cpu_fallback_is_a_positive_failure_not_a_silence(self):
        obs = att.parse_rocm_attestation(_CPU_FALLBACK)
        self.assertIsNotNone(obs)
        self.assertIsNone(obs.backend)
        self.assertIsNotNone(obs.failure_signature)

    def test_output_with_no_attestation_at_all_returns_none(self):
        # llama-server's own log format currently carries no ggml_cuda_init
        # line -- absence must never read as success.
        self.assertIsNone(att.parse_rocm_attestation("some log\nwith no init line\n"))
        self.assertIsNone(att.parse_rocm_attestation(""))

    def test_count_disagreeing_with_device_lines_is_refused(self):
        # Claims 2, lists 1. Do not guess which is right.
        obs = att.parse_rocm_attestation(
            "ggml_cuda_init: found 2 ROCm devices:\n"
            "  Device 0: AMD Radeon Graphics, gfx1201 (0x1201), VMM: no\n"
        )
        self.assertIsNone(obs.backend)
        self.assertIn("device line", obs.failure_signature)


# Real llama-server output, captured on Brutus 2026-09-05. Note it carries the
# PCI locator, which ggml_cuda_init's line does NOT.
_REAL_SERVER = """
0.00.025.981 I cmn  common_param:   - ROCm0   : AMD Radeon RX 7900 XTX (24560 MiB, 24520 MiB free)
0.00.109.723 I llama_prepare_model_devices: using device ROCm0 (AMD Radeon RX 7900 XTX) (0000:03:00.0) - 24520 MiB free
0.00.256.263 D load_tensors: layer   0 assigned to device ROCm0, is_swa = 0
0.00.256.264 D load_tensors: layer   1 assigned to device ROCm0, is_swa = 0
"""

# RD73's actual archived log shape: I and W levels present, no ROCm device
# line at all, because --fit off skips the device-fitting path that prints it.
_RD73_SHAPED_SERVER = """
0.00.025.217 I cmn  common_param: common_params_print_info: verbosity = 3
0.04.974.632 W internal AllReduce init failed (n_devices != 2?); falling back
0.05.393.687 W set_sampler: backend sampling not supported with SPLIT_MODE_TENSOR
"""

_LOCATOR_MAP = {"0000:03:00.0": "gfx1100", "0000:0c:00.0": "gfx1100"}


class ParseLlamaServerAttestationTests(unittest.TestCase):
    def test_real_server_output_yields_locator_backed_device(self):
        obs = att.parse_llama_server_attestation(
            _REAL_SERVER, architecture_by_locator=_LOCATOR_MAP)
        self.assertEqual(obs.backend, "ROCm")
        self.assertEqual(obs.device_count, 1)
        self.assertEqual(obs.devices[0].architecture, "gfx1100")
        self.assertEqual(obs.devices[0].locator, "0000:03:00.0")
        # Layer assignment is stronger than detection: it proves tensors
        # actually went to the device.
        self.assertEqual(obs.telemetry["layers_assigned_to_devices"], [0])

    def test_rd73_shaped_log_carries_no_attestation(self):
        # This is why RD73's six promoted sessions are unattested: --fit off
        # (required for -sm tensor) skips the path that prints the device line.
        self.assertIsNone(
            att.parse_llama_server_attestation(
                _RD73_SHAPED_SERVER, architecture_by_locator=_LOCATOR_MAP)
        )

    def test_unknown_locator_does_not_silently_match(self):
        obs = att.parse_llama_server_attestation(_REAL_SERVER, architecture_by_locator={})
        self.assertEqual(obs.devices[0].architecture, "<unknown>")
        expected = att.ExecutionIdentity(backend="ROCm", architectures=("gfx1100",))
        self.assertEqual(
            att.compare_execution_identity(expected, obs), (att.ARCH_MISMATCH,)
        )

    def test_server_cpu_fallback_is_a_positive_failure(self):
        obs = att.parse_llama_server_attestation(
            "some log\nfailed to initialize ROCm: no ROCm-capable device is detected\n")
        self.assertIsNone(obs.backend)
        self.assertIsNotNone(obs.failure_signature)

    def test_locator_distinguishes_the_two_gfx1100_cards(self):
        # The concrete reason locator is identity rather than telemetry.
        obs = att.parse_llama_server_attestation(
            _REAL_SERVER, architecture_by_locator=_LOCATOR_MAP)
        pinned_to_other_card = att.ExecutionIdentity(
            backend="ROCm", architectures=("gfx1100",), locators=("0000:0c:00.0",),
        )
        self.assertEqual(
            att.compare_execution_identity(pinned_to_other_card, obs),
            (att.DEVICE_ID_MISMATCH,),
        )


class CompareExecutionIdentityTests(unittest.TestCase):
    ONE_GFX1201 = att.ExecutionIdentity(backend="ROCm", architectures=("gfx1201",))
    TWO_GFX1100 = att.ExecutionIdentity(
        backend="ROCm", architectures=("gfx1100", "gfx1100")
    )

    def test_matching_environment_passes(self):
        obs = att.parse_rocm_attestation(_REAL_GFX1201)
        self.assertEqual(att.compare_execution_identity(self.ONE_GFX1201, obs), ())

    def test_cpu_fallback_is_rejected(self):
        obs = att.parse_rocm_attestation(_CPU_FALLBACK)
        self.assertEqual(
            att.compare_execution_identity(self.ONE_GFX1201, obs),
            (att.ATTESTATION_CORRUPT,),
        )

    def test_missing_attestation_is_rejected(self):
        self.assertEqual(
            att.compare_execution_identity(self.ONE_GFX1201, None),
            (att.ATTESTATION_MISSING,),
        )

    def test_wrong_architecture_is_rejected(self):
        # Built and selected for gfx1100, actually ran on gfx1201.
        obs = att.parse_rocm_attestation(_REAL_GFX1201)
        expected = att.ExecutionIdentity(backend="ROCm", architectures=("gfx1100",))
        self.assertEqual(
            att.compare_execution_identity(expected, obs), (att.ARCH_MISMATCH,)
        )

    def test_wrong_device_count_is_rejected(self):
        # The VA22 hazard: env selectors leave one device visible when the
        # lane requires two, so -sm tensor silently measures something else.
        obs = att.parse_rocm_attestation(_REAL_GFX1201)
        self.assertEqual(
            att.compare_execution_identity(self.TWO_GFX1100, obs),
            (att.DEVICE_COUNT_MISMATCH,),
        )

    def test_cardinality_fault_is_reported_alone(self):
        # Per-device comparison at a different cardinality is meaningless and
        # would bury the real fault under derived noise.
        obs = att.parse_rocm_attestation(_REAL_DUAL_GFX1100)
        reasons = att.compare_execution_identity(self.ONE_GFX1201, obs)
        self.assertEqual(reasons, (att.DEVICE_COUNT_MISMATCH,))

    def test_device_order_matters(self):
        # Tensor split is order-dependent, so a set-equality pass would hide
        # a genuinely different configuration.
        expected = att.ExecutionIdentity(
            backend="ROCm", architectures=("gfx1100", "gfx1201")
        )
        obs = att.ExecutionAttestation(
            backend="ROCm",
            devices=(
                att.ObservedDevice("gfx1201"),
                att.ObservedDevice("gfx1100"),
            ),
        )
        self.assertEqual(
            att.compare_execution_identity(expected, obs), (att.ARCH_MISMATCH,)
        )

    def test_same_arch_different_locator_is_rejected(self):
        # This host has TWO gfx1100 cards: architecture alone cannot detect a
        # run silently moving between them.
        expected = att.ExecutionIdentity(
            backend="ROCm", architectures=("gfx1100",), locators=("0000:03:00.0",),
        )
        obs = att.ExecutionAttestation(
            backend="ROCm",
            devices=(att.ObservedDevice("gfx1100", locator="0000:0c:00.0"),),
        )
        self.assertEqual(
            att.compare_execution_identity(expected, obs), (att.DEVICE_ID_MISMATCH,)
        )

    def test_pinned_locator_not_reported_by_backend_is_rejected(self):
        # A lane that pinned locators must not pass because the backend
        # declined to report one.
        expected = att.ExecutionIdentity(
            backend="ROCm", architectures=("gfx1100",), locators=("0000:03:00.0",),
        )
        obs = att.ExecutionAttestation(
            backend="ROCm", devices=(att.ObservedDevice("gfx1100", locator=None),),
        )
        self.assertEqual(
            att.compare_execution_identity(expected, obs), (att.DEVICE_ID_MISMATCH,)
        )

    def test_telemetry_differences_do_not_gate(self):
        # VRAM/driver drift is recorded, never compared.
        obs = att.ExecutionAttestation(
            backend="ROCm",
            devices=(att.ObservedDevice("gfx1201"),),
            telemetry={"first_device_vram_mib": 1},
        )
        self.assertEqual(att.compare_execution_identity(self.ONE_GFX1201, obs), ())

    def test_backend_mismatch_is_rejected(self):
        obs = att.ExecutionAttestation(
            backend="Vulkan", devices=(att.ObservedDevice("gfx1201"),),
        )
        self.assertEqual(
            att.compare_execution_identity(self.ONE_GFX1201, obs),
            (att.BACKEND_MISMATCH,),
        )


class RequireExecutionIdentityTests(unittest.TestCase):
    def test_pass_is_silent(self):
        obs = att.parse_rocm_attestation(_REAL_GFX1201)
        att.require_execution_identity(
            att.ExecutionIdentity(backend="ROCm", architectures=("gfx1201",)),
            obs, context="unit",
        )

    def test_failure_raises_and_carries_reason_codes(self):
        obs = att.parse_rocm_attestation(_CPU_FALLBACK)
        with self.assertRaises(att.AttestationError) as caught:
            att.require_execution_identity(
                att.ExecutionIdentity(backend="ROCm", architectures=("gfx1201",)),
                obs, context="rd73 mtp lane",
            )
        self.assertIn(att.ATTESTATION_CORRUPT, caught.exception.reasons)
        self.assertIn("rd73 mtp lane", str(caught.exception))


class ExecutionIdentityConstructionTests(unittest.TestCase):
    def test_empty_architectures_rejected(self):
        # A lane declaring no expected device cannot attest anything, so an
        # empty identity must not be constructible.
        with self.assertRaises(ValueError):
            att.ExecutionIdentity(backend="ROCm", architectures=())

    def test_locator_count_must_match_architecture_count(self):
        with self.assertRaises(ValueError):
            att.ExecutionIdentity(
                backend="ROCm", architectures=("gfx1100", "gfx1100"),
                locators=("0000:03:00.0",),
            )

    def test_attestation_document_is_serialisable_and_complete(self):
        obs = att.parse_rocm_attestation(_REAL_DUAL_GFX1100)
        doc = obs.document()
        self.assertEqual(doc["schema_version"], 1)
        self.assertEqual(doc["device_count"], 2)
        self.assertEqual(len(doc["devices"]), 2)
        self.assertIn("telemetry", doc)


if __name__ == "__main__":
    unittest.main()
