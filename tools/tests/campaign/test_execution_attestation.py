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
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path, PurePosixPath

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


class LinuxKfdProcessEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.pid = 4242
        self.proc = self.root / "proc"
        self.kfd = self.root / "kfd"
        self.drm = self.root / "drm"
        self.binary = self.root / "bin" / "llama-server"
        self.binary.parent.mkdir()
        self.binary.write_bytes(b"server")
        proc_pid = self.proc / str(self.pid)
        proc_pid.mkdir(parents=True)
        self.exe_link = proc_pid / "exe"
        self.other_binary = self.root / "bin" / "other"
        self.other_binary.write_bytes(b"other")
        # /proc/<pid>/stat fields after the final ')' begin at field 3;
        # field 22 (starttime) is index 19 in this suffix.
        (proc_pid / "stat").write_text(") " + " ".join(["S"] + ["0"] * 18 + ["1234"]) + "\n", encoding="ascii")
        nodes = self.kfd / "topology" / "nodes"
        self.drm_targets = {}
        for node_id, gpu_id, minor in (("0", 10, 128), ("1", 11, 129)):
            node = nodes / node_id
            node.mkdir(parents=True)
            (node / "gpu_id").write_text(str(gpu_id), encoding="ascii")
            (node / "properties").write_text(
                f"drm_render_minor {minor}\ngfx_target_version 1100\n", encoding="ascii"
            )
            render = self.drm / f"renderD{minor}"
            render.mkdir(parents=True)
            self.drm_targets[render / "device"] = PurePosixPath(
                f"/sys/devices/pci0000:00/0000:00:01.0/0000:{gpu_id - 9:02x}:00.0"
            )

        def resolve_fixture(path, strict=False):
            if path == self.exe_link:
                return self.other_binary if getattr(self, "exe_changed", False) else self.binary
            return self.drm_targets.get(path, path.absolute())
        self.resolve_patch = patch.object(Path, "resolve", new=resolve_fixture)
        self.resolve_patch.start()
        self.addCleanup(self.resolve_patch.stop)
        process_kfd = self.kfd / "proc" / str(self.pid)
        process_kfd.mkdir(parents=True)
        (process_kfd / "vram_10").write_text("4096\n", encoding="ascii")
        (process_kfd / "vram_11").write_text("0\n", encoding="ascii")
        for qid, gpu_id, size, qtype in (("7", 10, 2048, 0), ("8", 11, 0, 1)):
            queue = process_kfd / "queues" / qid
            queue.mkdir(parents=True)
            (queue / "gpuid").write_text(str(gpu_id), encoding="ascii")
            (queue / "size").write_text(str(size), encoding="ascii")
            (queue / "type").write_text(str(qtype), encoding="ascii")

    def capture(self, **kwargs):
        return att.capture_linux_kfd_process_evidence(
            self.pid, self.binary, proc_root=self.proc, kfd_root=self.kfd,
            drm_root=self.drm, **kwargs,
        )

    def test_captures_all_nodes_including_zero_vram(self):
        evidence = self.capture()
        self.assertEqual(evidence["schema_version"], 1)
        self.assertEqual(evidence["kind"], "linux-kfd-process-v1")
        self.assertEqual(evidence["pid"], self.pid)
        self.assertEqual(evidence["starttime_ticks"], 1234)
        self.assertFalse(evidence["device_order_observed"])
        self.assertEqual([d["gpu_id"] for d in evidence["devices"]], [10, 11])
        self.assertEqual(evidence["devices"][1]["vram_bytes"], 0)
        self.assertEqual(evidence["devices"][0]["queues"][0]["type"], 0)

    def test_missing_proc_is_rejected(self):
        import shutil
        shutil.rmtree(self.proc / str(self.pid))
        with self.assertRaises(att.AttestationError):
            self.capture()

    def test_executable_mismatch_is_rejected(self):
        with self.assertRaisesRegex(att.AttestationError, "executable mismatch"):
            att.capture_linux_kfd_process_evidence(
                self.pid, self.other_binary, proc_root=self.proc, kfd_root=self.kfd, drm_root=self.drm,
            )

    def test_executable_change_after_capture_is_rejected(self):
        def starttime_with_change(path):
            if not self.exe_changed:
                self.exe_changed = True
                return 1234
            return 1234
        self.exe_changed = False
        with patch.object(att, "_linux_kfd_starttime", side_effect=starttime_with_change):
            with self.assertRaisesRegex(att.AttestationError, "executable changed"):
                self.capture()

    def test_cpu_node_is_skipped_and_gpu_is_required(self):
        node = self.kfd / "topology/nodes/2"
        node.mkdir()
        (node / "gpu_id").write_text("0", encoding="ascii")
        evidence = self.capture()
        self.assertEqual(len(evidence["devices"]), 2)
        (self.kfd / "topology/nodes/0/gpu_id").write_text("0", encoding="ascii")
        (self.kfd / "topology/nodes/1/gpu_id").write_text("0", encoding="ascii")
        with self.assertRaisesRegex(att.AttestationError, "no GPU nodes"):
            self.capture()

    def test_nested_bridge_path_uses_leaf_bdf(self):
        self.assertEqual(self.capture()["devices"][0]["bdf"], "0000:01:00.0")

    def test_zombie_process_is_rejected(self):
        (self.proc / str(self.pid) / "stat").write_text(") " + " ".join(["Z"] + ["0"] * 18 + ["1234"]) + "\n", encoding="ascii")
        with self.assertRaisesRegex(att.AttestationError, "zombie"):
            self.capture()

    def test_negative_property_or_vram_is_rejected(self):
        (self.kfd / "topology/nodes/1/properties").write_text("drm_render_minor -1\ngfx_target_version 1100\n", encoding="ascii")
        with self.assertRaisesRegex(att.AttestationError, "unsupported"):
            self.capture()
        (self.kfd / "topology/nodes/1/properties").write_text("drm_render_minor 129\ngfx_target_version 1100\n", encoding="ascii")
        (self.kfd / "proc" / str(self.pid) / "vram_11").write_text("-1\n", encoding="ascii")
        with self.assertRaisesRegex(att.AttestationError, "unsupported"):
            self.capture()

    def test_pid_reuse_is_rejected_when_starttime_changes(self):
        with patch.object(att, "_linux_kfd_starttime", side_effect=(1234, 1235)):
            with self.assertRaisesRegex(att.AttestationError, "PID reuse"):
                self.capture()

    def test_unknown_queue_gpu_is_rejected(self):
        queue = self.kfd / "proc" / str(self.pid) / "queues" / "9"
        queue.mkdir()
        (queue / "gpuid").write_text("99", encoding="ascii")
        (queue / "size").write_text("1", encoding="ascii")
        (queue / "type").write_text("0", encoding="ascii")
        with self.assertRaisesRegex(att.AttestationError, "unknown gpu_id"):
            self.capture()


if __name__ == "__main__":
    unittest.main()
