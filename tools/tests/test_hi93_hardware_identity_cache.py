"""HI93 (RP4): source-contract tests for the per-device cached hardware
key/digest. A device's hardware key/digest never changes for the life of
the process, so the cold-path resolver should construct it once per device
and reuse it, instead of rebuilding it (including a real blake2b digest)
on every cold signature. Pure caching, no behavior/semantic change.

Source-contract only, matching this repo's existing pattern for .cu files
-- no HIP compiler is assumed available offline (though one was used to
real-hardware-validate this change; see HI93.md's notes).
"""

from __future__ import annotations

import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DISPATCH = ROOT / "src" / "ggml" / "src" / "ggml-cuda" / "hip-autotune-dispatch.cu"


class Hi93HardwareIdentityCacheContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.src = DISPATCH.read_text(encoding="utf-8")

    def test_cache_is_keyed_directly_by_device_not_hipgetdevice(self):
        # Unlike HI64's PerDeviceState<T>, every call site here already has
        # the target device index (ctx.device) -- the cache must not call
        # hipGetDevice() to re-derive what the caller already knows.
        idx = self.src.index("cached_hardware_identity(int device) {")
        body_end = self.src.index("\n}", idx)
        body = self.src[idx:body_end]
        self.assertNotIn("hipGetDevice", body)
        self.assertIn("g_hardware_identity_by_device.find(device)", body)

    def test_cache_stores_both_key_and_digest(self):
        self.assertIn("struct HardwareIdentity {", self.src)
        idx = self.src.index("struct HardwareIdentity {")
        end = self.src.index("};", idx)
        body = self.src[idx:end]
        self.assertIn("ggml_hip_hardware_key_v1 key;", body)
        self.assertIn("ggml_hip_digest digest;", body)

    def test_entries_are_never_erased(self):
        # Reference stability into the map depends on this: unordered_map
        # references/pointers stay valid across insertion as long as no
        # element is ever erased.
        self.assertNotIn("g_hardware_identity_by_device.erase", self.src)

    def test_both_real_call_sites_use_the_cache_not_a_fresh_build(self):
        # ggml_hip_make_hardware_key() itself must only be called from
        # inside the cache's own miss path now -- not directly at either
        # of the two original resolver call sites (forced-candidate path,
        # main resolution path).
        self.assertEqual(self.src.count("ggml_hip_make_hardware_key("), 1)
        fn_idx = self.src.index("cached_hardware_identity(int device) {")
        call_idx = self.src.index("ggml_hip_make_hardware_key(")
        self.assertGreater(call_idx, fn_idx)
        self.assertLess(call_idx - fn_idx, 600)

    def test_forced_candidate_path_reads_the_cached_key(self):
        idx = self.src.index("ForcedCandidate::instance(); forced.candidate != nullptr")
        window = self.src[idx:idx + 200]
        self.assertIn("cached_hardware_identity(ctx.device).key", window)

    def test_main_resolution_path_reads_the_cached_identity(self):
        self.assertIn(
            "const HardwareIdentity & hw_identity = cached_hardware_identity(ctx.device);",
            self.src,
        )
        idx = self.src.index("const HardwareIdentity & hw_identity")
        window = self.src[idx:idx + 300]
        self.assertIn("hw_identity.key", window)
        self.assertIn("hw_identity.digest", window)

    def test_lookup_and_insert_are_mutex_guarded(self):
        idx = self.src.index("cached_hardware_identity(int device) {")
        window = self.src[idx:idx + 200]
        self.assertIn("std::lock_guard<std::mutex> lock(g_hardware_identity_mutex);", window)


if __name__ == "__main__":
    unittest.main()
