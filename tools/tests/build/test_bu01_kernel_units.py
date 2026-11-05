"""BU01: independent mailbox HIP unit and unchanged inverse-map semantics."""
from pathlib import Path
import re
import unittest

ROOT = Path(__file__).resolve().parents[3]
COMPONENT = ROOT / "engines/radiance/components/moe-remote"


class MailboxUnitTests(unittest.TestCase):
    def test_inverse_map_is_isolated_into_own_hip_source(self):
        cmake = (COMPONENT / "CMakeLists.txt").read_text()
        src = (COMPONENT / "moe_mailbox_ops.hip").read_text()
        split = (COMPONENT / "mailbox_inverse_kernel.hip").read_text()
        self.assertIn("HIP_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/moe_mailbox_ops.hip", cmake)
        self.assertIn("${CMAKE_CURRENT_SOURCE_DIR}/mailbox_inverse_kernel.hip", cmake)
        self.assertNotIn("__global__ void mailbox_inverse_kernel", src)
        self.assertIn("hipLaunchKernelGGL(r3_mailbox_inverse::mailbox_inverse_kernel,", src)
        self.assertRegex(split, r"__global__ void mailbox_inverse_kernel\(")

    def test_kernel_operation_and_thread_mapping_are_unchanged(self):
        src = (COMPONENT / "mailbox_inverse_kernel.hip").read_text()
        def compact(text):
            return re.sub(r"\s+", "", text)
        required = (
            "const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;",
            "if (i >= slots) return;",
            "const int32_t flat = sorted[i];",
            "if (flat >= 0 && (uint32_t)flat < slots) inv[flat] = (int32_t)i;",
        )
        code = compact(src)
        for fragment in required:
            self.assertIn(compact(fragment), code)

    def test_kernel_source_does_not_include_monolithic_headers(self):
        src = (COMPONENT / "mailbox_inverse_kernel.hip").read_text()
        self.assertNotIn("moe_remote_shm.h", src)
        self.assertNotIn("moe_remote_hook.h", src)
        self.assertNotIn("moe_lane_gemm.h", src)


if __name__ == "__main__":
    unittest.main()
