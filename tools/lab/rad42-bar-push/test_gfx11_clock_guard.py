"""Card-free regression against GFX11's broken clock64 deadline primitive.

This guards the tested lab waiter implementations only, not every HIP kernel
in BigCherry. A real GPU/host timeout check remains mandatory as well.
"""
from pathlib import Path
import re
import unittest

ROOT=Path(__file__).resolve().parent
SOURCES=("live_control.hip.h","live_relay.hip.h","live_rounds.hip.h")
RUNNER="bar_dmabuf_ab.hip"

class Gfx11ClockGuardTests(unittest.TestCase):
    def test_live_waiters_use_wallclock_not_clock64(self):
        for name in SOURCES:
            source=(ROOT/name).read_text()
            # Strip comments to avoid matching the explanatory warning.
            source=re.sub(r"//[^\n]*|/\*.*?\*/","",source,flags=re.DOTALL)
            self.assertNotRegex(source,r"(?<![\w])clock64\s*\(",
                                f"{name}: GFX11 uses wall_clock64 for watchdogs")

    def test_device_waiters_have_wallclock_deadline(self):
        for name in ("live_control.hip.h","live_relay.hip.h"):
            source=(ROOT/name).read_text()
            self.assertIn("wall_clock64()",source)
            self.assertIn("deadline_wall_ticks",source)
            self.assertIn("__HIP_MEMORY_SCOPE_SYSTEM",source)

    def test_rate_is_runtime_queried_and_checked(self):
        source=(ROOT/RUNNER).read_text()
        self.assertIn("hipDeviceAttributeWallClockRate",source)
        self.assertIn("gpu_wallclock_khz<=0",source)
        self.assertIn("static_cast<uint64_t>(gpu_wallclock_khz)*3000ull",source)

    def test_independent_host_timeouts_and_exact_ticket(self):
        for name in ("live_rounds.hip.h","live_relay.hip.h"):
            source=(ROOT/name).read_text()
            self.assertIn("host_wait_ticket(",source)
            self.assertIn("hipDeviceSynchronize()",source)
        source=(ROOT/"live_control.hip.h").read_text()
        self.assertIn("seen!=gen",source)

if __name__=="__main__":
    unittest.main()
