"""HI34 step 3 (Slice B0): cache-eviction experiment mechanism, source contract tests.

Source-contract tests pinning the structural invariants GPT's close-out
verdict requires of B0, independently of GPU runtime testing:

- default OFF, explicit env parsing for both knobs
- explicit megabyte sizing rather than 2 x hipDeviceAttributeL2CacheSize
  (which reports only the 6 MB L2 on Navi 31 and would size a flush against
  the wrong cache level)
- eviction enqueued before hipEventRecord(start) on the measurement stream,
  so none of it is inside the timed window
- flush mode threaded as a required parameter through every measurement path
  (time_candidate, run_counterbalanced_round), so a path that silently
  measures differently from its siblings is a compile error
- flush + batching cannot coexist: get_config forces max_launches_per_sample
  to 1 when flushing, where every calibration reads its ceiling
- fail-closed eviction: a requested flush whose buffer allocation failed must
  not proceed as an unflushed measurement
- flush configuration recorded in the artifact header, because flush=0 and
  flush=1 artifacts are not measurement-equivalent even with identical
  build/input digests
"""

from __future__ import annotations

import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TUNER = ROOT / "src" / "ggml" / "src" / "ggml-cuda" / "hip-autotune-tuner.cu"
TUNER_H = ROOT / "src" / "ggml" / "src" / "ggml-cuda" / "hip-autotune-tuner.cuh"


class FlushContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tuner = TUNER.read_text(encoding="utf-8")
        cls.tuner_h = TUNER_H.read_text(encoding="utf-8")

    # --- configuration ----------------------------------------------------

    def test_flush_defaults_off_with_explicit_mb_sizing(self):
        self.assertIn("int flush_l2       = 0;", self.tuner_h)
        self.assertIn("int flush_evict_mb = 256;", self.tuner_h)
        self.assertIn(
            'int_env("GGML_HIP_TUNE_FLUSH_L2", 0, 1, c.flush_l2);', self.tuner
        )
        self.assertIn(
            'int_env("GGML_HIP_TUNE_FLUSH_MB", 1, 65536, c.flush_evict_mb);',
            self.tuner,
        )

    def test_sizing_never_derived_from_the_l2_attribute(self):
        # The eviction buffer must come from the explicit megabyte count. A
        # hipDeviceGetAttribute call anywhere in the tuner would be a sizing
        # source that knows nothing about the Infinity Cache behind the L2.
        self.assertNotIn("hipDeviceGetAttribute", self.tuner)
        self.assertIn(
            "(size_t) std::max(\n"
            "        ggml_hip_tuner_get_config().flush_evict_mb, 1) << 20",
            self.tuner,
        )

    # --- eviction placement -------------------------------------------------

    def test_eviction_is_outside_the_timed_window(self):
        # Inside time_candidate's sample loop the eviction must come after
        # the loop opens and before hipEventRecord(start), AND before the
        # host clock starts: stream ordering alone keeps it out of the GPU
        # event interval, but the tuner also ranks on host wall time, and an
        # unfinished 256 MB eviction would ride inside that window via the
        # final hipEventSynchronize(stop).
        loop = self.tuner.find("for (int s = 0; s < samples; ++s)")
        evict = self.tuner.find("if (flush_l2 && !launch_cache_evict(lc))")
        host_start = self.tuner.find("const int64_t host_start = ggml_time_us();")
        record_start = self.tuner.find("hipEventRecord(start, lc.stream)")
        self.assertGreater(loop, 0)
        self.assertGreater(evict, loop)
        self.assertGreater(host_start, evict)
        self.assertGreater(record_start, evict)

    def test_eviction_completes_before_any_measurement_clock(self):
        # GPT B0 review (P0): enqueueing the eviction is not enough -- the
        # helper must check a stream synchronization so the eviction has
        # actually finished before launch_cache_evict returns true and the
        # sample's host_start / event pair begin.
        start = self.tuner.find("static bool launch_cache_evict")
        end = self.tuner.find("bool time_candidate(", start)
        fn = self.tuner[start:end]
        kernel = fn.find("ggml_hip_cache_evict<<<1024, 256, 0, lc.stream>>>")
        sync = fn.find("hipStreamSynchronize(lc.stream)")
        ok_ret = fn.rfind("return true;")
        self.assertGreater(kernel, 0)
        self.assertGreater(sync, kernel,
                           "eviction must complete before launch_cache_evict returns")
        self.assertGreater(ok_ret, sync)
        # The synchronization is checked, not fire-and-forget.
        sync_block = fn[sync:sync + 400]
        self.assertIn("!= hipSuccess", sync_block)
        self.assertIn("return false;", sync_block)

    def test_eviction_uses_volatile_stores(self):
        # A kernel whose only effect is dead stores flushes nothing and fails
        # silently; the store must be volatile.
        self.assertIn(
            "__global__ void ggml_hip_cache_evict(volatile char * __restrict__ buffer,",
            self.tuner,
        )

    def test_buffer_is_device_scoped_and_fail_closed(self):
        # GPT B0 review (P1): hipMalloc allocations are device-specific. A
        # single process-lifetime pointer would let a later GPU launch the
        # eviction with a foreign pointer -- invalid without peer access and
        # wrong for this experiment regardless. The storage must be keyed by
        # device, with the current device resolved at the eviction seam.
        self.assertNotIn("static char * buffer", self.tuner)
        self.assertIn("struct ggml_hip_flush_evict_slot {", self.tuner)
        self.assertIn("slot.device == device", self.tuner)
        seam = self.tuner.find("static bool launch_cache_evict")
        block = self.tuner[seam:seam + 700]
        self.assertIn("hipGetDevice(&device)", block)
        # A requested flush whose allocation failed returns false: the sample
        # is rejected, never silently measured unflushed.
        fn_end = self.tuner.find("bool time_candidate(", seam)
        fn = self.tuner[seam:fn_end]
        self.assertIn("buffer == nullptr", fn)
        self.assertIn("return false;", fn)

    # --- threading through every measurement path ---------------------------

    def test_flush_is_a_required_parameter_of_both_measurement_functions(self):
        # Required, not defaulted: a caller that forgets to choose a mode is
        # a compile error rather than a silent default.
        self.assertIn(
            "bool flush_l2,\n                    std::vector<double> & gpu_us,",
            self.tuner,
        )
        self.assertIn(
            "int launches_per_sample,\n"
            "        bool flush_l2,\n"
            "        const char * protocol_stage) {",
            self.tuner,
        )
        self.assertNotIn("bool flush_l2 =", self.tuner)

    def test_every_call_site_states_the_flush_mode(self):
        # Six direct decision sites (pilot, native, screening, final round,
        # canary re-measure, confirmation) read the config explicitly; the
        # counterbalanced round forwards its parameter to time_candidate.
        self.assertEqual(
            self.tuner.count("config.flush_l2 != 0"),
            6,
            "expected pilot + native + screening + final + canary-remeasure "
            "+ confirmation call sites",
        )
        self.assertIn(
            "launches_per_sample, flush_l2,\n"
            "                                one_gpu, one_host,",
            self.tuner,
        )

    # --- flush/batch mutual exclusion ---------------------------------------

    def test_flush_forces_one_launch_per_sample(self):
        # Enforced in get_config, where every calibration reads its ceiling:
        # a batched sample would measure one cold launch plus lps-1 hot ones
        # and report the mean as if it were one number.
        clamp = self.tuner.find("if (c.flush_l2 != 0 && c.max_launches_per_sample > 1)")
        self.assertGreater(clamp, 0)
        block = self.tuner[clamp : clamp + 900]
        self.assertIn("c.max_launches_per_sample = 1;", block)
        self.assertIn("a flush cannot ", block)

    # --- provenance ----------------------------------------------------------

    def test_flush_configuration_is_recorded_in_the_artifact_header(self):
        # The C string literal carries escaped quotes; match them literally.
        header = self.tuner.find('\\"flush_l2\\":%d,\\"flush_evict_mb\\":%d}')

        self.assertGreater(header, 0)
        self.assertIn("config.flush_l2, config.flush_evict_mb", self.tuner)
        # The rationale must stay next to the emission: flush=0 and flush=1
        # artifacts are not measurement-equivalent.
        block = self.tuner[header : header + 1600]
        self.assertIn("not measurement-equivalent", block)


if __name__ == "__main__":
    unittest.main()
