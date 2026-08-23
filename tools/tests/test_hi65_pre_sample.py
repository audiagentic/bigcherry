"""HI65 step 1: pre-sample intervention enum, source contract tests.

Source-contract tests pinning the structural invariants of the single-enum
pre-sample intervention (NONE / EVICT / EVICT_REWARM):

- the enum exists and is the default OFF mode (GGML_HIP_PRE_SAMPLE_NONE)
- measurement code branches on the enum, not on independent booleans: an
  invalid combination (e.g. evict + rewarm as separate flags reaching the
  loop) is unrepresentable rather than runtime-rejected
- the two env request flags map into ONE mode in get_config; setting both
  fails closed at parse time (valid = false, tuning disabled)
- EVICT_REWARM runs the untimed rewarm AFTER the eviction and BEFORE the
  measurement clocks (host_start / hipEventRecord(start)), with a checked
  stream synchronization so no tail rides inside host_us
- the resolved mode string is emitted in the artifact header alongside the
  legacy flush_l2 wire mirror
"""

from __future__ import annotations

import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TUNER = ROOT / "src" / "ggml" / "src" / "ggml-cuda" / "hip-autotune-tuner.cu"
TUNER_H = ROOT / "src" / "ggml" / "src" / "ggml-cuda" / "hip-autotune-tuner.cuh"


class PreSampleModeContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tuner = TUNER.read_text(encoding="utf-8")
        cls.tuner_h = TUNER_H.read_text(encoding="utf-8")

    # --- enum definition ----------------------------------------------------

    def test_enum_is_defined_with_none_as_default(self):
        self.assertIn("enum ggml_hip_pre_sample_mode {", self.tuner_h)
        for member in (
            "GGML_HIP_PRE_SAMPLE_NONE = 0,",
            "GGML_HIP_PRE_SAMPLE_EVICT = 1,",
            "GGML_HIP_PRE_SAMPLE_EVICT_REWARM = 2,",
        ):
            self.assertIn(member, self.tuner_h)
        # Production default is the no-intervention mode.
        self.assertIn(
            "ggml_hip_pre_sample_mode pre_sample_mode = GGML_HIP_PRE_SAMPLE_NONE;",
            self.tuner_h,
        )

    # --- single-mode resolution ----------------------------------------------

    def test_env_flags_resolve_into_one_mode(self):
        # Both request flags are parsed into locals, never straight into the
        # config: the tuner core must branch on exactly one mode.
        self.assertIn("int flush_l2_req = 0;", self.tuner)
        self.assertIn("int flush_rewarm_req = 0;", self.tuner)
        self.assertNotIn(
            "c.flush_l2 =",
            self.tuner.replace(
                "c.flush_l2 = (c.pre_sample_mode != GGML_HIP_PRE_SAMPLE_NONE) ? 1 : 0;",
                "",
            ),
        )
        self.assertIn(
            "c.pre_sample_mode = GGML_HIP_PRE_SAMPLE_EVICT_REWARM;", self.tuner
        )
        self.assertIn("c.pre_sample_mode = GGML_HIP_PRE_SAMPLE_EVICT;", self.tuner)

    def test_both_flags_set_fails_closed_at_parse_time(self):
        # Two different post-eviction states cannot both be the state.
        both = self.tuner.find("if (flush_l2_req != 0 && flush_rewarm_req != 0)")
        self.assertGreater(both, 0)
        block = self.tuner[both : both + 500]
        self.assertIn("c.valid = false;", block)
        self.assertIn("mutually exclusive", block)

    def test_wire_mirror_is_derived_from_the_mode(self):
        # flush_l2 stays only as the resolved 0/1 wire format for artifact
        # emission and backward compatibility.
        self.assertIn(
            "c.flush_l2 = (c.pre_sample_mode != GGML_HIP_PRE_SAMPLE_NONE) ? 1 : 0;",
            self.tuner,
        )

    def test_measurement_code_branches_on_the_enum_not_a_second_boolean(self):
        # No independent rewarm boolean may reach the measurement loop; the
        # sample loop branches on the enum alone.
        loop = self.tuner.find("for (int s = 0; s < samples; ++s)")
        sample_block = self.tuner[loop : loop + 2000]
        self.assertNotIn("rewarm_l2", sample_block)
        self.assertIn("pre_sample != GGML_HIP_PRE_SAMPLE_NONE", sample_block)

    # --- EVICT_REWARM ordering ------------------------------------------------

    def test_rewarm_sits_between_eviction_and_the_clocks(self):
        # The untimed rewarm launch must come after the eviction, BEFORE
        # host_start and hipEventRecord(start), and carry a checked stream
        # sync: an unsynchronized tail would ride inside host_us via the
        # final hipEventSynchronize(stop).
        loop = self.tuner.find("for (int s = 0; s < samples; ++s)")
        evict = self.tuner.find("&& !launch_cache_evict(lc))")
        rewarm = self.tuner.find("if (pre_sample == GGML_HIP_PRE_SAMPLE_EVICT_REWARM)")
        host_start = self.tuner.find("const int64_t host_start = ggml_time_us();")
        record_start = self.tuner.find("hipEventRecord(start, lc.stream)")
        for pos in (loop, evict, rewarm, host_start, record_start):
            self.assertGreater(pos, 0)
        self.assertGreater(rewarm, evict, "rewarm must follow the eviction")
        self.assertGreater(host_start, rewarm, "rewarm must precede the clocks")
        self.assertGreater(record_start, rewarm)
        # The rewarm block itself: launch -> checked error -> sync -> return.
        block = self.tuner[rewarm : rewarm + 900]
        # HI30: the raw launch is now issued through do_launch(), which
        # transparently routes through a routing transformation when one is
        # in play (GGML_HIP_ROUTING_TRANSFORM) and falls back to the same
        # effective.launch(&effective, lc) otherwise.
        launch = block.find("if (!do_launch()) {")
        sync = block.find("hipStreamSynchronize(lc.stream)")
        # The launch's own failure path returns before the sync exists to
        # check; the sync's failure path is the one that must come after it.
        ok_ret = block.find("return false;", sync)
        self.assertGreater(launch, 0)
        self.assertGreater(sync, launch, "rewarm must be synchronized")
        self.assertGreater(ok_ret, sync)
        self.assertIn("!= hipSuccess", block)

    # --- provenance -----------------------------------------------------------

    def test_resolved_mode_string_is_in_the_artifact_header(self):
        # HI37 Part 2 added workload/workload_label fields after this one, so
        # it is no longer immediately followed by the closing brace -- just
        # confirm the field itself and its value source are present.
        header = self.tuner.find('\\"pre_sample_mode\\":\\"%s\\"')
        self.assertGreater(header, 0)
        self.assertIn("pre_sample_mode_name(config.pre_sample_mode)", self.tuner)

    def test_mode_names_are_stable_strings(self):
        # The provenance values are a fixed vocabulary; the evaluator side
        # (tools/residency_gates.py and the HI65 matrix) keys on them.
        fn = self.tuner.find("static const char * pre_sample_mode_name")
        end = self.tuner.find("bool time_candidate(", fn)
        body = self.tuner[fn:end]
        for name in ('"evict"', '"evict_rewarm"', '"none"'):
            self.assertIn(name, body)


if __name__ == "__main__":
    unittest.main()
