"""HI99: source-contract tests for GGML_HIP_TUNER_CONFIG_FIELDS, the single
source of truth generating both the tuner's env-override parsing and its
measurements.jsonl header-emission fprintf from one field list.

Before this macro, the two were independently hand-maintained and had
already drifted: 11 of the ~17 direct scalar env-overridable fields were
never emitted into the header at all, so a real tuning run's own provenance
did not record that they had been set -- the same failure-mode class as the
historical incident where the `compiler` header field silently went missing
for weeks. Real-hardware validation (gfx1100, ROCm 7.1): a run with
GGML_HIP_TUNE_NOISE_PCT/MAX_WORKSPACE/ELAPSED_RETRY all set to non-default
values produced a header where all three now appear with the exact
overridden values, and existing wire names (alpha, not confidence_alpha)
are unchanged; alpha now records at full precision (0.025000000000000001,
not the old 0.0250) since macro-generated doubles use %.17g. See HI99.md
for the full gpt-dev-agent design conversation and real-run evidence.

Source-contract only, matching this repo's existing pattern for .cu/.cuh
files -- no HIP compiler is assumed available offline (though one was used
to real-hardware-validate this change).
"""

from __future__ import annotations

import re
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CUH = ROOT / "src" / "ggml" / "src" / "ggml-cuda" / "hip-autotune-tuner.cuh"
CU = ROOT / "src" / "ggml" / "src" / "ggml-cuda" / "hip-autotune-tuner.cu"

# Fields the macro must cover, in declared order: (TYPE, cpp_field, wire_key).
EXPECTED_FIELDS = [
    ("INT", "final_samples", "final_samples"),
    ("INT", "screen_samples", "screen_samples"),
    ("SIZE", "max_workspace_bytes", "max_workspace_bytes"),
    ("DOUBLE", "noise_canary_pct", "noise_canary_pct"),
    ("INT", "double_native", "double_native"),
    ("DOUBLE", "hot_share_pct", "hot_share_pct"),
    ("DOUBLE", "confidence_alpha", "alpha"),
    ("DOUBLE", "noisy_mad_ratio", "noisy_mad_ratio"),
    ("INT", "verify_determinism", "verify_determinism"),
    ("INT", "emit_samples", "emit_samples"),
    ("INT", "pilot_samples", "pilot_samples"),
    ("DOUBLE", "min_sample_us", "min_sample_us"),
    ("INT", "max_launches_per_sample", "max_launches_per_sample"),
    ("INT", "elapsed_time_retry_max", "elapsed_time_retry_max"),
    ("DOUBLE", "elapsed_time_retry_backoff_us", "elapsed_time_retry_backoff_us"),
    ("INT", "confirmation_samples", "confirmation_samples"),
    ("INT", "flush_evict_mb", "flush_evict_mb"),
]

# Fields the header legitimately carries that are NOT macro-driven: resolved
# flush state (hand-resolved from two mutually-exclusive request flags, not
# a 1:1 scalar override) and other tuner-config fields that were already
# hand-written before HI99 and stay that way (not env-overridable at all).
NON_MACRO_CONFIG_KEYS = {
    "warmup_launches", "replacement_threshold_pct",
    "production_policy", "active_policies", "flush_l2", "pre_sample_mode",
}

# Header keys that are build/workload identity or measured values, never
# tuner config at all.
NON_CONFIG_KEYS = {
    "kind", "artifact_version", "source_revision", "manifest_hash",
    "compiler", "hip_version", "variant_set", "build_descriptor_hash",
    "host_sync_overhead_us", "host_sync_overhead_valid",
    "workload", "workload_label",
}


def _macro_table_text(src: str) -> str:
    start = src.index("#define GGML_HIP_TUNER_CONFIG_FIELDS(F)")
    end = src.index("\n\nconst ggml_hip_tuner_config & ggml_hip_tuner_get_config();", start)
    return src[start:end]


class Hi99TunerConfigMacroTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cuh = CUH.read_text(encoding="utf-8")
        cls.cu = CU.read_text(encoding="utf-8")
        cls.macro_text = _macro_table_text(cls.cuh)

    def test_macro_declares_every_expected_field_in_order(self):
        rows = re.findall(
            r"F\((\w+),\s*(\w+),\s*\"([^\"]+)\"", self.macro_text
        )
        self.assertEqual(
            [(t, f, w) for t, f, w in rows], EXPECTED_FIELDS,
            "GGML_HIP_TUNER_CONFIG_FIELDS field list changed shape or order",
        )

    def test_alpha_wire_key_differs_from_cpp_field_deliberately(self):
        # The one field where JSON name and C++ field name genuinely differ
        # -- proves the macro carries an explicit wire_key column rather than
        # deriving one from cpp_field (which would either rename the wire
        # format or require a second hidden mapping).
        self.assertIn('confidence_alpha,"alpha"', self.macro_text.replace(" ", ""))

    def test_existing_bounds_preserved_exactly(self):
        # A representative sample of the pre-existing bespoke bounds, chosen
        # because getting any of them wrong silently changes what env values
        # are accepted -- not just a cosmetic refactor risk.
        self.assertIn('"GGML_HIP_TUNE_FINAL_SAMPLES",             2,    100000', self.macro_text)
        self.assertIn('"GGML_HIP_TUNE_ALPHA",                     0.0,  1.0', self.macro_text)
        self.assertIn('"GGML_HIP_TUNE_ELAPSED_RETRY",             0,    10', self.macro_text)
        self.assertIn('"GGML_HIP_TUNE_ELAPSED_RETRY_BACKOFF_US", -1.0,', self.macro_text)
        self.assertIn('"GGML_HIP_TUNE_HOT_SHARE",                 0.0,  100.0', self.macro_text)

    def test_flush_l2_and_flush_rewarm_are_not_macro_fields(self):
        # These resolve two mutually-exclusive request flags into one enum
        # plus a derived wire mirror -- not a 1:1 scalar override -- and stay
        # hand-written by design (see the macro's own comment).
        self.assertNotIn("GGML_HIP_TUNE_FLUSH_L2", self.macro_text)
        self.assertNotIn("GGML_HIP_TUNE_FLUSH_REWARM", self.macro_text)
        # But the hand-written resolution must still exist in the .cu file.
        self.assertIn('getenv("GGML_HIP_TUNE_FLUSH_L2")', self.cu)
        self.assertIn('getenv("GGML_HIP_TUNE_FLUSH_REWARM")', self.cu)

    def test_macro_expanded_at_the_env_override_site(self):
        self.assertEqual(
            self.cu.count(
                "GGML_HIP_TUNER_CONFIG_FIELDS(GGML_HIP_TUNER_APPLY_ENV_ONE)"
            ),
            1,
        )
        idx = self.cu.index("GGML_HIP_TUNER_CONFIG_FIELDS(GGML_HIP_TUNER_APPLY_ENV_ONE)")
        fn_idx = self.cu.index("const ggml_hip_tuner_config & ggml_hip_tuner_get_config()")
        self.assertGreater(idx, fn_idx)
        self.assertLess(idx - fn_idx, 6000)

    def test_macro_expanded_at_both_header_emission_sites(self):
        self.assertEqual(
            self.cu.count(
                "GGML_HIP_TUNER_CONFIG_FIELDS(GGML_HIP_TUNER_HEADER_FMT_ONE)"
            ),
            1,
        )
        self.assertEqual(
            self.cu.count(
                "GGML_HIP_TUNER_CONFIG_FIELDS(GGML_HIP_TUNER_HEADER_ARG_ONE)"
            ),
            1,
        )

    def test_no_outer_getenv_guard_around_the_macro_invocation(self):
        # Each *_env() lambda already no-ops on a missing variable, so an
        # outer `if (getenv(...))` around the macro-generated block would be
        # redundant, not protective -- confirm it was actually dropped.
        idx = self.cu.index("GGML_HIP_TUNER_CONFIG_FIELDS(GGML_HIP_TUNER_APPLY_ENV_ONE)")
        window = self.cu[max(0, idx - 200):idx]
        self.assertNotIn("if (const char * v = getenv(", window)

    def test_header_double_fields_use_full_precision_not_display_precision(self):
        self.assertIn('wire_key "\\":%.17g,"', self.cu)

    def test_governed_header_config_keys_equal_macro_plus_declared_exceptions(self):
        # The header legitimately contains non-config provenance too (build
        # identity, measured overhead, workload) -- equality must hold over
        # the GOVERNED config portion only, not the whole JSON object, per
        # the agreed design (a literal whole-header equality would be wrong;
        # a bare subset check would miss a hand-added config field that
        # bypasses the registry in the opposite direction).
        # Derived from the actual macro rows, not the EXPECTED_FIELDS
        # constant: test_macro_declares_every_expected_field_in_order proves
        # the two are equal today, but deriving independently here means a
        # future one-row addition to the macro table only needs that one
        # edit -- not also a matching edit to this test's own field list
        # (per dev-gpt-agent post-implementation review, req_c40a1876389a469b,
        # verdict CLOSE with this as a non-blocking hardening suggestion).
        macro_rows = re.findall(r"F\(\w+,\s*\w+,\s*\"([^\"]+)\"", self.macro_text)
        self.assertEqual(len(macro_rows), len(set(macro_rows)),
                          "duplicate wire_key in GGML_HIP_TUNER_CONFIG_FIELDS")
        macro_keys = set(macro_rows)
        governed = macro_keys | NON_MACRO_CONFIG_KEYS

        fn_idx = self.cu.index("void ggml_hip_tuner_flush(void)") \
            if "void ggml_hip_tuner_flush(void)" in self.cu \
            else self.cu.index("ggml_hip_tuner_flush(")
        fprintf_idx = self.cu.index('"{\\"kind\\":\\"header\\"', fn_idx)
        end_idx = self.cu.index("#undef GGML_HIP_TUNER_HEADER_ARG_ONE", fprintf_idx)
        block = self.cu[fprintf_idx:end_idx]

        # The macro-driven keys are never literal text at this call site --
        # they arrive via GGML_HIP_TUNER_CONFIG_FIELDS(...)'s expansion, whose
        # wire keys live in the .cuh table (already captured in macro_keys).
        # What's literally spelled out in the fprintf block is everything
        # ELSE: the non-macro config exceptions plus true provenance fields.
        literal_key_list = re.findall(r'\\"(\w+)\\":', block)
        self.assertEqual(len(literal_key_list), len(set(literal_key_list)),
                          "duplicate literal header key in the hand-written fprintf block")
        literal_keys = set(literal_key_list)
        self.assertTrue(
            macro_keys.isdisjoint(literal_keys),
            f"macro-driven keys duplicated as hand-written literals: "
            f"{macro_keys & literal_keys} -- this recreates two sources of truth",
        )
        all_expected = governed | NON_CONFIG_KEYS
        self.assertEqual(literal_keys | macro_keys, all_expected,
                          "header's key set (hand-written literals + macro-driven "
                          "wire keys) no longer matches the declared governed-config "
                          "+ provenance exceptions -- either a field was added/"
                          "removed by hand, or this test's own exception lists need "
                          "updating")

    def test_macro_field_count_matches_expected(self):
        self.assertEqual(len(EXPECTED_FIELDS), 17)


if __name__ == "__main__":
    unittest.main()
