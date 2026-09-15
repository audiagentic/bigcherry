"""PA30 gates 1-3 offline oracle: real-source composition-delta proof and
diagnostic-overlay construction, against the real recipes.toml/catalog.

Distinct from PA26's ``test_replay_equivalence.py`` (which proves PA26's
own ephemeral 8-module-removed candidate) -- this proves the REAL
``bigcherry-native`` vs ``bigcherry-serving-base``/migrated ``bigcherry``
comparison PA30 gates 1-3 actually need, with the real 7-module delta
(0700 excluded, since PA28 kept 0700 in serving-core).
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from bigcherry.core import config as campaign_config  # noqa: E402
from bigcherry.core import paths  # noqa: E402
from bigcherry.patch import patchset  # noqa: E402
from bigcherry.campaign import replay_equivalence_pa30 as pa30  # noqa: E402

_RECIPES_PATH = paths.RECIPES


class RealCompositionDeltaTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.cfg = campaign_config.load(_RECIPES_PATH)
        cls.catalog = patchset.catalog()

    def test_native_vs_serving_base_delta_is_exact_seven_modules(self) -> None:
        delta = pa30.resolve_real_composition_delta(
            self.cfg, self.catalog,
            control_source="bigcherry-native",
            candidate_source="bigcherry-serving-base",
        )
        self.assertEqual(
            frozenset(delta.removed), frozenset(pa30.REAL_EXPECTED_REMOVED_MODULES)
        )
        self.assertEqual(delta.added, ())
        self.assertNotIn("0700_coverage_counters", delta.removed)
        pa30.require_real_expected_composition_delta(delta)

    def test_native_vs_migrated_release_delta_matches_serving_base(self) -> None:
        """source.bigcherry is now serving-core + upstream-fixes (PA29) --
        identical membership to bigcherry-serving-base while
        validated-enhancements is empty (PA29's own recorded finding)."""
        native_vs_serving_base = pa30.resolve_real_composition_delta(
            self.cfg, self.catalog,
            control_source="bigcherry-native",
            candidate_source="bigcherry-serving-base",
        )
        native_vs_release = pa30.resolve_real_composition_delta(
            self.cfg, self.catalog,
            control_source="bigcherry-native",
            candidate_source="bigcherry",
        )
        self.assertEqual(
            native_vs_serving_base.candidate.patch_ids,
            native_vs_release.candidate.patch_ids,
        )
        self.assertEqual(
            native_vs_serving_base.candidate.module_hashes,
            native_vs_release.candidate.module_hashes,
        )
        pa30.require_real_expected_composition_delta(native_vs_release)

    def test_wrong_expected_delta_fails_closed(self) -> None:
        delta = pa30.resolve_real_composition_delta(
            self.cfg, self.catalog,
            control_source="bigcherry-native",
            candidate_source="bigcherry-serving-base",
        )
        with self.assertRaises(pa30.Pa30ReplayEquivalenceError):
            # PA26's own 8-module delta (includes 0700) is NOT the real delta.
            pa30.require_real_expected_composition_delta(
                delta, expected_removed=(
                    "0110_campaign_tune_record_build",
                    "0700_coverage_counters",
                    "0800_server_shutdown_endpoint",
                    "0810_replay_hit_diagnostics",
                    "0820_measurement_signature_shapes",
                    "0830_split_reduce_telemetry",
                    "0900_pool_workspace_metrics",
                    "1100_hi70_direct_op_evidence",
                ),
            )


class DiagnosticOverlayTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.cfg = campaign_config.load(_RECIPES_PATH)
        cls.catalog = patchset.catalog()

    def test_diagnostic_overlay_is_serving_base_plus_0810(self) -> None:
        from bigcherry.campaign import resolution

        production = resolution.resolve_canonical_selection(
            "bigcherry-serving-base", self.cfg, self.catalog
        )
        diagnostic_cfg = pa30.build_serving_base_diagnostic_config(self.cfg)
        diagnostic = resolution.resolve_canonical_selection(
            pa30.SERVING_BASE_DIAGNOSTIC_SOURCE_NAME, diagnostic_cfg, self.catalog
        )
        pa30.require_real_diagnostic_matches_production(
            production.identity.patch_ids, diagnostic.identity.patch_ids
        )
        # Resolution orders by canonical patch id, not insertion order, so
        # 0810 sorts into the middle of the sequence rather than appending
        # at the end -- require_real_diagnostic_matches_production (above)
        # is the real order-preserving-except-insertion check; here just
        # confirm set membership.
        self.assertEqual(
            frozenset(diagnostic.identity.patch_ids),
            frozenset(production.identity.patch_ids) | {"0810_replay_hit_diagnostics"},
        )

    def test_diagnostic_overlay_never_persisted(self) -> None:
        diagnostic_cfg = pa30.build_serving_base_diagnostic_config(self.cfg)
        self.assertNotIn(
            pa30.SERVING_BASE_DIAGNOSTIC_SOURCE_NAME, self.cfg.sources
        )
        self.assertIn(
            pa30.SERVING_BASE_DIAGNOSTIC_SOURCE_NAME, diagnostic_cfg.sources
        )

    def test_mismatched_diagnostic_fails_closed(self) -> None:
        with self.assertRaises(pa30.Pa30ReplayEquivalenceError):
            pa30.require_real_diagnostic_matches_production(
                ("0100_cmake_options",), ("0100_cmake_options", "0200_dispatch_hook"),
            )


if __name__ == "__main__":
    unittest.main()
