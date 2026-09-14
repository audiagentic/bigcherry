"""PA26 offline oracle: composition-delta proof + fixed-corpus binding.

Against the real recipes.toml/patch catalog, like test_campaign_resolution.py
and test_campaign_source.py -- resolution rejects any injected catalog whose
patch-id set does not exactly match the physical catalog on disk.
"""

from __future__ import annotations

import dataclasses
import struct
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from bigcherry.core import config as campaign_config  # noqa: E402
from bigcherry.core import paths  # noqa: E402
from bigcherry.patch import patchset  # noqa: E402
from bigcherry.campaign import replay_equivalence as re  # noqa: E402
from bigcherry.tuning import replay as replay_module  # noqa: E402

_RECIPES_PATH = paths.RECIPES


def _make_cache_blob(*, entry_count: int = 1, version: int | None = None) -> bytes:
    """Hand-craft a minimal, structurally valid v5 replay cache blob (one
    entry, one string), mirroring replay.build()'s own wire layout, so
    offline tests exercise the real reader without needing a GPU-produced
    measurements/manifest pipeline."""
    name = b"winner-a\0"
    entry = (
        b"\x11" * 16  # dispatch digest
        + b"\x22" * 16  # signature digest
        + struct.pack("<I", 0)  # name_offset into string table
        + struct.pack("<H", 1)  # implementation_version
        + struct.pack("<iii", 0, 0, 0)  # primary, secondary, width
        + struct.pack("<BBBB", 0, 0, 0, 0)  # acc_f16, fallback, small_k, src0_type
        + b"\x33" * 16  # manifest hash
        + b"\x44" * 16  # source revision digest
        + struct.pack("<I", 0)  # generation
        + struct.pack("<H", 0)  # transform_id
        + struct.pack("<B", 0)  # match_kind
    )
    assert len(entry) == replay_module.ENT_SIZE, len(entry)
    payload = entry * entry_count + name
    header = bytearray()
    header += struct.pack(
        "<III", replay_module.MAGIC, version or replay_module.REPLAY_VERSION,
        replay_module.ARTIFACT_VERSION,
    )
    header += struct.pack(
        "<HH", replay_module.SIGNATURE_SCHEMA_VERSION,
        replay_module.HARDWARE_SCHEMA_VERSION,
    )
    header += struct.pack("<II", entry_count, len(name))
    header += b"\x55" * 16  # manifest_hash
    header += replay_module.blake2b_digest(payload)
    assert len(header) == replay_module.REPLAY_HEADER_SIZE
    return bytes(header) + payload


class ServingCoreConfigTests(unittest.TestCase):
    """Real recipes.toml/catalog -- the ephemeral candidate must never be
    written back to disk, and must match the declared PA20 boundary."""

    @classmethod
    def setUpClass(cls):
        cls.cfg = campaign_config.load(_RECIPES_PATH)
        cls.catalog = patchset.catalog()

    def test_recipes_toml_is_untouched_on_disk(self):
        before = _RECIPES_PATH.read_bytes()
        re.build_serving_core_config(self.cfg)
        after = _RECIPES_PATH.read_bytes()
        self.assertEqual(before, after)

    def test_candidate_config_does_not_mutate_original(self):
        candidate_cfg = re.build_serving_core_config(self.cfg)
        self.assertNotIn(re.SERVING_CORE_SOURCE_NAME, self.cfg.sources)
        self.assertIn(re.SERVING_CORE_SOURCE_NAME, candidate_cfg.sources)
        self.assertNotIn(re.SERVING_CORE_PATCH_SET_NAME, self.cfg.patch_sets)

    def test_candidate_excludes_exactly_the_expected_modules(self):
        candidate_cfg = re.build_serving_core_config(self.cfg)
        candidate_ids = set(
            candidate_cfg.patch_sets[re.SERVING_CORE_PATCH_SET_NAME].patches
        )
        framework_ids = set(self.cfg.patch_sets["framework"].patches)
        self.assertEqual(
            framework_ids - candidate_ids, set(re.EXPECTED_REMOVED_MODULES)
        )
        self.assertEqual(candidate_ids - framework_ids, set())

    def test_stale_expected_removed_modules_fails_closed(self):
        pruned_framework = dataclasses.replace(
            self.cfg.patch_sets["framework"],
            patches=tuple(
                pid
                for pid in self.cfg.patch_sets["framework"].patches
                if pid != re.EXPECTED_REMOVED_MODULES[0]
            ),
        )
        stale_cfg = dataclasses.replace(
            self.cfg,
            patch_sets={**self.cfg.patch_sets, "framework": pruned_framework},
        )
        with self.assertRaises(re.ReplayEquivalenceError):
            re.build_serving_core_config(stale_cfg)


class CompositionDeltaTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg = campaign_config.load(_RECIPES_PATH)
        cls.catalog = patchset.catalog()

    def test_real_composition_delta_matches_pa20_boundary(self):
        delta = re.resolve_composition_delta(self.cfg, self.catalog)
        re.require_expected_composition_delta(delta)  # must not raise
        self.assertEqual(delta.removed, re.EXPECTED_REMOVED_MODULES)
        self.assertEqual(delta.added, ())

    def test_negative_missing_serving_module_is_detected(self):
        """The most important PA26 negative fixture (per design review):
        drop a genuine serving module (0650) from the candidate and prove
        the oracle fails rather than silently accepting a narrower arm."""
        candidate_cfg = re.build_serving_core_config(self.cfg)
        narrowed = dataclasses.replace(
            candidate_cfg.patch_sets[re.SERVING_CORE_PATCH_SET_NAME],
            patches=tuple(
                pid
                for pid in candidate_cfg.patch_sets[
                    re.SERVING_CORE_PATCH_SET_NAME
                ].patches
                if pid != "0650_mmvq_native_variant"
            ),
        )
        candidate_cfg = dataclasses.replace(
            candidate_cfg,
            patch_sets={
                **candidate_cfg.patch_sets,
                re.SERVING_CORE_PATCH_SET_NAME: narrowed,
            },
        )
        delta = re.resolve_composition_delta(
            self.cfg, self.catalog, candidate_cfg=candidate_cfg
        )
        with self.assertRaises(re.ReplayEquivalenceError):
            re.require_expected_composition_delta(delta)

    def test_negative_unexpected_extra_module_is_detected(self):
        """Candidate accidentally carrying a module the control arm never
        selected (e.g. an RD-experimental patch) must fail closed."""
        candidate_cfg = re.build_serving_core_config(self.cfg)
        widened = dataclasses.replace(
            candidate_cfg.patch_sets[re.SERVING_CORE_PATCH_SET_NAME],
            patches=candidate_cfg.patch_sets[re.SERVING_CORE_PATCH_SET_NAME].patches
            + ("1200_rd19_single_gpu_meta_bypass",),
        )
        candidate_cfg = dataclasses.replace(
            candidate_cfg,
            patch_sets={
                **candidate_cfg.patch_sets,
                re.SERVING_CORE_PATCH_SET_NAME: widened,
            },
        )
        delta = re.resolve_composition_delta(
            self.cfg, self.catalog, candidate_cfg=candidate_cfg
        )
        with self.assertRaises(re.ReplayEquivalenceError):
            re.require_expected_composition_delta(delta)


class WinnersCorpusTests(unittest.TestCase):
    def test_real_v5_blob_reads_back_with_entries(self):
        with tempfile.TemporaryDirectory() as directory:
            cache_path = Path(directory) / "winners.cache"
            cache_path.write_bytes(_make_cache_blob())
            corpus = re.load_winners_corpus(cache_path)
            self.assertEqual(len(corpus.entries), 1)
            self.assertEqual(corpus.header["version"], replay_module.REPLAY_VERSION)
            self.assertEqual(len(corpus.sha256), 64)

    def test_empty_cache_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            cache_path = Path(directory) / "empty.cache"
            cache_path.write_bytes(_make_cache_blob(entry_count=0))
            with self.assertRaises(re.ReplayEquivalenceError):
                re.load_winners_corpus(cache_path)

    def test_wrong_version_cache_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            cache_path = Path(directory) / "stale.cache"
            cache_path.write_bytes(_make_cache_blob(version=4))
            with self.assertRaises(SystemExit):
                re.load_winners_corpus(cache_path)


class OfflineReceiptTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg = campaign_config.load(_RECIPES_PATH)
        cls.catalog = patchset.catalog()

    def test_receipt_binds_identities_and_stays_not_evaluated(self):
        with tempfile.TemporaryDirectory() as directory:
            cache_path = Path(directory) / "winners.cache"
            cache_path.write_bytes(_make_cache_blob())
            receipt = re.build_offline_receipt(
                self.cfg,
                self.catalog,
                cache_path,
                bigcherry_revision="0" * 40,
            )
            self.assertEqual(receipt["schema_version"], re.RECEIPT_SCHEMA_VERSION)
            self.assertEqual(
                receipt["expected_composition_delta"],
                list(re.EXPECTED_REMOVED_MODULES),
            )
            self.assertEqual(
                receipt["actual_composition_delta"], receipt["expected_composition_delta"]
            )
            self.assertEqual(receipt["winner_count"], 1)
            self.assertEqual(len(receipt["winners_sha256"]), 64)
            self.assertTrue(receipt["decision_equivalent"])
            self.assertEqual(receipt["differences"], [])
            self.assertEqual(receipt["runtime"], {"status": "NOT_EVALUATED"})
            self.assertNotEqual(
                receipt["control_selector"]["patch_set_id"],
                receipt["candidate_selector"]["patch_set_id"],
            )

    def test_receipt_fails_closed_on_empty_corpus(self):
        with tempfile.TemporaryDirectory() as directory:
            cache_path = Path(directory) / "empty.cache"
            cache_path.write_bytes(_make_cache_blob(entry_count=0))
            with self.assertRaises(re.ReplayEquivalenceError):
                re.build_offline_receipt(
                    self.cfg, self.catalog, cache_path, bigcherry_revision="0" * 40
                )


if __name__ == "__main__":
    unittest.main()
