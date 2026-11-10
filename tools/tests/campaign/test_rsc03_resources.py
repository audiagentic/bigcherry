"""RSC11 S7 card-free resource ledger tests (no HIP/KFD, no model payload)."""
from __future__ import annotations

import unittest
from dataclasses import replace

from bigcherry.core import paths
from bigcherry.engine_adapters import radiance_inventory as inventory
from bigcherry.engine_adapters import radiance_layout as layouts
from bigcherry.engine_adapters import radiance_resources as resource

M = resource.MIB
ENGINE = inventory.Gpu("0000:03:00.0", 2, "gfx1201", 32624)
XTX = inventory.Gpu("0000:04:00.0", 0, "gfx1100", 24560)
DIGEST = "a" * 64


def selected(gpu=XTX):
    arch = layouts.load_archs()[gpu.arch]
    helper = layouts.HelperCard(gpu.ordinal, arch, gpu.vram_mib, 0, "head", 1, True)
    resolved = layouts.Resolved("test", (ENGINE.ordinal,), ENGINE.arch, "libr4d.so",
                                "models/model.rad", None, (), (helper,))
    return inventory.Layout(resolved, inventory.Inventory((ENGINE, gpu)),
                            (ENGINE.bdf,), (gpu.bdf,))


def measured():
    return resource.ModelMemory(
        owner_map_digest=DIGEST,
        host_embedding_bytes=2 * M,
        unowned_expert_bytes=7123 * M + 1,
        host_staging_bytes=200 * M,
        ngram_ram_bytes=4096 * M,
        checkpoint_bytes=512 * M,
        other_engine_bytes=256 * M,
        kv_bytes_per_token=1024,
    )


def targets():
    return resource.Targets(
        context_tokens=8192,
        concurrent_sessions=8,
        retained_kv_tokens=32768,
        host_pool_margin_bytes=128 * M,
        host_reserve_bytes=4096 * M,
    )


def host():
    return resource.HostLimits(
        mem_available_bytes=80000 * M,
        cgroup_headroom_bytes=90000 * M,
        pin_available_bytes=30000 * M,
    )


def derive(layout=None, model=None, objective=None, limits=None, worker=None):
    l = layout or selected()
    return l.derive_resources(
         model or measured(), objective or targets(), limits or host(),
        worker_footprints_bytes=worker if worker is not None else {l.helper_bdfs[0]: 250 * M + 1},
    )


class ResourceTests(unittest.TestCase):
    def test_v1_lab_launcher_cannot_silently_use_fixed_host_budgets(self):
        launcher = (paths.REPO_ROOT / "tools/lab/radiance/launch-flashnext-best.sh").read_text()
        self.assertNotIn("HOST_POOL_MIB=9216", launcher)
        self.assertNotIn("HOST_KV_MIB=12288", launcher)
        self.assertIn("resources derive-launch", launcher)
        self.assertIn("RSC03 resolved host pool", launcher)

    def test_shipped_layouts_have_no_literal_helper_budgets(self):
        for name, recipe in layouts.load_layouts().items():
            for helper in recipe.get("helper", []):
                with self.subTest(layout=name):
                    self.assertNotIn("budget-mib", helper)

    def test_explained_host_pool_kv_and_derived_expert_budget(self):
        p = derive()
        # The +1 byte rounds the host pool up once, never truncates it.
        self.assertEqual(p.host_pool_mib, 2 + 7123 + 1 + 200 + 128)
        self.assertEqual(p.host_kv_mib, 32)
        self.assertEqual(p.helpers[0].expert_budget_mib, 24560 - 1024 - 251)
        self.assertEqual(p.helpers[0].bdf, XTX.bdf)
        self.assertEqual(p.explain()["host"]["components"][1]["source"], "published owner map")
        self.assertEqual(p.explain()["owner_map_digest"], DIGEST)
        self.assertEqual(p.host_remaining_bytes, p.host_available_bytes - p.host_committed_bytes)

    def test_detector_inventory_change_rederives_budget(self):
        original = derive()
        changed = selected(replace(XTX, vram_mib=16384))
        amended = derive(layout=changed)
        self.assertNotEqual(original.helpers[0].expert_budget_mib,
                            amended.helpers[0].expert_budget_mib)
        self.assertEqual(original.helpers[0].expert_budget_mib -
                         amended.helpers[0].expert_budget_mib, 24560 - 16384)

    def test_owner_map_change_rederives_host_pool(self):
        p = derive()
        p2 = derive(model=replace(measured(), unowned_expert_bytes=1024 * M,
                                  owner_map_digest="b" * 64))
        self.assertGreater(p.host_pool_mib, p2.host_pool_mib)

    def test_no_guessed_ownership_or_worker_footprint(self):
        with self.assertRaisesRegex(resource.ResourceError, "owner_map_digest"):
            derive(model=replace(measured(), owner_map_digest=""))
        with self.assertRaisesRegex(resource.ResourceError, "missing"):
            derive(worker={})
        with self.assertRaisesRegex(resource.ResourceError, "unexpected"):
            derive(worker={XTX.bdf: 100 * M, ENGINE.bdf: M})
        with self.assertRaisesRegex(resource.ResourceError, "positive"):
            derive(worker={XTX.bdf: 0})

    def test_refuse_host_room_and_pin_overcommit(self):
        with self.assertRaisesRegex(resource.ResourceError, "host RAM admission refused"):
            derive(limits=replace(host(), cgroup_headroom_bytes=100 * M))
        with self.assertRaisesRegex(resource.ResourceError, "pin admission refused"):
            derive(limits=replace(host(), pin_available_bytes=M))

    def test_refuse_worker_that_does_not_fit_and_invalid_context(self):
        with self.assertRaisesRegex(resource.ResourceError, "no expert VRAM"):
            derive(worker={XTX.bdf: 24560 * M})
        with self.assertRaisesRegex(resource.ResourceError, "retained KV target"):
            derive(objective=replace(targets(), retained_kv_tokens=100000))
        with self.assertRaisesRegex(resource.ResourceError, "measured"):
            derive(model=replace(measured(), unowned_expert_bytes=-1))


if __name__ == "__main__":
    unittest.main()
