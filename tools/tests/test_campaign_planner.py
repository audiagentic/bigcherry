"""RE18: campaign_planner.plan()/run_campaign() -- the canonical multi-lane
planner and sequential runner over execute_campaign_lane() (RE16/RE17).

execute_campaign_lane() itself is faked here (its own correctness is
already covered by test_campaign_lane.py); this file's job is
campaign_planner.py's OWN orchestration: request-to-lane expansion against
real config identities, per-lane run_id derivation (the RE17-review-flagged
run_id collision trap), and fault isolation across lanes.
"""

from __future__ import annotations

import unittest
from pathlib import Path
from unittest.mock import patch

from bigcherry import config as campaign_config
from bigcherry.campaign_planner import (CampaignLane, CampaignPlannerError,
                                        CampaignRequest, lane_id, plan,
                                        run_campaign)


def _cfg(**overrides) -> campaign_config.Config:
    values = dict(
        pinned="rev1", patch_sets={},
        sources={
            "src-a": campaign_config.Source(name="src-a", ref="rev1", overlay=False, patch_sets=()),
            "src-b": campaign_config.Source(name="src-b", ref="rev1", overlay=False, patch_sets=()),
        },
        builds={
            "stock": campaign_config.Build(name="stock", options=(), variant_set=None, needs=frozenset()),
            "tune": campaign_config.Build(name="tune", options=(), variant_set="workload-max",
                                          needs=frozenset({"inventory"})),
        },
        platforms={
            "linux-multi": campaign_config.Platform(name="linux-multi", targets=("gfx1100",), options=()),
        },
        experiments={},
        campaigns={
            "standard": campaign_config.CampaignProfile(name="standard", lanes=(
                campaign_config.CampaignLaneSelector(source="src-a", build="stock", platform="linux-multi"),
                campaign_config.CampaignLaneSelector(source="src-a", build="tune", platform="linux-multi"),
            )),
        },
        path=Path("recipes.toml"),
    )
    values.update(overrides)
    return campaign_config.Config(**values)


class LaneIdTests(unittest.TestCase):
    def test_lane_id_is_source_build_platform(self):
        lane = CampaignLane(
            source_name="src-a", build_name="tune", platform_name="linux-multi",
            architectures=("gfx1100",))
        self.assertEqual(lane_id(lane), "src-a:tune:linux-multi")


class PlanTests(unittest.TestCase):
    def test_plan_expands_explicit_selectors(self):
        cfg = _cfg()
        request = CampaignRequest(
            selectors=(
                campaign_config.CampaignLaneSelector(source="src-a", build="stock", platform="linux-multi"),
                campaign_config.CampaignLaneSelector(source="src-b", build="tune", platform="linux-multi"),
            ),
            architectures=("gfx1100",),
            inputs_by_build={"tune": (("inventory", Path("inv.json")),)},
        )
        lanes = plan(request, cfg)
        self.assertEqual(len(lanes), 2)
        self.assertEqual(lanes[0].source_name, "src-a")
        self.assertEqual(lanes[0].build_name, "stock")
        self.assertEqual(lanes[0].inputs, ())
        self.assertEqual(lanes[1].source_name, "src-b")
        self.assertEqual(lanes[1].inputs, (("inventory", Path("inv.json")),))

    def test_plan_expands_named_profile(self):
        cfg = _cfg()
        request = CampaignRequest(profile_name="standard", architectures=("gfx1100",))
        lanes = plan(request, cfg)
        self.assertEqual(len(lanes), 2)
        self.assertEqual({lane_id(lane) for lane in lanes},
                         {"src-a:stock:linux-multi", "src-a:tune:linux-multi"})

    def test_plan_rejects_unknown_profile(self):
        cfg = _cfg()
        request = CampaignRequest(profile_name="does-not-exist")
        with self.assertRaises(CampaignPlannerError):
            plan(request, cfg)

    def test_plan_rejects_unknown_source_build_or_platform(self):
        cfg = _cfg()
        for selector in (
            campaign_config.CampaignLaneSelector(source="ghost", build="stock", platform="linux-multi"),
            campaign_config.CampaignLaneSelector(source="src-a", build="ghost", platform="linux-multi"),
            campaign_config.CampaignLaneSelector(source="src-a", build="stock", platform="ghost"),
        ):
            with self.assertRaises(CampaignPlannerError):
                plan(CampaignRequest(selectors=(selector,)), cfg)

    def test_plan_rejects_both_selectors_and_profile_name(self):
        cfg = _cfg()
        request = CampaignRequest(
            selectors=(campaign_config.CampaignLaneSelector(
                source="src-a", build="stock", platform="linux-multi"),),
            profile_name="standard",
        )
        with self.assertRaises(CampaignPlannerError):
            plan(request, cfg)

    def test_plan_rejects_neither_selectors_nor_profile_name(self):
        with self.assertRaises(CampaignPlannerError):
            plan(CampaignRequest(), _cfg())


class RunCampaignTests(unittest.TestCase):
    def test_lanes_receive_distinct_run_ids_sharing_one_campaign_run_id(self):
        # The RE17-review-flagged trap: two lanes sharing one campaign
        # run_id must not receive the same execution run_id, or their
        # run-scoped filesystem/ArtifactStore paths would collide.
        lanes = (
            CampaignLane(source_name="src-a", build_name="stock",
                        platform_name="linux-multi", architectures=("gfx1100",)),
            CampaignLane(source_name="src-b", build_name="tune",
                        platform_name="linux-multi", architectures=("gfx1100",)),
        )
        seen_run_ids: list[str] = []

        def fake_execute(spec, *, cfg, context, store, run_id):
            seen_run_ids.append(run_id)
            return f"result-for-{run_id}"

        with patch("bigcherry.campaign_planner.execute_campaign_lane", side_effect=fake_execute):
            results = run_campaign(
                lanes, cfg=object(), context=object(), store=object(), run_id="campaign1")

        self.assertEqual(len(seen_run_ids), 2)
        self.assertEqual(len(set(seen_run_ids)), 2)  # distinct
        for run_id in seen_run_ids:
            self.assertTrue(run_id.startswith("campaign1-"))
        self.assertEqual(set(results), {"src-a:stock:linux-multi", "src-b:tune:linux-multi"})

    def test_one_lane_failing_does_not_affect_sibling_lane_results(self):
        lanes = (
            CampaignLane(source_name="src-a", build_name="stock",
                        platform_name="linux-multi", architectures=("gfx1100",)),
            CampaignLane(source_name="src-b", build_name="tune",
                        platform_name="linux-multi", architectures=("gfx1100",)),
        )

        def fake_execute(spec, *, cfg, context, store, run_id):
            if spec.source_name == "src-a":
                raise RuntimeError("injected fault in src-a's lane")
            return f"ok-{spec.source_name}"

        with patch("bigcherry.campaign_planner.execute_campaign_lane", side_effect=fake_execute):
            results = run_campaign(lanes, cfg=object(), context=object(), store=object())

        self.assertIsInstance(results["src-a:stock:linux-multi"], RuntimeError)
        self.assertEqual(results["src-b:tune:linux-multi"], "ok-src-b")

    def test_run_id_defaults_to_a_generated_value_when_not_supplied(self):
        lanes = (CampaignLane(source_name="src-a", build_name="stock",
                              platform_name="linux-multi", architectures=("gfx1100",)),)
        captured = {}

        def fake_execute(spec, *, cfg, context, store, run_id):
            captured["run_id"] = run_id
            return "ok"

        with patch("bigcherry.campaign_planner.execute_campaign_lane", side_effect=fake_execute):
            run_campaign(lanes, cfg=object(), context=object(), store=object())

        self.assertTrue(captured["run_id"])


if __name__ == "__main__":
    unittest.main()
