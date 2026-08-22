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
from bigcherry import experiment_contract as ec
from bigcherry.campaign_planner import (CampaignLane, CampaignPlannerError,
                                        CampaignRequest, expand_contract,
                                        lane_id, plan, run_campaign)


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
            "linux-multi": campaign_config.Platform(
                name="linux-multi", targets=("gfx1100", "gfx1201", "gfx1030"), options=()),
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

    def test_plan_threads_experiment_onto_every_lane(self):
        # RE26 prep: one experiment name on the request applies to every
        # lane it plans -- isolated single-patch benching needs the SAME
        # experiment across the whole standard lane set, not a per-lane
        # mix that could silently omit it from one lane.
        cfg = _cfg()
        request = CampaignRequest(
            selectors=(
                campaign_config.CampaignLaneSelector(source="src-a", build="stock", platform="linux-multi"),
                campaign_config.CampaignLaneSelector(source="src-b", build="tune", platform="linux-multi"),
            ),
            architectures=("gfx1100",),
            inputs_by_build={"tune": (("inventory", Path("inv.json")),)},
            experiment="rd19-only",
        )
        lanes = plan(request, cfg)
        self.assertEqual(len(lanes), 2)
        self.assertTrue(all(lane.experiment == "rd19-only" for lane in lanes))

    def test_plan_defaults_experiment_to_none(self):
        cfg = _cfg()
        request = CampaignRequest(
            selectors=(
                campaign_config.CampaignLaneSelector(source="src-a", build="stock", platform="linux-multi"),
            ),
            architectures=("gfx1100",),
        )
        lanes = plan(request, cfg)
        self.assertIsNone(lanes[0].experiment)

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

    def test_plan_defaults_architectures_to_platform_targets_when_unspecified(self):
        # GPT review (round 20): empty/unspecified architectures must mean
        # "use this lane's Platform.targets", not "generate for nothing" --
        # architectures flows straight into candidate-universe generation
        # independently of platform.targets (which only drives
        # AMDGPU_TARGETS at compile time).
        cfg = _cfg()
        request = CampaignRequest(profile_name="standard")  # no architectures set
        lanes = plan(request, cfg)
        for lane in lanes:
            self.assertEqual(lane.architectures, ("gfx1100", "gfx1201", "gfx1030"))

    def test_plan_accepts_an_explicit_subset_of_platform_targets(self):
        cfg = _cfg()
        request = CampaignRequest(profile_name="standard", architectures=("gfx1100",))
        lanes = plan(request, cfg)
        for lane in lanes:
            self.assertEqual(lane.architectures, ("gfx1100",))

    def test_plan_rejects_an_architecture_not_in_platform_targets(self):
        cfg = _cfg()
        request = CampaignRequest(profile_name="standard", architectures=("gfx9999",))
        with self.assertRaises(CampaignPlannerError):
            plan(request, cfg)

    def test_plan_rejects_duplicate_lane_selectors(self):
        cfg = _cfg()
        selector = campaign_config.CampaignLaneSelector(
            source="src-a", build="stock", platform="linux-multi")
        request = CampaignRequest(selectors=(selector, selector))
        with self.assertRaises(CampaignPlannerError):
            plan(request, cfg)

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


def _contract(**overrides) -> ec.ExperimentContract:
    doc = {
        "title": "tiny-M Q8 MMQ specialization",
        "source": {"source_id": "s", "commits": ["c"], "atomic_part": "tiny-m-q8-gate"},
        "hypothesis": {"family": "mmq", "expected_effect": "performance", "rationale": "r"},
        "scope": {"backend": "hip", "architectures": ["gfx1100"], "weight_types": ["q8_0"]},
        "positive": {"models": ["model-a", "model-b"], "workloads": ["small_m"]},
        "controls": {"models": ["model-c"], "workloads": ["decode", "prefill"]},
        "boundary": {"dimensions": {"physical_m": [1, 2, 4]}},
        "acceptance": {"max_control_regression_pct": 1},
    }
    doc.update(overrides)
    return ec.parse_contract(doc, contract_id="RDNA-EXT-001")


class ExpandContractTests(unittest.TestCase):
    def test_expands_positive_control_and_boundary_lanes(self):
        cfg = _cfg()
        lanes = expand_contract(
            _contract(), cfg=cfg, source_name="src-a", build_name="stock",
            platform_name="linux-multi",
        )
        # positive: 2 models x 1 workload = 2; controls: 1 model x 2
        # workloads = 2; boundary: 3 physical_m values = 3. Total 7.
        self.assertEqual(len(lanes), 7)
        roles = sorted(lane.role for lane in lanes)
        self.assertEqual(roles, ["boundary"] * 3 + ["control"] * 2 + ["positive"] * 2)

    def test_every_lane_shares_build_identity_inputs(self):
        cfg = _cfg()
        lanes = expand_contract(
            _contract(), cfg=cfg, source_name="src-a", build_name="stock",
            platform_name="linux-multi",
        )
        for lane in lanes:
            self.assertEqual(lane.source_name, "src-a")
            self.assertEqual(lane.build_name, "stock")
            self.assertEqual(lane.platform_name, "linux-multi")
            self.assertEqual(lane.architectures, ("gfx1100", "gfx1201", "gfx1030"))

    def test_lanes_carry_contract_metadata_not_identity(self):
        cfg = _cfg()
        lanes = expand_contract(
            _contract(), cfg=cfg, source_name="src-a", build_name="stock",
            platform_name="linux-multi",
        )
        positive = [lane for lane in lanes if lane.role == "positive"]
        self.assertEqual({lane.contract_id for lane in positive}, {"RDNA-EXT-001"})
        self.assertEqual({lane.optimization_id for lane in positive}, {"tiny-m-q8-gate"})
        self.assertEqual({lane.workload_tag for lane in positive}, {"small_m"})
        self.assertEqual({lane.model_ref for lane in positive}, {"model-a", "model-b"})
        boundary = [lane for lane in lanes if lane.role == "boundary"]
        self.assertEqual({lane.boundary_dimension for lane in boundary}, {"physical_m"})
        self.assertEqual({lane.boundary_value for lane in boundary}, {"1", "2", "4"})

    def test_all_expanded_lane_ids_are_distinct(self):
        cfg = _cfg()
        lanes = expand_contract(
            _contract(), cfg=cfg, source_name="src-a", build_name="stock",
            platform_name="linux-multi",
        )
        ids = [lane_id(lane) for lane in lanes]
        self.assertEqual(len(ids), len(set(ids)))

    def test_unknown_source_build_or_platform_rejected(self):
        cfg = _cfg()
        with self.assertRaises(CampaignPlannerError):
            expand_contract(_contract(), cfg=cfg, source_name="ghost",
                            build_name="stock", platform_name="linux-multi")
        with self.assertRaises(CampaignPlannerError):
            expand_contract(_contract(), cfg=cfg, source_name="src-a",
                            build_name="ghost", platform_name="linux-multi")
        with self.assertRaises(CampaignPlannerError):
            expand_contract(_contract(), cfg=cfg, source_name="src-a",
                            build_name="stock", platform_name="ghost")

    def test_expanded_lanes_feed_run_campaign_without_collision(self):
        cfg = _cfg()
        lanes = expand_contract(
            _contract(), cfg=cfg, source_name="src-a", build_name="stock",
            platform_name="linux-multi",
        )

        def fake_execute(spec, *, cfg, context, store, run_id, allow_dirty_bigcherry=False):
            return f"ok-{spec.role}-{spec.workload_tag}-{spec.model_ref}-{spec.boundary_value}"

        with patch("bigcherry.campaign_planner.execute_campaign_lane", side_effect=fake_execute):
            results = run_campaign(lanes, cfg=cfg, context=object(), store=object())
        self.assertEqual(len(results), 7)


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

        def fake_execute(spec, *, cfg, context, store, run_id, allow_dirty_bigcherry=False):
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

        def fake_execute(spec, *, cfg, context, store, run_id, allow_dirty_bigcherry=False):
            if spec.source_name == "src-a":
                raise RuntimeError("injected fault in src-a's lane")
            return f"ok-{spec.source_name}"

        with patch("bigcherry.campaign_planner.execute_campaign_lane", side_effect=fake_execute):
            results = run_campaign(lanes, cfg=object(), context=object(), store=object())

        self.assertIsInstance(results["src-a:stock:linux-multi"], RuntimeError)
        self.assertEqual(results["src-b:tune:linux-multi"], "ok-src-b")

    def test_run_campaign_rejects_duplicate_lane_identities(self):
        # Defensive check even though plan() already rejects duplicates:
        # callers can construct CampaignLanes directly without plan().
        lane = CampaignLane(source_name="src-a", build_name="stock",
                            platform_name="linux-multi", architectures=("gfx1100",))
        with patch("bigcherry.campaign_planner.execute_campaign_lane") as fake:
            with self.assertRaises(CampaignPlannerError):
                run_campaign((lane, lane), cfg=object(), context=object(), store=object())
        fake.assert_not_called()

    def test_run_id_defaults_to_a_generated_value_when_not_supplied(self):
        lanes = (CampaignLane(source_name="src-a", build_name="stock",
                              platform_name="linux-multi", architectures=("gfx1100",)),)
        captured = {}

        def fake_execute(spec, *, cfg, context, store, run_id, allow_dirty_bigcherry=False):
            captured["run_id"] = run_id
            return "ok"

        with patch("bigcherry.campaign_planner.execute_campaign_lane", side_effect=fake_execute):
            run_campaign(lanes, cfg=object(), context=object(), store=object())

        self.assertTrue(captured["run_id"])


if __name__ == "__main__":
    unittest.main()
