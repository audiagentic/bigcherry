"""Unit harness: the card-free parts (resource diff, hash comparison, replay estimator, selftest edit, stub generator)."""
import importlib.util
import sys
import unittest
import tempfile
from unittest.mock import patch
from pathlib import Path

PATH = Path(__file__).resolve().parents[2] / "lab" / "radiance" / "unit_harness.py"
spec = importlib.util.spec_from_file_location("unit_harness", PATH)
uh = importlib.util.module_from_spec(spec)
sys.modules["unit_harness"] = uh
spec.loader.exec_module(uh)

REC = dict.fromkeys(uh.RESOURCE_KEYS, 0)


class HarnessTests(unittest.TestCase):
    def test_resource_diff(self):
        ref = {"a_iqr": [{**REC, "vgpr_count": 100}], "b_iqr": [dict(REC)], "c_other": [dict(REC), {**REC, "vgpr_count": 5}]}
        cand = {"a_iqr": [{**REC, "vgpr_count": 101}], "new_iqr": [dict(REC)], "c_other": [dict(REC), {**REC, "vgpr_count": 5}]}
        import re
        diff = uh.diff_records(ref, cand, re.compile("iqr"))
        self.assertEqual(diff["added"], ["new_iqr"])
        self.assertEqual(diff["removed"], ["b_iqr"])
        self.assertEqual(diff["changed"], {"a_iqr": {"vgpr_count": [[100], [101]]}})
        self.assertEqual(diff["unchanged"], 1)

    def test_hash_comparison(self):
        got = uh.compare_hashes({"x": "1", "y": "2", "z": "3"}, {"x": "1", "y": "9", "w": "4"})
        self.assertEqual((got["compared"], got["differ"], got["missing_in_candidate"], got["only_in_candidate"]),
                         (2, ["y"], ["z"], ["w"]))

    def test_estimator_ignores_slow_mode_windows(self):
        self.assertFalse(uh.converged([300.0, 301.0, 300.5], 0.015))      # too few windows
        self.assertTrue(uh.converged([300.0, 150.0, 151.0, 300.0], 0.015))
        self.assertFalse(uh.converged([300.0, 150.0, 190.0, 300.0], 0.015))
        self.assertEqual(uh.estimate([300.0, 150.0, 151.0]), 150.0)

    def test_classify(self):
        self.assertEqual([uh.classify(r, 0.04) for r in (0.5, 1.0, 1.1)], ["win", "same", "regress"])

    def test_selftest_edit_needs_both_anchors_once(self):
        text = (uh.DEVICE_ANCHOR + uh.SYM_ANCHOR + uh.HASH_ANCHOR + uh.CASES_ANCHOR + uh.FIND_ANCHOR + uh.EMBED_ANCHOR + uh.EMBED_CASES_ANCHOR + uh.VOCAB_ANCHOR + uh.VOCAB_CASES_ANCHOR)
        edited = uh.inject_selftest(text)
        self.assertIn("UNITHASH", edited)
        self.assertIn("UH_HASH_CASES", edited)
        self.assertIn("UH_TOLERATE_ABSENT", edited)
        self.assertIn("UH_ROW_PREFIX", edited)
        self.assertIn("UH_TIME_ITERS", edited)
        with self.assertRaises(uh.HarnessError):
            uh.inject_selftest(text + uh.CASES_ANCHOR)
        with self.assertRaises(uh.HarnessError):
            uh.inject_selftest("nothing")

    def test_families_are_well_formed(self):
        for name, fam in uh.families().items():
            self.assertTrue(fam["units"], name)
            self.assertIn("hash", fam)
            self.assertIn("replay", fam)
        cases = uh.hash_cases(uh.families()["iqr"]["hash"]).split(",")
        self.assertIn("5376:2560:129", cases)
        self.assertEqual(len(cases), len(set(cases)))

    def test_group_manifest_is_complete_and_unique(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "r3_groups.tsv"
            path.write_text("group\tunit\nnew_group\tr4d_one\n")
            actual = uh.read_group_manifest(path)
            expected = {"r4d_one": "new_group", "r4d_two": "other"}
            uh.verify_group_manifest(actual, expected, {"r4d_two"})
            with self.assertRaises(uh.HarnessError):
                uh.verify_group_manifest(actual, expected, set())
            with self.assertRaises(uh.HarnessError):
                uh.verify_group_manifest(actual, {"r4d_one": "wrong"}, set())
            path.write_text("group\tunit\nnew_group\tr4d_one\nother\tr4d_one\n")
            with self.assertRaises(uh.HarnessError):
                uh.read_group_manifest(path)

    def test_group_routes_do_not_count_support_or_substitute_base(self):
        inventory = {"r4d_a": "new", "r4d_b": "new", "r4d_c": "new", "r4d_d": "other"}
        families = {"custom": {"units": ["r4d_a"], "support_units": ["r4d_b"], "case": "row_a"}}
        origins = {"custom_overlay": {"units": ["r4d_c"], "base_rows": ["row_c", "row_d"],
                                      "overlay_rows": ["new_c", "new_d"], "selftest_case": "row_c"}}
        plan = uh.resolve_group_plan(inventory, {"r4d_c"}, families, origins, ["new"], [])
        self.assertEqual(plan["uncovered_units"], ["r4d_b"])
        self.assertEqual(plan["runs"][1]["other_rows_need_evidence"], ["row_d"])
        plan = uh.resolve_group_plan(inventory, set(), families, origins, ["new"], [])
        self.assertEqual(plan["uncovered_units"], ["r4d_b", "r4d_c"])
        plan = uh.resolve_group_plan(inventory, {"r4d_a", "r4d_c"}, families, origins, ["new"], [])
        self.assertEqual(plan["uncovered_units"], ["r4d_a", "r4d_b"])
        with self.assertRaises(uh.HarnessError):
            uh.resolve_group_plan(inventory, set(), families, origins, ["typo"], [])

    def test_outcome_requires_positive_complete_converged_evidence(self):
        point = {"row": "row", "N": 16, "K": 64, "M": 1, "B": 4,"mode":"cold"}
        base = {"hash": uh.compare_hashes({"a": "1"}, {"a": "1"}), "selftest_failures": {"ref": 0, "cand": 0},
                "required_replay": [point], "replay": [{**point, "converged": True, "verdict": "same"}]}
        self.assertEqual(uh.configured_measurement_outcome(base)[0], "checks_passed")
        for changes in (
            {"hash": uh.compare_hashes({}, {})},
            {"replay": []},
            {"replay": [{**point, "converged": False, "verdict": "win"}]},
            {"selftest_failures": {"ref": 1, "cand": 0}},
        ):
            self.assertEqual(uh.configured_measurement_outcome({**base, **changes})[0], "inconclusive")
        self.assertEqual(uh.configured_measurement_outcome({**base, "hash": uh.compare_hashes({"a": "1"}, {"a": "2"})})[0], "rejected")
        self.assertEqual(uh.configured_measurement_outcome({**base, "replay": [{**point, "converged": True, "verdict": "regress"}]})[0], "rejected")

    def test_all_overlay_rounds_must_match(self):
        line = "UNITHASH case 0123456789abcdef\n"
        self.assertFalse(uh.consistent_round_hashes([line, line])[1])
        self.assertTrue(uh.consistent_round_hashes([line, ""])[1])
        self.assertTrue(uh.consistent_round_hashes([line, line + "FAIL test\n"])[1])

    def test_canonical_groups_use_cmake_classifier(self):
        import shutil
        if not shutil.which("cmake"):
            self.skipTest("cmake not installed")
        with tempfile.TemporaryDirectory() as tmp:
            tree = Path(tmp)
            (tree / "libr4d").mkdir()
            cmake = tree / "libr4d/CMakeLists.txt"
            cmake.write_text('  "r4d_dflash_example|flags"\n  "r4d_ar_Nrank|flags"\n  "r4d_new_unit|"\n')
            with patch.object(uh, "work_root", return_value=tree):
                component = uh.ROOT / "engines/radiance/components/libr3"
                self.assertEqual(uh.canonical_groups(tree, component),
                                 {"r4d_dflash_example": "speculation", "r4d_ar_Nrank": "collectives", "r4d_new_unit": "core"})
                cmake.write_text(cmake.read_text() + '  "r4d_new_unit|"\n')
                with self.assertRaises(uh.HarnessError):
                    uh.canonical_groups(tree, component)

    def test_duplicate_and_malformed_hash_records(self):
        line="UNITHASH case 0123456789abcdef\n"
        self.assertEqual(uh.parse_hashes(line)[1],[])
        self.assertTrue(uh.parse_hashes(line+line)[1])
        self.assertTrue(uh.parse_hashes(line+"UNITHASH malformed\n")[1])
        self.assertTrue(uh.parse_hashes(line+"  skip  required row\n")[1])
        self.assertTrue(uh.parse_hashes(line+"0 failure(s), 1 skipped\n")[1])

    def test_affected_paths_widen_for_headers_removed_or_external(self):
        with tempfile.TemporaryDirectory() as tmp:
            tree=Path(tmp)
            (tree/"libr4d").mkdir()
            for n in ("r4d_a","r4d_b"):
                (tree/"libr4d"/(n+".hip")).write_text("")
            inventory={"r4d_a":"g","r4d_b":"g"}
            with patch("bigcherry.patch.compose_check.affected_units",return_value=["r4d_a"]):
                self.assertEqual(uh.affected_selection(tree,inventory,["libr4d/r4d_a.hip"])["units"],["r4d_a"])
                for path in ("libr4d/common.h","libr4d/removed.hip","abi/rad_abi.h"):
                    result=uh.affected_selection(tree,inventory,[path])
                    self.assertEqual(result["units"],sorted(inventory))
                    self.assertEqual(result["scope"],"full_inventory_conservative")
                with self.assertRaises(uh.HarnessError):
                    uh.affected_selection(tree,inventory,["../outside"])

    def test_exact_row_selection_does_not_grant_other_overlay_rows(self):
        fam={"f":{"units":["r4d_a"],"case":"base","replay":{"rows":["other"]}}}
        ov={"o":{"units":["r4d_a"],"base_rows":["base","other"],"overlay_rows":["bc_base","bc_other"],"selftest_case":"base"}}
        units,rows=uh.resolve_rows(["bc_base","bc_other","missing","base","other"],fam,ov)
        self.assertEqual(units,["r4d_a"])
        self.assertEqual([r["status"] for r in rows],["configured_route","uncovered","uncovered","configured_route","uncovered"])
        with self.assertRaises(uh.HarnessError):
            uh.resolve_rows(["base","base"],fam,ov)

    def test_model_graph_scope_uses_declared_rows_and_checks_order(self):
        import json
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/"graph.json"
            graph={"model":{"rank":0,"world":2},"counts":{"ops":2},
                   "plugins":[{"name":"libbciqr"}],"order":[0,0],"misses":[],
                   "ops":[{"op":"gemm","count":2,"geometry":"M=1","bands":[
                       {"span":"1","device":{"kernel":"bc_gemm_iqr_tiled","plugin":"libbciqr","geometry":"M=1"}}]}]}
            path.write_text(json.dumps(graph))
            scope=uh.model_graph_scope(path)
            self.assertEqual(scope["declared"][0]["row"],"bc_gemm_iqr_tiled")
            self.assertIn("other ranks require their own graph",scope["gaps"])
            graph["order"]=[0]
            path.write_text(json.dumps(graph))
            with self.assertRaises(uh.HarnessError):
                uh.model_graph_scope(path)

    def test_overlay_timing_and_layout_every_round(self):
        text="UNITHASH a 0123456789abcdef\nUNITTIME a 100\nUNITLAYOUT plain_bf16 rc=-2\n"
        good={"ref":[text,text],"cand":[text,text]}
        timing,gaps=uh.overlay_timing(good,expect_timing=True)
        self.assertEqual(gaps,[])
        self.assertEqual(timing[0]["ratio"],1)
        self.assertEqual(uh.overlay_layout(good)[1],[])
        for bad in ("",text+"UNITTIME a 100\n",text.replace("100","0"),text+"UNITTIME extra 12\n"):
            self.assertTrue(uh.overlay_timing({"ref":[text,text],"cand":[text,bad]},expect_timing=True)[1])
        self.assertTrue(uh.overlay_layout({"ref":[text,text],"cand":[text,text.replace("-2","0")]})[1])

    def test_cold_and_warm_replay_share_exact_shape_policy(self):
        spec=uh.families()["iqr"]["replay"]
        points=uh.replay_points(spec,[6],[65])
        self.assertEqual(len(points),2*len(spec["shapes"]))
        self.assertEqual(set(p["mode"] for p in points),{"cold","warm"})
        self.assertIn("--cold",uh.replay_command(spec,points[0],15,"cold"))
        self.assertNotIn("--cold",uh.replay_command(spec,points[1],15,"warm"))
        with self.assertRaises(uh.HarnessError):
            uh.replay_command({"command":["--perfiqr"]},points[0],15,"cold")

    def test_explicit_tree_is_copied_before_any_build_edit(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            source=root/"source"
            source.mkdir()
            (source/"CMakeLists.txt").write_text("original")
            scratch=root/"build"
            got=uh.compose([],source,scratch)
            (got/"CMakeLists.txt").write_text("private")
            self.assertEqual((source/"CMakeLists.txt").read_text(),"original")
            with self.assertRaises(uh.HarnessError):
                uh.compose([],source,scratch)
            with self.assertRaises(uh.HarnessError):
                uh.compose(["patch"],source,root/"other")

    def test_artifact_identity_detects_replaced_binary(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            state={}
            for key in ("plugin","reference_plugin","selftest"):
                path=root/key
                path.write_text(key)
                state[key]=str(path)
            with self.assertRaises(uh.HarnessError):
                uh.require_artifacts(state)
            state["artifact_sha256"]=uh.artifact_identity(state)
            uh.require_artifacts(state)
            (root/"plugin").write_text("replaced")
            with self.assertRaises(uh.HarnessError):
                uh.require_artifacts(state)

    def test_selftest_requires_physical_identity_and_lease(self):
        import os
        with patch("bigcherry.hardware.lease.held_by_this_process",return_value=False):
            with self.assertRaises(uh.HarnessError):
                uh.device_index("gfx1100","0000:03:00.0")
        with patch("bigcherry.hardware.lease.held_by_this_process",return_value=True), patch.dict(os.environ,{"HIP_VISIBLE_DEVICES":"0"}):
            with self.assertRaises(uh.HarnessError):
                uh.device_index("gfx1100","0000:03:00.0")
        env=uh.selftest_env("gfx1100",("1","0000:03:00.0"))
        self.assertEqual(env["UH_EXPECT_BDF"],"0000:03:00.0")
        self.assertEqual(env["HIP_VISIBLE_DEVICES"],"1")
        with self.assertRaises(uh.HarnessError):
            uh.selftest_env("gfx1100",("1","0000:03:00.0"),{"UH_EXPECT_BDF":"wrong"})


if __name__ == "__main__":
    unittest.main()
