"""RCS19 ownership and naming checks: fixtures for each failure, and the real tree passes."""
import unittest

from bigcherry.patch import unit_ownership as u

RULES = {"patch": {"suffixes": ["_{arch}_{id}", "_{id}"]}, "rewrite": {"suffixes": ["_gfx11_r3"]}}


class OwnershipTests(unittest.TestCase):
    def test_declared_units_must_equal_the_units_the_edits_touch(self):
        got = u.compare({"rad0001_x": ("r4d_a",)}, {"rad0001_x": ()}, {}, {}, [], set(), RULES)
        self.assertEqual(len(got), 1)
        self.assertIn("must equal the units its edits touch", got[0])
        self.assertEqual(u.compare({"rad0001_x": ("r4d_a",)}, {"rad0001_x": ("r4d_a",)}, {}, {}, [], set(), RULES), [])
        self.assertTrue(u.compare({"rad0001_x": ()}, {"rad0001_x": ("r4d_b",)}, {}, {}, [], set(), RULES))

    def test_a_unit_edited_by_two_layers_is_listed_until_recorded(self):
        shared = {"r4d_a": ["patch:rad0001_x", "rewrite:libr3"]}
        self.assertIn("edited by 2 layers", u.compare({}, {}, shared, {}, [], set(), RULES)[0])
        self.assertEqual(u.compare({}, {}, shared, {"r4d_a": {"reason": "disjoint"}}, [], set(), RULES), [])
        self.assertIn("no longer shared", u.compare({}, {}, {}, {"r4d_a": {}}, [], set(), RULES)[0])

    def test_layers_of_the_real_tree_list_the_fused_tile_unit_twice(self):
        self.assertEqual(sorted(u.layers("radiance")["r4d_gemm_iqr"]),
                         ["patch:rad0065_iqr_fused_gfx1201", "rewrite:libr3"])


class NamingTests(unittest.TestCase):
    def test_declarations_are_found_and_the_anchor_is_excluded(self):
        text = ("struct Tile { int a; };\nconstexpr uint32_t kWave = 32, kBlock = 128;\nusing Alias = int;\n"
                "#define BC_FLAG 1\n__global__ void fused_kernel(int x) {}\ntemplate <class F> struct Old { };\n")
        self.assertEqual(u.declared_names(text), {"Tile", "kWave", "kBlock", "Alias", "BC_FLAG", "fused_kernel", "Old"})
        self.assertEqual(u.introduced_names(text, "struct Old {"), {"Tile", "kWave", "kBlock", "Alias", "BC_FLAG", "fused_kernel"})

    def test_suffix_rules_come_from_data(self):
        rules = u.rules("radiance")
        layer = "patch:rad0065_iqr_fused_gfx1201"
        self.assertTrue(u.suffix_ok(layer, "IqrFusedPre_gfx12_rad0065", rules))
        self.assertTrue(u.suffix_ok(layer, "IqrFusedPre_rad0065", rules))          # arch-neutral code
        self.assertFalse(u.suffix_ok(layer, "IqrFused12Pre", rules))
        self.assertFalse(u.suffix_ok(layer, "Tile_gfx12_rad0064", rules))          # another patch's id
        self.assertTrue(u.suffix_ok("rewrite:libr3", "kTile_gfx11_r3", rules))
        self.assertFalse(u.suffix_ok("rewrite:libr3", "kTile_gfx12_rad0065", rules))
        self.assertFalse(u.suffix_ok(layer, "IqrFusedPre_rad0065", {"patch": {"suffixes": ["_{arch}_{id}"]}}))

    def test_scope_rules(self):
        text = (
            'namespace r3 { struct Foo_a {}; constexpr char s[] = "{"; struct After_b {}; }\n'
            "template <class F>\n__global__ __launch_bounds__(kPWaves * kWave) R4D_WAVES(6)\n"
            "void kern_c(Ctx c, int x) { int local = 1; }\n"
            "namespace n {\n#define MAC_D 1\n}\n"
            'extern "C" { __device__ int dev_e(int); }\n'
            "struct Outer { constexpr static int mem = 3; using D = int; };\n"
            "// struct InComment { };\n"
        )
        self.assertEqual(u.declared_names(text), {"Foo_a", "s", "After_b", "kern_c", "MAC_D", "dev_e", "Outer"})

    def test_members_and_locals_are_scoped_by_their_type(self):
        text = "struct Outer_rad0001 {\n  static constexpr bool geometry = true;\n  using D = int;\n};\nconstexpr int kTop = 1;\n"
        self.assertEqual(u.declared_names(text), {"Outer_rad0001", "kTop"})

    def test_a_new_unsuffixed_name_fails_and_a_recorded_one_passes(self):
        layer = "patch:rad0065_iqr_fused_gfx1201"
        names = [(layer, "a.hip:e1", "NewTile")]
        self.assertIn("lacks a layer suffix", u.compare({}, {}, {}, {}, names, set(), RULES)[0])
        self.assertEqual(u.compare({}, {}, {}, {}, names, {(layer, "NewTile")}, RULES), [])
        self.assertEqual(u.compare({}, {}, {}, {}, [(layer, "a.hip:e1", "NewTile_gfx12_rad0065")], set(), RULES), [])
        self.assertIn("no longer introduced", u.compare({}, {}, {}, {}, [], {(layer, "Gone")}, RULES)[0])


class RealTreeTests(unittest.TestCase):
    def test_the_real_tree_passes(self):
        self.assertEqual(u.problems("radiance"), [])


if __name__ == "__main__":
    unittest.main()
