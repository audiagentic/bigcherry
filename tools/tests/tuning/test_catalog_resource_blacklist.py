import unittest

from bigcherry.tuning import catalog as catalog


class CatalogResourceBlacklistTests(unittest.TestCase):
    def test_blacklists_only_generated_candidate_on_affected_architecture(self):
        generated = catalog.Candidate(
            "mmvq:q8_0:w1:nw1:rpb1:sk0:v1", "mmvq", "new_generated_variant",
            ["gfx1100", "gfx1201"], {})
        native = catalog.Candidate(
            "native:mmvq:v1", "mmvq", "native_wrapper", ["gfx1100"], {})

        result = catalog.apply_resource_blacklist(
            [generated, native],
            {(generated.stable_name, "gfx1100"): ("scratch",)},
        )

        self.assertEqual(result[0].architectures, ["gfx1201"])
        self.assertEqual(result[1], native)

    def test_drops_generated_candidate_when_all_architectures_are_blacklisted(self):
        candidate = catalog.Candidate(
            "mmvq:q8_0:w1:nw1:rpb1:sk0:v1", "mmvq", "new_generated_variant",
            ["gfx1100", "gfx1201"], {})
        self.assertEqual(
            catalog.apply_resource_blacklist(
                [candidate],
                {(candidate.stable_name, "gfx1100"): ("scratch",),
                 (candidate.stable_name, "gfx1201"): ("register_spill",)},
            ),
            [],
        )


if __name__ == "__main__":
    unittest.main()
