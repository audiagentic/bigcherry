"""The release-notes line rewriter (.github/scripts/rewrite_release_lines.py): keyed by commit sha, idempotent, touches nothing else."""
import importlib.util
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
_SPEC = importlib.util.spec_from_file_location("rewrite_release_lines", ROOT / ".github" / "scripts" / "rewrite_release_lines.py")
mod = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(mod)

SHA_A = "a" * 40
SHA_B = "b" * 40
RULES = {SHA_A: "**lab:** clean subject (MEN11)", SHA_B: "clean other"}
BULLET_A = (f"* **lab:** old subject ([#259](https://github.com/o/r/issues/259)) "
            f"([aaaaaaa](https://github.com/o/r/commit/{SHA_A}))")
BULLET_C = "* **bench:** keep me ([#1](https://github.com/o/r/issues/1)) ([ccccccc](https://github.com/o/r/commit/" + "c" * 40 + "))"


class RewriteTests(unittest.TestCase):
    def test_a_bullet_whose_commit_link_matches_gets_the_clean_subject_and_keeps_its_links(self):
        out = mod.rewrite(BULLET_A, RULES)
        self.assertEqual(out, f"* **lab:** clean subject (MEN11) ([#259](https://github.com/o/r/issues/259)) "
                              f"([aaaaaaa](https://github.com/o/r/commit/{SHA_A}))")

    def test_other_lines_are_untouched(self):
        text = "## [1.4.0]\n\n### New\n\n" + BULLET_C + "\n\nplain text mentioning " + SHA_A + "\n"
        self.assertEqual(mod.rewrite(text, RULES), text)

    def test_idempotent(self):
        once = mod.rewrite(BULLET_A + "\n" + BULLET_C, RULES)
        self.assertEqual(mod.rewrite(once, RULES), once)

    def test_only_the_matching_bullet_changes_in_a_larger_body(self):
        body = "\n".join(["### New and changed", "", BULLET_C, BULLET_A, "", "### Fixes", BULLET_C])
        out = mod.rewrite(body, RULES).split("\n")
        self.assertEqual(out[2], BULLET_C)
        self.assertIn("clean subject", out[3])
        self.assertEqual(out[-1], BULLET_C)

    def test_a_bullet_without_an_issue_link_is_left_alone_rather_than_guessed(self):
        odd = f"* subject with no issue link ([aaaaaaa](https://github.com/o/r/commit/{SHA_A}))"
        self.assertEqual(mod.rewrite(odd, RULES), odd)

    def test_the_shipped_rules_load_and_carry_plain_replacement_text(self):
        # The forbidden-terms scan guards the rule file itself, so this test must not name the rejected words:
        # it checks structure only (a full sha, no environment-variable-style tokens in the replacement).
        rules = mod.load_rules(ROOT / ".github" / "release-line-rewrites.toml")
        self.assertGreaterEqual(len(rules), 2)
        for sha, text in rules.items():
            self.assertEqual(len(sha), 40)
            self.assertNotRegex(text, r"\b[A-Z][A-Z0-9]*_[A-Z0-9_]+\b")

    def test_rule_file_validation_fails_closed(self):
        for bad in ('[[line]]\nsha = "abc"\ntext = "x"\n', f'[[line]]\nsha = "{SHA_A}"\ntext = " "\n',
                    f'[[line]]\nsha = "{SHA_A}"\ntext = "x"\n[[line]]\nsha = "{SHA_A}"\ntext = "y"\n'):
            with tempfile.TemporaryDirectory() as d:
                p = Path(d) / "r.toml"
                p.write_text(bad)
                with self.assertRaises(ValueError):
                    mod.load_rules(p)


if __name__ == "__main__":
    unittest.main()
