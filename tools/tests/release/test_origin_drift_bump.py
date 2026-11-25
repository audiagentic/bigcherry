"""RCS23: the pin-bump drift phase stops on owned-unit origins that are blocked or changed upstream, and passes clean ones."""
import unittest
from pathlib import Path
from unittest import mock

from bigcherry.core.context import ProjectContext
from bigcherry.engine_adapters import radiance_bump
from bigcherry.release import pin_bump, pin_triage


def item(state, reason="", commits=()):
    return {"component": "bc-iqr", "unit": "r4d_x", "state": state, "reason": reason, "commits": list(commits)}


class OriginDriftBumpTests(unittest.TestCase):
    def phase(self, found):
        context = ProjectContext.resolve(engine="radiance")
        checks = {c.id: c for c in radiance_bump.source_drift_checks(context, Path("/src"))}
        # only the origin checks: the others read a real tree
        ours = [checks["owned-origins-blocked"], checks["owned-origins-changed"]]
        with mock.patch.object(pin_triage, "origin_drift", return_value=found), \
                mock.patch.object(radiance_bump.subprocess, "run", return_value=mock.Mock(stdout="0" * 40 + "\n")), \
                mock.patch("bigcherry.release.pin_bump.adapter") as adapters:
            adapters.return_value.source_drift_checks.return_value = tuple(ours)
            pin_bump.run_phase_drift(engine="radiance", vendor_root=Path("/src"), target_ref="v9")

    def test_a_clean_origin_passes(self):
        self.phase([item("clean")])

    def test_a_blocked_origin_stops_the_bump_with_its_reason(self):
        with self.assertRaises(pin_bump.PinBumpStop) as caught:
            self.phase([item("blocked", "upstream renamed libr4d/r4d_x.hip to libr4d/r4d_y.hip")])
        self.assertEqual(caught.exception.code, "OWNED_ORIGIN_BLOCKED")
        self.assertIn("renamed", str(caught.exception.evidence))

    def test_a_changed_origin_stops_for_review_and_lists_the_commits(self):
        with self.assertRaises(pin_bump.PinBumpStop) as caught:
            self.phase([item("candidate", "the origin path differs (1 upstream commit(s) touch it)", ["abc1234 tune the unit"])])
        self.assertEqual(caught.exception.code, "OWNED_ORIGIN_CHANGED")
        self.assertIn("abc1234 tune the unit", str(caught.exception.evidence))

    def test_blocked_is_reported_before_changed(self):
        with self.assertRaises(pin_bump.PinBumpStop) as caught:
            self.phase([item("candidate", "x", ["c1 y"]), item("blocked", "wrong path")])
        self.assertEqual(caught.exception.code, "OWNED_ORIGIN_BLOCKED")

    def test_the_bump_never_edits_origins(self):
        source = Path(pin_bump.__file__).read_text()
        self.assertNotIn("last-integrated", source)


if __name__ == "__main__":
    unittest.main()
