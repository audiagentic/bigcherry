"""tools.bigcherry.patch.docs -- per-patch SUMMARY.md merge into a release doc."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from bigcherry.patch import docs as patch_docs  # noqa: E402
from bigcherry.patch import registry as patch_registry  # noqa: E402


def _write_packaged_patch(root: Path, patch_id: str, *,
                           state: str = "validated", group: str = "core",
                           plan_item: str | None = None,
                           plan_ids: tuple[str, ...] = ()) -> None:
    """Every real production patch is a packaged directory ("<id>/patch.py"
    + "<id>/patch.toml", patch.toml authoritative for state/group/plan)."""
    # order must equal the id's own numeric prefix (registry.py's
    # _packaged_descriptor enforces this) -- derive it, don't hand-pick it.
    order = int(patch_id.split("_", 1)[0])
    patch_dir = root / patch_id
    patch_dir.mkdir()
    (patch_dir / "patch.py").write_text("PATCHES = []\n", encoding="utf-8")
    plan_ids_toml = "[" + ", ".join(f'"{p}"' for p in plan_ids) + "]"
    (patch_dir / "patch.toml").write_text(
        "schema = 1\n"
        f'id = "{patch_id}"\n'
        f"order = {order}\n"
        f'group = "{group}"\n'
        f'state = "{state}"\n'
        'kind = "framework"\n'
        'origin = "local"\n'
        'backend = "agnostic"\n'
        + (f'plan-item = "{plan_item}"\n' if plan_item else "")
        + f"plan-ids = {plan_ids_toml}\nrequires = []\nconflicts = []\n"
        "requires-options = []\nforbids-options = []\nsubsystems = []\n"
        "hardware = []\nvalidation-architectures = []\nbackends = []\n",
        encoding="utf-8",
    )


def _write_summary(root: Path, patch_id: str, *, status: str, group: str,
                    plan_item: str, body: str = "") -> None:
    (root / patch_id / "SUMMARY.md").write_text(
        f"# {patch_id}\n\n**Status:** {status}\n**Group:** {group}\n"
        f"**Plan item:** {plan_item}\n\n{body}",
        encoding="utf-8",
    )


def _descriptor(root: Path, patch_id: str) -> "patch_registry.PatchDescriptor":
    return patch_registry.load_registry(root).by_id[patch_id]


def _registry_root(root: Path) -> Path:
    return patch_registry.load_registry(root).root


class ReadPatchSummaryTests(unittest.TestCase):
    def test_reads_the_real_summary_file(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _write_packaged_patch(root, "0100_x")
            (root / "0100_x" / "SUMMARY.md").write_text(
                "# 0100_x\n\nreal content\n", encoding="utf-8",
            )
            descriptor = _descriptor(root, "0100_x")
            self.assertIn(
                "real content", patch_docs.read_patch_summary(descriptor, _registry_root(root)),
            )

    def test_missing_summary_renders_a_visible_placeholder_not_a_crash(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _write_packaged_patch(root, "0200_y", state="untested")
            descriptor = _descriptor(root, "0200_y")
            rendered = patch_docs.read_patch_summary(descriptor, _registry_root(root))
            self.assertIn("No SUMMARY.md found", rendered)
            self.assertIn("0200_y", rendered)
            self.assertIn("untested", rendered)


class RenderReleaseDocTests(unittest.TestCase):
    def test_merges_in_given_order_with_pin_header(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _write_packaged_patch(root, "0200_b")
            _write_packaged_patch(root, "0100_a")
            (root / "0200_b" / "SUMMARY.md").write_text("# 0200_b\n\nsecond patch\n", encoding="utf-8")
            (root / "0100_a" / "SUMMARY.md").write_text("# 0100_a\n\nfirst patch\n", encoding="utf-8")

            registry = patch_registry.load_registry(root)
            # Caller order is preserved, NOT re-sorted by numeric id --
            # pass 0200_b before 0100_a deliberately to prove it.
            descriptors = [registry.by_id["0200_b"], registry.by_id["0100_a"]]
            doc = patch_docs.render_release_doc(
                descriptors=descriptors,
                patches_root=registry.root,
                pin_info={"llama.cpp revision": "abc123"},
                selection_label="--recipe workstation",
            )
            self.assertIn("Selection: --recipe workstation", doc)
            self.assertIn("llama.cpp revision:** abc123", doc)
            self.assertIn("2 patch(es) included", doc)
            self.assertLess(doc.index("second patch"), doc.index("first patch"))


class ResolvePatchDescriptorsTests(unittest.TestCase):
    def test_raises_on_an_unresolvable_id_instead_of_silently_dropping_it(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _write_packaged_patch(root, "0100_x")
            with self.assertRaises(patch_docs.PatchDocError):
                patch_docs.resolve_patch_descriptors(
                    ("0100_x", "not-a-real-id"), patches_dir=root,
                )

    def test_preserves_caller_order(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _write_packaged_patch(root, "0200_b")
            _write_packaged_patch(root, "0100_a")
            resolved, patches_root = patch_docs.resolve_patch_descriptors(
                ("0200_b", "0100_a"), patches_dir=root,
            )
            self.assertEqual([d.patch_id for d in resolved], ["0200_b", "0100_a"])
            self.assertEqual(patches_root, root.resolve())


class RenderPatchSelectionDocTests(unittest.TestCase):
    def test_raises_when_a_requested_id_does_not_resolve(self):
        """A real regression this replaces: an unknown id used to be
        silently filtered out, so an N-patch selection could render as a
        doc claiming fewer patches than were actually requested."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _write_packaged_patch(root, "0100_x")
            _write_summary(root, "0100_x", status="validated", group="core", plan_item="none")
            with self.assertRaises(patch_docs.PatchDocError):
                patch_docs.render_patch_selection_doc(
                    patch_ids=("0100_x", "ghost-id"),
                    pin_info={}, selection_label="test", patches_dir=root,
                )

    def test_renders_the_full_requested_selection(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _write_packaged_patch(root, "0100_x")
            _write_summary(root, "0100_x", status="validated", group="core", plan_item="none",
                            body="real content\n")
            doc = patch_docs.render_patch_selection_doc(
                patch_ids=("0100_x",), pin_info={}, selection_label="test", patches_dir=root,
            )
            self.assertIn("1 patch(es) included", doc)
            self.assertIn("real content", doc)


class ParseSummaryHeaderTests(unittest.TestCase):
    def test_extracts_all_three_fields(self):
        header = patch_docs.parse_summary_header(
            "# x\n\n**Status:** validated\n**Group:** core\n**Plan item:** RD20\n\n## What it does\n"
        )
        self.assertEqual(header, {"status": "validated", "group": "core", "plan_item": "RD20"})

    def test_returns_none_when_header_is_missing(self):
        self.assertIsNone(patch_docs.parse_summary_header("# x\n\nno header here\n"))

    def test_a_blank_line_between_fields_does_not_get_absorbed_into_a_match(self):
        # \s* crossing a newline used to let this still parse; fields are
        # now anchored line-by-line and must be consecutive.
        text = "**Status:** validated\n\n**Group:** core\n**Plan item:** none\n"
        self.assertIsNone(patch_docs.parse_summary_header(text))


class CheckSummaryConsistencyTests(unittest.TestCase):
    def test_flags_missing_summary(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _write_packaged_patch(root, "0100_x", state="validated", group="core")
            problems = patch_docs.check_summary_consistency(root)
            self.assertEqual(problems, ["0100_x: missing SUMMARY.md"])

    def test_flags_status_drift(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _write_packaged_patch(root, "0100_x", state="superseded", group="core")
            _write_summary(root, "0100_x", status="untested", group="core", plan_item="none")
            problems = patch_docs.check_summary_consistency(root)
            self.assertEqual(len(problems), 1)
            self.assertIn("Status='untested'", problems[0])
            self.assertIn("state='superseded'", problems[0])

    def test_clean_when_everything_agrees(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _write_packaged_patch(root, "0100_x", state="validated", group="core")
            _write_summary(root, "0100_x", status="validated", group="core", plan_item="none")
            self.assertEqual(patch_docs.check_summary_consistency(root), [])

    def test_no_plan_item_declared_requires_summary_to_say_none_not_anything(self):
        """A real false-negative this replaces: an absent plan-item used to
        make ANY SUMMARY.md Plan item value report clean."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _write_packaged_patch(root, "0100_x", state="validated", group="core")
            _write_summary(root, "0100_x", status="validated", group="core", plan_item="RD999")
            problems = patch_docs.check_summary_consistency(root)
            self.assertEqual(len(problems), 1)
            self.assertIn("Plan item='RD999'", problems[0])
            self.assertIn("'none'", problems[0])

    def test_plural_plan_ids_render_as_the_canonical_joined_value(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _write_packaged_patch(
                root, "1215_x", state="untested", group="rdna-boosts",
                plan_ids=("RD39", "RD40", "RD41", "RD42"),
            )
            _write_summary(root, "1215_x", status="untested", group="rdna-boosts",
                            plan_item="RD39/RD40/RD41/RD42")
            self.assertEqual(patch_docs.check_summary_consistency(root), [])

    def test_plan_ids_take_priority_over_singular_plan_item_when_both_set(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _write_packaged_patch(
                root, "1215_x", state="untested", group="rdna-boosts",
                plan_item="RD39", plan_ids=("RD39", "RD40"),
            )
            _write_summary(root, "1215_x", status="untested", group="rdna-boosts",
                            plan_item="RD39")  # matches the stale singular field, not plan_ids
            problems = patch_docs.check_summary_consistency(root)
            self.assertEqual(len(problems), 1)
            self.assertIn("'RD39/RD40'", problems[0])


if __name__ == "__main__":
    unittest.main()
